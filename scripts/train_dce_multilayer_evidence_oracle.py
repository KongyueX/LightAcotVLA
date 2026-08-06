"""Train the shared four-layer DCE residual cross-attention oracle on Task8.

Separate EAR and final adapters reuse one low-rank evidence cross-attention at
frozen Gemma layers 0, 5, 11, and 17.  Each selected layer has its own scalar
ReZero gate, initialized to exact identity, while Q/K/V/O kernels are shared.
The opt-in Gemma hook changes no suffix tokens, masks, positions, frozen expert
weights, or action projections, and no post-suffix action residual is allowed.

Training remains staged EAR-then-final.  The EAR target is the direct-splice
response.  Final base/teacher paths share the same predicted EAR and stale IAR.
Different-episode evidence is a margin-ranked negative, zero evidence must be
an exact no-op, and full-fresh outputs appear only in the Test22 report.
This GPU-only script does not enable the hook in default inference.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, Callable

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import dce_multilayer_evidence_adapter
from openpi.models import mrr_block_selector
from openpi.models.pi0 import make_attn_mask

try:
    import train_dce_dual_path_adapter_oracle as dce_base
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_dce_dual_path_adapter_oracle as dce_base


LOGGER = logging.getLogger("train_dce_multilayer_evidence_oracle")
INJECTION_LAYERS = (0, 5, 11, 17)
EXPERT_DEPTH = 18
METHOD_NAMES = dce_base.METHOD_NAMES


@dataclasses.dataclass(frozen=True)
class Args(dce_base.Args):
    evidence_mode: str = "selected128_pair"


def _validate_args(args: Args) -> None:
    dce_base._validate_args(args)  # noqa: SLF001


def _adapted_coarse_endpoint(
    base_model: Any,
    prefix_state: dict[str, Any],
    adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    evidence: jax.Array,
) -> jax.Array:
    observation = prefix_state["observation"]
    prefix_mask = prefix_state["prefix_mask"]
    coarse_noise = prefix_state["ref_action_noise"]
    batch_size = observation.state.shape[0]
    time_value = jnp.ones((batch_size,), dtype=jnp.float32)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = base_model.embed_suffix(
        observation,
        coarse_noise,
        time_value,
        suf_type="reasoner",
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, suffix_out, _), _ = base_model.PaliGemma.llm(
        [None, suffix_tokens, None],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=prefix_state["kv_cache"],
        adarms_cond=[None, adarms_cond, None],
        evidence_hook=adapter.hook_payload(evidence),
    )
    velocity = base_model.coarse_action_out_proj(
        suffix_out[:, -base_model.coarse_action_horizon :]
    )
    return coarse_noise - velocity


def _adapted_final_endpoint(
    base_model: Any,
    prefix_state: dict[str, Any],
    adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    evidence: jax.Array,
    explicit_action_reason: jax.Array,
    implicit_action_reason: jax.Array,
) -> jax.Array:
    observation = prefix_state["observation"]
    prefix_mask = prefix_state["prefix_mask"]
    action_noise = prefix_state["expert_action_noise"]
    batch_size = observation.state.shape[0]
    time_value = jnp.ones((batch_size,), dtype=jnp.float32)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = base_model.embed_suffix(
        observation,
        action_noise,
        time_value,
        explicit_action_reason=explicit_action_reason,
        implicit_action_reason=implicit_action_reason,
        suf_type="expert",
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, _, suffix_out), _ = base_model.PaliGemma.llm(
        [None, None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=prefix_state["kv_cache"],
        adarms_cond=[None, None, adarms_cond],
        evidence_hook=adapter.hook_payload(evidence),
    )
    velocity = base_model.action_out_proj(suffix_out[:, -base_model.action_horizon :])
    return action_noise - velocity


def _gate_metrics(
    adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
) -> dict[str, jax.Array]:
    gates = jnp.tanh(adapter.layer_gates.value)
    return {
        "gate_mean_abs": jnp.mean(jnp.abs(gates)),
        **{f"gate_layer_{layer}": gates[index] for index, layer in enumerate(INJECTION_LAYERS)},
    }


def _ear_loss(
    base_model: Any,
    adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
        context["causal_prefix"],
        context["causal_prefix"]["ref_action_noise"],
    )
    stale_ear, direct_ear = teacher_ear[0:1], teacher_ear[1:2]
    zero_evidence = jnp.zeros_like(context["evidence"])
    if mismatched_evidence is None:
        candidate_evidence = jnp.concatenate([context["evidence"], zero_evidence], axis=0)
        candidate_count = 2
    else:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], mismatched_evidence, zero_evidence],
            axis=0,
        )
        candidate_count = 3
    candidate_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
        context["anchor_prefix"],
        context["fresh_prefix"],
        candidate_count,
    )
    candidate_ear = _adapted_coarse_endpoint(
        base_model,
        candidate_prefix,
        adapter,
        candidate_evidence,
    )
    matched_ear = candidate_ear[0:1]
    zero_ear = candidate_ear[-1:]
    teacher_response = direct_ear - stale_ear
    matched_response = matched_ear - stale_ear
    response_loss = dce_base._continuous_mse(matched_response, teacher_response)  # noqa: SLF001
    zero_loss = dce_base._continuous_mse(  # noqa: SLF001
        zero_ear - stale_ear,
        jnp.zeros_like(stale_ear),
    )
    if mismatched_evidence is None:
        mismatch_distance = jnp.zeros((), dtype=jnp.float32)
        ranking_loss = jnp.zeros((), dtype=jnp.float32)
    else:
        mismatch_response = candidate_ear[1:2] - stale_ear
        mismatch_distance = dce_base._continuous_mse(  # noqa: SLF001
            mismatch_response,
            teacher_response,
        )
        ranking_loss = jax.nn.relu(
            args.contrastive_margin + response_loss - mismatch_distance
        )
    gripper_loss, gripper_event_fraction = dce_base._event_gripper_margin(  # noqa: SLF001
        matched_ear,
        direct_ear,
        stale_ear,
        margin=args.gripper_sign_margin,
    )
    loss = (
        args.ear_response_loss_weight * response_loss
        + args.ear_zero_loss_weight * zero_loss
        + args.ear_ranking_loss_weight * ranking_loss
        + args.gripper_event_loss_weight * gripper_loss
    )
    return loss, {
        "matched_response_mse_6d": response_loss,
        "mismatch_distance_to_teacher_6d": mismatch_distance,
        "zero_response_mse_6d": zero_loss,
        "contrastive_ranking_loss": ranking_loss,
        "gripper_event_margin_loss": gripper_loss,
        "gripper_event_fraction": gripper_event_fraction,
        **_gate_metrics(adapter),
    }


def _final_loss(
    base_model: Any,
    ear_adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    final_adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    stale_iar = base_model.sample_actions_profile_implicit(context["stale_prefix"])[
        "implicit_action_reason"
    ]
    if stale_iar is None:
        raise ValueError("Multi-layer DCE requires the frozen implicit action reasoner.")
    predicted_ear = _adapted_coarse_endpoint(
        base_model,
        context["stale_prefix"],
        ear_adapter,
        context["evidence"],
    )
    causal_prefix = context["causal_prefix"]
    base_teacher_actions = base_model._one_step_action_endpoint(  # noqa: SLF001
        causal_prefix,
        causal_prefix["expert_action_noise"],
        jnp.repeat(predicted_ear, 2, axis=0),
        jnp.repeat(stale_iar, 2, axis=0),
    )
    base_action, teacher_action = base_teacher_actions[0:1], base_teacher_actions[1:2]
    zero_evidence = jnp.zeros_like(context["evidence"])
    if mismatched_evidence is None:
        candidate_evidence = jnp.concatenate([context["evidence"], zero_evidence], axis=0)
        candidate_count = 2
    else:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], mismatched_evidence, zero_evidence],
            axis=0,
        )
        candidate_count = 3
    candidate_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
        context["anchor_prefix"],
        context["fresh_prefix"],
        candidate_count,
    )
    candidate_actions = _adapted_final_endpoint(
        base_model,
        candidate_prefix,
        final_adapter,
        candidate_evidence,
        jnp.repeat(predicted_ear, candidate_count, axis=0),
        jnp.repeat(stale_iar, candidate_count, axis=0),
    )
    matched_action = candidate_actions[0:1]
    zero_action = candidate_actions[-1:]
    teacher_response = teacher_action - base_action
    matched_response = matched_action - base_action
    response_loss = dce_base._continuous_mse(matched_response, teacher_response)  # noqa: SLF001
    zero_loss = dce_base._continuous_mse(  # noqa: SLF001
        zero_action - base_action,
        jnp.zeros_like(base_action),
    )
    if mismatched_evidence is None:
        mismatch_distance = jnp.zeros((), dtype=jnp.float32)
        ranking_loss = jnp.zeros((), dtype=jnp.float32)
    else:
        mismatch_response = candidate_actions[1:2] - base_action
        mismatch_distance = dce_base._continuous_mse(  # noqa: SLF001
            mismatch_response,
            teacher_response,
        )
        ranking_loss = jax.nn.relu(
            args.contrastive_margin + response_loss - mismatch_distance
        )
    gripper_loss, gripper_event_fraction = dce_base._event_gripper_margin(  # noqa: SLF001
        matched_action,
        teacher_action,
        base_action,
        margin=args.gripper_sign_margin,
    )
    loss = (
        args.final_response_loss_weight * response_loss
        + args.final_zero_loss_weight * zero_loss
        + args.final_ranking_loss_weight * ranking_loss
        + args.gripper_event_loss_weight * gripper_loss
    )
    return loss, {
        "matched_response_mse_6d": response_loss,
        "mismatch_distance_to_teacher_6d": mismatch_distance,
        "zero_response_mse_6d": zero_loss,
        "contrastive_ranking_loss": ranking_loss,
        "gripper_event_margin_loss": gripper_loss,
        "gripper_event_fraction": gripper_event_fraction,
        "matched_teacher_gripper_accuracy": dce_base._gripper_accuracy(  # noqa: SLF001
            matched_action,
            teacher_action,
        ),
        **_gate_metrics(final_adapter),
    }


def _make_steps(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    ear_graphdef: Any,
    final_graphdef: Any,
    optimizer: optax.GradientTransformation,
    args: Args,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    @jax.jit
    def ear_train_step(
        base_state: nnx.State,
        scorer_state: nnx.State,
        params: nnx.State,
        optimizer_state: optax.OptState,
        batch: dict[str, Any],
        rng: jax.Array,
        mismatched_evidence: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        adapter = nnx.merge(ear_graphdef, params)

        def loss_fn(candidate: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter):
            return _ear_loss(
                base_model,
                candidate,
                context,
                args,
                mismatched_evidence=mismatched_evidence,
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(adapter)
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_optimizer_state, {
            **metrics,
            "loss": loss,
            "gradient_norm": optax.global_norm(gradients),
        }

    @jax.jit
    def ear_eval_step(
        base_state: nnx.State,
        scorer_state: nnx.State,
        params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        mismatched_evidence: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        adapter = nnx.merge(ear_graphdef, params)
        loss, metrics = _ear_loss(
            base_model,
            adapter,
            context,
            args,
            mismatched_evidence=mismatched_evidence,
        )
        return {**metrics, "loss": loss}

    @jax.jit
    def final_train_step(
        base_state: nnx.State,
        scorer_state: nnx.State,
        ear_params: nnx.State,
        params: nnx.State,
        optimizer_state: optax.OptState,
        batch: dict[str, Any],
        rng: jax.Array,
        mismatched_evidence: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, params)

        def loss_fn(candidate: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter):
            return _final_loss(
                base_model,
                ear_adapter,
                candidate,
                context,
                args,
                mismatched_evidence=mismatched_evidence,
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(final_adapter)
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_optimizer_state, {
            **metrics,
            "loss": loss,
            "gradient_norm": optax.global_norm(gradients),
        }

    @jax.jit
    def final_eval_step(
        base_state: nnx.State,
        scorer_state: nnx.State,
        ear_params: nnx.State,
        final_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        mismatched_evidence: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, final_params)
        loss, metrics = _final_loss(
            base_model,
            ear_adapter,
            final_adapter,
            context,
            args,
            mismatched_evidence=mismatched_evidence,
        )
        return {**metrics, "loss": loss}

    return ear_train_step, ear_eval_step, final_train_step, final_eval_step


def _make_evidence_exporter(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    args: Args,
):
    @jax.jit
    def export(
        base_state: nnx.State,
        scorer_state: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        return {
            "evidence": context["evidence"],
            "selected_ids": context["selected_ids"],
        }

    return export


def _all_action_mse(predicted: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(
        jnp.square(predicted.astype(jnp.float32) - target.astype(jnp.float32))
    )


def _make_identity_drift_evaluator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    ear_graphdef: Any,
    final_graphdef: Any,
    args: Args,
):
    @jax.jit
    def evaluate(
        base_state: nnx.State,
        scorer_state: nnx.State,
        ear_params: nnx.State,
        final_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, final_params)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        original_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            context["stale_prefix"],
            context["stale_prefix"]["ref_action_noise"],
        )
        adapted_ear = _adapted_coarse_endpoint(
            base_model,
            context["stale_prefix"],
            ear_adapter,
            context["evidence"],
        )
        stale_iar = base_model.sample_actions_profile_implicit(context["stale_prefix"])[
            "implicit_action_reason"
        ]
        if stale_iar is None:
            raise ValueError("Multi-layer DCE requires the frozen implicit action reasoner.")
        original_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["stale_prefix"],
            context["stale_prefix"]["expert_action_noise"],
            original_ear,
            stale_iar,
        )
        adapted_action = _adapted_final_endpoint(
            base_model,
            context["stale_prefix"],
            final_adapter,
            context["evidence"],
            original_ear,
            stale_iar,
        )
        return {
            "ear_mse_7d": _all_action_mse(adapted_ear, original_ear),
            "action_mse_7d": _all_action_mse(adapted_action, original_action),
        }

    return evaluate


def _make_test_evaluator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    ear_graphdef: Any,
    final_graphdef: Any,
    args: Args,
):
    @jax.jit
    def evaluate(
        base_state: nnx.State,
        scorer_state: nnx.State,
        ear_params: nnx.State,
        final_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        shuffled_evidence: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, final_params)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        comparison = context["comparison"]

        teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            comparison,
            comparison["ref_action_noise"],
        )
        teacher_iar = base_model.sample_actions_profile_implicit(comparison)[
            "implicit_action_reason"
        ]
        if teacher_iar is None:
            raise ValueError("Multi-layer DCE requires the frozen implicit action reasoner.")
        stale_ear, fresh_ear = teacher_ear[0:1], teacher_ear[2:3]
        stale_iar, fresh_iar = teacher_iar[0:1], teacher_iar[2:3]
        predicted_ear = _adapted_coarse_endpoint(
            base_model,
            context["stale_prefix"],
            ear_adapter,
            context["evidence"],
        )
        shuffled_ear = _adapted_coarse_endpoint(
            base_model,
            context["stale_prefix"],
            ear_adapter,
            shuffled_evidence,
        )

        plain_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
            context["anchor_prefix"],
            context["fresh_prefix"],
            2,
        )
        plain_actions = base_model._one_step_action_endpoint(  # noqa: SLF001
            plain_prefix,
            plain_prefix["expert_action_noise"],
            jnp.concatenate([stale_ear, predicted_ear], axis=0),
            jnp.repeat(stale_iar, 2, axis=0),
        )
        stale_action, plan_action = plain_actions[0:1], plain_actions[1:2]

        adapted_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
            context["anchor_prefix"],
            context["fresh_prefix"],
            2,
        )
        adapted_actions = _adapted_final_endpoint(
            base_model,
            adapted_prefix,
            final_adapter,
            jnp.repeat(context["evidence"], 2, axis=0),
            jnp.concatenate([stale_ear, predicted_ear], axis=0),
            jnp.repeat(stale_iar, 2, axis=0),
        )
        direct_action, joint_action = adapted_actions[0:1], adapted_actions[1:2]
        shuffled_action = _adapted_final_endpoint(
            base_model,
            context["stale_prefix"],
            final_adapter,
            shuffled_evidence,
            shuffled_ear,
            stale_iar,
        )
        fresh_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["fresh_prefix"],
            context["fresh_prefix"]["expert_action_noise"],
            fresh_ear,
            fresh_iar,
        )

        zero_evidence = jnp.zeros_like(context["evidence"])
        zero_ear = _adapted_coarse_endpoint(
            base_model,
            context["stale_prefix"],
            ear_adapter,
            zero_evidence,
        )
        zero_action = _adapted_final_endpoint(
            base_model,
            context["stale_prefix"],
            final_adapter,
            zero_evidence,
            stale_ear,
            stale_iar,
        )
        original_zero_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["stale_prefix"],
            context["stale_prefix"]["expert_action_noise"],
            stale_ear,
            stale_iar,
        )

        variant_ear = jnp.concatenate(
            [stale_ear, predicted_ear, stale_ear, predicted_ear, fresh_ear, shuffled_ear],
            axis=0,
        )
        variant_iar = jnp.concatenate(
            [stale_iar, stale_iar, stale_iar, stale_iar, fresh_iar, stale_iar],
            axis=0,
        )
        variant_actions = jnp.concatenate(
            [stale_action, plan_action, direct_action, joint_action, fresh_action, shuffled_action],
            axis=0,
        )
        metrics = dce_base.mrr_oracle._downstream_metrics(  # noqa: SLF001
            variant_iar,
            variant_ear,
            variant_actions,
            fresh_iar,
            fresh_ear,
            fresh_action,
        )
        return {
            "metrics": metrics,
            "selected_ids": context["selected_ids"],
            "ear_gates": jnp.tanh(ear_adapter.layer_gates.value),
            "final_gates": jnp.tanh(final_adapter.layer_gates.value),
            "zero_evidence_ear_drift_mse_7d": _all_action_mse(zero_ear, stale_ear),
            "zero_evidence_action_drift_mse_7d": _all_action_mse(
                zero_action,
                original_zero_action,
            ),
        }

    return evaluate


def _aggregate_test(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    result = dce_base._aggregate_test(records, args)  # noqa: SLF001
    result["zero_evidence_identity_drift"] = {
        "ear_mse_7d": float(
            np.mean([record["zero_evidence_ear_drift_mse_7d"] for record in records])
        ),
        "action_mse_7d": float(
            np.mean([record["zero_evidence_action_drift_mse_7d"] for record in records])
        ),
    }
    if not result["gate"]["pass"]:
        result["gate"]["decision"] = "no_go_shared_four_layer_residual_ca"
    return result


def _evidence_token_contract(mode: str) -> str:
    if mode == "block16":
        return "one absolute-current mean plus one current-minus-anchor mean per block"
    if mode == "selected128":
        return "all current pre-Gemma visual tokens in selector top-8 rank order"
    if mode == "selected128_pair":
        return (
            "per-token concat(current, current-minus-anchor) for all selected tokens "
            "in selector top-8 rank order"
        )
    raise ValueError(f"Unsupported evidence mode {mode!r}.")


def _validation_metrics(
    evaluator: Callable[..., dict[str, jax.Array]],
    indices: np.ndarray,
    pairs: Any,
    records: Any,
    arguments: tuple[Any, ...],
    *,
    seed: int,
    mismatch_map: dict[int, int],
    evidence_by_index: dict[int, np.ndarray],
) -> dict[str, float]:
    values: list[dict[str, float]] = []
    for record_index_value in indices:
        record_index = int(record_index_value)
        batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
            records,
            np.asarray([record_index], dtype=np.int64),
        )
        anchor = int(pairs.anchor_indices[record_index])
        mismatch_index = mismatch_map[record_index]
        output = jax.device_get(
            evaluator(
                *arguments,
                batch,
                jax.random.fold_in(jax.random.key(seed), anchor),
                jnp.asarray(evidence_by_index[mismatch_index]),
            )
        )
        values.append({name: float(value) for name, value in output.items()})
    return dce_base._metric_mean(values)  # noqa: SLF001


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = dce_base.p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    per_pair_path = output_dir / "test_pairs.jsonl"
    summary_path = output_dir / "summary.json"
    ear_params_path = output_dir / "final" / "ear_multilayer_adapter" / "params"
    final_params_path = output_dir / "final" / "final_multilayer_adapter" / "params"
    targets = (metrics_path, per_pair_path, summary_path, ear_params_path, final_params_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"Multi-layer DCE output already exists in {output_dir}; pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    trainer_args = dce_base.p3t_trainer.Args(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        endpoint_student_params=args.endpoint_student_params,
        output_dir=args.output_dir,
        config_name=args.config_name,
        dataset_task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        seed=args.seed,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    arrays = multirate_dataset.load_multirate_arrays(
        args.dataset,
        fields=("anchor_index", "task_id", "episode_id", "frame_id", "fresh_ear"),
    )
    base_graphdef, base_state, observation_dataset, raw_dataset, model_metadata = (
        dce_base.p3t_trainer._load_model_and_dataset(trainer_args)  # noqa: SLF001
    )
    pairs = dce_base.p3t_trainer._select_pairs(  # noqa: SLF001
        arrays,
        raw_dataset,
        task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        maximum_pairs=200,
        seed=args.seed,
    )
    train_indices, validation_indices, test_indices = dce_base.p3t_trainer._split_pairs(  # noqa: SLF001
        pairs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    actual_split = (len(pairs), train_indices.size, validation_indices.size, test_indices.size)
    expected_split = (
        args.expected_pairs,
        args.expected_train_pairs,
        args.expected_validation_pairs,
        args.expected_test_pairs,
    )
    if actual_split != expected_split:
        raise ValueError(f"Expected multi-layer split {expected_split}, got {actual_split}.")
    records = dce_base.p3t_trainer._materialize_pairs(  # noqa: SLF001
        pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    selector_runtime = mrr_block_selector.load_mrr_block_selector(args.selector_checkpoint)

    base_model = nnx.merge(base_graphdef, base_state)
    query_dim = int(base_model.action_in_proj.out_features)
    coarse_query_dim = int(base_model.coarse_action_in_proj.out_features)
    if query_dim != coarse_query_dim:
        raise ValueError(f"EAR/final suffix widths differ: {coarse_query_dim} vs {query_dim}.")
    evidence_tokens = dce_base.EVIDENCE_TOKENS_BY_MODE[args.evidence_mode]
    evidence_dim = int(mrr_block_selector.TOKEN_EMBEDDING_DIM) * (
        2 if args.evidence_mode == "selected128_pair" else 1
    )
    adapter_config = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapterConfig(
        evidence_dim=evidence_dim,
        evidence_tokens=evidence_tokens,
        expert_dim=query_dim,
        attention_dim=args.attention_dim,
        num_heads=args.attention_heads,
        expert_depth=EXPERT_DEPTH,
        injection_layers=INJECTION_LAYERS,
    )
    ear_adapter = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter(
        adapter_config,
        rngs=nnx.Rngs(args.seed + 505),
    )
    final_adapter = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter(
        adapter_config,
        rngs=nnx.Rngs(args.seed + 606),
    )
    ear_graphdef, ear_params = nnx.split(ear_adapter)
    final_graphdef, final_params = nnx.split(final_adapter)
    ear_parameter_count = int(
        sum(np.asarray(value).size for value in jax.tree.leaves(ear_params.to_pure_dict()))
    )
    final_parameter_count = int(
        sum(np.asarray(value).size for value in jax.tree.leaves(final_params.to_pure_dict()))
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    ear_optimizer_state = optimizer.init(ear_params)
    final_optimizer_state = optimizer.init(final_params)
    ear_train_step, ear_eval_step, final_train_step, final_eval_step = _make_steps(
        base_graphdef,
        selector_runtime,
        ear_graphdef,
        final_graphdef,
        optimizer,
        args,
    )

    evidence_exporter = _make_evidence_exporter(base_graphdef, selector_runtime, args)
    mismatch_map = dce_base._deterministic_mismatch_map(train_indices, pairs)  # noqa: SLF001
    validation_mismatch_map = dce_base._deterministic_mismatch_map(  # noqa: SLF001
        validation_indices,
        pairs,
    )
    train_evidence: dict[int, np.ndarray] = {}
    for position, record_index_value in enumerate(train_indices):
        record_index = int(record_index_value)
        anchor = int(pairs.anchor_indices[record_index])
        batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
            records,
            np.asarray([record_index], dtype=np.int64),
        )
        output = jax.device_get(
            evidence_exporter(
                base_state,
                selector_runtime.scorer_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), anchor),
            )
        )
        train_evidence[record_index] = np.asarray(output["evidence"], dtype=np.float32)
        if position == 0 or (position + 1) % 20 == 0 or position + 1 == train_indices.size:
            LOGGER.info(
                "Exported train multi-layer evidence %d/%d",
                position + 1,
                train_indices.size,
            )

    validation_evidence: dict[int, np.ndarray] = {}
    for record_index_value in validation_indices:
        record_index = int(record_index_value)
        anchor = int(pairs.anchor_indices[record_index])
        batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
            records,
            np.asarray([record_index], dtype=np.int64),
        )
        output = jax.device_get(
            evidence_exporter(
                base_state,
                selector_runtime.scorer_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), anchor),
            )
        )
        validation_evidence[record_index] = np.asarray(output["evidence"], dtype=np.float32)

    drift_evaluator = _make_identity_drift_evaluator(
        base_graphdef,
        selector_runtime,
        ear_graphdef,
        final_graphdef,
        args,
    )
    drift_record_index = int(validation_indices[0])
    drift_anchor = int(pairs.anchor_indices[drift_record_index])
    drift_batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
        records,
        np.asarray([drift_record_index], dtype=np.int64),
    )
    initial_rezero_identity_drift = {
        name: float(value)
        for name, value in jax.device_get(
            drift_evaluator(
                base_state,
                selector_runtime.scorer_state,
                ear_params,
                final_params,
                drift_batch,
                jax.random.fold_in(jax.random.key(args.seed), drift_anchor),
            )
        ).items()
    }

    sampling_rng = np.random.default_rng(args.seed)
    mode = "w" if args.overwrite else "a"
    with metrics_path.open(mode, encoding="utf-8") as metrics_file:
        metrics_file.write(
            json.dumps(
                {"stage": "initial_rezero_identity_drift", **initial_rezero_identity_drift},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        metrics_file.flush()

        for step in range(1, args.ear_steps + 1):
            record_index = int(sampling_rng.choice(train_indices))
            mismatch_index = mismatch_map[record_index]
            anchor = int(pairs.anchor_indices[record_index])
            batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
                records,
                np.asarray([record_index], dtype=np.int64),
            )
            ear_params, ear_optimizer_state, output = ear_train_step(
                base_state,
                selector_runtime.scorer_state,
                ear_params,
                ear_optimizer_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), anchor),
                jnp.asarray(train_evidence[mismatch_index]),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.ear_steps:
                host = jax.device_get(output)
                record = {
                    "stage": "ear",
                    "step": step,
                    "pair_index": record_index,
                    "mismatch_pair_index": mismatch_index,
                    "elapsed_seconds": time.monotonic() - started,
                    **{name: float(value) for name, value in host.items()},
                }
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                LOGGER.info(
                    "Multi-layer EAR step %d/%d loss=%.6f",
                    step,
                    args.ear_steps,
                    record["loss"],
                )

        ear_validation = _validation_metrics(
            ear_eval_step,
            validation_indices,
            pairs,
            records,
            (base_state, selector_runtime.scorer_state, ear_params),
            seed=args.seed,
            mismatch_map=validation_mismatch_map,
            evidence_by_index=validation_evidence,
        )
        metrics_file.write(
            json.dumps(
                {"stage": "ear_validation", **ear_validation},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        metrics_file.flush()

        for step in range(1, args.final_steps + 1):
            record_index = int(sampling_rng.choice(train_indices))
            mismatch_index = mismatch_map[record_index]
            anchor = int(pairs.anchor_indices[record_index])
            batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
                records,
                np.asarray([record_index], dtype=np.int64),
            )
            final_params, final_optimizer_state, output = final_train_step(
                base_state,
                selector_runtime.scorer_state,
                ear_params,
                final_params,
                final_optimizer_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), anchor),
                jnp.asarray(train_evidence[mismatch_index]),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.final_steps:
                host = jax.device_get(output)
                record = {
                    "stage": "final",
                    "step": step,
                    "pair_index": record_index,
                    "mismatch_pair_index": mismatch_index,
                    "elapsed_seconds": time.monotonic() - started,
                    **{name: float(value) for name, value in host.items()},
                }
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                LOGGER.info(
                    "Multi-layer final step %d/%d loss=%.6f",
                    step,
                    args.final_steps,
                    record["loss"],
                )

        final_validation = _validation_metrics(
            final_eval_step,
            validation_indices,
            pairs,
            records,
            (base_state, selector_runtime.scorer_state, ear_params, final_params),
            seed=args.seed,
            mismatch_map=validation_mismatch_map,
            evidence_by_index=validation_evidence,
        )
        metrics_file.write(
            json.dumps(
                {"stage": "final_validation", **final_validation},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        metrics_file.flush()

    saved_ear = dce_base._save_params(  # noqa: SLF001
        ear_params,
        ear_params_path,
        name="dce_ear_multilayer_evidence_adapter",
        overwrite=args.overwrite,
    )
    saved_final = dce_base._save_params(  # noqa: SLF001
        final_params,
        final_params_path,
        name="dce_final_multilayer_evidence_adapter",
        overwrite=args.overwrite,
    )

    test_evidence = []
    for record_index_value in test_indices:
        record_index = int(record_index_value)
        anchor = int(pairs.anchor_indices[record_index])
        batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
            records,
            np.asarray([record_index], dtype=np.int64),
        )
        output = jax.device_get(
            evidence_exporter(
                base_state,
                selector_runtime.scorer_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), anchor),
            )
        )
        test_evidence.append(np.asarray(output["evidence"], dtype=np.float32))
    shuffled_test_evidence = np.roll(np.stack(test_evidence), shift=1, axis=0)

    test_evaluator = _make_test_evaluator(
        base_graphdef,
        selector_runtime,
        ear_graphdef,
        final_graphdef,
        args,
    )
    test_records: list[dict[str, Any]] = []
    with per_pair_path.open("w", encoding="utf-8") as output_file:
        for test_position, record_index_value in enumerate(test_indices):
            record_index = int(record_index_value)
            anchor = int(pairs.anchor_indices[record_index])
            batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
                records,
                np.asarray([record_index], dtype=np.int64),
            )
            output = jax.device_get(
                test_evaluator(
                    base_state,
                    selector_runtime.scorer_state,
                    ear_params,
                    final_params,
                    batch,
                    jax.random.fold_in(jax.random.key(args.seed), anchor),
                    jnp.asarray(shuffled_test_evidence[test_position]),
                )
            )
            metric_arrays = np.asarray(output["metrics"], dtype=np.float64)
            if not np.all(np.isfinite(metric_arrays)):
                raise FloatingPointError(
                    f"Non-finite multi-layer test output at pair {record_index}."
                )
            record = {
                "test_position": test_position,
                "pair_index": record_index,
                "anchor_index": anchor,
                "target_index": int(pairs.target_indices[record_index]),
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [
                    int(value) for value in np.asarray(output["selected_ids"])
                ],
                "metric_arrays": metric_arrays.tolist(),
                "methods": {
                    name: {
                        "metrics": dce_base.mrr_oracle._metric_dict(metric_arrays[index]),  # noqa: SLF001
                        "gap_closure_vs_stale": dce_base._closure(  # noqa: SLF001
                            metric_arrays[index],
                            metric_arrays[0],
                        ),
                    }
                    for index, name in enumerate(METHOD_NAMES)
                },
                "ear_layer_gates": [float(value) for value in np.asarray(output["ear_gates"])],
                "final_layer_gates": [
                    float(value) for value in np.asarray(output["final_gates"])
                ],
                "zero_evidence_ear_drift_mse_7d": float(
                    output["zero_evidence_ear_drift_mse_7d"]
                ),
                "zero_evidence_action_drift_mse_7d": float(
                    output["zero_evidence_action_drift_mse_7d"]
                ),
            }
            output_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output_file.flush()
            test_records.append(record)
            LOGGER.info(
                "Multi-layer test pair %d/%d anchor=%d",
                test_position + 1,
                test_indices.size,
                anchor,
            )

    evaluation = _aggregate_test(test_records, args)
    summary = {
        "method": "DCE shared four-layer residual cross-attention oracle",
        "status": "offline_test22_oracle_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
            "test_episodes": sorted(
                int(value) for value in np.unique(pairs.episode_ids[test_indices])
            ),
        },
        "evidence": {
            "mode": args.evidence_mode,
            "selector": "learned MRR logits",
            "top_k_blocks": dce_base.TOP_K,
            "source_visual_tokens": (
                dce_base.TOP_K * dce_base.mrr_oracle.TOKENS_PER_BLOCK
            ),
            "evidence_tokens": evidence_tokens,
            "evidence_dim": evidence_dim,
            "token_contract": _evidence_token_contract(args.evidence_mode),
            "privileged_encoder": "full current SigLIP/pre-Gemma visual representation",
        },
        "architecture": {
            "mechanism": "shared residual cross-attention after selected frozen Gemma blocks",
            "injection_layers": list(INJECTION_LAYERS),
            "expert_depth": EXPERT_DEPTH,
            "attention_dim": args.attention_dim,
            "attention_heads": args.attention_heads,
            "ear_adapter_parameters": ear_parameter_count,
            "final_adapter_parameters": final_parameter_count,
            "expert_dim": query_dim,
            "gates": "independent per-layer ReZero tanh gates initialized exactly zero",
            "shared_qkvo_across_layers": True,
            "suffix_tokens_unchanged": True,
            "suffix_mask_unchanged": True,
            "suffix_positions_unchanged": True,
            "base_frozen": True,
            "coarse_action_out_proj_frozen": True,
            "action_out_proj_frozen": True,
            "post_suffix_action_residual": False,
            "default_inference_hook": None,
        },
        "training": {
            "order": "EAR adapter first; frozen predicted EAR conditions final-adapter training",
            "teacher": "strict direct-splice response with no evidence hook",
            "mismatched_evidence": {
                "pairing": "deterministic different-episode pairing",
                "cached_train_evidence_pairs": len(train_evidence),
                "margin": args.contrastive_margin,
                "ear_ranking_weight": args.ear_ranking_loss_weight,
                "final_ranking_weight": args.final_ranking_loss_weight,
            },
            "zero_evidence": {
                "contract": "bias-free K/V makes every residual exactly zero",
                "ear_weight": args.ear_zero_loss_weight,
                "final_weight": args.final_zero_loss_weight,
            },
            "initial_rezero_identity_drift": initial_rezero_identity_drift,
            "ear_validation": ear_validation,
            "final_validation": final_validation,
            "ear_checkpoint": str(saved_ear),
            "final_checkpoint": str(saved_final),
        },
        "test": evaluation,
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "metrics": str(metrics_path),
            "test_pairs": str(per_pair_path),
            "summary": str(summary_path),
        },
        "constraints": {
            "runtime_integration": False,
            "deployable_speed_claim": False,
            "closed_loop_success_claim": False,
            "full_fresh_used_for_training": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Multi-layer DCE complete decision=%s elapsed=%.1fs",
        evaluation["gate"]["decision"],
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
