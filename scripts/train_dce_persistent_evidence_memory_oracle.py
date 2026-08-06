"""Train a persistent multi-layer evidence-memory oracle on Task8.

The frozen ACoT policy supplies anchor/current pairs and learned-selector top-8
visual evidence.  Separate EAR and final memory projectors compress that
evidence into a small trainable suffix memory.  Memory tokens are inserted
immediately before the action tokens and traverse the unchanged frozen
18-layer Gemma expert, so every layer can read and update evidence without a
Gemma scan hook.  Only the original action-token outputs reach the frozen
coarse/final action projections; there is no post-suffix action residual.

All counterfactual branches use the same static persistent-slot shape, but a
per-row validity mask activates slots only for matched/mismatched evidence.
Baseline, zero-evidence, and fresh-teacher rows mask every memory slot and keep
the original action-token RoPE positions, exactly recovering the frozen suffix
instead of diluting its attention softmax with null keys.
Training remains staged EAR-then-final with different-episode mismatch ranking,
zero-response supervision, and the six-way Test22 causal closure report.
This script is GPU-only and does not modify the default inference path.
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
from openpi.models import dce_persistent_evidence_memory
from openpi.models import mrr_block_selector
from openpi.models.pi0 import make_attn_mask

try:
    import train_dce_dual_path_adapter_oracle as dce_base
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_dce_dual_path_adapter_oracle as dce_base


LOGGER = logging.getLogger("train_dce_persistent_evidence_memory_oracle")
METHOD_NAMES = dce_base.METHOD_NAMES


@dataclasses.dataclass(frozen=True)
class Args(dce_base.Args):
    evidence_mode: str = "selected128_pair"
    memory_tokens: int = 8


def _validate_args(args: Args) -> None:
    dce_base._validate_args(args)  # noqa: SLF001
    if args.memory_tokens <= 0:
        raise ValueError("memory_tokens must be positive.")


def _extend_adarms_condition(
    condition: Any | None,
    *,
    original_tokens: int,
    memory_tokens: int,
) -> Any | None:
    """Insert memory-token conditioning when AdaRMS is token-wise."""

    if condition is None:
        return None

    def extend_tokenwise(values: jax.Array) -> jax.Array:
        if values.ndim != 3 or values.shape[1] != original_tokens:
            raise ValueError(
                "Token-wise AdaRMS condition must match the original suffix; "
                f"got {values.shape} for {original_tokens} tokens."
            )
        memory_value = jnp.repeat(values[:, :1], memory_tokens, axis=1)
        return jnp.concatenate([memory_value, values], axis=1)

    if isinstance(condition, tuple):
        basis_condition, token_weights = condition
        return basis_condition, extend_tokenwise(token_weights)
    if condition.ndim == 2:
        return condition
    return extend_tokenwise(condition)


def _prepend_persistent_memory(
    suffix_tokens: jax.Array,
    suffix_mask: jax.Array,
    suffix_ar_mask: jax.Array,
    adarms_cond: Any | None,
    memory_tokens: jax.Array,
    memory_active: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, Any | None]:
    """Prepend one masked memory block without changing original suffix order."""

    if memory_tokens.ndim != 3 or memory_tokens.shape[0] != suffix_tokens.shape[0]:
        raise ValueError(
            "Persistent memory must have shape [B,M,D] with the suffix batch; "
            f"got {memory_tokens.shape} and {suffix_tokens.shape}."
        )
    if memory_tokens.shape[-1] != suffix_tokens.shape[-1]:
        raise ValueError(
            "Persistent memory and suffix widths differ: "
            f"{memory_tokens.shape[-1]} vs {suffix_tokens.shape[-1]}."
        )
    memory_count = memory_tokens.shape[1]
    if memory_active.shape != (suffix_tokens.shape[0],):
        raise ValueError(
            f"memory_active must have shape {(suffix_tokens.shape[0],)}, "
            f"got {memory_active.shape}."
        )
    tokens = jnp.concatenate([memory_tokens, suffix_tokens], axis=1)
    memory_mask = jnp.repeat(memory_active[:, None], memory_count, axis=1)
    input_mask = jnp.concatenate([memory_mask, suffix_mask], axis=1)
    memory_ar_mask = jnp.concatenate(
        [
            jnp.ones((1,), dtype=jnp.bool_),
            jnp.zeros((memory_count - 1,), dtype=jnp.bool_),
        ]
    )
    ar_mask = jnp.concatenate([memory_ar_mask, suffix_ar_mask], axis=0)
    condition = _extend_adarms_condition(
        adarms_cond,
        original_tokens=suffix_tokens.shape[1],
        memory_tokens=memory_count,
    )
    return tokens, input_mask, ar_mask, condition


def _persistent_positions(
    prefix_mask: jax.Array,
    original_suffix_mask: jax.Array,
    memory_tokens: int,
) -> jax.Array:
    """Share prefix-end position for memory and preserve all suffix positions."""

    prefix_end = jnp.sum(prefix_mask, axis=-1)
    memory_positions = jnp.repeat(prefix_end[:, None], memory_tokens, axis=1)
    suffix_positions = (
        prefix_end[:, None] + jnp.cumsum(original_suffix_mask, axis=-1) - 1
    )
    return jnp.concatenate([memory_positions, suffix_positions], axis=1)


def _persistent_coarse_endpoint(
    base_model: Any,
    prefix_state: dict[str, Any],
    memory: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory,
    evidence: jax.Array,
    memory_active: jax.Array,
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
    original_suffix_mask = suffix_mask
    memory_values = memory(evidence).astype(suffix_tokens.dtype)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = _prepend_persistent_memory(
        suffix_tokens,
        original_suffix_mask,
        suffix_ar_mask,
        adarms_cond,
        memory_values,
        memory_active,
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = _persistent_positions(
        prefix_mask,
        original_suffix_mask,
        memory_values.shape[1],
    )
    (_, suffix_out, _), _ = base_model.PaliGemma.llm(
        [None, suffix_tokens, None],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=prefix_state["kv_cache"],
        adarms_cond=[None, adarms_cond, None],
    )
    velocity = base_model.coarse_action_out_proj(
        suffix_out[:, -base_model.coarse_action_horizon :]
    )
    return coarse_noise - velocity


def _persistent_final_endpoint(
    base_model: Any,
    prefix_state: dict[str, Any],
    memory: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory,
    evidence: jax.Array,
    explicit_action_reason: jax.Array,
    implicit_action_reason: jax.Array,
    memory_active: jax.Array,
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
    original_suffix_mask = suffix_mask
    memory_values = memory(evidence).astype(suffix_tokens.dtype)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = _prepend_persistent_memory(
        suffix_tokens,
        original_suffix_mask,
        suffix_ar_mask,
        adarms_cond,
        memory_values,
        memory_active,
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = _persistent_positions(
        prefix_mask,
        original_suffix_mask,
        memory_values.shape[1],
    )
    (_, _, suffix_out), _ = base_model.PaliGemma.llm(
        [None, None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=prefix_state["kv_cache"],
        adarms_cond=[None, None, adarms_cond],
    )
    velocity = base_model.action_out_proj(suffix_out[:, -base_model.action_horizon :])
    return action_noise - velocity


def _zero_evidence(evidence: jax.Array, repeats: int) -> jax.Array:
    return jnp.repeat(jnp.zeros_like(evidence), repeats, axis=0)


def _ear_loss(
    base_model: Any,
    memory: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    calibrated_teacher = _persistent_coarse_endpoint(
        base_model,
        context["causal_prefix"],
        memory,
        _zero_evidence(context["evidence"], 2),
        jnp.zeros((2,), dtype=jnp.bool_),
    )
    stale_ear, direct_ear = calibrated_teacher[0:1], calibrated_teacher[1:2]
    if mismatched_evidence is None:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], jnp.zeros_like(context["evidence"])],
            axis=0,
        )
        candidate_count = 2
        candidate_active = jnp.asarray([True, False], dtype=jnp.bool_)
    else:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], mismatched_evidence, jnp.zeros_like(context["evidence"])],
            axis=0,
        )
        candidate_count = 3
        candidate_active = jnp.asarray([True, True, False], dtype=jnp.bool_)
    candidate_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
        context["anchor_prefix"],
        context["fresh_prefix"],
        candidate_count,
    )
    candidate_ear = _persistent_coarse_endpoint(
        base_model,
        candidate_prefix,
        memory,
        candidate_evidence,
        candidate_active,
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
        "gate": jnp.tanh(memory.gate.value),
    }


def _final_loss(
    base_model: Any,
    ear_memory: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory,
    final_memory: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    stale_iar = base_model.sample_actions_profile_implicit(context["stale_prefix"])[
        "implicit_action_reason"
    ]
    if stale_iar is None:
        raise ValueError("Persistent evidence memory requires the frozen implicit reasoner.")
    predicted_ear = _persistent_coarse_endpoint(
        base_model,
        context["stale_prefix"],
        ear_memory,
        context["evidence"],
        jnp.ones((1,), dtype=jnp.bool_),
    )
    calibrated_teacher = _persistent_final_endpoint(
        base_model,
        context["causal_prefix"],
        final_memory,
        _zero_evidence(context["evidence"], 2),
        jnp.repeat(predicted_ear, 2, axis=0),
        jnp.repeat(stale_iar, 2, axis=0),
        jnp.zeros((2,), dtype=jnp.bool_),
    )
    base_action, teacher_action = calibrated_teacher[0:1], calibrated_teacher[1:2]
    if mismatched_evidence is None:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], jnp.zeros_like(context["evidence"])],
            axis=0,
        )
        candidate_count = 2
        candidate_active = jnp.asarray([True, False], dtype=jnp.bool_)
    else:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], mismatched_evidence, jnp.zeros_like(context["evidence"])],
            axis=0,
        )
        candidate_count = 3
        candidate_active = jnp.asarray([True, True, False], dtype=jnp.bool_)
    candidate_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
        context["anchor_prefix"],
        context["fresh_prefix"],
        candidate_count,
    )
    candidate_actions = _persistent_final_endpoint(
        base_model,
        candidate_prefix,
        final_memory,
        candidate_evidence,
        jnp.repeat(predicted_ear, candidate_count, axis=0),
        jnp.repeat(stale_iar, candidate_count, axis=0),
        candidate_active,
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
        "gate": jnp.tanh(final_memory.gate.value),
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
        memory = nnx.merge(ear_graphdef, params)

        def loss_fn(candidate: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory):
            return _ear_loss(
                base_model,
                candidate,
                context,
                args,
                mismatched_evidence=mismatched_evidence,
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(memory)
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
        memory = nnx.merge(ear_graphdef, params)
        loss, metrics = _ear_loss(
            base_model,
            memory,
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
        ear_memory = nnx.merge(ear_graphdef, ear_params)
        final_memory = nnx.merge(final_graphdef, params)

        def loss_fn(candidate: dce_persistent_evidence_memory.DCEPersistentEvidenceMemory):
            return _final_loss(
                base_model,
                ear_memory,
                candidate,
                context,
                args,
                mismatched_evidence=mismatched_evidence,
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(final_memory)
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
        ear_memory = nnx.merge(ear_graphdef, ear_params)
        final_memory = nnx.merge(final_graphdef, final_params)
        loss, metrics = _final_loss(
            base_model,
            ear_memory,
            final_memory,
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


def _make_no_evidence_drift_evaluator(
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
        ear_memory = nnx.merge(ear_graphdef, ear_params)
        final_memory = nnx.merge(final_graphdef, final_params)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        comparison = context["comparison"]
        count = comparison["observation"].state.shape[0]
        original_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            comparison,
            comparison["ref_action_noise"],
        )
        masked_ear = _persistent_coarse_endpoint(
            base_model,
            comparison,
            ear_memory,
            _zero_evidence(context["evidence"], count),
            jnp.zeros((count,), dtype=jnp.bool_),
        )
        active_zero_ear = _persistent_coarse_endpoint(
            base_model,
            comparison,
            ear_memory,
            _zero_evidence(context["evidence"], count),
            jnp.ones((count,), dtype=jnp.bool_),
        )
        implicit_reason = base_model.sample_actions_profile_implicit(comparison)[
            "implicit_action_reason"
        ]
        if implicit_reason is None:
            raise ValueError("Persistent evidence memory requires the frozen implicit reasoner.")
        original_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            comparison,
            comparison["expert_action_noise"],
            original_ear,
            implicit_reason,
        )
        masked_action = _persistent_final_endpoint(
            base_model,
            comparison,
            final_memory,
            _zero_evidence(context["evidence"], count),
            original_ear,
            implicit_reason,
            jnp.zeros((count,), dtype=jnp.bool_),
        )
        active_zero_action = _persistent_final_endpoint(
            base_model,
            comparison,
            final_memory,
            _zero_evidence(context["evidence"], count),
            original_ear,
            implicit_reason,
            jnp.ones((count,), dtype=jnp.bool_),
        )
        return {
            "no_evidence_ear_mse_7d": _all_action_mse(masked_ear, original_ear),
            "no_evidence_action_mse_7d": _all_action_mse(masked_action, original_action),
            "active_zero_token_ear_mse_7d": _all_action_mse(
                active_zero_ear,
                original_ear,
            ),
            "active_zero_token_action_mse_7d": _all_action_mse(
                active_zero_action,
                original_action,
            ),
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
        ear_memory = nnx.merge(ear_graphdef, ear_params)
        final_memory = nnx.merge(final_graphdef, final_params)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model,
            scorer,
            selector_runtime,
            batch,
            rng,
            args,
        )
        comparison = context["comparison"]
        comparison_count = comparison["observation"].state.shape[0]
        original_teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            comparison,
            comparison["ref_action_noise"],
        )
        teacher_ear = _persistent_coarse_endpoint(
            base_model,
            comparison,
            ear_memory,
            _zero_evidence(context["evidence"], comparison_count),
            jnp.zeros((comparison_count,), dtype=jnp.bool_),
        )
        teacher_iar = base_model.sample_actions_profile_implicit(comparison)[
            "implicit_action_reason"
        ]
        if teacher_iar is None:
            raise ValueError("Persistent evidence memory requires the frozen implicit reasoner.")
        stale_ear, fresh_ear = teacher_ear[0:1], teacher_ear[2:3]
        stale_iar, fresh_iar = teacher_iar[0:1], teacher_iar[2:3]
        predicted_ear = _persistent_coarse_endpoint(
            base_model,
            context["stale_prefix"],
            ear_memory,
            context["evidence"],
            jnp.ones((1,), dtype=jnp.bool_),
        )
        shuffled_ear = _persistent_coarse_endpoint(
            base_model,
            context["stale_prefix"],
            ear_memory,
            shuffled_evidence,
            jnp.ones((1,), dtype=jnp.bool_),
        )

        plain_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
            context["anchor_prefix"],
            context["fresh_prefix"],
            2,
        )
        plain_actions = _persistent_final_endpoint(
            base_model,
            plain_prefix,
            final_memory,
            _zero_evidence(context["evidence"], 2),
            jnp.concatenate([stale_ear, predicted_ear], axis=0),
            jnp.repeat(stale_iar, 2, axis=0),
            jnp.zeros((2,), dtype=jnp.bool_),
        )
        original_plain_actions = base_model._one_step_action_endpoint(  # noqa: SLF001
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
        adapted_actions = _persistent_final_endpoint(
            base_model,
            adapted_prefix,
            final_memory,
            jnp.repeat(context["evidence"], 2, axis=0),
            jnp.concatenate([stale_ear, predicted_ear], axis=0),
            jnp.repeat(stale_iar, 2, axis=0),
            jnp.ones((2,), dtype=jnp.bool_),
        )
        direct_action, joint_action = adapted_actions[0:1], adapted_actions[1:2]
        shuffled_action = _persistent_final_endpoint(
            base_model,
            context["stale_prefix"],
            final_memory,
            shuffled_evidence,
            shuffled_ear,
            stale_iar,
            jnp.ones((1,), dtype=jnp.bool_),
        )
        fresh_action = _persistent_final_endpoint(
            base_model,
            context["fresh_prefix"],
            final_memory,
            _zero_evidence(context["evidence"], 1),
            fresh_ear,
            fresh_iar,
            jnp.zeros((1,), dtype=jnp.bool_),
        )
        original_fresh_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["fresh_prefix"],
            context["fresh_prefix"]["expert_action_noise"],
            fresh_ear,
            fresh_iar,
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
            "ear_gate": jnp.tanh(ear_memory.gate.value),
            "final_gate": jnp.tanh(final_memory.gate.value),
            "no_evidence_ear_drift_mse_7d": _all_action_mse(
                teacher_ear,
                original_teacher_ear,
            ),
            "no_evidence_action_drift_mse_7d": _all_action_mse(
                jnp.concatenate([plain_actions, fresh_action], axis=0),
                jnp.concatenate([original_plain_actions, original_fresh_action], axis=0),
            ),
        }

    return evaluate


def _aggregate_test(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    result = dce_base._aggregate_test(records, args)  # noqa: SLF001
    result["no_evidence_identity_drift"] = {
        "ear_mse_7d": float(
            np.mean([record["no_evidence_ear_drift_mse_7d"] for record in records])
        ),
        "action_mse_7d": float(
            np.mean([record["no_evidence_action_drift_mse_7d"] for record in records])
        ),
    }
    if not result["gate"]["pass"]:
        result["gate"]["decision"] = "no_go_persistent_multi_layer_evidence_memory"
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
    ear_params_path = output_dir / "final" / "ear_persistent_memory" / "params"
    final_params_path = output_dir / "final" / "final_persistent_memory" / "params"
    targets = (metrics_path, per_pair_path, summary_path, ear_params_path, final_params_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"Persistent-memory output already exists in {output_dir}; pass --overwrite."
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
        raise ValueError(f"Expected persistent-memory split {expected_split}, got {actual_split}.")
    records = dce_base.p3t_trainer._materialize_pairs(  # noqa: SLF001
        pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    selector_runtime = mrr_block_selector.load_mrr_block_selector(args.selector_checkpoint)

    base_model = nnx.merge(base_graphdef, base_state)
    if not base_model.pi05:
        raise ValueError("Persistent-memory oracle currently requires the pi05 AdaRMS suffix.")
    query_dim = int(base_model.action_in_proj.out_features)
    coarse_query_dim = int(base_model.coarse_action_in_proj.out_features)
    if query_dim != coarse_query_dim:
        raise ValueError(f"EAR/final suffix widths differ: {coarse_query_dim} vs {query_dim}.")
    evidence_tokens = dce_base.EVIDENCE_TOKENS_BY_MODE[args.evidence_mode]
    evidence_dim = int(mrr_block_selector.TOKEN_EMBEDDING_DIM) * (
        2 if args.evidence_mode == "selected128_pair" else 1
    )
    memory_config = (
        dce_persistent_evidence_memory.DCEPersistentEvidenceMemoryConfig(
            evidence_dim=evidence_dim,
            evidence_tokens=evidence_tokens,
            expert_dim=query_dim,
            memory_tokens=args.memory_tokens,
            attention_dim=args.attention_dim,
            num_heads=args.attention_heads,
        )
    )
    ear_memory = dce_persistent_evidence_memory.DCEPersistentEvidenceMemory(
        memory_config,
        rngs=nnx.Rngs(args.seed + 303),
    )
    final_memory = dce_persistent_evidence_memory.DCEPersistentEvidenceMemory(
        memory_config,
        rngs=nnx.Rngs(args.seed + 404),
    )
    ear_graphdef, ear_params = nnx.split(ear_memory)
    final_graphdef, final_params = nnx.split(final_memory)
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
                "Exported train persistent evidence %d/%d",
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

    drift_evaluator = _make_no_evidence_drift_evaluator(
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
    initial_memory_drift = {
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
                {"stage": "initial_memory_drift", **initial_memory_drift},
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
                    "Persistent EAR step %d/%d loss=%.6f",
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
                    "Persistent final step %d/%d loss=%.6f",
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
        name="dce_ear_persistent_evidence_memory",
        overwrite=args.overwrite,
    )
    saved_final = dce_base._save_params(  # noqa: SLF001
        final_params,
        final_params_path,
        name="dce_final_persistent_evidence_memory",
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
                    f"Non-finite persistent-memory test output at pair {record_index}."
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
                "ear_gate": float(output["ear_gate"]),
                "final_gate": float(output["final_gate"]),
                "no_evidence_ear_drift_mse_7d": float(
                    output["no_evidence_ear_drift_mse_7d"]
                ),
                "no_evidence_action_drift_mse_7d": float(
                    output["no_evidence_action_drift_mse_7d"]
                ),
            }
            output_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output_file.flush()
            test_records.append(record)
            LOGGER.info(
                "Persistent test pair %d/%d anchor=%d",
                test_position + 1,
                test_indices.size,
                anchor,
            )

    evaluation = _aggregate_test(test_records, args)
    summary = {
        "method": "DCE persistent multi-layer evidence memory oracle",
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
            "mechanism": "persistent learned evidence tokens through all 18 frozen expert layers",
            "injection_site": "prepend memory block to EAR/final suffix before action tokens",
            "memory_tokens": args.memory_tokens,
            "memory_attention_dim": args.attention_dim,
            "memory_attention_heads": args.attention_heads,
            "memory_mask": "active only for matched/mismatch/shuffled rows",
            "memory_positions": "shared prefix-end position",
            "action_positions": "identical to the original suffix positions",
            "ear_memory_parameters": ear_parameter_count,
            "final_memory_parameters": final_parameter_count,
            "expert_dim": query_dim,
            "gate": "scalar ReZero tanh gate initialized exactly zero",
            "base_frozen": True,
            "coarse_action_out_proj_frozen": True,
            "action_out_proj_frozen": True,
            "post_suffix_action_residual": False,
            "default_inference_modified": False,
        },
        "training": {
            "order": "EAR memory first; frozen predicted EAR conditions final-memory training",
            "teacher": "direct-splice response under masked memory slots",
            "mismatched_evidence": {
                "pairing": "deterministic different-episode pairing",
                "cached_train_evidence_pairs": len(train_evidence),
                "margin": args.contrastive_margin,
                "ear_ranking_weight": args.ear_ranking_loss_weight,
                "final_ranking_weight": args.final_ranking_loss_weight,
            },
            "zero_evidence": {
                "contract": "all memory slots masked; exact original frozen suffix",
                "ear_weight": args.ear_zero_loss_weight,
                "final_weight": args.final_zero_loss_weight,
            },
            "initial_memory_drift": initial_memory_drift,
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
        "Persistent-memory oracle complete decision=%s elapsed=%.1fs",
        evaluation["gate"]["decision"],
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
