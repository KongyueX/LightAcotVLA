"""Train the minimal pre-LLM DCE dual-path adapter oracle on Task8.

The frozen base+IR policy supplies an immutable anchor prefix and a privileged
current top-8 visual representation.  ``block16`` evidence pools each selected
4x4 block into one absolute-current token and one current-minus-anchor token;
``selected128`` instead preserves all 128 current pre-Gemma visual tokens in
selector rank order.  Two independent ReZero cross-attention adapters inject
the selected evidence into the action suffix *before* the frozen 18-layer EAR
and final experts.  The original coarse/final output projections stay frozen,
and there is no action residual head.

Training is deliberately staged.  Stage 1 learns only the exact top-8
direct-splice EAR response.  Stage 2 freezes Stage 1 and learns only the final
response between direct-splice and anchor prefixes under the same predicted
EAR and stale IAR.  Full-fresh outputs are used only by the Test22 closure
report.  Zero evidence is supervised to produce zero response, while evidence
from another episode is a contrastive negative rather than a hard stale
target.  Test22 reports stale, plan-only, direct-only, joint, fresh, and
shuffled-evidence interventions.
This script is GPU-only and never modifies the default policy path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, Callable, Sequence

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import dce_evidence_adapter
from openpi.models import mrr_block_selector
from openpi.models.pi0 import make_attn_mask

try:
    import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    import probe_mrr_compiler_bottleneck_oracle as compiler_oracle
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    from scripts import probe_mrr_compiler_bottleneck_oracle as compiler_oracle
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("train_dce_dual_path_adapter_oracle")
METHOD_NAMES = (
    "stale",
    "plan_only",
    "direct_only",
    "joint",
    "fresh",
    "shuffled_evidence",
)
CONTINUOUS_ACTION_DIM = 6
TOP_K = 8
EVIDENCE_TOKENS_BY_MODE = {
    "block16": TOP_K * 2,
    "selected128": TOP_K * mrr_block_selector.TOKENS_PER_BLOCK,
}


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    endpoint_student_params: str
    selector_checkpoint: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    dataset_task_id: int = 6
    temporal_stride: int = 10
    seed: int = 7
    split_seed: int = 7
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    expected_pairs: int = 188
    expected_train_pairs: int = 144
    expected_validation_pairs: int = 22
    expected_test_pairs: int = 22
    ear_steps: int = 500
    final_steps: int = 500
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    log_interval: int = 25
    attention_dim: int = 128
    attention_heads: int = 4
    evidence_mode: str = "block16"
    ear_response_loss_weight: float = 1.0
    ear_zero_loss_weight: float = 1.0
    ear_ranking_loss_weight: float = 1.0
    final_response_loss_weight: float = 1.0
    final_zero_loss_weight: float = 1.0
    final_ranking_loss_weight: float = 1.0
    gripper_event_loss_weight: float = 1.0
    contrastive_margin: float = 0.01
    gripper_sign_margin: float = 0.1
    joint_action_closure_gate: float = 0.75
    plan_ear_closure_gate: float = 0.60
    conditional_direct_gain_gate: float = 0.25
    conditional_plan_gain_gate: float = 0.10
    shuffled_action_drop_gate: float = 0.10
    joint_gripper_accuracy_gate: float = 0.95
    delta_floor: float = 1e-8
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if any(value < 0 for value in (args.dataset_task_id, args.seed, args.split_seed)):
        raise ValueError("Task id and seeds must be non-negative.")
    if args.temporal_stride != 10:
        raise ValueError("DCE adapter oracle requires anchor-to-anchor+10 pairs.")
    if args.batch_size != 1:
        raise ValueError("The minimal online-prefix oracle currently requires --batch-size=1.")
    if args.ear_steps <= 0 or args.final_steps <= 0 or args.log_interval <= 0:
        raise ValueError("Training steps and log interval must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.gradient_clip_norm <= 0.0:
        raise ValueError("Optimizer scales are invalid.")
    if args.attention_dim <= 0 or args.attention_heads <= 0:
        raise ValueError("Attention dimensions must be positive.")
    if args.attention_dim % args.attention_heads:
        raise ValueError("attention_dim must be divisible by attention_heads.")
    if args.evidence_mode not in EVIDENCE_TOKENS_BY_MODE:
        raise ValueError(
            f"evidence_mode must be one of {tuple(EVIDENCE_TOKENS_BY_MODE)}, "
            f"got {args.evidence_mode!r}."
        )
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("Validation/test fractions must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    expected = (
        args.expected_pairs,
        args.expected_train_pairs,
        args.expected_validation_pairs,
        args.expected_test_pairs,
    )
    if any(value <= 0 for value in expected) or sum(expected[1:]) != expected[0]:
        raise ValueError("Expected split sizes must be positive and sum to expected_pairs.")
    weights = (
        args.ear_response_loss_weight,
        args.ear_zero_loss_weight,
        args.ear_ranking_loss_weight,
        args.final_response_loss_weight,
        args.final_zero_loss_weight,
        args.final_ranking_loss_weight,
        args.gripper_event_loss_weight,
    )
    if any(value < 0.0 for value in weights):
        raise ValueError("DCE loss weights must be non-negative.")
    if args.ear_response_loss_weight <= 0.0 or args.final_response_loss_weight <= 0.0:
        raise ValueError("Both matched-response losses must be active.")
    if args.contrastive_margin <= 0.0 or args.gripper_sign_margin <= 0.0:
        raise ValueError("Contrastive and gripper margins must be positive.")
    gates = (
        args.joint_action_closure_gate,
        args.plan_ear_closure_gate,
        args.conditional_direct_gain_gate,
        args.conditional_plan_gain_gate,
        args.shuffled_action_drop_gate,
        args.joint_gripper_accuracy_gate,
    )
    if any(not 0.0 <= value <= 1.0 for value in gates):
        raise ValueError("DCE evaluation gates must lie in [0, 1].")
    if args.delta_floor <= 0.0:
        raise ValueError("delta_floor must be positive.")


def _continuous_mse(predicted: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(
        jnp.square(
            predicted[..., :CONTINUOUS_ACTION_DIM].astype(jnp.float32)
            - target[..., :CONTINUOUS_ACTION_DIM].astype(jnp.float32)
        )
    )


def _gripper_accuracy(predicted: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean((predicted[..., 6] >= 0.0) == (target[..., 6] >= 0.0))


def _event_gripper_margin(
    predicted: jax.Array,
    teacher: jax.Array,
    base: jax.Array,
    *,
    margin: float,
) -> tuple[jax.Array, jax.Array]:
    teacher_positive = teacher[..., 6] >= 0.0
    base_positive = base[..., 6] >= 0.0
    event_mask = teacher_positive != base_positive
    target_sign = jnp.where(teacher_positive, 1.0, -1.0)
    per_token = jax.nn.relu(
        margin - target_sign * predicted[..., 6].astype(jnp.float32)
    )
    event_count = jnp.sum(event_mask.astype(jnp.float32))
    loss = jnp.sum(per_token * event_mask.astype(jnp.float32)) / jnp.maximum(event_count, 1.0)
    return loss, jnp.mean(event_mask.astype(jnp.float32))


def _repeat_stale_prefix(
    anchor_prefix: dict[str, Any],
    fresh_prefix: dict[str, Any],
    repeats: int,
) -> dict[str, Any]:
    kv_cache = (
        jnp.repeat(anchor_prefix["kv_cache"][0], repeats, axis=1),
        jnp.repeat(anchor_prefix["kv_cache"][1], repeats, axis=1),
    )
    return mrr_oracle._repeat_prefix_state(  # noqa: SLF001
        fresh_prefix,
        anchor_prefix["prefix_out"],
        kv_cache,
        repeats,
    )


def _top8_evidence(
    anchor_prefix: dict[str, Any],
    fresh_prefix: dict[str, Any],
    selected_ids: jax.Array,
    *,
    mode: str,
) -> jax.Array:
    """Build evidence in selector top-k order under the requested token contract."""

    anchor_visual = anchor_prefix["prefix_tokens"][:, : mrr_oracle.VISUAL_TOKENS]
    fresh_visual = fresh_prefix["prefix_tokens"][:, : mrr_oracle.VISUAL_TOKENS]
    token_indices = mrr_block_selector.BLOCK_TOKEN_INDICES[selected_ids]

    def gather(tokens: jax.Array, indices: jax.Array) -> jax.Array:
        return tokens[indices]

    fresh_blocks = jax.vmap(gather)(fresh_visual, token_indices)
    if mode == "selected128":
        return fresh_blocks.reshape(
            fresh_visual.shape[0],
            EVIDENCE_TOKENS_BY_MODE[mode],
            fresh_visual.shape[-1],
        ).astype(fresh_visual.dtype)

    if mode != "block16":
        raise ValueError(f"Unsupported evidence mode {mode!r}.")
    anchor_blocks = jax.vmap(gather)(anchor_visual, token_indices)
    current_token = jnp.mean(fresh_blocks.astype(jnp.float32), axis=2)
    delta_token = jnp.mean(
        fresh_blocks.astype(jnp.float32) - anchor_blocks.astype(jnp.float32),
        axis=2,
    )
    evidence = jnp.stack([current_token, delta_token], axis=2).reshape(
        fresh_visual.shape[0],
        EVIDENCE_TOKENS_BY_MODE[mode],
        fresh_visual.shape[-1],
    )
    return evidence.astype(fresh_visual.dtype)


def _pair_context(
    base_model: Any,
    scorer: mrr_block_selector.MRRBlockSelectorScorer,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    batch: dict[str, Any],
    rng: jax.Array,
    args: Args,
) -> dict[str, Any]:
    anchor_prefix, fresh_prefix, _, _, block_logits = compiler_oracle._learned_direct_context(  # noqa: SLF001
        base_model,
        scorer,
        batch,
        rng,
        selector_runtime.feature_mean,
        selector_runtime.feature_std,
        projection_seed=selector_runtime.config.feature_projection_seed,
        projection_rank=selector_runtime.config.projection_rank,
    )
    _, selected_ids = jax.lax.top_k(block_logits, TOP_K)
    selected_mask = mrr_oracle._selected_visual_mask(selected_ids[0])  # noqa: SLF001
    direct_kv = mrr_oracle._composite_cache(  # noqa: SLF001
        anchor_prefix["kv_cache"],
        fresh_prefix["kv_cache"],
        selected_mask[None],
        jnp.zeros((1,), dtype=jnp.bool_),
    )
    comparison = compiler_oracle._prefix_comparison(  # noqa: SLF001
        anchor_prefix,
        fresh_prefix,
        direct_kv,
    )
    causal_kv = (
        jnp.concatenate([anchor_prefix["kv_cache"][0], direct_kv[0]], axis=1),
        jnp.concatenate([anchor_prefix["kv_cache"][1], direct_kv[1]], axis=1),
    )
    causal_prefix = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
        fresh_prefix,
        anchor_prefix["prefix_out"],
        causal_kv,
        2,
    )
    return {
        "anchor_prefix": anchor_prefix,
        "fresh_prefix": fresh_prefix,
        "stale_prefix": _repeat_stale_prefix(anchor_prefix, fresh_prefix, 1),
        "causal_prefix": causal_prefix,
        "comparison": comparison,
        "evidence": _top8_evidence(
            anchor_prefix,
            fresh_prefix,
            selected_ids,
            mode=args.evidence_mode,
        ),
        "selected_ids": selected_ids[0],
        "block_logits": block_logits[0],
    }


def _adapted_coarse_endpoint(
    base_model: Any,
    prefix_state: dict[str, Any],
    adapter: dce_evidence_adapter.DCEEvidenceAdapter,
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
    action_tokens = suffix_tokens[:, -base_model.coarse_action_horizon :]
    adapted_tokens = adapter(action_tokens, evidence).astype(suffix_tokens.dtype)
    suffix_tokens = suffix_tokens.at[:, -base_model.coarse_action_horizon :].set(adapted_tokens)
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
    )
    velocity = base_model.coarse_action_out_proj(
        suffix_out[:, -base_model.coarse_action_horizon :]
    )
    return coarse_noise - velocity


def _adapted_final_endpoint(
    base_model: Any,
    prefix_state: dict[str, Any],
    adapter: dce_evidence_adapter.DCEEvidenceAdapter,
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
    action_tokens = suffix_tokens[:, -base_model.action_horizon :]
    adapted_tokens = adapter(action_tokens, evidence).astype(suffix_tokens.dtype)
    suffix_tokens = suffix_tokens.at[:, -base_model.action_horizon :].set(adapted_tokens)
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
    )
    velocity = base_model.action_out_proj(suffix_out[:, -base_model.action_horizon :])
    return action_noise - velocity


def _ear_loss(
    base_model: Any,
    adapter: dce_evidence_adapter.DCEEvidenceAdapter,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    causal_prefix = context["causal_prefix"]
    teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
        causal_prefix,
        causal_prefix["ref_action_noise"],
    )
    stale_ear, direct_ear = teacher_ear[0:1], teacher_ear[1:2]
    zero_evidence = jnp.zeros_like(context["evidence"])
    if mismatched_evidence is None:
        candidate_evidence = jnp.concatenate([context["evidence"], zero_evidence], axis=0)
        candidate_prefix = _repeat_stale_prefix(
            context["anchor_prefix"],
            context["fresh_prefix"],
            2,
        )
    else:
        candidate_evidence = jnp.concatenate(
            [context["evidence"], mismatched_evidence, zero_evidence],
            axis=0,
        )
        candidate_prefix = _repeat_stale_prefix(
            context["anchor_prefix"],
            context["fresh_prefix"],
            3,
        )
    candidate_ear = _adapted_coarse_endpoint(
        base_model,
        candidate_prefix,
        adapter,
        candidate_evidence,
    )
    predicted_ear = candidate_ear[0:1]
    zero_ear = candidate_ear[-1:]
    teacher_response = direct_ear - stale_ear
    matched_response = predicted_ear - stale_ear
    response_loss = _continuous_mse(matched_response, teacher_response)
    zero_loss = _continuous_mse(zero_ear - stale_ear, jnp.zeros_like(stale_ear))
    if mismatched_evidence is None:
        mismatch_distance = jnp.zeros((), dtype=jnp.float32)
        ranking_loss = jnp.zeros((), dtype=jnp.float32)
    else:
        mismatch_response = candidate_ear[1:2] - stale_ear
        mismatch_distance = _continuous_mse(mismatch_response, teacher_response)
        ranking_loss = jax.nn.relu(
            args.contrastive_margin + response_loss - mismatch_distance
        )
    gripper_loss, gripper_event_fraction = _event_gripper_margin(
        predicted_ear,
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
        "gate": jnp.tanh(adapter.gate.value),
    }


def _final_loss(
    base_model: Any,
    ear_adapter: dce_evidence_adapter.DCEEvidenceAdapter,
    final_adapter: dce_evidence_adapter.DCEEvidenceAdapter,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    stale_iar = base_model.sample_actions_profile_implicit(context["stale_prefix"])[
        "implicit_action_reason"
    ]
    if stale_iar is None:
        raise ValueError("DCE adapter requires the frozen implicit action reasoner.")
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
    candidate_prefix = _repeat_stale_prefix(
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
    response_loss = _continuous_mse(matched_response, teacher_response)
    zero_loss = _continuous_mse(zero_action - base_action, jnp.zeros_like(base_action))
    if mismatched_evidence is None:
        mismatch_distance = jnp.zeros((), dtype=jnp.float32)
        ranking_loss = jnp.zeros((), dtype=jnp.float32)
    else:
        mismatch_response = candidate_actions[1:2] - base_action
        mismatch_distance = _continuous_mse(mismatch_response, teacher_response)
        ranking_loss = jax.nn.relu(
            args.contrastive_margin + response_loss - mismatch_distance
        )
    gripper_loss, gripper_event_fraction = _event_gripper_margin(
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
        "matched_teacher_gripper_accuracy": _gripper_accuracy(matched_action, teacher_action),
        "gate": jnp.tanh(final_adapter.gate.value),
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
        context = _pair_context(base_model, scorer, selector_runtime, batch, rng, args)
        adapter = nnx.merge(ear_graphdef, params)

        def loss_fn(candidate: dce_evidence_adapter.DCEEvidenceAdapter):
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
        context = _pair_context(base_model, scorer, selector_runtime, batch, rng, args)
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
        context = _pair_context(base_model, scorer, selector_runtime, batch, rng, args)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, params)

        def loss_fn(candidate: dce_evidence_adapter.DCEEvidenceAdapter):
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
        context = _pair_context(base_model, scorer, selector_runtime, batch, rng, args)
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
        context = _pair_context(base_model, scorer, selector_runtime, batch, rng, args)
        return {
            "evidence": context["evidence"],
            "selected_ids": context["selected_ids"],
        }

    return export


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
        context = _pair_context(base_model, scorer, selector_runtime, batch, rng, args)
        comparison = context["comparison"]

        teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            comparison,
            comparison["ref_action_noise"],
        )
        teacher_iar = base_model.sample_actions_profile_implicit(comparison)["implicit_action_reason"]
        if teacher_iar is None:
            raise ValueError("DCE adapter requires the frozen implicit action reasoner.")
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

        plain_prefix = _repeat_stale_prefix(
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

        adapted_prefix = _repeat_stale_prefix(
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
        metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
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
            "ear_gate": jnp.tanh(ear_adapter.gate.value),
            "final_gate": jnp.tanh(final_adapter.gate.value),
        }

    return evaluate


def _metric_mean(records: Sequence[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        name: float(np.mean([record[name] for record in records]))
        for name in records[0]
    }


def _closure(candidate: np.ndarray, stale: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for index, name in enumerate(mrr_oracle.METRIC_NAMES[:3]):
        denominator = float(stale[index])
        result[name] = None if denominator <= 1e-12 else float(1.0 - candidate[index] / denominator)
    return result


def _aggregate_test(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    metrics = np.stack([np.asarray(record["metric_arrays"], dtype=np.float64) for record in records])
    means = np.mean(metrics, axis=0)
    stale_mean = means[0]
    methods = {
        name: {
            "metrics": mrr_oracle._metric_dict(means[index]),  # noqa: SLF001
            "global_gap_closure_vs_stale": _closure(means[index], stale_mean),
        }
        for index, name in enumerate(METHOD_NAMES)
    }
    plan_closure = methods["plan_only"]["global_gap_closure_vs_stale"]["action_mse_7d"]
    direct_closure = methods["direct_only"]["global_gap_closure_vs_stale"]["action_mse_7d"]
    joint_closure = methods["joint"]["global_gap_closure_vs_stale"]["action_mse_7d"]
    shuffled_closure = methods["shuffled_evidence"]["global_gap_closure_vs_stale"]["action_mse_7d"]
    decomposition = {
        "mediated_gain_stale_gap_units": plan_closure,
        "direct_gain_stale_gap_units": direct_closure,
        "interaction_gain_stale_gap_units": joint_closure - plan_closure - direct_closure,
        "joint_gain_stale_gap_units": joint_closure,
        "conditional_direct_gain_beyond_plan": joint_closure - plan_closure,
        "conditional_plan_gain_beyond_direct": joint_closure - direct_closure,
        "shuffled_evidence_action_drop": joint_closure - shuffled_closure,
    }
    checks = {
        "joint_action_gap_closure": joint_closure >= args.joint_action_closure_gate,
        "plan_ear_gap_closure": methods["plan_only"]["global_gap_closure_vs_stale"]["ear_mse_7d"]
        >= args.plan_ear_closure_gate,
        "conditional_direct_gain_beyond_plan": decomposition["conditional_direct_gain_beyond_plan"]
        >= args.conditional_direct_gain_gate,
        "conditional_plan_gain_beyond_direct": decomposition["conditional_plan_gain_beyond_direct"]
        >= args.conditional_plan_gain_gate,
        "shuffled_evidence_action_drop": decomposition["shuffled_evidence_action_drop"]
        >= args.shuffled_action_drop_gate,
        "joint_gripper_sign_accuracy": methods["joint"]["metrics"]["gripper_sign_accuracy"]
        >= args.joint_gripper_accuracy_gate,
    }
    return {
        "num_pairs": len(records),
        "methods": methods,
        "counterfactual_decomposition": decomposition,
        "gate": {
            "thresholds": {
                "joint_action_gap_closure": args.joint_action_closure_gate,
                "plan_ear_gap_closure": args.plan_ear_closure_gate,
                "conditional_direct_gain_beyond_plan": args.conditional_direct_gain_gate,
                "conditional_plan_gain_beyond_direct": args.conditional_plan_gain_gate,
                "shuffled_evidence_action_drop": args.shuffled_action_drop_gate,
                "joint_gripper_sign_accuracy": args.joint_gripper_accuracy_gate,
            },
            "checks": checks,
            "pass": all(checks.values()),
            "decision": (
                "continue_to_closed_loop_adapter_integration"
                if all(checks.values())
                else "no_go_learned_pre_llm_dual_path"
            ),
        },
    }


def _save_params(
    params: nnx.State,
    target: pathlib.Path,
    *,
    name: str,
    overwrite: bool,
) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {name: params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target.resolve(), item, force=overwrite)
    return target.resolve()


def _deterministic_mismatch_map(
    indices: np.ndarray,
    pairs: p3t_trainer.PairIndices,
) -> dict[int, int]:
    """Pair every training example with a stable example from another episode."""

    ordered = [int(value) for value in np.asarray(indices, dtype=np.int64)]
    if len(ordered) < 2:
        raise ValueError("Mismatched-evidence training requires at least two training pairs.")
    result: dict[int, int] = {}
    initial_shift = max(1, len(ordered) // 2)
    for position, record_index in enumerate(ordered):
        source_episode = int(pairs.episode_ids[record_index])
        for extra_shift in range(len(ordered)):
            candidate = ordered[(position + initial_shift + extra_shift) % len(ordered)]
            if candidate != record_index and int(pairs.episode_ids[candidate]) != source_episode:
                result[record_index] = candidate
                break
        else:
            raise ValueError(
                f"No different-episode mismatch exists for training pair {record_index}."
            )
    return result


def _validation_metrics(
    evaluator: Callable[..., dict[str, jax.Array]],
    indices: np.ndarray,
    pairs: p3t_trainer.PairIndices,
    records: Sequence[p3t_trainer.MaterializedPair],
    arguments: tuple[Any, ...],
    *,
    seed: int,
    mismatch_map: dict[int, int],
    evidence_by_index: dict[int, np.ndarray],
) -> dict[str, float]:
    values: list[dict[str, float]] = []
    for record_index in indices:
        batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
        anchor = int(pairs.anchor_indices[record_index])
        mismatch_index = mismatch_map[int(record_index)]
        output = jax.device_get(
            evaluator(
                *arguments,
                batch,
                jax.random.fold_in(jax.random.key(seed), anchor),
                jnp.asarray(evidence_by_index[mismatch_index]),
            )
        )
        values.append({name: float(value) for name, value in output.items()})
    return _metric_mean(values)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    per_pair_path = output_dir / "test_pairs.jsonl"
    summary_path = output_dir / "summary.json"
    ear_params_path = output_dir / "final" / "ear_adapter" / "params"
    final_params_path = output_dir / "final" / "final_adapter" / "params"
    targets = (metrics_path, per_pair_path, summary_path, ear_params_path, final_params_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"DCE output already exists in {output_dir}; pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    trainer_args = p3t_trainer.Args(
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
        p3t_trainer._load_model_and_dataset(trainer_args)  # noqa: SLF001
    )
    pairs = p3t_trainer._select_pairs(  # noqa: SLF001
        arrays,
        raw_dataset,
        task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        maximum_pairs=200,
        seed=args.seed,
    )
    train_indices, validation_indices, test_indices = p3t_trainer._split_pairs(  # noqa: SLF001
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
        raise ValueError(f"Expected DCE split {expected_split}, got {actual_split}.")
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
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
    evidence_dim = int(mrr_block_selector.TOKEN_EMBEDDING_DIM)
    evidence_tokens = EVIDENCE_TOKENS_BY_MODE[args.evidence_mode]
    if query_dim != coarse_query_dim:
        raise ValueError(f"EAR/final suffix widths differ: {coarse_query_dim} vs {query_dim}.")
    adapter_config = dce_evidence_adapter.DCEEvidenceAdapterConfig(
        query_dim=query_dim,
        evidence_dim=evidence_dim,
        evidence_tokens=evidence_tokens,
        attention_dim=args.attention_dim,
        num_heads=args.attention_heads,
    )
    ear_adapter = dce_evidence_adapter.DCEEvidenceAdapter(
        adapter_config,
        rngs=nnx.Rngs(args.seed + 101),
    )
    final_adapter = dce_evidence_adapter.DCEEvidenceAdapter(
        adapter_config,
        rngs=nnx.Rngs(args.seed + 202),
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
    mismatch_map = _deterministic_mismatch_map(train_indices, pairs)
    validation_mismatch_map = _deterministic_mismatch_map(validation_indices, pairs)
    train_evidence: dict[int, np.ndarray] = {}
    for position, record_index_value in enumerate(train_indices):
        record_index = int(record_index_value)
        anchor = int(pairs.anchor_indices[record_index])
        batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
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
            LOGGER.info("Exported train mismatch evidence %d/%d", position + 1, train_indices.size)

    validation_evidence: dict[int, np.ndarray] = {}
    for record_index_value in validation_indices:
        record_index = int(record_index_value)
        anchor = int(pairs.anchor_indices[record_index])
        batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
        output = jax.device_get(
            evidence_exporter(
                base_state,
                selector_runtime.scorer_state,
                batch,
                jax.random.fold_in(jax.random.key(args.seed), anchor),
            )
        )
        validation_evidence[record_index] = np.asarray(output["evidence"], dtype=np.float32)

    sampling_rng = np.random.default_rng(args.seed)
    mode = "w" if args.overwrite else "a"
    with metrics_path.open(mode, encoding="utf-8") as metrics_file:
        for step in range(1, args.ear_steps + 1):
            record_index = int(sampling_rng.choice(train_indices))
            mismatch_index = mismatch_map[record_index]
            anchor = int(pairs.anchor_indices[record_index])
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
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
                LOGGER.info("DCE EAR step %d/%d loss=%.6f", step, args.ear_steps, record["loss"])

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
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
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
                LOGGER.info("DCE final step %d/%d loss=%.6f", step, args.final_steps, record["loss"])

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

    saved_ear = _save_params(
        ear_params,
        ear_params_path,
        name="dce_ear_evidence_adapter",
        overwrite=args.overwrite,
    )
    saved_final = _save_params(
        final_params,
        final_params_path,
        name="dce_final_evidence_adapter",
        overwrite=args.overwrite,
    )

    test_evidence = []
    for record_index in test_indices:
        anchor = int(pairs.anchor_indices[record_index])
        batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
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
        for test_position, record_index in enumerate(test_indices):
            anchor = int(pairs.anchor_indices[record_index])
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
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
                raise FloatingPointError(f"Non-finite DCE test output at pair {record_index}.")
            record = {
                "test_position": test_position,
                "pair_index": int(record_index),
                "anchor_index": anchor,
                "target_index": int(pairs.target_indices[record_index]),
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [int(value) for value in np.asarray(output["selected_ids"])],
                "metric_arrays": metric_arrays.tolist(),
                "methods": {
                    name: {
                        "metrics": mrr_oracle._metric_dict(metric_arrays[index]),  # noqa: SLF001
                        "gap_closure_vs_stale": _closure(metric_arrays[index], metric_arrays[0]),
                    }
                    for index, name in enumerate(METHOD_NAMES)
                },
                "ear_gate": float(output["ear_gate"]),
                "final_gate": float(output["final_gate"]),
            }
            output_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output_file.flush()
            test_records.append(record)
            LOGGER.info("DCE test pair %d/%d anchor=%d", test_position + 1, test_indices.size, anchor)

    evaluation = _aggregate_test(test_records, args)
    summary = {
        "method": "DCE/CDVA-ACoT learned pre-LLM dual-path adapter oracle",
        "status": "offline_test22_oracle_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
            "test_episodes": sorted(int(value) for value in np.unique(pairs.episode_ids[test_indices])),
        },
        "evidence": {
            "mode": args.evidence_mode,
            "selector": "learned MRR logits",
            "top_k_blocks": TOP_K,
            "source_visual_tokens": TOP_K * mrr_oracle.TOKENS_PER_BLOCK,
            "evidence_tokens": evidence_tokens,
            "token_contract": (
                "one absolute-current mean plus one current-minus-anchor mean per block"
                if args.evidence_mode == "block16"
                else "all current pre-Gemma visual tokens, grouped in selector top-8 rank order"
            ),
            "privileged_encoder": "full current SigLIP/pre-Gemma visual representation",
        },
        "architecture": {
            "injection_site": "embed_suffix output before frozen 18-layer PaliGemma expert",
            "ear_adapter_parameters": ear_parameter_count,
            "final_adapter_parameters": final_parameter_count,
            "query_dim": query_dim,
            "evidence_dim": evidence_dim,
            "attention_dim": args.attention_dim,
            "attention_heads": args.attention_heads,
            "gate": "scalar ReZero tanh gate initialized exactly zero",
            "base_frozen": True,
            "coarse_action_out_proj_frozen": True,
            "action_out_proj_frozen": True,
            "direct_action_residual_head": False,
        },
        "training": {
            "order": "EAR adapter first; frozen predicted EAR conditions final adapter training",
            "mismatched_evidence": {
                "pairing": "deterministic cyclic pairing with a different training episode",
                "cached_train_evidence_pairs": len(train_evidence),
                "objective": "matched response must be closer to the causal teacher than mismatch by margin",
                "margin": args.contrastive_margin,
                "ear_ranking_weight": args.ear_ranking_loss_weight,
                "final_ranking_weight": args.final_ranking_loss_weight,
            },
            "zero_evidence": {
                "ear_target": "zero response relative to stale EAR",
                "final_target": "zero response relative to anchor-prefix predicted-EAR base action",
                "ear_weight": args.ear_zero_loss_weight,
                "final_weight": args.final_zero_loss_weight,
            },
            "gripper_event_margin": {
                "weight": args.gripper_event_loss_weight,
                "margin": args.gripper_sign_margin,
                "mask": "teacher/base gripper sign differs",
            },
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
            "default_inference_modified": False,
            "runtime_integration": False,
            "deployable_speed_claim": False,
            "closed_loop_success_claim": False,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
