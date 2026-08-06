"""Train three-level Evidence ACoT: IAR, EAR, then final action evidence.

Stage 1 updates frozen-anchor IAR tokens with selected128_pair current evidence
and learns the top-8 direct-splice IAR response.  Stage 2 trains the shared
four-layer EAR evidence hook.  Stage 3 freezes both predicted bottlenecks and
trains the shared four-layer final-action hook.  Final base and direct teachers
receive exactly the same predicted IAR and EAR, isolating the direct evidence
path.  Every stage uses different-episode mismatch ranking and exact zero-
evidence response supervision.  Full-fresh outputs remain Test22-only.

No action residual is introduced and the default inference path stays off.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import dce_iar_evidence_adapter
from openpi.models import dce_multilayer_evidence_adapter
from openpi.models import mrr_block_selector

try:
    import train_dce_multilayer_evidence_oracle as ml_base
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_dce_multilayer_evidence_oracle as ml_base


dce_base = ml_base.dce_base
LOGGER = logging.getLogger("train_dce_evidence_acot_tri_path_oracle")
METHOD_NAMES = ml_base.METHOD_NAMES
IAR_TOKENS = 18
IAR_DIM = 1024


@dataclasses.dataclass(frozen=True)
class Args(ml_base.Args):
    iar_steps: int = 500
    iar_response_loss_weight: float = 1.0
    iar_zero_loss_weight: float = 1.0
    iar_ranking_loss_weight: float = 1.0


def _validate_args(args: Args) -> None:
    ml_base._validate_args(args)  # noqa: SLF001
    if args.iar_steps <= 0:
        raise ValueError("iar_steps must be positive.")
    weights = (
        args.iar_response_loss_weight,
        args.iar_zero_loss_weight,
        args.iar_ranking_loss_weight,
    )
    if any(value < 0.0 for value in weights) or args.iar_response_loss_weight <= 0.0:
        raise ValueError("IAR loss weights are invalid.")


def _latent_mse(predicted: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(
        jnp.square(predicted.astype(jnp.float32) - target.astype(jnp.float32))
    )


def _frozen_iar(base_model: Any, prefix_state: dict[str, Any]) -> jax.Array:
    iar = base_model.sample_actions_profile_implicit(prefix_state)["implicit_action_reason"]
    if iar is None:
        raise ValueError("Tri-path Evidence ACoT requires the frozen implicit reasoner.")
    if iar.shape[1:] != (IAR_TOKENS, IAR_DIM):
        raise ValueError(f"Expected IAR [B,{IAR_TOKENS},{IAR_DIM}], got {iar.shape}.")
    return iar


def _iar_loss(
    base_model: Any,
    adapter: dce_iar_evidence_adapter.DCEIAREvidenceAdapter,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    causal_iar = _frozen_iar(base_model, context["causal_prefix"])
    stale_iar, direct_iar = causal_iar[0:1], causal_iar[1:2]
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
    candidate_iar = adapter(
        jnp.repeat(stale_iar, candidate_count, axis=0),
        candidate_evidence,
    )
    predicted_iar = candidate_iar[0:1]
    zero_iar = candidate_iar[-1:]
    teacher_response = direct_iar - stale_iar
    matched_response = predicted_iar - stale_iar
    response_loss = _latent_mse(matched_response, teacher_response)
    zero_loss = _latent_mse(zero_iar - stale_iar, jnp.zeros_like(stale_iar))
    if mismatched_evidence is None:
        mismatch_distance = jnp.zeros((), dtype=jnp.float32)
        ranking_loss = jnp.zeros((), dtype=jnp.float32)
    else:
        mismatch_response = candidate_iar[1:2] - stale_iar
        mismatch_distance = _latent_mse(mismatch_response, teacher_response)
        ranking_loss = jax.nn.relu(
            args.contrastive_margin + response_loss - mismatch_distance
        )
    loss = (
        args.iar_response_loss_weight * response_loss
        + args.iar_zero_loss_weight * zero_loss
        + args.iar_ranking_loss_weight * ranking_loss
    )
    gates = jnp.tanh(adapter.gates.value)
    return loss, {
        "matched_response_mse": response_loss,
        "mismatch_distance_to_teacher": mismatch_distance,
        "zero_response_mse": zero_loss,
        "contrastive_ranking_loss": ranking_loss,
        "gate_mean_abs": jnp.mean(jnp.abs(gates)),
        "gate_min": jnp.min(gates),
        "gate_max": jnp.max(gates),
    }


def _predicted_iar(
    base_model: Any,
    adapter: dce_iar_evidence_adapter.DCEIAREvidenceAdapter,
    context: dict[str, Any],
    evidence: jax.Array,
) -> jax.Array:
    return adapter(_frozen_iar(base_model, context["stale_prefix"]), evidence)


def _final_loss(
    base_model: Any,
    iar_adapter: dce_iar_evidence_adapter.DCEIAREvidenceAdapter,
    ear_adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    final_adapter: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter,
    context: dict[str, Any],
    args: Args,
    mismatched_evidence: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    predicted_iar = _predicted_iar(
        base_model,
        iar_adapter,
        context,
        context["evidence"],
    )
    predicted_ear = ml_base._adapted_coarse_endpoint(  # noqa: SLF001
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
        jnp.repeat(predicted_iar, 2, axis=0),
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
    candidate_actions = ml_base._adapted_final_endpoint(  # noqa: SLF001
        base_model,
        candidate_prefix,
        final_adapter,
        candidate_evidence,
        jnp.repeat(predicted_ear, candidate_count, axis=0),
        jnp.repeat(predicted_iar, candidate_count, axis=0),
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
        **ml_base._gate_metrics(final_adapter),  # noqa: SLF001
    }


def _make_steps(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    iar_graphdef: Any,
    ear_graphdef: Any,
    final_graphdef: Any,
    optimizer: optax.GradientTransformation,
    args: Args,
) -> tuple[Callable[..., Any], ...]:
    @jax.jit
    def iar_train_step(
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
            base_model, scorer, selector_runtime, batch, rng, args
        )
        adapter = nnx.merge(iar_graphdef, params)

        def loss_fn(candidate: dce_iar_evidence_adapter.DCEIAREvidenceAdapter):
            return _iar_loss(
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
    def iar_eval_step(
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
            base_model, scorer, selector_runtime, batch, rng, args
        )
        adapter = nnx.merge(iar_graphdef, params)
        loss, metrics = _iar_loss(
            base_model,
            adapter,
            context,
            args,
            mismatched_evidence=mismatched_evidence,
        )
        return {**metrics, "loss": loss}

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
            base_model, scorer, selector_runtime, batch, rng, args
        )
        adapter = nnx.merge(ear_graphdef, params)

        def loss_fn(candidate: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter):
            return ml_base._ear_loss(  # noqa: SLF001
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
            base_model, scorer, selector_runtime, batch, rng, args
        )
        adapter = nnx.merge(ear_graphdef, params)
        loss, metrics = ml_base._ear_loss(  # noqa: SLF001
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
        iar_params: nnx.State,
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
            base_model, scorer, selector_runtime, batch, rng, args
        )
        iar_adapter = nnx.merge(iar_graphdef, iar_params)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, params)

        def loss_fn(candidate: dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter):
            return _final_loss(
                base_model,
                iar_adapter,
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
        iar_params: nnx.State,
        ear_params: nnx.State,
        final_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        mismatched_evidence: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model, scorer, selector_runtime, batch, rng, args
        )
        iar_adapter = nnx.merge(iar_graphdef, iar_params)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, final_params)
        loss, metrics = _final_loss(
            base_model,
            iar_adapter,
            ear_adapter,
            final_adapter,
            context,
            args,
            mismatched_evidence=mismatched_evidence,
        )
        return {**metrics, "loss": loss}

    return (
        iar_train_step,
        iar_eval_step,
        ear_train_step,
        ear_eval_step,
        final_train_step,
        final_eval_step,
    )


def _make_test_evaluator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    iar_graphdef: Any,
    ear_graphdef: Any,
    final_graphdef: Any,
    args: Args,
):
    @jax.jit
    def evaluate(
        base_state: nnx.State,
        scorer_state: nnx.State,
        iar_params: nnx.State,
        ear_params: nnx.State,
        final_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        shuffled_evidence: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        iar_adapter = nnx.merge(iar_graphdef, iar_params)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, final_params)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model, scorer, selector_runtime, batch, rng, args
        )
        comparison = context["comparison"]
        teacher_iar = _frozen_iar(base_model, comparison)
        teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            comparison,
            comparison["ref_action_noise"],
        )
        stale_iar, direct_iar, fresh_iar = (
            teacher_iar[0:1],
            teacher_iar[1:2],
            teacher_iar[2:3],
        )
        stale_ear, fresh_ear = teacher_ear[0:1], teacher_ear[2:3]
        predicted_iar = iar_adapter(stale_iar, context["evidence"])
        shuffled_iar = iar_adapter(stale_iar, shuffled_evidence)
        predicted_ear = ml_base._adapted_coarse_endpoint(  # noqa: SLF001
            base_model,
            context["stale_prefix"],
            ear_adapter,
            context["evidence"],
        )
        shuffled_ear = ml_base._adapted_coarse_endpoint(  # noqa: SLF001
            base_model,
            context["stale_prefix"],
            ear_adapter,
            shuffled_evidence,
        )

        plain_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
            context["anchor_prefix"], context["fresh_prefix"], 2
        )
        plain_actions = base_model._one_step_action_endpoint(  # noqa: SLF001
            plain_prefix,
            plain_prefix["expert_action_noise"],
            jnp.concatenate([stale_ear, predicted_ear], axis=0),
            jnp.concatenate([stale_iar, predicted_iar], axis=0),
        )
        stale_action, plan_action = plain_actions[0:1], plain_actions[1:2]

        adapted_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
            context["anchor_prefix"], context["fresh_prefix"], 2
        )
        adapted_actions = ml_base._adapted_final_endpoint(  # noqa: SLF001
            base_model,
            adapted_prefix,
            final_adapter,
            jnp.repeat(context["evidence"], 2, axis=0),
            jnp.concatenate([stale_ear, predicted_ear], axis=0),
            jnp.concatenate([stale_iar, predicted_iar], axis=0),
        )
        direct_action, joint_action = adapted_actions[0:1], adapted_actions[1:2]
        shuffled_action = ml_base._adapted_final_endpoint(  # noqa: SLF001
            base_model,
            context["stale_prefix"],
            final_adapter,
            shuffled_evidence,
            shuffled_ear,
            shuffled_iar,
        )
        fresh_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["fresh_prefix"],
            context["fresh_prefix"]["expert_action_noise"],
            fresh_ear,
            fresh_iar,
        )

        zero_evidence = jnp.zeros_like(context["evidence"])
        zero_iar = iar_adapter(stale_iar, zero_evidence)
        zero_ear = ml_base._adapted_coarse_endpoint(  # noqa: SLF001
            base_model,
            context["stale_prefix"],
            ear_adapter,
            zero_evidence,
        )
        zero_action = ml_base._adapted_final_endpoint(  # noqa: SLF001
            base_model,
            context["stale_prefix"],
            final_adapter,
            zero_evidence,
            predicted_ear,
            predicted_iar,
        )
        zero_action_target = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["stale_prefix"],
            context["stale_prefix"]["expert_action_noise"],
            predicted_ear,
            predicted_iar,
        )

        variant_iar = jnp.concatenate(
            [stale_iar, predicted_iar, stale_iar, predicted_iar, fresh_iar, shuffled_iar],
            axis=0,
        )
        variant_ear = jnp.concatenate(
            [stale_ear, predicted_ear, stale_ear, predicted_ear, fresh_ear, shuffled_ear],
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
            "iar_gates": jnp.tanh(iar_adapter.gates.value),
            "ear_gates": jnp.tanh(ear_adapter.layer_gates.value),
            "final_gates": jnp.tanh(final_adapter.layer_gates.value),
            "predicted_direct_iar_mse": _latent_mse(predicted_iar, direct_iar),
            "shuffled_direct_iar_mse": _latent_mse(shuffled_iar, direct_iar),
            "zero_evidence_iar_drift_mse": _latent_mse(zero_iar, stale_iar),
            "zero_evidence_ear_drift_mse_7d": ml_base._all_action_mse(  # noqa: SLF001
                zero_ear,
                stale_ear,
            ),
            "zero_evidence_action_drift_mse_7d": ml_base._all_action_mse(  # noqa: SLF001
                zero_action,
                zero_action_target,
            ),
        }

    return evaluate


def _aggregate_test(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    result = dce_base._aggregate_test(records, args)  # noqa: SLF001
    result["iar_evidence_diagnostics"] = {
        "predicted_direct_iar_mse": float(
            np.mean([record["predicted_direct_iar_mse"] for record in records])
        ),
        "shuffled_direct_iar_mse": float(
            np.mean([record["shuffled_direct_iar_mse"] for record in records])
        ),
    }
    result["zero_evidence_identity_drift"] = {
        "iar_mse": float(
            np.mean([record["zero_evidence_iar_drift_mse"] for record in records])
        ),
        "ear_mse_7d": float(
            np.mean([record["zero_evidence_ear_drift_mse_7d"] for record in records])
        ),
        "action_mse_7d": float(
            np.mean([record["zero_evidence_action_drift_mse_7d"] for record in records])
        ),
    }
    if not result["gate"]["pass"]:
        result["gate"]["decision"] = "no_go_three_level_evidence_acot"
    return result


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = dce_base.p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    per_pair_path = output_dir / "test_pairs.jsonl"
    summary_path = output_dir / "summary.json"
    iar_path = output_dir / "final" / "iar_adapter" / "params"
    ear_path = output_dir / "final" / "ear_adapter" / "params"
    final_path = output_dir / "final" / "final_adapter" / "params"
    targets = (metrics_path, per_pair_path, summary_path, iar_path, ear_path, final_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"Tri-path output exists in {output_dir}; pass --overwrite.")
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
        raise ValueError(f"Expected tri-path split {expected_split}, got {actual_split}.")
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
    if query_dim != IAR_DIM or int(base_model.coarse_action_in_proj.out_features) != query_dim:
        raise ValueError("Tri-path oracle requires 1024-wide IAR/EAR/final experts.")
    evidence_tokens = dce_base.EVIDENCE_TOKENS_BY_MODE[args.evidence_mode]
    evidence_dim = int(mrr_block_selector.TOKEN_EMBEDDING_DIM) * (
        2 if args.evidence_mode == "selected128_pair" else 1
    )
    iar_adapter = dce_iar_evidence_adapter.DCEIAREvidenceAdapter(
        dce_iar_evidence_adapter.DCEIAREvidenceAdapterConfig(
            query_dim=IAR_DIM,
            query_tokens=IAR_TOKENS,
            evidence_dim=evidence_dim,
            evidence_tokens=evidence_tokens,
            attention_dim=args.attention_dim,
            num_heads=args.attention_heads,
        ),
        rngs=nnx.Rngs(args.seed + 707),
    )
    multilayer_config = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapterConfig(
        evidence_dim=evidence_dim,
        evidence_tokens=evidence_tokens,
        expert_dim=query_dim,
        attention_dim=args.attention_dim,
        num_heads=args.attention_heads,
        expert_depth=ml_base.EXPERT_DEPTH,
        injection_layers=ml_base.INJECTION_LAYERS,
    )
    ear_adapter = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter(
        multilayer_config, rngs=nnx.Rngs(args.seed + 808)
    )
    final_adapter = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter(
        multilayer_config, rngs=nnx.Rngs(args.seed + 909)
    )
    iar_graphdef, iar_params = nnx.split(iar_adapter)
    ear_graphdef, ear_params = nnx.split(ear_adapter)
    final_graphdef, final_params = nnx.split(final_adapter)
    parameter_counts = {
        name: int(sum(np.asarray(v).size for v in jax.tree.leaves(params.to_pure_dict())))
        for name, params in (("iar", iar_params), ("ear", ear_params), ("final", final_params))
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    iar_optimizer_state = optimizer.init(iar_params)
    ear_optimizer_state = optimizer.init(ear_params)
    final_optimizer_state = optimizer.init(final_params)
    (
        iar_train_step,
        iar_eval_step,
        ear_train_step,
        ear_eval_step,
        final_train_step,
        final_eval_step,
    ) = _make_steps(
        base_graphdef,
        selector_runtime,
        iar_graphdef,
        ear_graphdef,
        final_graphdef,
        optimizer,
        args,
    )

    evidence_exporter = ml_base._make_evidence_exporter(  # noqa: SLF001
        base_graphdef, selector_runtime, args
    )
    mismatch_map = dce_base._deterministic_mismatch_map(train_indices, pairs)  # noqa: SLF001
    validation_mismatch_map = dce_base._deterministic_mismatch_map(  # noqa: SLF001
        validation_indices, pairs
    )

    def export_cache(indices: np.ndarray) -> dict[int, np.ndarray]:
        cache: dict[int, np.ndarray] = {}
        for record_index_value in indices:
            record_index = int(record_index_value)
            anchor = int(pairs.anchor_indices[record_index])
            batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
                records, np.asarray([record_index], dtype=np.int64)
            )
            output = jax.device_get(
                evidence_exporter(
                    base_state,
                    selector_runtime.scorer_state,
                    batch,
                    jax.random.fold_in(jax.random.key(args.seed), anchor),
                )
            )
            cache[record_index] = np.asarray(output["evidence"], dtype=np.float32)
        return cache

    train_evidence = export_cache(train_indices)
    validation_evidence = export_cache(validation_indices)
    sampling_rng = np.random.default_rng(args.seed)

    def batch_and_inputs(record_index: int) -> tuple[dict[str, Any], jax.Array, jax.Array, int]:
        mismatch_index = mismatch_map[record_index]
        anchor = int(pairs.anchor_indices[record_index])
        batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
            records, np.asarray([record_index], dtype=np.int64)
        )
        key = jax.random.fold_in(jax.random.key(args.seed), anchor)
        return batch, key, jnp.asarray(train_evidence[mismatch_index]), mismatch_index

    with metrics_path.open("w" if args.overwrite else "a", encoding="utf-8") as metrics_file:
        for stage, steps, train_step in (
            ("iar", args.iar_steps, iar_train_step),
            ("ear", args.ear_steps, ear_train_step),
        ):
            for step in range(1, steps + 1):
                record_index = int(sampling_rng.choice(train_indices))
                batch, key, mismatch_evidence, mismatch_index = batch_and_inputs(record_index)
                if stage == "iar":
                    iar_params, iar_optimizer_state, output = train_step(
                        base_state,
                        selector_runtime.scorer_state,
                        iar_params,
                        iar_optimizer_state,
                        batch,
                        key,
                        mismatch_evidence,
                    )
                else:
                    ear_params, ear_optimizer_state, output = train_step(
                        base_state,
                        selector_runtime.scorer_state,
                        ear_params,
                        ear_optimizer_state,
                        batch,
                        key,
                        mismatch_evidence,
                    )
                if step == 1 or step % args.log_interval == 0 or step == steps:
                    host = jax.device_get(output)
                    record = {
                        "stage": stage,
                        "step": step,
                        "pair_index": record_index,
                        "mismatch_pair_index": mismatch_index,
                        "elapsed_seconds": time.monotonic() - started,
                        **{name: float(value) for name, value in host.items()},
                    }
                    metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    metrics_file.flush()
                    LOGGER.info("Tri-path %s step %d/%d loss=%.6f", stage, step, steps, record["loss"])

            evaluator = iar_eval_step if stage == "iar" else ear_eval_step
            params = iar_params if stage == "iar" else ear_params
            validation = ml_base._validation_metrics(  # noqa: SLF001
                evaluator,
                validation_indices,
                pairs,
                records,
                (base_state, selector_runtime.scorer_state, params),
                seed=args.seed,
                mismatch_map=validation_mismatch_map,
                evidence_by_index=validation_evidence,
            )
            if stage == "iar":
                iar_validation = validation
            else:
                ear_validation = validation
            metrics_file.write(
                json.dumps({"stage": f"{stage}_validation", **validation}, sort_keys=True, allow_nan=False)
                + "\n"
            )
            metrics_file.flush()

        for step in range(1, args.final_steps + 1):
            record_index = int(sampling_rng.choice(train_indices))
            batch, key, mismatch_evidence, mismatch_index = batch_and_inputs(record_index)
            final_params, final_optimizer_state, output = final_train_step(
                base_state,
                selector_runtime.scorer_state,
                iar_params,
                ear_params,
                final_params,
                final_optimizer_state,
                batch,
                key,
                mismatch_evidence,
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
                LOGGER.info("Tri-path final step %d/%d loss=%.6f", step, args.final_steps, record["loss"])

        final_validation = ml_base._validation_metrics(  # noqa: SLF001
            final_eval_step,
            validation_indices,
            pairs,
            records,
            (base_state, selector_runtime.scorer_state, iar_params, ear_params, final_params),
            seed=args.seed,
            mismatch_map=validation_mismatch_map,
            evidence_by_index=validation_evidence,
        )
        metrics_file.write(
            json.dumps({"stage": "final_validation", **final_validation}, sort_keys=True, allow_nan=False)
            + "\n"
        )
        metrics_file.flush()

    saved_iar = dce_base._save_params(  # noqa: SLF001
        iar_params, iar_path, name="dce_iar_evidence_adapter", overwrite=args.overwrite
    )
    saved_ear = dce_base._save_params(  # noqa: SLF001
        ear_params, ear_path, name="dce_ear_multilayer_evidence_adapter", overwrite=args.overwrite
    )
    saved_final = dce_base._save_params(  # noqa: SLF001
        final_params, final_path, name="dce_final_multilayer_evidence_adapter", overwrite=args.overwrite
    )
    test_evidence_cache = export_cache(test_indices)
    ordered_test_evidence = np.stack(
        [test_evidence_cache[int(index)] for index in test_indices], axis=0
    )
    shuffled_test_evidence = np.roll(ordered_test_evidence, shift=1, axis=0)
    test_evaluator = _make_test_evaluator(
        base_graphdef,
        selector_runtime,
        iar_graphdef,
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
                records, np.asarray([record_index], dtype=np.int64)
            )
            output = jax.device_get(
                test_evaluator(
                    base_state,
                    selector_runtime.scorer_state,
                    iar_params,
                    ear_params,
                    final_params,
                    batch,
                    jax.random.fold_in(jax.random.key(args.seed), anchor),
                    jnp.asarray(shuffled_test_evidence[test_position]),
                )
            )
            metric_arrays = np.asarray(output["metrics"], dtype=np.float64)
            if not np.all(np.isfinite(metric_arrays)):
                raise FloatingPointError(f"Non-finite tri-path test output at pair {record_index}.")
            record = {
                "test_position": test_position,
                "pair_index": record_index,
                "anchor_index": anchor,
                "target_index": int(pairs.target_indices[record_index]),
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [int(value) for value in np.asarray(output["selected_ids"])],
                "metric_arrays": metric_arrays.tolist(),
                "methods": {
                    name: {
                        "metrics": dce_base.mrr_oracle._metric_dict(metric_arrays[index]),  # noqa: SLF001
                        "gap_closure_vs_stale": dce_base._closure(  # noqa: SLF001
                            metric_arrays[index], metric_arrays[0]
                        ),
                    }
                    for index, name in enumerate(METHOD_NAMES)
                },
                "iar_gates": [float(value) for value in np.asarray(output["iar_gates"])],
                "ear_gates": [float(value) for value in np.asarray(output["ear_gates"])],
                "final_gates": [float(value) for value in np.asarray(output["final_gates"])],
                "predicted_direct_iar_mse": float(output["predicted_direct_iar_mse"]),
                "shuffled_direct_iar_mse": float(output["shuffled_direct_iar_mse"]),
                "zero_evidence_iar_drift_mse": float(output["zero_evidence_iar_drift_mse"]),
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
            LOGGER.info("Tri-path test %d/%d anchor=%d", test_position + 1, test_indices.size, anchor)

    evaluation = _aggregate_test(test_records, args)
    summary = {
        "method": "Three-level Evidence ACoT IAR-EAR-final oracle",
        "status": "offline_test22_oracle_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
        },
        "evidence": {
            "mode": args.evidence_mode,
            "tokens": evidence_tokens,
            "dim": evidence_dim,
            "selector": "learned MRR top-8",
            "token_contract": ml_base._evidence_token_contract(args.evidence_mode),  # noqa: SLF001
        },
        "architecture": {
            "paths": ["IAR evidence adapter", "EAR four-layer hook", "final four-layer hook"],
            "training_order": ["IAR", "EAR", "final"],
            "iar_shape": [IAR_TOKENS, IAR_DIM],
            "multilayer_injection_layers": list(ml_base.INJECTION_LAYERS),
            "parameter_counts": parameter_counts,
            "base_frozen": True,
            "action_residual": False,
            "default_inference_modified": False,
        },
        "training": {
            "final_conditioning": "same frozen predIAR and predEAR for base/direct teacher",
            "mismatch": "deterministic different-episode evidence with margin ranking",
            "zero_evidence": "bias-free exact response no-op",
            "iar_validation": iar_validation,
            "ear_validation": ear_validation,
            "final_validation": final_validation,
            "checkpoints": {
                "iar": str(saved_iar),
                "ear": str(saved_ear),
                "final": str(saved_final),
            },
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
        "Tri-path Evidence ACoT complete decision=%s elapsed=%.1fs",
        evaluation["gate"]["decision"],
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
