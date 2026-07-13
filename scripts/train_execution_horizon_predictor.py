"""Train the Budgeted Event V2-P predictor from counterfactual HDF5 shards.

The base ACoT-VLA checkpoint is deliberately not loaded by this process.  Only
the standalone predictor is optimized and written as an Orbax sidecar.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.models import model as model_lib
from openpi.models.execution_horizon_predictor import ExecutionHorizonLabelWeights
from openpi.models.execution_horizon_predictor import ExecutionHorizonLossWeights
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictor
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictorConfig
from openpi.models.execution_horizon_predictor import execution_horizon_loss


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    resume_params: str | None = None
    seed: int = 7
    train_steps: int = 20_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    log_interval: int = 100
    checkpoint_interval: int = 5_000
    select_best_validation: bool = True
    early_stopping_patience_logs: int = 0
    early_stopping_min_delta: float = 0.0
    hidden_dim: int = 256
    temporal_layers: int = 3

    focus_task_ids: tuple[int, ...] = (8, 9)
    focus_task_multiplier: float = 2.0
    high_risk_multiplier: float = 2.0
    gripper_multiplier: float = 1.5
    failure_multiplier: float = 2.0

    loss_success: float = 1.0
    loss_timeout: float = 0.5
    loss_remaining_calls: float = 0.25
    loss_remaining_steps: float = 0.25
    loss_final_risk: float = 0.5
    loss_action_cot_risk: float = 0.5
    loss_fused_risk: float = 1.0
    loss_event: float = 0.5
    loss_raw_h_classification: float = 0.5
    loss_raw_h_ordinal: float = 0.25

    success_failure_multiplier: float = 4.0
    timeout_positive_multiplier: float = 4.0
    event_positive_multiplier: float = 5.0
    risk_event_multiplier: float = 3.0
    event_risk_threshold: float = 1.5


_INPUT_FIELDS = (
    "prefix_feature",
    "state",
    "coarse_actions",
    "final_actions",
    "previous_actions",
    "previous_h",
    "budget_balance",
    "episode_progress",
    "previous_valid",
)
_LABEL_FIELDS = (
    "branch_success",
    "branch_timeout",
    "remaining_calls",
    "remaining_steps",
    "branch_valid",
    "final_risk",
    "action_cot_risk",
    "fused_risk",
    "event_mask",
    "risk_valid",
    "raw_h",
)


def _loss_weights(args: Args) -> ExecutionHorizonLossWeights:
    return ExecutionHorizonLossWeights(
        success=args.loss_success,
        timeout=args.loss_timeout,
        remaining_calls=args.loss_remaining_calls,
        remaining_steps=args.loss_remaining_steps,
        final_risk=args.loss_final_risk,
        action_cot_risk=args.loss_action_cot_risk,
        fused_risk=args.loss_fused_risk,
        event=args.loss_event,
        raw_h_classification=args.loss_raw_h_classification,
        raw_h_ordinal=args.loss_raw_h_ordinal,
    )


def _label_weights(args: Args) -> ExecutionHorizonLabelWeights:
    return ExecutionHorizonLabelWeights(
        success_failure=args.success_failure_multiplier,
        timeout_positive=args.timeout_positive_multiplier,
        event_positive=args.event_positive_multiplier,
        risk_event=args.risk_event_multiplier,
    )


def _split_indices(arrays: dict[str, np.ndarray], args: Args) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5).")
    groups = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    groups += np.asarray(arrays["episode_id"], dtype=np.uint64)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    if len(unique_groups) < 2:
        indices = np.arange(len(groups), dtype=np.int64)
        rng.shuffle(indices)
        if len(indices) == 1:
            return indices, indices
        validation_count = max(1, round(len(indices) * args.validation_fraction))
        validation_count = min(validation_count, len(indices) - 1)
        return indices[validation_count:], indices[:validation_count]
    rng.shuffle(unique_groups)
    validation_count = max(1, round(len(unique_groups) * args.validation_fraction))
    validation_groups = unique_groups[:validation_count]
    validation_mask = np.isin(groups, validation_groups)
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Episode-level split produced an empty train or validation partition.")
    return train_indices, validation_indices


def _batch(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, jax.Array]:
    fields = _INPUT_FIELDS + _LABEL_FIELDS
    return {name: jnp.asarray(arrays[name][indices]) for name in fields}


def _predict(module: ExecutionHorizonPredictor, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return module(**{name: batch[name] for name in _INPUT_FIELDS})


def _save_sidecar(params: nnx.State, target: pathlib.Path) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {"execution_horizon_predictor": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=True)


def _restore_predictor(module: ExecutionHorizonPredictor, params_path: str) -> ExecutionHorizonPredictor:
    loaded = model_lib.restore_params(params_path, dtype=jnp.float32)
    # Orbax serializes integer keys used by NNX list containers (for example
    # temporal_layers/0) as strings.  Convert them back before replacing the
    # freshly constructed predictor state so iterative SFT can warm-start from
    # the previous round's sidecar.
    loaded = model_lib.convert_str_keys_to_int(loaded)
    if "execution_horizon_predictor" in loaded:
        loaded = loaded["execution_horizon_predictor"]
    graphdef, state = nnx.split(module)
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def main(args: Args) -> None:
    if args.train_steps <= 0 or args.batch_size <= 0:
        raise ValueError("train_steps and batch_size must be positive.")
    if args.early_stopping_patience_logs < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and min delta must be non-negative.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    train_indices, validation_indices = _split_indices(arrays, args)
    all_weights = horizon_dataset.sampling_weights(
        arrays,
        focus_task_ids=args.focus_task_ids,
        focus_task_multiplier=args.focus_task_multiplier,
        high_risk_multiplier=args.high_risk_multiplier,
        gripper_multiplier=args.gripper_multiplier,
        failure_multiplier=args.failure_multiplier,
    )
    train_probabilities = all_weights[train_indices]
    train_probabilities /= train_probabilities.sum()

    predictor_config = ExecutionHorizonPredictorConfig(
        prefix_feature_dim=int(arrays["prefix_feature"].shape[-1]),
        state_dim=int(arrays["state"].shape[-1]),
        action_dim=int(arrays["final_actions"].shape[-1]),
        coarse_horizon=int(arrays["coarse_actions"].shape[-2]),
        action_horizon=int(arrays["final_actions"].shape[-2]),
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
    )
    module = ExecutionHorizonPredictor(predictor_config, rngs=nnx.Rngs(args.seed))
    if args.resume_params is not None:
        module = _restore_predictor(module, args.resume_params)
    graphdef, params = nnx.split(module)
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.train_steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(params)
    weights = _loss_weights(args)
    label_weights = _label_weights(args)

    @jax.jit
    def train_step(
        current_params: nnx.State,
        current_optimizer_state: optax.OptState,
        batch: dict[str, jax.Array],
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        current_module = nnx.merge(graphdef, current_params)

        def loss_fn(candidate: ExecutionHorizonPredictor) -> tuple[jax.Array, dict[str, jax.Array]]:
            predictions = _predict(candidate, batch)
            return execution_horizon_loss(
                predictions,
                {name: batch[name] for name in _LABEL_FIELDS},
                weights=weights,
                label_weights=label_weights,
                remaining_calls_scale=predictor_config.remaining_calls_scale,
                remaining_steps_scale=predictor_config.remaining_steps_scale,
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(current_module)
        updates, next_optimizer_state = optimizer.update(gradients, current_optimizer_state, current_params)
        updated_params = optax.apply_updates(current_params, updates)
        metrics = {**metrics, "loss": loss, "gradient_norm": optax.global_norm(gradients)}
        return updated_params, next_optimizer_state, metrics

    @jax.jit
    def validation_step(current_params: nnx.State, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        current_module = nnx.merge(graphdef, current_params)
        predictions = _predict(current_module, batch)
        _, metrics = execution_horizon_loss(
            predictions,
            {name: batch[name] for name in _LABEL_FIELDS},
            weights=weights,
            label_weights=label_weights,
            remaining_calls_scale=predictor_config.remaining_calls_scale,
            remaining_steps_scale=predictor_config.remaining_steps_scale,
        )
        success_prediction = jax.nn.sigmoid(predictions["success_logits"]) >= 0.5
        timeout_prediction = jax.nn.sigmoid(predictions["timeout_logits"]) >= 0.5
        event_prediction = predictions["event_logits"] >= 0.0
        event_label = batch["event_mask"].astype(jnp.bool_)
        risk_event_prediction = predictions["fused_risk"] >= args.event_risk_threshold
        event_valid = batch["risk_valid"].astype(jnp.bool_)
        branch_valid = batch["branch_valid"].astype(jnp.bool_)

        def binary_recall(
            prediction: jax.Array, target: jax.Array, valid: jax.Array
        ) -> jax.Array:
            positives = target & valid
            return jnp.sum(prediction & positives) / jnp.maximum(jnp.sum(positives), 1)

        def binary_precision(
            prediction: jax.Array, target: jax.Array, valid: jax.Array
        ) -> jax.Array:
            predicted_positives = prediction & valid
            return jnp.sum(target & predicted_positives) / jnp.maximum(jnp.sum(predicted_positives), 1)

        metrics["success_accuracy"] = jnp.mean(success_prediction == batch["branch_success"])
        metrics["timeout_accuracy"] = jnp.mean(timeout_prediction == batch["branch_timeout"])
        metrics["failure_recall"] = binary_recall(~success_prediction, ~batch["branch_success"], branch_valid)
        metrics["timeout_recall"] = binary_recall(timeout_prediction, batch["branch_timeout"], branch_valid)
        metrics["event_precision"] = binary_precision(event_prediction, event_label, event_valid)
        metrics["event_recall"] = binary_recall(event_prediction, event_label, event_valid)
        metrics["fused_risk_event_precision"] = binary_precision(
            risk_event_prediction, event_label, event_valid
        )
        metrics["fused_risk_event_recall"] = binary_recall(risk_event_prediction, event_label, event_valid)
        success_squared_error = (
            jax.nn.sigmoid(predictions["success_logits"]) - batch["branch_success"]
        ) ** 2
        metrics["success_brier"] = jnp.sum(success_squared_error * branch_valid) / jnp.maximum(
            jnp.sum(branch_valid), 1
        )
        metrics["raw_h_accuracy"] = jnp.mean(
            (jnp.argmax(predictions["raw_h_logits"], axis=-1) + 1) == batch["raw_h"]
        )
        metrics["raw_h_mae"] = jnp.mean(
            jnp.abs((jnp.argmax(predictions["raw_h_logits"], axis=-1) + 1) - batch["raw_h"])
        )
        return metrics

    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.monotonic()
    last_train_metrics: dict[str, float] = {}
    last_validation_metrics: dict[str, float] = {}
    best_validation_loss = float("inf")
    best_validation_step = 0
    best_params: nnx.State | None = None
    logs_without_improvement = 0
    completed_steps = 0
    stopped_early = False
    with metrics_path.open("a") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
                p=train_probabilities,
            )
            params, optimizer_state, train_metrics = train_step(params, optimizer_state, _batch(arrays, sampled))
            completed_steps = step
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_sample = rng.choice(
                    validation_indices,
                    size=min(args.batch_size * 4, validation_indices.size),
                    replace=False,
                )
                validation_metrics = validation_step(params, _batch(arrays, validation_sample))
                last_train_metrics = {
                    f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()
                }
                last_validation_metrics = {
                    f"validation/{name}": float(value)
                    for name, value in jax.device_get(validation_metrics).items()
                }
                record: dict[str, Any] = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - start_time,
                    **last_train_metrics,
                    **last_validation_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                validation_loss = last_validation_metrics["validation/loss"]
                if validation_loss < best_validation_loss - args.early_stopping_min_delta:
                    best_validation_loss = validation_loss
                    best_validation_step = step
                    # Optax returns a new state tree on each update, so keeping
                    # this immutable NNX State retains the best validation
                    # checkpoint without loading or copying the base policy.
                    best_params = params
                    logs_without_improvement = 0
                else:
                    logs_without_improvement += 1
                if (
                    args.early_stopping_patience_logs > 0
                    and logs_without_improvement >= args.early_stopping_patience_logs
                ):
                    stopped_early = True
                    break
            if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
                _save_sidecar(params, output_dir / "checkpoints" / f"step-{step:08d}" / "params")

    final_params = output_dir / "params"
    selected_params = best_params if args.select_best_validation and best_params is not None else params
    _save_sidecar(selected_params, final_params)
    summary = {
        "status": "complete",
        "base_policy_loaded": False,
        "base_policy_frozen": True,
        "dataset_inputs": list(args.dataset),
        "num_records": len(arrays["task_id"]),
        "num_train_records": int(train_indices.size),
        "num_validation_records": int(validation_indices.size),
        "train_steps": completed_steps,
        "requested_train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "elapsed_seconds": time.monotonic() - start_time,
        "predictor_params": str(final_params.resolve()),
        "selected_checkpoint": "best_validation" if args.select_best_validation else "last_step",
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "stopped_early": stopped_early,
        "loss_weights": dataclasses.asdict(weights),
        "label_weights": dataclasses.asdict(label_weights),
        "last_train_metrics": last_train_metrics,
        "last_validation_metrics": last_validation_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
