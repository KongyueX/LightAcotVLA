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
import flax.traverse_util as traverse_util
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
    split_seed: int | None = None
    stratify_splits_by_task: bool = False
    bootstrap_episode_groups: bool = False
    train_steps: int = 20_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    calibration_fraction: float = 0.0
    log_interval: int = 100
    checkpoint_interval: int = 5_000
    select_best_validation: bool = True
    early_stopping_patience_logs: int = 0
    early_stopping_min_delta: float = 0.0
    hidden_dim: int = 256
    temporal_layers: int = 3
    temporal_backbone: str = "local_mlp"
    num_heads: int = 4
    feed_forward_multiplier: int = 4
    reference_horizon: int = 10
    coarse_stride: int = 2
    final_stride: int = 1
    visual_num_queries: int = 0
    paired_advantage_heads: bool = False
    paired_distribution_heads: bool = False
    resume_legacy_paired_heads: bool = False
    physical_action_dim: int = 7
    minimum_trials_per_candidate: int = 3
    selection_false_long_upper_bound: float = 0.05
    selection_success_noninferiority: float = 0.01
    selection_max_long_event_probability: float = 0.20

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
    loss_survival: float = 1.0
    loss_success_advantage: float = 1.0
    loss_elapsed_advantage: float = 0.25
    loss_calls_advantage: float = 0.10
    loss_false_long: float = 2.0
    loss_danger_rescue: float = 0.0
    loss_paired_elapsed: float = 0.0
    loss_faster_long: float = 0.0

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
    "hazard_event_count",
    "hazard_at_risk_count",
    "raw_h",
    "success_count",
    "timeout_count",
    "trial_count",
    "remaining_steps_mean",
    "remaining_calls_mean",
    "remaining_calls_variance",
    "elapsed_mean",
    "elapsed_variance",
    "dangerous_long_count",
    "paired_trial_count",
    "trial_success",
    "trial_elapsed",
    "trial_valid",
)


def _loss_weights(args: Args) -> ExecutionHorizonLossWeights:
    hierarchical = args.temporal_backbone == "transformer"
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
        survival=args.loss_survival if hierarchical else 0.0,
        success_advantage=(args.loss_success_advantage if hierarchical and not args.paired_distribution_heads else 0.0),
        elapsed_advantage=(args.loss_elapsed_advantage if hierarchical and not args.paired_distribution_heads else 0.0),
        calls_advantage=args.loss_calls_advantage if hierarchical else 0.0,
        false_long=args.loss_false_long if hierarchical else 0.0,
        danger_rescue=args.loss_danger_rescue if hierarchical else 0.0,
        paired_elapsed=args.loss_paired_elapsed if hierarchical else 0.0,
        faster_long=args.loss_faster_long if hierarchical else 0.0,
    )


def _label_weights(args: Args) -> ExecutionHorizonLabelWeights:
    return ExecutionHorizonLabelWeights(
        success_failure=args.success_failure_multiplier,
        timeout_positive=args.timeout_positive_multiplier,
        event_positive=args.event_positive_multiplier,
        risk_event=args.risk_event_multiplier,
    )


def _validate_paired_args(args: Args) -> None:
    paired_losses_enabled = any(
        value > 0 for value in (args.loss_danger_rescue, args.loss_paired_elapsed, args.loss_faster_long)
    )
    if args.paired_advantage_heads and args.paired_distribution_heads:
        raise ValueError("--paired-advantage-heads and --paired-distribution-heads are mutually exclusive.")
    if args.paired_advantage_heads and args.temporal_backbone != "transformer":
        raise ValueError("paired_advantage_heads require temporal_backbone=transformer.")
    if args.paired_distribution_heads and args.temporal_backbone != "transformer":
        raise ValueError("paired_distribution_heads require temporal_backbone=transformer.")
    if paired_losses_enabled and not (args.paired_advantage_heads or args.paired_distribution_heads):
        raise ValueError("Paired loss weights require an explicit paired head mode.")
    if args.paired_distribution_heads and (args.loss_danger_rescue <= 0 or args.loss_paired_elapsed <= 0):
        raise ValueError("paired_distribution_heads require positive --loss-danger-rescue and --loss-paired-elapsed.")
    if args.resume_legacy_paired_heads and not args.paired_distribution_heads:
        raise ValueError("--resume-legacy-paired-heads is allowed only with --paired-distribution-heads.")
    if args.resume_legacy_paired_heads and args.resume_params is None:
        raise ValueError("--resume-legacy-paired-heads requires --resume-params.")


def _split_indices(arrays: dict[str, np.ndarray], args: Args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5).")
    if args.calibration_fraction < 0 or args.calibration_fraction >= 0.5:
        raise ValueError("calibration_fraction must be in [0, 0.5).")
    if args.validation_fraction + args.calibration_fraction >= 1.0:
        raise ValueError("validation_fraction + calibration_fraction must be below one.")
    groups = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    groups += np.asarray(arrays["episode_id"], dtype=np.uint64)
    unique_groups = np.unique(groups)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    rng = np.random.default_rng(split_seed)
    required_partitions = 3 if args.calibration_fraction > 0 else 2
    if len(unique_groups) < required_partitions:
        if args.temporal_backbone == "transformer":
            raise ValueError(
                "Transformer training requires enough episodes for disjoint train/validation/calibration splits."
            )
        indices = np.arange(len(groups), dtype=np.int64)
        rng.shuffle(indices)
        if len(indices) == 1:
            return indices, indices, np.empty((0,), dtype=np.int64)
        validation_count = max(1, round(len(indices) * args.validation_fraction))
        validation_count = min(validation_count, len(indices) - 1)
        return indices[validation_count:], indices[:validation_count], np.empty((0,), dtype=np.int64)
    if args.stratify_splits_by_task:
        task_ids = np.asarray(arrays["task_id"], dtype=np.int64)
        group_tasks = np.empty(unique_groups.shape, dtype=np.int64)
        for index, group in enumerate(unique_groups):
            tasks = np.unique(task_ids[groups == group])
            if tasks.size != 1:
                raise ValueError(f"Episode group {int(group)} spans multiple tasks: {tasks.tolist()}.")
            group_tasks[index] = tasks[0]
        validation_parts: list[np.ndarray] = []
        calibration_parts: list[np.ndarray] = []
        for task_id in np.unique(group_tasks):
            task_groups = unique_groups[group_tasks == task_id].copy()
            rng.shuffle(task_groups)
            validation_count = max(1, round(len(task_groups) * args.validation_fraction))
            calibration_count = (
                max(1, round(len(task_groups) * args.calibration_fraction)) if args.calibration_fraction > 0 else 0
            )
            if validation_count + calibration_count >= len(task_groups):
                raise ValueError(
                    "Task-stratified split leaves no training groups for "
                    f"task {int(task_id)} with {len(task_groups)} groups."
                )
            validation_parts.append(task_groups[:validation_count])
            calibration_parts.append(task_groups[validation_count : validation_count + calibration_count])
        validation_groups = np.concatenate(validation_parts)
        calibration_groups = (
            np.concatenate(calibration_parts) if args.calibration_fraction > 0 else np.empty((0,), dtype=np.uint64)
        )
    else:
        rng.shuffle(unique_groups)
        validation_count = max(1, round(len(unique_groups) * args.validation_fraction))
        calibration_count = (
            max(1, round(len(unique_groups) * args.calibration_fraction)) if args.calibration_fraction > 0 else 0
        )
        if validation_count + calibration_count >= len(unique_groups):
            raise ValueError("Episode-level split leaves no training groups.")
        validation_groups = unique_groups[:validation_count]
        calibration_groups = unique_groups[validation_count : validation_count + calibration_count]
    validation_mask = np.isin(groups, validation_groups)
    calibration_mask = np.isin(groups, calibration_groups)
    train_indices = np.flatnonzero(~validation_mask & ~calibration_mask)
    validation_indices = np.flatnonzero(validation_mask)
    calibration_indices = np.flatnonzero(calibration_mask)
    if not train_indices.size or not validation_indices.size:
        raise ValueError("Episode-level split produced an empty train or validation partition.")
    return train_indices, validation_indices, calibration_indices


def _batch(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, jax.Array]:
    fields = _INPUT_FIELDS + _LABEL_FIELDS
    batch = {name: jnp.asarray(arrays[name][indices]) for name in fields}
    if "prefix_tokens" in arrays:
        batch["prefix_tokens"] = jnp.asarray(arrays["prefix_tokens"][indices])
        batch["prefix_mask"] = jnp.asarray(arrays["prefix_token_mask"][indices])
    return batch


def _predict(module: ExecutionHorizonPredictor, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    inputs = {name: batch[name] for name in _INPUT_FIELDS}
    if "prefix_tokens" in batch:
        inputs["prefix_tokens"] = batch["prefix_tokens"]
        inputs["prefix_mask"] = batch["prefix_mask"]
    return module(**inputs)


def _save_sidecar(params: nnx.State, target: pathlib.Path) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {"execution_horizon_predictor": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=True)


def _path_text(path: tuple[Any, ...]) -> str:
    return "/".join(map(str, path))


def _migrate_legacy_paired_state(
    module: ExecutionHorizonPredictor,
    state: nnx.State,
    loaded: dict[str, Any],
) -> tuple[nnx.State, dict[str, Any]]:
    """Strictly migrate the two legacy Bernoulli heads to one categorical head."""

    if not module.config.paired_distribution_heads:
        raise ValueError("Legacy paired-head migration is allowed only with --paired-distribution-heads.")
    expected_flat = traverse_util.flatten_dict(state.to_pure_dict())
    loaded_flat = traverse_util.flatten_dict(loaded)
    outcome_paths = {
        ("paired_outcome_logits_head", "kernel"),
        ("paired_outcome_logits_head", "bias"),
    }
    legacy_paths = {
        (head, parameter)
        for head in ("danger_logits_head", "rescue_logits_head", "success_advantage_log_scale_head")
        for parameter in ("kernel", "bias")
    }
    required_legacy_paths = {
        (head, parameter) for head in ("danger_logits_head", "rescue_logits_head") for parameter in ("kernel", "bias")
    }
    reinitialized_paths = {
        ("elapsed_advantage_log_scale_head", "kernel"),
        ("elapsed_advantage_log_scale_head", "bias"),
    }
    missing = set(expected_flat).difference(loaded_flat)
    unexpected = set(loaded_flat).difference(expected_flat)
    if missing != outcome_paths:
        invalid = sorted(_path_text(path) for path in missing.symmetric_difference(outcome_paths))
        raise ValueError(f"Legacy paired checkpoint has non-migratable missing parameters: {invalid[:8]}")
    if not required_legacy_paths.issubset(loaded_flat):
        absent = sorted(_path_text(path) for path in required_legacy_paths.difference(loaded_flat))
        raise ValueError(f"Legacy paired checkpoint is missing source heads: {absent}")
    if not unexpected.issubset(legacy_paths):
        invalid = sorted(_path_text(path) for path in unexpected.difference(legacy_paths))
        raise ValueError(f"Legacy paired checkpoint has unexpected parameters: {invalid[:8]}")

    shared_paths = set(expected_flat).intersection(loaded_flat).difference(reinitialized_paths)
    mismatched = sorted(
        _path_text(path) for path in shared_paths if np.shape(loaded_flat[path]) != np.shape(expected_flat[path])
    )
    if mismatched:
        raise ValueError(f"Legacy paired checkpoint has shared parameter shape mismatches: {mismatched[:8]}")

    danger_kernel = jnp.asarray(loaded_flat[("danger_logits_head", "kernel")])
    danger_bias = jnp.asarray(loaded_flat[("danger_logits_head", "bias")])
    rescue_kernel = jnp.asarray(loaded_flat[("rescue_logits_head", "kernel")])
    rescue_bias = jnp.asarray(loaded_flat[("rescue_logits_head", "bias")])
    if danger_kernel.shape != rescue_kernel.shape or danger_bias.shape != rescue_bias.shape:
        raise ValueError("Legacy danger/rescue heads must have identical shapes for categorical migration.")
    migrated_kernel = jnp.stack(
        [danger_kernel, jnp.zeros_like(danger_kernel), rescue_kernel],
        axis=-1,
    ).reshape(expected_flat[("paired_outcome_logits_head", "kernel")].shape)
    migrated_bias = jnp.stack(
        [danger_bias, jnp.zeros_like(danger_bias), rescue_bias],
        axis=-1,
    ).reshape(expected_flat[("paired_outcome_logits_head", "bias")].shape)

    migrated_values = {
        ("paired_outcome_logits_head", "kernel"): migrated_kernel,
        ("paired_outcome_logits_head", "bias"): migrated_bias,
    }
    merged_flat = {}
    for path, expected in expected_flat.items():
        if path in migrated_values:
            value = migrated_values[path]
        elif path in reinitialized_paths:
            # This head used to emit log sigma. Distribution mode instead
            # applies softplus to a raw value, so the same-shaped legacy
            # tensor is intentionally not restored.
            value = expected
        else:
            value = loaded_flat[path]
        merged_flat[path] = jnp.asarray(value, dtype=expected.dtype)
    state.replace_by_pure_dict(traverse_util.unflatten_dict(merged_flat))
    report = {
        "mode": "legacy_paired_to_distribution",
        "enabled": True,
        "shared_parameter_leaves": len(shared_paths),
        "migrated_parameter_leaves": sorted(_path_text(path) for path in outcome_paths),
        "dropped_legacy_parameter_leaves": sorted(_path_text(path) for path in unexpected),
        "reinitialized_parameter_leaves": sorted(_path_text(path) for path in reinitialized_paths),
        "tie_logits_initialized_to_zero": True,
    }
    return state, report


def _restore_predictor(
    module: ExecutionHorizonPredictor,
    params_path: str,
    *,
    resume_legacy_paired_heads: bool = False,
) -> tuple[ExecutionHorizonPredictor, dict[str, Any]]:
    loaded = model_lib.restore_params(params_path, dtype=jnp.float32)
    # Orbax serializes integer keys used by NNX list containers (for example
    # temporal_layers/0) as strings.  Convert them back before replacing the
    # freshly constructed predictor state so iterative SFT can warm-start from
    # the previous round's sidecar.
    loaded = model_lib.convert_str_keys_to_int(loaded)
    if "execution_horizon_predictor" in loaded:
        loaded = loaded["execution_horizon_predictor"]
    graphdef, state = nnx.split(module)
    if resume_legacy_paired_heads:
        state, report = _migrate_legacy_paired_state(module, state, loaded)
    else:
        state.replace_by_pure_dict(loaded)
        report = {"mode": "strict", "enabled": False}
    return nnx.merge(graphdef, state), report


def _advantage_scale(
    values: np.ndarray,
    indices: np.ndarray,
    candidate_horizons: tuple[int, ...],
    reference_horizon: int,
    *,
    minimum: float,
) -> float:
    reference_index = candidate_horizons.index(reference_horizon)
    long_indices = [index for index, horizon in enumerate(candidate_horizons) if horizon > reference_horizon]
    selected = np.asarray(values[indices], dtype=np.float64)
    differences = selected[:, long_indices] - selected[:, reference_index : reference_index + 1]
    differences = differences[np.isfinite(differences)]
    if not differences.size:
        raise ValueError("Cannot determine advantage scale without finite training labels.")
    return float(
        max(
            np.quantile(np.abs(differences), 0.75),
            np.std(differences),
            minimum,
        )
    )


def main(args: Args) -> None:
    if args.train_steps <= 0 or args.batch_size <= 0:
        raise ValueError("train_steps and batch_size must be positive.")
    if args.early_stopping_patience_logs < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping patience and min delta must be non-negative.")
    if not 0 < args.selection_max_long_event_probability < 1:
        raise ValueError("selection_max_long_event_probability must lie in (0, 1).")
    if args.split_seed is not None and args.split_seed < 0:
        raise ValueError("split_seed must be non-negative when set.")
    _validate_paired_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    candidate_rows = np.asarray(arrays["candidate_horizons"], dtype=np.int64)
    if not np.all(candidate_rows == candidate_rows[:1]):
        raise ValueError("All roots must store the same ordered candidate_horizons.")
    candidate_horizons = tuple(int(value) for value in candidate_rows[0])
    if args.temporal_backbone not in {"local_mlp", "transformer"}:
        raise ValueError("temporal_backbone must be local_mlp or transformer.")
    if args.temporal_backbone == "transformer":
        if not np.all(np.asarray(arrays["schema_version"]) == horizon_dataset.SCHEMA_VERSION):
            raise ValueError("Transformer training requires count-aware schema v2 shards, not repeat-0 v1 labels.")
        if args.calibration_fraction <= 0:
            raise ValueError("Transformer training requires a non-zero independent calibration_fraction.")
        if args.split_seed is None:
            raise ValueError(
                "Transformer training requires an explicit split_seed so training-seed replications share splits."
            )
        trial_count = np.asarray(arrays["trial_count"], dtype=np.int64)
        if np.any(trial_count < args.minimum_trials_per_candidate):
            raise ValueError(
                "Transformer training requires multi-seed labels for every root/candidate; "
                f"minimum observed trial_count={int(trial_count.min())}, "
                f"required={args.minimum_trials_per_candidate}."
            )
        success_count = np.asarray(arrays["success_count"], dtype=np.int64)
        timeout_count = np.asarray(arrays["timeout_count"], dtype=np.int64)
        raw_trial_valid = np.asarray(arrays["trial_valid"], dtype=np.bool_)
        raw_trial_success = np.asarray(arrays["trial_success"], dtype=np.bool_)
        raw_trial_timeout = np.asarray(arrays["trial_timeout"], dtype=np.bool_)
        raw_trial_elapsed = np.asarray(arrays["trial_elapsed"], dtype=np.float32)
        if np.any(success_count + timeout_count != trial_count):
            raise ValueError("Every valid continuation trial must be exactly success or timeout.")
        if not np.array_equal(np.sum(raw_trial_valid, axis=-1), trial_count):
            raise ValueError("trial_count does not match trial_valid raw outcomes.")
        if not np.array_equal(np.sum(raw_trial_success & raw_trial_valid, axis=-1), success_count):
            raise ValueError("success_count does not match raw continuation outcomes.")
        if not np.array_equal(np.sum(raw_trial_timeout & raw_trial_valid, axis=-1), timeout_count):
            raise ValueError("timeout_count does not match raw continuation outcomes.")
        if not np.all(np.isfinite(raw_trial_elapsed[raw_trial_valid])):
            raise ValueError("Every valid continuation trial must contain a finite elapsed label.")
        elapsed_mean = np.asarray(arrays["elapsed_mean"], dtype=np.float32)
        elapsed_variance = np.asarray(arrays["elapsed_variance"], dtype=np.float32)
        if not np.all(np.isfinite(elapsed_mean)) or not np.all(np.isfinite(elapsed_variance)):
            raise ValueError("Transformer training requires finite elapsed mean/variance labels.")
        if np.any(elapsed_variance < 0):
            raise ValueError("elapsed_variance must be non-negative.")
        dangerous = np.asarray(arrays["dangerous_long_count"], dtype=np.int64)
        paired = np.asarray(arrays["paired_trial_count"], dtype=np.int64)
        if np.any(dangerous < 0) or np.any(dangerous > paired):
            raise ValueError("dangerous_long_count must lie between zero and paired_trial_count.")
        reference_index = candidate_horizons.index(args.reference_horizon)
        long_indices = [index for index, horizon in enumerate(candidate_horizons) if horizon > args.reference_horizon]
        paired_from_raw = (
            raw_trial_valid[:, long_indices] & raw_trial_valid[:, reference_index : reference_index + 1, :]
        )
        dangerous_from_raw = (
            paired_from_raw
            & raw_trial_success[:, reference_index : reference_index + 1, :]
            & ~raw_trial_success[:, long_indices]
        )
        expected_paired = np.zeros_like(paired)
        expected_dangerous = np.zeros_like(dangerous)
        expected_paired[:, long_indices] = np.sum(paired_from_raw, axis=-1)
        expected_dangerous[:, long_indices] = np.sum(dangerous_from_raw, axis=-1)
        if not np.array_equal(expected_paired, paired):
            raise ValueError("paired_trial_count does not match raw continuation outcomes.")
        if not np.array_equal(expected_dangerous, dangerous):
            raise ValueError("dangerous_long_count does not match raw continuation outcomes.")
        hazard_events = np.asarray(arrays["hazard_event_count"], dtype=np.int64)
        hazard_at_risk = np.asarray(arrays["hazard_at_risk_count"], dtype=np.int64)
        if np.any(hazard_events < 0) or np.any(hazard_events > hazard_at_risk):
            raise ValueError("hazard_event_count must lie between zero and hazard_at_risk_count.")
        if args.visual_num_queries and "prefix_tokens" not in arrays:
            raise ValueError("visual_num_queries requires prefix_tokens/prefix_token_mask in the dataset.")
        if args.visual_num_queries and np.any(
            np.sum(np.asarray(arrays["prefix_token_mask"], dtype=np.bool_), axis=-1) == 0
        ):
            raise ValueError("Every root must contain at least one valid prefix token.")
    train_indices, validation_indices, calibration_indices = _split_indices(arrays, args)
    all_weights = horizon_dataset.sampling_weights(
        arrays,
        focus_task_ids=args.focus_task_ids,
        focus_task_multiplier=args.focus_task_multiplier,
        high_risk_multiplier=args.high_risk_multiplier,
        gripper_multiplier=args.gripper_multiplier,
        failure_multiplier=args.failure_multiplier,
    )
    group_ids = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    group_ids += np.asarray(arrays["episode_id"], dtype=np.uint64)
    bootstrap_group_counts: dict[int, int] = {}
    train_multiplicity = np.ones((train_indices.size,), dtype=np.float64)
    if args.bootstrap_episode_groups:
        train_groups = np.unique(group_ids[train_indices])
        bootstrap_rng = np.random.default_rng(args.seed)
        sampled_groups = bootstrap_rng.choice(train_groups, size=train_groups.size, replace=True)
        values, counts = np.unique(sampled_groups, return_counts=True)
        bootstrap_group_counts = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
        train_multiplicity = np.asarray(
            [bootstrap_group_counts.get(int(group), 0) for group in group_ids[train_indices]],
            dtype=np.float64,
        )
    train_probabilities = all_weights[train_indices] * train_multiplicity
    if not np.any(train_probabilities > 0):
        raise ValueError("Episode bootstrap produced no positive-weight training roots.")
    train_probabilities /= train_probabilities.sum()
    train_support_size = int(np.count_nonzero(train_probabilities))

    elapsed_advantage_scale = 1.0
    calls_advantage_scale = 1.0
    if args.temporal_backbone == "transformer":
        elapsed_advantage_scale = _advantage_scale(
            arrays["elapsed_mean"],
            train_indices,
            candidate_horizons,
            args.reference_horizon,
            minimum=0.01,
        )
        calls_advantage_scale = _advantage_scale(
            arrays["remaining_calls_mean"],
            train_indices,
            candidate_horizons,
            args.reference_horizon,
            minimum=1.0,
        )

    predictor_config = ExecutionHorizonPredictorConfig(
        prefix_feature_dim=int(arrays["prefix_feature"].shape[-1]),
        state_dim=int(arrays["state"].shape[-1]),
        action_dim=int(arrays["final_actions"].shape[-1]),
        physical_action_dim=args.physical_action_dim,
        coarse_horizon=int(arrays["coarse_actions"].shape[-2]),
        action_horizon=int(arrays["final_actions"].shape[-2]),
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        temporal_backbone=args.temporal_backbone,
        num_heads=args.num_heads,
        feed_forward_multiplier=args.feed_forward_multiplier,
        candidate_horizons=candidate_horizons,
        reference_horizon=args.reference_horizon,
        coarse_stride=args.coarse_stride,
        final_stride=args.final_stride,
        visual_num_queries=args.visual_num_queries,
        paired_advantage_heads=args.paired_advantage_heads,
        paired_distribution_heads=args.paired_distribution_heads,
        elapsed_advantage_scale=elapsed_advantage_scale,
        calls_advantage_scale=calls_advantage_scale,
    )
    (output_dir / "predictor_config.json").write_text(
        json.dumps(dataclasses.asdict(predictor_config), indent=2, sort_keys=True) + "\n"
    )
    split_manifest = {
        "split_seed": args.seed if args.split_seed is None else args.split_seed,
        "training_seed": args.seed,
        "stratify_splits_by_task": args.stratify_splits_by_task,
        "bootstrap_episode_groups": args.bootstrap_episode_groups,
        "bootstrap_train_group_counts": bootstrap_group_counts,
        "train_group_ids": sorted({int(value) for value in group_ids[train_indices]}),
        "validation_group_ids": sorted({int(value) for value in group_ids[validation_indices]}),
        "calibration_group_ids": sorted({int(value) for value in group_ids[calibration_indices]}),
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    module = ExecutionHorizonPredictor(predictor_config, rngs=nnx.Rngs(args.seed))
    resume_report: dict[str, Any] | None = None
    if args.resume_params is not None:
        module, resume_report = _restore_predictor(
            module,
            args.resume_params,
            resume_legacy_paired_heads=args.resume_legacy_paired_heads,
        )
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
                elapsed_advantage_scale=predictor_config.elapsed_advantage_scale,
                candidate_horizons=predictor_config.candidate_horizons,
                reference_horizon=predictor_config.reference_horizon,
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
            elapsed_advantage_scale=predictor_config.elapsed_advantage_scale,
            candidate_horizons=predictor_config.candidate_horizons,
            reference_horizon=predictor_config.reference_horizon,
        )
        trial_count = jnp.maximum(batch["trial_count"].astype(jnp.float32), 1.0)
        success_rate = batch["success_count"].astype(jnp.float32) / trial_count
        timeout_rate = batch["timeout_count"].astype(jnp.float32) / trial_count
        success_label = success_rate >= 0.5
        timeout_label = timeout_rate >= 0.5
        success_prediction = jax.nn.sigmoid(predictions["success_logits"]) >= 0.5
        timeout_prediction = jax.nn.sigmoid(predictions["timeout_logits"]) >= 0.5
        event_prediction = predictions["event_logits"] >= 0.0
        event_label = batch["event_mask"].astype(jnp.bool_)
        risk_event_prediction = predictions["fused_risk"] >= args.event_risk_threshold
        event_valid = batch["risk_valid"].astype(jnp.bool_)
        branch_valid = batch["branch_valid"].astype(jnp.bool_)

        def binary_recall(prediction: jax.Array, target: jax.Array, valid: jax.Array) -> jax.Array:
            positives = target & valid
            return jnp.sum(prediction & positives) / jnp.maximum(jnp.sum(positives), 1)

        def binary_precision(prediction: jax.Array, target: jax.Array, valid: jax.Array) -> jax.Array:
            predicted_positives = prediction & valid
            return jnp.sum(target & predicted_positives) / jnp.maximum(jnp.sum(predicted_positives), 1)

        def masked_mean(values: jax.Array, valid: jax.Array) -> jax.Array:
            valid = jnp.asarray(valid, dtype=values.dtype)
            return jnp.sum(values * valid) / jnp.maximum(jnp.sum(valid), 1.0)

        metrics["success_accuracy"] = jnp.mean(success_prediction == success_label)
        metrics["timeout_accuracy"] = jnp.mean(timeout_prediction == timeout_label)
        metrics["failure_recall"] = binary_recall(~success_prediction, ~success_label, branch_valid)
        metrics["timeout_recall"] = binary_recall(timeout_prediction, timeout_label, branch_valid)
        metrics["event_precision"] = binary_precision(event_prediction, event_label, event_valid)
        metrics["event_recall"] = binary_recall(event_prediction, event_label, event_valid)
        metrics["fused_risk_event_precision"] = binary_precision(risk_event_prediction, event_label, event_valid)
        metrics["fused_risk_event_recall"] = binary_recall(risk_event_prediction, event_label, event_valid)
        success_squared_error = (jax.nn.sigmoid(predictions["success_logits"]) - success_rate) ** 2
        metrics["success_brier"] = jnp.sum(success_squared_error * branch_valid) / jnp.maximum(jnp.sum(branch_valid), 1)
        candidate_values = jnp.asarray(predictor_config.candidate_horizons, dtype=jnp.int32)
        predicted_raw_h = candidate_values[jnp.argmax(predictions["raw_h_logits"], axis=-1)]
        metrics["raw_h_accuracy"] = jnp.mean(predicted_raw_h == batch["raw_h"])
        metrics["raw_h_mae"] = jnp.mean(jnp.abs(predicted_raw_h - batch["raw_h"]))
        if predictor_config.temporal_backbone == "transformer":
            long_indices = jnp.asarray(
                [
                    index
                    for index, horizon in enumerate(predictor_config.candidate_horizons)
                    if horizon > predictor_config.reference_horizon
                ],
                dtype=jnp.int32,
            )
            long_horizons = jnp.asarray(
                [
                    horizon
                    for horizon in predictor_config.candidate_horizons
                    if horizon > predictor_config.reference_horizon
                ],
                dtype=jnp.int32,
            )
            reference_index = predictor_config.candidate_horizons.index(predictor_config.reference_horizon)
            eligible = (
                predictions["success_advantage"] - 1.96 * predictions["success_advantage_std"]
                >= -args.selection_success_noninferiority
            ) & (predictions["elapsed_advantage"] + 1.96 * predictions["elapsed_advantage_std"] < 0.0)
            long_event_probability = 1.0 - jnp.take(
                predictions["survival"],
                long_horizons - 1,
                axis=1,
            )
            eligible &= long_event_probability <= args.selection_max_long_event_probability
            eligible_index = jnp.max(
                jnp.where(eligible, jnp.arange(eligible.shape[-1], dtype=jnp.int32) + 1, 0),
                axis=-1,
            )
            selected = eligible_index > 0
            selected_long_index = jnp.maximum(eligible_index - 1, 0)
            metrics["long_coverage"] = jnp.mean(selected)
            selected_long_event_probability = jnp.take_along_axis(
                long_event_probability,
                selected_long_index[:, None],
                axis=1,
            )[:, 0]
            metrics["selected_long_event_probability"] = jnp.sum(
                jnp.where(selected, selected_long_event_probability, 0.0)
            ) / jnp.maximum(jnp.sum(selected), 1)
            dangerous = jnp.take(batch["dangerous_long_count"], long_indices, axis=1)
            paired = jnp.take(batch["paired_trial_count"], long_indices, axis=1)
            selected_dangerous = jnp.take_along_axis(dangerous, selected_long_index[:, None], axis=1)[:, 0]
            selected_paired = jnp.take_along_axis(paired, selected_long_index[:, None], axis=1)[:, 0]
            dangerous_total = jnp.sum(jnp.where(selected, selected_dangerous, 0.0))
            paired_total = jnp.sum(jnp.where(selected, selected_paired, 0.0))
            false_long_rate = dangerous_total / jnp.maximum(paired_total, 1.0)
            z_two_sided = jnp.asarray(1.96, dtype=jnp.float32)
            wilson_denominator = 1.0 + z_two_sided**2 / jnp.maximum(paired_total, 1.0)
            wilson_center = (
                false_long_rate + z_two_sided**2 / (2.0 * jnp.maximum(paired_total, 1.0))
            ) / wilson_denominator
            wilson_margin = (
                z_two_sided
                / wilson_denominator
                * jnp.sqrt(
                    false_long_rate * (1.0 - false_long_rate) / jnp.maximum(paired_total, 1.0)
                    + z_two_sided**2 / (4.0 * jnp.maximum(paired_total, 1.0) ** 2)
                )
            )
            metrics["false_long_rate"] = false_long_rate
            metrics["false_long_upper_95"] = jnp.where(
                paired_total > 0,
                jnp.minimum(1.0, wilson_center + wilson_margin),
                1.0,
            )
            success_advantage_target = (
                jnp.take(success_rate, long_indices, axis=1) - success_rate[:, reference_index : reference_index + 1]
            )
            selected_success = jnp.take_along_axis(success_advantage_target, selected_long_index[:, None], axis=1)[:, 0]
            elapsed_target = (
                jnp.take(batch["elapsed_mean"], long_indices, axis=1)
                - batch["elapsed_mean"][:, reference_index : reference_index + 1]
            )
            selected_elapsed = jnp.take_along_axis(elapsed_target, selected_long_index[:, None], axis=1)[:, 0]
            selected_count = jnp.sum(selected)

            def mean_and_standard_error(values: jax.Array) -> tuple[jax.Array, jax.Array]:
                mean = jnp.sum(jnp.where(selected, values, 0.0)) / jnp.maximum(selected_count, 1)
                centered = jnp.where(selected, values - mean, 0.0)
                variance = jnp.sum(jnp.square(centered)) / jnp.maximum(selected_count - 1, 1)
                return mean, jnp.sqrt(variance / jnp.maximum(selected_count, 1))

            success_mean, success_standard_error = mean_and_standard_error(selected_success)
            elapsed_mean, elapsed_standard_error = mean_and_standard_error(selected_elapsed)
            one_sided_z = jnp.asarray(1.645, dtype=jnp.float32)
            metrics["selected_success_advantage_target"] = success_mean
            metrics["selected_success_advantage_lcb95"] = jnp.where(
                selected_count > 0,
                success_mean - one_sided_z * success_standard_error,
                -1.0,
            )
            metrics["selected_elapsed_advantage_target"] = elapsed_mean
            metrics["selected_elapsed_advantage_ucb95"] = jnp.where(
                selected_count > 0,
                elapsed_mean + one_sided_z * elapsed_standard_error,
                1e9,
            )
            if predictor_config.paired_advantage_heads or predictor_config.paired_distribution_heads:
                raw_valid = batch["trial_valid"].astype(jnp.bool_)
                raw_success = batch["trial_success"].astype(jnp.bool_)
                raw_elapsed = batch["trial_elapsed"].astype(jnp.float32)
                reference_valid = raw_valid[:, reference_index : reference_index + 1, :]
                reference_success = raw_success[:, reference_index : reference_index + 1, :]
                reference_elapsed = raw_elapsed[:, reference_index : reference_index + 1, :]
                long_valid = jnp.take(raw_valid, long_indices, axis=1)
                long_success = jnp.take(raw_success, long_indices, axis=1)
                long_elapsed = jnp.take(raw_elapsed, long_indices, axis=1)
                pair_valid = reference_valid & long_valid
                pair_valid &= jnp.isfinite(reference_elapsed) & jnp.isfinite(long_elapsed)
                pair_count = jnp.sum(pair_valid, axis=-1)
                danger_rate = jnp.sum(pair_valid & reference_success & ~long_success, axis=-1) / jnp.maximum(
                    pair_count, 1
                )
                rescue_rate = jnp.sum(pair_valid & ~reference_success & long_success, axis=-1) / jnp.maximum(
                    pair_count, 1
                )
                elapsed_delta = long_elapsed - reference_elapsed
                faster_rate = jnp.sum(pair_valid & (elapsed_delta < 0.0), axis=-1) / jnp.maximum(pair_count, 1)
                pair_mask = pair_count > 0
                metrics["danger_brier"] = masked_mean(
                    jnp.square(predictions["danger_probability"] - danger_rate), pair_mask
                )
                metrics["rescue_brier"] = masked_mean(
                    jnp.square(predictions["rescue_probability"] - rescue_rate), pair_mask
                )
                metrics["faster_long_brier"] = masked_mean(
                    jnp.square(predictions["faster_long_probability"] - faster_rate), pair_mask
                )
                metrics["paired_elapsed_mae"] = masked_mean(
                    jnp.abs(predictions["elapsed_advantage"][..., None] - jnp.nan_to_num(elapsed_delta)),
                    pair_valid,
                )
                if predictor_config.paired_distribution_heads:
                    tie_rate = 1.0 - danger_rate - rescue_rate
                    outcome_target = jnp.stack([danger_rate, tie_rate, rescue_rate], axis=-1)
                    metrics["paired_outcome_brier"] = masked_mean(
                        jnp.sum(jnp.square(predictions["paired_outcome_probability"] - outcome_target), axis=-1),
                        pair_mask,
                    )
                    metrics["pairwise_selection_score"] = (
                        metrics["paired_outcome_multinomial_nll"]
                        + metrics["paired_elapsed_student_t_nll"]
                        + metrics["faster_long_binomial"]
                    )
                else:
                    metrics["pairwise_selection_score"] = (
                        metrics["danger_rescue_binomial"]
                        + metrics["paired_elapsed_huber"]
                        + metrics["faster_long_binomial"]
                        + 0.25 * metrics["success_advantage_nll"]
                        + 0.25 * metrics["elapsed_advantage_nll"]
                    )
        return metrics

    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    start_time = time.monotonic()
    last_train_metrics: dict[str, float] = {}
    last_validation_metrics: dict[str, float] = {}
    best_validation_loss = float("inf")
    best_validation_objective = float("inf")
    best_validation_step = 0
    best_params: nnx.State | None = None
    best_long_coverage = -1.0
    best_selection_feasible = False
    baseline_success_brier: float | None = None
    if args.temporal_backbone == "transformer":
        initial_validation = jax.device_get(validation_step(params, _batch(arrays, validation_indices)))
        baseline_success_brier = float(initial_validation["success_brier"])
    logs_without_improvement = 0
    completed_steps = 0
    stopped_early = False
    with metrics_path.open("a") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_support_size < args.batch_size,
                p=train_probabilities,
            )
            params, optimizer_state, train_metrics = train_step(params, optimizer_state, _batch(arrays, sampled))
            completed_steps = step
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_sample = (
                    validation_indices
                    if args.temporal_backbone == "transformer"
                    else rng.choice(
                        validation_indices,
                        size=min(args.batch_size * 4, validation_indices.size),
                        replace=False,
                    )
                )
                validation_metrics = validation_step(params, _batch(arrays, validation_sample))
                last_train_metrics = {
                    f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()
                }
                last_validation_metrics = {
                    f"validation/{name}": float(value) for name, value in jax.device_get(validation_metrics).items()
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
                validation_objective = (
                    last_validation_metrics["validation/pairwise_selection_score"]
                    if predictor_config.paired_advantage_heads or predictor_config.paired_distribution_heads
                    else validation_loss
                )
                if baseline_success_brier is None:
                    baseline_success_brier = last_validation_metrics["validation/success_brier"]
                selection_feasible = False
                long_coverage = -1.0
                if args.temporal_backbone == "transformer":
                    long_coverage = last_validation_metrics["validation/long_coverage"]
                    selection_feasible = (
                        long_coverage > 0
                        and last_validation_metrics["validation/false_long_upper_95"]
                        <= args.selection_false_long_upper_bound
                        and last_validation_metrics["validation/selected_success_advantage_lcb95"]
                        >= -args.selection_success_noninferiority
                        and last_validation_metrics["validation/selected_elapsed_advantage_ucb95"] < 0
                        and last_validation_metrics["validation/success_brier"] <= baseline_success_brier
                    )
                    improved = (
                        selection_feasible
                        and (
                            not best_selection_feasible
                            or long_coverage > best_long_coverage + 1e-6
                            or (
                                abs(long_coverage - best_long_coverage) <= 1e-6
                                and validation_objective < best_validation_objective - args.early_stopping_min_delta
                            )
                        )
                    ) or (
                        not selection_feasible
                        and not best_selection_feasible
                        and validation_objective < best_validation_objective - args.early_stopping_min_delta
                    )
                else:
                    improved = validation_loss < best_validation_loss - args.early_stopping_min_delta
                if improved:
                    best_validation_loss = validation_loss
                    best_validation_objective = validation_objective
                    best_validation_step = step
                    best_long_coverage = long_coverage
                    best_selection_feasible = selection_feasible
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
        "num_calibration_records": int(calibration_indices.size),
        "train_steps": completed_steps,
        "requested_train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "training_seed": args.seed,
        "split_seed": args.seed if args.split_seed is None else args.split_seed,
        "stratify_splits_by_task": args.stratify_splits_by_task,
        "bootstrap_episode_groups": args.bootstrap_episode_groups,
        "train_sampling_support_records": train_support_size,
        "elapsed_seconds": time.monotonic() - start_time,
        "predictor_params": str(final_params.resolve()),
        "selected_checkpoint": "best_validation" if args.select_best_validation else "last_step",
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "best_validation_objective": best_validation_objective,
        "best_long_coverage": best_long_coverage if args.temporal_backbone == "transformer" else None,
        "best_selection_feasible": best_selection_feasible if args.temporal_backbone == "transformer" else None,
        "initial_validation_success_brier": baseline_success_brier,
        "stopped_early": stopped_early,
        "loss_weights": dataclasses.asdict(weights),
        "label_weights": dataclasses.asdict(label_weights),
        "predictor_config": dataclasses.asdict(predictor_config),
        "resume": resume_report,
        "calibration_split_used_for_training": False,
        "last_train_metrics": last_train_metrics,
        "last_validation_metrics": last_validation_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
