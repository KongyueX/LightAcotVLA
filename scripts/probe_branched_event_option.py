"""Fast go/no-go probe for compiling cached Action-CoT into feedback options.

This is deliberately an offline probe, not a deployable LIBERO policy.  It
answers two prerequisite questions before implementing a runtime:

1. Does the cached final-action chunk still contain a useful recovery action
   after a four-tick disturbance?
2. Can a small, structured state-feedback model select and correct that action
   on held-out episodes without an unrestricted observation-to-action path?

EAR is used to compile event/geometry option descriptors.  Executable
feed-forward actions always come from the cached final-action chunk.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from openpi.action_cot import branched_dataset
from openpi.action_cot import compression


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    seed: int = 7
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    ridge_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    max_options: int = 4
    min_option_length: int = 2
    max_option_length: int = 6
    gripper_boundary_weight: float = 4.0
    calibration_grid_size: int = 181
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray


@dataclasses.dataclass(frozen=True)
class StructuredFeedbackModel:
    coefficient_model: RidgeModel
    coefficient_clip: np.ndarray
    gripper_event_model: RidgeModel
    gripper_event_threshold: float


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("dataset must contain at least one path.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")
    if not args.ridge_grid or any(value < 0 for value in args.ridge_grid):
        raise ValueError("ridge_grid must contain non-negative values.")
    if args.max_options <= 0:
        raise ValueError("max_options must be positive.")
    if args.min_option_length <= 0 or args.max_option_length < args.min_option_length:
        raise ValueError("Option lengths must satisfy 0 < min <= max.")
    if args.gripper_boundary_weight < 0:
        raise ValueError("gripper_boundary_weight must be non-negative.")
    if args.calibration_grid_size < 2:
        raise ValueError("calibration_grid_size must be at least two.")


def _split_roots(
    arrays: Mapping[str, np.ndarray],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return task-stratified, episode-disjoint root indices."""

    tasks = np.asarray(arrays["task_id"], dtype=np.int64)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    if tasks.ndim != 1 or episodes.shape != tasks.shape:
        raise ValueError("task_id and episode_id must be matching rank-one arrays.")
    rng = np.random.default_rng(seed)
    partitions: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
    for task_id in np.unique(tasks):
        roots = np.flatnonzero(tasks == task_id)
        task_episodes = np.unique(episodes[roots])
        if task_episodes.size < 3:
            raise ValueError(f"Task {task_id} needs at least three episodes for train/val/test.")
        rng.shuffle(task_episodes)
        test_count = max(1, round(task_episodes.size * test_fraction))
        validation_count = max(1, round(task_episodes.size * validation_fraction))
        if test_count + validation_count >= task_episodes.size:
            test_count = 1
            validation_count = 1
        episode_groups = (
            task_episodes[test_count + validation_count :],
            task_episodes[test_count : test_count + validation_count],
            task_episodes[:test_count],
        )
        for destination, selected_episodes in zip(partitions, episode_groups, strict=True):
            destination.append(roots[np.isin(episodes[roots], selected_episodes)])
    outputs = tuple(np.sort(np.concatenate(parts)) for parts in partitions)
    if any(not partition.size for partition in outputs):
        raise ValueError("Episode-level split produced an empty partition.")
    return outputs  # type: ignore[return-value]


def _flatten_valid(
    arrays: Mapping[str, np.ndarray],
    root_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = np.asarray(arrays["branch_valid"], dtype=np.bool_)[root_indices]
    local_roots, branches = np.nonzero(valid)
    if not local_roots.size:
        raise ValueError("Partition contains no valid branches.")
    roots = np.asarray(root_indices, dtype=np.int64)[local_roots]
    return {
        "root": roots,
        "root_id": np.asarray(arrays["root_id"])[roots],
        "task_id": np.asarray(arrays["task_id"])[roots],
        "episode_id": np.asarray(arrays["episode_id"])[roots],
        "branch_id": branches.astype(np.int64),
        "anchor_state": np.asarray(arrays["anchor_state"], dtype=np.float32)[roots],
        "current_state": np.asarray(arrays["current_state"], dtype=np.float32)[roots, branches],
        "cached_ear": np.asarray(arrays["cached_ear"], dtype=np.float32)[roots],
        "cached_actions": np.asarray(arrays["cached_actions"], dtype=np.float32)[roots],
        "cached_actions_env": np.asarray(arrays["cached_actions_env"], dtype=np.float32)[roots],
        "fresh_actions": np.asarray(arrays["fresh_actions"], dtype=np.float32)[roots, branches],
        "executed_actions": np.asarray(arrays["executed_actions"], dtype=np.float32)[roots, branches],
        "executed_valid": np.asarray(arrays["executed_valid"], dtype=np.bool_)[roots, branches],
    }


def _option_descriptor(
    trajectory: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    segment = np.asarray(trajectory[start:end, :7], dtype=np.float32)
    if not segment.size:
        raise ValueError("An option segment must not be empty.")
    delta = segment[-1] - segment[0]
    motion = np.linalg.norm(np.diff(segment[:, :6], axis=0), axis=-1)
    gripper = segment[:, 6] >= 0
    gripper_event = float(np.any(gripper[1:] != gripper[:-1]))
    return np.concatenate(
        [
            np.asarray(
                [
                    start / trajectory.shape[0],
                    end / trajectory.shape[0],
                    (end - start) / trajectory.shape[0],
                    float(np.mean(motion)) if motion.size else 0.0,
                    float(np.max(motion)) if motion.size else 0.0,
                    float(np.mean(gripper)),
                    gripper_event,
                ],
                dtype=np.float32,
            ),
            segment[0],
            segment[-1],
            np.mean(segment, axis=0),
            delta,
        ]
    ).astype(np.float32)


def compile_options(
    cached_ear: np.ndarray,
    *,
    max_options: int,
    min_length: int,
    max_length: int,
    gripper_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile every EAR into padded option descriptors and boundaries."""

    trajectories = np.asarray(cached_ear, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[-1] < 7:
        raise ValueError(f"cached_ear must be [N,H,D>=7], got {trajectories.shape}.")
    descriptor_dim = 7 + 4 * 7
    descriptors = np.zeros((trajectories.shape[0], max_options, descriptor_dim), dtype=np.float32)
    boundaries = np.zeros((trajectories.shape[0], max_options, 2), dtype=np.int64)
    valid = np.zeros((trajectories.shape[0], max_options), dtype=np.bool_)
    for sample, trajectory in enumerate(trajectories):
        segments = compression.segment_adaptive(
            trajectory[:, :7],
            min_len=min_length,
            max_len=max_length,
            max_segments=max_options,
            gripper_indices=(6,),
            gamma=gripper_weight,
        )
        if len(segments) > max_options:
            raise RuntimeError(f"Compiler returned {len(segments)} options; limit={max_options}.")
        for option_id, (start, end) in enumerate(segments):
            boundaries[sample, option_id] = (start, end)
            descriptors[sample, option_id] = _option_descriptor(trajectory, start, end)
            valid[sample, option_id] = True
    return descriptors, boundaries, valid


def _discrete_action_oracle(
    cached_actions: np.ndarray,
    target_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the cached action index with minimum continuous six-DoF MSE."""

    cached = np.asarray(cached_actions, dtype=np.float32)
    target = np.asarray(target_actions, dtype=np.float32)
    if cached.ndim != 3 or target.ndim != 2 or cached.shape[0] != target.shape[0]:
        raise ValueError(f"Expected cached [N,T,D] and target [N,D], got {cached.shape}, {target.shape}.")
    if cached.shape[-1] < 6 or target.shape[-1] < 6:
        raise ValueError("Action arrays need at least six dimensions.")
    errors = np.mean(np.square(cached[..., :6] - target[:, None, :6]), axis=-1)
    indices = np.argmin(errors, axis=1).astype(np.int64)
    return indices, np.take_along_axis(errors, indices[:, None], axis=1)[:, 0]


def _interpolate_actions(actions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    position = np.clip(np.asarray(indices, dtype=np.float32), 0.0, values.shape[1] - 1.0)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, values.shape[1] - 1)
    fraction = position - lower
    batch = np.arange(values.shape[0], dtype=np.int64)
    return values[batch, lower] + fraction[:, None] * (values[batch, upper] - values[batch, lower])


def _selector_actions(
    cached_actions: np.ndarray,
    indices: np.ndarray,
    fixed_actions: np.ndarray,
) -> np.ndarray:
    """Select six-DoF progress while preserving the fixed-age gripper mode."""

    output = _interpolate_actions(cached_actions, indices)
    output[:, 6] = np.asarray(fixed_actions, dtype=np.float32)[:, 6]
    return output


def _calibrate_continuous_index(
    cached_actions: np.ndarray,
    target_actions: np.ndarray,
    *,
    grid_size: int,
) -> float:
    """Fit one train-only index by directly minimizing aggregate six-DoF MSE."""

    cached = np.asarray(cached_actions, dtype=np.float32)
    target = np.asarray(target_actions, dtype=np.float32)
    grid = np.linspace(0.0, cached.shape[1] - 1.0, grid_size, dtype=np.float32)
    errors = np.empty((grid.size,), dtype=np.float64)
    for position, value in enumerate(grid):
        prediction = _interpolate_actions(
            cached,
            np.full((cached.shape[0],), value, dtype=np.float32),
        )
        errors[position] = np.mean(np.square(prediction[:, :6] - target[:, :6]))
    return float(grid[int(np.argmin(errors))])


def _option_ids(
    action_indices: np.ndarray,
    boundaries: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Map final-action phase to the corresponding coarse EAR option."""

    coarse_phase = np.asarray(action_indices, dtype=np.float32) / 2.0
    output = np.zeros((coarse_phase.size,), dtype=np.int64)
    for sample, phase in enumerate(coarse_phase):
        candidates = np.flatnonzero(valid[sample])
        if not candidates.size:
            raise ValueError("Every sample must contain at least one valid option.")
        selected = int(candidates[-1])
        for option_id in candidates:
            start, end = boundaries[sample, option_id]
            if float(start) <= phase < float(end):
                selected = int(option_id)
                break
        output[sample] = selected
    return output


def _structured_features(
    flat: Mapping[str, np.ndarray],
    descriptors: np.ndarray,
    valid_options: np.ndarray,
    *,
    current_equals_anchor: bool = False,
) -> np.ndarray:
    anchor = np.asarray(flat["anchor_state"], dtype=np.float32)[:, :8]
    current = (
        anchor
        if current_equals_anchor
        else np.asarray(flat["current_state"], dtype=np.float32)[:, :8]
    )
    state = np.concatenate([anchor, current, current - anchor], axis=-1)

    executed = np.asarray(flat["executed_actions"], dtype=np.float32)
    valid = np.asarray(flat["executed_valid"], dtype=np.float32)
    masked = executed * valid[..., None]
    counts = np.maximum(np.sum(valid, axis=1, keepdims=True), 1.0)
    mean_executed = np.sum(masked, axis=1) / counts
    last_index = np.maximum(np.sum(valid, axis=1).astype(np.int64) - 1, 0)
    last_executed = masked[np.arange(masked.shape[0]), last_index]
    expected = np.asarray(flat["cached_actions_env"], dtype=np.float32)[:, : executed.shape[1], :7]
    execution_mismatch = np.sum((masked - expected) * valid[..., None], axis=1)
    prefix = np.concatenate(
        [
            masked.reshape((masked.shape[0], -1)),
            valid,
            mean_executed,
            last_executed,
            execution_mismatch,
            counts / executed.shape[1],
        ],
        axis=-1,
    )

    ear = np.asarray(flat["cached_ear"], dtype=np.float32)[..., :7]
    plan = np.concatenate(
        [ear[:, 0], ear[:, min(2, ear.shape[1] - 1)], ear[:, -1], np.mean(ear, axis=1), np.std(ear, axis=1)],
        axis=-1,
    )
    action = np.asarray(flat["cached_actions"], dtype=np.float32)[..., :7]
    action_summary = np.concatenate(
        [
            action[:, min(4, action.shape[1] - 1)],
            np.mean(action, axis=1),
            np.std(action, axis=1),
        ],
        axis=-1,
    )
    option_features = np.concatenate(
        [descriptors.reshape((descriptors.shape[0], -1)), valid_options.astype(np.float32)],
        axis=-1,
    )
    return np.concatenate([state, prefix, plan, action_summary, option_features], axis=-1).astype(np.float64)


def _selected_option_features(
    base: np.ndarray,
    descriptors: np.ndarray,
    option_ids: np.ndarray,
    reference_actions: np.ndarray,
) -> np.ndarray:
    batch = np.arange(base.shape[0], dtype=np.int64)
    selected = descriptors[batch, np.asarray(option_ids, dtype=np.int64)]
    return np.concatenate([base, selected, np.asarray(reference_actions, dtype=np.float32)[:, :7]], axis=-1)


def _option_action_basis(
    cached_actions: np.ndarray,
    descriptors: np.ndarray,
    option_ids: np.ndarray,
    action_indices: np.ndarray,
) -> np.ndarray:
    """Return three plan-derived six-DoF correction directions per sample."""

    actions = np.asarray(cached_actions, dtype=np.float32)
    indices = np.asarray(action_indices, dtype=np.float32)
    reference = _interpolate_actions(actions, indices)[:, :6]
    lower = np.floor(np.clip(indices, 0.0, actions.shape[1] - 1.0)).astype(np.int64)
    previous = np.maximum(lower - 1, 0)
    following = np.minimum(lower + 1, actions.shape[1] - 1)
    batch = np.arange(actions.shape[0], dtype=np.int64)
    tangent = actions[batch, following, :6] - actions[batch, previous, :6]
    selected_descriptor = descriptors[batch, np.asarray(option_ids, dtype=np.int64)]
    option_delta = selected_descriptor[:, 28:34]
    basis = np.stack([reference, tangent, option_delta], axis=-1)
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    return basis / np.maximum(norms, 1e-6)


def _project_residual_to_basis(residual: np.ndarray, basis: np.ndarray) -> np.ndarray:
    targets = np.asarray(residual, dtype=np.float64)
    matrices = np.asarray(basis, dtype=np.float64)
    coefficients = np.empty((targets.shape[0], matrices.shape[-1]), dtype=np.float32)
    for sample in range(targets.shape[0]):
        coefficients[sample] = np.linalg.lstsq(
            matrices[sample],
            targets[sample],
            rcond=1e-4,
        )[0]
    return coefficients


def _best_binary_threshold(scores: np.ndarray, target: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(target, dtype=np.bool_)
    candidates = np.unique(np.concatenate([values, np.asarray([0.0, 0.5, 1.0], dtype=np.float32)]))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        f1 = _binary_f1(values >= threshold, labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def _fit_ridge(features: np.ndarray, targets: np.ndarray, regularization: float) -> RidgeModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim not in {1, 2} or y.shape[0] != x.shape[0]:
        raise ValueError(f"Invalid ridge shapes: features={x.shape}, targets={y.shape}.")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = (x - mean) / scale
    design = np.concatenate([normalized, np.ones((normalized.shape[0], 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeModel(mean=mean, scale=scale, coefficients=coefficients)


def _predict_ridge(model: RidgeModel, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - model.mean) / model.scale
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return np.asarray(design @ model.coefficients, dtype=np.float32)


def _branch_means(
    labels: np.ndarray,
    branch_ids: np.ndarray,
    *,
    fallback: float,
) -> np.ndarray:
    means = np.full((len(branched_dataset.BRANCH_NAMES),), fallback, dtype=np.float32)
    for branch_id in range(len(means)):
        selected = np.asarray(branch_ids) == branch_id
        if np.any(selected):
            means[branch_id] = float(np.mean(np.asarray(labels)[selected]))
    return means


def _calibrate_branch_indices(
    cached_actions: np.ndarray,
    target_actions: np.ndarray,
    branch_ids: np.ndarray,
    *,
    fallback: float,
    grid_size: int,
) -> np.ndarray:
    indices = np.full((len(branched_dataset.BRANCH_NAMES),), fallback, dtype=np.float32)
    for branch_id in range(len(indices)):
        selected = np.asarray(branch_ids) == branch_id
        if np.any(selected):
            indices[branch_id] = _calibrate_continuous_index(
                np.asarray(cached_actions)[selected],
                np.asarray(target_actions)[selected],
                grid_size=grid_size,
            )
    return indices


def _binary_f1(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.bool_)
    target = np.asarray(target, dtype=np.bool_)
    true_positive = int(np.sum(predicted & target))
    false_positive = int(np.sum(predicted & ~target))
    false_negative = int(np.sum(~predicted & target))
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 1.0


def _safe_gap_closure(value: float, stale: float, oracle: float) -> float | None:
    denominator = stale - oracle
    if denominator <= 1e-12:
        return None
    return float((stale - value) / denominator)


def _metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    fixed: np.ndarray,
    oracle: np.ndarray,
    predicted_index: np.ndarray,
    oracle_index: np.ndarray,
    predicted_option: np.ndarray,
    oracle_option: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=np.bool_)
    prediction = np.asarray(predicted, dtype=np.float32)[selected, :7]
    target_values = np.asarray(target, dtype=np.float32)[selected, :7]
    fixed_values = np.asarray(fixed, dtype=np.float32)[selected, :7]
    oracle_values = np.asarray(oracle, dtype=np.float32)[selected, :7]
    if not prediction.size:
        return {"count": 0}
    mse_6d = float(np.mean(np.square(prediction[:, :6] - target_values[:, :6])))
    mse_7d = float(np.mean(np.square(prediction - target_values)))
    fixed_mse_6d = float(np.mean(np.square(fixed_values[:, :6] - target_values[:, :6])))
    oracle_mse_6d = float(np.mean(np.square(oracle_values[:, :6] - target_values[:, :6])))
    target_sign = target_values[:, 6] >= 0
    predicted_sign = prediction[:, 6] >= 0
    fixed_sign = fixed_values[:, 6] >= 0
    target_event = target_sign != fixed_sign
    predicted_event = predicted_sign != fixed_sign
    return {
        "count": int(np.sum(selected)),
        "action_mse_6d": mse_6d,
        "action_mse_7d": mse_7d,
        "gap_closure_6d_fixed_to_oracle": _safe_gap_closure(mse_6d, fixed_mse_6d, oracle_mse_6d),
        "relative_improvement_6d_vs_fixed": float((fixed_mse_6d - mse_6d) / max(fixed_mse_6d, 1e-12)),
        "index_mae": float(np.mean(np.abs(np.asarray(predicted_index)[selected] - np.asarray(oracle_index)[selected]))),
        "option_accuracy": float(
            np.mean(np.asarray(predicted_option)[selected] == np.asarray(oracle_option)[selected])
        ),
        "gripper_sign_accuracy": float(np.mean(predicted_sign == target_sign)),
        "gripper_event_count": int(np.sum(target_event)),
        "gripper_event_f1": _binary_f1(predicted_event, target_event),
    }


def _stratified_metrics(
    predicted: np.ndarray,
    flat: Mapping[str, np.ndarray],
    *,
    fixed: np.ndarray,
    oracle: np.ndarray,
    predicted_index: np.ndarray,
    oracle_index: np.ndarray,
    predicted_option: np.ndarray,
    oracle_option: np.ndarray,
) -> dict[str, Any]:
    branches = np.asarray(flat["branch_id"], dtype=np.int64)
    target = np.asarray(flat["fresh_actions"], dtype=np.float32)[:, 0]

    def evaluate(mask: np.ndarray) -> dict[str, Any]:
        return _metrics(
            predicted,
            target,
            fixed,
            oracle,
            predicted_index,
            oracle_index,
            predicted_option,
            oracle_option,
            mask,
        )

    return {
        "overall": evaluate(np.ones((branches.size,), dtype=np.bool_)),
        "nominal_disturbed": {
            "nominal": evaluate(branches == 0),
            "disturbed": evaluate(branches != 0),
        },
        "by_branch": {
            name: evaluate(branches == branch_id)
            for branch_id, name in enumerate(branched_dataset.BRANCH_NAMES)
        },
    }


def _same_task_shuffle(flat: Mapping[str, np.ndarray], *, seed: int) -> np.ndarray:
    tasks = np.asarray(flat["task_id"], dtype=np.int64)
    roots = np.asarray(flat["root_id"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    mapping = np.arange(tasks.size, dtype=np.int64)
    for task in np.unique(tasks):
        selected = np.flatnonzero(tasks == task)
        if selected.size < 2:
            continue
        donors = selected.copy()
        for _ in range(32):
            rng.shuffle(donors)
            if np.all(roots[donors] != roots[selected]):
                break
        mapping[selected] = donors
    return mapping


def _prepare(
    flat: dict[str, np.ndarray],
    *,
    args: Args,
) -> dict[str, Any]:
    descriptors, boundaries, valid_options = compile_options(
        flat["cached_ear"],
        max_options=args.max_options,
        min_length=args.min_option_length,
        max_length=args.max_option_length,
        gripper_weight=args.gripper_boundary_weight,
    )
    target = np.asarray(flat["fresh_actions"], dtype=np.float32)[:, 0]
    oracle_index, oracle_error = _discrete_action_oracle(flat["cached_actions"], target)
    oracle_option = _option_ids(oracle_index, boundaries, valid_options)
    fixed_index = np.sum(np.asarray(flat["executed_valid"], dtype=np.int64), axis=1)
    fixed_index = np.clip(fixed_index, 0, flat["cached_actions"].shape[1] - 1)
    return {
        "descriptors": descriptors,
        "boundaries": boundaries,
        "valid_options": valid_options,
        "base_features": _structured_features(flat, descriptors, valid_options),
        "oracle_index": oracle_index,
        "oracle_error_6d": oracle_error,
        "oracle_option": oracle_option,
        "fixed_index": fixed_index,
        "fixed_actions": _interpolate_actions(flat["cached_actions"], fixed_index),
        "oracle_actions": _interpolate_actions(flat["cached_actions"], oracle_index),
        "target": target,
    }


def _fit_structured_feedback(
    flat: Mapping[str, np.ndarray],
    prepared: Mapping[str, Any],
    *,
    regularization: float,
) -> StructuredFeedbackModel:
    reference = _selector_actions(
        flat["cached_actions"],
        prepared["oracle_index"],
        prepared["fixed_actions"],
    )
    option_ids = np.asarray(prepared["oracle_option"], dtype=np.int64)
    features = _selected_option_features(
        prepared["base_features"],
        prepared["descriptors"],
        option_ids,
        reference,
    )
    basis = _option_action_basis(
        flat["cached_actions"],
        prepared["descriptors"],
        option_ids,
        prepared["oracle_index"],
    )
    target = np.asarray(prepared["target"], dtype=np.float32)
    coefficients = _project_residual_to_basis(target[:, :6] - reference[:, :6], basis)
    coefficient_clip = np.maximum(np.quantile(np.abs(coefficients), 0.99, axis=0), 1e-3)
    coefficient_model = _fit_ridge(features, coefficients, regularization)

    target_sign = target[:, 6] >= 0
    fixed_sign = np.asarray(prepared["fixed_actions"], dtype=np.float32)[:, 6] >= 0
    event_target = target_sign != fixed_sign
    gripper_event_model = _fit_ridge(features, event_target.astype(np.float32), regularization)
    event_scores = _predict_ridge(gripper_event_model, features)
    threshold = _best_binary_threshold(event_scores, event_target)
    return StructuredFeedbackModel(
        coefficient_model=coefficient_model,
        coefficient_clip=coefficient_clip,
        gripper_event_model=gripper_event_model,
        gripper_event_threshold=threshold,
    )


def _predict_feedback(
    flat: Mapping[str, np.ndarray],
    prepared: Mapping[str, Any],
    *,
    selector_index: np.ndarray,
    feedback_model: StructuredFeedbackModel,
    base_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference = _selector_actions(
        flat["cached_actions"],
        selector_index,
        prepared["fixed_actions"],
    )
    option = _option_ids(
        selector_index,
        prepared["boundaries"],
        prepared["valid_options"],
    )
    base = prepared["base_features"] if base_features is None else base_features
    features = _selected_option_features(base, prepared["descriptors"], option, reference)
    basis = _option_action_basis(
        flat["cached_actions"],
        prepared["descriptors"],
        option,
        selector_index,
    )
    coefficients = _predict_ridge(feedback_model.coefficient_model, features)
    coefficients = np.clip(
        coefficients,
        -feedback_model.coefficient_clip,
        feedback_model.coefficient_clip,
    )
    residual = np.einsum("ndk,nk->nd", basis, coefficients)
    event_score = _predict_ridge(feedback_model.gripper_event_model, features)
    gripper_event = event_score >= feedback_model.gripper_event_threshold
    output = np.asarray(reference, dtype=np.float32).copy()
    output[:, :6] += residual
    output[gripper_event, 6] *= -1.0
    return output, option


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {summary_path}")

    arrays = branched_dataset.load_branched_arrays(
        args.dataset,
        fields=(
            "root_id",
            "task_id",
            "episode_id",
            "branch_valid",
            "anchor_state",
            "current_state",
            "cached_ear",
            "cached_actions",
            "cached_actions_env",
            "fresh_actions",
            "executed_actions",
            "executed_valid",
        ),
    )
    train_roots, validation_roots, test_roots = _split_roots(
        arrays,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_flat = _flatten_valid(arrays, train_roots)
    validation_flat = _flatten_valid(arrays, validation_roots)
    test_flat = _flatten_valid(arrays, test_roots)
    train = _prepare(train_flat, args=args)
    validation = _prepare(validation_flat, args=args)
    test = _prepare(test_flat, args=args)

    global_index = _calibrate_continuous_index(
        train_flat["cached_actions"],
        train["target"],
        grid_size=args.calibration_grid_size,
    )
    branch_index = _calibrate_branch_indices(
        train_flat["cached_actions"],
        train["target"],
        train_flat["branch_id"],
        fallback=global_index,
        grid_size=args.calibration_grid_size,
    )

    disturbed_validation = np.asarray(validation_flat["branch_id"], dtype=np.int64) != 0
    ridge_selection: dict[str, float] = {}
    no_current_ridge_selection: dict[str, float] = {}
    selector_candidates: dict[float, RidgeModel] = {}
    no_current_selector_candidates: dict[float, RidgeModel] = {}
    train_no_current_features = _structured_features(
        train_flat,
        train["descriptors"],
        train["valid_options"],
        current_equals_anchor=True,
    )
    validation_no_current_features = _structured_features(
        validation_flat,
        validation["descriptors"],
        validation["valid_options"],
        current_equals_anchor=True,
    )
    for regularization in args.ridge_grid:
        candidate = _fit_ridge(
            train["base_features"],
            train["oracle_index"],
            regularization,
        )
        selector_candidates[regularization] = candidate
        candidate_index = np.clip(
            _predict_ridge(candidate, validation["base_features"]),
            0.0,
            validation_flat["cached_actions"].shape[1] - 1.0,
        )
        candidate_action = _interpolate_actions(validation_flat["cached_actions"], candidate_index)
        candidate_error = candidate_action[disturbed_validation, :6] - validation["target"][
            disturbed_validation, :6
        ]
        ridge_selection[str(regularization)] = float(np.mean(np.square(candidate_error)))
        no_current_candidate = _fit_ridge(
            train_no_current_features,
            train["oracle_index"],
            regularization,
        )
        no_current_selector_candidates[regularization] = no_current_candidate
        no_current_index = np.clip(
            _predict_ridge(no_current_candidate, validation_no_current_features),
            0.0,
            validation_flat["cached_actions"].shape[1] - 1.0,
        )
        no_current_action = _interpolate_actions(validation_flat["cached_actions"], no_current_index)
        no_current_error = no_current_action[disturbed_validation, :6] - validation["target"][
            disturbed_validation, :6
        ]
        no_current_ridge_selection[str(regularization)] = float(np.mean(np.square(no_current_error)))
    selected_ridge = min(
        args.ridge_grid,
        key=lambda value: (ridge_selection[str(value)], value),
    )
    selected_no_current_ridge = min(
        args.ridge_grid,
        key=lambda value: (no_current_ridge_selection[str(value)], value),
    )
    selector_model = selector_candidates[selected_ridge]
    no_current_selector_model = no_current_selector_candidates[selected_no_current_ridge]
    feedback_model = _fit_structured_feedback(
        train_flat,
        train,
        regularization=selected_ridge,
    )
    train_selector = np.clip(
        _predict_ridge(selector_model, train["base_features"]),
        0.0,
        train_flat["cached_actions"].shape[1] - 1.0,
    )

    def evaluate_partition(
        name: str,
        flat: dict[str, np.ndarray],
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        learned_index = np.clip(
            _predict_ridge(selector_model, prepared["base_features"]),
            0.0,
            flat["cached_actions"].shape[1] - 1.0,
        )
        global_indices = np.full_like(learned_index, global_index)
        privileged_indices = branch_index[np.asarray(flat["branch_id"], dtype=np.int64)]
        learned_selector_action = _selector_actions(
            flat["cached_actions"],
            learned_index,
            prepared["fixed_actions"],
        )
        learned_selector_option = _option_ids(
            learned_index,
            prepared["boundaries"],
            prepared["valid_options"],
        )
        learned_action, learned_option = _predict_feedback(
            flat,
            prepared,
            selector_index=learned_index,
            feedback_model=feedback_model,
        )
        privileged_action, privileged_option = _predict_feedback(
            flat,
            prepared,
            selector_index=prepared["oracle_index"],
            feedback_model=feedback_model,
        )
        branch_action, branch_option = _predict_feedback(
            flat,
            prepared,
            selector_index=privileged_indices,
            feedback_model=feedback_model,
        )
        global_action = _selector_actions(
            flat["cached_actions"],
            global_indices,
            prepared["fixed_actions"],
        )
        global_feedback_action, global_feedback_option = _predict_feedback(
            flat,
            prepared,
            selector_index=global_indices,
            feedback_model=feedback_model,
        )
        global_option = _option_ids(
            global_indices,
            prepared["boundaries"],
            prepared["valid_options"],
        )
        fixed_option = _option_ids(
            prepared["fixed_index"],
            prepared["boundaries"],
            prepared["valid_options"],
        )

        baselines = {
            "fixed_age_cached_action": (
                prepared["fixed_actions"],
                prepared["fixed_index"],
                fixed_option,
            ),
            "train_global_index": (global_action, global_indices, global_option),
            "train_global_index_structured_feedback": (
                global_feedback_action,
                global_indices,
                global_feedback_option,
            ),
            "privileged_branch_calibrated_index_feedback": (
                branch_action,
                privileged_indices,
                branch_option,
            ),
            "learned_structured_selector_only": (
                learned_selector_action,
                learned_index,
                learned_selector_option,
            ),
            "learned_structured_event_option_feedback_diagnostic": (
                learned_action,
                learned_index,
                learned_option,
            ),
            "privileged_discrete_option_feedback": (
                privileged_action,
                prepared["oracle_index"],
                privileged_option,
            ),
            "discrete_cached_action_oracle": (
                prepared["oracle_actions"],
                prepared["oracle_index"],
                prepared["oracle_option"],
            ),
        }
        results = {
            baseline_name: _stratified_metrics(
                prediction,
                flat,
                fixed=prepared["fixed_actions"],
                oracle=prepared["oracle_actions"],
                predicted_index=indices,
                oracle_index=prepared["oracle_index"],
                predicted_option=options,
                oracle_option=prepared["oracle_option"],
            )
            for baseline_name, (prediction, indices, options) in baselines.items()
        }

        ablated_base = _structured_features(
            flat,
            prepared["descriptors"],
            prepared["valid_options"],
            current_equals_anchor=True,
        )
        ablated_index = np.clip(
            _predict_ridge(no_current_selector_model, ablated_base),
            0.0,
            flat["cached_actions"].shape[1] - 1.0,
        )
        ablated_selector_action = _selector_actions(
            flat["cached_actions"],
            ablated_index,
            prepared["fixed_actions"],
        )
        ablated_selector_option = _option_ids(
            ablated_index,
            prepared["boundaries"],
            prepared["valid_options"],
        )
        ablated_action, ablated_option = _predict_feedback(
            flat,
            prepared,
            selector_index=ablated_index,
            feedback_model=feedback_model,
            base_features=ablated_base,
        )
        results["learned_selector_current_equals_anchor"] = _stratified_metrics(
            ablated_selector_action,
            flat,
            fixed=prepared["fixed_actions"],
            oracle=prepared["oracle_actions"],
            predicted_index=ablated_index,
            oracle_index=prepared["oracle_index"],
            predicted_option=ablated_selector_option,
            oracle_option=prepared["oracle_option"],
        )
        results["learned_feedback_current_equals_anchor_ood_diagnostic"] = _stratified_metrics(
            ablated_action,
            flat,
            fixed=prepared["fixed_actions"],
            oracle=prepared["oracle_actions"],
            predicted_index=ablated_index,
            oracle_index=prepared["oracle_index"],
            predicted_option=ablated_option,
            oracle_option=prepared["oracle_option"],
        )

        if name == "test":
            donors = _same_task_shuffle(flat, seed=args.seed + 17)
            shuffled_flat = dict(flat)
            for field in ("cached_ear", "cached_actions", "cached_actions_env"):
                shuffled_flat[field] = np.asarray(flat[field])[donors]
            shuffled = _prepare(shuffled_flat, args=args)
            shuffled_index = np.clip(
                _predict_ridge(selector_model, shuffled["base_features"]),
                0.0,
                shuffled_flat["cached_actions"].shape[1] - 1.0,
            )
            shuffled_selector_action = _selector_actions(
                shuffled_flat["cached_actions"],
                shuffled_index,
                shuffled["fixed_actions"],
            )
            shuffled_selector_option = _option_ids(
                shuffled_index,
                shuffled["boundaries"],
                shuffled["valid_options"],
            )
            shuffled_action, shuffled_option = _predict_feedback(
                shuffled_flat,
                shuffled,
                selector_index=shuffled_index,
                feedback_model=feedback_model,
            )
            results["learned_selector_same_task_shuffled_option"] = _stratified_metrics(
                shuffled_selector_action,
                flat,
                fixed=prepared["fixed_actions"],
                oracle=prepared["oracle_actions"],
                predicted_index=shuffled_index,
                oracle_index=prepared["oracle_index"],
                predicted_option=shuffled_selector_option,
                oracle_option=prepared["oracle_option"],
            )
            results["learned_feedback_same_task_shuffled_option_diagnostic"] = _stratified_metrics(
                shuffled_action,
                flat,
                fixed=prepared["fixed_actions"],
                oracle=prepared["oracle_actions"],
                predicted_index=shuffled_index,
                oracle_index=prepared["oracle_index"],
                predicted_option=shuffled_option,
                oracle_option=prepared["oracle_option"],
            )
        return results

    partition_results = {
        "validation": evaluate_partition("validation", validation_flat, validation),
        "test": evaluate_partition("test", test_flat, test),
    }
    test_metrics = partition_results["test"]
    fixed_disturbed = test_metrics["fixed_age_cached_action"]["nominal_disturbed"]["disturbed"]
    oracle_disturbed = test_metrics["discrete_cached_action_oracle"]["nominal_disturbed"]["disturbed"]
    learned_disturbed = test_metrics["learned_structured_selector_only"]["nominal_disturbed"]["disturbed"]
    global_disturbed = test_metrics["train_global_index"]["nominal_disturbed"]["disturbed"]
    learned_nominal = test_metrics["learned_structured_selector_only"]["nominal_disturbed"]["nominal"]
    fixed_nominal = test_metrics["fixed_age_cached_action"]["nominal_disturbed"]["nominal"]
    ablated_disturbed = test_metrics["learned_selector_current_equals_anchor"]["nominal_disturbed"][
        "disturbed"
    ]
    shuffled_disturbed = test_metrics["learned_selector_same_task_shuffled_option"]["nominal_disturbed"][
        "disturbed"
    ]
    feedback_disturbed = test_metrics["learned_structured_event_option_feedback_diagnostic"][
        "nominal_disturbed"
    ]["disturbed"]
    global_feedback_disturbed = test_metrics["train_global_index_structured_feedback"][
        "nominal_disturbed"
    ]["disturbed"]

    def relative_gain(reference: float, value: float) -> float:
        return float((reference - value) / max(reference, 1e-12))

    gate_values = {
        "oracle_gap_closure_disturbed_6d": _safe_gap_closure(
            oracle_disturbed["action_mse_6d"],
            fixed_disturbed["action_mse_6d"],
            0.0,
        ),
        "learned_gap_closure_fixed_to_oracle_disturbed_6d": learned_disturbed[
            "gap_closure_6d_fixed_to_oracle"
        ],
        "learned_gain_vs_global_disturbed_6d": relative_gain(
            global_disturbed["action_mse_6d"],
            learned_disturbed["action_mse_6d"],
        ),
        "learned_nominal_mse_ratio_vs_fixed": float(
            learned_nominal["action_mse_7d"] / max(fixed_nominal["action_mse_7d"], 1e-12)
        ),
        "current_state_gain_disturbed_6d": relative_gain(
            ablated_disturbed["action_mse_6d"],
            learned_disturbed["action_mse_6d"],
        ),
        "shuffled_plan_degradation_disturbed_6d": float(
            shuffled_disturbed["action_mse_6d"] / max(learned_disturbed["action_mse_6d"], 1e-12) - 1.0
        ),
        "structured_feedback_gain_vs_global_feedback_disturbed_6d": relative_gain(
            global_feedback_disturbed["action_mse_6d"],
            feedback_disturbed["action_mse_6d"],
        ),
        "structured_gripper_event_f1": feedback_disturbed["gripper_event_f1"],
    }
    checks = {
        "cached_recovery_exists": gate_values["oracle_gap_closure_disturbed_6d"] >= 0.50,
        "learned_closes_gap": (
            gate_values["learned_gap_closure_fixed_to_oracle_disturbed_6d"] is not None
            and gate_values["learned_gap_closure_fixed_to_oracle_disturbed_6d"] >= 0.25
        ),
        "beats_global": gate_values["learned_gain_vs_global_disturbed_6d"] >= 0.10,
        "preserves_nominal": gate_values["learned_nominal_mse_ratio_vs_fixed"] <= 1.05,
        "uses_current_state": gate_values["current_state_gain_disturbed_6d"] >= 0.05,
        "uses_option": gate_values["shuffled_plan_degradation_disturbed_6d"] >= 0.10,
        "structured_feedback_beats_matched_global": (
            gate_values["structured_feedback_gain_vs_global_feedback_disturbed_6d"] >= 0.10
        ),
        "gripper_event": gate_values["structured_gripper_event_f1"] > 0.488,
    }
    summary = {
        "protocol": {
            "method": "structured_event_option_offline_probe",
            "split": "task-stratified and episode-disjoint",
            "target": "fresh normalized final action at the disturbed endpoint",
            "fixed_reference": "cached normalized final action after the actually executed prefix length",
            "oracle": "discrete search over the ten cached final actions using six-DoF action MSE",
            "compiler": "adaptive EAR segmentation; executable actions always come from the final-action chunk",
            "note": "Offline teacher-fidelity probe only; not a LIBERO success-rate result.",
        },
        "counts": {
            "roots": {
                "train": int(train_roots.size),
                "validation": int(validation_roots.size),
                "test": int(test_roots.size),
            },
            "branches": {
                "train": int(train_flat["branch_id"].size),
                "validation": int(validation_flat["branch_id"].size),
                "test": int(test_flat["branch_id"].size),
            },
        },
        "train_calibration": {
            "calibration_grid_size": args.calibration_grid_size,
            "global_action_index": global_index,
            "branch_mse_optimal_action_index": {
                name: float(branch_index[index])
                for index, name in enumerate(branched_dataset.BRANCH_NAMES)
            },
            "ridge_validation_disturbed_mse_6d": ridge_selection,
            "selected_ridge": selected_ridge,
            "no_current_ridge_validation_disturbed_mse_6d": no_current_ridge_selection,
            "selected_no_current_ridge": selected_no_current_ridge,
            "selector_train_index_mae": float(np.mean(np.abs(train_selector - train["oracle_index"]))),
            "feedback_constraint": (
                "six-DoF residual is restricted to reference-action, local-tangent, and EAR-option-delta "
                "directions; gripper is an independent event-flip head"
            ),
        },
        "results": partition_results,
        "gates": {
            "values": gate_values,
            "checks": checks,
            "go": all(checks.values()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
