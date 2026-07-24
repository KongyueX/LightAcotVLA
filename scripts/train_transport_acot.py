"""Train and evaluate observation-conditioned temporal transport for Action-CoT.

The exported multirate windows contain four fresh, same-seed Action-CoT
inferences.  This script turns each window into synthetic time-warp pairs:

* frame zero is always the cached-plan anchor;
* ``elapsed_age`` is the wall-clock cache age in ``{1, 2, 3}``;
* ``physical_progress`` is sampled from
  ``{elapsed_age - 1, elapsed_age, elapsed_age + 1} intersect [0, 3]``;
* the current observation and targets come from ``physical_progress`` while
  the model receives ``elapsed_age``.

Consequently, the model cannot solve every example by advancing the cached
EAR at a fixed rate.  Validation and test evaluation report both nominal
``physical_progress == elapsed_age`` samples and all legal time-warp pairs.
Splits are task-stratified and episode-disjoint.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import json
import pathlib
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.action_cot import multirate_dataset
from openpi.models import transported_action_cot
import optax
import orbax.checkpoint as ocp
import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    seed: int = 7
    train_steps: int = 1_500
    batch_size: int = 128
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    log_interval: int = 100
    early_stopping_patience_logs: int = 5
    early_stopping_min_delta: float = 1e-5
    action_loss_weight: float = 1.0
    transport_loss_weight: float = 1.0
    phase_loss_weight: float = 0.25
    huber_delta: float = 0.1
    gripper_weight: float = 4.0
    minimum_action_residual_scale: float = 0.5
    output_margin: float = 1.10
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class PairIndices:
    windows: np.ndarray
    elapsed_age: np.ndarray
    physical_progress: np.ndarray

    def __post_init__(self) -> None:
        windows = np.asarray(self.windows)
        elapsed_age = np.asarray(self.elapsed_age)
        physical_progress = np.asarray(self.physical_progress)
        if windows.ndim != 1 or elapsed_age.shape != windows.shape or physical_progress.shape != windows.shape:
            raise ValueError("Time-warp pair arrays must be matching rank-one arrays.")
        if windows.dtype.kind not in {"i", "u"}:
            raise TypeError("windows must contain integer indices.")
        if elapsed_age.dtype.kind not in {"i", "u"} or physical_progress.dtype.kind not in {"i", "u"}:
            raise TypeError("elapsed_age and physical_progress must contain integers.")
        if np.any((elapsed_age < 1) | (elapsed_age > 3)):
            raise ValueError("elapsed_age must be in [1, 3].")
        legal = (
            (physical_progress >= 0)
            & (physical_progress <= 3)
            & (np.abs(physical_progress - elapsed_age) <= 1)
        )
        if not np.all(legal):
            raise ValueError("physical_progress must be a legal local time warp around elapsed_age.")

    def __len__(self) -> int:
        return int(self.windows.size)

    def take(self, indices: np.ndarray | slice) -> PairIndices:
        return PairIndices(
            windows=self.windows[indices],
            elapsed_age=self.elapsed_age[indices],
            physical_progress=self.physical_progress[indices],
        )


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.train_steps <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("train_steps and batch sizes must be positive.")
    if args.log_interval <= 0:
        raise ValueError("log_interval must be positive.")
    if args.early_stopping_patience_logs < 0 or args.early_stopping_min_delta < 0:
        raise ValueError("Early-stopping settings must be non-negative.")
    if (
        args.learning_rate <= 0
        or args.weight_decay < 0
        or args.gradient_clip_norm <= 0
        or args.huber_delta <= 0
        or args.gripper_weight <= 0
        or args.minimum_action_residual_scale <= 0
        or args.output_margin <= 1.0
    ):
        raise ValueError(
            "Optimizer/loss scales must be positive, weight_decay non-negative, and output_margin greater than one."
        )
    if args.action_loss_weight < 0 or args.transport_loss_weight < 0 or args.phase_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.action_loss_weight + args.transport_loss_weight + args.phase_loss_weight <= 0:
        raise ValueError("At least one loss weight must be positive.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must be in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")


def _split_windows(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return task-stratified, episode-disjoint train/validation/test windows."""

    tasks = np.asarray(arrays["task_id"], dtype=np.int64)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for task_id in np.unique(tasks):
        task_indices = np.flatnonzero(tasks == task_id)
        task_episodes = np.unique(episodes[task_indices])
        if task_episodes.size < 3:
            raise ValueError(f"Task {task_id} needs at least three episodes for train/val/test.")
        rng.shuffle(task_episodes)
        test_count = max(1, round(task_episodes.size * test_fraction))
        validation_count = max(1, round(task_episodes.size * validation_fraction))
        if test_count + validation_count >= task_episodes.size:
            validation_count = 1
            test_count = 1
        test_episodes = task_episodes[:test_count]
        validation_episodes = task_episodes[test_count : test_count + validation_count]
        test.append(task_indices[np.isin(episodes[task_indices], test_episodes)])
        validation.append(task_indices[np.isin(episodes[task_indices], validation_episodes)])
        train.append(
            task_indices[
                ~np.isin(
                    episodes[task_indices],
                    np.concatenate([test_episodes, validation_episodes]),
                )
            ]
        )
    outputs = tuple(np.sort(np.concatenate(parts)) for parts in (train, validation, test))
    if any(not values.size for values in outputs):
        raise ValueError("Episode-level split produced an empty partition.")
    return outputs  # type: ignore[return-value]


def _nominal_pairs(window_indices: np.ndarray) -> PairIndices:
    windows = np.repeat(np.asarray(window_indices, dtype=np.int64), 3)
    elapsed_age = np.tile(np.arange(1, 4, dtype=np.int64), window_indices.size)
    return PairIndices(
        windows=windows,
        elapsed_age=elapsed_age,
        physical_progress=elapsed_age.copy(),
    )


def _all_time_warp_pairs(window_indices: np.ndarray) -> PairIndices:
    triples: list[tuple[int, int, int]] = []
    for window in np.asarray(window_indices, dtype=np.int64):
        for elapsed_age in range(1, 4):
            lower = max(0, elapsed_age - 1)
            upper = min(3, elapsed_age + 1)
            triples.extend((int(window), elapsed_age, progress) for progress in range(lower, upper + 1))
    values = np.asarray(triples, dtype=np.int64)
    return PairIndices(
        windows=values[:, 0],
        elapsed_age=values[:, 1],
        physical_progress=values[:, 2],
    )


def _sample_training_pairs(
    window_indices: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> PairIndices:
    windows = rng.choice(
        np.asarray(window_indices, dtype=np.int64),
        size=batch_size,
        replace=window_indices.size < batch_size,
    )
    elapsed_age = rng.integers(1, 4, size=batch_size, dtype=np.int64)
    physical_progress = np.empty_like(elapsed_age)
    for index, age in enumerate(elapsed_age):
        physical_progress[index] = rng.integers(max(0, int(age) - 1), min(3, int(age) + 1) + 1)
    return PairIndices(
        windows=windows,
        elapsed_age=elapsed_age,
        physical_progress=physical_progress,
    )


def _batch(arrays: dict[str, np.ndarray], pairs: PairIndices) -> dict[str, jax.Array]:
    windows = pairs.windows
    progress = pairs.physical_progress
    return {
        "anchor_images": jnp.asarray(arrays["images"][windows, 0].astype(np.float32) / 255.0),
        "current_images": jnp.asarray(arrays["images"][windows, progress].astype(np.float32) / 255.0),
        "anchor_state": jnp.asarray(arrays["states"][windows, 0], dtype=jnp.float32),
        "current_state": jnp.asarray(arrays["states"][windows, progress], dtype=jnp.float32),
        "cached_ear": jnp.asarray(arrays["fresh_ear"][windows, 0], dtype=jnp.float32),
        "cached_iar": jnp.asarray(arrays["fresh_iar"][windows, 0], dtype=jnp.float32),
        "cache_age": jnp.asarray(pairs.elapsed_age, dtype=jnp.int32),
        "target_action": jnp.asarray(arrays["teacher_actions"][windows, progress], dtype=jnp.float32),
        "target_ear": jnp.asarray(arrays["fresh_ear"][windows, progress], dtype=jnp.float32),
        "phase_label": jnp.asarray(progress.astype(np.float32) / 2.0),
    }


def _huber(error: jax.Array, delta: float) -> jax.Array:
    absolute = jnp.abs(error)
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * jnp.square(quadratic) + delta * (absolute - quadratic)


def _weighted_huber_7d(
    predicted: jax.Array,
    target: jax.Array,
    *,
    delta: float,
    gripper_weight: float,
) -> jax.Array:
    if predicted.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {predicted.shape} != {target.shape}.")
    if predicted.shape[-1] < 7:
        raise ValueError(f"Action tensors must have at least seven dimensions, got {predicted.shape}.")
    values = _huber(predicted[..., :7] - target[..., :7], delta)
    weights = jnp.ones((7,), dtype=values.dtype).at[6].set(gripper_weight)
    weighted = values * weights
    denominator = np.prod(values.shape[:-1], dtype=np.int64) * (6.0 + gripper_weight)
    return jnp.sum(weighted) / jnp.asarray(denominator, dtype=values.dtype)


def _loss(
    predicted_action: jax.Array,
    predicted_ear: jax.Array,
    predicted_phase: jax.Array,
    batch: dict[str, jax.Array],
    *,
    args: Args,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    action_loss = _weighted_huber_7d(
        predicted_action,
        batch["target_action"],
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    # Supervise all EAR time tokens, but only the seven physical action
    # dimensions.  Remaining padded dimensions must not dominate transport.
    transport_loss = _weighted_huber_7d(
        predicted_ear,
        batch["target_ear"],
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    if predicted_phase.ndim != 2 or predicted_phase.shape[0] != batch["phase_label"].shape[0]:
        raise ValueError(
            "Predicted phase must have shape [batch, EAR horizon], got "
            f"{predicted_phase.shape} for labels {batch['phase_label'].shape}."
        )
    # The explicit time-warp label supervises the transported plan's starting
    # phase.  The full-EAR loss supplies the corresponding speed/shape signal.
    phase_start = predicted_phase[:, 0]
    phase_loss = jnp.mean(jnp.square(phase_start - batch["phase_label"]))
    total = (
        args.action_loss_weight * action_loss
        + args.transport_loss_weight * transport_loss
        + args.phase_loss_weight * phase_loss
    )
    return total, {
        "action_huber": action_loss,
        "transport_huber": transport_loss,
        "phase_mse": phase_loss,
        "total": total,
    }


def _transport_cached_ear(cached_ear: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Shift an EAR forward by a continuous coarse-token phase."""

    cached_ear = np.asarray(cached_ear, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32).reshape(-1)
    if cached_ear.ndim != 3 or cached_ear.shape[0] != phase.size:
        raise ValueError(f"Expected cached EAR [N,H,D] and phase [N], got {cached_ear.shape} and {phase.shape}.")
    horizon = cached_ear.shape[1]
    positions = phase[:, None] + np.arange(horizon, dtype=np.float32)[None, :]
    lower = np.minimum(np.floor(positions).astype(np.int64), horizon - 1)
    upper = np.minimum(lower + 1, horizon - 1)
    interpolation = positions - lower
    lower_values = np.take_along_axis(cached_ear, lower[..., None], axis=1)
    upper_values = np.take_along_axis(cached_ear, upper[..., None], axis=1)
    return lower_values + interpolation[..., None] * (upper_values - lower_values)


def _training_ranges(
    arrays: dict[str, np.ndarray],
    train_windows: np.ndarray,
    *,
    minimum_action_residual_scale: float,
    output_margin: float,
) -> dict[str, float]:
    """Calibrate bounded action outputs using training windows only."""

    pairs = _all_time_warp_pairs(train_windows)
    cached_ear = arrays["fresh_ear"][pairs.windows, 0].astype(np.float32)
    oracle_transport = _transport_cached_ear(
        cached_ear,
        pairs.physical_progress.astype(np.float32) / 2.0,
    )
    target_action = arrays["teacher_actions"][pairs.windows, pairs.physical_progress].astype(np.float32)
    action_target_max_abs_7d = float(np.max(np.abs(target_action[..., :7])))
    action_residual_max_abs_7d = float(
        np.max(np.abs(target_action[..., :7] - oracle_transport[:, 0, :7]))
    )
    transport_target = arrays["fresh_ear"][pairs.windows, pairs.physical_progress].astype(np.float32)
    transport_target_max_abs_7d = float(np.max(np.abs(transport_target[..., :7])))
    effective_action_residual_scale = max(
        minimum_action_residual_scale,
        output_margin * max(action_target_max_abs_7d, action_residual_max_abs_7d),
    )
    return {
        "action_target_max_abs_7d": action_target_max_abs_7d,
        "action_residual_max_abs_7d": action_residual_max_abs_7d,
        "transport_target_max_abs_7d": transport_target_max_abs_7d,
        "suggested_action_residual_scale": effective_action_residual_scale,
    }


def _save_params(params: nnx.State, target: pathlib.Path, *, overwrite: bool) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {"transported_action_cot": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=overwrite)


def _numpy_huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    return 0.5 * np.square(quadratic) + delta * (absolute - quadratic)


def _numpy_weighted_huber_7d(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    delta: float,
    gripper_weight: float,
) -> float:
    error = np.asarray(predicted, dtype=np.float32)[..., :7] - np.asarray(target, dtype=np.float32)[..., :7]
    values = _numpy_huber(error, delta)
    weights = np.ones((7,), dtype=np.float32)
    weights[6] = gripper_weight
    return float(np.sum(values * weights) / (np.prod(values.shape[:-1]) * np.sum(weights)))


def _action_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    error = predicted[..., :7] - target[..., :7]
    predicted_gripper = predicted[..., 6] >= 0
    target_gripper = target[..., 6] >= 0
    return {
        "mse_7d": float(np.mean(np.square(error))),
        "translation_mse": float(np.mean(np.square(error[..., :3]))),
        "rotation_mse": float(np.mean(np.square(error[..., 3:6]))),
        "gripper_mse": float(np.mean(np.square(error[..., 6]))),
        "gripper_sign_accuracy": float(np.mean(predicted_gripper == target_gripper)),
        "count": int(error.reshape((-1, error.shape[-1])).shape[0]),
    }


def _prediction_metrics(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    pairs: PairIndices,
    *,
    args: Args,
) -> dict[str, Any]:
    target_action = arrays["teacher_actions"][pairs.windows, pairs.physical_progress].astype(np.float32)
    target_ear = arrays["fresh_ear"][pairs.windows, pairs.physical_progress].astype(np.float32)
    phase_label = pairs.physical_progress.astype(np.float32) / 2.0
    predicted_phase = np.asarray(predictions["phase"], dtype=np.float32)
    if predicted_phase.ndim != 2 or predicted_phase.shape[0] != len(pairs):
        raise ValueError(
            f"Predicted phase must have shape [pairs, EAR horizon], got {predicted_phase.shape}."
        )
    phase = predicted_phase[:, 0]
    action_huber = _numpy_weighted_huber_7d(
        predictions["action"],
        target_action,
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    transport_huber = _numpy_weighted_huber_7d(
        predictions["ear"],
        target_ear,
        delta=args.huber_delta,
        gripper_weight=args.gripper_weight,
    )
    phase_mse = float(np.mean(np.square(phase - phase_label)))
    total = (
        args.action_loss_weight * action_huber
        + args.transport_loss_weight * transport_huber
        + args.phase_loss_weight * phase_mse
    )

    def subset(mask: np.ndarray) -> dict[str, Any]:
        action = _action_metrics(predictions["action"][mask], target_action[mask])
        return {
            **action,
            "transport_mse_7d": float(np.mean(np.square(predictions["ear"][mask, :, :7] - target_ear[mask, :, :7]))),
            "transport_token0_mse_7d": float(
                np.mean(np.square(predictions["ear"][mask, 0, :7] - target_ear[mask, 0, :7]))
            ),
            "phase_mse": float(np.mean(np.square(phase[mask] - phase_label[mask]))),
            "phase_mae": float(np.mean(np.abs(phase[mask] - phase_label[mask]))),
        }

    metrics = {
        "loss": {
            "total": total,
            "action_huber": action_huber,
            "transport_huber": transport_huber,
            "phase_mse": phase_mse,
        },
        "overall": subset(np.ones((len(pairs),), dtype=np.bool_)),
        "by_elapsed_age": {
            str(age): subset(pairs.elapsed_age == age)
            for age in range(1, 4)
            if np.any(pairs.elapsed_age == age)
        },
        "by_progress_offset": {
            str(offset): subset((pairs.physical_progress - pairs.elapsed_age) == offset)
            for offset in (-1, 0, 1)
            if np.any((pairs.physical_progress - pairs.elapsed_age) == offset)
        },
    }
    return metrics


def _transport_baseline_metrics(
    predicted_ear: np.ndarray,
    predicted_phase: np.ndarray,
    arrays: dict[str, np.ndarray],
    pairs: PairIndices,
) -> dict[str, Any]:
    target_action = arrays["teacher_actions"][pairs.windows, pairs.physical_progress].astype(np.float32)
    target_ear = arrays["fresh_ear"][pairs.windows, pairs.physical_progress].astype(np.float32)
    phase_label = pairs.physical_progress.astype(np.float32) / 2.0
    return {
        **_action_metrics(predicted_ear[:, 0], target_action),
        "transport_mse_7d": float(np.mean(np.square(predicted_ear[..., :7] - target_ear[..., :7]))),
        "transport_token0_mse_7d": float(
            np.mean(np.square(predicted_ear[:, 0, :7] - target_ear[:, 0, :7]))
        ),
        "phase_mse": float(np.mean(np.square(predicted_phase - phase_label))),
        "phase_mae": float(np.mean(np.abs(predicted_phase - phase_label))),
    }


def _baselines(
    arrays: dict[str, np.ndarray],
    pairs: PairIndices,
    *,
    include_nominal_action_baselines: bool,
) -> dict[str, Any]:
    cached_ear = arrays["fresh_ear"][pairs.windows, 0].astype(np.float32)
    stale_phase = pairs.elapsed_age.astype(np.float32) / 2.0
    oracle_phase = pairs.physical_progress.astype(np.float32) / 2.0
    stale_transport = _transport_cached_ear(cached_ear, stale_phase)
    oracle_transport = _transport_cached_ear(cached_ear, oracle_phase)
    result: dict[str, Any] = {
        "stale_fixed_age": _transport_baseline_metrics(
            stale_transport,
            stale_phase,
            arrays,
            pairs,
        ),
        "oracle_label_transport": _transport_baseline_metrics(
            oracle_transport,
            oracle_phase,
            arrays,
            pairs,
        ),
    }
    if include_nominal_action_baselines:
        if not np.array_equal(pairs.elapsed_age, pairs.physical_progress):
            raise ValueError("Hold4/B6 baselines are only defined for nominal pairs.")
        target = arrays["teacher_actions"][pairs.windows, pairs.physical_progress].astype(np.float32)
        result["hold4"] = _action_metrics(
            arrays["hold_actions"][pairs.windows, pairs.physical_progress].astype(np.float32),
            target,
        )
        result["b6"] = _action_metrics(
            arrays["b6_actions"][pairs.windows, pairs.physical_progress].astype(np.float32),
            target,
        )
    return result


def _predict_pairs(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], tuple[jax.Array, jax.Array, jax.Array]],
    params: nnx.State,
    arrays: dict[str, np.ndarray],
    pairs: PairIndices,
    *,
    eval_batch_size: int,
    ablate_current_observation: bool = False,
) -> dict[str, np.ndarray]:
    action_pieces: list[np.ndarray] = []
    ear_pieces: list[np.ndarray] = []
    phase_pieces: list[np.ndarray] = []
    for start in range(0, len(pairs), eval_batch_size):
        selected = pairs.take(slice(start, start + eval_batch_size))
        batch = _batch(arrays, selected)
        if ablate_current_observation:
            batch["current_images"] = batch["anchor_images"]
            batch["current_state"] = batch["anchor_state"]
        predicted_action, predicted_ear, predicted_phase = predict_step(params, batch)
        action_pieces.append(np.asarray(predicted_action))
        ear_pieces.append(np.asarray(predicted_ear))
        phase_pieces.append(np.asarray(predicted_phase))
    outputs = {
        "action": np.concatenate(action_pieces, axis=0),
        "ear": np.concatenate(ear_pieces, axis=0),
        "phase": np.concatenate(phase_pieces, axis=0),
    }
    non_finite = {name: int(np.sum(~np.isfinite(value))) for name, value in outputs.items()}
    if any(non_finite.values()):
        raise FloatingPointError(f"Transported Action-CoT produced non-finite predictions: {non_finite}.")
    return outputs


def _evaluate_partition(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], tuple[jax.Array, jax.Array, jax.Array]],
    params: nnx.State,
    arrays: dict[str, np.ndarray],
    window_indices: np.ndarray,
    *,
    args: Args,
) -> dict[str, Any]:
    nominal = _nominal_pairs(window_indices)
    all_pairs = _all_time_warp_pairs(window_indices)
    nominal_predictions = _predict_pairs(
        predict_step,
        params,
        arrays,
        nominal,
        eval_batch_size=args.eval_batch_size,
    )
    all_predictions = _predict_pairs(
        predict_step,
        params,
        arrays,
        all_pairs,
        eval_batch_size=args.eval_batch_size,
    )
    nominal_anchor_ablation = _predict_pairs(
        predict_step,
        params,
        arrays,
        nominal,
        eval_batch_size=args.eval_batch_size,
        ablate_current_observation=True,
    )
    all_anchor_ablation = _predict_pairs(
        predict_step,
        params,
        arrays,
        all_pairs,
        eval_batch_size=args.eval_batch_size,
        ablate_current_observation=True,
    )
    return {
        "nominal": {
            "num_pairs": len(nominal),
            "model": _prediction_metrics(nominal_predictions, arrays, nominal, args=args),
            "model_anchor_observation_ablation": _prediction_metrics(
                nominal_anchor_ablation,
                arrays,
                nominal,
                args=args,
            ),
            "baselines": _baselines(
                arrays,
                nominal,
                include_nominal_action_baselines=True,
            ),
        },
        "all_time_warp_pairs": {
            "num_pairs": len(all_pairs),
            "model": _prediction_metrics(all_predictions, arrays, all_pairs, args=args),
            "model_anchor_observation_ablation": _prediction_metrics(
                all_anchor_ablation,
                arrays,
                all_pairs,
                args=args,
            ),
            "baselines": _baselines(
                arrays,
                all_pairs,
                include_nominal_action_baselines=False,
            ),
        },
    }


def _train(
    arrays: dict[str, np.ndarray],
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    *,
    args: Args,
    config: transported_action_cot.TransportedActionCoTConfig,
    output_dir: pathlib.Path,
) -> tuple[Any, nnx.State, dict[str, Any]]:
    model = transported_action_cot.TransportedActionCoTExecutor(
        config,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, params = nnx.split(model)
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.train_steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: nnx.State,
        current_optimizer_state: optax.OptState,
        batch: dict[str, jax.Array],
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        current_model = nnx.merge(graphdef, current_params)

        def loss_fn(candidate: transported_action_cot.TransportedActionCoTExecutor):
            predicted_action, predicted_ear, predicted_phase = candidate.forward_with_aux(
                batch["anchor_images"],
                batch["current_images"],
                batch["anchor_state"],
                batch["current_state"],
                batch["cached_ear"],
                batch["cached_iar"],
                batch["cache_age"],
            )
            return _loss(
                predicted_action,
                predicted_ear,
                predicted_phase,
                batch,
                args=args,
            )

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(current_model)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            current_optimizer_state,
            current_params,
        )
        next_params = optax.apply_updates(current_params, updates)
        return (
            next_params,
            next_optimizer_state,
            {
                **metrics,
                "loss": loss,
                "gradient_norm": optax.global_norm(gradients),
            },
        )

    @jax.jit
    def predict_step(
        current_params: nnx.State,
        batch: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        current_model = nnx.merge(graphdef, current_params)
        return current_model.forward_with_aux(
            batch["anchor_images"],
            batch["current_images"],
            batch["anchor_state"],
            batch["current_state"],
            batch["cached_ear"],
            batch["cached_iar"],
            batch["cache_age"],
        )

    validation_pairs = _all_time_warp_pairs(validation_windows)
    validation_nominal_baselines = _baselines(
        arrays,
        _nominal_pairs(validation_windows),
        include_nominal_action_baselines=True,
    )
    validation_warp_baselines = _baselines(
        arrays,
        validation_pairs,
        include_nominal_action_baselines=False,
    )
    baseline_nominal_action_mse = float(validation_nominal_baselines["stale_fixed_age"]["mse_7d"])
    baseline_warp_phase_mae = float(validation_warp_baselines["stale_fixed_age"]["phase_mae"])
    if baseline_nominal_action_mse <= 0 or baseline_warp_phase_mae <= 0:
        raise ValueError(
            "Validation baselines must have positive nominal action MSE and time-warp phase MAE."
        )
    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Metrics already exist: {metrics_path}")
    best_score = float("inf")
    best_step = 0
    best_params: nnx.State | None = None
    stale_logs = 0
    completed_steps = 0
    started = time.monotonic()

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            pairs = _sample_training_pairs(
                train_windows,
                batch_size=args.batch_size,
                rng=rng,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                _batch(arrays, pairs),
            )
            completed_steps = step
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_predictions = _predict_pairs(
                    predict_step,
                    params,
                    arrays,
                    validation_pairs,
                    eval_batch_size=args.eval_batch_size,
                )
                validation_metrics = _prediction_metrics(
                    validation_predictions,
                    arrays,
                    validation_pairs,
                    args=args,
                )
                validation_nominal_action_mse = validation_metrics["by_progress_offset"]["0"]["mse_7d"]
                validation_warp_phase_mae = validation_metrics["overall"]["phase_mae"]
                # Select checkpoints with a unitless, balanced score aligned to
                # the two go/no-go questions: nominal action fidelity and
                # observation-conditioned time-warp tracking.  Raw training
                # loss is phase-scale dominated and is not used for selection.
                validation_score = 0.5 * (
                    validation_nominal_action_mse / baseline_nominal_action_mse
                ) + 0.5 * (validation_warp_phase_mae / baseline_warp_phase_mae)
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{
                        f"train/{name}": float(value)
                        for name, value in jax.device_get(train_metrics).items()
                    },
                    **{
                        f"validation/{name}": float(value)
                        for name, value in validation_metrics["loss"].items()
                    },
                    "validation/action_mse_7d": validation_metrics["overall"]["mse_7d"],
                    "validation/nominal_action_mse_7d": validation_nominal_action_mse,
                    "validation/transport_mse_7d": validation_metrics["overall"]["transport_mse_7d"],
                    "validation/phase_mae": validation_metrics["overall"]["phase_mae"],
                    "validation/selection_score": validation_score,
                }
                non_finite = {
                    name: value
                    for name, value in record.items()
                    if isinstance(value, float) and not np.isfinite(value)
                }
                if non_finite:
                    raise FloatingPointError(f"Non-finite transport training metrics: {non_finite}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                if validation_score < best_score - args.early_stopping_min_delta:
                    best_score = validation_score
                    best_step = step
                    best_params = params
                    stale_logs = 0
                else:
                    stale_logs += 1
                if args.early_stopping_patience_logs > 0 and stale_logs >= args.early_stopping_patience_logs:
                    break

    selected_params = best_params if best_params is not None else params
    params_path = output_dir / "final" / "params"
    _save_params(selected_params, params_path, overwrite=args.overwrite)
    summary = {
        "completed_steps": completed_steps,
        "requested_steps": args.train_steps,
        "best_validation_step": best_step,
        "best_validation_score": best_score,
        "selection_criterion": (
            "0.5 * nominal_action_mse / fixed_transport_nominal_action_mse + "
            "0.5 * all_time_warp_phase_mae / fixed_age_phase_mae"
        ),
        "selection_baselines": {
            "nominal_action_mse_7d": baseline_nominal_action_mse,
            "all_time_warp_phase_mae": baseline_warp_phase_mae,
        },
        "elapsed_seconds": time.monotonic() - started,
        "params_path": str(params_path.resolve()),
        "validation_pair_count": len(validation_pairs),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return predict_step, selected_params, summary


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")

    arrays = multirate_dataset.load_multirate_arrays(args.dataset)
    if arrays["images"].shape[1] != 4:
        raise ValueError(f"Expected four-frame windows, got {arrays['images'].shape}.")
    train_windows, validation_windows, test_windows = _split_windows(
        arrays,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    ranges = _training_ranges(
        arrays,
        train_windows,
        minimum_action_residual_scale=args.minimum_action_residual_scale,
        output_margin=args.output_margin,
    )
    config = transported_action_cot.TransportedActionCoTConfig(
        image_size=int(arrays["images"].shape[3]),
        state_dim=int(arrays["states"].shape[-1]),
        action_dim=int(arrays["teacher_actions"].shape[-1]),
        ear_horizon=int(arrays["fresh_ear"].shape[-2]),
        iar_tokens=int(arrays["fresh_iar"].shape[-2]),
        iar_dim=int(arrays["fresh_iar"].shape[-1]),
        coarse_time_stride=2,
        max_phase=float(arrays["fresh_ear"].shape[-2] - 1),
    )
    predict_step, params, train_summary = _train(
        arrays,
        train_windows,
        validation_windows,
        args=args,
        config=config,
        output_dir=output_dir,
    )
    validation_evaluation = _evaluate_partition(
        predict_step,
        params,
        arrays,
        validation_windows,
        args=args,
    )
    test_evaluation = _evaluate_partition(
        predict_step,
        params,
        arrays,
        test_windows,
        args=args,
    )
    summary = {
        "status": "complete",
        "method": "transported_action_cot_time_warp",
        "dataset": list(args.dataset),
        "num_windows": int(arrays["anchor_index"].shape[0]),
        "num_train_windows": int(train_windows.size),
        "num_validation_windows": int(validation_windows.size),
        "num_test_windows": int(test_windows.size),
        "split": "task-stratified and episode-disjoint",
        "time_warp_protocol": {
            "anchor_age": 0,
            "elapsed_ages": [1, 2, 3],
            "legal_physical_progress": "{d-1,d,d+1} intersect [0,3]",
            "phase_label": "physical_progress / 2",
            "training_sampling": "uniform elapsed age, then uniform legal physical progress per sampled window",
            "validation_and_test": "nominal and exhaustive legal time-warp pairs",
        },
        "range_calibration": {
            "source": "training partition only",
            **ranges,
        },
        "config": dataclasses.asdict(config),
        "args": dataclasses.asdict(args),
        "train": train_summary,
        "validation": validation_evaluation,
        "test": test_evaluation,
        "note": "Episode-held-out open-loop transport/action fidelity; not a LIBERO success-rate result.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
