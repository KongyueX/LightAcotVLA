"""Probe whether disturbed observations reveal effective cached-plan progress.

This deliberately narrow experiment learns one bounded scalar: where the
current state lies along a cached EAR.  The scalar transports the cached plan;
there is no observation-to-action residual or action decoder in this script.

Targets are obtained by a dense, continuous phase search that minimizes the
six-dimensional EAR error to the same-seed fresh teacher plan.  Roots are
split by ``(task_id, episode_id)`` before valid branches are flattened.
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
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import branched_dataset
from openpi.models import transported_action_cot


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    output_dir: str
    seed: int = 7
    split_seed: int = 7
    train_steps: int = 500
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    log_interval: int = 50
    phase_loss_weight: float = 0.25
    transport_loss_weight: float = 1.0
    oracle_max_phase: float = 4.0
    oracle_grid_size: int = 161
    oracle_chunk_size: int = 256
    max_parameters: int = 1_000_000
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class BranchIndices:
    roots: np.ndarray
    branches: np.ndarray

    def __post_init__(self) -> None:
        roots = np.asarray(self.roots)
        branches = np.asarray(self.branches)
        if roots.ndim != 1 or branches.shape != roots.shape:
            raise ValueError("roots and branches must be matching rank-one arrays.")
        if roots.dtype.kind not in {"i", "u"} or branches.dtype.kind not in {"i", "u"}:
            raise TypeError("roots and branches must contain integer indices.")

    def __len__(self) -> int:
        return int(self.roots.size)

    def take(self, indices: np.ndarray | slice) -> BranchIndices:
        return BranchIndices(roots=self.roots[indices], branches=self.branches[indices])


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("seed and split_seed must be non-negative.")
    if (
        args.train_steps <= 0
        or args.batch_size <= 0
        or args.eval_batch_size <= 0
        or args.log_interval <= 0
        or args.oracle_chunk_size <= 0
    ):
        raise ValueError("Training, batching, logging, and oracle chunk sizes must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm <= 0:
        raise ValueError("Optimizer scales must be positive and weight_decay non-negative.")
    if args.phase_loss_weight < 0 or args.transport_loss_weight <= 0:
        raise ValueError("phase_loss_weight must be non-negative and transport_loss_weight positive.")
    if not 0 < args.validation_fraction < 0.5 or not 0 < args.test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")
    if args.oracle_max_phase <= 0:
        raise ValueError("oracle_max_phase must be positive.")
    if args.oracle_grid_size < 2:
        raise ValueError("oracle_grid_size must be at least two.")
    if args.max_parameters <= 0 or args.max_parameters > 1_000_000:
        raise ValueError("max_parameters must lie in [1, 1_000_000].")


def _split_roots(
    arrays: dict[str, np.ndarray],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return task-stratified, episode-disjoint root indices.

    In the canonical three-episode pilot this deterministically assigns one
    shuffled episode per task to each partition.
    """

    tasks = np.asarray(arrays["task_id"], dtype=np.int64)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    if tasks.ndim != 1 or episodes.shape != tasks.shape:
        raise ValueError("task_id and episode_id must be matching rank-one root arrays.")
    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for task_id in np.unique(tasks):
        task_roots = np.flatnonzero(tasks == task_id)
        task_episodes = np.unique(episodes[task_roots])
        if task_episodes.size < 3:
            raise ValueError(f"Task {task_id} needs at least three episodes for train/val/test.")
        rng.shuffle(task_episodes)
        test_count = max(1, round(task_episodes.size * test_fraction))
        validation_count = max(1, round(task_episodes.size * validation_fraction))
        if test_count + validation_count >= task_episodes.size:
            test_count = 1
            validation_count = 1
        test_episodes = task_episodes[:test_count]
        validation_episodes = task_episodes[test_count : test_count + validation_count]
        train_episodes = task_episodes[test_count + validation_count :]
        train.append(task_roots[np.isin(episodes[task_roots], train_episodes)])
        validation.append(task_roots[np.isin(episodes[task_roots], validation_episodes)])
        test.append(task_roots[np.isin(episodes[task_roots], test_episodes)])
    outputs = tuple(np.sort(np.concatenate(parts)) for parts in (train, validation, test))
    if any(not partition.size for partition in outputs):
        raise ValueError("Episode-level split produced an empty partition.")
    return outputs  # type: ignore[return-value]


def _flatten_valid_branches(
    arrays: dict[str, np.ndarray],
    root_indices: np.ndarray,
) -> BranchIndices:
    valid = np.asarray(arrays["branch_valid"], dtype=np.bool_)[root_indices]
    if valid.ndim != 2:
        raise ValueError(f"branch_valid must have shape [roots, branches], got {valid.shape}.")
    local_roots, branches = np.nonzero(valid)
    if not local_roots.size:
        raise ValueError("Partition contains no valid branches.")
    return BranchIndices(
        roots=np.asarray(root_indices, dtype=np.int64)[local_roots],
        branches=branches.astype(np.int64),
    )


def _transport_ear_numpy(cached_ear: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    cached = np.asarray(cached_ear, dtype=np.float32)
    if cached.ndim != 3 or cached.shape[1] == 0:
        raise ValueError(f"cached_ear must have shape [N,H,D] with H > 0, got {cached.shape}.")
    phases = np.broadcast_to(np.asarray(phase, dtype=np.float32), (cached.shape[0],))
    positions = np.clip(
        phases[:, None] + np.arange(cached.shape[1], dtype=np.float32)[None, :],
        0.0,
        float(cached.shape[1] - 1),
    )
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, cached.shape[1] - 1)
    interpolation = positions - lower
    batch = np.arange(cached.shape[0], dtype=np.int64)[:, None]
    lower_values = cached[batch, lower]
    upper_values = cached[batch, upper]
    return lower_values + interpolation[..., None] * (upper_values - lower_values)


def _dense_phase_oracle(
    cached_ear: np.ndarray,
    fresh_ear: np.ndarray,
    *,
    max_phase: float,
    grid_size: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find each sample's scalar phase using only continuous six-DoF EAR."""

    cached = np.asarray(cached_ear, dtype=np.float32)
    target = np.asarray(fresh_ear, dtype=np.float32)
    if cached.shape != target.shape or cached.ndim != 3 or cached.shape[-1] < 6:
        raise ValueError(f"EAR arrays must match as [N,H,D>=6], got {cached.shape} and {target.shape}.")
    grid = np.linspace(0.0, max_phase, grid_size, dtype=np.float32)
    token_positions = np.arange(cached.shape[1], dtype=np.float32)
    positions = np.clip(
        grid[:, None] + token_positions[None, :],
        0.0,
        float(cached.shape[1] - 1),
    )
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, cached.shape[1] - 1)
    interpolation = positions - lower
    best_phase = np.empty((cached.shape[0],), dtype=np.float32)
    best_error = np.empty((cached.shape[0],), dtype=np.float32)
    for start in range(0, cached.shape[0], chunk_size):
        end = min(start + chunk_size, cached.shape[0])
        candidate_lower = cached[start:end, lower, :6]
        candidate_upper = cached[start:end, upper, :6]
        candidates = candidate_lower + interpolation[None, ..., None] * (candidate_upper - candidate_lower)
        errors = np.mean(
            np.square(candidates - target[start:end, None, :, :6]),
            axis=(2, 3),
        )
        best = np.argmin(errors, axis=1)
        best_phase[start:end] = grid[best]
        best_error[start:end] = errors[np.arange(end - start), best]
    return best_phase, best_error


def _flat_arrays(
    arrays: dict[str, np.ndarray],
    indices: BranchIndices,
) -> dict[str, np.ndarray]:
    roots = indices.roots
    branches = indices.branches
    return {
        "root_id": np.asarray(arrays["root_id"])[roots],
        "task_id": np.asarray(arrays["task_id"])[roots],
        "episode_id": np.asarray(arrays["episode_id"])[roots],
        "branch_id": np.asarray(arrays["branch_ids"])[roots, branches],
        "branch_steps": np.asarray(arrays["branch_steps"])[roots, branches],
        "anchor_images": np.asarray(arrays["anchor_images"])[roots],
        "current_images": np.asarray(arrays["current_images"])[roots, branches],
        "anchor_state": np.asarray(arrays["anchor_state"])[roots],
        "current_state": np.asarray(arrays["current_state"])[roots, branches],
        "cached_ear": np.asarray(arrays["cached_ear"])[roots],
        "fresh_ear": np.asarray(arrays["fresh_ear"])[roots, branches],
        "executed_actions": np.asarray(arrays["executed_actions"])[roots, branches],
        "executed_valid": np.asarray(arrays["executed_valid"])[roots, branches],
    }


def _calibrate_global_phase(
    cached_ear: np.ndarray,
    fresh_ear: np.ndarray,
    *,
    max_phase: float,
    grid_size: int,
) -> float:
    grid = np.linspace(0.0, max_phase, grid_size, dtype=np.float32)
    errors = np.empty((grid.size,), dtype=np.float64)
    for index, phase in enumerate(grid):
        prediction = _transport_ear_numpy(cached_ear, float(phase))
        errors[index] = float(np.mean(np.square(prediction[..., :6] - fresh_ear[..., :6])))
    return float(grid[int(np.argmin(errors))])


def _branch_mean_phases(
    branch_ids: np.ndarray,
    oracle_phase: np.ndarray,
    *,
    global_phase: float,
) -> dict[int, float]:
    means: dict[int, float] = {}
    for branch_id in range(len(branched_dataset.BRANCH_NAMES)):
        selected = np.asarray(branch_ids) == branch_id
        means[branch_id] = float(np.mean(oracle_phase[selected])) if np.any(selected) else global_phase
    return means


class _Conv2D(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        fan_in = kernel_size * kernel_size * in_channels
        kernel = jax.random.normal(
            rngs.params(),
            (kernel_size, kernel_size, in_channels, out_channels),
            dtype=jnp.float32,
        )
        self.kernel = nnx.Param(kernel * jnp.sqrt(jnp.asarray(2.0 / fan_in, dtype=jnp.float32)))
        self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=jnp.float32))

    def __call__(self, images: jax.Array) -> jax.Array:
        return (
            jax.lax.conv_general_dilated(
                images,
                self.kernel.value,
                window_strides=(2, 2),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + self.bias.value
        )


class EffectiveProgressProbe(nnx.Module):
    """Shared visual/state/prefix/plan encoder with one bounded phase output."""

    def __init__(
        self,
        *,
        image_views: int,
        image_channels: int,
        state_dim: int,
        action_dim: int,
        env_action_dim: int,
        max_executed_steps: int,
        max_phase: float,
        rngs: nnx.Rngs,
        hidden_dim: int = 128,
        cnn_channels: tuple[int, ...] = (16, 32, 48),
    ) -> None:
        self.image_views = image_views
        self.max_phase = max_phase
        self.max_executed_steps = max_executed_steps
        image_convs = []
        in_channels = image_channels
        for out_channels, kernel_size in zip(cnn_channels, (5, 3, 3), strict=True):
            image_convs.append(_Conv2D(in_channels, out_channels, kernel_size, rngs=rngs))
            in_channels = out_channels
        self.image_convs = image_convs
        image_summary = image_views * cnn_channels[-1]
        self.image_proj = nnx.Linear(4 * image_summary, hidden_dim, rngs=rngs)
        self.state_proj = nnx.Linear(3 * state_dim, hidden_dim, rngs=rngs)
        self.plan_token_proj = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.plan_summary_proj = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        prefix_dim = max_executed_steps * (env_action_dim + 1)
        self.prefix_proj = nnx.Linear(prefix_dim, hidden_dim, rngs=rngs)
        self.age_proj = nnx.Linear(1, hidden_dim, rngs=rngs)
        self.fusion = nnx.Linear(5 * hidden_dim, hidden_dim, rngs=rngs)
        self.hidden = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.phase_out = nnx.Linear(
            hidden_dim,
            1,
            rngs=rngs,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    @staticmethod
    def _normalize_images(images: jax.Array) -> jax.Array:
        mean = jnp.mean(images, axis=(1, 2), keepdims=True)
        variance = jnp.mean(jnp.square(images - mean), axis=(1, 2), keepdims=True)
        return (images - mean) * jax.lax.rsqrt(variance + 1e-6)

    def _encode_images(self, images: jax.Array) -> jax.Array:
        batch_size, views, height, width, channels = images.shape
        if views != self.image_views:
            raise ValueError(f"Expected {self.image_views} image views, got {views}.")
        encoded = images.reshape((batch_size * views, height, width, channels))
        encoded = self._normalize_images(encoded)
        for convolution in self.image_convs:
            encoded = jax.nn.silu(convolution(encoded))
        encoded = jnp.mean(encoded, axis=(1, 2))
        return encoded.reshape((batch_size, -1))

    def __call__(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_ear: jax.Array,
        executed_actions: jax.Array,
        executed_valid: jax.Array,
    ) -> jax.Array:
        anchor_visual = self._encode_images(anchor_images)
        current_visual = self._encode_images(current_images)
        visual_delta = current_visual - anchor_visual
        image_feature = jax.nn.silu(
            self.image_proj(
                jnp.concatenate(
                    [
                        anchor_visual,
                        current_visual,
                        visual_delta,
                        jnp.abs(visual_delta),
                    ],
                    axis=-1,
                )
            )
        )
        state_delta = current_state - anchor_state
        state_feature = jax.nn.silu(
            self.state_proj(jnp.concatenate([anchor_state, current_state, state_delta], axis=-1))
        )
        plan_tokens = jax.nn.silu(self.plan_token_proj(cached_ear))
        plan_positions = jnp.linspace(-1.0, 1.0, cached_ear.shape[1], dtype=plan_tokens.dtype)
        plan_feature = jax.nn.silu(
            self.plan_summary_proj(
                jnp.concatenate(
                    [
                        jnp.mean(plan_tokens, axis=1),
                        jnp.mean(plan_tokens * plan_positions[None, :, None], axis=1),
                    ],
                    axis=-1,
                )
            )
        )
        valid = executed_valid.astype(executed_actions.dtype)
        masked_actions = executed_actions * valid[..., None]
        prefix_feature = jax.nn.silu(
            self.prefix_proj(
                jnp.concatenate([masked_actions, valid[..., None]], axis=-1).reshape((cached_ear.shape[0], -1))
            )
        )
        elapsed_fraction = jnp.sum(valid, axis=1, keepdims=True) / float(self.max_executed_steps)
        age_feature = jax.nn.silu(self.age_proj(elapsed_fraction))
        hidden = jax.nn.silu(
            self.fusion(
                jnp.concatenate(
                    [
                        image_feature,
                        state_feature,
                        plan_feature,
                        prefix_feature,
                        age_feature,
                    ],
                    axis=-1,
                )
            )
        )
        hidden = hidden + jax.nn.silu(self.hidden(hidden))
        return self.max_phase * jax.nn.sigmoid(self.phase_out(hidden)[..., 0])


def _linear_parameters(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim + output_dim


def estimate_parameter_count(
    *,
    image_views: int,
    image_channels: int,
    state_dim: int,
    action_dim: int,
    env_action_dim: int,
    max_executed_steps: int,
    hidden_dim: int = 128,
    cnn_channels: tuple[int, ...] = (16, 32, 48),
) -> int:
    count = 0
    in_channels = image_channels
    for out_channels, kernel_size in zip(cnn_channels, (5, 3, 3), strict=True):
        count += kernel_size * kernel_size * in_channels * out_channels + out_channels
        in_channels = out_channels
    image_summary = image_views * cnn_channels[-1]
    count += _linear_parameters(4 * image_summary, hidden_dim)
    count += _linear_parameters(3 * state_dim, hidden_dim)
    count += _linear_parameters(action_dim, hidden_dim)
    count += _linear_parameters(2 * hidden_dim, hidden_dim)
    count += _linear_parameters(max_executed_steps * (env_action_dim + 1), hidden_dim)
    count += _linear_parameters(1, hidden_dim)
    count += _linear_parameters(5 * hidden_dim, hidden_dim)
    count += _linear_parameters(hidden_dim, hidden_dim)
    count += _linear_parameters(hidden_dim, 1)
    return count


def _batch(
    flat: dict[str, np.ndarray],
    phase_labels: np.ndarray,
    indices: np.ndarray,
    *,
    current_equals_anchor: bool = False,
) -> dict[str, jax.Array]:
    anchor_images = flat["anchor_images"][indices].astype(np.float32) / 255.0
    anchor_state = flat["anchor_state"][indices].astype(np.float32)
    current_images = (
        anchor_images if current_equals_anchor else flat["current_images"][indices].astype(np.float32) / 255.0
    )
    current_state = anchor_state if current_equals_anchor else flat["current_state"][indices].astype(np.float32)
    return {
        "anchor_images": jnp.asarray(anchor_images),
        "current_images": jnp.asarray(current_images),
        "anchor_state": jnp.asarray(anchor_state),
        "current_state": jnp.asarray(current_state),
        "cached_ear": jnp.asarray(flat["cached_ear"][indices], dtype=jnp.float32),
        "fresh_ear": jnp.asarray(flat["fresh_ear"][indices], dtype=jnp.float32),
        "executed_actions": jnp.asarray(flat["executed_actions"][indices], dtype=jnp.float32),
        "executed_valid": jnp.asarray(flat["executed_valid"][indices], dtype=jnp.bool_),
        "phase_label": jnp.asarray(phase_labels[indices], dtype=jnp.float32),
    }


def _loss(
    model: EffectiveProgressProbe,
    batch: dict[str, jax.Array],
    *,
    max_phase: float,
    transport_loss_weight: float,
    phase_loss_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    phase = model(
        batch["anchor_images"],
        batch["current_images"],
        batch["anchor_state"],
        batch["current_state"],
        batch["cached_ear"],
        batch["executed_actions"],
        batch["executed_valid"],
    )
    phase_tokens = phase[:, None] + jnp.arange(batch["cached_ear"].shape[1], dtype=jnp.float32)[None]
    transported = transported_action_cot.interpolate_ear(batch["cached_ear"], phase_tokens)
    transport_mse = jnp.mean(jnp.square(transported[..., :6] - batch["fresh_ear"][..., :6]))
    phase_mse = jnp.mean(jnp.square((phase - batch["phase_label"]) / max_phase))
    total = transport_loss_weight * transport_mse + phase_loss_weight * phase_mse
    return total, {
        "loss": total,
        "transport_mse_6d": transport_mse,
        "phase_mse_normalized": phase_mse,
        "phase_mae": jnp.mean(jnp.abs(phase - batch["phase_label"])),
    }


def _predict_all(
    predict_step: Callable[[nnx.State, dict[str, jax.Array]], jax.Array],
    params: nnx.State,
    flat: dict[str, np.ndarray],
    phase_labels: np.ndarray,
    *,
    batch_size: int,
    current_equals_anchor: bool = False,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(phase_labels), batch_size):
        indices = np.arange(start, min(start + batch_size, len(phase_labels)))
        prediction = predict_step(
            params,
            _batch(
                flat,
                phase_labels,
                indices,
                current_equals_anchor=current_equals_anchor,
            ),
        )
        outputs.append(np.asarray(jax.device_get(prediction), dtype=np.float32))
    return np.concatenate(outputs)


def _safe_gap_closure(value: float, stale: float, oracle: float) -> float | None:
    denominator = stale - oracle
    if denominator <= 1e-12:
        return None
    return float((stale - value) / denominator)


def _metrics_for_mask(
    phase: np.ndarray,
    oracle_phase: np.ndarray,
    cached_ear: np.ndarray,
    fresh_ear: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_phase = np.asarray(phase, dtype=np.float32)[mask]
    selected_oracle = np.asarray(oracle_phase, dtype=np.float32)[mask]
    selected_cached = np.asarray(cached_ear, dtype=np.float32)[mask]
    selected_fresh = np.asarray(fresh_ear, dtype=np.float32)[mask]
    prediction = _transport_ear_numpy(selected_cached, selected_phase)
    stale = _transport_ear_numpy(selected_cached, 0.0)
    oracle = _transport_ear_numpy(selected_cached, selected_oracle)
    mse_6d = float(np.mean(np.square(prediction[..., :6] - selected_fresh[..., :6])))
    mse_7d = float(np.mean(np.square(prediction[..., :7] - selected_fresh[..., :7])))
    stale_mse_6d = float(np.mean(np.square(stale[..., :6] - selected_fresh[..., :6])))
    stale_mse_7d = float(np.mean(np.square(stale[..., :7] - selected_fresh[..., :7])))
    oracle_mse_6d = float(np.mean(np.square(oracle[..., :6] - selected_fresh[..., :6])))
    oracle_mse_7d = float(np.mean(np.square(oracle[..., :7] - selected_fresh[..., :7])))
    return {
        "count": int(np.sum(mask)),
        "ear_mse_6d": mse_6d,
        "ear_mse_7d": mse_7d,
        "phase_mae": float(np.mean(np.abs(selected_phase - selected_oracle))),
        "gap_closure_6d_vs_stale_oracle": _safe_gap_closure(mse_6d, stale_mse_6d, oracle_mse_6d),
        "gap_closure_7d_vs_stale_oracle": _safe_gap_closure(mse_7d, stale_mse_7d, oracle_mse_7d),
    }


def _stratified_metrics(
    phase: np.ndarray,
    oracle_phase: np.ndarray,
    flat: dict[str, np.ndarray],
) -> dict[str, Any]:
    branch_ids = np.asarray(flat["branch_id"], dtype=np.int64)

    def evaluate(mask: np.ndarray) -> dict[str, Any]:
        return _metrics_for_mask(
            phase,
            oracle_phase,
            flat["cached_ear"],
            flat["fresh_ear"],
            mask,
        )

    return {
        "overall": evaluate(np.ones((len(phase),), dtype=np.bool_)),
        "nominal_disturbed": {
            "nominal": evaluate(branch_ids == 0),
            "disturbed": evaluate(branch_ids != 0),
        },
        "by_branch": {
            name: evaluate(branch_ids == branch_id)
            for branch_id, name in enumerate(branched_dataset.BRANCH_NAMES)
            if np.any(branch_ids == branch_id)
        },
    }


def _episode_summary(arrays: dict[str, np.ndarray], roots: np.ndarray) -> dict[str, list[int]]:
    tasks = np.asarray(arrays["task_id"], dtype=np.int64)
    episodes = np.asarray(arrays["episode_id"], dtype=np.int64)
    return {
        str(task_id): sorted(np.unique(episodes[roots][tasks[roots] == task_id]).astype(int).tolist())
        for task_id in np.unique(tasks[roots])
    }


def _save_params(params: nnx.State, target: pathlib.Path, *, overwrite: bool) -> None:
    item = {"params": {"effective_progress_probe": params.to_pure_dict()}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target.resolve(), item, force=overwrite)


def _train(
    train_flat: dict[str, np.ndarray],
    train_phase: np.ndarray,
    validation_flat: dict[str, np.ndarray],
    validation_phase: np.ndarray,
    *,
    args: Args,
    output_dir: pathlib.Path,
) -> tuple[Any, nnx.State, dict[str, Any]]:
    image_shape = train_flat["anchor_images"].shape
    model = EffectiveProgressProbe(
        image_views=image_shape[1],
        image_channels=image_shape[-1],
        state_dim=train_flat["anchor_state"].shape[-1],
        action_dim=train_flat["cached_ear"].shape[-1],
        env_action_dim=train_flat["executed_actions"].shape[-1],
        max_executed_steps=train_flat["executed_actions"].shape[-2],
        max_phase=args.oracle_max_phase,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, params = nnx.split(model)
    actual_parameters = int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))
    expected_parameters = estimate_parameter_count(
        image_views=image_shape[1],
        image_channels=image_shape[-1],
        state_dim=train_flat["anchor_state"].shape[-1],
        action_dim=train_flat["cached_ear"].shape[-1],
        env_action_dim=train_flat["executed_actions"].shape[-1],
        max_executed_steps=train_flat["executed_actions"].shape[-2],
    )
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Parameter accounting mismatch: actual={actual_parameters}, expected={expected_parameters}."
        )
    if actual_parameters >= args.max_parameters:
        raise ValueError(
            f"Effective-progress probe has {actual_parameters:,} parameters; limit={args.max_parameters:,}."
        )
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

        def loss_fn(candidate: EffectiveProgressProbe):
            return _loss(
                candidate,
                batch,
                max_phase=args.oracle_max_phase,
                transport_loss_weight=args.transport_loss_weight,
                phase_loss_weight=args.phase_loss_weight,
            )

        (_, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(current_model)
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
                "gradient_norm": optax.global_norm(gradients),
            },
        )

    @jax.jit
    def predict_step(current_params: nnx.State, batch: dict[str, jax.Array]) -> jax.Array:
        current_model = nnx.merge(graphdef, current_params)
        return current_model(
            batch["anchor_images"],
            batch["current_images"],
            batch["anchor_state"],
            batch["current_state"],
            batch["cached_ear"],
            batch["executed_actions"],
            batch["executed_valid"],
        )

    rng = np.random.default_rng(args.seed)
    metrics_path = output_dir / "metrics.jsonl"
    best_score = float("inf")
    best_step = 0
    best_params: nnx.State | None = None
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                len(train_phase),
                size=args.batch_size,
                replace=len(train_phase) < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                _batch(train_flat, train_phase, sampled),
            )
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_prediction = _predict_all(
                    predict_step,
                    params,
                    validation_flat,
                    validation_phase,
                    batch_size=args.eval_batch_size,
                )
                validation_transport = _transport_ear_numpy(
                    validation_flat["cached_ear"],
                    validation_prediction,
                )
                validation_score = float(
                    np.mean(np.square(validation_transport[..., :6] - validation_flat["fresh_ear"][..., :6]))
                )
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{f"train/{name}": float(value) for name, value in jax.device_get(train_metrics).items()},
                    "validation/transport_mse_6d": validation_score,
                    "validation/phase_mae": float(np.mean(np.abs(validation_prediction - validation_phase))),
                }
                if not all(np.isfinite(value) for value in record.values()):
                    raise FloatingPointError(f"Non-finite progress-probe metric: {record}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                if validation_score < best_score:
                    best_score = validation_score
                    best_step = step
                    best_params = params
    selected_params = best_params if best_params is not None else params
    params_path = output_dir / "final" / "params"
    _save_params(selected_params, params_path, overwrite=args.overwrite)
    return (
        predict_step,
        selected_params,
        {
            "parameter_count": actual_parameters,
            "completed_steps": args.train_steps,
            "best_validation_step": best_step,
            "best_validation_transport_mse_6d": best_score,
            "selection_criterion": "validation transported EAR MSE over continuous dimensions 0:6",
            "elapsed_seconds": time.monotonic() - started,
            "params_path": str(params_path.resolve()),
        },
    )


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_path}")
    arrays = branched_dataset.load_branched_arrays(
        args.dataset,
        fields=(
            "root_id",
            "task_id",
            "episode_id",
            "branch_ids",
            "branch_steps",
            "branch_valid",
            "anchor_images",
            "current_images",
            "anchor_state",
            "current_state",
            "cached_ear",
            "fresh_ear",
            "executed_actions",
            "executed_valid",
        ),
    )
    train_roots, validation_roots, test_roots = _split_roots(
        arrays,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    partitions: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for name, roots in (
        ("train", train_roots),
        ("validation", validation_roots),
        ("test", test_roots),
    ):
        indices = _flatten_valid_branches(arrays, roots)
        partitions[name] = (roots, _flat_arrays(arrays, indices))

    phase_labels: dict[str, np.ndarray] = {}
    oracle_errors: dict[str, np.ndarray] = {}
    for name, (_, flat) in partitions.items():
        phase_labels[name], oracle_errors[name] = _dense_phase_oracle(
            flat["cached_ear"],
            flat["fresh_ear"],
            max_phase=args.oracle_max_phase,
            grid_size=args.oracle_grid_size,
            chunk_size=args.oracle_chunk_size,
        )

    train_flat = partitions["train"][1]
    validation_flat = partitions["validation"][1]
    test_flat = partitions["test"][1]
    global_phase = _calibrate_global_phase(
        train_flat["cached_ear"],
        train_flat["fresh_ear"],
        max_phase=args.oracle_max_phase,
        grid_size=args.oracle_grid_size,
    )
    branch_means = _branch_mean_phases(
        train_flat["branch_id"],
        phase_labels["train"],
        global_phase=global_phase,
    )
    predict_step, params, train_summary = _train(
        train_flat,
        phase_labels["train"],
        validation_flat,
        phase_labels["validation"],
        args=args,
        output_dir=output_dir,
    )
    learned_phase = _predict_all(
        predict_step,
        params,
        test_flat,
        phase_labels["test"],
        batch_size=args.eval_batch_size,
    )
    anchor_ablation_phase = _predict_all(
        predict_step,
        params,
        test_flat,
        phase_labels["test"],
        batch_size=args.eval_batch_size,
        current_equals_anchor=True,
    )
    test_branch_ids = np.asarray(test_flat["branch_id"], dtype=np.int64)
    baseline_phases = {
        "stale_phase0": np.zeros_like(phase_labels["test"]),
        "fixed_phase2": np.full_like(
            phase_labels["test"],
            min(2.0, args.oracle_max_phase),
        ),
        "train_global_calibrated": np.full_like(phase_labels["test"], global_phase),
        "branch_id_train_mean_privileged": np.asarray(
            [branch_means[int(branch_id)] for branch_id in test_branch_ids],
            dtype=np.float32,
        ),
        "learned_observation_conditioned": learned_phase,
        "learned_current_equals_anchor_ablation": anchor_ablation_phase,
        "dense_phase_oracle": phase_labels["test"],
    }
    evaluations = {
        name: _stratified_metrics(phase, phase_labels["test"], test_flat) for name, phase in baseline_phases.items()
    }
    fixed2_mse = evaluations["fixed_phase2"]["overall"]["ear_mse_6d"]
    for evaluation in evaluations.values():
        current = evaluation["overall"]["ear_mse_6d"]
        evaluation["overall"]["relative_improvement_6d_vs_fixed_phase2"] = (
            float((fixed2_mse - current) / fixed2_mse) if fixed2_mse > 0 else None
        )

    summary = {
        "method": {
            "name": "branched_effective_progress_probe",
            "causal_bottleneck": "observation -> one bounded scalar phase -> cached EAR transport",
            "forbidden_path": "no observation-to-action residual and no action decoder",
            "oracle": (
                f"{args.oracle_grid_size}-point dense phase search over [0, {args.oracle_max_phase}] "
                "using full-horizon EAR dimensions 0:6"
            ),
        },
        "args": dataclasses.asdict(args),
        "split": {
            name: {
                "root_count": len(roots),
                "valid_branch_count": len(flat["branch_id"]),
                "episodes_by_task": _episode_summary(arrays, roots),
            }
            for name, (roots, flat) in partitions.items()
        },
        "train_calibration": {
            "global_phase": global_phase,
            "branch_id_mean_phase": {
                branched_dataset.BRANCH_NAMES[branch_id]: phase for branch_id, phase in branch_means.items()
            },
            "oracle_phase_mean": float(np.mean(phase_labels["train"])),
            "oracle_phase_std": float(np.std(phase_labels["train"])),
            "oracle_6d_mse_mean": float(np.mean(oracle_errors["train"])),
        },
        "train": train_summary,
        "test": {
            "root_count": len(test_roots),
            "branch_count": len(phase_labels["test"]),
            "oracle_phase_mean": float(np.mean(phase_labels["test"])),
            "oracle_phase_std": float(np.std(phase_labels["test"])),
            "evaluations": evaluations,
        },
        "interpretation_guard": (
            "This is an offline same-root representation probe. It does not establish closed-loop success or "
            "end-to-end wall-clock speedup."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
