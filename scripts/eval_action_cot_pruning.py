"""Evaluate static Action-CoT pruning from Stage A entropy exports.

This script is intentionally offline-first. It can validate entropy-ranked
coarse Action-CoT pruning directly from the Stage A .npz files without running
rollouts or training. If --enable_action_injection is set and a local policy
checkpoint plus dataset are available, it also re-runs the final action head
with masked coarse actions through the optional policy override hook.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from openpi.action_cot import compression as acot_compression
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

LOGGER = logging.getLogger("eval_action_cot_pruning")

DEFAULT_CONFIG = "acot_libero_action_cot_explicit_implicit_co_fusion"
METRIC_FIELDS = [
    "coarse_l1_to_full",
    "coarse_mse_to_full",
    "action_l1_to_full",
    "action_mse_to_full",
    "action_l1_to_expert",
    "action_mse_to_expert",
    "trajectory_jerk",
    "gripper_error",
    "skip_ratio",
    "avg_inference_time",
    "success_rate",
    "average_return",
    "collision_rate",
    "timeout_rate",
]
_SAMPLE_RE = re.compile(r"sample_(\d+)\.npz$")


def _status(message: str) -> None:
    print(f"[eval_action_cot_pruning] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy_dir", "--entropy-dir", required=True, help="Directory produced by Stage A export.")
    parser.add_argument(
        "--policy.config",
        "--policy-config",
        "--config_name",
        "--config-name",
        dest="config_name",
        default=DEFAULT_CONFIG,
        help="Training config name. Dotted alias matches the Stage B spec.",
    )
    parser.add_argument(
        "--policy.dir",
        "--policy-dir",
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=None,
        help="Policy checkpoint directory. Dotted alias matches the Stage B spec.",
    )
    parser.add_argument("--default_prompt", "--default-prompt", default=None)

    parser.add_argument("--strategy", choices=("low_entropy", "high_entropy", "random", "oracle"), required=True)
    parser.add_argument("--segment_mode", "--segment-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=5)
    parser.add_argument("--prune_ratio", "--prune-ratio", type=float, default=0.3)
    parser.add_argument("--replacement", choices=("interp", "hold", "zero"), default="interp")
    parser.add_argument("--num_random_trials", "--num-random-trials", type=int, default=5)
    parser.add_argument("--max_items", "--max-items", type=int, default=None)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--min_keep_segments", "--min-keep-segments", type=int, default=1)

    parser.add_argument("--min_len", "--min-len", type=int, default=3)
    parser.add_argument("--max_len", "--max-len", type=int, default=6)
    parser.add_argument("--max_segments", "--max-segments", type=int, default=5)
    parser.add_argument("--gripper_indices", "--gripper-indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--gripper_index",
        "--gripper-index",
        type=int,
        default=None,
        help="Final action gripper dimension. If omitted, dim -1 is used only for 7-D actions.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue_on_error", "--continue-on-error", action="store_true")
    parser.add_argument("--no_expert_actions", "--no-expert-actions", action="store_true")
    parser.add_argument(
        "--enable_action_injection",
        "--enable-action-injection",
        action="store_true",
        help="Re-run the policy with coarse_actions_override. Expensive; off by default for static sweeps.",
    )
    parser.add_argument(
        "--require_action_injection",
        "--require-action-injection",
        action="store_true",
        help="Fail instead of falling back to static action proxy if injection cannot be initialized.",
    )

    # Closed-loop / environment-shaped arguments are parsed for compatibility
    # with existing eval scripts. Closed-loop pruning rollout is not implemented here.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize_size", "--resize-size", type=int, default=224)
    parser.add_argument("--replan_steps", "--replan-steps", type=int, default=5)
    parser.add_argument("--task_suite_name", "--task-suite-name", default="libero_spatial")
    parser.add_argument("--num_steps_wait", "--num-steps-wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", "--num-trials-per-task", type=int, default=1)
    parser.add_argument("--video_out_path", "--video-out-path", default="./libero_pruning_videos")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive.")
    if not 0.0 <= args.prune_ratio <= 1.0:
        raise ValueError("--prune_ratio must be in [0, 1].")
    if args.num_random_trials <= 0:
        raise ValueError("--num_random_trials must be positive.")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("--max_items must be positive when provided.")
    if args.min_keep_segments < 0:
        raise ValueError("--min_keep_segments must be non-negative.")
    if args.require_action_injection and not args.enable_action_injection:
        raise ValueError("--require_action_injection requires --enable_action_injection.")


def _stat_attr(stats: Any, name: str) -> np.ndarray | None:
    if isinstance(stats, dict):
        value = stats.get(name)
    else:
        value = getattr(stats, name, None)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _fit_last_dim(values: np.ndarray, dim: int, pad_value: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] >= dim:
        return values[..., :dim]

    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, dim - values.shape[-1])
    return np.pad(values, pad_width, constant_values=pad_value)


def _resolve_checkpoint_dir(
    train_config: _config.TrainConfig,
    checkpoint_dir: str | None,
    *,
    allow_config_default: bool,
) -> pathlib.Path | None:
    if checkpoint_dir:
        return download.maybe_download(checkpoint_dir)
    if not allow_config_default:
        return None
    try:
        return download.maybe_download(str(train_config.checkpoint_dir))
    except Exception as exc:
        LOGGER.warning("Could not resolve config checkpoint_dir: %s", exc)
        return None


def _load_norm_stats(
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
    checkpoint_dir: pathlib.Path | None,
) -> dict[str, Any] | None:
    if data_config.norm_stats is not None:
        return data_config.norm_stats
    if checkpoint_dir is None or data_config.asset_id is None:
        return None
    try:
        return _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    except FileNotFoundError:
        LOGGER.warning("No normalization stats found under %s.", checkpoint_dir / "assets")
    except Exception as exc:
        LOGGER.warning("Could not load normalization stats: %s", exc)
    return None


def _normalize_actions(
    actions: np.ndarray,
    norm_stats: dict[str, Any] | None,
    *,
    use_quantiles: bool,
    preferred_key: str,
) -> tuple[np.ndarray, str | None]:
    actions = np.asarray(actions, dtype=np.float64)
    if norm_stats is None:
        return actions, None

    stats_key = preferred_key if preferred_key in norm_stats else "actions" if "actions" in norm_stats else None
    if stats_key is None:
        return actions, None

    stats = norm_stats[stats_key]
    action_dim = actions.shape[-1]
    if use_quantiles:
        q01 = _stat_attr(stats, "q01")
        q99 = _stat_attr(stats, "q99")
        if q01 is not None and q99 is not None:
            q01 = _fit_last_dim(q01, action_dim, pad_value=0.0)
            q99 = _fit_last_dim(q99, action_dim, pad_value=1.0)
            return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0, stats_key

    mean = _stat_attr(stats, "mean")
    std = _stat_attr(stats, "std")
    if mean is None or std is None:
        return actions, None
    mean = _fit_last_dim(mean, action_dim, pad_value=0.0)
    std = _fit_last_dim(std, action_dim, pad_value=1.0)
    return (actions - mean) / (std + 1e-6), stats_key


def _zero_action_value(
    norm_stats: dict[str, Any] | None,
    *,
    use_quantiles: bool,
    preferred_key: str,
    dim: int,
) -> np.ndarray:
    if norm_stats is None:
        return np.zeros((dim,), dtype=np.float64)
    stats_key = preferred_key if preferred_key in norm_stats else "actions" if "actions" in norm_stats else None
    if stats_key is None:
        return np.zeros((dim,), dtype=np.float64)

    stats = norm_stats[stats_key]
    if use_quantiles:
        q01 = _stat_attr(stats, "q01")
        q99 = _stat_attr(stats, "q99")
        if q01 is not None and q99 is not None:
            q01 = _fit_last_dim(q01, dim, pad_value=0.0)
            q99 = _fit_last_dim(q99, dim, pad_value=1.0)
            return ((q99 - q01) + 1e-6) * 0.5 + q01

    mean = _stat_attr(stats, "mean")
    if mean is not None:
        return _fit_last_dim(mean, dim, pad_value=0.0)
    return np.zeros((dim,), dtype=np.float64)


def _create_policy_dataset(
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> _data_loader.Dataset:
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("Stage B static expert-action lookup supports random-access LeRobot datasets.")
    base_dataset = _data_loader.create_torch_dataset(data_config, train_config.model)
    return _data_loader.TransformedDataset(base_dataset, [*data_config.repack_transforms.inputs])


def _maybe_create_expert_dataset(
    args: argparse.Namespace,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> _data_loader.Dataset | None:
    if args.no_expert_actions:
        return None
    try:
        dataset = _create_policy_dataset(train_config, data_config)
        _status(f"Expert dataset ready: len={len(dataset)}")
        return dataset
    except Exception as exc:
        LOGGER.warning("Expert action lookup unavailable; continuing without expert metrics: %s", exc)
        return None


def _maybe_create_injection_policy(
    args: argparse.Namespace,
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | None,
    norm_stats: dict[str, Any] | None,
    dataset: _data_loader.Dataset | None,
):
    if not args.enable_action_injection:
        return None
    if checkpoint_dir is None:
        message = "Action injection requires --policy.dir or a resolvable config checkpoint_dir."
        if args.require_action_injection:
            raise ValueError(message)
        LOGGER.warning("%s Falling back to static action proxy.", message)
        return None
    if dataset is None:
        message = "Action injection requires the policy input dataset to reconstruct observations."
        if args.require_action_injection:
            raise ValueError(message)
        LOGGER.warning("%s Falling back to static action proxy.", message)
        return None
    try:
        _status(f"Loading policy for action injection from {checkpoint_dir}")
        return _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=args.default_prompt,
            norm_stats=norm_stats,
        )
    except Exception as exc:
        if args.require_action_injection:
            raise
        LOGGER.warning("Could not initialize action injection policy; using static proxy: %s", exc)
        return None


def _set_policy_seed(policy: Any, seed: int) -> None:
    if not hasattr(policy, "_rng"):
        return
    import jax

    setattr(policy, "_rng", jax.random.key(seed))


def _entropy_files(entropy_dir: pathlib.Path, max_items: int | None) -> list[pathlib.Path]:
    files = sorted(entropy_dir.glob("sample_*.npz"))
    if not files:
        files = sorted(entropy_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz entropy files found in {entropy_dir}.")
    return files[:max_items] if max_items is not None else files


def _item_index_from_path(path: pathlib.Path, fallback: int) -> int:
    match = _SAMPLE_RE.match(path.name)
    if match is None:
        return fallback
    return int(match.group(1))


def _np_scalar_to_str(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return fallback


def _load_npz(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _mean_trajectory(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        return np.mean(values, axis=0)
    if values.ndim == 2:
        return values
    return None


def _first_trajectory(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        return values[0]
    if values.ndim == 2:
        return values
    return None


def _segment(coarse_actions_normalized: np.ndarray, args: argparse.Namespace) -> list[acot_compression.Segment]:
    if args.segment_mode == "fixed":
        return acot_compression.segment_fixed(coarse_actions_normalized, chunk_size=args.chunk_size)
    return acot_compression.segment_adaptive(
        coarse_actions_normalized,
        min_len=args.min_len,
        max_len=args.max_len,
        max_segments=args.max_segments,
        gripper_indices=args.gripper_indices,
    )


def _num_segments_to_skip(num_segments: int, prune_ratio: float, min_keep_segments: int) -> int:
    if num_segments <= 0:
        return 0
    requested = int(math.ceil(prune_ratio * num_segments))
    max_skip = max(num_segments - min_keep_segments, 0)
    return min(max(requested, 0), max_skip)


def _select_skip_mask(
    entropy: np.ndarray,
    *,
    strategy: str,
    prune_ratio: float,
    min_keep_segments: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    entropy = np.asarray(entropy, dtype=np.float64)
    num_segments = entropy.shape[0]
    skip_mask = np.zeros((num_segments,), dtype=np.int8)
    num_skip = _num_segments_to_skip(num_segments, prune_ratio, min_keep_segments)
    if num_skip == 0:
        return skip_mask

    sortable = np.nan_to_num(entropy, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    if strategy == "low_entropy":
        selected = np.argsort(sortable, kind="stable")[:num_skip]
    elif strategy == "high_entropy":
        selected = np.argsort(-sortable, kind="stable")[:num_skip]
    elif strategy == "random":
        if rng is None:
            raise ValueError("rng is required for random pruning.")
        selected = rng.choice(num_segments, size=num_skip, replace=False)
    else:
        raise ValueError(f"Unsupported pruning strategy for skip mask: {strategy}")

    skip_mask[np.asarray(selected, dtype=np.int64)] = 1
    return skip_mask


def _replace_masked_frames(
    trajectory: np.ndarray,
    frame_skip_mask: np.ndarray,
    *,
    replacement: str,
    zero_value: np.ndarray,
) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float64)
    mask = np.asarray(frame_skip_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"trajectory must have shape [T, D], got {values.shape}.")
    if mask.shape != (values.shape[0],):
        raise ValueError(f"frame_skip_mask shape {mask.shape} does not match trajectory length {values.shape[0]}.")

    output = values.copy()
    if not np.any(mask):
        return output

    zero_value = np.asarray(zero_value, dtype=np.float64)
    if zero_value.shape != (values.shape[1],):
        zero_value = _fit_last_dim(zero_value, values.shape[1], pad_value=0.0)

    kept = ~mask
    if replacement == "zero" or not np.any(kept):
        output[mask] = zero_value
        return output

    kept_indices = np.flatnonzero(kept)
    all_indices = np.arange(values.shape[0])
    if replacement == "interp":
        for dim in range(values.shape[1]):
            output[:, dim] = np.interp(all_indices, kept_indices, values[kept_indices, dim])
        return output

    if replacement == "hold":
        first_kept = int(kept_indices[0])
        last_value = values[first_kept]
        for idx in range(values.shape[0]):
            if kept[idx]:
                last_value = values[idx]
            elif idx < first_kept:
                output[idx] = values[first_kept]
            else:
                output[idx] = last_value
        return output

    raise ValueError(f"Unknown replacement: {replacement}")


def _project_frame_mask(frame_skip_mask: np.ndarray, target_len: int) -> np.ndarray:
    mask = np.asarray(frame_skip_mask, dtype=np.int8)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.int8)
    if mask.shape[0] == target_len:
        return mask.copy()
    if mask.shape[0] == 0:
        return np.zeros((target_len,), dtype=np.int8)
    source_indices = np.floor((np.arange(target_len) + 0.5) * mask.shape[0] / target_len).astype(np.int64)
    source_indices = np.clip(source_indices, 0, mask.shape[0] - 1)
    return mask[source_indices].astype(np.int8)


def _align_pair(left: np.ndarray | None, right: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if left is None or right is None:
        return None
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        return None
    t_len = min(left.shape[0], right.shape[0])
    dim = min(left.shape[1], right.shape[1])
    if t_len == 0 or dim == 0:
        return None
    return left[:t_len, :dim], right[:t_len, :dim]


def _l1_mse(left: np.ndarray | None, right: np.ndarray | None) -> tuple[float, float]:
    aligned = _align_pair(left, right)
    if aligned is None:
        return float("nan"), float("nan")
    a, b = aligned
    diff = a - b
    return float(np.mean(np.abs(diff))), float(np.mean(np.square(diff)))


def _trajectory_jerk(trajectory: np.ndarray | None) -> float:
    if trajectory is None:
        return float("nan")
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[0] < 4:
        return 0.0
    jerk = np.diff(trajectory, n=3, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def _resolve_gripper_index(dim: int, configured_index: int | None) -> int | None:
    if configured_index is not None:
        index = configured_index if configured_index >= 0 else dim + configured_index
        return index if 0 <= index < dim else None
    if dim == 7:
        return dim - 1
    return None


def _gripper_error(
    pruned_actions: np.ndarray | None,
    reference_actions: np.ndarray | None,
    configured_index: int | None,
) -> float:
    aligned = _align_pair(pruned_actions, reference_actions)
    if aligned is None:
        return float("nan")
    left, right = aligned
    index = _resolve_gripper_index(left.shape[1], configured_index)
    if index is None:
        return float("nan")
    return float(np.mean(np.abs(left[:, index] - right[:, index])))


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.std(finite)) if finite else float("nan")


class MetricAccumulator:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {field: [] for field in METRIC_FIELDS}

    def update(self, metrics: dict[str, float]) -> None:
        for field in METRIC_FIELDS:
            self._values[field].append(float(metrics.get(field, float("nan"))))

    def mean(self) -> dict[str, float]:
        return {field: _safe_mean(values) for field, values in self._values.items()}


def _aggregate_std(aggregates: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        field: _safe_std([aggregate.get(field, float("nan")) for aggregate in aggregates])
        for field in METRIC_FIELDS
    }


def _expert_actions_for_item(
    dataset: _data_loader.Dataset | None,
    item_index: int,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if dataset is None:
        return None, None
    data = dataset[item_index]
    if data is None:
        return None, None
    if "actions" not in data:
        return None, data

    raw_actions = np.asarray(data["actions"], dtype=np.float64)
    action_horizon = train_config.model.action_horizon
    joint_action_shifts = getattr(data_config, "joint_action_shifts", (1, 1))
    action_shift = int(joint_action_shifts[1]) if len(joint_action_shifts) > 1 else 1
    required_length = (action_horizon - 1) * action_shift + 1
    expert = raw_actions[:required_length:action_shift]
    return expert[:action_horizon], data


def _compute_metrics(
    *,
    coarse_full: np.ndarray,
    coarse_pruned: np.ndarray,
    action_full: np.ndarray | None,
    action_pruned: np.ndarray | None,
    expert_actions: np.ndarray | None,
    frame_skip_mask: np.ndarray,
    gripper_index: int | None,
    avg_inference_time: float,
) -> dict[str, float]:
    coarse_l1, coarse_mse = _l1_mse(coarse_pruned, coarse_full)
    action_l1, action_mse = _l1_mse(action_pruned, action_full)
    expert_l1, expert_mse = _l1_mse(action_pruned, expert_actions)
    gripper_reference = expert_actions if expert_actions is not None else action_full
    jerk_source = action_pruned if action_pruned is not None else coarse_pruned
    return {
        "coarse_l1_to_full": coarse_l1,
        "coarse_mse_to_full": coarse_mse,
        "action_l1_to_full": action_l1,
        "action_mse_to_full": action_mse,
        "action_l1_to_expert": expert_l1,
        "action_mse_to_expert": expert_mse,
        "trajectory_jerk": _trajectory_jerk(jerk_source),
        "gripper_error": _gripper_error(action_pruned, gripper_reference, gripper_index),
        "skip_ratio": float(np.mean(frame_skip_mask)) if frame_skip_mask.size else 0.0,
        "avg_inference_time": avg_inference_time,
        "success_rate": float("nan"),
        "average_return": float("nan"),
        "collision_rate": float("nan"),
        "timeout_rate": float("nan"),
    }


def _prepare_sample(
    path: pathlib.Path,
    *,
    file_position: int,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
    data_config: _config.DataConfig,
) -> dict[str, Any]:
    loaded = _load_npz(path)
    if "coarse_samples" in loaded:
        coarse_samples = np.asarray(loaded["coarse_samples"], dtype=np.float64)
    elif "coarse_mean" in loaded:
        coarse_samples = np.asarray(loaded["coarse_mean"], dtype=np.float64)[None, ...]
    else:
        raise KeyError(f"{path.name} does not contain coarse_samples or coarse_mean.")
    if coarse_samples.ndim != 3:
        raise ValueError(f"{path.name} coarse_samples must have shape [K, T, D], got {coarse_samples.shape}.")

    coarse_samples_normalized, norm_key = _normalize_actions(
        coarse_samples,
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        preferred_key="coarse_actions",
    )
    coarse_mean_normalized = np.mean(coarse_samples_normalized, axis=0)
    segments = _segment(coarse_mean_normalized, args)
    entropy = acot_compression.compute_mc_predictive_entropy(coarse_samples_normalized, segments)

    actions_full = np.asarray(loaded["actions_full"], dtype=np.float64) if "actions_full" in loaded else None
    item_index = _item_index_from_path(path, file_position)
    sample_id = _np_scalar_to_str(loaded.get("sample_id"), fallback=str(item_index))
    timing = np.asarray(loaded["timing"], dtype=np.float64) if "timing" in loaded else np.asarray([])
    return {
        "path": path,
        "item_index": item_index,
        "sample_id": sample_id,
        "coarse_samples": coarse_samples,
        "coarse_mean": np.mean(coarse_samples, axis=0),
        "actions_full": actions_full,
        "segments": segments,
        "entropy": entropy,
        "normalization_key": norm_key,
        "timing": timing,
    }


def _evaluate_once(
    *,
    sample: dict[str, Any],
    skip_mask: np.ndarray,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
    data_config: _config.DataConfig,
    train_config: _config.TrainConfig,
    dataset: _data_loader.Dataset | None,
    policy: Any | None,
) -> tuple[dict[str, float], list[int], str]:
    segments = sample["segments"]
    frame_skip_mask = acot_compression.expand_segment_mask(
        skip_mask,
        segments,
        t_len=sample["coarse_samples"].shape[1],
    )
    injection_enabled = policy is not None
    coarse_full = _first_trajectory(sample["coarse_samples"]) if injection_enabled else sample["coarse_mean"]
    if coarse_full is None:
        raise ValueError("Could not resolve full coarse trajectory.")
    coarse_zero = _zero_action_value(
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        preferred_key="coarse_actions",
        dim=coarse_full.shape[-1],
    )
    coarse_pruned = _replace_masked_frames(
        coarse_full,
        frame_skip_mask,
        replacement=args.replacement,
        zero_value=coarse_zero,
    )

    actions_full_raw = sample["actions_full"]
    action_full = _first_trajectory(actions_full_raw) if injection_enabled else _mean_trajectory(actions_full_raw)
    expert_actions, policy_input = _expert_actions_for_item(dataset, sample["item_index"], train_config, data_config)
    action_pruned = None
    avg_inference_time = _safe_mean(sample["timing"].tolist()) if sample["timing"].size else float("nan")
    action_source = "static_projected_action_proxy"

    if injection_enabled:
        if policy_input is None:
            raise RuntimeError("Action injection policy is available but the matching dataset item is unavailable.")
        infer_input = dict(policy_input)
        infer_input["coarse_actions_override"] = coarse_pruned
        seed_offset = sample["item_index"] * max(sample["coarse_samples"].shape[0], 1)
        _set_policy_seed(policy, args.seed + seed_offset)
        start = time.perf_counter()
        result = policy.infer(infer_input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        action_pruned = np.asarray(result.get("actions"), dtype=np.float64)
        policy_timing = result.get("policy_timing", {}) if isinstance(result, dict) else {}
        avg_inference_time = float(policy_timing.get("infer_ms", elapsed_ms))
        action_source = "model_injected_coarse_actions"
    elif action_full is not None:
        action_mask = _project_frame_mask(frame_skip_mask, action_full.shape[0])
        action_zero = _zero_action_value(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            preferred_key="actions",
            dim=action_full.shape[-1],
        )
        action_pruned = _replace_masked_frames(
            action_full,
            action_mask,
            replacement=args.replacement,
            zero_value=action_zero,
        )

    metrics = _compute_metrics(
        coarse_full=coarse_full,
        coarse_pruned=coarse_pruned,
        action_full=action_full,
        action_pruned=action_pruned,
        expert_actions=expert_actions,
        frame_skip_mask=frame_skip_mask,
        gripper_index=args.gripper_index,
        avg_inference_time=avg_inference_time,
    )
    skipped_segments = [idx for idx, value in enumerate(skip_mask.tolist()) if value]
    return metrics, skipped_segments, action_source


def _write_metadata(
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    *,
    checkpoint_dir: pathlib.Path | None,
    action_injection_available: bool,
    action_metrics_source: str,
) -> None:
    metadata = {
        "stage": "B",
        "note": (
            "Static Stage B validates entropy ranking by masking saved or injected Action-CoT trajectories. "
            "It does not train and does not imply real inference speedup."
        ),
        "entropy_dir": str(args.entropy_dir),
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "strategy": args.strategy,
        "segment_mode": args.segment_mode,
        "chunk_size": args.chunk_size,
        "prune_ratio": args.prune_ratio,
        "replacement": args.replacement,
        "num_random_trials": args.num_random_trials,
        "max_items": args.max_items,
        "min_keep_segments": args.min_keep_segments,
        "action_injection_available": action_injection_available,
        "action_metrics_source": action_metrics_source,
        "closed_loop_pruning_rollout_available": False,
        "closed_loop_note": "Closed-loop pruning rollout is not implemented in this static evaluator.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_summary_csv(
    path: pathlib.Path,
    *,
    args: argparse.Namespace,
    aggregate: dict[str, float],
    random_std: dict[str, float] | None,
    action_metrics_source: str,
    action_injection_available: bool,
) -> None:
    fields = [
        "segment_mode",
        "chunk_size",
        "prune_ratio",
        "strategy",
        "replacement",
        "action_metrics_source",
        "action_injection_available",
        *METRIC_FIELDS,
    ]
    if random_std is not None:
        fields.extend([f"{field}_std" for field in METRIC_FIELDS])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        row = {
            "segment_mode": args.segment_mode,
            "chunk_size": args.chunk_size,
            "prune_ratio": args.prune_ratio,
            "strategy": args.strategy,
            "replacement": args.replacement,
            "action_metrics_source": action_metrics_source,
            "action_injection_available": action_injection_available,
            **aggregate,
        }
        if random_std is not None:
            row.update({f"{field}_std": random_std[field] for field in METRIC_FIELDS})
        writer.writerow(row)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearmanr(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.shape[0] < 2:
        return float("nan")
    try:
        from scipy import stats as scipy_stats

        return float(scipy_stats.spearmanr(x, y, nan_policy="omit").correlation)
    except Exception:
        x_rank = _rankdata(x)
        y_rank = _rankdata(y)
        if np.std(x_rank) == 0 or np.std(y_rank) == 0:
            return float("nan")
        return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _run_oracle(
    *,
    args: argparse.Namespace,
    files: Sequence[pathlib.Path],
    output_dir: pathlib.Path,
    norm_stats: dict[str, Any] | None,
    data_config: _config.DataConfig,
    train_config: _config.TrainConfig,
    dataset: _data_loader.Dataset | None,
    policy: Any | None,
) -> dict[str, Any]:
    importance_path = output_dir / "oracle_importance.csv"
    entropies: list[float] = []
    importances: list[float] = []
    metric_name = "action_mse_to_full" if policy is not None else "coarse_mse_to_full"
    action_source = "model_injected_coarse_actions" if policy is not None else "static_projected_action_proxy"

    with importance_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "segment_id", "start", "end", "entropy", "importance", "importance_metric"],
        )
        writer.writeheader()
        for file_position, path in enumerate(files):
            try:
                sample = _prepare_sample(
                    path,
                    file_position=file_position,
                    args=args,
                    norm_stats=norm_stats,
                    data_config=data_config,
                )
                for segment_id, (start, end) in enumerate(sample["segments"]):
                    skip_mask = np.zeros((len(sample["segments"]),), dtype=np.int8)
                    skip_mask[segment_id] = 1
                    metrics, _, _ = _evaluate_once(
                        sample=sample,
                        skip_mask=skip_mask,
                        args=args,
                        norm_stats=norm_stats,
                        data_config=data_config,
                        train_config=train_config,
                        dataset=dataset,
                        policy=policy,
                    )
                    importance = float(metrics.get(metric_name, float("nan")))
                    entropy = float(sample["entropy"][segment_id])
                    writer.writerow(
                        {
                            "sample_id": sample["sample_id"],
                            "segment_id": segment_id,
                            "start": start,
                            "end": end,
                            "entropy": entropy,
                            "importance": importance,
                            "importance_metric": metric_name,
                        }
                    )
                    entropies.append(entropy)
                    importances.append(importance)
            except Exception as exc:
                LOGGER.exception("Oracle failed for %s", path)
                if not args.continue_on_error:
                    raise
                _status(f"Skipping oracle item {path.name}: {exc}")

    spearman = _spearmanr(entropies, importances)
    return {
        "aggregate": {
            "oracle_num_segments": len(importances),
            "oracle_importance_metric": metric_name,
            "entropy_importance_spearman": spearman,
        },
        "action_metrics_source": action_source,
        "oracle_importance_csv": str(importance_path),
    }


def _run_pruning(
    *,
    args: argparse.Namespace,
    files: Sequence[pathlib.Path],
    output_dir: pathlib.Path,
    norm_stats: dict[str, Any] | None,
    data_config: _config.DataConfig,
    train_config: _config.TrainConfig,
    dataset: _data_loader.Dataset | None,
    policy: Any | None,
) -> dict[str, Any]:
    per_sample_path = output_dir / "per_sample_metrics.csv"
    fields = [
        "sample_id",
        "item_index",
        "strategy",
        "trial",
        "segment_mode",
        "chunk_size",
        "prune_ratio",
        "replacement",
        "num_segments",
        "skipped_segments",
        "action_metrics_source",
        *METRIC_FIELDS,
    ]
    trial_count = args.num_random_trials if args.strategy == "random" else 1
    accumulators = [MetricAccumulator() for _ in range(trial_count)]
    observed_action_sources: set[str] = set()

    with per_sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for file_position, path in enumerate(files):
            try:
                sample = _prepare_sample(
                    path,
                    file_position=file_position,
                    args=args,
                    norm_stats=norm_stats,
                    data_config=data_config,
                )
                for trial in range(trial_count):
                    if args.strategy == "random":
                        rng = np.random.default_rng(args.seed + sample["item_index"] * 1009 + trial)
                        skip_mask = _select_skip_mask(
                            sample["entropy"],
                            strategy=args.strategy,
                            prune_ratio=args.prune_ratio,
                            min_keep_segments=args.min_keep_segments,
                            rng=rng,
                        )
                    else:
                        skip_mask = _select_skip_mask(
                            sample["entropy"],
                            strategy=args.strategy,
                            prune_ratio=args.prune_ratio,
                            min_keep_segments=args.min_keep_segments,
                        )

                    metrics, skipped_segments, action_source = _evaluate_once(
                        sample=sample,
                        skip_mask=skip_mask,
                        args=args,
                        norm_stats=norm_stats,
                        data_config=data_config,
                        train_config=train_config,
                        dataset=dataset,
                        policy=policy,
                    )
                    observed_action_sources.add(action_source)
                    accumulators[trial].update(metrics)
                    writer.writerow(
                        {
                            "sample_id": sample["sample_id"],
                            "item_index": sample["item_index"],
                            "strategy": args.strategy,
                            "trial": trial if args.strategy == "random" else "",
                            "segment_mode": args.segment_mode,
                            "chunk_size": args.chunk_size,
                            "prune_ratio": args.prune_ratio,
                            "replacement": args.replacement,
                            "num_segments": len(sample["segments"]),
                            "skipped_segments": ";".join(str(idx) for idx in skipped_segments),
                            "action_metrics_source": action_source,
                            **metrics,
                        }
                    )
            except Exception as exc:
                LOGGER.exception("Pruning eval failed for %s", path)
                if not args.continue_on_error:
                    raise
                _status(f"Skipping pruning item {path.name}: {exc}")

    trial_aggregates = [accumulator.mean() for accumulator in accumulators]
    if args.strategy == "random":
        aggregate = {field: _safe_mean([trial[field] for trial in trial_aggregates]) for field in METRIC_FIELDS}
        random_std = _aggregate_std(trial_aggregates)
    else:
        aggregate = trial_aggregates[0]
        random_std = None
    action_source = (
        sorted(observed_action_sources)[0]
        if len(observed_action_sources) == 1
        else ",".join(sorted(observed_action_sources))
    )
    return {
        "aggregate": aggregate,
        "random_trial_aggregates": trial_aggregates if args.strategy == "random" else None,
        "random_trial_std": random_std,
        "action_metrics_source": action_source,
        "per_sample_metrics_csv": str(per_sample_path),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    entropy_dir = pathlib.Path(args.entropy_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    checkpoint_dir = _resolve_checkpoint_dir(
        train_config,
        args.checkpoint_dir,
        allow_config_default=args.enable_action_injection,
    )
    norm_stats = _load_norm_stats(train_config, data_config, checkpoint_dir)
    dataset = _maybe_create_expert_dataset(args, train_config, data_config)
    policy = _maybe_create_injection_policy(args, train_config, checkpoint_dir, norm_stats, dataset)
    files = _entropy_files(entropy_dir, args.max_items)

    _status(
        f"Running strategy={args.strategy}, segment_mode={args.segment_mode}, "
        f"chunk_size={args.chunk_size}, prune_ratio={args.prune_ratio}, files={len(files)}"
    )
    if policy is None:
        _status("Action injection unavailable or disabled; using static projected action metrics.")
    else:
        _status("Action injection enabled; pruned coarse actions will be sent through the final action head.")

    if args.strategy == "oracle":
        result = _run_oracle(
            args=args,
            files=files,
            output_dir=output_dir,
            norm_stats=norm_stats,
            data_config=data_config,
            train_config=train_config,
            dataset=dataset,
            policy=policy,
        )
        random_std = None
    else:
        result = _run_pruning(
            args=args,
            files=files,
            output_dir=output_dir,
            norm_stats=norm_stats,
            data_config=data_config,
            train_config=train_config,
            dataset=dataset,
            policy=policy,
        )
        random_std = result.get("random_trial_std")

    action_source = result["action_metrics_source"]
    action_injection_available = policy is not None
    _write_metadata(
        args,
        output_dir,
        checkpoint_dir=checkpoint_dir,
        action_injection_available=action_injection_available,
        action_metrics_source=action_source,
    )
    if args.strategy != "oracle":
        _write_summary_csv(
            output_dir / "pruning_summary.csv",
            args=args,
            aggregate=result["aggregate"],
            random_std=random_std,
            action_metrics_source=action_source,
            action_injection_available=action_injection_available,
        )

    metrics_json = {
        "aggregate": result["aggregate"],
        "random_trial_aggregates": result.get("random_trial_aggregates"),
        "random_trial_std": random_std,
        "action_metrics_source": action_source,
        "action_injection_available": action_injection_available,
        "closed_loop_pruning_rollout_available": False,
        "outputs": {key: value for key, value in result.items() if key.endswith("_csv")},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    _status(f"Wrote metrics to {output_dir / 'metrics.json'}")
    return metrics_json


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = build_arg_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
