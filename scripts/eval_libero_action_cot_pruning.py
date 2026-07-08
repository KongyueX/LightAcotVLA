"""Closed-loop LIBERO evaluation for Action-CoT pruning quality.

This evaluator compares rollout success under:

* full: normal ACoT policy inference.
* cached_override: generate full coarse_actions, then re-run the final action
  head with those same coarse_actions through the override hook.
* pruned_override: estimate online MC entropy from coarse_actions, prune low
  entropy Action-CoT segments, then re-run the final action head with the
  pruned coarse trajectory.
* true_entropy_skip: estimate online MC entropy, select one fixed L=5 segment,
  then run the true segment-skip model path that generates fewer explicit
  Action-CoT tokens.

Important: pruned_override is a closed-loop quality test, not a deployable
speed test. It still calls full ACoT first to estimate online entropy. Real
speedup requires a trained skip head or a model path that avoids generating or
consuming skipped Action-CoT tokens.
The deployable timing fields for true_entropy_skip report only the final
true-skip model call, assuming the skip decision is supplied by a future skip
head or offline selector.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import eval_action_cot_pruning as stage_b
from openpi.action_cot import compression as acot_compression
from openpi.shared import normalize as _normalize


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
PROFILE_TIMING_FIELDS = (
    "vlm_ms",
    "implicit_action_reasoner_ms",
    "coarse_action_expert_ms",
    "action_expert_ms",
    "profile_overhead_ms",
)


def _ensure_libero_import_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get("LIBERO_ROOT"),
        repo_root / "third_party" / "libero",
        repo_root / "LIBERO-plus",
        repo_root / "LIBERO",
        repo_root.parent / "LIBERO-plus",
        repo_root.parent / "LIBERO",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = pathlib.Path(candidate)
        if (path / "libero" / "libero").exists() or (path / "libero").exists():
            sys.path.insert(0, str(path))


_ensure_libero_import_path()
try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as exc:
    if exc.name != "libero":
        raise
    raise ModuleNotFoundError(
        "The current Python process cannot import LIBERO. If LIBERO is already available on this server, "
        "run with the same PYTHONPATH/environment used by the old LIBERO eval, or set LIBERO_ROOT to the "
        "directory that contains the top-level libero package, for example:\n"
        "  export LIBERO_ROOT=/root/ACoT-VLA/LIBERO-plus\n"
        "  export PYTHONPATH=$LIBERO_ROOT:$PYTHONPATH\n"
        "If the repo submodule is actually missing, initialize it with:\n"
        "  cd /root/ACoT-VLA\n"
        "  git submodule update --init --recursive third_party/libero\n"
        "  uv pip install -e third_party/libero\n"
        "Then rerun this script."
    ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy_api_key", "--policy-api-key", default=None)
    parser.add_argument("--resize_size", "--resize-size", type=int, default=224)
    parser.add_argument("--replan_steps", "--replan-steps", type=int, default=5)
    parser.add_argument(
        "--adaptive_replanning",
        "--adaptive-replanning",
        choices=("none", "action", "entropy", "action_entropy"),
        default="none",
        help=(
            "Dynamically choose how many low-level actions to execute before replanning. "
            "action uses action-chunk stability, entropy uses Action-CoT uncertainty, and "
            "action_entropy combines both. The baseline is none, equivalent to fixed --replan_steps."
        ),
    )
    parser.add_argument(
        "--adaptive_replan_horizons",
        "--adaptive-replan-horizons",
        nargs="*",
        type=int,
        default=[5, 6, 7, 8, 9, 10],
        help=(
            "Candidate execution horizons H in environment/control steps. For entropy-based adaptive replanning, "
            "horizons below --replan_steps are ignored so entropy can only keep or lengthen the baseline horizon."
        ),
    )
    parser.add_argument(
        "--adaptive_replan_entropy_mode",
        "--adaptive-replan-entropy-mode",
        choices=("none", "coarse_proxy", "online_mc"),
        default="none",
        help=(
            "Entropy source for adaptive replanning. coarse_proxy is a cheap single-sample coarse-action "
            "variation proxy. online_mc computes Stage-B-style MC predictive entropy and is useful as an oracle, "
            "but its extra policy calls are included in non-deployable timing."
        ),
    )
    parser.add_argument(
        "--adaptive_replan_entropy_samples",
        "--adaptive-replan-entropy-samples",
        type=int,
        default=4,
        help="Number of MC samples for --adaptive_replan_entropy_mode online_mc.",
    )
    parser.add_argument(
        "--adaptive_replan_entropy_low_quantile",
        "--adaptive-replan-entropy-low-quantile",
        type=float,
        default=0.33,
        help="Running-history quantile below which entropy lengthens the horizon.",
    )
    parser.add_argument(
        "--adaptive_replan_entropy_high_quantile",
        "--adaptive-replan-entropy-high-quantile",
        type=float,
        default=0.67,
        help="Running-history quantile above which entropy shortens the horizon.",
    )
    parser.add_argument(
        "--adaptive_replan_entropy_warmup",
        "--adaptive-replan-entropy-warmup",
        type=int,
        default=20,
        help="Number of entropy observations to collect before quantile-based entropy gating affects H.",
    )
    parser.add_argument(
        "--adaptive_replan_entropy_low",
        "--adaptive-replan-entropy-low",
        type=float,
        default=None,
        help="Optional absolute entropy threshold for lengthening H. Overrides the low quantile when set.",
    )
    parser.add_argument(
        "--adaptive_replan_entropy_high",
        "--adaptive-replan-entropy-high",
        type=float,
        default=None,
        help="Optional absolute entropy threshold for shortening H. Overrides the high quantile when set.",
    )
    parser.add_argument(
        "--adaptive_replan_jerk_low",
        "--adaptive-replan-jerk-low",
        type=float,
        default=0.25,
        help="Low action jerk ratio threshold for choosing a long horizon.",
    )
    parser.add_argument(
        "--adaptive_replan_jerk_high",
        "--adaptive-replan-jerk-high",
        type=float,
        default=0.75,
        help="High action jerk ratio threshold for choosing a short horizon.",
    )
    parser.add_argument(
        "--adaptive_replan_gripper_change_threshold",
        "--adaptive-replan-gripper-change-threshold",
        type=float,
        default=0.25,
        help="Max gripper change above which the horizon is capped at the default replan horizon.",
    )
    parser.add_argument(
        "--disable_adaptive_replan_stage_guard",
        "--disable-adaptive-replan-stage-guard",
        action="store_true",
        help=(
            "Disable the default stage-aware guard. By default entropy adaptive replanning caps H at "
            "--replan_steps during gripper/action-instability phases."
        ),
    )
    parser.add_argument(
        "--adaptive_replan_action_delta_high",
        "--adaptive-replan-action-delta-high",
        type=float,
        default=0.08,
        help="Action-delta threshold above which stage-aware entropy replanning is capped at --replan_steps.",
    )
    parser.add_argument(
        "--adaptive_replan_stage_guard_jerk_high",
        "--adaptive-replan-stage-guard-jerk-high",
        type=float,
        default=0.65,
        help="Action jerk-ratio threshold above which stage-aware entropy replanning is capped at --replan_steps.",
    )
    parser.add_argument("--task_suite_name", "--task-suite-name", default="libero_spatial")
    parser.add_argument("--num_steps_wait", "--num-steps-wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", "--num-trials-per-task", type=int, default=1)
    parser.add_argument("--max_tasks", "--max-tasks", type=int, default=None)
    parser.add_argument("--task_start", "--task-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--video_out_path", "--video-out-path", default=None)
    parser.add_argument("--save_videos", "--save-videos", action="store_true")
    parser.add_argument(
        "--action_cot_denoising_steps",
        "--action-cot-denoising-steps",
        type=int,
        default=10,
        help="Denoising iterations for explicit Action-CoT coarse-action generation.",
    )
    parser.add_argument(
        "--action_cot_dynamic_denoising_steps",
        action="store_true",
        help="Use the trained Action-CoT step head to choose coarse denoising iterations.",
    )

    parser.add_argument(
        "--mode",
        choices=("full", "cached_override", "pruned_override", "true_segment_skip", "true_entropy_skip", "all"),
        default="all",
        help=(
            "Use all to run full, cached_override, pruned_override, and true_entropy_skip "
            "on the same task initial states."
        ),
    )
    parser.add_argument("--entropy_samples", "--entropy-samples", type=int, default=4)
    parser.add_argument("--strategy", choices=("low_entropy", "high_entropy", "random"), default="low_entropy")
    parser.add_argument("--segment_mode", "--segment-mode", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=5)
    parser.add_argument("--prune_ratio", "--prune-ratio", type=float, default=0.3)
    parser.add_argument("--replacement", choices=("interp", "hold", "zero"), default="interp")
    parser.add_argument("--min_keep_segments", "--min-keep-segments", type=int, default=1)
    parser.add_argument("--min_len", "--min-len", type=int, default=3)
    parser.add_argument("--max_len", "--max-len", type=int, default=6)
    parser.add_argument("--max_segments", "--max-segments", type=int, default=5)
    parser.add_argument(
        "--true_skip_segment",
        "--true-skip-segment",
        type=int,
        default=1,
        help="Segment id for true_segment_skip. true_entropy_skip ignores this and selects by entropy.",
    )
    parser.add_argument(
        "--true_skip_chunk_size",
        "--true-skip-chunk-size",
        type=int,
        default=5,
        help="Fixed chunk size used by true segment-skip modes. The current model path supports L=5.",
    )
    parser.add_argument("--gripper_indices", "--gripper-indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--norm_stats_dir",
        "--norm-stats-dir",
        default=None,
        help="Optional norm_stats directory for entropy computation. If omitted, raw coarse_actions are used.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.replan_steps <= 0:
        raise ValueError("--replan_steps must be positive.")
    if not args.adaptive_replan_horizons:
        raise ValueError("--adaptive_replan_horizons must contain at least one value.")
    if any(horizon <= 0 for horizon in args.adaptive_replan_horizons):
        raise ValueError("--adaptive_replan_horizons values must be positive.")
    if args.adaptive_replan_entropy_samples <= 0:
        raise ValueError("--adaptive_replan_entropy_samples must be positive.")
    if not 0.0 <= args.adaptive_replan_entropy_low_quantile <= 1.0:
        raise ValueError("--adaptive_replan_entropy_low_quantile must be in [0, 1].")
    if not 0.0 <= args.adaptive_replan_entropy_high_quantile <= 1.0:
        raise ValueError("--adaptive_replan_entropy_high_quantile must be in [0, 1].")
    if args.adaptive_replan_entropy_low_quantile > args.adaptive_replan_entropy_high_quantile:
        raise ValueError("--adaptive_replan_entropy_low_quantile must be <= high quantile.")
    if args.adaptive_replan_entropy_warmup < 0:
        raise ValueError("--adaptive_replan_entropy_warmup must be non-negative.")
    if args.adaptive_replan_jerk_low < 0 or args.adaptive_replan_jerk_high < 0:
        raise ValueError("--adaptive_replan_jerk thresholds must be non-negative.")
    if args.adaptive_replan_jerk_low > args.adaptive_replan_jerk_high:
        raise ValueError("--adaptive_replan_jerk_low must be <= --adaptive_replan_jerk_high.")
    if args.adaptive_replan_action_delta_high < 0:
        raise ValueError("--adaptive_replan_action_delta_high must be non-negative.")
    if args.adaptive_replan_stage_guard_jerk_high < 0:
        raise ValueError("--adaptive_replan_stage_guard_jerk_high must be non-negative.")
    if args.adaptive_replanning in ("entropy", "action_entropy") and args.adaptive_replan_entropy_mode == "none":
        raise ValueError(
            "--adaptive_replanning entropy/action_entropy requires --adaptive_replan_entropy_mode "
            "coarse_proxy or online_mc."
        )
    if args.adaptive_replanning in ("entropy", "action_entropy") and args.replan_steps < 5:
        raise ValueError("Entropy adaptive replanning requires --replan_steps >= 5 for speed validation.")
    if args.num_trials_per_task <= 0:
        raise ValueError("--num_trials_per_task must be positive.")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max_tasks must be positive when set.")
    if args.entropy_samples <= 0:
        raise ValueError("--entropy_samples must be positive.")
    if args.action_cot_denoising_steps is not None and args.action_cot_denoising_steps <= 0:
        raise ValueError("--action_cot_denoising_steps must be positive when set.")
    if not 0.0 <= args.prune_ratio <= 1.0:
        raise ValueError("--prune_ratio must be in [0, 1].")
    if args.true_skip_chunk_size != 5:
        raise ValueError("The current true segment-skip model path supports --true_skip_chunk_size 5 only.")
    if not 0 <= args.true_skip_segment <= 2:
        raise ValueError("--true_skip_segment must be 0, 1, or 2.")


def _status(message: str) -> None:
    print(f"[eval_libero_action_cot_pruning] {message}", flush=True)


def _with_action_cot_denoising_steps(element: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    request = dict(element)
    if args.action_cot_denoising_steps is not None and not args.action_cot_dynamic_denoising_steps:
        request["action_cot_denoising_steps"] = np.asarray(args.action_cot_denoising_steps, dtype=np.int32)
    if args.action_cot_dynamic_denoising_steps:
        request["action_cot_dynamic_denoising_steps"] = np.asarray(True)
    return request


def _max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220 * 3
    if task_suite_name == "libero_object":
        return 280 * 3
    if task_suite_name == "libero_goal":
        return 300 * 3
    if task_suite_name == "libero_10":
        return 520 * 3
    if task_suite_name == "libero_90":
        return 400 * 3
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _safe_close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _env_success(env) -> bool:
    candidates = [env, getattr(env, "env", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for name in ("_check_success", "check_success"):
            fn = getattr(candidate, name, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    continue
    return False


def _is_terminated_episode_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "executing action in terminated episode" in str(exc)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _observation_to_policy_input(obs: dict[str, Any], task_description: str, resize_size: int) -> dict[str, Any]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }


def _load_norm_stats(norm_stats_dir: str | None) -> dict[str, Any] | None:
    if norm_stats_dir is None:
        return None
    return _normalize.load(norm_stats_dir)


def _segment(coarse_normalized: np.ndarray, args: argparse.Namespace) -> list[acot_compression.Segment]:
    if args.segment_mode == "fixed":
        return acot_compression.segment_fixed(coarse_normalized, args.chunk_size)
    return acot_compression.segment_adaptive(
        coarse_normalized,
        min_len=args.min_len,
        max_len=args.max_len,
        max_segments=args.max_segments,
        gripper_indices=args.gripper_indices,
    )


def _prune_online_coarse(
    coarse_samples: np.ndarray,
    *,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, list[int], int]:
    coarse_samples = np.asarray(coarse_samples, dtype=np.float64)
    coarse_normalized, _ = stage_b._normalize_actions(
        coarse_samples,
        norm_stats,
        use_quantiles=False,
        preferred_key="coarse_actions",
    )
    coarse_mean_normalized = np.mean(coarse_normalized, axis=0)
    segments = _segment(coarse_mean_normalized, args)
    entropy = acot_compression.compute_mc_predictive_entropy(coarse_normalized, segments)
    skip_mask = stage_b._select_skip_mask(
        entropy,
        strategy=args.strategy,
        prune_ratio=args.prune_ratio,
        min_keep_segments=args.min_keep_segments,
        rng=rng,
    )
    frame_skip_mask = acot_compression.expand_segment_mask(skip_mask, segments, t_len=coarse_samples.shape[1])
    zero_value = stage_b._zero_action_value(
        norm_stats,
        use_quantiles=False,
        preferred_key="coarse_actions",
        dim=coarse_samples.shape[-1],
    )
    coarse_pruned = stage_b._replace_masked_frames(
        coarse_samples[0],
        frame_skip_mask,
        replacement=args.replacement,
        zero_value=zero_value,
    )
    skipped_segments = [idx for idx, value in enumerate(skip_mask.tolist()) if value]
    return coarse_pruned.astype(np.float32), float(np.mean(frame_skip_mask)), skipped_segments, len(segments)


def _select_online_true_skip_segment(
    coarse_samples: np.ndarray,
    *,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[int, float, np.ndarray]:
    coarse_samples = np.asarray(coarse_samples, dtype=np.float64)
    if coarse_samples.ndim != 3:
        raise ValueError(f"coarse_samples must have shape [K, T, D], got {coarse_samples.shape}.")
    if coarse_samples.shape[1] != 15:
        raise ValueError(f"true segment-skip currently requires a 15-frame coarse horizon, got {coarse_samples.shape[1]}.")

    coarse_normalized, _ = stage_b._normalize_actions(
        coarse_samples,
        norm_stats,
        use_quantiles=False,
        preferred_key="coarse_actions",
    )
    coarse_mean_normalized = np.mean(coarse_normalized, axis=0)
    segments = acot_compression.segment_fixed(coarse_mean_normalized, chunk_size=args.true_skip_chunk_size)
    if len(segments) != 3 or any((end - start) != args.true_skip_chunk_size for start, end in segments):
        raise ValueError(f"true segment-skip expects three L=5 segments, got {segments}.")

    entropy = acot_compression.compute_mc_predictive_entropy(coarse_normalized, segments)
    skip_mask = stage_b._select_skip_mask(
        entropy,
        strategy=args.strategy,
        prune_ratio=args.prune_ratio,
        min_keep_segments=args.min_keep_segments,
        rng=rng,
    )
    skipped_segments = np.flatnonzero(skip_mask)
    if skipped_segments.size != 1:
        raise ValueError(
            "true segment-skip currently supports exactly one skipped L=5 segment. "
            f"Got {skipped_segments.size}; use prune_ratio in (0, 1/3] with three segments."
        )
    return int(skipped_segments[0]), float(args.true_skip_chunk_size / coarse_samples.shape[1]), entropy


def _timing_ms(result: dict[str, Any], wall_ms: float) -> tuple[float, float, dict[str, float]]:
    policy_timing = result.get("policy_timing", {}) if isinstance(result, dict) else {}
    server_timing = result.get("server_timing", {}) if isinstance(result, dict) else {}
    policy_ms = float(policy_timing.get("infer_ms", np.nan))
    server_ms = float(server_timing.get("infer_ms", wall_ms))
    stage_timing = {field: _timing_float(policy_timing, field) for field in PROFILE_TIMING_FIELDS}
    return policy_ms, server_ms, stage_timing


def _timing_float(timing: dict[str, Any], field: str) -> float:
    value = timing.get(field)
    if value is None:
        return float("nan")
    return float(value)


def _sum_stage_timing(timings: list[dict[str, float]]) -> dict[str, float]:
    result = {}
    for field in PROFILE_TIMING_FIELDS:
        values = [float(timing.get(field, float("nan"))) for timing in timings]
        finite = [value for value in values if np.isfinite(value)]
        result[field] = float(np.sum(finite)) if finite else float("nan")
    return result


def _prefixed_stage_timing(prefix: str, timing: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{field}": float(timing.get(field, float("nan"))) for field in PROFILE_TIMING_FIELDS}


def _infer(
    client,
    element: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], float, float, float, dict[str, float]]:
    request = dict(element)
    request["policy_seed"] = np.asarray(seed, dtype=np.int64)
    request["profile_policy_timing"] = np.asarray(True)
    start = time.perf_counter()
    result = client.infer(request)
    wall_ms = (time.perf_counter() - start) * 1000.0
    policy_ms, server_ms, stage_timing = _timing_ms(result, wall_ms)
    return result, wall_ms, policy_ms, server_ms, stage_timing


def _denoising_steps_from_result(result: dict[str, Any]) -> float:
    if "action_cot_denoising_steps" not in result:
        return float("nan")
    value = np.asarray(result["action_cot_denoising_steps"])
    if value.size == 0:
        return float("nan")
    return float(value.reshape(-1)[0])


def _adaptive_replanning_enabled(args: argparse.Namespace) -> bool:
    return args.adaptive_replanning != "none"


def _adaptive_uses_entropy(args: argparse.Namespace) -> bool:
    return args.adaptive_replanning in ("entropy", "action_entropy") and args.adaptive_replan_entropy_mode != "none"


def _adaptive_uses_action(args: argparse.Namespace) -> bool:
    return args.adaptive_replanning in ("action", "action_entropy")


def _candidate_horizons(args: argparse.Namespace, action_len: int) -> list[int]:
    horizons = sorted(set(int(value) for value in args.adaptive_replan_horizons + [args.replan_steps]))
    if _adaptive_uses_entropy(args):
        horizons = [value for value in horizons if value >= args.replan_steps]
    horizons = [value for value in horizons if value > 0 and value <= action_len]
    if horizons:
        return horizons
    return [min(args.replan_steps, action_len)]


def _nearest_horizon_index(horizons: list[int], target: int) -> int:
    return min(range(len(horizons)), key=lambda idx: (abs(horizons[idx] - target), horizons[idx]))


def _gripper_action_indices(action_dim: int, args: argparse.Namespace) -> np.ndarray:
    if args.gripper_indices:
        indices = np.asarray(args.gripper_indices, dtype=np.int64)
        indices = np.where(indices < 0, indices + action_dim, indices)
        indices = indices[(0 <= indices) & (indices < action_dim)]
        if indices.size:
            return indices
    return np.asarray([action_dim - 1], dtype=np.int64)


def _action_stability_metrics(action_chunk: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        return {
            "action_delta": float("nan"),
            "action_jerk": float("nan"),
            "action_jerk_ratio": float("nan"),
            "gripper_change": float("nan"),
            "gripper_event": 0.0,
        }

    action_dim = actions.shape[-1]
    gripper_indices = _gripper_action_indices(action_dim, args)
    non_gripper_mask = np.ones(action_dim, dtype=bool)
    non_gripper_mask[gripper_indices] = False
    body_actions = actions[:, non_gripper_mask] if np.any(non_gripper_mask) else actions

    if body_actions.shape[0] >= 2:
        deltas = body_actions[1:] - body_actions[:-1]
        delta_norm = np.linalg.norm(deltas, axis=-1)
        action_delta = float(np.mean(delta_norm))
    else:
        deltas = np.zeros((0, body_actions.shape[-1]), dtype=np.float64)
        action_delta = 0.0

    if deltas.shape[0] >= 2:
        curvature = deltas[1:] - deltas[:-1]
        jerk_norm = np.linalg.norm(curvature, axis=-1)
        action_jerk = float(np.mean(jerk_norm))
    else:
        action_jerk = 0.0

    action_jerk_ratio = float(action_jerk / (action_delta + 1e-6))

    if gripper_indices.size and actions.shape[0] >= 2:
        gripper = actions[:, gripper_indices]
        gripper_change = float(np.max(np.abs(gripper[1:] - gripper[:-1])))
    else:
        gripper_change = 0.0

    return {
        "action_delta": action_delta,
        "action_jerk": action_jerk,
        "action_jerk_ratio": action_jerk_ratio,
        "gripper_change": gripper_change,
        "gripper_event": float(gripper_change >= args.adaptive_replan_gripper_change_threshold),
    }


def _coarse_variation_proxy(
    coarse_actions: np.ndarray,
    *,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    coarse_actions = np.asarray(coarse_actions, dtype=np.float64)
    coarse_normalized, _ = stage_b._normalize_actions(
        coarse_actions[None, ...],
        norm_stats,
        use_quantiles=False,
        preferred_key="coarse_actions",
    )
    coarse = coarse_normalized[0]
    segments = _segment(coarse, args)
    scores = []
    for start, end in segments:
        segment = coarse[start:end]
        if segment.shape[0] <= 1:
            scores.append(0.0)
            continue
        deltas = segment[1:] - segment[:-1]
        delta_score = float(np.mean(np.linalg.norm(deltas, axis=-1)))
        if deltas.shape[0] >= 2:
            curvature = deltas[1:] - deltas[:-1]
            curve_score = float(np.mean(np.linalg.norm(curvature, axis=-1)))
        else:
            curve_score = 0.0
        scores.append(delta_score + 0.5 * curve_score)

    values = np.asarray(scores, dtype=np.float64)
    return {
        "adaptive_entropy_source": "coarse_proxy",
        "adaptive_entropy_score": float(np.max(values)) if values.size else float("nan"),
        "adaptive_entropy_mean": float(np.mean(values)) if values.size else float("nan"),
        "adaptive_entropy_max": float(np.max(values)) if values.size else float("nan"),
        "adaptive_entropy_std": float(np.std(values)) if values.size else float("nan"),
        "adaptive_entropy_num_segments": int(len(segments)),
    }


def _mc_entropy_info(
    coarse_samples: np.ndarray,
    *,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    coarse_samples = np.asarray(coarse_samples, dtype=np.float64)
    coarse_normalized, _ = stage_b._normalize_actions(
        coarse_samples,
        norm_stats,
        use_quantiles=False,
        preferred_key="coarse_actions",
    )
    coarse_mean = np.mean(coarse_normalized, axis=0)
    segments = _segment(coarse_mean, args)
    entropy = acot_compression.compute_mc_predictive_entropy(coarse_normalized, segments)
    return {
        "adaptive_entropy_source": "online_mc",
        "adaptive_entropy_score": float(np.max(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_mean": float(np.mean(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_max": float(np.max(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_std": float(np.std(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_num_segments": int(len(segments)),
    }


def _entropy_thresholds(
    entropy_history: list[float],
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    if args.adaptive_replan_entropy_low is not None and args.adaptive_replan_entropy_high is not None:
        return float(args.adaptive_replan_entropy_low), float(args.adaptive_replan_entropy_high), "absolute"

    if len(entropy_history) < args.adaptive_replan_entropy_warmup:
        return float("nan"), float("nan"), "warmup"

    values = np.asarray(entropy_history, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), "missing"

    low = float(np.quantile(values, args.adaptive_replan_entropy_low_quantile))
    high = float(np.quantile(values, args.adaptive_replan_entropy_high_quantile))
    return low, high, "running_quantile"


def _low_entropy_horizon_index(
    horizons: list[int],
    default_idx: int,
    entropy_score: float,
    entropy_history: list[float],
    args: argparse.Namespace,
) -> int:
    if default_idx >= len(horizons) - 1:
        return default_idx

    values = np.asarray(entropy_history, dtype=np.float64)
    values = values[np.isfinite(values)]
    low_quantile = float(args.adaptive_replan_entropy_low_quantile)
    if values.size == 0 or low_quantile <= 0:
        return min(default_idx + 1, len(horizons) - 1)

    percentile = float(np.mean(values <= entropy_score))
    extension = float(np.clip((low_quantile - percentile) / max(low_quantile, 1e-6), 0.0, 1.0))
    available = len(horizons) - default_idx - 1
    return min(default_idx + 1 + int(round(extension * max(available - 1, 0))), len(horizons) - 1)


def _stage_guard_info(metrics: dict[str, float], args: argparse.Namespace) -> tuple[bool, str]:
    if args.disable_adaptive_replan_stage_guard:
        return False, ""

    reasons = []
    if bool(metrics["gripper_event"]):
        reasons.append("gripper_change")

    action_delta = float(metrics["action_delta"])
    if np.isfinite(action_delta) and action_delta >= args.adaptive_replan_action_delta_high:
        reasons.append("action_delta")

    jerk_ratio = float(metrics["action_jerk_ratio"])
    if np.isfinite(jerk_ratio) and jerk_ratio >= args.adaptive_replan_stage_guard_jerk_high:
        reasons.append("jerk_ratio")

    return bool(reasons), ",".join(reasons)


def _select_replan_horizon(
    action_chunk: np.ndarray,
    *,
    timing: dict[str, Any],
    args: argparse.Namespace,
    entropy_history: list[float],
) -> tuple[int, dict[str, Any]]:
    action_len = int(np.asarray(action_chunk).shape[0])
    if not _adaptive_replanning_enabled(args):
        horizon = min(args.replan_steps, action_len)
        return horizon, {
            "adaptive_replan_horizon": horizon,
            "adaptive_replan_reason": "fixed",
            "adaptive_entropy_decision": 0,
        }

    horizons = _candidate_horizons(args, action_len)
    default_idx = _nearest_horizon_index(horizons, args.replan_steps)
    bucket_idx = default_idx
    reasons = []

    metrics = _action_stability_metrics(action_chunk, args)
    if _adaptive_uses_action(args):
        jerk_ratio = float(metrics["action_jerk_ratio"])
        if np.isfinite(jerk_ratio) and jerk_ratio <= args.adaptive_replan_jerk_low:
            bucket_idx = len(horizons) - 1
            reasons.append("action_low_jerk")
        elif np.isfinite(jerk_ratio) and jerk_ratio >= args.adaptive_replan_jerk_high:
            bucket_idx = 0
            reasons.append("action_high_jerk")
        else:
            bucket_idx = default_idx
            reasons.append("action_mid_jerk")

        if bool(metrics["gripper_event"]):
            bucket_idx = min(bucket_idx, default_idx)
            reasons.append("gripper_guard")

    entropy_score = float(timing.get("adaptive_entropy_score", float("nan")))
    entropy_low, entropy_high, entropy_threshold_source = _entropy_thresholds(entropy_history, args)
    entropy_decision = 0
    low_entropy_target_idx = default_idx
    if _adaptive_uses_entropy(args) and np.isfinite(entropy_score):
        if np.isfinite(entropy_low) and np.isfinite(entropy_high):
            if entropy_score <= entropy_low:
                entropy_decision = 1
                low_entropy_target_idx = _low_entropy_horizon_index(
                    horizons,
                    default_idx,
                    entropy_score,
                    entropy_history,
                    args,
                )
                bucket_idx = max(bucket_idx, low_entropy_target_idx)
                reasons.append("entropy_low")
            elif entropy_score >= entropy_high:
                entropy_decision = -1
                bucket_idx = default_idx
                reasons.append("entropy_high")
            else:
                bucket_idx = min(bucket_idx, default_idx)
                reasons.append("entropy_mid")
        else:
            reasons.append(f"entropy_{entropy_threshold_source}")

    stage_guarded, stage_guard_reason = _stage_guard_info(metrics, args)
    if _adaptive_uses_entropy(args) and stage_guarded:
        bucket_idx = min(bucket_idx, default_idx)
        reasons.append(f"stage_guard:{stage_guard_reason}")

    horizon = horizons[bucket_idx]
    info = {
        "adaptive_replan_horizon": horizon,
        "adaptive_replan_reason": "+".join(reasons) if reasons else "default",
        "adaptive_action_delta": float(metrics["action_delta"]),
        "adaptive_action_jerk": float(metrics["action_jerk"]),
        "adaptive_action_jerk_ratio": float(metrics["action_jerk_ratio"]),
        "adaptive_gripper_change": float(metrics["gripper_change"]),
        "adaptive_gripper_event": float(metrics["gripper_event"]),
        "adaptive_entropy_score": entropy_score,
        "adaptive_entropy_mean": float(timing.get("adaptive_entropy_mean", float("nan"))),
        "adaptive_entropy_max": float(timing.get("adaptive_entropy_max", float("nan"))),
        "adaptive_entropy_std": float(timing.get("adaptive_entropy_std", float("nan"))),
        "adaptive_entropy_num_segments": int(timing.get("adaptive_entropy_num_segments", 0) or 0),
        "adaptive_entropy_low_threshold": entropy_low,
        "adaptive_entropy_high_threshold": entropy_high,
        "adaptive_entropy_threshold_source": entropy_threshold_source,
        "adaptive_entropy_decision": entropy_decision,
        "adaptive_low_entropy_target_horizon": horizons[low_entropy_target_idx],
        "adaptive_stage_guard": float(stage_guarded),
        "adaptive_stage_guard_reason": stage_guard_reason,
    }
    return horizon, info


def _query_action(
    client,
    element: dict[str, Any],
    *,
    mode: str,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
    seed: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "full":
        result, wall_ms, policy_ms, server_ms, stage_timing = _infer(
            client,
            _with_action_cot_denoising_steps(element, args),
            seed=seed,
        )
        full_wall_ms = [wall_ms]
        full_policy_ms = [policy_ms]
        full_server_ms = [server_ms]
        full_stage_timings = [stage_timing]
        full_denoising_steps = [_denoising_steps_from_result(result)]
        entropy_info: dict[str, Any] = {}
        coarse_actions = np.asarray(result["coarse_actions"], dtype=np.float32) if "coarse_actions" in result else None

        if _adaptive_uses_entropy(args) and coarse_actions is not None:
            if args.adaptive_replan_entropy_mode == "coarse_proxy":
                entropy_info = _coarse_variation_proxy(coarse_actions, args=args, norm_stats=norm_stats)
            elif args.adaptive_replan_entropy_mode == "online_mc":
                coarse_samples = [coarse_actions]
                full_element = _with_action_cot_denoising_steps(element, args)
                for sample_idx in range(1, args.adaptive_replan_entropy_samples):
                    extra_result, extra_wall_ms, extra_policy_ms, extra_server_ms, extra_stage_timing = _infer(
                        client,
                        full_element,
                        seed=seed + sample_idx,
                    )
                    if "coarse_actions" not in extra_result:
                        raise KeyError("Policy output does not contain coarse_actions for online entropy sampling.")
                    coarse_samples.append(np.asarray(extra_result["coarse_actions"], dtype=np.float32))
                    full_wall_ms.append(extra_wall_ms)
                    full_policy_ms.append(extra_policy_ms)
                    full_server_ms.append(extra_server_ms)
                    full_stage_timings.append(extra_stage_timing)
                    full_denoising_steps.append(_denoising_steps_from_result(extra_result))
                entropy_info = _mc_entropy_info(np.stack(coarse_samples, axis=0), args=args, norm_stats=norm_stats)

        total_wall_ms = float(np.nansum(full_wall_ms))
        total_policy_ms = float(np.nansum(full_policy_ms))
        total_server_ms = float(np.nansum(full_server_ms))
        total_stage_timing = _sum_stage_timing(full_stage_timings)
        return np.asarray(result["actions"]), {
            "wall_ms": total_wall_ms,
            "policy_ms": total_policy_ms,
            "server_ms": total_server_ms,
            "deployable_wall_ms": wall_ms,
            "deployable_policy_ms": policy_ms,
            "deployable_server_ms": server_ms,
            **_prefixed_stage_timing("policy", total_stage_timing),
            **_prefixed_stage_timing("deployable_policy", stage_timing),
            "skip_ratio": float("nan"),
            "num_segments": 0,
            "skipped_segments": "",
            "full_calls": len(full_wall_ms),
            "override_calls": 0,
            "true_skip_calls": 0,
            "action_cot_denoising_steps_used": _mean(full_denoising_steps),
            **entropy_info,
        }

    if mode == "true_segment_skip":
        true_element = _with_action_cot_denoising_steps(element, args)
        true_element["action_cot_skip_segment"] = np.asarray(args.true_skip_segment, dtype=np.int32)
        result, wall_ms, policy_ms, server_ms, stage_timing = _infer(client, true_element, seed=seed)
        return np.asarray(result["actions"]), {
            "wall_ms": wall_ms,
            "policy_ms": policy_ms,
            "server_ms": server_ms,
            "deployable_wall_ms": wall_ms,
            "deployable_policy_ms": policy_ms,
            "deployable_server_ms": server_ms,
            **_prefixed_stage_timing("policy", stage_timing),
            **_prefixed_stage_timing("deployable_policy", stage_timing),
            "skip_ratio": float(args.true_skip_chunk_size / 15),
            "num_segments": 3,
            "skipped_segments": str(args.true_skip_segment),
            "full_calls": 0,
            "override_calls": 0,
            "true_skip_calls": 1,
            "action_cot_denoising_steps_used": _denoising_steps_from_result(result),
        }

    entropy_samples = 1 if mode == "cached_override" else args.entropy_samples
    coarse_samples = []
    full_wall_ms = []
    full_policy_ms = []
    full_server_ms = []
    full_stage_timings = []
    full_denoising_steps = []
    full_element = _with_action_cot_denoising_steps(element, args)
    for sample_idx in range(entropy_samples):
        result, wall_ms, policy_ms, server_ms, stage_timing = _infer(client, full_element, seed=seed + sample_idx)
        if "coarse_actions" not in result:
            raise KeyError("Policy output does not contain coarse_actions.")
        coarse_samples.append(np.asarray(result["coarse_actions"], dtype=np.float32))
        full_wall_ms.append(wall_ms)
        full_policy_ms.append(policy_ms)
        full_server_ms.append(server_ms)
        full_stage_timings.append(stage_timing)
        full_denoising_steps.append(_denoising_steps_from_result(result))

    if mode == "cached_override":
        coarse_override = coarse_samples[0]
        skip_ratio = float("nan")
        skipped_segments = []
        num_segments = 0
    elif mode == "true_entropy_skip":
        skip_segment, skip_ratio, entropy = _select_online_true_skip_segment(
            np.stack(coarse_samples, axis=0),
            args=args,
            norm_stats=norm_stats,
            rng=rng,
        )
        true_element = _with_action_cot_denoising_steps(element, args)
        true_element["action_cot_skip_segment"] = np.asarray(skip_segment, dtype=np.int32)
        result, skip_wall_ms, skip_policy_ms, skip_server_ms, skip_stage_timing = _infer(
            client,
            true_element,
            seed=seed,
        )
        return np.asarray(result["actions"]), {
            "wall_ms": float(np.nansum(full_wall_ms) + skip_wall_ms),
            "policy_ms": float(np.nansum(full_policy_ms) + skip_policy_ms),
            "server_ms": float(np.nansum(full_server_ms) + skip_server_ms),
            "deployable_wall_ms": skip_wall_ms,
            "deployable_policy_ms": skip_policy_ms,
            "deployable_server_ms": skip_server_ms,
            **_prefixed_stage_timing("policy", _sum_stage_timing([*full_stage_timings, skip_stage_timing])),
            **_prefixed_stage_timing("deployable_policy", skip_stage_timing),
            "skip_ratio": skip_ratio,
            "num_segments": 3,
            "skipped_segments": str(skip_segment),
            "skip_segment_entropy": float(entropy[skip_segment]),
            "full_calls": entropy_samples,
            "override_calls": 0,
            "true_skip_calls": 1,
            "action_cot_denoising_steps_used": _denoising_steps_from_result(result),
        }
    else:
        coarse_override, skip_ratio, skipped_segments, num_segments = _prune_online_coarse(
            np.stack(coarse_samples, axis=0),
            args=args,
            norm_stats=norm_stats,
            rng=rng,
        )

    override_element = _with_action_cot_denoising_steps(element, args)
    override_element["coarse_actions_override"] = coarse_override
    result, override_wall_ms, override_policy_ms, override_server_ms, override_stage_timing = _infer(
        client,
        override_element,
        seed=seed,
    )
    return np.asarray(result["actions"]), {
        "wall_ms": float(np.nansum(full_wall_ms) + override_wall_ms),
        "policy_ms": float(np.nansum(full_policy_ms) + override_policy_ms),
        "server_ms": float(np.nansum(full_server_ms) + override_server_ms),
        "deployable_wall_ms": override_wall_ms,
        "deployable_policy_ms": override_policy_ms,
        "deployable_server_ms": override_server_ms,
        **_prefixed_stage_timing("policy", _sum_stage_timing([*full_stage_timings, override_stage_timing])),
        **_prefixed_stage_timing("deployable_policy", override_stage_timing),
        "override_wall_ms": override_wall_ms,
        "override_policy_ms": override_policy_ms,
        "override_server_ms": override_server_ms,
        "skip_ratio": skip_ratio,
        "num_segments": num_segments,
        "skipped_segments": ";".join(str(idx) for idx in skipped_segments),
        "full_calls": entropy_samples,
        "override_calls": 1,
        "true_skip_calls": 0,
        "action_cot_denoising_steps_used": _mean(full_denoising_steps),
    }


def _modes(args: argparse.Namespace) -> list[str]:
    if args.mode == "all":
        return ["full", "cached_override", "pruned_override", "true_entropy_skip"]
    return [args.mode]


def _task_ids(args: argparse.Namespace, num_tasks: int) -> range:
    end = num_tasks if args.max_tasks is None else min(num_tasks, args.task_start + args.max_tasks)
    return range(args.task_start, end)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _run_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    client,
    task_suite,
    norm_stats: dict[str, Any] | None,
    output_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(args.seed)
    max_steps = _max_steps(args.task_suite_name)
    video_root = pathlib.Path(args.video_out_path) if args.video_out_path else output_dir / "videos"
    entropy_history: list[float] = []

    for task_id in _task_ids(args, task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language
        env = None

        for episode_idx in range(args.num_trials_per_task):
            if env is not None:
                _safe_close_env(env)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            _status(f"mode={mode} task={task_id} episode={episode_idx} task='{task_description}'")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            replay_images = []
            t = 0
            done = False
            total_return = 0.0
            infer_wall_ms = []
            infer_policy_ms = []
            infer_server_ms = []
            deployable_wall_ms = []
            deployable_policy_ms = []
            deployable_server_ms = []
            policy_stage_ms = {field: [] for field in PROFILE_TIMING_FIELDS}
            deployable_policy_stage_ms = {field: [] for field in PROFILE_TIMING_FIELDS}
            skip_ratios = []
            denoising_steps_used_values = []
            replan_horizons = []
            adaptive_action_delta = []
            adaptive_action_jerk = []
            adaptive_action_jerk_ratio = []
            adaptive_gripper_change = []
            adaptive_entropy_scores = []
            adaptive_entropy_means = []
            adaptive_entropy_maxes = []
            adaptive_entropy_stds = []
            adaptive_entropy_decisions = []
            adaptive_low_entropy_target_horizons = []
            adaptive_stage_guards = []
            adaptive_replan_reasons = []
            full_calls = 0
            override_calls = 0
            true_skip_calls = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    try:
                        obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    except Exception as exc:
                        if not _is_terminated_episode_error(exc):
                            raise
                        done = _env_success(env)
                        _status(
                            f"WARNING: mode={mode} task={task_id} episode={episode_idx} "
                            f"ended during wait step; success={done}"
                        )
                        break
                    total_return += float(reward)
                    t += 1
                    if done:
                        break
                    continue

                if not action_plan:
                    element = _observation_to_policy_input(obs, task_description, args.resize_size)
                    replay_images.append(element["observation/image"])
                    seed = args.seed + task_id * 1_000_000 + episode_idx * 10_000 + t
                    action_chunk, timing = _query_action(
                        client,
                        element,
                        mode=mode,
                        args=args,
                        norm_stats=norm_stats,
                        seed=seed,
                        rng=rng,
                    )
                    if not _adaptive_replanning_enabled(args) and len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"Need at least {args.replan_steps} actions, got {len(action_chunk)} for mode={mode}."
                        )
                    if len(action_chunk) <= 0:
                        raise ValueError(f"Policy returned an empty action chunk for mode={mode}.")
                    replan_horizon, replan_info = _select_replan_horizon(
                        action_chunk,
                        timing=timing,
                        args=args,
                        entropy_history=entropy_history,
                    )
                    replan_horizon = min(int(replan_horizon), len(action_chunk))
                    action_plan.extend(action_chunk[:replan_horizon])
                    replan_horizons.append(float(replan_horizon))
                    adaptive_replan_reasons.append(str(replan_info.get("adaptive_replan_reason", "")))
                    for key, target in (
                        ("adaptive_action_delta", adaptive_action_delta),
                        ("adaptive_action_jerk", adaptive_action_jerk),
                        ("adaptive_action_jerk_ratio", adaptive_action_jerk_ratio),
                        ("adaptive_gripper_change", adaptive_gripper_change),
                        ("adaptive_entropy_score", adaptive_entropy_scores),
                        ("adaptive_entropy_mean", adaptive_entropy_means),
                        ("adaptive_entropy_max", adaptive_entropy_maxes),
                        ("adaptive_entropy_std", adaptive_entropy_stds),
                        ("adaptive_entropy_decision", adaptive_entropy_decisions),
                        ("adaptive_low_entropy_target_horizon", adaptive_low_entropy_target_horizons),
                        ("adaptive_stage_guard", adaptive_stage_guards),
                    ):
                        value = float(replan_info.get(key, float("nan")))
                        if np.isfinite(value):
                            target.append(value)
                    entropy_score = float(replan_info.get("adaptive_entropy_score", float("nan")))
                    if _adaptive_uses_entropy(args) and np.isfinite(entropy_score):
                        entropy_history.append(entropy_score)

                    infer_wall_ms.append(float(timing["wall_ms"]))
                    infer_policy_ms.append(float(timing["policy_ms"]))
                    infer_server_ms.append(float(timing["server_ms"]))
                    deployable_wall_ms.append(float(timing.get("deployable_wall_ms", timing["wall_ms"])))
                    deployable_policy_ms.append(float(timing.get("deployable_policy_ms", timing["policy_ms"])))
                    deployable_server_ms.append(float(timing.get("deployable_server_ms", timing["server_ms"])))
                    for field in PROFILE_TIMING_FIELDS:
                        value = float(timing.get(f"policy_{field}", float("nan")))
                        if np.isfinite(value):
                            policy_stage_ms[field].append(value)
                        deployable_value = float(timing.get(f"deployable_policy_{field}", float("nan")))
                        if np.isfinite(deployable_value):
                            deployable_policy_stage_ms[field].append(deployable_value)
                    if np.isfinite(float(timing["skip_ratio"])):
                        skip_ratios.append(float(timing["skip_ratio"]))
                    denoising_steps_used = timing.get("action_cot_denoising_steps_used", float("nan"))
                    if np.isfinite(float(denoising_steps_used)):
                        denoising_steps_used_values.append(float(denoising_steps_used))
                    full_calls += int(timing["full_calls"])
                    override_calls += int(timing["override_calls"])
                    true_skip_calls += int(timing.get("true_skip_calls", 0))

                action = action_plan.popleft()
                try:
                    obs, reward, done, _ = env.step(np.asarray(action).tolist())
                except Exception as exc:
                    if not _is_terminated_episode_error(exc):
                        raise
                    done = _env_success(env)
                    _status(
                        f"WARNING: mode={mode} task={task_id} episode={episode_idx} "
                        f"ended before action step; success={done}"
                    )
                    break
                total_return += float(reward)
                if done:
                    break
                t += 1

            success = bool(done)
            timeout = not success
            total_policy_calls = full_calls + override_calls + true_skip_calls
            deployable_policy_calls = len(deployable_wall_ms)
            row = {
                "mode": mode,
                "task_suite": args.task_suite_name,
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task_description,
                "episode": episode_idx,
                "success": int(success),
                "return": total_return,
                "steps": t,
                "timeout": int(timeout),
                "avg_wall_inference_ms": _mean(infer_wall_ms),
                "avg_policy_inference_ms": _mean(infer_policy_ms),
                "avg_server_inference_ms": _mean(infer_server_ms),
                "total_wall_inference_ms": float(np.nansum(infer_wall_ms)),
                "total_policy_inference_ms": float(np.nansum(infer_policy_ms)),
                "total_server_inference_ms": float(np.nansum(infer_server_ms)),
                "avg_deployable_wall_inference_ms": _mean(deployable_wall_ms),
                "avg_deployable_policy_inference_ms": _mean(deployable_policy_ms),
                "avg_deployable_server_inference_ms": _mean(deployable_server_ms),
                "primary_wall_inference_ms": _mean(deployable_wall_ms),
                "primary_policy_inference_ms": _mean(deployable_policy_ms),
                "primary_server_inference_ms": _mean(deployable_server_ms),
                "total_deployable_wall_inference_ms": float(np.nansum(deployable_wall_ms)),
                "total_deployable_policy_inference_ms": float(np.nansum(deployable_policy_ms)),
                "total_deployable_server_inference_ms": float(np.nansum(deployable_server_ms)),
                **{f"avg_policy_{field}": _mean(policy_stage_ms[field]) for field in PROFILE_TIMING_FIELDS},
                **{
                    f"avg_deployable_policy_{field}": _mean(deployable_policy_stage_ms[field])
                    for field in PROFILE_TIMING_FIELDS
                },
                "avg_skip_ratio": _mean(skip_ratios),
                "avg_action_cot_denoising_steps_used": _mean(denoising_steps_used_values),
                "avg_replan_horizon": _mean(replan_horizons),
                "min_replan_horizon": float(np.min(replan_horizons)) if replan_horizons else float("nan"),
                "max_replan_horizon": float(np.max(replan_horizons)) if replan_horizons else float("nan"),
                "avg_adaptive_action_delta": _mean(adaptive_action_delta),
                "avg_adaptive_action_jerk": _mean(adaptive_action_jerk),
                "avg_adaptive_action_jerk_ratio": _mean(adaptive_action_jerk_ratio),
                "avg_adaptive_gripper_change": _mean(adaptive_gripper_change),
                "avg_adaptive_entropy_score": _mean(adaptive_entropy_scores),
                "avg_adaptive_entropy_mean": _mean(adaptive_entropy_means),
                "avg_adaptive_entropy_max": _mean(adaptive_entropy_maxes),
                "avg_adaptive_entropy_std": _mean(adaptive_entropy_stds),
                "avg_adaptive_entropy_decision": _mean(adaptive_entropy_decisions),
                "avg_adaptive_low_entropy_target_horizon": _mean(adaptive_low_entropy_target_horizons),
                "avg_adaptive_stage_guard": _mean(adaptive_stage_guards),
                "adaptive_replan_reasons": ";".join(adaptive_replan_reasons),
                "num_replans": len(infer_wall_ms),
                "total_policy_calls": total_policy_calls,
                "deployable_policy_calls": deployable_policy_calls,
                "entropy_oracle_extra_calls": max(total_policy_calls - deployable_policy_calls, 0),
                "full_calls": full_calls,
                "override_calls": override_calls,
                "true_skip_calls": true_skip_calls,
            }
            rows.append(row)

            if args.save_videos:
                suffix = "success" if success else "failure"
                video_dir = video_root / mode / args.task_suite_name / suffix
                video_dir.mkdir(parents=True, exist_ok=True)
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    video_dir / f"task{task_id}_ep{episode_idx}_{task_segment}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            _status(
                f"mode={mode} task={task_id} episode={episode_idx} "
                f"success={success} observed_avg_wall_ms={row['avg_wall_inference_ms']:.2f} "
                f"primary_avg_wall_ms={row['primary_wall_inference_ms']:.2f} "
                f"entropy_extra_calls={row['entropy_oracle_extra_calls']}"
            )

        if env is not None:
            _safe_close_env(env)

    return rows


def _write_results(output_dir: pathlib.Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rollout_rows.csv"
    fieldnames = [
        "mode",
        "task_suite",
        "task_id",
        "task_name",
        "task_description",
        "episode",
        "success",
        "return",
        "steps",
        "timeout",
        "avg_wall_inference_ms",
        "avg_policy_inference_ms",
        "avg_server_inference_ms",
        "total_wall_inference_ms",
        "total_policy_inference_ms",
        "total_server_inference_ms",
        "avg_deployable_wall_inference_ms",
        "avg_deployable_policy_inference_ms",
        "avg_deployable_server_inference_ms",
        "primary_wall_inference_ms",
        "primary_policy_inference_ms",
        "primary_server_inference_ms",
        "total_deployable_wall_inference_ms",
        "total_deployable_policy_inference_ms",
        "total_deployable_server_inference_ms",
        *[f"avg_policy_{field}" for field in PROFILE_TIMING_FIELDS],
        *[f"avg_deployable_policy_{field}" for field in PROFILE_TIMING_FIELDS],
        "avg_skip_ratio",
        "avg_action_cot_denoising_steps_used",
        "avg_replan_horizon",
        "min_replan_horizon",
        "max_replan_horizon",
        "avg_adaptive_action_delta",
        "avg_adaptive_action_jerk",
        "avg_adaptive_action_jerk_ratio",
        "avg_adaptive_gripper_change",
        "avg_adaptive_entropy_score",
        "avg_adaptive_entropy_mean",
        "avg_adaptive_entropy_max",
        "avg_adaptive_entropy_std",
        "avg_adaptive_entropy_decision",
        "avg_adaptive_low_entropy_target_horizon",
        "avg_adaptive_stage_guard",
        "adaptive_replan_reasons",
        "num_replans",
        "total_policy_calls",
        "deployable_policy_calls",
        "entropy_oracle_extra_calls",
        "full_calls",
        "override_calls",
        "true_skip_calls",
    ]
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_task_path = output_dir / "per_task_summary.csv"
    per_task_fieldnames = [
        "mode",
        "task_suite",
        "task_id",
        "task_name",
        "task_description",
        "episodes",
        "success_rate",
        "average_return",
        "timeout_rate",
        "avg_steps",
        "sum_steps",
        "avg_wall_inference_ms",
        "avg_policy_inference_ms",
        "avg_server_inference_ms",
        "avg_total_wall_inference_ms_per_episode",
        "avg_total_policy_inference_ms_per_episode",
        "avg_total_server_inference_ms_per_episode",
        "sum_total_wall_inference_ms",
        "sum_total_policy_inference_ms",
        "sum_total_server_inference_ms",
        "avg_deployable_wall_inference_ms",
        "avg_deployable_policy_inference_ms",
        "avg_deployable_server_inference_ms",
        "avg_total_deployable_wall_inference_ms_per_episode",
        "avg_total_deployable_policy_inference_ms_per_episode",
        "avg_total_deployable_server_inference_ms_per_episode",
        "sum_total_deployable_wall_inference_ms",
        "sum_total_deployable_policy_inference_ms",
        "sum_total_deployable_server_inference_ms",
        "avg_num_replans_per_episode",
        "avg_total_policy_calls_per_episode",
        "avg_deployable_policy_calls_per_episode",
        "avg_entropy_oracle_extra_calls_per_episode",
        "avg_replan_horizon",
        "avg_min_replan_horizon",
        "avg_max_replan_horizon",
        "avg_adaptive_entropy_score",
        "avg_adaptive_entropy_decision",
        "avg_adaptive_low_entropy_target_horizon",
        "avg_adaptive_stage_guard",
    ]
    per_task_rows = []
    task_keys = sorted(
        {(row["mode"], row["task_id"]) for row in rows},
        key=lambda item: (str(item[0]), int(item[1])),
    )
    for mode, task_id in task_keys:
        subset = [row for row in rows if row["mode"] == mode and row["task_id"] == task_id]
        per_task_rows.append(
            {
                "mode": mode,
                "task_suite": subset[0]["task_suite"],
                "task_id": task_id,
                "task_name": subset[0]["task_name"],
                "task_description": subset[0]["task_description"],
                "episodes": len(subset),
                "success_rate": _mean([float(row["success"]) for row in subset]),
                "average_return": _mean([float(row["return"]) for row in subset]),
                "timeout_rate": _mean([float(row["timeout"]) for row in subset]),
                "avg_steps": _mean([float(row["steps"]) for row in subset]),
                "sum_steps": float(np.nansum([float(row["steps"]) for row in subset])),
                "avg_wall_inference_ms": _mean([float(row["avg_wall_inference_ms"]) for row in subset]),
                "avg_policy_inference_ms": _mean([float(row["avg_policy_inference_ms"]) for row in subset]),
                "avg_server_inference_ms": _mean([float(row["avg_server_inference_ms"]) for row in subset]),
                "avg_total_wall_inference_ms_per_episode": _mean(
                    [float(row["total_wall_inference_ms"]) for row in subset]
                ),
                "avg_total_policy_inference_ms_per_episode": _mean(
                    [float(row["total_policy_inference_ms"]) for row in subset]
                ),
                "avg_total_server_inference_ms_per_episode": _mean(
                    [float(row["total_server_inference_ms"]) for row in subset]
                ),
                "sum_total_wall_inference_ms": float(
                    np.nansum([float(row["total_wall_inference_ms"]) for row in subset])
                ),
                "sum_total_policy_inference_ms": float(
                    np.nansum([float(row["total_policy_inference_ms"]) for row in subset])
                ),
                "sum_total_server_inference_ms": float(
                    np.nansum([float(row["total_server_inference_ms"]) for row in subset])
                ),
                "avg_deployable_wall_inference_ms": _mean(
                    [float(row["avg_deployable_wall_inference_ms"]) for row in subset]
                ),
                "avg_deployable_policy_inference_ms": _mean(
                    [float(row["avg_deployable_policy_inference_ms"]) for row in subset]
                ),
                "avg_deployable_server_inference_ms": _mean(
                    [float(row["avg_deployable_server_inference_ms"]) for row in subset]
                ),
                "avg_total_deployable_wall_inference_ms_per_episode": _mean(
                    [float(row["total_deployable_wall_inference_ms"]) for row in subset]
                ),
                "avg_total_deployable_policy_inference_ms_per_episode": _mean(
                    [float(row["total_deployable_policy_inference_ms"]) for row in subset]
                ),
                "avg_total_deployable_server_inference_ms_per_episode": _mean(
                    [float(row["total_deployable_server_inference_ms"]) for row in subset]
                ),
                "sum_total_deployable_wall_inference_ms": float(
                    np.nansum([float(row["total_deployable_wall_inference_ms"]) for row in subset])
                ),
                "sum_total_deployable_policy_inference_ms": float(
                    np.nansum([float(row["total_deployable_policy_inference_ms"]) for row in subset])
                ),
                "sum_total_deployable_server_inference_ms": float(
                    np.nansum([float(row["total_deployable_server_inference_ms"]) for row in subset])
                ),
                "avg_num_replans_per_episode": _mean([float(row["num_replans"]) for row in subset]),
                "avg_total_policy_calls_per_episode": _mean(
                    [float(row["total_policy_calls"]) for row in subset]
                ),
                "avg_deployable_policy_calls_per_episode": _mean(
                    [float(row["deployable_policy_calls"]) for row in subset]
                ),
                "avg_entropy_oracle_extra_calls_per_episode": _mean(
                    [float(row["entropy_oracle_extra_calls"]) for row in subset]
                ),
                "avg_replan_horizon": _mean([float(row["avg_replan_horizon"]) for row in subset]),
                "avg_min_replan_horizon": _mean([float(row["min_replan_horizon"]) for row in subset]),
                "avg_max_replan_horizon": _mean([float(row["max_replan_horizon"]) for row in subset]),
                "avg_adaptive_entropy_score": _mean(
                    [float(row["avg_adaptive_entropy_score"]) for row in subset]
                ),
                "avg_adaptive_entropy_decision": _mean(
                    [float(row["avg_adaptive_entropy_decision"]) for row in subset]
                ),
                "avg_adaptive_low_entropy_target_horizon": _mean(
                    [float(row["avg_adaptive_low_entropy_target_horizon"]) for row in subset]
                ),
                "avg_adaptive_stage_guard": _mean(
                    [float(row["avg_adaptive_stage_guard"]) for row in subset]
                ),
            }
        )
    with per_task_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_task_fieldnames)
        writer.writeheader()
        writer.writerows(per_task_rows)

    by_mode = {}
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {
            "episodes": len(subset),
            "success_rate": _mean([float(row["success"]) for row in subset]),
            "average_return": _mean([float(row["return"]) for row in subset]),
            "timeout_rate": _mean([float(row["timeout"]) for row in subset]),
            "avg_wall_inference_ms": _mean([float(row["avg_wall_inference_ms"]) for row in subset]),
            "avg_policy_inference_ms": _mean([float(row["avg_policy_inference_ms"]) for row in subset]),
            "avg_server_inference_ms": _mean([float(row["avg_server_inference_ms"]) for row in subset]),
            "avg_total_wall_inference_ms_per_episode": _mean(
                [float(row["total_wall_inference_ms"]) for row in subset]
            ),
            "avg_total_policy_inference_ms_per_episode": _mean(
                [float(row["total_policy_inference_ms"]) for row in subset]
            ),
            "avg_total_server_inference_ms_per_episode": _mean(
                [float(row["total_server_inference_ms"]) for row in subset]
            ),
            "avg_deployable_wall_inference_ms": _mean(
                [float(row["avg_deployable_wall_inference_ms"]) for row in subset]
            ),
            "avg_deployable_policy_inference_ms": _mean(
                [float(row["avg_deployable_policy_inference_ms"]) for row in subset]
            ),
            "avg_deployable_server_inference_ms": _mean(
                [float(row["avg_deployable_server_inference_ms"]) for row in subset]
            ),
            "primary_wall_inference_ms": _mean([float(row["primary_wall_inference_ms"]) for row in subset]),
            "primary_policy_inference_ms": _mean([float(row["primary_policy_inference_ms"]) for row in subset]),
            "primary_server_inference_ms": _mean([float(row["primary_server_inference_ms"]) for row in subset]),
            "avg_total_deployable_wall_inference_ms_per_episode": _mean(
                [float(row["total_deployable_wall_inference_ms"]) for row in subset]
            ),
            "avg_total_deployable_policy_inference_ms_per_episode": _mean(
                [float(row["total_deployable_policy_inference_ms"]) for row in subset]
            ),
            "avg_total_deployable_server_inference_ms_per_episode": _mean(
                [float(row["total_deployable_server_inference_ms"]) for row in subset]
            ),
            "primary_total_wall_inference_ms_per_episode": _mean(
                [float(row["total_deployable_wall_inference_ms"]) for row in subset]
            ),
            "primary_total_policy_inference_ms_per_episode": _mean(
                [float(row["total_deployable_policy_inference_ms"]) for row in subset]
            ),
            "primary_total_server_inference_ms_per_episode": _mean(
                [float(row["total_deployable_server_inference_ms"]) for row in subset]
            ),
            **{
                f"avg_policy_{field}": _mean([float(row[f"avg_policy_{field}"]) for row in subset])
                for field in PROFILE_TIMING_FIELDS
            },
            **{
                f"avg_deployable_policy_{field}": _mean(
                    [float(row[f"avg_deployable_policy_{field}"]) for row in subset]
                )
                for field in PROFILE_TIMING_FIELDS
            },
            "avg_skip_ratio": _mean([float(row["avg_skip_ratio"]) for row in subset]),
            "avg_action_cot_denoising_steps_used": _mean(
                [float(row["avg_action_cot_denoising_steps_used"]) for row in subset]
            ),
            "avg_replan_horizon": _mean([float(row["avg_replan_horizon"]) for row in subset]),
            "avg_min_replan_horizon": _mean([float(row["min_replan_horizon"]) for row in subset]),
            "avg_max_replan_horizon": _mean([float(row["max_replan_horizon"]) for row in subset]),
            "avg_adaptive_action_delta": _mean([float(row["avg_adaptive_action_delta"]) for row in subset]),
            "avg_adaptive_action_jerk": _mean([float(row["avg_adaptive_action_jerk"]) for row in subset]),
            "avg_adaptive_action_jerk_ratio": _mean(
                [float(row["avg_adaptive_action_jerk_ratio"]) for row in subset]
            ),
            "avg_adaptive_gripper_change": _mean([float(row["avg_adaptive_gripper_change"]) for row in subset]),
            "avg_adaptive_entropy_score": _mean([float(row["avg_adaptive_entropy_score"]) for row in subset]),
            "avg_adaptive_entropy_mean": _mean([float(row["avg_adaptive_entropy_mean"]) for row in subset]),
            "avg_adaptive_entropy_max": _mean([float(row["avg_adaptive_entropy_max"]) for row in subset]),
            "avg_adaptive_entropy_std": _mean([float(row["avg_adaptive_entropy_std"]) for row in subset]),
            "avg_adaptive_entropy_decision": _mean(
                [float(row["avg_adaptive_entropy_decision"]) for row in subset]
            ),
            "avg_adaptive_low_entropy_target_horizon": _mean(
                [float(row["avg_adaptive_low_entropy_target_horizon"]) for row in subset]
            ),
            "avg_adaptive_stage_guard": _mean([float(row["avg_adaptive_stage_guard"]) for row in subset]),
            "avg_num_replans_per_episode": _mean([float(row["num_replans"]) for row in subset]),
            "avg_total_policy_calls_per_episode": _mean([float(row["total_policy_calls"]) for row in subset]),
            "avg_deployable_policy_calls_per_episode": _mean(
                [float(row["deployable_policy_calls"]) for row in subset]
            ),
            "avg_entropy_oracle_extra_calls_per_episode": _mean(
                [float(row["entropy_oracle_extra_calls"]) for row in subset]
            ),
            "avg_full_calls_per_episode": _mean([float(row["full_calls"]) for row in subset]),
            "avg_override_calls_per_episode": _mean([float(row["override_calls"]) for row in subset]),
            "avg_true_skip_calls_per_episode": _mean([float(row["true_skip_calls"]) for row in subset]),
        }

    summary = {
        "config": {
            "host": args.host,
            "port": args.port,
            "task_suite_name": args.task_suite_name,
            "num_trials_per_task": args.num_trials_per_task,
            "max_tasks": args.max_tasks,
            "task_start": args.task_start,
            "mode": args.mode,
            "replan_steps": args.replan_steps,
            "adaptive_replanning": args.adaptive_replanning,
            "adaptive_replan_horizons": args.adaptive_replan_horizons,
            "adaptive_replan_entropy_mode": args.adaptive_replan_entropy_mode,
            "adaptive_replan_entropy_samples": args.adaptive_replan_entropy_samples,
            "adaptive_replan_entropy_low_quantile": args.adaptive_replan_entropy_low_quantile,
            "adaptive_replan_entropy_high_quantile": args.adaptive_replan_entropy_high_quantile,
            "adaptive_replan_entropy_warmup": args.adaptive_replan_entropy_warmup,
            "adaptive_replan_entropy_low": args.adaptive_replan_entropy_low,
            "adaptive_replan_entropy_high": args.adaptive_replan_entropy_high,
            "adaptive_replan_jerk_low": args.adaptive_replan_jerk_low,
            "adaptive_replan_jerk_high": args.adaptive_replan_jerk_high,
            "adaptive_replan_gripper_change_threshold": args.adaptive_replan_gripper_change_threshold,
            "adaptive_replan_stage_guard": not args.disable_adaptive_replan_stage_guard,
            "adaptive_replan_action_delta_high": args.adaptive_replan_action_delta_high,
            "adaptive_replan_stage_guard_jerk_high": args.adaptive_replan_stage_guard_jerk_high,
            "action_cot_denoising_steps": args.action_cot_denoising_steps,
            "action_cot_dynamic_denoising_steps": args.action_cot_dynamic_denoising_steps,
            "entropy_samples": args.entropy_samples,
            "strategy": args.strategy,
            "segment_mode": args.segment_mode,
            "chunk_size": args.chunk_size,
            "prune_ratio": args.prune_ratio,
            "replacement": args.replacement,
            "true_skip_segment": args.true_skip_segment,
            "true_skip_chunk_size": args.true_skip_chunk_size,
            "quality_mode": "online_mc_then_override",
            "note": (
                "pruned_override measures closed-loop quality. It is not a deployable speed path because it first "
                "runs full ACoT to estimate online entropy. true_entropy_skip also estimates entropy online here; "
                "its deployable timing fields measure only the final true-skip model call. For adaptive replanning "
                "with online_mc entropy, avg_total_* includes the Stage-B entropy MC calls, while "
                "avg_total_deployable_* treats entropy as an oracle/predicted signal and counts only the optimized "
                "action-producing policy calls after the entropy decision. primary_* fields mirror this deployable "
                "timing and are the intended fields for speed-success comparison."
            ),
        },
        "aggregate": by_mode,
        "outputs": {"rollout_rows_csv": str(rows_path), "per_task_summary_csv": str(per_task_path)},
    }
    if "full" in by_mode and "pruned_override" in by_mode:
        summary["comparison"] = {
            "success_drop_pruned_vs_full": by_mode["full"]["success_rate"]
            - by_mode["pruned_override"]["success_rate"],
            "return_drop_pruned_vs_full": by_mode["full"]["average_return"]
            - by_mode["pruned_override"]["average_return"],
        }
    if "full" in by_mode and "true_entropy_skip" in by_mode:
        comparison = summary.setdefault("comparison", {})
        comparison.update(
            {
                "success_drop_true_entropy_skip_vs_full": by_mode["full"]["success_rate"]
                - by_mode["true_entropy_skip"]["success_rate"],
                "return_drop_true_entropy_skip_vs_full": by_mode["full"]["average_return"]
                - by_mode["true_entropy_skip"]["average_return"],
                "deployable_wall_speedup_true_entropy_skip_vs_full_pct": (
                    1.0
                    - by_mode["true_entropy_skip"]["avg_deployable_wall_inference_ms"]
                    / by_mode["full"]["avg_deployable_wall_inference_ms"]
                )
                * 100.0,
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_args(args)
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    norm_stats = _load_norm_stats(args.norm_stats_dir)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )

    rows = []
    for mode in _modes(args):
        rows.extend(
            _run_mode(
                mode=mode,
                args=args,
                client=client,
                task_suite=task_suite,
                norm_stats=norm_stats,
                output_dir=output_dir,
            )
        )

    _write_results(output_dir, rows, args)
    _status(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
