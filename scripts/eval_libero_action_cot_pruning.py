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
import dataclasses
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


@dataclasses.dataclass
class AdaptiveHState:
    previous_horizon: int
    low_risk_streak: int = 0
    guard_cooldown: int = 0
    budget_balance: float = 0.0
    budget_horizon_sum: float = 0.0
    budget_decisions: int = 0
    budget_interventions: int = 0
    budget_limited_decisions: int = 0


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
            "Candidate execution horizons H in environment/control steps. Legacy entropy selectors ignore values "
            "below --replan_steps; budgeted_event_v2 fills every integer horizon from "
            "--adaptive_h_v2_min_horizon through the configured maximum."
        ),
    )
    parser.add_argument(
        "--adaptive_h_selector",
        "--adaptive-h-selector",
        choices=("legacy", "final_aac", "cot_aac", "guarded_cot_aac", "budgeted_event_v2"),
        default="guarded_cot_aac",
        help=(
            "Execution-horizon selector used by entropy-based adaptive replanning. final_aac uses final-action "
            "MC entropy, cot_aac uses time-aligned Action-CoT MC entropy, guarded_cot_aac adds stage guards and "
            "hysteresis, budgeted_event_v2 fuses per-timestep final/Action-CoT risk and permits sparse short-H "
            "interventions under an episode horizon budget, and legacy retains the previous mapping."
        ),
    )
    parser.add_argument(
        "--adaptive_h_entropy_algorithm",
        "--adaptive-h-entropy-algorithm",
        choices=("diagonal_logvar", "aac_grouped"),
        default="diagonal_logvar",
        help=(
            "Per-timestep MC entropy estimator. diagonal_logvar is the stable Stage-B-style baseline; "
            "aac_grouped uses translation/rotation Gaussian entropy plus binary gripper entropy."
        ),
    )
    parser.add_argument(
        "--adaptive_h_coarse_stride",
        "--adaptive-h-coarse-stride",
        type=float,
        default=2.0,
        help="Raw control-step stride represented by adjacent coarse Action-CoT tokens.",
    )
    parser.add_argument(
        "--adaptive_h_jump_mad_scale",
        "--adaptive-h-jump-mad-scale",
        type=float,
        default=1.5,
        help="MAD multiplier used to decide whether a prefix-entropy increase is a significant AAC jump.",
    )
    parser.add_argument(
        "--adaptive_h_entropy_eps",
        "--adaptive-h-entropy-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for horizon entropy estimation.",
    )
    parser.add_argument(
        "--adaptive_h_cov_shrinkage",
        "--adaptive-h-cov-shrinkage",
        type=float,
        default=1e-4,
        help="Diagonal covariance shrinkage for --adaptive_h_entropy_algorithm aac_grouped.",
    )
    parser.add_argument(
        "--adaptive_h_growth_limit",
        "--adaptive-h-growth-limit",
        type=int,
        default=1,
        help="Maximum increase in guarded execution horizon per replan decision.",
    )
    parser.add_argument(
        "--adaptive_h_low_risk_required",
        "--adaptive-h-low-risk-required",
        type=int,
        default=2,
        help="Consecutive low-risk decisions required before guarded execution horizon can increase.",
    )
    parser.add_argument(
        "--adaptive_h_guard_cooldown",
        "--adaptive-h-guard-cooldown",
        type=int,
        default=2,
        help="Number of subsequent replan decisions held at the baseline horizon after a stage guard fires.",
    )
    parser.add_argument(
        "--adaptive_h_v2_min_horizon",
        "--adaptive-h-v2-min-horizon",
        type=int,
        default=3,
        help="Minimum execution horizon allowed for a budgeted_event_v2 risk intervention.",
    )
    parser.add_argument(
        "--adaptive_h_v2_target_avg_horizon",
        "--adaptive-h-v2-target-avg-horizon",
        type=float,
        default=9.0,
        help=(
            "Target episode-average execution horizon for budgeted_event_v2. Horizons above this target earn "
            "budget credit and shorter horizons spend it."
        ),
    )
    parser.add_argument(
        "--adaptive_h_v2_initial_budget",
        "--adaptive-h-v2-initial-budget",
        type=float,
        default=6.0,
        help="Initial horizon-step credit, allowing one early H=3 intervention when the target is H=9.",
    )
    parser.add_argument(
        "--adaptive_h_v2_budget_capacity",
        "--adaptive-h-v2-budget-capacity",
        type=float,
        default=12.0,
        help="Maximum horizon-step credit retained by the V2 episode budget controller.",
    )
    parser.add_argument(
        "--adaptive_h_v2_risk_threshold",
        "--adaptive-h-v2-risk-threshold",
        type=float,
        default=1.5,
        help="Threshold on the fused robust per-timestep risk curve that triggers a V2 event.",
    )
    parser.add_argument(
        "--adaptive_h_v2_final_weight",
        "--adaptive-h-v2-final-weight",
        type=float,
        default=0.5,
        help="Fusion weight for final-action entropy and component disagreement in budgeted_event_v2.",
    )
    parser.add_argument(
        "--adaptive_h_v2_cot_weight",
        "--adaptive-h-v2-cot-weight",
        type=float,
        default=0.5,
        help="Fusion weight for time-aligned Action-CoT entropy in budgeted_event_v2.",
    )
    parser.add_argument(
        "--adaptive_h_v2_final_entropy_threshold",
        "--adaptive-h-v2-final-entropy-threshold",
        type=float,
        default=None,
        help="Optional globally calibrated absolute final-action entropy event threshold.",
    )
    parser.add_argument(
        "--adaptive_h_v2_cot_entropy_threshold",
        "--adaptive-h-v2-cot-entropy-threshold",
        type=float,
        default=None,
        help="Optional globally calibrated absolute time-aligned Action-CoT entropy event threshold.",
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
        default=5,
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
    if args.adaptive_h_coarse_stride <= 0:
        raise ValueError("--adaptive_h_coarse_stride must be positive.")
    if args.adaptive_h_jump_mad_scale < 0:
        raise ValueError("--adaptive_h_jump_mad_scale must be non-negative.")
    if args.adaptive_h_entropy_eps <= 0:
        raise ValueError("--adaptive_h_entropy_eps must be positive.")
    if args.adaptive_h_cov_shrinkage <= 0:
        raise ValueError("--adaptive_h_cov_shrinkage must be positive.")
    if args.adaptive_h_growth_limit <= 0:
        raise ValueError("--adaptive_h_growth_limit must be positive.")
    if args.adaptive_h_low_risk_required <= 0:
        raise ValueError("--adaptive_h_low_risk_required must be positive.")
    if args.adaptive_h_guard_cooldown < 0:
        raise ValueError("--adaptive_h_guard_cooldown must be non-negative.")
    if args.adaptive_h_v2_min_horizon <= 0:
        raise ValueError("--adaptive_h_v2_min_horizon must be positive.")
    if args.adaptive_h_v2_target_avg_horizon <= 0:
        raise ValueError("--adaptive_h_v2_target_avg_horizon must be positive.")
    if args.adaptive_h_v2_initial_budget < 0:
        raise ValueError("--adaptive_h_v2_initial_budget must be non-negative.")
    if args.adaptive_h_v2_budget_capacity <= 0:
        raise ValueError("--adaptive_h_v2_budget_capacity must be positive.")
    if args.adaptive_h_v2_initial_budget > args.adaptive_h_v2_budget_capacity:
        raise ValueError("--adaptive_h_v2_initial_budget must not exceed --adaptive_h_v2_budget_capacity.")
    if args.adaptive_h_v2_risk_threshold < 0:
        raise ValueError("--adaptive_h_v2_risk_threshold must be non-negative.")
    if args.adaptive_h_v2_final_weight < 0 or args.adaptive_h_v2_cot_weight < 0:
        raise ValueError("V2 entropy fusion weights must be non-negative.")
    if args.adaptive_h_v2_final_weight + args.adaptive_h_v2_cot_weight <= 0:
        raise ValueError("At least one V2 entropy fusion weight must be positive.")
    if args.adaptive_h_selector == "budgeted_event_v2":
        max_configured_horizon = max(args.adaptive_replan_horizons + [args.replan_steps])
        if args.adaptive_h_v2_min_horizon > max_configured_horizon:
            raise ValueError("--adaptive_h_v2_min_horizon exceeds every configured candidate horizon.")
        if args.adaptive_h_v2_target_avg_horizon > max_configured_horizon:
            raise ValueError("--adaptive_h_v2_target_avg_horizon exceeds the maximum configured horizon.")
        if args.adaptive_replanning not in ("entropy", "action_entropy"):
            raise ValueError("budgeted_event_v2 requires --adaptive_replanning entropy or action_entropy.")
        if args.adaptive_replan_entropy_mode != "online_mc":
            raise ValueError("budgeted_event_v2 requires --adaptive_replan_entropy_mode online_mc.")
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
    if (
        args.adaptive_replanning in ("entropy", "action_entropy")
        and args.adaptive_h_selector != "legacy"
        and args.adaptive_replan_entropy_mode != "online_mc"
    ):
        raise ValueError("AAC adaptive-H selectors require --adaptive_replan_entropy_mode online_mc.")
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


def _env_horizon(env) -> int | None:
    """Returns the smallest positive episode horizon exposed by an env wrapper chain."""
    queue = collections.deque([env])
    seen = set()
    horizons = []
    while queue:
        candidate = queue.popleft()
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        for name in ("horizon", "_horizon"):
            try:
                value = getattr(candidate, name, None)
                array = np.asarray(value)
                if array.size != 1:
                    continue
                horizon = int(array.reshape(()).item())
            except (TypeError, ValueError, OverflowError):
                continue
            if horizon > 0:
                horizons.append(horizon)

        for name in ("env", "_env", "unwrapped"):
            try:
                child = getattr(candidate, name, None)
            except Exception:
                continue
            if child is not None and id(child) not in seen:
                queue.append(child)

    return min(horizons) if horizons else None


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
    if args.adaptive_h_selector == "budgeted_event_v2":
        horizons = sorted(set(horizons + list(range(args.adaptive_h_v2_min_horizon, max(horizons) + 1))))
        horizons = [value for value in horizons if value >= args.adaptive_h_v2_min_horizon]
    elif _adaptive_uses_entropy(args):
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


def _diagonal_frame_entropy(samples: np.ndarray, eps: float) -> np.ndarray:
    variance = np.var(samples, axis=0)
    return np.mean(np.log(variance + eps), axis=-1)


def _gaussian_group_entropy(samples: np.ndarray, shrinkage: float) -> np.ndarray:
    sample_count, time_len, dim = samples.shape
    entropy = np.empty((time_len,), dtype=np.float64)
    constant = dim * (1.0 + math.log(2.0 * math.pi))
    for time_idx in range(time_len):
        values = samples[:, time_idx, :]
        centered = values - np.mean(values, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(sample_count, 1)
        covariance = covariance + shrinkage * np.eye(dim, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(covariance)
        entropy[time_idx] = 0.5 * (constant + logdet) if sign > 0 else float("nan")
    return entropy


def _aac_grouped_frame_entropy(samples: np.ndarray, eps: float, shrinkage: float) -> np.ndarray:
    if samples.shape[-1] < 7:
        return _diagonal_frame_entropy(samples, eps)

    actions = samples[..., :7]
    translation = _gaussian_group_entropy(actions[..., :3], shrinkage)
    rotation = _gaussian_group_entropy(actions[..., 3:6], shrinkage)
    gripper_closed = actions[..., 6] > 0
    probability = np.clip(np.mean(gripper_closed, axis=0), eps, 1.0 - eps)
    gripper = -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))
    return translation + rotation + gripper


def _frame_entropy(samples: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if args.adaptive_h_entropy_algorithm == "aac_grouped":
        return _aac_grouped_frame_entropy(
            samples,
            eps=args.adaptive_h_entropy_eps,
            shrinkage=args.adaptive_h_cov_shrinkage,
        )
    return _diagonal_frame_entropy(samples, args.adaptive_h_entropy_eps)


def _final_action_component_entropy(samples: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    time_len = samples.shape[1]
    action_dim = samples.shape[-1]

    translation_end = min(3, action_dim)
    rotation_start = min(3, action_dim)
    rotation_end = min(6, action_dim)
    translation = (
        _diagonal_frame_entropy(samples[..., :translation_end], args.adaptive_h_entropy_eps)
        if translation_end > 0
        else np.zeros((time_len,), dtype=np.float64)
    )
    rotation = (
        _diagonal_frame_entropy(samples[..., rotation_start:rotation_end], args.adaptive_h_entropy_eps)
        if rotation_end > rotation_start
        else np.zeros((time_len,), dtype=np.float64)
    )

    gripper_indices = _gripper_action_indices(action_dim, args)
    if gripper_indices.size:
        gripper_closed = samples[..., gripper_indices] > 0
        probability = np.clip(
            np.mean(gripper_closed, axis=0),
            args.adaptive_h_entropy_eps,
            1.0 - args.adaptive_h_entropy_eps,
        )
        gripper = np.mean(
            -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability)),
            axis=-1,
        )
    else:
        gripper = np.zeros((time_len,), dtype=np.float64)

    return {
        "translation": translation,
        "rotation": rotation,
        "gripper": gripper,
    }


def _align_coarse_entropy_to_actions(
    coarse_entropy: np.ndarray,
    *,
    action_len: int,
    coarse_stride: float,
) -> np.ndarray:
    coarse_entropy = np.asarray(coarse_entropy, dtype=np.float64)
    if coarse_entropy.size == 0 or action_len <= 0:
        return np.zeros((0,), dtype=np.float64)
    coarse_times = np.arange(coarse_entropy.size, dtype=np.float64) * coarse_stride
    action_times = np.arange(action_len, dtype=np.float64)
    return np.interp(action_times, coarse_times, coarse_entropy)


def _mc_entropy_info(
    coarse_samples: np.ndarray,
    action_samples: np.ndarray,
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
    coarse_frame_entropy = _frame_entropy(coarse_normalized, args)

    action_samples = np.asarray(action_samples, dtype=np.float64)
    action_normalized, _ = stage_b._normalize_actions(
        action_samples,
        norm_stats,
        use_quantiles=False,
        preferred_key="actions",
    )
    final_frame_entropy = _frame_entropy(action_normalized, args)
    final_components = _final_action_component_entropy(action_normalized, args)
    aligned_cot_entropy = _align_coarse_entropy_to_actions(
        coarse_frame_entropy,
        action_len=action_samples.shape[1],
        coarse_stride=args.adaptive_h_coarse_stride,
    )
    return {
        "adaptive_entropy_source": "online_mc",
        "adaptive_entropy_score": float(np.max(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_mean": float(np.mean(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_max": float(np.max(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_std": float(np.std(entropy)) if entropy.size else float("nan"),
        "adaptive_entropy_num_segments": int(len(segments)),
        "adaptive_cot_entropy_curve": aligned_cot_entropy.tolist(),
        "adaptive_final_entropy_curve": final_frame_entropy.tolist(),
        "adaptive_coarse_entropy_curve": coarse_frame_entropy.tolist(),
        "adaptive_final_translation_entropy_curve": final_components["translation"].tolist(),
        "adaptive_final_rotation_entropy_curve": final_components["rotation"].tolist(),
        "adaptive_final_gripper_entropy_curve": final_components["gripper"].tolist(),
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


def _aac_horizon_from_curve(
    entropy_curve: np.ndarray,
    horizons: list[int],
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    curve = np.asarray(entropy_curve, dtype=np.float64)
    prefix_entropy = {
        horizon: float(np.mean(curve[:horizon]))
        for horizon in horizons
        if 0 < horizon <= curve.size
    }
    valid_horizons = [horizon for horizon in horizons if horizon in prefix_entropy]
    if not valid_horizons:
        raise ValueError("No candidate execution horizon fits the entropy curve.")
    if len(valid_horizons) == 1:
        return valid_horizons[0], {
            "prefix_entropy": prefix_entropy,
            "entropy_jumps": {},
            "max_entropy_jump": float("nan"),
            "entropy_jump_threshold": float("nan"),
            "entropy_jump_significant": False,
        }

    jumps = {
        f"{left}->{right}": prefix_entropy[right] - prefix_entropy[left]
        for left, right in zip(valid_horizons[:-1], valid_horizons[1:], strict=True)
    }
    jump_values = np.asarray(list(jumps.values()), dtype=np.float64)
    median = float(np.median(jump_values))
    mad = float(np.median(np.abs(jump_values - median)))
    jump_threshold = max(0.0, median + args.adaptive_h_jump_mad_scale * mad)
    max_idx = int(np.argmax(jump_values))
    max_jump = float(jump_values[max_idx])
    significant = bool(max_jump > jump_threshold and max_jump > 0.0)
    raw_horizon = valid_horizons[max_idx] if significant else valid_horizons[-1]
    return raw_horizon, {
        "prefix_entropy": prefix_entropy,
        "entropy_jumps": jumps,
        "max_entropy_jump": max_jump,
        "entropy_jump_threshold": jump_threshold,
        "entropy_jump_significant": significant,
    }


def _robust_positive_risk(curve: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)

    center = float(np.median(finite))
    filled = np.where(np.isfinite(values), values, center)
    mad_scale = 1.4826 * float(np.median(np.abs(finite - center)))
    standard_scale = float(np.std(finite))
    scale = max(mad_scale, standard_scale, eps)
    return np.maximum((filled - center) / scale, 0.0)


def _v2_risk_curve(
    timing: dict[str, Any],
    *,
    action_len: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int | None, str, dict[str, np.ndarray]]:
    curve_keys = {
        "final": "adaptive_final_entropy_curve",
        "cot": "adaptive_cot_entropy_curve",
        "translation": "adaptive_final_translation_entropy_curve",
        "rotation": "adaptive_final_rotation_entropy_curve",
        "gripper": "adaptive_final_gripper_entropy_curve",
    }
    curves = {
        name: np.asarray(timing.get(key, []), dtype=np.float64)
        for name, key in curve_keys.items()
    }
    missing = [name for name, curve in curves.items() if curve.size < action_len]
    if missing:
        raise ValueError(
            "budgeted_event_v2 requires full per-timestep online-MC curves; "
            f"missing or short curves: {', '.join(missing)}."
        )
    curves = {name: curve[:action_len] for name, curve in curves.items()}

    component_risk = np.maximum.reduce(
        [
            _robust_positive_risk(curves["final"], args.adaptive_h_entropy_eps),
            _robust_positive_risk(curves["translation"], args.adaptive_h_entropy_eps),
            _robust_positive_risk(curves["rotation"], args.adaptive_h_entropy_eps),
            _robust_positive_risk(curves["gripper"], args.adaptive_h_entropy_eps),
        ]
    )
    cot_risk = _robust_positive_risk(curves["cot"], args.adaptive_h_entropy_eps)
    weight_sum = args.adaptive_h_v2_final_weight + args.adaptive_h_v2_cot_weight
    fused_risk = (
        args.adaptive_h_v2_final_weight * component_risk
        + args.adaptive_h_v2_cot_weight * cot_risk
    ) / weight_sum

    robust_event = fused_risk >= args.adaptive_h_v2_risk_threshold
    final_absolute_event = np.zeros((action_len,), dtype=bool)
    if args.adaptive_h_v2_final_entropy_threshold is not None:
        final_absolute_event = curves["final"] >= args.adaptive_h_v2_final_entropy_threshold
    cot_absolute_event = np.zeros((action_len,), dtype=bool)
    if args.adaptive_h_v2_cot_entropy_threshold is not None:
        cot_absolute_event = curves["cot"] >= args.adaptive_h_v2_cot_entropy_threshold

    event_mask = robust_event | final_absolute_event | cot_absolute_event
    event_indices = np.flatnonzero(event_mask)
    event_index = int(event_indices[0]) if event_indices.size else None
    event_sources = []
    if event_index is not None:
        if robust_event[event_index]:
            event_sources.append("robust_fusion")
        if final_absolute_event[event_index]:
            event_sources.append("final_absolute")
        if cot_absolute_event[event_index]:
            event_sources.append("cot_absolute")
    return fused_risk, event_index, ",".join(event_sources) if event_sources else "none", curves


def _event_horizon(event_index: int | None, horizons: list[int]) -> int:
    if event_index is None:
        return horizons[-1]

    safe_horizon = max(horizons[0], event_index)
    safe_candidates = [horizon for horizon in horizons if horizon <= safe_horizon]
    return safe_candidates[-1] if safe_candidates else horizons[0]


def _apply_horizon_budget(
    raw_horizon: int,
    horizons: list[int],
    *,
    args: argparse.Namespace,
    state: AdaptiveHState,
) -> tuple[int, dict[str, float]]:
    target = min(float(args.adaptive_h_v2_target_avg_horizon), float(horizons[-1]))
    balance_before = float(state.budget_balance)
    required_credit = max(target - raw_horizon, 0.0)
    final_horizon = raw_horizon
    budget_limited = False

    if required_credit > balance_before + 1e-9:
        affordable_floor = target - balance_before
        affordable = [
            horizon
            for horizon in horizons
            if horizon >= raw_horizon and horizon + 1e-9 >= affordable_floor
        ]
        final_horizon = affordable[0] if affordable else horizons[-1]
        budget_limited = final_horizon > raw_horizon

    balance_after = balance_before + final_horizon - target
    balance_after = float(np.clip(balance_after, 0.0, args.adaptive_h_v2_budget_capacity))
    state.budget_balance = balance_after
    state.budget_decisions += 1
    state.budget_horizon_sum += final_horizon
    intervention = final_horizon < horizons[-1]
    state.budget_interventions += int(intervention)
    state.budget_limited_decisions += int(budget_limited)

    return final_horizon, {
        "target_horizon": target,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "required_credit": required_credit,
        "budget_limited": float(budget_limited),
        "intervention": float(intervention),
        "cumulative_avg_horizon": state.budget_horizon_sum / state.budget_decisions,
        "intervention_rate": state.budget_interventions / state.budget_decisions,
        "budget_limited_rate": state.budget_limited_decisions / state.budget_decisions,
    }


def _select_budgeted_event_execution_horizon(
    action_chunk: np.ndarray,
    *,
    timing: dict[str, Any],
    args: argparse.Namespace,
    state: AdaptiveHState,
) -> tuple[int, dict[str, Any]]:
    action_len = int(np.asarray(action_chunk).shape[0])
    horizons = _candidate_horizons(args, action_len)
    if horizons[0] < args.adaptive_h_v2_min_horizon:
        raise ValueError("No V2 candidate horizon satisfies --adaptive_h_v2_min_horizon.")

    fused_risk, event_index, event_source, curves = _v2_risk_curve(
        timing,
        action_len=action_len,
        args=args,
    )
    raw_horizon = _event_horizon(event_index, horizons)
    previous_horizon = state.previous_horizon
    final_horizon, budget_info = _apply_horizon_budget(
        raw_horizon,
        horizons,
        args=args,
        state=state,
    )
    state.previous_horizon = final_horizon

    metrics = _action_stability_metrics(np.asarray(action_chunk)[:final_horizon], args)
    reasons = ["no_risk_event" if event_index is None else f"risk_event_t{event_index}"]
    if budget_info["budget_limited"]:
        reasons.append("budget_limited")

    event_risk = float(fused_risk[event_index]) if event_index is not None else float("nan")
    return final_horizon, {
        "adaptive_replan_horizon": final_horizon,
        "adaptive_replan_reason": "+".join(reasons),
        "adaptive_h_selector": "budgeted_event_v2",
        "adaptive_raw_execution_horizon": int(raw_horizon),
        "adaptive_guard_cap": int(horizons[-1]),
        "adaptive_previous_execution_horizon": int(previous_horizon),
        "adaptive_hysteresis_limited": 0.0,
        "adaptive_low_risk_streak": 0,
        "adaptive_guard_cooldown_remaining": 0,
        "adaptive_action_delta": float(metrics["action_delta"]),
        "adaptive_action_jerk": float(metrics["action_jerk"]),
        "adaptive_action_jerk_ratio": float(metrics["action_jerk_ratio"]),
        "adaptive_gripper_change": float(metrics["gripper_change"]),
        "adaptive_gripper_event": float(metrics["gripper_event"]),
        "adaptive_entropy_score": float(np.max(fused_risk)),
        "adaptive_entropy_mean": float(np.mean(fused_risk)),
        "adaptive_entropy_max": float(np.max(fused_risk)),
        "adaptive_entropy_std": float(np.std(fused_risk)),
        "adaptive_entropy_num_segments": int(fused_risk.size),
        "adaptive_entropy_low_threshold": float("nan"),
        "adaptive_entropy_high_threshold": float(args.adaptive_h_v2_risk_threshold),
        "adaptive_entropy_threshold_source": "v2_robust_or_global_absolute",
        "adaptive_entropy_decision": -1 if event_index is not None else 1,
        "adaptive_low_entropy_target_horizon": int(raw_horizon),
        "adaptive_stage_guard": 0.0,
        "adaptive_stage_guard_reason": "",
        "adaptive_entropy_curve": fused_risk.tolist(),
        "adaptive_prefix_entropy": {},
        "adaptive_entropy_jumps": {},
        "adaptive_max_entropy_jump": float("nan"),
        "adaptive_entropy_jump_threshold": float("nan"),
        "adaptive_entropy_jump_significant": float("nan"),
        "adaptive_v2_risk_event_index": float(event_index) if event_index is not None else float("nan"),
        "adaptive_v2_event_risk": event_risk,
        "adaptive_v2_event_source": event_source,
        "adaptive_v2_final_risk_curve": _robust_positive_risk(
            curves["final"], args.adaptive_h_entropy_eps
        ).tolist(),
        "adaptive_v2_cot_risk_curve": _robust_positive_risk(
            curves["cot"], args.adaptive_h_entropy_eps
        ).tolist(),
        "adaptive_v2_budget_balance_before": budget_info["balance_before"],
        "adaptive_v2_budget_balance_after": budget_info["balance_after"],
        "adaptive_v2_budget_required_credit": budget_info["required_credit"],
        "adaptive_v2_budget_limited": budget_info["budget_limited"],
        "adaptive_v2_intervention": budget_info["intervention"],
        "adaptive_v2_cumulative_avg_horizon": budget_info["cumulative_avg_horizon"],
        "adaptive_v2_intervention_rate": budget_info["intervention_rate"],
        "adaptive_v2_budget_limited_rate": budget_info["budget_limited_rate"],
    }


def _select_aac_execution_horizon(
    action_chunk: np.ndarray,
    *,
    timing: dict[str, Any],
    args: argparse.Namespace,
    entropy_history: list[float],
    state: AdaptiveHState,
) -> tuple[int, dict[str, Any]]:
    action_len = int(np.asarray(action_chunk).shape[0])
    horizons = _candidate_horizons(args, action_len)
    baseline_horizon = min(args.replan_steps, action_len)
    selector = args.adaptive_h_selector
    curve_key = "adaptive_final_entropy_curve" if selector == "final_aac" else "adaptive_cot_entropy_curve"
    entropy_curve = np.asarray(timing.get(curve_key, []), dtype=np.float64)
    if entropy_curve.size < baseline_horizon:
        raise ValueError(f"{selector} requires an entropy curve with at least {baseline_horizon} values.")

    raw_horizon, curve_info = _aac_horizon_from_curve(entropy_curve, horizons, args)
    entropy_score = float(np.max(entropy_curve[:baseline_horizon]))
    entropy_low, entropy_high, threshold_source = _entropy_thresholds(entropy_history, args)
    entropy_decision = 1 if raw_horizon > baseline_horizon else -1
    reasons = ["aac_entropy_jump" if curve_info["entropy_jump_significant"] else "aac_no_significant_jump"]

    guarded = selector == "guarded_cot_aac"
    if guarded:
        if not (np.isfinite(entropy_low) and np.isfinite(entropy_high)):
            raw_horizon = baseline_horizon
            entropy_decision = 0
            reasons.append(f"entropy_{threshold_source}")
        elif entropy_score >= entropy_high:
            raw_horizon = baseline_horizon
            entropy_decision = -1
            reasons.append("entropy_high")
        elif entropy_score <= entropy_low:
            entropy_decision = 1
            reasons.append("entropy_low")
        else:
            baseline_idx = _nearest_horizon_index(horizons, baseline_horizon)
            cautious_horizon = horizons[min(baseline_idx + 1, len(horizons) - 1)]
            raw_horizon = min(raw_horizon, cautious_horizon)
            entropy_decision = 0
            reasons.append("entropy_mid")

    metrics = _action_stability_metrics(np.asarray(action_chunk)[:raw_horizon], args)
    stage_guarded, stage_guard_reason = _stage_guard_info(metrics, args)
    guard_cap = horizons[-1]
    previous_horizon = state.previous_horizon
    hysteresis_limited = False

    if guarded:
        if stage_guarded:
            guard_cap = baseline_horizon
            state.guard_cooldown = args.adaptive_h_guard_cooldown
            state.low_risk_streak = 0
            reasons.append(f"stage_guard:{stage_guard_reason}")
        elif state.guard_cooldown > 0:
            guard_cap = baseline_horizon
            state.guard_cooldown -= 1
            stage_guarded = True
            stage_guard_reason = "cooldown"
            state.low_risk_streak = 0
            reasons.append("stage_guard:cooldown")

        candidate_horizon = min(raw_horizon, guard_cap)
        low_risk = entropy_decision > 0 and not stage_guarded
        if candidate_horizon > previous_horizon:
            state.low_risk_streak = state.low_risk_streak + 1 if low_risk else 0
            if state.low_risk_streak < args.adaptive_h_low_risk_required:
                final_horizon = previous_horizon
                hysteresis_limited = True
                reasons.append("hysteresis_wait")
            else:
                final_horizon = min(candidate_horizon, previous_horizon + args.adaptive_h_growth_limit)
                hysteresis_limited = final_horizon < candidate_horizon
                if hysteresis_limited:
                    reasons.append("hysteresis_growth_limit")
        else:
            final_horizon = candidate_horizon
            if candidate_horizon < previous_horizon:
                reasons.append("immediate_shrink")
            state.low_risk_streak = 0
    else:
        final_horizon = raw_horizon

    final_horizon = min(int(final_horizon), action_len)
    state.previous_horizon = final_horizon
    info = {
        "adaptive_replan_horizon": final_horizon,
        "adaptive_replan_reason": "+".join(reasons),
        "adaptive_h_selector": selector,
        "adaptive_raw_execution_horizon": int(raw_horizon),
        "adaptive_guard_cap": int(guard_cap),
        "adaptive_previous_execution_horizon": int(previous_horizon),
        "adaptive_hysteresis_limited": float(hysteresis_limited),
        "adaptive_low_risk_streak": int(state.low_risk_streak),
        "adaptive_guard_cooldown_remaining": int(state.guard_cooldown),
        "adaptive_action_delta": float(metrics["action_delta"]),
        "adaptive_action_jerk": float(metrics["action_jerk"]),
        "adaptive_action_jerk_ratio": float(metrics["action_jerk_ratio"]),
        "adaptive_gripper_change": float(metrics["gripper_change"]),
        "adaptive_gripper_event": float(metrics["gripper_event"]),
        "adaptive_entropy_score": entropy_score,
        "adaptive_entropy_mean": float(np.mean(entropy_curve)),
        "adaptive_entropy_max": float(np.max(entropy_curve)),
        "adaptive_entropy_std": float(np.std(entropy_curve)),
        "adaptive_entropy_num_segments": int(entropy_curve.size),
        "adaptive_entropy_low_threshold": entropy_low,
        "adaptive_entropy_high_threshold": entropy_high,
        "adaptive_entropy_threshold_source": threshold_source if guarded else "aac_curve",
        "adaptive_entropy_decision": entropy_decision,
        "adaptive_low_entropy_target_horizon": int(raw_horizon),
        "adaptive_stage_guard": float(stage_guarded),
        "adaptive_stage_guard_reason": stage_guard_reason,
        "adaptive_entropy_curve": entropy_curve.tolist(),
        "adaptive_prefix_entropy": curve_info["prefix_entropy"],
        "adaptive_entropy_jumps": curve_info["entropy_jumps"],
        "adaptive_max_entropy_jump": float(curve_info["max_entropy_jump"]),
        "adaptive_entropy_jump_threshold": float(curve_info["entropy_jump_threshold"]),
        "adaptive_entropy_jump_significant": float(curve_info["entropy_jump_significant"]),
    }
    return final_horizon, info


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
    state: AdaptiveHState,
) -> tuple[int, dict[str, Any]]:
    action_len = int(np.asarray(action_chunk).shape[0])
    if not _adaptive_replanning_enabled(args):
        horizon = min(args.replan_steps, action_len)
        return horizon, {
            "adaptive_replan_horizon": horizon,
            "adaptive_replan_reason": "fixed",
            "adaptive_entropy_decision": 0,
        }

    if _adaptive_uses_entropy(args) and args.adaptive_h_selector == "budgeted_event_v2":
        return _select_budgeted_event_execution_horizon(
            action_chunk,
            timing=timing,
            args=args,
            state=state,
        )

    if _adaptive_uses_entropy(args) and args.adaptive_h_selector != "legacy":
        return _select_aac_execution_horizon(
            action_chunk,
            timing=timing,
            args=args,
            entropy_history=entropy_history,
            state=state,
        )

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
    state.previous_horizon = horizon
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
                action_samples = [np.asarray(result["actions"], dtype=np.float32)]
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
                    action_samples.append(np.asarray(extra_result["actions"], dtype=np.float32))
                    full_wall_ms.append(extra_wall_ms)
                    full_policy_ms.append(extra_policy_ms)
                    full_server_ms.append(extra_server_ms)
                    full_stage_timings.append(extra_stage_timing)
                    full_denoising_steps.append(_denoising_steps_from_result(extra_result))
                entropy_info = _mc_entropy_info(
                    np.stack(coarse_samples, axis=0),
                    np.stack(action_samples, axis=0),
                    args=args,
                    norm_stats=norm_stats,
                )

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

    for task_id in _task_ids(args, task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_description = task.language
        env = None
        entropy_history: list[float] = []

        for episode_idx in range(args.num_trials_per_task):
            if env is not None:
                _safe_close_env(env)
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            _status(f"mode={mode} task={task_id} episode={episode_idx} task='{task_description}'")
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            environment_horizon = _env_horizon(env)
            episode_step_limit = max_steps + args.num_steps_wait
            if environment_horizon is not None:
                episode_step_limit = min(episode_step_limit, environment_horizon)
            replay_images = []
            t = 0
            done = False
            termination_reason = ""
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
            adaptive_raw_execution_horizons = []
            adaptive_guard_caps = []
            adaptive_max_entropy_jumps = []
            adaptive_hysteresis_limited = []
            adaptive_v2_risk_event_indices = []
            adaptive_v2_event_risks = []
            adaptive_v2_budget_balances = []
            adaptive_v2_budget_limited = []
            adaptive_v2_interventions = []
            adaptive_replan_reasons = []
            adaptive_h_decisions = []
            v2_enabled = _adaptive_uses_entropy(args) and args.adaptive_h_selector == "budgeted_event_v2"
            initial_horizon = (
                max(args.adaptive_replan_horizons)
                if v2_enabled
                else args.replan_steps
            )
            adaptive_h_state = AdaptiveHState(
                previous_horizon=initial_horizon,
                budget_balance=(
                    min(args.adaptive_h_v2_initial_budget, args.adaptive_h_v2_budget_capacity)
                    if v2_enabled
                    else 0.0
                ),
            )
            full_calls = 0
            override_calls = 0
            true_skip_calls = 0

            while t < episode_step_limit:
                if t < args.num_steps_wait:
                    try:
                        obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    except Exception as exc:
                        if not _is_terminated_episode_error(exc):
                            raise
                        done = _env_success(env)
                        termination_reason = "environment_terminated"
                        _status(
                            f"mode={mode} task={task_id} episode={episode_idx} environment terminated "
                            f"during wait at step={t}/{episode_step_limit}; success={done}"
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
                        state=adaptive_h_state,
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
                        ("adaptive_raw_execution_horizon", adaptive_raw_execution_horizons),
                        ("adaptive_guard_cap", adaptive_guard_caps),
                        ("adaptive_max_entropy_jump", adaptive_max_entropy_jumps),
                        ("adaptive_hysteresis_limited", adaptive_hysteresis_limited),
                        ("adaptive_v2_risk_event_index", adaptive_v2_risk_event_indices),
                        ("adaptive_v2_event_risk", adaptive_v2_event_risks),
                        ("adaptive_v2_budget_balance_after", adaptive_v2_budget_balances),
                        ("adaptive_v2_budget_limited", adaptive_v2_budget_limited),
                        ("adaptive_v2_intervention", adaptive_v2_interventions),
                    ):
                        value = float(replan_info.get(key, float("nan")))
                        if np.isfinite(value):
                            target.append(value)
                    entropy_score = float(replan_info.get("adaptive_entropy_score", float("nan")))
                    if _adaptive_uses_entropy(args) and np.isfinite(entropy_score):
                        entropy_history.append(entropy_score)
                    adaptive_h_decisions.append(
                        {
                            "environment_step": int(t),
                            "selector": str(replan_info.get("adaptive_h_selector", "legacy")),
                            "entropy_source": str(timing.get("adaptive_entropy_source", "none")),
                            "raw_execution_horizon": int(
                                replan_info.get("adaptive_raw_execution_horizon", replan_horizon)
                            ),
                            "guard_cap": int(replan_info.get("adaptive_guard_cap", len(action_chunk))),
                            "previous_execution_horizon": int(
                                replan_info.get("adaptive_previous_execution_horizon", replan_horizon)
                            ),
                            "execution_horizon": int(replan_horizon),
                            "entropy_score": entropy_score,
                            "entropy_low_threshold": float(
                                replan_info.get("adaptive_entropy_low_threshold", float("nan"))
                            ),
                            "entropy_high_threshold": float(
                                replan_info.get("adaptive_entropy_high_threshold", float("nan"))
                            ),
                            "entropy_threshold_source": str(
                                replan_info.get("adaptive_entropy_threshold_source", "none")
                            ),
                            "max_entropy_jump": float(
                                replan_info.get("adaptive_max_entropy_jump", float("nan"))
                            ),
                            "entropy_jump_threshold": float(
                                replan_info.get("adaptive_entropy_jump_threshold", float("nan"))
                            ),
                            "entropy_jump_significant": float(
                                replan_info.get("adaptive_entropy_jump_significant", float("nan"))
                            ),
                            "stage_guard": float(replan_info.get("adaptive_stage_guard", float("nan"))),
                            "stage_guard_reason": str(replan_info.get("adaptive_stage_guard_reason", "")),
                            "hysteresis_limited": float(
                                replan_info.get("adaptive_hysteresis_limited", float("nan"))
                            ),
                            "low_risk_streak": int(replan_info.get("adaptive_low_risk_streak", 0)),
                            "guard_cooldown_remaining": int(
                                replan_info.get("adaptive_guard_cooldown_remaining", 0)
                            ),
                            "v2_risk_event_index": float(
                                replan_info.get("adaptive_v2_risk_event_index", float("nan"))
                            ),
                            "v2_event_risk": float(
                                replan_info.get("adaptive_v2_event_risk", float("nan"))
                            ),
                            "v2_event_source": str(replan_info.get("adaptive_v2_event_source", "")),
                            "v2_budget_balance_before": float(
                                replan_info.get("adaptive_v2_budget_balance_before", float("nan"))
                            ),
                            "v2_budget_balance_after": float(
                                replan_info.get("adaptive_v2_budget_balance_after", float("nan"))
                            ),
                            "v2_budget_required_credit": float(
                                replan_info.get("adaptive_v2_budget_required_credit", float("nan"))
                            ),
                            "v2_budget_limited": float(
                                replan_info.get("adaptive_v2_budget_limited", float("nan"))
                            ),
                            "v2_intervention": float(
                                replan_info.get("adaptive_v2_intervention", float("nan"))
                            ),
                            "v2_cumulative_avg_horizon": float(
                                replan_info.get("adaptive_v2_cumulative_avg_horizon", float("nan"))
                            ),
                            "v2_intervention_rate": float(
                                replan_info.get("adaptive_v2_intervention_rate", float("nan"))
                            ),
                            "v2_budget_limited_rate": float(
                                replan_info.get("adaptive_v2_budget_limited_rate", float("nan"))
                            ),
                            "decision_reason": str(replan_info.get("adaptive_replan_reason", "")),
                            "entropy_curve": replan_info.get("adaptive_entropy_curve", []),
                            "v2_final_risk_curve": replan_info.get("adaptive_v2_final_risk_curve", []),
                            "v2_cot_risk_curve": replan_info.get("adaptive_v2_cot_risk_curve", []),
                            "prefix_entropy": replan_info.get("adaptive_prefix_entropy", {}),
                            "entropy_jumps": replan_info.get("adaptive_entropy_jumps", {}),
                        }
                    )

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
                    termination_reason = "environment_terminated"
                    _status(
                        f"mode={mode} task={task_id} episode={episode_idx} environment terminated "
                        f"before action at step={t}/{episode_step_limit}; success={done}"
                    )
                    break
                total_return += float(reward)
                if done:
                    break
                t += 1

            success = bool(done)
            if success:
                termination_reason = "success"
            elif not termination_reason and t >= episode_step_limit:
                termination_reason = "step_limit"
                _status(
                    f"mode={mode} task={task_id} episode={episode_idx} reached step limit "
                    f"step={t}/{episode_step_limit}; success=False"
                )
            elif not termination_reason:
                termination_reason = "stopped"
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
                "termination_reason": termination_reason,
                "episode_step_limit": episode_step_limit,
                "environment_horizon": (
                    float(environment_horizon) if environment_horizon is not None else float("nan")
                ),
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
                "avg_raw_execution_horizon": _mean(adaptive_raw_execution_horizons),
                "avg_guard_cap": _mean(adaptive_guard_caps),
                "avg_max_entropy_jump": _mean(adaptive_max_entropy_jumps),
                "avg_hysteresis_limited": _mean(adaptive_hysteresis_limited),
                "avg_v2_risk_event_index": _mean(adaptive_v2_risk_event_indices),
                "avg_v2_event_risk": _mean(adaptive_v2_event_risks),
                "avg_v2_budget_balance": _mean(adaptive_v2_budget_balances),
                "v2_budget_limited_rate": _mean(adaptive_v2_budget_limited),
                "v2_intervention_rate": _mean(adaptive_v2_interventions),
                "v2_final_budget_balance": (
                    float(adaptive_h_state.budget_balance) if v2_enabled else float("nan")
                ),
                "v2_cumulative_avg_horizon": (
                    adaptive_h_state.budget_horizon_sum / adaptive_h_state.budget_decisions
                    if adaptive_h_state.budget_decisions
                    else float("nan")
                ),
                "adaptive_replan_reasons": ";".join(adaptive_replan_reasons),
                "adaptive_h_decisions_json": json.dumps(adaptive_h_decisions, separators=(",", ":")),
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
        "termination_reason",
        "episode_step_limit",
        "environment_horizon",
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
        "avg_raw_execution_horizon",
        "avg_guard_cap",
        "avg_max_entropy_jump",
        "avg_hysteresis_limited",
        "avg_v2_risk_event_index",
        "avg_v2_event_risk",
        "avg_v2_budget_balance",
        "v2_budget_limited_rate",
        "v2_intervention_rate",
        "v2_final_budget_balance",
        "v2_cumulative_avg_horizon",
        "adaptive_replan_reasons",
        "adaptive_h_decisions_json",
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

    decisions_path = output_dir / "adaptive_h_decisions.csv"
    decision_fieldnames = [
        "mode",
        "task_suite",
        "task_id",
        "task_name",
        "episode",
        "decision_index",
        "environment_step",
        "selector",
        "entropy_source",
        "raw_execution_horizon",
        "guard_cap",
        "previous_execution_horizon",
        "execution_horizon",
        "entropy_score",
        "entropy_low_threshold",
        "entropy_high_threshold",
        "entropy_threshold_source",
        "max_entropy_jump",
        "entropy_jump_threshold",
        "entropy_jump_significant",
        "stage_guard",
        "stage_guard_reason",
        "hysteresis_limited",
        "low_risk_streak",
        "guard_cooldown_remaining",
        "v2_risk_event_index",
        "v2_event_risk",
        "v2_event_source",
        "v2_budget_balance_before",
        "v2_budget_balance_after",
        "v2_budget_required_credit",
        "v2_budget_limited",
        "v2_intervention",
        "v2_cumulative_avg_horizon",
        "v2_intervention_rate",
        "v2_budget_limited_rate",
        "decision_reason",
        "entropy_curve",
        "v2_final_risk_curve",
        "v2_cot_risk_curve",
        "prefix_entropy",
        "entropy_jumps",
    ]
    decision_rows = []
    for row in rows:
        for decision_index, decision in enumerate(json.loads(row["adaptive_h_decisions_json"])):
            decision_rows.append(
                {
                    "mode": row["mode"],
                    "task_suite": row["task_suite"],
                    "task_id": row["task_id"],
                    "task_name": row["task_name"],
                    "episode": row["episode"],
                    "decision_index": decision_index,
                    **{
                        key: json.dumps(value, separators=(",", ":"))
                        if key
                        in (
                            "entropy_curve",
                            "v2_final_risk_curve",
                            "v2_cot_risk_curve",
                            "prefix_entropy",
                            "entropy_jumps",
                        )
                        else value
                        for key, value in decision.items()
                    },
                }
            )
    with decisions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=decision_fieldnames)
        writer.writeheader()
        writer.writerows(decision_rows)

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
        "avg_successful_deployable_policy_calls",
        "avg_failed_deployable_policy_calls",
        "deployable_policy_ms_per_success",
        "avg_replan_horizon",
        "avg_min_replan_horizon",
        "avg_max_replan_horizon",
        "avg_adaptive_entropy_score",
        "avg_adaptive_entropy_decision",
        "avg_adaptive_low_entropy_target_horizon",
        "avg_adaptive_stage_guard",
        "avg_raw_execution_horizon",
        "avg_guard_cap",
        "avg_max_entropy_jump",
        "avg_hysteresis_limited",
        "avg_v2_risk_event_index",
        "avg_v2_event_risk",
        "avg_v2_budget_balance",
        "avg_v2_budget_limited_rate",
        "avg_v2_intervention_rate",
        "avg_v2_final_budget_balance",
        "avg_v2_cumulative_avg_horizon",
    ]
    per_task_rows = []
    task_keys = sorted(
        {(row["mode"], row["task_id"]) for row in rows},
        key=lambda item: (str(item[0]), int(item[1])),
    )
    for mode, task_id in task_keys:
        subset = [row for row in rows if row["mode"] == mode and row["task_id"] == task_id]
        successful_subset = [row for row in subset if int(row["success"]) == 1]
        failed_subset = [row for row in subset if int(row["success"]) == 0]
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
                "avg_successful_deployable_policy_calls": _mean(
                    [float(row["deployable_policy_calls"]) for row in successful_subset]
                ),
                "avg_failed_deployable_policy_calls": _mean(
                    [float(row["deployable_policy_calls"]) for row in failed_subset]
                ),
                "deployable_policy_ms_per_success": (
                    float(np.nansum([float(row["total_deployable_policy_inference_ms"]) for row in subset]))
                    / len(successful_subset)
                    if successful_subset
                    else float("nan")
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
                "avg_raw_execution_horizon": _mean(
                    [float(row["avg_raw_execution_horizon"]) for row in subset]
                ),
                "avg_guard_cap": _mean([float(row["avg_guard_cap"]) for row in subset]),
                "avg_max_entropy_jump": _mean([float(row["avg_max_entropy_jump"]) for row in subset]),
                "avg_hysteresis_limited": _mean(
                    [float(row["avg_hysteresis_limited"]) for row in subset]
                ),
                "avg_v2_risk_event_index": _mean(
                    [float(row["avg_v2_risk_event_index"]) for row in subset]
                ),
                "avg_v2_event_risk": _mean([float(row["avg_v2_event_risk"]) for row in subset]),
                "avg_v2_budget_balance": _mean(
                    [float(row["avg_v2_budget_balance"]) for row in subset]
                ),
                "avg_v2_budget_limited_rate": _mean(
                    [float(row["v2_budget_limited_rate"]) for row in subset]
                ),
                "avg_v2_intervention_rate": _mean(
                    [float(row["v2_intervention_rate"]) for row in subset]
                ),
                "avg_v2_final_budget_balance": _mean(
                    [float(row["v2_final_budget_balance"]) for row in subset]
                ),
                "avg_v2_cumulative_avg_horizon": _mean(
                    [float(row["v2_cumulative_avg_horizon"]) for row in subset]
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
        successful_subset = [row for row in subset if int(row["success"]) == 1]
        failed_subset = [row for row in subset if int(row["success"]) == 0]
        decisions = [
            decision
            for row in subset
            for decision in json.loads(row["adaptive_h_decisions_json"])
        ]
        horizon_counts = collections.Counter(int(decision["execution_horizon"]) for decision in decisions)
        raw_horizon_counts = collections.Counter(int(decision["raw_execution_horizon"]) for decision in decisions)
        event_source_counts = collections.Counter(
            str(decision["v2_event_source"])
            for decision in decisions
            if str(decision.get("v2_event_source", ""))
        )
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
            "avg_raw_execution_horizon": _mean(
                [float(row["avg_raw_execution_horizon"]) for row in subset]
            ),
            "avg_guard_cap": _mean([float(row["avg_guard_cap"]) for row in subset]),
            "avg_max_entropy_jump": _mean([float(row["avg_max_entropy_jump"]) for row in subset]),
            "avg_hysteresis_limited": _mean(
                [float(row["avg_hysteresis_limited"]) for row in subset]
            ),
            "avg_v2_risk_event_index": _mean(
                [float(row["avg_v2_risk_event_index"]) for row in subset]
            ),
            "avg_v2_event_risk": _mean([float(row["avg_v2_event_risk"]) for row in subset]),
            "avg_v2_budget_balance": _mean(
                [float(row["avg_v2_budget_balance"]) for row in subset]
            ),
            "avg_v2_budget_limited_rate": _mean(
                [float(row["v2_budget_limited_rate"]) for row in subset]
            ),
            "avg_v2_intervention_rate": _mean(
                [float(row["v2_intervention_rate"]) for row in subset]
            ),
            "avg_v2_final_budget_balance": _mean(
                [float(row["v2_final_budget_balance"]) for row in subset]
            ),
            "avg_v2_cumulative_avg_horizon": _mean(
                [float(row["v2_cumulative_avg_horizon"]) for row in subset]
            ),
            "avg_num_replans_per_episode": _mean([float(row["num_replans"]) for row in subset]),
            "avg_total_policy_calls_per_episode": _mean([float(row["total_policy_calls"]) for row in subset]),
            "avg_deployable_policy_calls_per_episode": _mean(
                [float(row["deployable_policy_calls"]) for row in subset]
            ),
            "avg_entropy_oracle_extra_calls_per_episode": _mean(
                [float(row["entropy_oracle_extra_calls"]) for row in subset]
            ),
            "avg_successful_deployable_policy_calls": _mean(
                [float(row["deployable_policy_calls"]) for row in successful_subset]
            ),
            "avg_failed_deployable_policy_calls": _mean(
                [float(row["deployable_policy_calls"]) for row in failed_subset]
            ),
            "deployable_policy_ms_per_success": (
                float(np.nansum([float(row["total_deployable_policy_inference_ms"]) for row in subset]))
                / len(successful_subset)
                if successful_subset
                else float("nan")
            ),
            "execution_horizon_counts": dict(sorted(horizon_counts.items())),
            "raw_execution_horizon_counts": dict(sorted(raw_horizon_counts.items())),
            "v2_event_source_counts": dict(sorted(event_source_counts.items())),
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
            "adaptive_h_selector": args.adaptive_h_selector,
            "adaptive_h_entropy_algorithm": args.adaptive_h_entropy_algorithm,
            "adaptive_h_coarse_stride": args.adaptive_h_coarse_stride,
            "adaptive_h_jump_mad_scale": args.adaptive_h_jump_mad_scale,
            "adaptive_h_entropy_eps": args.adaptive_h_entropy_eps,
            "adaptive_h_cov_shrinkage": args.adaptive_h_cov_shrinkage,
            "adaptive_h_growth_limit": args.adaptive_h_growth_limit,
            "adaptive_h_low_risk_required": args.adaptive_h_low_risk_required,
            "adaptive_h_guard_cooldown": args.adaptive_h_guard_cooldown,
            "adaptive_h_v2_min_horizon": args.adaptive_h_v2_min_horizon,
            "adaptive_h_v2_target_avg_horizon": args.adaptive_h_v2_target_avg_horizon,
            "adaptive_h_v2_initial_budget": args.adaptive_h_v2_initial_budget,
            "adaptive_h_v2_budget_capacity": args.adaptive_h_v2_budget_capacity,
            "adaptive_h_v2_risk_threshold": args.adaptive_h_v2_risk_threshold,
            "adaptive_h_v2_final_weight": args.adaptive_h_v2_final_weight,
            "adaptive_h_v2_cot_weight": args.adaptive_h_v2_cot_weight,
            "adaptive_h_v2_final_entropy_threshold": args.adaptive_h_v2_final_entropy_threshold,
            "adaptive_h_v2_cot_entropy_threshold": args.adaptive_h_v2_cot_entropy_threshold,
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
                "timing and are the intended fields for speed-success comparison. budgeted_event_v2 defaults to "
                "H=10, shortens H at the earliest fused per-timestep risk event, and constrains short-H decisions "
                "with an episode horizon-credit budget targeting the configured average H. Its current online-MC "
                "implementation remains an oracle; actual observed timing includes all entropy samples."
            ),
        },
        "aggregate": by_mode,
        "outputs": {
            "rollout_rows_csv": str(rows_path),
            "per_task_summary_csv": str(per_task_path),
            "adaptive_h_decisions_csv": str(decisions_path),
        },
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
