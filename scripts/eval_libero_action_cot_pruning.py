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
        "--coarse_num_steps",
        "--coarse-num-steps",
        type=int,
        default=None,
        help="Denoising steps for explicit coarse Action-CoT only. Defaults to the policy server sample default.",
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
    if args.num_trials_per_task <= 0:
        raise ValueError("--num_trials_per_task must be positive.")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("--max_tasks must be positive when set.")
    if args.entropy_samples <= 0:
        raise ValueError("--entropy_samples must be positive.")
    if args.coarse_num_steps is not None and args.coarse_num_steps <= 0:
        raise ValueError("--coarse_num_steps must be positive when set.")
    if not 0.0 <= args.prune_ratio <= 1.0:
        raise ValueError("--prune_ratio must be in [0, 1].")
    if args.true_skip_chunk_size != 5:
        raise ValueError("The current true segment-skip model path supports --true_skip_chunk_size 5 only.")
    if not 0 <= args.true_skip_segment <= 2:
        raise ValueError("--true_skip_segment must be 0, 1, or 2.")


def _status(message: str) -> None:
    print(f"[eval_libero_action_cot_pruning] {message}", flush=True)


def _with_coarse_num_steps(element: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    request = dict(element)
    if args.coarse_num_steps is not None:
        request["action_cot_coarse_num_steps"] = np.asarray(args.coarse_num_steps, dtype=np.int32)
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


def _timing_ms(result: dict[str, Any], wall_ms: float) -> tuple[float, float]:
    policy_timing = result.get("policy_timing", {}) if isinstance(result, dict) else {}
    server_timing = result.get("server_timing", {}) if isinstance(result, dict) else {}
    policy_ms = float(policy_timing.get("infer_ms", np.nan))
    server_ms = float(server_timing.get("infer_ms", wall_ms))
    return policy_ms, server_ms


def _infer(client, element: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], float, float, float]:
    request = dict(element)
    request["policy_seed"] = np.asarray(seed, dtype=np.int64)
    start = time.perf_counter()
    result = client.infer(request)
    wall_ms = (time.perf_counter() - start) * 1000.0
    policy_ms, server_ms = _timing_ms(result, wall_ms)
    return result, wall_ms, policy_ms, server_ms


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
        result, wall_ms, policy_ms, server_ms = _infer(client, _with_coarse_num_steps(element, args), seed=seed)
        return np.asarray(result["actions"]), {
            "wall_ms": wall_ms,
            "policy_ms": policy_ms,
            "server_ms": server_ms,
            "deployable_wall_ms": wall_ms,
            "deployable_policy_ms": policy_ms,
            "deployable_server_ms": server_ms,
            "skip_ratio": float("nan"),
            "num_segments": 0,
            "skipped_segments": "",
            "full_calls": 1,
            "override_calls": 0,
            "true_skip_calls": 0,
        }

    if mode == "true_segment_skip":
        true_element = _with_coarse_num_steps(element, args)
        true_element["action_cot_skip_segment"] = np.asarray(args.true_skip_segment, dtype=np.int32)
        result, wall_ms, policy_ms, server_ms = _infer(client, true_element, seed=seed)
        return np.asarray(result["actions"]), {
            "wall_ms": wall_ms,
            "policy_ms": policy_ms,
            "server_ms": server_ms,
            "deployable_wall_ms": wall_ms,
            "deployable_policy_ms": policy_ms,
            "deployable_server_ms": server_ms,
            "skip_ratio": float(args.true_skip_chunk_size / 15),
            "num_segments": 3,
            "skipped_segments": str(args.true_skip_segment),
            "full_calls": 0,
            "override_calls": 0,
            "true_skip_calls": 1,
        }

    entropy_samples = 1 if mode == "cached_override" else args.entropy_samples
    coarse_samples = []
    full_wall_ms = []
    full_policy_ms = []
    full_server_ms = []
    full_element = _with_coarse_num_steps(element, args)
    for sample_idx in range(entropy_samples):
        result, wall_ms, policy_ms, server_ms = _infer(client, full_element, seed=seed + sample_idx)
        if "coarse_actions" not in result:
            raise KeyError("Policy output does not contain coarse_actions.")
        coarse_samples.append(np.asarray(result["coarse_actions"], dtype=np.float32))
        full_wall_ms.append(wall_ms)
        full_policy_ms.append(policy_ms)
        full_server_ms.append(server_ms)

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
        true_element = _with_coarse_num_steps(element, args)
        true_element["action_cot_skip_segment"] = np.asarray(skip_segment, dtype=np.int32)
        result, skip_wall_ms, skip_policy_ms, skip_server_ms = _infer(client, true_element, seed=seed)
        return np.asarray(result["actions"]), {
            "wall_ms": float(np.nansum(full_wall_ms) + skip_wall_ms),
            "policy_ms": float(np.nansum(full_policy_ms) + skip_policy_ms),
            "server_ms": float(np.nansum(full_server_ms) + skip_server_ms),
            "deployable_wall_ms": skip_wall_ms,
            "deployable_policy_ms": skip_policy_ms,
            "deployable_server_ms": skip_server_ms,
            "skip_ratio": skip_ratio,
            "num_segments": 3,
            "skipped_segments": str(skip_segment),
            "skip_segment_entropy": float(entropy[skip_segment]),
            "full_calls": entropy_samples,
            "override_calls": 0,
            "true_skip_calls": 1,
        }
    else:
        coarse_override, skip_ratio, skipped_segments, num_segments = _prune_online_coarse(
            np.stack(coarse_samples, axis=0),
            args=args,
            norm_stats=norm_stats,
            rng=rng,
        )

    override_element = _with_coarse_num_steps(element, args)
    override_element["coarse_actions_override"] = coarse_override
    result, override_wall_ms, override_policy_ms, override_server_ms = _infer(client, override_element, seed=seed)
    return np.asarray(result["actions"]), {
        "wall_ms": float(np.nansum(full_wall_ms) + override_wall_ms),
        "policy_ms": float(np.nansum(full_policy_ms) + override_policy_ms),
        "server_ms": float(np.nansum(full_server_ms) + override_server_ms),
        "deployable_wall_ms": override_wall_ms,
        "deployable_policy_ms": override_policy_ms,
        "deployable_server_ms": override_server_ms,
        "override_wall_ms": override_wall_ms,
        "override_policy_ms": override_policy_ms,
        "override_server_ms": override_server_ms,
        "skip_ratio": skip_ratio,
        "num_segments": num_segments,
        "skipped_segments": ";".join(str(idx) for idx in skipped_segments),
        "full_calls": entropy_samples,
        "override_calls": 1,
        "true_skip_calls": 0,
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
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for episode_idx in range(args.num_trials_per_task):
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
            skip_ratios = []
            full_calls = 0
            override_calls = 0
            true_skip_calls = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    total_return += float(reward)
                    t += 1
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
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"Need at least {args.replan_steps} actions, got {len(action_chunk)} for mode={mode}."
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])
                    infer_wall_ms.append(float(timing["wall_ms"]))
                    infer_policy_ms.append(float(timing["policy_ms"]))
                    infer_server_ms.append(float(timing["server_ms"]))
                    deployable_wall_ms.append(float(timing.get("deployable_wall_ms", timing["wall_ms"])))
                    deployable_policy_ms.append(float(timing.get("deployable_policy_ms", timing["policy_ms"])))
                    deployable_server_ms.append(float(timing.get("deployable_server_ms", timing["server_ms"])))
                    if np.isfinite(float(timing["skip_ratio"])):
                        skip_ratios.append(float(timing["skip_ratio"]))
                    full_calls += int(timing["full_calls"])
                    override_calls += int(timing["override_calls"])
                    true_skip_calls += int(timing.get("true_skip_calls", 0))

                action = action_plan.popleft()
                obs, reward, done, _ = env.step(np.asarray(action).tolist())
                total_return += float(reward)
                if done:
                    break
                t += 1

            success = bool(done)
            timeout = not success
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
                "avg_deployable_wall_inference_ms": _mean(deployable_wall_ms),
                "avg_deployable_policy_inference_ms": _mean(deployable_policy_ms),
                "avg_deployable_server_inference_ms": _mean(deployable_server_ms),
                "avg_skip_ratio": _mean(skip_ratios),
                "num_replans": len(infer_wall_ms),
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
                f"success={success} avg_wall_ms={row['avg_wall_inference_ms']:.2f}"
            )

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
        "avg_deployable_wall_inference_ms",
        "avg_deployable_policy_inference_ms",
        "avg_deployable_server_inference_ms",
        "avg_skip_ratio",
        "num_replans",
        "full_calls",
        "override_calls",
        "true_skip_calls",
    ]
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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
            "avg_deployable_wall_inference_ms": _mean(
                [float(row["avg_deployable_wall_inference_ms"]) for row in subset]
            ),
            "avg_deployable_policy_inference_ms": _mean(
                [float(row["avg_deployable_policy_inference_ms"]) for row in subset]
            ),
            "avg_deployable_server_inference_ms": _mean(
                [float(row["avg_deployable_server_inference_ms"]) for row in subset]
            ),
            "avg_skip_ratio": _mean([float(row["avg_skip_ratio"]) for row in subset]),
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
            "coarse_num_steps": args.coarse_num_steps,
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
                "its deployable timing fields measure only the final true-skip model call."
            ),
        },
        "aggregate": by_mode,
        "outputs": {"rollout_rows_csv": str(rows_path)},
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
