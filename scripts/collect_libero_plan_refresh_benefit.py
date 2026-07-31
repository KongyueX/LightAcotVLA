"""Collect same-snapshot labels for deciding whether to refresh a stale plan.

The collector follows the ordinary stale-H10 policy trajectory.  At selected
macro roots it executes the common H4 prefix, captures one canonical simulator
snapshot, and compares two paired branches from that exact age-four state:

* ``stale`` executes cached actions 4:10;
* ``fresh`` makes an exact full-policy request and executes fresh actions 0:6.

Both H6 endpoints then receive an independently computed fresh H9 continuation
with the same policy seed.  Generic privileged LIBERO goal progress is recorded
after H6 and after the continuation.  The policy trajectory itself always
continues through the stale H6 endpoint.

Every completed root is written atomically as a compressed NPZ.  The NPZ files
are the resume authority; ``roots.jsonl`` is a compact searchable index.  This
script is intentionally separate from the existing evaluation entrypoints.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import time
from typing import Any

import collect_action_cot_state_branches as branch_collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.action_cot import branched_dataset
from openpi.execution_horizon import privileged_progress


SCHEMA_VERSION = 1
DEFAULT_TASK_ID = 8
DEFAULT_EPISODE_IDS = (*range(0, 10), *range(30, 50))
DEFAULT_ROOT_INDICES = tuple(range(6, 15))
IMAGE_SIZE = 64
ACTION_DIM = 7
ANCHOR_HORIZON = 4
BRANCH_HORIZON = 6
CONTINUATION_HORIZON = 9
TEMPORAL_FEATURE_DIM = 256
DATASET_SHAPE = branched_dataset.BranchedDatasetShape(
    image_height=IMAGE_SIZE,
    image_width=IMAGE_SIZE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPISODE_IDS),
        help="Explicit LIBERO initial-state IDs; defaults to 0-9 and 30-49.",
    )
    parser.add_argument(
        "--root-indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_ROOT_INDICES),
        help="Stale-H10 macro roots to label while all intervening roots still advance.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument(
        "--continuation-seed-offset",
        type=int,
        default=100_000_000,
        help="Offset from the canonical root seed shared by both fresh-H9 arms.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.task_id < 0:
        raise ValueError("--task-id must be non-negative.")
    if not args.episode_ids or any(value < 0 for value in args.episode_ids):
        raise ValueError("--episode-ids must contain non-negative IDs.")
    if len(set(args.episode_ids)) != len(args.episode_ids):
        raise ValueError("--episode-ids must not contain duplicates.")
    if not args.root_indices or any(value < 0 for value in args.root_indices):
        raise ValueError("--root-indices must contain non-negative indices.")
    if len(set(args.root_indices)) != len(args.root_indices):
        raise ValueError("--root-indices must not contain duplicates.")
    if args.seed < 0 or args.num_steps_wait < 0:
        raise ValueError("--seed and --num-steps-wait must be non-negative.")
    if args.resize_size <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("--resize-size and --action-cot-denoising-steps must be positive.")
    if args.continuation_seed_offset <= 0:
        raise ValueError("--continuation-seed-offset must be positive.")


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logging.warning("Ignoring malformed JSONL line %s in %s.", line_number, path)
    return rows


def _append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def _root_path(root_dir: pathlib.Path, task_id: int, episode_id: int, root_index: int) -> pathlib.Path:
    return root_dir / f"task{task_id:02d}_episode{episode_id:03d}_root{root_index:03d}.npz"


def _policy_input(
    observation: dict[str, Any],
    task_description: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return libero_eval._observation_to_policy_input(
        observation,
        task_description,
        args.resize_size,
    )


def _anchor_request(
    client: websocket_policy.WebsocketClientPolicy,
    element: dict[str, Any],
    *,
    policy_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Request a full anchor and its deployable V2-P temporal feature."""

    request = {
        **element,
        "policy_seed": np.asarray(policy_seed, dtype=np.int64),
        "profile_policy_timing": np.zeros((), dtype=np.bool_),
        "export_acot_cache": np.ones((), dtype=np.bool_),
        "action_cot_denoising_steps": np.asarray(
            args.action_cot_denoising_steps,
            dtype=np.int32,
        ),
        "run_execution_horizon_predictor": np.ones((), dtype=np.bool_),
        "execution_horizon_previous_actions": np.zeros((10, ACTION_DIM), dtype=np.float32),
        "execution_horizon_previous_h": np.zeros((), dtype=np.int32),
        "execution_horizon_budget_balance": np.zeros((), dtype=np.float32),
        "execution_horizon_episode_progress": np.zeros((), dtype=np.float32),
        "execution_horizon_previous_valid": np.zeros((), dtype=np.bool_),
    }
    started = time.perf_counter()
    result = client.infer(request)
    result["collector_wall_ms"] = (time.perf_counter() - started) * 1000.0
    required = (
        "actions",
        "execution_horizon_state_normalized",
        "execution_horizon_coarse_actions_normalized",
        "execution_horizon_final_actions_normalized",
        "execution_horizon_temporal_feature",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError(f"Anchor policy response is missing fields: {missing}.")
    temporal = np.asarray(result["execution_horizon_temporal_feature"], dtype=np.float32).reshape(-1)
    if temporal.shape != (TEMPORAL_FEATURE_DIM,):
        raise ValueError(
            "Expected execution_horizon_temporal_feature "
            f"[{TEMPORAL_FEATURE_DIM}], got {temporal.shape}."
        )
    return result


def _fresh_request(
    client: websocket_policy.WebsocketClientPolicy,
    observation: dict[str, Any],
    task_description: str,
    *,
    policy_seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    element = _policy_input(observation, task_description, args)
    result = branch_collector._teacher_request(
        client,
        element,
        policy_seed=policy_seed,
        denoising_steps=args.action_cot_denoising_steps,
    )
    return element, result


def _teacher_tensors(result: dict[str, Any]) -> dict[str, np.ndarray]:
    return branch_collector._teacher_tensors(result, DATASET_SHAPE)


def _progress(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    score = privileged_progress.score_libero_goal_progress(env, observation)
    return {
        "score": float(score.score),
        "normalized_score": float(score.normalized_score),
        "satisfied_count": int(score.satisfied_count),
        "total_goals": int(score.total_goals),
        "active_kind": str(score.active_kind),
        "active_progress": float(score.active_progress),
        "success": bool(libero_eval._env_success(env)),
    }


def _empty_actions(horizon: int) -> np.ndarray:
    return np.zeros((horizon, ACTION_DIM), dtype=np.float32)


def _padded_executed(executed: list[np.ndarray], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    actions = _empty_actions(horizon)
    valid = np.zeros((horizon,), dtype=np.bool_)
    if executed:
        count = min(len(executed), horizon)
        actions[:count] = np.asarray(executed[:count], dtype=np.float32)[:, :ACTION_DIM]
        valid[:count] = True
    return actions, valid


def _run_branch(
    env: Any,
    midpoint: branch_collector.CanonicalSimulatorSnapshot,
    h6_actions: np.ndarray,
    *,
    continuation_seed: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> dict[str, Any]:
    observation = branch_collector._restore_canonical_snapshot(env, midpoint)
    before = _progress(env, observation)
    observation, done, executed_h6 = branch_collector._step_actions(
        env,
        observation,
        np.asarray(h6_actions, dtype=np.float32)[:BRANCH_HORIZON, :ACTION_DIM],
    )
    after_h6 = _progress(env, observation)

    continuation_actions = _empty_actions(CONTINUATION_HORIZON)
    continuation_teacher_wall_ms = 0.0
    continuation_requested = False
    executed_h9: list[np.ndarray] = []
    if not done and not after_h6["success"]:
        _, continuation_result = _fresh_request(
            client,
            observation,
            task_description,
            policy_seed=continuation_seed,
            args=args,
        )
        continuation_tensors = _teacher_tensors(continuation_result)
        continuation_actions = np.asarray(
            continuation_tensors["actions_env"][:CONTINUATION_HORIZON, :ACTION_DIM],
            dtype=np.float32,
        )
        continuation_requested = True
        continuation_teacher_wall_ms = float(continuation_result["collector_wall_ms"])
        observation, done, executed_h9 = branch_collector._step_actions(
            env,
            observation,
            continuation_actions,
        )
    after_handoff = _progress(env, observation)
    executed_h6_array, executed_h6_valid = _padded_executed(executed_h6, BRANCH_HORIZON)
    executed_h9_array, executed_h9_valid = _padded_executed(
        executed_h9,
        CONTINUATION_HORIZON,
    )
    return {
        "before": before,
        "after_h6": after_h6,
        "after_handoff": after_handoff,
        "h6_actions": np.asarray(h6_actions, dtype=np.float32)[:BRANCH_HORIZON, :ACTION_DIM],
        "h6_executed_actions": executed_h6_array,
        "h6_executed_valid": executed_h6_valid,
        "h6_executed_steps": int(np.sum(executed_h6_valid)),
        "h9_actions": continuation_actions,
        "h9_executed_actions": executed_h9_array,
        "h9_executed_valid": executed_h9_valid,
        "h9_executed_steps": int(np.sum(executed_h9_valid)),
        "continuation_requested": continuation_requested,
        "continuation_teacher_wall_ms": continuation_teacher_wall_ms,
        "done": bool(done),
    }


def _progress_arrays(prefix: str, progress: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_score": np.asarray(progress["score"], dtype=np.float32),
        f"{prefix}_normalized_score": np.asarray(
            progress["normalized_score"],
            dtype=np.float32,
        ),
        f"{prefix}_satisfied_count": np.asarray(
            progress["satisfied_count"],
            dtype=np.int16,
        ),
        f"{prefix}_total_goals": np.asarray(progress["total_goals"], dtype=np.int16),
        f"{prefix}_active_progress": np.asarray(
            progress["active_progress"],
            dtype=np.float32,
        ),
        f"{prefix}_success": np.asarray(progress["success"], dtype=np.bool_),
    }


def _advantage_arrays(
    endpoint: str,
    stale: dict[str, Any],
    fresh: dict[str, Any],
) -> dict[str, np.ndarray]:
    normalized = float(fresh["normalized_score"] - stale["normalized_score"])
    satisfied = int(fresh["satisfied_count"] - stale["satisfied_count"])
    success = int(bool(fresh["success"])) - int(bool(stale["success"]))
    values = {
        f"fresh_minus_stale_{endpoint}_normalized_score": np.asarray(
            normalized,
            dtype=np.float32,
        ),
        f"fresh_minus_stale_{endpoint}_satisfied_count": np.asarray(
            satisfied,
            dtype=np.int16,
        ),
        f"fresh_minus_stale_{endpoint}_success": np.asarray(success, dtype=np.int8),
    }
    # Stable short aliases used by the first training-head implementation.
    values.update(
        {
            f"advantage_{endpoint}_normalized_score": np.asarray(normalized, dtype=np.float32),
            f"advantage_{endpoint}_satisfied_count": np.asarray(satisfied, dtype=np.int16),
            f"advantage_{endpoint}_success": np.asarray(success, dtype=np.int8),
        }
    )
    return values


def _root_arrays(
    *,
    args: argparse.Namespace,
    episode_id: int,
    initial_state_id: int,
    root_index: int,
    decision_step: int,
    root_seed: int,
    continuation_seed: int,
    anchor_element: dict[str, Any],
    age4_element: dict[str, Any],
    anchor_result: dict[str, Any],
    fresh_result: dict[str, Any],
    stale: dict[str, Any],
    fresh: dict[str, Any],
) -> dict[str, np.ndarray]:
    anchor = _teacher_tensors(anchor_result)
    refreshed = _teacher_tensors(fresh_result)
    temporal = np.asarray(
        anchor_result["execution_horizon_temporal_feature"],
        dtype=np.float32,
    ).reshape(TEMPORAL_FEATURE_DIM)
    before = stale["before"]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "valid": np.ones((), dtype=np.bool_),
        "task_id": np.asarray(args.task_id, dtype=np.int16),
        "episode_id": np.asarray(episode_id, dtype=np.int32),
        "initial_state_id": np.asarray(initial_state_id, dtype=np.int32),
        "root_index": np.asarray(root_index, dtype=np.int32),
        "anchor_decision_step": np.asarray(decision_step, dtype=np.int32),
        "age4_decision_step": np.asarray(decision_step + ANCHOR_HORIZON, dtype=np.int32),
        "anchor_control_step": np.asarray(
            decision_step - args.num_steps_wait,
            dtype=np.int32,
        ),
        "age4_control_step": np.asarray(
            decision_step - args.num_steps_wait + ANCHOR_HORIZON,
            dtype=np.int32,
        ),
        "policy_seed": np.asarray(root_seed, dtype=np.uint32),
        "continuation_seed": np.asarray(continuation_seed, dtype=np.uint32),
        "temporal_feature": temporal,
        "anchor_images": branch_collector._images(anchor_element, IMAGE_SIZE),
        "current_images": branch_collector._images(age4_element, IMAGE_SIZE),
        "anchor_state": np.asarray(anchor["state"], dtype=np.float32),
        "current_state": np.asarray(refreshed["state"], dtype=np.float32),
        "cached_coarse_actions": np.asarray(anchor["ear"], dtype=np.float32),
        "cached_final_actions": np.asarray(anchor["actions"], dtype=np.float32),
        "cached_env_actions": np.asarray(anchor["actions_env"], dtype=np.float32),
        "fresh_coarse_actions": np.asarray(refreshed["ear"], dtype=np.float32),
        "fresh_final_actions": np.asarray(refreshed["actions"], dtype=np.float32),
        "fresh_env_actions": np.asarray(refreshed["actions_env"], dtype=np.float32),
        "intended_prefix_env": np.asarray(
            anchor["actions_env"][:ANCHOR_HORIZON, :ACTION_DIM],
            dtype=np.float32,
        ),
        "stale_h6_actions": stale["h6_actions"],
        "fresh_h6_actions": fresh["h6_actions"],
        "stale_h6_executed_actions": stale["h6_executed_actions"],
        "fresh_h6_executed_actions": fresh["h6_executed_actions"],
        "stale_h6_executed_valid": stale["h6_executed_valid"],
        "fresh_h6_executed_valid": fresh["h6_executed_valid"],
        "stale_h9_actions": stale["h9_actions"],
        "fresh_h9_actions": fresh["h9_actions"],
        "stale_h9_executed_actions": stale["h9_executed_actions"],
        "fresh_h9_executed_actions": fresh["h9_executed_actions"],
        "stale_h9_executed_valid": stale["h9_executed_valid"],
        "fresh_h9_executed_valid": fresh["h9_executed_valid"],
        "stale_h6_executed_steps": np.asarray(stale["h6_executed_steps"], dtype=np.int16),
        "fresh_h6_executed_steps": np.asarray(fresh["h6_executed_steps"], dtype=np.int16),
        "stale_h9_executed_steps": np.asarray(stale["h9_executed_steps"], dtype=np.int16),
        "fresh_h9_executed_steps": np.asarray(fresh["h9_executed_steps"], dtype=np.int16),
        "stale_continuation_requested": np.asarray(
            stale["continuation_requested"],
            dtype=np.bool_,
        ),
        "fresh_continuation_requested": np.asarray(
            fresh["continuation_requested"],
            dtype=np.bool_,
        ),
        "anchor_teacher_wall_ms": np.asarray(
            anchor_result["collector_wall_ms"],
            dtype=np.float32,
        ),
        "fresh_teacher_wall_ms": np.asarray(
            fresh_result["collector_wall_ms"],
            dtype=np.float32,
        ),
        "stale_continuation_teacher_wall_ms": np.asarray(
            stale["continuation_teacher_wall_ms"],
            dtype=np.float32,
        ),
        "fresh_continuation_teacher_wall_ms": np.asarray(
            fresh["continuation_teacher_wall_ms"],
            dtype=np.float32,
        ),
    }
    arrays.update(_progress_arrays("before", before))
    arrays.update(_progress_arrays("stale_h6", stale["after_h6"]))
    arrays.update(_progress_arrays("fresh_h6", fresh["after_h6"]))
    arrays.update(_progress_arrays("stale_handoff", stale["after_handoff"]))
    arrays.update(_progress_arrays("fresh_handoff", fresh["after_handoff"]))
    arrays.update(
        _advantage_arrays(
            "h6",
            stale["after_h6"],
            fresh["after_h6"],
        )
    )
    arrays.update(
        _advantage_arrays(
            "handoff",
            stale["after_handoff"],
            fresh["after_handoff"],
        )
    )
    return arrays


def _write_npz_atomic(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _compact_progress(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalized_score": float(progress["normalized_score"]),
        "satisfied_count": int(progress["satisfied_count"]),
        "success": bool(progress["success"]),
    }


def _root_index_row(path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        def scalar(name: str) -> Any:
            return np.asarray(arrays[name]).item()

        return {
            "schema_version": int(scalar("schema_version")),
            "valid": bool(scalar("valid")),
            "task_id": int(scalar("task_id")),
            "episode_id": int(scalar("episode_id")),
            "initial_state_id": int(scalar("initial_state_id")),
            "root_index": int(scalar("root_index")),
            "anchor_decision_step": int(scalar("anchor_decision_step")),
            "age4_decision_step": int(scalar("age4_decision_step")),
            "anchor_control_step": int(scalar("anchor_control_step")),
            "age4_control_step": int(scalar("age4_control_step")),
            "policy_seed": int(scalar("policy_seed")),
            "continuation_seed": int(scalar("continuation_seed")),
            "npz_file": str(path.relative_to(output_dir)),
            "before": {
                "normalized_score": float(scalar("before_normalized_score")),
                "satisfied_count": int(scalar("before_satisfied_count")),
                "success": bool(scalar("before_success")),
            },
            "stale_h6": {
                "normalized_score": float(scalar("stale_h6_normalized_score")),
                "satisfied_count": int(scalar("stale_h6_satisfied_count")),
                "success": bool(scalar("stale_h6_success")),
            },
            "fresh_h6": {
                "normalized_score": float(scalar("fresh_h6_normalized_score")),
                "satisfied_count": int(scalar("fresh_h6_satisfied_count")),
                "success": bool(scalar("fresh_h6_success")),
            },
            "stale_handoff": {
                "normalized_score": float(scalar("stale_handoff_normalized_score")),
                "satisfied_count": int(scalar("stale_handoff_satisfied_count")),
                "success": bool(scalar("stale_handoff_success")),
            },
            "fresh_handoff": {
                "normalized_score": float(scalar("fresh_handoff_normalized_score")),
                "satisfied_count": int(scalar("fresh_handoff_satisfied_count")),
                "success": bool(scalar("fresh_handoff_success")),
            },
            "fresh_minus_stale_h6_normalized_score": float(
                scalar("fresh_minus_stale_h6_normalized_score")
            ),
            "fresh_minus_stale_h6_satisfied_count": int(
                scalar("fresh_minus_stale_h6_satisfied_count")
            ),
            "fresh_minus_stale_h6_success": int(
                scalar("fresh_minus_stale_h6_success")
            ),
            "fresh_minus_stale_handoff_normalized_score": float(
                scalar("fresh_minus_stale_handoff_normalized_score")
            ),
            "fresh_minus_stale_handoff_satisfied_count": int(
                scalar("fresh_minus_stale_handoff_satisfied_count")
            ),
            "fresh_minus_stale_handoff_success": int(
                scalar("fresh_minus_stale_handoff_success")
            ),
            "timing_ms": {
                "anchor_teacher": float(scalar("anchor_teacher_wall_ms")),
                "fresh_teacher": float(scalar("fresh_teacher_wall_ms")),
                "stale_continuation_teacher": float(
                    scalar("stale_continuation_teacher_wall_ms")
                ),
                "fresh_continuation_teacher": float(
                    scalar("fresh_continuation_teacher_wall_ms")
                ),
            },
        }


def _collect_root(
    env: Any,
    observation: dict[str, Any],
    *,
    task_description: str,
    episode_id: int,
    initial_state_id: int,
    root_index: int,
    decision_step: int,
    root_seed: int,
    anchor_element: dict[str, Any],
    anchor_result: dict[str, Any],
    anchor_tensors: dict[str, np.ndarray],
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    output_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], bool, int]:
    observation, done, executed_prefix = branch_collector._step_actions(
        env,
        observation,
        anchor_tensors["actions_env"][:ANCHOR_HORIZON, :ACTION_DIM],
    )
    actual_steps = len(executed_prefix)
    if done or libero_eval._env_success(env) or actual_steps < ANCHOR_HORIZON:
        return {}, observation, True, actual_steps

    midpoint = branch_collector._capture_canonical_snapshot(env)
    stale_h6_actions = np.asarray(
        anchor_tensors["actions_env"][ANCHOR_HORIZON : ANCHOR_HORIZON + BRANCH_HORIZON, :ACTION_DIM],
        dtype=np.float32,
    )
    continuation_seed = root_seed + args.continuation_seed_offset
    if continuation_seed > np.iinfo(np.uint32).max:
        raise ValueError(f"Continuation seed {continuation_seed} exceeds uint32.")

    try:
        age4_element, fresh_result = _fresh_request(
            client,
            observation,
            task_description,
            policy_seed=root_seed,
            args=args,
        )
        fresh_tensors = _teacher_tensors(fresh_result)
        fresh_h6_actions = np.asarray(
            fresh_tensors["actions_env"][:BRANCH_HORIZON, :ACTION_DIM],
            dtype=np.float32,
        )
        stale = _run_branch(
            env,
            midpoint,
            stale_h6_actions,
            continuation_seed=continuation_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        fresh = _run_branch(
            env,
            midpoint,
            fresh_h6_actions,
            continuation_seed=continuation_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        arrays = _root_arrays(
            args=args,
            episode_id=episode_id,
            initial_state_id=initial_state_id,
            root_index=root_index,
            decision_step=decision_step,
            root_seed=root_seed,
            continuation_seed=continuation_seed,
            anchor_element=anchor_element,
            age4_element=age4_element,
            anchor_result=anchor_result,
            fresh_result=fresh_result,
            stale=stale,
            fresh=fresh,
        )
        _write_npz_atomic(output_path, arrays)
        row = _root_index_row(output_path, output_path.parent.parent)
    finally:
        observation = branch_collector._restore_canonical_snapshot(env, midpoint)

    observation, done, executed_stale = branch_collector._step_actions(
        env,
        observation,
        stale_h6_actions,
    )
    return (
        row,
        observation,
        bool(done or libero_eval._env_success(env)),
        actual_steps + len(executed_stale),
    )


def _run_episode(
    *,
    task: Any,
    initial_state: Any,
    initial_state_id: int,
    episode_id: int,
    root_indices: set[int],
    indexed_root_keys: set[tuple[int, int, int]],
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    output_dir: pathlib.Path,
    root_dir: pathlib.Path,
    roots_file: Any,
) -> dict[str, Any]:
    env, task_description = libero_eval._get_libero_env(
        task,
        libero_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    started = time.monotonic()
    collected = 0
    resumed = 0
    repaired_index_rows = 0
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        step = 0
        done = False
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                break

        environment_horizon = libero_eval._env_horizon(env)
        step_limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        if environment_horizon is not None:
            step_limit = min(step_limit, environment_horizon)

        root_index = 0
        termination_reason = ""
        maximum_root = max(root_indices)
        while not done and step < step_limit and root_index <= maximum_root:
            decision_step = step
            root_seed = branch_collector.canonical_policy_seed(
                args.seed,
                args.task_id,
                episode_id,
                decision_step,
            )
            anchor_element = _policy_input(observation, task_description, args)
            anchor_result = _anchor_request(
                client,
                anchor_element,
                policy_seed=root_seed,
                args=args,
            )
            anchor_tensors = _teacher_tensors(anchor_result)
            requested = root_index in root_indices
            output_path = _root_path(root_dir, args.task_id, episode_id, root_index)

            if requested and output_path.exists():
                key = (args.task_id, episode_id, root_index)
                if key not in indexed_root_keys:
                    row = _root_index_row(output_path, output_dir)
                    _append_jsonl(roots_file, row)
                    indexed_root_keys.add(key)
                    repaired_index_rows += 1
                observation, done, executed = branch_collector._step_actions(
                    env,
                    observation,
                    anchor_tensors["actions_env"][: ANCHOR_HORIZON + BRANCH_HORIZON, :ACTION_DIM],
                )
                actual_steps = len(executed)
                resumed += 1
            elif requested:
                row, observation, done, actual_steps = _collect_root(
                    env,
                    observation,
                    task_description=task_description,
                    episode_id=episode_id,
                    initial_state_id=initial_state_id,
                    root_index=root_index,
                    decision_step=decision_step,
                    root_seed=root_seed,
                    anchor_element=anchor_element,
                    anchor_result=anchor_result,
                    anchor_tensors=anchor_tensors,
                    args=args,
                    client=client,
                    output_path=output_path,
                )
                if row:
                    _append_jsonl(roots_file, row)
                    indexed_root_keys.add((args.task_id, episode_id, root_index))
                    collected += 1
                    print(
                        json.dumps(
                            {
                                "task_id": args.task_id,
                                "episode_id": episode_id,
                                "root_index": root_index,
                                "age4_control_step": row["age4_control_step"],
                                "fresh_minus_stale_h6_normalized_score": row[
                                    "fresh_minus_stale_h6_normalized_score"
                                ],
                                "fresh_minus_stale_handoff_normalized_score": row[
                                    "fresh_minus_stale_handoff_normalized_score"
                                ],
                                "npz_file": row["npz_file"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            else:
                observation, done, executed = branch_collector._step_actions(
                    env,
                    observation,
                    anchor_tensors["actions_env"][: ANCHOR_HORIZON + BRANCH_HORIZON, :ACTION_DIM],
                )
                actual_steps = len(executed)

            step += actual_steps
            root_index += 1
            if actual_steps == 0:
                termination_reason = "no_environment_progress"
                break

        success = bool(done or libero_eval._env_success(env))
        if success:
            termination_reason = "success"
        elif not termination_reason and root_index > maximum_root:
            termination_reason = "requested_roots_complete"
        elif not termination_reason:
            termination_reason = "step_limit"
        return {
            "schema_version": SCHEMA_VERSION,
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "task_description": str(task_description),
            "episode_id": episode_id,
            "initial_state_id": initial_state_id,
            "success": success,
            "termination_reason": termination_reason,
            "environment_steps": step,
            "control_steps": max(0, step - args.num_steps_wait),
            "roots_visited": root_index,
            "new_root_files": collected,
            "resumed_root_files": resumed,
            "repaired_root_index_rows": repaired_index_rows,
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        libero_eval._safe_close_env(env)


def _summary(
    args: argparse.Namespace,
    root_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_roots = [
        row
        for row in root_rows
        if int(row.get("task_id", -1)) == args.task_id
        and int(row.get("episode_id", -1)) in set(args.episode_ids)
        and int(row.get("root_index", -1)) in set(args.root_indices)
    ]
    selected_episodes = {
        int(row["episode_id"]): row
        for row in episode_rows
        if int(row.get("task_id", -1)) == args.task_id
        and int(row.get("episode_id", -1)) in set(args.episode_ids)
    }

    def mean(name: str) -> float | None:
        values = [float(row[name]) for row in selected_roots]
        return float(np.mean(values)) if values else None

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if len(selected_episodes) == len(args.episode_ids) else "running",
        "protocol": {
            "task_suite_name": args.task_suite_name,
            "task_id": args.task_id,
            "episode_ids": list(args.episode_ids),
            "root_indices": sorted(args.root_indices),
            "trajectory": "stale_h10",
            "anchor_horizon": ANCHOR_HORIZON,
            "branch_horizon": BRANCH_HORIZON,
            "continuation_horizon": CONTINUATION_HORIZON,
            "same_seed_fresh_h9_continuations": True,
            "continuation_seed_offset": args.continuation_seed_offset,
            "seed": args.seed,
            "action_cot_denoising_steps": args.action_cot_denoising_steps,
            "anchor_predictor_inputs": {
                "run_execution_horizon_predictor": True,
                "episode_progress": 0.0,
                "budget_balance": 0.0,
                "previous_valid": False,
            },
            "progress": "generic privileged_progress normalized_score and satisfied_count",
            "resume_authority": "one compressed NPZ per root",
        },
        "completed_episodes": len(selected_episodes),
        "requested_episodes": len(args.episode_ids),
        "root_files": len(selected_roots),
        "expected_root_files_if_all_reachable": len(args.episode_ids) * len(args.root_indices),
        "mean_fresh_minus_stale_h6_normalized_score": mean(
            "fresh_minus_stale_h6_normalized_score"
        ),
        "mean_fresh_minus_stale_handoff_normalized_score": mean(
            "fresh_minus_stale_handoff_normalized_score"
        ),
        "fresh_handoff_wins": int(
            sum(row["fresh_minus_stale_handoff_normalized_score"] > 0 for row in selected_roots)
        ),
        "fresh_handoff_losses": int(
            sum(row["fresh_minus_stale_handoff_normalized_score"] < 0 for row in selected_roots)
        ),
        "fresh_handoff_ties": int(
            sum(row["fresh_minus_stale_handoff_normalized_score"] == 0 for row in selected_roots)
        ),
        "fresh_handoff_predicate_regressions": int(
            sum(row["fresh_minus_stale_handoff_satisfied_count"] < 0 for row in selected_roots)
        ),
    }


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    root_dir = output_dir / "roots"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)
    roots_path = output_dir / "roots.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"

    root_rows = _read_jsonl(roots_path)
    episode_rows = _read_jsonl(episodes_path)
    indexed_root_keys = {
        (int(row["task_id"]), int(row["episode_id"]), int(row["root_index"]))
        for row in root_rows
        if all(name in row for name in ("task_id", "episode_id", "root_index"))
    }
    completed_episode_ids = {
        int(row["episode_id"])
        for row in episode_rows
        if int(row.get("task_id", -1)) == args.task_id
    }

    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.task_id >= suite.n_tasks:
        raise ValueError(f"Task ID {args.task_id} is outside suite size {suite.n_tasks}.")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    root_indices = set(args.root_indices)

    with roots_path.open("a", encoding="utf-8") as roots_file, episodes_path.open(
        "a",
        encoding="utf-8",
    ) as episodes_file:
        for episode_id in args.episode_ids:
            if episode_id in completed_episode_ids:
                continue
            initial_state_id = episode_id % len(initial_states)
            episode_row = _run_episode(
                task=task,
                initial_state=initial_states[initial_state_id],
                initial_state_id=initial_state_id,
                episode_id=episode_id,
                root_indices=root_indices,
                indexed_root_keys=indexed_root_keys,
                args=args,
                client=client,
                output_dir=output_dir,
                root_dir=root_dir,
                roots_file=roots_file,
            )
            _append_jsonl(episodes_file, episode_row)
            episode_rows.append(episode_row)
            completed_episode_ids.add(episode_id)
            root_rows = _read_jsonl(roots_path)
            summary = _summary(args, root_rows, episode_rows)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(episode_row, sort_keys=True), flush=True)

    root_rows = _read_jsonl(roots_path)
    episode_rows = _read_jsonl(episodes_path)
    summary = _summary(args, root_rows, episode_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
