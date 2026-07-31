"""Collect paired stale-vs-fresh plan validity after controlled H4 prefixes.

At every requested Task8 macro root this collector captures one canonical
pre-H4 simulator snapshot and obtains one cached 10-step Action-CoT plan.  It
then restores that snapshot for each selected physical prefix branch, executes
the branch H4, and compares from the exact branch endpoint:

* cached stale actions 4:10 (H6), and
* an exact fresh policy request followed by fresh actions 0:6 (H6).

Both H6 arms receive independently requested fresh H9 continuations with the
same continuation seed.  One compressed NPZ is the resume authority for one
``(episode, root, branch)`` label.  After all branch labels, the environment is
restored to the pre-H4 snapshot and the nominal cached H10 is the only path
used to advance the episode.  This file is independent from existing
collectors and evaluation entrypoints.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os
import pathlib
import time
from typing import Any, Sequence

import collect_action_cot_state_branches as state_branches
import collect_libero_plan_refresh_benefit as refresh_collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.action_cot import branched_dataset


SCHEMA_VERSION = 1
DEFAULT_TASK_ID = 8
DEFAULT_EPISODE_IDS = (*range(0, 10), *range(30, 50))
DEFAULT_ROOT_INDICES = tuple(range(8, 13))
DEFAULT_BRANCHES = (
    "nominal",
    "underact",
    "overact",
    "translation_pulse",
    "gripper_fault",
)
ANCHOR_HORIZON = 4
BRANCH_HORIZON = 6
CONTINUATION_HORIZON = 9
ACTION_DIM = 7
IMAGE_SIZE = 64
DATASET_SHAPE = branched_dataset.BranchedDatasetShape(
    image_height=IMAGE_SIZE,
    image_width=IMAGE_SIZE,
)
BRANCH_ALIASES = {
    "underact": "action_scale_down",
    "overact": "action_scale_up",
}
CANONICAL_TO_DISPLAY = {
    "action_scale_down": "underact",
    "action_scale_up": "overact",
}


@dataclasses.dataclass(frozen=True)
class BranchSpec:
    branch_id: int
    name: str
    canonical_name: str
    strength: float
    actions: np.ndarray


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
    )
    parser.add_argument(
        "--root-indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_ROOT_INDICES),
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=list(DEFAULT_BRANCHES),
        help=(
            "Selected make_branch_actions branches. underact/overact are aliases for "
            "action_scale_down/action_scale_up."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument(
        "--continuation-seed-offset",
        type=int,
        default=100_000_000,
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def _canonical_branch_name(name: str) -> str:
    return BRANCH_ALIASES.get(name, name)


def _display_branch_name(canonical_name: str) -> str:
    return CANONICAL_TO_DISPLAY.get(canonical_name, canonical_name)


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
    canonical = [_canonical_branch_name(name) for name in args.branches]
    unknown = sorted(set(canonical).difference(branched_dataset.BRANCH_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown --branches {unknown}; canonical choices are "
            f"{list(branched_dataset.BRANCH_NAMES)}."
        )
    if len(set(canonical)) != len(canonical):
        raise ValueError("--branches must not contain duplicate canonical branches.")
    if args.seed < 0 or args.num_steps_wait < 0:
        raise ValueError("--seed and --num-steps-wait must be non-negative.")
    if args.resize_size <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("--resize-size and --action-cot-denoising-steps must be positive.")
    if args.continuation_seed_offset <= 0:
        raise ValueError("--continuation-seed-offset must be positive.")


def _protocol_branches(args: argparse.Namespace) -> list[tuple[int, str, str]]:
    result = []
    for requested_name in args.branches:
        canonical = _canonical_branch_name(requested_name)
        branch_id = branched_dataset.BRANCH_IDS[canonical]
        result.append((branch_id, _display_branch_name(canonical), canonical))
    return result


def _branch_specs(primary_actions: np.ndarray, args: argparse.Namespace) -> list[BranchSpec]:
    actions, strengths = state_branches.make_branch_actions(primary_actions)
    return [
        BranchSpec(
            branch_id=branch_id,
            name=display_name,
            canonical_name=canonical_name,
            strength=float(strengths[branch_id]),
            actions=np.asarray(actions[branch_id], dtype=np.float32),
        )
        for branch_id, display_name, canonical_name in _protocol_branches(args)
    ]


def _branch_path(
    root_dir: pathlib.Path,
    task_id: int,
    episode_id: int,
    root_index: int,
    branch_id: int,
    branch_name: str,
) -> pathlib.Path:
    return root_dir / (
        f"task{task_id:02d}_episode{episode_id:03d}_root{root_index:03d}_"
        f"branch{branch_id:02d}_{branch_name}.npz"
    )


def _teacher_request(
    client: websocket_policy.WebsocketClientPolicy,
    policy_input: dict[str, Any],
    *,
    policy_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return state_branches._teacher_request(
        client,
        policy_input,
        policy_seed=policy_seed,
        denoising_steps=args.action_cot_denoising_steps,
    )


def _teacher_tensors(result: dict[str, Any]) -> dict[str, np.ndarray]:
    return state_branches._teacher_tensors(result, DATASET_SHAPE)


def _policy_input(
    observation: dict[str, Any], task_description: str, args: argparse.Namespace
) -> dict[str, Any]:
    return libero_eval._observation_to_policy_input(
        observation,
        task_description,
        args.resize_size,
    )


def _continuation_seed(root_seed: int, args: argparse.Namespace) -> int:
    result = root_seed + args.continuation_seed_offset
    if result > np.iinfo(np.uint32).max:
        raise ValueError(f"Continuation seed {result} exceeds uint32.")
    return result


def _base_arrays(
    *,
    args: argparse.Namespace,
    episode_id: int,
    initial_state_id: int,
    root_index: int,
    decision_step: int,
    root_seed: int,
    continuation_seed: int,
    branch: BranchSpec,
    anchor_input: dict[str, Any],
    endpoint_input: dict[str, Any],
    anchor_result: dict[str, Any],
    current_tensors: dict[str, np.ndarray],
    actual_prefix: np.ndarray,
    actual_prefix_valid: np.ndarray,
    endpoint_done: bool,
    valid: bool,
    invalid_reason: str,
) -> dict[str, np.ndarray]:
    anchor = _teacher_tensors(anchor_result)
    return {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "valid": np.asarray(valid, dtype=np.bool_),
        "invalid_reason": np.asarray(invalid_reason),
        "task_id": np.asarray(args.task_id, dtype=np.int16),
        "episode_id": np.asarray(episode_id, dtype=np.int32),
        "initial_state_id": np.asarray(initial_state_id, dtype=np.int32),
        "root_index": np.asarray(root_index, dtype=np.int32),
        "anchor_decision_step": np.asarray(decision_step, dtype=np.int32),
        "endpoint_decision_step": np.asarray(
            decision_step + int(np.sum(actual_prefix_valid)),
            dtype=np.int32,
        ),
        "anchor_control_step": np.asarray(
            decision_step - args.num_steps_wait,
            dtype=np.int32,
        ),
        "endpoint_control_step": np.asarray(
            decision_step - args.num_steps_wait + int(np.sum(actual_prefix_valid)),
            dtype=np.int32,
        ),
        "policy_seed": np.asarray(root_seed, dtype=np.uint32),
        "continuation_seed": np.asarray(continuation_seed, dtype=np.uint32),
        "branch_id": np.asarray(branch.branch_id, dtype=np.uint8),
        "branch_name": np.asarray(branch.name),
        "branch_canonical_name": np.asarray(branch.canonical_name),
        "branch_strength": np.asarray(branch.strength, dtype=np.float32),
        "branch_endpoint_done": np.asarray(endpoint_done, dtype=np.bool_),
        "anchor_images": state_branches._images(anchor_input, IMAGE_SIZE),
        "current_images": state_branches._images(endpoint_input, IMAGE_SIZE),
        "anchor_state": np.asarray(anchor["state"], dtype=np.float32),
        "current_state": np.asarray(current_tensors["state"], dtype=np.float32),
        "cached_ear": np.asarray(anchor["ear"], dtype=np.float32),
        "cached_iar": np.asarray(anchor["iar"], dtype=np.float32),
        "cached_final_actions": np.asarray(anchor["actions"], dtype=np.float32),
        "cached_env_actions": np.asarray(anchor["actions_env"], dtype=np.float32),
        "fresh_ear": np.asarray(current_tensors["ear"], dtype=np.float32),
        "fresh_iar": np.asarray(current_tensors["iar"], dtype=np.float32),
        "fresh_final_actions": np.asarray(current_tensors["actions"], dtype=np.float32),
        "fresh_env_actions": np.asarray(current_tensors["actions_env"], dtype=np.float32),
        "intended_prefix_env": np.asarray(
            anchor["actions_env"][:ANCHOR_HORIZON, :ACTION_DIM],
            dtype=np.float32,
        ),
        "actual_prefix_env": np.asarray(branch.actions, dtype=np.float32),
        "actual_prefix_executed_env": np.asarray(actual_prefix, dtype=np.float32),
        "actual_prefix_executed_valid": np.asarray(actual_prefix_valid, dtype=np.bool_),
        "actual_prefix_executed_steps": np.asarray(
            np.sum(actual_prefix_valid),
            dtype=np.int16,
        ),
        "anchor_teacher_wall_ms": np.asarray(
            anchor_result["collector_wall_ms"],
            dtype=np.float32,
        ),
    }


def _valid_arrays(
    *,
    args: argparse.Namespace,
    episode_id: int,
    initial_state_id: int,
    root_index: int,
    decision_step: int,
    root_seed: int,
    continuation_seed: int,
    branch: BranchSpec,
    anchor_input: dict[str, Any],
    endpoint_input: dict[str, Any],
    anchor_result: dict[str, Any],
    fresh_result: dict[str, Any],
    actual_prefix: np.ndarray,
    actual_prefix_valid: np.ndarray,
    stale: dict[str, Any],
    fresh: dict[str, Any],
) -> dict[str, np.ndarray]:
    refreshed = _teacher_tensors(fresh_result)
    arrays = _base_arrays(
        args=args,
        episode_id=episode_id,
        initial_state_id=initial_state_id,
        root_index=root_index,
        decision_step=decision_step,
        root_seed=root_seed,
        continuation_seed=continuation_seed,
        branch=branch,
        anchor_input=anchor_input,
        endpoint_input=endpoint_input,
        anchor_result=anchor_result,
        current_tensors=refreshed,
        actual_prefix=actual_prefix,
        actual_prefix_valid=actual_prefix_valid,
        endpoint_done=False,
        valid=True,
        invalid_reason="",
    )
    arrays.update(
        {
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
            "stale_h6_executed_steps": np.asarray(
                stale["h6_executed_steps"], dtype=np.int16
            ),
            "fresh_h6_executed_steps": np.asarray(
                fresh["h6_executed_steps"], dtype=np.int16
            ),
            "stale_h9_executed_steps": np.asarray(
                stale["h9_executed_steps"], dtype=np.int16
            ),
            "fresh_h9_executed_steps": np.asarray(
                fresh["h9_executed_steps"], dtype=np.int16
            ),
            "stale_continuation_requested": np.asarray(
                stale["continuation_requested"], dtype=np.bool_
            ),
            "fresh_continuation_requested": np.asarray(
                fresh["continuation_requested"], dtype=np.bool_
            ),
            "fresh_teacher_wall_ms": np.asarray(
                fresh_result["collector_wall_ms"], dtype=np.float32
            ),
            "stale_continuation_teacher_wall_ms": np.asarray(
                stale["continuation_teacher_wall_ms"], dtype=np.float32
            ),
            "fresh_continuation_teacher_wall_ms": np.asarray(
                fresh["continuation_teacher_wall_ms"], dtype=np.float32
            ),
        }
    )
    arrays.update(refresh_collector._progress_arrays("before", stale["before"]))
    arrays.update(refresh_collector._progress_arrays("stale_h6", stale["after_h6"]))
    arrays.update(refresh_collector._progress_arrays("fresh_h6", fresh["after_h6"]))
    arrays.update(
        refresh_collector._progress_arrays("stale_handoff", stale["after_handoff"])
    )
    arrays.update(
        refresh_collector._progress_arrays("fresh_handoff", fresh["after_handoff"])
    )
    arrays.update(
        refresh_collector._advantage_arrays(
            "h6",
            stale["after_h6"],
            fresh["after_h6"],
        )
    )
    arrays.update(
        refresh_collector._advantage_arrays(
            "handoff",
            stale["after_handoff"],
            fresh["after_handoff"],
        )
    )
    return arrays


def _invalid_arrays(
    *,
    args: argparse.Namespace,
    episode_id: int,
    initial_state_id: int,
    root_index: int,
    decision_step: int,
    root_seed: int,
    continuation_seed: int,
    branch: BranchSpec,
    anchor_input: dict[str, Any],
    endpoint_input: dict[str, Any],
    anchor_result: dict[str, Any],
    actual_prefix: np.ndarray,
    actual_prefix_valid: np.ndarray,
    endpoint_done: bool,
    invalid_reason: str,
    endpoint_progress: dict[str, Any],
) -> dict[str, np.ndarray]:
    anchor = _teacher_tensors(anchor_result)
    arrays = _base_arrays(
        args=args,
        episode_id=episode_id,
        initial_state_id=initial_state_id,
        root_index=root_index,
        decision_step=decision_step,
        root_seed=root_seed,
        continuation_seed=continuation_seed,
        branch=branch,
        anchor_input=anchor_input,
        endpoint_input=endpoint_input,
        anchor_result=anchor_result,
        current_tensors=anchor,
        actual_prefix=actual_prefix,
        actual_prefix_valid=actual_prefix_valid,
        endpoint_done=endpoint_done,
        valid=False,
        invalid_reason=invalid_reason,
    )
    zero_h6 = np.zeros((BRANCH_HORIZON, ACTION_DIM), dtype=np.float32)
    zero_h9 = np.zeros((CONTINUATION_HORIZON, ACTION_DIM), dtype=np.float32)
    arrays.update(
        {
            "stale_h6_actions": zero_h6,
            "fresh_h6_actions": zero_h6.copy(),
            "stale_h6_executed_actions": zero_h6.copy(),
            "fresh_h6_executed_actions": zero_h6.copy(),
            "stale_h6_executed_valid": np.zeros((BRANCH_HORIZON,), dtype=np.bool_),
            "fresh_h6_executed_valid": np.zeros((BRANCH_HORIZON,), dtype=np.bool_),
            "stale_h9_actions": zero_h9,
            "fresh_h9_actions": zero_h9.copy(),
            "stale_h9_executed_actions": zero_h9.copy(),
            "fresh_h9_executed_actions": zero_h9.copy(),
            "stale_h9_executed_valid": np.zeros((CONTINUATION_HORIZON,), dtype=np.bool_),
            "fresh_h9_executed_valid": np.zeros((CONTINUATION_HORIZON,), dtype=np.bool_),
            "stale_h6_executed_steps": np.asarray(0, dtype=np.int16),
            "fresh_h6_executed_steps": np.asarray(0, dtype=np.int16),
            "stale_h9_executed_steps": np.asarray(0, dtype=np.int16),
            "fresh_h9_executed_steps": np.asarray(0, dtype=np.int16),
            "stale_continuation_requested": np.asarray(False, dtype=np.bool_),
            "fresh_continuation_requested": np.asarray(False, dtype=np.bool_),
            "fresh_teacher_wall_ms": np.asarray(0.0, dtype=np.float32),
            "stale_continuation_teacher_wall_ms": np.asarray(0.0, dtype=np.float32),
            "fresh_continuation_teacher_wall_ms": np.asarray(0.0, dtype=np.float32),
        }
    )
    for prefix in ("before", "stale_h6", "fresh_h6", "stale_handoff", "fresh_handoff"):
        arrays.update(refresh_collector._progress_arrays(prefix, endpoint_progress))
    arrays.update(
        refresh_collector._advantage_arrays("h6", endpoint_progress, endpoint_progress)
    )
    arrays.update(
        refresh_collector._advantage_arrays(
            "handoff",
            endpoint_progress,
            endpoint_progress,
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


def _scalar(arrays: Any, name: str) -> Any:
    return np.asarray(arrays[name]).reshape(()).item()


def _index_row(path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        return {
            "schema_version": int(_scalar(arrays, "schema_version")),
            "valid": bool(_scalar(arrays, "valid")),
            "invalid_reason": str(_scalar(arrays, "invalid_reason")),
            "task_id": int(_scalar(arrays, "task_id")),
            "episode_id": int(_scalar(arrays, "episode_id")),
            "initial_state_id": int(_scalar(arrays, "initial_state_id")),
            "root_index": int(_scalar(arrays, "root_index")),
            "anchor_decision_step": int(_scalar(arrays, "anchor_decision_step")),
            "endpoint_decision_step": int(_scalar(arrays, "endpoint_decision_step")),
            "branch_id": int(_scalar(arrays, "branch_id")),
            "branch_name": str(_scalar(arrays, "branch_name")),
            "branch_canonical_name": str(_scalar(arrays, "branch_canonical_name")),
            "branch_strength": float(_scalar(arrays, "branch_strength")),
            "policy_seed": int(_scalar(arrays, "policy_seed")),
            "continuation_seed": int(_scalar(arrays, "continuation_seed")),
            "npz_file": str(path.relative_to(output_dir)),
            "fresh_minus_stale_h6_normalized_score": float(
                _scalar(arrays, "fresh_minus_stale_h6_normalized_score")
            ),
            "fresh_minus_stale_h6_satisfied_count": int(
                _scalar(arrays, "fresh_minus_stale_h6_satisfied_count")
            ),
            "fresh_minus_stale_handoff_normalized_score": float(
                _scalar(arrays, "fresh_minus_stale_handoff_normalized_score")
            ),
            "fresh_minus_stale_handoff_satisfied_count": int(
                _scalar(arrays, "fresh_minus_stale_handoff_satisfied_count")
            ),
            "fresh_minus_stale_handoff_success": int(
                _scalar(arrays, "fresh_minus_stale_handoff_success")
            ),
        }


def _collect_branch(
    env: Any,
    pre_h4_snapshot: state_branches.CanonicalSimulatorSnapshot,
    *,
    branch: BranchSpec,
    task_description: str,
    episode_id: int,
    initial_state_id: int,
    root_index: int,
    decision_step: int,
    root_seed: int,
    anchor_input: dict[str, Any],
    anchor_result: dict[str, Any],
    anchor_tensors: dict[str, np.ndarray],
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    observation = state_branches._restore_canonical_snapshot(env, pre_h4_snapshot)
    observation, done, executed = state_branches._step_actions(
        env,
        observation,
        branch.actions,
    )
    actual_prefix, actual_prefix_valid = refresh_collector._padded_executed(
        executed,
        ANCHOR_HORIZON,
    )
    endpoint_input = _policy_input(observation, task_description, args)
    continuation_seed = _continuation_seed(root_seed, args)
    if done or libero_eval._env_success(env) or len(executed) < ANCHOR_HORIZON:
        reason = (
            "terminal_or_success_during_prefix"
            if done or libero_eval._env_success(env)
            else "short_prefix_execution"
        )
        arrays = _invalid_arrays(
            args=args,
            episode_id=episode_id,
            initial_state_id=initial_state_id,
            root_index=root_index,
            decision_step=decision_step,
            root_seed=root_seed,
            continuation_seed=continuation_seed,
            branch=branch,
            anchor_input=anchor_input,
            endpoint_input=endpoint_input,
            anchor_result=anchor_result,
            actual_prefix=actual_prefix,
            actual_prefix_valid=actual_prefix_valid,
            endpoint_done=bool(done or libero_eval._env_success(env)),
            invalid_reason=reason,
            endpoint_progress=refresh_collector._progress(env, observation),
        )
        _write_npz_atomic(output_path, arrays)
        return _index_row(output_path, output_path.parent.parent)

    endpoint_snapshot = state_branches._capture_canonical_snapshot(env)
    try:
        fresh_result = _teacher_request(
            client,
            endpoint_input,
            policy_seed=root_seed,
            args=args,
        )
        fresh_tensors = _teacher_tensors(fresh_result)
        stale_h6 = np.asarray(
            anchor_tensors["actions_env"][
                ANCHOR_HORIZON : ANCHOR_HORIZON + BRANCH_HORIZON,
                :ACTION_DIM,
            ],
            dtype=np.float32,
        )
        fresh_h6 = np.asarray(
            fresh_tensors["actions_env"][:BRANCH_HORIZON, :ACTION_DIM],
            dtype=np.float32,
        )
        stale = refresh_collector._run_branch(
            env,
            endpoint_snapshot,
            stale_h6,
            continuation_seed=continuation_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        fresh = refresh_collector._run_branch(
            env,
            endpoint_snapshot,
            fresh_h6,
            continuation_seed=continuation_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        arrays = _valid_arrays(
            args=args,
            episode_id=episode_id,
            initial_state_id=initial_state_id,
            root_index=root_index,
            decision_step=decision_step,
            root_seed=root_seed,
            continuation_seed=continuation_seed,
            branch=branch,
            anchor_input=anchor_input,
            endpoint_input=endpoint_input,
            anchor_result=anchor_result,
            fresh_result=fresh_result,
            actual_prefix=actual_prefix,
            actual_prefix_valid=actual_prefix_valid,
            stale=stale,
            fresh=fresh,
        )
        _write_npz_atomic(output_path, arrays)
        return _index_row(output_path, output_path.parent.parent)
    finally:
        state_branches._restore_canonical_snapshot(env, pre_h4_snapshot)


def _protocol_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_id": args.task_id,
        "episode_ids": list(args.episode_ids),
        "root_indices": sorted(args.root_indices),
        "branches": [item[1] for item in _protocol_branches(args)],
        "canonical_branches": [item[2] for item in _protocol_branches(args)],
        "seed": args.seed,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "continuation_seed_offset": args.continuation_seed_offset,
    }


def _episode_already_complete(
    rows: Sequence[dict[str, Any]], episode_id: int, args: argparse.Namespace
) -> bool:
    signature = _protocol_signature(args)
    for row in reversed(rows):
        if int(row.get("episode_id", -1)) != episode_id:
            continue
        return bool(row.get("complete", False)) and row.get("protocol") == signature
    return False


def _run_episode(
    *,
    task: Any,
    initial_state: Any,
    initial_state_id: int,
    episode_id: int,
    root_indices: set[int],
    indexed_keys: set[tuple[int, int, int, int]],
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    output_dir: pathlib.Path,
    root_dir: pathlib.Path,
    index_file: Any,
) -> dict[str, Any]:
    env, task_description = libero_eval._get_libero_env(
        task,
        libero_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    started = time.monotonic()
    new_files = 0
    resumed_files = 0
    repaired_rows = 0
    failed = 0
    invalid = 0
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
        maximum_root = max(root_indices)
        termination_reason = ""
        while not done and step < step_limit and root_index <= maximum_root:
            decision_step = step
            root_seed = state_branches.canonical_policy_seed(
                args.seed,
                args.task_id,
                episode_id,
                decision_step,
            )
            anchor_input = _policy_input(observation, task_description, args)
            anchor_result = _teacher_request(
                client,
                anchor_input,
                policy_seed=root_seed,
                args=args,
            )
            anchor_tensors = _teacher_tensors(anchor_result)

            if root_index in root_indices:
                pre_h4_snapshot = state_branches._capture_canonical_snapshot(env)
                for branch in _branch_specs(anchor_tensors["actions_env"], args):
                    output_path = _branch_path(
                        root_dir,
                        args.task_id,
                        episode_id,
                        root_index,
                        branch.branch_id,
                        branch.name,
                    )
                    key = (args.task_id, episode_id, root_index, branch.branch_id)
                    if output_path.exists():
                        if key not in indexed_keys:
                            row = _index_row(output_path, output_dir)
                            refresh_collector._append_jsonl(index_file, row)
                            indexed_keys.add(key)
                            repaired_rows += 1
                        resumed_files += 1
                        continue
                    try:
                        row = _collect_branch(
                            env,
                            pre_h4_snapshot,
                            branch=branch,
                            task_description=task_description,
                            episode_id=episode_id,
                            initial_state_id=initial_state_id,
                            root_index=root_index,
                            decision_step=decision_step,
                            root_seed=root_seed,
                            anchor_input=anchor_input,
                            anchor_result=anchor_result,
                            anchor_tensors=anchor_tensors,
                            args=args,
                            client=client,
                            output_path=output_path,
                        )
                        refresh_collector._append_jsonl(index_file, row)
                        indexed_keys.add(key)
                        new_files += 1
                        invalid += int(not row["valid"])
                        print(
                            json.dumps(
                                {
                                    "task_id": args.task_id,
                                    "episode_id": episode_id,
                                    "root_index": root_index,
                                    "branch": branch.name,
                                    "valid": row["valid"],
                                    "h6_advantage": row[
                                        "fresh_minus_stale_h6_normalized_score"
                                    ],
                                    "handoff_advantage": row[
                                        "fresh_minus_stale_handoff_normalized_score"
                                    ],
                                    "npz_file": row["npz_file"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        failed += 1
                        print(
                            json.dumps(
                                {
                                    "task_id": args.task_id,
                                    "episode_id": episode_id,
                                    "root_index": root_index,
                                    "branch": branch.name,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if not args.continue_on_error:
                            raise
                    finally:
                        observation = state_branches._restore_canonical_snapshot(
                            env,
                            pre_h4_snapshot,
                        )

            observation, done, executed = state_branches._step_actions(
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
            "new_branch_files": new_files,
            "resumed_branch_files": resumed_files,
            "repaired_index_rows": repaired_rows,
            "invalid_branch_files": invalid,
            "failed_branches": failed,
            "complete": failed == 0,
            "protocol": _protocol_signature(args),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        libero_eval._safe_close_env(env)


def _summary(
    args: argparse.Namespace,
    branch_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    episode_set = set(args.episode_ids)
    root_set = set(args.root_indices)
    branch_ids = {item[0] for item in _protocol_branches(args)}
    selected = [
        row
        for row in branch_rows
        if int(row.get("task_id", -1)) == args.task_id
        and int(row.get("episode_id", -1)) in episode_set
        and int(row.get("root_index", -1)) in root_set
        and int(row.get("branch_id", -1)) in branch_ids
    ]
    deduplicated = {
        (
            int(row["task_id"]),
            int(row["episode_id"]),
            int(row["root_index"]),
            int(row["branch_id"]),
        ): row
        for row in selected
    }
    selected = list(deduplicated.values())
    complete_episodes = {
        int(row["episode_id"])
        for row in episode_rows
        if int(row.get("task_id", -1)) == args.task_id
        and int(row.get("episode_id", -1)) in episode_set
        and bool(row.get("complete", False))
        and row.get("protocol") == _protocol_signature(args)
    }
    by_branch: dict[str, Any] = {}
    for branch_id, display_name, _ in _protocol_branches(args):
        rows = [row for row in selected if int(row["branch_id"]) == branch_id]
        valid = [row for row in rows if bool(row["valid"])]
        handoff = [float(row["fresh_minus_stale_handoff_normalized_score"]) for row in valid]
        h6 = [float(row["fresh_minus_stale_h6_normalized_score"]) for row in valid]
        by_branch[display_name] = {
            "files": len(rows),
            "valid": len(valid),
            "invalid": len(rows) - len(valid),
            "mean_h6_advantage": float(np.mean(h6)) if h6 else None,
            "mean_handoff_advantage": float(np.mean(handoff)) if handoff else None,
            "handoff_wins": int(sum(value > 0 for value in handoff)),
            "handoff_losses": int(sum(value < 0 for value in handoff)),
            "handoff_ties": int(sum(value == 0 for value in handoff)),
            "handoff_predicate_regressions": int(
                sum(
                    int(row["fresh_minus_stale_handoff_satisfied_count"]) < 0
                    for row in valid
                )
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete" if len(complete_episodes) == len(args.episode_ids) else "running"
        ),
        "protocol": {
            **_protocol_signature(args),
            "task_suite_name": args.task_suite_name,
            "trajectory": "restore_pre_h4_then_nominal_cached_h10_only",
            "prefix_horizon": ANCHOR_HORIZON,
            "comparison_horizon": BRANCH_HORIZON,
            "continuation_horizon": CONTINUATION_HORIZON,
            "same_seed_stale_fresh_h9": True,
            "resume_authority": "one compressed NPZ per episode/root/branch",
            "progress": "generic privileged progress; label-only",
        },
        "completed_episodes": len(complete_episodes),
        "requested_episodes": len(args.episode_ids),
        "branch_files": len(selected),
        "expected_branch_files_if_all_roots_reachable": (
            len(args.episode_ids) * len(args.root_indices) * len(args.branches)
        ),
        "by_branch": by_branch,
        "note": (
            "Paired same-snapshot branch labels only; no deployment or closed-loop success claim."
        ),
    }


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    root_dir = output_dir / "roots"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "branches.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"

    branch_rows = refresh_collector._read_jsonl(index_path)
    episode_rows = refresh_collector._read_jsonl(episodes_path)
    indexed_keys = {
        (
            int(row["task_id"]),
            int(row["episode_id"]),
            int(row["root_index"]),
            int(row["branch_id"]),
        )
        for row in branch_rows
        if all(
            name in row
            for name in ("task_id", "episode_id", "root_index", "branch_id")
        )
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

    with index_path.open("a", encoding="utf-8") as index_file, episodes_path.open(
        "a",
        encoding="utf-8",
    ) as episodes_file:
        for episode_id in args.episode_ids:
            if _episode_already_complete(episode_rows, episode_id, args):
                continue
            initial_state_id = episode_id % len(initial_states)
            row = _run_episode(
                task=task,
                initial_state=initial_states[initial_state_id],
                initial_state_id=initial_state_id,
                episode_id=episode_id,
                root_indices=root_indices,
                indexed_keys=indexed_keys,
                args=args,
                client=client,
                output_dir=output_dir,
                root_dir=root_dir,
                index_file=index_file,
            )
            refresh_collector._append_jsonl(episodes_file, row)
            episode_rows.append(row)
            branch_rows = refresh_collector._read_jsonl(index_path)
            summary = _summary(args, branch_rows, episode_rows)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    branch_rows = refresh_collector._read_jsonl(index_path)
    episode_rows = refresh_collector._read_jsonl(episodes_path)
    summary = _summary(args, branch_rows, episode_rows)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
