"""Collect same-snapshot alpha=0 versus alpha=0.05 Task8 labels.

The default pilot replays compact MuJoCo roots already stored by the execution-
horizon collector.  At every root it regenerates two action chunks with the
same policy seed, executes H10 from the exact same simulator snapshot, and then
uses one common alpha=0 H10 continuation policy for both arms.  Privileged
LIBERO progress is recorded at H10 and H20.  With ``--terminal``, that same
control continuation is repeated to the original episode limit and terminal
success becomes the primary label.

Episodes 10-29 are reserved for formal Task8 evaluation and are rejected by
default.  One compressed NPZ per root is the resume authority; the physics
state is retained so a cheap progress pilot can later be upgraded to terminal
labels without recollecting the trajectory.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import time
from typing import Any, Sequence

import collect_action_cot_state_branches as state_branches
import collect_execution_horizon_counterfactuals as horizon_collector
import collect_libero_plan_refresh_benefit as progress_collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
import replay_execution_horizon_branches as replay_collector
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import dataset as horizon_dataset


SCHEMA_VERSION = 1
TASK_ID = 8
FORMAL_EPISODES = frozenset(range(10, 30))
DEFAULT_EPISODES = tuple(range(10))
ACTION_HORIZON = 10
ACTION_DIM = 7
IMAGE_SIZE = 64


@dataclasses.dataclass(frozen=True)
class LiveRoot:
    episode_id: int
    decision_step: int
    root_seed: int
    call_index: int
    snapshot: state_branches.CanonicalSimulatorSnapshot
    router_score: float
    router_alpha: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Counterfactual dataset root containing saved physics_state; repeat as needed.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--episode-ids", nargs="+", type=int, default=list(DEFAULT_EPISODES))
    parser.add_argument("--max-roots", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=1)
    parser.add_argument("--alternative-alpha", type=float, default=0.05)
    parser.add_argument(
        "--live-on-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collect canonical roots from a live compact-router rollout instead of --dataset. "
            "The candidate alpha=0/.05 first chunk is followed by that same compact router, "
            "yielding Q^pi terminal labels without H1-H10 branch collection."
        ),
    )
    parser.add_argument(
        "--live-roots-per-episode",
        type=int,
        default=3,
        help="Number of evenly-spaced early/middle/late roots retained per live episode.",
    )
    parser.add_argument(
        "--terminal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue both arms with alpha=0 H10 replanning to terminal success/timeout.",
    )
    parser.add_argument(
        "--allow-formal-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly allow reserved Task8 episode IDs 10-29.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.episode_ids or any(episode < 0 for episode in args.episode_ids):
        raise ValueError("--episode-ids must contain non-negative IDs.")
    if len(set(args.episode_ids)) != len(args.episode_ids):
        raise ValueError("--episode-ids must not contain duplicates.")
    forbidden = sorted(set(args.episode_ids) & FORMAL_EPISODES)
    if forbidden and not args.allow_formal_episodes:
        raise ValueError(
            f"Task8 formal episodes 10-29 are reserved and forbidden by default: {forbidden}."
        )
    if args.max_roots < 0:
        raise ValueError("--max-roots must be non-negative; zero means all matching roots.")
    if args.live_roots_per_episode <= 0:
        raise ValueError("--live-roots-per-episode must be positive.")
    if args.live_on_policy and args.dataset:
        raise ValueError("--live-on-policy and --dataset are mutually exclusive.")
    if not args.live_on_policy and not args.dataset:
        raise ValueError("At least one --dataset is required unless --live-on-policy is enabled.")
    if args.seed < 0 or args.num_steps_wait < 0:
        raise ValueError("--seed and --num-steps-wait must be non-negative.")
    if args.resize_size <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("Resize size and Action-CoT denoising steps must be positive.")
    if not 0.0 < args.alternative_alpha < 1.0:
        raise ValueError("--alternative-alpha must lie strictly between zero and one.")


def _physics_key(physics_state: np.ndarray) -> str:
    payload = np.asarray(physics_state, dtype=np.float64).tobytes()
    return hashlib.sha1(payload).hexdigest()


def _selected_indices(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> list[int]:
    task_ids = np.asarray(arrays["task_id"], dtype=np.int64)
    episode_ids = np.asarray(arrays["episode_id"], dtype=np.int64)
    decision_steps = np.asarray(arrays["decision_step"], dtype=np.int64)
    source_iterations = np.asarray(
        arrays.get("source_iteration", np.zeros_like(task_ids)),
        dtype=np.int64,
    )
    mask = (task_ids == TASK_ID) & np.isin(
        episode_ids,
        np.asarray(args.episode_ids, dtype=np.int64),
    )
    candidates = np.flatnonzero(mask)
    order = np.lexsort(
        (
            source_iterations[candidates],
            decision_steps[candidates],
            episode_ids[candidates],
        )
    )
    selected: list[int] = []
    seen_physics: set[str] = set()
    for index in candidates[order]:
        key = _physics_key(arrays["physics_state"][index])
        if key in seen_physics:
            continue
        selected.append(int(index))
        seen_physics.add(key)
        if args.max_roots and len(selected) >= args.max_roots:
            break
    return selected


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


def _request(
    client: websocket_policy.WebsocketClientPolicy,
    element: dict[str, Any],
    *,
    policy_seed: int,
    alpha: float,
    args: argparse.Namespace,
    export_cache: bool = False,
    compact_router: bool = False,
    absolute_step: int | None = None,
    profile_policy_timing: bool = False,
    force_final_one_step: bool = False,
) -> dict[str, Any]:
    request = {
        **element,
        "policy_seed": np.asarray(policy_seed, dtype=np.int64),
        "profile_policy_timing": np.asarray(profile_policy_timing, dtype=np.bool_),
        "action_cot_denoising_steps": np.asarray(
            args.action_cot_denoising_steps,
            dtype=np.int32,
        ),
    }
    if compact_router:
        if absolute_step is None or absolute_step < 0:
            raise ValueError("compact_router requests require a non-negative absolute_step.")
        if alpha != 0.0:
            raise ValueError("compact_router owns alpha; do not also pass a fixed alpha.")
        request.update(
            {
                "action_cot_compact_alpha_router": np.asarray(True, dtype=np.bool_),
                "action_cot_absolute_decision_step": np.asarray(
                    absolute_step,
                    dtype=np.int32,
                ),
            }
        )
    elif alpha > 0.0:
        request["action_cot_final_time_warp_alpha"] = np.asarray(alpha, dtype=np.float32)
    if force_final_one_step:
        request["action_cot_final_denoising_steps"] = np.asarray(1, dtype=np.int32)
    if export_cache:
        request["export_acot_cache"] = np.asarray(True, dtype=np.bool_)
    started = time.perf_counter()
    result = client.infer(request)
    result["collector_wall_ms"] = (time.perf_counter() - started) * 1000.0
    if "actions" not in result:
        raise KeyError("Policy response does not contain actions.")
    return result


def _progress_arrays(prefix: str, value: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_score": np.asarray(value["score"], dtype=np.float32),
        f"{prefix}_normalized_score": np.asarray(value["normalized_score"], dtype=np.float32),
        f"{prefix}_satisfied_count": np.asarray(value["satisfied_count"], dtype=np.int16),
        f"{prefix}_total_goals": np.asarray(value["total_goals"], dtype=np.int16),
        f"{prefix}_active_progress": np.asarray(value["active_progress"], dtype=np.float32),
        f"{prefix}_success": np.asarray(value["success"], dtype=np.bool_),
    }


def _padded_actions(actions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    valid = np.zeros((ACTION_HORIZON,), dtype=np.bool_)
    count = min(len(actions), ACTION_HORIZON)
    if count:
        padded[:count] = np.asarray(actions[:count], dtype=np.float32)[:, :ACTION_DIM]
        valid[:count] = True
    return padded, valid


def _restore_branch_snapshot(
    env: Any,
    snapshot: horizon_collector.SimulatorSnapshot
    | state_branches.CanonicalSimulatorSnapshot,
) -> dict[str, Any]:
    if isinstance(snapshot, state_branches.CanonicalSimulatorSnapshot):
        return state_branches._restore_canonical_snapshot(env, snapshot)
    return horizon_collector._restore_snapshot(env, snapshot)


def _run_arm(
    env: Any,
    snapshot: horizon_collector.SimulatorSnapshot
    | state_branches.CanonicalSimulatorSnapshot,
    initial_actions: np.ndarray,
    *,
    root_seed: int,
    root_step: int,
    episode_step_limit: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    continuation_policy: str = "alpha0",
    profile_policy_timing: bool = False,
) -> dict[str, Any]:
    if continuation_policy not in {"alpha0", "current_router"}:
        raise ValueError(f"Unsupported continuation_policy: {continuation_policy!r}.")
    observation = _restore_branch_snapshot(env, snapshot)
    before = progress_collector._progress(env, observation)
    observation, done, executed = state_branches._step_actions(
        env,
        observation,
        np.asarray(initial_actions, dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM],
    )
    initial_executed, initial_valid = _padded_actions(executed)
    absolute_step = root_step + len(executed)
    after_h10 = progress_collector._progress(env, observation)
    continuation_calls = 0
    continuation_wall_ms = 0.0
    continuation_steps = 0
    logical_policy_seeds: list[int] = []
    first_continuation_actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    first_continuation_executed = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    first_continuation_valid = np.zeros((ACTION_HORIZON,), dtype=np.bool_)
    after_h20 = after_h10
    no_progress = False

    while (
        not done
        and not libero_eval._env_success(env)
        and absolute_step < episode_step_limit
        and (args.terminal or continuation_calls < 1)
    ):
        policy_seed = root_seed + (absolute_step - root_step)
        if policy_seed > np.iinfo(np.uint32).max:
            raise ValueError(f"Continuation seed {policy_seed} exceeds uint32.")
        element = _policy_input(observation, task_description, args)
        result = _request(
            client,
            element,
            policy_seed=policy_seed,
            alpha=0.0,
            args=args,
            compact_router=continuation_policy == "current_router",
            absolute_step=(
                absolute_step if continuation_policy == "current_router" else None
            ),
            profile_policy_timing=profile_policy_timing,
        )
        actions = np.asarray(result["actions"], dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM]
        if continuation_calls == 0:
            first_continuation_actions[: len(actions)] = actions
        observation, done, executed = state_branches._step_actions(
            env,
            observation,
            actions,
        )
        if continuation_calls == 0:
            first_continuation_executed, first_continuation_valid = _padded_actions(executed)
        executed_count = len(executed)
        continuation_steps += executed_count
        absolute_step += executed_count
        continuation_calls += 1
        continuation_wall_ms += float(result["collector_wall_ms"])
        logical_policy_seeds.append(policy_seed)
        if continuation_calls == 1:
            after_h20 = progress_collector._progress(env, observation)
        if executed_count == 0:
            no_progress = True
            break

    terminal_progress = progress_collector._progress(env, observation)
    terminal_success = bool(terminal_progress["success"])
    terminal_reason = "not_evaluated"
    if args.terminal:
        if terminal_success:
            terminal_reason = "success"
        elif done:
            terminal_reason = "environment_done"
        elif absolute_step >= episode_step_limit:
            terminal_reason = "step_limit"
        elif no_progress:
            terminal_reason = "no_environment_progress"
        else:
            terminal_reason = "stopped"
    return {
        "before": before,
        "after_h10": after_h10,
        "after_h20": after_h20,
        "terminal": terminal_progress,
        "initial_executed_actions": initial_executed,
        "initial_executed_valid": initial_valid,
        "first_continuation_actions": first_continuation_actions,
        "first_continuation_executed_actions": first_continuation_executed,
        "first_continuation_executed_valid": first_continuation_valid,
        "continuation_calls": continuation_calls,
        "continuation_wall_ms": continuation_wall_ms,
        "continuation_steps": continuation_steps,
        "logical_policy_seeds": np.asarray(logical_policy_seeds, dtype=np.uint32),
        "absolute_step": absolute_step,
        "terminal_success": terminal_success,
        "terminal_reason": terminal_reason,
    }


def _preference(
    alpha0: dict[str, Any],
    alternative: dict[str, Any],
    *,
    terminal: bool,
    tolerance: float = 1e-6,
) -> tuple[int, str]:
    if terminal:
        success_difference = int(alternative["terminal_success"]) - int(alpha0["terminal_success"])
        if success_difference > 0:
            return 1, "terminal_rescue"
        if success_difference < 0:
            return -1, "terminal_regression"
        if alpha0["terminal_success"] and alternative["terminal_success"]:
            if alternative["continuation_calls"] < alpha0["continuation_calls"]:
                return 1, "both_success_fewer_calls"
            if alternative["continuation_calls"] > alpha0["continuation_calls"]:
                return -1, "both_success_more_calls"
            if alternative["absolute_step"] < alpha0["absolute_step"]:
                return 1, "both_success_fewer_steps"
            if alternative["absolute_step"] > alpha0["absolute_step"]:
                return -1, "both_success_more_steps"
        delta = (
            alternative["terminal"]["normalized_score"]
            - alpha0["terminal"]["normalized_score"]
        )
        if delta > tolerance:
            return 1, "terminal_progress_gain"
        if delta < -tolerance:
            return -1, "terminal_progress_loss"
        return 0, "terminal_tie"

    h20_delta = (
        alternative["after_h20"]["normalized_score"]
        - alpha0["after_h20"]["normalized_score"]
    )
    if h20_delta > tolerance:
        return 1, "h20_progress_gain"
    if h20_delta < -tolerance:
        return -1, "h20_progress_loss"
    h10_delta = (
        alternative["after_h10"]["normalized_score"]
        - alpha0["after_h10"]["normalized_score"]
    )
    if h10_delta > tolerance:
        return 1, "h10_progress_gain"
    if h10_delta < -tolerance:
        return -1, "h10_progress_loss"
    return 0, "progress_tie"


def _root_path(
    root_dir: pathlib.Path,
    episode_id: int,
    decision_step: int,
    physics_key: str,
) -> pathlib.Path:
    return root_dir / (
        f"task{TASK_ID:02d}_episode{episode_id:03d}_step{decision_step:04d}_"
        f"{physics_key[:12]}.npz"
    )


def _scalar(data: Any, name: str) -> Any:
    return np.asarray(data[name]).reshape(()).item()


def _index_row(path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "schema_version": int(_scalar(data, "schema_version")),
            "task_id": int(_scalar(data, "task_id")),
            "episode_id": int(_scalar(data, "episode_id")),
            "decision_step": int(_scalar(data, "decision_step")),
            "physics_key": str(_scalar(data, "physics_key")),
            "terminal_evaluated": bool(_scalar(data, "terminal_evaluated")),
            "preference_label": int(_scalar(data, "preference_label")),
            "preference_reason": str(_scalar(data, "preference_reason")),
            "alternative_minus_alpha0_h10_progress": float(
                _scalar(data, "alternative_minus_alpha0_h10_progress")
            ),
            "alternative_minus_alpha0_h20_progress": float(
                _scalar(data, "alternative_minus_alpha0_h20_progress")
            ),
            "terminal_success_difference": int(_scalar(data, "terminal_success_difference")),
            "action_rmse": float(_scalar(data, "action_rmse")),
            "coarse_max_abs_difference": float(
                _scalar(data, "coarse_max_abs_difference")
            ),
            "npz_file": str(path.relative_to(output_dir)),
        }


def _record_arrays(
    *,
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    index: int,
    physics_key: str,
    element: dict[str, Any],
    alpha0_result: dict[str, Any],
    alternative_result: dict[str, Any],
    alpha0: dict[str, Any],
    alternative: dict[str, Any],
    preference_label: int,
    preference_reason: str,
) -> dict[str, np.ndarray]:
    alpha0_actions = np.asarray(alpha0_result["actions"], dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM]
    alternative_actions = np.asarray(
        alternative_result["actions"],
        dtype=np.float32,
    )[:ACTION_HORIZON, :ACTION_DIM]
    h10_delta = (
        alternative["after_h10"]["normalized_score"]
        - alpha0["after_h10"]["normalized_score"]
    )
    h20_delta = (
        alternative["after_h20"]["normalized_score"]
        - alpha0["after_h20"]["normalized_score"]
    )
    terminal_delta = int(alternative["terminal_success"]) - int(alpha0["terminal_success"])
    result: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "task_id": np.asarray(TASK_ID, dtype=np.int16),
        "episode_id": np.asarray(arrays["episode_id"][index], dtype=np.int32),
        "decision_step": np.asarray(arrays["decision_step"][index], dtype=np.int32),
        "root_seed": np.asarray(arrays["root_seed"][index], dtype=np.uint32),
        "source_iteration": np.asarray(arrays["source_iteration"][index], dtype=np.int16),
        "physics_key": np.asarray(physics_key),
        "physics_state": np.asarray(arrays["physics_state"][index], dtype=np.float64),
        "current_images": state_branches._images(element, IMAGE_SIZE),
        "normalized_state": np.asarray(
            alpha0_result["execution_horizon_state_normalized"],
            dtype=np.float32,
        ),
        "acot_prefix_feature": np.asarray(alpha0_result["acot_prefix_feature"], dtype=np.float16),
        "acot_iar_tokens": np.asarray(alpha0_result["acot_iar_tokens"], dtype=np.float16),
        "coarse_actions": np.asarray(alpha0_result["coarse_actions"], dtype=np.float16),
        "alpha0_actions": alpha0_actions,
        "alternative_actions": alternative_actions,
        "alternative_alpha": np.asarray(args.alternative_alpha, dtype=np.float32),
        "action_rmse": np.asarray(
            np.sqrt(np.mean((alternative_actions - alpha0_actions) ** 2)),
            dtype=np.float32,
        ),
        "coarse_max_abs_difference": np.asarray(
            np.max(
                np.abs(
                    np.asarray(alternative_result["coarse_actions"], dtype=np.float32)
                    - np.asarray(alpha0_result["coarse_actions"], dtype=np.float32)
                )
            ),
            dtype=np.float32,
        ),
        "alpha0_request_wall_ms": np.asarray(alpha0_result["collector_wall_ms"], dtype=np.float32),
        "alternative_request_wall_ms": np.asarray(
            alternative_result["collector_wall_ms"],
            dtype=np.float32,
        ),
        "terminal_evaluated": np.asarray(args.terminal, dtype=np.bool_),
        "preference_label": np.asarray(preference_label, dtype=np.int8),
        "preference_reason": np.asarray(preference_reason),
        "alternative_minus_alpha0_h10_progress": np.asarray(h10_delta, dtype=np.float32),
        "alternative_minus_alpha0_h20_progress": np.asarray(h20_delta, dtype=np.float32),
        "terminal_success_difference": np.asarray(terminal_delta, dtype=np.int8),
        "alpha0_terminal_success": np.asarray(alpha0["terminal_success"], dtype=np.bool_),
        "alternative_terminal_success": np.asarray(alternative["terminal_success"], dtype=np.bool_),
        "alpha0_terminal_reason": np.asarray(alpha0["terminal_reason"]),
        "alternative_terminal_reason": np.asarray(alternative["terminal_reason"]),
        "alpha0_terminal_absolute_step": np.asarray(alpha0["absolute_step"], dtype=np.int32),
        "alternative_terminal_absolute_step": np.asarray(
            alternative["absolute_step"],
            dtype=np.int32,
        ),
        "alpha0_continuation_calls": np.asarray(alpha0["continuation_calls"], dtype=np.int16),
        "alternative_continuation_calls": np.asarray(
            alternative["continuation_calls"],
            dtype=np.int16,
        ),
        "alpha0_continuation_wall_ms": np.asarray(alpha0["continuation_wall_ms"], dtype=np.float32),
        "alternative_continuation_wall_ms": np.asarray(
            alternative["continuation_wall_ms"],
            dtype=np.float32,
        ),
        "alpha0_initial_executed_actions": alpha0["initial_executed_actions"],
        "alternative_initial_executed_actions": alternative["initial_executed_actions"],
        "alpha0_initial_executed_valid": alpha0["initial_executed_valid"],
        "alternative_initial_executed_valid": alternative["initial_executed_valid"],
        "alpha0_first_continuation_actions": alpha0["first_continuation_actions"],
        "alternative_first_continuation_actions": alternative["first_continuation_actions"],
        "alpha0_first_continuation_executed_actions": alpha0[
            "first_continuation_executed_actions"
        ],
        "alternative_first_continuation_executed_actions": alternative[
            "first_continuation_executed_actions"
        ],
        "alpha0_first_continuation_executed_valid": alpha0[
            "first_continuation_executed_valid"
        ],
        "alternative_first_continuation_executed_valid": alternative[
            "first_continuation_executed_valid"
        ],
        "alpha0_logical_policy_seeds": alpha0["logical_policy_seeds"],
        "alternative_logical_policy_seeds": alternative["logical_policy_seeds"],
    }
    for prefix, arm in (("alpha0", alpha0), ("alternative", alternative)):
        result.update(_progress_arrays(f"{prefix}_before", arm["before"]))
        result.update(_progress_arrays(f"{prefix}_h10", arm["after_h10"]))
        result.update(_progress_arrays(f"{prefix}_h20", arm["after_h20"]))
        result.update(_progress_arrays(f"{prefix}_terminal", arm["terminal"]))
    return result


def _collect_root(
    *,
    client: websocket_policy.WebsocketClientPolicy,
    task_suite: Any,
    arrays: dict[str, np.ndarray],
    index: int,
    args: argparse.Namespace,
    output_path: pathlib.Path,
    physics_key: str,
) -> dict[str, Any]:
    task_id = int(arrays["task_id"][index])
    episode_id = int(arrays["episode_id"][index])
    root_step = int(arrays["decision_step"][index])
    root_seed = int(arrays["root_seed"][index])
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = libero_eval._get_libero_env(
        task,
        libero_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    try:
        env.reset()
        observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            if done:
                break
        snapshot = replay_collector._saved_snapshot(
            env,
            arrays["physics_state"][index],
            root_step,
        )
        observation = horizon_collector._restore_snapshot(env, snapshot)
        element = _policy_input(observation, task_description, args)
        alpha0_result = _request(
            client,
            element,
            policy_seed=root_seed,
            alpha=0.0,
            args=args,
            export_cache=True,
        )
        alternative_result = _request(
            client,
            element,
            policy_seed=root_seed,
            alpha=args.alternative_alpha,
            args=args,
        )
        required_cache = ("acot_prefix_feature", "acot_iar_tokens", "coarse_actions")
        missing = [name for name in required_cache if name not in alpha0_result]
        if missing:
            raise KeyError(f"Alpha=0 response is missing deployable gate inputs: {missing}.")
        if "coarse_actions" not in alternative_result:
            raise KeyError("Alternative-alpha response is missing coarse_actions.")

        environment_horizon = libero_eval._env_horizon(env)
        episode_step_limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        if environment_horizon is not None:
            episode_step_limit = min(episode_step_limit, environment_horizon)
        alpha0 = _run_arm(
            env,
            snapshot,
            alpha0_result["actions"],
            root_seed=root_seed,
            root_step=root_step,
            episode_step_limit=episode_step_limit,
            task_description=task_description,
            args=args,
            client=client,
        )
        alternative = _run_arm(
            env,
            snapshot,
            alternative_result["actions"],
            root_seed=root_seed,
            root_step=root_step,
            episode_step_limit=episode_step_limit,
            task_description=task_description,
            args=args,
            client=client,
        )
        preference_label, preference_reason = _preference(
            alpha0,
            alternative,
            terminal=args.terminal,
        )
        record = _record_arrays(
            args=args,
            arrays=arrays,
            index=index,
            physics_key=physics_key,
            element=element,
            alpha0_result=alpha0_result,
            alternative_result=alternative_result,
            alpha0=alpha0,
            alternative=alternative,
            preference_label=preference_label,
            preference_reason=preference_reason,
        )
        progress_collector._write_npz_atomic(output_path, record)
        return _index_row(output_path, output_path.parent.parent)
    finally:
        libero_eval._safe_close_env(env)


def _collect_live_pair(
    *,
    client: websocket_policy.WebsocketClientPolicy,
    env: Any,
    task_description: str,
    root: LiveRoot,
    episode_step_limit: int,
    behavior_success: bool,
    behavior_calls: int,
    args: argparse.Namespace,
    output_path: pathlib.Path,
    physics_key: str,
) -> dict[str, Any]:
    """Label one canonical live root with two first actions and shared pi continuation."""

    observation = state_branches._restore_canonical_snapshot(env, root.snapshot)
    element = _policy_input(observation, task_description, args)
    alpha0_result = _request(
        client,
        element,
        policy_seed=root.root_seed,
        alpha=0.0,
        args=args,
        export_cache=True,
        profile_policy_timing=True,
        force_final_one_step=True,
    )
    alternative_result = _request(
        client,
        element,
        policy_seed=root.root_seed,
        alpha=args.alternative_alpha,
        args=args,
        profile_policy_timing=True,
        force_final_one_step=True,
    )
    required_cache = ("acot_prefix_feature", "acot_iar_tokens", "coarse_actions")
    missing = [name for name in required_cache if name not in alpha0_result]
    if missing:
        raise KeyError(f"Alpha=0 response is missing deployable gate inputs: {missing}.")
    if "coarse_actions" not in alternative_result:
        raise KeyError("Alternative-alpha response is missing coarse_actions.")

    alpha0 = _run_arm(
        env,
        root.snapshot,
        alpha0_result["actions"],
        root_seed=root.root_seed,
        root_step=root.decision_step,
        episode_step_limit=episode_step_limit,
        task_description=task_description,
        args=args,
        client=client,
        continuation_policy="current_router",
        profile_policy_timing=True,
    )
    alternative = _run_arm(
        env,
        root.snapshot,
        alternative_result["actions"],
        root_seed=root.root_seed,
        root_step=root.decision_step,
        episode_step_limit=episode_step_limit,
        task_description=task_description,
        args=args,
        client=client,
        continuation_policy="current_router",
        profile_policy_timing=True,
    )
    preference_label, preference_reason = _preference(
        alpha0,
        alternative,
        terminal=args.terminal,
    )
    arrays = {
        "episode_id": np.asarray([root.episode_id], dtype=np.int32),
        "decision_step": np.asarray([root.decision_step], dtype=np.int32),
        "root_seed": np.asarray([root.root_seed], dtype=np.uint32),
        "source_iteration": np.asarray([1], dtype=np.int16),
        "physics_state": np.asarray(
            [root.snapshot.simulator.physics_state],
            dtype=np.float64,
        ),
    }
    record = _record_arrays(
        args=args,
        arrays=arrays,
        index=0,
        physics_key=physics_key,
        element=element,
        alpha0_result=alpha0_result,
        alternative_result=alternative_result,
        alpha0=alpha0,
        alternative=alternative,
        preference_label=preference_label,
        preference_reason=preference_reason,
    )
    record.update(
        {
            "behavior_policy": np.asarray("current_compact_alpha_router"),
            "continuation_policy": np.asarray("current_compact_alpha_router"),
            "behavior_call_index": np.asarray(root.call_index, dtype=np.int16),
            "behavior_router_score": np.asarray(root.router_score, dtype=np.float32),
            "behavior_router_selected_alpha": np.asarray(root.router_alpha, dtype=np.float32),
            "behavior_rollout_success": np.asarray(behavior_success, dtype=np.bool_),
            "behavior_rollout_calls": np.asarray(behavior_calls, dtype=np.int16),
        }
    )
    progress_collector._write_npz_atomic(output_path, record)
    return _index_row(output_path, output_path.parent.parent)


def _live_root_indices(num_calls: int, roots_per_episode: int) -> list[int]:
    if num_calls <= 0:
        return []
    count = min(num_calls, roots_per_episode)
    return sorted(
        set(
            int(value)
            for value in np.rint(np.linspace(0, num_calls - 1, num=count)).tolist()
        )
    )


def _collect_live_episode(
    *,
    client: websocket_policy.WebsocketClientPolicy,
    task_suite: Any,
    episode_id: int,
    args: argparse.Namespace,
    root_dir: pathlib.Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task = task_suite.get_task(TASK_ID)
    initial_states = task_suite.get_task_init_states(TASK_ID)
    env, task_description = libero_eval._get_libero_env(
        task,
        libero_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    try:
        env.reset()
        observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
        environment_horizon = libero_eval._env_horizon(env)
        episode_step_limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        if environment_horizon is not None:
            episode_step_limit = min(episode_step_limit, environment_horizon)
        step = 0
        done = False
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                break

        live_roots: list[LiveRoot] = []
        while not done and not libero_eval._env_success(env) and step < episode_step_limit:
            root_seed = state_branches.canonical_policy_seed(
                args.seed,
                TASK_ID,
                episode_id,
                step,
            )
            snapshot = state_branches._capture_canonical_snapshot(env)
            element = _policy_input(observation, task_description, args)
            result = _request(
                client,
                element,
                policy_seed=root_seed,
                alpha=0.0,
                args=args,
                compact_router=True,
                absolute_step=step,
                profile_policy_timing=True,
            )
            required = {
                "compact_alpha_router_score",
                "compact_alpha_router_selected_alpha",
            }
            missing = sorted(required.difference(result))
            if missing:
                raise KeyError(f"Compact-router behavior response is missing: {missing}.")
            router_score = float(np.asarray(result["compact_alpha_router_score"]).item())
            router_alpha = float(
                np.asarray(result["compact_alpha_router_selected_alpha"]).item()
            )
            if not np.isfinite(router_score):
                raise ValueError("Compact-router behavior score must be finite.")
            if not (
                np.isclose(router_alpha, 0.0)
                or np.isclose(router_alpha, args.alternative_alpha)
            ):
                raise ValueError(f"Unexpected compact-router alpha: {router_alpha}.")
            router_alpha = args.alternative_alpha if router_alpha > 0.0 else 0.0
            live_roots.append(
                LiveRoot(
                    episode_id=episode_id,
                    decision_step=step,
                    root_seed=root_seed,
                    call_index=len(live_roots),
                    snapshot=snapshot,
                    router_score=router_score,
                    router_alpha=router_alpha,
                )
            )
            actions = np.asarray(result["actions"], dtype=np.float32)[
                :ACTION_HORIZON, :ACTION_DIM
            ]
            observation, done, executed = state_branches._step_actions(
                env,
                observation,
                actions,
            )
            if not executed:
                break
            step += len(executed)

        behavior_success = bool(libero_eval._env_success(env))
        selected = [
            live_roots[index]
            for index in _live_root_indices(
                len(live_roots),
                args.live_roots_per_episode,
            )
        ]
        rows: list[dict[str, Any]] = []
        for root in selected:
            physics_key = _physics_key(root.snapshot.simulator.physics_state)
            output_path = _root_path(
                root_dir,
                episode_id,
                root.decision_step,
                physics_key,
            )
            row = _index_row(output_path, root_dir.parent) if output_path.exists() else None
            if row is None or (args.terminal and not row["terminal_evaluated"]):
                row = _collect_live_pair(
                    client=client,
                    env=env,
                    task_description=task_description,
                    root=root,
                    episode_step_limit=episode_step_limit,
                    behavior_success=behavior_success,
                    behavior_calls=len(live_roots),
                    args=args,
                    output_path=output_path,
                    physics_key=physics_key,
                )
            rows.append(row)
        episode_row = {
            "episode_id": episode_id,
            "initial_state_id": episode_id % len(initial_states),
            "success": behavior_success,
            "behavior_calls": len(live_roots),
            "final_environment_step": step,
            "selected_root_steps": [root.decision_step for root in selected],
            "selected_root_alphas": [root.router_alpha for root in selected],
            "selected_root_scores": [root.router_score for root in selected],
        }
        return rows, episode_row
    finally:
        libero_eval._safe_close_env(env)


def _summary(
    args: argparse.Namespace,
    records: Sequence[dict[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    preferences = [int(row["preference_label"]) for row in records]
    h10 = [float(row["alternative_minus_alpha0_h10_progress"]) for row in records]
    h20 = [float(row["alternative_minus_alpha0_h20_progress"]) for row in records]
    terminal = [int(row["terminal_success_difference"]) for row in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "elapsed_seconds": elapsed_seconds,
        "num_roots": len(records),
        "protocol": {
            "task_id": TASK_ID,
            "episode_ids": list(args.episode_ids),
            "formal_episodes_allowed": bool(args.allow_formal_episodes),
            "formal_episode_ids": sorted(FORMAL_EPISODES),
            "root_source": "saved counterfactual physics_state",
            "initial_arms": [0.0, args.alternative_alpha],
            "initial_horizon": ACTION_HORIZON,
            "continuation": "shared alpha=0 H10 policy with absolute-step seeds",
            "terminal_evaluated": bool(args.terminal),
            "label_priority": (
                "terminal success, both-success calls/steps, terminal progress"
                if args.terminal
                else "H20 privileged progress, then H10 privileged progress"
            ),
            "policy_seed": "same stored root_seed for alpha=0 and alternative alpha",
            "action_cot_denoising_steps": args.action_cot_denoising_steps,
            "resume_authority": "one compressed NPZ per unique physics root",
        },
        "preference": {
            "alternative_wins": int(sum(value > 0 for value in preferences)),
            "alpha0_wins": int(sum(value < 0 for value in preferences)),
            "ties": int(sum(value == 0 for value in preferences)),
        },
        "h10_progress": {
            "alternative_wins": int(sum(value > 0 for value in h10)),
            "alpha0_wins": int(sum(value < 0 for value in h10)),
            "ties": int(sum(value == 0 for value in h10)),
            "mean_advantage": float(np.mean(h10)) if h10 else None,
        },
        "h20_progress": {
            "alternative_wins": int(sum(value > 0 for value in h20)),
            "alpha0_wins": int(sum(value < 0 for value in h20)),
            "ties": int(sum(value == 0 for value in h20)),
            "mean_advantage": float(np.mean(h20)) if h20 else None,
        },
        "terminal_success": {
            "alternative_rescues": int(sum(value > 0 for value in terminal)) if args.terminal else None,
            "alternative_regressions": int(sum(value < 0 for value in terminal)) if args.terminal else None,
            "ties": int(sum(value == 0 for value in terminal)) if args.terminal else None,
        },
        "mean_action_rmse": (
            float(np.mean([float(row["action_rmse"]) for row in records]))
            if records
            else None
        ),
        "max_coarse_abs_difference": (
            float(max(float(row["coarse_max_abs_difference"]) for row in records))
            if records
            else None
        ),
        "records": list(records),
    }


def _main_live(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir)
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    episodes_path = output_dir / "behavior_episodes.jsonl"
    summary_path = output_dir / "summary.json"
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    started = time.monotonic()
    with (
        records_path.open("w", encoding="utf-8") as records_writer,
        episodes_path.open("w", encoding="utf-8") as episodes_writer,
    ):
        for episode_id in args.episode_ids:
            episode_records, episode_row = _collect_live_episode(
                client=client,
                task_suite=task_suite,
                episode_id=episode_id,
                args=args,
                root_dir=root_dir,
            )
            for row in episode_records:
                records.append(row)
                records_writer.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                records_writer.flush()
                print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            episodes.append(episode_row)
            episodes_writer.write(
                json.dumps(episode_row, sort_keys=True, allow_nan=False) + "\n"
            )
            episodes_writer.flush()
            print(json.dumps(episode_row, sort_keys=True, allow_nan=False), flush=True)

    summary = _summary(
        args,
        records,
        elapsed_seconds=time.monotonic() - started,
    )
    summary["protocol"].update(
        {
            "root_source": "live current compact-router canonical simulator snapshots",
            "behavior_policy": "current compact alpha router, H10",
            "continuation": "shared current compact alpha router with absolute-step seeds",
            "q_semantics": (
                "candidate alpha=0/.05 first chunk followed by frozen current-router pi"
            ),
            "live_roots_per_episode": args.live_roots_per_episode,
            "live_root_selection": "evenly-spaced early/middle/late behavior calls",
            "profile_policy_timing": True,
            "h1_h10_branch_collection": False,
        }
    )
    summary["behavior"] = {
        "episodes": episodes,
        "successes": int(sum(bool(row["success"]) for row in episodes)),
        "num_episodes": len(episodes),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    summary_path.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    if args.live_on_policy:
        _main_live(args)
        return
    output_dir = pathlib.Path(args.output_dir)
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset, include_physics=True)
    indices = _selected_indices(arrays, args)
    if not indices:
        raise ValueError("No unique Task8 saved roots matched the requested development episodes.")

    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with records_path.open("w", encoding="utf-8") as writer:
        for index in indices:
            episode_id = int(arrays["episode_id"][index])
            decision_step = int(arrays["decision_step"][index])
            physics_key = _physics_key(arrays["physics_state"][index])
            output_path = _root_path(root_dir, episode_id, decision_step, physics_key)
            row = _index_row(output_path, output_dir) if output_path.exists() else None
            if row is None or (args.terminal and not row["terminal_evaluated"]):
                row = _collect_root(
                    client=client,
                    task_suite=task_suite,
                    arrays=arrays,
                    index=index,
                    args=args,
                    output_path=output_path,
                    physics_key=physics_key,
                )
            records.append(row)
            writer.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            writer.flush()
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    summary = _summary(
        args,
        records,
        elapsed_seconds=time.monotonic() - started,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    summary_path.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
