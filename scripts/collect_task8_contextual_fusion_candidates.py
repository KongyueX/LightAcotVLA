"""Collect same-root outcome labels for phase-aware contextual chunk routing.

For every selected non-formal Task8 root, this collector asks one loaded policy
server for two deterministic chunks with the same policy seed:

* ``compiler``: the cheap contextual Action-CoT compiler draft;
* ``expert``: the B6 one-step final-action expert.

It then finds one action-dependent phase boundary at the lowest shared control
speed and constructs four candidates: compiler, expert, compiler->expert, and
expert->compiler.  The two phase candidates cross-fade the six continuous
controls for one step while switching the gripper discretely.  Every candidate
is executed from the exact same MuJoCo snapshot and followed by one frozen
continuation policy.  Terminal success, remaining calls/steps, and privileged
task progress are stored separately so a router can learn a lexicographic
Q^pi ranking without using privileged inputs at deployment time.

Task8 initial-state IDs 10-29 are reserved for formal evaluation and are
rejected.  One compressed NPZ per root is the resume authority.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import time
from typing import Any

import collect_action_cot_state_branches as state_branches
import collect_libero_plan_refresh_benefit as progress_collector
import collect_task8_alpha_snapshot_pairs as pair_collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy


SCHEMA_VERSION = 1
TASK_ID = 8
FORMAL_EPISODES = frozenset(range(10, 30))
ACTION_HORIZON = 10
ACTION_DIM = 7
CONTROL_DIM = 6
IMAGE_SIZE = 64
CANDIDATE_NAMES = ("compiler", "expert", "compiler_to_expert", "expert_to_compiler")


@dataclasses.dataclass(frozen=True)
class Root:
    episode_id: int
    decision_step: int
    policy_seed: int
    call_index: int
    snapshot: state_branches.CanonicalSimulatorSnapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--episode-ids", nargs="+", type=int, default=list(range(30, 50)))
    parser.add_argument("--roots-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=1)
    parser.add_argument(
        "--behavior-mode",
        choices=("compiler", "expert"),
        default="compiler",
    )
    parser.add_argument(
        "--continuation-mode",
        choices=("compiler", "expert"),
        default="compiler",
    )
    parser.add_argument(
        "--terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.episode_ids or any(episode < 0 for episode in args.episode_ids):
        raise ValueError("--episode-ids must contain unique non-negative IDs.")
    if len(set(args.episode_ids)) != len(args.episode_ids):
        raise ValueError("--episode-ids must not contain duplicates.")
    forbidden = sorted(set(args.episode_ids) & FORMAL_EPISODES)
    if forbidden:
        raise ValueError(f"Formal Task8 episode IDs 10-29 are forbidden: {forbidden}.")
    if args.roots_per_episode <= 0:
        raise ValueError("--roots-per-episode must be positive.")
    if args.seed < 0 or args.num_steps_wait < 0:
        raise ValueError("--seed and --num-steps-wait must be non-negative.")
    if args.resize_size <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("Image size and Action-CoT NFE must be positive.")


def _policy_input(observation: dict[str, Any], task_description: str, args: argparse.Namespace) -> dict[str, Any]:
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
    fusion_mode: str,
    args: argparse.Namespace,
    export_cache: bool = False,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        **element,
        "policy_seed": np.asarray(policy_seed, dtype=np.int64),
        "profile_policy_timing": np.asarray(True, dtype=np.bool_),
        "action_cot_denoising_steps": np.asarray(
            args.action_cot_denoising_steps,
            dtype=np.int32,
        ),
    }
    if fusion_mode != "compiler":
        request["action_cot_contextual_fusion_mode"] = fusion_mode
    if export_cache:
        request["export_acot_cache"] = np.asarray(True, dtype=np.bool_)
    started = time.perf_counter()
    result = client.infer(request)
    result["collector_wall_ms"] = (time.perf_counter() - started) * 1000.0
    if "actions" not in result:
        raise KeyError("Policy response does not contain actions.")
    return result


def _progress_arrays(prefix: str, progress: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_score": np.asarray(progress["score"], dtype=np.float32),
        f"{prefix}_normalized_score": np.asarray(
            progress["normalized_score"], dtype=np.float32
        ),
        f"{prefix}_satisfied_count": np.asarray(progress["satisfied_count"], dtype=np.int16),
        f"{prefix}_total_goals": np.asarray(progress["total_goals"], dtype=np.int16),
        f"{prefix}_active_progress": np.asarray(progress["active_progress"], dtype=np.float32),
        f"{prefix}_success": np.asarray(progress["success"], dtype=np.bool_),
    }


def _run_arm(
    env: Any,
    root: Root,
    initial_actions: np.ndarray,
    *,
    episode_step_limit: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> dict[str, Any]:
    observation = state_branches._restore_canonical_snapshot(env, root.snapshot)
    before = progress_collector._progress(env, observation)
    observation, done, executed = state_branches._step_actions(
        env,
        observation,
        np.asarray(initial_actions, dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM],
    )
    absolute_step = root.decision_step + len(executed)
    after_h10 = progress_collector._progress(env, observation)
    continuation_calls = 0
    continuation_steps = 0
    continuation_wall_ms = 0.0
    no_progress = False

    while (
        not done
        and not libero_eval._env_success(env)
        and absolute_step < episode_step_limit
        and (args.terminal or continuation_calls < 1)
    ):
        policy_seed = root.policy_seed + (absolute_step - root.decision_step)
        element = _policy_input(observation, task_description, args)
        result = _request(
            client,
            element,
            policy_seed=policy_seed,
            fusion_mode=args.continuation_mode,
            args=args,
        )
        actions = np.asarray(result["actions"], dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM]
        observation, done, executed = state_branches._step_actions(env, observation, actions)
        executed_count = len(executed)
        continuation_calls += 1
        continuation_steps += executed_count
        continuation_wall_ms += float(result["collector_wall_ms"])
        absolute_step += executed_count
        if executed_count == 0:
            no_progress = True
            break

    terminal = progress_collector._progress(env, observation)
    terminal_success = bool(terminal["success"])
    if not args.terminal:
        terminal_reason = "not_evaluated"
    elif terminal_success:
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
        "terminal": terminal,
        "terminal_success": terminal_success,
        "terminal_reason": terminal_reason,
        "absolute_step": absolute_step,
        "continuation_calls": continuation_calls,
        "continuation_steps": continuation_steps,
        "continuation_wall_ms": continuation_wall_ms,
    }


def _phase_boundary(compiler: np.ndarray, expert: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Return a PACE-style internal boundary at the shared speed valley."""

    shared_speed = 0.5 * (
        np.linalg.norm(compiler[:, :CONTROL_DIM], axis=-1)
        + np.linalg.norm(expert[:, :CONTROL_DIM], axis=-1)
    )
    residual = expert[:, :CONTROL_DIM] - compiler[:, :CONTROL_DIM]
    residual_jump = np.zeros((ACTION_HORIZON,), dtype=np.float32)
    residual_jump[1:] = np.linalg.norm(np.diff(residual, axis=0), axis=-1)
    internal = np.arange(2, ACTION_HORIZON - 1, dtype=np.int32)
    minimum_speed = np.min(shared_speed[internal])
    tied = internal[np.isclose(shared_speed[internal], minimum_speed, rtol=1e-5, atol=1e-7)]
    boundary = int(tied[np.argmax(residual_jump[tied])])
    return boundary, shared_speed.astype(np.float32), residual_jump


def _phase_candidate(source: np.ndarray, destination: np.ndarray, boundary: int) -> np.ndarray:
    actions = np.asarray(source, dtype=np.float32).copy()
    actions[boundary:] = np.asarray(destination, dtype=np.float32)[boundary:]
    actions[boundary, :CONTROL_DIM] = 0.5 * (
        source[boundary, :CONTROL_DIM] + destination[boundary, :CONTROL_DIM]
    )
    actions[boundary, CONTROL_DIM] = destination[boundary, CONTROL_DIM]
    return actions


def _candidates(compiler: np.ndarray, expert: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    boundary, speed, residual_jump = _phase_boundary(compiler, expert)
    actions = np.stack(
        (
            compiler,
            expert,
            _phase_candidate(compiler, expert, boundary),
            _phase_candidate(expert, compiler, boundary),
        ),
        axis=0,
    ).astype(np.float32)
    return boundary, speed, residual_jump, actions


def _ranking_key(arm: dict[str, Any]) -> tuple[float, float, float, float]:
    success = float(bool(arm["terminal_success"]))
    return (
        success,
        -float(arm["continuation_calls"]) if success else 0.0,
        -float(arm["absolute_step"]) if success else 0.0,
        float(arm["terminal"]["normalized_score"]),
    )


def _rank_arms(arms: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    keys = [_ranking_key(arm) for arm in arms]
    order = np.asarray(sorted(range(len(arms)), key=lambda index: keys[index], reverse=True), dtype=np.int8)
    ranks = np.empty((len(arms),), dtype=np.int8)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return order, ranks


def _physics_key(snapshot: state_branches.CanonicalSimulatorSnapshot) -> str:
    payload = np.asarray(snapshot.simulator.physics_state, dtype=np.float64).tobytes()
    return hashlib.sha1(payload).hexdigest()


def _root_path(root_dir: pathlib.Path, root: Root) -> pathlib.Path:
    key = _physics_key(root.snapshot)
    return root_dir / (
        f"task{TASK_ID:02d}_episode{root.episode_id:03d}_"
        f"step{root.decision_step:04d}_{key[:12]}.npz"
    )


def _scalar(data: Any, name: str) -> Any:
    return np.asarray(data[name]).reshape(()).item()


def _index_row(path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "episode_id": int(_scalar(data, "episode_id")),
            "decision_step": int(_scalar(data, "decision_step")),
            "phase_boundary": int(_scalar(data, "phase_boundary")),
            "best_candidate": str(data["candidate_names"][int(data["candidate_order"][0])]),
            "candidate_terminal_success": data["candidate_terminal_success"].astype(int).tolist(),
            "candidate_continuation_calls": data["candidate_continuation_calls"].astype(int).tolist(),
            "candidate_terminal_progress": data["candidate_terminal_normalized_score"].astype(float).tolist(),
            "outcome_discordant": bool(np.ptp(data["candidate_terminal_success"].astype(np.int8))),
            "npz_file": str(path.relative_to(output_dir)),
        }


def _collect_root(
    *,
    client: websocket_policy.WebsocketClientPolicy,
    env: Any,
    task_description: str,
    root: Root,
    episode_step_limit: int,
    args: argparse.Namespace,
    output_path: pathlib.Path,
) -> dict[str, Any]:
    observation = state_branches._restore_canonical_snapshot(env, root.snapshot)
    element = _policy_input(observation, task_description, args)
    compiler_result = _request(
        client,
        element,
        policy_seed=root.policy_seed,
        fusion_mode="compiler",
        args=args,
        export_cache=True,
    )
    expert_result = _request(
        client,
        element,
        policy_seed=root.policy_seed,
        fusion_mode="expert",
        args=args,
    )
    required = ("coarse_actions", "acot_prefix_feature", "acot_iar_tokens")
    missing = [name for name in required if name not in compiler_result]
    if missing:
        raise KeyError(f"Compiler response is missing router inputs: {missing}.")
    compiler = np.asarray(compiler_result["actions"], dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM]
    expert = np.asarray(expert_result["actions"], dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM]
    boundary, speed, residual_jump, candidates = _candidates(compiler, expert)
    arms = [
        _run_arm(
            env,
            root,
            actions,
            episode_step_limit=episode_step_limit,
            task_description=task_description,
            args=args,
            client=client,
        )
        for actions in candidates
    ]
    order, ranks = _rank_arms(arms)
    candidate_terminal_success = np.asarray(
        [arm["terminal_success"] for arm in arms], dtype=np.bool_
    )
    record: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "task_id": np.asarray(TASK_ID, dtype=np.int16),
        "episode_id": np.asarray(root.episode_id, dtype=np.int32),
        "decision_step": np.asarray(root.decision_step, dtype=np.int32),
        "policy_seed": np.asarray(root.policy_seed, dtype=np.uint32),
        "call_index": np.asarray(root.call_index, dtype=np.int16),
        "physics_key": np.asarray(_physics_key(root.snapshot)),
        "physics_state": np.asarray(root.snapshot.simulator.physics_state, dtype=np.float64),
        "current_images": state_branches._images(element, IMAGE_SIZE),
        "normalized_state": np.asarray(
            compiler_result.get("execution_horizon_state_normalized", np.zeros((32,))),
            dtype=np.float32,
        ),
        "acot_prefix_feature": np.asarray(compiler_result["acot_prefix_feature"], dtype=np.float16),
        "acot_iar_tokens": np.asarray(compiler_result["acot_iar_tokens"], dtype=np.float16),
        "coarse_actions": np.asarray(compiler_result["coarse_actions"], dtype=np.float16),
        "compiler_actions": compiler,
        "expert_actions": expert,
        "candidate_names": np.asarray(CANDIDATE_NAMES),
        "candidate_actions": candidates,
        "phase_boundary": np.asarray(boundary, dtype=np.int8),
        "shared_speed": speed,
        "residual_jump": residual_jump,
        "translation_disagreement": np.asarray(
            np.sqrt(np.mean((expert[:, :3] - compiler[:, :3]) ** 2)), dtype=np.float32
        ),
        "rotation_disagreement": np.asarray(
            np.sqrt(np.mean((expert[:, 3:6] - compiler[:, 3:6]) ** 2)), dtype=np.float32
        ),
        "gripper_conflict_rate": np.asarray(
            np.mean(np.signbit(expert[:, 6]) != np.signbit(compiler[:, 6])), dtype=np.float32
        ),
        "candidate_order": order,
        "candidate_rank": ranks,
        "candidate_terminal_success": candidate_terminal_success,
        "candidate_terminal_absolute_step": np.asarray(
            [arm["absolute_step"] for arm in arms], dtype=np.int32
        ),
        "candidate_continuation_calls": np.asarray(
            [arm["continuation_calls"] for arm in arms], dtype=np.int16
        ),
        "candidate_continuation_steps": np.asarray(
            [arm["continuation_steps"] for arm in arms], dtype=np.int16
        ),
        "candidate_continuation_wall_ms": np.asarray(
            [arm["continuation_wall_ms"] for arm in arms], dtype=np.float32
        ),
        "candidate_terminal_normalized_score": np.asarray(
            [arm["terminal"]["normalized_score"] for arm in arms], dtype=np.float32
        ),
        "candidate_h10_normalized_score": np.asarray(
            [arm["after_h10"]["normalized_score"] for arm in arms], dtype=np.float32
        ),
        "candidate_terminal_reason": np.asarray([arm["terminal_reason"] for arm in arms]),
        "compiler_request_wall_ms": np.asarray(
            compiler_result["collector_wall_ms"], dtype=np.float32
        ),
        "expert_request_wall_ms": np.asarray(expert_result["collector_wall_ms"], dtype=np.float32),
    }
    for index, arm in enumerate(arms):
        prefix = CANDIDATE_NAMES[index]
        record.update(_progress_arrays(f"{prefix}_before", arm["before"]))
        record.update(_progress_arrays(f"{prefix}_h10", arm["after_h10"]))
        record.update(_progress_arrays(f"{prefix}_terminal", arm["terminal"]))
    progress_collector._write_npz_atomic(output_path, record)
    return _index_row(output_path, output_path.parent.parent)


def _collect_episode(
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
        task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed
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

        roots: list[Root] = []
        while not done and not libero_eval._env_success(env) and step < episode_step_limit:
            policy_seed = state_branches.canonical_policy_seed(
                args.seed, TASK_ID, episode_id, step
            )
            roots.append(
                Root(
                    episode_id=episode_id,
                    decision_step=step,
                    policy_seed=policy_seed,
                    call_index=len(roots),
                    snapshot=state_branches._capture_canonical_snapshot(env),
                )
            )
            element = _policy_input(observation, task_description, args)
            result = _request(
                client,
                element,
                policy_seed=policy_seed,
                fusion_mode=args.behavior_mode,
                args=args,
            )
            actions = np.asarray(result["actions"], dtype=np.float32)[:ACTION_HORIZON, :ACTION_DIM]
            observation, done, executed = state_branches._step_actions(env, observation, actions)
            if not executed:
                break
            step += len(executed)

        behavior_success = bool(libero_eval._env_success(env))
        selected = [
            roots[index]
            for index in pair_collector._live_root_indices(len(roots), args.roots_per_episode)
        ]
        rows: list[dict[str, Any]] = []
        for root in selected:
            output_path = _root_path(root_dir, root)
            row = (
                _index_row(output_path, root_dir.parent)
                if output_path.exists()
                else _collect_root(
                    client=client,
                    env=env,
                    task_description=task_description,
                    root=root,
                    episode_step_limit=episode_step_limit,
                    args=args,
                    output_path=output_path,
                )
            )
            rows.append(row)
        return rows, {
            "episode_id": episode_id,
            "initial_state_id": episode_id % len(initial_states),
            "behavior_success": behavior_success,
            "behavior_calls": len(roots),
            "final_environment_step": step,
            "selected_root_steps": [root.decision_step for root in selected],
        }
    finally:
        libero_eval._safe_close_env(env)


def _summary(args: argparse.Namespace, rows: list[dict[str, Any]], episodes: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    candidate_successes = np.asarray(
        [row["candidate_terminal_success"] for row in rows], dtype=np.int32
    )
    best_counts = collections.Counter(row["best_candidate"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "elapsed_seconds": elapsed,
        "protocol": {
            "task_id": TASK_ID,
            "episode_ids": list(args.episode_ids),
            "formal_episode_ids_forbidden": sorted(FORMAL_EPISODES),
            "roots_per_episode": args.roots_per_episode,
            "behavior_mode": args.behavior_mode,
            "continuation_mode": args.continuation_mode,
            "candidate_names": list(CANDIDATE_NAMES),
            "phase_boundary": "minimum shared compiler/expert control speed over internal steps 2..8",
            "phase_transition": "one-step continuous crossfade and hard gripper switch",
            "label_priority": "terminal success, successful remaining calls/steps, terminal progress",
            "same_root": True,
            "same_policy_seed": True,
        },
        "num_roots": len(rows),
        "outcome_discordant_roots": int(sum(row["outcome_discordant"] for row in rows)),
        "candidate_terminal_success_rate": {
            name: (float(np.mean(candidate_successes[:, index])) if len(rows) else None)
            for index, name in enumerate(CANDIDATE_NAMES)
        },
        "candidate_oracle_terminal_success_rate": (
            float(np.mean(np.max(candidate_successes, axis=1))) if len(rows) else None
        ),
        "best_candidate_counts": dict(sorted(best_counts.items())),
        "behavior": {
            "successes": int(sum(row["behavior_success"] for row in episodes)),
            "episodes": episodes,
        },
        "records": rows,
    }


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    root_dir = output_dir / "roots"
    root_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    records_path = output_dir / "records.jsonl"
    episodes_path = output_dir / "behavior_episodes.jsonl"
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    started = time.monotonic()
    with (
        records_path.open("w", encoding="utf-8") as records_writer,
        episodes_path.open("w", encoding="utf-8") as episodes_writer,
    ):
        for episode_id in args.episode_ids:
            episode_rows, episode = _collect_episode(
                client=client,
                task_suite=task_suite,
                episode_id=episode_id,
                args=args,
                root_dir=root_dir,
            )
            for row in episode_rows:
                rows.append(row)
                records_writer.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                records_writer.flush()
                print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            episodes.append(episode)
            episodes_writer.write(json.dumps(episode, sort_keys=True, allow_nan=False) + "\n")
            episodes_writer.flush()
            print(json.dumps(episode, sort_keys=True, allow_nan=False), flush=True)

    summary = _summary(args, rows, episodes, time.monotonic() - started)
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    (output_dir / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
