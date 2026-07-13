"""Collect H=1..10 counterfactual execution-horizon labels in LIBERO."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import json
import pathlib
import time
from typing import Any

import eval_libero_action_cot_pruning as libero_eval
import imageio
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import v2


@dataclasses.dataclass
class SimulatorSnapshot:
    physics_state: np.ndarray
    scalar_attributes: list[tuple[Any, str, Any]]
    random_states: list[tuple[Any, str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=20)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--root-stride-calls", type=int, default=1)
    parser.add_argument("--max-roots-per-episode", type=int, default=0)
    parser.add_argument("--records-per-shard", type=int, default=1024)
    parser.add_argument("--source-iteration", type=int, default=0)
    parser.add_argument("--continuation-policy", choices=("fixed_h9", "current_student"), default="fixed_h9")
    parser.add_argument("--student-mode", choices=("v2_distilled", "v2_value_refined"), default="v2_value_refined")
    parser.add_argument("--student-candidates", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--v2-min-horizon", type=int, default=3)
    parser.add_argument("--v2-risk-threshold", type=float, default=1.5)
    parser.add_argument("--v2-final-weight", type=float, default=0.5)
    parser.add_argument("--v2-action-cot-weight", type=float, default=0.5)
    parser.add_argument("--v2-target-average-horizon", type=float, default=9.0)
    parser.add_argument("--v2-initial-budget", type=float, default=6.0)
    parser.add_argument("--v2-budget-capacity", type=float, default=12.0)
    parser.add_argument("--q-min-success-probability", type=float, default=0.90)
    parser.add_argument("--q-max-timeout-probability", type=float, default=0.20)
    parser.add_argument("--q-risk-slack-steps", type=int, default=0)
    parser.add_argument("--debug-failure-videos", type=int, default=3)
    parser.add_argument("--debug-video-stride", type=int, default=5)
    return parser


def _walk_env(env: Any) -> list[Any]:
    queue = [env]
    result = []
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        result.append(candidate)
        for name in ("env", "_env", "unwrapped"):
            try:
                child = getattr(candidate, name, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                queue.append(child)
    return result


def _simulator(env: Any) -> Any:
    for candidate in _walk_env(env):
        simulator = getattr(candidate, "sim", None)
        if simulator is not None and hasattr(simulator, "get_state"):
            return simulator
    raise AttributeError("Could not find a MuJoCo simulator in the LIBERO wrapper chain.")


def _capture_snapshot(env: Any) -> SimulatorSnapshot:
    simulator = _simulator(env)
    physics_state = np.asarray(simulator.get_state().flatten(), dtype=np.float64).copy()
    scalar_attributes: list[tuple[Any, str, Any]] = []
    random_states: list[tuple[Any, str, Any]] = []
    for candidate in _walk_env(env):
        for name in ("timestep", "_timestep", "done", "_done"):
            if hasattr(candidate, name):
                value = getattr(candidate, name)
                if np.asarray(value).size == 1:
                    scalar_attributes.append((candidate, name, copy.deepcopy(value)))
        for name in ("np_random", "_np_random"):
            generator = getattr(candidate, name, None)
            if hasattr(generator, "get_state"):
                random_states.append((candidate, name, copy.deepcopy(generator.get_state())))
            elif hasattr(generator, "bit_generator"):
                random_states.append((candidate, name, copy.deepcopy(generator.bit_generator.state)))
    return SimulatorSnapshot(physics_state, scalar_attributes, random_states)


def _restore_snapshot(env: Any, snapshot: SimulatorSnapshot) -> dict[str, Any]:
    simulator = _simulator(env)
    if hasattr(simulator, "set_state_from_flattened"):
        simulator.set_state_from_flattened(snapshot.physics_state)
    else:
        simulator.set_state(snapshot.physics_state)
    simulator.forward()
    for candidate, name, value in snapshot.scalar_attributes:
        with contextlib.suppress(Exception):
            setattr(candidate, name, copy.deepcopy(value))
    for candidate, name, state in snapshot.random_states:
        generator = getattr(candidate, name, None)
        try:
            if hasattr(generator, "set_state"):
                generator.set_state(copy.deepcopy(state))
            elif hasattr(generator, "bit_generator"):
                generator.bit_generator.state = copy.deepcopy(state)
        except Exception:
            pass

    for candidate in _walk_env(env):
        for method_name in ("_get_observations", "get_observations", "_get_observation"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    observation = method()
                    if isinstance(observation, dict) and "agentview_image" in observation:
                        return observation
                except Exception:
                    pass
    regenerate = getattr(env, "regenerate_obs_from_state", None)
    if callable(regenerate):
        observation = regenerate(snapshot.physics_state)
        for candidate, name, value in snapshot.scalar_attributes:
            with contextlib.suppress(Exception):
                setattr(candidate, name, copy.deepcopy(value))
        return observation
    raise RuntimeError("Could not regenerate a LIBERO observation after restoring physics state.")


def _policy_request(
    client: websocket_policy.WebsocketClientPolicy,
    observation: dict[str, Any],
    *,
    seed: int,
    args: argparse.Namespace,
    teacher: bool = False,
    run_student: bool = False,
    previous_actions: np.ndarray | None = None,
    previous_h: int = 1,
    budget_balance: float = 0.0,
    episode_progress: float = 0.0,
) -> dict[str, Any]:
    request = dict(observation)
    request["policy_seed"] = np.asarray(seed, dtype=np.int64)
    request["action_cot_denoising_steps"] = np.asarray(args.action_cot_denoising_steps, dtype=np.int32)
    request["profile_policy_timing"] = np.asarray(1, dtype=np.bool_)
    if teacher:
        request["batched_mc_samples"] = np.asarray(args.teacher_samples, dtype=np.int32)
    if run_student:
        request["run_execution_horizon_predictor"] = np.asarray(1, dtype=np.bool_)
        request["execution_horizon_previous_actions"] = (
            np.asarray(previous_actions, dtype=np.float32)
            if previous_actions is not None
            else np.zeros((10, 7), dtype=np.float32)
        )
        request["execution_horizon_previous_h"] = np.asarray(previous_h, dtype=np.int32)
        request["execution_horizon_budget_balance"] = np.asarray(budget_balance, dtype=np.float32)
        request["execution_horizon_episode_progress"] = np.asarray(episode_progress, dtype=np.float32)
        request["execution_horizon_previous_valid"] = np.asarray(previous_actions is not None)
    started = time.perf_counter()
    result = client.infer(request)
    result["collector_wall_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _student_horizon(
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    budget_state: v2.EpisodeBudgetState,
) -> tuple[int, int]:
    final_risk = np.asarray(result["execution_horizon_final_risk"], dtype=np.float64)
    action_cot_risk = np.asarray(result["execution_horizon_action_cot_risk"], dtype=np.float64)
    fused_risk = np.asarray(result["execution_horizon_fused_risk"], dtype=np.float64)
    risk_config = v2.V2RiskConfig(
        risk_threshold=args.v2_risk_threshold,
        final_weight=args.v2_final_weight,
        action_cot_weight=args.v2_action_cot_weight,
    )
    entropy_candidates = list(range(args.v2_min_horizon, 11))
    entropy_raw_horizon, _ = v2.distilled_raw_horizon(
        final_risk,
        action_cot_risk,
        fused_risk,
        candidates=entropy_candidates,
        config=risk_config,
    )
    candidates = sorted(set(args.student_candidates))
    raw_horizon = entropy_raw_horizon
    if args.student_mode == "v2_value_refined":
        raw_horizon, _ = v2.value_refined_raw_horizon(
            entropy_raw_horizon=entropy_raw_horizon,
            success_probability=_sigmoid(result["execution_horizon_success_logits"]),
            timeout_probability=_sigmoid(result["execution_horizon_timeout_logits"]),
            fused_risk=fused_risk,
            config=v2.ValueRefinementConfig(
                minimum_success_probability=args.q_min_success_probability,
                maximum_timeout_probability=args.q_max_timeout_probability,
                risk_threshold=args.v2_risk_threshold,
                risk_slack_steps=args.q_risk_slack_steps,
                candidates=tuple(candidates),
            ),
        )
    else:
        candidates = entropy_candidates
    final_horizon, _ = v2.apply_episode_budget(
        raw_horizon,
        candidates,
        config=v2.EpisodeBudgetConfig(
            target_average_horizon=args.v2_target_average_horizon,
            capacity=args.v2_budget_capacity,
        ),
        state=budget_state,
    )
    return raw_horizon, final_horizon


def _advance_forced_budget(horizon: int, args: argparse.Namespace, state: v2.EpisodeBudgetState) -> None:
    target = min(args.v2_target_average_horizon, max(args.student_candidates))
    state.balance = float(np.clip(state.balance + horizon - target, 0.0, args.v2_budget_capacity))
    state.decisions += 1
    state.horizon_sum += horizon
    state.interventions += int(horizon < max(args.student_candidates))


def _frame(observation: dict[str, Any]) -> np.ndarray | None:
    image = observation.get("agentview_image")
    return np.asarray(image)[::-1, ::-1] if image is not None else None


def _run_branch(
    env: Any,
    snapshot: SimulatorSnapshot,
    primary_actions: np.ndarray,
    *,
    forced_horizon: int,
    root_step: int,
    episode_step_limit: int,
    root_seed: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    root_budget_state: v2.EpisodeBudgetState,
    capture_video: bool,
) -> tuple[bool, bool, int, int, list[np.ndarray]]:
    observation = _restore_snapshot(env, snapshot)
    steps = 0
    calls = 1  # The primary root request is shared across all ten branches.
    done = False
    frames: list[np.ndarray] = []
    previous_actions: np.ndarray | None = np.asarray(primary_actions, dtype=np.float32)
    previous_h = forced_horizon
    budget_state = copy.deepcopy(root_budget_state)
    if args.continuation_policy == "current_student":
        _advance_forced_budget(forced_horizon, args, budget_state)

    action_plan = np.asarray(primary_actions)[:forced_horizon]
    continuation_index = 0
    while root_step + steps < episode_step_limit:
        for action in action_plan:
            if root_step + steps >= episode_step_limit:
                break
            try:
                observation, _, done, _ = env.step(np.asarray(action).tolist())
            except Exception as exc:
                if not libero_eval._is_terminated_episode_error(exc):
                    raise
                done = libero_eval._env_success(env)
                break
            steps += 1
            if capture_video and steps % args.debug_video_stride == 0:
                frame = _frame(observation)
                if frame is not None:
                    frames.append(frame)
            if done or libero_eval._env_success(env):
                done = True
                break
        if done or root_step + steps >= episode_step_limit:
            break

        continuation_seed = root_seed + 100_000 + continuation_index
        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        progress = np.clip((root_step + steps) / max(episode_step_limit, 1), 0.0, 1.0)
        use_student = args.continuation_policy == "current_student"
        result = _policy_request(
            client,
            policy_input,
            seed=continuation_seed,
            args=args,
            run_student=use_student,
            previous_actions=previous_actions,
            previous_h=previous_h,
            budget_balance=budget_state.balance / args.v2_budget_capacity,
            episode_progress=float(progress),
        )
        calls += 1
        action_plan = np.asarray(result["actions"], dtype=np.float32)
        if use_student:
            _, continuation_horizon = _student_horizon(result, args=args, budget_state=budget_state)
        else:
            continuation_horizon = 9
        continuation_horizon = min(continuation_horizon, len(action_plan))
        previous_actions = action_plan
        previous_h = continuation_horizon
        action_plan = action_plan[:continuation_horizon]
        continuation_index += 1

    success = bool(done or libero_eval._env_success(env))
    timeout = not success
    return success, timeout, steps, calls, frames


def _root_record(
    *,
    result: dict[str, Any],
    risk: dict[str, np.ndarray | int],
    branches: list[tuple[bool, bool, int, int]],
    snapshot: SimulatorSnapshot,
    task_id: int,
    episode_id: int,
    decision_step: int,
    root_seed: int,
    previous_actions_normalized: np.ndarray,
    previous_h: int,
    previous_valid: bool,
    budget_balance: float,
    episode_progress: float,
    source_iteration: int,
    v2_min_horizon: int,
) -> dict[str, Any]:
    final_mc = np.asarray(result["mc_actions_normalized"], dtype=np.float32)
    coarse_mc = np.asarray(result["mc_coarse_actions_normalized"], dtype=np.float32)
    event_index = int(risk["event_index"])
    raw_h = v2.event_horizon(event_index, range(v2_min_horizon, 11))
    return {
        "prefix_feature": np.asarray(result["execution_horizon_prefix_feature"], dtype=np.float32),
        "state": np.asarray(result["execution_horizon_state_normalized"], dtype=np.float32),
        "coarse_actions": coarse_mc[0],
        "final_actions": final_mc[0],
        "previous_actions": previous_actions_normalized,
        "previous_h": previous_h,
        "previous_valid": previous_valid,
        "budget_balance": budget_balance,
        "episode_progress": episode_progress,
        "final_risk": risk["final_risk"],
        "action_cot_risk": risk["action_cot_risk"],
        "fused_risk": risk["fused_risk"],
        "event_mask": risk["event_mask"],
        "risk_valid": np.ones((10,), dtype=np.bool_),
        "raw_h": raw_h,
        "branch_success": [branch[0] for branch in branches],
        "branch_timeout": [branch[1] for branch in branches],
        "remaining_steps": [branch[2] for branch in branches],
        "remaining_calls": [branch[3] for branch in branches],
        "branch_valid": np.ones((10,), dtype=np.bool_),
        "physics_state": snapshot.physics_state,
        "task_id": task_id,
        "episode_id": episode_id,
        "decision_step": decision_step,
        "root_seed": root_seed,
        "source_iteration": source_iteration,
    }


def main(args: argparse.Namespace) -> None:
    if args.root_stride_calls <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("root_stride_calls and action_cot_denoising_steps must be positive.")
    if args.continuation_policy == "current_student" and args.v2_budget_capacity <= 0:
        raise ValueError("v2_budget_capacity must be positive.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_failures"
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = libero_eval._max_steps(args.task_suite_name)
    task_end = (
        task_suite.n_tasks if args.max_tasks is None else min(task_suite.n_tasks, args.task_start + args.max_tasks)
    )
    risk_config = v2.V2RiskConfig(
        risk_threshold=args.v2_risk_threshold,
        final_weight=args.v2_final_weight,
        action_cot_weight=args.v2_action_cot_weight,
    )
    metadata = {
        "task_suite": args.task_suite_name,
        "teacher_samples": args.teacher_samples,
        "continuation_policy": args.continuation_policy,
        "student_mode": args.student_mode,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "source_iteration": args.source_iteration,
        "risk_config": dataclasses.asdict(risk_config),
    }
    total_records = 0
    branch_successes = np.zeros((10,), dtype=np.int64)
    debug_videos = 0
    started = time.monotonic()
    with horizon_dataset.ShardedCounterfactualWriter(
        output_dir,
        records_per_shard=args.records_per_shard,
        metadata=metadata,
    ) as writer:
        for task_id in range(args.task_start, task_end):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            for episode_id in range(args.num_trials_per_task):
                env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
                try:
                    env.reset()
                    observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
                    environment_horizon = libero_eval._env_horizon(env)
                    episode_step_limit = max_steps + args.num_steps_wait
                    if environment_horizon is not None:
                        episode_step_limit = min(episode_step_limit, environment_horizon)
                    step = 0
                    done = False
                    for _ in range(args.num_steps_wait):
                        observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
                        step += 1
                        if done:
                            break
                    decision_index = 0
                    roots_this_episode = 0
                    previous_actions_raw: np.ndarray | None = None
                    previous_actions_normalized = np.zeros((10, 32), dtype=np.float32)
                    previous_h = 10
                    budget_state = v2.EpisodeBudgetState(balance=min(args.v2_initial_budget, args.v2_budget_capacity))
                    while not done and step < episode_step_limit:
                        collect_root = decision_index % args.root_stride_calls == 0
                        if args.max_roots_per_episode and roots_this_episode >= args.max_roots_per_episode:
                            break
                        root_seed = args.seed + task_id * 1_000_000 + episode_id * 10_000 + step
                        policy_input = libero_eval._observation_to_policy_input(
                            observation, task_description, args.resize_size
                        )
                        progress = float(np.clip(step / max(episode_step_limit, 1), 0.0, 1.0))
                        use_student = args.continuation_policy == "current_student"
                        result = _policy_request(
                            client,
                            policy_input,
                            seed=root_seed,
                            args=args,
                            teacher=collect_root,
                            run_student=use_student,
                            previous_actions=previous_actions_raw,
                            previous_h=previous_h,
                            budget_balance=budget_state.balance / args.v2_budget_capacity,
                            episode_progress=progress,
                        )
                        primary_actions = np.asarray(result["actions"], dtype=np.float32)
                        if collect_root:
                            snapshot = _capture_snapshot(env)
                            risk = v2.risk_targets_from_normalized_mc(
                                result["mc_coarse_actions_normalized"],
                                result["mc_actions_normalized"],
                                config=risk_config,
                            )
                            branch_rows: list[tuple[bool, bool, int, int]] = []
                            for forced_horizon in range(1, 11):
                                capture_video = debug_videos < args.debug_failure_videos
                                success, timeout, remaining_steps, remaining_calls, frames = _run_branch(
                                    env,
                                    snapshot,
                                    primary_actions,
                                    forced_horizon=forced_horizon,
                                    root_step=step,
                                    episode_step_limit=episode_step_limit,
                                    root_seed=root_seed,
                                    task_description=task_description,
                                    args=args,
                                    client=client,
                                    root_budget_state=budget_state,
                                    capture_video=capture_video,
                                )
                                branch_rows.append((success, timeout, remaining_steps, remaining_calls))
                                branch_successes[forced_horizon - 1] += int(success)
                                if timeout and frames and debug_videos < args.debug_failure_videos:
                                    debug_dir.mkdir(parents=True, exist_ok=True)
                                    imageio.mimwrite(
                                        debug_dir / f"task{task_id}_ep{episode_id}_step{step}_h{forced_horizon}.mp4",
                                        frames,
                                        fps=10,
                                    )
                                    debug_videos += 1
                            writer.append(
                                _root_record(
                                    result=result,
                                    risk=risk,
                                    branches=branch_rows,
                                    snapshot=snapshot,
                                    task_id=task_id,
                                    episode_id=episode_id,
                                    decision_step=step,
                                    root_seed=root_seed,
                                    previous_actions_normalized=previous_actions_normalized,
                                    previous_h=previous_h,
                                    previous_valid=previous_actions_raw is not None,
                                    budget_balance=budget_state.balance / args.v2_budget_capacity,
                                    episode_progress=progress,
                                    source_iteration=args.source_iteration,
                                    v2_min_horizon=args.v2_min_horizon,
                                )
                            )
                            total_records += 1
                            roots_this_episode += 1
                            observation = _restore_snapshot(env, snapshot)

                        if use_student:
                            _, rollout_horizon = _student_horizon(result, args=args, budget_state=budget_state)
                        else:
                            rollout_horizon = 9
                        rollout_horizon = min(rollout_horizon, len(primary_actions))
                        for action in primary_actions[:rollout_horizon]:
                            if step >= episode_step_limit:
                                break
                            try:
                                observation, _, done, _ = env.step(np.asarray(action).tolist())
                            except Exception as exc:
                                if not libero_eval._is_terminated_episode_error(exc):
                                    raise
                                done = libero_eval._env_success(env)
                                break
                            step += 1
                            if done:
                                break
                        if "execution_horizon_final_actions_normalized" in result:
                            previous_actions_normalized = np.asarray(
                                result["execution_horizon_final_actions_normalized"], dtype=np.float32
                            )
                        elif "mc_actions_normalized" in result:
                            previous_actions_normalized = np.asarray(result["mc_actions_normalized"], dtype=np.float32)[
                                0
                            ]
                        else:
                            previous_actions_normalized = np.zeros((10, 32), dtype=np.float32)
                        previous_actions_raw = primary_actions
                        previous_h = rollout_horizon
                        decision_index += 1
                        print(
                            json.dumps(
                                {
                                    "task": task_id,
                                    "episode": episode_id,
                                    "step": step,
                                    "records": total_records,
                                    "last_h": rollout_horizon,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                finally:
                    libero_eval._safe_close_env(env)

    summary = {
        "status": "complete",
        "num_records": total_records,
        "teacher_samples": args.teacher_samples,
        "branch_success_count_by_h": branch_successes.tolist(),
        "branch_success_rate_by_h": (branch_successes / max(total_records, 1)).tolist(),
        "debug_failure_videos": debug_videos,
        "elapsed_seconds": time.monotonic() - started,
        "metadata": metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
