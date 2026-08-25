"""Collect multi-seed counterfactual execution-horizon labels in LIBERO."""
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
from openpi.execution_horizon import hierarchical
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
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit episode IDs; overrides the 0..num-trials-per-task-1 range.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=20)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--root-stride-calls", type=int, default=1)
    parser.add_argument(
        "--root-call-offset-cycle",
        type=int,
        default=1,
        help="Episode e starts collecting at policy call e modulo this value.",
    )
    parser.add_argument("--max-roots-per-episode", type=int, default=0)
    parser.add_argument("--records-per-shard", type=int, default=1024)
    parser.add_argument("--source-iteration", type=int, default=0)
    parser.add_argument("--candidate-horizons", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--reference-horizon", type=int, default=10)
    parser.add_argument("--model-action-horizon", type=int, default=10)
    parser.add_argument("--model-coarse-horizon", type=int, default=15)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--model-state-dim", type=int, default=32)
    parser.add_argument("--prefix-feature-dim", type=int, default=2048)
    parser.add_argument(
        "--prefix-token-count",
        type=int,
        default=0,
        help="Store this many already-computed prefix tokens per root; zero preserves the compact legacy input.",
    )
    parser.add_argument("--continuation-policy", choices=("fixed_h9", "current_student"), default="fixed_h9")
    parser.add_argument(
        "--branch-repeats",
        type=int,
        default=1,
        help="Number of continuation-policy seeds to evaluate for selected forced horizons.",
    )
    parser.add_argument(
        "--repeat-branch-horizons",
        nargs="+",
        type=int,
        default=None,
        help="Forced horizons that receive --branch-repeats trials; defaults to every candidate horizon.",
    )
    parser.add_argument(
        "--branch-repeat-seed-stride",
        type=int,
        default=20_000_000,
        help="Seed offset between repeated branches from the same simulator snapshot.",
    )
    parser.add_argument(
        "--student-mode",
        choices=("v2_distilled", "v2_value_refined", "hierarchical_transformer"),
        default="v2_value_refined",
    )
    parser.add_argument("--hierarchical-calibration-json", default=None)
    parser.add_argument("--long-success-noninferiority", type=float, default=0.01)
    parser.add_argument("--short-max-event-probability", type=float, default=0.20)
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
    profile: bool | None = None,
    run_student: bool = False,
    previous_actions: np.ndarray | None = None,
    previous_h: int = 1,
    budget_balance: float = 0.0,
    episode_progress: float = 0.0,
) -> dict[str, Any]:
    request = dict(observation)
    request["policy_seed"] = np.asarray(seed, dtype=np.int64)
    request["action_cot_denoising_steps"] = np.asarray(args.action_cot_denoising_steps, dtype=np.int32)
    # Branch continuation can contain hundreds of calls.  Only the root
    # teacher needs stage-synchronized timings; disabling it elsewhere keeps
    # labels identical while avoiding four host synchronization points/call.
    request["profile_policy_timing"] = np.asarray(teacher if profile is None else profile, dtype=np.bool_)
    if teacher:
        request["batched_mc_samples"] = np.asarray(args.teacher_samples, dtype=np.int32)
    if run_student:
        request["run_execution_horizon_predictor"] = np.asarray(1, dtype=np.bool_)
        request["execution_horizon_previous_actions"] = (
            np.asarray(previous_actions, dtype=np.float32)
            if previous_actions is not None
            else np.zeros((args.model_action_horizon, 7), dtype=np.float32)
        )
        request["execution_horizon_previous_h"] = np.asarray(previous_h, dtype=np.int32)
        request["execution_horizon_budget_balance"] = np.asarray(budget_balance, dtype=np.float32)
        request["execution_horizon_episode_progress"] = np.asarray(episode_progress, dtype=np.float32)
        request["execution_horizon_previous_valid"] = np.asarray(previous_actions is not None)
    if args.prefix_token_count and teacher:
        request["export_execution_horizon_prefix_tokens"] = np.asarray(1, dtype=np.bool_)
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
    if args.student_mode == "hierarchical_transformer":
        predictor_outputs = {
            name.removeprefix("execution_horizon_"): value
            for name, value in result.items()
            if name.startswith("execution_horizon_")
        }
        decision = hierarchical.select_horizon(
            predictor_outputs,
            calibration=args._hierarchical_calibration,
            config=hierarchical.HierarchicalSelectorConfig(
                success_noninferiority_margin=args.long_success_noninferiority,
                maximum_short_event_probability=args.short_max_event_probability,
                require_calibration_for_long_h=True,
            ),
        )
        return decision.selected_horizon, decision.selected_horizon

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
) -> tuple[bool, bool, int, int, float, list[np.ndarray]]:
    branch_started = time.perf_counter()
    observation = _restore_snapshot(env, snapshot)
    steps = 0
    calls = 1  # The primary root request is shared across all ten branches.
    done = False
    frames: list[np.ndarray] = []
    diagnostic_overhead = 0.0
    previous_actions: np.ndarray | None = np.asarray(primary_actions, dtype=np.float32)
    previous_h = forced_horizon
    budget_state = copy.deepcopy(root_budget_state)
    if args.continuation_policy == "current_student" and args.student_mode != "hierarchical_transformer":
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
                diagnostic_started = time.perf_counter()
                frame = _frame(observation)
                if frame is not None:
                    frames.append(frame)
                diagnostic_overhead += time.perf_counter() - diagnostic_started
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
    return (
        success,
        timeout,
        steps,
        calls,
        time.perf_counter() - branch_started - diagnostic_overhead,
        frames,
    )


def _root_record(
    *,
    result: dict[str, Any],
    risk: dict[str, np.ndarray | int],
    branches: list[list[dict[str, Any]]],
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
    shape: horizon_dataset.DatasetShape,
    reference_horizon: int,
) -> dict[str, Any]:
    final_mc = np.asarray(result["mc_actions_normalized"], dtype=np.float32)
    coarse_mc = np.asarray(result["mc_coarse_actions_normalized"], dtype=np.float32)
    event_index = int(risk["event_index"])
    short_candidates = tuple(
        horizon for horizon in shape.candidate_horizons if v2_min_horizon <= horizon <= reference_horizon
    )
    if not short_candidates:
        short_candidates = (reference_horizon,)
    raw_h = v2.event_horizon(event_index, short_candidates)

    num_candidates = shape.num_candidates
    trial_success = np.zeros((num_candidates, shape.max_trials), dtype=np.bool_)
    trial_timeout = np.zeros((num_candidates, shape.max_trials), dtype=np.bool_)
    trial_steps = np.zeros((num_candidates, shape.max_trials), dtype=np.uint16)
    trial_calls = np.zeros((num_candidates, shape.max_trials), dtype=np.uint16)
    trial_elapsed = np.full((num_candidates, shape.max_trials), np.nan, dtype=np.float32)
    trial_valid = np.zeros((num_candidates, shape.max_trials), dtype=np.bool_)
    for candidate_index, outcomes in enumerate(branches):
        if not outcomes or len(outcomes) > shape.max_trials:
            raise ValueError(
                f"Candidate {shape.candidate_horizons[candidate_index]} has {len(outcomes)} trials; "
                f"expected between 1 and {shape.max_trials}."
            )
        for repeat_index, outcome in enumerate(outcomes):
            trial_success[candidate_index, repeat_index] = bool(outcome["success"])
            trial_timeout[candidate_index, repeat_index] = bool(outcome["timeout"])
            trial_steps[candidate_index, repeat_index] = int(outcome["remaining_steps"])
            trial_calls[candidate_index, repeat_index] = int(outcome["remaining_calls"])
            trial_elapsed[candidate_index, repeat_index] = float(outcome["elapsed_seconds"])
            trial_valid[candidate_index, repeat_index] = True

    trial_count = np.sum(trial_valid, axis=-1, dtype=np.uint16)

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid_count = np.maximum(trial_count.astype(np.float32), 1.0)
        safe_values = np.where(trial_valid, values.astype(np.float32), 0.0)
        mean = np.sum(safe_values, axis=-1) / valid_count
        centered = np.where(trial_valid, safe_values - mean[:, None], 0.0)
        variance = np.sum(np.square(centered), axis=-1)
        variance /= np.maximum(valid_count - 1.0, 1.0)
        variance = np.where(trial_count > 1, variance, 0.0)
        return mean.astype(np.float32), variance.astype(np.float32)

    steps_mean, steps_variance = moments(trial_steps)
    calls_mean, calls_variance = moments(trial_calls)
    elapsed_mean, elapsed_variance = moments(trial_elapsed)
    reference_index = shape.candidate_horizons.index(reference_horizon)
    dangerous_long_count = np.zeros((num_candidates,), dtype=np.uint16)
    paired_trial_count = np.zeros((num_candidates,), dtype=np.uint16)
    for candidate_index, horizon in enumerate(shape.candidate_horizons):
        if horizon <= reference_horizon:
            continue
        paired = trial_valid[reference_index] & trial_valid[candidate_index]
        dangerous = paired & trial_success[reference_index] & ~trial_success[candidate_index]
        paired_trial_count[candidate_index] = int(np.sum(paired))
        dangerous_long_count[candidate_index] = int(np.sum(dangerous))

    hazard_event_count = np.zeros((shape.action_horizon,), dtype=np.uint16)
    hazard_at_risk_count = np.zeros((shape.action_horizon,), dtype=np.uint16)
    short_indices = [index for index, horizon in enumerate(shape.candidate_horizons) if horizon <= reference_horizon]
    for repeat_index in range(shape.max_trials):
        if not np.all(trial_valid[short_indices, repeat_index]):
            continue
        short_success = trial_success[short_indices, repeat_index]
        failed_positions = np.flatnonzero(~short_success)
        if failed_positions.size and np.any(short_success[failed_positions[0] + 1 :]):
            # Episode success is a noisy, non-local outcome. A pattern such as
            # H3=fail,H5=success cannot identify a monotone re-observation
            # boundary, so exclude it from survival supervision rather than
            # injecting a contradictory event time.
            continue
        event_step = (
            shape.candidate_horizons[short_indices[int(failed_positions[0])]] - 1 if failed_positions.size else None
        )
        observed_last_step = reference_horizon - 1 if event_step is None else event_step
        observed_last_step = min(observed_last_step, shape.action_horizon - 1)
        hazard_at_risk_count[: observed_last_step + 1] += 1
        if event_step is not None and event_step < shape.action_horizon:
            hazard_event_count[event_step] += 1

    record = {
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
        "risk_valid": np.ones((shape.action_horizon,), dtype=np.bool_),
        "hazard_event_count": hazard_event_count,
        "hazard_at_risk_count": hazard_at_risk_count,
        "raw_h": raw_h,
        "candidate_horizons": shape.candidate_horizons,
        "branch_success": trial_success[:, 0],
        "branch_timeout": trial_timeout[:, 0],
        "remaining_steps": trial_steps[:, 0],
        "remaining_calls": trial_calls[:, 0],
        "branch_valid": trial_valid[:, 0],
        "success_count": np.sum(trial_success & trial_valid, axis=-1, dtype=np.uint16),
        "timeout_count": np.sum(trial_timeout & trial_valid, axis=-1, dtype=np.uint16),
        "trial_count": trial_count,
        "remaining_steps_mean": steps_mean,
        "remaining_steps_variance": steps_variance,
        "remaining_calls_mean": calls_mean,
        "remaining_calls_variance": calls_variance,
        "elapsed_mean": elapsed_mean,
        "elapsed_variance": elapsed_variance,
        "trial_success": trial_success,
        "trial_timeout": trial_timeout,
        "trial_remaining_steps": trial_steps,
        "trial_remaining_calls": trial_calls,
        "trial_elapsed": trial_elapsed,
        "trial_valid": trial_valid,
        "dangerous_long_count": dangerous_long_count,
        "paired_trial_count": paired_trial_count,
        "physics_state": snapshot.physics_state,
        "task_id": task_id,
        "episode_id": episode_id,
        "decision_step": decision_step,
        "root_seed": root_seed,
        "source_iteration": source_iteration,
    }
    if shape.prefix_token_count:
        prefix_tokens = np.asarray(result["execution_horizon_prefix_tokens"], dtype=np.float32)
        prefix_mask = np.asarray(result["execution_horizon_prefix_mask"], dtype=np.bool_)
        if prefix_tokens.ndim != 2 or prefix_tokens.shape[-1] != shape.prefix_feature_dim:
            raise ValueError(
                f"Exported prefix tokens must have shape [tokens, prefix_feature_dim]; got {prefix_tokens.shape}."
            )
        if prefix_mask.shape != (prefix_tokens.shape[0],):
            raise ValueError(
                "Exported prefix mask must have one entry per prefix token; "
                f"got tokens={prefix_tokens.shape}, mask={prefix_mask.shape}."
            )
        stored_tokens = np.zeros(
            (shape.prefix_token_count, shape.prefix_feature_dim),
            dtype=np.float32,
        )
        stored_mask = np.zeros((shape.prefix_token_count,), dtype=np.bool_)
        copied = min(shape.prefix_token_count, prefix_tokens.shape[0])
        stored_tokens[:copied] = prefix_tokens[:copied]
        stored_mask[:copied] = prefix_mask[:copied]
        record["prefix_tokens"] = stored_tokens
        record["prefix_token_mask"] = stored_mask
    return record


def main(args: argparse.Namespace) -> None:
    if (
        args.root_stride_calls <= 0
        or args.root_call_offset_cycle <= 0
        or args.action_cot_denoising_steps <= 0
        or args.branch_repeats <= 0
        or args.branch_repeat_seed_stride <= 0
    ):
        raise ValueError(
            "root_stride_calls, root_call_offset_cycle, action_cot_denoising_steps, branch_repeats and "
            "branch_repeat_seed_stride must be positive."
        )
    candidate_horizons = tuple(sorted(set(args.candidate_horizons)))
    if not candidate_horizons or candidate_horizons[0] <= 0 or candidate_horizons[-1] > args.model_action_horizon:
        raise ValueError("candidate_horizons must be positive and no larger than model_action_horizon.")
    if args.reference_horizon not in candidate_horizons:
        raise ValueError("reference_horizon must be included in candidate_horizons.")
    repeated_horizons = sorted(
        set(candidate_horizons if args.repeat_branch_horizons is None else args.repeat_branch_horizons)
    )
    if not repeated_horizons or not set(repeated_horizons).issubset(candidate_horizons):
        raise ValueError("repeat_branch_horizons must be a non-empty subset of candidate_horizons.")
    shape = horizon_dataset.DatasetShape(
        prefix_feature_dim=args.prefix_feature_dim,
        state_dim=args.model_state_dim,
        action_dim=args.model_action_dim,
        coarse_horizon=args.model_coarse_horizon,
        action_horizon=args.model_action_horizon,
        candidate_horizons=candidate_horizons,
        max_trials=args.branch_repeats,
        prefix_token_count=args.prefix_token_count,
    )
    episode_ids = (
        list(range(args.num_trials_per_task)) if args.episode_ids is None else list(dict.fromkeys(args.episode_ids))
    )
    if not episode_ids or any(episode_id < 0 for episode_id in episode_ids):
        raise ValueError("episode_ids must contain non-negative values.")
    if args.continuation_policy == "current_student" and args.v2_budget_capacity <= 0:
        raise ValueError("v2_budget_capacity must be positive.")
    if args.student_mode == "hierarchical_transformer":
        if args.continuation_policy != "current_student":
            raise ValueError("hierarchical_transformer is meaningful only with current_student continuation.")
        if args.hierarchical_calibration_json is None:
            raise ValueError("hierarchical_transformer requires --hierarchical-calibration-json.")
        args._hierarchical_calibration = hierarchical.HierarchicalCalibration.load(args.hierarchical_calibration_json)
        if args._hierarchical_calibration.candidate_horizons != candidate_horizons:
            raise ValueError("Hierarchical calibration candidates must match collection candidate_horizons.")
        if max(args._hierarchical_calibration.candidate_horizons) > args.model_action_horizon:
            raise ValueError("Hierarchical calibration candidates exceed model_action_horizon.")
    else:
        args._hierarchical_calibration = None
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
        "root_call_offset_cycle": args.root_call_offset_cycle,
        "episode_ids": episode_ids,
        "branch_repeats": args.branch_repeats,
        "repeat_branch_horizons": repeated_horizons,
        "branch_repeat_seed_stride": args.branch_repeat_seed_stride,
        "branch_schedule": "repeat_interleaved_deterministic_permutation",
        "candidate_horizons": candidate_horizons,
        "reference_horizon": args.reference_horizon,
        "dataset_shape": dataclasses.asdict(shape),
        "risk_config": dataclasses.asdict(risk_config),
    }
    total_records = 0
    branch_successes = np.zeros((shape.num_candidates,), dtype=np.int64)
    repeated_branch_successes = np.zeros((shape.num_candidates,), dtype=np.int64)
    repeated_branch_trials = np.zeros((shape.num_candidates,), dtype=np.int64)
    debug_videos = 0
    started = time.monotonic()
    repeated_outcomes_path = output_dir / "repeated_branch_outcomes.jsonl"
    with contextlib.ExitStack() as stack:
        writer = stack.enter_context(
            horizon_dataset.ShardedCounterfactualWriter(
                output_dir,
                shape=shape,
                records_per_shard=args.records_per_shard,
                metadata=metadata,
            )
        )
        repeated_outcomes_writer = (
            stack.enter_context(repeated_outcomes_path.open("w", encoding="utf-8")) if args.branch_repeats > 1 else None
        )
        for task_id in range(args.task_start, task_end):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            for episode_id in episode_ids:
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
                    root_call_offset = episode_id % args.root_call_offset_cycle
                    roots_this_episode = 0
                    previous_actions_raw: np.ndarray | None = None
                    previous_actions_normalized = np.zeros((shape.action_horizon, shape.action_dim), dtype=np.float32)
                    previous_h = args.reference_horizon
                    budget_state = v2.EpisodeBudgetState(balance=min(args.v2_initial_budget, args.v2_budget_capacity))
                    while not done and step < episode_step_limit:
                        collect_root = (
                            decision_index >= root_call_offset
                            and (decision_index - root_call_offset) % args.root_stride_calls == 0
                        )
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
                            branch_rows: list[list[dict[str, Any]]] = [[] for _ in candidate_horizons]
                            # Interleave paired repeats and deterministically
                            # randomize H order within each repeat. This avoids
                            # making H20 elapsed labels systematically last on
                            # a warming/throttling policy server.
                            for repeat_index in range(args.branch_repeats):
                                scheduled_candidates = [
                                    index
                                    for index, horizon in enumerate(candidate_horizons)
                                    if repeat_index == 0 or horizon in repeated_horizons
                                ]
                                schedule_rng = np.random.default_rng(
                                    root_seed + repeat_index * args.branch_repeat_seed_stride + 17
                                )
                                for candidate_index in schedule_rng.permutation(scheduled_candidates):
                                    forced_horizon = candidate_horizons[candidate_index]
                                    branch_seed = root_seed + repeat_index * args.branch_repeat_seed_stride
                                    capture_video = repeat_index == 0 and debug_videos < args.debug_failure_videos
                                    (
                                        success,
                                        timeout,
                                        remaining_steps,
                                        remaining_calls,
                                        elapsed_seconds,
                                        frames,
                                    ) = _run_branch(
                                        env,
                                        snapshot,
                                        primary_actions,
                                        forced_horizon=forced_horizon,
                                        root_step=step,
                                        episode_step_limit=episode_step_limit,
                                        root_seed=branch_seed,
                                        task_description=task_description,
                                        args=args,
                                        client=client,
                                        root_budget_state=budget_state,
                                        capture_video=capture_video,
                                    )
                                    repeated_branch_successes[candidate_index] += int(success)
                                    repeated_branch_trials[candidate_index] += 1
                                    branch_rows[candidate_index].append(
                                        {
                                            "repeat_index": repeat_index,
                                            "policy_seed": branch_seed,
                                            "success": success,
                                            "timeout": timeout,
                                            "remaining_steps": remaining_steps,
                                            "remaining_calls": remaining_calls,
                                            "elapsed_seconds": elapsed_seconds,
                                        }
                                    )
                                    if repeat_index == 0:
                                        branch_successes[candidate_index] += int(success)
                                        if timeout and frames and debug_videos < args.debug_failure_videos:
                                            debug_dir.mkdir(parents=True, exist_ok=True)
                                            imageio.mimwrite(
                                                debug_dir
                                                / (f"task{task_id}_ep{episode_id}_step{step}_h{forced_horizon}.mp4"),
                                                frames,
                                                fps=10,
                                            )
                                            debug_videos += 1
                            repeated_outcomes = {
                                str(horizon): outcomes
                                for horizon, outcomes in zip(candidate_horizons, branch_rows, strict=True)
                            }
                            root_record = _root_record(
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
                                shape=shape,
                                reference_horizon=args.reference_horizon,
                            )
                            writer.append(root_record)
                            if repeated_outcomes_writer is not None:
                                repeated_outcomes_writer.write(
                                    json.dumps(
                                        {
                                            "schema_version": horizon_dataset.SCHEMA_VERSION,
                                            "task_id": task_id,
                                            "episode_id": episode_id,
                                            "decision_step": step,
                                            "root_seed": root_seed,
                                            "raw_h": int(root_record["raw_h"]),
                                            "continuation_policy": args.continuation_policy,
                                            "branch_repeats": args.branch_repeats,
                                            "candidate_horizons": candidate_horizons,
                                            "repeated_horizons": repeated_horizons,
                                            "outcomes_by_h": repeated_outcomes,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                repeated_outcomes_writer.flush()
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
                            previous_actions_normalized = np.zeros(
                                (shape.action_horizon, shape.action_dim), dtype=np.float32
                            )
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

    repeated_branch_rates = np.divide(
        repeated_branch_successes,
        repeated_branch_trials,
        out=np.zeros((shape.num_candidates,), dtype=np.float64),
        where=repeated_branch_trials > 0,
    )
    summary = {
        "status": "complete",
        "num_records": total_records,
        "teacher_samples": args.teacher_samples,
        "candidate_horizons": candidate_horizons,
        "branch_success_count_by_h": branch_successes.tolist(),
        "branch_success_rate_by_h": (branch_successes / max(total_records, 1)).tolist(),
        "repeated_branch_success_count_by_h": repeated_branch_successes.tolist(),
        "repeated_branch_trial_count_by_h": repeated_branch_trials.tolist(),
        "repeated_branch_success_rate_by_h": repeated_branch_rates.tolist(),
        "repeated_branch_outcomes_path": (str(repeated_outcomes_path) if args.branch_repeats > 1 else None),
        "debug_failure_videos": debug_videos,
        "elapsed_seconds": time.monotonic() - started,
        "metadata": metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
