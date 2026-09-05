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
from openpi.execution_horizon import initial_states as horizon_initial_states
from openpi.execution_horizon import ordered
from openpi.execution_horizon import v2

ROOT_SEED_EPISODE_STRIDE = 10_000
LEGACY_ROOT_SEED_TASK_STRIDE = 1_000_000
LEGACY_ROOT_SEED_BRANCH_CONTINUATION_OFFSET = 100_000
ROOT_SEED_BRANCH_SCHEDULE_OFFSET = 17
LEGACY_ROOT_SEED_SCHEME = "affine_task_episode_step_uint32_v1"
LANED_ROOT_SEED_SCHEME = "affine_task_episode_repeat_schedule_continuation_lanes_uint32_v2"
MAX_POLICY_SEED = int(np.iinfo(np.uint32).max)
HIERARCHICAL_MODE = "hierarchical_transformer"
ORDERED_MODE = "ordered_transformer"


@dataclasses.dataclass
class SimulatorSnapshot:
    physics_state: np.ndarray
    scalar_attributes: list[tuple[Any, str, Any]]
    random_states: list[tuple[Any, str, Any]]


@dataclasses.dataclass
class TrajectoryRoot:
    snapshot: SimulatorSnapshot
    policy_input: dict[str, Any]
    result: dict[str, Any]
    primary_actions: np.ndarray
    previous_actions_raw: np.ndarray | None
    previous_actions_normalized: np.ndarray
    previous_h: int
    budget_state: v2.EpisodeBudgetState
    progress: float
    step: int
    root_seed: int
    decision_index: int


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
        "--initial-state-bank",
        default=None,
        help="Optional frozen initial-state bank; episode IDs are looked up exactly, never modulo the presets.",
    )
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
    parser.add_argument(
        "--root-seed-task-stride",
        type=int,
        default=LEGACY_ROOT_SEED_TASK_STRIDE,
        help=(
            "Task namespace stride in the deterministic root/policy seed scheme. The legacy 1,000,000 "
            "default is unchanged; use a larger opt-in value when collecting wide episode-ID ranges."
        ),
    )
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
    parser.add_argument(
        "--root-sampling",
        choices=("call_offset", "trajectory_reservoir"),
        default="call_offset",
        help="trajectory_reservoir samples one call uniformly from a complete current-student source rollout.",
    )
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
    parser.add_argument(
        "--continuation-policy",
        choices=("fixed_h9", "fixed_h", "current_student"),
        default="fixed_h9",
        help=(
            "Policy used after the one forced candidate-H chunk. fixed_h9 preserves the legacy protocol; "
            "fixed_h uses --fixed-continuation-horizon."
        ),
    )
    parser.add_argument(
        "--source-policy",
        choices=("match_continuation", "fixed_reference", "current_student"),
        default="match_continuation",
        help=(
            "Policy used to reach collection roots. match_continuation preserves the legacy behavior; "
            "current_student enables on-policy state collection while the counterfactual branches may "
            "still use a fixed continuation policy."
        ),
    )
    parser.add_argument(
        "--fixed-continuation-horizon",
        type=int,
        default=5,
        help="Execution horizon after the forced root chunk when --continuation-policy=fixed_h.",
    )
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
        choices=("v2_distilled", "v2_value_refined", HIERARCHICAL_MODE, ORDERED_MODE),
        default="v2_value_refined",
    )
    parser.add_argument("--hierarchical-calibration-json", default=None)
    parser.add_argument(
        "--hierarchical-aggregate-calibration-json",
        default=None,
        help="Optional frozen aggregate-risk selector for current-student source trajectories.",
    )
    parser.add_argument("--long-success-noninferiority", type=float, default=0.01)
    parser.add_argument("--short-max-event-probability", type=float, default=0.20)
    parser.add_argument("--long-max-event-probability", type=float, default=0.20)
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


def _uint32_seed(value: int, *, description: str) -> int:
    value = int(value)
    if not 0 <= value <= MAX_POLICY_SEED:
        raise ValueError(f"{description}={value} lies outside the uint32 policy-seed range [0, {MAX_POLICY_SEED}].")
    return value


def _root_seed(
    base_seed: int,
    task_id: int,
    episode_id: int,
    decision_step: int,
    *,
    task_stride: int,
) -> int:
    values = (base_seed, task_id, episode_id, decision_step)
    if any(int(value) < 0 for value in values):
        raise ValueError("Root seed inputs must be non-negative.")
    if int(task_stride) <= 0:
        raise ValueError("root_seed_task_stride must be positive.")
    return _uint32_seed(
        int(base_seed)
        + int(task_id) * int(task_stride)
        + int(episode_id) * ROOT_SEED_EPISODE_STRIDE
        + int(decision_step),
        description="root_seed",
    )


def _branch_seed(root_seed: int, repeat_index: int, repeat_stride: int) -> int:
    if int(repeat_index) < 0 or int(repeat_stride) <= 0:
        raise ValueError("repeat_index must be non-negative and branch_repeat_seed_stride must be positive.")
    return _uint32_seed(
        int(root_seed) + int(repeat_index) * int(repeat_stride),
        description="branch_seed",
    )


def _branch_schedule_seed(
    branch_seed: int,
    *,
    schedule_offset: int = ROOT_SEED_BRANCH_SCHEDULE_OFFSET,
) -> int:
    if int(schedule_offset) <= 0:
        raise ValueError("schedule_offset must be positive.")
    return _uint32_seed(
        int(branch_seed) + int(schedule_offset),
        description="branch_schedule_seed",
    )


def _branch_continuation_seed(
    branch_seed: int,
    continuation_index: int,
    *,
    continuation_offset: int = LEGACY_ROOT_SEED_BRANCH_CONTINUATION_OFFSET,
) -> int:
    if int(continuation_index) < 0:
        raise ValueError("continuation_index must be non-negative.")
    if int(continuation_offset) <= 0:
        raise ValueError("continuation_offset must be positive.")
    return _uint32_seed(
        int(branch_seed) + int(continuation_offset) + int(continuation_index),
        description="branch_continuation_seed",
    )


def _seed_scheme(task_stride: int, branch_repeat_seed_stride: int) -> tuple[str, int, int, bool]:
    if int(task_stride) == LEGACY_ROOT_SEED_TASK_STRIDE:
        return (
            LEGACY_ROOT_SEED_SCHEME,
            ROOT_SEED_BRANCH_SCHEDULE_OFFSET,
            LEGACY_ROOT_SEED_BRANCH_CONTINUATION_OFFSET,
            False,
        )
    if int(branch_repeat_seed_stride) < 4:
        raise ValueError("The opt-in laned seed scheme requires branch_repeat_seed_stride >= 4.")
    # Each repeat owns one branch_repeat_seed_stride-wide region. Root/branch,
    # schedule, and continuation seeds occupy distinct quarter/half lanes.
    return (
        LANED_ROOT_SEED_SCHEME,
        int(branch_repeat_seed_stride) // 4,
        int(branch_repeat_seed_stride) // 2,
        True,
    )


def _seed_scheme_metadata(args: argparse.Namespace) -> dict[str, int | str]:
    scheme, schedule_offset, continuation_offset, strict_namespace = _seed_scheme(
        args.root_seed_task_stride,
        args.branch_repeat_seed_stride,
    )
    return {
        "root_seed_scheme": scheme,
        "root_seed_task_stride": int(args.root_seed_task_stride),
        "root_seed_episode_stride": ROOT_SEED_EPISODE_STRIDE,
        "root_seed_branch_repeat_stride": int(args.branch_repeat_seed_stride),
        "root_seed_branch_schedule_offset": schedule_offset,
        "root_seed_branch_continuation_offset": continuation_offset,
        "root_seed_strict_namespace_validation": strict_namespace,
        "root_seed_max_value": MAX_POLICY_SEED,
    }


def _merge_seed_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _validate_seed_namespace(
    *,
    base_seed: int,
    task_ids: list[int],
    episode_ids: list[int],
    maximum_episode_step: int,
    maximum_continuation_calls: int,
    task_stride: int,
    branch_repeats: int,
    branch_repeat_seed_stride: int,
    teacher_samples: int,
) -> None:
    """Prove opt-in seed intervals are disjoint across actual task/episode identities."""

    if not task_ids or any(int(task_id) < 0 for task_id in task_ids):
        raise ValueError("task_ids must contain non-negative values.")
    if not episode_ids or any(int(episode_id) < 0 for episode_id in episode_ids):
        raise ValueError("episode_ids must contain non-negative values.")
    if maximum_episode_step < 0 or maximum_episode_step >= ROOT_SEED_EPISODE_STRIDE:
        raise ValueError(
            f"maximum_episode_step must lie in [0, {ROOT_SEED_EPISODE_STRIDE}); "
            "otherwise episode seed namespaces can overlap."
        )
    if maximum_continuation_calls < 0:
        raise ValueError("maximum_continuation_calls must be non-negative.")
    if task_stride <= 0 or branch_repeats <= 0 or branch_repeat_seed_stride <= 0 or teacher_samples <= 0:
        raise ValueError("Seed strides, branch_repeats and teacher_samples must be positive.")

    _, schedule_offset, continuation_offset, strict_namespace = _seed_scheme(
        task_stride,
        branch_repeat_seed_stride,
    )
    intervals_by_identity: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for task_id in task_ids:
        for episode_id in episode_ids:
            labelled_intervals: list[tuple[int, int, str]] = []
            root_base = _root_seed(base_seed, task_id, episode_id, 0, task_stride=task_stride)
            root_end = _root_seed(
                base_seed,
                task_id,
                episode_id,
                maximum_episode_step,
                task_stride=task_stride,
            )
            # The root request also constructs batched teacher keys from a
            # uint32 arange, whose exclusive stop must remain representable.
            teacher_end = _uint32_seed(
                root_end + teacher_samples,
                description="root_teacher_seed_stop",
            )
            labelled_intervals.append((root_base, teacher_end, "repeat0_root_branch_teacher"))
            for repeat_index in range(branch_repeats):
                branch_base = _branch_seed(root_base, repeat_index, branch_repeat_seed_stride)
                branch_end = _branch_seed(root_end, repeat_index, branch_repeat_seed_stride)
                if repeat_index:
                    labelled_intervals.append((branch_base, branch_end, f"repeat{repeat_index}_branch"))
                labelled_intervals.append(
                    (
                        _branch_schedule_seed(branch_base, schedule_offset=schedule_offset),
                        _branch_schedule_seed(branch_end, schedule_offset=schedule_offset),
                        f"repeat{repeat_index}_schedule",
                    )
                )
                labelled_intervals.append(
                    (
                        _branch_continuation_seed(
                            branch_base,
                            0,
                            continuation_offset=continuation_offset,
                        ),
                        _branch_continuation_seed(
                            branch_end,
                            maximum_continuation_calls,
                            continuation_offset=continuation_offset,
                        ),
                        f"repeat{repeat_index}_continuation",
                    )
                )
            if strict_namespace:
                sorted_intervals = sorted(labelled_intervals)
                previous_start, previous_end, previous_name = sorted_intervals[0]
                for start, end, name in sorted_intervals[1:]:
                    if start <= previous_end:
                        raise ValueError(
                            "Opt-in seed lanes overlap within task/episode identity "
                            f"{(int(task_id), int(episode_id))}: {previous_name}=[{previous_start},{previous_end}] "
                            f"and {name}=[{start},{end}]. Increase --branch-repeat-seed-stride or narrow the "
                            "episode/call range."
                        )
                    previous_start, previous_end, previous_name = start, end, name
                task_namespace_start = _uint32_seed(
                    int(base_seed) + int(task_id) * int(task_stride),
                    description="task_seed_namespace_start",
                )
                task_namespace_end = _uint32_seed(
                    int(base_seed) + (int(task_id) + 1) * int(task_stride) - 1,
                    description="task_seed_namespace_end",
                )
                for start, end, name in labelled_intervals:
                    if start < task_namespace_start or end > task_namespace_end:
                        raise ValueError(
                            f"Opt-in {name} interval [{start},{end}] for task{task_id}/episode{episode_id} "
                            f"escapes task namespace [{task_namespace_start},{task_namespace_end}]; increase "
                            "--root-seed-task-stride or narrow the episode/repeat range."
                        )
            intervals_by_identity[(int(task_id), int(episode_id))] = _merge_seed_intervals(
                [(start, end) for start, end, _ in labelled_intervals]
            )

    if not strict_namespace:
        # Preserve legacy collection behavior byte-for-byte by keeping its
        # historical +100k continuation namespace. The widened opt-in scheme
        # below is the collision-proof choice for new high-episode data.
        return

    flattened = sorted(
        (start, end, identity)
        for identity, intervals in intervals_by_identity.items()
        for start, end in intervals
    )
    furthest_end = -1
    furthest_identity: tuple[int, int] | None = None
    for start, end, identity in flattened:
        if start <= furthest_end:
            # Per-identity intervals were merged above, so an active interval
            # at this point necessarily belongs to a different identity.
            raise ValueError(
                "Seed namespaces overlap across task/episode identities "
                f"{furthest_identity} and {identity} at uint32 seed {start}; increase "
                "--root-seed-task-stride/--branch-repeat-seed-stride or narrow the episode/repeat range."
            )
        if end > furthest_end:
            furthest_end = end
            furthest_identity = identity


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
    export_prefix_tokens: bool = False,
) -> dict[str, Any]:
    request = dict(observation)
    seed = _uint32_seed(seed, description="policy_seed")
    if teacher:
        _uint32_seed(seed + int(args.teacher_samples), description="teacher_policy_seed_stop")
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
    if args.prefix_token_count and (teacher or export_prefix_tokens):
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
    if args.student_mode == ORDERED_MODE:
        selected = ordered.selected_horizon(
            result,
            model_action_horizon=args.model_action_horizon,
        )
        return selected, selected

    if args.student_mode == HIERARCHICAL_MODE:
        predictor_outputs = {
            name.removeprefix("execution_horizon_"): value
            for name, value in result.items()
            if name.startswith("execution_horizon_")
        }
        aggregate_calibration = getattr(args, "_hierarchical_aggregate_calibration", None)
        if aggregate_calibration is not None:
            decision = aggregate_calibration.apply(predictor_outputs)
        else:
            decision = hierarchical.select_horizon(
                predictor_outputs,
                calibration=args._hierarchical_calibration,
                config=hierarchical.HierarchicalSelectorConfig(
                    success_noninferiority_margin=args.long_success_noninferiority,
                    maximum_short_event_probability=args.short_max_event_probability,
                    maximum_long_event_probability=args.long_max_event_probability,
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


def _fixed_continuation_horizon(args: argparse.Namespace) -> int:
    if args.continuation_policy == "fixed_h9":
        return 9
    if args.continuation_policy == "fixed_h":
        return int(args.fixed_continuation_horizon)
    raise ValueError("A fixed continuation horizon was requested for a non-fixed continuation policy.")


def _nonstudent_source_horizon(args: argparse.Namespace) -> int:
    """Choose the rollout horizon used to reach later collection roots."""
    if getattr(args, "source_policy", "match_continuation") == "fixed_reference":
        return int(args.reference_horizon)
    if args.continuation_policy == "fixed_h9":
        return 9
    if args.continuation_policy == "fixed_h":
        return int(args.reference_horizon)
    raise ValueError("A non-student source horizon was requested for current_student continuation.")


def _source_uses_student(args: argparse.Namespace) -> bool:
    source_policy = getattr(args, "source_policy", "match_continuation")
    if source_policy == "current_student":
        return True
    if source_policy in {"match_continuation", "fixed_reference"}:
        return source_policy == "match_continuation" and args.continuation_policy == "current_student"
    raise ValueError(f"Unsupported source_policy: {source_policy!r}.")


def _trajectory_reservoir_root(
    env: Any,
    observation: dict[str, Any],
    *,
    step: int,
    episode_step_limit: int,
    task_id: int,
    episode_id: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> tuple[TrajectoryRoot | None, int]:
    """Finish one source trajectory, retaining one uniformly sampled policy call."""
    # This local generator never consumes simulator or policy randomness.
    sampling_rng = np.random.default_rng(np.random.SeedSequence([args.seed, task_id, episode_id, 0x524F4F54]))
    root = None
    decision_index = 0
    done = False
    previous_actions_raw = None
    previous_actions_normalized = np.zeros((args.model_action_horizon, args.model_action_dim), dtype=np.float32)
    previous_h = args.reference_horizon
    budget_state = v2.EpisodeBudgetState(balance=min(args.v2_initial_budget, args.v2_budget_capacity))
    while not done and step < episode_step_limit:
        replace_root = int(sampling_rng.integers(decision_index + 1)) == 0
        root_seed = _root_seed(args.seed, task_id, episode_id, step, task_stride=args.root_seed_task_stride)
        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        progress = float(np.clip(step / max(episode_step_limit, 1), 0.0, 1.0))
        result = _policy_request(
            client,
            policy_input,
            seed=root_seed,
            args=args,
            run_student=True,
            previous_actions=previous_actions_raw,
            previous_h=previous_h,
            budget_balance=budget_state.balance / args.v2_budget_capacity,
            episode_progress=progress,
            export_prefix_tokens=replace_root,
        )
        primary_actions = np.asarray(result["actions"], dtype=np.float32)
        if replace_root:
            root = TrajectoryRoot(
                snapshot=_capture_snapshot(env),
                policy_input=copy.deepcopy(policy_input),
                result=copy.deepcopy(result),
                primary_actions=primary_actions.copy(),
                previous_actions_raw=None if previous_actions_raw is None else previous_actions_raw.copy(),
                previous_actions_normalized=previous_actions_normalized.copy(),
                previous_h=previous_h,
                budget_state=copy.deepcopy(budget_state),
                progress=progress,
                step=step,
                root_seed=root_seed,
                decision_index=decision_index,
            )
        _, rollout_horizon = _student_horizon(result, args=args, budget_state=budget_state)
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
            if done or libero_eval._env_success(env):
                done = True
                break
        previous_actions_normalized = np.asarray(
            result["execution_horizon_final_actions_normalized"], dtype=np.float32
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
                    "last_h": rollout_horizon,
                    "source_decision_count": decision_index,
                    "root_sampling": "trajectory_reservoir",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return root, decision_index


def _trajectory_root_with_mc(
    root: TrajectoryRoot,
    *,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> dict[str, Any]:
    """Add auxiliary MC risk samples without replacing the executed source plan."""
    teacher_result = _policy_request(
        client,
        root.policy_input,
        seed=root.root_seed,
        args=args,
        teacher=True,
    )
    result = dict(root.result)
    for name in ("mc_actions_normalized", "mc_coarse_actions_normalized"):
        result[name] = teacher_result[name]
    return result


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
    if args.continuation_policy == "current_student" and args.student_mode not in {
        HIERARCHICAL_MODE,
        ORDERED_MODE,
    }:
        _advance_forced_budget(forced_horizon, args, budget_state)

    action_plan = np.asarray(primary_actions)[:forced_horizon]
    continuation_index = 0
    _, _, continuation_offset, _ = _seed_scheme(
        getattr(args, "root_seed_task_stride", LEGACY_ROOT_SEED_TASK_STRIDE),
        args.branch_repeat_seed_stride,
    )
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

        continuation_seed = _branch_continuation_seed(
            root_seed,
            continuation_index,
            continuation_offset=continuation_offset,
        )
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
            continuation_horizon = _fixed_continuation_horizon(args)
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
    primary_final_actions_normalized: np.ndarray | None = None,
    primary_coarse_actions_normalized: np.ndarray | None = None,
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
    hazard_indices = list(range(num_candidates))
    for repeat_index in range(shape.max_trials):
        if not np.all(trial_valid[hazard_indices, repeat_index]):
            continue
        candidate_success = trial_success[hazard_indices, repeat_index]
        failed_positions = np.flatnonzero(~candidate_success)
        if failed_positions.size and np.any(candidate_success[failed_positions[0] + 1 :]):
            # Episode success is a noisy, non-local outcome. A pattern such as
            # H5=fail,H10=success cannot identify a monotone re-observation
            # boundary, so exclude it from survival supervision rather than
            # injecting a contradictory event time.
            continue
        event_step = (
            shape.candidate_horizons[hazard_indices[int(failed_positions[0])]] - 1 if failed_positions.size else None
        )
        observed_last_step = shape.candidate_horizons[-1] - 1 if event_step is None else event_step
        observed_last_step = min(observed_last_step, shape.action_horizon - 1)
        hazard_at_risk_count[: observed_last_step + 1] += 1
        if event_step is not None and event_step < shape.action_horizon:
            hazard_event_count[event_step] += 1

    record = {
        "prefix_feature": np.asarray(result["execution_horizon_prefix_feature"], dtype=np.float32),
        "state": np.asarray(result["execution_horizon_state_normalized"], dtype=np.float32),
        "coarse_actions": (
            coarse_mc[0]
            if primary_coarse_actions_normalized is None
            else np.asarray(primary_coarse_actions_normalized, dtype=np.float32)
        ),
        "final_actions": (
            final_mc[0]
            if primary_final_actions_normalized is None
            else np.asarray(primary_final_actions_normalized, dtype=np.float32)
        ),
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
    reservoir_sampling = args.root_sampling == "trajectory_reservoir"
    if reservoir_sampling and (args.max_roots_per_episode != 1 or not _source_uses_student(args)):
        raise ValueError("trajectory_reservoir requires max_roots_per_episode=1 and a current_student source.")
    if (
        args.root_stride_calls <= 0
        or args.root_call_offset_cycle <= 0
        or args.root_seed_task_stride <= 0
        or args.action_cot_denoising_steps <= 0
        or args.branch_repeats <= 0
        or args.branch_repeat_seed_stride <= 0
    ):
        raise ValueError(
            "root_stride_calls, root_call_offset_cycle, root_seed_task_stride, action_cot_denoising_steps, "
            "branch_repeats and branch_repeat_seed_stride must be positive."
        )
    candidate_horizons = tuple(sorted(set(args.candidate_horizons)))
    if not candidate_horizons or candidate_horizons[0] <= 0 or candidate_horizons[-1] > args.model_action_horizon:
        raise ValueError("candidate_horizons must be positive and no larger than model_action_horizon.")
    if args.reference_horizon not in candidate_horizons:
        raise ValueError("reference_horizon must be included in candidate_horizons.")
    if args.continuation_policy == "fixed_h" and not (
        1 <= args.fixed_continuation_horizon <= args.model_action_horizon
    ):
        raise ValueError("fixed_continuation_horizon must lie within the model action horizon.")
    if not 0 < args.short_max_event_probability < 1 or not 0 < args.long_max_event_probability < 1:
        raise ValueError("short/long max event probabilities must lie in (0, 1).")
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
    student_is_used = args.continuation_policy == "current_student" or _source_uses_student(args)
    if student_is_used and args.v2_budget_capacity <= 0:
        raise ValueError("v2_budget_capacity must be positive.")
    if args.student_mode in {HIERARCHICAL_MODE, ORDERED_MODE} and not student_is_used:
        raise ValueError(f"{args.student_mode} requires current_student source or continuation.")
    if args.student_mode == HIERARCHICAL_MODE:
        if args.hierarchical_calibration_json is not None and args.hierarchical_aggregate_calibration_json is not None:
            raise ValueError(
                "Provide only one of --hierarchical-calibration-json and "
                "--hierarchical-aggregate-calibration-json."
            )
        if args.hierarchical_aggregate_calibration_json is not None:
            args._hierarchical_aggregate_calibration = hierarchical.AggregateSelectorCalibration.load(
                args.hierarchical_aggregate_calibration_json
            )
            args._hierarchical_calibration = args._hierarchical_aggregate_calibration.pointwise_calibration
        elif args.hierarchical_calibration_json is not None:
            args._hierarchical_aggregate_calibration = None
            args._hierarchical_calibration = hierarchical.HierarchicalCalibration.load(
                args.hierarchical_calibration_json
            )
        else:
            raise ValueError(
                "hierarchical_transformer requires --hierarchical-calibration-json or "
                "--hierarchical-aggregate-calibration-json."
            )
        if args._hierarchical_calibration.candidate_horizons != candidate_horizons:
            raise ValueError("Hierarchical calibration candidates must match collection candidate_horizons.")
        if max(args._hierarchical_calibration.candidate_horizons) > args.model_action_horizon:
            raise ValueError("Hierarchical calibration candidates exceed model_action_horizon.")
    else:
        args._hierarchical_calibration = None
        args._hierarchical_aggregate_calibration = None
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = libero_eval._max_steps(args.task_suite_name)
    task_end = (
        task_suite.n_tasks if args.max_tasks is None else min(task_suite.n_tasks, args.task_start + args.max_tasks)
    )
    task_ids = list(range(args.task_start, task_end))
    _validate_seed_namespace(
        base_seed=args.seed,
        task_ids=task_ids,
        episode_ids=episode_ids,
        maximum_episode_step=max_steps + args.num_steps_wait,
        maximum_continuation_calls=max_steps + args.num_steps_wait,
        task_stride=args.root_seed_task_stride,
        branch_repeats=args.branch_repeats,
        branch_repeat_seed_stride=args.branch_repeat_seed_stride,
        teacher_samples=args.teacher_samples,
    )
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
    initial_state_bank = (
        horizon_initial_states.InitialStateBank(args.initial_state_bank)
        if args.initial_state_bank is not None
        else None
    )
    if initial_state_bank is not None:
        if initial_state_bank.manifest["task_suite"] != args.task_suite_name:
            raise ValueError("Initial-state bank uses a different task suite.")
        for task_id in range(args.task_start, task_end):
            initial_state_bank.validate_presets(task_id, task_suite.get_task_init_states(task_id))
            for episode_id in episode_ids:
                initial_state_bank.state(task_id, episode_id)
    risk_config = v2.V2RiskConfig(
        risk_threshold=args.v2_risk_threshold,
        final_weight=args.v2_final_weight,
        action_cot_weight=args.v2_action_cot_weight,
    )
    metadata = {
        "task_suite": args.task_suite_name,
        "teacher_samples": args.teacher_samples,
        "source_policy": args.source_policy,
        "continuation_policy": args.continuation_policy,
        "fixed_continuation_horizon": (
            _fixed_continuation_horizon(args) if args.continuation_policy != "current_student" else None
        ),
        "student_mode": args.student_mode,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "source_iteration": args.source_iteration,
        "root_call_offset_cycle": args.root_call_offset_cycle,
        "root_sampling": args.root_sampling,
        **_seed_scheme_metadata(args),
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
    if initial_state_bank is not None:
        metadata.update(initial_state_bank.metadata())
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
                    initial_state = (
                        initial_states[episode_id % len(initial_states)]
                        if initial_state_bank is None
                        else initial_state_bank.state(task_id, episode_id)
                    )
                    observation = env.set_init_state(initial_state)
                    if initial_state_bank is not None:
                        np.testing.assert_allclose(
                            env.env.sim.get_state().flatten(),
                            initial_state,
                            rtol=0.0,
                            atol=1e-12,
                            err_msg="Frozen initial state was not restored faithfully.",
                        )
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
                    trajectory_root = None
                    source_decision_count = None
                    if reservoir_sampling and not done:
                        trajectory_root, source_decision_count = _trajectory_reservoir_root(
                            env,
                            observation,
                            step=step,
                            episode_step_limit=episode_step_limit,
                            task_id=task_id,
                            episode_id=episode_id,
                            task_description=task_description,
                            args=args,
                            client=client,
                        )
                        if trajectory_root is None:
                            continue
                        step = trajectory_root.step
                        decision_index = trajectory_root.decision_index
                        previous_actions_raw = trajectory_root.previous_actions_raw
                        previous_actions_normalized = trajectory_root.previous_actions_normalized
                        previous_h = trajectory_root.previous_h
                        budget_state = trajectory_root.budget_state
                    while not done and step < episode_step_limit:
                        collect_root = reservoir_sampling or (
                            decision_index >= root_call_offset
                            and (decision_index - root_call_offset) % args.root_stride_calls == 0
                        )
                        if args.max_roots_per_episode and roots_this_episode >= args.max_roots_per_episode:
                            break
                        root_seed = _root_seed(
                            args.seed,
                            task_id,
                            episode_id,
                            step,
                            task_stride=args.root_seed_task_stride,
                        )
                        progress = float(np.clip(step / max(episode_step_limit, 1), 0.0, 1.0))
                        use_student = _source_uses_student(args)
                        if trajectory_root is not None:
                            root_seed = trajectory_root.root_seed
                            progress = trajectory_root.progress
                            result = _trajectory_root_with_mc(trajectory_root, args=args, client=client)
                            primary_actions = trajectory_root.primary_actions
                        else:
                            policy_input = libero_eval._observation_to_policy_input(
                                observation, task_description, args.resize_size
                            )
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
                            snapshot = trajectory_root.snapshot if trajectory_root is not None else _capture_snapshot(env)
                            risk = v2.risk_targets_from_normalized_mc(
                                result["mc_coarse_actions_normalized"],
                                result["mc_actions_normalized"],
                                config=risk_config,
                            )
                            branch_rows: list[list[dict[str, Any]]] = [[] for _ in candidate_horizons]
                            _, branch_schedule_offset, _, _ = _seed_scheme(
                                args.root_seed_task_stride,
                                args.branch_repeat_seed_stride,
                            )
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
                                branch_seed = _branch_seed(
                                    root_seed,
                                    repeat_index,
                                    args.branch_repeat_seed_stride,
                                )
                                schedule_rng = np.random.default_rng(
                                    _branch_schedule_seed(
                                        branch_seed,
                                        schedule_offset=branch_schedule_offset,
                                    )
                                )
                                for candidate_index in schedule_rng.permutation(scheduled_candidates):
                                    forced_horizon = candidate_horizons[candidate_index]
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
                                primary_final_actions_normalized=(
                                    trajectory_root.result["execution_horizon_final_actions_normalized"]
                                    if trajectory_root is not None
                                    else None
                                ),
                                primary_coarse_actions_normalized=(
                                    trajectory_root.result["execution_horizon_coarse_actions_normalized"]
                                    if trajectory_root is not None
                                    else None
                                ),
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
                                            "root_sampling": args.root_sampling,
                                            "source_decision_index": decision_index,
                                            "source_decision_count": source_decision_count,
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
                            if reservoir_sampling:
                                break
                            observation = _restore_snapshot(env, snapshot)

                        if use_student:
                            _, rollout_horizon = _student_horizon(result, args=args, budget_state=budget_state)
                        else:
                            rollout_horizon = _nonstudent_source_horizon(args)
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
