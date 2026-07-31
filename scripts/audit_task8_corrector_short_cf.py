"""Task8 same-snapshot audit for short-horizon corrector action quality.

The audit follows a stale-H10 trajectory through held-out Task8 initial states.
At every fresh policy decision it executes the first four anchor actions, takes
one canonical MuJoCo/controller snapshot, and compares three H6 branches:

* ``stale_h6``: cached anchor actions 4:10;
* ``corrector_h6``: the saved direct age-4 corrector output;
* ``fresh_teacher_h6``: a fresh 10/10 ACoT request at the age-4 state.

The stale and corrector endpoints each receive a fresh H9 teacher continuation.
Those two continuation requests use the same policy seed, so their difference
is caused by the H6 endpoint state rather than diffusion noise.  Scores are
computed from Task8's two ``On(moka_pot, stove_cook_region)`` predicates: the
number satisfied and the sum of their dense continuous progress.

This is intentionally independent of the ordinary LIBERO evaluation scripts.
Root records and episode completion records are appended immediately so a
partially interrupted run can resume deterministically.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

import collect_action_cot_state_branches as branch_collector
import eval_libero_action_cot_pruning as libero_eval
import eval_libero_branched_corrector as corrector_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import privileged_progress
from openpi.shared import normalize as normalize_lib


TASK_ID = 8
TASK_DESCRIPTION = "put both moka pots on the stove"
ACTION_DIM = 7
ANCHOR_HORIZON = 4
AUDIT_HORIZON = 6
CONTINUATION_HORIZON = 9
HELD_OUT_START = 10
ARMS = ("stale_h6", "corrector_h6", "fresh_teacher_h6")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrector-summary", required=True)
    parser.add_argument("--corrector-params", default=None)
    parser.add_argument("--norm-stats-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument(
        "--trial-start",
        type=int,
        default=HELD_OUT_START,
        help="First Task8 initial-state ID. IDs 0-9 were used by the canonical corrector dataset.",
    )
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument(
        "--max-roots-per-episode",
        type=int,
        default=0,
        help="Maximum audited age-4 roots per episode; zero audits the full stale-H10 trajectory.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument(
        "--continuation-seed-offset",
        type=int,
        default=100_000_000,
        help="Offset from the canonical root seed used by both H9 continuation arms.",
    )
    parser.add_argument(
        "--progress-tie-tolerance",
        type=float,
        default=1e-6,
        help="Absolute two-pot progress-sum difference counted as a tie.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.task_suite_name != "libero_10":
        raise ValueError("This Task8-specific audit requires --task-suite-name libero_10.")
    if args.trial_start < HELD_OUT_START:
        raise ValueError(
            f"Task8 initial-state IDs 0-{HELD_OUT_START - 1} were used during canonical "
            f"corrector collection; --trial-start must be at least {HELD_OUT_START}."
        )
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")
    if args.max_roots_per_episode < 0:
        raise ValueError("--max-roots-per-episode must be non-negative.")
    if args.seed < 0 or args.num_steps_wait < 0:
        raise ValueError("--seed and --num-steps-wait must be non-negative.")
    if args.resize_size <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("--resize-size and --action-cot-denoising-steps must be positive.")
    if args.continuation_seed_offset <= 0:
        raise ValueError("--continuation-seed-offset must be positive.")
    if args.progress_tie_tolerance < 0:
        raise ValueError("--progress-tie-tolerance must be non-negative.")


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


def _task8_progress(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    overall = privileged_progress.score_libero_goal_progress(env, observation)
    target_components = []
    for component in overall.components:
        arguments = [str(value) for value in component.get("arguments", ())]
        if (
            str(component.get("predicate", "")).lower() == "on"
            and len(arguments) == 2
            and "moka" in arguments[0].lower()
            and "stove" in arguments[1].lower()
        ):
            target_components.append(component)
    if len(target_components) != 2:
        available = [
            {
                "predicate": component.get("predicate"),
                "arguments": component.get("arguments"),
            }
            for component in overall.components
        ]
        raise RuntimeError(
            "Task8 audit expected exactly two On(moka_pot, stove) goal components; "
            f"found {len(target_components)} in {available}."
        )
    target_components.sort(key=lambda item: tuple(str(value) for value in item["arguments"]))
    compact_components = [
        {
            "predicate": str(component["predicate"]),
            "arguments": [str(value) for value in component["arguments"]],
            "satisfied": bool(component["satisfied"]),
            "continuous_progress": float(component["continuous_progress"]),
            "grasped": bool(component.get("grasped", False)),
            "eef_to_object_distance": float(component.get("eef_to_object_distance", float("nan"))),
            "object_to_target_distance": float(
                component.get("object_to_target_distance", float("nan"))
            ),
        }
        for component in target_components
    ]
    progress_sum = float(
        sum(float(component["continuous_progress"]) for component in compact_components)
    )
    satisfied_count = int(sum(bool(component["satisfied"]) for component in compact_components))
    return {
        "pot_progress_sum": progress_sum,
        "pot_progress_mean": progress_sum / 2.0,
        "on_satisfied_count": satisfied_count,
        "task_success": bool(libero_eval._env_success(env)),
        "privileged_goal_score": float(overall.score),
        "pot_components": compact_components,
    }


def _teacher_request(
    client: websocket_policy.WebsocketClientPolicy,
    observation: dict[str, Any],
    task_description: str,
    *,
    policy_seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_input = libero_eval._observation_to_policy_input(
        observation,
        task_description,
        args.resize_size,
    )
    result = branch_collector._teacher_request(
        client,
        policy_input,
        policy_seed=policy_seed,
        denoising_steps=args.action_cot_denoising_steps,
    )
    return policy_input, result


def _env_actions(result: dict[str, Any], minimum_horizon: int) -> np.ndarray:
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < minimum_horizon or actions.shape[1] < ACTION_DIM:
        raise ValueError(
            f"Expected at least [{minimum_horizon}, {ACTION_DIM}] environment actions, "
            f"got {actions.shape}."
        )
    return actions[:, :ACTION_DIM]


def _action_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Action arrays must have equal shape, got {left.shape} and {right.shape}.")
    continuous_difference = left[:, :6] - right[:, :6]
    return {
        "continuous_rmse": float(np.sqrt(np.mean(np.square(continuous_difference)))),
        "continuous_mean_l2": float(np.mean(np.linalg.norm(continuous_difference, axis=-1))),
        "continuous_max_abs": float(np.max(np.abs(continuous_difference))),
        "gripper_disagreement_fraction": float(
            np.mean((left[:, 6] >= 0.0) != (right[:, 6] >= 0.0))
        ),
    }


def _run_arm(
    env: Any,
    snapshot: branch_collector.CanonicalSimulatorSnapshot,
    actions: np.ndarray,
    *,
    arm: str,
    with_teacher_h9: bool,
    continuation_seed: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> dict[str, Any]:
    observation = branch_collector._restore_canonical_snapshot(env, snapshot)
    before = _task8_progress(env, observation)
    try:
        observation, done, executed_h6 = branch_collector._step_actions(
            env,
            observation,
            np.asarray(actions, dtype=np.float32)[:AUDIT_HORIZON],
        )
        after_h6 = _task8_progress(env, observation)
        continuation_steps = 0
        continuation_wall_ms = 0.0
        continuation_requested = False
        after_handoff = None
        if with_teacher_h9:
            if not done and not libero_eval._env_success(env):
                _, continuation_result = _teacher_request(
                    client,
                    observation,
                    task_description,
                    policy_seed=continuation_seed,
                    args=args,
                )
                continuation_actions = _env_actions(
                    continuation_result,
                    CONTINUATION_HORIZON,
                )
                continuation_requested = True
                continuation_wall_ms = float(continuation_result["collector_wall_ms"])
                observation, done, executed_h9 = branch_collector._step_actions(
                    env,
                    observation,
                    continuation_actions[:CONTINUATION_HORIZON],
                )
                continuation_steps = len(executed_h9)
            after_handoff = _task8_progress(env, observation)
        return {
            "arm": arm,
            "before": before,
            "after_h6": after_h6,
            "h6_steps": len(executed_h6),
            "h6_progress_delta": float(
                after_h6["pot_progress_sum"] - before["pot_progress_sum"]
            ),
            "success_after_h6": bool(after_h6["task_success"]),
            "continuation_seed": continuation_seed if with_teacher_h9 else None,
            "continuation_requested": continuation_requested,
            "continuation_steps": continuation_steps,
            "continuation_teacher_wall_ms": continuation_wall_ms,
            "after_handoff": after_handoff,
            "handoff_progress_delta": (
                float(after_handoff["pot_progress_sum"] - before["pot_progress_sum"])
                if after_handoff is not None
                else None
            ),
            "success_after_handoff": (
                bool(after_handoff["task_success"]) if after_handoff is not None else None
            ),
            "terminated_during_h6": bool(done and len(executed_h6) < AUDIT_HORIZON),
        }
    finally:
        branch_collector._restore_canonical_snapshot(env, snapshot)


def _audit_root(
    env: Any,
    observation: dict[str, Any],
    *,
    task_description: str,
    trial_id: int,
    root_index: int,
    decision_step: int,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    corrector: corrector_eval.DirectCorrector,
    norm_stats: dict[str, Any],
    collect_record: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any], bool, int]:
    root_seed = branch_collector.canonical_policy_seed(
        args.seed,
        TASK_ID,
        trial_id,
        decision_step,
    )
    anchor_input, anchor_result = _teacher_request(
        client,
        observation,
        task_description,
        policy_seed=root_seed,
        args=args,
    )
    anchor_actions = _env_actions(anchor_result, ANCHOR_HORIZON + AUDIT_HORIZON)
    observation, done, executed_prefix = branch_collector._step_actions(
        env,
        observation,
        anchor_actions[:ANCHOR_HORIZON],
    )
    actual_steps = len(executed_prefix)
    if done or libero_eval._env_success(env) or len(executed_prefix) < ANCHOR_HORIZON:
        return None, observation, bool(done or libero_eval._env_success(env)), actual_steps

    age4_snapshot = branch_collector._capture_canonical_snapshot(env)
    stale_actions = anchor_actions[ANCHOR_HORIZON : ANCHOR_HORIZON + AUDIT_HORIZON]
    if not collect_record:
        observation = branch_collector._restore_canonical_snapshot(env, age4_snapshot)
        observation, done, executed_stale = branch_collector._step_actions(
            env,
            observation,
            stale_actions,
        )
        return (
            None,
            observation,
            bool(done or libero_eval._env_success(env)),
            actual_steps + len(executed_stale),
        )

    age4_input, fresh_result = _teacher_request(
        client,
        observation,
        task_description,
        policy_seed=root_seed,
        args=args,
    )
    fresh_actions = _env_actions(fresh_result, AUDIT_HORIZON)[:AUDIT_HORIZON]
    cache = corrector_eval._make_cache(anchor_input, anchor_result, norm_stats)
    current_images = corrector_eval._small_images(age4_input)
    current_state = corrector_eval._normalise_state(
        age4_input["observation/state"],
        norm_stats,
    )
    corrected_actions, corrector_ms = corrector(
        cache,
        current_images,
        current_state,
    )
    corrected_actions = np.asarray(corrected_actions, dtype=np.float32)
    if corrected_actions.shape != (AUDIT_HORIZON, ACTION_DIM):
        raise ValueError(
            f"Corrector returned {corrected_actions.shape}; "
            f"expected {(AUDIT_HORIZON, ACTION_DIM)}."
        )

    continuation_seed = root_seed + args.continuation_seed_offset
    if continuation_seed > np.iinfo(np.uint32).max:
        raise ValueError(f"Continuation seed {continuation_seed} exceeds uint32.")
    arm_actions = {
        "stale_h6": stale_actions,
        "corrector_h6": corrected_actions,
        "fresh_teacher_h6": fresh_actions,
    }
    arm_records = {
        arm: _run_arm(
            env,
            age4_snapshot,
            arm_actions[arm],
            arm=arm,
            with_teacher_h9=arm != "fresh_teacher_h6",
            continuation_seed=continuation_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        for arm in ARMS
    }
    before_scores = [
        float(arm_records[arm]["before"]["pot_progress_sum"])
        for arm in ARMS
    ]
    record = {
        "schema_version": 1,
        "task_suite_name": args.task_suite_name,
        "task_id": TASK_ID,
        "task_description": task_description,
        "trial_id": trial_id,
        "root_index": root_index,
        "anchor_decision_step": decision_step,
        "age4_decision_step": decision_step + ANCHOR_HORIZON,
        "root_seed": root_seed,
        "continuation_seed": continuation_seed,
        "anchor_teacher_wall_ms": float(anchor_result["collector_wall_ms"]),
        "fresh_teacher_wall_ms": float(fresh_result["collector_wall_ms"]),
        "corrector_ms": float(corrector_ms),
        "restore_before_progress_spread": float(max(before_scores) - min(before_scores)),
        "action_distances": {
            "corrector_vs_stale": _action_distance(corrected_actions, stale_actions),
            "corrector_vs_fresh_teacher": _action_distance(
                corrected_actions,
                fresh_actions,
            ),
            "stale_vs_fresh_teacher": _action_distance(stale_actions, fresh_actions),
        },
        "arms": arm_records,
    }

    observation = branch_collector._restore_canonical_snapshot(env, age4_snapshot)
    observation, done, executed_stale = branch_collector._step_actions(
        env,
        observation,
        stale_actions,
    )
    return (
        record,
        observation,
        bool(done or libero_eval._env_success(env)),
        actual_steps + len(executed_stale),
    )


def _run_episode(
    *,
    task: Any,
    initial_state: Any,
    trial_id: int,
    existing_root_keys: set[tuple[int, int]],
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    corrector: corrector_eval.DirectCorrector,
    norm_stats: dict[str, Any],
    roots_file: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, task_description = libero_eval._get_libero_env(
        task,
        libero_eval.LIBERO_ENV_RESOLUTION,
        args.seed,
    )
    if str(task_description).strip().lower() != TASK_DESCRIPTION:
        libero_eval._safe_close_env(env)
        raise RuntimeError(
            f"Task ID {TASK_ID} resolved to {task_description!r}, expected {TASK_DESCRIPTION!r}."
        )
    new_records: list[dict[str, Any]] = []
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
        while not done and step < step_limit:
            if args.max_roots_per_episode and root_index >= args.max_roots_per_episode:
                termination_reason = "root_limit"
                break
            root_key = (trial_id, root_index)
            record, observation, done, actual_steps = _audit_root(
                env,
                observation,
                task_description=task_description,
                trial_id=trial_id,
                root_index=root_index,
                decision_step=step,
                args=args,
                client=client,
                corrector=corrector,
                norm_stats=norm_stats,
                collect_record=root_key not in existing_root_keys,
            )
            step += actual_steps
            if record is not None:
                roots_file.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
                roots_file.flush()
                new_records.append(record)
                existing_root_keys.add(root_key)
                print(
                    json.dumps(
                        {
                            "task_id": TASK_ID,
                            "trial_id": trial_id,
                            "root_index": root_index,
                            "age4_decision_step": record["age4_decision_step"],
                            "corrector_minus_stale_h6_progress": (
                                record["arms"]["corrector_h6"]["after_h6"]["pot_progress_sum"]
                                - record["arms"]["stale_h6"]["after_h6"]["pot_progress_sum"]
                            ),
                            "corrector_minus_stale_handoff_progress": (
                                record["arms"]["corrector_h6"]["after_handoff"][
                                    "pot_progress_sum"
                                ]
                                - record["arms"]["stale_h6"]["after_handoff"][
                                    "pot_progress_sum"
                                ]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            root_index += 1
            if actual_steps == 0:
                termination_reason = "no_environment_progress"
                break

        success = bool(done or libero_eval._env_success(env))
        if success:
            termination_reason = "success"
        elif not termination_reason:
            termination_reason = "step_limit"
        return {
            "schema_version": 1,
            "task_suite_name": args.task_suite_name,
            "task_id": TASK_ID,
            "task_description": task_description,
            "trial_id": trial_id,
            "trajectory": "stale_h10",
            "success": success,
            "termination_reason": termination_reason,
            "environment_steps": step,
            "control_steps": max(0, step - args.num_steps_wait),
            "roots_visited": root_index,
            "root_limit": args.max_roots_per_episode,
        }, new_records
    finally:
        libero_eval._safe_close_env(env)


def _finite_mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _arm_summary(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows = [record["arms"][arm] for record in records]
    result = {
        "roots": len(rows),
        "mean_before_progress_sum": _finite_mean(
            [row["before"]["pot_progress_sum"] for row in rows]
        ),
        "mean_after_h6_progress_sum": _finite_mean(
            [row["after_h6"]["pot_progress_sum"] for row in rows]
        ),
        "mean_h6_progress_delta": _finite_mean([row["h6_progress_delta"] for row in rows]),
        "satisfied_predicates_after_h6": int(
            sum(int(row["after_h6"]["on_satisfied_count"]) for row in rows)
        ),
        "roots_success_after_h6": int(sum(bool(row["success_after_h6"]) for row in rows)),
    }
    handoff_rows = [row for row in rows if row["after_handoff"] is not None]
    if handoff_rows:
        result.update(
            {
                "mean_after_handoff_progress_sum": _finite_mean(
                    [row["after_handoff"]["pot_progress_sum"] for row in handoff_rows]
                ),
                "mean_handoff_progress_delta": _finite_mean(
                    [row["handoff_progress_delta"] for row in handoff_rows]
                ),
                "satisfied_predicates_after_handoff": int(
                    sum(
                        int(row["after_handoff"]["on_satisfied_count"])
                        for row in handoff_rows
                    )
                ),
                "roots_success_after_handoff": int(
                    sum(bool(row["success_after_handoff"]) for row in handoff_rows)
                ),
                "continuation_teacher_calls": int(
                    sum(bool(row["continuation_requested"]) for row in handoff_rows)
                ),
            }
        )
    return result


def _wins_losses_ties(
    records: list[dict[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
    endpoint: str,
    tolerance: float,
) -> dict[str, Any]:
    differences = []
    satisfied_differences = []
    for record in records:
        left = record["arms"][left_arm][endpoint]
        right = record["arms"][right_arm][endpoint]
        if left is None or right is None:
            continue
        differences.append(float(left["pot_progress_sum"] - right["pot_progress_sum"]))
        satisfied_differences.append(
            int(left["on_satisfied_count"]) - int(right["on_satisfied_count"])
        )
    wins = sum(value > tolerance for value in differences)
    losses = sum(value < -tolerance for value in differences)
    ties = len(differences) - wins - losses
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "endpoint": endpoint,
        "paired_roots": len(differences),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "mean_progress_sum_delta": _finite_mean(differences),
        "median_progress_sum_delta": (
            float(np.median(np.asarray(differences, dtype=np.float64)))
            if differences
            else None
        ),
        "predicate_wins": sum(value > 0 for value in satisfied_differences),
        "predicate_losses": sum(value < 0 for value in satisfied_differences),
        "predicate_ties": sum(value == 0 for value in satisfied_differences),
        "tie_tolerance": tolerance,
    }


def _summarize(
    root_records: list[dict[str, Any]],
    episode_records: list[dict[str, Any]],
    args: argparse.Namespace,
    corrector: corrector_eval.DirectCorrector,
) -> dict[str, Any]:
    trial_end = args.trial_start + args.num_trials
    selected_roots = [
        row
        for row in root_records
        if int(row["task_id"]) == TASK_ID
        and args.trial_start <= int(row["trial_id"]) < trial_end
    ]
    selected_episodes_by_trial = {
        int(row["trial_id"]): row
        for row in episode_records
        if int(row["task_id"]) == TASK_ID
        and args.trial_start <= int(row["trial_id"]) < trial_end
    }
    episode_values = list(selected_episodes_by_trial.values())
    comparisons = {
        "corrector_vs_stale_after_h6": _wins_losses_ties(
            selected_roots,
            left_arm="corrector_h6",
            right_arm="stale_h6",
            endpoint="after_h6",
            tolerance=args.progress_tie_tolerance,
        ),
        "corrector_vs_stale_after_handoff": _wins_losses_ties(
            selected_roots,
            left_arm="corrector_h6",
            right_arm="stale_h6",
            endpoint="after_handoff",
            tolerance=args.progress_tie_tolerance,
        ),
        "corrector_vs_fresh_teacher_after_h6": _wins_losses_ties(
            selected_roots,
            left_arm="corrector_h6",
            right_arm="fresh_teacher_h6",
            endpoint="after_h6",
            tolerance=args.progress_tie_tolerance,
        ),
        "stale_vs_fresh_teacher_after_h6": _wins_losses_ties(
            selected_roots,
            left_arm="stale_h6",
            right_arm="fresh_teacher_h6",
            endpoint="after_h6",
            tolerance=args.progress_tie_tolerance,
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "complete"
            if len(selected_episodes_by_trial) == args.num_trials
            else "running"
        ),
        "protocol": {
            "task_suite_name": args.task_suite_name,
            "task_id": TASK_ID,
            "task_description": TASK_DESCRIPTION,
            "trial_start": args.trial_start,
            "num_trials": args.num_trials,
            "held_out_initial_state_ids": list(range(args.trial_start, trial_end)),
            "max_roots_per_episode": args.max_roots_per_episode,
            "trajectory": "stale_h10",
            "anchor_horizon": ANCHOR_HORIZON,
            "audit_horizon": AUDIT_HORIZON,
            "continuation_horizon": CONTINUATION_HORIZON,
            "same_seed_handoff": True,
            "continuation_seed_offset": args.continuation_seed_offset,
            "seed": args.seed,
            "action_cot_denoising_steps": args.action_cot_denoising_steps,
            "score": (
                "sum of continuous progress for Task8's two "
                "On(moka_pot, stove_cook_region) predicates"
            ),
        },
        "corrector": {
            "summary_path": corrector.summary_path,
            "params_path": corrector.params_path,
        },
        "completed_trials": len(selected_episodes_by_trial),
        "requested_trials": args.num_trials,
        "audited_roots": len(selected_roots),
        "stale_trajectory_episode_successes": int(
            sum(bool(row["success"]) for row in episode_values)
        ),
        "stale_trajectory_episode_success_rate": (
            float(np.mean([bool(row["success"]) for row in episode_values]))
            if episode_values
            else None
        ),
        "mean_roots_per_completed_episode": _finite_mean(
            [row["roots_visited"] for row in episode_values]
        ),
        "restore_before_progress_max_spread": (
            float(
                max(
                    float(row["restore_before_progress_spread"])
                    for row in selected_roots
                )
            )
            if selected_roots
            else None
        ),
        "by_arm": {
            arm: _arm_summary(selected_roots, arm)
            for arm in ARMS
        },
        "wins_losses_ties": comparisons,
        "action_distance_means": {
            pair: {
                metric: _finite_mean(
                    [
                        row["action_distances"][pair][metric]
                        for row in selected_roots
                    ]
                )
                for metric in (
                    "continuous_rmse",
                    "continuous_mean_l2",
                    "continuous_max_abs",
                    "gripper_disagreement_fraction",
                )
            }
            for pair in (
                "corrector_vs_stale",
                "corrector_vs_fresh_teacher",
                "stale_vs_fresh_teacher",
            )
        },
        "mean_corrector_ms": _finite_mean(
            [row["corrector_ms"] for row in selected_roots]
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots_path = output_dir / "roots.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    root_records = _read_jsonl(roots_path)
    episode_records = _read_jsonl(episodes_path)
    existing_root_keys = {
        (int(row["trial_id"]), int(row["root_index"]))
        for row in root_records
        if int(row.get("task_id", -1)) == TASK_ID
    }
    completed_trials = {
        int(row["trial_id"])
        for row in episode_records
        if int(row.get("task_id", -1)) == TASK_ID
    }

    norm_stats = normalize_lib.load(args.norm_stats_dir)
    corrector = corrector_eval.DirectCorrector(
        args.corrector_summary,
        args.corrector_params,
    )
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(TASK_ID)
    initial_states = suite.get_task_init_states(TASK_ID)
    trial_end = args.trial_start + args.num_trials
    if trial_end > len(initial_states):
        raise ValueError(
            f"Requested Task8 trials [{args.trial_start}, {trial_end}) but only "
            f"{len(initial_states)} initial states are available."
        )

    with (
        roots_path.open("a", encoding="utf-8") as roots_file,
        episodes_path.open("a", encoding="utf-8") as episodes_file,
    ):
        for trial_id in range(args.trial_start, trial_end):
            if trial_id in completed_trials:
                logging.info("Skipping completed Task8 trial %s.", trial_id)
                continue
            logging.info("Auditing Task8 trial %s.", trial_id)
            episode_record, new_root_records = _run_episode(
                task=task,
                initial_state=initial_states[trial_id],
                trial_id=trial_id,
                existing_root_keys=existing_root_keys,
                args=args,
                client=client,
                corrector=corrector,
                norm_stats=norm_stats,
                roots_file=roots_file,
            )
            root_records.extend(new_root_records)
            episodes_file.write(
                json.dumps(episode_record, sort_keys=True, allow_nan=True) + "\n"
            )
            episodes_file.flush()
            episode_records.append(episode_record)
            completed_trials.add(trial_id)
            summary = _summarize(
                root_records,
                episode_records,
                args,
                corrector,
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
                encoding="utf-8",
            )
            logging.info(
                "Task8 trial %s complete: success=%s roots=%s.",
                trial_id,
                episode_record["success"],
                episode_record["roots_visited"],
            )

    summary = _summarize(root_records, episode_records, args, corrector)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    logging.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
