"""Replay saved LIBERO root states with repeated execution-horizon branches.

This is a small robustness audit for counterfactual labels.  It regenerates the
primary action chunk at a saved MuJoCo root, then repeats selected H branches
with controlled continuation-policy seeds.  The output is diagnostic only: it
does not replace a closed-loop episode evaluation or create training records.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import pathlib
import time
from typing import Any

import collect_execution_horizon_counterfactuals as collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--tasks", nargs="+", type=int, default=[8, 9])
    parser.add_argument("--reference-horizon", type=int, default=10)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-roots-per-task", type=int, default=0)
    parser.add_argument("--repeat-seed-stride", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
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
    parser.add_argument("--debug-video-stride", type=int, default=5)
    return parser


def _replacement_scalar(value: Any, replacement: int | bool) -> Any:
    if isinstance(value, np.ndarray):
        return np.full_like(value, replacement)
    if isinstance(value, np.generic):
        return value.dtype.type(replacement)
    try:
        return type(value)(replacement)
    except (TypeError, ValueError):
        return replacement


def _saved_snapshot(
    env: Any,
    physics_state: np.ndarray,
    decision_step: int,
) -> collector.SimulatorSnapshot:
    template = collector._capture_snapshot(env)
    scalar_attributes = []
    for owner, name, value in template.scalar_attributes:
        if name in ("timestep", "_timestep"):
            value = _replacement_scalar(value, decision_step)
        elif name in ("done", "_done"):
            value = _replacement_scalar(value, False)
        scalar_attributes.append((owner, name, value))
    return collector.SimulatorSnapshot(
        physics_state=np.asarray(physics_state, dtype=np.float64).copy(),
        scalar_attributes=scalar_attributes,
        random_states=copy.deepcopy(template.random_states),
    )


def _selected_indices(arrays: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    task_ids = np.asarray(arrays["task_id"], dtype=np.int64)
    success = np.asarray(arrays["branch_success"], dtype=np.bool_)
    valid = np.asarray(arrays["branch_valid"], dtype=np.bool_)
    reference = args.reference_horizon - 1
    mask = np.isin(task_ids, np.asarray(args.tasks, dtype=np.int64))
    mask &= valid[:, reference] & ~success[:, reference]
    mask &= np.any(success & valid, axis=1)
    selected = []
    for task_id in args.tasks:
        task_indices = np.flatnonzero(mask & (task_ids == task_id))
        order = np.lexsort(
            (
                np.asarray(arrays["decision_step"])[task_indices],
                np.asarray(arrays["episode_id"])[task_indices],
            )
        )
        task_indices = task_indices[order]
        if args.max_roots_per_task:
            task_indices = task_indices[: args.max_roots_per_task]
        selected.extend(task_indices.tolist())
    return np.asarray(selected, dtype=np.int64)


def _root_budget_state(arrays: dict[str, np.ndarray], index: int, args: argparse.Namespace) -> v2.EpisodeBudgetState:
    normalized_balance = float(np.asarray(arrays["budget_balance"])[index])
    return v2.EpisodeBudgetState(
        balance=float(np.clip(normalized_balance, 0.0, 1.0) * args.v2_budget_capacity)
    )


def _prepare_root(
    task_suite: Any,
    arrays: dict[str, np.ndarray],
    index: int,
    args: argparse.Namespace,
) -> tuple[Any, str, collector.SimulatorSnapshot, int]:
    task_id = int(arrays["task_id"][index])
    episode_id = int(arrays["episode_id"][index])
    decision_step = int(arrays["decision_step"][index])
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
    env.reset()
    observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
    for _ in range(args.num_steps_wait):
        observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
        if done:
            break
    snapshot = _saved_snapshot(env, arrays["physics_state"][index], decision_step)
    collector._restore_snapshot(env, snapshot)
    environment_horizon = libero_eval._env_horizon(env)
    episode_step_limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
    if environment_horizon is not None:
        episode_step_limit = min(episode_step_limit, environment_horizon)
    return env, task_description, snapshot, episode_step_limit


def _replay_root(
    client: websocket_policy.WebsocketClientPolicy,
    task_suite: Any,
    arrays: dict[str, np.ndarray],
    index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    env, task_description, snapshot, episode_step_limit = _prepare_root(task_suite, arrays, index, args)
    task_id = int(arrays["task_id"][index])
    episode_id = int(arrays["episode_id"][index])
    decision_step = int(arrays["decision_step"][index])
    root_seed = int(arrays["root_seed"][index])
    try:
        observation = collector._restore_snapshot(env, snapshot)
        policy_input = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        result = collector._policy_request(
            client,
            policy_input,
            seed=root_seed,
            args=args,
        )
        primary_actions = np.asarray(result["actions"], dtype=np.float32)
        regenerated_normalized = result.get("execution_horizon_final_actions_normalized")
        action_max_abs_difference = None
        if regenerated_normalized is not None:
            regenerated_normalized = np.asarray(regenerated_normalized, dtype=np.float32)
            stored_normalized = np.asarray(arrays["final_actions"][index], dtype=np.float32)
            if regenerated_normalized.shape == stored_normalized.shape:
                action_max_abs_difference = float(np.max(np.abs(regenerated_normalized - stored_normalized)))

        stored_success = np.asarray(arrays["branch_success"][index], dtype=np.bool_)
        replay: dict[str, list[dict[str, Any]]] = {}
        for horizon in args.horizons:
            outcomes = []
            for repeat in range(args.repeats):
                continuation_seed = root_seed + repeat * args.repeat_seed_stride
                success, timeout, steps, calls, _ = collector._run_branch(
                    env,
                    snapshot,
                    primary_actions,
                    forced_horizon=horizon,
                    root_step=decision_step,
                    episode_step_limit=episode_step_limit,
                    root_seed=continuation_seed,
                    task_description=task_description,
                    args=args,
                    client=client,
                    root_budget_state=_root_budget_state(arrays, index, args),
                    capture_video=False,
                )
                outcome = {
                    "repeat": repeat,
                    "continuation_seed": continuation_seed,
                    "success": success,
                    "timeout": timeout,
                    "remaining_steps": steps,
                    "remaining_calls": calls,
                }
                outcomes.append(outcome)
                print(
                    json.dumps(
                        {
                            "task": task_id,
                            "episode": episode_id,
                            "decision_step": decision_step,
                            "horizon": horizon,
                            **outcome,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            replay[str(horizon)] = outcomes

        original_rescue_candidates = [
            horizon
            for horizon in args.horizons
            if horizon != args.reference_horizon and stored_success[horizon - 1]
        ]
        original_rescue_horizon = max(original_rescue_candidates, default=None)
        return {
            "dataset_index": int(index),
            "task_id": task_id,
            "episode_id": episode_id,
            "decision_step": decision_step,
            "root_seed": root_seed,
            "root_action_max_abs_difference_vs_float16_record": action_max_abs_difference,
            "stored_success_by_h": {
                str(horizon): bool(stored_success[horizon - 1]) for horizon in args.horizons
            },
            "original_rescue_horizon": original_rescue_horizon,
            "replay": replay,
        }
    finally:
        libero_eval._safe_close_env(env)


def _summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    per_horizon: dict[str, Any] = {}
    repeat_zero_matches = 0
    repeat_zero_total = 0
    for horizon in args.horizons:
        outcomes = [item for record in records for item in record["replay"][str(horizon)]]
        successes = sum(bool(item["success"]) for item in outcomes)
        per_horizon[str(horizon)] = {
            "num_branch_rollouts": len(outcomes),
            "success_count": successes,
            "success_rate": successes / len(outcomes) if outcomes else None,
            "average_remaining_calls": (
                float(np.mean([item["remaining_calls"] for item in outcomes])) if outcomes else None
            ),
        }
        for record in records:
            first = record["replay"][str(horizon)][0]
            repeat_zero_matches += int(first["success"] == record["stored_success_by_h"][str(horizon)])
            repeat_zero_total += 1

    reference_key = str(args.reference_horizon)
    paired_alt_wins = 0
    paired_reference_wins = 0
    paired_both_success = 0
    paired_both_failure = 0
    roots_with_more_stable_original_rescue = 0
    roots_with_majority_original_rescue = 0
    root_comparisons = []
    for record in records:
        rescue_horizon = record["original_rescue_horizon"]
        if rescue_horizon is None:
            continue
        reference = record["replay"][reference_key]
        rescue = record["replay"][str(rescue_horizon)]
        reference_rate = float(np.mean([item["success"] for item in reference]))
        rescue_rate = float(np.mean([item["success"] for item in rescue]))
        roots_with_more_stable_original_rescue += int(rescue_rate > reference_rate)
        roots_with_majority_original_rescue += int(rescue_rate >= 2 / 3 and reference_rate <= 1 / 3)
        for rescue_item, reference_item in zip(rescue, reference, strict=True):
            rescue_success = bool(rescue_item["success"])
            reference_success = bool(reference_item["success"])
            paired_alt_wins += int(rescue_success and not reference_success)
            paired_reference_wins += int(reference_success and not rescue_success)
            paired_both_success += int(rescue_success and reference_success)
            paired_both_failure += int(not rescue_success and not reference_success)
        root_comparisons.append(
            {
                "task_id": record["task_id"],
                "episode_id": record["episode_id"],
                "decision_step": record["decision_step"],
                "rescue_horizon": rescue_horizon,
                "rescue_success_rate": rescue_rate,
                "reference_success_rate": reference_rate,
                "success_rate_delta": rescue_rate - reference_rate,
            }
        )

    return {
        "num_roots": len(records),
        "num_branch_rollouts": len(records) * len(args.horizons) * args.repeats,
        "per_horizon": per_horizon,
        "repeat_zero_stored_outcome_match_count": repeat_zero_matches,
        "repeat_zero_stored_outcome_total": repeat_zero_total,
        "repeat_zero_stored_outcome_match_rate": (
            repeat_zero_matches / repeat_zero_total if repeat_zero_total else None
        ),
        "original_rescue_vs_reference": {
            "num_roots_with_candidate": len(root_comparisons),
            "roots_with_higher_repeated_rescue_rate": roots_with_more_stable_original_rescue,
            "roots_with_rescue_rate_at_least_two_thirds_and_reference_at_most_one_third": (
                roots_with_majority_original_rescue
            ),
            "paired_rescue_wins": paired_alt_wins,
            "paired_reference_wins": paired_reference_wins,
            "paired_both_success": paired_both_success,
            "paired_both_failure": paired_both_failure,
            "roots": root_comparisons,
        },
    }


def main(args: argparse.Namespace) -> None:
    if args.repeats <= 0 or args.repeat_seed_stride <= 0:
        raise ValueError("repeats and repeat-seed-stride must be positive.")
    if args.max_roots_per_task < 0:
        raise ValueError("max-roots-per-task must be non-negative.")
    horizons = sorted(set(args.horizons))
    if not horizons or any(horizon < 1 or horizon > 10 for horizon in horizons):
        raise ValueError("horizons must be a non-empty subset of H1-H10.")
    if args.reference_horizon not in horizons:
        raise ValueError("reference-horizon must be included in horizons.")
    args.horizons = horizons
    if args.continuation_policy == "current_student" and args.v2_budget_capacity <= 0:
        raise ValueError("v2-budget-capacity must be positive.")

    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset, include_physics=True)
    indices = _selected_indices(arrays, args)
    if not indices.size:
        raise ValueError("No fixable reference-failure roots matched the requested tasks.")
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    started = time.monotonic()
    records = [_replay_root(client, task_suite, arrays, int(index), args) for index in indices]
    report = {
        "status": "complete",
        "semantics": (
            "Repeated saved-root branch replay with one regenerated primary action chunk and controlled "
            "continuation-policy seeds. This audits label stability; it is not a closed-loop episode success rate."
        ),
        "dataset_inputs": list(args.dataset),
        "tasks": list(args.tasks),
        "reference_horizon": args.reference_horizon,
        "horizons": list(args.horizons),
        "repeats": args.repeats,
        "continuation_policy": args.continuation_policy,
        "elapsed_seconds": time.monotonic() - started,
        "summary": _summarize(records, args),
        "records": records,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    target = pathlib.Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
