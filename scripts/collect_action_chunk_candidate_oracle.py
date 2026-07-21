"""Audit action-chunk candidate headroom from the same live LIBERO snapshot.

The audit generates K action chunks with one shared policy prefix, selects a
small outcome-independent subset, and compares paired simulator branches:

* candidate 0 executed for a long reference horizon;
* candidate 0 executed for the short recovery horizon; and
* alternative candidates executed for that same short horizon.

This separates gains caused by early replanning from gains caused by changing
the action chunk.  Candidate-oracle metrics use future branch outcomes and are
diagnostic upper bounds, not deployable policy results.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import collect_execution_horizon_counterfactuals as collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument(
        "--episode-ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit initial-state IDs; overrides num-trials-per-task.",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=10)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument(
        "--candidate-selection",
        choices=("farthest", "first"),
        default="farthest",
        help="Outcome-independent selection from the batched MC chunks.",
    )
    parser.add_argument(
        "--candidate-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit indices; overrides candidate-count/selection.",
    )
    parser.add_argument("--candidate-horizon", type=int, default=3)
    parser.add_argument("--reference-horizon", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeat-seed-stride", type=int, default=20_000_000)
    parser.add_argument("--root-stride-calls", type=int, default=1)
    parser.add_argument(
        "--root-call-offset-cycle",
        type=int,
        default=20,
        help="Episode e audits policy-call e modulo this value.",
    )
    parser.add_argument("--max-roots-per-episode", type=int, default=1)
    parser.add_argument(
        "--minimum-root-max-fused-risk",
        type=float,
        default=None,
        help="Optional risk trigger; skipped scheduled roots do not count toward the root limit.",
    )
    parser.add_argument("--v2-risk-threshold", type=float, default=1.5)
    parser.add_argument("--v2-final-weight", type=float, default=0.5)
    parser.add_argument("--v2-action-cot-weight", type=float, default=0.5)
    parser.add_argument(
        "--v2-budget-capacity",
        type=float,
        default=12.0,
        help="Shared branch-runner normalization constant; fixed-H9 does not update this budget.",
    )
    # collector._run_branch reads this only when video capture is enabled.  The
    # candidate audit never records videos, but retaining the field keeps the
    # shared branch runner's argument contract explicit.
    parser.add_argument("--debug-video-stride", type=int, default=5)
    parser.set_defaults(continuation_policy="fixed_h9")
    return parser


def _candidate_indices(normalized: np.ndarray, args: argparse.Namespace) -> list[int]:
    sample_count = normalized.shape[0]
    if args.candidate_indices is not None:
        indices = list(dict.fromkeys(args.candidate_indices))
        if 0 not in indices:
            indices.insert(0, 0)
        return indices
    count = min(args.candidate_count, sample_count)
    if args.candidate_selection == "first":
        return list(range(count))

    prefix = np.asarray(normalized[:, : args.candidate_horizon, :7], dtype=np.float64)
    selected = [0]
    while len(selected) < count:
        remaining = [index for index in range(sample_count) if index not in selected]
        min_distances = []
        for index in remaining:
            distances = [float(np.sqrt(np.mean((prefix[index] - prefix[chosen]) ** 2))) for chosen in selected]
            min_distances.append(min(distances))
        selected.append(remaining[int(np.argmax(min_distances))])
    return selected


def _risk_record(risk: dict[str, np.ndarray | int], args: argparse.Namespace) -> dict[str, Any]:
    event_index = int(risk["event_index"])
    return {
        "event_index": event_index,
        "teacher_raw_h": v2.event_horizon(event_index, range(3, 11)),
        "max_fused_risk": float(np.max(risk["fused_risk"])),
        "final_risk": np.asarray(risk["final_risk"], dtype=np.float64).tolist(),
        "action_cot_risk": np.asarray(risk["action_cot_risk"], dtype=np.float64).tolist(),
        "fused_risk": np.asarray(risk["fused_risk"], dtype=np.float64).tolist(),
        "risk_threshold": args.v2_risk_threshold,
    }


def _branch_outcome(
    env: Any,
    snapshot: collector.SimulatorSnapshot,
    actions: np.ndarray,
    *,
    horizon: int,
    root_step: int,
    episode_step_limit: int,
    branch_seed: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
) -> dict[str, Any]:
    success, timeout, steps, calls, _ = collector._run_branch(
        env,
        snapshot,
        actions,
        forced_horizon=horizon,
        root_step=root_step,
        episode_step_limit=episode_step_limit,
        root_seed=branch_seed,
        task_description=task_description,
        args=args,
        client=client,
        root_budget_state=v2.EpisodeBudgetState(balance=0.0),
        capture_video=False,
    )
    return {
        "success": success,
        "timeout": timeout,
        "remaining_steps": steps,
        "remaining_calls": calls,
    }


def _evaluate_root(
    env: Any,
    snapshot: collector.SimulatorSnapshot,
    result: dict[str, Any],
    *,
    task_id: int,
    episode_id: int,
    root_step: int,
    episode_step_limit: int,
    root_seed: int,
    task_description: str,
    args: argparse.Namespace,
    client: websocket_policy.WebsocketClientPolicy,
    risk: dict[str, np.ndarray | int],
) -> dict[str, Any]:
    if "mc_actions" not in result:
        raise KeyError(
            "Policy response has no mc_actions. Pull the policy-side candidate-output change and restart the server."
        )
    candidates = np.asarray(result["mc_actions"], dtype=np.float32)
    normalized = np.asarray(result["mc_actions_normalized"], dtype=np.float32)
    if candidates.ndim != 3 or normalized.ndim != 3:
        raise ValueError(f"Expected KxTxD candidates, got raw={candidates.shape}, normalized={normalized.shape}.")
    if candidates.shape[0] != args.teacher_samples or normalized.shape[0] != args.teacher_samples:
        raise ValueError("Policy returned a different candidate count from teacher-samples.")
    selected = _candidate_indices(normalized, args)
    if not selected or selected[0] != 0 or any(index < 0 or index >= args.teacher_samples for index in selected):
        raise ValueError("candidate-indices must be valid teacher sample indices and include candidate 0.")

    primary_actions = np.asarray(result["actions"], dtype=np.float32)
    candidate_metadata = []
    for rank, index in enumerate(selected):
        prefix = slice(0, args.candidate_horizon)
        candidate_metadata.append(
            {
                "rank": rank,
                "sample_index": index,
                "policy_seed": root_seed + index,
                "normalized_prefix_rmse_vs_candidate0": float(
                    np.sqrt(np.mean((normalized[index, prefix, :7] - normalized[0, prefix, :7]) ** 2))
                ),
                "raw_prefix_rmse_vs_candidate0": float(
                    np.sqrt(np.mean((candidates[index, prefix] - candidates[0, prefix]) ** 2))
                ),
            }
        )

    arms: dict[str, list[dict[str, Any]]] = {"reference_candidate0": []}
    for rank in range(len(selected)):
        arms[f"candidate_rank{rank}"] = []

    for repeat_index in range(args.repeats):
        branch_seed = root_seed + repeat_index * args.repeat_seed_stride
        arm_specs = [("reference_candidate0", 0, args.reference_horizon)]
        arm_specs.extend(
            (f"candidate_rank{rank}", index, args.candidate_horizon) for rank, index in enumerate(selected)
        )
        for arm_name, candidate_index, horizon in arm_specs:
            outcome = _branch_outcome(
                env,
                snapshot,
                candidates[candidate_index],
                horizon=horizon,
                root_step=root_step,
                episode_step_limit=episode_step_limit,
                branch_seed=branch_seed,
                task_description=task_description,
                args=args,
                client=client,
            )
            outcome.update(
                {
                    "repeat_index": repeat_index,
                    "continuation_seed": branch_seed,
                    "candidate_rank": (
                        0 if arm_name == "reference_candidate0" else int(arm_name.removeprefix("candidate_rank"))
                    ),
                    "candidate_index": candidate_index,
                    "forced_horizon": horizon,
                }
            )
            arms[arm_name].append(outcome)
            print(
                json.dumps(
                    {
                        "task": task_id,
                        "episode": episode_id,
                        "root_step": root_step,
                        "arm": arm_name,
                        **outcome,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return {
        "schema_version": 1,
        "task_id": task_id,
        "episode_id": episode_id,
        "decision_step": root_step,
        "root_seed": root_seed,
        "candidate_selection": args.candidate_selection if args.candidate_indices is None else "explicit",
        "candidate_horizon": args.candidate_horizon,
        "reference_horizon": args.reference_horizon,
        "repeats": args.repeats,
        "risk": _risk_record(risk, args),
        "candidate0_action_max_abs_difference_vs_primary": float(np.max(np.abs(candidates[0] - primary_actions))),
        "candidates": candidate_metadata,
        "arms": arms,
    }


def _paired_counts(left: list[bool], right: list[bool]) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("Paired outcome vectors must have equal lengths.")
    return {
        "left_only_success": sum(a and not b for a, b in zip(left, right, strict=True)),
        "right_only_success": sum(b and not a for a, b in zip(left, right, strict=True)),
        "both_success": sum(a and b for a, b in zip(left, right, strict=True)),
        "both_failure": sum(not a and not b for a, b in zip(left, right, strict=True)),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_roots": 0, "num_branch_rollouts": 0}
    candidate_arm_names = [name for name in records[0]["arms"] if name.startswith("candidate_rank")]
    all_arm_names = ["reference_candidate0", *candidate_arm_names]
    per_arm = {}
    for name in all_arm_names:
        outcomes = [outcome for record in records for outcome in record["arms"][name]]
        successes = sum(bool(outcome["success"]) for outcome in outcomes)
        per_arm[name] = {
            "success_count": successes,
            "num_trials": len(outcomes),
            "success_rate": successes / len(outcomes),
            "average_remaining_calls": float(np.mean([outcome["remaining_calls"] for outcome in outcomes])),
            "average_remaining_steps": float(np.mean([outcome["remaining_steps"] for outcome in outcomes])),
        }

    action_oracle: list[bool] = []
    full_oracle: list[bool] = []
    candidate0_short: list[bool] = []
    reference: list[bool] = []
    empirical_best_successes = 0
    stable_action_headroom_roots = 0
    stable_overall_headroom_roots = 0
    root_comparisons = []
    for record in records:
        reference_root = [bool(item["success"]) for item in record["arms"]["reference_candidate0"]]
        candidate_root = [[bool(item["success"]) for item in record["arms"][name]] for name in candidate_arm_names]
        candidate0_root = candidate_root[0]
        alternative_root = candidate_root[1:]
        per_repeat_any_alternative = (
            [any(outcomes[repeat] for outcomes in alternative_root) for repeat in range(record["repeats"])]
            if alternative_root
            else list(candidate0_root)
        )
        per_repeat_any_candidate = [
            any(outcomes[repeat] for outcomes in candidate_root) for repeat in range(record["repeats"])
        ]
        reference.extend(reference_root)
        candidate0_short.extend(candidate0_root)
        action_oracle.extend(per_repeat_any_alternative)
        full_oracle.extend(per_repeat_any_candidate)

        candidate_success_counts = [sum(outcomes) for outcomes in candidate_root]
        best_rank = int(np.argmax(candidate_success_counts))
        empirical_best_successes += candidate_success_counts[best_rank]
        reference_rate = float(np.mean(reference_root))
        candidate0_rate = float(np.mean(candidate0_root))
        best_rate = candidate_success_counts[best_rank] / record["repeats"]
        best_alternative_rate = (
            max(sum(outcomes) for outcomes in alternative_root) / record["repeats"]
            if alternative_root
            else candidate0_rate
        )
        stable_action_headroom_roots += int(best_alternative_rate >= 2 / 3 and candidate0_rate <= 1 / 3)
        stable_overall_headroom_roots += int(best_rate >= 2 / 3 and reference_rate <= 1 / 3)
        root_comparisons.append(
            {
                "task_id": record["task_id"],
                "episode_id": record["episode_id"],
                "decision_step": record["decision_step"],
                "reference_h10_success_rate": reference_rate,
                "candidate0_short_h_success_rate": candidate0_rate,
                "empirical_best_candidate_rank": best_rank,
                "empirical_best_candidate_success_rate": best_rate,
                "best_alternative_candidate_success_rate": best_alternative_rate,
            }
        )

    trial_count = len(reference)
    task_summary = {}
    for task_id in sorted({int(record["task_id"]) for record in records}):
        task_records = [record for record in records if int(record["task_id"]) == task_id]
        task_summary[str(task_id)] = {
            name: {
                "success_count": sum(
                    bool(outcome["success"]) for record in task_records for outcome in record["arms"][name]
                ),
                "num_trials": len(task_records) * task_records[0]["repeats"],
            }
            for name in all_arm_names
        }

    return {
        "num_roots": len(records),
        "num_branch_rollouts": sum(len(outcomes) for record in records for outcomes in record["arms"].values()),
        "per_arm": per_arm,
        "candidate0_short_h_vs_reference_h": _paired_counts(candidate0_short, reference),
        "hindsight_any_alternative_chunk_vs_candidate0_same_h": _paired_counts(action_oracle, candidate0_short),
        "hindsight_any_candidate_short_h_vs_reference_h": _paired_counts(full_oracle, reference),
        "hindsight_any_candidate_short_h_success_rate": sum(full_oracle) / trial_count,
        "in_sample_root_empirical_best_candidate_success_rate": empirical_best_successes / trial_count,
        "roots_with_best_alternative_rate_at_least_two_thirds_and_candidate0_at_most_one_third": (
            stable_action_headroom_roots
        ),
        "roots_with_best_candidate_rate_at_least_two_thirds_and_reference_at_most_one_third": (
            stable_overall_headroom_roots
        ),
        "root_comparisons": root_comparisons,
        "per_task": task_summary,
    }


def main(args: argparse.Namespace) -> None:
    positive_values = (
        args.teacher_samples,
        args.candidate_count,
        args.candidate_horizon,
        args.reference_horizon,
        args.repeats,
        args.repeat_seed_stride,
        args.root_stride_calls,
        args.root_call_offset_cycle,
        args.action_cot_denoising_steps,
        args.v2_budget_capacity,
    )
    if any(value <= 0 for value in positive_values):
        raise ValueError("Sample, horizon, repeat, stride, cycle, and denoising values must be positive.")
    if args.candidate_count > args.teacher_samples:
        raise ValueError("candidate-count cannot exceed teacher-samples.")
    if args.candidate_horizon > 10 or args.reference_horizon > 10:
        raise ValueError("candidate-horizon and reference-horizon must be within H1-H10.")
    if args.candidate_indices is not None and any(
        index < 0 or index >= args.teacher_samples for index in args.candidate_indices
    ):
        raise ValueError("candidate-indices must be valid batched teacher sample indices.")
    episode_ids = (
        list(range(args.num_trials_per_task)) if args.episode_ids is None else list(dict.fromkeys(args.episode_ids))
    )
    if not episode_ids or any(episode_id < 0 for episode_id in episode_ids):
        raise ValueError("episode-ids must contain non-negative values.")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "candidate_oracle_records.jsonl"
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
    records = []
    scheduled_roots_skipped_by_risk = 0
    started = time.monotonic()
    with records_path.open("w", encoding="utf-8") as writer:
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
                    while not done and step < episode_step_limit:
                        if args.max_roots_per_episode and roots_this_episode >= args.max_roots_per_episode:
                            break
                        scheduled_root = (
                            decision_index >= root_call_offset
                            and (decision_index - root_call_offset) % args.root_stride_calls == 0
                        )
                        root_seed = args.seed + task_id * 1_000_000 + episode_id * 10_000 + step
                        policy_input = libero_eval._observation_to_policy_input(
                            observation, task_description, args.resize_size
                        )
                        result = collector._policy_request(
                            client,
                            policy_input,
                            seed=root_seed,
                            args=args,
                            teacher=scheduled_root,
                        )
                        primary_actions = np.asarray(result["actions"], dtype=np.float32)
                        if scheduled_root:
                            risk = v2.risk_targets_from_normalized_mc(
                                result["mc_coarse_actions_normalized"],
                                result["mc_actions_normalized"],
                                config=risk_config,
                            )
                            max_fused_risk = float(np.max(risk["fused_risk"]))
                            accepted = (
                                args.minimum_root_max_fused_risk is None
                                or max_fused_risk >= args.minimum_root_max_fused_risk
                            )
                            if accepted:
                                snapshot = collector._capture_snapshot(env)
                                record = _evaluate_root(
                                    env,
                                    snapshot,
                                    result,
                                    task_id=task_id,
                                    episode_id=episode_id,
                                    root_step=step,
                                    episode_step_limit=episode_step_limit,
                                    root_seed=root_seed,
                                    task_description=task_description,
                                    args=args,
                                    client=client,
                                    risk=risk,
                                )
                                records.append(record)
                                writer.write(json.dumps(record, sort_keys=True) + "\n")
                                writer.flush()
                                roots_this_episode += 1
                                observation = collector._restore_snapshot(env, snapshot)
                            else:
                                scheduled_roots_skipped_by_risk += 1

                        if args.max_roots_per_episode and roots_this_episode >= args.max_roots_per_episode:
                            break
                        rollout_horizon = min(9, len(primary_actions))
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
                        decision_index += 1
                finally:
                    libero_eval._safe_close_env(env)

    report = {
        "status": "complete",
        "semantics": (
            "Same-live-snapshot paired branch audit with fixed-H9 continuation. "
            "Hindsight oracle and in-sample best-candidate metrics are diagnostic upper bounds, "
            "not deployable closed-loop success rates."
        ),
        "elapsed_seconds": time.monotonic() - started,
        "records_path": str(records_path),
        "scheduled_roots_skipped_by_risk": scheduled_roots_skipped_by_risk,
        "config": {
            "task_suite_name": args.task_suite_name,
            "task_start": args.task_start,
            "max_tasks": args.max_tasks,
            "episode_ids": episode_ids,
            "teacher_samples": args.teacher_samples,
            "candidate_count": args.candidate_count,
            "candidate_selection": args.candidate_selection,
            "candidate_indices": args.candidate_indices,
            "candidate_horizon": args.candidate_horizon,
            "reference_horizon": args.reference_horizon,
            "repeats": args.repeats,
            "repeat_seed_stride": args.repeat_seed_stride,
            "root_call_offset_cycle": args.root_call_offset_cycle,
            "minimum_root_max_fused_risk": args.minimum_root_max_fused_risk,
            "continuation_policy": args.continuation_policy,
        },
        "summary": _summarize(records),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
