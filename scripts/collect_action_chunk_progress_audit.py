"""Select action chunks by H3 privileged progress, then audit held-out success.

Each candidate score uses only the state reached after its short prefix. The
selected candidate is then evaluated with independently seeded fixed-H9
continuations from the same live snapshot. Terminal outcomes never enter the
candidate selection rule.
"""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import collect_action_chunk_candidate_oracle as candidate_audit
import collect_execution_horizon_counterfactuals as collector
import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import privileged_progress
from openpi.execution_horizon import v2


def build_parser() -> argparse.ArgumentParser:
    parser = candidate_audit.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--progress-minimum-margin",
        type=float,
        default=0.0005,
        help="Minimum raw dense-score advantage over candidate 0 before selecting an alternative.",
    )
    return parser


def _step_prefix(
    env: Any,
    snapshot: collector.SimulatorSnapshot,
    actions: np.ndarray,
    *,
    horizon: int,
) -> tuple[privileged_progress.LiberoGoalProgress, privileged_progress.LiberoGoalProgress, int]:
    observation = collector._restore_snapshot(env, snapshot)
    before = privileged_progress.score_libero_goal_progress(env, observation)
    steps = 0
    for action in np.asarray(actions)[:horizon]:
        try:
            observation, _, done, _ = env.step(np.asarray(action).tolist())
        except Exception as exc:
            if not libero_eval._is_terminated_episode_error(exc):
                raise
            break
        steps += 1
        if done:
            break
    after = privileged_progress.score_libero_goal_progress(env, observation)
    collector._restore_snapshot(env, snapshot)
    return before, after, steps


def _progress_candidates(
    env: Any,
    snapshot: collector.SimulatorSnapshot,
    candidates: np.ndarray,
    normalized: np.ndarray,
    selected: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int, int]:
    records = []
    for rank, index in enumerate(selected):
        before, after, steps = _step_prefix(
            env,
            snapshot,
            candidates[index],
            horizon=args.candidate_horizon,
        )
        prefix = slice(0, args.candidate_horizon)
        records.append(
            {
                "rank": rank,
                "sample_index": index,
                "policy_seed_offset": index,
                "normalized_prefix_rmse_vs_candidate0": float(
                    np.sqrt(np.mean((normalized[index, prefix, :7] - normalized[0, prefix, :7]) ** 2))
                ),
                "raw_prefix_rmse_vs_candidate0": float(
                    np.sqrt(np.mean((candidates[index, prefix] - candidates[0, prefix]) ** 2))
                ),
                "prefix_steps": steps,
                "progress_before": before.as_dict(),
                "progress_after": after.as_dict(),
                "progress_delta": after.score - before.score,
            }
        )

    best_rank = max(range(len(records)), key=lambda rank: records[rank]["progress_after"]["score"])
    candidate0_score = float(records[0]["progress_after"]["score"])
    best_score = float(records[best_rank]["progress_after"]["score"])
    chosen_rank = best_rank if best_rank != 0 and best_score >= candidate0_score + args.progress_minimum_margin else 0
    return records, best_rank, chosen_rank


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
        raise KeyError("Policy response has no mc_actions; restart a server containing the candidate-output change.")
    candidates = np.asarray(result["mc_actions"], dtype=np.float32)
    normalized = np.asarray(result["mc_actions_normalized"], dtype=np.float32)
    if candidates.ndim != 3 or normalized.ndim != 3:
        raise ValueError(f"Expected KxTxD candidates, got raw={candidates.shape}, normalized={normalized.shape}.")
    selected = candidate_audit._candidate_indices(normalized, args)
    if not selected or selected[0] != 0:
        raise ValueError("The candidate pool must include candidate 0 as rank 0.")

    progress_records, ungated_best_rank, chosen_rank = _progress_candidates(
        env,
        snapshot,
        candidates,
        normalized,
        selected,
        args,
    )
    chosen_index = selected[chosen_rank]
    primary_actions = np.asarray(result["actions"], dtype=np.float32)
    arms: dict[str, list[dict[str, Any]]] = {
        "reference_candidate0": [],
        "candidate0_short": [],
        "progress_selected_short": [],
    }
    for repeat_index in range(args.repeats):
        branch_seed = root_seed + repeat_index * args.repeat_seed_stride
        reference = candidate_audit._branch_outcome(
            env,
            snapshot,
            candidates[0],
            horizon=args.reference_horizon,
            root_step=root_step,
            episode_step_limit=episode_step_limit,
            branch_seed=branch_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        candidate0 = candidate_audit._branch_outcome(
            env,
            snapshot,
            candidates[0],
            horizon=args.candidate_horizon,
            root_step=root_step,
            episode_step_limit=episode_step_limit,
            branch_seed=branch_seed,
            task_description=task_description,
            args=args,
            client=client,
        )
        if chosen_rank == 0:
            progress_selected = dict(candidate0)
            progress_selected["reused_candidate0_branch"] = True
        else:
            progress_selected = candidate_audit._branch_outcome(
                env,
                snapshot,
                candidates[chosen_index],
                horizon=args.candidate_horizon,
                root_step=root_step,
                episode_step_limit=episode_step_limit,
                branch_seed=branch_seed,
                task_description=task_description,
                args=args,
                client=client,
            )
            progress_selected["reused_candidate0_branch"] = False

        outcomes = {
            "reference_candidate0": (reference, 0, args.reference_horizon),
            "candidate0_short": (candidate0, 0, args.candidate_horizon),
            "progress_selected_short": (progress_selected, chosen_index, args.candidate_horizon),
        }
        for arm_name, (outcome, candidate_index, horizon) in outcomes.items():
            outcome.update(
                {
                    "repeat_index": repeat_index,
                    "continuation_seed": branch_seed,
                    "candidate_rank": 0 if arm_name != "progress_selected_short" else chosen_rank,
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
        "candidate_pool_selection": args.candidate_selection if args.candidate_indices is None else "explicit",
        "candidate_horizon": args.candidate_horizon,
        "reference_horizon": args.reference_horizon,
        "repeats": args.repeats,
        "progress_minimum_margin": args.progress_minimum_margin,
        "risk": candidate_audit._risk_record(risk, args),
        "candidate0_action_max_abs_difference_vs_primary": float(np.max(np.abs(candidates[0] - primary_actions))),
        "progress_candidates": progress_records,
        "ungated_best_progress_rank": ungated_best_rank,
        "selected_progress_rank": chosen_rank,
        "selected_progress_candidate_index": chosen_index,
        "selected_alternative": chosen_rank != 0,
        "arms": arms,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_roots": 0, "num_logical_branch_rollouts": 0}
    arm_names = ["reference_candidate0", "candidate0_short", "progress_selected_short"]
    per_arm = {}
    for arm_name in arm_names:
        outcomes = [outcome for record in records for outcome in record["arms"][arm_name]]
        success_count = sum(bool(outcome["success"]) for outcome in outcomes)
        per_arm[arm_name] = {
            "success_count": success_count,
            "num_trials": len(outcomes),
            "success_rate": success_count / len(outcomes),
            "average_remaining_calls": float(np.mean([outcome["remaining_calls"] for outcome in outcomes])),
            "average_remaining_steps": float(np.mean([outcome["remaining_steps"] for outcome in outcomes])),
        }

    def outcome_vector(arm_name: str, subset: list[dict[str, Any]] = records) -> list[bool]:
        return [bool(outcome["success"]) for record in subset for outcome in record["arms"][arm_name]]

    per_task = {}
    for task_id in sorted({int(record["task_id"]) for record in records}):
        subset = [record for record in records if int(record["task_id"]) == task_id]
        per_task[str(task_id)] = {
            "num_roots": len(subset),
            "selected_alternative_roots": sum(bool(record["selected_alternative"]) for record in subset),
            "progress_selected_vs_candidate0": candidate_audit._paired_counts(
                outcome_vector("progress_selected_short", subset),
                outcome_vector("candidate0_short", subset),
            ),
            "per_arm_success_count": {arm_name: sum(outcome_vector(arm_name, subset)) for arm_name in arm_names},
            "num_trials": len(subset) * subset[0]["repeats"],
        }

    root_comparisons = []
    for record in records:
        candidate0 = [bool(outcome["success"]) for outcome in record["arms"]["candidate0_short"]]
        chosen = [bool(outcome["success"]) for outcome in record["arms"]["progress_selected_short"]]
        root_comparisons.append(
            {
                "task_id": record["task_id"],
                "episode_id": record["episode_id"],
                "decision_step": record["decision_step"],
                "selected_progress_rank": record["selected_progress_rank"],
                "selected_progress_candidate_index": record["selected_progress_candidate_index"],
                "candidate0_after_score": record["progress_candidates"][0]["progress_after"]["score"],
                "selected_after_score": record["progress_candidates"][record["selected_progress_rank"]][
                    "progress_after"
                ]["score"],
                "candidate0_success_count": sum(candidate0),
                "progress_selected_success_count": sum(chosen),
            }
        )

    selected_count = sum(bool(record["selected_alternative"]) for record in records)
    actual_branch_rollouts = sum(
        2 * record["repeats"] + int(record["selected_alternative"]) * record["repeats"] for record in records
    )
    return {
        "num_roots": len(records),
        "num_logical_branch_rollouts": len(records) * records[0]["repeats"] * len(arm_names),
        "num_actual_branch_rollouts": actual_branch_rollouts,
        "selected_alternative_roots": selected_count,
        "selected_alternative_fraction": selected_count / len(records),
        "per_arm": per_arm,
        "progress_selected_vs_candidate0_same_h": candidate_audit._paired_counts(
            outcome_vector("progress_selected_short"),
            outcome_vector("candidate0_short"),
        ),
        "progress_selected_short_h_vs_reference_h": candidate_audit._paired_counts(
            outcome_vector("progress_selected_short"),
            outcome_vector("reference_candidate0"),
        ),
        "candidate0_short_h_vs_reference_h": candidate_audit._paired_counts(
            outcome_vector("candidate0_short"),
            outcome_vector("reference_candidate0"),
        ),
        "mean_selected_score_advantage_over_candidate0": float(
            np.mean(
                [
                    record["progress_candidates"][record["selected_progress_rank"]]["progress_after"]["score"]
                    - record["progress_candidates"][0]["progress_after"]["score"]
                    for record in records
                ]
            )
        ),
        "root_comparisons": root_comparisons,
        "per_task": per_task,
    }


def _validate_args(args: argparse.Namespace) -> list[int]:
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
    if args.progress_minimum_margin < 0:
        raise ValueError("progress-minimum-margin must be non-negative.")
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
    return episode_ids


def main(args: argparse.Namespace) -> None:
    episode_ids = _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "progress_audit_records.jsonl"
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
                env, task_description = libero_eval._get_libero_env(
                    task,
                    libero_eval.LIBERO_ENV_RESOLUTION,
                    args.seed,
                )
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
                            observation,
                            task_description,
                            args.resize_size,
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
            "Candidate selection uses only BDDL predicate and MuJoCo state progress after H3; "
            "paired terminal success comes from independently seeded fixed-H9 continuations."
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
            "progress_minimum_margin": args.progress_minimum_margin,
            "continuation_policy": args.continuation_policy,
        },
        "summary": _summarize(records),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    (output_dir / "summary.json").write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
