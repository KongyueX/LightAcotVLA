"""Closed-loop LIBERO evaluation for Budgeted Event V2-P execution horizons."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import time
from typing import Any

import eval_libero_action_cot_pruning as libero_eval
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

from openpi.execution_horizon import v2

MODES = (
    "original",
    "fixed_h9",
    "exact_batched_mc_v2",
    "v2_distilled",
    "v2_value_refined",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=20)
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--original-horizon", type=int, default=5)
    parser.add_argument("--fixed-horizon", type=int, default=9)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=20)
    parser.add_argument("--v2-min-horizon", type=int, default=3)
    parser.add_argument("--v2-risk-threshold", type=float, default=1.5)
    parser.add_argument("--v2-final-weight", type=float, default=0.5)
    parser.add_argument("--v2-action-cot-weight", type=float, default=0.5)
    parser.add_argument("--v2-final-risk-threshold", type=float, default=None)
    parser.add_argument("--v2-action-cot-risk-threshold", type=float, default=None)
    parser.add_argument("--v2-target-average-horizon", type=float, default=9.0)
    parser.add_argument("--v2-initial-budget", type=float, default=6.0)
    parser.add_argument("--v2-budget-capacity", type=float, default=12.0)
    parser.add_argument("--value-candidates", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--q-min-success-probability", type=float, default=0.90)
    parser.add_argument("--q-max-timeout-probability", type=float, default=0.20)
    parser.add_argument("--q-risk-slack-steps", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume a matching interrupted evaluation from its per-episode CSV journal.",
    )
    return parser


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _request(
    client: websocket_policy.WebsocketClientPolicy,
    element: dict[str, Any],
    *,
    mode: str,
    seed: int,
    previous_actions: np.ndarray | None,
    previous_horizon: int,
    budget_fraction: float,
    episode_progress: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    request = {
        **element,
        "policy_seed": np.asarray(seed, dtype=np.int64),
        "profile_policy_timing": np.asarray(1, dtype=np.bool_),
        "action_cot_denoising_steps": np.asarray(args.action_cot_denoising_steps, dtype=np.int32),
    }
    if mode == "exact_batched_mc_v2":
        request["batched_mc_samples"] = np.asarray(args.teacher_samples, dtype=np.int32)
    if mode in {"v2_distilled", "v2_value_refined"}:
        request.update(
            {
                "run_execution_horizon_predictor": np.asarray(1, dtype=np.bool_),
                "execution_horizon_previous_actions": (
                    np.asarray(previous_actions, dtype=np.float32)
                    if previous_actions is not None
                    else np.zeros((10, 7), dtype=np.float32)
                ),
                "execution_horizon_previous_h": np.asarray(previous_horizon, dtype=np.int32),
                "execution_horizon_budget_balance": np.asarray(budget_fraction, dtype=np.float32),
                "execution_horizon_episode_progress": np.asarray(episode_progress, dtype=np.float32),
                "execution_horizon_previous_valid": np.asarray(previous_actions is not None),
            }
        )
    started = time.perf_counter()
    result = client.infer(request)
    wall_ms = (time.perf_counter() - started) * 1000.0
    policy_timing = result.get("policy_timing", {})
    server_timing = result.get("server_timing", {})
    return result, {
        "wall_ms": wall_ms,
        "policy_ms": float(policy_timing.get("infer_ms", np.nan)),
        "server_ms": float(server_timing.get("infer_ms", wall_ms)),
        "predictor_ms": float(policy_timing.get("execution_horizon_predictor_ms", np.nan)),
        "batched_teacher_ms": float(policy_timing.get("batched_mc_teacher_ms", np.nan)),
    }


def _risk_config(args: argparse.Namespace) -> v2.V2RiskConfig:
    return v2.V2RiskConfig(
        risk_threshold=args.v2_risk_threshold,
        final_weight=args.v2_final_weight,
        action_cot_weight=args.v2_action_cot_weight,
        final_risk_threshold=args.v2_final_risk_threshold,
        action_cot_risk_threshold=args.v2_action_cot_risk_threshold,
    )


def _select_horizon(
    mode: str,
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    budget_state: v2.EpisodeBudgetState,
) -> tuple[int, dict[str, Any]]:
    if mode == "original":
        return args.original_horizon, {"raw_horizon": args.original_horizon, "budget_limited": 0.0}
    if mode == "fixed_h9":
        return args.fixed_horizon, {"raw_horizon": args.fixed_horizon, "budget_limited": 0.0}

    risk_config = _risk_config(args)
    entropy_candidates = list(range(args.v2_min_horizon, 11))
    if mode == "exact_batched_mc_v2":
        risk = v2.risk_targets_from_normalized_mc(
            result["mc_coarse_actions_normalized"],
            result["mc_actions_normalized"],
            config=risk_config,
        )
        final_risk = np.asarray(risk["final_risk"])
        action_cot_risk = np.asarray(risk["action_cot_risk"])
        fused_risk = np.asarray(risk["fused_risk"])
    else:
        final_risk = np.asarray(result["execution_horizon_final_risk"], dtype=np.float64)
        action_cot_risk = np.asarray(result["execution_horizon_action_cot_risk"], dtype=np.float64)
        fused_risk = np.asarray(result["execution_horizon_fused_risk"], dtype=np.float64)

    entropy_raw_horizon, event_mask = v2.distilled_raw_horizon(
        final_risk,
        action_cot_risk,
        fused_risk,
        candidates=entropy_candidates,
        config=risk_config,
    )
    raw_horizon = entropy_raw_horizon
    candidates = entropy_candidates
    q_info: dict[str, Any] = {}
    if mode == "v2_value_refined":
        success_probability = _sigmoid(result["execution_horizon_success_logits"])
        timeout_probability = _sigmoid(result["execution_horizon_timeout_logits"])
        candidates = sorted(set(args.value_candidates))
        raw_horizon, filters = v2.value_refined_raw_horizon(
            entropy_raw_horizon=entropy_raw_horizon,
            success_probability=success_probability,
            timeout_probability=timeout_probability,
            fused_risk=fused_risk,
            config=v2.ValueRefinementConfig(
                minimum_success_probability=args.q_min_success_probability,
                maximum_timeout_probability=args.q_max_timeout_probability,
                risk_threshold=args.v2_risk_threshold,
                risk_slack_steps=args.q_risk_slack_steps,
                candidates=tuple(candidates),
            ),
        )
        q_info = {
            "success_probability": success_probability.tolist(),
            "timeout_probability": timeout_probability.tolist(),
            "q_eligible": np.asarray(filters["eligible"], dtype=np.int8).tolist(),
            "predicted_remaining_calls": np.asarray(
                result["execution_horizon_remaining_calls"], dtype=np.float64
            ).tolist(),
            "predicted_remaining_steps": np.asarray(
                result["execution_horizon_remaining_steps"], dtype=np.float64
            ).tolist(),
        }

    final_horizon, budget_info = v2.apply_episode_budget(
        raw_horizon,
        candidates,
        config=v2.EpisodeBudgetConfig(
            target_average_horizon=args.v2_target_average_horizon,
            capacity=args.v2_budget_capacity,
        ),
        state=budget_state,
    )
    raw_h_prediction = None
    if "execution_horizon_raw_h_logits" in result:
        raw_h_prediction = int(np.argmax(result["execution_horizon_raw_h_logits"]) + 1)
    return final_horizon, {
        "raw_horizon": raw_horizon,
        "entropy_raw_horizon": entropy_raw_horizon,
        "raw_h_prediction": raw_h_prediction,
        "event_mask": np.asarray(event_mask, dtype=np.int8).tolist(),
        "final_risk": final_risk.tolist(),
        "action_cot_risk": action_cot_risk.tolist(),
        "fused_risk": fused_risk.tolist(),
        **budget_info,
        **q_info,
    }


def _warmup(
    client: websocket_policy.WebsocketClientPolicy,
    task_suite,
    args: argparse.Namespace,
) -> None:
    if args.warmup_requests <= 0:
        return
    task = task_suite.get_task(args.task_start)
    states = task_suite.get_task_init_states(args.task_start)
    env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        observation = env.set_init_state(states[args.initial_state_offset % len(states)])
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            if done:
                break
        element = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
        for mode in args.modes:
            for repeat in range(args.warmup_requests):
                _request(
                    client,
                    element,
                    mode=mode,
                    seed=args.seed + repeat,
                    previous_actions=None,
                    previous_horizon=10,
                    budget_fraction=args.v2_initial_budget / args.v2_budget_capacity,
                    episode_progress=0.0,
                    args=args,
                )
    finally:
        libero_eval._safe_close_env(env)


def _run_episode(
    *,
    mode: str,
    task_id: int,
    episode: int,
    task_suite,
    client: websocket_policy.WebsocketClientPolicy,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = task_suite.get_task(task_id)
    states = task_suite.get_task_init_states(task_id)
    state_id = (args.initial_state_offset + episode) % len(states)
    env, task_description = libero_eval._get_libero_env(task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed)
    timings: list[dict[str, float]] = []
    decisions: list[dict[str, Any]] = []
    horizons: list[int] = []
    policy_calls = 0
    sampled_chunks = 0
    step = 0
    success = False
    previous_actions: np.ndarray | None = None
    previous_horizon = 10
    budget_state = v2.EpisodeBudgetState(balance=min(args.v2_initial_budget, args.v2_budget_capacity))
    max_steps = libero_eval._max_steps(args.task_suite_name)
    try:
        env.reset()
        observation = env.set_init_state(states[state_id])
        environment_horizon = libero_eval._env_horizon(env)
        episode_step_limit = max_steps + args.num_steps_wait
        if environment_horizon is not None:
            episode_step_limit = min(episode_step_limit, environment_horizon)
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                success = True
                break

        while not success and step < episode_step_limit:
            element = libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
            request_seed = args.seed + task_id * 1_000_000 + episode * 10_000 + step
            result, timing = _request(
                client,
                element,
                mode=mode,
                seed=request_seed,
                previous_actions=previous_actions,
                previous_horizon=previous_horizon,
                budget_fraction=budget_state.balance / args.v2_budget_capacity,
                episode_progress=step / max(episode_step_limit, 1),
                args=args,
            )
            policy_calls += 1
            sampled_chunks += args.teacher_samples if mode == "exact_batched_mc_v2" else 1
            timings.append(timing)
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            horizon, selector_info = _select_horizon(mode, result, args=args, budget_state=budget_state)
            horizon = min(horizon, len(action_chunk), episode_step_limit - step)
            if horizon <= 0:
                break
            horizons.append(horizon)
            decisions.append(
                {
                    "mode": mode,
                    "task_id": task_id,
                    "episode": episode,
                    "initial_state_id": state_id,
                    "environment_step": step,
                    "execution_horizon": horizon,
                    "wall_ms": timing["wall_ms"],
                    "policy_ms": timing["policy_ms"],
                    "server_ms": timing["server_ms"],
                    "predictor_ms": timing["predictor_ms"],
                    "batched_teacher_ms": timing["batched_teacher_ms"],
                    "selector_json": json.dumps(selector_info, separators=(",", ":")),
                }
            )
            previous_actions = action_chunk
            previous_horizon = horizon
            for action in action_chunk[:horizon]:
                try:
                    observation, _, done, _ = env.step(np.asarray(action).tolist())
                except Exception as exc:
                    if not libero_eval._is_terminated_episode_error(exc):
                        raise
                    done = libero_eval._env_success(env)
                step += 1
                if done or libero_eval._env_success(env):
                    success = True
                    break
    finally:
        libero_eval._safe_close_env(env)

    def total(field: str) -> float:
        values = [timing[field] for timing in timings if np.isfinite(timing[field])]
        return float(np.sum(values)) if values else float("nan")

    histogram = collections.Counter(horizons)
    row = {
        "mode": mode,
        "task_suite": args.task_suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "episode": episode,
        "initial_state_id": state_id,
        "success": int(success),
        "timeout": int(not success),
        "steps": step,
        "policy_calls": policy_calls,
        "sampled_action_chunks": sampled_chunks,
        "avg_h": float(np.mean(horizons)) if horizons else float("nan"),
        "h_distribution_json": json.dumps(dict(sorted(histogram.items()))),
        "actual_wall_total_ms": total("wall_ms"),
        "actual_policy_total_ms": total("policy_ms"),
        "actual_server_total_ms": total("server_ms"),
        "actual_predictor_total_ms": total("predictor_ms"),
        "actual_batched_teacher_total_ms": total("batched_teacher_ms"),
    }
    return row, decisions


def _aggregate(rows: list[dict[str, Any]], mode: str, task_id: int | None = None) -> dict[str, Any]:
    subset = [row for row in rows if row["mode"] == mode and (task_id is None or row["task_id"] == task_id)]
    all_horizons: list[int] = []
    for row in subset:
        histogram = json.loads(row["h_distribution_json"])
        for horizon, count in histogram.items():
            all_horizons.extend([int(horizon)] * int(count))

    def mean(field: str) -> float:
        values = [float(row[field]) for row in subset if np.isfinite(float(row[field]))]
        return float(np.mean(values)) if values else float("nan")

    histogram = collections.Counter(all_horizons)
    return {
        "mode": mode,
        "task_id": task_id if task_id is not None else "overall",
        "episodes": len(subset),
        "success_rate": mean("success"),
        "timeout_rate": mean("timeout"),
        "calls_per_episode": mean("policy_calls"),
        "sampled_action_chunks_per_episode": mean("sampled_action_chunks"),
        "avg_h": float(np.mean(all_horizons)) if all_horizons else float("nan"),
        "h_distribution": dict(sorted(histogram.items())),
        "actual_wall_ms_per_episode": mean("actual_wall_total_ms"),
        "actual_policy_ms_per_episode": mean("actual_policy_total_ms"),
        "actual_server_ms_per_episode": mean("actual_server_total_ms"),
        "predictor_ms_per_episode": mean("actual_predictor_total_ms"),
        "batched_teacher_ms_per_episode": mean("actual_batched_teacher_total_ms"),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _coerce_rollout_row(row: dict[str, str]) -> dict[str, Any]:
    integers = {
        "task_id",
        "episode",
        "initial_state_id",
        "success",
        "timeout",
        "steps",
        "policy_calls",
        "sampled_action_chunks",
    }
    floats = {
        "avg_h",
        "actual_wall_total_ms",
        "actual_policy_total_ms",
        "actual_server_total_ms",
        "actual_predictor_total_ms",
        "actual_batched_teacher_total_ms",
    }
    return {
        key: int(value) if key in integers else float(value) if key in floats else value
        for key, value in row.items()
    }


def _run_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"output_dir", "resume"}
    }


def _prepare_journal(
    output_dir: pathlib.Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], set[tuple[str, int, int]]]:
    rollout_path = output_dir / "rollout_rows.csv"
    decisions_path = output_dir / "decisions.csv"
    signature_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    signature = _run_signature(args)

    if summary_path.exists():
        if args.resume:
            print(summary_path.read_text(), flush=True)
            return [], set()
        raise FileExistsError(f"Evaluation is already complete: {summary_path}")

    if not args.resume:
        existing = [path for path in (rollout_path, decisions_path, signature_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing evaluation journal: {existing}")
        signature_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
        return [], set()

    if not signature_path.exists():
        raise FileNotFoundError(
            f"Cannot safely resume without the configuration signature: {signature_path}"
        )
    saved_signature = json.loads(signature_path.read_text())
    if saved_signature != signature:
        raise ValueError(
            "Resume configuration differs from the saved evaluation configuration. "
            f"saved={saved_signature}, requested={signature}"
        )

    rows = [_coerce_rollout_row(row) for row in _read_csv(rollout_path)]
    completed: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (str(row["mode"]), int(row["task_id"]), int(row["episode"]))
        if key in completed:
            raise ValueError(f"Duplicate completed rollout row while resuming: {key}")
        completed.add(key)

    # Decisions are journaled before their rollout completion row.  If the
    # process died in that small window, discard the incomplete episode's
    # decisions.  Also deduplicate a previously interrupted append.
    cleaned_decisions: list[dict[str, Any]] = []
    seen_decisions: set[tuple[str, int, int, int]] = set()
    for row in _read_csv(decisions_path):
        episode_key = (str(row["mode"]), int(row["task_id"]), int(row["episode"]))
        decision_key = (*episode_key, int(row["environment_step"]))
        if episode_key in completed and decision_key not in seen_decisions:
            cleaned_decisions.append(row)
            seen_decisions.add(decision_key)
    decisions_path.unlink(missing_ok=True)
    _write_csv(decisions_path, cleaned_decisions)
    return rows, completed


def main(args: argparse.Namespace) -> None:
    if args.action_cot_denoising_steps <= 0:
        raise ValueError("action_cot_denoising_steps must be positive.")
    if args.v2_budget_capacity <= 0 or args.v2_initial_budget > args.v2_budget_capacity:
        raise ValueError("Invalid V2 budget configuration.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, completed = _prepare_journal(output_dir, args)
    if (output_dir / "summary.json").exists():
        return
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    _warmup(client, task_suite, args)
    task_end = min(task_suite.n_tasks, args.task_start + args.max_tasks)
    for mode in args.modes:
        for task_id in range(args.task_start, task_end):
            for episode in range(args.num_trials_per_task):
                episode_key = (mode, task_id, episode)
                if episode_key in completed:
                    continue
                row, episode_decisions = _run_episode(
                    mode=mode,
                    task_id=task_id,
                    episode=episode,
                    task_suite=task_suite,
                    client=client,
                    args=args,
                )
                _append_csv(output_dir / "decisions.csv", episode_decisions)
                _append_csv(output_dir / "rollout_rows.csv", [row])
                rows.append(row)
                completed.add(episode_key)
                print(json.dumps(row, sort_keys=True), flush=True)

    per_task = [_aggregate(rows, mode, task_id) for mode in args.modes for task_id in range(args.task_start, task_end)]
    overall = {mode: _aggregate(rows, mode) for mode in args.modes}
    flat_per_task = [
        {**item, "h_distribution": json.dumps(item["h_distribution"], sort_keys=True)} for item in per_task
    ]
    _write_csv(output_dir / "per_task_summary.csv", flat_per_task)
    summary = {
        "status": "complete",
        "paired_initial_states": True,
        "task_suite": args.task_suite_name,
        "num_tasks": task_end - args.task_start,
        "num_trials_per_task": args.num_trials_per_task,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "teacher_samples": args.teacher_samples,
        "timing_semantics": (
            "actual synchronized policy/server/client-wall totals; predictor and batched sampling are included"
        ),
        "config": vars(args),
        "overall": overall,
        "per_task": per_task,
        "outputs": {
            "rollout_rows": str(output_dir / "rollout_rows.csv"),
            "decisions": str(output_dir / "decisions.csv"),
            "per_task_summary": str(output_dir / "per_task_summary.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
