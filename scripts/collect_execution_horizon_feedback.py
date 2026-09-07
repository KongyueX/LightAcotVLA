"""Collect paired A-continuation labels with one real observation-history step."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import pathlib
import time
from typing import Any

import numpy as np

CANDIDATE_HORIZONS = (5, 10, 15, 20, 25)
TASK_SEED_STRIDE = 250_000_000
REPEAT_SEED_STRIDE = 20_000_000
SCHEMA_VERSION = 1


def _collector():
    return importlib.import_module("collect_execution_horizon_counterfactuals")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state-bank", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--seed", type=int, default=67007)
    parser.add_argument("--branch-repeats", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--final-denoising-steps", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.set_defaults(
        task_suite_name="libero_10", model_action_horizon=25, prefix_token_count=0,
        student_mode="ordered_transformer",
    )
    return parser


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _vector(result: dict[str, Any], key: str, width: int) -> np.ndarray:
    value = np.asarray(result[key], dtype=np.float32).reshape(-1)
    if value.shape != (width,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected finite {key}[{width}], got {value.shape}.")
    return value.copy()


def _selected_h(result: dict[str, Any]) -> int:
    base = _collector()
    candidates = tuple(np.asarray(result["execution_horizon_candidate_horizons"]).reshape(-1).tolist())
    if candidates != CANDIDATE_HORIZONS:
        raise ValueError(f"Expected H25 A candidates {CANDIDATE_HORIZONS}, got {candidates}.")
    return base.ordered.selected_horizon(result, model_action_horizon=25)


def _request(
    client: Any, observation: dict[str, Any], task_description: str, *, args: argparse.Namespace,
    seed: int, step: int, step_limit: int, previous_actions: np.ndarray | None, previous_h: int,
) -> dict[str, Any]:
    base = _collector()
    policy_input = base.libero_eval._observation_to_policy_input(observation, task_description, args.resize_size)
    policy_input["action_cot_final_denoising_steps"] = np.asarray(args.final_denoising_steps, dtype=np.int32)
    policy_input["action_cot_absolute_decision_step"] = np.asarray(step, dtype=np.int32)
    return base._policy_request(
        client, policy_input, seed=seed, args=args, teacher=False, profile=True, run_student=True,
        previous_actions=previous_actions, previous_h=previous_h, budget_balance=0.5,
        episode_progress=float(np.clip(step / max(step_limit, 1), 0.0, 1.0)),
    )


def _execute(env: Any, actions: np.ndarray, *, step: int, step_limit: int) -> tuple[dict[str, Any], int, bool]:
    libero = _collector().libero_eval
    observation: dict[str, Any] = {}
    for action in actions:
        if step >= step_limit:
            break
        try:
            observation, _, done, _ = env.step(np.asarray(action).tolist())
        except Exception as exc:
            if not libero._is_terminated_episode_error(exc):
                raise
            return observation, step, bool(libero._env_success(env))
        step += 1
        if done or libero._env_success(env):
            return observation, step, True
    return observation, step, bool(libero._env_success(env))


@dataclasses.dataclass
class FeedbackRoot:
    snapshot: Any
    result: dict[str, Any]
    primary_actions: np.ndarray
    previous_prefix_feature: np.ndarray
    previous_state: np.ndarray
    previous_h: int
    elapsed_steps: int
    history_valid: bool
    step: int
    seed: int
    decision_index: int


def collect_source_root(
    env: Any, observation: dict[str, Any], task_description: str, *, args: argparse.Namespace,
    client: Any, task_id: int, episode_id: int, step: int, step_limit: int,
    progress_callback: Any | None = None,
) -> tuple[FeedbackRoot | None, bool, int]:
    """Complete A's rollout and retain a call without using its eventual outcome."""
    base = _collector()
    sampler = np.random.default_rng(np.random.SeedSequence([args.seed, task_id, episode_id, 0x46454544]))
    previous_actions = None
    previous_prefix = np.zeros(2048, dtype=np.float32)
    previous_state = np.zeros(32, dtype=np.float32)
    previous_h = 10
    previous_step = step
    decision_index = 0
    root = None
    success = False
    while step < step_limit and not success:
        seed = base._root_seed(args.seed, task_id, episode_id, step, task_stride=TASK_SEED_STRIDE)
        result = _request(
            client, observation, task_description, args=args, seed=seed, step=step, step_limit=step_limit,
            previous_actions=previous_actions, previous_h=previous_h,
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        selected_h = _selected_h(result)
        if actions.shape != (25, 7):
            raise ValueError(f"Expected raw H25 actions[25,7], got {actions.shape}.")
        current_prefix = _vector(result, "execution_horizon_prefix_feature", 2048)
        current_state = _vector(result, "execution_horizon_state_normalized", 32)
        if int(sampler.integers(decision_index + 1)) == 0:
            root = FeedbackRoot(
                snapshot=base._capture_snapshot(env),
                result={key: np.asarray(value).copy() for key, value in result.items() if key in {
                    "execution_horizon_temporal_feature", "execution_horizon_prefix_feature",
                    "execution_horizon_state_normalized", "execution_horizon_ordered_continuation_logits",
                    "execution_horizon_ordered_horizon_probability", "execution_horizon_ordered_selected_h",
                    "execution_horizon_candidate_horizons", "collector_wall_ms",
                }},
                primary_actions=actions.copy(), previous_prefix_feature=previous_prefix.copy(),
                previous_state=previous_state.copy(), previous_h=previous_h,
                elapsed_steps=step - previous_step if previous_actions is not None else 0,
                history_valid=previous_actions is not None, step=step, seed=seed, decision_index=decision_index,
            )
        decision_step = step
        observation, step, success = _execute(env, actions[:selected_h], step=step, step_limit=step_limit)
        previous_prefix, previous_state = current_prefix, current_state
        previous_actions = actions.copy()
        previous_h = step - decision_step
        previous_step = decision_step
        decision_index += 1
        if progress_callback is not None:
            progress_callback(phase="source", source_calls=decision_index, source_step=step)
        if step == decision_step:
            break
    return root, success, decision_index


def run_branch(
    env: Any, root: FeedbackRoot, task_description: str, *, args: argparse.Namespace, client: Any,
    forced_horizon: int, repeat_seed: int, step_limit: int,
) -> dict[str, Any]:
    """Force one saved root prefix, then use the unchanged A for every subsequent call."""
    base = _collector()
    started = time.perf_counter()
    observation = base._restore_snapshot(env, root.snapshot)
    step = root.step
    actions = root.primary_actions
    planned_h = forced_horizon
    continuation_index = 0
    root_rpc = float(np.asarray(root.result["collector_wall_ms"]).item()) / 1000.0
    rpc_seconds = root_rpc
    success = False
    while step < step_limit and not success:
        decision_step = step
        observation, step, success = _execute(env, actions[:planned_h], step=step, step_limit=step_limit)
        if success or step >= step_limit or step == decision_step:
            break
        seed = base._branch_continuation_seed(
            repeat_seed, continuation_index, continuation_offset=REPEAT_SEED_STRIDE // 2,
        )
        result = _request(
            client, observation, task_description, args=args, seed=seed, step=step, step_limit=step_limit,
            previous_actions=actions, previous_h=step - decision_step,
        )
        rpc_seconds += float(np.asarray(result["collector_wall_ms"]).item()) / 1000.0
        actions = np.asarray(result["actions"], dtype=np.float32)
        planned_h = _selected_h(result)
        continuation_index += 1
    return {
        "success": success, "elapsed": time.perf_counter() - started + root_rpc,
        "rpc": rpc_seconds, "calls": 1 + continuation_index, "steps": step - root.step,
    }


def collect_branches(
    env: Any, root: FeedbackRoot, task_description: str, *, args: argparse.Namespace, client: Any,
    task_id: int, episode_id: int, step_limit: int, source_success: bool, source_calls: int,
    progress_callback: Any | None = None,
) -> dict[str, np.ndarray]:
    base = _collector()
    shape = (len(CANDIDATE_HORIZONS), args.branch_repeats)
    success = np.zeros(shape, dtype=np.bool_)
    elapsed = np.zeros(shape, dtype=np.float64)
    rpc = np.zeros(shape, dtype=np.float64)
    calls = np.zeros(shape, dtype=np.int32)
    steps = np.zeros(shape, dtype=np.int32)
    seeds = np.zeros(args.branch_repeats, dtype=np.uint32)
    for repeat in range(args.branch_repeats):
        seed = base._branch_seed(root.seed, repeat, REPEAT_SEED_STRIDE)
        seeds[repeat] = seed
        schedule_seed = base._branch_schedule_seed(seed, schedule_offset=REPEAT_SEED_STRIDE // 4)
        for candidate_index in np.random.default_rng(schedule_seed).permutation(len(CANDIDATE_HORIZONS)):
            horizon = CANDIDATE_HORIZONS[candidate_index]
            outcome = run_branch(
                env, root, task_description, args=args, client=client, forced_horizon=horizon,
                repeat_seed=seed, step_limit=step_limit,
            )
            success[candidate_index, repeat] = outcome["success"]
            elapsed[candidate_index, repeat] = outcome["elapsed"]
            rpc[candidate_index, repeat] = outcome["rpc"]
            calls[candidate_index, repeat] = outcome["calls"]
            steps[candidate_index, repeat] = outcome["steps"]
            if progress_callback is not None:
                progress_callback(phase="branches", repeat_index=repeat, horizon=horizon)
    return {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "temporal_feature": _vector(root.result, "execution_horizon_temporal_feature", 256),
        "prefix_feature": _vector(root.result, "execution_horizon_prefix_feature", 2048),
        "state": _vector(root.result, "execution_horizon_state_normalized", 32),
        "previous_prefix_feature": root.previous_prefix_feature.copy(),
        "previous_state": root.previous_state.copy(),
        "previous_h": np.asarray(root.previous_h, dtype=np.int32),
        "elapsed_steps": np.asarray(root.elapsed_steps, dtype=np.int32),
        "history_valid": np.asarray(root.history_valid, dtype=np.bool_),
        "continuation_logits": _vector(root.result, "execution_horizon_ordered_continuation_logits", 4),
        "ordered_probability": _vector(root.result, "execution_horizon_ordered_horizon_probability", 5),
        "selected_h": np.asarray(_selected_h(root.result), dtype=np.int32),
        "trial_success": success, "trial_elapsed": elapsed, "trial_rpc": rpc,
        "trial_calls": calls, "trial_steps": steps, "trial_valid": np.ones(shape, dtype=np.bool_),
        "candidate_horizons": np.asarray(CANDIDATE_HORIZONS, dtype=np.int32),
        "branch_seeds": seeds, "task_id": np.asarray(task_id, dtype=np.int32),
        "episode_id": np.asarray(episode_id, dtype=np.int32), "root_step": np.asarray(root.step, dtype=np.int32),
        "root_seed": np.asarray(root.seed, dtype=np.uint32),
        "source_success": np.asarray(source_success, dtype=np.bool_),
        "source_calls": np.asarray(source_calls, dtype=np.int32),
        "source_decision_index": np.asarray(root.decision_index, dtype=np.int32),
        "physics_state": np.asarray(root.snapshot.physics_state, dtype=np.float64).copy(),
        "primary_actions": root.primary_actions.copy(),
        "root_rpc_seconds": np.asarray(root.result["collector_wall_ms"], dtype=np.float64).reshape(()) / 1000.0,
    }


def root_diagnostics(record: dict[str, np.ndarray]) -> dict[str, Any]:
    successes = np.asarray(record["trial_success"], dtype=bool)
    costs = np.asarray(record["trial_rpc"], dtype=np.float64)
    valid = np.asarray(record["trial_valid"], dtype=bool)
    if successes.ndim != 2 or successes.shape[0] != 5 or successes.shape != costs.shape or not np.all(valid):
        raise ValueError("Feedback records must contain five complete paired-horizon branches.")
    repeats = successes.shape[1]
    if repeats < 2 or valid.shape != successes.shape:
        raise ValueError("At least two complete paired repeats are required.")
    selected_h = int(record["selected_h"])
    a_index = CANDIDATE_HORIZONS.index(selected_h)
    reference = successes[a_index]
    per_h = {}
    for index, horizon in enumerate(CANDIDATE_HORIZONS):
        per_h[str(horizon)] = {
            "success_count": int(successes[index].sum()),
            "rescues_vs_a": int(np.sum(successes[index] & ~reference)),
            "regressions_vs_a": int(np.sum(~successes[index] & reference)),
            "mean_rpc_seconds": float(costs[index].mean()),
        }
    loo_successes = 0
    loo_cost = 0.0
    loo_h = []
    for held_out in range(repeats):
        training = np.arange(repeats) != held_out
        best = max(range(5), key=lambda index: (
            int(successes[index, training].sum()), -float(costs[index, training].mean()), index == a_index,
        ))
        loo_h.append(CANDIDATE_HORIZONS[best])
        loo_successes += int(successes[best, held_out])
        loo_cost += float(costs[best, held_out])
    return {
        "task_id": int(record["task_id"]), "episode_id": int(record["episode_id"]),
        "root_step": int(record["root_step"]), "selected_h": selected_h,
        "history_valid": bool(record["history_valid"]), "source_success": bool(record["source_success"]),
        "num_repeats": repeats, "a_success_count": int(reference.sum()), "by_h": per_h,
        "all_h_failed_all_repeats": bool(not np.any(successes)),
        "any_paired_success_difference": bool(np.any(successes != reference[None, :])),
        "any_mean_success_difference": bool(np.any(successes.sum(axis=1) != reference.sum())),
        "empirical_best_success_count": int(successes.sum(axis=1).max()),
        "loo_diagnostic_success_count": loo_successes,
        "loo_diagnostic_mean_rpc_seconds": loo_cost / repeats,
        "loo_diagnostic_selected_h": loo_h,
    }


def _read_record(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def summarize_records(paths: list[pathlib.Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        row = root_diagnostics(_read_record(path))
        rows.append({"path": str(path.resolve()), **row})
    repeats = sum(row["num_repeats"] for row in rows)
    aggregate = {
        "num_roots": len(rows), "num_paired_trials": repeats,
        "source_success_count": sum(row["source_success"] for row in rows),
        "history_valid_roots": sum(row["history_valid"] for row in rows),
        "all_h_failed_roots": sum(row["all_h_failed_all_repeats"] for row in rows),
        "roots_with_any_paired_success_difference": sum(row["any_paired_success_difference"] for row in rows),
        "roots_with_any_mean_success_difference": sum(row["any_mean_success_difference"] for row in rows),
        "a_success_count": sum(row["a_success_count"] for row in rows),
        "loo_diagnostic_success_count": sum(row["loo_diagnostic_success_count"] for row in rows),
        "empirical_best_success_count": sum(row["empirical_best_success_count"] for row in rows),
        "paired_vs_a": {
            str(horizon): {
                key: sum(row["by_h"][str(horizon)][key] for row in rows)
                for key in ("success_count", "rescues_vs_a", "regressions_vs_a")
            }
            for horizon in CANDIDATE_HORIZONS
        },
        "diagnostic_semantics": (
            "Root-level paired branches with fixed source A and continuation A; empirical and leave-one-seed-out "
            "best-H metrics are diagnostics, not deployable closed-loop performance or stability guarantees."
        ),
    }
    return {**aggregate, "rows": rows}


def _save_record(path: pathlib.Path, record: dict[str, np.ndarray]) -> None:
    root_diagnostics(record)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **record)
    temporary.replace(path)


def _refresh_summary(output: pathlib.Path, *, expected: int, metadata: dict[str, Any]) -> dict[str, Any]:
    report = summarize_records(sorted(output.glob("task*_ep*.npz")))
    rows = report.pop("rows")
    temporary = output / "rows.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(output / "rows.jsonl")
    report.update({
        "status": "complete" if report["num_roots"] == expected else "collecting",
        "expected_roots": expected, **metadata,
    })
    _write_json(output / "summary.json", report)
    return report


def main(args: argparse.Namespace) -> None:
    if (
        args.task_start < 0 or args.max_tasks <= 0 or args.seed < 0 or args.branch_repeats < 2
        or args.num_steps_wait < 0 or args.resize_size <= 0 or args.warmup_requests < 0
        or args.action_cot_denoising_steps != 10 or args.final_denoising_steps != 10
        or not args.episodes or min(args.episodes) < 0 or len(set(args.episodes)) != len(args.episodes)
    ):
        raise ValueError("Use distinct nonnegative episodes, positive task/repeat dimensions, and CoT/final NFE10.")
    base = _collector()
    libero = base.libero_eval
    libero._ensure_libero_import_path()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    tasks = list(range(args.task_start, args.task_start + args.max_tasks))
    if tasks[-1] >= suite.n_tasks:
        raise ValueError("Requested task range exceeds LIBERO-10.")
    step_limit_default = libero._max_steps(args.task_suite_name) + args.num_steps_wait
    base._validate_seed_namespace(
        base_seed=args.seed, task_ids=tasks, episode_ids=args.episodes, maximum_episode_step=step_limit_default,
        maximum_continuation_calls=step_limit_default, task_stride=TASK_SEED_STRIDE,
        branch_repeats=args.branch_repeats, branch_repeat_seed_stride=REPEAT_SEED_STRIDE,
        teacher_samples=1,  # Reserve the existing validator's minimal root lane; no teacher request is made.
    )
    bank = base.horizon_initial_states.InitialStateBank(args.initial_state_bank)
    if bank.manifest["task_suite"] != args.task_suite_name:
        raise ValueError("Initial-state bank task suite differs from this collector.")
    for task in tasks:
        bank.validate_presets(task, suite.get_task_init_states(task))
        for episode in args.episodes:
            bank.state(task, episode)
    output = pathlib.Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION, "source_policy": "A_ordered_transformer",
        "continuation_policy": "A_ordered_transformer", "candidate_horizons": list(CANDIDATE_HORIZONS),
        "root_sampling": "outcome_blind_full_trajectory_reservoir",
        "timing_semantics": (
            "trial_rpc sums measured client RPC seconds; trial_elapsed is branch wall time including snapshot "
            "restore and simulation. Both include the same saved source-root RPC once per candidate. "
            "The root is never regenerated; this shared cost cancels in paired differences. All policy calls "
            "use profile_policy_timing=true and CoT/final NFE10. No teacher/MC calls or full prefix export."
        ),
        **bank.metadata(),
    }
    config = {key: value for key, value in vars(args).items() if key not in {"policy_api_key", "output_dir"}}
    config["initial_state_bank"] = str(pathlib.Path(args.initial_state_bank).resolve())
    config.update(metadata)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to a different feedback collection configuration.")
    _write_json(config_path, config)
    expected = len(tasks) * len(args.episodes)
    expected_files = {f"task{task:02d}_ep{episode:06d}.npz" for task in tasks for episode in args.episodes}
    if any(path.name not in expected_files for path in output.glob("task*_ep*.npz")):
        raise ValueError("Output contains roots outside the requested tasks and episodes.")
    report = _refresh_summary(output, expected=expected, metadata=metadata)
    if report["status"] == "complete":
        _write_json(output / "status.json", {"status": "complete", "completed_roots": expected})
        return
    client = base.websocket_policy.WebsocketClientPolicy(
        args.host, args.port, api_key=args.policy_api_key, ping_interval=None, ping_timeout=None,
    )
    warmed = False
    try:
        for task in tasks:
            for episode in args.episodes:
                path = output / f"task{task:02d}_ep{episode:06d}.npz"
                if path.exists():
                    continue
                def progress(
                    *, _task: int = task, _episode: int = episode,
                    _completed: int = report["num_roots"], **details: Any,
                ) -> None:
                    _write_json(output / "status.json", {
                        "status": "collecting", "task_id": _task, "episode_id": _episode,
                        "completed_roots": _completed, "expected_roots": expected, **details,
                    })
                progress(phase="initializing")
                env, description = libero._get_libero_env(suite.get_task(task), libero.LIBERO_ENV_RESOLUTION, args.seed)
                try:
                    env.reset()
                    observation = env.set_init_state(bank.state(task, episode))
                    environment_horizon = libero._env_horizon(env)
                    step_limit = min(step_limit_default, environment_horizon) if environment_horizon else step_limit_default
                    step = 0
                    done = False
                    for _ in range(args.num_steps_wait):
                        observation, _, done, _ = env.step(libero.LIBERO_DUMMY_ACTION)
                        step += 1
                        if done:
                            break
                    if done:
                        raise RuntimeError(f"Initial wait ended task{task}/episode{episode} before any source decision.")
                    if not warmed:
                        for repeat in range(args.warmup_requests):
                            _request(
                                client, observation, description, args=args, seed=args.seed + repeat, step=step,
                                step_limit=step_limit, previous_actions=None, previous_h=10,
                            )
                        warmed = True
                    root, source_success, source_calls = collect_source_root(
                        env, observation, description, args=args, client=client, task_id=task, episode_id=episode,
                        step=step, step_limit=step_limit, progress_callback=progress,
                    )
                    if root is None:
                        raise RuntimeError(f"Source trajectory has no root for task{task}/episode{episode}.")
                    record = collect_branches(
                        env, root, description, args=args, client=client, task_id=task, episode_id=episode,
                        step_limit=step_limit, source_success=source_success, source_calls=source_calls,
                        progress_callback=progress,
                    )
                    _save_record(path, record)
                finally:
                    libero._safe_close_env(env)
                report = _refresh_summary(output, expected=expected, metadata=metadata)
                print(json.dumps({"completed_roots": report["num_roots"], "expected_roots": expected,
                                  "task_id": task, "episode_id": episode,
                                  "paired_success_difference_roots": report["roots_with_any_paired_success_difference"]}), flush=True)
        _write_json(output / "status.json", {"status": "complete", "completed_roots": expected})
    except Exception as exc:
        _write_json(output / "status.json", {"status": "failed", "completed_roots": report["num_roots"],
                                            "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
