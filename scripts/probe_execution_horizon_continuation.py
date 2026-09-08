"""Measure first-H advantages under A and candidate continuation on training roots."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import fcntl
import json
import pathlib
import time
from typing import Any

import collect_execution_horizon_feedback as feedback
import eval_libero_action_cot_pruning as libero_eval
import eval_libero_execution_horizon as evaluator
import numpy as np
from openpi_client import websocket_client_policy as websocket_policy
import probe_execution_horizon_first_divergence as first_probe

from openpi.execution_horizon.feedback import FeedbackSelector
from openpi.execution_horizon.initial_states import InitialStateBank
from openpi.execution_horizon.trace import _physics_state


TASKS = tuple(range(10))
EPISODES = (300, 301, 302, 303)
ROOTS_PER_TASK = 2
REPEATS = 3
CELLS = (("A", "A"), ("C", "A"), ("A", "C"), ("C", "C"))
ROOT_ARRAYS = (
    "prefix_actions", "physics_state", "saved_actions", "cache_prefix", "cache_state",
    "root_proprio", "root_image", "root_wrist_image", "anchor_logits", "anchor_probabilities",
    "candidate_logits", "candidate_probabilities",
)


class InputMismatch(RuntimeError):
    """Stop a diagnostic whose reconstructed input no longer matches its saved root."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state-bank", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-params", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=87007)
    return parser


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feedback._write_json(path, value)


def _eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return evaluator.build_parser().parse_args([
        "--output-dir", str(args.output_dir), "--initial-state-bank", str(args.initial_state_bank),
        "--modes", "ordered_transformer", "--model-action-horizon", "25",
        "--host", args.host, "--port", str(args.port), "--seed", str(args.seed),
        "--num-steps-wait", "10", "--resize-size", "224",
        "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
    ])


def _action_chunk(result: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.shape != (25, 7) or not np.all(np.isfinite(actions)):
        raise InputMismatch("The A service must return finite raw actions[25,7].")
    return actions


def _request_once(
    client: Any, observation: dict[str, Any], description: str, *, args: argparse.Namespace,
    step: int, limit: int, seed: int, previous_actions: np.ndarray | None, previous_h: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    element = libero_eval._observation_to_policy_input(observation, description, args.resize_size)
    result, timing = evaluator._request(
        client, element, mode="ordered_transformer", seed=seed,
        previous_actions=previous_actions, previous_horizon=previous_h,
        budget_fraction=0.5, episode_progress=step / limit, absolute_decision_step=step, args=args,
    )
    return result, timing, element


def _both_h(
    result: dict[str, Any], candidate: FeedbackSelector, previous_cache: dict[str, Any] | None,
    *, step: int, previous_h: int,
) -> tuple[int, int, dict[str, Any], dict[str, Any], float]:
    started = time.perf_counter()
    a_h = feedback._selected_h(result)
    inputs, cache = evaluator._feedback_inputs(result, previous_cache, step=step, previous_h=previous_h)
    c_h, info = candidate.decide(inputs)
    if c_h not in feedback.CANDIDATE_HORIZONS:
        raise InputMismatch("The candidate selected an unsupported horizon.")
    return a_h, c_h, cache, info, time.perf_counter() - started


def _execute(
    env: Any, actions: np.ndarray, *, step: int, limit: int,
    recorded_actions: list[np.ndarray] | None = None,
) -> tuple[dict[str, Any], int, bool]:
    observation = {}
    success = False
    for action in actions[:max(limit - step, 0)]:
        command = np.asarray(action, dtype=np.float32)
        observation, _, done, _ = env.step(command.tolist())
        if recorded_actions is not None:
            recorded_actions.append(command.copy())
        step += 1
        if done or libero_eval._env_success(env):
            success = True
            break
    return observation, step, success


def save_root(path: pathlib.Path, root: dict[str, Any]) -> None:
    metadata = {key: value for key, value in root.items() if key not in ROOT_ARRAYS}
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, **{name: root[name] for name in ROOT_ARRAYS},
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    temporary.replace(path)


def load_root(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        root = json.loads(str(archive["metadata_json"].item()))
        root.update({name: archive[name].copy() for name in ROOT_ARRAYS})
    if root["prefix_actions"].shape != (root["step"], 7) or root["saved_actions"].shape != (25, 7):
        raise InputMismatch("Saved root action shapes do not match the complete prefix and H25 chunk.")
    return root


def find_first_disagreement(
    *, task_id: int, episode: int, suite: Any, bank: InitialStateBank, client: Any,
    candidate: FeedbackSelector, args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    env, description = libero_eval._get_libero_env(
        suite.get_task(task_id), libero_eval.LIBERO_ENV_RESOLUTION, args.seed
    )
    started = time.perf_counter()
    prefix: list[np.ndarray] = []
    calls = 0
    step = 0
    source_rpc = 0.0
    previous_actions = None
    previous_h = 10
    previous_cache = None
    try:
        env.reset()
        env.set_init_state(bank.state(task_id, episode))
        limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        environment_horizon = libero_eval._env_horizon(env)
        if environment_horizon is not None:
            limit = min(limit, environment_horizon)
        wait_actions = np.repeat(np.asarray(libero_eval.LIBERO_DUMMY_ACTION, dtype=np.float32)[None], 10, axis=0)
        observation, step, success = _execute(env, wait_actions, step=step, limit=limit, recorded_actions=prefix)
        while not success and step < limit:
            seed = feedback._collector()._root_seed(
                args.seed, task_id, episode, step, task_stride=feedback.TASK_SEED_STRIDE
            )
            result, timing, element = _request_once(
                client, observation, description, args=args, step=step, limit=limit, seed=seed,
                previous_actions=previous_actions, previous_h=previous_h,
            )
            calls += 1
            source_rpc += float(timing["wall_ms"]) / 1000.0
            actions = _action_chunk(result)
            a_h, c_h, current_cache, candidate_info, selection_seconds = _both_h(
                result, candidate, previous_cache, step=step, previous_h=previous_h
            )
            if a_h != c_h:
                report = {
                    "status": "found_root", "task_id": task_id, "episode_id": episode,
                    "source_policy": "A", "source_calls": calls, "source_steps": step,
                    "source_rpc_seconds": source_rpc, "source_seconds": time.perf_counter() - started,
                    "source_terminal_success": None, "a_h": a_h, "c_h": c_h,
                }
                physics, _ = _physics_state(env)
                root = {
                    "schema_version": 1, "task_id": task_id, "episode_id": episode, "step": step,
                    "simulator_seed": args.seed, "root_request_seed": seed,
                    "a_h": a_h, "c_h": c_h, "previous_h": previous_h,
                    "prefix_actions": np.asarray(prefix, dtype=np.float32), "physics_state": physics,
                    "saved_actions": actions.copy(), "cache_prefix": current_cache["prefix"].copy(),
                    "cache_state": current_cache["state"].copy(),
                    "root_proprio": np.asarray(element["observation/state"]).copy(),
                    "root_image": np.asarray(element["observation/image"]).copy(),
                    "root_wrist_image": np.asarray(element["observation/wrist_image"]).copy(),
                    "anchor_logits": np.asarray(result["execution_horizon_ordered_continuation_logits"]).copy(),
                    "anchor_probabilities": np.asarray(result["execution_horizon_ordered_horizon_probability"]).copy(),
                    "candidate_logits": np.asarray(candidate_info["ordered_continuation_logits"]),
                    "candidate_probabilities": np.asarray(candidate_info["ordered_horizon_probability"]),
                    "root_rpc_seconds": float(timing["wall_ms"]) / 1000.0,
                    "root_policy_seconds": float(timing["policy_ms"]) / 1000.0,
                    "root_selection_seconds": selection_seconds, "source_report": report,
                }
                return root, report
            previous_actions = actions.copy()
            previous_cache = current_cache
            before = step
            observation, step, success = _execute(
                env, actions[:a_h], step=step, limit=limit, recorded_actions=prefix
            )
            previous_h = step - before
        return None, {
            "status": "no_disagreement", "task_id": task_id, "episode_id": episode,
            "source_policy": "A", "source_calls": calls, "source_steps": step,
            "source_rpc_seconds": source_rpc, "source_seconds": time.perf_counter() - started,
            "source_terminal_success": bool(success),
        }
    finally:
        libero_eval._safe_close_env(env)


def run_branch(
    *, root: dict[str, Any], first: str, continuation: str, repeat: int,
    suite: Any, bank: InitialStateBank, client: Any, candidate: FeedbackSelector, args: argparse.Namespace,
) -> dict[str, Any]:
    env = None
    try:
        env, description, observation, limit, reconstruction = first_probe._prepare_live_root(suite, bank, root, args)
        if not reconstruction["root_valid"]:
            raise InputMismatch(f"Root reconstruction failed: {reconstruction}")
        cache = {"prefix": root["cache_prefix"].copy(), "state": root["cache_state"].copy(), "step": root["step"]}
        actions = root["saved_actions"].copy()
        h = root["a_h"] if first == "A" else root["c_h"]
        step = root["step"]
        continuation_calls = 0
        continuation_rpc = 0.0
        continuation_policy = 0.0
        seeds = []
        decisions = []
        first_actual_h = 0
        success = False
        started = time.perf_counter()
        while step < limit:
            before = step
            observation, step, success = _execute(env, actions[:h], step=step, limit=limit)
            actual_h = step - before
            if continuation_calls == 0:
                first_actual_h = actual_h
            if success or step >= limit:
                break
            if actual_h <= 0:
                raise InputMismatch("A continuation decision did not execute any environment action.")
            seed = first_probe._continuation_seed(root["root_request_seed"], repeat, continuation_calls)
            result, timing, _ = _request_once(
                client, observation, description, args=args, step=step, limit=limit, seed=seed,
                previous_actions=actions, previous_h=actual_h,
            )
            continuation_rpc += float(timing["wall_ms"]) / 1000.0
            continuation_policy += float(timing["policy_ms"]) / 1000.0
            a_h, c_h, cache, _, _ = _both_h(result, candidate, cache, step=step, previous_h=actual_h)
            h = a_h if continuation == "A" else c_h
            actions = _action_chunk(result)
            seeds.append(seed)
            decisions.append({"step": step, "a_h": a_h, "c_h": c_h, "selected_h": h})
            continuation_calls += 1
        segment_seconds = time.perf_counter() - started
        return {
            "repeat": repeat, "first": first, "continuation": continuation,
            "forced_h": root["a_h"] if first == "A" else root["c_h"], "actual_first_h": first_actual_h,
            "success": bool(success), "steps": step - root["step"], "final_step": step,
            "calls": 1 + continuation_calls, "continuation_calls": continuation_calls,
            "rpc_seconds": root["root_rpc_seconds"] + continuation_rpc,
            "policy_seconds": root["root_policy_seconds"] + continuation_policy,
            "full_seconds": root["root_rpc_seconds"] + root["root_selection_seconds"] + segment_seconds,
            "segment_seconds_excluding_shared_root_rpc": segment_seconds,
            "continuation_rpc_seconds": continuation_rpc, "root_rpc_shared": True,
            "root_rpc_seconds": root["root_rpc_seconds"], "continuation_seeds": seeds,
            "decisions": decisions, **reconstruction,
        }
    finally:
        if env is not None:
            libero_eval._safe_close_env(env)


def effects(branches: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {(row["repeat"], row["first"], row["continuation"]): row for row in branches}
    if len(indexed) != 12 or len(branches) != 12:
        raise ValueError("Continuation effects require exactly 2 x 2 x 3 completed branches.")
    counts = {first + continuation: sum(int(indexed[r, first, continuation]["success"]) for r in range(REPEATS))
              for first, continuation in CELLS}
    delta_a = counts["CA"] - counts["AA"]
    delta_c = counts["CC"] - counts["AC"]
    paired = []
    for repeat in range(REPEATS):
        values = {f + c: int(indexed[repeat, f, c]["success"]) for f, c in CELLS}
        da, dc = values["CA"] - values["AA"], values["CC"] - values["AC"]
        paired.append({"repeat": repeat, "success": values, "delta_A": da, "delta_C": dc, "interaction": dc - da})
    means = {f + c: {name: float(np.mean([indexed[r, f, c][name] for r in range(REPEATS)]))
                     for name in ("rpc_seconds", "policy_seconds", "full_seconds", "calls", "steps")}
             for f, c in CELLS}
    return {
        "success_counts": counts, "trials_per_cell": REPEATS, "delta_A": delta_a, "delta_C": delta_c,
        "interaction": delta_c - delta_a,
        "strict_reversal": delta_a * delta_c < 0,
        "harmful_reversal": delta_a > 0 and delta_c < 0,
        "nonnegative_to_harmful": delta_a >= 0 and delta_c < 0,
        "same_sign": int(np.sign(delta_a)) == int(np.sign(delta_c)), "paired": paired, "cell_means": means,
    }


def run_root(
    root: dict[str, Any], path: pathlib.Path, *, suite: Any, bank: InitialStateBank,
    client: Any, candidate: FeedbackSelector, args: argparse.Namespace, status_path: pathlib.Path,
) -> dict[str, Any]:
    report = json.loads(path.read_text()) if path.exists() else {
        "status": "running", "task_id": root["task_id"], "episode_id": root["episode_id"],
        "root_step": root["step"], "a_h": root["a_h"], "c_h": root["c_h"], "branches": [],
        "root_npz": str(path.with_suffix(".npz")), "diagnostic_only": True,
    }
    if report["status"] == "complete":
        return report
    completed = {(row["repeat"], row["first"], row["continuation"]) for row in report["branches"]}
    for repeat in range(REPEATS):
        order = CELLS if repeat % 2 == 0 else tuple(reversed(CELLS))
        for first, continuation in order:
            if (repeat, first, continuation) in completed:
                continue
            _write_json(status_path, {
                "status": "running", "phase": "branches", "task_id": root["task_id"],
                "episode_id": root["episode_id"], "repeat": repeat, "first": first,
                "continuation": continuation, "completed_root_branches": len(report["branches"]),
            })
            outcome = run_branch(
                root=root, first=first, continuation=continuation, repeat=repeat, suite=suite,
                bank=bank, client=client, candidate=candidate, args=args,
            )
            report["branches"].append(outcome)
            _write_json(path, report)
    report.update(status="complete", effects=effects(report["branches"]))
    _write_json(path, report)
    return report


def summarize(root_reports: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(root_reports)
    counts = {f + c: sum(root["effects"]["success_counts"][f + c] for root in root_reports) for f, c in CELLS}
    da, dc = counts["CA"] - counts["AA"], counts["CC"] - counts["AC"]
    tasks = []
    for task in TASKS:
        rows = [row for row in root_reports if row["task_id"] == task]
        ta = sum(row["effects"]["delta_A"] for row in rows)
        tc = sum(row["effects"]["delta_C"] for row in rows)
        tasks.append({
            "task_id": task, "num_roots": len(rows), "delta_A": ta, "delta_C": tc,
            "interaction": tc - ta, "same_sign": int(np.sign(ta)) == int(np.sign(tc)),
            "strict_reversal": ta * tc < 0, "harmful_reversal": ta > 0 and tc < 0,
        })
    def identities(name):
        return [[row["task_id"], row["episode_id"]] for row in root_reports if row["effects"][name]]
    return {
        "status": "complete", "diagnostic_only": True, "num_roots": count,
        "num_branches": sum(len(row["branches"]) for row in root_reports),
        "aggregate": {
            "success_counts": counts, "trials_per_cell": count * REPEATS,
            "delta_A": da, "delta_C": dc, "interaction": dc - da,
            "mean_delta_A": da / (count * REPEATS) if count else None,
            "mean_delta_C": dc / (count * REPEATS) if count else None,
            "mean_interaction": (dc - da) / (count * REPEATS) if count else None,
        },
        "strict_reversal_roots": identities("strict_reversal"),
        "harmful_reversal_roots": identities("harmful_reversal"),
        "nonnegative_to_harmful_roots": identities("nonnegative_to_harmful"),
        "strict_reversal_tasks": [row["task_id"] for row in tasks if row["strict_reversal"]],
        "harmful_reversal_tasks": [row["task_id"] for row in tasks if row["harmful_reversal"]],
        "same_sign_tasks": [row["task_id"] for row in tasks if row["num_roots"] and row["same_sign"]],
        "different_sign_tasks": [row["task_id"] for row in tasks if row["num_roots"] and not row["same_sign"]],
        "no_disagreement_episodes": [[row["task_id"], row["episode_id"]]
                                     for row in episodes if row["status"] == "no_disagreement"],
        "tasks": tasks, "episodes": episodes,
        "root_results": [{key: row[key] for key in ("task_id", "episode_id", "root_step", "a_h", "c_h", "effects")}
                         for row in root_reports],
        "actual_policy_requests": sum(row.get("source_calls", 0) for row in episodes)
        + sum(branch["continuation_calls"] for row in root_reports for branch in row["branches"]),
        "definitions": {
            "cell_order": "first H source, then continuation policy; A=original A, C=current step200",
            "delta_A": "success(C first H, A continuation) minus success(A first H, A continuation)",
            "delta_C": "success(C first H, C continuation) minus success(A first H, C continuation)",
            "interaction": "delta_C minus delta_A",
            "strict_reversal": "delta_A * delta_C < 0", "harmful_reversal": "delta_A > 0 and delta_C < 0",
            "full_seconds": "live continuation segment walltime plus shared root RPC and root selector time",
        },
        "scope": [
            "Only first-disagreement states reached by A on training IDs300-303; at most two roots per task.",
            "No terminal-outcome selection; episodes without a disagreement and task-cap omissions are recorded.",
            "Each branch reconstructs the full recorded prefix in a new same-seed environment without policy calls.",
            "One A-service VLA response supplies both H choices; the saved root chunk and observation cache are shared.",
            "Seeds pair by repeat and continuation call index; reconstruction time is excluded from branch timing.",
            "These conditional counterfactual outcomes are not whole-policy success rates or generalization results.",
        ],
    }


def main(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "initial_state_bank": str(args.initial_state_bank.resolve()),
        "candidate_params": str(args.candidate_params.resolve()), "host": args.host, "port": args.port,
        "seed": args.seed, "tasks": list(TASKS), "episodes": list(EPISODES),
        "roots_per_task": ROOTS_PER_TASK, "repeats": REPEATS,
        "model_action_horizon": 25, "action_cot_denoising_steps": 10, "final_denoising_steps": 10,
        "resize_size": 224, "num_steps_wait": 10,
    }
    with (output / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "run_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Existing continuation diagnostic uses a different configuration.")
        _write_json(config_path, config)
        if (output / "summary.json").exists():
            print((output / "summary.json").read_text(), flush=True)
            return
        candidate = FeedbackSelector.load(args.candidate_params)
        if candidate.variant != "current" or int(candidate.metadata.get("best_step", -1)) != 200:
            raise ValueError("This diagnostic requires the existing current step200 candidate.")
        bank = InitialStateBank(args.initial_state_bank)
        libero_eval._ensure_libero_import_path()
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        client = websocket_policy.WebsocketClientPolicy(args.host, args.port, ping_interval=None, ping_timeout=None)
        eval_args = _eval_args(args)
        roots_dir, episodes_dir = output / "roots", output / "episodes"
        roots_dir.mkdir(exist_ok=True)
        episodes_dir.mkdir(exist_ok=True)
        status_path = output / "status.json"
        reports, episode_reports = [], []
        try:
            for task in TASKS:
                bank.validate_presets(task, suite.get_task_init_states(task))
                task_roots = 0
                for episode in EPISODES:
                    filename = f"task{task:02d}_ep{episode:06d}"
                    root_path = roots_dir / f"{filename}.npz"
                    episode_path = episodes_dir / f"{filename}.json"
                    if task_roots >= ROOTS_PER_TASK:
                        episode_report = {
                            "status": "not_attempted_task_root_cap", "task_id": task, "episode_id": episode
                        }
                        _write_json(episode_path, episode_report)
                        episode_reports.append(episode_report)
                        continue
                    root = None
                    if root_path.exists():
                        root = load_root(root_path)
                        episode_report = root["source_report"]
                    elif episode_path.exists() and json.loads(episode_path.read_text())["status"] == "no_disagreement":
                        episode_report = json.loads(episode_path.read_text())
                    else:
                        _write_json(status_path, {
                            "status": "running", "phase": "source", "task_id": task, "episode_id": episode,
                            "completed_roots": len(reports),
                        })
                        root, episode_report = find_first_disagreement(
                            task_id=task, episode=episode, suite=suite, bank=bank, client=client,
                            candidate=candidate, args=eval_args,
                        )
                        if root is not None:
                            save_root(root_path, root)
                    _write_json(episode_path, episode_report)
                    episode_reports.append(episode_report)
                    if root is None:
                        continue
                    task_roots += 1
                    report = run_root(
                        root, root_path.with_suffix(".json"), suite=suite, bank=bank, client=client,
                        candidate=candidate, args=eval_args, status_path=status_path,
                    )
                    reports.append(report)
                    print(json.dumps({
                        "task_id": task, "episode_id": episode, "effects": report["effects"]
                    }), flush=True)
            summary = summarize(reports, episode_reports)
            _write_json(output / "summary.json", summary)
            _write_json(status_path, {
                "status": "complete", "num_roots": len(reports), "num_branches": summary["num_branches"]
            })
        except Exception as exc:
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(status="paused" if isinstance(exc, InputMismatch) else "failed",
                          error=str(exc), error_type=type(exc).__name__)
            _write_json(status_path, status)
            raise


if __name__ == "__main__":
    main(build_parser().parse_args())
