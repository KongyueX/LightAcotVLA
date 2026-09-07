"""Replay the fixed A roots and attach predictor inputs to their paired labels."""
# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import collect_execution_horizon_feedback as feedback
import numpy as np

CACHE_FIELDS = {
    "prefix_feature": "execution_horizon_prefix_feature",
    "state": "execution_horizon_state_normalized",
    "coarse_actions": "execution_horizon_coarse_actions_normalized",
    "final_actions": "execution_horizon_final_actions_normalized",
    "previous_actions": "execution_horizon_previous_actions_normalized",
    "previous_h": "execution_horizon_previous_h",
    "previous_valid": "execution_horizon_previous_valid",
    "budget_balance": "execution_horizon_budget_balance",
    "episode_progress": "execution_horizon_episode_progress",
    "prefix_tokens": "execution_horizon_prefix_tokens",
    "prefix_mask": "execution_horizon_prefix_mask",
    "expert_hidden": "execution_horizon_expert_hidden",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fresh-paired", action="store_true")
    return parser


def collect_fresh(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for path in sorted(args.source_dir.resolve().rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            task, episode = int(data["task_id"]), int(data["episode_id"])
        groups.setdefault(task, {})[episode] = path
    if not groups:
        raise ValueError("Source split has no task/episode identities.")
    total, completed = sum(len(group) for group in groups.values()), 0
    for task, group in sorted(groups.items()):
        original = json.loads((next(iter(group.values())).parent / "run_config.json").read_text())
        command = [
            "--initial-state-bank", original["initial_state_bank"], "--episodes", *map(str, sorted(group)),
            "--task-start", str(task), "--max-tasks", "1", "--output-dir", str(output / f"task{task:02d}"),
            "--host", args.host, "--port", str(args.port), "--seed", str(original["seed"]),
            "--branch-repeats", "5", "--architecture-cache",
        ]
        feedback._write_json(output / "status.json", {
            "status": "running", "task_id": task, "completed_roots": completed, "expected_roots": total,
            "task_status": str(output / f"task{task:02d}/status.json"), "labels_reused": False,
        })
        feedback.main(feedback.build_parser().parse_args(command))
        completed += len(group)
    summary = {
        "status": "complete", "completed_roots": completed, "expected_roots": total,
        "labels_reused": False, "counterfactual_branches_run": completed * 25,
        "source_split": str(args.source_dir.resolve()), "feature_label_alignment": "same real source call",
    }
    feedback._write_json(output / "summary.json", summary)
    feedback._write_json(output / "status.json", summary)


def check_replay(
    record: dict[str, np.ndarray], result: dict[str, Any], physics: np.ndarray,
    *, step: int, seed: int, previous_h: int, elapsed_steps: int, history_valid: bool,
    previous_prefix: np.ndarray, previous_state: np.ndarray,
) -> dict[str, float]:
    if (step, seed, previous_h) != tuple(int(record[name]) for name in ("root_step", "root_seed", "previous_h")):
        raise ValueError("Replayed root step, seed or previous H differs from the labeled root.")
    if (elapsed_steps, history_valid) != (int(record["elapsed_steps"]), bool(record["history_valid"])):
        raise ValueError("Replayed root history validity or elapsed steps differ from the labeled root.")
    comparisons = {
        "physics": (physics, record["physics_state"], 1e-8),
        "actions": (result["actions"], record["primary_actions"], 1e-5),
        "prefix": (result["execution_horizon_prefix_feature"], record["prefix_feature"], 1e-5),
        "state": (result["execution_horizon_state_normalized"], record["state"], 1e-6),
        "previous_prefix": (previous_prefix, record["previous_prefix_feature"], 1e-5),
        "previous_state": (previous_state, record["previous_state"], 1e-6),
        "logits": (result["execution_horizon_ordered_continuation_logits"], record["continuation_logits"], 1e-5),
        "probability": (result["execution_horizon_ordered_horizon_probability"], record["ordered_probability"], 1e-5),
        "temporal": (result["execution_horizon_temporal_feature"], record["temporal_feature"], 1e-5),
    }
    differences = {}
    for name, (actual_value, expected_value, tolerance) in comparisons.items():
        actual, expected = np.asarray(actual_value), np.asarray(expected_value)
        if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
            raise ValueError(f"Replayed {name} has the wrong shape or non-finite values.")
        difference = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        differences[name] = difference
        if difference > tolerance:
            raise ValueError(f"Replayed {name} differs by {difference:.9g}, above {tolerance}; do not reuse labels.")
    if feedback._selected_h(result) != int(record["selected_h"]):
        raise ValueError("Replayed A selected a different H; do not reuse labels.")
    return differences


def replay_root(path: pathlib.Path, *, client: Any, suite: Any, output: pathlib.Path) -> dict[str, Any]:
    base = feedback._collector()
    libero = base.libero_eval
    with np.load(path, allow_pickle=False) as archive:
        record = {name: archive[name] for name in archive.files}
    source_config = json.loads((path.parent / "run_config.json").read_text())
    args = argparse.Namespace(**source_config)
    task_id, episode_id = int(record["task_id"]), int(record["episode_id"])
    bank = base.horizon_initial_states.InitialStateBank(args.initial_state_bank)
    bank.validate_presets(task_id, suite.get_task_init_states(task_id))
    target_index = int(record["source_decision_index"])
    env, description = libero._get_libero_env(suite.get_task(task_id), libero.LIBERO_ENV_RESOLUTION, args.seed)
    try:
        env.reset()
        observation = env.set_init_state(bank.state(task_id, episode_id))
        step_limit = libero._max_steps(args.task_suite_name) + args.num_steps_wait
        environment_horizon = libero._env_horizon(env)
        if environment_horizon:
            step_limit = min(step_limit, environment_horizon)
        step = 0
        for _ in range(args.num_steps_wait):
            observation, _, done, _ = env.step(libero.LIBERO_DUMMY_ACTION)
            step += 1
            if done:
                raise RuntimeError("Episode ended during the fixed initial wait.")
        previous_actions = None
        previous_h = 10
        previous_step = step
        previous_prefix = np.zeros(2048, dtype=np.float32)
        previous_state = np.zeros(32, dtype=np.float32)
        for decision_index in range(target_index + 1):
            seed = base._root_seed(args.seed, task_id, episode_id, step, task_stride=feedback.TASK_SEED_STRIDE)
            policy_input = libero._observation_to_policy_input(observation, description, args.resize_size)
            policy_input["action_cot_final_denoising_steps"] = np.asarray(args.final_denoising_steps, dtype=np.int32)
            policy_input["action_cot_absolute_decision_step"] = np.asarray(step, dtype=np.int32)
            target = decision_index == target_index
            if target:
                policy_input["execution_horizon_export_architecture_cache"] = np.asarray(1, dtype=np.bool_)
            result = base._policy_request(
                client, policy_input, seed=seed, args=args, teacher=False, profile=True, run_student=True,
                previous_actions=previous_actions, previous_h=previous_h, budget_balance=0.5,
                episode_progress=float(np.clip(step / max(step_limit, 1), 0.0, 1.0)),
            )
            if target:
                differences = check_replay(
                    record, result, base._capture_snapshot(env).physics_state,
                    step=step, seed=seed, previous_h=previous_h,
                    elapsed_steps=step - previous_step if previous_actions is not None else 0,
                    history_valid=previous_actions is not None,
                    previous_prefix=previous_prefix, previous_state=previous_state,
                )
                inputs = {"input_" + name: np.asarray(result[key]).copy() for name, key in CACHE_FIELDS.items()}
                for name, value in inputs.items():
                    if not np.all(np.isfinite(value)):
                        raise ValueError(f"Non-finite architecture input: {name}")
                if inputs["input_expert_hidden"].shape != (25, 1024):
                    raise ValueError("The H25 gemma_300m expert must expose hidden[25,1024].")
                mask = inputs["input_prefix_mask"].astype(bool)
                pooled = inputs["input_prefix_tokens"][mask].mean(axis=0, dtype=np.float64)
                if not np.allclose(pooled, record["prefix_feature"], rtol=0.0, atol=1e-5):
                    raise ValueError("Full prefix cache does not reproduce the labeled pooled prefix.")
                record.update(inputs)
                record["architecture_cache_schema"] = np.asarray(1, dtype=np.int32)
                destination = output / path.name
                with destination.with_suffix(".npz.tmp").open("wb") as handle:
                    np.savez_compressed(handle, **record)
                destination.with_suffix(".npz.tmp").replace(destination)
                audit = {
                    "source": str(path), "cache": str(destination), "task_id": task_id,
                    "episode_id": episode_id, "root_step": step, "root_seed": seed,
                    "replayed_policy_calls": decision_index + 1, "max_abs_difference": differences,
                    "prefix_tokens": list(inputs["input_prefix_tokens"].shape),
                    "expert_hidden": list(inputs["input_expert_hidden"].shape),
                    "labels_reused": True,
                }
                feedback._write_json(destination.with_suffix(".json"), audit)
                return audit
            actions = np.asarray(result["actions"], dtype=np.float32)
            selected_h = feedback._selected_h(result)
            decision_step = step
            observation, step, success = feedback._execute(
                env, actions[:selected_h], step=step, step_limit=step_limit,
            )
            if success or step == decision_step or step >= step_limit:
                raise RuntimeError(f"A replay ended before labeled root {path.name}.")
            previous_actions, previous_h = actions.copy(), step - decision_step
            previous_step = decision_step
            previous_prefix = np.asarray(result["execution_horizon_prefix_feature"], dtype=np.float32).copy()
            previous_state = np.asarray(result["execution_horizon_state_normalized"], dtype=np.float32).copy()
    finally:
        libero._safe_close_env(env)
    raise RuntimeError("No labeled root reached.")


def main(args: argparse.Namespace) -> None:
    if args.fresh_paired:
        collect_fresh(args)
        return
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    paths = sorted(source.rglob("*.npz"))
    if not paths or len({path.name for path in paths}) != len(paths):
        raise ValueError("Feature source must contain unique labeled root NPZ names.")
    if args.limit:
        paths = paths[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    config = {"source_dir": str(source), "host": args.host, "port": args.port, "source_paths": list(map(str, paths))}
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Architecture cache directory belongs to a different fixed collection.")
    feedback._write_json(config_path, config)
    base = feedback._collector()
    base.libero_eval._ensure_libero_import_path()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    client = base.websocket_policy.WebsocketClientPolicy(args.host, args.port, ping_interval=None, ping_timeout=None)
    completed = 0
    try:
        for path in paths:
            destination = output / path.name
            if not (destination.exists() and destination.with_suffix(".json").exists()):
                feedback._write_json(output / "status.json", {
                    "status": "running", "root": path.name, "completed_roots": completed, "expected_roots": len(paths),
                })
                audit = replay_root(path, client=client, suite=suite, output=output)
                print(json.dumps(audit), flush=True)
            completed += 1
        result = {
            "status": "complete", "completed_roots": completed, "expected_roots": len(paths),
            "labels_reused": True, "counterfactual_branches_run": 0, **config,
        }
        feedback._write_json(output / "summary.json", result)
        feedback._write_json(output / "status.json", result)
    except Exception as exc:
        feedback._write_json(output / "status.json", {
            "status": "failed", "completed_roots": completed, "expected_roots": len(paths),
            "error": str(exc), "error_type": type(exc).__name__,
        })
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
