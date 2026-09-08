"""Collect paired continue/replan labels at scheduled observations inside an A chunk."""
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
import probe_execution_horizon_continuation as continuation
import probe_execution_horizon_first_divergence as first_probe

from openpi.execution_horizon.initial_states import InitialStateBank
from openpi.execution_horizon.midchunk import CHECK_AFTER
from openpi.execution_horizon.midchunk import MidchunkMonitor
from openpi.execution_horizon.midchunk import planned_gripper_check_step
from openpi.execution_horizon.midchunk import proprio_from_observation
from openpi.execution_horizon.trace import _physics_state


PARTITIONS = {"train": tuple(range(300, 308)), "early": (330, 331)}
EVENT_PARTITIONS = {"train": tuple(range(308, 313)), "early": (332, 333)}
TASKS = tuple(range(10))
REPEATS = 3
BRANCH_NAMES = ("continue", "replan")
TRIAL_FIELDS = ("success", "rpc", "policy", "segment_seconds", "calls", "steps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-state-bank", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--seed", type=int, default=97007)
    parser.add_argument("--protocol", choices=("fixed5", "gripper_event"), default="fixed5")
    return parser


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feedback._write_json(path, data)


def _eval_args(args: argparse.Namespace) -> argparse.Namespace:
    result = evaluator.build_parser().parse_args([
        "--output-dir", str(args.output_dir), "--initial-state-bank", str(args.initial_state_bank),
        "--host", args.host, "--port", str(args.port), "--seed", str(args.seed),
        "--modes", "ordered_transformer", "--model-action-horizon", "25",
        "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
        "--resize-size", "224", "--num-steps-wait", "10",
    ])
    result.protocol = args.protocol
    return result


def _gray_views(observation: dict[str, Any]) -> np.ndarray:
    views = []
    for name in ("agentview_image", "robot0_eye_in_hand_image"):
        image = observation.get(name)
        if image is None:
            return np.empty((0, 8, 8), dtype=np.float32)
        image = np.asarray(image)[::-1, ::-1, :3]
        gray = image.astype(np.float32).mean(axis=-1) / 255.0
        height, width = gray.shape
        blocks = gray.reshape(8, height // 8, 8, width // 8).mean(axis=(1, 3))
        views.append(blocks)
    return np.stack(views).astype(np.float32)


def _save_root(path: pathlib.Path, record: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **record)
    temporary.replace(path)


def _load_root(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _new_trials(repeats: int = REPEATS) -> dict[str, np.ndarray]:
    return {
        "trial_success": np.zeros((2, repeats), dtype=np.bool_),
        "trial_rpc": np.full((2, repeats), np.nan, dtype=np.float64),
        "trial_policy": np.full((2, repeats), np.nan, dtype=np.float64),
        "trial_segment_seconds": np.full((2, repeats), np.nan, dtype=np.float64),
        "trial_calls": np.zeros((2, repeats), dtype=np.int32),
        "trial_steps": np.zeros((2, repeats), dtype=np.int32),
        "trial_valid": np.zeros((2, repeats), dtype=np.bool_),
        "trial_root_physics_difference": np.full((2, repeats), np.nan, dtype=np.float64),
    }


def collect_source(
    *, task_id: int, episode: int, suite: Any, bank: InitialStateBank,
    client: Any, args: argparse.Namespace, feature_builder: MidchunkMonitor,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    env, description = libero_eval._get_libero_env(
        suite.get_task(task_id), libero_eval.LIBERO_ENV_RESOLUTION, args.seed
    )
    sampler = np.random.default_rng(np.random.SeedSequence([args.seed, task_id, episode, 0x4D4944]))
    prefix: list[np.ndarray] = []
    root = None
    eligible = 0
    calls = 0
    step = 0
    success = False
    previous_actions = None
    previous_h = 10
    previous_gripper = float(libero_eval.LIBERO_DUMMY_ACTION[6])
    protocol = getattr(args, "protocol", "fixed5")
    repeats = 2 if protocol == "gripper_event" else REPEATS
    started = time.perf_counter()
    try:
        env.reset()
        env.set_init_state(bank.state(task_id, episode))
        limit = libero_eval._max_steps(args.task_suite_name) + args.num_steps_wait
        environment_horizon = libero_eval._env_horizon(env)
        if environment_horizon is not None:
            limit = min(limit, environment_horizon)
        wait = np.repeat(np.asarray(libero_eval.LIBERO_DUMMY_ACTION, dtype=np.float32)[None], 10, axis=0)
        observation, step, success = continuation._execute(
            env, wait, step=step, limit=limit, recorded_actions=prefix
        )
        while not success and step < limit:
            start_step = step
            start_proprio = proprio_from_observation(observation)
            seed = feedback._collector()._root_seed(
                args.seed, task_id, episode, step, task_stride=feedback.TASK_SEED_STRIDE
            )
            result, _, _ = continuation._request_once(
                client, observation, description, args=args, step=step, limit=limit, seed=seed,
                previous_actions=previous_actions, previous_h=previous_h,
            )
            calls += 1
            actions = continuation._action_chunk(result)
            planned_h = feedback._selected_h(result)
            check_step = (
                planned_gripper_check_step(actions, planned_h, previous_gripper)
                if protocol == "gripper_event" else CHECK_AFTER if planned_h > CHECK_AFTER else None
            )
            start_gray = _gray_views(observation) if check_step is not None else None
            if check_step is not None:
                start_agentview_rgb = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
                start_wrist_rgb = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
            first_count = check_step if check_step is not None else planned_h
            observation, step, success = continuation._execute(
                env, actions[:first_count], step=step, limit=limit, recorded_actions=prefix
            )
            if check_step is not None and step - start_step == check_step and not success and step < limit:
                eligible += 1
                if int(sampler.integers(eligible)) == 0:
                    current_proprio = proprio_from_observation(observation)
                    temporal = feedback._vector(result, "execution_horizon_temporal_feature", 256)
                    inputs = {
                        "temporal_feature": temporal, "start_proprio": start_proprio,
                        "chunk_actions": actions, "planned_h": planned_h,
                        "episode_progress": step / limit, "current_proprio": current_proprio,
                        "executed_in_chunk": check_step,
                    }
                    physics, _ = _physics_state(env)
                    root = {
                        "schema_version": np.asarray(1, dtype=np.int32),
                        "task_id": np.asarray(task_id, dtype=np.int32), "episode_id": np.asarray(episode, dtype=np.int32),
                        "root_step": np.asarray(step, dtype=np.int32),
                        "chunk_start_step": np.asarray(start_step, dtype=np.int32),
                        "episode_step_limit": np.asarray(limit, dtype=np.int32),
                        "executed_in_chunk": np.asarray(check_step, dtype=np.int32),
                        "check_schedule": np.asarray(protocol),
                        "previous_gripper_command": np.asarray(previous_gripper, dtype=np.float32),
                        "planned_h": np.asarray(planned_h, dtype=np.int32),
                        "episode_progress": np.asarray(step / limit, dtype=np.float32),
                        "root_request_seed": np.asarray(seed, dtype=np.uint32),
                        "simulator_seed": np.asarray(args.seed, dtype=np.int32),
                        "source_decision_index": np.asarray(calls - 1, dtype=np.int32),
                        "prefix_actions": np.asarray(prefix, dtype=np.float32), "physics_state": physics,
                        "chunk_actions": actions.copy(), "remaining_actions": actions[check_step:planned_h].copy(),
                        "temporal_feature": temporal, "start_proprio": start_proprio.copy(),
                        "current_proprio": current_proprio, "raw_feature": feature_builder.build_features(inputs),
                        "start_gray": start_gray, "current_gray": _gray_views(observation),
                        "start_agentview_rgb": start_agentview_rgb, "start_wrist_rgb": start_wrist_rgb,
                        "current_agentview_rgb": np.asarray(observation["agentview_image"], dtype=np.uint8).copy(),
                        "current_wrist_rgb": np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy(),
                        **_new_trials(repeats),
                    }
            if not success and step < limit and planned_h > first_count:
                observation, step, success = continuation._execute(
                    env, actions[first_count:planned_h], step=step, limit=limit, recorded_actions=prefix
                )
            previous_actions = actions.copy()
            previous_h = step - start_step
            previous_gripper = float(prefix[-1][6])
        report = {
            "status": "complete" if root is not None else "no_eligible_chunk",
            "task_id": task_id, "episode_id": episode, "source_success": bool(success),
            "source_steps": step, "source_calls": calls, "eligible_chunks": eligible,
            "source_seconds": time.perf_counter() - started,
            "check_schedule": protocol,
            "sampling": "one uniform reservoir over nonterminal eligible scheduled checks in a complete A episode",
        }
        if root is not None:
            root.update(
                source_success=np.asarray(success, dtype=np.bool_), source_steps=np.asarray(step, dtype=np.int32),
                source_calls=np.asarray(calls, dtype=np.int32), eligible_chunks=np.asarray(eligible, dtype=np.int32),
                source_seconds=np.asarray(report["source_seconds"], dtype=np.float64),
            )
        return root, report
    finally:
        libero_eval._safe_close_env(env)


def collect_branch(
    record: dict[str, np.ndarray], *, action: int, repeat: int, suite: Any,
    bank: InitialStateBank, client: Any, args: argparse.Namespace,
) -> dict[str, Any]:
    root = {
        "task_id": int(record["task_id"]), "episode_id": int(record["episode_id"]),
        "step": int(record["root_step"]), "prefix_actions": record["prefix_actions"],
        "physics_state": record["physics_state"],
    }
    env = None
    try:
        env, description, observation, limit, reconstruction = first_probe._prepare_live_root(suite, bank, root, args)
        if not reconstruction["root_valid"] or limit != int(record["episode_step_limit"]):
            raise continuation.InputMismatch(f"Midchunk root reconstruction does not match: {reconstruction}")
        step = root["step"]
        executed_in_chunk = int(record["executed_in_chunk"])
        previous_actions = record["chunk_actions"].copy()
        previous_h = executed_in_chunk
        success = False
        calls = 0
        rpc = 0.0
        policy = 0.0
        started = time.perf_counter()
        if action == 0:
            observation, step, success = continuation._execute(
                env, record["remaining_actions"], step=step, limit=limit
            )
            previous_h = executed_in_chunk + step - root["step"]
        while not success and step < limit:
            seed = first_probe._continuation_seed(int(record["root_request_seed"]), repeat, calls)
            result, timing, _ = continuation._request_once(
                client, observation, description, args=args, step=step, limit=limit, seed=seed,
                previous_actions=previous_actions, previous_h=previous_h,
            )
            calls += 1
            rpc += float(timing["wall_ms"]) / 1000.0
            policy += float(timing["policy_ms"]) / 1000.0
            actions = continuation._action_chunk(result)
            planned_h = feedback._selected_h(result)
            before = step
            observation, step, success = continuation._execute(env, actions[:planned_h], step=step, limit=limit)
            previous_actions = actions.copy()
            previous_h = step - before
            if previous_h <= 0:
                raise continuation.InputMismatch("A continuation did not execute any action.")
        return {
            "success": bool(success), "rpc": rpc, "policy": policy,
            "segment_seconds": time.perf_counter() - started, "calls": calls, "steps": step - root["step"],
            "root_physics_difference": reconstruction["root_physics_max_abs_difference"],
        }
    finally:
        if env is not None:
            libero_eval._safe_close_env(env)


def _root_report(record: dict[str, np.ndarray]) -> dict[str, Any]:
    branches = []
    valid = np.asarray(record["trial_valid"], dtype=bool)
    for action in range(2):
        for repeat in range(valid.shape[1]):
            if valid[action, repeat]:
                row = {"choice": BRANCH_NAMES[action], "action": action, "repeat": repeat}
                for field in TRIAL_FIELDS:
                    row[field] = np.asarray(record["trial_" + field][action, repeat]).item()
                row["root_physics_difference"] = float(record["trial_root_physics_difference"][action, repeat])
                branches.append(row)
    return {
        "status": "complete" if bool(valid.all()) else "running",
        "task_id": int(record["task_id"]), "episode_id": int(record["episode_id"]),
        "root_step": int(record["root_step"]), "chunk_start_step": int(record["chunk_start_step"]),
        "planned_h": int(record["planned_h"]), "executed_in_chunk": int(record["executed_in_chunk"]),
        "check_schedule": str(record.get("check_schedule", "fixed5")),
        "episode_step_limit": int(record["episode_step_limit"]), "branches": branches,
        "branch_order": list(BRANCH_NAMES), "continuation": "A for both choices",
        "root_reconstruction": "fresh same-seed environment and complete recorded float32 action prefix",
        "timing": "starts at midchunk after reconstruction; the already-paid chunk-start RPC is excluded",
        "model_inputs": "raw_feature only; physics, terminal source outcome and gray caches are excluded",
    }


def main(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    partitions = EVENT_PARTITIONS if args.protocol == "gripper_event" else PARTITIONS
    repeats = 2 if args.protocol == "gripper_event" else REPEATS
    configuration = {
        "initial_state_bank": str(args.initial_state_bank.resolve()), "host": args.host, "port": args.port,
        "seed": args.seed, "partitions": {name: list(ids) for name, ids in partitions.items()},
        "tasks": list(TASKS), "repeats": repeats, "check_after": CHECK_AFTER,
        "NFE": [10, 10], "resize_size": 224, "num_steps_wait": 10,
    }
    if args.protocol == "gripper_event":
        configuration.update(protocol=args.protocol, check_schedule=args.protocol, check_after=None)
    with (output / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "run_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise ValueError("The midchunk output belongs to a different collection protocol.")
        _write_json(config_path, configuration)
        if (output / "summary.json").exists():
            print((output / "summary.json").read_text(), flush=True)
            return
        bank = InitialStateBank(args.initial_state_bank)
        libero_eval._ensure_libero_import_path()
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()["libero_10"]()
        client = websocket_policy.WebsocketClientPolicy(args.host, args.port, ping_interval=None, ping_timeout=None)
        eval_args = _eval_args(args)
        builder = MidchunkMonitor.initialize("fresh", check_schedule=args.protocol)
        status_path = output / "status.json"
        counts = {name: 0 for name in partitions}
        completed_branches = 0
        no_eligible = []
        try:
            for partition, episodes in partitions.items():
                roots_dir = output / partition / "roots"
                roots_dir.mkdir(parents=True, exist_ok=True)
                for task in TASKS:
                    bank.validate_presets(task, suite.get_task_init_states(task))
                    for episode in episodes:
                        filename = f"task{task:02d}_ep{episode:06d}"
                        path = roots_dir / f"{filename}.npz"
                        source_path = output / partition / "episodes" / f"{filename}.json"
                        if path.exists():
                            record = _load_root(path)
                        elif source_path.exists() and json.loads(source_path.read_text())["status"] == "no_eligible_chunk":
                            no_eligible.append({"partition": partition, "task_id": task, "episode_id": episode})
                            continue
                        else:
                            _write_json(status_path, {
                                "status": "running", "phase": "source", "partition": partition,
                                "task_id": task, "episode_id": episode, "completed_branches": completed_branches,
                            })
                            record, source_report = collect_source(
                                task_id=task, episode=episode, suite=suite, bank=bank, client=client,
                                args=eval_args, feature_builder=builder,
                            )
                            if record is not None:
                                _save_root(path, record)
                            _write_json(source_path, source_report)
                            if record is None:
                                no_eligible.append({"partition": partition, "task_id": task, "episode_id": episode})
                                continue
                        for repeat in range(record["trial_valid"].shape[1]):
                            order = (0, 1) if repeat % 2 == 0 else (1, 0)
                            for action in order:
                                if not bool(record["trial_valid"][action, repeat]):
                                    _write_json(status_path, {
                                        "status": "running", "phase": "branches", "partition": partition,
                                        "task_id": task, "episode_id": episode, "root_step": int(record["root_step"]),
                                        "choice": BRANCH_NAMES[action], "repeat": repeat,
                                        "completed_branches": completed_branches,
                                    })
                                    outcome = collect_branch(
                                        record, action=action, repeat=repeat, suite=suite, bank=bank,
                                        client=client, args=eval_args,
                                    )
                                    for field in TRIAL_FIELDS:
                                        record["trial_" + field][action, repeat] = outcome[field]
                                    record["trial_root_physics_difference"][action, repeat] = outcome["root_physics_difference"]
                                    record["trial_valid"][action, repeat] = True
                                    _save_root(path, record)
                                completed_branches += 1
                                _write_json(path.with_suffix(".json"), _root_report(record))
                        counts[partition] += 1
                        print(json.dumps({"partition": partition, "task_id": task, "episode_id": episode,
                                          "completed_roots": counts, "completed_branches": completed_branches}), flush=True)
            summary = {
                "status": "complete", "num_roots": sum(counts.values()), "num_branches": completed_branches,
                "train_roots": counts["train"], "early_roots": counts["early"],
                "source_episode_budget": sum(map(len, partitions.values())) * len(TASKS),
                "branch_budget": sum(map(len, partitions.values())) * len(TASKS) * 2 * repeats,
                "no_eligible_episodes": no_eligible,
                "branch_order": list(BRANCH_NAMES), "feature_dim": builder.feature_dim,
                "fresh_feature_start": builder.pre_feature_dim, "check_schedule": args.protocol,
                "check_after": CHECK_AFTER if args.protocol == "fixed5" else None, "gray_cache_used_by_model": False,
                "seed_formula": "chunk-start root seed + 100000000 + repeat * 20000000 + continuation call index",
                "training_root_selection": "one reservoir-sampled nonterminal eligible scheduled state per complete A episode",
            }
            _write_json(output / "summary.json", summary)
            _write_json(status_path, summary)
        except Exception as exc:
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(status="paused" if isinstance(exc, continuation.InputMismatch) else "failed",
                          error=str(exc), error_type=type(exc).__name__)
            _write_json(status_path, status)
            raise


if __name__ == "__main__":
    main(build_parser().parse_args())
