#!/usr/bin/env python3
"""Collect ordered-policy counterfactuals with dynamic branch continuation."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import pathlib
import socket
import subprocess
import sys
from typing import Any

PROTOCOL = "h25_ordered_dynamic_continuation_round4_v1"
TASK_IDS = tuple(range(10))
ROLE_TARGETS = {"train": 10, "early_stop": 3, "calibration": 3, "dev_audit": 2}
CANDIDATE_HORIZONS = (5, 10, 15, 20, 25)
SEED = 47_007
TASK_SEED_STRIDE = 250_000_000
BRANCH_SEED_STRIDE = 20_000_000
SOURCE_ITERATION = 4
GROUP_ID_TASK_MULTIPLIER = 1_000_000_000
ALLOWED_OFFSETS = tuple(range(3, 11))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-state-bank", required=True)
    parser.add_argument("--round3-breadth-summary", required=True)
    parser.add_argument("--round3-selection", required=True)
    parser.add_argument("--base-split-manifest", required=True)
    parser.add_argument("--predictor-dir", required=True)
    parser.add_argument("--round3-collection-dir", default=None)
    parser.add_argument("--collector-script", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--episodes-per-attempt", type=int, default=10)
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any], *, once: bool = False) -> None:
    normalized = json.loads(json.dumps(payload))
    if once and path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"Existing immutable JSON differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _balanced_order(episodes: list[int], rotation: int) -> list[int]:
    pools = {offset: [] for offset in ALLOWED_OFFSETS}
    for episode in sorted(set(episodes)):
        if episode % 12 in pools:
            pools[episode % 12].append(episode)
    offsets = list(ALLOWED_OFFSETS)
    rotation %= len(offsets)
    offsets = offsets[rotation:] + offsets[:rotation]
    ordered: list[int] = []
    while any(pools.values()):
        for offset in offsets:
            if pools[offset]:
                ordered.append(pools[offset].pop(0))
    return ordered


def _old_attempted(collection: pathlib.Path, task: int, role: str) -> set[int]:
    attempted: set[int] = set()
    for config_path in collection.glob(f"task{task:02d}/{role}/attempt-*/run_config.json"):
        attempted.update(int(value) for value in json.loads(config_path.read_text()).get("episodes", ()))
    return attempted


def _base_validation_by_task(manifest: dict[str, Any]) -> dict[int, list[int]]:
    values = manifest.get("validation_group_ids")
    if values is None:
        values = manifest.get("early_stop_group_ids")
    if values is None:
        raise ValueError("Base split manifest has neither validation_group_ids nor early_stop_group_ids.")
    result = {task: [] for task in TASK_IDS}
    for raw in values:
        group = int(raw)
        task, episode = divmod(group, GROUP_ID_TASK_MULTIPLIER)
        if task in result and episode % 12 in ALLOWED_OFFSETS:
            result[task].append(episode)
    return result


def build_selection(
    breadth: dict[str, Any],
    round3: dict[str, Any],
    base_manifest: dict[str, Any],
    round3_collection: pathlib.Path,
) -> dict[str, Any]:
    if breadth.get("status") != "complete" or breadth.get("purpose") != "breadth_first_warm_start":
        raise ValueError("Round-3 breadth summary is not complete.")
    base_validation = _base_validation_by_task(base_manifest)
    tasks: dict[str, Any] = {}
    for task in TASK_IDS:
        breadth_roles = breadth["realized_by_task"][str(task)]
        round3_roles = round3["tasks"][str(task)]["roles"]
        role_plan: dict[str, Any] = {}

        train_actual = [int(value) for value in breadth_roles["train"]["episodes"]]
        if len(train_actual) != ROLE_TARGETS["train"]:
            raise ValueError(f"Task {task} must provide exactly ten realized Round-3 train groups.")
        train_pool = [
            int(value)
            for key in ("primary", "fallback")
            for value in round3_roles["train"][key]
        ]
        train_fallback = [
            episode
            for episode in _balanced_order(train_pool, task)
            if episode not in _old_attempted(round3_collection, task, "train")
        ]
        role_plan["train"] = {"primary": _balanced_order(train_actual, task), "fallback": train_fallback}

        early_order = _balanced_order(base_validation[task], task + 1)
        if len(early_order) < ROLE_TARGETS["early_stop"]:
            raise ValueError(f"Task {task} lacks three eligible base validation groups.")
        role_plan["early_stop"] = {
            "primary": early_order[: ROLE_TARGETS["early_stop"]],
            "fallback": early_order[ROLE_TARGETS["early_stop"] :],
        }

        for role, rotation in (("calibration", task + 2), ("dev_audit", task + 3)):
            actual = _balanced_order([int(value) for value in breadth_roles[role]["episodes"]], rotation)
            target = ROLE_TARGETS[role]
            if len(actual) < target:
                raise ValueError(f"Task {task}/{role} lacks realized Round-3 groups.")
            old_pool = [
                int(value)
                for key in ("primary", "fallback")
                for value in round3_roles[role][key]
            ]
            old_attempted = _old_attempted(round3_collection, task, role)
            unused = [episode for episode in _balanced_order(old_pool, rotation) if episode not in old_attempted]
            role_plan[role] = {"primary": actual[:target], "fallback": [*actual[target:], *unused]}

        seen: dict[int, str] = {}
        for role, plan in role_plan.items():
            if len(plan["primary"]) != ROLE_TARGETS[role]:
                raise ValueError(f"Task {task}/{role} has the wrong primary count.")
            if len(plan["fallback"]) != len(set(plan["fallback"])):
                raise ValueError(f"Task {task}/{role} fallback contains duplicates.")
            for episode in (*plan["primary"], *plan["fallback"]):
                previous = seen.setdefault(episode, role)
                if previous != role:
                    raise ValueError(f"Task {task} episode {episode} crosses roles {previous}/{role}.")
        tasks[str(task)] = {"roles": role_plan}
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "selection_semantics": "Role-preserving, outcome-blind Round-4 primary and fallback episode groups.",
        "role_targets_per_task": ROLE_TARGETS,
        "tasks": tasks,
    }


def build_collector_command(
    *,
    python: pathlib.Path,
    collector: pathlib.Path,
    output: pathlib.Path,
    bank: pathlib.Path,
    host: str,
    port: int,
    task: int,
    episodes: list[int],
) -> list[str]:
    return [
        str(python), str(collector), "--host", host, "--port", str(port), "--output-dir", str(output),
        "--task-suite-name", "libero_10", "--task-start", str(task), "--max-tasks", "1",
        "--episode-ids", *[str(value) for value in episodes], "--initial-state-bank", str(bank),
        "--num-steps-wait", "10", "--resize-size", "224", "--seed", str(SEED),
        "--root-seed-task-stride", str(TASK_SEED_STRIDE), "--teacher-samples", "20",
        "--action-cot-denoising-steps", "10", "--root-stride-calls", "1",
        "--root-call-offset-cycle", "12", "--max-roots-per-episode", "1", "--records-per-shard", "10",
        "--source-iteration", str(SOURCE_ITERATION), "--candidate-horizons",
        *[str(value) for value in CANDIDATE_HORIZONS], "--reference-horizon", "10",
        "--model-action-horizon", "25", "--model-coarse-horizon", "15", "--model-action-dim", "32",
        "--model-state-dim", "32", "--prefix-feature-dim", "2048", "--prefix-token-count", "1024",
        "--source-policy", "current_student", "--continuation-policy", "current_student",
        "--branch-repeats", "3", "--repeat-branch-horizons", *[str(value) for value in CANDIDATE_HORIZONS],
        "--branch-repeat-seed-stride", str(BRANCH_SEED_STRIDE), "--student-mode", "ordered_transformer",
        "--debug-failure-videos", "0",
    ]


def _scan_attempts(role_dir: pathlib.Path, task: int) -> tuple[set[int], set[tuple[int, int]], int, list[str]]:
    h5py = importlib.import_module("h5py")

    attempted: set[int] = set()
    roots: set[tuple[int, int]] = set()
    data_dirs: list[str] = []
    maximum = -1
    for attempt in sorted(role_dir.glob("attempt-*")):
        maximum = max(maximum, int(attempt.name.removeprefix("attempt-")))
        config = json.loads((attempt / "run_config.json").read_text())
        attempted.update(int(value) for value in config["episodes"])
        exit_path = attempt / "collector.exit"
        if not exit_path.exists():
            raise RuntimeError(f"Unclosed attempt must not be rerun: {attempt}")
        if exit_path.read_text().strip() != "0":
            raise RuntimeError(f"Failed attempt requires diagnosis: {attempt}")
        data = attempt / "data"
        summary = json.loads((data / "summary.json").read_text())
        if summary.get("status") != "complete":
            raise RuntimeError(f"Attempt lacks a complete collector summary: {attempt}")
        attempt_roots: set[tuple[int, int]] = set()
        for shard in data.glob("shard-*.h5"):
            with h5py.File(shard, "r") as handle:
                attempt_roots.update((int(t), int(e)) for t, e in zip(handle["task_id"][:], handle["episode_id"][:], strict=True))
        if any(root[0] != task or root[1] not in config["episodes"] for root in attempt_roots):
            raise ValueError(f"Attempt contains an unexpected task/episode root: {attempt}")
        if len(attempt_roots) != int(summary.get("num_records", -1)):
            raise ValueError(f"Collector summary/root count mismatch: {attempt}")
        if roots.intersection(attempt_roots):
            raise ValueError(f"Attempts contain duplicate roots under {role_dir}")
        roots.update(attempt_roots)
        data_dirs.append(str(data))
    return attempted, roots, maximum, data_dirs


def _run_attempt(
    role_dir: pathlib.Path,
    *,
    index: int,
    role: str,
    task: int,
    episodes: list[int],
    command: list[str],
) -> None:
    attempt = role_dir / f"attempt-{index:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    _write_json(attempt / "run_config.json", {"protocol": PROTOCOL, "role": role, "task": task, "episodes": episodes}, once=True)
    with (attempt / "collector.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    (attempt / "collector.exit").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def main(args: argparse.Namespace) -> None:
    if args.episodes_per_attempt <= 0 or args.episodes_per_attempt > 10:
        raise ValueError("episodes_per_attempt must lie in [1, 10].")
    output = pathlib.Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        breadth_path = pathlib.Path(args.round3_breadth_summary).resolve()
        round3_path = pathlib.Path(args.round3_selection).resolve()
        base_path = pathlib.Path(args.base_split_manifest).resolve()
        predictor = pathlib.Path(args.predictor_dir).resolve()
        bank = pathlib.Path(args.initial_state_bank).resolve()
        code = pathlib.Path(args.code_dir).resolve()
        python = pathlib.Path(args.python).absolute()
        collector = pathlib.Path(args.collector_script).resolve() if args.collector_script else code / "scripts/collect_execution_horizon_counterfactuals.py"
        round3_collection = pathlib.Path(args.round3_collection_dir).resolve() if args.round3_collection_dir else breadth_path.parent.parent
        for required in (breadth_path, round3_path, base_path, predictor / "params", bank, collector, python):
            if not required.exists():
                raise FileNotFoundError(required)
        protocol = {
            "protocol": PROTOCOL, "code_dir": str(code), "code_commit": args.code_commit,
            "collector_script": str(collector), "python": str(python), "initial_state_bank": str(bank),
            "round3_breadth_summary": str(breadth_path), "round3_selection": str(round3_path),
            "base_split_manifest": str(base_path), "predictor_dir": str(predictor),
            "predictor_params": str(predictor / "params"), "host": args.host, "port": args.port,
            "seed": SEED, "task_seed_stride": TASK_SEED_STRIDE, "branch_seed_stride": BRANCH_SEED_STRIDE,
            "source_iteration": SOURCE_ITERATION, "candidate_horizons": CANDIDATE_HORIZONS,
            "source_policy": "current_student", "continuation_policy": "current_student",
            "student_mode": "ordered_transformer", "role_targets_per_task": ROLE_TARGETS,
        }
        _write_json(output / "run_config.json", protocol, once=True)
        selection_path = output / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
        else:
            selection = build_selection(
                json.loads(breadth_path.read_text()), json.loads(round3_path.read_text()),
                json.loads(base_path.read_text()), round3_collection,
            )
            _write_json(selection_path, selection, once=True)
        if (output / "summary.json").exists():
            summary = json.loads((output / "summary.json").read_text())
            if summary.get("status") != "complete" or int(summary.get("num_roots", -1)) != 180:
                raise ValueError("Existing Round-4 summary is not a complete 180-root result.")
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        with socket.create_connection((args.host, args.port), timeout=3):
            pass

        groups = {role: [] for role in ROLE_TARGETS}
        data_dirs: list[str] = []
        by_task: dict[str, Any] = {}
        for task in TASK_IDS:
            by_task[str(task)] = {}
            for role, target in ROLE_TARGETS.items():
                role_dir = output / f"task{task:02d}/{role}"
                role_dir.mkdir(parents=True, exist_ok=True)
                while True:
                    attempted, roots, maximum, completed_dirs = _scan_attempts(role_dir, task)
                    if len(roots) >= target:
                        break
                    plan = selection["tasks"][str(task)]["roles"][role]
                    available = [episode for episode in (*plan["primary"], *plan["fallback"]) if episode not in attempted]
                    if not available:
                        raise RuntimeError(f"Task {task}/{role} exhausted its same-role fallback pool.")
                    episodes = available[: min(target - len(roots), args.episodes_per_attempt)]
                    data = role_dir / f"attempt-{maximum + 1:04d}/data"
                    command = build_collector_command(
                        python=python, collector=collector, output=data, bank=bank, host=args.host,
                        port=args.port, task=task, episodes=episodes,
                    )
                    _run_attempt(role_dir, index=maximum + 1, role=role, task=task, episodes=episodes, command=command)
                if len(roots) != target:
                    raise ValueError(f"Task {task}/{role} produced {len(roots)} roots; expected {target}.")
                episodes = sorted(episode for _, episode in roots)
                groups[role].extend(task * GROUP_ID_TASK_MULTIPLIER + episode for episode in episodes)
                data_dirs.extend(completed_dirs)
                by_task[str(task)][role] = {"roots": target, "episodes": episodes, "data_dirs": completed_dirs}
                _write_json(output / "status.json", {"status": "collecting", "task": task, "role": role, "roots": sum(map(len, groups.values())), "target": 180})

        split = {
            "split_schema_version": 2,
            "split_roles": {"train": "train", "early_stop": "early_stop", "calibration": "calibration", "dev_audit": "development_audit"},
            **{f"{role}_group_ids": sorted(values) for role, values in groups.items()},
        }
        _write_json(output / "split_manifest.json", split, once=True)
        summary = {
            "status": "complete", "protocol": PROTOCOL, "num_roots": sum(map(len, groups.values())),
            "roots_by_role": {role: len(values) for role, values in groups.items()},
            "realized_by_task": by_task, "data_dirs": sorted(set(data_dirs)),
            "split_manifest": str(output / "split_manifest.json"), "predictor_dir": str(predictor),
            "source_policy": "current_student", "continuation_policy": "current_student",
        }
        _write_json(output / "summary.json", summary, once=True)
        _write_json(output / "status.json", {"status": "complete", "roots": 180, "target": 180})
        (output / "controller.exit").write_text("0\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
