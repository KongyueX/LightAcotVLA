"""Collect one independent counterfactual root per requested LIBERO episode."""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import subprocess
import sys
import time
from typing import Any

import h5py

from openpi.execution_horizon import dataset as horizon_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-start", type=int, required=True)
    parser.add_argument("--episode-end", type=int, required=True, help="Inclusive episode ID.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--collector-script", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--root-call-offset-cycle", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--branch-repeats", type=int, default=3)
    parser.add_argument("--teacher-samples", type=int, default=20)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--prefix-token-count", type=int, default=1024)
    parser.add_argument("--source-iteration", type=int, default=1)
    parser.add_argument("--records-per-shard", type=int, default=10)
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _server_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3.0):
            return True
    except OSError:
        return False


def _episode_roots(data_dir: pathlib.Path, task_id: int) -> dict[int, tuple[int, int]]:
    roots: dict[int, tuple[int, int]] = {}
    for shard in horizon_dataset.discover_shards((data_dir,)):
        with h5py.File(shard, "r") as handle:
            tasks = handle["task_id"][:]
            episodes = handle["episode_id"][:]
            steps = handle["decision_step"][:]
            seeds = handle["root_seed"][:]
            for task, episode_value, step, seed in zip(tasks, episodes, steps, seeds, strict=True):
                if int(task) != task_id:
                    raise ValueError(f"Unexpected task {int(task)} in task{task_id} fresh-root output.")
                episode = int(episode_value)
                if episode in roots:
                    raise ValueError(f"Episode {episode} has multiple roots in task{task_id} output.")
                roots[episode] = (int(step), int(seed))
    return roots


def _complete_output(data_dir: pathlib.Path, task_id: int, expected_episodes: set[int]) -> bool:
    summary_path = data_dir / "summary.json"
    manifest_path = data_dir / "manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        return False
    observed = set(_episode_roots(data_dir, task_id))
    if not observed.issubset(expected_episodes):
        raise ValueError(f"Unexpected episodes in {data_dir}: {sorted(observed - expected_episodes)}")
    return True


def _collector_command(
    *,
    args: argparse.Namespace,
    collector_script: pathlib.Path,
    task_id: int,
    episode_ids: list[int],
    output_dir: pathlib.Path,
    offset_cycle: int,
) -> list[str]:
    return [
        sys.executable,
        str(collector_script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output-dir",
        str(output_dir),
        "--task-start",
        str(task_id),
        "--max-tasks",
        "1",
        "--episode-ids",
        *[str(value) for value in episode_ids],
        "--candidate-horizons",
        "5",
        "10",
        "15",
        "20",
        "25",
        "--reference-horizon",
        "10",
        "--model-action-horizon",
        "25",
        "--model-coarse-horizon",
        "15",
        "--continuation-policy",
        "fixed_h",
        "--fixed-continuation-horizon",
        "5",
        "--branch-repeats",
        str(args.branch_repeats),
        "--repeat-branch-horizons",
        "5",
        "10",
        "15",
        "20",
        "25",
        "--prefix-token-count",
        str(args.prefix_token_count),
        "--root-stride-calls",
        "1",
        "--root-call-offset-cycle",
        str(offset_cycle),
        "--max-roots-per-episode",
        "1",
        "--records-per-shard",
        str(args.records_per_shard),
        "--task-suite-name",
        args.task_suite_name,
        "--seed",
        str(args.seed),
        "--teacher-samples",
        str(args.teacher_samples),
        "--action-cot-denoising-steps",
        str(args.action_cot_denoising_steps),
        "--source-iteration",
        str(args.source_iteration),
    ]


def _run_group(command: list[str], log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main(args: argparse.Namespace) -> None:
    if args.episode_start < 0 or args.episode_end < args.episode_start:
        raise ValueError("episode_start/end must define a non-empty non-negative inclusive range.")
    if args.task_start < 0 or args.max_tasks <= 0:
        raise ValueError("task_start must be non-negative and max_tasks must be positive.")
    if args.root_call_offset_cycle <= 0 or args.branch_repeats < 3:
        raise ValueError("root_call_offset_cycle must be positive and branch_repeats must be at least three.")
    if not _server_ready(args.host, args.port):
        raise ConnectionError(f"Policy server is not reachable at {args.host}:{args.port}.")
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    collector_script = (
        pathlib.Path(args.collector_script).resolve()
        if args.collector_script is not None
        else pathlib.Path(__file__).with_name("collect_execution_horizon_counterfactuals.py")
    )
    if not collector_script.exists():
        raise FileNotFoundError(collector_script)
    episodes = list(range(args.episode_start, args.episode_end + 1))
    expected_episodes = set(episodes)
    task_end = args.task_start + args.max_tasks
    data_dirs: list[str] = []
    total_roots = 0
    started = time.monotonic()

    for task_id in range(args.task_start, task_end):
        _write_json(
            status_path,
            {
                "status": "collecting_main",
                "task_id": task_id,
                "task_start": args.task_start,
                "task_end": task_end,
                "episode_start": args.episode_start,
                "episode_end": args.episode_end,
                "completed_roots": total_roots,
            },
        )
        main_dir = output_dir / f"task{task_id:02d}_main" / "data"
        if main_dir.exists():
            if not _complete_output(main_dir, task_id, expected_episodes):
                raise FileExistsError(f"Refusing to overwrite incomplete main fresh-root output: {main_dir}")
        else:
            command = _collector_command(
                args=args,
                collector_script=collector_script,
                task_id=task_id,
                episode_ids=episodes,
                output_dir=main_dir,
                offset_cycle=args.root_call_offset_cycle,
            )
            _run_group(command, main_dir.parent / "collector_stdout.log")
            if not _complete_output(main_dir, task_id, expected_episodes):
                raise RuntimeError(f"Main fresh-root collector did not finish cleanly: {main_dir}")
        observed_main = _episode_roots(main_dir, task_id)
        missing = sorted(expected_episodes.difference(observed_main))
        task_dirs = [main_dir]
        if missing:
            _write_json(
                status_path,
                {
                    "status": "supplementing_missing",
                    "task_id": task_id,
                    "missing_episode_ids": missing,
                    "completed_roots": total_roots + len(observed_main),
                },
            )
            supplement_dir = output_dir / f"task{task_id:02d}_supplement" / "data"
            if supplement_dir.exists():
                if not _complete_output(supplement_dir, task_id, set(missing)):
                    raise FileExistsError(f"Refusing to overwrite incomplete supplement: {supplement_dir}")
            else:
                command = _collector_command(
                    args=args,
                    collector_script=collector_script,
                    task_id=task_id,
                    episode_ids=missing,
                    output_dir=supplement_dir,
                    offset_cycle=1,
                )
                _run_group(command, supplement_dir.parent / "collector_stdout.log")
                if not _complete_output(supplement_dir, task_id, set(missing)):
                    raise RuntimeError(f"Fresh-root supplement did not finish cleanly: {supplement_dir}")
            observed_supplement = _episode_roots(supplement_dir, task_id)
            if set(observed_supplement) != set(missing):
                raise ValueError(
                    f"Supplement did not produce exactly the missing task{task_id} episodes: "
                    f"expected={missing}, observed={sorted(observed_supplement)}"
                )
            task_dirs.append(supplement_dir)
        combined_episodes: set[int] = set()
        for data_dir in task_dirs:
            roots = _episode_roots(data_dir, task_id)
            overlap = combined_episodes.intersection(roots)
            if overlap:
                raise ValueError(f"Main and supplement overlap for task{task_id}: {sorted(overlap)}")
            combined_episodes.update(roots)
            data_dirs.append(str(data_dir))
        if combined_episodes != expected_episodes:
            raise ValueError(
                f"Fresh-root task{task_id} remains incomplete: "
                f"missing={sorted(expected_episodes - combined_episodes)}"
            )
        total_roots += len(combined_episodes)

    expected_total = args.max_tasks * len(episodes)
    if total_roots != expected_total:
        raise ValueError(f"Expected {expected_total} fresh roots, found {total_roots}.")
    summary = {
        "status": "complete",
        "semantics": "Independent episode groups; one root per task and episode after safe early-offset supplementation.",
        "task_start": args.task_start,
        "max_tasks": args.max_tasks,
        "episode_start": args.episode_start,
        "episode_end": args.episode_end,
        "root_call_offset_cycle": args.root_call_offset_cycle,
        "branch_repeats": args.branch_repeats,
        "num_roots": total_roots,
        "data_dirs": data_dirs,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(status_path, {"status": "complete", "num_roots": total_roots})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
