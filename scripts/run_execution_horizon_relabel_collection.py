"""Replay a selected root manifest with denser paired continuation seeds."""

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
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--collector-script", default=None)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--seed", type=int, default=7)
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


def _observed_roots(data_dir: pathlib.Path) -> set[tuple[int, int, int, int]]:
    roots: set[tuple[int, int, int, int]] = set()
    for shard in horizon_dataset.discover_shards((data_dir,)):
        with h5py.File(shard, "r") as handle:
            for row in range(int(handle["task_id"].shape[0])):
                key = tuple(
                    int(handle[name][row]) for name in ("task_id", "episode_id", "decision_step", "root_seed")
                )
                if key in roots:
                    raise ValueError(f"Duplicate exact root in relabel output: {key}.")
                roots.add(key)
    return roots


def _complete_group(
    data_dir: pathlib.Path,
    expected: set[tuple[int, int, int, int]],
    *,
    target_trials: int,
) -> bool:
    summary_path = data_dir / "summary.json"
    manifest_path = data_dir / "manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if summary.get("status") != "complete" or int(summary.get("num_records", -1)) != len(expected):
        return False
    if int(manifest["shape"]["max_trials"]) != target_trials:
        raise ValueError(f"Existing relabel output has the wrong max_trials: {data_dir}")
    observed = _observed_roots(data_dir)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(f"Relabel root mismatch in {data_dir}: missing={missing[:3]}, extra={extra[:3]}.")
    return True


def _groups(records: list[dict[str, Any]]) -> list[tuple[int, int, list[dict[str, Any]]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    seen: set[tuple[int, int, int, int]] = set()
    for record in records:
        key = (
            int(record["task_id"]),
            int(record["episode_id"]),
            int(record["decision_step"]),
            int(record["root_seed"]),
        )
        if key in seen:
            raise ValueError(f"Selection manifest contains duplicate exact root {key}.")
        seen.add(key)
        group = (int(record["task_id"]), int(record["root_call_offset_cycle"]))
        grouped.setdefault(group, []).append(record)
    return [
        (task, cycle, sorted(rows, key=lambda row: int(row["episode_id"])))
        for (task, cycle), rows in sorted(grouped.items())
    ]


def main(args: argparse.Namespace) -> None:
    selection_path = pathlib.Path(args.selection_manifest).resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "complete":
        raise ValueError("Selection manifest is not complete.")
    records = list(selection.get("records", []))
    if not records or len(records) != int(selection["num_selected_roots"]):
        raise ValueError("Selection manifest record count is inconsistent.")
    candidates = [int(value) for value in selection["candidate_horizons"]]
    reference_horizon = int(selection["reference_horizon"])
    target_trials = int(selection["target_trials"])
    if any(int(record["target_trials"]) != target_trials for record in records):
        raise ValueError("Selected roots do not share one target_trials value.")
    if target_trials < 3:
        raise ValueError("Dense relabel collection requires at least three paired trials.")
    if not _server_ready(args.host, args.port):
        raise ConnectionError(f"Policy server is not reachable at {args.host}:{args.port}.")
    if args.records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive.")

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
    groups = _groups(records)
    completed_roots = 0
    started = time.monotonic()
    for group_index, (task_id, offset_cycle, group_records) in enumerate(groups):
        expected = {
            (
                int(record["task_id"]),
                int(record["episode_id"]),
                int(record["decision_step"]),
                int(record["root_seed"]),
            )
            for record in group_records
        }
        group_dir = output_dir / f"task{task_id:02d}_cycle{offset_cycle}" / "data"
        if group_dir.exists():
            if _complete_group(group_dir, expected, target_trials=target_trials):
                completed_roots += len(expected)
                continue
            raise FileExistsError(f"Refusing to overwrite incomplete relabel group: {group_dir}")
        group_dir.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            status_path,
            {
                "status": "collecting",
                "task_id": task_id,
                "root_call_offset_cycle": offset_cycle,
                "group_index": group_index,
                "num_groups": len(groups),
                "completed_roots": completed_roots,
                "total_roots": len(records),
            },
        )
        command = [
            sys.executable,
            str(collector_script),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--output-dir",
            str(group_dir),
            "--task-start",
            str(task_id),
            "--max-tasks",
            "1",
            "--episode-ids",
            *[str(record["episode_id"]) for record in group_records],
            "--candidate-horizons",
            *[str(value) for value in candidates],
            "--reference-horizon",
            str(reference_horizon),
            "--model-action-horizon",
            "25",
            "--model-coarse-horizon",
            "15",
            "--continuation-policy",
            "fixed_h",
            "--fixed-continuation-horizon",
            "5",
            "--branch-repeats",
            str(target_trials),
            "--repeat-branch-horizons",
            *[str(value) for value in candidates],
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
        log_path = group_dir.parent / "collector_stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not _complete_group(group_dir, expected, target_trials=target_trials):
            raise RuntimeError(f"Collector exited without a complete relabel group: {group_dir}")
        completed_roots += len(expected)

    summary = {
        "status": "complete",
        "selection_manifest": str(selection_path),
        "num_groups": len(groups),
        "num_roots": len(records),
        "target_trials": target_trials,
        "candidate_horizons": candidates,
        "reference_horizon": reference_horizon,
        "elapsed_seconds": time.monotonic() - started,
        "group_data_dirs": [
            str(output_dir / f"task{task_id:02d}_cycle{offset_cycle}" / "data")
            for task_id, offset_cycle, _ in groups
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(status_path, {"status": "complete", "completed_roots": len(records), "total_roots": len(records)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
