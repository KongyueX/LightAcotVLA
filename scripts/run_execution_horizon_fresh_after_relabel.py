"""Wait for dense relabeling, then collect isolated fresh development and final roots."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-exit-file", required=True)
    parser.add_argument("--relabel-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--expected-relabel-roots", type=int, default=137)
    parser.add_argument("--expected-relabel-trials", type=int, default=10)
    parser.add_argument("--fresh-runner", default=None)
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_phase(
    *,
    runner: pathlib.Path,
    output_dir: pathlib.Path,
    episode_start: int,
    episode_end: int,
    host: str,
    port: int,
) -> None:
    command = [
        sys.executable,
        str(runner),
        "--output-dir",
        str(output_dir),
        "--episode-start",
        str(episode_start),
        "--episode-end",
        str(episode_end),
        "--host",
        host,
        "--port",
        str(port),
        "--task-suite-name",
        "libero_10",
        "--task-start",
        "0",
        "--max-tasks",
        "10",
        "--root-call-offset-cycle",
        "20",
        "--seed",
        "7",
        "--branch-repeats",
        "3",
        "--teacher-samples",
        "20",
        "--action-cot-denoising-steps",
        "10",
        "--prefix-token-count",
        "1024",
        "--source-iteration",
        "1",
        "--records-per-shard",
        "10",
    ]
    log_path = output_dir.with_name(output_dir.name + "_controller.log")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    relabel_exit = pathlib.Path(args.relabel_exit_file).resolve()
    relabel_summary = pathlib.Path(args.relabel_summary).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    runner = (
        pathlib.Path(args.fresh_runner).resolve()
        if args.fresh_runner is not None
        else pathlib.Path(__file__).with_name("run_execution_horizon_fresh_root_collection.py")
    )
    if not runner.exists():
        raise FileNotFoundError(runner)
    _write_json(status_path, {"status": "waiting_for_relabel", "relabel_exit_file": str(relabel_exit)})
    while not relabel_exit.exists():
        time.sleep(args.poll_seconds)
    exit_code = int(relabel_exit.read_text().strip())
    if exit_code != 0:
        raise RuntimeError(f"Dense relabel collection exited with code {exit_code}.")
    if not relabel_summary.exists():
        raise FileNotFoundError(f"Dense relabel exit was zero but summary is absent: {relabel_summary}")
    summary = json.loads(relabel_summary.read_text())
    if (
        summary.get("status") != "complete"
        or int(summary.get("num_roots", -1)) != args.expected_relabel_roots
        or int(summary.get("target_trials", -1)) != args.expected_relabel_trials
    ):
        raise ValueError(f"Dense relabel summary failed the completion contract: {summary}")

    development_dir = output_dir / "fresh_development_e50_89_r3"
    _write_json(status_path, {"status": "collecting_fresh_development", "output_dir": str(development_dir)})
    _run_phase(
        runner=runner,
        output_dir=development_dir,
        episode_start=50,
        episode_end=89,
        host=args.host,
        port=args.port,
    )
    development_summary = json.loads((development_dir / "summary.json").read_text())
    if development_summary.get("status") != "complete" or int(development_summary.get("num_roots", -1)) != 400:
        raise ValueError(f"Fresh development completion contract failed: {development_summary}")

    final_dir = output_dir / "fresh_final_e90_99_r3"
    _write_json(status_path, {"status": "collecting_fresh_final", "output_dir": str(final_dir)})
    _run_phase(
        runner=runner,
        output_dir=final_dir,
        episode_start=90,
        episode_end=99,
        host=args.host,
        port=args.port,
    )
    final_summary = json.loads((final_dir / "summary.json").read_text())
    if final_summary.get("status") != "complete" or int(final_summary.get("num_roots", -1)) != 100:
        raise ValueError(f"Fresh final completion contract failed: {final_summary}")

    result = {
        "status": "complete",
        "relabel_summary": str(relabel_summary),
        "fresh_development_summary": str(development_dir / "summary.json"),
        "fresh_final_summary": str(final_dir / "summary.json"),
        "num_relabel_roots": args.expected_relabel_roots,
        "num_fresh_development_roots": 400,
        "num_fresh_final_roots": 100,
        "semantics": "Fresh final roots are isolated and must not enter training, checkpoint selection, or calibration.",
    }
    _write_json(output_dir / "summary.json", result)
    _write_json(status_path, {"status": "complete"})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
