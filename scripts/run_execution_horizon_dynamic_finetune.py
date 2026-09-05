#!/usr/bin/env python3
"""Fine-tune the ordered horizon selector on completed dynamic-rollout data."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

import run_execution_horizon_post_breadth_ordered_warmstart as post_breadth

EXPECTED_SPLITS = {
    "train_group_ids": 100, "early_stop_group_ids": 30, "calibration_group_ids": 30, "dev_audit_group_ids": 20
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--collection-summary", "--resume-predictor-dir", "--output-dir", "--policy-server-tmux"):
        parser.add_argument(option, required=True)
    parser.add_argument("--python")
    parser.add_argument("--ordered-listwise-elapsed-mode", choices=("root_minmax", "paired_noise"), default="root_minmax")
    parser.add_argument("--ordered-listwise-elapsed-floor-seconds", type=float, default=1.0)
    return parser


def _validate_elapsed_target(mode: str, floor_seconds: float) -> None:
    if mode not in ("root_minmax", "paired_noise"):
        raise ValueError("ordered_listwise_elapsed_mode must be root_minmax or paired_noise.")
    if not math.isfinite(floor_seconds) or floor_seconds <= 0:
        raise ValueError("ordered_listwise_elapsed_floor_seconds must be finite and positive.")


def _summary_path(parent: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def _status(output_dir: pathlib.Path, status: str, **details: object) -> None:
    post_breadth._write_json(output_dir / "runner_status.json", {"status": status, **details})  # noqa: SLF001


def load_collection(summary_path: pathlib.Path) -> tuple[dict[str, object], pathlib.Path, list[pathlib.Path]]:
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or int(summary.get("num_roots", -1)) != 180:
        raise ValueError("Dynamic collection must be complete with exactly 180 roots.")
    manifest_path = _summary_path(summary_path.parent, str(summary.get("split_manifest", "")))
    manifest = json.loads(manifest_path.read_text())
    split_groups: list[int] = []
    for field, expected_count in EXPECTED_SPLITS.items():
        values = [int(value) for value in manifest.get(field, ())]
        if len(values) != expected_count or len(set(values)) != expected_count:
            raise ValueError(f"Dynamic split {field} must contain {expected_count} unique groups.")
        split_groups.extend(values)
    if len(set(split_groups)) != 180:
        raise ValueError("Dynamic four-way split groups must be mutually disjoint.")
    data_dirs = [_summary_path(summary_path.parent, str(value)) for value in summary.get("data_dirs", ())]
    if not data_dirs or any(not path.is_dir() for path in data_dirs):
        raise FileNotFoundError("A dynamic collection data directory is missing.")
    return summary, manifest_path, list(dict.fromkeys(data_dirs))


def validate_resume_predictor(predictor_dir: pathlib.Path) -> pathlib.Path:
    params = post_breadth.validate_resume_predictor(predictor_dir)
    config = json.loads((predictor_dir.resolve() / "predictor_config.json").read_text())
    if not config.get("ordered_continuation_head"):
        raise ValueError("Resume predictor must contain the ordered continuation warm start.")
    return params


def build_train_command(
    *,
    python: pathlib.Path,
    train_script: pathlib.Path,
    data_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    resume_params: pathlib.Path,
    split_manifest: pathlib.Path,
    ordered_listwise_elapsed_mode: str = "root_minmax",
    ordered_listwise_elapsed_floor_seconds: float = 1.0,
) -> list[str]:
    _validate_elapsed_target(ordered_listwise_elapsed_mode, ordered_listwise_elapsed_floor_seconds)
    command = post_breadth.build_train_command(
        python=python,
        train_script=train_script,
        datasets=data_dirs,
        output_dir=output_dir,
        resume_params=resume_params,
        split_manifest=split_manifest,
    )
    for option, value in {
        "--train-steps": "5000",
        "--batch-size": "128",
        "--learning-rate": "1e-4",
        "--checkpoint-interval": "0",
        "--early-stopping-patience-logs": "8",
    }.items():
        command[command.index(option) + 1] = value
    if ordered_listwise_elapsed_mode != "root_minmax" or ordered_listwise_elapsed_floor_seconds != 1.0:
        command.extend((
            "--ordered-listwise-elapsed-mode", ordered_listwise_elapsed_mode,
            "--ordered-listwise-elapsed-floor-seconds", str(ordered_listwise_elapsed_floor_seconds),
        ))
    return command


def _tmux_process_tree(name: str) -> set[int]:
    if not name or "\n" in name:
        raise ValueError("policy-server tmux name must be a non-empty single line.")
    result = subprocess.run(
        ["tmux", "list-panes", "-t", f"={name}", "-F", "#{pane_pid}"], check=False, capture_output=True, text=True
    )
    if result.returncode:
        return set()
    process_ids = {int(value) for value in result.stdout.split()}
    frontier = list(process_ids)
    while frontier:
        pid = frontier.pop()
        children_path = pathlib.Path(f"/proc/{pid}/task/{pid}/children")
        descendants = {int(value) for value in children_path.read_text().split()} if children_path.is_file() else set()
        descendants.difference_update(process_ids)
        process_ids.update(descendants)
        frontier.extend(descendants)
    return process_ids


def stop_policy_server(name: str, timeout_seconds: float = 60.0) -> bool:
    process_ids = _tmux_process_tree(name)
    if not process_ids:
        return False
    subprocess.run(["tmux", "kill-session", "-t", f"={name}"], check=True)
    deadline = time.monotonic() + timeout_seconds
    while process_ids:
        process_ids = {pid for pid in process_ids if pathlib.Path(f"/proc/{pid}").exists()}
        if not process_ids:
            return True
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Policy-server child processes did not exit: {sorted(process_ids)}")
        time.sleep(0.25)
    return True


def main(args: argparse.Namespace) -> None:
    _validate_elapsed_target(args.ordered_listwise_elapsed_mode, args.ordered_listwise_elapsed_floor_seconds)
    collection_path = pathlib.Path(args.collection_summary).resolve()
    collection, split_manifest, data_dirs = load_collection(collection_path)
    resume_dir = pathlib.Path(args.resume_predictor_dir).resolve()
    resume_params = validate_resume_predictor(resume_dir)
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dynamic fine-tune output: {output_dir}")
    python = pathlib.Path(args.python).absolute() if args.python else pathlib.Path(sys.executable).absolute()
    train_script = pathlib.Path(__file__).resolve().with_name("train_execution_horizon_predictor.py")
    if not python.is_file() or not train_script.is_file():
        raise FileNotFoundError("Python executable or predictor training script is missing.")

    command = build_train_command(
        python=python,
        train_script=train_script,
        data_dirs=data_dirs,
        output_dir=output_dir,
        resume_params=resume_params,
        split_manifest=split_manifest,
        ordered_listwise_elapsed_mode=args.ordered_listwise_elapsed_mode,
        ordered_listwise_elapsed_floor_seconds=args.ordered_listwise_elapsed_floor_seconds,
    )
    elapsed_target = {
        "ordered_listwise_elapsed_mode": args.ordered_listwise_elapsed_mode,
        "ordered_listwise_elapsed_floor_seconds": args.ordered_listwise_elapsed_floor_seconds,
    }
    output_dir.mkdir(parents=True)
    post_breadth._write_json(output_dir / "run_config.json", {  # noqa: SLF001
        "collection_summary": str(collection_path), "resume_predictor_dir": str(resume_dir),
        "split_manifest": str(split_manifest), "data_dirs": [str(path) for path in data_dirs],
        "policy_server_tmux": args.policy_server_tmux, "command": command, **elapsed_target,
    })
    _status(output_dir, "stopping_policy_server")
    server_stopped = stop_policy_server(args.policy_server_tmux)
    _status(output_dir, "training", command=command)
    with (output_dir / "train.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    (output_dir / "train.exit").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        _status(output_dir, "failed", returncode=completed.returncode)
        raise subprocess.CalledProcessError(completed.returncode, command)

    training_summary = json.loads((output_dir / "summary.json").read_text())
    config = training_summary.get("predictor_config", {})
    if (
        training_summary.get("status") != "complete"
        or training_summary.get("resume", {}).get("mode") != "strict"
        or training_summary.get("best_validation_objective_name") != "ordered_listwise_nll"
        or [pathlib.Path(value).resolve() for value in training_summary.get("dataset_inputs", ())] != data_dirs
        or not config.get("paired_distribution_heads")
        or not config.get("ordered_continuation_head")
        or training_summary.get("ordered_listwise_elapsed_mode", "root_minmax") != args.ordered_listwise_elapsed_mode
        or training_summary.get("ordered_listwise_elapsed_floor_seconds", 1.0) != args.ordered_listwise_elapsed_floor_seconds
    ):
        _status(output_dir, "failed", reason="training_summary_contract")
        raise ValueError("Dynamic fine-tune training summary does not match the requested contract.")
    finetune_summary = {
        "status": "complete",
        "purpose": "dynamic_on_policy_ordered_finetune",
        "collection_summary": str(collection_path),
        "num_roots": int(collection["num_roots"]),
        "data_dirs": [str(path) for path in data_dirs],
        "split_manifest": str(split_manifest),
        "resume_predictor_dir": str(resume_dir),
        "predictor_dir": str(output_dir),
        "predictor_params": training_summary["predictor_params"],
        "training_summary": str(output_dir / "summary.json"),
        "policy_server_stopped": server_stopped,
        "calibration_or_gate_run": False,
        **elapsed_target,
    }
    post_breadth._write_json(output_dir / "finetune_summary.json", finetune_summary)  # noqa: SLF001
    _status(output_dir, "complete", summary=str(output_dir / "finetune_summary.json"))


if __name__ == "__main__":
    main(build_parser().parse_args())
