#!/usr/bin/env python3
"""Train an ordered execution-horizon warm start after breadth-first collection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import fcntl
import json
import pathlib
import subprocess
import sys
from typing import Any


CANDIDATE_HORIZONS = (5, 10, 15, 20, 25)
TRAINING_SEED = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breadth-dir", required=True)
    parser.add_argument("--base-dataset", action="append", required=True)
    parser.add_argument("--resume-predictor-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-server-tmux", required=True)
    return parser


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_breadth_snapshot(breadth_dir: pathlib.Path) -> tuple[dict[str, Any], pathlib.Path, list[pathlib.Path]]:
    breadth_dir = breadth_dir.resolve()
    summary_path = breadth_dir / "summary.json"
    manifest_path = breadth_dir / "split_manifest.json"
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if summary.get("status") != "complete" or summary.get("purpose") != "breadth_first_warm_start":
        raise ValueError("Breadth-first summary is not complete.")
    if pathlib.Path(summary.get("split_manifest", "")).resolve() != manifest_path:
        raise ValueError("Breadth-first summary points to a different split manifest.")
    if not summary.get("resumable_to_full_round3") or summary.get("full_round3_outputs_written"):
        raise ValueError("Breadth-first summary does not preserve full Round-3 resumability.")
    for task in range(10):
        roles = summary.get("realized_by_task", {}).get(str(task), {})
        if any(int(roles.get(role, {}).get("roots", 0)) < 10 for role in ("train", "calibration", "dev_audit")):
            raise ValueError(f"Task {task} lacks breadth-first train/calibration/dev_audit coverage.")
    required_manifest_fields = (
        "train_group_ids",
        "early_stop_group_ids",
        "calibration_group_ids",
        "dev_audit_group_ids",
    )
    if any(not manifest.get(field) for field in required_manifest_fields):
        raise ValueError("Breadth-first split manifest is missing a non-empty four-way split.")
    data_dirs = [pathlib.Path(value).resolve() for value in summary.get("data_dirs", ())]
    if not data_dirs or any(not path.is_dir() for path in data_dirs):
        raise FileNotFoundError("A breadth-first data directory is missing.")
    return summary, manifest_path, data_dirs


def validate_resume_predictor(predictor_dir: pathlib.Path) -> pathlib.Path:
    predictor_dir = predictor_dir.resolve()
    config = json.loads((predictor_dir / "predictor_config.json").read_text())
    summary = json.loads((predictor_dir / "summary.json").read_text())
    expected = {
        "temporal_backbone": "transformer",
        "temporal_layers": 2,
        "hidden_dim": 256,
        "num_heads": 4,
        "feed_forward_multiplier": 4,
        "reference_horizon": 10,
        "visual_num_queries": 4,
        "paired_distribution_heads": True,
        "paired_advantage_heads": False,
    }
    mismatches = {name: (config.get(name), value) for name, value in expected.items() if config.get(name) != value}
    if tuple(config.get("candidate_horizons", ())) != CANDIDATE_HORIZONS:
        mismatches["candidate_horizons"] = (config.get("candidate_horizons"), CANDIDATE_HORIZONS)
    if mismatches:
        raise ValueError(f"Resume predictor is not the paired 2-layer H25 Transformer: {mismatches}")
    if summary.get("status") != "complete" or int(summary.get("training_seed", -1)) != TRAINING_SEED:
        raise ValueError("Resume predictor must be the completed seed7 predictor.")
    params = predictor_dir / "params"
    if not params.is_dir():
        raise FileNotFoundError(params)
    return params


def build_train_command(
    *,
    python: pathlib.Path,
    train_script: pathlib.Path,
    datasets: Sequence[pathlib.Path],
    output_dir: pathlib.Path,
    resume_params: pathlib.Path,
    split_manifest: pathlib.Path,
) -> list[str]:
    return [
        str(python),
        str(train_script),
        "--dataset",
        *[str(path) for path in datasets],
        "--output-dir",
        str(output_dir),
        "--resume-params",
        str(resume_params),
        "--seed",
        str(TRAINING_SEED),
        "--input-split-manifest",
        str(split_manifest),
        "--train-steps",
        "20000",
        "--batch-size",
        "256",
        "--learning-rate",
        "3e-4",
        "--weight-decay",
        "1e-4",
        "--gradient-clip-norm",
        "1.0",
        "--log-interval",
        "25",
        "--checkpoint-interval",
        "5000",
        "--early-stopping-patience-logs",
        "10",
        "--early-stopping-min-delta",
        "1e-4",
        "--temporal-backbone",
        "transformer",
        "--temporal-layers",
        "2",
        "--hidden-dim",
        "256",
        "--num-heads",
        "4",
        "--feed-forward-multiplier",
        "4",
        "--reference-horizon",
        "10",
        "--coarse-stride",
        "2",
        "--final-stride",
        "1",
        "--physical-action-dim",
        "7",
        "--visual-num-queries",
        "4",
        "--paired-distribution-heads",
        "--ordered-continuation-head",
        "--minimum-trials-per-candidate",
        "3",
        "--focus-task-multiplier",
        "1",
        "--high-risk-multiplier",
        "1",
        "--gripper-multiplier",
        "1",
        "--failure-multiplier",
        "1",
        "--loss-success",
        "0.5",
        "--loss-timeout",
        "0",
        "--loss-remaining-calls",
        "0",
        "--loss-remaining-steps",
        "0",
        "--loss-final-risk",
        "0",
        "--loss-action-cot-risk",
        "0",
        "--loss-fused-risk",
        "0",
        "--loss-event",
        "0",
        "--loss-raw-h-classification",
        "0",
        "--loss-raw-h-ordinal",
        "0",
        "--loss-survival",
        "0",
        "--loss-success-advantage",
        "0",
        "--loss-elapsed-advantage",
        "0",
        "--loss-calls-advantage",
        "0",
        "--loss-false-long",
        "0",
        "--loss-danger-rescue",
        "1",
        "--loss-paired-elapsed",
        "0.5",
        "--loss-faster-long",
        "0",
        "--loss-ordered-listwise",
        "2",
        "--ordered-listwise-elapsed-temperature",
        "0.25",
    ]


def _stop_exact_tmux_session(name: str) -> None:
    if not name or "\n" in name:
        raise ValueError("policy-server tmux name must be a non-empty single line.")
    sessions = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if name not in sessions:
        raise RuntimeError(f"Policy-server tmux session is absent: {name}")
    subprocess.run(["tmux", "kill-session", "-t", f"={name}"], check=True)


def main(args: argparse.Namespace) -> None:
    breadth_dir = pathlib.Path(args.breadth_dir).resolve()
    breadth_summary, split_manifest, round3_data = load_breadth_snapshot(breadth_dir)
    base_data = [pathlib.Path(value).resolve() for value in args.base_dataset]
    if any(not path.exists() for path in base_data):
        raise FileNotFoundError("A base dataset path is missing.")
    datasets = list(dict.fromkeys((*base_data, *round3_data)))
    resume_params = validate_resume_predictor(pathlib.Path(args.resume_predictor_dir))

    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite ordered warm-start output: {output_dir}")
    train_script = pathlib.Path(__file__).resolve().with_name("train_execution_horizon_predictor.py")
    # Preserve a virtualenv executable symlink so Python keeps the environment's
    # site-packages instead of falling through to the underlying interpreter.
    python = pathlib.Path(sys.executable).absolute()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)

    collection_dir = breadth_dir.parent
    with (collection_dir / "controller.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Round-3 collection is still active; post-breadth training cannot start.") from error

        output_dir.mkdir(parents=True)
        _write_json(
            output_dir / "runner_status.json",
            {
                "status": "stopping_policy_server",
                "breadth_summary": str(breadth_dir / "summary.json"),
                "policy_server_tmux": args.policy_server_tmux,
            },
        )
        _stop_exact_tmux_session(args.policy_server_tmux)
        command = build_train_command(
            python=python,
            train_script=train_script,
            datasets=datasets,
            output_dir=output_dir,
            resume_params=resume_params,
            split_manifest=split_manifest,
        )
        _write_json(
            output_dir / "runner_status.json",
            {
                "status": "training",
                "breadth_summary": str(breadth_dir / "summary.json"),
                "policy_server_tmux": args.policy_server_tmux,
                "dataset_inputs": [str(path) for path in datasets],
            },
        )
        with (output_dir / "train_stdout.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        (output_dir / "train.exit").write_text(f"{completed.returncode}\n")
        if completed.returncode:
            _write_json(output_dir / "runner_status.json", {"status": "failed", "returncode": completed.returncode})
            raise subprocess.CalledProcessError(completed.returncode, command)

        training_summary = json.loads((output_dir / "summary.json").read_text())
        config = training_summary.get("predictor_config", {})
        if (
            training_summary.get("status") != "complete"
            or training_summary.get("resume", {}).get("mode") != "strict"
            or training_summary.get("selected_checkpoint") != "best_validation"
            or training_summary.get("best_validation_objective_name") != "ordered_listwise_nll"
            or not config.get("paired_distribution_heads")
            or not config.get("ordered_continuation_head")
        ):
            raise ValueError("Ordered warm-start training summary does not match the requested contract.")
        post_summary = {
            "status": "complete",
            "purpose": "ordered_continuation_warm_start",
            "predictor_dir": str(output_dir),
            "predictor_params": training_summary["predictor_params"],
            "training_summary": str(output_dir / "summary.json"),
            "breadth_summary": str(breadth_dir / "summary.json"),
            "breadth_roots": int(breadth_summary["num_roots"]),
            "dataset_inputs": [str(path) for path in datasets],
            "policy_server_tmux_stopped": args.policy_server_tmux,
            "formal_gate_or_elapsed_audit_run": False,
        }
        _write_json(output_dir / "post_breadth_summary.json", post_summary)
        _write_json(output_dir / "runner_status.json", {"status": "complete", "predictor_dir": str(output_dir)})
        print(json.dumps(post_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
