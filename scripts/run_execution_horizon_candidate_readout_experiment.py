#!/usr/bin/env python3
"""Compare candidate-aware readout with the completed Round-4 ordered baseline."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys

import run_execution_horizon_dynamic_finetune as dynamic
import run_execution_horizon_post_breadth_ordered_warmstart as post_breadth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "--collection-summary",
        "--resume-predictor-dir",
        "--baseline-predictor-dir",
        "--output-dir",
        "--policy-server-tmux",
    ):
        parser.add_argument(option, required=True)
    parser.add_argument("--temporal-layers", type=int, choices=(2, 4), default=2)
    parser.add_argument("--python")
    return parser


def build_train_command(
    *,
    python: pathlib.Path,
    train_script: pathlib.Path,
    data_dirs: list[pathlib.Path],
    output_dir: pathlib.Path,
    resume_params: pathlib.Path,
    split_manifest: pathlib.Path,
    temporal_layers: int = 2,
) -> list[str]:
    command = dynamic.build_train_command(
        python=python,
        train_script=train_script,
        data_dirs=data_dirs,
        output_dir=output_dir,
        resume_params=resume_params,
        split_manifest=split_manifest,
    )
    command[command.index("--temporal-layers") + 1] = str(temporal_layers)
    command.extend(("--ordered-readout", "candidate", "--resume-candidate-readout"))
    return command


def _read_validation_objective(
    predictor_dir: pathlib.Path, *, data_dirs: list[pathlib.Path], split_manifest: pathlib.Path
) -> tuple[dict[str, object], float]:
    summary = json.loads((predictor_dir / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("best_validation_objective_name") != "ordered_listwise_nll":
        raise ValueError(f"Predictor must have completed ordered-listwise training: {predictor_dir}")
    if [pathlib.Path(value).resolve() for value in summary.get("dataset_inputs", ())] != data_dirs:
        raise ValueError("Candidate and baseline must use the same dynamic collection data.")
    if pathlib.Path(summary.get("input_split_manifest") or "").resolve() != split_manifest:
        raise ValueError("Candidate and baseline must use the same early-stop split manifest.")
    objective = float(summary["best_validation_objective"])
    if not math.isfinite(objective):
        raise ValueError("The best early-stop ordered-listwise NLL must be finite.")
    return summary, objective


def _run(args: argparse.Namespace, output_dir: pathlib.Path) -> None:
    collection_path = pathlib.Path(args.collection_summary).resolve()
    collection, split_manifest, data_dirs = dynamic.load_collection(collection_path)
    resume_dir = pathlib.Path(args.resume_predictor_dir).resolve()
    resume_params = dynamic.validate_resume_predictor(resume_dir)
    baseline_dir = pathlib.Path(args.baseline_predictor_dir).resolve()
    baseline_summary, baseline_objective = _read_validation_objective(
        baseline_dir, data_dirs=data_dirs, split_manifest=split_manifest
    )
    baseline_config = baseline_summary.get("predictor_config", {})
    if baseline_config.get("ordered_readout", "global") != "global" or baseline_config.get("temporal_layers") != 2:
        raise ValueError("The reference must be the completed two-layer mean-readout Round-4 baseline.")
    python = pathlib.Path(args.python).absolute() if args.python else pathlib.Path(sys.executable).absolute()
    train_script = pathlib.Path(__file__).resolve().with_name("train_execution_horizon_predictor.py")
    if not python.is_file() or not train_script.is_file():
        raise FileNotFoundError("Python executable or predictor training script is missing.")
    predictor_dir = output_dir / "predictor"
    command = build_train_command(
        python=python,
        train_script=train_script,
        data_dirs=data_dirs,
        output_dir=predictor_dir,
        resume_params=resume_params,
        split_manifest=split_manifest,
        temporal_layers=args.temporal_layers,
    )
    run_config = {
        "purpose": "candidate_readout_dynamic_comparison",
        "collection_summary": str(collection_path),
        "num_roots": int(collection["num_roots"]),
        "data_dirs": [str(path) for path in data_dirs],
        "split_manifest": str(split_manifest),
        "resume_predictor_dir": str(resume_dir),
        "baseline_predictor_dir": str(baseline_dir),
        "predictor_dir": str(predictor_dir),
        "ordered_readout": "candidate",
        "temporal_layers": args.temporal_layers,
        "selection_split": "early_stop",
        "selection_objective": "ordered_listwise_nll",
        "train_command": command,
    }
    post_breadth._write_json(output_dir / "run_config.json", run_config)  # noqa: SLF001
    dynamic._status(output_dir, "stopping_policy_server")  # noqa: SLF001
    server_stopped = dynamic.stop_policy_server(args.policy_server_tmux)
    dynamic._status(output_dir, "training", command=command)  # noqa: SLF001
    with (output_dir / "train.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    (output_dir / "train.exit").write_text(f"{completed.returncode}\n")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)

    training_summary, candidate_objective = _read_validation_objective(
        predictor_dir, data_dirs=data_dirs, split_manifest=split_manifest
    )
    config = training_summary.get("predictor_config", {})
    if config.get("ordered_readout") != "candidate" or config.get("temporal_layers") != args.temporal_layers:
        raise ValueError("Completed predictor does not use the requested candidate-readout architecture.")
    summary = {
        "status": "complete",
        **run_config,
        "training_summary": str(predictor_dir / "summary.json"),
        "predictor_params": training_summary["predictor_params"],
        "baseline_training_summary": str(baseline_dir / "summary.json"),
        "baseline_ordered_listwise_nll": baseline_objective,
        "candidate_ordered_listwise_nll": candidate_objective,
        "ordered_listwise_nll_delta": candidate_objective - baseline_objective,
        "candidate_improved": candidate_objective < baseline_objective,
        "closedloop_pilot_run": False,
        "policy_server_stopped": server_stopped,
        "calibration_or_dev_used_for_selection": False,
    }
    post_breadth._write_json(output_dir / "summary.json", summary)  # noqa: SLF001
    dynamic._status(  # noqa: SLF001
        output_dir,
        "complete",
        summary=str(output_dir / "summary.json"),
        candidate_improved=summary["candidate_improved"],
    )


def main(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate-readout output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        _run(args, output_dir)
    except Exception as error:
        dynamic._status(output_dir, "failed", error=str(error))  # noqa: SLF001
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
