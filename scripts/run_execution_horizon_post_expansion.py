"""Consolidate expanded labels and run the first paired-head train/calibrate/audit cycle."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

from openpi.execution_horizon import initial_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-exit-file", required=True)
    parser.add_argument("--fresh-summary", required=True)
    parser.add_argument("--relabel-summary", required=True)
    parser.add_argument("--base-dataset", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-server-tmux", default="h25_relabel_server")
    parser.add_argument("--expected-development-roots", type=int, default=900)
    parser.add_argument("--expected-final-roots", type=int, default=100)
    parser.add_argument("--expected-replacements", type=int, default=137)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-initial-state-isolation", action="store_true")
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run(command: list[str], log_path: pathlib.Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def _complete_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"Artifact is not complete: {path}")
    return payload


def _episode_groups(data_dirs: list[str]) -> set[int]:
    from openpi.execution_horizon import dataset as horizon_dataset

    arrays = horizon_dataset.load_counterfactual_arrays(tuple(data_dirs))
    task = arrays["task_id"].astype("uint64")
    episode = arrays["episode_id"].astype("uint64")
    return set((task * 1_000_000_000 + episode).tolist())


def _stop_policy_server(tmux_name: str) -> None:
    present = (
        subprocess.run(
            ["tmux", "has-session", "-t", tmux_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    if present:
        subprocess.run(["tmux", "kill-session", "-t", tmux_name], check=True)
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not pids:
            return
        time.sleep(5.0)
    raise RuntimeError("GPU compute processes remain after stopping the scoped H25 policy server.")


def _prepare_mean_only_calibration(source: pathlib.Path, target: pathlib.Path) -> None:
    calibration = json.loads(source.read_text())
    calibration["success_residual_quantiles"] = [0.0 for _ in calibration["success_residual_quantiles"]]
    calibration["elapsed_residual_quantiles"] = [0.0 for _ in calibration["elapsed_residual_quantiles"]]
    target.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")


def _audit_summary(path: pathlib.Path) -> dict[str, Any]:
    audit = _complete_json(path)
    return {
        "num_roots": int(audit["num_roots"]),
        "long_h_coverage": float(audit["long_h_coverage"]),
        "selected_h_distribution": audit["selected_h_distribution"],
        "success_advantage": audit["success_advantage_vs_h10_cluster_bootstrap"],
        "elapsed_advantage": audit["elapsed_advantage_vs_h10_cluster_bootstrap"],
        "calls_advantage": audit["calls_advantage_vs_h10_cluster_bootstrap"],
        "false_long_rate": float(audit["false_long_rate"]),
        "false_long_upper_95": float(audit["false_long_upper_95"]),
        "offline_engineering_gate": bool(audit["offline_engineering_gate"]),
    }


def main(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    fresh_exit = pathlib.Path(args.fresh_exit_file).resolve()
    fresh_summary_path = pathlib.Path(args.fresh_summary).resolve()
    relabel_summary_path = pathlib.Path(args.relabel_summary).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    _write_json(status_path, {"status": "waiting_for_fresh_collection", "fresh_exit_file": str(fresh_exit)})
    while not fresh_exit.exists():
        time.sleep(args.poll_seconds)
    fresh_exit_code = int(fresh_exit.read_text().strip())
    if fresh_exit_code != 0:
        raise RuntimeError(f"Fresh-root followup exited with code {fresh_exit_code}.")
    fresh_summary = _complete_json(fresh_summary_path)
    relabel_summary = _complete_json(relabel_summary_path)
    if int(fresh_summary["num_fresh_development_roots"]) != args.expected_development_roots - 500:
        raise ValueError("Fresh development root count does not match the preregistered expansion.")
    if int(fresh_summary["num_fresh_final_roots"]) != args.expected_final_roots:
        raise ValueError("Fresh final root count does not match the preregistered expansion.")
    if int(relabel_summary["num_roots"]) != args.expected_replacements:
        raise ValueError("Dense relabel root count does not match expected replacements.")

    fresh_development_summary = _complete_json(pathlib.Path(fresh_summary["fresh_development_summary"]))
    fresh_final_summary = _complete_json(pathlib.Path(fresh_summary["fresh_final_summary"]))
    development_dirs = [str(pathlib.Path(value).resolve()) for value in fresh_development_summary["data_dirs"]]
    final_dirs = [str(pathlib.Path(value).resolve()) for value in fresh_final_summary["data_dirs"]]
    overlay_dirs = [str(pathlib.Path(value).resolve()) for value in relabel_summary["group_data_dirs"]]
    base_dirs = [str(pathlib.Path(value).resolve()) for value in args.base_dataset]
    bank_dir = fresh_summary.get("initial_state_bank")
    isolation = None
    if bank_dir is not None:
        bank = initial_states.InitialStateBank(bank_dir)
        if fresh_summary.get("initial_state_bank_sha256") != bank.sha256:
            raise ValueError("Fresh summary initial-state bank changed after collection.")
        isolation = bank.audit_partitions(
            {
                "base": initial_states.dataset_groups(base_dirs, bank, allow_legacy_presets=True),
                "fresh_development": initial_states.dataset_groups(development_dirs, bank),
                "final": initial_states.dataset_groups(final_dirs, bank),
            }
        )
        expected_groups = {
            "base": 500,
            "fresh_development": args.expected_development_roots - 500,
            "final": args.expected_final_roots,
        }
        if isolation["partition_group_counts"] != expected_groups:
            raise ValueError("Initial-state partition counts differ from the expansion contract.")
        _write_json(output_dir / "initial_state_isolation.json", isolation)
    elif args.require_initial_state_isolation:
        raise ValueError("Fresh expansion lacks frozen initial-state provenance; training is forbidden.")
    elif _episode_groups(base_dirs + development_dirs).intersection(_episode_groups(final_dirs)):
        raise ValueError("Fresh final episode groups overlap development/training inputs.")

    _write_json(status_path, {"status": "stopping_policy_server", "tmux": args.policy_server_tmux})
    _stop_policy_server(args.policy_server_tmux)

    code_dir = pathlib.Path(__file__).resolve().parent
    consolidated_dir = output_dir / "consolidated_development_900_max10"
    consolidation_report = consolidated_dir / "consolidation_report.json"
    if consolidated_dir.exists():
        report = _complete_json(consolidation_report)
        if (
            int(report["num_roots"]) != args.expected_development_roots
            or int(report["num_replacements"]) != args.expected_replacements
        ):
            raise ValueError("Existing consolidated dataset does not match expected counts.")
    else:
        _write_json(status_path, {"status": "consolidating", "output_dir": str(consolidated_dir)})
        command = [
            sys.executable,
            str(code_dir / "consolidate_execution_horizon_counterfactuals.py"),
            *[item for path in base_dirs + development_dirs for item in ("--base", path)],
            *[item for path in overlay_dirs for item in ("--overlay", path)],
            "--output-dir",
            str(consolidated_dir),
            "--records-per-shard",
            "10",
            "--expected-roots",
            str(args.expected_development_roots),
            "--expected-replacements",
            str(args.expected_replacements),
        ]
        _run(command, output_dir / "consolidation.log")
        _complete_json(consolidation_report)

    predictor_dir = output_dir / f"predictor_pairwise_dev900_seed{args.seed}_lr3e4"
    predictor_summary_path = predictor_dir / "summary.json"
    if predictor_dir.exists():
        summary = _complete_json(predictor_summary_path)
        if not summary["predictor_config"].get("paired_advantage_heads", False):
            raise ValueError("Existing predictor is not the paired-head architecture.")
    else:
        predictor_dir.mkdir(parents=True)
        _write_json(status_path, {"status": "training", "seed": args.seed, "output_dir": str(predictor_dir)})
        train_command = [
            sys.executable,
            str(code_dir / "train_execution_horizon_predictor.py"),
            "--dataset",
            str(consolidated_dir),
            "--output-dir",
            str(predictor_dir),
            "--seed",
            str(args.seed),
            "--split-seed",
            "42",
            "--stratify-splits-by-task",
            "--bootstrap-episode-groups",
            "--train-steps",
            "20000",
            "--batch-size",
            "256",
            "--learning-rate",
            "3e-4",
            "--weight-decay",
            "1e-4",
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
            "--paired-advantage-heads",
            "--validation-fraction",
            "0.15",
            "--calibration-fraction",
            "0.15",
            "--minimum-trials-per-candidate",
            "3",
            "--loss-success",
            "0.5",
            "--loss-timeout",
            "0.25",
            "--loss-remaining-calls",
            "0.1",
            "--loss-remaining-steps",
            "0.1",
            "--loss-final-risk",
            "0.25",
            "--loss-action-cot-risk",
            "0.25",
            "--loss-fused-risk",
            "0.5",
            "--loss-event",
            "0.25",
            "--loss-raw-h-classification",
            "0.1",
            "--loss-raw-h-ordinal",
            "0.1",
            "--loss-survival",
            "1.0",
            "--loss-success-advantage",
            "1.0",
            "--loss-elapsed-advantage",
            "0.25",
            "--loss-calls-advantage",
            "0.1",
            "--loss-false-long",
            "2.0",
            "--loss-danger-rescue",
            "2.0",
            "--loss-paired-elapsed",
            "1.0",
            "--loss-faster-long",
            "1.0",
        ]
        _run(train_command, predictor_dir / "train_stdout.log")
        _complete_json(predictor_summary_path)

    _write_json(status_path, {"status": "calibrating", "predictor_dir": str(predictor_dir)})
    calibration_path = predictor_dir / "calibration.json"
    if not calibration_path.exists():
        calibrate_command = [
            sys.executable,
            str(code_dir / "calibrate_execution_horizon_predictor.py"),
            "--dataset",
            str(consolidated_dir),
            "--predictor-dir",
            str(predictor_dir),
            "--confidence-level",
            "0.95",
            "--ood-probability-threshold",
            "0.95",
            "--seed",
            str(args.seed),
        ]
        _run(calibrate_command, predictor_dir / "calibration_stdout.log")
    _complete_json(predictor_dir / "calibration_report.json")

    audits = {
        "development_official": (consolidated_dir,),
        "final_official": tuple(pathlib.Path(value) for value in final_dirs),
    }
    for name, data_paths in audits.items():
        output_path = predictor_dir / f"{name}_audit.json"
        if output_path.exists():
            _complete_json(output_path)
            continue
        command = [
            sys.executable,
            str(code_dir / "audit_hierarchical_execution_horizon.py"),
            "--dataset",
            *[str(path) for path in data_paths],
            "--predictor-dir",
            str(predictor_dir),
            "--bootstrap-samples",
            "5000",
            "--seed",
            str(args.seed),
            "--output-json",
            str(output_path),
        ]
        if name == "development_official":
            command.extend(("--split-name", "validation"))
        _run(command, predictor_dir / f"{name}_audit.log")

    mean_calibration = predictor_dir / "diagnostic_mean_only_calibration.json"
    _prepare_mean_only_calibration(calibration_path, mean_calibration)
    for name, data_paths in (
        ("development_mean_only", (consolidated_dir,)),
        ("final_mean_only", tuple(pathlib.Path(value) for value in final_dirs)),
    ):
        output_path = predictor_dir / f"{name}_audit.json"
        command = [
            sys.executable,
            str(code_dir / "audit_hierarchical_execution_horizon.py"),
            "--dataset",
            *[str(path) for path in data_paths],
            "--predictor-dir",
            str(predictor_dir),
            "--calibration-json",
            str(mean_calibration),
            "--bootstrap-samples",
            "5000",
            "--seed",
            str(args.seed),
            "--output-json",
            str(output_path),
        ]
        if name == "development_mean_only":
            command.extend(("--split-name", "validation"))
        _run(command, predictor_dir / f"{name}_audit.log")

    result = {
        "status": "complete",
        "consolidation_report": str(consolidation_report),
        "predictor_dir": str(predictor_dir),
        "fresh_final_data_dirs": final_dirs,
        "initial_state_isolation": isolation,
        "development_official": _audit_summary(predictor_dir / "development_official_audit.json"),
        "final_official": _audit_summary(predictor_dir / "final_official_audit.json"),
        "development_mean_only": _audit_summary(predictor_dir / "development_mean_only_audit.json"),
        "final_mean_only": _audit_summary(predictor_dir / "final_mean_only_audit.json"),
    }
    result["dual_official_gate"] = bool(
        result["development_official"]["offline_engineering_gate"]
        and result["final_official"]["offline_engineering_gate"]
    )
    _write_json(output_dir / "summary.json", result)
    _write_json(status_path, {"status": "complete", "dual_official_gate": result["dual_official_gate"]})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
