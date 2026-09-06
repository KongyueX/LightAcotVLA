"""Run one bounded two-round head-only SMDP PPO pilot against a frozen H25+A server."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np

MODE = "ordered_smdp"
ANCHOR_MODE = "ordered_transformer"
TRAIN_EPISODES = (tuple(range(100, 110)), tuple(range(110, 120)))
VALIDATION_EPISODES = tuple(range(20, 30))
FINAL_EPISODES = tuple(range(20))
METRICS = {
    "mean_calls": ("policy_calls", 1.0),
    "mean_policy_seconds": ("actual_policy_total_ms", 1000.0),
    "mean_rpc_seconds": ("policy_rpc_wall_total_ms", 1000.0),
    "mean_elapsed_seconds": ("actual_episode_elapsed_total_ms", 1000.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-dir", "code-dir", "train-state-bank", "reference-eval-dir", "anchor-predictor-dir",
    ):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--original-reference-eval-dir", type=pathlib.Path)
    parser.add_argument("--python", type=pathlib.Path, default=pathlib.Path(sys.executable))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status(output_dir: pathlib.Path, phase: str, **details: Any) -> None:
    _write_json(output_dir / "status.json", {"phase": phase, "updated_at": time.time(), **details})


def validate_train_bank(bank_dir: pathlib.Path) -> dict[str, Any]:
    """Read the existing ID columns; do not generate states or rehash the bank."""
    manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("task_suite") != "libero_10":
        raise ValueError("Training bank must be complete and use libero_10.")
    train_ids = set().union(*TRAIN_EPISODES)
    if train_ids & (set(VALIDATION_EPISODES) | set(FINAL_EPISODES)):
        raise ValueError("Training episode IDs overlap this pilot's validation/final IDs.")
    tasks = {int(entry["task_id"]): entry for entry in manifest["tasks"]}
    counts = {}
    for task_id in range(10):
        entry = tasks[task_id]
        with np.load(bank_dir / entry["file"], allow_pickle=False) as archive:
            episode_ids = set(int(value) for value in archive["episode_ids"])
        if not train_ids <= episode_ids:
            raise ValueError(f"Training bank task {task_id} lacks requested episode IDs 100–119.")
        counts[str(task_id)] = len(episode_ids)
    return {
        "directory": str(bank_dir), "status": "complete", "task_suite": "libero_10",
        "round_episode_ids": [list(ids) for ids in TRAIN_EPISODES],
        "available_ids_per_task": counts, "validation_or_final_ids_in_training": False,
    }


def _read_eval(
    directory: pathlib.Path, *, mode: str, episodes: tuple[int, ...], seed: int,
) -> tuple[dict[tuple[int, int], dict[str, str]], dict[str, Any]]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"Evaluation is not complete: {directory}")
    config = summary["config"]
    for name, expected in {
        "seed": seed, "num_steps_wait": 10, "resize_size": 224, "action_cot_denoising_steps": 10,
    }.items():
        if config.get(name) is not None and config[name] != expected:
            raise ValueError(f"Evaluation {name} differs from pilot protocol at {directory}.")
    if summary.get("task_suite", config.get("task_suite_name")) != "libero_10":
        raise ValueError("Pilot references must use libero_10.")
    if config.get("final_denoising_steps") not in (None, 10):
        raise ValueError("Pilot references must use final NFE10 or its recorded server default.")
    rows = {}
    with (directory / "rollout_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != mode or int(row["episode"]) not in episodes:
                continue
            key = (int(row["task_id"]), int(row["episode"]))
            if key in rows:
                raise ValueError(f"Duplicate {mode} rollout at {key}.")
            rows[key] = row
    expected_keys = {(task, episode) for task in range(10) for episode in episodes}
    if set(rows) != expected_keys:
        raise ValueError(f"Incomplete {mode} task/episode grid at {directory}.")
    return rows, summary


def summarize(rows: dict[tuple[int, int], dict[str, str]]) -> dict[str, Any]:
    items = list(rows.values())
    result = {
        "episodes": len(items),
        "success_count": sum(int(row["success"]) for row in items),
        "success_rate": float(np.mean([int(row["success"]) for row in items])),
    }
    for name, (field, divisor) in METRICS.items():
        values = [float(row[field]) / divisor for row in items]
        if not all(np.isfinite(values)):
            raise ValueError(f"Missing/nonfinite {field} in rollout results.")
        result[name] = float(np.mean(values))
    histogram: collections.Counter[str] = collections.Counter()
    for row in items:
        histogram.update(json.loads(row.get("h_distribution_json") or "{}"))
    result["execution_horizon_counts"] = dict(sorted(histogram.items(), key=lambda item: int(item[0])))
    result["per_task"] = {}
    for task_id in sorted({key[0] for key in rows}):
        subset = [row for (task, _), row in rows.items() if task == task_id]
        result["per_task"][str(task_id)] = {
            "episodes": len(subset),
            "success_count": sum(int(row["success"]) for row in subset),
            **{
                name: float(np.mean([float(row[field]) / divisor for row in subset]))
                for name, (field, divisor) in METRICS.items()
            },
        }
    return result


def compare(
    reference: dict[tuple[int, int], dict[str, str]], candidate: dict[tuple[int, int], dict[str, str]],
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise ValueError("Comparison task/episode keys differ.")
    rescues, regressions = [], []
    for key in sorted(reference):
        if int(reference[key]["initial_state_id"]) != int(candidate[key]["initial_state_id"]):
            raise ValueError(f"Comparison initial_state_id differs at {key}.")
        before, after = int(reference[key]["success"]), int(candidate[key]["success"])
        identity = {"task_id": key[0], "episode": key[1], "initial_state_id": int(reference[key]["initial_state_id"])}
        if after > before:
            rescues.append(identity)
        elif after < before:
            regressions.append(identity)
    baseline, new = summarize(reference), summarize(candidate)
    return {
        "paired_episodes": len(reference), "initial_state_id_matched": True,
        "reference": baseline, "candidate": new,
        "success_delta_pp": 100 * (new["success_rate"] - baseline["success_rate"]),
        "rescues": len(rescues), "regressions": len(regressions),
        "rescue_states": rescues, "regression_states": regressions,
        "metric_delta_candidate_minus_reference": {name: new[name] - baseline[name] for name in METRICS},
        "timing_scope": "Observed cross-run comparison, not a controlled same-run timing estimate.",
    }


def eligible(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return candidate["mean_rpc_seconds"] <= 3.0 and (
        candidate["success_count"], -candidate["mean_rpc_seconds"]
    ) > (baseline["success_count"], -baseline["mean_rpc_seconds"])


def build_eval_command(
    args: argparse.Namespace, *, checkpoint: pathlib.Path, output_dir: pathlib.Path,
    episodes: tuple[int, ...], sample: bool,
) -> list[str]:
    command = [
        str(args.python), "-u", str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
        "--host", args.host, "--port", str(args.port), "--output-dir", str(output_dir),
        "--modes", MODE, "--ordered-smdp-params", str(checkpoint), "--record-ordered-diagnostics",
        "--task-suite-name", "libero_10", "--task-start", "0", "--max-tasks", "10",
        "--num-trials-per-task", str(len(episodes)), "--episode-ids", *(str(value) for value in episodes),
        "--initial-state-offset", "0", "--seed", str(args.seed), "--num-steps-wait", "10",
        "--resize-size", "224", "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
        "--model-action-horizon", "25", "--warmup-requests", "1",
    ]
    if sample:
        command.extend(["--ordered-smdp-sample", "--initial-state-bank", str(args.train_state_bank)])
    return command


def build_train_command(
    args: argparse.Namespace, *, checkpoint: pathlib.Path, rollout_dir: pathlib.Path, output_dir: pathlib.Path,
) -> list[str]:
    return [
        str(args.python), "-u", str(args.code_dir / "scripts/train_ordered_horizon_smdp.py"),
        "--rollout-dir", str(rollout_dir), "--input-checkpoint", str(checkpoint),
        "--output-dir", str(output_dir), "--epochs", "4", "--seed", str(args.seed),
    ]


def _environment(args: argparse.Namespace, *, cpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    paths = [args.code_dir / "src", args.code_dir / "packages/openpi-client/src", args.code_dir / "scripts"]
    env["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in paths), env.get("PYTHONPATH", "")])
    if cpu:
        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    return env


def _run_stage(
    args: argparse.Namespace, parent: pathlib.Path, name: str, command: list[str], *, cpu: bool,
) -> None:
    _status(args.output_dir, name, stage_dir=str(parent), command=command)
    _write_json(parent / f"{name}.command.json", {"command": command, "cpu": cpu})
    with (parent / f"{name}.log").open("x", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command, cwd=args.code_dir, env=_environment(args, cpu=cpu),
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            code = completed.returncode
        except OSError as error:
            log.write(str(error) + "\n")
            code = 127
    (parent / f"{name}.exit").write_text(f"{code}\n", encoding="utf-8")
    if code:
        raise subprocess.CalledProcessError(code, command)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    bank = validate_train_bank(args.train_state_bank)
    baseline_rows, reference_summary = _read_eval(
        args.reference_eval_dir, mode=ANCHOR_MODE, episodes=VALIDATION_EPISODES, seed=args.seed,
    )
    baseline = summarize(baseline_rows)
    protocol = {
        "experiment": "bounded_head_only_ordered_smdp_ppo",
        "frozen_vla_and_A_encoder": True, "maximum_rounds": 2, "ppo_epochs_per_round": 4,
        "code_dir": str(args.code_dir), "server": {"host": args.host, "port": args.port},
        "anchor_predictor_dir": str(args.anchor_predictor_dir), "seed": args.seed,
        "reference_eval_dir": str(args.reference_eval_dir),
        "original_reference_eval_dir": str(args.original_reference_eval_dir) if args.original_reference_eval_dir else None,
        "training_bank": bank, "validation_episode_ids": list(VALIDATION_EPISODES),
        "final_episode_ids": list(FINAL_EPISODES), "maximum_validation_rpc_seconds": 3.0,
        "selection": "lexicographic (validation success_count, -mean_rpc_seconds), strictly above A",
        "all_results_development": True,
        "caveat": "Validation/final presets were previously used in development; this is not independent confirmation.",
    }
    _write_json(args.output_dir / "run_config.json", protocol)
    step0 = args.output_dir / "step0"
    step0.mkdir()
    checkpoint = step0 / "selector.npz"
    initialize = (
        "from openpi.execution_horizon.ordered_smdp import OrderedSMDPSelector; "
        f"selector=OrderedSMDPSelector.initialize(feature_dim=256, seed={args.seed}); "
        f"selector.save({str(checkpoint)!r}); print('INITIALIZED_ZERO_ACTOR')"
    )
    _run_stage(args, step0, "initialize", [str(args.python), "-c", initialize], cpu=True)
    _write_json(step0 / "summary.json", {
        "status": "complete", "selector_checkpoint": str(checkpoint), "validation": baseline,
        "validation_source": "existing A rows; zero actor preserves greedy A",
        "reference_source_summary": reference_summary,
    })
    rounds = []
    for round_index, episodes in enumerate(TRAIN_EPISODES, start=1):
        round_dir = args.output_dir / f"round{round_index:02d}"
        round_dir.mkdir()
        input_checkpoint = checkpoint
        collection_dir = round_dir / "collection"
        _run_stage(args, round_dir, "collect", build_eval_command(
            args, checkpoint=input_checkpoint, output_dir=collection_dir, episodes=episodes, sample=True,
        ), cpu=False)
        collection_rows, collection_summary = _read_eval(collection_dir, mode=MODE, episodes=episodes, seed=args.seed)
        train_dir = round_dir / "training"
        _run_stage(args, round_dir, "train", build_train_command(
            args, checkpoint=input_checkpoint, rollout_dir=collection_dir, output_dir=train_dir,
        ), cpu=True)
        training = json.loads((train_dir / "summary.json").read_text(encoding="utf-8"))
        if training.get("status") != "complete":
            raise ValueError(f"Trainer did not complete: {train_dir}")
        checkpoint = train_dir / "selector.npz"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        validation_dir = round_dir / "validation"
        _run_stage(args, round_dir, "validate", build_eval_command(
            args, checkpoint=checkpoint, output_dir=validation_dir, episodes=VALIDATION_EPISODES, sample=False,
        ), cpu=False)
        validation_rows, _ = _read_eval(validation_dir, mode=MODE, episodes=VALIDATION_EPISODES, seed=args.seed)
        validation = summarize(validation_rows)
        result = {
            "round": round_index, "status": "complete", "input_checkpoint": str(input_checkpoint),
            "selector_checkpoint": str(checkpoint), "collection": summarize(collection_rows),
            "collection_source_summary": collection_summary, "training_summary": training,
            "validation": validation, "validation_vs_A": compare(baseline_rows, validation_rows),
            "eligible": eligible(validation, baseline),
        }
        _write_json(round_dir / "summary.json", result)
        rounds.append(result)
    candidates = [result for result in rounds if result["eligible"]]
    result = {
        "status": "complete", **protocol, "baseline_validation": baseline, "rounds": rounds,
        "final_run": False, "selected_system": "A", "selected_checkpoint": None,
    }
    if not candidates:
        result["conclusion"] = "No candidate exceeded A under the validation/RPC criterion; retain A, skip final."
        return result
    best = max(candidates, key=lambda item: (item["validation"]["success_count"], -item["validation"]["mean_rpc_seconds"]))
    best_checkpoint = pathlib.Path(best["selector_checkpoint"])
    final_dir = args.output_dir / "final"
    _run_stage(args, args.output_dir, "final_eval", build_eval_command(
        args, checkpoint=best_checkpoint, output_dir=final_dir, episodes=FINAL_EPISODES, sample=False,
    ), cpu=False)
    final_rows, _ = _read_eval(final_dir, mode=MODE, episodes=FINAL_EPISODES, seed=args.seed)
    anchor_rows, _ = _read_eval(args.reference_eval_dir, mode=ANCHOR_MODE, episodes=FINAL_EPISODES, seed=args.seed)
    result.update(
        final_run=True, selected_system=MODE, selected_checkpoint=str(best_checkpoint), selected_round=best["round"],
        final=summarize(final_rows), final_vs_A=compare(anchor_rows, final_rows),
        conclusion="One validation-selected candidate completed the sole development final evaluation; no further rounds.",
        legacy_A_pilot_caveat=(
            "Earlier A pilot reported 189/200; the reference here is episode0–19 from the current 1000-episode A run. "
            "Identical task/episode IDs do not imply byte-identical simulator trajectories across runs."
        ),
    )
    if args.original_reference_eval_dir is not None:
        original_rows, _ = _read_eval(
            args.original_reference_eval_dir, mode="original", episodes=FINAL_EPISODES, seed=args.seed,
        )
        result["final_vs_historical_original"] = compare(original_rows, final_rows)
    return result


def main(args: argparse.Namespace) -> None:
    for name in ("output_dir", "code_dir", "python", "train_state_bank", "reference_eval_dir", "anchor_predictor_dir"):
        setattr(args, name, getattr(args, name).absolute())
    if args.original_reference_eval_dir is not None:
        args.original_reference_eval_dir = args.original_reference_eval_dir.absolute()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        summary = _run(args)
        _write_json(args.output_dir / "summary.json", summary)
        _status(args.output_dir, "complete", selected_system=summary["selected_system"], final_run=summary["final_run"])
    except Exception as error:
        _status(args.output_dir, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
