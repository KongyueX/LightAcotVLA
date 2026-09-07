"""Replay discordant A/history episodes with passive action and state traces."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

MODES = ("ordered_transformer", "ordered_feedback_history")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-eval-dir", type=pathlib.Path, required=True)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser


def write_json(path: pathlib.Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def select_cases(reference: pathlib.Path) -> tuple[dict, list[dict]]:
    config = json.loads((reference / "run_config.json").read_text())
    if json.loads((reference / "summary.json").read_text()).get("status") != "complete":
        raise ValueError("Diagnosis requires a completed reference evaluation.")
    runs = {mode: {} for mode in MODES}
    with (reference / "rollout_rows.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["mode"] not in runs:
                continue
            key = (int(row["task_id"]), int(row["episode"]))
            if key in runs[row["mode"]]:
                raise ValueError(f"Duplicate reference episode: {row['mode']}/{key}.")
            runs[row["mode"]][key] = row
    if set(runs[MODES[0]]) != set(runs[MODES[1]]):
        raise ValueError("Reference A and history episodes are not paired.")
    cases = []
    episode_ids = config["episode_ids"]
    for key, anchor in sorted(runs[MODES[0]].items()):
        candidate = runs[MODES[1]][key]
        if anchor["initial_state_id"] != candidate["initial_state_id"]:
            raise ValueError("Reference initial states differ.")
        if anchor["success"] == candidate["success"]:
            continue
        pair_index = (key[0] - config["task_start"]) * len(episode_ids) + episode_ids.index(key[1])
        modes = list(MODES)
        if config.get("interleave_modes", False) and pair_index % 2:
            modes.reverse()
        cases.append({
            "task_id": key[0], "episode": key[1], "initial_state_id": int(anchor["initial_state_id"]),
            "category": "regression" if int(anchor["success"]) else "rescue",
            "a_success": int(anchor["success"]), "history_success": int(candidate["success"]),
            "mode_order": modes,
        })
    if not cases:
        raise ValueError("Reference has no discordant A/history episodes.")
    return config, cases


def replay_command(args: argparse.Namespace, config: dict, case: dict) -> list[str]:
    output = args.output_dir / "replays" / f"task{case['task_id']:02d}_ep{case['episode']:06d}"
    command = [
        args.python, str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
        "--output-dir", str(output), "--trace-output-dir", str(args.output_dir / "traces"),
        "--trace-video-stride", "5", "--modes", *case["mode_order"], "--interleave-modes",
        "--record-ordered-diagnostics", "--task-start", str(case["task_id"]), "--max-tasks", "1",
        "--episode-ids", str(case["episode"]),
    ]
    for name in (
        "host", "port", "task_suite_name", "seed", "initial_state_offset", "initial_state_bank",
        "resize_size", "num_steps_wait", "action_cot_denoising_steps", "final_denoising_steps",
        "model_action_horizon", "feedback_history_params",
    ):
        if config.get(name) is not None:
            command += ["--" + name.replace("_", "-"), str(config[name])]
    if (output / "run_config.json").exists():
        command.append("--resume")
    return command


def run_stage(args: argparse.Namespace, name: str, command: list[str]) -> None:
    exit_path = args.output_dir / f"{name}.exit"
    if exit_path.exists() and exit_path.read_text().strip() == "0":
        return
    write_json(args.output_dir / f"{name}.command.json", command)
    with (args.output_dir / f"{name}.log").open("a") as log:
        process = subprocess.Popen(command, cwd=args.code_dir, stdout=log, stderr=subprocess.STDOUT)
        write_json(args.output_dir / "status.json", {
            "status": "running", "stage": name, "child_pid": process.pid, "started_at": time.time(),
        })
        result = process.wait()
    exit_path.write_text(f"{result}\n")
    if result:
        write_json(args.output_dir / "status.json", {"status": "failed", "stage": name, "exit_code": result})
        raise RuntimeError(f"{name} failed; see {args.output_dir / (name + '.log')}")


def main(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    args.code_dir = args.code_dir.resolve()
    args.reference_eval_dir = args.reference_eval_dir.resolve()
    config, cases = select_cases(args.reference_eval_dir)
    manifest = {
        "reference_eval_dir": str(args.reference_eval_dir), "cases": cases,
        "purpose": "Passive diagnosis of discordant development episodes; timing is not a performance result.",
        "code_dir": str(args.code_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "case_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Diagnosis directory belongs to another fixed case selection.")
    write_json(manifest_path, manifest)
    for case in cases:
        name = f"replay_task{case['task_id']:02d}_ep{case['episode']:06d}"
        run_stage(args, name, replay_command(args, config, case))
    run_stage(args, "analysis", [
        args.python, str(args.code_dir / "scripts/analyze_execution_horizon_replanning_traces.py"),
        "--trace-dir", str(args.output_dir / "traces"),
        "--baseline-rows", str(args.reference_eval_dir / "rollout_rows.csv"),
        "--output-dir", str(args.output_dir / "analysis"),
    ])
    write_json(args.output_dir / "status.json", {
        "status": "complete", "stage": "complete", "pairs": len(cases), "episodes": 2 * len(cases),
    })


if __name__ == "__main__":
    main(build_parser().parse_args())
