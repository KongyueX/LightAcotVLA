"""Compare A with one current-selector intervention per LIBERO episode."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

from run_execution_horizon_feedback_experiment import DEV_EPISODES
from run_execution_horizon_feedback_experiment import notify
from run_execution_horizon_feedback_experiment import result_message
from run_execution_horizon_feedback_experiment import run_stage
from run_execution_horizon_feedback_experiment import summarize_eval
from run_execution_horizon_feedback_experiment import write_json


ANCHOR = "ordered_transformer"
ONCE = "ordered_feedback_current_once"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--initial-state-bank", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-params", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    return parser


def intervention_summary(directory: pathlib.Path) -> dict:
    events = {(task, episode): [] for task in range(10) for episode in DEV_EPISODES}
    with (directory / "decisions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != ONCE:
                continue
            info = json.loads(row["selector_json"])
            changed = info["first_disagreement_intervened"]
            expected_h = info["candidate_h"] if changed else info["anchor_h"]
            if int(row["selected_horizon"]) != expected_h:
                raise ValueError("Single-intervention output does not match the selected policy.")
            if changed:
                events[int(row["task_id"]), int(row["episode"])].append({
                    "environment_step": int(row["environment_step"]),
                    "anchor_h": info["anchor_h"], "candidate_h": info["candidate_h"],
                })
    if any(len(values) > 1 for values in events.values()):
        raise ValueError("An episode applied more than one current intervention.")
    return {
        "episodes_with_intervention": sum(bool(values) for values in events.values()),
        "episodes_without_intervention": sum(not values for values in events.values()),
        "episodes": [
            {"task_id": task, "episode_id": episode, "interventions": values}
            for (task, episode), values in events.items()
        ],
    }


def main(args: argparse.Namespace) -> None:
    csv.field_size_limit(sys.maxsize)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = output / "development"
    command = [
        sys.executable, str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
        "--host", args.host, "--port", str(args.port), "--output-dir", str(evaluation),
        "--initial-state-bank", str(args.initial_state_bank),
        "--episode-ids", *map(str, DEV_EPISODES), "--modes", ANCHOR, ONCE,
        "--feedback-current-params", str(args.candidate_params),
        "--interleave-modes", "--record-ordered-diagnostics",
        "--model-action-horizon", "25", "--seed", "7", "--max-tasks", "10",
        "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
        "--num-steps-wait", "10", "--resize-size", "224",
    ]
    if (evaluation / "run_config.json").exists():
        command.append("--resume")
    notify(output, "start", "单次介入测试已启动：复用current step200，默认A控制，每局首次H分歧只采用一次current H，随后回A。同100个开发初态两种策略交错运行，共200局，比较成功数与实际RPC/整局耗时；不重训。")
    try:
        run_stage(args, "development", command)
        analysis = summarize_eval(evaluation, DEV_EPISODES, ["current_once"])
        interventions = intervention_summary(evaluation)
        write_json(output / "summary.json", {
            "status": "complete", "development": analysis,
            "interventions": interventions,
            "candidate_params": str(args.candidate_params), "training_performed": False,
            "scope": "100 development initial states, A versus one first-disagreement current intervention; no held-out test.",
        })
        write_json(output / "status.json", {
            "status": "awaiting_analysis", "stage": "development",
            "finished_at": time.time(), "summary": str(output / "summary.json"),
        })
        notify(output, "development_complete", result_message("单次介入完整开发对照已完成", analysis)
               + f"\n其中{interventions['episodes_with_intervention']}/100局发生一次介入。")
    except Exception as exc:
        write_json(output / "status.json", {
            "status": "failed", "stage": "development", "finished_at": time.time(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        notify(output, "failed", f"单次介入测试暂停：{type(exc).__name__}。已完成episode保留，按实际错误处理后续跑。")
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
