"""Evaluate a frozen cached-context event policy against A on the fixed dev cohort."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import pathlib
import sys
import time

from openpi.execution_horizon.midchunk import MidchunkMonitor
from run_execution_horizon_feedback_experiment import DEV_EPISODES, notify, result_message, write_json
from run_execution_horizon_midchunk_experiment import WindowEnded, bounded_stage, summarize

MODES = ("ordered_transformer", "ordered_midchunk_masked")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--initial-state-bank", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-params", type=pathlib.Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    return parser


def main(args: argparse.Namespace) -> None:
    csv.field_size_limit(sys.maxsize)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.deadline_epoch = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    candidate = MidchunkMonitor.load(args.candidate_params)
    if candidate.variant != "masked" or candidate.check_schedule != "gripper_event":
        raise ValueError("This comparison requires the frozen masked gripper-event checkpoint.")
    directory = args.output_dir / "development"
    command = [
        sys.executable, str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
        "--host", args.host, "--port", str(args.port), "--output-dir", str(directory),
        "--initial-state-bank", str(args.initial_state_bank), "--episode-ids", *map(str, DEV_EPISODES),
        "--modes", *MODES, "--interleave-modes", "--record-ordered-diagnostics",
        "--midchunk-masked-params", str(args.candidate_params),
        "--model-action-horizon", "25", "--seed", "7", "--max-tasks", "10",
        "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
        "--num-steps-wait", "10", "--resize-size", "224",
    ]
    if (directory / "run_config.json").exists():
        command.append("--resume")
    notify(args.output_dir, "start", "开始固定cached-context事件策略与A的完整开发对照，各100局共200局。复用事件masked已选checkpoint与阈值，不再训练或校准；这检验已有上下文的事件调度，不宣称新proprio增益。")
    try:
        bounded_stage(args, "development", command)
        analysis = summarize(directory, MODES)
        write_json(args.output_dir / "summary.json", {
            "status": "complete", "development": analysis, "candidate_params": str(args.candidate_params),
            "candidate_metadata": candidate.metadata, "threshold": candidate.threshold,
            "training_performed": False, "scope": "Frozen cached-context event policy; no fresh-proprio benefit claim.",
        })
        write_json(args.output_dir / "status.json", {
            "status": "awaiting_review", "stage": "development", "finished_at": time.time(),
            "summary": str(args.output_dir / "summary.json"),
        })
        notify(args.output_dir, "development_complete", result_message("cached-context事件策略完整开发对照完成", analysis))
    except WindowEnded as exc:
        write_json(args.output_dir / "status.json", {
            "status": "time_budget_exhausted", "error": str(exc), "finished_at": time.time(),
        })
        notify(args.output_dir, "deadline", "六小时窗口已结束，已完成开发episode保留，不再启动后续实验。")
    except Exception as exc:
        write_json(args.output_dir / "status.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "finished_at": time.time(),
        })
        notify(args.output_dir, "failed", f"cached-context事件对照暂停：{type(exc).__name__}，按实际错误处理并保留完成episode。")
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
