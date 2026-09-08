"""Run a time-bounded paired midchunk experiment with frozen A and VLA."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import pathlib
import sys
import time

import audit_fixed_h_paired as paired
from run_execution_horizon_feedback_experiment import DEV_EPISODES
from run_execution_horizon_feedback_experiment import notify
from run_execution_horizon_feedback_experiment import run_stage
from run_execution_horizon_feedback_experiment import write_json

MODES = ("ordered_transformer", "ordered_midchunk_fresh", "ordered_midchunk_masked")


class WindowEnded(RuntimeError):
    """The authorized wall-clock experiment window has ended."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--initial-state-bank", type=pathlib.Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    return parser


def bounded_stage(args: argparse.Namespace, name: str, command: list[str]) -> None:
    remaining = int(args.deadline_epoch - time.time())
    if remaining <= 0:
        raise WindowEnded("No remaining time in the six-hour window.")
    try:
        run_stage(args, name, ["/usr/bin/timeout", "--signal=TERM", "--kill-after=15s", str(remaining), *command])
    except RuntimeError as exc:
        path = args.output_dir / f"{name}.exit"
        if path.exists() and path.read_text().strip() in {"124", "137"}:
            raise WindowEnded(f"{name} reached the authorized deadline; partial output is preserved.") from exc
        raise


def summarize(directory: pathlib.Path) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("Cannot summarize a partial development cohort as complete.")
    runs = {mode: {} for mode in MODES}
    with (directory / "rollout_rows.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["task_id"]), int(row["episode"]))
            if row["mode"] not in runs or key in runs[row["mode"]]:
                raise ValueError("Unexpected or duplicate development result.")
            runs[row["mode"]][key] = row
    keys = paired._validate_pairing(runs)
    if set(keys) != {(task, episode) for task in range(10) for episode in DEV_EPISODES}:
        raise ValueError("Development results do not cover the fixed 100 initial states.")
    metrics = {mode: paired._run_summary(rows, keys) for mode, rows in runs.items()}
    comparisons = {
        mode: paired._pairwise_audit(runs[MODES[0]], runs[mode], keys, samples=5000, seed=7,
                                    noninferiority_margin=.01)
        for mode in MODES[1:]
    }
    comparisons["fresh_vs_masked"] = paired._pairwise_audit(
        runs[MODES[2]], runs[MODES[1]], keys, samples=5000, seed=7, noninferiority_margin=.01,
    )
    monitor = {mode: {"checks": 0, "replans": 0, "monitor_ms": 0.0} for mode in MODES[1:]}
    with (directory / "decisions.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["mode"] not in monitor:
                continue
            info = json.loads(row["selector_json"])
            record = monitor[row["mode"]]
            record["checks"] += int(info["midchunk_checked"])
            record["replans"] += int(info["replanned_early"])
            record["monitor_ms"] += float(info.get("midchunk_monitor_ms", 0.0))
            if info["replanned_early"] and int(row["execution_horizon"]) != 5:
                raise ValueError("A first-version midchunk replan must occur after five executed actions.")
    for record in monitor.values():
        record["trigger_fraction"] = record["replans"] / record["checks"] if record["checks"] else 0.0
        record["monitor_ms_per_episode"] = record["monitor_ms"] / len(keys)
    result = {"status": "complete", "runs": metrics, "pairwise": comparisons, "monitor": monitor}
    write_json(directory / "paired_analysis.json", result)
    return result


def main(args: argparse.Namespace) -> None:
    csv.field_size_limit(sys.maxsize)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.deadline_epoch = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    collect = args.output_dir / "collection"
    training = args.output_dir / "training"
    development = args.output_dir / "development"
    notify(args.output_dir, "start", "六小时研究第一轮已启动：chunk内第5步的新proprio决定继续/重规划。先采最多80训练+20early状态、两分支各3次，训练fresh/masked小头；A与VLA冻结，所有worker受09:12截止时间约束。")
    try:
        bounded_stage(args, "collect", [
            sys.executable, str(args.code_dir / "scripts/collect_execution_horizon_midchunk.py"),
            "--initial-state-bank", str(args.initial_state_bank), "--output-dir", str(collect),
            "--host", args.host, "--port", str(args.port), "--seed", "97007",
        ])
        notify(args.output_dir, "collection_complete", "第一轮中段continue/replan配对采集完成，进入两个冻结基线之上的轻量小头训练与early选模。")
        bounded_stage(args, "train", [
            sys.executable, str(args.code_dir / "scripts/train_execution_horizon_midchunk.py"),
            "--data-dir", str(collect), "--output-dir", str(training),
        ])
        training_summary = json.loads((training / "summary.json").read_text())
        from openpi.execution_horizon.midchunk import MidchunkMonitor

        fresh = MidchunkMonitor.load(training / "fresh/checkpoint.npz")
        if not fresh.params["output_w"].any() and not fresh.params["output_b"].any() and fresh.threshold >= 0:
            write_json(args.output_dir / "status.json", {
                "status": "awaiting_review", "stage": "training",
                "reason": "Selected fresh monitor is identically continue; no new closed-loop policy to evaluate.",
                "training_summary": str(training / "summary.json"), "finished_at": time.time(),
            })
            notify(args.output_dir, "fresh_zero_selected", "early选择了恒定继续执行的fresh模型，尚无新的闭环策略。本轮训练与标签保留，先交结果复盘，不重复评测等同A的零输出模型，也不据此否定新反馈方向。")
            return
        notify(args.output_dir, "training_complete", "fresh/masked小头训练与early锁模完成；接下来使用同100开发初态比较原A与两个中段监视器，全部实际成本计入闭环。")
        if args.deadline_epoch - time.time() < 3600:
            write_json(args.output_dir / "status.json", {
                "status": "awaiting_review", "stage": "training", "reason": "Less than one hour remains for a complete three-way development cohort.",
                "training_summary": str(training / "summary.json"), "finished_at": time.time(),
            })
            notify(args.output_dir, "development_deferred", "当前窗口剩余不足一小时，完整三组开发对照暂缓；已保留训练结果，交由本任务复盘时间安排，不把未跑开发当完成。")
            return
        command = [
            sys.executable, str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
            "--host", args.host, "--port", str(args.port), "--output-dir", str(development),
            "--initial-state-bank", str(args.initial_state_bank), "--episode-ids", *map(str, DEV_EPISODES),
            "--modes", *MODES, "--interleave-modes", "--record-ordered-diagnostics",
            "--midchunk-fresh-params", str(training / "fresh/checkpoint.npz"),
            "--midchunk-masked-params", str(training / "masked/checkpoint.npz"),
            "--model-action-horizon", "25", "--seed", "7", "--max-tasks", "10",
            "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
            "--num-steps-wait", "10", "--resize-size", "224",
        ]
        if (development / "run_config.json").exists():
            command.append("--resume")
        bounded_stage(args, "development", command)
        analysis = summarize(development)
        write_json(args.output_dir / "summary.json", {
            "status": "complete", "training": training_summary, "development": analysis,
            "deadline_utc": args.deadline_utc,
        })
        write_json(args.output_dir / "status.json", {
            "status": "awaiting_review", "stage": "development", "finished_at": time.time(),
            "summary": str(args.output_dir / "summary.json"),
        })
        lines = ["第一轮三组开发闭环完成，等待一次结果复盘："]
        for mode, row in analysis["runs"].items():
            means = row["means"]
            lines.append(f"{mode}: {row['success_count']}/100成功，RPC {means['policy_rpc_wall_total_ms']/1000:.3f}秒，整局 {means['actual_episode_elapsed_total_ms']/1000:.3f}秒。")
        notify(args.output_dir, "development_complete", "\n".join(lines))
    except WindowEnded as exc:
        write_json(args.output_dir / "status.json", {
            "status": "time_budget_exhausted", "error": str(exc), "finished_at": time.time(),
        })
        notify(args.output_dir, "deadline", "六小时窗口已结束，本轮未完成输出已保留；不再启动实验，等待汇总实际完成范围。")
    except Exception as exc:
        write_json(args.output_dir / "status.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "finished_at": time.time(),
        })
        notify(args.output_dir, "failed", f"中段反馈实验暂停：{type(exc).__name__}。已完成分支保留，按真实错误最小修复后续跑。")
        raise


if __name__ == "__main__":
    main(build_parser().parse_args())
