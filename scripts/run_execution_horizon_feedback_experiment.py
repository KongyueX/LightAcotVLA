"""Collect paired labels, fit observation-feedback residuals, and evaluate them."""
# ruff: noqa: SLF001, RUF001

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np

ANCHOR = "ordered_transformer"
VARIANTS = ("current", "history")
TRAIN_EPISODES = tuple(range(300, 330))
EARLY_EPISODES = tuple(range(330, 336))
DEV_EPISODES = tuple(range(336, 346))
FINAL_EPISODES = tuple(range(346, 366))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--a-predictor-dir", type=pathlib.Path, required=True)
    parser.add_argument("--parent-bank", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8040)
    return parser


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def diagnostic_episodes(task: int) -> tuple[int, ...]:
    count = 10 if task in (3, 4) else 6 if task == 8 else 2
    return TRAIN_EPISODES[:count]


def notify(output: pathlib.Path, key: str, message: str) -> None:
    marker = output / "notifications" / f"{key}.json"
    if marker.exists():
        return
    webhook = os.environ.get("FEISHU_WEBHOOK_URL")
    secret = os.environ.get("FEISHU_SIGNING_SECRET")
    if not webhook or not secret:
        print("FEISHU_NOT_CONFIGURED", flush=True)
        return
    from watch_experiment_feishu import _send_feishu

    try:
        _send_feishu(message, webhook_url=webhook, secret=secret)
    except Exception as exc:
        print(f"FEISHU_FAILED {type(exc).__name__}", flush=True)
        return
    write_json(marker, {"sent_at": time.time(), "message": message, "result": "FEISHU_OK"})
    print(f"FEISHU_OK {key}", flush=True)


def run_stage(args: argparse.Namespace, name: str, command: list[str]) -> None:
    output = args.output_dir
    exit_path = output / f"{name}.exit"
    if exit_path.exists() and exit_path.read_text().strip() == "0":
        return
    status_path = output / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        pid = previous.get("child_pid")
        proc = pathlib.Path(f"/proc/{pid}/cmdline")
        if previous.get("stage") == name and pid and proc.exists():
            actual_command = proc.read_bytes().rstrip(b"\0").split(b"\0")
            if actual_command == [value.encode() for value in previous["command"]]:
                raise RuntimeError(f"Stage {name} is still running in PID {pid}; do not start a duplicate.")
    if name in ("train_current", "train_history", "generate_bank"):
        destination = pathlib.Path(command[command.index("--output-dir") + 1])
        completion = destination / ("manifest.json" if name == "generate_bank" else "summary.json")
        if completion.exists() and json.loads(completion.read_text()).get("status") == "complete":
            exit_path.write_text("0\n")
            return
        if destination.exists() and any(destination.iterdir()):
            destination.rename(destination.with_name(f"{destination.name}.interrupted_{time.time_ns()}"))
    write_json(output / f"{name}.command.json", command)
    environment = dict(os.environ, JAX_PLATFORMS="cpu")
    with (output / f"{name}.log").open("a") as log:
        process = subprocess.Popen(command, cwd=args.code_dir, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write_json(status_path, {
            "status": "running", "stage": name, "started_at": time.time(), "command": command,
            "child_pid": process.pid,
        })
        returncode = process.wait()
    exit_path.write_text(f"{returncode}\n")
    if returncode:
        raise RuntimeError(f"{name} exited {returncode}; see {output / (name + '.log')}")


def collector_command(args: argparse.Namespace, task: int, episodes: tuple[int, ...], output: pathlib.Path) -> list[str]:
    return [
        args.python, str(args.code_dir / "scripts/collect_execution_horizon_feedback.py"),
        "--initial-state-bank", str(args.output_dir / "initial_state_bank_0_365"),
        "--episodes", *map(str, episodes), "--task-start", str(task), "--max-tasks", "1",
        "--output-dir", str(output), "--host", args.host, "--port", str(args.port),
        "--seed", "67007", "--branch-repeats", "5", "--action-cot-denoising-steps", "10",
        "--final-denoising-steps", "10", "--num-steps-wait", "10", "--resize-size", "224",
    ]


def eval_command(args: argparse.Namespace, phase: str, episodes: tuple[int, ...], variants: list[str]) -> list[str]:
    command = [
        args.python, str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
        "--host", args.host, "--port", str(args.port), "--output-dir", str(args.output_dir / phase),
        "--initial-state-bank", str(args.output_dir / "initial_state_bank_0_365"),
        "--episode-ids", *map(str, episodes), "--modes", ANCHOR,
        *(f"ordered_feedback_{variant}" for variant in variants),
        "--interleave-modes", "--record-ordered-diagnostics",
        "--model-action-horizon", "25", "--seed", "7", "--max-tasks", "10",
        "--action-cot-denoising-steps", "10", "--final-denoising-steps", "10",
        "--num-steps-wait", "10", "--resize-size", "224",
    ]
    for variant in variants:
        command += [f"--feedback-{variant}-params", str(args.output_dir / f"training_{variant}/checkpoint.npz")]
    if (args.output_dir / phase / "run_config.json").exists():
        command.append("--resume")
    return command


def summarize_eval(directory: pathlib.Path, episodes: tuple[int, ...], variants: list[str]) -> dict[str, Any]:
    import audit_fixed_h_paired as paired

    target = directory / "paired_analysis.json"
    if target.exists():
        return json.loads(target.read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("Cannot summarize incomplete feedback evaluation.")
    modes = [ANCHOR, *(f"ordered_feedback_{variant}" for variant in variants)]
    runs: dict[str, dict] = {mode: {} for mode in modes}
    with (directory / "rollout_rows.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["task_id"]), int(row["episode"]))
            if row["mode"] not in runs or key in runs[row["mode"]]:
                raise ValueError("Unexpected or duplicate feedback rollout key.")
            if not all(np.isfinite(float(row[field])) for field in paired._METRICS):
                raise ValueError("Feedback evaluation has incomplete timing metrics.")
            runs[row["mode"]][key] = row
    keys = paired._validate_pairing(runs)
    if set(keys) != {(task, episode) for task in range(10) for episode in episodes}:
        raise ValueError("Feedback evaluation does not cover its complete fixed episode split.")
    metrics = {mode: paired._run_summary(rows, keys) for mode, rows in runs.items()}
    comparisons = {
        mode: paired._pairwise_audit(runs[ANCHOR], runs[mode], keys, samples=5000, seed=7, noninferiority_margin=.01)
        for mode in modes[1:]
    }
    if len(modes) == 3:
        comparisons["history_vs_current"] = paired._pairwise_audit(
            runs[modes[1]], runs[modes[2]], keys, samples=5000, seed=7, noninferiority_margin=.01,
        )
    overhead = {mode: [] for mode in modes}
    with (directory / "decisions.csv").open() as handle:
        for row in csv.DictReader(handle):
            info = json.loads(row["selector_json"])
            overhead[row["mode"]].append(float(info.get("selector_postprocess_ms", 0.0)))
    result = {
        "status": "complete", "runs": metrics, "pairwise": comparisons,
        "feedback_selector_ms_per_call": {
            mode: float(np.mean(values)) if values else 0.0 for mode, values in overhead.items()
        },
        "semantics": "Same H25/A service and new bank states; modes interleaved per episode. Feedback time is outside RPC and inside full episode.",
    }
    write_json(target, result)
    return result


def select_development_candidate(analysis: dict[str, Any]) -> str | None:
    def score(mode: str) -> tuple[int, float]:
        row = analysis["runs"][mode]
        return row["success_count"], -row["means"]["policy_rpc_wall_total_ms"]

    winner = max(analysis["runs"], key=score)
    return winner.removeprefix("ordered_feedback_") if winner != ANCHOR and score(winner) > score(ANCHOR) else None


def result_message(label: str, analysis: dict[str, Any]) -> str:
    lines = [label]
    for mode, row in analysis["runs"].items():
        means = row["means"]
        lines.append(
            f"{mode}: {row['success_count']}/{row['episodes']}成功，"
            f"policy {means['actual_policy_total_ms']/1000:.3f}秒，"
            f"RPC {means['policy_rpc_wall_total_ms']/1000:.3f}秒，"
            f"整局 {means['actual_episode_elapsed_total_ms']/1000:.3f}秒。"
        )
    return "\n".join(lines)


def write_report(args: argparse.Namespace, development: dict, final: dict | None, winner: str | None) -> None:
    lines = ["# Predictor A 配对监督与历史反馈实验", "", "H25 VLA与A参数冻结；新300个训练root、60个早停root，各五H×五次配对分支。", ""]
    for label, result in (("开发验证（每模式100局）", development), ("预留测试（每模式200局）", final)):
        if result is None:
            continue
        lines += [f"## {label}", "", "| 模式 | 成功 | Policy秒 | RPC秒 | 整局秒 |", "| --- | ---: | ---: | ---: | ---: |"]
        for mode, row in result["runs"].items():
            means = row["means"]
            lines.append(f"| {mode} | {row['success_count']}/{row['episodes']} | {means['actual_policy_total_ms']/1000:.4f} | {means['policy_rpc_wall_total_ms']/1000:.4f} | {means['actual_episode_elapsed_total_ms']/1000:.4f} |")
        lines.append("")
        anchor = result["runs"][ANCHOR]
        for mode, row in result["runs"].items():
            if mode == ANCHOR:
                continue
            comparison = result["pairwise"][mode]
            rpc_change = row["means"]["policy_rpc_wall_total_ms"] / anchor["means"]["policy_rpc_wall_total_ms"] - 1
            elapsed_change = row["means"]["actual_episode_elapsed_total_ms"] / anchor["means"]["actual_episode_elapsed_total_ms"] - 1
            lines.append(
                f"{mode}相对A：救回{comparison['rescues']}局、退化{comparison['regressions']}局，"
                f"成功率差{100 * (row['success_rate'] - anchor['success_rate']):+.2f}个百分点，"
                f"RPC变化{100 * rpc_change:+.2f}%，整局变化{100 * elapsed_change:+.2f}%。"
            )
        lines.append("")
    lines += [
        f"开发阶段候选：{winner or '保留A'}。配对救回/退化、置信区间及逐任务结果见各评测目录 paired_analysis.json。",
        "", "## Limitations", "",
        "单训练seed和有限配对重复；反事实标签只改变当前H，后续由A继续，离线优势不等于候选闭环收益。",
        "扩展bank保留旧0–299，仅使用新300–365，训练、早停、开发及预留测试互不交叉；预留测试不参与选模。",
        "候选残差耗时在RPC以外、整局以内，原A predictor耗时已计入RPC。所有原模型保留，结果不自动替换A服务。",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def notify_completion(output: pathlib.Path, summary: dict[str, Any]) -> None:
    final = summary.get("final")
    message = result_message("Predictor反馈预留测试完成（每模式200局）", final) if final else result_message(
        "Predictor反馈开发比较已结束，无胜出候选", summary["development"],
    )
    notify(output, "complete", message + "\n本批已结束，A服务与全部新旧模型/数据保留；完整配对结果与报告已落盘，不追加采集或扫参。")


def execute(args: argparse.Namespace) -> None:
    from collect_execution_horizon_feedback import summarize_records

    from openpi.execution_horizon.initial_states import InitialStateBank
    from openpi.execution_horizon.initial_states import audit_bank_prefix

    output = args.output_dir
    parent = InitialStateBank(args.parent_bank)
    if any(len(parent.states[task]) != 300 for task in range(10)):
        raise ValueError("This experiment extends the existing 0–299 bank.")
    bank = output / "initial_state_bank_0_365"
    config = parent.manifest["generation_config"]
    command = [args.python, str(args.code_dir / "scripts/generate_execution_horizon_initial_states.py"), "--output-dir", str(bank)]
    for key, value in dict(config, generated_per_task=316).items():
        command += ["--" + key.replace("_", "-"), str(value)]
    run_stage(args, "generate_bank", command)
    if not (output / "bank_lineage.json").exists():
        write_json(output / "bank_lineage.json", audit_bank_prefix(parent, InitialStateBank(bank)))
    write_json(output / "split_manifest.json", {
        "bank": str(bank), "train": TRAIN_EPISODES, "early_stop": EARLY_EPISODES,
        "development": DEV_EPISODES, "final": FINAL_EPISODES, "tasks": list(range(10)),
    })
    notify(output, "bank_ready", "Predictor A新实验：初态bank扩展完成，旧0–299保留；新300–329训练、330–335早停、336–345开发、346–365预留测试，每个角色跨10任务。开始40个训练root配对诊断，每root五H×五次；所有root正常/失败均保留。")
    for task in (3, 4, 8, 0, 1, 2, 5, 6, 7, 9):
        run_stage(args, f"diagnostic_task{task:02d}", collector_command(
            args, task, diagnostic_episodes(task), output / "train/diagnostic" / f"task{task:02d}",
        ))
    diagnostic = summarize_records(sorted((output / "train/diagnostic").rglob("*.npz")))
    write_json(output / "diagnostic_summary.json", diagnostic)
    notify(output, "diagnostic_complete", "Predictor A的40个新训练root配对诊断完成，摘要：\n" + json.dumps(diagnostic, ensure_ascii=False)[:3200] + "\n这是同状态H后果诊断，不是新模型闭环成绩；继续既定300训练/60早停roots。")
    for task in range(10):
        remaining = tuple(ep for ep in TRAIN_EPISODES if ep not in diagnostic_episodes(task))
        run_stage(args, f"train_roots_task{task:02d}", collector_command(args, task, remaining, output / "train/remaining" / f"task{task:02d}"))
        run_stage(args, f"early_roots_task{task:02d}", collector_command(args, task, EARLY_EPISODES, output / "early_stop" / f"task{task:02d}"))
    collection = {
        "training": summarize_records(sorted((output / "train").rglob("*.npz"))),
        "early_stop": summarize_records(sorted((output / "early_stop").rglob("*.npz"))),
    }
    write_json(output / "collection_summary.json", collection)
    notify(output, "collection_complete", "Predictor A配对采集完成：300个独立训练root＋60个早停root，各五H×五次，开始current-only与history两个相同训练设置的残差候选。冻结A和VLA；目标为配对成功差−0.02×RPC秒差＋0.05 KL约束，无critic、无扫参。")
    for variant in VARIANTS:
        run_stage(args, f"train_{variant}", [
            args.python, str(args.code_dir / "scripts/train_execution_horizon_feedback.py"),
            "--train-dir", str(output / "train"), "--validation-dir", str(output / "early_stop"),
            "--a-predictor-dir", str(args.a_predictor_dir), "--output-dir", str(output / f"training_{variant}"),
            "--variant", variant, "--seed", "7",
        ])
    notify(output, "training_complete", "两个残差候选训练完成。开始新100初态的A/current-only/history三模式闭环，逐episode交错、同H25与A服务；离线loss不作为成功率结果。")
    run_stage(args, "development", eval_command(args, "development", DEV_EPISODES, list(VARIANTS)))
    development = summarize_eval(output / "development", DEV_EPISODES, list(VARIANTS))
    winner = select_development_candidate(development)
    write_json(output / "selection.json", {"winner": winner, "criterion": "success_count then lower RPC; strictly above A", "stage": "development"})
    notify(output, "development_complete", result_message("Predictor反馈开发闭环完成（每模式100局）", development) + (f"\n候选{winner}进入唯一预留200局对A测试。" if winner else "\n无候选优于A，本批结束，保留A，不追加训练或预留测试。"))
    final = None
    if winner is not None:
        run_stage(args, "final", eval_command(args, "final", FINAL_EPISODES, [winner]))
        final = summarize_eval(output / "final", FINAL_EPISODES, [winner])
    write_report(args, development, final, winner)
    summary = {"status": "complete", "development_candidate": winner, "final_run": final is not None,
               "development": development, "final": final, "default_service": "A", "report": str(output / "report.md")}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", {"status": "complete", "stage": "complete", "finished_at": time.time()})
    notify_completion(output, summary)


def main(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    args.code_dir = args.code_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()}
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Feedback run directory already has a different configuration.")
    write_json(config_path, config)
    with (args.output_dir / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output_dir / "summary.json").exists():
            notify_completion(args.output_dir, json.loads((args.output_dir / "summary.json").read_text()))
            return
        try:
            execute(args)
        except Exception as exc:
            write_json(args.output_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc), "time": time.time()})
            status_path = args.output_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update(status="failed", error=str(exc), failed_at=time.time())
            write_json(status_path, status)
            notify(args.output_dir, "failed", f"Predictor反馈实验需要处理：{type(exc).__name__}: {exc}。现有A服务与数据保留。")
            raise


if __name__ == "__main__":
    main(build_parser().parse_args())
