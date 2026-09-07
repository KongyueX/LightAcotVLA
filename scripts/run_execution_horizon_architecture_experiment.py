"""Run A, conditioned-query, and expert-hidden matched architecture comparisons."""
# ruff: noqa: RUF001, SLF001

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

import numpy as np
import run_execution_horizon_feedback_experiment as feedback

MODES = {"A": "ordered_transformer", "visual_query": "ordered_visual_query", "expert_hidden": "ordered_expert_hidden"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--a-predictor-dir", type=pathlib.Path, required=True)
    parser.add_argument("--policy-dir", type=pathlib.Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8050)
    return parser


def live_command(pid: int) -> list[str] | None:
    path = pathlib.Path(f"/proc/{pid}/cmdline")
    return path.read_bytes().rstrip(b"\0").decode().split("\0") if path.exists() else None


def server_command(args, name: str, port: int):
    predictor = args.a_predictor_dir if name == "A" else args.output_dir / f"training_{name}"
    return [
        args.python, str(args.code_dir / "scripts/serve_policy.py"), "--port", str(port),
        "policy:checkpoint", "--policy.config", "acot_libero_long_chunk_h25",
        "--policy.dir", str(args.policy_dir), "--policy.execution-horizon-predictor-params", str(predictor),
    ]


def ensure_server(args, name: str, port: int):
    command = server_command(args, name, port)
    state_path = args.output_dir / f"server_{name}.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else None
    pid = previous["pid"] if previous else None
    if pid and live_command(pid) is not None:
        if live_command(pid) != command:
            raise RuntimeError(f"Recorded {name} PID now belongs to another command.")
    else:
        with (args.output_dir / f"server_{name}.log").open("a") as log:
            process = subprocess.Popen(
                command, cwd=args.code_dir, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false"),
                start_new_session=True,
            )
        pid = process.pid
        feedback.write_json(state_path, {"pid": pid, "command": command, "port": port, "started_at": time.time()})
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if live_command(pid) is None:
            raise RuntimeError(f"{name} server exited; see server_{name}.log.")
        try:
            with socket.create_connection((args.host, port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"{name} server did not listen on {port} within ten minutes.")


def stop_servers(args):
    for name in MODES:
        path = args.output_dir / f"server_{name}.json"
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if live_command(state["pid"]) == state["command"]:
            os.killpg(state["pid"], signal.SIGTERM)
            feedback.write_json(args.output_dir / f"server_{name}.stopped.json", {"pid": state["pid"], "stopped_at": time.time()})


def require_idle_stage(output: pathlib.Path):
    status_path = output / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("child_pid") and live_command(previous["child_pid"]) == previous.get("command"):
        raise RuntimeError(f"Previous stage is still running in PID {previous['child_pid']}.")


def run_stage(args, name: str, command: list[str], *, gpu: bool = False):
    output = args.output_dir
    status_path = output / "status.json"
    exit_path = output / f"{name}.exit"
    if exit_path.exists() and exit_path.read_text().strip() == "0":
        return
    require_idle_stage(output)
    if name.startswith("train_"):
        destination = pathlib.Path(command[command.index("--output-dir") + 1])
        summary_path = destination / "summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "complete":
            exit_path.write_text("0\n")
            return
        if destination.exists() and any(destination.iterdir()):
            destination.rename(destination.with_name(f"{destination.name}.interrupted_{time.time_ns()}"))
    feedback.write_json(output / f"{name}.command.json", command)
    with (output / f"{name}.log").open("a") as log:
        process = subprocess.Popen(
            command, cwd=args.code_dir, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, JAX_PLATFORMS="cuda" if gpu else "cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        )
        feedback.write_json(status_path, {
            "stage": name, "status": "running", "child_pid": process.pid, "command": command, "started_at": time.time(),
        })
        returncode = process.wait()
    exit_path.write_text(f"{returncode}\n")
    if returncode:
        raise RuntimeError(f"{name} exited {returncode}; see {output / (name + '.log')}")


def evaluation_command(args, phase: str, variants: list[str]):
    episodes = feedback.DEV_EPISODES if phase == "development" else feedback.FINAL_EPISODES
    command = [
        args.python, str(args.code_dir / "scripts/eval_libero_execution_horizon.py"),
        "--host", args.host, "--port", str(args.port), "--output-dir", str(args.output_dir / phase),
        "--initial-state-bank", str(args.source_dir / "initial_state_bank_0_365"),
        "--episode-ids", *map(str, episodes), "--modes", MODES["A"], *(MODES[name] for name in variants),
        "--interleave-modes", "--record-ordered-diagnostics", "--model-action-horizon", "25",
        "--seed", "7", "--max-tasks", "10", "--action-cot-denoising-steps", "10",
        "--final-denoising-steps", "10", "--num-steps-wait", "10", "--resize-size", "224",
    ]
    if "visual_query" in variants:
        command += ["--visual-query-port", str(args.port + 1)]
    if "expert_hidden" in variants:
        command += ["--expert-hidden-port", str(args.port + 2)]
    if (args.output_dir / phase / "run_config.json").exists():
        command.append("--resume")
    return command


def summarize(directory: pathlib.Path, episodes, variants):
    import audit_fixed_h_paired as paired

    path = directory / "paired_analysis.json"
    if path.exists():
        return json.loads(path.read_text())
    if json.loads((directory / "summary.json").read_text()).get("status") != "complete":
        raise ValueError("Cannot summarize an incomplete architecture evaluation.")
    runs = {MODES[name]: {} for name in ("A", *variants)}
    with (directory / "rollout_rows.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["task_id"]), int(row["episode"]))
            if row["mode"] not in runs or key in runs[row["mode"]]:
                raise ValueError("Unexpected or duplicate architecture evaluation row.")
            if not all(np.isfinite(float(row[field])) for field in paired._METRICS):
                raise ValueError("Incomplete architecture timing metrics.")
            runs[row["mode"]][key] = row
    keys = paired._validate_pairing(runs)
    if set(keys) != {(task, episode) for task in range(10) for episode in episodes}:
        raise ValueError("Architecture evaluation must cover every fixed task/episode.")
    result = {
        "status": "complete", "runs": {mode: paired._run_summary(rows, keys) for mode, rows in runs.items()},
        "pairwise": {MODES[name]: paired._pairwise_audit(
            runs[MODES["A"]], runs[MODES[name]], keys, samples=5000, seed=7, noninferiority_margin=.01,
        ) for name in variants},
        "semantics": "Same frozen VLA and external observations; episode-interleaved endpoints, one active RPC at a time. Adapter and hidden-capture costs are included in policy/RPC/full-episode timing. No architecture cache exported during evaluation.",
    }
    feedback.write_json(path, result)
    return result


def winner(analysis):
    def score(mode):
        return analysis["runs"][mode]["success_count"], -analysis["runs"][mode]["means"]["policy_rpc_wall_total_ms"]
    selected = max(analysis["runs"], key=score)
    if selected == MODES["A"] or score(selected) <= score(MODES["A"]):
        return None
    return next(name for name, mode in MODES.items() if mode == selected)


def execute(args):
    try:
        feedback.write_json(args.output_dir / "status.json", {"status": "preparing", "stage": "server_A"})
        ensure_server(args, "A", args.port)
        feedback.notify(args.output_dir, "started", "三组架构实验开始：A、状态/任务条件视觉query、动作专家hidden。冻结VLA与A原参数，在同一真实调用采集特征及配对分支标签；所有三组完整开发闭环都会测试。")
        for split in ("train", "early_stop"):
            run_stage(args, f"features_{split}", [
                args.python, str(args.code_dir / "scripts/collect_execution_horizon_architecture_features.py"),
                "--source-dir", str(args.source_dir / split), "--output-dir", str(args.output_dir / f"features_{split}"),
                "--host", args.host, "--port", str(args.port),
                "--fresh-paired",
            ])
        feedback.notify(args.output_dir, "features_complete", "360个root的特征与9000条配对分支结果采集完成，输入和标签来自相同真实调用。开始两个独立零增量模块训练，各固定650次更新，step0参与部署greedy选模。")
        for variant in ("visual_query", "expert_hidden"):
            run_stage(args, f"train_{variant}", [
                args.python, str(args.code_dir / "scripts/train_execution_horizon_architecture.py"),
                "--train-dir", str(args.output_dir / "features_train"),
                "--validation-dir", str(args.output_dir / "features_early_stop"),
                "--a-predictor-dir", str(args.a_predictor_dir), "--output-dir", str(args.output_dir / f"training_{variant}"),
                "--variant", variant, "--seed", "7", "--learning-rate", "0.0001", "--batch-size", "64",
                "--max-updates", "650", "--log-every", "25",
            ], gpu=True)
        ensure_server(args, "visual_query", args.port + 1)
        ensure_server(args, "expert_hidden", args.port + 2)
        feedback.notify(args.output_dir, "training_complete", "两个架构模块训练完成，开始A/query/hidden三模式同100初态交错测试，共300局；三端点只有一个RPC在执行，不并发评测。训练loss不作为收益结论。")
        variants = ["visual_query", "expert_hidden"]
        run_stage(args, "development", evaluation_command(args, "development", variants))
        development = summarize(args.output_dir / "development", feedback.DEV_EPISODES, variants)
        selected = winner(development)
        feedback.notify(args.output_dir, "development_complete", feedback.result_message("三组架构开发闭环完成", development) + (f"\n{selected}进入唯一预留200初态对A测试。" if selected else "\n无胜出候选，保留A，不运行预留测试。"))
        final = None
        if selected:
            run_stage(args, "final", evaluation_command(args, "final", [selected]))
            final = summarize(args.output_dir / "final", feedback.FINAL_EPISODES, [selected])
        result = {
            "status": "complete", "development": development, "development_winner": selected, "final": final,
            "training": {name: json.loads((args.output_dir / f"training_{name}/summary.json").read_text()) for name in variants},
            "source_dir": str(args.source_dir), "default_service": "A", "counterfactual_branches_recollected": 9000,
        }
        feedback.write_json(args.output_dir / "summary.json", result)
        feedback.write_json(args.output_dir / "status.json", {"status": "complete", "stage": "complete"})
        feedback.notify(args.output_dir, "complete", feedback.result_message("架构实验完成", final or development) + "\n原A服务和全部权重/数据保留；本批不追加组合模型、扫参或训练轮数。")
        stop_servers(args)
    except Exception as exc:
        feedback.write_json(args.output_dir / "failure.json", {"error": str(exc), "error_type": type(exc).__name__, "time": time.time()})
        feedback.notify(args.output_dir, "failure", "架构实验暂停于实际错误：" + str(exc) + "\n保留现有数据与进程记录，等待修复后从完成阶段继续。")
        raise


def main(args):
    for name in ("code_dir", "source_dir", "output_dir", "a_predictor_dir", "policy_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {name: str(value) if isinstance(value, pathlib.Path) else value for name, value in vars(args).items()}
        path = args.output_dir / "run_config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Architecture run configuration changed.")
        require_idle_stage(args.output_dir)
        feedback.write_json(path, config)
        if (args.output_dir / "summary.json").exists():
            stop_servers(args)
            return
        execute(args)


if __name__ == "__main__":
    main(build_parser().parse_args())
