"""Run the hierarchical H25 closed-loop pilot only after both offline gates pass."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import socket
import subprocess
import sys
import time
from typing import Any

from openpi.execution_horizon import initial_states


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pilot-output-dir", required=True)
    parser.add_argument("--policy-config", default="acot_libero_long_chunk_h25")
    parser.add_argument("--policy-server-tmux", default="h25_dynamic_pilot_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--server-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-trials-per-task", type=int, default=20)
    parser.add_argument("--require-initial-state-isolation", action="store_true")
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _complete_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"Artifact is not complete: {path}")
    return payload


def _tmux_present(name: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _stop_tmux(name: str) -> None:
    if _tmux_present(name):
        subprocess.run(["tmux", "kill-session", "-t", name], check=True)


def _wait_for_server(host: str, port: int, tmux_name: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _tmux_present(tmux_name):
            raise RuntimeError(f"Policy server tmux {tmux_name!r} exited before becoming ready.")
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(5.0)
    raise TimeoutError(f"Policy server did not become ready at {host}:{port} within {timeout_seconds}s.")


def _notify(message: str, output_dir: pathlib.Path, code_dir: pathlib.Path) -> None:
    if not os.environ.get("FEISHU_WEBHOOK_URL") or not os.environ.get("FEISHU_SIGNING_SECRET"):
        return
    completed = subprocess.run(
        [
            sys.executable,
            str(code_dir / "watch_experiment_feishu.py"),
            "--output-dir",
            str(output_dir),
            "--test-message",
            message,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode:
        print(
            f"Warning: Feishu notification exited with code {completed.returncode}; experiment continues.",
            file=sys.stderr,
        )


def _find_pilot_summary(pilot_output_dir: pathlib.Path) -> pathlib.Path:
    summaries = sorted(pilot_output_dir.rglob("summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected exactly one pilot summary under {pilot_output_dir}, found {summaries}.")
    return summaries[0]


def _verify_pilot_summary(summary: dict[str, Any], expected_episodes: int) -> None:
    if int(summary.get("num_tasks", -1)) != 10:
        raise ValueError("Dynamic pilot summary does not contain ten tasks.")
    if int(summary.get("num_trials_per_task", -1)) * 10 != expected_episodes:
        raise ValueError("Dynamic pilot summary trial count does not match the requested pilot.")
    overall = summary.get("overall")
    if isinstance(overall, dict):
        hierarchical = overall.get("hierarchical_transformer")
    elif isinstance(overall, list):
        matching = [row for row in overall if row.get("mode") == "hierarchical_transformer"]
        hierarchical = matching[0] if len(matching) == 1 else None
    else:
        hierarchical = None
    if not isinstance(hierarchical, dict) or int(hierarchical.get("episodes", -1)) != expected_episodes:
        raise ValueError("Dynamic pilot summary does not contain the expected hierarchical episodes.")


def main(args: argparse.Namespace) -> None:
    if args.poll_seconds <= 0 or args.server_timeout_seconds <= 0:
        raise ValueError("Poll and server timeout values must be positive.")
    if args.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive.")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must lie in [1, 65535].")

    post_summary_path = pathlib.Path(args.post_summary).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).resolve()
    pilot_output_dir = pathlib.Path(args.pilot_output_dir).resolve()
    code_dir = pathlib.Path(__file__).resolve().parent
    status_path = output_dir / "status.json"
    result_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        print(json.dumps(_complete_json(result_path), indent=2, sort_keys=True))
        return
    _write_json(status_path, {"status": "waiting_for_dual_offline_gate", "post_summary": str(post_summary_path)})
    while not post_summary_path.exists():
        time.sleep(args.poll_seconds)
    post_summary = _complete_json(post_summary_path)
    if not bool(post_summary.get("dual_official_gate", False)):
        result = {
            "status": "complete",
            "pilot_started": False,
            "reason": "dual_official_gate_not_met",
            "post_summary": str(post_summary_path),
        }
        _write_json(result_path, result)
        _write_json(status_path, {"status": "offline_gate_not_met", "pilot_started": False})
        _notify(
            "【H25 predictor门控】seed7未同时通过development与fresh-final的预注册95%门, 动态10x20 pilot未启动; 将按held-out失败模式继续定向relabel/DAgger。",
            output_dir,
            code_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    for name in ("development_official", "final_official"):
        audit = post_summary.get(name)
        if not isinstance(audit, dict) or not bool(audit.get("offline_engineering_gate", False)):
            raise ValueError(f"dual_official_gate conflicts with {name} audit.")
    if args.require_initial_state_isolation:
        isolation = post_summary.get("initial_state_isolation") or {}
        if isolation.get("status") != "complete" or isolation.get("pairwise_initial_state_overlap") != 0:
            raise ValueError("Dynamic pilot requires verified initial-state isolation.")
        bank = initial_states.InitialStateBank(isolation["initial_state_bank"])
        if bank.sha256 != isolation.get("initial_state_bank_sha256"):
            raise ValueError("Initial-state bank changed after the offline audit.")
    predictor_dir = pathlib.Path(post_summary["predictor_dir"]).resolve()
    calibration_path = predictor_dir / "calibration.json"
    predictor_summary = _complete_json(predictor_dir / "summary.json")
    if not predictor_summary.get("predictor_config", {}).get("paired_advantage_heads", False):
        raise ValueError("Pilot requires the paired-head Transformer predictor.")
    if not checkpoint_dir.exists() or not calibration_path.exists():
        raise FileNotFoundError(f"Missing checkpoint or calibration: {checkpoint_dir}, {calibration_path}")
    pilot_output_dir.mkdir(parents=True, exist_ok=True)
    existing_summaries = sorted(pilot_output_dir.rglob("summary.json"))
    if len(existing_summaries) > 1:
        raise RuntimeError(f"Multiple pilot summaries found under {pilot_output_dir}: {existing_summaries}")
    expected_episodes = 10 * args.num_trials_per_task
    if existing_summaries:
        pilot_summary = _complete_json(existing_summaries[0])
        _verify_pilot_summary(pilot_summary, expected_episodes)
        result = {
            "status": "complete",
            "pilot_started": True,
            "post_summary": str(post_summary_path),
            "predictor_dir": str(predictor_dir),
            "calibration_path": str(calibration_path),
            "pilot_summary": str(existing_summaries[0]),
            "pilot_summary_status": pilot_summary["status"],
            "expected_episodes": expected_episodes,
            "resumed_completed_pilot": True,
        }
        _write_json(result_path, result)
        _write_json(status_path, {"status": "complete", "pilot_summary": str(existing_summaries[0])})
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    resume_pilot = (pilot_output_dir / "run_config.json").exists()
    if any(pilot_output_dir.iterdir()) and not resume_pilot:
        raise FileExistsError(f"Pilot output is non-empty but has no resume signature: {pilot_output_dir}")

    _write_json(
        status_path,
        {
            "status": "starting_dynamic_pilot_server",
            "predictor_dir": str(predictor_dir),
            "pilot_output_dir": str(pilot_output_dir),
        },
    )
    _notify(
        "【H25 predictor门控】seed7已同时通过development与fresh-final的预注册95%长H门, 开始动态10 tasks x 20闭环pilot。",
        output_dir,
        code_dir,
    )
    if _tmux_present(args.policy_server_tmux):
        raise RuntimeError(f"Scoped pilot server tmux already exists: {args.policy_server_tmux}")
    server_log = output_dir / "policy_server.log"
    server_command = [
        sys.executable,
        str(code_dir / "serve_policy.py"),
        "--env",
        "LIBERO",
        "--port",
        str(args.port),
        "policy:checkpoint",
        "--policy.config",
        args.policy_config,
        "--policy.dir",
        str(checkpoint_dir),
        "--policy.execution-horizon-predictor-params",
        str(predictor_dir),
    ]
    shell_command = f"{shlex.join(server_command)} >{shlex.quote(str(server_log))} 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", args.policy_server_tmux, "-c", str(code_dir.parent), shell_command],
        check=True,
    )
    try:
        _wait_for_server(args.host, args.port, args.policy_server_tmux, args.server_timeout_seconds)
        _write_json(
            status_path,
            {
                "status": "dynamic_pilot_running",
                "predictor_dir": str(predictor_dir),
                "pilot_output_dir": str(pilot_output_dir),
            },
        )
        eval_command = [
            sys.executable,
            str(code_dir / "eval_libero_execution_horizon.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--output-dir",
            str(pilot_output_dir),
            "--modes",
            "hierarchical_transformer",
            "--hierarchical-calibration-json",
            str(calibration_path),
            "--model-action-horizon",
            "25",
            "--task-suite-name",
            "libero_10",
            "--max-tasks",
            "10",
            "--num-trials-per-task",
            str(args.num_trials_per_task),
            "--seed",
            str(args.seed),
            "--action-cot-denoising-steps",
            "10",
            "--teacher-samples",
            "20",
        ]
        if resume_pilot:
            eval_command.append("--resume")
        with (output_dir / "dynamic_pilot.log").open("w", encoding="utf-8") as log:
            subprocess.run(eval_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        _stop_tmux(args.policy_server_tmux)

    pilot_summary_path = _find_pilot_summary(pilot_output_dir)
    pilot_summary = _complete_json(pilot_summary_path)
    _verify_pilot_summary(pilot_summary, expected_episodes)
    result = {
        "status": "complete",
        "pilot_started": True,
        "post_summary": str(post_summary_path),
        "predictor_dir": str(predictor_dir),
        "calibration_path": str(calibration_path),
        "pilot_summary": str(pilot_summary_path),
        "pilot_summary_status": pilot_summary["status"],
        "expected_episodes": expected_episodes,
    }
    _write_json(result_path, result)
    _write_json(status_path, {"status": "complete", "pilot_summary": str(pilot_summary_path)})
    _notify(
        f"【H25 predictor完成】通过双重95%门控的分层Transformer已完成动态10x{args.num_trials_per_task} pilot; 下一步审计success、实际elapsed及H分布。",
        output_dir,
        code_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
