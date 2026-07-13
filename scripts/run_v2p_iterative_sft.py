"""Reproducible collect -> SFT -> student rebranch -> aggregate -> SFT loop."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dataset", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-config", default="acot_libero_action_cot_explicit_implicit_co_fusion")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--train-steps-initial", type=int, default=20_000)
    parser.add_argument("--train-steps-per-round", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--teacher-samples", type=int, choices=(10, 20, 32), default=20)
    parser.add_argument("--num-trials-per-task", type=int, default=2)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-roots-per-episode", type=int, default=0)
    parser.add_argument("--root-stride-calls", type=int, default=1)
    parser.add_argument("--student-mode", choices=("v2_distilled", "v2_value_refined"), default="v2_value_refined")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--python", default=sys.executable)
    return parser


def _run(command: list[str], log_path: pathlib.Path, *, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {process.returncode}.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Policy server did not become ready on port {port} within {timeout}s.")


def _train_command(
    args: argparse.Namespace,
    datasets: list[str],
    output_dir: pathlib.Path,
    *,
    train_steps: int,
    resume_params: pathlib.Path | None,
) -> list[str]:
    command = [
        args.python,
        "scripts/train_execution_horizon_predictor.py",
        "--dataset",
        *datasets,
        "--output-dir",
        str(output_dir),
        "--train-steps",
        str(train_steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
    ]
    if resume_params is not None:
        command.extend(["--resume-params", str(resume_params)])
    return command


def main(args: argparse.Namespace) -> None:
    if args.rounds < 0:
        raise ValueError("rounds must be non-negative.")
    output_root = pathlib.Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    datasets = [str(pathlib.Path(path).resolve()) for path in args.initial_dataset]
    training_runs: list[str] = []
    relabel_runs: list[str] = []

    initial_train = output_root / "train_round_00"
    _run(
        _train_command(
            args,
            datasets,
            initial_train,
            train_steps=args.train_steps_initial,
            resume_params=None,
        ),
        initial_train / "run.log",
        env=environment,
    )
    training_runs.append(str(initial_train))
    predictor_params = initial_train / "params"

    for round_index in range(1, args.rounds + 1):
        relabel_dir = output_root / f"counterfactual_round_{round_index:02d}"
        server_log = output_root / f"server_round_{round_index:02d}.log"
        server_command = [
            args.python,
            "scripts/serve_policy.py",
            "--env",
            "libero",
            "--port",
            str(args.port),
            "policy:checkpoint",
            f"--policy.config={args.base_config}",
            f"--policy.dir={args.base_checkpoint}",
            f"--policy.execution-horizon-predictor-params={predictor_params}",
        ]
        with server_log.open("w") as server_file:
            server = subprocess.Popen(
                server_command,
                stdout=server_file,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            try:
                _wait_for_server(server, args.port)
                collector_command = [
                    args.python,
                    "scripts/collect_execution_horizon_counterfactuals.py",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--output-dir",
                    str(relabel_dir),
                    "--continuation-policy",
                    "current_student",
                    "--student-mode",
                    args.student_mode,
                    "--teacher-samples",
                    str(args.teacher_samples),
                    "--num-trials-per-task",
                    str(args.num_trials_per_task),
                    "--task-start",
                    str(args.task_start),
                    "--max-tasks",
                    str(args.max_tasks),
                    "--root-stride-calls",
                    str(args.root_stride_calls),
                    "--max-roots-per-episode",
                    str(args.max_roots_per_episode),
                    "--source-iteration",
                    str(round_index),
                    "--seed",
                    str(args.seed + round_index * 100),
                ]
                _run(collector_command, relabel_dir / "run.log", env=environment)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        datasets.append(str(relabel_dir.resolve()))
        relabel_runs.append(str(relabel_dir))

        train_dir = output_root / f"train_round_{round_index:02d}"
        _run(
            _train_command(
                args,
                datasets,
                train_dir,
                train_steps=args.train_steps_per_round,
                resume_params=predictor_params,
            ),
            train_dir / "run.log",
            env=environment,
        )
        training_runs.append(str(train_dir))
        predictor_params = train_dir / "params"

    summary = {
        "status": "complete",
        "ppo_implemented": False,
        "initial_datasets": list(args.initial_dataset),
        "aggregated_datasets": datasets,
        "training_runs": training_runs,
        "student_relabel_runs": relabel_runs,
        "final_predictor_params": str(predictor_params.resolve()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
