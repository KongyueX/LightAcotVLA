"""Benchmark actual end-to-end latency of the shared-prefix batched MC teacher."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np
from openpi_client import websocket_client_policy as websocket_policy

import eval_libero_action_cot_pruning as libero_eval


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(statistics.mean(finite)) if finite else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-api-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--samples", nargs="+", type=int, default=[10, 20, 32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--sequential-comparison", action="store_true")
    return parser


def _infer(client, request: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    result = client.infer(request)
    return result, (time.perf_counter() - started) * 1000.0


def main(args: argparse.Namespace) -> None:
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup must be non-negative.")
    if any(sample_count not in (10, 20, 32) for sample_count in args.samples):
        raise ValueError("Every --samples value must be one of 10, 20, or 32.")
    client = websocket_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        api_key=args.policy_api_key,
        ping_interval=None,
        ping_timeout=None,
    )
    task_suite = libero_eval.benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    states = task_suite.get_task_init_states(args.task_id)
    env, task_description = libero_eval._get_libero_env(
        task, libero_eval.LIBERO_ENV_RESOLUTION, args.seed
    )
    try:
        env.reset()
        observation = env.set_init_state(states[args.episode % len(states)])
        for _ in range(10):
            observation, _, done, _ = env.step(libero_eval.LIBERO_DUMMY_ACTION)
            if done:
                break
        element = libero_eval._observation_to_policy_input(
            observation, task_description, args.resize_size
        )
    finally:
        libero_eval._safe_close_env(env)

    rows = []
    for sample_count in args.samples:
        measurements: list[dict[str, float]] = []
        for repeat in range(args.warmup + args.repeats):
            request = {
                **element,
                "policy_seed": np.asarray(args.seed + repeat * 1000, dtype=np.int64),
                "profile_policy_timing": np.asarray(True),
                "batched_mc_samples": np.asarray(sample_count, dtype=np.int32),
                "action_cot_denoising_steps": np.asarray(
                    args.action_cot_denoising_steps, dtype=np.int32
                ),
            }
            result, wall_ms = _infer(client, request)
            if repeat < args.warmup:
                continue
            policy_timing = result.get("policy_timing", {})
            server_timing = result.get("server_timing", {})
            measurements.append(
                {
                    "wall_ms": wall_ms,
                    "policy_ms": float(policy_timing.get("infer_ms", np.nan)),
                    "server_ms": float(server_timing.get("infer_ms", np.nan)),
                    "batched_teacher_ms": float(policy_timing.get("batched_mc_teacher_ms", np.nan)),
                    "predictor_ms": float(
                        policy_timing.get("execution_horizon_predictor_ms", np.nan)
                    ),
                }
            )
        row = {
            "samples": sample_count,
            "repeats": args.repeats,
            **{
                f"actual_{field}": _mean([measurement[field] for measurement in measurements])
                for field in measurements[0]
            },
            "measurements": measurements,
        }
        if args.sequential_comparison:
            sequential_wall = []
            sequential_policy = []
            for repeat in range(args.repeats):
                wall_sum = 0.0
                policy_sum = 0.0
                for sample_index in range(sample_count):
                    request = {
                        **element,
                        "policy_seed": np.asarray(
                            args.seed + repeat * 1000 + sample_index, dtype=np.int64
                        ),
                        "profile_policy_timing": np.asarray(True),
                        "action_cot_denoising_steps": np.asarray(
                            args.action_cot_denoising_steps, dtype=np.int32
                        ),
                    }
                    result, wall_ms = _infer(client, request)
                    wall_sum += wall_ms
                    policy_sum += float(result.get("policy_timing", {}).get("infer_ms", np.nan))
                sequential_wall.append(wall_sum)
                sequential_policy.append(policy_sum)
            row["actual_sequential_wall_ms"] = _mean(sequential_wall)
            row["actual_sequential_policy_ms"] = _mean(sequential_policy)
            row["actual_wall_speedup"] = row["actual_sequential_wall_ms"] / row["actual_wall_ms"]
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete",
        "timing_semantics": "actual client wall, server and synchronized policy latency; no potential timing",
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(build_parser().parse_args())
