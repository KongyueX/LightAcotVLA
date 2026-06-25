"""Sweep explicit Action-CoT denoising steps for speed and closed-loop quality."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from typing import Any


DEFAULT_STEPS = [10, 7, 5, 3, 1]


def _status(message: str) -> None:
    print(f"[sweep_action_cot_coarse_steps] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("speed", "closed_loop", "both"), default="speed")
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--coarse_steps", "--coarse-steps", nargs="*", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_plots", "--no-plots", action="store_true")

    parser.add_argument("--entropy_dir", "--entropy-dir", default=None)
    parser.add_argument("--policy.config", "--policy-config", dest="config_name", default=None)
    parser.add_argument("--policy.dir", "--policy-dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--default_prompt", "--default-prompt", default=None)
    parser.add_argument("--num_steps", "--num-steps", type=int, default=10)

    parser.add_argument("--max_items", "--max-items", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--strategy", choices=("low_entropy", "high_entropy", "random"), default="low_entropy")
    parser.add_argument("--segment_mode", "--segment-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=5)
    parser.add_argument("--prune_ratio", "--prune-ratio", type=float, default=0.3)
    parser.add_argument("--replacement", choices=("interp", "hold", "zero"), default="interp")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy_api_key", "--policy-api-key", default=None)
    parser.add_argument("--task_suite_name", "--task-suite-name", default="libero_spatial")
    parser.add_argument("--max_tasks", "--max-tasks", type=int, default=10)
    parser.add_argument("--task_start", "--task-start", type=int, default=0)
    parser.add_argument("--num_trials_per_task", "--num-trials-per-task", type=int, default=1)
    parser.add_argument("--rollout_mode", "--rollout-mode", default="full")
    parser.add_argument("--entropy_samples", "--entropy-samples", type=int, default=4)
    parser.add_argument("--norm_stats_dir", "--norm-stats-dir", default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.coarse_steps:
        raise ValueError("--coarse_steps must contain at least one value.")
    if any(step <= 0 for step in args.coarse_steps):
        raise ValueError("--coarse_steps values must be positive.")
    if args.mode in ("speed", "both"):
        if args.entropy_dir is None:
            raise ValueError("--entropy_dir is required for --mode speed or --mode both.")
        if args.config_name is None:
            raise ValueError("--policy.config is required for --mode speed or --mode both.")
        if args.checkpoint_dir is None:
            raise ValueError("--policy.dir is required for --mode speed or --mode both.")


def _run(command: list[str]) -> None:
    _status(" ".join(command))
    subprocess.run(command, check=True)


def _add_common_policy_args(command: list[str], args: argparse.Namespace) -> None:
    if args.config_name is not None:
        command.extend(["--policy.config", args.config_name])
    if args.checkpoint_dir is not None:
        command.extend(["--policy.dir", args.checkpoint_dir])
    if args.default_prompt is not None:
        command.extend(["--default_prompt", args.default_prompt])


def _speed_command(args: argparse.Namespace, step: int, run_dir: pathlib.Path) -> list[str]:
    script = pathlib.Path(__file__).with_name("benchmark_action_cot_speed.py")
    command = [
        sys.executable,
        str(script),
        "--entropy_dir",
        str(args.entropy_dir),
        "--output_dir",
        str(run_dir),
        "--num_steps",
        str(args.num_steps),
        "--coarse_num_steps",
        str(step),
        "--max_items",
        str(args.max_items),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--strategy",
        args.strategy,
        "--segment_mode",
        args.segment_mode,
        "--chunk_size",
        str(args.chunk_size),
        "--prune_ratio",
        str(args.prune_ratio),
        "--replacement",
        args.replacement,
        "--seed",
        str(args.seed),
    ]
    _add_common_policy_args(command, args)
    return command


def _closed_loop_command(args: argparse.Namespace, step: int, run_dir: pathlib.Path) -> list[str]:
    script = pathlib.Path(__file__).with_name("eval_libero_action_cot_pruning.py")
    command = [
        sys.executable,
        str(script),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--task_suite_name",
        args.task_suite_name,
        "--max_tasks",
        str(args.max_tasks),
        "--task_start",
        str(args.task_start),
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--mode",
        args.rollout_mode,
        "--entropy_samples",
        str(args.entropy_samples),
        "--strategy",
        args.strategy,
        "--segment_mode",
        args.segment_mode,
        "--chunk_size",
        str(args.chunk_size),
        "--prune_ratio",
        str(args.prune_ratio),
        "--replacement",
        args.replacement,
        "--coarse_num_steps",
        str(step),
        "--output_dir",
        str(run_dir),
    ]
    if args.policy_api_key is not None:
        command.extend(["--policy_api_key", args.policy_api_key])
    if args.norm_stats_dir is not None:
        command.extend(["--norm_stats_dir", args.norm_stats_dir])
    return command


def _load_summary(run_dir: pathlib.Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _safe_get(mapping: dict[str, Any], key: str) -> Any:
    return mapping.get(key, float("nan"))


def _speed_row(step: int, summary: dict[str, Any], baseline_ms: float | None) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    full_ms = float(_safe_get(aggregate, "full_acot_ms_mean"))
    speedup = float("nan") if baseline_ms in (None, 0.0) else (1.0 - full_ms / baseline_ms) * 100.0
    return {
        "coarse_num_steps": step,
        "full_acot_ms_mean": full_ms,
        "full_acot_ms_std": _safe_get(aggregate, "full_acot_ms_std"),
        "speedup_vs_coarse10_pct": speedup,
        "cached_coarse_override_ms_mean": _safe_get(aggregate, "cached_coarse_override_ms_mean"),
        "true_entropy_segment_skip_ms_mean": _safe_get(aggregate, "true_entropy_segment_skip_ms_mean"),
        "full_to_cached_speedup_pct": _safe_get(aggregate, "full_to_cached_speedup_pct"),
    }


def _closed_loop_row(step: int, summary: dict[str, Any], target_mode: str, baseline_ms: float | None) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    mode_metrics = aggregate.get(target_mode, {})
    wall_ms = float(_safe_get(mode_metrics, "avg_wall_inference_ms"))
    speedup = float("nan") if baseline_ms in (None, 0.0) else (1.0 - wall_ms / baseline_ms) * 100.0
    return {
        "coarse_num_steps": step,
        "mode": target_mode,
        "success_rate": _safe_get(mode_metrics, "success_rate"),
        "average_return": _safe_get(mode_metrics, "average_return"),
        "timeout_rate": _safe_get(mode_metrics, "timeout_rate"),
        "avg_wall_inference_ms": wall_ms,
        "avg_policy_inference_ms": _safe_get(mode_metrics, "avg_policy_inference_ms"),
        "avg_server_inference_ms": _safe_get(mode_metrics, "avg_server_inference_ms"),
        "avg_coarse_num_steps_used": _safe_get(mode_metrics, "avg_coarse_num_steps_used"),
        "speedup_vs_coarse10_pct": speedup,
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_line(rows: list[dict[str, Any]], *, x_key: str, y_key: str, ylabel: str, title: str, path: pathlib.Path) -> None:
    if not rows:
        return
    points = sorted((int(float(row[x_key])), _float(row, y_key)) for row in rows)
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", linewidth=2)
    ax.set_xlabel("Coarse denoising steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tradeoff(speed_rows: list[dict[str, Any]], closed_rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    if not speed_rows or not closed_rows:
        return
    speed_by_step = {int(float(row["coarse_num_steps"])): _float(row, "full_acot_ms_mean") for row in speed_rows}
    points = []
    for row in closed_rows:
        step = int(float(row["coarse_num_steps"]))
        if step in speed_by_step:
            points.append((step, speed_by_step[step], _float(row, "success_rate")))
    if not points:
        return

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for step, latency, success in sorted(points):
        ax.scatter(latency, success, s=70)
        ax.annotate(str(step), (latency, success), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Open-loop policy latency (ms)")
    ax.set_ylabel("Closed-loop success rate")
    ax.set_title("Speed-success tradeoff by coarse denoising steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_combined_report(output_dir: pathlib.Path) -> None:
    speed_rows = _read_csv(output_dir / "speed" / "coarse_steps_speed_summary.csv")
    closed_rows = _read_csv(output_dir / "closed_loop" / "coarse_steps_closed_loop_summary.csv")
    if not speed_rows and not closed_rows:
        return

    speed_by_step = {int(float(row["coarse_num_steps"])): row for row in speed_rows}
    closed_by_step = {int(float(row["coarse_num_steps"])): row for row in closed_rows}
    steps = sorted(set(speed_by_step) | set(closed_by_step), reverse=True)
    combined_rows = []
    for step in steps:
        speed = speed_by_step.get(step, {})
        closed = closed_by_step.get(step, {})
        combined_rows.append(
            {
                "coarse_num_steps": step,
                "speed_full_acot_ms_mean": speed.get("full_acot_ms_mean", ""),
                "speedup_vs_coarse10_pct": speed.get("speedup_vs_coarse10_pct", closed.get("speedup_vs_coarse10_pct", "")),
                "closed_loop_success_rate": closed.get("success_rate", ""),
                "closed_loop_average_return": closed.get("average_return", ""),
                "closed_loop_timeout_rate": closed.get("timeout_rate", ""),
                "closed_loop_avg_wall_inference_ms": closed.get("avg_wall_inference_ms", ""),
                "avg_coarse_num_steps_used": closed.get("avg_coarse_num_steps_used", ""),
            }
        )
    _write_csv(output_dir / "coarse_steps_systematic_summary.csv", combined_rows)

    plot_dir = output_dir / "plots"
    _plot_line(
        speed_rows,
        x_key="coarse_num_steps",
        y_key="full_acot_ms_mean",
        ylabel="Open-loop latency (ms)",
        title="Latency vs coarse denoising steps",
        path=plot_dir / "latency_vs_steps.png",
    )
    _plot_line(
        speed_rows,
        x_key="coarse_num_steps",
        y_key="speedup_vs_coarse10_pct",
        ylabel="Speedup vs 10 steps (%)",
        title="Speedup vs coarse denoising steps",
        path=plot_dir / "speedup_vs_steps.png",
    )
    _plot_line(
        closed_rows,
        x_key="coarse_num_steps",
        y_key="success_rate",
        ylabel="Success rate",
        title="Closed-loop success vs coarse denoising steps",
        path=plot_dir / "success_vs_steps.png",
    )
    _plot_tradeoff(speed_rows, closed_rows, plot_dir / "speed_success_tradeoff.png")


def _run_mode(args: argparse.Namespace, mode: str, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[int, dict[str, Any]] = {}
    for step in args.coarse_steps:
        run_dir = output_dir / f"coarse_steps_{step}"
        if mode == "speed":
            command = _speed_command(args, step, run_dir)
        else:
            command = _closed_loop_command(args, step, run_dir)

        if not (run_dir / "summary.json").exists() or args.overwrite:
            _run(command)
        else:
            _status(f"Reuse existing {run_dir / 'summary.json'}")
        summaries[step] = _load_summary(run_dir)

    baseline_summary = summaries.get(10)
    if mode == "speed":
        baseline_ms = None
        if baseline_summary is not None:
            baseline_ms = float(baseline_summary.get("aggregate", {}).get("full_acot_ms_mean", float("nan")))
        rows = [_speed_row(step, summaries[step], baseline_ms) for step in args.coarse_steps]
        csv_path = output_dir / "coarse_steps_speed_summary.csv"
    else:
        target_mode = "full" if args.rollout_mode == "all" else args.rollout_mode
        baseline_ms = None
        if baseline_summary is not None:
            baseline_ms = float(
                baseline_summary.get("aggregate", {}).get(target_mode, {}).get("avg_wall_inference_ms", float("nan"))
            )
        rows = [_closed_loop_row(step, summaries[step], target_mode, baseline_ms) for step in args.coarse_steps]
        csv_path = output_dir / "coarse_steps_closed_loop_summary.csv"

    _write_csv(csv_path, rows)
    _status(f"Wrote {csv_path}")
    return csv_path


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "both":
        _run_mode(args, "speed", output_dir / "speed")
        _run_mode(args, "closed_loop", output_dir / "closed_loop")
        if not args.no_plots:
            _write_combined_report(output_dir)
            _status(f"Wrote combined report to {output_dir}")
        return

    _run_mode(args, args.mode, output_dir)
    if not args.no_plots:
        report_dir = output_dir.parent if output_dir.name in ("speed", "closed_loop") else output_dir
        _write_combined_report(report_dir)


if __name__ == "__main__":
    main()
