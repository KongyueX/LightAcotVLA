"""Sweep explicit Action-CoT denoising steps for speed and closed-loop quality."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np


DEFAULT_STEPS = [10, 7, 5, 3, 1]
PROFILE_TIMING_FIELDS = (
    "vlm_ms",
    "implicit_action_reasoner_ms",
    "coarse_action_expert_ms",
    "action_expert_ms",
    "profile_overhead_ms",
)


def _status(message: str) -> None:
    print(f"[sweep_action_cot_denoising_steps] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("speed", "closed_loop", "both"), default="speed")
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument(
        "--action_cot_denoising_steps",
        "--action-cot-denoising-steps",
        nargs="*",
        type=int,
        default=DEFAULT_STEPS,
        help="Explicit Action-CoT denoising step values to sweep.",
    )
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
    parser.add_argument(
        "--adaptive_replanning",
        "--adaptive-replanning",
        choices=("none", "action", "entropy", "action_entropy"),
        default="none",
    )
    parser.add_argument("--adaptive_replan_horizons", "--adaptive-replan-horizons", nargs="*", type=int, default=None)
    parser.add_argument(
        "--adaptive_h_selector",
        "--adaptive-h-selector",
        choices=("legacy", "final_aac", "cot_aac", "guarded_cot_aac"),
        default="guarded_cot_aac",
    )
    parser.add_argument(
        "--adaptive_h_entropy_algorithm",
        "--adaptive-h-entropy-algorithm",
        choices=("diagonal_logvar", "aac_grouped"),
        default="diagonal_logvar",
    )
    parser.add_argument("--adaptive_h_coarse_stride", "--adaptive-h-coarse-stride", type=float, default=2.0)
    parser.add_argument("--adaptive_h_jump_mad_scale", "--adaptive-h-jump-mad-scale", type=float, default=1.5)
    parser.add_argument("--adaptive_h_entropy_eps", "--adaptive-h-entropy-eps", type=float, default=1e-6)
    parser.add_argument("--adaptive_h_cov_shrinkage", "--adaptive-h-cov-shrinkage", type=float, default=1e-4)
    parser.add_argument("--adaptive_h_growth_limit", "--adaptive-h-growth-limit", type=int, default=1)
    parser.add_argument("--adaptive_h_low_risk_required", "--adaptive-h-low-risk-required", type=int, default=2)
    parser.add_argument("--adaptive_h_guard_cooldown", "--adaptive-h-guard-cooldown", type=int, default=2)
    parser.add_argument(
        "--adaptive_replan_entropy_mode",
        "--adaptive-replan-entropy-mode",
        choices=("none", "coarse_proxy", "online_mc"),
        default="none",
    )
    parser.add_argument("--adaptive_replan_entropy_samples", "--adaptive-replan-entropy-samples", type=int, default=5)
    parser.add_argument(
        "--adaptive_replan_entropy_low_quantile",
        "--adaptive-replan-entropy-low-quantile",
        type=float,
        default=0.33,
    )
    parser.add_argument(
        "--adaptive_replan_entropy_high_quantile",
        "--adaptive-replan-entropy-high-quantile",
        type=float,
        default=0.67,
    )
    parser.add_argument("--adaptive_replan_entropy_warmup", "--adaptive-replan-entropy-warmup", type=int, default=20)
    parser.add_argument("--adaptive_replan_entropy_low", "--adaptive-replan-entropy-low", type=float, default=None)
    parser.add_argument("--adaptive_replan_entropy_high", "--adaptive-replan-entropy-high", type=float, default=None)
    parser.add_argument("--adaptive_replan_jerk_low", "--adaptive-replan-jerk-low", type=float, default=0.25)
    parser.add_argument("--adaptive_replan_jerk_high", "--adaptive-replan-jerk-high", type=float, default=0.75)
    parser.add_argument(
        "--adaptive_replan_gripper_change_threshold",
        "--adaptive-replan-gripper-change-threshold",
        type=float,
        default=0.25,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.action_cot_denoising_steps:
        raise ValueError("--action_cot_denoising_steps must contain at least one value.")
    if any(step <= 0 for step in args.action_cot_denoising_steps):
        raise ValueError("--action_cot_denoising_steps values must be positive.")
    if args.adaptive_replan_horizons is not None and any(horizon <= 0 for horizon in args.adaptive_replan_horizons):
        raise ValueError("--adaptive_replan_horizons values must be positive.")
    if args.adaptive_h_coarse_stride <= 0:
        raise ValueError("--adaptive_h_coarse_stride must be positive.")
    if args.adaptive_h_jump_mad_scale < 0:
        raise ValueError("--adaptive_h_jump_mad_scale must be non-negative.")
    if args.adaptive_h_entropy_eps <= 0:
        raise ValueError("--adaptive_h_entropy_eps must be positive.")
    if args.adaptive_h_cov_shrinkage <= 0:
        raise ValueError("--adaptive_h_cov_shrinkage must be positive.")
    if args.adaptive_h_growth_limit <= 0:
        raise ValueError("--adaptive_h_growth_limit must be positive.")
    if args.adaptive_h_low_risk_required <= 0:
        raise ValueError("--adaptive_h_low_risk_required must be positive.")
    if args.adaptive_h_guard_cooldown < 0:
        raise ValueError("--adaptive_h_guard_cooldown must be non-negative.")
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
        "--action_cot_denoising_steps",
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
        "--action_cot_denoising_steps",
        str(step),
        "--output_dir",
        str(run_dir),
    ]
    if args.policy_api_key is not None:
        command.extend(["--policy_api_key", args.policy_api_key])
    if args.norm_stats_dir is not None:
        command.extend(["--norm_stats_dir", args.norm_stats_dir])
    if args.adaptive_replanning != "none":
        command.extend(["--adaptive_replanning", args.adaptive_replanning])
        command.extend(["--adaptive_h_selector", args.adaptive_h_selector])
        command.extend(["--adaptive_h_entropy_algorithm", args.adaptive_h_entropy_algorithm])
        command.extend(["--adaptive_h_coarse_stride", str(args.adaptive_h_coarse_stride)])
        command.extend(["--adaptive_h_jump_mad_scale", str(args.adaptive_h_jump_mad_scale)])
        command.extend(["--adaptive_h_entropy_eps", str(args.adaptive_h_entropy_eps)])
        command.extend(["--adaptive_h_cov_shrinkage", str(args.adaptive_h_cov_shrinkage)])
        command.extend(["--adaptive_h_growth_limit", str(args.adaptive_h_growth_limit)])
        command.extend(["--adaptive_h_low_risk_required", str(args.adaptive_h_low_risk_required)])
        command.extend(["--adaptive_h_guard_cooldown", str(args.adaptive_h_guard_cooldown)])
        command.extend(["--adaptive_replan_entropy_mode", args.adaptive_replan_entropy_mode])
        command.extend(["--adaptive_replan_entropy_samples", str(args.adaptive_replan_entropy_samples)])
        command.extend(["--adaptive_replan_entropy_low_quantile", str(args.adaptive_replan_entropy_low_quantile)])
        command.extend(["--adaptive_replan_entropy_high_quantile", str(args.adaptive_replan_entropy_high_quantile)])
        command.extend(["--adaptive_replan_entropy_warmup", str(args.adaptive_replan_entropy_warmup)])
        command.extend(["--adaptive_replan_jerk_low", str(args.adaptive_replan_jerk_low)])
        command.extend(["--adaptive_replan_jerk_high", str(args.adaptive_replan_jerk_high)])
        command.extend(
            [
                "--adaptive_replan_gripper_change_threshold",
                str(args.adaptive_replan_gripper_change_threshold),
            ]
        )
        if args.adaptive_replan_horizons is not None:
            command.append("--adaptive_replan_horizons")
            command.extend(str(horizon) for horizon in args.adaptive_replan_horizons)
        if args.adaptive_replan_entropy_low is not None:
            command.extend(["--adaptive_replan_entropy_low", str(args.adaptive_replan_entropy_low)])
        if args.adaptive_replan_entropy_high is not None:
            command.extend(["--adaptive_replan_entropy_high", str(args.adaptive_replan_entropy_high)])
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
        "action_cot_denoising_steps": step,
        "full_acot_ms_mean": full_ms,
        "full_acot_ms_std": _safe_get(aggregate, "full_acot_ms_std"),
        "speedup_vs_denoising10_pct": speedup,
        "cached_coarse_override_ms_mean": _safe_get(aggregate, "cached_coarse_override_ms_mean"),
        "true_entropy_segment_skip_ms_mean": _safe_get(aggregate, "true_entropy_segment_skip_ms_mean"),
        "full_to_cached_speedup_pct": _safe_get(aggregate, "full_to_cached_speedup_pct"),
        **{f"full_{field}_mean": _safe_get(aggregate, f"full_{field}_mean") for field in PROFILE_TIMING_FIELDS},
        **{
            f"true_entropy_segment_skip_{field}_mean": _safe_get(
                aggregate,
                f"true_entropy_segment_skip_{field}_mean",
            )
            for field in PROFILE_TIMING_FIELDS
        },
    }


def _closed_loop_row(step: int, summary: dict[str, Any], target_mode: str, baseline_ms: float | None) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    mode_metrics = aggregate.get(target_mode, {})
    primary_wall_ms = float(_safe_get(mode_metrics, "primary_wall_inference_ms"))
    if not np.isfinite(primary_wall_ms):
        primary_wall_ms = float(_safe_get(mode_metrics, "avg_deployable_wall_inference_ms"))
    if not np.isfinite(primary_wall_ms):
        primary_wall_ms = float(_safe_get(mode_metrics, "avg_wall_inference_ms"))
    speedup = float("nan") if baseline_ms in (None, 0.0) else (1.0 - primary_wall_ms / baseline_ms) * 100.0
    return {
        "action_cot_denoising_steps": step,
        "mode": target_mode,
        "success_rate": _safe_get(mode_metrics, "success_rate"),
        "average_return": _safe_get(mode_metrics, "average_return"),
        "timeout_rate": _safe_get(mode_metrics, "timeout_rate"),
        "primary_wall_inference_ms": primary_wall_ms,
        "primary_policy_inference_ms": _safe_get(mode_metrics, "primary_policy_inference_ms"),
        "primary_server_inference_ms": _safe_get(mode_metrics, "primary_server_inference_ms"),
        "observed_avg_wall_inference_ms": _safe_get(mode_metrics, "avg_wall_inference_ms"),
        "observed_avg_policy_inference_ms": _safe_get(mode_metrics, "avg_policy_inference_ms"),
        "observed_avg_server_inference_ms": _safe_get(mode_metrics, "avg_server_inference_ms"),
        **{
            f"primary_policy_{field}": _safe_get(mode_metrics, f"avg_deployable_policy_{field}")
            for field in PROFILE_TIMING_FIELDS
        },
        **{
            f"observed_policy_{field}": _safe_get(mode_metrics, f"avg_policy_{field}")
            for field in PROFILE_TIMING_FIELDS
        },
        "avg_total_wall_inference_ms_per_episode": _safe_get(
            mode_metrics,
            "avg_total_wall_inference_ms_per_episode",
        ),
        "avg_total_policy_inference_ms_per_episode": _safe_get(
            mode_metrics,
            "avg_total_policy_inference_ms_per_episode",
        ),
        "avg_total_deployable_wall_inference_ms_per_episode": _safe_get(
            mode_metrics,
            "avg_total_deployable_wall_inference_ms_per_episode",
        ),
        "avg_total_deployable_policy_inference_ms_per_episode": _safe_get(
            mode_metrics,
            "avg_total_deployable_policy_inference_ms_per_episode",
        ),
        "avg_num_replans_per_episode": _safe_get(mode_metrics, "avg_num_replans_per_episode"),
        "avg_total_policy_calls_per_episode": _safe_get(mode_metrics, "avg_total_policy_calls_per_episode"),
        "avg_deployable_policy_calls_per_episode": _safe_get(
            mode_metrics,
            "avg_deployable_policy_calls_per_episode",
        ),
        "avg_entropy_oracle_extra_calls_per_episode": _safe_get(
            mode_metrics,
            "avg_entropy_oracle_extra_calls_per_episode",
        ),
        "avg_replan_horizon": _safe_get(mode_metrics, "avg_replan_horizon"),
        "avg_raw_execution_horizon": _safe_get(mode_metrics, "avg_raw_execution_horizon"),
        "avg_guard_cap": _safe_get(mode_metrics, "avg_guard_cap"),
        "avg_hysteresis_limited": _safe_get(mode_metrics, "avg_hysteresis_limited"),
        "avg_action_cot_denoising_steps_used": _safe_get(
            mode_metrics,
            "avg_action_cot_denoising_steps_used",
        ),
        "speedup_vs_denoising10_pct": speedup,
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


def _step_from_row(row: dict[str, Any]) -> int:
    return int(float(row["action_cot_denoising_steps"]))


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_line(rows: list[dict[str, Any]], *, x_key: str, y_key: str, ylabel: str, title: str, path: pathlib.Path) -> None:
    if not rows:
        return
    points = sorted(
        (
            _step_from_row(row) if x_key == "action_cot_denoising_steps" else int(float(row[x_key])),
            _float(row, y_key),
        )
        for row in rows
    )
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", linewidth=2)
    ax.set_xlabel("Action-CoT denoising steps")
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
    speed_by_step = {_step_from_row(row): _float(row, "full_acot_ms_mean") for row in speed_rows}
    points = []
    for row in closed_rows:
        step = _step_from_row(row)
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
    ax.set_title("Speed-success tradeoff by Action-CoT denoising steps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_combined_report(output_dir: pathlib.Path) -> None:
    speed_rows = _read_csv(output_dir / "speed" / "denoising_steps_speed_summary.csv")
    closed_rows = _read_csv(output_dir / "closed_loop" / "denoising_steps_closed_loop_summary.csv")
    if not speed_rows and not closed_rows:
        return

    speed_by_step = {_step_from_row(row): row for row in speed_rows}
    closed_by_step = {_step_from_row(row): row for row in closed_rows}
    steps = sorted(set(speed_by_step) | set(closed_by_step), reverse=True)
    combined_rows = []
    for step in steps:
        speed = speed_by_step.get(step, {})
        closed = closed_by_step.get(step, {})
        combined_rows.append(
            {
                "action_cot_denoising_steps": step,
                "speed_full_acot_ms_mean": speed.get("full_acot_ms_mean", ""),
                "speedup_vs_denoising10_pct": speed.get(
                    "speedup_vs_denoising10_pct",
                    closed.get("speedup_vs_denoising10_pct", ""),
                ),
                "closed_loop_success_rate": closed.get("success_rate", ""),
                "closed_loop_average_return": closed.get("average_return", ""),
                "closed_loop_timeout_rate": closed.get("timeout_rate", ""),
                "closed_loop_primary_wall_inference_ms": closed.get("primary_wall_inference_ms", ""),
                "closed_loop_primary_policy_inference_ms": closed.get("primary_policy_inference_ms", ""),
                "closed_loop_observed_avg_wall_inference_ms": closed.get("observed_avg_wall_inference_ms", ""),
                "avg_action_cot_denoising_steps_used": closed.get("avg_action_cot_denoising_steps_used", ""),
            }
        )
    _write_csv(output_dir / "denoising_steps_systematic_summary.csv", combined_rows)

    plot_dir = output_dir / "plots"
    _plot_line(
        speed_rows,
        x_key="action_cot_denoising_steps",
        y_key="full_acot_ms_mean",
        ylabel="Open-loop latency (ms)",
        title="Latency vs Action-CoT denoising steps",
        path=plot_dir / "latency_vs_steps.png",
    )
    _plot_line(
        speed_rows,
        x_key="action_cot_denoising_steps",
        y_key="speedup_vs_denoising10_pct",
        ylabel="Speedup vs 10 steps (%)",
        title="Speedup vs Action-CoT denoising steps",
        path=plot_dir / "speedup_vs_steps.png",
    )
    _plot_line(
        closed_rows,
        x_key="action_cot_denoising_steps",
        y_key="success_rate",
        ylabel="Success rate",
        title="Closed-loop success vs Action-CoT denoising steps",
        path=plot_dir / "success_vs_steps.png",
    )
    _plot_tradeoff(speed_rows, closed_rows, plot_dir / "speed_success_tradeoff.png")


def _run_mode(args: argparse.Namespace, mode: str, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[int, dict[str, Any]] = {}
    for step in args.action_cot_denoising_steps:
        run_dir = output_dir / f"denoising_steps_{step}"
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
        rows = [_speed_row(step, summaries[step], baseline_ms) for step in args.action_cot_denoising_steps]
        csv_path = output_dir / "denoising_steps_speed_summary.csv"
    else:
        target_mode = "full" if args.rollout_mode == "all" else args.rollout_mode
        baseline_ms = None
        if baseline_summary is not None:
            baseline_metrics = baseline_summary.get("aggregate", {}).get(target_mode, {})
            baseline_ms = float(
                baseline_metrics.get(
                    "primary_wall_inference_ms",
                    baseline_metrics.get(
                        "avg_deployable_wall_inference_ms",
                        baseline_metrics.get("avg_wall_inference_ms", float("nan")),
                    ),
                )
            )
        rows = [
            _closed_loop_row(step, summaries[step], target_mode, baseline_ms)
            for step in args.action_cot_denoising_steps
        ]
        csv_path = output_dir / "denoising_steps_closed_loop_summary.csv"

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
