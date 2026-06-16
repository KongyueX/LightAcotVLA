"""Run the Stage B Action-CoT pruning sweep and generate summaries/plots."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import shutil
from collections import defaultdict
from typing import Any

import numpy as np

try:
    import eval_action_cot_pruning as pruning_eval
except ImportError:  # pragma: no cover - supports python -m scripts.sweep_action_cot_pruning
    from scripts import eval_action_cot_pruning as pruning_eval

LOGGER = logging.getLogger("sweep_action_cot_pruning")

RESULT_COLUMNS = [
    "segment_mode",
    "chunk_size",
    "prune_ratio",
    "strategy",
    "replacement",
    "success_rate",
    "avg_return",
    "action_l1",
    "action_mse",
    "coarse_l1",
    "coarse_mse",
    "jerk",
    "skip_ratio",
    "avg_inference_time",
]


def _status(message: str) -> None:
    print(f"[sweep_action_cot_pruning] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy_dir", "--entropy-dir", required=True)
    parser.add_argument(
        "--policy.config",
        "--policy-config",
        "--config_name",
        "--config-name",
        dest="config_name",
        default=pruning_eval.DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--policy.dir",
        "--policy-dir",
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        default=None,
    )
    parser.add_argument("--default_prompt", "--default-prompt", default=None)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--max_items", "--max-items", type=int, default=None)
    parser.add_argument("--replacement", choices=("interp", "hold", "zero"), default="interp")
    parser.add_argument("--num_random_trials", "--num-random-trials", type=int, default=5)
    parser.add_argument("--fixed_chunk_sizes", "--fixed-chunk-sizes", nargs="*", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--prune_ratios", "--prune-ratios", nargs="*", type=float, default=[0.2, 0.3, 0.5, 0.7])
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=("low_entropy", "random", "high_entropy"),
        default=["low_entropy", "random", "high_entropy"],
    )
    parser.add_argument("--min_keep_segments", "--min-keep-segments", type=int, default=1)
    parser.add_argument("--min_len", "--min-len", type=int, default=3)
    parser.add_argument("--max_len", "--max-len", type=int, default=6)
    parser.add_argument("--max_segments", "--max-segments", type=int, default=5)
    parser.add_argument("--gripper_indices", "--gripper-indices", nargs="*", type=int, default=None)
    parser.add_argument("--gripper_index", "--gripper-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue_on_error", "--continue-on-error", action="store_true")
    parser.add_argument("--no_expert_actions", "--no-expert-actions", action="store_true")
    parser.add_argument(
        "--enable_action_injection",
        "--enable-action-injection",
        action="store_true",
        help="Very expensive for full sweeps; off by default.",
    )
    parser.add_argument("--require_action_injection", "--require-action-injection", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_oracle", "--skip-oracle", action="store_true")
    parser.add_argument(
        "--oracle_segment_mode",
        "--oracle-segment-mode",
        choices=("fixed", "adaptive"),
        default="adaptive",
    )
    parser.add_argument("--oracle_chunk_size", "--oracle-chunk-size", type=int, default=5)

    # Compatibility pass-throughs for eval-shaped arguments.
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize_size", "--resize-size", type=int, default=224)
    parser.add_argument("--replan_steps", "--replan-steps", type=int, default=5)
    parser.add_argument("--task_suite_name", "--task-suite-name", default="libero_spatial")
    parser.add_argument("--num_steps_wait", "--num-steps-wait", type=int, default=10)
    parser.add_argument("--num_trials_per_task", "--num-trials-per-task", type=int, default=1)
    parser.add_argument("--video_out_path", "--video-out-path", default="./libero_pruning_videos")
    return parser.parse_args()


def _run_name(segment_mode: str, chunk_size: int, prune_ratio: float, strategy: str) -> str:
    ratio = f"{prune_ratio:g}".replace(".", "p")
    if segment_mode == "fixed":
        return f"fixed_L{chunk_size}_p{ratio}_{strategy}"
    return f"adaptive_p{ratio}_{strategy}"


def _eval_arg_list(
    args: argparse.Namespace,
    *,
    segment_mode: str,
    chunk_size: int,
    prune_ratio: float,
    strategy: str,
    output_dir: pathlib.Path,
) -> list[str]:
    argv = [
        "--entropy_dir",
        args.entropy_dir,
        "--policy.config",
        args.config_name,
        "--strategy",
        strategy,
        "--segment_mode",
        segment_mode,
        "--chunk_size",
        str(chunk_size),
        "--prune_ratio",
        str(prune_ratio),
        "--replacement",
        args.replacement,
        "--num_random_trials",
        str(args.num_random_trials),
        "--output_dir",
        str(output_dir),
        "--min_keep_segments",
        str(args.min_keep_segments),
        "--min_len",
        str(args.min_len),
        "--max_len",
        str(args.max_len),
        "--max_segments",
        str(args.max_segments),
        "--seed",
        str(args.seed),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--resize_size",
        str(args.resize_size),
        "--replan_steps",
        str(args.replan_steps),
        "--task_suite_name",
        args.task_suite_name,
        "--num_steps_wait",
        str(args.num_steps_wait),
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--video_out_path",
        args.video_out_path,
    ]
    if args.checkpoint_dir:
        argv.extend(["--policy.dir", args.checkpoint_dir])
    if args.default_prompt:
        argv.extend(["--default_prompt", args.default_prompt])
    if args.max_items is not None:
        argv.extend(["--max_items", str(args.max_items)])
    if args.gripper_indices:
        argv.append("--gripper_indices")
        argv.extend(str(index) for index in args.gripper_indices)
    if args.gripper_index is not None:
        argv.extend(["--gripper_index", str(args.gripper_index)])
    if args.continue_on_error:
        argv.append("--continue_on_error")
    if args.no_expert_actions:
        argv.append("--no_expert_actions")
    if args.enable_action_injection:
        argv.append("--enable_action_injection")
    if args.require_action_injection:
        argv.append("--require_action_injection")
    return argv


def _load_or_run_eval(
    args: argparse.Namespace,
    *,
    segment_mode: str,
    chunk_size: int,
    prune_ratio: float,
    strategy: str,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_args = pruning_eval.build_arg_parser().parse_args(
        _eval_arg_list(
            args,
            segment_mode=segment_mode,
            chunk_size=chunk_size,
            prune_ratio=prune_ratio,
            strategy=strategy,
            output_dir=output_dir,
        )
    )
    return pruning_eval.run_evaluation(eval_args)


def _row_from_metrics(
    *,
    segment_mode: str,
    chunk_size: int,
    prune_ratio: float,
    strategy: str,
    replacement: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    aggregate = metrics.get("aggregate", {})
    return {
        "segment_mode": segment_mode,
        "chunk_size": chunk_size,
        "prune_ratio": prune_ratio,
        "strategy": strategy,
        "replacement": replacement,
        "success_rate": aggregate.get("success_rate", float("nan")),
        "avg_return": aggregate.get("average_return", float("nan")),
        "action_l1": aggregate.get("action_l1_to_full", float("nan")),
        "action_mse": aggregate.get("action_mse_to_full", float("nan")),
        "coarse_l1": aggregate.get("coarse_l1_to_full", float("nan")),
        "coarse_mse": aggregate.get("coarse_mse_to_full", float("nan")),
        "jerk": aggregate.get("trajectory_jerk", float("nan")),
        "skip_ratio": aggregate.get("skip_ratio", float("nan")),
        "avg_inference_time": aggregate.get("avg_inference_time", float("nan")),
    }


def _write_results_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _finite_xy(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for x_value, y_value in points:
        if math.isfinite(x_value) and math.isfinite(y_value):
            xs.append(x_value)
            ys.append(y_value)
    return xs, ys


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _empty_plot(path: pathlib.Path, title: str, message: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_metric_vs_prune(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    path: pathlib.Path,
    include_adaptive: bool = True,
) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["segment_mode"] == "fixed" and int(row["chunk_size"]) != 5:
            continue
        if row["segment_mode"] == "adaptive" and not include_adaptive:
            continue
        label = f"{row['segment_mode']} {row['strategy']}"
        grouped[label].append((_as_float(row["prune_ratio"]), _as_float(row[metric])))

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for label, points in sorted(grouped.items()):
        xs, ys = _finite_xy(sorted(points))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=label)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Prune ratio")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_low_random_high(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    fixed_l5 = [row for row in rows if row["segment_mode"] == "fixed" and int(row["chunk_size"]) == 5]
    _plot_metric_vs_prune(
        fixed_l5,
        metric="action_mse",
        ylabel="Action MSE to full",
        title="Low vs Random vs High Pruning (Fixed L=5)",
        path=path,
        include_adaptive=False,
    )


def _plot_chunk_ablation(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["segment_mode"] != "fixed" or row["strategy"] != "low_entropy":
            continue
        grouped[_as_float(row["prune_ratio"])].append((_as_float(row["chunk_size"]), _as_float(row["action_mse"])))

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for prune_ratio, points in sorted(grouped.items()):
        xs, ys = _finite_xy(sorted(points))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=f"prune={prune_ratio:g}")
        plotted = True
    ax.set_title("Chunk Size Ablation (Low Entropy)")
    ax.set_xlabel("Fixed chunk size")
    ax.set_ylabel("Action MSE to full")
    ax.grid(True, linestyle=":", alpha=0.5)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_adaptive_vs_fixed(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["strategy"] != "low_entropy":
            continue
        if row["segment_mode"] == "fixed" and int(row["chunk_size"]) != 5:
            continue
        label = "adaptive" if row["segment_mode"] == "adaptive" else "fixed L=5"
        grouped[label].append((_as_float(row["prune_ratio"]), _as_float(row["action_mse"])))

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for label, points in sorted(grouped.items()):
        xs, ys = _finite_xy(sorted(points))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=label)
        plotted = True
    ax.set_title("Adaptive vs Fixed L=5 (Low Entropy)")
    ax.set_xlabel("Prune ratio")
    ax.set_ylabel("Action MSE to full")
    ax.grid(True, linestyle=":", alpha=0.5)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_oracle_scatter(oracle_csv: pathlib.Path, path: pathlib.Path, *, max_points: int = 20000) -> None:
    if not oracle_csv.exists():
        _empty_plot(path, "Entropy vs Oracle Importance", "oracle_importance.csv not found")
        return

    entropy_values: list[float] = []
    importance_values: list[float] = []
    with oracle_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entropy = _as_float(row.get("entropy"))
            importance = _as_float(row.get("importance"))
            if math.isfinite(entropy) and math.isfinite(importance):
                entropy_values.append(entropy)
                importance_values.append(importance)

    if not entropy_values:
        _empty_plot(path, "Entropy vs Oracle Importance", "No finite oracle points")
        return

    entropy = np.asarray(entropy_values)
    importance = np.asarray(importance_values)
    if entropy.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(entropy.shape[0], size=max_points, replace=False)
        entropy = entropy[indices]
        importance = importance[indices]

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(entropy, importance, s=5, alpha=0.25)
    ax.set_title("Entropy vs Oracle Importance")
    ax.set_xlabel("Segment entropy")
    ax.set_ylabel("Importance")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _generate_plots(rows: list[dict[str, Any]], oracle_csv: pathlib.Path, plot_dir: pathlib.Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_vs_prune(
        rows,
        metric="success_rate",
        ylabel="Success rate",
        title="Success Rate vs Prune Ratio",
        path=plot_dir / "success_rate_vs_prune_ratio.png",
    )
    _plot_metric_vs_prune(
        rows,
        metric="action_mse",
        ylabel="Action MSE to full",
        title="Action MSE vs Prune Ratio",
        path=plot_dir / "action_mse_vs_prune_ratio.png",
    )
    _plot_low_random_high(rows, plot_dir / "low_random_high_pruning.png")
    _plot_chunk_ablation(rows, plot_dir / "chunk_size_ablation.png")
    _plot_adaptive_vs_fixed(rows, plot_dir / "adaptive_vs_fixed_l5.png")
    _plot_oracle_scatter(oracle_csv, plot_dir / "entropy_vs_oracle_importance.png")


def _copy_oracle_csv(source: pathlib.Path, target: pathlib.Path) -> None:
    if source.resolve() == target.resolve():
        return
    if source.exists():
        shutil.copyfile(source, target)
    else:
        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "segment_id", "start", "end", "entropy", "importance", "importance_metric"],
            )
            writer.writeheader()


def run_sweep(args: argparse.Namespace) -> dict[str, pathlib.Path]:
    output_dir = pathlib.Path(args.output_dir)
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for chunk_size in args.fixed_chunk_sizes:
        for prune_ratio in args.prune_ratios:
            for strategy in args.strategies:
                run_dir = runs_dir / _run_name("fixed", chunk_size, prune_ratio, strategy)
                _status(f"Run fixed L={chunk_size}, prune={prune_ratio:g}, strategy={strategy}")
                metrics = _load_or_run_eval(
                    args,
                    segment_mode="fixed",
                    chunk_size=chunk_size,
                    prune_ratio=prune_ratio,
                    strategy=strategy,
                    output_dir=run_dir,
                )
                rows.append(
                    _row_from_metrics(
                        segment_mode="fixed",
                        chunk_size=chunk_size,
                        prune_ratio=prune_ratio,
                        strategy=strategy,
                        replacement=args.replacement,
                        metrics=metrics,
                    )
                )

    adaptive_chunk_size = 5
    for prune_ratio in args.prune_ratios:
        for strategy in args.strategies:
            run_dir = runs_dir / _run_name("adaptive", adaptive_chunk_size, prune_ratio, strategy)
            _status(f"Run adaptive, prune={prune_ratio:g}, strategy={strategy}")
            metrics = _load_or_run_eval(
                args,
                segment_mode="adaptive",
                chunk_size=adaptive_chunk_size,
                prune_ratio=prune_ratio,
                strategy=strategy,
                output_dir=run_dir,
            )
            rows.append(
                _row_from_metrics(
                    segment_mode="adaptive",
                    chunk_size=adaptive_chunk_size,
                    prune_ratio=prune_ratio,
                    strategy=strategy,
                    replacement=args.replacement,
                    metrics=metrics,
                )
            )

    results_csv = output_dir / "pruning_results.csv"
    _write_results_csv(results_csv, rows)
    _status(f"Wrote {results_csv}")

    oracle_csv = output_dir / "oracle_importance.csv"
    if args.skip_oracle:
        _copy_oracle_csv(pathlib.Path("__missing_oracle__"), oracle_csv)
    else:
        oracle_run_dir = runs_dir / "oracle"
        _status(f"Run oracle importance ({args.oracle_segment_mode})")
        oracle_metrics = _load_or_run_eval(
            args,
            segment_mode=args.oracle_segment_mode,
            chunk_size=args.oracle_chunk_size,
            prune_ratio=0.3,
            strategy="oracle",
            output_dir=oracle_run_dir,
        )
        source = pathlib.Path(
            oracle_metrics.get("outputs", {}).get(
                "oracle_importance_csv",
                oracle_run_dir / "oracle_importance.csv",
            )
        )
        _copy_oracle_csv(source, oracle_csv)
        _status(f"Wrote {oracle_csv}")

    _generate_plots(rows, oracle_csv, output_dir / "plots")
    _status(f"Wrote plots to {output_dir / 'plots'}")
    return {
        "pruning_results_csv": results_csv,
        "oracle_importance_csv": oracle_csv,
        "plots_dir": output_dir / "plots",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run_sweep(_parse_args())


if __name__ == "__main__":
    main()
