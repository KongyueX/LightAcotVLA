"""Replay fixed feedback training updates and select using deployment argmax."""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import pathlib
import sys
import time

import numpy as np
import run_execution_horizon_feedback_experiment as experiment

from openpi.execution_horizon.feedback import FeedbackSelector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--code-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--wait-for-exit", type=pathlib.Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser


def train_command(args: argparse.Namespace, source: dict, variant: str) -> list[str]:
    if source.get("status") != "complete" or source["variant"] != variant:
        raise ValueError("Greedy reselection requires the completed original training run.")
    original_args = source["args"]
    values = dict(original_args, output_dir=str(args.output_dir / f"training_{variant}"),
                  max_updates=int(source["updates"]), patience=1000, selection_metric="greedy")
    command = [args.python, str(args.code_dir / "scripts/train_execution_horizon_feedback.py")]
    for key, value in values.items():
        command += ["--" + key.replace("_", "-"), str(value)]
    return command


def greedy_score(summary: dict) -> tuple[Fraction, float]:
    metrics = summary["best_greedy_validation"]
    fraction = metrics["greedy_success_delta_fraction"]
    return Fraction(fraction["numerator"], fraction["denominator"]), -metrics["greedy_rpc_delta_seconds"]


def same_effective_model(left: pathlib.Path, right: pathlib.Path) -> bool:
    a, b = FeedbackSelector.load(left), FeedbackSelector.load(right)
    return (
        a.variant == b.variant and a.candidates == b.candidates
        and all(np.array_equal(getattr(a, key), getattr(b, key)) for key in (
            "prefix_kernel", "prefix_bias", "feature_mean", "feature_std",
        ))
        and all(np.array_equal(a.params[key], b.params[key]) for key in a.params)
    )


def evaluation_command(args: argparse.Namespace, phase: str, episodes: tuple[int, ...], variant: str) -> list[str]:
    command = experiment.eval_command(args, phase, episodes, [variant])
    command[command.index("--initial-state-bank") + 1] = str(args.source_dir / "initial_state_bank_0_365")
    return command


def finish(args: argparse.Namespace, selection: dict, *, development=None, final=None) -> None:
    result = {
        "status": "complete", "selection": selection, "development": development, "final": final,
        "source_dir": str(args.source_dir), "data_recollected": False, "default_service": "A",
        "training_change": "Same original updates, loss, data, architecture and optimizer; checkpoint selected by greedy validation.",
    }
    experiment.write_json(args.output_dir / "summary.json", result)
    experiment.write_json(args.output_dir / "status.json", {"status": "complete", "stage": "complete"})
    message = "Greedy选模试验结束：" + selection["reason"]
    if final is not None:
        message += "\n" + experiment.result_message("预留测试", final)
    elif development is not None:
        message += "\n" + experiment.result_message("开发闭环", development)
    experiment.notify(args.output_dir, "complete", message + "\nA服务与所有权重保留，不自动替换模型或追加轮次。")


def main(args: argparse.Namespace) -> None:
    args.source_dir = args.source_dir.resolve()
    args.code_dir = args.code_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((args.source_dir / "run_config.json").read_text())
    args.host, args.port = source_config["host"], source_config["port"]
    manifest = {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()}
    path = args.output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Greedy selection output belongs to a different fixed run.")
    experiment.write_json(path, manifest)
    if (args.output_dir / "summary.json").exists():
        return
    training = {}
    for variant in ("current", "history"):
        original = json.loads((args.source_dir / f"training_{variant}/summary.json").read_text())
        experiment.run_stage(args, f"train_{variant}", train_command(args, original, variant))
        training[variant] = json.loads((args.output_dir / f"training_{variant}/summary.json").read_text())
        if training[variant]["updates"] != original["updates"]:
            raise ValueError("Greedy reselection did not replay the complete original update sequence.")
    variant = max(training, key=lambda key: greedy_score(training[key]))
    positive = greedy_score(training[variant]) > (Fraction(0), 0.0)
    unchanged = same_effective_model(
        args.output_dir / f"training_{variant}/checkpoint.npz",
        args.source_dir / f"training_{variant}/checkpoint.npz",
    )
    selection = {
        "variant": variant, "validation_better_than_A": positive, "same_as_previously_evaluated": unchanged,
        "criterion": "Highest root-equal greedy validation success delta, then lower RPC delta; no development look-ahead.",
        "models": {key: {name: value[name] for name in (
            "best_step", "updates", "selection_metric", "best_greedy_validation", "residual_changed",
        )} for key, value in training.items()},
        "reason": "greedy validation selected one changed candidate",
    }
    if not positive:
        selection["reason"] = "No greedy validation improvement over A; keep A."
    elif unchanged:
        selection["reason"] = "Selected checkpoint is identical to the already evaluated candidate; reuse its result and keep A."
    experiment.write_json(args.output_dir / "selection.json", selection)
    if not positive or unchanged:
        finish(args, selection)
        return
    experiment.notify(args.output_dir, "selected", (
        f"Greedy选模完成：{variant}，best step {training[variant]['best_step']}；复用原数据与固定训练轨迹。"
        "单点H诊断完成后，串行进行候选与A的100初态开发闭环，不并发污染计时。"
    ))
    experiment.write_json(args.output_dir / "status.json", {
        "status": "waiting", "stage": "wait_for_probe", "wait_for_exit": str(args.wait_for_exit),
    })
    while not args.wait_for_exit.exists():
        time.sleep(15)
    if args.wait_for_exit.read_text().strip() != "0":
        experiment.write_json(args.output_dir / "status.json", {"status": "failed", "stage": "wait_for_probe"})
        raise RuntimeError("The diagnostic probe failed; leave evaluation queued for diagnosis.")
    experiment.run_stage(args, "development", evaluation_command(args, "development", experiment.DEV_EPISODES, variant))
    development = experiment.summarize_eval(args.output_dir / "development", experiment.DEV_EPISODES, [variant])
    winner = experiment.select_development_candidate(development)
    final = None
    if winner is not None:
        experiment.notify(args.output_dir, "development_complete", experiment.result_message("Greedy选模开发闭环完成，候选进入预留测试", development))
        experiment.run_stage(args, "final", evaluation_command(args, "final", experiment.FINAL_EPISODES, variant))
        final = experiment.summarize_eval(args.output_dir / "final", experiment.FINAL_EPISODES, [variant])
        selection["reason"] = "Development winner evaluated once on the reserved 200 initial states."
    else:
        selection["reason"] = "Greedy-selected candidate did not beat A in development; keep A."
    finish(args, selection, development=development, final=final)


if __name__ == "__main__":
    main(build_parser().parse_args())
