"""Fit fresh and masked mid-chunk monitors from paired continuation outcomes."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.execution_horizon.midchunk import FEATURE_DIM
from openpi.execution_horizon.midchunk import MidchunkMonitor

UPDATES = 400
LEARNING_RATE = 1e-3
SEED = 7
BATCH_SIZE = 64
LOG_EVERY = 25
TIME_BONUS = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_split(directory: pathlib.Path, *, episode_ids: range) -> dict[str, Any]:
    paths = sorted(directory.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No closed mid-chunk roots in {directory}.")
    records = []
    identities = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            record = {key: archive[key] for key in (
                "raw_feature", "trial_success", "trial_rpc", "trial_valid", "task_id", "episode_id", "root_step",
            )}
        feature = np.asarray(record["raw_feature"], dtype=np.float32)
        success = np.asarray(record["trial_success"], dtype=np.float64)
        rpc = np.asarray(record["trial_rpc"], dtype=np.float64)
        valid = np.asarray(record["trial_valid"], dtype=bool)
        if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
            raise ValueError(f"Expected finite raw_feature[{FEATURE_DIM}] in {path}.")
        if success.shape != (2, 3) or rpc.shape != success.shape or valid.shape != success.shape:
            raise ValueError(f"Expected continue/replan by three paired repeats in {path}.")
        paired = valid[0] & valid[1]
        if not paired.any():
            raise ValueError(f"Root has no valid paired outcomes: {path}.")
        if (
            not np.all(np.isin(success[:, paired], [0.0, 1.0]))
            or not np.all(np.isfinite(rpc[:, paired])) or np.any(rpc[:, paired] < 0)
        ):
            raise ValueError(f"Paired outcomes require binary success and nonnegative RPC seconds: {path}.")
        task, episode, step = (int(record[key]) for key in ("task_id", "episode_id", "root_step"))
        if task not in range(10) or episode not in episode_ids:
            raise ValueError(f"Root {(task, episode)} is outside the fixed split {directory}.")
        identity = (task, episode)
        if identity in identities:
            raise ValueError(f"Only one mid-chunk root per source episode is supported: {identity}.")
        identities.append(identity)
        continue_cost = rpc[0, paired]
        replan_cost = rpc[1, paired]
        scale = np.maximum(1.0, np.maximum(continue_cost, replan_cost))
        replan_reward = success[1, paired] * (1.0 + TIME_BONUS * (continue_cost - replan_cost) / scale)
        target = float(np.mean(replan_reward - success[0, paired]))
        records.append({
            "feature": feature, "success": success[:, paired], "rpc": rpc[:, paired], "target": target,
            "task_id": task, "episode_id": episode, "root_step": step, "path": str(path.resolve()),
        })
    return {
        "records": records,
        "features": np.stack([record["feature"] for record in records]),
        "targets": np.asarray([record["target"] for record in records], dtype=np.float32),
        "identities": np.asarray([(record["task_id"], record["episode_id"], record["root_step"]) for record in records]),
        "summary": {
            "directory": str(directory.resolve()), "roots": len(records),
            "roots_by_task": {str(task): sum(record["task_id"] == task for record in records) for task in range(10)},
            "episode_groups": [list(identity) for identity in identities],
            "paired_repeats": sum(record["success"].shape[1] for record in records),
            "target_min": float(min(record["target"] for record in records)),
            "target_max": float(max(record["target"] for record in records)),
            "target_mean": float(np.mean([record["target"] for record in records])),
        },
    }


def branch_metrics(scores: np.ndarray, split: dict[str, Any], *, threshold: float = 0.0) -> dict[str, Any]:
    actions = np.asarray(scores > threshold, dtype=np.int32)
    success_fractions = []
    costs = []
    success_count = trial_count = rescues = regressions = 0
    for action, record in zip(actions, split["records"], strict=True):
        success = record["success"][action].astype(bool)
        reference = record["success"][0].astype(bool)
        successes, trials = int(success.sum()), len(success)
        success_fractions.append(Fraction(successes, trials))
        costs.append(float(record["rpc"][action].mean()))
        success_count += successes
        trial_count += trials
        rescues += int(np.sum(success & ~reference))
        regressions += int(np.sum(~success & reference))
    fraction = sum(success_fractions, Fraction(0)) / len(success_fractions)
    return {
        "roots": len(actions), "success_count": success_count, "paired_trials": trial_count,
        "root_equal_success": float(fraction),
        "success_fraction": {"numerator": fraction.numerator, "denominator": fraction.denominator},
        "root_equal_rpc_seconds": float(np.mean(costs)),
        "rescues_vs_continue": rescues, "regressions_vs_continue": regressions,
        "threshold": float(threshold), "replan_roots": int(actions.sum()), "replan_rate": float(actions.mean()),
        "selected_actions": actions.tolist(),
    }


def _selection_score(metrics: dict[str, Any]) -> tuple[Fraction, float]:
    fraction = metrics["success_fraction"]
    return Fraction(fraction["numerator"], fraction["denominator"]), -metrics["root_equal_rpc_seconds"]


def _predict(monitor: MidchunkMonitor, features: np.ndarray) -> np.ndarray:
    scores = np.asarray([monitor.forward(feature) for feature in features], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Mid-chunk monitor produced nonfinite gains.")
    return scores


def _copy_params(params: dict[str, Any]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in params.items()}


def train_variant(
    variant: str, training: dict[str, Any], early: dict[str, Any], mean: np.ndarray, std: np.ndarray, output: pathlib.Path,
) -> tuple[MidchunkMonitor, dict[str, Any], dict[str, np.ndarray]]:
    monitor = MidchunkMonitor.initialize(variant=variant, seed=SEED, threshold=0.0)
    monitor.feature_mean, monitor.feature_std = mean.copy(), std.copy()
    initial_params = _copy_params(monitor.params)
    output.mkdir()
    train_features = monitor.normalized_features(training["features"])
    initial_train_scores = _predict(monitor, training["features"])
    initial_early_scores = _predict(monitor, early["features"])
    initial_metrics = branch_metrics(initial_early_scores, early)
    best_metrics = initial_metrics
    best_params = _copy_params(initial_params)
    best_step = 0
    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(SEED)
    started = time.monotonic()
    with jax.default_device(cpu):
        params = {name: jnp.asarray(value) for name, value in initial_params.items()}
        optimizer = optax.adam(LEARNING_RATE)
        optimizer_state = optimizer.init(params)
        features = jnp.asarray(train_features, dtype=jnp.float32)
        targets = jnp.asarray(training["targets"], dtype=jnp.float32)

        def loss(parameters, batch_features, batch_targets):
            hidden = jnp.tanh(batch_features @ parameters["hidden_w"] + parameters["hidden_b"])
            prediction = (hidden @ parameters["output_w"] + parameters["output_b"])[..., 0]
            return jnp.mean(jnp.square(prediction - batch_targets))

        @jax.jit
        def update(parameters, state, batch_features, batch_targets):
            objective, gradients = jax.value_and_grad(loss)(parameters, batch_features, batch_targets)
            changes, state = optimizer.update(gradients, state, parameters)
            return optax.apply_updates(parameters, changes), state, objective

        initial_log = {
            "step": 0, "train_mse": float(np.mean((initial_train_scores - training["targets"]) ** 2)),
            "early_mse": float(np.mean((initial_early_scores - early["targets"]) ** 2)),
            "early": initial_metrics, "best_step": 0,
        }
        with (output / "training_log.jsonl").open("w") as log:
            log.write(json.dumps(initial_log, sort_keys=True) + "\n")
            log.flush()
            for step in range(1, UPDATES + 1):
                indices = rng.choice(len(training["records"]), size=min(BATCH_SIZE, len(training["records"])), replace=False)
                params, optimizer_state, objective = update(params, optimizer_state, features[indices], targets[indices])
                if not np.isfinite(float(objective)):
                    raise ValueError(f"Nonfinite mid-chunk loss at {variant} update {step}.")
                if step % LOG_EVERY and step != UPDATES:
                    continue
                monitor.params = _copy_params(params)
                train_scores = _predict(monitor, training["features"])
                early_scores = _predict(monitor, early["features"])
                metrics = branch_metrics(early_scores, early)
                if _selection_score(metrics) > _selection_score(best_metrics):
                    best_metrics, best_step, best_params = metrics, step, _copy_params(params)
                row = {
                    "step": step, "train_mse": float(np.mean((train_scores - training["targets"]) ** 2)),
                    "early_mse": float(np.mean((early_scores - early["targets"]) ** 2)),
                    "early": metrics, "best_step": best_step,
                }
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps({"variant": variant, **row}, sort_keys=True), flush=True)
    last_params = _copy_params(params)
    monitor.params = best_params
    selected_train_scores = _predict(monitor, training["features"])
    selected_early_scores = _predict(monitor, early["features"])
    monitor.metadata.update({
        "training_objective": "mse_success_gated_bounded_relative_rpc_gain",
        "updates": UPDATES, "best_step": best_step, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "normalization_fit": "common_raw_training_features_only",
        "checkpoint_selection": "root-equal greedy early success, then lower RPC seconds; threshold0; step0 included",
    })
    summary = {
        "status": "complete", "variant": variant, "updates": UPDATES, "best_step": best_step,
        "initial_early": initial_metrics, "best_early_before_calibration": best_metrics,
        "last_early": metrics, "selected_train_mse": float(np.mean((selected_train_scores - training["targets"]) ** 2)),
        "selected_early_mse": float(np.mean((selected_early_scores - early["targets"]) ** 2)),
        "selected_parameters_changed": any(not np.array_equal(initial_params[name], best_params[name]) for name in initial_params),
        "last_parameters_changed": any(not np.array_equal(initial_params[name], last_params[name]) for name in initial_params),
        "training_seconds": time.monotonic() - started,
    }
    predictions = {
        "train_initial_gain": initial_train_scores, "early_initial_gain": initial_early_scores,
        "train_gain": selected_train_scores, "early_gain": selected_early_scores,
        "train_last_gain": train_scores, "early_last_gain": early_scores,
        "train_target": training["targets"], "early_target": early["targets"],
        "train_identities": training["identities"], "early_identities": early["identities"],
    }
    return monitor, summary, predictions


def match_trigger_rate(scores: np.ndarray, target_count: int) -> dict[str, Any]:
    """Choose one score-order threshold; no outcomes enter this calibration."""
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.all(np.isfinite(scores)) or not 0 <= target_count <= len(scores):
        raise ValueError("Calibration needs finite scores and a feasible target trigger count.")
    if target_count == 0:
        threshold = float(scores.max())
    elif target_count == len(scores):
        threshold = float(np.nextafter(scores.min(), -np.inf))
    else:
        pivot = float(np.sort(scores)[::-1][target_count])
        count_above, count_including_tie = int(np.sum(scores > pivot)), int(np.sum(scores >= pivot))
        threshold = pivot if abs(count_above - target_count) <= abs(count_including_tie - target_count) else float(np.nextafter(pivot, -np.inf))
    actual_count = int(np.sum(scores > threshold))
    return {
        "method": "single_early_score_order_statistic; nearest attainable count; tie favors fewer replans",
        "threshold": threshold, "target_count": target_count, "actual_count": actual_count,
        "roots": len(scores), "target_rate": target_count / len(scores), "actual_rate": actual_count / len(scores),
        "exact_match": actual_count == target_count, "outcomes_used": False,
        "scope": "Early-root trigger matching only; deployment trigger rates and RPC costs may differ.",
    }


def main(args: argparse.Namespace) -> None:
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory for the fixed mid-chunk training run.")
    training = load_split(data_dir / "train", episode_ids=range(300, 308))
    early = load_split(data_dir / "early", episode_ids=range(330, 332))
    normalization = MidchunkMonitor.initialize(variant="fresh", seed=SEED)
    normalization.fit_normalization(training["features"])
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "data_dir": str(data_dir), "updates_per_variant": UPDATES, "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "seed": SEED, "log_every": LOG_EVERY,
        "target": "mean_paired[S_replan * (1 + .02*(C_continue-C_replan)/max(1 second,C_continue,C_replan)) - S_continue]",
        "time_bonus": TIME_BONUS, "critic": False, "normalization": "same train-only mean/std; masked tail16 cleared after normalization",
        "training": training["summary"], "early": early["summary"],
    }
    _write_json(output / "run_config.json", contract)
    trained = {}
    for variant in ("fresh", "masked"):
        trained[variant] = train_variant(
            variant, training, early, normalization.feature_mean, normalization.feature_std, output / variant,
        )
    fresh_model, _, fresh_predictions = trained["fresh"]
    target_count = int(np.sum(fresh_predictions["early_gain"] > fresh_model.threshold))
    masked_model, _, masked_predictions = trained["masked"]
    calibration = match_trigger_rate(masked_predictions["early_gain"], target_count)
    masked_model.threshold = calibration["threshold"]
    masked_model.metadata["trigger_calibration"] = calibration
    summaries = {}
    for variant, (monitor, summary, predictions) in trained.items():
        checkpoint = output / variant / "checkpoint.npz"
        monitor.save(checkpoint)
        predictions["train_selected_action"] = (predictions["train_gain"] > monitor.threshold).astype(np.int32)
        predictions["early_selected_action"] = (predictions["early_gain"] > monitor.threshold).astype(np.int32)
        np.savez_compressed(output / variant / "predictions.npz", **predictions)
        summary.update({
            "checkpoint": str(checkpoint), "threshold": monitor.threshold,
            "early": branch_metrics(predictions["early_gain"], early, threshold=monitor.threshold),
            "calibration": calibration if variant == "masked" else {"method": "fixed zero threshold", "outcomes_used": False},
        })
        _write_json(output / variant / "summary.json", summary)
        summaries[variant] = summary
    report = {
        "status": "complete", **contract, "variants": summaries, "trigger_calibration": calibration,
        "semantics": "Early cached branch outcomes are diagnostics, not closed-loop model performance. Step0 selection means no learned improvement passed this early criterion.",
    }
    _write_json(output / "summary.json", report)
    print(json.dumps({"status": "complete", "models": {
        variant: {key: value[key] for key in ("best_step", "threshold", "early", "selected_parameters_changed")}
        for variant, value in summaries.items()
    }}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
