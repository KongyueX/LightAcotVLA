"""Fit temperature and conformal calibration on the held-out predictor split."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import hierarchical
from openpi.models import model as model_lib
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictor
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictorConfig


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    predictor_dir: str
    output_json: str | None = None
    split_manifest: str | None = None
    params: str | None = None
    batch_size: int = 256
    confidence_level: float = 0.95
    ood_probability_threshold: float = 0.95
    seed: int = 7


_BASE_INPUT_FIELDS = (
    "prefix_feature",
    "state",
    "coarse_actions",
    "final_actions",
    "previous_actions",
    "previous_h",
    "budget_balance",
    "episode_progress",
    "previous_valid",
)


def _load_config(path: pathlib.Path) -> ExecutionHorizonPredictorConfig:
    values = json.loads(path.read_text())
    values["candidate_horizons"] = tuple(values["candidate_horizons"])
    return ExecutionHorizonPredictorConfig(**values)


def _restore(config: ExecutionHorizonPredictorConfig, params_path: pathlib.Path, seed: int):
    module = ExecutionHorizonPredictor(config, rngs=nnx.Rngs(seed))
    loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(params_path, dtype=jnp.float32))
    if "execution_horizon_predictor" in loaded:
        loaded = loaded["execution_horizon_predictor"]
    graphdef, state = nnx.split(module)
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def _split_indices(arrays: dict[str, np.ndarray], manifest_path: pathlib.Path, split_name: str) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text())
    selected_groups = np.asarray(manifest[f"{split_name}_group_ids"], dtype=np.uint64)
    if not selected_groups.size:
        raise ValueError(f"split_manifest contains no {split_name} groups.")
    groups = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    groups += np.asarray(arrays["episode_id"], dtype=np.uint64)
    if split_name == "train" and manifest.get("bootstrap_episode_groups"):
        multiplicities = {int(group): int(count) for group, count in manifest["bootstrap_train_group_counts"].items()}
        repeated = [
            np.flatnonzero(groups == np.uint64(group)) for group, count in multiplicities.items() for _ in range(count)
        ]
        if not repeated:
            raise ValueError("Bootstrap split manifest contains no sampled train groups.")
        return np.concatenate(repeated)
    indices = np.flatnonzero(np.isin(groups, selected_groups))
    if not indices.size:
        raise ValueError(f"No dataset roots match the {split_name} groups in split_manifest.")
    return indices


def _predict(
    module: ExecutionHorizonPredictor,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    @jax.jit
    def infer(batch):
        return module(**batch)

    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = {name: jnp.asarray(arrays[name][selected]) for name in _BASE_INPUT_FIELDS}
        if module.config.visual_num_queries:
            batch["prefix_tokens"] = jnp.asarray(arrays["prefix_tokens"][selected])
            batch["prefix_mask"] = jnp.asarray(arrays["prefix_token_mask"][selected])
        outputs = jax.device_get(infer(batch))
        for name, value in outputs.items():
            pieces.setdefault(name, []).append(np.asarray(value))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _expected_calibration_error(
    probabilities: np.ndarray,
    success_rate: np.ndarray,
    trial_count: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape((-1,))
    success_rate = np.asarray(success_rate, dtype=np.float64).reshape((-1,))
    weights = np.asarray(trial_count, dtype=np.float64).reshape((-1,))
    valid = (weights > 0) & np.isfinite(probabilities) & np.isfinite(success_rate)
    probabilities = probabilities[valid]
    success_rate = success_rate[valid]
    weights = weights[valid]
    total_weight = np.sum(weights)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1] if index + 1 < bins else probabilities <= 1.0
        )
        bin_weight = np.sum(weights[selected])
        if bin_weight <= 0:
            continue
        confidence = np.average(probabilities[selected], weights=weights[selected])
        accuracy = np.average(success_rate[selected], weights=weights[selected])
        error += bin_weight / max(total_weight, 1.0) * abs(confidence - accuracy)
    return float(error)


def main(args: Args) -> None:
    predictor_dir = pathlib.Path(args.predictor_dir).resolve()
    config = _load_config(predictor_dir / "predictor_config.json")
    if config.temporal_backbone != "transformer":
        raise ValueError("Hierarchical calibration requires a transformer predictor.")
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    split_path = (
        pathlib.Path(args.split_manifest).resolve()
        if args.split_manifest is not None
        else predictor_dir / "split_manifest.json"
    )
    indices = _split_indices(arrays, split_path, "calibration")
    train_indices = _split_indices(arrays, split_path, "train")
    params_path = pathlib.Path(args.params).resolve() if args.params is not None else predictor_dir / "params"
    module = _restore(config, params_path, args.seed)
    predictions = _predict(module, arrays, indices, args.batch_size)
    training_predictions = _predict(module, arrays, train_indices, args.batch_size)
    labels = {
        name: arrays[name][indices]
        for name in (
            "success_count",
            "trial_count",
            "elapsed_mean",
            "event_mask",
            "risk_valid",
            "hazard_event_count",
            "hazard_at_risk_count",
            "dangerous_long_count",
            "paired_trial_count",
        )
    }
    calibration = hierarchical.fit_calibration(
        predictions,
        labels,
        candidate_horizons=config.candidate_horizons,
        reference_horizon=config.reference_horizon,
        confidence_level=args.confidence_level,
        training_features=training_predictions["temporal_feature"],
        ood_probability_threshold=args.ood_probability_threshold,
    )
    output_path = (
        pathlib.Path(args.output_json).resolve() if args.output_json is not None else predictor_dir / "calibration.json"
    )
    calibration.save(output_path)

    trials = np.asarray(labels["trial_count"], dtype=np.float64)
    success_rate = np.asarray(labels["success_count"], dtype=np.float64) / np.maximum(trials, 1.0)
    uncalibrated_probability = _sigmoid(predictions["success_logits"])
    calibrated_probability = _sigmoid(predictions["success_logits"] / calibration.success_temperature)
    valid = trials > 0
    uncalibrated_brier = float(np.mean(np.square(uncalibrated_probability[valid] - success_rate[valid])))
    calibrated_brier = float(np.mean(np.square(calibrated_probability[valid] - success_rate[valid])))
    uncalibrated_ece = _expected_calibration_error(uncalibrated_probability, success_rate, trials)
    calibrated_ece = _expected_calibration_error(calibrated_probability, success_rate, trials)
    reference_index = config.candidate_horizons.index(config.reference_horizon)
    long_indices = [
        index for index, horizon in enumerate(config.candidate_horizons) if horizon > config.reference_horizon
    ]
    success_target = success_rate[:, long_indices] - success_rate[:, reference_index : reference_index + 1]
    elapsed = np.asarray(labels["elapsed_mean"], dtype=np.float64)
    elapsed_target = elapsed[:, long_indices] - elapsed[:, reference_index : reference_index + 1]
    success_lcb = np.asarray(predictions["success_advantage"]) - np.asarray(calibration.success_residual_quantiles)[
        None, :
    ] * np.asarray(predictions["success_advantage_std"])
    elapsed_ucb = np.asarray(predictions["elapsed_advantage"]) + np.asarray(calibration.elapsed_residual_quantiles)[
        None, :
    ] * np.asarray(predictions["elapsed_advantage_std"])
    decisions = []
    for index in range(indices.size):
        row = {name: value[index] for name, value in predictions.items()}
        decisions.append(hierarchical.select_horizon(row, calibration=calibration))
    selected_long_positions = [
        calibration.long_horizons.index(decision.selected_horizon)
        if decision.selected_horizon in calibration.long_horizons
        else -1
        for decision in decisions
    ]
    selected_long = np.asarray(selected_long_positions) >= 0
    dangerous = np.asarray(labels["dangerous_long_count"], dtype=np.float64)[:, long_indices]
    paired = np.asarray(labels["paired_trial_count"], dtype=np.float64)[:, long_indices]
    dangerous_total = 0.0
    paired_total = 0.0
    for root_index, long_position in enumerate(selected_long_positions):
        if long_position >= 0:
            dangerous_total += dangerous[root_index, long_position]
            paired_total += paired[root_index, long_position]
    report = {
        "status": "complete",
        "calibration_path": str(output_path),
        "num_calibration_roots": int(indices.size),
        "calibration": dataclasses.asdict(calibration),
        "success_brier_before_temperature": uncalibrated_brier,
        "success_brier_after_temperature": calibrated_brier,
        "success_ece_before_temperature": uncalibrated_ece,
        "success_ece_after_temperature": calibrated_ece,
        "success_lcb_empirical_coverage_by_long_h": np.mean(success_target >= success_lcb, axis=0).tolist(),
        "elapsed_ucb_empirical_coverage_by_long_h": np.mean(elapsed_target <= elapsed_ucb, axis=0).tolist(),
        "ood_calibration_distance_95th_percentile": float(
            np.quantile(np.asarray(calibration.ood_calibration_distances), 0.95)
        ),
        "calibrated_long_h_coverage": float(np.mean(selected_long)),
        "calibrated_false_long_rate": float(dangerous_total / max(paired_total, 1.0)),
        "calibrated_false_long_paired_trials": int(paired_total),
        "calibrated_ood_fallback_rate": float(
            np.mean([decision.ood_probability >= calibration.ood_probability_threshold for decision in decisions])
        ),
    }
    report_path = output_path.with_name(output_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
