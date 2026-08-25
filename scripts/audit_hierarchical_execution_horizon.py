"""Offline audit for a calibrated hierarchical Transformer sidecar.

This consumes held-out multi-seed counterfactual roots. It does not replace the
paired closed-loop LIBERO pilot or formal evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

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
    calibration_json: str | None = None
    output_json: str | None = None
    split_manifest: str | None = None
    split_name: str | None = None
    batch_size: int = 256
    seed: int = 7
    bootstrap_samples: int = 5_000
    success_noninferiority_margin: float = 0.01
    maximum_short_event_probability: float = 0.20
    false_long_upper_bound: float = 0.05


_INPUT_FIELDS = (
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


def _indices(arrays: dict[str, np.ndarray], args: Args, predictor_dir: pathlib.Path) -> np.ndarray:
    if args.split_name is None:
        return np.arange(len(arrays["task_id"]), dtype=np.int64)
    if args.split_name not in {"train", "validation", "calibration"}:
        raise ValueError("split_name must be train, validation, calibration, or omitted.")
    manifest_path = (
        pathlib.Path(args.split_manifest).resolve()
        if args.split_manifest is not None
        else predictor_dir / "split_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    selected_groups = np.asarray(manifest[f"{args.split_name}_group_ids"], dtype=np.uint64)
    groups = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    groups += np.asarray(arrays["episode_id"], dtype=np.uint64)
    indices = np.flatnonzero(np.isin(groups, selected_groups))
    if not indices.size:
        raise ValueError(f"No roots matched split {args.split_name!r}.")
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
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        batch = {name: jnp.asarray(arrays[name][selected]) for name in _INPUT_FIELDS}
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


def _distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _cluster_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.uint64)
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    if not values.size:
        return {"mean": None, "ci95": [None, None], "num_roots": 0, "num_clusters": 0}
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    unique_groups = np.unique(groups)
    cluster_means = np.asarray(
        [np.mean(values[groups == group]) for group in unique_groups],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    means = np.empty((samples,), dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        sampled = rng.integers(0, cluster_means.size, size=(count, cluster_means.size))
        means[start : start + count] = np.mean(cluster_means[sampled], axis=1)
    return {
        "mean": float(np.mean(cluster_means)),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "num_roots": int(values.size),
        "num_clusters": int(cluster_means.size),
    }


def _wilson_upper(successes: float, trials: float, z: float = 1.96) -> float:
    if trials <= 0:
        return 1.0
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    margin = z / denominator * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
    return float(min(1.0, center + margin))


def main(args: Args) -> None:
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0 <= args.false_long_upper_bound <= 1:
        raise ValueError("false_long_upper_bound must lie in [0, 1].")
    predictor_dir = pathlib.Path(args.predictor_dir).resolve()
    config = _load_config(predictor_dir / "predictor_config.json")
    if config.temporal_backbone != "transformer":
        raise ValueError("This audit requires a hierarchical Transformer predictor.")
    calibration_path = (
        pathlib.Path(args.calibration_json).resolve()
        if args.calibration_json is not None
        else predictor_dir / "calibration.json"
    )
    calibration = hierarchical.HierarchicalCalibration.load(calibration_path)
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    candidate_rows = np.asarray(arrays["candidate_horizons"], dtype=np.int64)
    if not np.all(candidate_rows == np.asarray(config.candidate_horizons)[None, :]):
        raise ValueError("Dataset and predictor candidate horizons differ.")
    indices = _indices(arrays, args, predictor_dir)
    module = _restore(config, predictor_dir / "params", args.seed)
    predictions = _predict(module, arrays, indices, args.batch_size)

    selector_config = hierarchical.HierarchicalSelectorConfig(
        success_noninferiority_margin=args.success_noninferiority_margin,
        maximum_short_event_probability=args.maximum_short_event_probability,
        require_calibration_for_long_h=True,
    )
    decisions = [
        hierarchical.select_horizon(
            {name: value[root] for name, value in predictions.items()},
            calibration=calibration,
            config=selector_config,
        )
        for root in range(indices.size)
    ]
    selected_horizons = np.asarray([decision.selected_horizon for decision in decisions], dtype=np.int64)
    candidate_to_index = {horizon: index for index, horizon in enumerate(config.candidate_horizons)}
    selected_indices = np.asarray([candidate_to_index[value] for value in selected_horizons], dtype=np.int64)
    reference_index = candidate_to_index[config.reference_horizon]
    row = np.arange(indices.size)

    trial_count = np.asarray(arrays["trial_count"][indices], dtype=np.float64)
    success_rate = np.asarray(arrays["success_count"][indices], dtype=np.float64) / np.maximum(trial_count, 1.0)
    timeout_rate = np.asarray(arrays["timeout_count"][indices], dtype=np.float64) / np.maximum(trial_count, 1.0)
    elapsed = np.asarray(arrays["elapsed_mean"][indices], dtype=np.float64)
    calls = np.asarray(arrays["remaining_calls_mean"][indices], dtype=np.float64)
    selected_success = success_rate[row, selected_indices]
    selected_timeout = timeout_rate[row, selected_indices]
    selected_elapsed = elapsed[row, selected_indices]
    selected_calls = calls[row, selected_indices]
    success_advantage = selected_success - success_rate[:, reference_index]
    elapsed_advantage = selected_elapsed - elapsed[:, reference_index]
    calls_advantage = selected_calls - calls[:, reference_index]

    dangerous = np.asarray(arrays["dangerous_long_count"][indices], dtype=np.float64)
    paired = np.asarray(arrays["paired_trial_count"][indices], dtype=np.float64)
    selected_long = selected_horizons > config.reference_horizon
    dangerous_total = float(np.sum(dangerous[row[selected_long], selected_indices[selected_long]]))
    paired_total = float(np.sum(paired[row[selected_long], selected_indices[selected_long]]))
    false_long_upper_95 = _wilson_upper(dangerous_total, paired_total)

    trial_success = np.asarray(arrays["trial_success"][indices], dtype=np.bool_)
    trial_valid = np.asarray(arrays["trial_valid"][indices], dtype=np.bool_)
    rescues = 0
    regressions = 0
    paired_outcomes = 0
    for root, candidate_index in enumerate(selected_indices):
        valid = trial_valid[root, reference_index] & trial_valid[root, candidate_index]
        reference_success = trial_success[root, reference_index]
        candidate_success = trial_success[root, candidate_index]
        rescues += int(np.sum(valid & ~reference_success & candidate_success))
        regressions += int(np.sum(valid & reference_success & ~candidate_success))
        paired_outcomes += int(np.sum(valid))

    cluster_ids = np.asarray(arrays["task_id"][indices], dtype=np.uint64) * np.uint64(1_000_000_000)
    cluster_ids += np.asarray(arrays["episode_id"][indices], dtype=np.uint64)
    success_interval = _cluster_bootstrap_interval(
        success_advantage,
        cluster_ids,
        seed=args.seed + 1,
        samples=args.bootstrap_samples,
    )
    elapsed_interval = _cluster_bootstrap_interval(
        elapsed_advantage,
        cluster_ids,
        seed=args.seed + 2,
        samples=args.bootstrap_samples,
    )
    calls_interval = _cluster_bootstrap_interval(
        calls_advantage,
        cluster_ids,
        seed=args.seed + 3,
        samples=args.bootstrap_samples,
    )
    calibrated_success_probability = _sigmoid(predictions["success_logits"] / calibration.success_temperature)
    valid = trial_count > 0
    brier = float(np.mean(np.square(calibrated_success_probability[valid] - success_rate[valid])))

    fixed_baselines = {}
    for candidate_index, horizon in enumerate(config.candidate_horizons):
        fixed_baselines[str(horizon)] = {
            "counterfactual_success_rate": float(np.mean(success_rate[:, candidate_index])),
            "counterfactual_timeout_rate": float(np.mean(timeout_rate[:, candidate_index])),
            "mean_elapsed_seconds": float(np.mean(elapsed[:, candidate_index])),
            "mean_remaining_calls": float(np.mean(calls[:, candidate_index])),
        }
    result = {
        "status": "complete",
        "semantics": (
            "Held-out multi-seed counterfactual audit with task-by-initial-state cluster bootstrap; "
            "paired closed-loop LIBERO evaluation remains required."
        ),
        "predictor_dir": str(predictor_dir),
        "calibration_json": str(calibration_path),
        "dataset_inputs": list(args.dataset),
        "split_name": args.split_name,
        "num_roots": int(indices.size),
        "selected_h_distribution": _distribution(selected_horizons),
        "long_h_coverage": float(np.mean(selected_long)),
        "short_h_intervention_rate": float(np.mean(selected_horizons < config.reference_horizon)),
        "ood_fallback_rate": float(
            np.mean([decision.ood_probability >= calibration.ood_probability_threshold for decision in decisions])
        ),
        "counterfactual_success_rate": float(np.mean(selected_success)),
        "counterfactual_timeout_rate": float(np.mean(selected_timeout)),
        "success_advantage_vs_h10_cluster_bootstrap": success_interval,
        "elapsed_advantage_vs_h10_cluster_bootstrap": elapsed_interval,
        "calls_advantage_vs_h10_cluster_bootstrap": calls_interval,
        "false_long_rate": dangerous_total / max(paired_total, 1.0),
        "false_long_upper_95": false_long_upper_95,
        "false_long_paired_trials": int(paired_total),
        "rescues_vs_h10": rescues,
        "regressions_vs_h10": regressions,
        "paired_seed_outcomes_vs_h10": paired_outcomes,
        "calibrated_success_brier": brier,
        "offline_engineering_gate": bool(
            success_interval["ci95"][0] is not None
            and success_interval["ci95"][0] >= -args.success_noninferiority_margin
            and elapsed_interval["ci95"][1] is not None
            and elapsed_interval["ci95"][1] < 0.0
            and np.any(selected_long)
            and false_long_upper_95 <= args.false_long_upper_bound
        ),
        "fixed_h_baselines": fixed_baselines,
        "config": dataclasses.asdict(args),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_path = pathlib.Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
