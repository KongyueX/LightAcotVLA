"""Audit V2-P sidecars against held-out H1-H10 counterfactual labels.

This is an offline model-selection utility.  It deliberately does not load the
base policy and does not claim to replace closed-loop LIBERO evaluation.
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
from openpi.execution_horizon import v2
from openpi.models import model as model_lib
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictor
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictorConfig


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    sidecars: tuple[str, ...]
    output_json: str | None = None
    seed: int = 7
    validation_fraction: float = 0.2
    batch_size: int = 256
    hidden_dim: int = 256
    temporal_layers: int = 3
    risk_threshold: float = 1.5
    minimum_success_probabilities: tuple[float, ...] = (0.90, 0.925, 0.95, 0.975)
    maximum_timeout_probabilities: tuple[float, ...] = (0.10, 0.15, 0.20)
    risk_slack_steps: tuple[int, ...] = (0,)


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


def _validation_indices(arrays: dict[str, np.ndarray], args: Args) -> np.ndarray:
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5).")
    groups = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    groups += np.asarray(arrays["episode_id"], dtype=np.uint64)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two episode groups are required for a held-out audit.")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    validation_count = max(1, round(len(unique_groups) * args.validation_fraction))
    validation_groups = unique_groups[:validation_count]
    indices = np.flatnonzero(np.isin(groups, validation_groups))
    if not indices.size:
        raise ValueError("Validation split is empty.")
    return indices


def _predictor_config(arrays: dict[str, np.ndarray], args: Args) -> ExecutionHorizonPredictorConfig:
    return ExecutionHorizonPredictorConfig(
        prefix_feature_dim=int(arrays["prefix_feature"].shape[-1]),
        state_dim=int(arrays["state"].shape[-1]),
        action_dim=int(arrays["final_actions"].shape[-1]),
        coarse_horizon=int(arrays["coarse_actions"].shape[-2]),
        action_horizon=int(arrays["final_actions"].shape[-2]),
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
    )


def _restore_predictor(
    config: ExecutionHorizonPredictorConfig,
    sidecar: str,
    *,
    seed: int,
) -> ExecutionHorizonPredictor:
    module = ExecutionHorizonPredictor(config, rngs=nnx.Rngs(seed))
    loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(sidecar, dtype=jnp.float32))
    if "execution_horizon_predictor" in loaded:
        loaded = loaded["execution_horizon_predictor"]
    graphdef, state = nnx.split(module)
    state.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, state)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _predict(
    module: ExecutionHorizonPredictor,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    @jax.jit
    def infer(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return module(**batch)

    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = {name: jnp.asarray(arrays[name][selected]) for name in _INPUT_FIELDS}
        prediction = jax.device_get(infer(batch))
        for name, value in prediction.items():
            pieces.setdefault(name, []).append(np.asarray(value))
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def _distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _selection_metrics(
    horizons: np.ndarray,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    mask: np.ndarray,
    *,
    entropy_horizons: np.ndarray | None = None,
    any_eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    selected_indices = indices[mask]
    selected_horizons = np.asarray(horizons[mask], dtype=np.int64)
    # Keep this assertion close to label indexing; malformed horizons must not
    # silently wrap around and produce optimistic audit metrics.
    if not np.all((selected_horizons >= 1) & (selected_horizons <= arrays["branch_success"].shape[1])):
        raise ValueError("Selected horizon is outside the stored counterfactual label range.")
    success = np.asarray(arrays["branch_success"])[selected_indices, selected_horizons - 1]
    timeout = np.asarray(arrays["branch_timeout"])[selected_indices, selected_horizons - 1]
    result: dict[str, Any] = {
        "num_roots": int(mask.sum()),
        "counterfactual_success_rate": float(np.mean(success)),
        "counterfactual_timeout_rate": float(np.mean(timeout)),
        "average_raw_horizon": float(np.mean(selected_horizons)),
        "h_distribution": _distribution(selected_horizons),
    }
    if entropy_horizons is not None:
        entropy_selected = np.asarray(entropy_horizons[mask], dtype=np.int64)
        result["q_reduction_rate"] = float(np.mean(selected_horizons < entropy_selected))
    if any_eligible is not None:
        result["any_q_eligible_rate"] = float(np.mean(any_eligible[mask]))
    return result


def _subset_metrics(
    horizons: np.ndarray,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    entropy_horizons: np.ndarray | None = None,
    any_eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    task_ids = np.asarray(arrays["task_id"])[indices]
    subsets = {
        "overall": np.ones(len(indices), dtype=np.bool_),
        "task8": task_ids == 8,
        "task9": task_ids == 9,
        "hard_tasks_8_9": np.isin(task_ids, (8, 9)),
    }
    return {
        name: _selection_metrics(
            horizons,
            arrays,
            indices,
            mask,
            entropy_horizons=entropy_horizons,
            any_eligible=any_eligible,
        )
        for name, mask in subsets.items()
        if np.any(mask)
    }


def _audit_sidecar(
    sidecar: str,
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    args: Args,
) -> dict[str, Any]:
    risk_config = v2.V2RiskConfig(risk_threshold=args.risk_threshold)
    num_roots = len(indices)
    entropy_horizons = np.empty(num_roots, dtype=np.int64)
    predicted_event = np.empty(num_roots, dtype=np.bool_)
    for index in range(num_roots):
        entropy_horizons[index], event_mask = v2.distilled_raw_horizon(
            predictions["final_risk"][index],
            predictions["action_cot_risk"][index],
            predictions["fused_risk"][index],
            candidates=range(3, 11),
            config=risk_config,
        )
        predicted_event[index] = bool(np.any(event_mask))

    report: dict[str, Any] = {
        "sidecar": str(pathlib.Path(sidecar).resolve()),
        "predicted_curve_event_rate": float(np.mean(predicted_event)),
        "distilled": _subset_metrics(entropy_horizons, arrays, indices),
        "threshold_sweep": [],
    }
    success_probability = _sigmoid(predictions["success_logits"])
    timeout_probability = _sigmoid(predictions["timeout_logits"])
    for minimum_success in args.minimum_success_probabilities:
        for maximum_timeout in args.maximum_timeout_probabilities:
            for risk_slack_steps in args.risk_slack_steps:
                value_horizons = np.empty(num_roots, dtype=np.int64)
                any_eligible = np.empty(num_roots, dtype=np.bool_)
                config = v2.ValueRefinementConfig(
                    minimum_success_probability=minimum_success,
                    maximum_timeout_probability=maximum_timeout,
                    risk_threshold=args.risk_threshold,
                    risk_slack_steps=risk_slack_steps,
                    candidates=tuple(range(1, 11)),
                )
                for index in range(num_roots):
                    value_horizons[index], filters = v2.value_refined_raw_horizon(
                        entropy_raw_horizon=int(entropy_horizons[index]),
                        success_probability=success_probability[index],
                        timeout_probability=timeout_probability[index],
                        fused_risk=predictions["fused_risk"][index],
                        config=config,
                    )
                    any_eligible[index] = bool(np.any(filters["eligible"]))
                report["threshold_sweep"].append(
                    {
                        "minimum_success_probability": minimum_success,
                        "maximum_timeout_probability": maximum_timeout,
                        "risk_slack_steps": risk_slack_steps,
                        "metrics": _subset_metrics(
                            value_horizons,
                            arrays,
                            indices,
                            entropy_horizons=entropy_horizons,
                            any_eligible=any_eligible,
                        ),
                    }
                )
    return report


def main(args: Args) -> None:
    if not args.sidecars:
        raise ValueError("At least one sidecar is required.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    indices = _validation_indices(arrays, args)
    config = _predictor_config(arrays, args)
    reports = []
    for sidecar in args.sidecars:
        module = _restore_predictor(config, sidecar, seed=args.seed)
        predictions = _predict(module, arrays, indices, args.batch_size)
        reports.append(_audit_sidecar(sidecar, predictions, arrays, indices, args))
    result = {
        "status": "complete",
        "semantics": (
            "Held-out root-state counterfactual audit without episode budget dynamics; "
            "closed-loop LIBERO evaluation remains required."
        ),
        "base_policy_loaded": False,
        "base_policy_frozen": True,
        "dataset_inputs": list(args.dataset),
        "num_records": len(arrays["task_id"]),
        "num_validation_records": len(indices),
        "validation_task_distribution": _distribution(np.asarray(arrays["task_id"])[indices]),
        "config": dataclasses.asdict(args),
        "sidecars": reports,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        target = pathlib.Path(args.output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
