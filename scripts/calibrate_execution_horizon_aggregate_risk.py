"""Fit a frozen aggregate-risk execution-horizon rule on development calibration groups.

This command is intentionally development-only.  It requires an explicit
split manifest and split name, and accepts only a split whose declared role is
``calibration``.  It never reads a final/test/holdout-labelled dataset path.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import execution_horizon_aggregate_risk_common as common
import numpy as np
import tyro

from openpi.execution_horizon import hierarchical


@dataclasses.dataclass(frozen=True)
class Args:
    development_dataset: tuple[str, ...]
    predictor_dir: str
    pointwise_calibration_json: str
    split_manifest: str
    calibration_split_name: str
    output_json: str
    report_json: str | None = None
    params: str | None = None
    batch_size: int = 256
    inference_initialization_seed: int | None = None
    success_thresholds: tuple[float, ...] = (-0.01, -0.005, 0.0)
    elapsed_margins: tuple[float, ...] = (0.0, 0.1, 0.25)
    maximum_danger_probabilities: tuple[float, ...] = (0.20, 0.10, 0.05)
    require_faster_options: tuple[bool, ...] = (False, True)
    maximum_hazard_probabilities: tuple[float, ...] = (0.20, 0.10)
    maximum_ood_probabilities: tuple[float, ...] = (0.95, 0.90)
    minimum_faster_probability: float = 0.50
    candidate_selection_strategy: str = "minimum_predicted_elapsed"
    confidence_level: float = 0.95
    success_noninferiority_margin: float = 0.01
    false_long_upper_bound: float = 0.05
    bootstrap_samples: int = 5_000
    bootstrap_seed: int = 7


def _row(predictions: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[index] for name, value in predictions.items()}


def _risk_coverage_payload(artifact: hierarchical.AggregateSelectorCalibration) -> dict[str, Any]:
    """Expose calibration-only threshold search without implying audit tuning."""

    evaluations = [
        {
            "rule": common.jsonable(item.rule),
            "coverage": float(item.metrics.long_coverage),
            "selected_long_roots": int(item.metrics.selected_long_roots),
            "success_advantage_lcb": float(item.metrics.success_advantage_lcb),
            "elapsed_advantage_ucb": float(item.metrics.elapsed_advantage_ucb),
            "false_long_upper": float(item.metrics.false_long_upper),
            "passed": bool(item.metrics.aggregate_gate_passed),
            "selected_h_distribution": dict(item.metrics.selected_h_distribution),
        }
        for item in artifact.search_evaluations
    ]
    return {
        "source_split": artifact.source_split,
        "selection_policy": "highest feasible long-H coverage on calibration; no validation labels used",
        "num_rules_evaluated": int(artifact.num_rules_evaluated),
        "num_unique_selections": int(artifact.num_unique_selections),
        "num_feasible_rules": int(artifact.num_feasible_rules),
        "evaluations": evaluations,
    }


def main(args: Args) -> None:
    if args.calibration_split_name != "calibration":
        raise ValueError("Aggregate-risk fitting requires the explicit split name 'calibration'.")
    if not np.isclose(args.confidence_level, 0.95):
        raise ValueError("This development gate is preregistered at 95% confidence.")
    if args.bootstrap_samples < 5_000:
        raise ValueError("Aggregate calibration requires at least 5000 bootstrap samples.")
    artifact_target = pathlib.Path(args.output_json).resolve()
    report_target = (
        pathlib.Path(args.report_json).resolve()
        if args.report_json is not None
        else artifact_target.with_name(artifact_target.stem + "_report.json")
    )
    if artifact_target == report_target:
        raise ValueError("Aggregate calibration artifact and report paths must differ.")
    existing_outputs = [str(path) for path in (artifact_target, report_target) if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite frozen aggregate calibration outputs: {existing_outputs}.")
    arrays, indices, cluster_ids, _, inputs = common.load_development_split(
        args.development_dataset,
        split_manifest=args.split_manifest,
        split_name=args.calibration_split_name,
        required_role="calibration",
    )
    inference_seed, predictor_summary_path = common.resolve_inference_initialization_seed(
        args.predictor_dir,
        args.inference_initialization_seed,
    )
    predictor_config, predictions, params_path = common.predict_split(
        args.predictor_dir,
        arrays,
        indices,
        params=args.params,
        batch_size=args.batch_size,
        inference_initialization_seed=inference_seed,
    )
    if predictor_config.temporal_backbone != "transformer":
        raise ValueError("Aggregate-risk calibration requires a Transformer predictor.")
    if not getattr(predictor_config, "paired_distribution_heads", False):
        raise ValueError("Aggregate-risk calibration requires coherent paired_distribution_heads predictions.")

    pointwise_path = pathlib.Path(args.pointwise_calibration_json).resolve()
    pointwise = hierarchical.HierarchicalCalibration.load(pointwise_path)
    if tuple(predictor_config.candidate_horizons) != pointwise.candidate_horizons:
        raise ValueError("Predictor and pointwise calibration candidate horizons differ.")
    if int(predictor_config.reference_horizon) != pointwise.reference_horizon:
        raise ValueError("Predictor and pointwise calibration reference horizons differ.")
    labels = common.split_labels(arrays, indices)
    provenance_values = common.provenance_values(
        predictor_dir=args.predictor_dir,
        params_path=params_path,
        pointwise_calibration_json=pointwise_path,
        split_manifest=args.split_manifest,
        development_dataset=inputs,
        calibration_group_ids=np.unique(cluster_ids),
    )
    provenance = hierarchical.AggregateSelectorProvenance(**provenance_values)
    search_config = hierarchical.AggregateSelectorCalibrationConfig(
        success_thresholds=args.success_thresholds,
        elapsed_margins=args.elapsed_margins,
        maximum_danger_probabilities=args.maximum_danger_probabilities,
        require_faster_options=args.require_faster_options,
        maximum_hazard_probabilities=args.maximum_hazard_probabilities,
        maximum_ood_probabilities=args.maximum_ood_probabilities,
        minimum_faster_probability=args.minimum_faster_probability,
        candidate_selection_strategy=args.candidate_selection_strategy,
        confidence_level=args.confidence_level,
        success_noninferiority_margin=args.success_noninferiority_margin,
        false_long_upper_bound=args.false_long_upper_bound,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        labels,
        calibration=pointwise,
        provenance=provenance,
        cluster_ids=cluster_ids,
        config=search_config,
        split_name=args.calibration_split_name,
    )
    artifact_path = artifact.save(artifact_target)
    decisions = [artifact.apply(_row(predictions, index)) for index in range(indices.size)]
    selected = np.asarray([decision.selected_horizon for decision in decisions], dtype=np.int64)
    metrics = common.selection_metrics(
        labels,
        selected_horizons=selected,
        candidate_horizons=predictor_config.candidate_horizons,
        reference_horizon=predictor_config.reference_horizon,
        cluster_ids=cluster_ids,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        success_noninferiority_margin=args.success_noninferiority_margin,
        false_long_upper_bound=args.false_long_upper_bound,
    )
    report = {
        "status": "complete",
        "semantics": (
            "Aggregate-risk thresholds were fitted only on the explicitly declared development calibration "
            "groups. The saved rule is frozen before any validation or final audit."
        ),
        "aggregate_calibration_json": str(pathlib.Path(artifact_path).resolve()),
        "pointwise_calibration_json": str(pointwise_path),
        "predictor_dir": str(pathlib.Path(args.predictor_dir).resolve()),
        "params": str(params_path),
        "predictor_summary": str(predictor_summary_path),
        "inference_initialization_seed": inference_seed,
        "inference_seed_semantics": (
            "Used only to initialize the predictor graph before complete parameter restoration; required to "
            "match predictor summary training_seed. Model identity is bound by config and params digests."
        ),
        "provenance": provenance_values,
        "provenance_digest_algorithms": {
            "json": "SHA-256 over parsed canonical JSON",
            "params": "SHA-256 over sorted relative checkpoint entries and exact file SHA-256 values",
            "development_dataset": (
                "SHA-256 over ordered development inputs, canonical manifest digests, and exact shard SHA-256 values"
            ),
        },
        "development_dataset_inputs": list(inputs),
        "split_manifest": str(pathlib.Path(args.split_manifest).resolve()),
        "split_name": args.calibration_split_name,
        "split_role": "calibration",
        "num_roots": int(indices.size),
        "num_clusters": int(np.unique(cluster_ids).size),
        "selected_group_ids": [int(value) for value in np.unique(cluster_ids)],
        "aggregate_gate_passed": bool(artifact.aggregate_gate_passed),
        "selected_rule": common.jsonable(artifact.selected_rule),
        "artifact_calibration_metrics": common.jsonable(artifact.calibration_metrics),
        "frozen_rule_replay_on_calibration": metrics,
        "constraint_elimination": common.decision_diagnostics(decisions),
        "risk_coverage": _risk_coverage_payload(artifact),
        "config": dataclasses.asdict(args),
    }
    common.write_json(report_target, report)
    print(json.dumps(common.jsonable(report), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
