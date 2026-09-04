"""Audit one frozen aggregate-risk rule on an explicit development audit split.

The audit labels are used exactly once for reporting. Thresholds and the
candidate-selection rule are loaded from the calibration artifact and are
never refitted or selected by this command. Four-way manifests use the
``dev_audit`` role; legacy manifests may still use ``validation``. Final/test/
holdout inputs are refused before any dataset is opened.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import execution_horizon_aggregate_risk_common as common
import numpy as np
import tyro

from openpi.execution_horizon import hierarchical


@dataclasses.dataclass(frozen=True)
class Args:
    development_dataset: tuple[str, ...]
    predictor_dir: str
    aggregate_calibration_json: str
    pointwise_calibration_json: str
    split_manifest: str
    audit_split_name: str
    output_json: str
    params: str | None = None
    batch_size: int = 256
    inference_initialization_seed: int | None = None


def _row(predictions: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[index] for name, value in predictions.items()}


def _fixed_baselines(
    labels: dict[str, np.ndarray],
    candidate_horizons: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    trials = np.asarray(labels["trial_count"], dtype=np.float64)
    success = np.asarray(labels["success_count"], dtype=np.float64) / np.maximum(trials, 1.0)
    elapsed = np.asarray(labels["elapsed_mean"], dtype=np.float64)
    calls = np.asarray(labels["remaining_calls_mean"], dtype=np.float64)
    result = {}
    for index, horizon in enumerate(candidate_horizons):
        result[str(horizon)] = {
            "counterfactual_success_rate": float(np.mean(success[:, index])),
            "mean_elapsed_seconds": float(np.mean(elapsed[:, index])),
            "mean_remaining_calls": float(np.mean(calls[:, index])),
        }
    return result


def _calibration_risk_coverage(artifact: hierarchical.AggregateSelectorCalibration) -> dict[str, object]:
    return {
        "source_split": artifact.source_split,
        "audit_split_used_for_rule_selection": False,
        "interpretation": (
            "Calibration-only risk/coverage frontier copied from the frozen artifact. "
            "Audit labels were not swept and did not choose a rule."
        ),
        "selected_rule": common.jsonable(artifact.selected_rule),
        "selected_calibration_metrics": common.jsonable(artifact.calibration_metrics),
        "search_evaluations": [common.jsonable(item) for item in artifact.search_evaluations],
    }


def _audit_split_role(split_name: str) -> str:
    if split_name == "validation":
        return "validation"
    if split_name == "dev_audit":
        return "development_audit"
    raise ValueError("Aggregate-risk audit split must be legacy 'validation' or four-way 'dev_audit'.")


def main(args: Args) -> None:
    audit_split_role = _audit_split_role(args.audit_split_name)
    output_path = pathlib.Path(args.output_json).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot aggregate audit output: {output_path}.")
    arrays, indices, cluster_ids, manifest, inputs = common.load_development_split(
        args.development_dataset,
        split_manifest=args.split_manifest,
        split_name=args.audit_split_name,
        required_role=audit_split_role,
    )
    artifact_path = pathlib.Path(args.aggregate_calibration_json).resolve()
    artifact = hierarchical.AggregateSelectorCalibration.load(artifact_path)
    if artifact.source_split != "calibration":
        raise ValueError("Aggregate rule does not prove calibration-only provenance.")
    if not np.isclose(artifact.search_config.confidence_level, 0.95):
        raise ValueError("This development audit requires a calibration artifact preregistered at 95% confidence.")
    calibration_groups = tuple(sorted({int(value) for value in manifest["calibration_group_ids"]}))
    audit_groups = tuple(int(value) for value in np.unique(cluster_ids))
    if calibration_groups != artifact.provenance.calibration_group_ids:
        raise ValueError("Manifest calibration groups differ from the frozen aggregate provenance.")
    overlap = sorted(set(calibration_groups).intersection(audit_groups))
    if overlap:
        raise ValueError(f"Development audit groups overlap calibration groups: {overlap[:10]}.")
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
    if tuple(predictor_config.candidate_horizons) != artifact.candidate_horizons:
        raise ValueError("Predictor and aggregate calibration candidate horizons differ.")
    if int(predictor_config.reference_horizon) != artifact.reference_horizon:
        raise ValueError("Predictor and aggregate calibration reference horizons differ.")
    if not getattr(predictor_config, "paired_distribution_heads", False):
        raise ValueError("Aggregate-risk audit requires coherent paired_distribution_heads predictions.")
    pointwise_path = pathlib.Path(args.pointwise_calibration_json).resolve()
    pointwise = hierarchical.HierarchicalCalibration.load(pointwise_path)
    pointwise_values = common.jsonable(dataclasses.asdict(pointwise))
    embedded_pointwise_values = common.jsonable(dataclasses.asdict(artifact.pointwise_calibration))
    if pointwise_values != embedded_pointwise_values:
        raise ValueError("Provided pointwise calibration content differs from the embedded frozen calibration.")
    live_provenance_values = common.provenance_values(
        predictor_dir=args.predictor_dir,
        params_path=params_path,
        pointwise_calibration_json=pointwise_path,
        split_manifest=args.split_manifest,
        development_dataset=inputs,
        calibration_group_ids=calibration_groups,
    )
    frozen_provenance_values = dataclasses.asdict(artifact.provenance)
    common.verify_provenance(frozen_provenance_values, live_provenance_values)
    labels = common.split_labels(arrays, indices)
    decisions = [artifact.apply(_row(predictions, index)) for index in range(indices.size)]
    selected = np.asarray([decision.selected_horizon for decision in decisions], dtype=np.int64)
    metrics = common.selection_metrics(
        labels,
        selected_horizons=selected,
        candidate_horizons=artifact.candidate_horizons,
        reference_horizon=artifact.reference_horizon,
        cluster_ids=cluster_ids,
        bootstrap_samples=artifact.search_config.bootstrap_samples,
        seed=artifact.search_config.bootstrap_seed,
        success_noninferiority_margin=artifact.search_config.success_noninferiority_margin,
        false_long_upper_bound=artifact.search_config.false_long_upper_bound,
    )
    if artifact.reference_horizon == 10:
        metrics.update(
            {
                "success_advantage_vs_h10_cluster_bootstrap": metrics[
                    "success_advantage_vs_reference_cluster_bootstrap"
                ],
                "elapsed_advantage_vs_h10_cluster_bootstrap": metrics[
                    "elapsed_advantage_vs_reference_cluster_bootstrap"
                ],
                "calls_advantage_vs_h10_cluster_bootstrap": metrics["calls_advantage_vs_reference_cluster_bootstrap"],
            }
        )
    report = {
        "status": "complete",
        "semantics": (
            "One evaluation of a calibration-frozen aggregate selector on an explicit development validation "
            "split. This is an offline counterfactual audit, not a closed-loop result."
        ),
        "aggregate_calibration_json": str(artifact_path),
        "predictor_dir": str(pathlib.Path(args.predictor_dir).resolve()),
        "params": str(params_path),
        "pointwise_calibration_json": str(pointwise_path),
        "predictor_summary": str(predictor_summary_path),
        "inference_initialization_seed": inference_seed,
        "inference_seed_semantics": (
            "Used only to initialize the predictor graph before complete parameter restoration; required to "
            "match predictor summary training_seed. Model identity is bound by config and params digests."
        ),
        "provenance_verified": True,
        "provenance": live_provenance_values,
        "development_dataset_inputs": list(inputs),
        "split_manifest": str(pathlib.Path(args.split_manifest).resolve()),
        "split_name": args.audit_split_name,
        "split_role": audit_split_role,
        "num_clusters": len(audit_groups),
        "selected_group_ids": list(audit_groups),
        "calibration_group_ids": list(calibration_groups),
        "calibration_audit_group_overlap": [],
        "calibration_validation_group_overlap": [],
        "rule_frozen_before_audit": True,
        "audit_split_used_for_threshold_fit": False,
        "aggregate_calibration_gate_passed": bool(artifact.aggregate_gate_passed),
        "selected_rule": common.jsonable(artifact.selected_rule),
        **metrics,
        "constraint_elimination": common.decision_diagnostics(decisions),
        "risk_coverage": _calibration_risk_coverage(artifact),
        "fixed_h_baselines": _fixed_baselines(labels, artifact.candidate_horizons),
        "config": dataclasses.asdict(args),
        "audit_statistics_config": {
            "confidence_level": artifact.search_config.confidence_level,
            "bootstrap_samples": artifact.search_config.bootstrap_samples,
            "bootstrap_seed": artifact.search_config.bootstrap_seed,
            "source": "frozen aggregate calibration artifact",
        },
    }
    common.write_json(output_path, report)
    print(json.dumps(common.jsonable(report), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
