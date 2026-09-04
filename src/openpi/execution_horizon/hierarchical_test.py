from __future__ import annotations

import dataclasses
import json

import numpy as np

from openpi.execution_horizon import hierarchical


def _predictions() -> dict[str, np.ndarray]:
    return {
        "candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
        "reference_horizon": np.asarray(10),
        "hazard_logits": np.full((25,), -8.0),
        "success_advantage": np.asarray([0.0, 0.01, 0.01]),
        "success_advantage_std": np.asarray([0.002, 0.002, 0.002]),
        "elapsed_advantage": np.asarray([-0.5, -1.0, -1.2]),
        "elapsed_advantage_std": np.asarray([0.05, 0.05, 0.05]),
        "ood_probability": np.asarray(0.1),
        "temporal_feature": np.asarray([0.0, 0.0]),
    }


def _calibration() -> hierarchical.HierarchicalCalibration:
    return hierarchical.HierarchicalCalibration(
        candidate_horizons=(5, 10, 15, 20, 25),
        success_residual_quantiles=(2.0, 2.0, 2.0),
        elapsed_residual_quantiles=(2.0, 2.0, 2.0),
        num_calibration_roots=100,
    )


def _aggregate_predictions(num_roots: int) -> dict[str, np.ndarray]:
    return {
        "candidate_horizons": np.tile(np.asarray([5, 10, 15, 20, 25]), (num_roots, 1)),
        "reference_horizon": np.full((num_roots,), 10),
        "long_horizons": np.tile(np.asarray([15, 20, 25]), (num_roots, 1)),
        "hazard_logits": np.full((num_roots, 25), -8.0),
        "success_advantage": np.tile(np.asarray([0.01, 0.02, 0.01]), (num_roots, 1)),
        "elapsed_advantage": np.tile(np.asarray([-0.5, -2.0, -1.0]), (num_roots, 1)),
        "danger_probability": np.tile(np.asarray([0.10, 0.04, 0.03]), (num_roots, 1)),
        "faster_long_probability": np.full((num_roots, 3), 0.8),
        "ood_probability": np.full((num_roots,), 0.1),
        "temporal_feature": np.zeros((num_roots, 2)),
    }


def _aggregate_labels(num_roots: int, *, long_success: int = 3) -> dict[str, np.ndarray]:
    if not 0 <= long_success <= 3:
        raise ValueError("long_success must lie in [0, 3].")
    success_count = np.full((num_roots, 5), 3)
    success_count[:, 2:] = long_success
    trial_valid = np.ones((num_roots, 5, 3), dtype=np.bool_)
    trial_success = np.zeros_like(trial_valid)
    for candidate, count in enumerate(success_count[0]):
        trial_success[:, candidate, :count] = True
    dangerous = np.zeros((num_roots, 5), dtype=np.int64)
    dangerous[:, 2:] = 3 - long_success
    return {
        "success_count": success_count,
        "trial_count": np.full((num_roots, 5), 3),
        "elapsed_mean": np.tile(np.asarray([11.0, 10.0, 9.5, 8.0, 9.0]), (num_roots, 1)),
        "dangerous_long_count": dangerous,
        "paired_trial_count": np.tile(np.asarray([0, 0, 3, 3, 3]), (num_roots, 1)),
        "trial_success": trial_success,
        "trial_valid": trial_valid,
    }


def _aggregate_config(**updates) -> hierarchical.AggregateSelectorCalibrationConfig:
    values = {
        "success_thresholds": (-0.01, 0.0),
        "elapsed_margins": (0.0,),
        "maximum_danger_probabilities": (0.20,),
        "require_faster_options": (False,),
        "maximum_hazard_probabilities": (0.20,),
        "maximum_ood_probabilities": (0.95,),
        "bootstrap_samples": 5_000,
        "bootstrap_seed": 11,
    }
    values.update(updates)
    return hierarchical.AggregateSelectorCalibrationConfig(**values)


def _aggregate_provenance(num_roots: int) -> hierarchical.AggregateSelectorProvenance:
    return hierarchical.AggregateSelectorProvenance(
        predictor_config_digest="1" * 64,
        params_digest="2" * 64,
        pointwise_calibration_digest="3" * 64,
        split_manifest_digest="4" * 64,
        development_dataset_fingerprint="5" * 64,
        calibration_group_ids=tuple(range(num_roots)),
    )


def test_selector_uses_largest_calibrated_safe_long_horizon():
    decision = hierarchical.select_horizon(_predictions(), calibration=_calibration())

    assert decision.selected_horizon == 25
    assert decision.reason == "calibrated_long_h"


def test_selector_never_defaults_to_h20_without_calibration():
    decision = hierarchical.select_horizon(_predictions(), calibration=None)

    assert decision.selected_horizon == 10
    assert decision.reason == "uncalibrated_fallback_reference"


def test_selector_reads_parameterized_reference_from_predictor():
    predictions = _predictions()
    predictions.update(
        {
            "candidate_horizons": np.asarray([2, 4, 6, 8]),
            "reference_horizon": np.asarray(6),
            "success_advantage": np.asarray([0.0]),
            "success_advantage_std": np.asarray([0.002]),
            "elapsed_advantage": np.asarray([-0.5]),
            "elapsed_advantage_std": np.asarray([0.05]),
        }
    )

    decision = hierarchical.select_horizon(predictions, calibration=None)

    assert decision.selected_horizon == 6
    assert decision.reason == "uncalibrated_fallback_reference"


def test_selector_shortens_when_hazard_is_high():
    predictions = _predictions()
    predictions["success_advantage"] = np.asarray([-0.5, -0.5, -0.5])
    predictions["hazard_logits"] = np.concatenate([np.full((5,), -8.0), np.full((20,), 8.0)])

    decision = hierarchical.select_horizon(predictions, calibration=_calibration())

    assert decision.selected_horizon == 5
    assert decision.reason == "hazard_short_gate"


def test_selector_rejects_long_horizon_when_cumulative_hazard_is_high():
    predictions = _predictions()
    predictions["hazard_logits"] = np.concatenate([np.full((10,), -8.0), np.full((15,), 8.0)])

    decision = hierarchical.select_horizon(predictions, calibration=_calibration())

    assert decision.selected_horizon == 10
    assert decision.reason == "default_reference_h"
    assert not any(decision.long_eligible)
    assert all(value > 0.20 for value in decision.long_event_probability)


def test_selector_uses_conformal_feature_distance_for_ood_fallback():
    calibration = dataclasses.replace(
        _calibration(),
        ood_feature_center=(0.0, 0.0),
        ood_feature_scale=(1.0, 1.0),
        ood_calibration_distances=(0.1, 0.2, 0.3),
        ood_probability_threshold=0.75,
    )
    predictions = _predictions()
    predictions["temporal_feature"] = np.asarray([10.0, 10.0])

    decision = hierarchical.select_horizon(predictions, calibration=calibration)

    assert decision.selected_horizon == 10
    assert decision.ood_probability == 0.75


def test_count_temperature_and_conformal_fit_are_finite():
    predictions = {
        "success_logits": np.zeros((4, 5)),
        "hazard_logits": np.zeros((4, 25)),
        "success_advantage": np.zeros((4, 3)),
        "success_advantage_std": np.ones((4, 3)),
        "elapsed_advantage": np.zeros((4, 3)),
        "elapsed_advantage_std": np.ones((4, 3)),
        "temporal_feature": np.zeros((4, 2)),
    }
    labels = {
        "success_count": np.asarray([[3, 3, 2, 2, 2]] * 4),
        "trial_count": np.full((4, 5), 3),
        "elapsed_mean": np.asarray([[8, 7, 6, 5, 4]] * 4, dtype=np.float32),
        "event_mask": np.zeros((4, 25), dtype=np.bool_),
        "risk_valid": np.ones((4, 25), dtype=np.bool_),
        "hazard_event_count": np.zeros((4, 25), dtype=np.int64),
        "hazard_at_risk_count": np.full((4, 25), 3, dtype=np.int64),
    }

    artifact = hierarchical.fit_calibration(
        predictions,
        labels,
        candidate_horizons=(5, 10, 15, 20, 25),
        training_features=np.zeros((8, 2)),
    )

    assert np.isfinite(artifact.hazard_temperature)
    assert np.isfinite(artifact.success_temperature)
    assert len(artifact.success_residual_quantiles) == 3
    assert artifact.has_ood_calibration


def test_aggregate_selector_fits_only_on_calibration_and_round_trips(tmp_path):
    num_roots = 40
    predictions = _aggregate_predictions(num_roots)
    labels = _aggregate_labels(num_roots)

    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        labels,
        calibration=_calibration(),
        provenance=_aggregate_provenance(num_roots),
        cluster_ids=np.arange(num_roots),
        config=_aggregate_config(),
        split_name="calibration",
    )

    assert artifact.aggregate_gate_passed
    assert artifact.source_split == "calibration"
    assert artifact.selected_rule is not None
    # Both thresholds produce the same selections, so the documented
    # deterministic safety tie-break keeps the stricter threshold.
    assert artifact.selected_rule.success_threshold == 0.0
    decision = artifact.apply({name: value[0] for name, value in predictions.items()})
    assert decision.selected_horizon == 20
    assert decision.reason == "aggregate_calibrated_long_h_min_elapsed"
    assert all(decision.long_eligible)

    path = artifact.save(tmp_path / "aggregate_calibration.json")
    loaded = hierarchical.AggregateSelectorCalibration.load(path)
    assert loaded == artifact
    saved = json.loads(path.read_text())
    assert saved["source_split"] == "calibration"
    assert saved["schema_version"] == hierarchical.AGGREGATE_SELECTOR_SCHEMA_VERSION
    assert saved["provenance"]["params_digest"] == "2" * 64


def test_aggregate_selector_rejects_validation_label_selection():
    num_roots = 40
    with np.testing.assert_raises_regex(ValueError, "calibration-only"):
        hierarchical.fit_aggregate_selector_calibration(
            _aggregate_predictions(num_roots),
            _aggregate_labels(num_roots),
            calibration=_calibration(),
            provenance=_aggregate_provenance(num_roots),
            cluster_ids=np.arange(num_roots),
            config=_aggregate_config(),
            split_name="validation",
        )


def test_aggregate_selector_no_passing_rule_falls_back_to_h10_then_h5():
    num_roots = 40
    predictions = _aggregate_predictions(num_roots)
    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        _aggregate_labels(num_roots, long_success=0),
        calibration=_calibration(),
        provenance=_aggregate_provenance(num_roots),
        cluster_ids=np.arange(num_roots),
        config=_aggregate_config(),
    )

    assert not artifact.aggregate_gate_passed
    row = {name: value[0] for name, value in predictions.items()}
    decision = artifact.apply(row)
    assert decision.selected_horizon == 10
    assert decision.reason == "aggregate_calibration_fallback_reference"
    assert not any(decision.long_eligible)
    assert all("aggregate_gate" in reasons for reasons in decision.rejection_reasons)

    row["hazard_logits"] = np.concatenate([np.full((5,), -8.0), np.full((20,), 8.0)])
    decision = artifact.apply(row)
    assert decision.selected_horizon == 5
    assert decision.reason == "aggregate_hazard_short_gate"

    for name in ("success_advantage", "elapsed_advantage"):
        row.pop(name)
    assert artifact.apply(row).selected_horizon == 5
    row.pop("hazard_logits")
    with np.testing.assert_raises_regex(ValueError, "requires hazard_logits"):
        artifact.apply(row)


def test_aggregate_minimum_predicted_elapsed_does_not_default_to_largest_h():
    num_roots = 40
    predictions = _aggregate_predictions(num_roots)
    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        _aggregate_labels(num_roots),
        calibration=_calibration(),
        provenance=_aggregate_provenance(num_roots),
        cluster_ids=np.arange(num_roots),
        config=_aggregate_config(success_thresholds=(-0.01,)),
    )

    row = {name: value[0] for name, value in predictions.items()}
    decision = artifact.apply(row)

    assert decision.long_eligible == (True, True, True)
    assert decision.elapsed_score == (-0.5, -2.0, -1.0)
    assert decision.selected_horizon == 20

    # Equal elapsed scores deterministically prefer the smaller safe horizon.
    row["elapsed_advantage"] = np.asarray([-0.5, -2.0, -2.0])
    assert artifact.apply(row).selected_horizon == 20


def test_aggregate_path_does_not_change_legacy_selector_default():
    legacy_before = hierarchical.select_horizon(_predictions(), calibration=_calibration())
    artifact = hierarchical.fit_aggregate_selector_calibration(
        _aggregate_predictions(40),
        _aggregate_labels(40),
        calibration=_calibration(),
        provenance=_aggregate_provenance(40),
        cluster_ids=np.arange(40),
        config=_aggregate_config(success_thresholds=(-0.01,)),
    )
    assert artifact.aggregate_gate_passed
    legacy_after = hierarchical.select_horizon(_predictions(), calibration=_calibration())

    assert legacy_before == legacy_after
    assert legacy_after.selected_horizon == 25


def test_aggregate_provenance_must_match_calibration_groups_and_valid_sha256():
    num_roots = 40
    with np.testing.assert_raises_regex(ValueError, "calibration_group_ids do not match"):
        hierarchical.fit_aggregate_selector_calibration(
            _aggregate_predictions(num_roots),
            _aggregate_labels(num_roots),
            calibration=_calibration(),
            provenance=_aggregate_provenance(num_roots),
            cluster_ids=np.arange(1, num_roots + 1),
            config=_aggregate_config(),
        )

    values = dataclasses.asdict(_aggregate_provenance(num_roots))
    values["params_digest"] = "NOT-A-DIGEST"
    with np.testing.assert_raises_regex(ValueError, "params_digest"):
        hierarchical.AggregateSelectorProvenance(**values)


def test_aggregate_rejects_impossible_or_fractional_paired_counts():
    num_roots = 40
    labels = _aggregate_labels(num_roots)
    labels["paired_trial_count"][:, 2:] = 100
    with np.testing.assert_raises_regex(ValueError, "does not match raw paired"):
        hierarchical.fit_aggregate_selector_calibration(
            _aggregate_predictions(num_roots),
            labels,
            calibration=_calibration(),
            provenance=_aggregate_provenance(num_roots),
            cluster_ids=np.arange(num_roots),
            config=_aggregate_config(),
        )

    labels = _aggregate_labels(num_roots)
    labels["dangerous_long_count"] = labels["dangerous_long_count"].astype(np.float64)
    labels["dangerous_long_count"][0, 2] = 0.5
    with np.testing.assert_raises_regex(ValueError, "integer counts"):
        hierarchical.fit_aggregate_selector_calibration(
            _aggregate_predictions(num_roots),
            labels,
            calibration=_calibration(),
            provenance=_aggregate_provenance(num_roots),
            cluster_ids=np.arange(num_roots),
            config=_aggregate_config(),
        )


def test_aggregate_danger_requires_probability_or_complete_categorical_logits():
    num_roots = 40
    predictions = _aggregate_predictions(num_roots)
    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        _aggregate_labels(num_roots),
        calibration=_calibration(),
        provenance=_aggregate_provenance(num_roots),
        cluster_ids=np.arange(num_roots),
        config=_aggregate_config(success_thresholds=(-0.01,)),
    )
    row = {name: value[0] for name, value in predictions.items()}
    row.pop("danger_probability")
    row["paired_outcome_logits"] = np.log(
        np.asarray(
            [
                [0.10, 0.80, 0.10],
                [0.04, 0.80, 0.16],
                [0.03, 0.80, 0.17],
            ]
        )
    )
    assert artifact.apply(row).selected_horizon == 20

    row.pop("paired_outcome_logits")
    row["danger_logits"] = np.full((3,), -8.0)
    with np.testing.assert_raises_regex(ValueError, "categorical danger logit"):
        artifact.apply(row)


def test_aggregate_rejects_short_or_nonfinite_hazard_and_invalid_probability():
    num_roots = 40
    predictions = _aggregate_predictions(num_roots)
    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        _aggregate_labels(num_roots),
        calibration=_calibration(),
        provenance=_aggregate_provenance(num_roots),
        cluster_ids=np.arange(num_roots),
        config=_aggregate_config(success_thresholds=(-0.01,)),
    )
    row = {name: value[0].copy() for name, value in predictions.items()}
    row["hazard_logits"] = row["hazard_logits"][:10]
    with np.testing.assert_raises_regex(ValueError, "cover every step"):
        artifact.apply(row)

    row = {name: value[0].copy() for name, value in predictions.items()}
    row["hazard_logits"][0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "hazard_logits must contain only finite"):
        artifact.apply(row)

    row = {name: value[0].copy() for name, value in predictions.items()}
    row["danger_probability"][0] = -0.1
    with np.testing.assert_raises_regex(ValueError, "finite probabilities"):
        artifact.apply(row)


def test_aggregate_config_cannot_weaken_preregistered_search():
    for updates, message in (
        ({"minimum_faster_probability": 0.49}, "pre-registered 0.50"),
        ({"candidate_selection_strategy": "largest_horizon"}, "minimum_predicted_elapsed"),
        ({"bootstrap_samples": 4_999}, "pre-registered 5000"),
    ):
        with np.testing.assert_raises_regex(ValueError, message):
            _aggregate_config(**updates)


def test_aggregate_frontier_rule_cannot_replace_uniquely_frozen_rule():
    num_roots = 40
    predictions = _aggregate_predictions(num_roots)
    artifact = hierarchical.fit_aggregate_selector_calibration(
        predictions,
        _aggregate_labels(num_roots),
        calibration=_calibration(),
        provenance=_aggregate_provenance(num_roots),
        cluster_ids=np.arange(num_roots),
        config=_aggregate_config(),
    )
    assert artifact.selected_rule is not None
    alternate = next(
        item.rule
        for item in artifact.search_evaluations
        if item.metrics.aggregate_gate_passed and item.rule != artifact.selected_rule
    )
    row = {name: value[0] for name, value in predictions.items()}
    assert any(artifact.evaluate_candidates(row, rule=alternate).long_eligible)
    with np.testing.assert_raises_regex(ValueError, "uniquely selected_rule"):
        artifact.apply_with_rule(row, alternate, aggregate_gate_passed=True)
