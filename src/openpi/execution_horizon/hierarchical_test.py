from __future__ import annotations

import dataclasses

import numpy as np

from openpi.execution_horizon import hierarchical


def _predictions() -> dict[str, np.ndarray]:
    return {
        "candidate_horizons": np.asarray([3, 5, 7, 10, 15, 20]),
        "reference_horizon": np.asarray(10),
        "hazard_logits": np.full((20,), -8.0),
        "success_advantage": np.asarray([0.0, 0.01]),
        "success_advantage_std": np.asarray([0.002, 0.002]),
        "elapsed_advantage": np.asarray([-0.5, -1.0]),
        "elapsed_advantage_std": np.asarray([0.05, 0.05]),
        "ood_probability": np.asarray(0.1),
        "temporal_feature": np.asarray([0.0, 0.0]),
    }


def _calibration() -> hierarchical.HierarchicalCalibration:
    return hierarchical.HierarchicalCalibration(
        candidate_horizons=(3, 5, 7, 10, 15, 20),
        success_residual_quantiles=(2.0, 2.0),
        elapsed_residual_quantiles=(2.0, 2.0),
        num_calibration_roots=100,
    )


def test_selector_uses_largest_calibrated_safe_long_horizon():
    decision = hierarchical.select_horizon(_predictions(), calibration=_calibration())

    assert decision.selected_horizon == 20
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
    predictions["success_advantage"] = np.asarray([-0.5, -0.5])
    predictions["hazard_logits"] = np.concatenate([np.full((4,), -8.0), np.full((16,), 8.0)])

    decision = hierarchical.select_horizon(predictions, calibration=_calibration())

    assert decision.selected_horizon == 3
    assert decision.reason == "hazard_short_gate"


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
        "success_logits": np.zeros((4, 6)),
        "hazard_logits": np.zeros((4, 20)),
        "success_advantage": np.zeros((4, 2)),
        "success_advantage_std": np.ones((4, 2)),
        "elapsed_advantage": np.zeros((4, 2)),
        "elapsed_advantage_std": np.ones((4, 2)),
        "temporal_feature": np.zeros((4, 2)),
    }
    labels = {
        "success_count": np.asarray([[3, 3, 3, 3, 2, 2]] * 4),
        "trial_count": np.full((4, 6), 3),
        "elapsed_mean": np.asarray([[8, 7, 6, 5, 4, 3]] * 4, dtype=np.float32),
        "event_mask": np.zeros((4, 20), dtype=np.bool_),
        "risk_valid": np.ones((4, 20), dtype=np.bool_),
        "hazard_event_count": np.zeros((4, 20), dtype=np.int64),
        "hazard_at_risk_count": np.full((4, 20), 3, dtype=np.int64),
    }

    artifact = hierarchical.fit_calibration(
        predictions,
        labels,
        candidate_horizons=(3, 5, 7, 10, 15, 20),
        training_features=np.zeros((8, 2)),
    )

    assert np.isfinite(artifact.hazard_temperature)
    assert np.isfinite(artifact.success_temperature)
    assert len(artifact.success_residual_quantiles) == 2
    assert artifact.has_ood_calibration
