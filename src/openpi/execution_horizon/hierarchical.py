"""Calibrated hierarchical selection over parameterized execution horizons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class HierarchicalCalibration:
    candidate_horizons: tuple[int, ...]
    reference_horizon: int = 10
    confidence_level: float = 0.95
    hazard_temperature: float = 1.0
    success_temperature: float = 1.0
    success_residual_quantiles: tuple[float, ...] = ()
    elapsed_residual_quantiles: tuple[float, ...] = ()
    ood_feature_center: tuple[float, ...] = ()
    ood_feature_scale: tuple[float, ...] = ()
    ood_calibration_distances: tuple[float, ...] = ()
    # One disables the gate for legacy calibration artifacts without feature
    # statistics. New artifacts use a conformal feature-distance score.
    ood_probability_threshold: float = 1.0
    num_calibration_roots: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_horizons)
        object.__setattr__(self, "candidate_horizons", candidates)
        if candidates != tuple(sorted(set(candidates))) or self.reference_horizon not in candidates:
            raise ValueError("Calibration candidates must be sorted/unique and include reference_horizon.")
        long_count = sum(value > self.reference_horizon for value in candidates)
        if len(self.success_residual_quantiles) != long_count:
            raise ValueError("success_residual_quantiles must have one value per long horizon.")
        if len(self.elapsed_residual_quantiles) != long_count:
            raise ValueError("elapsed_residual_quantiles must have one value per long horizon.")
        for name, values in (
            ("success_residual_quantiles", self.success_residual_quantiles),
            ("elapsed_residual_quantiles", self.elapsed_residual_quantiles),
        ):
            if any(not np.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{name} must contain finite non-negative values.")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0.5, 1).")
        if self.hazard_temperature <= 0 or self.success_temperature <= 0:
            raise ValueError("Calibration temperatures must be positive.")
        if not 0 < self.ood_probability_threshold <= 1:
            raise ValueError("ood_probability_threshold must lie in (0, 1].")
        ood_fields_present = (
            bool(self.ood_feature_center),
            bool(self.ood_feature_scale),
            bool(self.ood_calibration_distances),
        )
        if any(ood_fields_present) and not all(ood_fields_present):
            raise ValueError("OOD center, scale, and calibration distances must be provided together.")
        if self.ood_feature_center:
            if len(self.ood_feature_center) != len(self.ood_feature_scale):
                raise ValueError("OOD feature center and scale must have equal widths.")
            if any(value <= 0 for value in self.ood_feature_scale):
                raise ValueError("Every OOD feature scale must be positive.")
            if not np.all(np.isfinite(np.asarray(self.ood_feature_center))):
                raise ValueError("OOD feature center must be finite.")
            if not np.all(np.isfinite(np.asarray(self.ood_feature_scale))):
                raise ValueError("OOD feature scale must be finite.")
            distances = np.asarray(self.ood_calibration_distances)
            if not np.all(np.isfinite(distances)) or np.any(distances < 0):
                raise ValueError("OOD calibration distances must be finite and non-negative.")

    @property
    def long_horizons(self) -> tuple[int, ...]:
        return tuple(value for value in self.candidate_horizons if value > self.reference_horizon)

    @property
    def has_ood_calibration(self) -> bool:
        return bool(self.ood_feature_center)

    @classmethod
    def load(cls, path: pathlib.Path | str) -> HierarchicalCalibration:
        return cls(**json.loads(pathlib.Path(path).read_text()))

    def save(self, path: pathlib.Path | str) -> pathlib.Path:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n")
        return target


@dataclasses.dataclass(frozen=True)
class HierarchicalSelectorConfig:
    success_noninferiority_margin: float = 0.01
    maximum_short_event_probability: float = 0.20
    maximum_long_event_probability: float = 0.20
    require_calibration_for_long_h: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.success_noninferiority_margin < 1:
            raise ValueError("success_noninferiority_margin must lie in [0, 1).")
        for name in ("maximum_short_event_probability", "maximum_long_event_probability"):
            value = float(getattr(self, name))
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie in (0, 1).")


DEFAULT_SELECTOR_CONFIG = HierarchicalSelectorConfig()


@dataclasses.dataclass(frozen=True)
class HierarchicalDecision:
    selected_horizon: int
    reason: str
    success_lcb: tuple[float, ...]
    elapsed_ucb: tuple[float, ...]
    long_eligible: tuple[bool, ...]
    long_event_probability: tuple[float, ...]
    short_event_probability: tuple[float, ...]
    ood_probability: float


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def select_horizon(
    predictions: Mapping[str, Any],
    *,
    calibration: HierarchicalCalibration | None,
    config: HierarchicalSelectorConfig = DEFAULT_SELECTOR_CONFIG,
) -> HierarchicalDecision:
    """Select one horizon from an unbatched transformer predictor output."""

    if "candidate_horizons" not in predictions:
        raise ValueError("Hierarchical selection requires transformer candidate_horizons output.")
    candidates = tuple(int(value) for value in np.asarray(predictions["candidate_horizons"]).reshape((-1,)))
    predicted_reference = int(np.asarray(predictions.get("reference_horizon", 10)).reshape(()))
    reference_horizon = calibration.reference_horizon if calibration is not None else predicted_reference
    if calibration is not None and candidates != calibration.candidate_horizons:
        raise ValueError(
            f"Predictor/calibration candidates differ: predictor={candidates}, "
            f"calibration={calibration.candidate_horizons}."
        )
    if calibration is not None and predicted_reference != calibration.reference_horizon:
        raise ValueError(
            "Predictor/calibration reference horizons differ: "
            f"predictor={predicted_reference}, calibration={calibration.reference_horizon}."
        )
    if reference_horizon not in candidates:
        raise ValueError("reference_horizon is absent from predictor candidates.")
    long_horizons = tuple(value for value in candidates if value > reference_horizon)
    short_horizons = tuple(value for value in candidates if value <= reference_horizon)

    hazard_logits = np.asarray(predictions["hazard_logits"], dtype=np.float64).reshape((-1,))
    temperature = calibration.hazard_temperature if calibration is not None else 1.0
    hazard = _sigmoid(hazard_logits / temperature)
    survival = np.cumprod(1.0 - hazard)

    def event_probability(horizons: tuple[int, ...]) -> np.ndarray:
        return np.asarray(
            [1.0 - survival[min(horizon, survival.size) - 1] for horizon in horizons],
            dtype=np.float64,
        )

    long_event_probability = event_probability(long_horizons)
    short_event_probability = event_probability(short_horizons)

    if calibration is not None and calibration.has_ood_calibration:
        feature = np.asarray(predictions["temporal_feature"], dtype=np.float64).reshape((-1,))
        center = np.asarray(calibration.ood_feature_center, dtype=np.float64)
        scale = np.asarray(calibration.ood_feature_scale, dtype=np.float64)
        if feature.shape != center.shape:
            raise ValueError(f"Predictor/calibration OOD feature widths differ: {feature.shape} versus {center.shape}.")
        distance = float(np.sqrt(np.mean(np.square((feature - center) / scale))))
        calibration_distances = np.asarray(calibration.ood_calibration_distances, dtype=np.float64)
        conformal_p_value = (1.0 + np.sum(calibration_distances >= distance)) / (calibration_distances.size + 1.0)
        ood_probability = float(1.0 - conformal_p_value)
    else:
        ood_probability = float(np.asarray(predictions.get("ood_probability", 0.0)).reshape(()))
    success_lcb = np.full((len(long_horizons),), -np.inf, dtype=np.float64)
    elapsed_ucb = np.full((len(long_horizons),), np.inf, dtype=np.float64)
    eligible = np.zeros((len(long_horizons),), dtype=np.bool_)
    calibration_allows_long = calibration is not None or not config.require_calibration_for_long_h
    if calibration_allows_long and long_horizons:
        success_mean = np.asarray(predictions["success_advantage"], dtype=np.float64).reshape((-1,))
        success_std = np.asarray(predictions["success_advantage_std"], dtype=np.float64).reshape((-1,))
        elapsed_mean = np.asarray(predictions["elapsed_advantage"], dtype=np.float64).reshape((-1,))
        elapsed_std = np.asarray(predictions["elapsed_advantage_std"], dtype=np.float64).reshape((-1,))
        expected_long_width = len(long_horizons)
        if any(values.size != expected_long_width for values in (success_mean, success_std, elapsed_mean, elapsed_std)):
            raise ValueError("Predictor long-H head widths do not match candidate_horizons.")
        if calibration is None:
            success_quantile = np.full_like(success_mean, 1.96)
            elapsed_quantile = np.full_like(elapsed_mean, 1.96)
            ood_threshold = 1.0
        else:
            success_quantile = np.asarray(calibration.success_residual_quantiles, dtype=np.float64)
            elapsed_quantile = np.asarray(calibration.elapsed_residual_quantiles, dtype=np.float64)
            ood_threshold = calibration.ood_probability_threshold
        success_lcb = success_mean - success_quantile * np.maximum(success_std, 1e-6)
        elapsed_ucb = elapsed_mean + elapsed_quantile * np.maximum(elapsed_std, 1e-6)
        eligible = (success_lcb >= -config.success_noninferiority_margin) & (elapsed_ucb < 0.0)
        eligible &= long_event_probability <= config.maximum_long_event_probability
        eligible &= ood_probability < ood_threshold
        if np.any(eligible):
            selected = long_horizons[int(np.flatnonzero(eligible)[-1])]
            return HierarchicalDecision(
                selected_horizon=selected,
                reason="calibrated_long_h",
                success_lcb=tuple(float(value) for value in success_lcb),
                elapsed_ucb=tuple(float(value) for value in elapsed_ucb),
                long_eligible=tuple(bool(value) for value in eligible),
                long_event_probability=tuple(float(value) for value in long_event_probability),
                short_event_probability=tuple(float(value) for value in short_event_probability),
                ood_probability=ood_probability,
            )

    short_eligible = short_event_probability <= config.maximum_short_event_probability
    if np.any(short_eligible):
        selected = short_horizons[int(np.flatnonzero(short_eligible)[-1])]
        reason = "default_reference_h" if selected == reference_horizon else "hazard_short_gate"
    else:
        selected = short_horizons[0]
        reason = "hazard_minimum_h"
    if calibration is None and config.require_calibration_for_long_h and selected == reference_horizon:
        reason = "uncalibrated_fallback_reference"
    return HierarchicalDecision(
        selected_horizon=selected,
        reason=reason,
        success_lcb=tuple(float(value) for value in success_lcb),
        elapsed_ucb=tuple(float(value) for value in elapsed_ucb),
        long_eligible=tuple(bool(value) for value in eligible),
        long_event_probability=tuple(float(value) for value in long_event_probability),
        short_event_probability=tuple(float(value) for value in short_event_probability),
        ood_probability=ood_probability,
    )


def fit_temperature(
    logits: np.ndarray,
    success_count: np.ndarray,
    trial_count: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> float:
    """Fit one positive temperature by deterministic count-weighted NLL search."""

    logits = np.asarray(logits, dtype=np.float64)
    successes = np.asarray(success_count, dtype=np.float64)
    trials = np.asarray(trial_count, dtype=np.float64)
    mask = trials > 0
    if valid is not None:
        mask &= np.asarray(valid, dtype=np.bool_)
    if not np.any(mask):
        raise ValueError("Temperature fitting received no valid count labels.")
    temperatures = np.exp(np.linspace(np.log(0.05), np.log(20.0), 401))
    scaled = logits[mask][None, :] / temperatures[:, None]
    losses = np.sum(
        trials[mask][None, :] * np.logaddexp(0.0, scaled) - successes[mask][None, :] * scaled,
        axis=-1,
    ) / np.sum(trials[mask])
    return float(temperatures[int(np.argmin(losses))])


def conformal_quantile(values: np.ndarray, confidence_level: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Conformal calibration received no finite residuals.")
    rank = min(values.size - 1, int(np.ceil((values.size + 1) * confidence_level)) - 1)
    # A negative one-sided residual would make the interval narrower than the
    # predictive mean. Clamp at zero for conservative deployment.
    return max(0.0, float(np.partition(values, rank)[rank]))


def fit_calibration(
    predictions: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    candidate_horizons: Sequence[int],
    reference_horizon: int = 10,
    confidence_level: float = 0.95,
    training_features: np.ndarray | None = None,
    ood_probability_threshold: float = 0.95,
) -> HierarchicalCalibration:
    candidates = tuple(int(value) for value in candidate_horizons)
    reference_index = candidates.index(reference_horizon)
    long_indices = [index for index, value in enumerate(candidates) if value > reference_horizon]
    success_temperature = fit_temperature(
        predictions["success_logits"],
        labels["success_count"],
        labels["trial_count"],
    )
    if "hazard_event_count" in labels and "hazard_at_risk_count" in labels:
        hazard_temperature = fit_temperature(
            predictions["hazard_logits"],
            labels["hazard_event_count"],
            labels["hazard_at_risk_count"],
        )
    else:
        hazard_event = np.asarray(labels["event_mask"], dtype=np.bool_)
        prior_event = np.concatenate(
            [np.zeros_like(hazard_event[:, :1]), np.cumsum(hazard_event[:, :-1], axis=-1)],
            axis=-1,
        )
        hazard_valid = (prior_event == 0) & np.asarray(labels["risk_valid"], dtype=np.bool_)
        hazard_temperature = fit_temperature(
            predictions["hazard_logits"],
            hazard_event.astype(np.float32),
            np.ones_like(hazard_event, dtype=np.float32),
            valid=hazard_valid,
        )
    trials = np.asarray(labels["trial_count"], dtype=np.float64)
    rates = np.asarray(labels["success_count"], dtype=np.float64) / np.maximum(trials, 1.0)
    success_target = rates[:, long_indices] - rates[:, reference_index : reference_index + 1]
    elapsed = np.asarray(labels["elapsed_mean"], dtype=np.float64)
    elapsed_target = elapsed[:, long_indices] - elapsed[:, reference_index : reference_index + 1]
    success_mean = np.asarray(predictions["success_advantage"], dtype=np.float64)
    success_std = np.maximum(np.asarray(predictions["success_advantage_std"], dtype=np.float64), 1e-6)
    elapsed_mean = np.asarray(predictions["elapsed_advantage"], dtype=np.float64)
    elapsed_std = np.maximum(np.asarray(predictions["elapsed_advantage_std"], dtype=np.float64), 1e-6)
    success_quantiles = []
    elapsed_quantiles = []
    for position, candidate_index in enumerate(long_indices):
        valid = (trials[:, reference_index] > 0) & (trials[:, candidate_index] > 0)
        success_quantiles.append(
            conformal_quantile(
                ((success_mean[:, position] - success_target[:, position]) / success_std[:, position])[valid],
                confidence_level,
            )
        )
        elapsed_quantiles.append(
            conformal_quantile(
                ((elapsed_target[:, position] - elapsed_mean[:, position]) / elapsed_std[:, position])[valid],
                confidence_level,
            )
        )
    feature_center: tuple[float, ...] = ()
    feature_scale: tuple[float, ...] = ()
    calibration_distances: tuple[float, ...] = ()
    if training_features is not None:
        training_features = np.asarray(training_features, dtype=np.float64)
        calibration_features = np.asarray(predictions["temporal_feature"], dtype=np.float64)
        if training_features.ndim != 2 or calibration_features.ndim != 2:
            raise ValueError("OOD calibration features must have shape [roots, feature_dim].")
        if training_features.shape[1] != calibration_features.shape[1]:
            raise ValueError("Training/calibration OOD feature widths differ.")
        center = np.median(training_features, axis=0)
        scale = 1.4826 * np.median(np.abs(training_features - center), axis=0)
        fallback_scale = np.std(training_features, axis=0)
        scale = np.maximum(np.where(scale > 1e-6, scale, fallback_scale), 1e-3)
        distances = np.sqrt(np.mean(np.square((calibration_features - center) / scale), axis=-1))
        feature_center = tuple(float(value) for value in center)
        feature_scale = tuple(float(value) for value in scale)
        calibration_distances = tuple(float(value) for value in np.sort(distances))
    return HierarchicalCalibration(
        candidate_horizons=candidates,
        reference_horizon=reference_horizon,
        confidence_level=confidence_level,
        hazard_temperature=hazard_temperature,
        success_temperature=success_temperature,
        success_residual_quantiles=tuple(success_quantiles),
        elapsed_residual_quantiles=tuple(elapsed_quantiles),
        ood_feature_center=feature_center,
        ood_feature_scale=feature_scale,
        ood_calibration_distances=calibration_distances,
        ood_probability_threshold=ood_probability_threshold if feature_center else 1.0,
        num_calibration_roots=int(rates.shape[0]),
    )
