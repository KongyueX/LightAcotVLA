"""Calibrated hierarchical selection over parameterized execution horizons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import json
import pathlib
from statistics import NormalDist
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


@dataclasses.dataclass(frozen=True)
class AggregateSelectorCalibrationConfig:
    """Pre-registered, tightening-only search space for an aggregate selector.

    These thresholds are applied to predictor point estimates.  Their safety is
    established by the aggregate calibration gates below, rather than by
    requiring a simultaneous 95% interval to pass independently at every root.
    """

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

    def __post_init__(self) -> None:
        for name in (
            "success_thresholds",
            "elapsed_margins",
            "maximum_danger_probabilities",
            "maximum_hazard_probabilities",
            "maximum_ood_probabilities",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)
            if not values or len(values) != len(set(values)) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a non-empty finite grid without duplicates.")
        faster_options = tuple(bool(value) for value in self.require_faster_options)
        object.__setattr__(self, "require_faster_options", faster_options)
        if not faster_options or len(faster_options) != len(set(faster_options)):
            raise ValueError("require_faster_options must be a non-empty grid without duplicates.")
        if self.success_thresholds != tuple(sorted(self.success_thresholds)):
            raise ValueError("success_thresholds must run from the least to the most strict value.")
        if self.elapsed_margins != tuple(sorted(self.elapsed_margins)):
            raise ValueError("elapsed_margins must run from the least to the most strict value.")
        for name in (
            "maximum_danger_probabilities",
            "maximum_hazard_probabilities",
            "maximum_ood_probabilities",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(values, reverse=True)):
                raise ValueError(f"{name} must run from the least to the most strict value.")
        if faster_options != tuple(sorted(faster_options)):
            raise ValueError("require_faster_options must run from False to True.")
        if not 0 <= self.success_noninferiority_margin <= 0.01:
            raise ValueError("success_noninferiority_margin may only tighten the pre-registered 0.01 gate.")
        if self.success_thresholds[0] < -0.01:
            raise ValueError("success_thresholds may only tighten the pre-registered -0.01 threshold.")
        if self.elapsed_margins[0] < 0:
            raise ValueError("elapsed_margins may only tighten the zero-second deployment margin.")
        if any(not 0 <= value <= 0.20 for value in self.maximum_danger_probabilities):
            raise ValueError("maximum_danger_probabilities may only tighten the pre-registered 0.20 gate.")
        if any(not 0 < value <= 0.20 for value in self.maximum_hazard_probabilities):
            raise ValueError("maximum_hazard_probabilities may only tighten the 0.20 hazard gate.")
        if any(not 0 < value <= 0.95 for value in self.maximum_ood_probabilities):
            raise ValueError("maximum_ood_probabilities may only tighten the 0.95 OOD gate.")
        if not 0.50 <= self.minimum_faster_probability <= 1:
            raise ValueError(
                "minimum_faster_probability may only tighten the pre-registered 0.50 gate."
            )
        if self.candidate_selection_strategy != "minimum_predicted_elapsed":
            raise ValueError("Aggregate calibration must use minimum_predicted_elapsed selection.")
        if not 0.95 <= self.confidence_level < 1:
            raise ValueError("confidence_level may not be lower than the pre-registered 0.95 level.")
        if not 0 <= self.false_long_upper_bound <= 0.05:
            raise ValueError("false_long_upper_bound may only tighten the pre-registered 0.05 gate.")
        if self.bootstrap_samples < 5_000:
            raise ValueError("bootstrap_samples may not be lower than the pre-registered 5000 samples.")
        if not isinstance(self.bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer.")


DEFAULT_AGGREGATE_SELECTOR_CALIBRATION_CONFIG = AggregateSelectorCalibrationConfig()
AGGREGATE_SELECTOR_SCHEMA_VERSION = 2


@dataclasses.dataclass(frozen=True)
class AggregateSelectorProvenance:
    """Immutable identities needed to replay a frozen selector safely."""

    predictor_config_digest: str
    params_digest: str
    pointwise_calibration_digest: str
    split_manifest_digest: str
    development_dataset_fingerprint: str
    calibration_group_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "predictor_config_digest",
            "params_digest",
            "pointwise_calibration_digest",
            "split_manifest_digest",
            "development_dataset_fingerprint",
        ):
            value = str(getattr(self, name))
            object.__setattr__(self, name, value)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest.")
        groups = tuple(int(value) for value in self.calibration_group_ids)
        object.__setattr__(self, "calibration_group_ids", groups)
        if not groups or groups != tuple(sorted(set(groups))) or any(value < 0 for value in groups):
            raise ValueError("calibration_group_ids must be a non-empty sorted unique tuple of non-negative IDs.")


@dataclasses.dataclass(frozen=True)
class AggregateSelectorRule:
    success_threshold: float
    elapsed_margin: float
    maximum_danger_probability: float
    require_faster: bool
    minimum_faster_probability: float
    maximum_hazard_probability: float
    maximum_ood_probability: float
    candidate_selection_strategy: str = "minimum_predicted_elapsed"

    def __post_init__(self) -> None:
        numeric = (
            self.success_threshold,
            self.elapsed_margin,
            self.maximum_danger_probability,
            self.minimum_faster_probability,
            self.maximum_hazard_probability,
            self.maximum_ood_probability,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Every aggregate selector rule threshold must be finite.")
        if self.success_threshold < -0.01:
            raise ValueError("Aggregate success threshold may not be lower than -0.01.")
        if self.elapsed_margin < 0:
            raise ValueError("Aggregate elapsed margin must be non-negative.")
        if not 0 <= self.maximum_danger_probability <= 0.20:
            raise ValueError("Aggregate danger threshold must lie in [0, 0.20].")
        if not 0.50 <= self.minimum_faster_probability <= 1:
            raise ValueError("Aggregate faster threshold may not weaken the pre-registered 0.50 gate.")
        if not 0 < self.maximum_hazard_probability <= 0.20:
            raise ValueError("Aggregate hazard threshold must lie in (0, 0.20].")
        if not 0 < self.maximum_ood_probability <= 0.95:
            raise ValueError("Aggregate OOD threshold must lie in (0, 0.95].")
        if self.candidate_selection_strategy != "minimum_predicted_elapsed":
            raise ValueError("Aggregate rules must use minimum_predicted_elapsed selection.")


@dataclasses.dataclass(frozen=True)
class AggregateSelectorMetrics:
    long_coverage: float
    selected_long_roots: int
    success_advantage_mean: float
    success_advantage_lcb: float
    elapsed_advantage_mean: float
    elapsed_advantage_ucb: float
    false_long_rate: float
    false_long_upper: float
    false_long_paired_trials: float
    aggregate_gate_passed: bool
    selected_h_distribution: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        distribution = tuple((int(horizon), int(count)) for horizon, count in self.selected_h_distribution)
        object.__setattr__(self, "selected_h_distribution", distribution)


@dataclasses.dataclass(frozen=True)
class AggregateRuleEvaluation:
    rule: AggregateSelectorRule
    metrics: AggregateSelectorMetrics


@dataclasses.dataclass(frozen=True)
class AggregateCandidateEvaluation:
    long_horizons: tuple[int, ...]
    success_score: tuple[float, ...]
    elapsed_score: tuple[float, ...]
    danger_probability: tuple[float, ...]
    faster_probability: tuple[float, ...]
    long_event_probability: tuple[float, ...]
    short_event_probability: tuple[float, ...]
    ood_probability: float
    success_pass: tuple[bool, ...]
    elapsed_pass: tuple[bool, ...]
    danger_pass: tuple[bool, ...]
    faster_pass: tuple[bool, ...]
    hazard_pass: tuple[bool, ...]
    ood_pass: tuple[bool, ...]
    long_eligible: tuple[bool, ...]
    rejection_reasons: tuple[tuple[str, ...], ...]

    @property
    def constraint_pass(self) -> dict[str, tuple[bool, ...]]:
        return {
            "success": self.success_pass,
            "elapsed": self.elapsed_pass,
            "danger": self.danger_pass,
            "faster": self.faster_pass,
            "hazard": self.hazard_pass,
            "ood": self.ood_pass,
        }


@dataclasses.dataclass(frozen=True)
class AggregateSelectorDecision:
    selected_horizon: int
    reason: str
    long_horizons: tuple[int, ...]
    success_score: tuple[float, ...]
    elapsed_score: tuple[float, ...]
    danger_probability: tuple[float, ...]
    faster_probability: tuple[float, ...]
    long_event_probability: tuple[float, ...]
    short_event_probability: tuple[float, ...]
    ood_probability: float
    success_pass: tuple[bool, ...]
    elapsed_pass: tuple[bool, ...]
    danger_pass: tuple[bool, ...]
    faster_pass: tuple[bool, ...]
    hazard_pass: tuple[bool, ...]
    ood_pass: tuple[bool, ...]
    long_eligible: tuple[bool, ...]
    rejection_reasons: tuple[tuple[str, ...], ...]

    @property
    def constraint_pass(self) -> dict[str, tuple[bool, ...]]:
        return {
            "success": self.success_pass,
            "elapsed": self.elapsed_pass,
            "danger": self.danger_pass,
            "faster": self.faster_pass,
            "hazard": self.hazard_pass,
            "ood": self.ood_pass,
        }


@dataclasses.dataclass(frozen=True)
class AggregateSelectorCalibration:
    """Frozen aggregate-risk selector fitted only on a calibration split."""

    candidate_horizons: tuple[int, ...]
    reference_horizon: int
    pointwise_calibration: HierarchicalCalibration
    provenance: AggregateSelectorProvenance
    search_config: AggregateSelectorCalibrationConfig
    selected_rule: AggregateSelectorRule | None
    calibration_metrics: AggregateSelectorMetrics | None
    search_evaluations: tuple[AggregateRuleEvaluation, ...]
    num_rules_evaluated: int
    num_feasible_rules: int
    num_unique_selections: int
    source_split: str = "calibration"
    schema_version: int = AGGREGATE_SELECTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_horizons)
        object.__setattr__(self, "candidate_horizons", candidates)
        object.__setattr__(self, "search_evaluations", tuple(self.search_evaluations))
        if (
            candidates != tuple(sorted(set(candidates)))
            or any(value <= 0 for value in candidates)
            or self.reference_horizon not in candidates
        ):
            raise ValueError("Aggregate selector candidates must be sorted/unique and include reference_horizon.")
        if candidates != self.pointwise_calibration.candidate_horizons:
            raise ValueError("Aggregate and pointwise calibration candidates differ.")
        if self.reference_horizon != self.pointwise_calibration.reference_horizon:
            raise ValueError("Aggregate and pointwise calibration reference horizons differ.")
        if not isinstance(self.provenance, AggregateSelectorProvenance):
            raise TypeError("Aggregate selector provenance must be an AggregateSelectorProvenance instance.")
        if self.source_split != "calibration":
            raise ValueError("Aggregate selector rules may only be fitted on the calibration split.")
        if self.schema_version != AGGREGATE_SELECTOR_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported aggregate selector schema_version {self.schema_version}; "
                f"expected {AGGREGATE_SELECTOR_SCHEMA_VERSION}."
            )
        if (self.selected_rule is None) != (self.calibration_metrics is None):
            raise ValueError("selected_rule and calibration_metrics must either both be present or both be absent.")
        if self.selected_rule is not None and not self.calibration_metrics.aggregate_gate_passed:
            raise ValueError("A selected aggregate rule must pass every aggregate calibration gate.")
        expected_rules = _aggregate_rule_grid(self.search_config)
        evaluation_rules = tuple(item.rule for item in self.search_evaluations)
        if evaluation_rules != expected_rules:
            raise ValueError("Aggregate search evaluations do not match the pre-registered rule grid.")
        if self.num_rules_evaluated != len(self.search_evaluations):
            raise ValueError("num_rules_evaluated does not match search_evaluations.")
        feasible_count = sum(item.metrics.aggregate_gate_passed for item in self.search_evaluations)
        if self.num_feasible_rules != feasible_count:
            raise ValueError("num_feasible_rules does not match search_evaluations.")
        if not 0 < self.num_unique_selections <= max(1, self.num_rules_evaluated):
            raise ValueError("num_unique_selections is inconsistent with the search grid.")
        if self.selected_rule is not None:
            selected_matches = [item for item in self.search_evaluations if item.rule == self.selected_rule]
            if not selected_matches or all(item.metrics != self.calibration_metrics for item in selected_matches):
                raise ValueError("Selected aggregate rule/metrics are absent from search_evaluations.")

    @property
    def aggregate_gate_passed(self) -> bool:
        return self.selected_rule is not None

    @property
    def long_horizons(self) -> tuple[int, ...]:
        return tuple(value for value in self.candidate_horizons if value > self.reference_horizon)

    @property
    def fallback_rule(self) -> AggregateSelectorRule:
        config = self.search_config
        return AggregateSelectorRule(
            success_threshold=max(config.success_thresholds),
            elapsed_margin=max(config.elapsed_margins),
            maximum_danger_probability=min(config.maximum_danger_probabilities),
            require_faster=True in config.require_faster_options,
            minimum_faster_probability=config.minimum_faster_probability,
            maximum_hazard_probability=min(config.maximum_hazard_probabilities),
            maximum_ood_probability=min(config.maximum_ood_probabilities),
            candidate_selection_strategy=config.candidate_selection_strategy,
        )

    def evaluate_candidates(
        self,
        predictions: Mapping[str, Any],
        *,
        rule: AggregateSelectorRule | None = None,
    ) -> AggregateCandidateEvaluation:
        return _evaluate_aggregate_candidates(
            predictions,
            calibration=self.pointwise_calibration,
            rule=rule or self.selected_rule or self.fallback_rule,
        )

    def apply(self, predictions: Mapping[str, Any]) -> AggregateSelectorDecision:
        rule = self.selected_rule or self.fallback_rule
        evaluation = self.evaluate_candidates(predictions, rule=rule)
        return _aggregate_decision(
            predictions,
            calibration=self.pointwise_calibration,
            rule=rule,
            evaluation=evaluation,
            aggregate_gate_passed=self.aggregate_gate_passed,
        )

    def apply_with_rule(
        self,
        predictions: Mapping[str, Any],
        rule: AggregateSelectorRule,
        *,
        aggregate_gate_passed: bool = False,
    ) -> AggregateSelectorDecision:
        """Apply only the uniquely frozen deployment rule; frontier rules remain diagnostic."""

        if self.selected_rule is None or rule != self.selected_rule:
            raise ValueError("Only the aggregate artifact's uniquely selected_rule may be deployed.")
        if aggregate_gate_passed and not self.aggregate_gate_passed:
            raise ValueError("Cannot enable long horizons without a passing aggregate calibration gate.")
        evaluation = self.evaluate_candidates(predictions, rule=rule)
        return _aggregate_decision(
            predictions,
            calibration=self.pointwise_calibration,
            rule=rule,
            evaluation=evaluation,
            aggregate_gate_passed=aggregate_gate_passed,
        )

    @classmethod
    def load(cls, path: pathlib.Path | str) -> AggregateSelectorCalibration:
        values = json.loads(pathlib.Path(path).read_text())
        pointwise_values = values["pointwise_calibration"]
        for name in (
            "candidate_horizons",
            "success_residual_quantiles",
            "elapsed_residual_quantiles",
            "ood_feature_center",
            "ood_feature_scale",
            "ood_calibration_distances",
        ):
            pointwise_values[name] = tuple(pointwise_values[name])
        values["pointwise_calibration"] = HierarchicalCalibration(**pointwise_values)
        provenance_values = values["provenance"]
        provenance_values["calibration_group_ids"] = tuple(provenance_values["calibration_group_ids"])
        values["provenance"] = AggregateSelectorProvenance(**provenance_values)
        values["search_config"] = AggregateSelectorCalibrationConfig(**values["search_config"])
        if values["selected_rule"] is not None:
            values["selected_rule"] = AggregateSelectorRule(**values["selected_rule"])
        if values["calibration_metrics"] is not None:
            values["calibration_metrics"] = AggregateSelectorMetrics(**values["calibration_metrics"])
        values["search_evaluations"] = tuple(
            AggregateRuleEvaluation(
                rule=AggregateSelectorRule(**item["rule"]),
                metrics=AggregateSelectorMetrics(**item["metrics"]),
            )
            for item in values["search_evaluations"]
        )
        return cls(**values)

    def save(self, path: pathlib.Path | str) -> pathlib.Path:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n")
        return target


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _aggregate_ood_probability(
    predictions: Mapping[str, Any],
    calibration: HierarchicalCalibration,
) -> float:
    if calibration.has_ood_calibration:
        if "temporal_feature" not in predictions:
            return 1.0
        feature = np.asarray(predictions["temporal_feature"], dtype=np.float64).reshape((-1,))
        center = np.asarray(calibration.ood_feature_center, dtype=np.float64)
        scale = np.asarray(calibration.ood_feature_scale, dtype=np.float64)
        if feature.shape != center.shape or not np.all(np.isfinite(feature)):
            return 1.0
        distance = float(np.sqrt(np.mean(np.square((feature - center) / scale))))
        distances = np.asarray(calibration.ood_calibration_distances, dtype=np.float64)
        conformal_p_value = (1.0 + np.sum(distances >= distance)) / (distances.size + 1.0)
        return float(1.0 - conformal_p_value)
    value = float(np.asarray(predictions.get("ood_probability", 1.0), dtype=np.float64).reshape(()))
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("ood_probability must be a finite probability in [0, 1].")
    return value


def _aggregate_probability_head(
    predictions: Mapping[str, Any],
    probability_name: str,
    logit_name: str,
    width: int,
    *,
    missing_value: float,
) -> np.ndarray:
    if probability_name in predictions:
        values = np.asarray(predictions[probability_name], dtype=np.float64).reshape((-1,))
    elif logit_name in predictions:
        logits = np.asarray(predictions[logit_name], dtype=np.float64).reshape((-1,))
        if np.any(~np.isfinite(logits)):
            raise ValueError(f"{logit_name} must contain only finite values.")
        values = _sigmoid(logits)
    else:
        values = np.full((width,), missing_value, dtype=np.float64)
    if values.size != width:
        raise ValueError(f"{probability_name} width does not match the number of long horizons.")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{probability_name} must contain only finite probabilities in [0, 1].")
    return values


def _aggregate_danger_probability(predictions: Mapping[str, Any], width: int) -> np.ndarray:
    if "danger_probability" in predictions:
        values = np.asarray(predictions["danger_probability"], dtype=np.float64).reshape((-1,))
    elif "paired_outcome_logits" in predictions:
        logits = np.asarray(predictions["paired_outcome_logits"], dtype=np.float64)
        if logits.shape != (width, 3):
            raise ValueError("paired_outcome_logits must have shape [long_horizons, 3].")
        if np.any(~np.isfinite(logits)):
            raise ValueError("paired_outcome_logits must contain only finite values.")
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponential = np.exp(shifted)
        values = exponential[:, 0] / np.sum(exponential, axis=-1)
    else:
        raise ValueError(
            "Aggregate selection requires danger_probability or complete paired_outcome_logits; "
            "a categorical danger logit cannot be sigmoid-calibrated independently."
        )
    if values.size != width:
        raise ValueError("danger_probability width does not match the number of long horizons.")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("danger_probability must contain only finite probabilities in [0, 1].")
    return values


def _evaluate_aggregate_candidates(
    predictions: Mapping[str, Any],
    *,
    calibration: HierarchicalCalibration,
    rule: AggregateSelectorRule,
) -> AggregateCandidateEvaluation:
    if "candidate_horizons" not in predictions:
        raise ValueError("Aggregate selection requires predictor candidate_horizons output.")
    candidates = tuple(int(value) for value in np.asarray(predictions["candidate_horizons"]).reshape((-1,)))
    reference = int(np.asarray(predictions.get("reference_horizon", calibration.reference_horizon)).reshape(()))
    if candidates != calibration.candidate_horizons or reference != calibration.reference_horizon:
        raise ValueError("Predictor and aggregate calibration horizon contracts differ.")
    long_horizons = calibration.long_horizons
    if "long_horizons" in predictions:
        predicted_long = tuple(int(value) for value in np.asarray(predictions["long_horizons"]).reshape((-1,)))
        if predicted_long != long_horizons:
            raise ValueError("Predictor and aggregate calibration long-horizon ordering differs.")
    short_horizons = tuple(value for value in candidates if value <= reference)
    long_width = len(long_horizons)
    success_score = np.asarray(
        predictions.get("success_advantage", np.full((long_width,), -np.inf)),
        dtype=np.float64,
    ).reshape((-1,))
    elapsed_score = np.asarray(
        predictions.get("elapsed_advantage", np.full((long_width,), np.inf)),
        dtype=np.float64,
    ).reshape((-1,))
    if success_score.size != long_width or elapsed_score.size != long_width:
        raise ValueError("Predictor advantage widths do not match the number of long horizons.")
    danger_probability = _aggregate_danger_probability(predictions, long_width)
    faster_probability = _aggregate_probability_head(
        predictions,
        "faster_long_probability",
        "faster_long_logits",
        long_width,
        missing_value=0.0,
    )
    if "hazard_logits" not in predictions:
        raise ValueError("Aggregate selection requires hazard_logits.")
    hazard_logits = np.asarray(predictions["hazard_logits"], dtype=np.float64).reshape((-1,))
    if hazard_logits.size < max(candidates):
        raise ValueError("hazard_logits must cover every step through max(candidate_horizons).")
    if np.any(~np.isfinite(hazard_logits)):
        raise ValueError("hazard_logits must contain only finite values.")
    hazard = _sigmoid(hazard_logits / calibration.hazard_temperature)
    if np.any(~np.isfinite(hazard)) or np.any((hazard < 0.0) | (hazard > 1.0)):
        raise ValueError("Calibrated hazard values must be finite probabilities in [0, 1].")
    survival = np.cumprod(1.0 - hazard)

    def event_probability(horizons: tuple[int, ...]) -> np.ndarray:
        return np.asarray(
            [1.0 - survival[horizon - 1] for horizon in horizons],
            dtype=np.float64,
        )

    long_event_probability = event_probability(long_horizons)
    short_event_probability = event_probability(short_horizons)
    ood_probability = _aggregate_ood_probability(predictions, calibration)
    success_pass = np.isfinite(success_score) & (success_score >= rule.success_threshold)
    elapsed_pass = np.isfinite(elapsed_score) & (elapsed_score < -rule.elapsed_margin)
    danger_pass = danger_probability <= rule.maximum_danger_probability
    faster_pass = (
        faster_probability >= 0.0
        if not rule.require_faster
        else faster_probability >= rule.minimum_faster_probability
    )
    hazard_pass = long_event_probability <= rule.maximum_hazard_probability
    ood_pass = np.full((long_width,), ood_probability < rule.maximum_ood_probability, dtype=np.bool_)
    eligible = success_pass & elapsed_pass & danger_pass & faster_pass & hazard_pass & ood_pass
    names_and_masks = (
        ("success", success_pass),
        ("elapsed", elapsed_pass),
        ("danger", danger_pass),
        ("faster", faster_pass),
        ("hazard", hazard_pass),
        ("ood", ood_pass),
    )
    rejection_reasons = tuple(
        tuple(name for name, mask in names_and_masks if not bool(mask[position]))
        for position in range(long_width)
    )
    return AggregateCandidateEvaluation(
        long_horizons=long_horizons,
        success_score=tuple(float(value) for value in success_score),
        elapsed_score=tuple(float(value) for value in elapsed_score),
        danger_probability=tuple(float(value) for value in danger_probability),
        faster_probability=tuple(float(value) for value in faster_probability),
        long_event_probability=tuple(float(value) for value in long_event_probability),
        short_event_probability=tuple(float(value) for value in short_event_probability),
        ood_probability=ood_probability,
        success_pass=tuple(bool(value) for value in success_pass),
        elapsed_pass=tuple(bool(value) for value in elapsed_pass),
        danger_pass=tuple(bool(value) for value in danger_pass),
        faster_pass=tuple(bool(value) for value in faster_pass),
        hazard_pass=tuple(bool(value) for value in hazard_pass),
        ood_pass=tuple(bool(value) for value in ood_pass),
        long_eligible=tuple(bool(value) for value in eligible),
        rejection_reasons=rejection_reasons,
    )


def _aggregate_decision(
    predictions: Mapping[str, Any],
    *,
    calibration: HierarchicalCalibration,
    rule: AggregateSelectorRule,
    evaluation: AggregateCandidateEvaluation,
    aggregate_gate_passed: bool,
) -> AggregateSelectorDecision:
    del predictions  # Candidate diagnostics already validate the prediction contract.
    eligible = np.asarray(evaluation.long_eligible, dtype=np.bool_)
    reasons = evaluation.rejection_reasons
    if not aggregate_gate_passed:
        eligible = np.zeros_like(eligible)
        reasons = tuple((*values, "aggregate_gate") for values in reasons)
    if np.any(eligible):
        positions = np.flatnonzero(eligible)
        if rule.candidate_selection_strategy == "minimum_predicted_elapsed":
            elapsed = np.asarray(evaluation.elapsed_score, dtype=np.float64)
            position = int(positions[np.argmin(elapsed[positions])])
            reason = "aggregate_calibrated_long_h_min_elapsed"
        else:
            position = int(positions[-1])
            reason = "aggregate_calibrated_long_h_largest"
        selected = evaluation.long_horizons[position]
    else:
        short_horizons = tuple(
            value for value in calibration.candidate_horizons if value <= calibration.reference_horizon
        )
        short_event = np.asarray(evaluation.short_event_probability, dtype=np.float64)
        short_eligible = short_event <= rule.maximum_hazard_probability
        if np.any(short_eligible):
            selected = short_horizons[int(np.flatnonzero(short_eligible)[-1])]
            if selected == calibration.reference_horizon:
                reason = (
                    "aggregate_calibration_fallback_reference"
                    if not aggregate_gate_passed
                    else "aggregate_fallback_reference"
                )
            else:
                reason = "aggregate_hazard_short_gate"
        else:
            selected = short_horizons[0]
            reason = "aggregate_hazard_minimum_h"
    return AggregateSelectorDecision(
        selected_horizon=selected,
        reason=reason,
        long_horizons=evaluation.long_horizons,
        success_score=evaluation.success_score,
        elapsed_score=evaluation.elapsed_score,
        danger_probability=evaluation.danger_probability,
        faster_probability=evaluation.faster_probability,
        long_event_probability=evaluation.long_event_probability,
        short_event_probability=evaluation.short_event_probability,
        ood_probability=evaluation.ood_probability,
        success_pass=evaluation.success_pass,
        elapsed_pass=evaluation.elapsed_pass,
        danger_pass=evaluation.danger_pass,
        faster_pass=evaluation.faster_pass,
        hazard_pass=evaluation.hazard_pass,
        ood_pass=evaluation.ood_pass,
        long_eligible=tuple(bool(value) for value in eligible),
        rejection_reasons=reasons,
    )


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


def _unbatched_prediction_row(
    predictions: Mapping[str, Any],
    root: int,
    num_roots: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, value in predictions.items():
        array = np.asarray(value)
        if name == "candidate_horizons":
            row[name] = array[root] if array.ndim == 2 and array.shape[0] == num_roots else array
        elif name == "reference_horizon":
            row[name] = array[root] if array.ndim == 1 and array.shape[0] == num_roots else array
        elif array.ndim > 0 and array.shape[0] == num_roots:
            row[name] = array[root]
        else:
            row[name] = array
    return row


def _cluster_bootstrap_weights(
    cluster_ids: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    if not unique_clusters.size:
        raise ValueError("Aggregate calibration received no clusters.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, unique_clusters.size, size=(samples, unique_clusters.size))
    weights = np.zeros((samples, unique_clusters.size), dtype=np.float64)
    sample_rows = np.repeat(np.arange(samples), unique_clusters.size)
    np.add.at(weights, (sample_rows, sampled.reshape((-1,))), 1.0 / unique_clusters.size)
    cluster_counts = np.bincount(inverse, minlength=unique_clusters.size).astype(np.float64)
    return inverse, cluster_counts, weights


def _cluster_means(values: np.ndarray, inverse: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64), minlength=counts.size)
    return sums / counts


def _wilson_upper(successes: float, trials: float, confidence_level: float) -> float:
    if trials <= 0:
        return 1.0
    # Match the existing official auditor exactly at the pre-registered level.
    z = 1.96 if confidence_level == 0.95 else NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    margin = z / denominator * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
    return float(min(1.0, center + margin))


def _aggregate_rule_grid(config: AggregateSelectorCalibrationConfig) -> tuple[AggregateSelectorRule, ...]:
    return tuple(
        AggregateSelectorRule(
            success_threshold=success_threshold,
            elapsed_margin=elapsed_margin,
            maximum_danger_probability=maximum_danger_probability,
            require_faster=require_faster,
            minimum_faster_probability=config.minimum_faster_probability,
            maximum_hazard_probability=maximum_hazard_probability,
            maximum_ood_probability=maximum_ood_probability,
            candidate_selection_strategy=config.candidate_selection_strategy,
        )
        for (
            success_threshold,
            elapsed_margin,
            maximum_danger_probability,
            require_faster,
            maximum_hazard_probability,
            maximum_ood_probability,
        ) in itertools.product(
            config.success_thresholds,
            config.elapsed_margins,
            config.maximum_danger_probabilities,
            config.require_faster_options,
            config.maximum_hazard_probabilities,
            config.maximum_ood_probabilities,
        )
    )


def _validated_count_array(name: str, values: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != expected_shape:
        raise ValueError(f"{name} has shape {result.shape}; expected {expected_shape}.")
    if np.any(~np.isfinite(result)) or np.any(result < 0) or np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain finite non-negative integer counts.")
    return result


def _validated_binary_trials(name: str, values: Any, expected_prefix: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 3 or raw.shape[:2] != expected_prefix or raw.shape[2] <= 0:
        raise ValueError(f"{name} must have shape [roots, candidate_horizons, positive max_trials].")
    if raw.dtype != np.bool_:
        numeric = np.asarray(raw, dtype=np.float64)
        if np.any(~np.isfinite(numeric)) or np.any((numeric != 0) & (numeric != 1)):
            raise ValueError(f"{name} must contain only boolean or binary values.")
    return raw.astype(np.bool_, copy=False)


def fit_aggregate_selector_calibration(
    predictions: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    calibration: HierarchicalCalibration,
    provenance: AggregateSelectorProvenance,
    cluster_ids: np.ndarray,
    candidate_horizons: Sequence[int] | None = None,
    reference_horizon: int | None = None,
    config: AggregateSelectorCalibrationConfig = DEFAULT_AGGREGATE_SELECTOR_CALIBRATION_CONFIG,
    split_name: str = "calibration",
) -> AggregateSelectorCalibration:
    """Freeze the highest-coverage aggregate-safe rule on calibration data only.

    Predictor outputs determine every per-root action. Labels are used only to
    accept or reject a complete frozen rule using aggregate paired outcomes.
    Passing validation or test arrays to this fitter is rejected explicitly.
    """

    if split_name != "calibration":
        raise ValueError("Aggregate selector fitting is calibration-only; validation/test labels are forbidden.")
    candidates = (
        tuple(int(value) for value in candidate_horizons)
        if candidate_horizons is not None
        else calibration.candidate_horizons
    )
    reference = calibration.reference_horizon if reference_horizon is None else int(reference_horizon)
    if candidates != calibration.candidate_horizons or reference != calibration.reference_horizon:
        raise ValueError("Aggregate and pointwise calibration horizon contracts differ.")
    if candidates != tuple(sorted(set(candidates))) or reference not in candidates:
        raise ValueError("candidate_horizons must be sorted/unique and contain reference_horizon.")
    long_horizons = tuple(value for value in candidates if value > reference)
    if not long_horizons:
        raise ValueError("Aggregate selector calibration requires at least one long horizon.")
    success_prediction = np.asarray(predictions["success_advantage"], dtype=np.float64)
    if success_prediction.ndim != 2 or success_prediction.shape[1] != len(long_horizons):
        raise ValueError("Batched success_advantage must have shape [roots, long_horizons].")
    num_roots = success_prediction.shape[0]
    if num_roots == 0:
        raise ValueError("Aggregate selector calibration received no roots.")
    cluster_ids = np.asarray(cluster_ids).reshape((-1,))
    if cluster_ids.size != num_roots:
        raise ValueError("cluster_ids must contain one identifier per calibration root.")
    if not np.issubdtype(cluster_ids.dtype, np.integer) or np.any(cluster_ids < 0):
        raise ValueError("cluster_ids must contain non-negative integer identifiers.")
    calibration_groups = tuple(int(value) for value in np.unique(cluster_ids))
    if provenance.calibration_group_ids != calibration_groups:
        raise ValueError("Provenance calibration_group_ids do not match the fitted calibration roots.")

    expected_shape = (num_roots, len(candidates))
    trial_count = _validated_count_array("trial_count", labels["trial_count"], expected_shape)
    success_count = _validated_count_array("success_count", labels["success_count"], expected_shape)
    elapsed_mean = np.asarray(labels["elapsed_mean"], dtype=np.float64)
    if elapsed_mean.shape != expected_shape:
        raise ValueError(f"elapsed_mean has shape {elapsed_mean.shape}; expected {expected_shape}.")
    if np.any(trial_count <= 0) or np.any(success_count > trial_count) or np.any(~np.isfinite(elapsed_mean)):
        raise ValueError("Aggregate calibration requires finite, valid labels for every root and candidate.")

    trial_valid = _validated_binary_trials("trial_valid", labels["trial_valid"], expected_shape)
    trial_success = _validated_binary_trials("trial_success", labels["trial_success"], expected_shape)
    if trial_success.shape != trial_valid.shape:
        raise ValueError("trial_success and trial_valid must have identical shapes.")
    recomputed_trial_count = np.sum(trial_valid, axis=-1, dtype=np.int64)
    recomputed_success_count = np.sum(trial_valid & trial_success, axis=-1, dtype=np.int64)
    if not np.array_equal(trial_count, recomputed_trial_count):
        raise ValueError("trial_count does not match raw trial_valid outcomes.")
    if not np.array_equal(success_count, recomputed_success_count):
        raise ValueError("success_count does not match raw trial_valid/trial_success outcomes.")

    dangerous = np.asarray(labels["dangerous_long_count"])
    paired = np.asarray(labels["paired_trial_count"])
    if dangerous.shape == expected_shape and paired.shape == expected_shape:
        dangerous_full = _validated_count_array("dangerous_long_count", dangerous, expected_shape)
        paired_full = _validated_count_array("paired_trial_count", paired, expected_shape)
    elif dangerous.shape == (num_roots, len(long_horizons)) and paired.shape == dangerous.shape:
        dangerous_full = np.zeros(expected_shape, dtype=np.float64)
        paired_full = np.zeros(expected_shape, dtype=np.float64)
        long_indices = [candidates.index(horizon) for horizon in long_horizons]
        long_shape = (num_roots, len(long_horizons))
        dangerous_full[:, long_indices] = _validated_count_array("dangerous_long_count", dangerous, long_shape)
        paired_full[:, long_indices] = _validated_count_array("paired_trial_count", paired, long_shape)
    else:
        raise ValueError("Paired danger labels must use either all-candidate or long-candidate width.")
    recomputed_paired = np.zeros(expected_shape, dtype=np.int64)
    recomputed_dangerous = np.zeros(expected_shape, dtype=np.int64)
    reference_index = candidates.index(reference)
    for horizon in long_horizons:
        candidate_index = candidates.index(horizon)
        paired_valid = trial_valid[:, reference_index] & trial_valid[:, candidate_index]
        recomputed_paired[:, candidate_index] = np.sum(paired_valid, axis=-1, dtype=np.int64)
        recomputed_dangerous[:, candidate_index] = np.sum(
            paired_valid & trial_success[:, reference_index] & ~trial_success[:, candidate_index],
            axis=-1,
            dtype=np.int64,
        )
    if not np.array_equal(paired_full, recomputed_paired):
        raise ValueError("paired_trial_count does not match raw paired trial_valid outcomes.")
    if not np.array_equal(dangerous_full, recomputed_dangerous):
        raise ValueError("dangerous_long_count does not match raw paired success outcomes.")

    placeholder_rule = _aggregate_rule_grid(config)[0]
    candidate_evaluations = [
        _evaluate_aggregate_candidates(
            _unbatched_prediction_row(predictions, root, num_roots),
            calibration=calibration,
            rule=placeholder_rule,
        )
        for root in range(num_roots)
    ]
    success_score = np.asarray([item.success_score for item in candidate_evaluations], dtype=np.float64)
    elapsed_score = np.asarray([item.elapsed_score for item in candidate_evaluations], dtype=np.float64)
    danger_probability = np.asarray([item.danger_probability for item in candidate_evaluations], dtype=np.float64)
    faster_probability = np.asarray([item.faster_probability for item in candidate_evaluations], dtype=np.float64)
    long_event_probability = np.asarray(
        [item.long_event_probability for item in candidate_evaluations], dtype=np.float64
    )
    short_event_probability = np.asarray(
        [item.short_event_probability for item in candidate_evaluations], dtype=np.float64
    )
    ood_probability = np.asarray([item.ood_probability for item in candidate_evaluations], dtype=np.float64)

    candidate_to_index = {horizon: index for index, horizon in enumerate(candidates)}
    reference_index = candidate_to_index[reference]
    long_indices = np.asarray([candidate_to_index[horizon] for horizon in long_horizons], dtype=np.int64)
    short_horizons = tuple(value for value in candidates if value <= reference)
    short_indices = np.asarray([candidate_to_index[horizon] for horizon in short_horizons], dtype=np.int64)
    rows = np.arange(num_roots)
    success_rate = success_count / trial_count
    success_delta = success_rate - success_rate[:, reference_index : reference_index + 1]
    elapsed_delta = elapsed_mean - elapsed_mean[:, reference_index : reference_index + 1]
    inverse, cluster_counts, success_bootstrap_weights = _cluster_bootstrap_weights(
        cluster_ids,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + 1,
    )
    _, _, elapsed_bootstrap_weights = _cluster_bootstrap_weights(
        cluster_ids,
        samples=config.bootstrap_samples,
        seed=config.bootstrap_seed + 2,
    )
    alpha = (1.0 - config.confidence_level) / 2.0
    metrics_cache: dict[bytes, AggregateSelectorMetrics] = {}

    def metrics_for_selection(selected_indices: np.ndarray) -> AggregateSelectorMetrics:
        key = np.asarray(selected_indices, dtype=np.int64).tobytes()
        if key in metrics_cache:
            return metrics_cache[key]
        selected_horizons = np.asarray(candidates, dtype=np.int64)[selected_indices]
        selected_long = selected_horizons > reference
        selected_success_delta = success_delta[rows, selected_indices]
        selected_elapsed_delta = elapsed_delta[rows, selected_indices]
        success_clusters = _cluster_means(selected_success_delta, inverse, cluster_counts)
        elapsed_clusters = _cluster_means(selected_elapsed_delta, inverse, cluster_counts)
        success_bootstrap = success_bootstrap_weights @ success_clusters
        elapsed_bootstrap = elapsed_bootstrap_weights @ elapsed_clusters
        success_lcb = float(np.quantile(success_bootstrap, alpha))
        elapsed_ucb = float(np.quantile(elapsed_bootstrap, 1.0 - alpha))
        dangerous_total = float(np.sum(dangerous_full[rows[selected_long], selected_indices[selected_long]]))
        paired_total = float(np.sum(paired_full[rows[selected_long], selected_indices[selected_long]]))
        false_long_rate = dangerous_total / max(paired_total, 1.0)
        false_long_upper = _wilson_upper(dangerous_total, paired_total, config.confidence_level)
        unique, counts = np.unique(selected_horizons, return_counts=True)
        coverage = float(np.mean(selected_long))
        gate_passed = bool(
            np.any(selected_long)
            and success_lcb >= -config.success_noninferiority_margin
            and elapsed_ucb < 0.0
            and false_long_upper <= config.false_long_upper_bound
        )
        metrics = AggregateSelectorMetrics(
            long_coverage=coverage,
            selected_long_roots=int(np.sum(selected_long)),
            success_advantage_mean=float(np.mean(success_clusters)),
            success_advantage_lcb=success_lcb,
            elapsed_advantage_mean=float(np.mean(elapsed_clusters)),
            elapsed_advantage_ucb=elapsed_ucb,
            false_long_rate=false_long_rate,
            false_long_upper=false_long_upper,
            false_long_paired_trials=paired_total,
            aggregate_gate_passed=gate_passed,
            selected_h_distribution=tuple(
                (int(horizon), int(count)) for horizon, count in zip(unique, counts, strict=True)
            ),
        )
        metrics_cache[key] = metrics
        return metrics

    evaluations: list[AggregateRuleEvaluation] = []
    best_evaluation: AggregateRuleEvaluation | None = None
    best_rank: tuple[float, ...] | None = None
    for rule in _aggregate_rule_grid(config):
        eligible = (
            np.isfinite(success_score)
            & (success_score >= rule.success_threshold)
            & np.isfinite(elapsed_score)
            & (elapsed_score < -rule.elapsed_margin)
            & (danger_probability <= rule.maximum_danger_probability)
            & (long_event_probability <= rule.maximum_hazard_probability)
            & (ood_probability[:, None] < rule.maximum_ood_probability)
        )
        if rule.require_faster:
            eligible &= faster_probability >= rule.minimum_faster_probability
        if rule.candidate_selection_strategy == "minimum_predicted_elapsed":
            ranked_elapsed = np.where(eligible, elapsed_score, np.inf)
            long_positions = np.argmin(ranked_elapsed, axis=1)
        else:
            long_positions = np.max(
                np.where(eligible, np.arange(len(long_horizons), dtype=np.int64)[None, :] + 1, 0),
                axis=1,
            ) - 1
        has_long = np.any(eligible, axis=1)
        selected_indices = np.empty((num_roots,), dtype=np.int64)
        short_eligible = short_event_probability <= rule.maximum_hazard_probability
        short_positions = np.max(
            np.where(short_eligible, np.arange(len(short_horizons), dtype=np.int64)[None, :] + 1, 0),
            axis=1,
        ) - 1
        short_positions = np.maximum(short_positions, 0)
        selected_indices[:] = short_indices[short_positions]
        selected_indices[has_long] = long_indices[long_positions[has_long]]
        metrics = metrics_for_selection(selected_indices)
        evaluation = AggregateRuleEvaluation(rule=rule, metrics=metrics)
        evaluations.append(evaluation)
        if not metrics.aggregate_gate_passed:
            continue
        rank = (
            metrics.long_coverage,
            metrics.success_advantage_lcb,
            -metrics.elapsed_advantage_ucb,
            -metrics.false_long_upper,
            rule.success_threshold,
            rule.elapsed_margin,
            -rule.maximum_danger_probability,
            float(rule.require_faster),
            -rule.maximum_hazard_probability,
            -rule.maximum_ood_probability,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_evaluation = evaluation

    return AggregateSelectorCalibration(
        candidate_horizons=candidates,
        reference_horizon=reference,
        pointwise_calibration=calibration,
        provenance=provenance,
        search_config=config,
        selected_rule=best_evaluation.rule if best_evaluation is not None else None,
        calibration_metrics=best_evaluation.metrics if best_evaluation is not None else None,
        search_evaluations=tuple(evaluations),
        num_rules_evaluated=len(evaluations),
        num_feasible_rules=sum(item.metrics.aggregate_gate_passed for item in evaluations),
        num_unique_selections=len(metrics_cache),
        source_split=split_name,
    )
