"""Budgeted Event V2 teacher targets and closed-loop selection primitives."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class V2RiskConfig:
    entropy_algorithm: str = "diagonal_logvar"
    entropy_eps: float = 1e-6
    covariance_shrinkage: float = 1e-4
    coarse_stride: float = 2.0
    risk_threshold: float = 1.5
    final_weight: float = 0.5
    action_cot_weight: float = 0.5
    final_risk_threshold: float | None = None
    action_cot_risk_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.entropy_algorithm not in {"diagonal_logvar", "aac_grouped"}:
            raise ValueError(f"Unknown entropy algorithm: {self.entropy_algorithm}")
        if self.final_weight < 0 or self.action_cot_weight < 0:
            raise ValueError("Risk fusion weights must be non-negative.")
        if self.final_weight + self.action_cot_weight <= 0:
            raise ValueError("At least one risk fusion weight must be positive.")


@dataclasses.dataclass
class EpisodeBudgetState:
    balance: float
    horizon_sum: float = 0.0
    decisions: int = 0
    interventions: int = 0
    limited_decisions: int = 0


@dataclasses.dataclass(frozen=True)
class EpisodeBudgetConfig:
    target_average_horizon: float = 9.0
    capacity: float = 12.0


@dataclasses.dataclass(frozen=True)
class ValueRefinementConfig:
    minimum_success_probability: float = 0.90
    maximum_timeout_probability: float = 0.20
    risk_threshold: float = 1.5
    risk_slack_steps: int = 0
    candidates: tuple[int, ...] = tuple(range(1, 11))


@dataclasses.dataclass(frozen=True)
class SMDPDecision:
    """Future constrained-SMDP boundary; no PPO implementation is provided."""

    state_feature: np.ndarray
    selected_horizon: int
    success_probability: np.ndarray
    timeout_probability: np.ndarray
    predicted_remaining_calls: np.ndarray
    predicted_remaining_steps: np.ndarray
    budget_balance: float


DEFAULT_RISK_CONFIG = V2RiskConfig()
DEFAULT_VALUE_REFINEMENT_CONFIG = ValueRefinementConfig()


def _diagonal_frame_entropy(samples: np.ndarray, eps: float) -> np.ndarray:
    variance = np.var(samples, axis=0)
    return np.mean(np.log(variance + eps), axis=-1)


def _gaussian_group_entropy(samples: np.ndarray, shrinkage: float) -> np.ndarray:
    sample_count, time_len, dim = samples.shape
    entropy = np.empty((time_len,), dtype=np.float64)
    constant = dim * (1.0 + math.log(2.0 * math.pi))
    for time_index in range(time_len):
        values = samples[:, time_index, :]
        centered = values - np.mean(values, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(sample_count, 1)
        covariance += shrinkage * np.eye(dim, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(covariance)
        entropy[time_index] = 0.5 * (constant + logdet) if sign > 0 else float("nan")
    return entropy


def _aac_grouped_frame_entropy(samples: np.ndarray, config: V2RiskConfig) -> np.ndarray:
    if samples.shape[-1] < 7:
        return _diagonal_frame_entropy(samples, config.entropy_eps)
    translation = _gaussian_group_entropy(samples[..., :3], config.covariance_shrinkage)
    rotation = _gaussian_group_entropy(samples[..., 3:6], config.covariance_shrinkage)
    probability = np.clip(
        np.mean(samples[..., 6] > 0, axis=0), config.entropy_eps, 1.0 - config.entropy_eps
    )
    gripper = -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))
    return translation + rotation + gripper


def frame_entropy(samples: np.ndarray, config: V2RiskConfig) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if config.entropy_algorithm == "aac_grouped":
        return _aac_grouped_frame_entropy(samples, config)
    return _diagonal_frame_entropy(samples, config.entropy_eps)


def final_component_entropy(samples: np.ndarray, config: V2RiskConfig) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    time_len = samples.shape[1]
    translation = _diagonal_frame_entropy(samples[..., :3], config.entropy_eps)
    rotation = _diagonal_frame_entropy(samples[..., 3:6], config.entropy_eps)
    if samples.shape[-1] >= 7:
        probability = np.clip(
            np.mean(samples[..., 6] > 0, axis=0), config.entropy_eps, 1.0 - config.entropy_eps
        )
        gripper = -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))
    else:
        gripper = np.zeros((time_len,), dtype=np.float64)
    return {"translation": translation, "rotation": rotation, "gripper": gripper}


def align_coarse_curve(curve: np.ndarray, *, action_horizon: int, stride: float) -> np.ndarray:
    coarse_times = np.arange(len(curve), dtype=np.float64) * stride
    action_times = np.arange(action_horizon, dtype=np.float64)
    return np.interp(action_times, coarse_times, np.asarray(curve, dtype=np.float64))


def robust_positive_risk(curve: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(curve, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values)
    center = float(np.median(finite))
    filled = np.where(np.isfinite(values), values, center)
    mad_scale = 1.4826 * float(np.median(np.abs(finite - center)))
    scale = max(mad_scale, float(np.std(finite)), eps)
    return np.maximum((filled - center) / scale, 0.0)


def risk_targets_from_normalized_mc(
    coarse_samples: np.ndarray,
    action_samples: np.ndarray,
    *,
    config: V2RiskConfig = DEFAULT_RISK_CONFIG,
) -> dict[str, np.ndarray | int]:
    """Compute the same V2 curves from already-normalized K-sample chunks."""
    coarse_samples = np.asarray(coarse_samples, dtype=np.float64)[..., :7]
    action_samples = np.asarray(action_samples, dtype=np.float64)[..., :7]
    if coarse_samples.ndim != 3 or action_samples.ndim != 3:
        raise ValueError("MC samples must have shapes [K, coarse_T, D] and [K, action_T, D].")
    if coarse_samples.shape[0] != action_samples.shape[0]:
        raise ValueError("Coarse and final MC sample counts differ.")
    final_entropy = frame_entropy(action_samples, config)
    coarse_entropy = frame_entropy(coarse_samples, config)
    action_cot_entropy = align_coarse_curve(
        coarse_entropy, action_horizon=action_samples.shape[1], stride=config.coarse_stride
    )
    components = final_component_entropy(action_samples, config)

    final_risk = robust_positive_risk(final_entropy, config.entropy_eps)
    action_cot_risk = robust_positive_risk(action_cot_entropy, config.entropy_eps)
    component_risk = np.maximum.reduce(
        [
            final_risk,
            robust_positive_risk(components["translation"], config.entropy_eps),
            robust_positive_risk(components["rotation"], config.entropy_eps),
            robust_positive_risk(components["gripper"], config.entropy_eps),
        ]
    )
    fused_risk = (
        config.final_weight * component_risk + config.action_cot_weight * action_cot_risk
    ) / (config.final_weight + config.action_cot_weight)
    event_mask = fused_risk >= config.risk_threshold
    if config.final_risk_threshold is not None:
        event_mask |= final_risk >= config.final_risk_threshold
    if config.action_cot_risk_threshold is not None:
        event_mask |= action_cot_risk >= config.action_cot_risk_threshold
    event_indices = np.flatnonzero(event_mask)
    event_index = int(event_indices[0]) if event_indices.size else -1
    return {
        "final_entropy": final_entropy,
        "action_cot_entropy": action_cot_entropy,
        "translation_entropy": components["translation"],
        "rotation_entropy": components["rotation"],
        "gripper_entropy": components["gripper"],
        "final_risk": final_risk,
        "action_cot_risk": action_cot_risk,
        "fused_risk": fused_risk,
        "event_mask": event_mask,
        "event_index": event_index,
    }


def event_horizon(event_index: int | None, candidates: Sequence[int]) -> int:
    candidates = sorted({int(value) for value in candidates})
    if not candidates:
        raise ValueError("At least one horizon candidate is required.")
    if event_index is None or event_index < 0:
        return candidates[-1]
    safe_horizon = max(candidates[0], event_index)
    safe = [horizon for horizon in candidates if horizon <= safe_horizon]
    return safe[-1] if safe else candidates[0]


def distilled_raw_horizon(
    final_risk: np.ndarray,
    action_cot_risk: np.ndarray,
    fused_risk: np.ndarray,
    *,
    candidates: Sequence[int],
    config: V2RiskConfig = DEFAULT_RISK_CONFIG,
) -> tuple[int, np.ndarray]:
    """Apply the original V2 event mapping to predicted risk curves."""
    event_mask = np.asarray(fused_risk) >= config.risk_threshold
    if config.final_risk_threshold is not None:
        event_mask |= np.asarray(final_risk) >= config.final_risk_threshold
    if config.action_cot_risk_threshold is not None:
        event_mask |= np.asarray(action_cot_risk) >= config.action_cot_risk_threshold
    indices = np.flatnonzero(event_mask)
    index = int(indices[0]) if indices.size else None
    return event_horizon(index, candidates), event_mask


def value_refined_raw_horizon(
    *,
    entropy_raw_horizon: int,
    success_probability: np.ndarray,
    timeout_probability: np.ndarray,
    fused_risk: np.ndarray,
    config: ValueRefinementConfig = DEFAULT_VALUE_REFINEMENT_CONFIG,
) -> tuple[int, dict[str, np.ndarray]]:
    """Choose the largest entropy-safe H with acceptable counterfactual value."""
    success_probability = np.asarray(success_probability, dtype=np.float64)
    timeout_probability = np.asarray(timeout_probability, dtype=np.float64)
    fused_risk = np.asarray(fused_risk, dtype=np.float64)
    candidates = np.asarray(sorted(set(config.candidates)), dtype=np.int64)
    candidates = candidates[(candidates >= 1) & (candidates <= min(10, fused_risk.size))]
    entropy_cap = min(10, entropy_raw_horizon + max(config.risk_slack_steps, 0))
    risk_safe = np.asarray(
        [np.max(fused_risk[:horizon]) < config.risk_threshold for horizon in candidates], dtype=bool
    )
    value_safe = (
        success_probability[candidates - 1] >= config.minimum_success_probability
    ) & (timeout_probability[candidates - 1] <= config.maximum_timeout_probability)
    eligible = (candidates <= entropy_cap) & risk_safe & value_safe
    if np.any(eligible):
        selected = int(candidates[np.flatnonzero(eligible)[-1]])
    else:
        # Preserve V2 safety behavior when Q is uncertain rather than taking an
        # out-of-distribution long action chunk.
        selected = int(max(candidates[0], min(entropy_raw_horizon, candidates[-1])))
    return selected, {
        "candidates": candidates,
        "risk_safe": risk_safe,
        "value_safe": value_safe,
        "eligible": eligible,
    }


def apply_episode_budget(
    raw_horizon: int,
    candidates: Sequence[int],
    *,
    config: EpisodeBudgetConfig,
    state: EpisodeBudgetState,
) -> tuple[int, dict[str, float]]:
    """Original Budgeted Event V2 credit controller, unchanged."""
    horizons = sorted({int(value) for value in candidates})
    target = min(float(config.target_average_horizon), float(horizons[-1]))
    balance_before = float(state.balance)
    required_credit = max(target - raw_horizon, 0.0)
    final_horizon = raw_horizon
    budget_limited = False
    if required_credit > balance_before + 1e-9:
        affordable_floor = target - balance_before
        affordable = [
            horizon for horizon in horizons if horizon >= raw_horizon and horizon + 1e-9 >= affordable_floor
        ]
        final_horizon = affordable[0] if affordable else horizons[-1]
        budget_limited = final_horizon > raw_horizon
    balance_after = float(np.clip(balance_before + final_horizon - target, 0.0, config.capacity))
    state.balance = balance_after
    state.decisions += 1
    state.horizon_sum += final_horizon
    intervention = final_horizon < horizons[-1]
    state.interventions += int(intervention)
    state.limited_decisions += int(budget_limited)
    return final_horizon, {
        "target_horizon": target,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "required_credit": required_credit,
        "budget_limited": float(budget_limited),
        "intervention": float(intervention),
        "cumulative_avg_horizon": state.horizon_sum / state.decisions,
        "intervention_rate": state.interventions / state.decisions,
        "budget_limited_rate": state.limited_decisions / state.decisions,
    }
