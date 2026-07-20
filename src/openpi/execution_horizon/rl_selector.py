"""Lightweight frozen-feature selector for execution-horizon RL experiments.

The ACoT-VLA policy and the V2-P temporal encoder stay frozen.  This module
only consumes the predictor outputs already returned by the policy server and
applies a small actor/critic head on the client.  Keeping the selector in an
independent ``.npz`` sidecar makes the pilot reversible and preserves all
existing V2-P checkpoints and evaluation modes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

DEFAULT_CANDIDATES = (1, 3, 6, 10)
_PARAMETER_NAMES = (
    "trunk_w",
    "trunk_b",
    "actor_w",
    "actor_b",
    "q_success_w",
    "q_success_b",
    "q_cost_w",
    "q_cost_b",
    "value_w",
    "value_b",
)


def _get_output(outputs: Mapping[str, Any], name: str) -> np.ndarray:
    """Accept both predictor-only and policy-server-prefixed dictionaries."""
    if name in outputs:
        return np.asarray(outputs[name], dtype=np.float32)
    prefixed = f"execution_horizon_{name}"
    if prefixed in outputs:
        return np.asarray(outputs[prefixed], dtype=np.float32)
    raise KeyError(f"Missing predictor output {name!r} (or {prefixed!r}).")


def build_selector_feature(outputs: Mapping[str, Any]) -> np.ndarray:
    """Build the stable selector feature shared by offline and online code."""
    temporal = _get_output(outputs, "temporal_feature")
    success_logits = _get_output(outputs, "success_logits")
    timeout_logits = _get_output(outputs, "timeout_logits")
    remaining_calls = np.clip(_get_output(outputs, "remaining_calls") / 64.0, 0.0, 4.0)
    remaining_steps = np.clip(_get_output(outputs, "remaining_steps") / 512.0, 0.0, 4.0)
    final_risk = np.log1p(np.maximum(_get_output(outputs, "final_risk"), 0.0))
    action_cot_risk = np.log1p(np.maximum(_get_output(outputs, "action_cot_risk"), 0.0))
    fused_risk = np.log1p(np.maximum(_get_output(outputs, "fused_risk"), 0.0))
    return np.concatenate(
        [
            temporal,
            success_logits,
            timeout_logits,
            remaining_calls,
            remaining_steps,
            final_risk,
            action_cot_risk,
            fused_risk,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def softmax(values: np.ndarray, *, mask: np.ndarray | None = None, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logits = np.asarray(values, dtype=np.float64) / temperature
    if mask is not None:
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != logits.shape:
            raise ValueError(f"mask shape {mask.shape} does not match logits shape {logits.shape}.")
        if not np.any(mask):
            raise ValueError("At least one action must remain eligible.")
        logits = np.where(mask, logits, -np.inf)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


@dataclasses.dataclass(frozen=True)
class SelectorDecision:
    horizon: int
    action_index: int
    probabilities: np.ndarray
    log_probability: float
    value: float
    q_success_probability: np.ndarray
    q_cost: np.ndarray
    eligible: np.ndarray
    feature: np.ndarray
    actor_logits: np.ndarray

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "selector_horizon": self.horizon,
            "selector_action_index": self.action_index,
            "selector_probabilities": self.probabilities.tolist(),
            "selector_old_log_prob": self.log_probability,
            "selector_value": self.value,
            "selector_q_success_probability": self.q_success_probability.tolist(),
            "selector_q_cost": self.q_cost.tolist(),
            "selector_eligible": self.eligible.astype(np.int8).tolist(),
            "selector_feature": self.feature.tolist(),
            "selector_actor_logits": self.actor_logits.tolist(),
        }


@dataclasses.dataclass
class FrozenFeatureSelector:
    candidates: tuple[int, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    params: dict[str, np.ndarray]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidates or tuple(sorted(set(self.candidates))) != self.candidates:
            raise ValueError("candidates must be unique and sorted.")
        missing = sorted(set(_PARAMETER_NAMES).difference(self.params))
        if missing:
            raise KeyError(f"Selector parameters are missing: {missing}")
        feature_dim = int(self.params["trunk_w"].shape[0])
        if self.feature_mean.shape != (feature_dim,) or self.feature_std.shape != (feature_dim,):
            raise ValueError("Feature normalization shape does not match trunk input.")
        num_actions = len(self.candidates)
        for name in ("actor_b", "q_success_b", "q_cost_b"):
            if self.params[name].shape != (num_actions,):
                raise ValueError(f"{name} must have shape {(num_actions,)}, got {self.params[name].shape}.")

    @property
    def reference_index(self) -> int:
        if 10 not in self.candidates:
            raise ValueError("The safety reference H=10 is required.")
        return self.candidates.index(10)

    def forward(self, feature: np.ndarray) -> dict[str, np.ndarray | float]:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.ndim != 1:
            raise ValueError(f"Selector inference expects one feature vector, got {feature.shape}.")
        normalized = (feature - self.feature_mean) / self.feature_std
        hidden = np.tanh(normalized @ self.params["trunk_w"] + self.params["trunk_b"])
        actor_logits = hidden @ self.params["actor_w"] + self.params["actor_b"]
        q_success_logits = hidden @ self.params["q_success_w"] + self.params["q_success_b"]
        q_cost_raw = hidden @ self.params["q_cost_w"] + self.params["q_cost_b"]
        value_logit = float(hidden @ self.params["value_w"] + self.params["value_b"])
        return {
            "actor_logits": np.asarray(actor_logits, dtype=np.float64),
            "q_success_probability": sigmoid(q_success_logits),
            "q_cost": np.logaddexp(q_cost_raw, 0.0).astype(np.float64),
            "value": float(sigmoid(np.asarray(value_logit))),
        }

    def safety_mask(
        self,
        q_success_probability: np.ndarray,
        *,
        minimum_success_probability: float,
        reference_slack: float,
    ) -> np.ndarray:
        q_success_probability = np.asarray(q_success_probability, dtype=np.float64)
        reference_probability = float(q_success_probability[self.reference_index])
        threshold = max(minimum_success_probability, reference_probability - reference_slack)
        eligible = q_success_probability >= threshold
        # The reference is never removed by its own learned critic.  This is
        # the conservative fallback used by both Q distillation and PPO.
        eligible[self.reference_index] = True
        return eligible

    def decide(
        self,
        feature: np.ndarray,
        *,
        policy: str,
        minimum_success_probability: float = 0.5,
        reference_slack: float = 0.05,
        q_tie_margin: float = 0.03,
        sample: bool = False,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> SelectorDecision:
        outputs = self.forward(feature)
        actor_logits = np.asarray(outputs["actor_logits"])
        q_success = np.asarray(outputs["q_success_probability"])
        q_cost = np.asarray(outputs["q_cost"])
        eligible = self.safety_mask(
            q_success,
            minimum_success_probability=minimum_success_probability,
            reference_slack=reference_slack,
        )
        if policy == "q":
            best_success = float(np.max(q_success[eligible]))
            near_best = eligible & (q_success >= best_success - q_tie_margin)
            # Cost is predicted remaining policy calls / 100.  It is used only
            # inside a narrow success-probability tie band.
            score = np.where(near_best, q_cost, np.inf)
            action_index = int(np.argmin(score))
            probabilities = np.zeros(len(self.candidates), dtype=np.float64)
            probabilities[action_index] = 1.0
        elif policy == "actor":
            probabilities = softmax(actor_logits, mask=eligible, temperature=temperature)
            if sample:
                rng = rng or np.random.default_rng()
                action_index = int(rng.choice(len(self.candidates), p=probabilities))
            else:
                action_index = int(np.argmax(probabilities))
        else:
            raise ValueError(f"Unknown selector policy {policy!r}; expected 'q' or 'actor'.")
        return SelectorDecision(
            horizon=self.candidates[action_index],
            action_index=action_index,
            probabilities=probabilities,
            log_probability=float(np.log(max(probabilities[action_index], 1e-12))),
            value=float(outputs["value"]),
            q_success_probability=q_success,
            q_cost=q_cost,
            eligible=eligible,
            feature=np.asarray(feature, dtype=np.float32),
            actor_logits=actor_logits,
        )

    def save(self, path: pathlib.Path | str) -> pathlib.Path:
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "format": "openpi.execution_horizon.frozen_feature_selector",
            "format_version": 1,
            "candidates": list(self.candidates),
        }
        payload = {
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float32),
            "feature_std": np.asarray(self.feature_std, dtype=np.float32),
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            **{name: np.asarray(self.params[name], dtype=np.float32) for name in _PARAMETER_NAMES},
        }
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, **payload)
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: pathlib.Path | str) -> FrozenFeatureSelector:
        source = pathlib.Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            params = {name: np.asarray(archive[name], dtype=np.float32) for name in _PARAMETER_NAMES}
            return cls(
                candidates=tuple(int(value) for value in metadata["candidates"]),
                feature_mean=np.asarray(archive["feature_mean"], dtype=np.float32),
                feature_std=np.asarray(archive["feature_std"], dtype=np.float32),
                params=params,
                metadata=metadata,
            )


def copy_selector(
    selector: FrozenFeatureSelector,
    *,
    params: Mapping[str, np.ndarray] | None = None,
    metadata_updates: Mapping[str, Any] | None = None,
) -> FrozenFeatureSelector:
    merged_params = {
        name: np.asarray((params or {}).get(name, selector.params[name]), dtype=np.float32) for name in _PARAMETER_NAMES
    }
    return FrozenFeatureSelector(
        candidates=selector.candidates,
        feature_mean=np.asarray(selector.feature_mean, dtype=np.float32),
        feature_std=np.asarray(selector.feature_std, dtype=np.float32),
        params=merged_params,
        metadata={**selector.metadata, **dict(metadata_updates or {})},
    )


def candidate_indices(candidates: Sequence[int]) -> np.ndarray:
    values = np.asarray(tuple(candidates), dtype=np.int64)
    if np.any((values < 1) | (values > 10)):
        raise ValueError("Execution horizons must be between 1 and 10.")
    return values - 1
