"""Frozen-feature ordered actor and value heads for SMDP policy updates."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


DEFAULT_CANDIDATES = (5, 10, 15, 20, 25)
PARAMETER_NAMES = ("actor_w", "actor_b", "critic_w", "critic_b", "value_w", "value_b")


def ordered_log_probabilities(continuation_logits: np.ndarray) -> np.ndarray:
    """Convert one vector of continue logits to normalized horizon log-probabilities."""
    logits = np.asarray(continuation_logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size < 1 or not np.all(np.isfinite(logits)):
        raise ValueError("continuation_logits must be a finite, non-empty vector.")
    log_continue = -np.logaddexp(0.0, -logits)
    log_stop = -np.logaddexp(0.0, logits)
    log_prefix = np.concatenate(([0.0], np.cumsum(log_continue)))
    log_probability = np.concatenate((log_prefix[:-1] + log_stop, log_prefix[-1:]))
    return log_probability - np.logaddexp.reduce(log_probability)


@dataclasses.dataclass
class OrderedSMDPSelector:
    candidates: tuple[int, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    params: dict[str, np.ndarray]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_candidates = tuple(self.candidates)
        self.candidates = tuple(int(value) for value in raw_candidates)
        if (
            raw_candidates != self.candidates
            or len(self.candidates) < 2
            or self.candidates[0] <= 0
            or tuple(sorted(set(self.candidates))) != self.candidates
        ):
            raise ValueError("candidates must be at least two sorted, unique positive integers.")
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(self.feature_std, dtype=np.float32)
        if (
            self.feature_mean.ndim != 1
            or self.feature_mean.size == 0
            or self.feature_std.shape != self.feature_mean.shape
        ):
            raise ValueError("feature_mean and feature_std must be matching non-empty vectors.")
        if (
            not np.all(np.isfinite(self.feature_mean))
            or not np.all(np.isfinite(self.feature_std))
            or np.any(self.feature_std <= 0)
        ):
            raise ValueError("Feature normalization must be finite with positive standard deviations.")
        if set(self.params) != set(PARAMETER_NAMES):
            raise ValueError(f"Selector parameters must be exactly {PARAMETER_NAMES}.")
        feature_dim, continuation_dim = self.feature_mean.size, len(self.candidates) - 1
        shapes = {
            "actor_w": (feature_dim, continuation_dim), "actor_b": (continuation_dim,),
            "critic_w": (feature_dim, 64), "critic_b": (64,),
            "value_w": (64, 2), "value_b": (2,),
        }
        for name, shape in shapes.items():
            self.params[name] = np.asarray(self.params[name], dtype=np.float32)
            if self.params[name].shape != shape or not np.all(np.isfinite(self.params[name])):
                raise ValueError(f"{name} must have shape {shape} and finite values.")
        self.metadata = dict(self.metadata)
        saved_candidates = self.metadata.setdefault("candidate_horizons", list(self.candidates))
        if tuple(saved_candidates) != self.candidates:
            raise ValueError("Metadata candidate_horizons does not match selector candidates.")

    @classmethod
    def initialize(cls, feature_dim: int = 256, seed: int = 7) -> OrderedSMDPSelector:
        if int(feature_dim) != feature_dim or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        feature_dim = int(feature_dim)
        rng = np.random.default_rng(seed)
        return cls(
            candidates=DEFAULT_CANDIDATES,
            feature_mean=np.zeros(feature_dim, dtype=np.float32),
            feature_std=np.ones(feature_dim, dtype=np.float32),
            params={
                "actor_w": np.zeros((feature_dim, len(DEFAULT_CANDIDATES) - 1), dtype=np.float32),
                "actor_b": np.zeros(len(DEFAULT_CANDIDATES) - 1, dtype=np.float32),
                "critic_w": rng.normal(0.0, 0.02, (feature_dim, 64)).astype(np.float32),
                "critic_b": np.zeros(64, dtype=np.float32),
                "value_w": np.zeros((64, 2), dtype=np.float32),
                "value_b": np.asarray([np.log(0.9 / 0.1), np.log(np.expm1(3.0))], dtype=np.float32),
            },
            metadata={
                "candidate_horizons": list(DEFAULT_CANDIDATES), "feature_dim": feature_dim,
                "critic_hidden_dim": 64, "seed": seed, "cost_multiplier": 0.02,
            },
        )

    def forward(self, feature: np.ndarray, anchor_logits: np.ndarray) -> dict[str, np.ndarray | float]:
        feature = np.asarray(feature, dtype=np.float32)
        anchor_logits = np.asarray(anchor_logits, dtype=np.float32)
        if feature.shape != self.feature_mean.shape or not np.all(np.isfinite(feature)):
            raise ValueError(f"feature must be a finite vector with shape {self.feature_mean.shape}.")
        if anchor_logits.shape != (len(self.candidates) - 1,) or not np.all(np.isfinite(anchor_logits)):
            raise ValueError("anchor_logits must be a finite vector with one logit per candidate transition.")
        normalized = (feature - self.feature_mean) / self.feature_std
        continuation_logits = anchor_logits + normalized @ self.params["actor_w"] + self.params["actor_b"]
        log_probabilities = ordered_log_probabilities(continuation_logits)
        hidden = np.tanh(normalized @ self.params["critic_w"] + self.params["critic_b"])
        values = hidden @ self.params["value_w"] + self.params["value_b"]
        if not np.all(np.isfinite(values)):
            raise ValueError("Critic produced non-finite values.")
        return {
            "continuation_logits": continuation_logits,
            "log_probabilities": log_probabilities,
            "probabilities": np.exp(log_probabilities),
            "success_value": float(np.exp(-np.logaddexp(0.0, -float(values[0])))),
            "cost_value": float(np.logaddexp(0.0, float(values[1]))),
        }

    def decide(
        self,
        outputs: Mapping[str, Any],
        *,
        sample: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, dict[str, Any]]:
        returned_candidates = np.asarray(outputs["execution_horizon_candidate_horizons"]).reshape(-1)
        if not np.array_equal(returned_candidates, np.asarray(self.candidates)):
            raise ValueError("Policy candidate_horizons differs from the selector metadata.")
        feature = np.asarray(outputs["execution_horizon_temporal_feature"], dtype=np.float32).reshape(-1)
        anchor_logits = np.asarray(
            outputs["execution_horizon_ordered_continuation_logits"], dtype=np.float32
        ).reshape(-1)
        prediction = self.forward(feature, anchor_logits)
        probabilities = np.asarray(prediction["probabilities"])
        if sample:
            rng = rng if rng is not None else np.random.default_rng()
            action_index = int(rng.choice(len(self.candidates), p=probabilities))
        else:
            action_index = int(np.argmax(probabilities))
        raw_index = int(np.argmax(ordered_log_probabilities(anchor_logits)))
        return self.candidates[action_index], {
            "smdp_action_index": action_index,
            "smdp_old_log_prob": float(np.asarray(prediction["log_probabilities"])[action_index]),
            "smdp_success_value": prediction["success_value"],
            "smdp_cost_value": prediction["cost_value"],
            "smdp_feature": feature.tolist(),
            "smdp_anchor_logits": anchor_logits.tolist(),
            "smdp_probabilities": probabilities.tolist(),
            "raw_horizon": self.candidates[raw_index],
            "selector_policy": "ordered_smdp",
        }

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle, **self.params, feature_mean=self.feature_mean, feature_std=self.feature_std,
                metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> OrderedSMDPSelector:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            return cls(
                candidates=tuple(metadata["candidate_horizons"]),
                feature_mean=archive["feature_mean"], feature_std=archive["feature_std"],
                params={name: archive[name] for name in PARAMETER_NAMES}, metadata=metadata,
            )


def compute_smdp_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    durations: np.ndarray,
    gamma: float = 1.0,
    gae_lambda: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute duration-discounted GAE for one complete episode with zero terminal value.

    Rewards are expressed at each chunk's start. The caller discounts a terminal
    success within the last chunk and supplies measured, not planned, durations.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size == 0 or values.shape != rewards.shape or durations.shape != rewards.shape:
        raise ValueError("rewards, values, and durations must be matching non-empty episode vectors.")
    if not all(np.all(np.isfinite(array)) for array in (rewards, values, durations)):
        raise ValueError("Episode rewards, values, and durations must be finite.")
    if np.any(durations <= 0) or np.any(durations != np.floor(durations)):
        raise ValueError("durations must be positive integer environment-step counts.")
    if not (np.isfinite(gamma) and np.isfinite(gae_lambda) and 0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must be finite and lie in [0, 1].")
    advantages = np.empty_like(values)
    next_value = 0.0
    next_advantage = 0.0
    for index in range(values.size - 1, -1, -1):
        discount = gamma ** durations[index]
        delta = rewards[index] + discount * next_value - values[index]
        next_advantage = delta + (gamma * gae_lambda) ** durations[index] * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values
