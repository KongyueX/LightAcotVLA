"""Single-call execution-horizon predictor used by Budgeted Event V2-P.

This module is intentionally independent from ``action_cot_step_head``.  The
latter predicts the number of flow-denoising iterations, while this module
predicts how many environment actions (H=1..10) should be executed before the
next policy call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ExecutionHorizonPredictorConfig:
    prefix_feature_dim: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    coarse_horizon: int = 15
    action_horizon: int = 10
    hidden_dim: int = 256
    temporal_layers: int = 3
    remaining_calls_scale: float = 64.0
    remaining_steps_scale: float = 512.0

    def __post_init__(self) -> None:
        if self.action_horizon != 10:
            raise ValueError("Budgeted Event V2-P currently requires action_horizon=10.")
        if self.coarse_horizon < self.action_horizon:
            raise ValueError("coarse_horizon must be at least action_horizon.")


@dataclasses.dataclass(frozen=True)
class ExecutionHorizonLossWeights:
    success: float = 1.0
    timeout: float = 0.5
    remaining_calls: float = 0.25
    remaining_steps: float = 0.25
    final_risk: float = 0.5
    action_cot_risk: float = 0.5
    fused_risk: float = 1.0
    event: float = 0.5
    raw_h_classification: float = 0.5
    raw_h_ordinal: float = 0.25


DEFAULT_LOSS_WEIGHTS = ExecutionHorizonLossWeights()


def _bce_with_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    labels = labels.astype(logits.dtype)
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _masked_mean(values: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    if mask is None:
        return jnp.mean(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _huber(values: jax.Array, delta: float = 1.0) -> jax.Array:
    absolute = jnp.abs(values)
    quadratic = jnp.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


class ExecutionHorizonPredictor(nnx.Module):
    """Shared temporal encoder with entropy/event and counterfactual-Q heads."""

    def __init__(
        self,
        config: ExecutionHorizonPredictorConfig,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        h = config.hidden_dim
        self.prefix_proj = nnx.Linear(config.prefix_feature_dim, h, rngs=rngs, param_dtype=param_dtype)
        self.state_proj = nnx.Linear(config.state_dim, h, rngs=rngs, param_dtype=param_dtype)
        self.controller_proj = nnx.Linear(4, h, rngs=rngs, param_dtype=param_dtype)
        # final, aligned coarse, previous overlap and their delta, plus two
        # scalar overlap/consistency channels.
        self.action_proj = nnx.Linear(4 * config.action_dim + 2, h, rngs=rngs, param_dtype=param_dtype)
        self.temporal_layers = [
            nnx.Linear(3 * h, h, rngs=rngs, param_dtype=param_dtype)
            for _ in range(config.temporal_layers)
        ]
        self.summary_proj = nnx.Linear(2 * h, h, rngs=rngs, param_dtype=param_dtype)

        self.final_risk_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
        self.action_cot_risk_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
        self.fused_risk_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
        self.event_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)

        self.raw_h_logits_head = nnx.Linear(h, config.action_horizon, rngs=rngs, param_dtype=param_dtype)
        self.raw_h_ordinal_head = nnx.Linear(h, config.action_horizon - 1, rngs=rngs, param_dtype=param_dtype)
        self.success_head = nnx.Linear(h, config.action_horizon, rngs=rngs, param_dtype=param_dtype)
        self.timeout_head = nnx.Linear(h, config.action_horizon, rngs=rngs, param_dtype=param_dtype)
        self.remaining_calls_head = nnx.Linear(h, config.action_horizon, rngs=rngs, param_dtype=param_dtype)
        self.remaining_steps_head = nnx.Linear(h, config.action_horizon, rngs=rngs, param_dtype=param_dtype)

    def _align_coarse(self, coarse_actions: jax.Array) -> jax.Array:
        indices = jnp.rint(
            jnp.linspace(0, self.config.coarse_horizon - 1, self.config.action_horizon)
        ).astype(jnp.int32)
        return jnp.take(coarse_actions, indices, axis=1)

    def _previous_overlap(
        self,
        final_actions: jax.Array,
        previous_actions: jax.Array,
        previous_h: jax.Array,
        previous_valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        horizon = self.config.action_horizon
        previous_h = jnp.clip(jnp.asarray(previous_h, dtype=jnp.int32).reshape((-1,)), 1, horizon)
        indices = previous_h[:, None] + jnp.arange(horizon, dtype=jnp.int32)[None, :]
        valid = (indices < horizon) & jnp.asarray(previous_valid, dtype=jnp.bool_).reshape((-1, 1))
        clipped = jnp.minimum(indices, horizon - 1)
        aligned = jnp.take_along_axis(previous_actions, clipped[..., None], axis=1)
        aligned = jnp.where(valid[..., None], aligned, 0.0)
        delta = final_actions - aligned
        consistency = jnp.mean(jnp.abs(delta), axis=-1, keepdims=True)
        return aligned, valid[..., None].astype(final_actions.dtype), consistency

    def __call__(
        self,
        *,
        prefix_feature: jax.Array,
        state: jax.Array,
        coarse_actions: jax.Array,
        final_actions: jax.Array,
        previous_actions: jax.Array,
        previous_h: jax.Array,
        budget_balance: jax.Array,
        episode_progress: jax.Array,
        previous_valid: jax.Array,
    ) -> dict[str, jax.Array]:
        cfg = self.config
        prefix_feature = jnp.asarray(prefix_feature, dtype=jnp.float32)
        state = jnp.asarray(state, dtype=jnp.float32)[..., : cfg.state_dim]
        coarse_actions = jnp.asarray(coarse_actions, dtype=jnp.float32)[
            ..., : cfg.coarse_horizon, : cfg.action_dim
        ]
        final_actions = jnp.asarray(final_actions, dtype=jnp.float32)[
            ..., : cfg.action_horizon, : cfg.action_dim
        ]
        previous_actions = jnp.asarray(previous_actions, dtype=jnp.float32)[
            ..., : cfg.action_horizon, : cfg.action_dim
        ]

        aligned_coarse = self._align_coarse(coarse_actions)
        aligned_previous, overlap_valid, consistency = self._previous_overlap(
            final_actions, previous_actions, previous_h, previous_valid
        )
        action_features = jnp.concatenate(
            [
                final_actions,
                aligned_coarse,
                aligned_previous,
                final_actions - aligned_previous,
                overlap_valid,
                consistency,
            ],
            axis=-1,
        )

        previous_h_float = jnp.asarray(previous_h, dtype=jnp.float32).reshape((-1, 1)) / cfg.action_horizon
        controller = jnp.concatenate(
            [
                previous_h_float,
                jnp.asarray(budget_balance, dtype=jnp.float32).reshape((-1, 1)),
                jnp.asarray(episode_progress, dtype=jnp.float32).reshape((-1, 1)),
                jnp.asarray(previous_valid, dtype=jnp.float32).reshape((-1, 1)),
            ],
            axis=-1,
        )
        context = nnx.swish(self.prefix_proj(prefix_feature))
        context = context + nnx.swish(self.state_proj(state)) + nnx.swish(self.controller_proj(controller))
        tokens = nnx.swish(self.action_proj(action_features)) + context[:, None, :]

        for layer in self.temporal_layers:
            left = jnp.concatenate([tokens[:, :1], tokens[:, :-1]], axis=1)
            right = jnp.concatenate([tokens[:, 1:], tokens[:, -1:]], axis=1)
            tokens = tokens + nnx.swish(layer(jnp.concatenate([left, tokens, right], axis=-1)))

        temporal_summary = jnp.mean(tokens, axis=1)
        summary = nnx.swish(self.summary_proj(jnp.concatenate([temporal_summary, context], axis=-1)))
        remaining_calls = nnx.softplus(self.remaining_calls_head(summary)) * cfg.remaining_calls_scale
        remaining_steps = nnx.softplus(self.remaining_steps_head(summary)) * cfg.remaining_steps_scale
        return {
            "final_risk": nnx.softplus(self.final_risk_head(tokens)[..., 0]),
            "action_cot_risk": nnx.softplus(self.action_cot_risk_head(tokens)[..., 0]),
            "fused_risk": nnx.softplus(self.fused_risk_head(tokens)[..., 0]),
            "event_logits": self.event_head(tokens)[..., 0],
            "raw_h_logits": self.raw_h_logits_head(summary),
            "raw_h_ordinal_logits": self.raw_h_ordinal_head(summary),
            "success_logits": self.success_head(summary),
            "timeout_logits": self.timeout_head(summary),
            "remaining_calls": remaining_calls,
            "remaining_steps": remaining_steps,
            "temporal_feature": summary,
            "overlap_consistency": consistency[..., 0],
        }


def execution_horizon_loss(
    predictions: Mapping[str, jax.Array],
    labels: Mapping[str, jax.Array],
    *,
    weights: ExecutionHorizonLossWeights = DEFAULT_LOSS_WEIGHTS,
    remaining_calls_scale: float = 64.0,
    remaining_steps_scale: float = 512.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the configurable SFT objective over all counterfactual H values."""

    branch_mask = jnp.asarray(labels.get("branch_valid", jnp.ones_like(labels["branch_success"])))
    risk_mask = jnp.asarray(labels.get("risk_valid", jnp.ones_like(labels["final_risk"])))
    success_loss = _masked_mean(
        _bce_with_logits(predictions["success_logits"], labels["branch_success"]), branch_mask
    )
    timeout_loss = _masked_mean(
        _bce_with_logits(predictions["timeout_logits"], labels["branch_timeout"]), branch_mask
    )
    calls_loss = _masked_mean(
        _huber((predictions["remaining_calls"] - labels["remaining_calls"]) / remaining_calls_scale),
        branch_mask,
    )
    steps_loss = _masked_mean(
        _huber((predictions["remaining_steps"] - labels["remaining_steps"]) / remaining_steps_scale),
        branch_mask,
    )
    final_risk_loss = _masked_mean(_huber(predictions["final_risk"] - labels["final_risk"]), risk_mask)
    cot_risk_loss = _masked_mean(
        _huber(predictions["action_cot_risk"] - labels["action_cot_risk"]), risk_mask
    )
    fused_risk_loss = _masked_mean(_huber(predictions["fused_risk"] - labels["fused_risk"]), risk_mask)
    event_loss = _masked_mean(_bce_with_logits(predictions["event_logits"], labels["event_mask"]), risk_mask)

    raw_h = jnp.clip(jnp.asarray(labels["raw_h"], dtype=jnp.int32).reshape((-1,)), 1, 10)
    raw_h_classification_loss = jnp.mean(
        -jnp.take_along_axis(jax.nn.log_softmax(predictions["raw_h_logits"], axis=-1), (raw_h - 1)[:, None], axis=-1)
    )
    ordinal_targets = raw_h[:, None] > jnp.arange(1, 10, dtype=jnp.int32)[None, :]
    raw_h_ordinal_loss = jnp.mean(
        _bce_with_logits(predictions["raw_h_ordinal_logits"], ordinal_targets)
    )

    metrics = {
        "success_bce": success_loss,
        "timeout_bce": timeout_loss,
        "remaining_calls_huber": calls_loss,
        "remaining_steps_huber": steps_loss,
        "final_risk_huber": final_risk_loss,
        "action_cot_risk_huber": cot_risk_loss,
        "fused_risk_huber": fused_risk_loss,
        "event_bce": event_loss,
        "raw_h_classification": raw_h_classification_loss,
        "raw_h_ordinal": raw_h_ordinal_loss,
    }
    total = (
        weights.success * success_loss
        + weights.timeout * timeout_loss
        + weights.remaining_calls * calls_loss
        + weights.remaining_steps * steps_loss
        + weights.final_risk * final_risk_loss
        + weights.action_cot_risk * cot_risk_loss
        + weights.fused_risk * fused_risk_loss
        + weights.event * event_loss
        + weights.raw_h_classification * raw_h_classification_loss
        + weights.raw_h_ordinal * raw_h_ordinal_loss
    )
    metrics["loss"] = total
    return total, metrics
