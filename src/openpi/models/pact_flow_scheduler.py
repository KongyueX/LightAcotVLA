"""Lightweight Plan-Anchored Cross-level Token Flow scheduler utilities.

The scheduler consumes token-aligned explicit action reasoning (EAR), implicit
action reasoning (IAR), and robot state.  It emits a dynamic per-action-token
transport time without imposing a hand-written positional schedule.  Its time
head is zero initialized, so a new module starts from one uniform, verified
endpoint condition while retaining fully dynamic gradients.

This module is deliberately independent from ``acot_vla.py``.  It contains no
policy routing or host-side schedule logic and can therefore be integrated
inside one JAX-compiled inference graph later.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp


Array = jax.Array


@dataclasses.dataclass(frozen=True)
class PACTFlowSchedulerConfig:
    """Static dimensions and initialization for :class:`PACTFlowScheduler`."""

    action_horizon: int = 10
    action_dim: int = 32
    iar_dim: int = 1024
    state_dim: int = 32
    prefix_dim: int = 2048
    hidden_dim: int = 96
    prefix_bottleneck_dim: int = 32
    use_prefix: bool = False
    initial_tau: float = 0.975
    initial_log_std: float = -2.0
    tau_epsilon: float = 1e-4
    norm_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        positive_dimensions = {
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "iar_dim": self.iar_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
        }
        if self.use_prefix:
            positive_dimensions.update(
                prefix_dim=self.prefix_dim,
                prefix_bottleneck_dim=self.prefix_bottleneck_dim,
            )
        invalid = {name: value for name, value in positive_dimensions.items() if value <= 0}
        if invalid:
            raise ValueError(f"PACT-Flow dimensions must be positive, got {invalid}.")
        if not 0.0 < self.initial_tau < 1.0:
            raise ValueError("initial_tau must lie strictly inside (0, 1).")
        if not 0.0 <= self.tau_epsilon < 0.5:
            raise ValueError("tau_epsilon must lie in [0, 0.5).")
        if self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be positive.")


class PACTFlowSchedulerDiagnostics(NamedTuple):
    """Calibrated uncertainty and schedule-analysis outputs."""

    tau_logits: Array
    risk: Array
    iar_attention_entropy: Array
    tau_mean: Array
    tau_total_variation: Array


class PACTFlowSchedulerOutput(NamedTuple):
    """JAX-pytree output of :class:`PACTFlowScheduler`."""

    tau: Array
    log_std: Array
    plan_anchor: Array
    diagnostics: PACTFlowSchedulerDiagnostics


def _rms_normalize(x: Array, epsilon: float) -> Array:
    scale = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + epsilon)
    return x * scale


def _validate_scheduler_inputs(
    config: PACTFlowSchedulerConfig,
    aligned_ear: Array,
    iar_tokens: Array,
    state: Array,
    prefix_pooled: Array | None,
) -> None:
    if aligned_ear.ndim != 3 or aligned_ear.shape[1:] != (
        config.action_horizon,
        config.action_dim,
    ):
        raise ValueError(
            "aligned_ear must have shape "
            f"[B, {config.action_horizon}, {config.action_dim}], got {aligned_ear.shape}."
        )
    if iar_tokens.ndim != 3 or iar_tokens.shape[2] != config.iar_dim:
        raise ValueError(
            f"iar_tokens must have shape [B, S, {config.iar_dim}], got {iar_tokens.shape}."
        )
    if state.ndim != 2 or state.shape[1] != config.state_dim:
        raise ValueError(f"state must have shape [B, {config.state_dim}], got {state.shape}.")
    batch_size = aligned_ear.shape[0]
    if iar_tokens.shape[0] != batch_size or state.shape[0] != batch_size:
        raise ValueError("aligned_ear, iar_tokens, and state must have the same batch size.")
    if iar_tokens.shape[1] == 0:
        raise ValueError("iar_tokens must contain at least one token.")
    if config.use_prefix:
        if prefix_pooled is None:
            raise ValueError("prefix_pooled is required when config.use_prefix=True.")
        if prefix_pooled.ndim != 2 or prefix_pooled.shape != (batch_size, config.prefix_dim):
            raise ValueError(
                f"prefix_pooled must have shape [B, {config.prefix_dim}], got {prefix_pooled.shape}."
            )


class PACTFlowScheduler(nnx.Module):
    """Predict a cross-level, per-token PACT-Flow transport-time field.

    The default configuration has 147,106 trainable parameters.  Enabling the
    optional low-rank prefix path raises this to 225,058 parameters, keeping
    both variants below the 300k parameter budget.
    """

    def __init__(
        self,
        config: PACTFlowSchedulerConfig,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        hidden = config.hidden_dim

        self.position_embedding = nnx.Param(
            jnp.zeros((config.action_horizon, hidden), dtype=param_dtype)
        )
        self.ear_proj = nnx.Linear(
            config.action_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.iar_proj = nnx.Linear(
            config.iar_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.state_proj = nnx.Linear(
            config.state_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        fused_streams = 3
        if config.use_prefix:
            self.prefix_down_proj = nnx.Linear(
                config.prefix_dim,
                config.prefix_bottleneck_dim,
                rngs=rngs,
                param_dtype=param_dtype,
            )
            self.prefix_proj = nnx.Linear(
                config.prefix_bottleneck_dim,
                hidden,
                rngs=rngs,
                param_dtype=param_dtype,
            )
            fused_streams += 1

        self.fusion_in = nnx.Linear(
            fused_streams * hidden,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.fusion_hidden = nnx.Linear(
            hidden,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        zero_kernel = jax.nn.initializers.zeros
        self.tau_head = nnx.Linear(
            hidden,
            1,
            kernel_init=zero_kernel,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.log_std_head = nnx.Linear(
            hidden,
            1,
            kernel_init=zero_kernel,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        # EAR and final actions need not share an external normalization
        # affine.  Learn their bridge explicitly instead of treating raw EAR
        # as an action-space anchor.  Both paths start at zero, so the initial
        # tau=.975 bridge stays close to the verified Gaussian endpoint path.
        self.plan_anchor_from_ear = nnx.Linear(
            config.action_dim,
            config.action_dim,
            kernel_init=zero_kernel,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.plan_anchor_from_hidden = nnx.Linear(
            hidden,
            config.action_dim,
            kernel_init=zero_kernel,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(
        self,
        aligned_ear: Array,
        iar_tokens: Array,
        state: Array,
        prefix_pooled: Array | None = None,
    ) -> PACTFlowSchedulerOutput:
        """Return dynamic ``tau`` and uncertainty for each action token.

        Args:
            aligned_ear: EAR actions aligned to the final horizon, ``[B, 10, 32]``.
            iar_tokens: IAR representation tokens, ``[B, 18, 1024]`` by default.
            state: Normalized robot state, ``[B, 32]``.
            prefix_pooled: Optional pooled VLM prefix, ``[B, 2048]``.  It is
                ignored by the default ``use_prefix=False`` configuration.
        """

        config = self.config
        _validate_scheduler_inputs(config, aligned_ear, iar_tokens, state, prefix_pooled)

        compute_dtype = aligned_ear.dtype
        position = jnp.asarray(self.position_embedding[...], dtype=compute_dtype)
        ear_hidden = self.ear_proj(aligned_ear) + position[None, :, :]
        ear_hidden = _rms_normalize(ear_hidden, config.norm_epsilon)

        iar_hidden = self.iar_proj(iar_tokens)
        iar_hidden = _rms_normalize(iar_hidden, config.norm_epsilon)
        attention_logits = jnp.einsum("bth,bsh->bts", ear_hidden, iar_hidden)
        attention_logits = attention_logits * jnp.asarray(
            config.hidden_dim**-0.5,
            dtype=attention_logits.dtype,
        )
        attention = jax.nn.softmax(attention_logits, axis=-1)
        iar_context = jnp.einsum("bts,bsh->bth", attention, iar_hidden)

        state_hidden = self.state_proj(state)
        state_hidden = _rms_normalize(state_hidden, config.norm_epsilon)
        state_hidden = jnp.broadcast_to(
            state_hidden[:, None, :],
            ear_hidden.shape,
        )

        streams = [ear_hidden, iar_context, state_hidden]
        if config.use_prefix:
            assert prefix_pooled is not None
            prefix_hidden = nnx.swish(self.prefix_down_proj(prefix_pooled))
            prefix_hidden = self.prefix_proj(prefix_hidden)
            prefix_hidden = _rms_normalize(prefix_hidden, config.norm_epsilon)
            streams.append(jnp.broadcast_to(prefix_hidden[:, None, :], ear_hidden.shape))

        hidden = nnx.swish(self.fusion_in(jnp.concatenate(streams, axis=-1)))
        hidden = _rms_normalize(hidden + ear_hidden, config.norm_epsilon)
        hidden = _rms_normalize(
            hidden + nnx.swish(self.fusion_hidden(hidden)),
            config.norm_epsilon,
        )

        initial_tau_logit = math.log(config.initial_tau / (1.0 - config.initial_tau))
        tau_logits = self.tau_head(hidden)[..., 0] + jnp.asarray(
            initial_tau_logit,
            dtype=hidden.dtype,
        )
        tau = jax.nn.sigmoid(tau_logits)
        tau = jnp.clip(tau, config.tau_epsilon, 1.0 - config.tau_epsilon)
        log_std = self.log_std_head(hidden)[..., 0] + jnp.asarray(
            config.initial_log_std,
            dtype=hidden.dtype,
        )
        # Risk is the calibrated residual scale itself, not an unsupervised
        # decorative head.  The endpoint trainer supervises log_std through a
        # heteroscedastic NLL and calibration objective.
        risk = jnp.exp(jnp.clip(log_std, -5.0, 2.0))
        plan_anchor = self.plan_anchor_from_ear(aligned_ear) + self.plan_anchor_from_hidden(hidden)

        entropy_denominator = jnp.log(
            jnp.asarray(iar_tokens.shape[1], dtype=attention.dtype)
        )
        attention_entropy = -jnp.sum(
            attention * jnp.log(jnp.maximum(attention, 1e-8)),
            axis=-1,
        ) / jnp.maximum(entropy_denominator, 1e-8)
        tau_total_variation = jnp.mean(jnp.abs(tau[:, 1:] - tau[:, :-1]), axis=-1)

        return PACTFlowSchedulerOutput(
            tau=tau,
            log_std=log_std,
            plan_anchor=plan_anchor,
            diagnostics=PACTFlowSchedulerDiagnostics(
                tau_logits=tau_logits,
                risk=risk,
                iar_attention_entropy=attention_entropy,
                tau_mean=jnp.mean(tau, axis=-1),
                tau_total_variation=tau_total_variation,
            ),
        )


def build_plan_anchored_bridge(
    noise: Array,
    plan_anchor: Array,
    tau: Array,
    *,
    stop_gradient_anchor: bool = True,
) -> Array:
    """Construct ``x_i=tau_i*noise_i+(1-tau_i)*plan_anchor_i``.

    ``plan_anchor`` is normally a cheap action draft derived from EAR/IAR.  The
    function deliberately does not clip ``tau`` so invalid schedules remain
    visible to callers; :class:`PACTFlowScheduler` already emits values in the
    open unit interval.
    """

    if noise.ndim != 3 or plan_anchor.shape != noise.shape:
        raise ValueError(
            "noise and plan_anchor must have the same shape [B, T, D], got "
            f"{noise.shape} and {plan_anchor.shape}."
        )
    if tau.shape != noise.shape[:2]:
        raise ValueError(f"tau must have shape {noise.shape[:2]}, got {tau.shape}.")
    anchor = jax.lax.stop_gradient(plan_anchor) if stop_gradient_anchor else plan_anchor
    tau_expanded = jnp.asarray(tau, dtype=noise.dtype)[..., None]
    return tau_expanded * noise + (1.0 - tau_expanded) * anchor


def endpoint_from_bridge_velocity(
    bridge_state: Array,
    velocity: Array,
    tau: Array,
) -> Array:
    """Apply one token-wise interval update: ``a_hat=x_bridge-tau*v``."""

    if bridge_state.ndim != 3 or velocity.shape != bridge_state.shape:
        raise ValueError(
            "bridge_state and velocity must have the same shape [B, T, D], got "
            f"{bridge_state.shape} and {velocity.shape}."
        )
    if tau.shape != bridge_state.shape[:2]:
        raise ValueError(f"tau must have shape {bridge_state.shape[:2]}, got {tau.shape}.")
    tau_expanded = jnp.asarray(tau, dtype=bridge_state.dtype)[..., None]
    return bridge_state - tau_expanded * velocity


def bridge_velocity_target(
    bridge_state: Array,
    endpoint: Array,
    tau: Array,
    *,
    minimum_tau: float = 1e-4,
) -> Array:
    """Return the straight-path velocity target from a bridge state to data."""

    if minimum_tau <= 0.0:
        raise ValueError("minimum_tau must be positive.")
    if bridge_state.ndim != 3 or endpoint.shape != bridge_state.shape:
        raise ValueError(
            "bridge_state and endpoint must have the same shape [B, T, D], got "
            f"{bridge_state.shape} and {endpoint.shape}."
        )
    if tau.shape != bridge_state.shape[:2]:
        raise ValueError(f"tau must have shape {bridge_state.shape[:2]}, got {tau.shape}.")
    safe_tau = jnp.maximum(
        jnp.asarray(tau, dtype=bridge_state.dtype),
        jnp.asarray(minimum_tau, dtype=bridge_state.dtype),
    )[..., None]
    return (bridge_state - endpoint) / safe_tau


def scheduler_parameter_count(config: PACTFlowSchedulerConfig) -> int:
    """Return the exact trainable scalar count implied by ``config``."""

    hidden = config.hidden_dim

    def linear(in_features: int, out_features: int) -> int:
        return in_features * out_features + out_features

    count = config.action_horizon * hidden
    count += linear(config.action_dim, hidden)
    count += linear(config.iar_dim, hidden)
    count += linear(config.state_dim, hidden)
    fused_streams = 3
    if config.use_prefix:
        count += linear(config.prefix_dim, config.prefix_bottleneck_dim)
        count += linear(config.prefix_bottleneck_dim, hidden)
        fused_streams += 1
    count += linear(fused_streams * hidden, hidden)
    count += linear(hidden, hidden)
    count += 2 * linear(hidden, 1)
    count += linear(config.action_dim, config.action_dim)
    count += linear(hidden, config.action_dim)
    return count


__all__ = [
    "PACTFlowScheduler",
    "PACTFlowSchedulerConfig",
    "PACTFlowSchedulerDiagnostics",
    "PACTFlowSchedulerOutput",
    "bridge_velocity_target",
    "build_plan_anchored_bridge",
    "endpoint_from_bridge_velocity",
    "scheduler_parameter_count",
]
