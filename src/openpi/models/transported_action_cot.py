"""Lightweight observation-conditioned transport of a cached Action-CoT plan.

The latest observation is allowed to change *where* the cached EAR is sampled,
but it has no direct path to the returned action.  The action decoder consumes
only the first few tokens of the transported EAR.  This makes plan transport an
explicit causal bottleneck instead of learning another observation-to-action
residual policy.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

_PARAMETER_LIMIT = 5_000_000
_GEOMETRY_DIM = 6
_GRIPPER_INDEX = 6

CorrectionMode = Literal["phase", "plan", "direct", "event"]


class TransportedActionCoTOutput(NamedTuple):
    """Detailed outputs for training, diagnostics, and causal ablations.

    ``transported_ear`` is the phase-warped cached plan. ``revised_ear`` also
    includes the explicit low-rank plan correction when ``correction_mode`` is
    ``"plan"``.  In ``"direct"`` mode the plan is unchanged and
    ``direct_action_residual`` exposes the observation-to-action baseline.
    """

    action: jax.Array
    transported_ear: jax.Array
    revised_ear: jax.Array
    phase: jax.Array
    event_phase_offset: jax.Array
    geometry_residual: jax.Array
    gripper_logits: jax.Array
    event_prob: jax.Array
    direct_action_residual: jax.Array


@dataclasses.dataclass(frozen=True)
class TransportedActionCoTConfig:
    """Architecture and continuous phase-transport settings."""

    image_views: int = 2
    image_size: int = 64
    image_channels: int = 3
    state_dim: int = 32
    action_dim: int = 32
    ear_horizon: int = 15
    iar_tokens: int = 18
    iar_dim: int = 1024
    coarse_time_stride: int = 2
    decoder_tokens: int = 3
    max_phase: float = 14.0
    max_phase_offset: float = 3.0
    max_event_phase_offset: float = 2.0
    max_log_speed: float = 0.25
    cnn_channels: tuple[int, ...] = (16, 32, 64, 96)
    cnn_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3)
    hidden_dim: int = 128
    correction_mode: CorrectionMode = "phase"
    geometry_rank: int = 4
    geometry_scale: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    enable_geometry_correction: bool = True
    enable_event_shift: bool = True
    isolate_event_gradients: bool = False
    gripper_logit_residual_scale: float = 2.0
    direct_residual_scale: float = 0.5
    max_parameters: int = _PARAMETER_LIMIT

    def __post_init__(self) -> None:
        integer_fields = (
            "image_views",
            "image_size",
            "image_channels",
            "state_dim",
            "action_dim",
            "ear_horizon",
            "iar_tokens",
            "iar_dim",
            "coarse_time_stride",
            "decoder_tokens",
            "hidden_dim",
            "geometry_rank",
            "max_parameters",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.image_views != 2:
            raise ValueError("TransportedActionCoTExecutor requires exactly two image views.")
        if len(self.cnn_channels) != len(self.cnn_kernel_sizes) or not self.cnn_channels:
            raise ValueError("cnn_channels and cnn_kernel_sizes must have the same non-zero length.")
        if any(channel <= 0 for channel in self.cnn_channels):
            raise ValueError("All CNN channel counts must be positive.")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.cnn_kernel_sizes):
            raise ValueError("CNN kernel sizes must be positive odd integers.")
        if self.decoder_tokens > self.ear_horizon:
            raise ValueError("decoder_tokens may not exceed ear_horizon.")
        if self.max_phase <= 0:
            raise ValueError("max_phase must be positive.")
        if self.max_phase > self.ear_horizon - 1:
            raise ValueError("max_phase may not exceed the last EAR token index.")
        if self.max_phase_offset <= 0:
            raise ValueError("max_phase_offset must be positive.")
        if self.max_event_phase_offset <= 0:
            raise ValueError("max_event_phase_offset must be positive.")
        if self.max_log_speed <= 0:
            raise ValueError("max_log_speed must be positive.")
        if self.correction_mode not in ("phase", "plan", "direct", "event"):
            raise ValueError(
                "correction_mode must be one of 'phase', 'plan', 'direct', "
                f"or 'event', got {self.correction_mode!r}."
            )
        if self.geometry_rank != 4:
            raise ValueError("geometry_rank must be four for the matched plan/direct pilot.")
        if len(self.geometry_scale) != _GEOMETRY_DIM:
            raise ValueError(f"geometry_scale must contain {_GEOMETRY_DIM} values.")
        if any(scale <= 0 for scale in self.geometry_scale):
            raise ValueError("All geometry_scale values must be positive.")
        if self.gripper_logit_residual_scale <= 0:
            raise ValueError("gripper_logit_residual_scale must be positive.")
        if self.direct_residual_scale <= 0:
            raise ValueError("direct_residual_scale must be positive.")
        if self.correction_mode in ("plan", "event") and self.action_dim <= _GRIPPER_INDEX:
            raise ValueError(f"Gripper correction requires action_dim greater than {_GRIPPER_INDEX}.")
        if self.max_parameters > _PARAMETER_LIMIT:
            raise ValueError(f"max_parameters may not exceed {_PARAMETER_LIMIT}.")
        estimated = estimate_parameter_count(self)
        if estimated >= self.max_parameters:
            raise ValueError(
                f"Transported Action-CoT configuration has an estimated {estimated:,} parameters; "
                f"the limit is {self.max_parameters:,}."
            )


def _linear_parameter_count(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def estimate_parameter_count(config: TransportedActionCoTConfig) -> int:
    """Return the exact trainable parameter count for this implementation."""

    count = 0
    in_channels = config.image_channels
    for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernel_sizes, strict=True):
        count += kernel_size * kernel_size * in_channels * out_channels + out_channels
        in_channels = out_channels

    hidden = config.hidden_dim
    image_summary_dim = config.image_views * config.cnn_channels[-1]
    count += _linear_parameter_count(4 * image_summary_dim, hidden)
    count += _linear_parameter_count(3 * config.state_dim, hidden)
    count += _linear_parameter_count(config.action_dim, hidden)
    count += _linear_parameter_count(2 * hidden, hidden)
    count += _linear_parameter_count(config.iar_dim, hidden)
    count += _linear_parameter_count(1, hidden)
    count += _linear_parameter_count(5 * hidden, hidden)
    count += _linear_parameter_count(hidden, hidden)
    count += _linear_parameter_count(hidden, 2)
    count += _linear_parameter_count(config.decoder_tokens * config.action_dim, hidden)
    count += _linear_parameter_count(hidden, config.action_dim)
    if config.correction_mode == "plan":
        count += _linear_parameter_count(hidden, config.geometry_rank)
        count += _linear_parameter_count(
            hidden,
            config.geometry_rank * _GEOMETRY_DIM,
        )
        count += _linear_parameter_count(hidden, config.geometry_rank)
    elif config.correction_mode == "direct":
        count += _linear_parameter_count(hidden, config.action_dim)
    elif config.correction_mode == "event":
        count += _linear_parameter_count(hidden, 1)
    return count


def _require_shape(array: jax.Array, expected: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(expected):
        raise ValueError(f"{name} must have rank {len(expected)}, got shape {array.shape}.")
    for axis, (actual, wanted) in enumerate(zip(array.shape, expected, strict=True)):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} axis {axis} must be {wanted}, got shape {array.shape}.")


def interpolate_ear(cached_ear: jax.Array, phase: jax.Array) -> jax.Array:
    """Differentiably sample every EAR token at a continuous phase.

    The integer neighbours are selected discretely, while interpolation
    weights remain differentiable with respect to ``phase`` almost everywhere.
    """

    cached_ear = jnp.asarray(cached_ear, dtype=jnp.float32)
    phase = jnp.asarray(phase, dtype=jnp.float32)
    _require_shape(cached_ear, (None, None, None), "cached_ear")
    _require_shape(phase, (cached_ear.shape[0], None), "phase")
    if cached_ear.shape[1] == 0:
        raise ValueError("cached_ear must contain at least one token.")

    clipped_phase = jnp.clip(phase, 0.0, float(cached_ear.shape[1] - 1))
    lower_index = jnp.floor(clipped_phase).astype(jnp.int32)
    upper_index = jnp.minimum(lower_index + 1, cached_ear.shape[1] - 1)
    interpolation = clipped_phase - lower_index.astype(clipped_phase.dtype)
    batch_indices = jnp.arange(cached_ear.shape[0], dtype=jnp.int32)[:, None]
    lower = cached_ear[batch_indices, lower_index]
    upper = cached_ear[batch_indices, upper_index]
    return lower + interpolation[..., None] * (upper - lower)


class _Conv2D(nnx.Module):
    """Small shared NHWC convolution used for both anchor and current images."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: Any,
    ) -> None:
        fan_in = kernel_size * kernel_size * in_channels
        scale = jnp.sqrt(jnp.asarray(2.0 / fan_in, dtype=jnp.float32))
        kernel = jax.random.normal(
            rngs.params(),
            (kernel_size, kernel_size, in_channels, out_channels),
            dtype=param_dtype,
        )
        self.kernel = nnx.Param(kernel * scale.astype(param_dtype))
        self.bias = nnx.Param(jnp.zeros((out_channels,), dtype=param_dtype))

    def __call__(self, images: jax.Array) -> jax.Array:
        outputs = jax.lax.conv_general_dilated(
            images,
            self.kernel.value,
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return outputs + self.bias.value


class TransportedActionCoTExecutor(nnx.Module):
    """Transport a cached EAR, then decode an action solely from that plan."""

    def __init__(
        self,
        config: TransportedActionCoTConfig,
        *,
        rngs: nnx.Rngs,
        param_dtype: Any = jnp.float32,
    ) -> None:
        self.config = config
        hidden = config.hidden_dim

        image_convs = []
        in_channels = config.image_channels
        for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernel_sizes, strict=True):
            image_convs.append(
                _Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
            )
            in_channels = out_channels
        self.image_convs = image_convs

        image_summary_dim = config.image_views * config.cnn_channels[-1]
        self.image_delta_proj = nnx.Linear(
            4 * image_summary_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.state_delta_proj = nnx.Linear(
            3 * config.state_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.ear_token_proj = nnx.Linear(
            config.action_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.ear_summary_proj = nnx.Linear(
            2 * hidden,
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
        self.age_proj = nnx.Linear(1, hidden, rngs=rngs, param_dtype=param_dtype)
        self.phase_fusion = nnx.Linear(
            5 * hidden,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.phase_hidden = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        # Zero initialization gives phase_j = age / stride + j exactly.  The
        # observation pathway becomes active as soon as this head is updated.
        self.phase_out = nnx.Linear(
            hidden,
            2,
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

        # This decoder has no observation feature input.  Its zero-initialized
        # correction starts from the first transported token while retaining
        # all first-three-token information for learning.
        self.decoder_hidden = nnx.Linear(
            config.decoder_tokens * config.action_dim,
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.decoder_out = nnx.Linear(
            hidden,
            config.action_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

        if config.correction_mode == "plan":
            # A shared rank-four temporal basis keeps both geometry and
            # gripper correction inside the explicit EAR plan.  The
            # observation-conditioned coefficient heads are zero initialized,
            # so enabling this mode starts from phase transport.
            self.plan_temporal_basis = nnx.Linear(
                hidden,
                config.geometry_rank,
                rngs=rngs,
                param_dtype=param_dtype,
            )
            self.geometry_coefficients = nnx.Linear(
                hidden,
                config.geometry_rank * _GEOMETRY_DIM,
                rngs=rngs,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
            self.gripper_coefficients = nnx.Linear(
                hidden,
                config.geometry_rank,
                rngs=rngs,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        elif config.correction_mode == "direct":
            # Matched observation-to-action residual baseline.  With the
            # default dimensions this has exactly as many extra parameters as
            # the three plan-correction projections above.
            self.direct_action_out = nnx.Linear(
                hidden,
                config.action_dim,
                rngs=rngs,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )
        elif config.correction_mode == "event":
            # The event-factorized model receives exactly one additional
            # scalar and spends it on *when* to sample the cached gripper plan.
            self.event_scalar_out = nnx.Linear(
                hidden,
                1,
                rngs=rngs,
                param_dtype=param_dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
            )

    @staticmethod
    def _normalize_images(images: jax.Array) -> jax.Array:
        mean = jnp.mean(images, axis=(1, 2), keepdims=True)
        variance = jnp.mean(jnp.square(images - mean), axis=(1, 2), keepdims=True)
        return (images - mean) * jax.lax.rsqrt(variance + 1e-6)

    def _encode_images(self, images: jax.Array) -> jax.Array:
        config = self.config
        batch_size = images.shape[0]
        encoded = images.reshape(
            (
                batch_size * config.image_views,
                config.image_size,
                config.image_size,
                config.image_channels,
            )
        )
        encoded = self._normalize_images(encoded)
        for convolution in self.image_convs:
            encoded = jax.nn.silu(convolution(encoded))
        encoded = jnp.mean(encoded, axis=(1, 2))
        return encoded.reshape((batch_size, config.image_views * config.cnn_channels[-1]))

    def _encode_ear(self, cached_ear: jax.Array) -> jax.Array:
        tokens = jax.nn.silu(self.ear_token_proj(cached_ear))
        positions = jnp.linspace(-1.0, 1.0, cached_ear.shape[1], dtype=tokens.dtype)
        content = jnp.mean(tokens, axis=1)
        trend = jnp.mean(tokens * positions[None, :, None], axis=1)
        return jax.nn.silu(self.ear_summary_proj(jnp.concatenate([content, trend], axis=-1)))

    def _encode_iar(self, cached_iar: jax.Array | None, *, batch_size: int) -> jax.Array:
        config = self.config
        if cached_iar is None:
            return jnp.zeros((batch_size, config.hidden_dim), dtype=jnp.float32)
        if cached_iar.shape[1] == 0:
            raise ValueError("cached_iar must contain at least one token.")
        if cached_iar.shape[1] > config.iar_tokens:
            raise ValueError(
                f"cached_iar has {cached_iar.shape[1]} tokens; the configured maximum is {config.iar_tokens}."
            )
        return jnp.mean(jax.nn.silu(self.iar_proj(cached_iar)), axis=1)

    def _predict_phase_and_context(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        config = self.config
        batch_size = cached_ear.shape[0]
        anchor_visual = self._encode_images(anchor_images)
        current_visual = self._encode_images(current_images)
        visual_delta = current_visual - anchor_visual
        image_feature = jax.nn.silu(
            self.image_delta_proj(
                jnp.concatenate(
                    [
                        anchor_visual,
                        current_visual,
                        visual_delta,
                        jnp.abs(visual_delta),
                    ],
                    axis=-1,
                )
            )
        )

        state_delta = current_state - anchor_state
        state_feature = jax.nn.silu(
            self.state_delta_proj(jnp.concatenate([anchor_state, current_state, state_delta], axis=-1))
        )
        ear_feature = self._encode_ear(cached_ear)
        iar_feature = self._encode_iar(cached_iar, batch_size=batch_size)
        normalized_age = cache_age[:, None] / jnp.maximum(
            jnp.asarray(config.coarse_time_stride * config.max_phase, dtype=jnp.float32),
            1.0,
        )
        age_feature = jax.nn.silu(self.age_proj(normalized_age))
        hidden = jax.nn.silu(
            self.phase_fusion(
                jnp.concatenate(
                    [image_feature, state_feature, ear_feature, iar_feature, age_feature],
                    axis=-1,
                )
            )
        )
        hidden = hidden + jax.nn.silu(self.phase_hidden(hidden))
        phase_parameters = self.phase_out(hidden)

        nominal_start = jnp.clip(
            cache_age / jnp.asarray(config.coarse_time_stride, dtype=jnp.float32),
            0.0,
            config.max_phase,
        )
        start_offset = config.max_phase_offset * jnp.tanh(phase_parameters[:, 0])
        speed = jnp.exp(config.max_log_speed * jnp.tanh(phase_parameters[:, 1]))
        token_positions = jnp.arange(config.ear_horizon, dtype=jnp.float32)
        phase = nominal_start[:, None] + start_offset[:, None] + speed[:, None] * token_positions[None, :]
        return jnp.clip(phase, 0.0, config.max_phase), hidden

    def _predict_phase(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> jax.Array:
        """Return phase only, preserving the original private helper API."""

        phase, _ = self._predict_phase_and_context(
            anchor_images,
            current_images,
            anchor_state,
            current_state,
            cached_ear,
            cached_iar,
            cache_age,
        )
        return phase

    def _decode_action(self, ear: jax.Array) -> jax.Array:
        """Decode an action from EAR tokens without an observation bypass."""

        config = self.config
        batch_size = ear.shape[0]
        decoder_input = ear[:, : config.decoder_tokens].reshape((batch_size, config.decoder_tokens * config.action_dim))
        decoder_feature = jax.nn.silu(self.decoder_hidden(decoder_input))
        return ear[:, 0] + self.decoder_out(decoder_feature)

    def _gripper_statistics(self, ear: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return gripper state logits and adjacent-token event probabilities."""

        if ear.shape[-1] <= _GRIPPER_INDEX:
            logits = jnp.zeros(ear.shape[:2], dtype=ear.dtype)
            return logits, jnp.zeros((ear.shape[0], max(ear.shape[1] - 1, 0)), dtype=ear.dtype)
        clipped_gripper = jnp.clip(
            ear[..., _GRIPPER_INDEX],
            -1.0 + 1e-4,
            1.0 - 1e-4,
        )
        logits = jnp.arctanh(clipped_gripper)
        probability = jax.nn.sigmoid(2.0 * logits)
        event_probability = (
            probability[:, 1:] * (1.0 - probability[:, :-1]) + (1.0 - probability[:, 1:]) * probability[:, :-1]
        )
        return logits, event_probability

    def _revise_plan(
        self,
        transported_ear: jax.Array,
        context: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Apply the explicit rank-four plan correction, when configured."""

        config = self.config
        batch_size = transported_ear.shape[0]
        geometry_residual = jnp.zeros(
            (batch_size, config.ear_horizon, _GEOMETRY_DIM),
            dtype=transported_ear.dtype,
        )
        if config.correction_mode != "plan":
            gripper_logits, event_probability = self._gripper_statistics(transported_ear)
            return transported_ear, geometry_residual, gripper_logits, event_probability

        plan_tokens = jax.nn.silu(self.ear_token_proj(transported_ear))
        temporal_basis = jnp.tanh(self.plan_temporal_basis(plan_tokens))
        rank_normalizer = jnp.sqrt(jnp.asarray(config.geometry_rank, dtype=transported_ear.dtype))
        if config.enable_geometry_correction:
            geometry_coefficients = self.geometry_coefficients(context).reshape(
                (batch_size, config.geometry_rank, _GEOMETRY_DIM)
            )
            geometry_scale = jnp.asarray(config.geometry_scale, dtype=transported_ear.dtype)
            geometry_coefficients = jnp.tanh(geometry_coefficients) * geometry_scale[None, None, :]
            geometry_residual = (
                jnp.einsum(
                    "bhr,brd->bhd",
                    temporal_basis,
                    geometry_coefficients,
                )
                / rank_normalizer
            )
        revised_ear = transported_ear.at[..., :_GEOMETRY_DIM].add(geometry_residual)

        event_context = context
        event_temporal_basis = temporal_basis
        event_plan = transported_ear
        if config.isolate_event_gradients:
            event_context = jax.lax.stop_gradient(context)
            event_temporal_basis = jnp.tanh(self.plan_temporal_basis(jax.lax.stop_gradient(plan_tokens)))
            event_plan = jax.lax.stop_gradient(transported_ear)
        base_gripper = jnp.clip(
            event_plan[..., _GRIPPER_INDEX],
            -1.0 + 1e-4,
            1.0 - 1e-4,
        )
        base_gripper_logits = jnp.arctanh(base_gripper)
        gripper_coefficients = config.gripper_logit_residual_scale * jnp.tanh(self.gripper_coefficients(event_context))
        gripper_logit_residual = (
            jnp.einsum(
                "bhr,br->bh",
                event_temporal_basis,
                gripper_coefficients,
            )
            / rank_normalizer
        )
        gripper_logits = base_gripper_logits + gripper_logit_residual
        revised_ear = revised_ear.at[..., _GRIPPER_INDEX].set(jnp.tanh(gripper_logits))
        gripper_probability = jax.nn.sigmoid(2.0 * gripper_logits)
        event_probability = (
            gripper_probability[:, 1:] * (1.0 - gripper_probability[:, :-1])
            + (1.0 - gripper_probability[:, 1:]) * gripper_probability[:, :-1]
        )
        return revised_ear, geometry_residual, gripper_logits, event_probability

    def _revise_event_timing(
        self,
        transported_ear: jax.Array,
        cached_ear: jax.Array,
        phase: jax.Array,
        context: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Factor gripper timing from the shared continuous-action phase.

        ``event`` predicts one bounded phase offset and resamples only the
        cached gripper channel.  The continuous six-dimensional plan remains
        exactly the ordinary phase transport.
        """

        config = self.config
        batch_size = transported_ear.shape[0]
        event_context = context
        event_phase = phase
        event_cache = cached_ear
        if config.isolate_event_gradients:
            # Event losses must not update the shared observation encoder or
            # the base phase path in the strict-isolation ablation.
            event_context = jax.lax.stop_gradient(context)
            event_phase = jax.lax.stop_gradient(phase)
            event_cache = jax.lax.stop_gradient(cached_ear)

        event_scalar = self.event_scalar_out(event_context)[:, 0]
        event_phase_offset = jnp.zeros((batch_size,), dtype=transported_ear.dtype)
        if config.correction_mode == "event":
            learned_offset = config.max_event_phase_offset * jnp.tanh(event_scalar)
            if config.enable_event_shift:
                event_phase_offset = learned_offset
            else:
                # Keep the identical parameter tree for a matched delta=0
                # baseline while intentionally blocking head gradients.
                event_phase_offset = jnp.zeros_like(learned_offset)
            shifted_phase = jnp.clip(
                event_phase + event_phase_offset[:, None],
                0.0,
                config.max_phase,
            )
            shifted_gripper = interpolate_ear(
                event_cache[..., _GRIPPER_INDEX : _GRIPPER_INDEX + 1],
                shifted_phase,
            )[..., 0]
            if config.isolate_event_gradients:
                # Straight-through identity for the base gripper path: forward
                # uses the shifted event plan, while base-plan gradients remain
                # exactly those of the matched phase transport at delta=0.
                base_gripper = transported_ear[..., _GRIPPER_INDEX]
                shifted_gripper = base_gripper + shifted_gripper - jax.lax.stop_gradient(base_gripper)
            revised_ear = transported_ear.at[..., _GRIPPER_INDEX].set(shifted_gripper)
        else:
            raise ValueError(f"Event timing revision does not support mode {config.correction_mode!r}.")

        gripper_logits, event_probability = self._gripper_statistics(revised_ear)
        return revised_ear, event_phase_offset, gripper_logits, event_probability

    def forward_with_details(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> TransportedActionCoTOutput:
        """Return detailed phase, plan-correction, event, and action outputs."""

        config = self.config
        anchor_images = jnp.asarray(anchor_images, dtype=jnp.float32)
        current_images = jnp.asarray(current_images, dtype=jnp.float32)
        anchor_state = jnp.asarray(anchor_state, dtype=jnp.float32)
        current_state = jnp.asarray(current_state, dtype=jnp.float32)
        cached_ear = jnp.asarray(cached_ear, dtype=jnp.float32)
        cache_age = jnp.asarray(cache_age, dtype=jnp.float32)
        batch_size = anchor_images.shape[0]

        image_shape = (
            None,
            config.image_views,
            config.image_size,
            config.image_size,
            config.image_channels,
        )
        _require_shape(anchor_images, image_shape, "anchor_images")
        _require_shape(
            current_images,
            (
                batch_size,
                config.image_views,
                config.image_size,
                config.image_size,
                config.image_channels,
            ),
            "current_images",
        )
        _require_shape(anchor_state, (batch_size, config.state_dim), "anchor_state")
        _require_shape(current_state, (batch_size, config.state_dim), "current_state")
        _require_shape(
            cached_ear,
            (batch_size, config.ear_horizon, config.action_dim),
            "cached_ear",
        )
        _require_shape(cache_age, (batch_size,), "cache_age")
        if cached_iar is not None:
            cached_iar = jnp.asarray(cached_iar, dtype=jnp.float32)
            _require_shape(cached_iar, (batch_size, None, config.iar_dim), "cached_iar")

        phase, context = self._predict_phase_and_context(
            anchor_images,
            current_images,
            anchor_state,
            current_state,
            cached_ear,
            cached_iar,
            cache_age,
        )
        transported_ear = interpolate_ear(cached_ear, phase)
        event_phase_offset = jnp.zeros((batch_size,), dtype=transported_ear.dtype)
        if config.correction_mode == "event":
            revised_ear, event_phase_offset, gripper_logits, event_probability = self._revise_event_timing(
                transported_ear,
                cached_ear,
                phase,
                context,
            )
            geometry_residual = jnp.zeros(
                (batch_size, config.ear_horizon, _GEOMETRY_DIM),
                dtype=transported_ear.dtype,
            )
        else:
            revised_ear, geometry_residual, gripper_logits, event_probability = self._revise_plan(
                transported_ear,
                context,
            )
        # Event-factorized modes must not feed their gripper revision through
        # the decoder, because that would also perturb the continuous action.
        decoder_ear = revised_ear if config.correction_mode == "plan" else transported_ear
        action = self._decode_action(decoder_ear)
        direct_action_residual = jnp.zeros_like(action)
        if config.correction_mode == "plan":
            # The discrete gripper state is part of the revised EAR and cannot
            # be overwritten by a continuous decoder residual.
            action = action.at[..., _GRIPPER_INDEX].set(revised_ear[:, 0, _GRIPPER_INDEX])
        elif config.correction_mode == "event":
            # Express the event revision as a delta around the ordinary
            # phase-only decoder. Therefore delta=0 is exactly matched to the
            # phase baseline even after the decoder has learned a residual.
            action = action.at[..., _GRIPPER_INDEX].add(
                revised_ear[:, 0, _GRIPPER_INDEX] - transported_ear[:, 0, _GRIPPER_INDEX]
            )
        elif config.correction_mode == "direct":
            direct_action_residual = config.direct_residual_scale * jnp.tanh(self.direct_action_out(context))
            action = action + direct_action_residual

        return TransportedActionCoTOutput(
            action=action,
            transported_ear=transported_ear,
            revised_ear=revised_ear,
            phase=phase,
            event_phase_offset=event_phase_offset,
            geometry_residual=geometry_residual,
            gripper_logits=gripper_logits,
            event_prob=event_probability,
            direct_action_residual=direct_action_residual,
        )

    def forward_with_aux(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(action, revised_ear, phase)`` for the current tick.

        With the default ``correction_mode="phase"``, ``revised_ear`` is
        exactly the original phase-transported EAR and the parameter tree is
        unchanged.
        """

        output = self.forward_with_details(
            anchor_images,
            current_images,
            anchor_state,
            current_state,
            cached_ear,
            cached_iar,
            cache_age,
        )
        return output.action, output.revised_ear, output.phase

    def __call__(
        self,
        anchor_images: jax.Array,
        current_images: jax.Array,
        anchor_state: jax.Array,
        current_state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> jax.Array:
        action, _, _ = self.forward_with_aux(
            anchor_images,
            current_images,
            anchor_state,
            current_state,
            cached_ear,
            cached_iar,
            cache_age,
        )
        return action
