"""Lightweight observation-conditioned transport of a cached Action-CoT plan.

The latest observation is allowed to change *where* the cached EAR is sampled,
but it has no direct path to the returned action.  The action decoder consumes
only the first few tokens of the transported EAR.  This makes plan transport an
explicit causal bottleneck instead of learning another observation-to-action
residual policy.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

_PARAMETER_LIMIT = 5_000_000


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
    max_log_speed: float = 0.25
    cnn_channels: tuple[int, ...] = (16, 32, 64, 96)
    cnn_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3)
    hidden_dim: int = 128
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
        if self.max_log_speed <= 0:
            raise ValueError("max_log_speed must be positive.")
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
            self.state_delta_proj(
                jnp.concatenate([anchor_state, current_state, state_delta], axis=-1)
            )
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
        return jnp.clip(phase, 0.0, config.max_phase)

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
        """Return ``(action, transported_ear, phase)`` for the current tick."""

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

        phase = self._predict_phase(
            anchor_images,
            current_images,
            anchor_state,
            current_state,
            cached_ear,
            cached_iar,
            cache_age,
        )
        transported_ear = interpolate_ear(cached_ear, phase)
        decoder_input = transported_ear[:, : config.decoder_tokens].reshape(
            (batch_size, config.decoder_tokens * config.action_dim)
        )
        decoder_feature = jax.nn.silu(self.decoder_hidden(decoder_input))
        action = transported_ear[:, 0] + self.decoder_out(decoder_feature)
        return action, transported_ear, phase

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
