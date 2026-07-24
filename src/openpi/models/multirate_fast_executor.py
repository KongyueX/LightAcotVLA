"""Lightweight high-rate executor for a cached Action-CoT plan.

The slow ACoT branch can refresh ``cached_ear`` and ``cached_iar`` at a lower
frequency.  Between refreshes, this module consumes the latest two camera
views and proprioception, selects the EAR token aligned with ``cache_age``,
and predicts a bounded residual around that token.

This file is deliberately independent from ``ACOT_VLA`` so the executor can
be trained and profiled before changing the existing policy path.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

_PARAMETER_LIMIT = 10_000_000


@dataclasses.dataclass(frozen=True)
class MultiRateFastExecutorConfig:
    """Architecture and phase-alignment settings for the fast executor."""

    image_views: int = 2
    image_size: int = 64
    image_channels: int = 3
    state_dim: int = 32
    action_dim: int = 32
    ear_horizon: int = 15
    iar_tokens: int = 18
    iar_dim: int = 1024
    max_cache_age: int = 3
    coarse_time_stride: int = 2
    cnn_channels: tuple[int, ...] = (32, 64, 96, 128)
    cnn_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3)
    hidden_dim: int = 256
    attention_heads: int = 4
    residual_scale: float = 0.5
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
            "hidden_dim",
            "attention_heads",
            "coarse_time_stride",
            "max_parameters",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_cache_age < 0:
            raise ValueError("max_cache_age must be non-negative.")
        if self.image_views != 2:
            raise ValueError("The fast executor interface requires exactly two image views.")
        if len(self.cnn_channels) != len(self.cnn_kernel_sizes) or not self.cnn_channels:
            raise ValueError("cnn_channels and cnn_kernel_sizes must have the same non-zero length.")
        if any(channel <= 0 for channel in self.cnn_channels):
            raise ValueError("All CNN channel counts must be positive.")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.cnn_kernel_sizes):
            raise ValueError("CNN kernel sizes must be positive odd integers.")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        if self.max_cache_age > self.coarse_time_stride * (self.ear_horizon - 1):
            raise ValueError("The largest cache age must be covered by the coarse EAR time range.")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive.")
        if self.max_parameters > _PARAMETER_LIMIT:
            raise ValueError(f"max_parameters may not exceed {_PARAMETER_LIMIT}.")
        estimated = estimate_parameter_count(self)
        if estimated >= self.max_parameters:
            raise ValueError(
                f"Fast executor configuration has an estimated {estimated:,} parameters; "
                f"the limit is {self.max_parameters:,}."
            )


@dataclasses.dataclass(frozen=True)
class FastExecutorLossConfig:
    """Huber objective with extra weight on gripper action dimensions."""

    action_dim: int = 32
    huber_delta: float = 0.1
    gripper_indices: tuple[int, ...] = (6,)
    gripper_weight: float = 4.0

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        if self.gripper_weight <= 0:
            raise ValueError("gripper_weight must be positive.")
        if len(set(self.gripper_indices)) != len(self.gripper_indices):
            raise ValueError("gripper_indices must not contain duplicates.")
        if any(index < 0 or index >= self.action_dim for index in self.gripper_indices):
            raise ValueError("Every gripper index must be inside the action dimension.")


DEFAULT_FAST_EXECUTOR_LOSS_CONFIG = FastExecutorLossConfig()


def _linear_parameter_count(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def _downsampled_image_size(config: MultiRateFastExecutorConfig) -> int:
    size = config.image_size
    for _ in config.cnn_channels:
        size = (size + 1) // 2
    return size


def estimate_parameter_count(config: MultiRateFastExecutorConfig) -> int:
    """Return the exact trainable parameter count for this implementation."""

    count = 0
    in_channels = config.image_channels
    for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernel_sizes, strict=True):
        count += kernel_size * kernel_size * in_channels * out_channels + out_channels
        in_channels = out_channels

    hidden = config.hidden_dim
    spatial_tokens = _downsampled_image_size(config) ** 2
    count += _linear_parameter_count(config.cnn_channels[-1], hidden)
    count += _linear_parameter_count(hidden, hidden)
    count += config.image_views * spatial_tokens * hidden
    count += _linear_parameter_count(config.action_dim, hidden)
    count += 4 * _linear_parameter_count(hidden, hidden)
    count += _linear_parameter_count(2 * hidden, hidden)
    count += _linear_parameter_count(3 * hidden, hidden)
    count += _linear_parameter_count(config.iar_dim, hidden)
    count += _linear_parameter_count(hidden, hidden)
    count += config.iar_tokens * hidden
    count += _linear_parameter_count(config.state_dim, hidden)
    count += _linear_parameter_count(2 * hidden, hidden)
    count += _linear_parameter_count(3 * hidden, hidden)
    count += _linear_parameter_count(2 * hidden, hidden)
    count += _linear_parameter_count(hidden, hidden)
    count += 2 * _linear_parameter_count(hidden, config.action_dim)
    count += config.ear_horizon * hidden
    count += (config.max_cache_age + 1) * hidden
    return count


def _require_shape(array: jax.Array, expected: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(expected):
        raise ValueError(f"{name} must have rank {len(expected)}, got shape {array.shape}.")
    for axis, (actual, wanted) in enumerate(zip(array.shape, expected, strict=True)):
        if wanted is not None and actual != wanted:
            raise ValueError(f"{name} axis {axis} must be {wanted}, got shape {array.shape}.")


def phase_aligned_ear_token(
    cached_ear: jax.Array,
    cache_age: jax.Array,
    *,
    max_cache_age: int = 3,
    coarse_time_stride: int = 2,
) -> jax.Array:
    """Interpolate the EAR action corresponding to the current raw-action time.

    EAR tokens are sampled every ``coarse_time_stride`` raw control steps.  For
    the default stride of two, ages 0/1/2/3 map to token phases
    0/0.5/1/1.5 rather than directly selecting tokens 0/1/2/3.
    """

    cached_ear = jnp.asarray(cached_ear, dtype=jnp.float32)
    cache_age = jnp.asarray(cache_age, dtype=jnp.int32)
    _require_shape(cached_ear, (None, None, None), "cached_ear")
    _require_shape(cache_age, (cached_ear.shape[0],), "cache_age")
    if max_cache_age < 0:
        raise ValueError("max_cache_age must be non-negative.")
    if coarse_time_stride <= 0:
        raise ValueError("coarse_time_stride must be positive.")

    clipped_age = jnp.clip(cache_age, 0, max_cache_age)
    phase = clipped_age.astype(jnp.float32) / coarse_time_stride
    lower_index = jnp.minimum(jnp.floor(phase).astype(jnp.int32), cached_ear.shape[1] - 1)
    upper_index = jnp.minimum(lower_index + 1, cached_ear.shape[1] - 1)
    interpolation = phase - lower_index.astype(phase.dtype)
    lower = jnp.take_along_axis(cached_ear, lower_index[:, None, None], axis=1)[:, 0, :]
    upper = jnp.take_along_axis(cached_ear, upper_index[:, None, None], axis=1)[:, 0, :]
    return lower + interpolation[:, None] * (upper - lower)


class _Conv2D(nnx.Module):
    """Minimal NHWC convolution to avoid coupling the pilot to a vision stack."""

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


class MultiRateFastExecutor(nnx.Module):
    """Plan-conditioned residual policy intended to run every control cycle."""

    def __init__(
        self,
        config: MultiRateFastExecutorConfig,
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
        self.image_token_proj = nnx.Linear(
            config.cnn_channels[-1],
            hidden,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.image_query_proj = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        spatial_tokens = _downsampled_image_size(config) ** 2
        self.image_position_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (config.image_views, spatial_tokens, hidden),
                dtype=param_dtype,
            )
        )

        self.ear_input_proj = nnx.Linear(config.action_dim, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_query_proj = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_key_proj = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_value_proj = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_output_proj = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_fusion = nnx.Linear(2 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_position_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (config.ear_horizon, hidden),
                dtype=param_dtype,
            )
        )
        self.age_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (config.max_cache_age + 1, hidden),
                dtype=param_dtype,
            )
        )

        self.state_proj = nnx.Linear(config.state_dim, hidden, rngs=rngs, param_dtype=param_dtype)
        self.observation_query_proj = nnx.Linear(3 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.iar_proj = nnx.Linear(config.iar_dim, hidden, rngs=rngs, param_dtype=param_dtype)
        self.iar_query_proj = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.iar_position_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (config.iar_tokens, hidden),
                dtype=param_dtype,
            )
        )
        self.observation_fusion = nnx.Linear(2 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.plan_gate = nnx.Linear(3 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.fusion_in = nnx.Linear(2 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.fusion_hidden = nnx.Linear(hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.residual_out = nnx.Linear(
            hidden,
            config.action_dim,
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )
        self.refresh_out = nnx.Linear(
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

    def _attention_pool(
        self,
        query: jax.Array,
        tokens: jax.Array,
        query_projection: nnx.Linear,
    ) -> jax.Array:
        config = self.config
        batch_size = query.shape[0]
        heads = config.attention_heads
        head_dim = config.hidden_dim // heads
        projected_query = query_projection(query).reshape((batch_size, heads, head_dim))
        token_heads = tokens.reshape((batch_size, tokens.shape[1], heads, head_dim))
        attention_logits = jnp.einsum("bhd,bthd->bht", projected_query, token_heads)
        attention_logits = attention_logits / jnp.sqrt(jnp.asarray(head_dim, dtype=attention_logits.dtype))
        attention = jax.nn.softmax(attention_logits, axis=-1)
        return jnp.einsum("bht,bthd->bhd", attention, token_heads).reshape((batch_size, config.hidden_dim))

    def _encode_images(self, current_images: jax.Array, observation_query: jax.Array) -> jax.Array:
        config = self.config
        batch_size = current_images.shape[0]
        images = current_images.reshape(
            (
                batch_size * config.image_views,
                config.image_size,
                config.image_size,
                config.image_channels,
            )
        )
        images = self._normalize_images(images)
        for convolution in self.image_convs:
            images = jax.nn.silu(convolution(images))
        spatial_tokens = images.shape[1] * images.shape[2]
        image_tokens = images.reshape((batch_size, config.image_views, spatial_tokens, config.cnn_channels[-1]))
        image_tokens = jax.nn.silu(self.image_token_proj(image_tokens))
        image_tokens = image_tokens + self.image_position_embedding.value[None, :, :spatial_tokens, :]
        image_tokens = image_tokens.reshape((batch_size, config.image_views * spatial_tokens, config.hidden_dim))
        return self._attention_pool(observation_query, image_tokens, self.image_query_proj)

    def _encode_iar(
        self,
        cached_iar: jax.Array | None,
        observation_query: jax.Array,
        *,
        batch_size: int,
    ) -> jax.Array:
        config = self.config
        if cached_iar is None:
            return jnp.zeros((batch_size, config.hidden_dim), dtype=jnp.float32)
        if cached_iar.shape[1] == 0:
            raise ValueError("cached_iar must contain at least one token.")
        if cached_iar.shape[1] > config.iar_tokens:
            raise ValueError(
                f"cached_iar has {cached_iar.shape[1]} tokens; the configured maximum is {config.iar_tokens}."
            )
        iar_tokens = jax.nn.silu(self.iar_proj(cached_iar))
        iar_tokens = iar_tokens + self.iar_position_embedding.value[None, : cached_iar.shape[1], :]
        return self._attention_pool(observation_query, iar_tokens, self.iar_query_proj)

    def _encode_ear(
        self,
        cached_ear: jax.Array,
        cache_age: jax.Array,
        base_action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        config = self.config
        batch_size = cached_ear.shape[0]
        heads = config.attention_heads
        head_dim = config.hidden_dim // heads
        clipped_age = jnp.clip(cache_age, 0, config.max_cache_age)
        age_feature = self.age_embedding.value[clipped_age]

        ear_tokens = jax.nn.silu(self.ear_input_proj(cached_ear))
        ear_tokens = ear_tokens + self.ear_position_embedding.value[None, :, :]
        phase_feature = jax.nn.silu(self.ear_input_proj(base_action)) + age_feature

        query = self.ear_query_proj(phase_feature).reshape((batch_size, heads, head_dim))
        keys = self.ear_key_proj(ear_tokens).reshape((batch_size, config.ear_horizon, heads, head_dim))
        values = self.ear_value_proj(ear_tokens).reshape((batch_size, config.ear_horizon, heads, head_dim))
        attention_logits = jnp.einsum("bhd,bthd->bht", query, keys) / jnp.sqrt(jnp.asarray(head_dim, dtype=query.dtype))
        attention = jax.nn.softmax(attention_logits, axis=-1)
        context = jnp.einsum("bht,bthd->bhd", attention, values).reshape((batch_size, config.hidden_dim))
        context = self.ear_output_proj(context)
        ear_feature = jax.nn.silu(self.ear_fusion(jnp.concatenate([phase_feature, context], axis=-1)))
        return ear_feature, age_feature

    def forward_with_aux(
        self,
        current_images: jax.Array,
        state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        config = self.config
        current_images = jnp.asarray(current_images, dtype=jnp.float32)
        state = jnp.asarray(state, dtype=jnp.float32)
        cached_ear = jnp.asarray(cached_ear, dtype=jnp.float32)
        cache_age = jnp.asarray(cache_age, dtype=jnp.int32)
        batch_size = current_images.shape[0]

        _require_shape(
            current_images,
            (
                None,
                config.image_views,
                config.image_size,
                config.image_size,
                config.image_channels,
            ),
            "current_images",
        )
        _require_shape(state, (batch_size, config.state_dim), "state")
        _require_shape(
            cached_ear,
            (batch_size, config.ear_horizon, config.action_dim),
            "cached_ear",
        )
        _require_shape(cache_age, (batch_size,), "cache_age")

        if cached_iar is not None:
            cached_iar = jnp.asarray(cached_iar, dtype=jnp.float32)
            _require_shape(cached_iar, (batch_size, None, config.iar_dim), "cached_iar")

        base_action = phase_aligned_ear_token(
            cached_ear,
            cache_age,
            max_cache_age=config.max_cache_age,
            coarse_time_stride=config.coarse_time_stride,
        )
        ear_feature, age_feature = self._encode_ear(cached_ear, cache_age, base_action)
        state_feature = jax.nn.silu(self.state_proj(state))
        observation_query = jax.nn.silu(
            self.observation_query_proj(jnp.concatenate([ear_feature, state_feature, age_feature], axis=-1))
        )
        image_feature = self._encode_images(current_images, observation_query)
        iar_feature = self._encode_iar(cached_iar, observation_query, batch_size=batch_size)
        observation_feature = jax.nn.silu(
            self.observation_fusion(jnp.concatenate([image_feature, state_feature], axis=-1))
        )
        plan_gate = jnp.tanh(self.plan_gate(jnp.concatenate([ear_feature, iar_feature, age_feature], axis=-1)))
        interaction = observation_feature * (1.0 + plan_gate)
        hidden = jax.nn.silu(self.fusion_in(jnp.concatenate([observation_feature, interaction], axis=-1)))
        hidden = hidden + jax.nn.silu(self.fusion_hidden(hidden))
        residual = config.residual_scale * jnp.tanh(self.residual_out(hidden))
        # Both heads read an observation-conditioned hidden state.  Cached
        # reasoning can modulate the correction but cannot bypass the latest
        # image/state path with a plan-only additive residual.
        refresh_residual = config.residual_scale * jnp.tanh(self.refresh_out(hidden))
        return base_action + residual, base_action + refresh_residual

    def __call__(
        self,
        current_images: jax.Array,
        state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array | None,
        cache_age: jax.Array,
    ) -> jax.Array:
        predicted_action, _ = self.forward_with_aux(
            current_images,
            state,
            cached_ear,
            cached_iar,
            cache_age,
        )
        return predicted_action


def _huber(error: jax.Array, delta: float) -> jax.Array:
    absolute = jnp.abs(error)
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * jnp.square(quadratic) + delta * (absolute - quadratic)


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = jnp.asarray(mask, dtype=values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def multirate_fast_executor_loss(
    predicted_action: jax.Array,
    target_action: jax.Array,
    *,
    config: FastExecutorLossConfig = DEFAULT_FAST_EXECUTOR_LOSS_CONFIG,
    valid_mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute a gripper-weighted Huber loss and diagnostic scalar metrics."""

    predicted_action = jnp.asarray(predicted_action, dtype=jnp.float32)
    target_action = jnp.asarray(target_action, dtype=jnp.float32)
    if predicted_action.shape != target_action.shape:
        raise ValueError(
            f"predicted_action and target_action shapes differ: {predicted_action.shape} != {target_action.shape}."
        )
    _require_shape(predicted_action, (None, config.action_dim), "predicted_action")

    if valid_mask is None:
        mask = jnp.ones_like(predicted_action)
    else:
        mask = jnp.asarray(valid_mask, dtype=jnp.float32)
        if mask.shape == predicted_action.shape[:1]:
            mask = jnp.broadcast_to(mask[:, None], predicted_action.shape)
        elif mask.shape != predicted_action.shape:
            raise ValueError(f"valid_mask must have shape [B] or [B, action_dim], got {mask.shape}.")

    error = predicted_action - target_action
    element_huber = _huber(error, config.huber_delta)
    dimension_weights = jnp.ones((config.action_dim,), dtype=element_huber.dtype)
    if config.gripper_indices:
        gripper_indices = jnp.asarray(config.gripper_indices, dtype=jnp.int32)
        dimension_weights = dimension_weights.at[gripper_indices].set(config.gripper_weight)
    weighted_mask = mask * dimension_weights[None, :]
    loss = _masked_mean(element_huber, weighted_mask)

    gripper_dimension_mask = jnp.zeros((config.action_dim,), dtype=element_huber.dtype)
    if config.gripper_indices:
        gripper_dimension_mask = gripper_dimension_mask.at[gripper_indices].set(1.0)
    gripper_mask = mask * gripper_dimension_mask[None, :]
    non_gripper_mask = mask * (1.0 - gripper_dimension_mask[None, :])
    metrics = {
        "loss": loss,
        "huber": _masked_mean(element_huber, mask),
        "l1": _masked_mean(jnp.abs(error), mask),
        "gripper_huber": _masked_mean(element_huber, gripper_mask),
        "non_gripper_huber": _masked_mean(element_huber, non_gripper_mask),
    }
    return loss, metrics
