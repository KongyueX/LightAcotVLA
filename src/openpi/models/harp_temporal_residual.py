"""Tiny Hierarchy-Aligned Residual Projection (HARP) sidecar.

HARP is deliberately separate from the ACoT-VLA checkpoint.  It observes the
frozen IR endpoint student's one-step action, the physically aligned EAR, IAR
statistics, proprioception, and the exact final-flow noise.  A small temporal
head predicts only the six continuous-action residuals needed to approximate
the same model's two-step endpoint.  Gripper and any padded action dimensions
are copied bit-for-bit from the one-step action.

Loading a sidecar does not activate it.  Policy requests must additionally set
``action_cot_harp_residual=True``; this opt-in contract keeps legacy inference
unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

from flax import linen as nn
from flax import traverse_util
from flax.core import freeze
import jax
import jax.numpy as jnp
import numpy as np


SCHEMA_VERSION = 1
CONTROL_DIM = 6
STATE_FEATURE_DIM = 8
IAR_SUMMARY_DIM = 1024
FEATURE_DIM = CONTROL_DIM * 6 + STATE_FEATURE_DIM + IAR_SUMMARY_DIM
OUTPUT_DIM = CONTROL_DIM + 2
PARAM_PREFIX = "param/"


@dataclasses.dataclass(frozen=True)
class HARPHeadConfig:
    input_dim: int = FEATURE_DIM
    hidden_dim: int = 32
    temporal_layers: int = 2
    kernel_size: int = 3
    control_dim: int = CONTROL_DIM


class HARPTemporalResidualHead(nn.Module):
    """Sub-10k temporal residual, heteroscedasticity, and EAR-reliability head."""

    config: HARPHeadConfig

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"HARP feature width {features.shape[-1]} != {self.config.input_dim}."
            )
        hidden = nn.Dense(self.config.hidden_dim, name="input_projection")(features)
        hidden = nn.gelu(hidden)
        for layer in range(self.config.temporal_layers):
            residual = hidden
            hidden = nn.LayerNorm(name=f"temporal_norm_{layer}")(hidden)
            hidden = nn.Conv(
                features=self.config.hidden_dim,
                kernel_size=(self.config.kernel_size,),
                padding="SAME",
                name=f"temporal_conv_{layer}",
            )(hidden)
            hidden = nn.gelu(hidden)
            hidden = hidden + residual
        return nn.Dense(self.config.control_dim + 2, name="output_projection")(hidden)


def align_ear_to_action_time(ear: jax.Array, action_horizon: int) -> jax.Array:
    """Align EAR samples at ticks 0,2,... to final-action ticks 0,1,... ."""

    ear = jnp.asarray(ear, dtype=jnp.float32)
    if ear.ndim != 3:
        raise ValueError(f"EAR must have shape [B,T,D], got {ear.shape}.")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")
    coarse_horizon = ear.shape[1]
    if coarse_horizon <= 0:
        raise ValueError("EAR horizon must be positive.")
    last_position = 0.5 * float(action_horizon - 1)
    if last_position > float(coarse_horizon - 1):
        raise ValueError(
            "EAR horizon is too short for two-action-ticks-per-EAR-sample alignment."
        )
    positions = 0.5 * jnp.arange(action_horizon, dtype=jnp.float32)
    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, coarse_horizon - 1)
    weight = (positions - lower.astype(jnp.float32))[None, :, None]
    return (1.0 - weight) * jnp.take(ear, lower, axis=1) + weight * jnp.take(
        ear, upper, axis=1
    )


def _temporal_delta(values: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [jnp.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]],
        axis=1,
    )


def _iar_summary(iar: jax.Array) -> jax.Array:
    iar = jnp.asarray(iar, dtype=jnp.float32)
    if iar.ndim != 3:
        raise ValueError(f"IAR must have shape [B,N,D], got {iar.shape}.")
    if iar.shape[-1] != IAR_SUMMARY_DIM:
        raise ValueError(
            f"HARP expects {IAR_SUMMARY_DIM}-wide IAR tokens, got {iar.shape[-1]}."
        )
    # Preserve the semantic channel basis.  Four global moments erase which
    # latent features are active and cannot condition the midpoint velocity.
    return jnp.mean(iar, axis=1)


def build_harp_features(
    action_nfe1: jax.Array,
    ear: jax.Array,
    iar: jax.Array,
    action_noise: jax.Array,
    state: jax.Array,
) -> jax.Array:
    """Build the causal per-timestep feature contract used by train and serve."""

    action_nfe1 = jnp.asarray(action_nfe1, dtype=jnp.float32)
    action_noise = jnp.asarray(action_noise, dtype=jnp.float32)
    state = jnp.asarray(state, dtype=jnp.float32)
    if action_nfe1.ndim != 3:
        raise ValueError(f"action_nfe1 must have shape [B,T,D], got {action_nfe1.shape}.")
    if action_noise.shape != action_nfe1.shape:
        raise ValueError(
            f"action_noise shape {action_noise.shape} != action_nfe1 {action_nfe1.shape}."
        )
    if state.ndim != 2 or state.shape[0] != action_nfe1.shape[0]:
        raise ValueError(f"state must have shape [B,D], got {state.shape}.")
    if min(action_nfe1.shape[-1], ear.shape[-1]) < CONTROL_DIM:
        raise ValueError("HARP requires at least six continuous action dimensions.")
    if state.shape[-1] < STATE_FEATURE_DIM:
        raise ValueError(f"HARP requires at least {STATE_FEATURE_DIM} state dimensions.")

    action = action_nfe1[..., :CONTROL_DIM]
    aligned_ear = align_ear_to_action_time(ear, action_nfe1.shape[1])[..., :CONTROL_DIM]
    noise = action_noise[..., :CONTROL_DIM]
    state_features = jnp.broadcast_to(
        state[:, None, :STATE_FEATURE_DIM],
        (*action.shape[:2], STATE_FEATURE_DIM),
    )
    iar_features = jnp.broadcast_to(
        _iar_summary(iar)[:, None, :],
        (*action.shape[:2], IAR_SUMMARY_DIM),
    )
    features = jnp.concatenate(
        [
            action,
            aligned_ear,
            action - aligned_ear,
            _temporal_delta(action),
            _temporal_delta(aligned_ear),
            noise,
            state_features,
            iar_features,
        ],
        axis=-1,
    )
    if features.shape[-1] != FEATURE_DIM:
        raise AssertionError(f"Internal HARP feature width {features.shape[-1]} != {FEATURE_DIM}.")
    return features


@dataclasses.dataclass(frozen=True)
class HARPResidualSidecar:
    config: HARPHeadConfig
    params: Mapping[str, Any]
    feature_mean: jax.Array
    feature_std: jax.Array
    target_scale: jax.Array
    residual_abs_limit: jax.Array
    residual_norm_limit: jax.Array
    confidence_temperature: jax.Array
    confidence_bias: jax.Array
    trust_scale: jax.Array
    parameter_count: int
    metadata: Mapping[str, Any]

    def predict_and_correct(
        self,
        action_nfe1: jax.Array,
        ear: jax.Array,
        iar: jax.Array,
        action_noise: jax.Array,
        state: jax.Array,
    ) -> dict[str, jax.Array]:
        """Apply confidence shrink and a calibrated trust projection on device."""

        features = build_harp_features(action_nfe1, ear, iar, action_noise, state)
        normalized = (features - self.feature_mean) / self.feature_std
        head_output = HARPTemporalResidualHead(self.config).apply(
            {"params": self.params}, normalized
        )
        raw_residual = head_output[..., :CONTROL_DIM] * self.target_scale
        log_variance = jnp.clip(head_output[..., CONTROL_DIM], -8.0, 8.0)
        ear_reliability = jax.nn.sigmoid(head_output[..., CONTROL_DIM + 1])

        confidence = jax.nn.sigmoid(
            (self.confidence_bias - log_variance)
            / jnp.maximum(self.confidence_temperature, 1e-3)
        )
        # Reliability 0.5 is neutral; only evidence of unreliable EAR shrinks
        # the correction further.  This avoids an unconditional 0.5 penalty at
        # the head's zero-logit initialization.
        reliability_gate = jnp.clip(2.0 * ear_reliability, 0.0, 1.0)
        shrink = confidence * reliability_gate

        coordinate_limit = jnp.maximum(
            self.residual_abs_limit * self.trust_scale,
            1e-6,
        )
        projected = jnp.clip(raw_residual, -coordinate_limit, coordinate_limit)
        residual_norm = jnp.linalg.norm(projected, axis=-1, keepdims=True)
        norm_limit = jnp.maximum(self.residual_norm_limit * self.trust_scale, 1e-6)
        projected = projected * jnp.minimum(1.0, norm_limit / (residual_norm + 1e-6))
        applied_residual = projected * shrink[..., None]

        corrected_control = jnp.asarray(action_nfe1[..., :CONTROL_DIM], dtype=jnp.float32) + applied_residual
        corrected_actions = jnp.concatenate(
            [corrected_control.astype(action_nfe1.dtype), action_nfe1[..., CONTROL_DIM:]],
            axis=-1,
        )
        return {
            "actions": corrected_actions,
            "raw_residual": raw_residual,
            "applied_residual": applied_residual,
            "log_variance": log_variance,
            "confidence": confidence,
            "ear_reliability": ear_reliability,
        }


def parameter_count(params: Mapping[str, Any]) -> int:
    return int(
        sum(np.prod(np.asarray(value).shape) for value in traverse_util.flatten_dict(params).values())
    )


def _scalar(data: Mapping[str, np.ndarray], name: str, cast: type) -> Any:
    if name not in data:
        raise KeyError(f"HARP sidecar is missing {name!r}.")
    return cast(np.asarray(data[name]).reshape(()).item())


def load_harp_residual_sidecar(path: pathlib.Path | str) -> HARPResidualSidecar:
    """Load and strictly validate a standalone HARP NPZ sidecar."""

    resolved = pathlib.Path(path)
    if resolved.is_dir():
        resolved = resolved / "model_params.npz"
    if not resolved.exists():
        raise FileNotFoundError(f"HARP sidecar not found: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    schema_version = _scalar(data, "schema_version", int)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported HARP schema {schema_version}; expected {SCHEMA_VERSION}."
        )
    config = HARPHeadConfig(
        input_dim=_scalar(data, "input_dim", int),
        hidden_dim=_scalar(data, "hidden_dim", int),
        temporal_layers=_scalar(data, "temporal_layers", int),
        kernel_size=_scalar(data, "kernel_size", int),
        control_dim=_scalar(data, "control_dim", int),
    )
    if config.input_dim != FEATURE_DIM or config.control_dim != CONTROL_DIM:
        raise ValueError(
            "HARP sidecar feature/control contract differs from this runtime: "
            f"input={config.input_dim}, control={config.control_dim}."
        )
    flat_params = {
        tuple(name.removeprefix(PARAM_PREFIX).split("/")): jnp.asarray(value)
        for name, value in data.items()
        if name.startswith(PARAM_PREFIX)
    }
    if not flat_params:
        raise ValueError("HARP sidecar contains no model parameters.")
    params = freeze(traverse_util.unflatten_dict(flat_params))
    loaded_parameter_count = parameter_count(params)
    saved_parameter_count = _scalar(data, "parameter_count", int)
    if loaded_parameter_count != saved_parameter_count:
        raise ValueError(
            f"HARP parameter count mismatch: loaded={loaded_parameter_count}, saved={saved_parameter_count}."
        )
    if loaded_parameter_count >= 100_000:
        raise ValueError(
            f"HARP pilot must remain below 100k parameters, got {loaded_parameter_count}."
        )

    def vector(name: str, width: int) -> jax.Array:
        if name not in data:
            raise KeyError(f"HARP sidecar is missing {name!r}.")
        value = np.asarray(data[name], dtype=np.float32)
        if value.shape != (width,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid HARP {name}: shape={value.shape}.")
        return jnp.asarray(value)

    metadata: Mapping[str, Any] = {}
    if "metadata_json" in data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(()).item()))
    return HARPResidualSidecar(
        config=config,
        params=params,
        feature_mean=vector("feature_mean", FEATURE_DIM),
        feature_std=jnp.maximum(vector("feature_std", FEATURE_DIM), 1e-6),
        target_scale=jnp.maximum(vector("target_scale", CONTROL_DIM), 1e-6),
        residual_abs_limit=jnp.maximum(vector("residual_abs_limit", CONTROL_DIM), 1e-6),
        residual_norm_limit=jnp.asarray(
            max(_scalar(data, "residual_norm_limit", float), 1e-6),
            dtype=jnp.float32,
        ),
        confidence_temperature=jnp.asarray(
            max(_scalar(data, "confidence_temperature", float), 1e-3),
            dtype=jnp.float32,
        ),
        confidence_bias=jnp.asarray(_scalar(data, "confidence_bias", float), dtype=jnp.float32),
        trust_scale=jnp.asarray(max(_scalar(data, "trust_scale", float), 1e-3), dtype=jnp.float32),
        parameter_count=loaded_parameter_count,
        metadata=metadata,
    )
