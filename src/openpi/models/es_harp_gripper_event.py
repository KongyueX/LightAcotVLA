"""Endpoint-Sign HARP sidecar for sparse gripper flip events.

This student is independent of the continuous HARP residual sidecar.  It may
only replace action dimension seven and directly copies every other A1 value.
Loading the sidecar is inert; serving requests must opt in explicitly.
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
REQUIRED_DRAFT_FINAL_TIME_WARP_ALPHA = 0.05
CONTROL_DIM = 6
GRIPPER_INDEX = 6
STATE_FEATURE_DIM = 8
IAR_SUMMARY_DIM = 1024
GRIPPER_FEATURE_DIM = 6
CONTINUOUS_CONTEXT_DIM = CONTROL_DIM * 3
FEATURE_DIM = (
    GRIPPER_FEATURE_DIM
    + CONTINUOUS_CONTEXT_DIM
    + STATE_FEATURE_DIM
    + IAR_SUMMARY_DIM
)
OUTPUT_DIM = 2
PARAM_PREFIX = "param/"


@dataclasses.dataclass(frozen=True)
class GripperEventHeadConfig:
    input_dim: int = FEATURE_DIM
    hidden_dim: int = 32
    temporal_layers: int = 1
    kernel_size: int = 3


class GripperEventHead(nn.Module):
    """Small temporal head producing A2-sign and A1-to-A2-flip logits."""

    config: GripperEventHeadConfig

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"ES-HARP feature width {features.shape[-1]} != {self.config.input_dim}."
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
            hidden = nn.gelu(hidden) + residual
        return nn.Dense(OUTPUT_DIM, name="output_projection")(hidden)


def _align_ear_to_action_time(ear: jax.Array, action_horizon: int) -> jax.Array:
    ear = jnp.asarray(ear, dtype=jnp.float32)
    if ear.ndim != 3:
        raise ValueError(f"EAR must have shape [B,T,D], got {ear.shape}.")
    if action_horizon <= 0 or ear.shape[1] <= 0:
        raise ValueError("EAR and action horizons must be positive.")
    last_position = 0.5 * float(action_horizon - 1)
    if last_position > float(ear.shape[1] - 1):
        raise ValueError("EAR horizon is too short for two-action-ticks alignment.")
    positions = 0.5 * jnp.arange(action_horizon, dtype=jnp.float32)
    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, ear.shape[1] - 1)
    weight = (positions - lower.astype(jnp.float32))[None, :, None]
    return (1.0 - weight) * jnp.take(ear, lower, axis=1) + weight * jnp.take(
        ear, upper, axis=1
    )


def _temporal_delta(values: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [jnp.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]],
        axis=1,
    )


def build_gripper_event_features(
    action_nfe1: jax.Array,
    raw_ear: jax.Array,
    iar: jax.Array,
    action_noise: jax.Array,
    state: jax.Array,
    ear_control_scale: jax.Array,
    ear_control_bias: jax.Array,
) -> jax.Array:
    """Build the shared train/serve ES-HARP feature contract."""

    action_nfe1 = jnp.asarray(action_nfe1, dtype=jnp.float32)
    raw_ear = jnp.asarray(raw_ear, dtype=jnp.float32)
    iar = jnp.asarray(iar, dtype=jnp.float32)
    action_noise = jnp.asarray(action_noise, dtype=jnp.float32)
    state = jnp.asarray(state, dtype=jnp.float32)
    ear_control_scale = jnp.asarray(ear_control_scale, dtype=jnp.float32)
    ear_control_bias = jnp.asarray(ear_control_bias, dtype=jnp.float32)
    if action_nfe1.ndim != 3 or action_nfe1.shape[-1] <= GRIPPER_INDEX:
        raise ValueError(f"A1 must have shape [B,T,D>=7], got {action_nfe1.shape}.")
    if raw_ear.ndim != 3 or raw_ear.shape[-1] <= GRIPPER_INDEX:
        raise ValueError(f"Raw EAR must have shape [B,T,D>=7], got {raw_ear.shape}.")
    if action_noise.shape != action_nfe1.shape:
        raise ValueError(
            f"action_noise shape {action_noise.shape} != A1 shape {action_nfe1.shape}."
        )
    if state.ndim != 2 or state.shape[0] != action_nfe1.shape[0]:
        raise ValueError(f"state must have shape [B,D], got {state.shape}.")
    if state.shape[-1] < STATE_FEATURE_DIM:
        raise ValueError(f"ES-HARP requires at least {STATE_FEATURE_DIM} state dimensions.")
    if iar.ndim != 3 or iar.shape[0] != action_nfe1.shape[0]:
        raise ValueError(f"IAR must have shape [B,N,D], got {iar.shape}.")
    if iar.shape[-1] != IAR_SUMMARY_DIM:
        raise ValueError(
            f"ES-HARP expects {IAR_SUMMARY_DIM}-wide IAR tokens, got {iar.shape[-1]}."
        )
    if ear_control_scale.shape != (CONTROL_DIM,) or ear_control_bias.shape != (CONTROL_DIM,):
        raise ValueError("ES-HARP EAR control affine must contain six scale and bias values.")

    aligned_raw_ear = _align_ear_to_action_time(raw_ear, action_nfe1.shape[1])
    action_gripper = action_nfe1[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]
    ear_gripper = aligned_raw_ear[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]
    noise_gripper = action_noise[..., GRIPPER_INDEX : GRIPPER_INDEX + 1]
    action_control = action_nfe1[..., :CONTROL_DIM]
    ear_control = (
        aligned_raw_ear[..., :CONTROL_DIM] * ear_control_scale + ear_control_bias
    )
    state_features = jnp.broadcast_to(
        state[:, None, :STATE_FEATURE_DIM],
        (*action_nfe1.shape[:2], STATE_FEATURE_DIM),
    )
    iar_mean = jnp.mean(iar, axis=1)
    iar_features = jnp.broadcast_to(
        iar_mean[:, None, :],
        (*action_nfe1.shape[:2], IAR_SUMMARY_DIM),
    )
    features = jnp.concatenate(
        [
            action_gripper,
            ear_gripper,
            action_gripper - ear_gripper,
            _temporal_delta(action_gripper),
            _temporal_delta(ear_gripper),
            noise_gripper,
            action_control,
            ear_control,
            action_control - ear_control,
            state_features,
            iar_features,
        ],
        axis=-1,
    )
    if features.shape[-1] != FEATURE_DIM:
        raise AssertionError(
            f"Internal ES-HARP feature width {features.shape[-1]} != {FEATURE_DIM}."
        )
    return features


@dataclasses.dataclass(frozen=True)
class GripperEventSidecar:
    config: GripperEventHeadConfig
    params: Mapping[str, Any]
    feature_mean: jax.Array
    feature_std: jax.Array
    ear_control_scale: jax.Array
    ear_control_bias: jax.Array
    event_threshold: jax.Array
    positive_gripper_value: jax.Array
    negative_gripper_value: jax.Array
    draft_final_time_warp_alpha: float
    parameter_count: int
    metadata: Mapping[str, Any]

    def predict_and_correct(
        self,
        action_nfe1: jax.Array,
        raw_ear: jax.Array,
        iar: jax.Array,
        action_noise: jax.Array,
        state: jax.Array,
    ) -> dict[str, jax.Array]:
        """Replace only the gripper value when a calibrated sign flip fires."""

        features = build_gripper_event_features(
            action_nfe1,
            raw_ear,
            iar,
            action_noise,
            state,
            self.ear_control_scale,
            self.ear_control_bias,
        )
        normalized = (features - self.feature_mean) / self.feature_std
        logits = GripperEventHead(self.config).apply({"params": self.params}, normalized)
        sign_probability = jax.nn.sigmoid(logits[..., 0])
        flip_probability = jax.nn.sigmoid(logits[..., 1])
        predicted_positive = sign_probability >= 0.5
        original_gripper = action_nfe1[..., GRIPPER_INDEX]
        original_positive = original_gripper >= 0.0
        flip_consistent = predicted_positive != original_positive
        event_gate = (flip_probability >= self.event_threshold) & flip_consistent
        event_value = jnp.where(
            predicted_positive,
            self.positive_gripper_value,
            self.negative_gripper_value,
        ).astype(original_gripper.dtype)
        corrected_gripper = jnp.where(event_gate, event_value, original_gripper)
        corrected_actions = jnp.concatenate(
            [
                action_nfe1[..., :GRIPPER_INDEX],
                corrected_gripper[..., None],
                action_nfe1[..., GRIPPER_INDEX + 1 :],
            ],
            axis=-1,
        )
        return {
            "actions": corrected_actions,
            "sign_probability": sign_probability,
            "flip_probability": flip_probability,
            "predicted_positive": predicted_positive,
            "flip_consistent": flip_consistent,
            "event_gate": event_gate,
            "original_gripper": original_gripper,
            "corrected_gripper": corrected_gripper,
        }


def parameter_count(params: Mapping[str, Any]) -> int:
    return int(
        sum(
            np.prod(np.asarray(value).shape)
            for value in traverse_util.flatten_dict(params).values()
        )
    )


def _scalar(data: Mapping[str, np.ndarray], name: str, cast: type) -> Any:
    if name not in data:
        raise KeyError(f"ES-HARP sidecar is missing {name!r}.")
    return cast(np.asarray(data[name]).reshape(()).item())


def load_gripper_event_sidecar(path: pathlib.Path | str) -> GripperEventSidecar:
    """Load and strictly validate an ES-HARP sidecar."""

    resolved = pathlib.Path(path)
    if resolved.is_dir():
        resolved = resolved / "model_params.npz"
    if not resolved.exists():
        raise FileNotFoundError(f"ES-HARP sidecar not found: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    schema = _scalar(data, "schema_version", int)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"Unsupported ES-HARP schema {schema}; expected {SCHEMA_VERSION}.")

    config = GripperEventHeadConfig(
        input_dim=_scalar(data, "input_dim", int),
        hidden_dim=_scalar(data, "hidden_dim", int),
        temporal_layers=_scalar(data, "temporal_layers", int),
        kernel_size=_scalar(data, "kernel_size", int),
    )
    if config.input_dim != FEATURE_DIM:
        raise ValueError(
            f"ES-HARP feature contract differs from runtime: {config.input_dim} != {FEATURE_DIM}."
        )
    if config.hidden_dim <= 0 or config.temporal_layers <= 0 or config.kernel_size <= 0:
        raise ValueError("Invalid non-positive ES-HARP model configuration.")

    flat_params: dict[tuple[str, ...], jax.Array] = {}
    for name, value in data.items():
        if not name.startswith(PARAM_PREFIX):
            continue
        if not np.all(np.isfinite(value)):
            raise ValueError(f"ES-HARP parameter {name!r} contains non-finite values.")
        flat_params[tuple(name.removeprefix(PARAM_PREFIX).split("/"))] = jnp.asarray(value)
    if not flat_params:
        raise ValueError("ES-HARP sidecar contains no model parameters.")
    params = freeze(traverse_util.unflatten_dict(flat_params))
    loaded_count = parameter_count(params)
    saved_count = _scalar(data, "parameter_count", int)
    if loaded_count != saved_count:
        raise ValueError(
            f"ES-HARP parameter count mismatch: loaded={loaded_count}, saved={saved_count}."
        )
    if loaded_count >= 50_000:
        raise ValueError(f"ES-HARP must remain below 50k parameters, got {loaded_count}.")

    def vector(name: str, width: int) -> jax.Array:
        if name not in data:
            raise KeyError(f"ES-HARP sidecar is missing {name!r}.")
        value = np.asarray(data[name], dtype=np.float32)
        if value.shape != (width,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid ES-HARP {name}: shape={value.shape}.")
        return jnp.asarray(value)

    def finite_scalar(name: str) -> float:
        value = _scalar(data, name, float)
        if not np.isfinite(value):
            raise ValueError(f"Invalid non-finite ES-HARP {name}: {value}.")
        return value

    feature_std = vector("feature_std", FEATURE_DIM)
    if np.any(np.asarray(feature_std) <= 0.0):
        raise ValueError("ES-HARP feature_std must be strictly positive.")
    ear_control_scale = vector("ear_control_scale", CONTROL_DIM)
    if np.any(np.asarray(ear_control_scale) <= 0.0):
        raise ValueError("ES-HARP ear_control_scale must be strictly positive.")
    event_threshold = finite_scalar("event_threshold")
    if not 0.0 < event_threshold < 1.0:
        raise ValueError("ES-HARP event_threshold must lie in (0, 1).")
    positive_value = finite_scalar("positive_gripper_value")
    negative_value = finite_scalar("negative_gripper_value")
    if positive_value <= 0.0 or negative_value >= 0.0:
        raise ValueError(
            "ES-HARP gripper prototypes must have positive/negative signs respectively."
        )
    alpha = finite_scalar("draft_final_time_warp_alpha")
    if abs(alpha - REQUIRED_DRAFT_FINAL_TIME_WARP_ALPHA) > 1e-7:
        raise ValueError(
            "ES-HARP is strictly versioned for draft final_time_warp_alpha=0.05; "
            f"got {alpha}."
        )
    metadata: Mapping[str, Any] = {}
    if "metadata_json" in data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(()).item()))
        if not isinstance(metadata, dict):
            raise ValueError("ES-HARP metadata_json must decode to an object.")
    return GripperEventSidecar(
        config=config,
        params=params,
        feature_mean=vector("feature_mean", FEATURE_DIM),
        feature_std=feature_std,
        ear_control_scale=ear_control_scale,
        ear_control_bias=vector("ear_control_bias", CONTROL_DIM),
        event_threshold=jnp.asarray(event_threshold, dtype=jnp.float32),
        positive_gripper_value=jnp.asarray(positive_value, dtype=jnp.float32),
        negative_gripper_value=jnp.asarray(negative_value, dtype=jnp.float32),
        draft_final_time_warp_alpha=alpha,
        parameter_count=loaded_count,
        metadata=metadata,
    )
