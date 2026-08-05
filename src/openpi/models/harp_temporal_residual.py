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


SCHEMA_VERSION = 3
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
    """Sub-100k temporal residual and no-harm margin head."""

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


def map_ear_to_action_normalization(
    ear: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    """Map the EAR's coarse-action normalization into final-action normalization."""

    ear = jnp.asarray(ear, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32)
    bias = jnp.asarray(bias, dtype=jnp.float32)
    if ear.ndim != 3 or ear.shape[-1] < CONTROL_DIM:
        raise ValueError(f"EAR must have shape [B,T,D>=6], got {ear.shape}.")
    if scale.shape != (CONTROL_DIM,) or bias.shape != (CONTROL_DIM,):
        raise ValueError(
            "EAR normalization affine must contain six scale and bias values; "
            f"got scale={scale.shape}, bias={bias.shape}."
        )
    mapped_control = ear[..., :CONTROL_DIM] * scale + bias
    return jnp.concatenate([mapped_control, ear[..., CONTROL_DIM:]], axis=-1)


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
    ear_action_normalized: jax.Array,
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
    if min(action_nfe1.shape[-1], ear_action_normalized.shape[-1]) < CONTROL_DIM:
        raise ValueError("HARP requires at least six continuous action dimensions.")
    if state.shape[-1] < STATE_FEATURE_DIM:
        raise ValueError(f"HARP requires at least {STATE_FEATURE_DIM} state dimensions.")

    action = action_nfe1[..., :CONTROL_DIM]
    aligned_ear = align_ear_to_action_time(
        ear_action_normalized, action_nfe1.shape[1]
    )[..., :CONTROL_DIM]
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
    ear_normalization_scale: jax.Array
    ear_normalization_bias: jax.Array
    residual_gain: jax.Array
    margin_center: jax.Array
    margin_scale: jax.Array
    conformal_quantile: jax.Array
    conformal_alpha: float
    conformal_group_count: int
    conformal_rank: int
    margin_threshold: jax.Array
    draft_final_time_warp_alpha: float
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
        """Apply a residual only when its grouped-conformal margin LCB is positive."""

        ear_action_normalized = map_ear_to_action_normalization(
            ear,
            self.ear_normalization_scale,
            self.ear_normalization_bias,
        )
        features = build_harp_features(
            action_nfe1,
            ear_action_normalized,
            iar,
            action_noise,
            state,
        )
        normalized = (features - self.feature_mean) / self.feature_std
        head_output = HARPTemporalResidualHead(self.config).apply(
            {"params": self.params}, normalized
        )
        raw_residual = head_output[..., :CONTROL_DIM] * self.target_scale
        candidate_residual = raw_residual * self.residual_gain
        margin_mean = head_output[..., CONTROL_DIM] * self.margin_scale + self.margin_center
        margin_log_variance = jnp.clip(head_output[..., CONTROL_DIM + 1], -8.0, 8.0)
        margin_std = jnp.exp(0.5 * margin_log_variance) * self.margin_scale
        margin_lcb = margin_mean - self.conformal_quantile * margin_std
        margin_gate = margin_lcb > self.margin_threshold
        applied_residual = jnp.where(
            margin_gate[..., None], candidate_residual, jnp.zeros_like(candidate_residual)
        )

        original_control = action_nfe1[..., :CONTROL_DIM]
        candidate_control = (
            jnp.asarray(original_control, dtype=jnp.float32) + candidate_residual
        ).astype(original_control.dtype)
        # The false branch directly selects A1, rather than relying on A1 + 0,
        # so a rejected correction is an exact fallback in the action tensor.
        corrected_control = jnp.where(
            margin_gate[..., None], candidate_control, original_control
        )
        corrected_actions = jnp.concatenate(
            [corrected_control, action_nfe1[..., CONTROL_DIM:]],
            axis=-1,
        )
        return {
            "actions": corrected_actions,
            "raw_residual": raw_residual,
            "candidate_residual": candidate_residual,
            "applied_residual": applied_residual,
            "margin_mean": margin_mean,
            "margin_log_variance": margin_log_variance,
            "margin_std": margin_std,
            "margin_lcb": margin_lcb,
            "margin_gate": margin_gate,
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
        if schema_version == 2:
            raise ValueError(
                "HARP schema 2 has no conformal no-harm margin gate; retrain a schema-3 sidecar."
            )
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

    def finite_scalar(name: str) -> float:
        value = _scalar(data, name, float)
        if not np.isfinite(value):
            raise ValueError(f"Invalid non-finite HARP {name}: {value}.")
        return value

    metadata: Mapping[str, Any] = {}
    if "metadata_json" in data:
        metadata = json.loads(str(np.asarray(data["metadata_json"]).reshape(()).item()))
    ear_normalization_scale = vector("ear_normalization_scale", CONTROL_DIM)
    if np.any(np.asarray(ear_normalization_scale) <= 0.0):
        raise ValueError("HARP ear_normalization_scale must be strictly positive.")
    residual_gain = vector("residual_gain", CONTROL_DIM)
    if np.any((np.asarray(residual_gain) < 0.0) | (np.asarray(residual_gain) > 1.0)):
        raise ValueError("HARP residual_gain must lie in [0, 1].")
    feature_std = vector("feature_std", FEATURE_DIM)
    target_scale = vector("target_scale", CONTROL_DIM)
    if np.any(np.asarray(feature_std) <= 0.0):
        raise ValueError("HARP feature_std must be strictly positive.")
    if np.any(np.asarray(target_scale) <= 0.0):
        raise ValueError("HARP target_scale must be strictly positive.")
    margin_scale = finite_scalar("margin_scale")
    if margin_scale <= 0.0:
        raise ValueError("HARP margin_scale must be strictly positive.")
    conformal_alpha = finite_scalar("conformal_alpha")
    if not 0.0 < conformal_alpha < 1.0:
        raise ValueError("HARP conformal_alpha must lie in (0, 1).")
    conformal_group_count = _scalar(data, "conformal_group_count", int)
    conformal_rank = _scalar(data, "conformal_rank", int)
    if conformal_group_count <= 0 or not 1 <= conformal_rank <= conformal_group_count:
        raise ValueError(
            "Invalid HARP conformal group/rank calibration: "
            f"groups={conformal_group_count}, rank={conformal_rank}."
        )
    draft_final_time_warp_alpha = finite_scalar("draft_final_time_warp_alpha")
    if not 0.0 <= draft_final_time_warp_alpha < 1.0:
        raise ValueError(
            "HARP draft_final_time_warp_alpha must lie in [0, 1); "
            f"got {draft_final_time_warp_alpha}."
        )
    return HARPResidualSidecar(
        config=config,
        params=params,
        feature_mean=vector("feature_mean", FEATURE_DIM),
        feature_std=feature_std,
        target_scale=target_scale,
        ear_normalization_scale=ear_normalization_scale,
        ear_normalization_bias=vector("ear_normalization_bias", CONTROL_DIM),
        residual_gain=residual_gain,
        margin_center=jnp.asarray(finite_scalar("margin_center"), dtype=jnp.float32),
        margin_scale=jnp.asarray(margin_scale, dtype=jnp.float32),
        conformal_quantile=jnp.asarray(
            finite_scalar("conformal_quantile"), dtype=jnp.float32
        ),
        conformal_alpha=conformal_alpha,
        conformal_group_count=conformal_group_count,
        conformal_rank=conformal_rank,
        margin_threshold=jnp.asarray(
            finite_scalar("margin_threshold"), dtype=jnp.float32
        ),
        draft_final_time_warp_alpha=draft_final_time_warp_alpha,
        parameter_count=loaded_parameter_count,
        metadata=metadata,
    )
