"""Frozen JAX inference for the contextual causal plan compiler sidecar.

This module is the deployment counterpart of
``scripts/train_contextual_causal_plan_compiler_jax.py``.  It loads the
train-only ridge anchor, train-only context PCA, and the selected best residual
checkpoint from ``model_params.npz`` and reproduces that trainer's ``_predict``
function without importing training code.

The public prediction contract keeps the repository's full action shapes::

    EAR [B, 15, 32] + action noise [B, 10, 32]
      + pooled prefix [B, 2048] + IAR [B, 18, 1024] + state [B, 32]
      -> actions [B, 10, 32]

Only the leading seven action dimensions were trained.  The remaining 25
dimensions are deterministically zero-filled, matching the normalized ACoT
dataset convention.  ``predict_batch`` is a pure JAX function and can be used
directly inside a larger ``jax.jit`` computation.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


ACTIVE_ACTION_DIM = 7
FULL_ACTION_DIM = 32
COARSE_HORIZON = 15
ACTION_HORIZON = 10
PREFIX_DIM = 2048
IAR_TOKENS = 18
IAR_DIM = 1024
STATE_DIM = 32
RAW_CONTEXT_DIM = PREFIX_DIM + IAR_TOKENS * IAR_DIM + STATE_DIM


class ResidualParameters(NamedTuple):
    """Best held-out residual parameters saved under ``model/*``."""

    plan_projection: jax.Array
    context_projection: jax.Array
    base_projection: jax.Array
    base_bias: jax.Array
    gamma_projection: jax.Array
    beta_projection: jax.Array
    output_projection: jax.Array


class RidgeParameters(NamedTuple):
    """Frozen train-only EAR/noise ridge anchor saved under ``ridge/*``."""

    feature_mean: jax.Array
    feature_std: jax.Array
    target_mean: jax.Array
    weights: jax.Array


class ContextPCA(NamedTuple):
    """Frozen train-only standardization and PCA saved under ``context_pca/*``."""

    mean: jax.Array
    std: jax.Array
    components: jax.Array


class ContextualPlanCompilerState(NamedTuple):
    """JAX pytree consumed by :func:`predict_batch`."""

    model: ResidualParameters
    ridge: RidgeParameters
    context_pca: ContextPCA
    residual_scale: jax.Array


@dataclasses.dataclass(frozen=True)
class ContextualPlanCompilerMetadata:
    checkpoint_path: str
    residual_scale_source: str
    residual_scale: float
    context_group_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    context_pca_dim: int
    interaction_rank: int
    hidden_dim: int
    active_action_dim: int = ACTIVE_ACTION_DIM
    full_action_dim: int = FULL_ACTION_DIM


@dataclasses.dataclass(frozen=True)
class ContextualPlanCompiler:
    """Loaded sidecar plus a convenient full-shape prediction method."""

    state: ContextualPlanCompilerState
    metadata: ContextualPlanCompilerMetadata

    def predict_batch(
        self,
        ear: jax.Array,
        action_noise: jax.Array,
        prefix_feature: jax.Array,
        iar: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        return predict_batch(
            self.state,
            ear,
            action_noise,
            prefix_feature,
            iar,
            state,
        )

    __call__ = predict_batch


_MODEL_KEYS = (
    "plan_projection",
    "context_projection",
    "base_projection",
    "base_bias",
    "gamma_projection",
    "beta_projection",
    "output_projection",
)
_RIDGE_KEYS = ("feature_mean", "feature_std", "target_mean", "weights")
_PCA_KEYS = ("mean", "std", "components")
_EXPECTED_GROUP_SLICES = np.asarray(
    (
        (0, PREFIX_DIM),
        (PREFIX_DIM, PREFIX_DIM + IAR_TOKENS * IAR_DIM),
        (PREFIX_DIM + IAR_TOKENS * IAR_DIM, RAW_CONTEXT_DIM),
    ),
    dtype=np.int32,
)


def _read_tree(
    archive: Any,
    prefix: str,
    names: tuple[str, ...],
    *,
    device: jax.Device | None,
) -> dict[str, jax.Array]:
    missing = [f"{prefix}/{name}" for name in names if f"{prefix}/{name}" not in archive]
    if missing:
        raise KeyError(f"Contextual compiler checkpoint is missing keys: {missing}.")
    return {
        name: jax.device_put(
            jnp.asarray(np.asarray(archive[f"{prefix}/{name}"]), dtype=jnp.float32),
            device,
        )
        for name in names
    }


def _nested_number(value: Any, *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        current = value
        for name in path:
            if not isinstance(current, dict) or name not in current:
                break
            current = current[name]
        else:
            if isinstance(current, (float, int)) and not isinstance(current, bool):
                return float(current)
    return None


def _resolve_residual_scale(
    checkpoint_path: pathlib.Path,
    archive: Any,
    explicit: float | None,
) -> tuple[float, str]:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("residual_scale must be positive.")
        return float(explicit), "explicit loader argument"
    if "residual_scale" in archive:
        stored = np.asarray(archive["residual_scale"])
        if stored.shape == () and float(stored) > 0:
            return float(stored), "model_params.npz/residual_scale"
    summary_path = checkpoint_path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = _nested_number(
            summary,
            ("architecture", "residual_scale"),
            ("configuration", "residual_scale"),
        )
        if result is not None and result > 0:
            return result, str(summary_path)
    raise ValueError(
        "The trainer's residual_scale is not stored in this NPZ. Strict checkpoint "
        "reproduction requires either the sibling summary.json or an explicit "
        "residual_scale= value from that run."
    )


def _validate_checkpoint(
    model: ResidualParameters,
    ridge: RidgeParameters,
    pca: ContextPCA,
    group_slices: np.ndarray,
) -> None:
    if group_slices.shape != (3, 2) or not np.array_equal(group_slices, _EXPECTED_GROUP_SLICES):
        raise ValueError(
            "Checkpoint context slices do not match prefix[2048]+IAR[18,1024]+state[32]: "
            f"got {group_slices.tolist()}, expected {_EXPECTED_GROUP_SLICES.tolist()}."
        )
    plan_dim = COARSE_HORIZON * ACTIVE_ACTION_DIM
    noise_dim = ACTION_HORIZON * ACTIVE_ACTION_DIM
    target_dim = ACTION_HORIZON * ACTIVE_ACTION_DIM
    if model.plan_projection.ndim != 2 or model.plan_projection.shape[0] != plan_dim:
        raise ValueError(f"model/plan_projection must start with {plan_dim}, got {model.plan_projection.shape}.")
    rank = model.plan_projection.shape[1]
    if model.context_projection.ndim != 2 or model.context_projection.shape[1] != rank:
        raise ValueError("model/context_projection is incompatible with plan_projection.")
    context_dim = model.context_projection.shape[0]
    if pca.mean.shape != (RAW_CONTEXT_DIM,) or pca.std.shape != (RAW_CONTEXT_DIM,):
        raise ValueError(
            f"Context PCA mean/std must have width {RAW_CONTEXT_DIM}; got {pca.mean.shape}/{pca.std.shape}."
        )
    if pca.components.shape != (context_dim, RAW_CONTEXT_DIM):
        raise ValueError(
            "context_pca/components must be [context_code_dim, raw_context_dim]; "
            f"got {pca.components.shape}, expected {(context_dim, RAW_CONTEXT_DIM)}."
        )
    feature_dim = plan_dim + noise_dim
    if ridge.feature_mean.shape != (feature_dim,) or ridge.feature_std.shape != (feature_dim,):
        raise ValueError("Ridge feature statistics do not match flattened EAR plus action noise.")
    if ridge.target_mean.shape != (target_dim,) or ridge.weights.shape != (feature_dim, target_dim):
        raise ValueError("Ridge target tensors do not match the active ten-step action chunk.")
    hidden_dim = model.base_bias.shape[0]
    expected_model_shapes = {
        "base_projection": (feature_dim, hidden_dim),
        "gamma_projection": (rank, hidden_dim),
        "beta_projection": (rank, hidden_dim),
        "output_projection": (hidden_dim, target_dim),
    }
    for name, expected in expected_model_shapes.items():
        actual = getattr(model, name).shape
        if actual != expected:
            raise ValueError(f"model/{name} has shape {actual}; expected {expected}.")
    if np.any(np.asarray(jax.device_get(ridge.feature_std)) <= 0):
        raise ValueError("ridge/feature_std must be strictly positive.")
    if np.any(np.asarray(jax.device_get(pca.std)) <= 0):
        raise ValueError("context_pca/std must be strictly positive.")


def load_contextual_plan_compiler(
    checkpoint: pathlib.Path | str,
    *,
    device: jax.Device | None = None,
    residual_scale: float | None = None,
) -> ContextualPlanCompiler:
    """Load a trainer-produced ``model_params.npz`` for frozen inference.

    ``residual_scale`` normally comes from the sibling ``summary.json`` because
    the current trainer stores it in run metadata rather than in the NPZ.  An
    explicit value is accepted for relocated checkpoints, but must be the
    value from the same training run to preserve exact predictions.
    """

    path = pathlib.Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Contextual compiler checkpoint does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        model_values = _read_tree(archive, "model", _MODEL_KEYS, device=device)
        ridge_values = _read_tree(archive, "ridge", _RIDGE_KEYS, device=device)
        pca_values = _read_tree(archive, "context_pca", _PCA_KEYS, device=device)
        if "group_slices" not in archive:
            raise KeyError("Contextual compiler checkpoint is missing group_slices.")
        group_slices = np.asarray(archive["group_slices"], dtype=np.int32)
        scale, scale_source = _resolve_residual_scale(path, archive, residual_scale)
    model = ResidualParameters(**model_values)
    ridge = RidgeParameters(**ridge_values)
    pca = ContextPCA(**pca_values)
    _validate_checkpoint(model, ridge, pca, group_slices)
    state = ContextualPlanCompilerState(
        model=model,
        ridge=ridge,
        context_pca=pca,
        residual_scale=jax.device_put(jnp.asarray(scale, dtype=jnp.float32), device),
    )
    return ContextualPlanCompiler(
        state=state,
        metadata=ContextualPlanCompilerMetadata(
            checkpoint_path=str(path),
            residual_scale_source=scale_source,
            residual_scale=scale,
            context_group_slices=tuple(
                (int(start), int(end)) for start, end in group_slices.tolist()
            ),  # type: ignore[arg-type]
            context_pca_dim=int(pca.components.shape[0]),
            interaction_rank=int(model.plan_projection.shape[1]),
            hidden_dim=int(model.base_bias.shape[0]),
        ),
    )


def _validate_input_shapes(
    ear: jax.Array,
    action_noise: jax.Array,
    prefix_feature: jax.Array,
    iar: jax.Array,
    state: jax.Array,
) -> None:
    expected = {
        "ear": (COARSE_HORIZON, FULL_ACTION_DIM),
        "action_noise": (ACTION_HORIZON, FULL_ACTION_DIM),
        "prefix_feature": (PREFIX_DIM,),
        "iar": (IAR_TOKENS, IAR_DIM),
        "state": (STATE_DIM,),
    }
    values = {
        "ear": ear,
        "action_noise": action_noise,
        "prefix_feature": prefix_feature,
        "iar": iar,
        "state": state,
    }
    batch_size: int | None = None
    for name, value in values.items():
        if value.ndim != len(expected[name]) + 1 or value.shape[1:] != expected[name]:
            raise ValueError(f"{name} must have shape [B,{','.join(map(str, expected[name]))}], got {value.shape}.")
        if batch_size is None:
            batch_size = value.shape[0]
        elif value.shape[0] != batch_size:
            raise ValueError("All contextual compiler inputs must have the same batch dimension.")


def _predict_active(
    compiler: ContextualPlanCompilerState,
    plan: jax.Array,
    noise: jax.Array,
    context_code: jax.Array,
) -> jax.Array:
    features = jnp.concatenate((plan, noise), axis=-1)
    normalized = (features - compiler.ridge.feature_mean) / compiler.ridge.feature_std
    anchor = normalized @ compiler.ridge.weights + compiler.ridge.target_mean
    plan_factor = jnp.tanh(plan @ compiler.model.plan_projection)
    context_factor = jnp.tanh(context_code @ compiler.model.context_projection)
    interaction = plan_factor * context_factor
    base_hidden = jax.nn.gelu(
        normalized @ compiler.model.base_projection + compiler.model.base_bias,
        approximate=True,
    )
    gamma = jnp.tanh(interaction @ compiler.model.gamma_projection)
    beta = interaction @ compiler.model.beta_projection
    delta_hidden = base_hidden * gamma + beta
    residual = (
        jax.nn.gelu(delta_hidden, approximate=True)
        @ compiler.model.output_projection
    )
    return anchor + compiler.residual_scale * jnp.tanh(residual)


@jax.jit
def predict_batch(
    compiler: ContextualPlanCompilerState,
    ear: jax.Array,
    action_noise: jax.Array,
    prefix_feature: jax.Array,
    iar: jax.Array,
    state: jax.Array,
) -> jax.Array:
    """Predict a full normalized action chunk with the frozen best checkpoint."""

    _validate_input_shapes(ear, action_noise, prefix_feature, iar, state)
    plan = jnp.asarray(ear[..., :ACTIVE_ACTION_DIM], dtype=jnp.float32).reshape(
        (ear.shape[0], COARSE_HORIZON * ACTIVE_ACTION_DIM)
    )
    noise = jnp.asarray(action_noise[..., :ACTIVE_ACTION_DIM], dtype=jnp.float32).reshape(
        (action_noise.shape[0], ACTION_HORIZON * ACTIVE_ACTION_DIM)
    )
    raw_context = jnp.concatenate(
        (
            jnp.asarray(prefix_feature, dtype=jnp.float32),
            jnp.asarray(iar, dtype=jnp.float32).reshape((iar.shape[0], IAR_TOKENS * IAR_DIM)),
            jnp.asarray(state, dtype=jnp.float32),
        ),
        axis=-1,
    )
    standardized_context = (
        raw_context - compiler.context_pca.mean
    ) / compiler.context_pca.std
    context_code = standardized_context @ compiler.context_pca.components.T
    active = _predict_active(compiler, plan, noise, context_code).reshape(
        (ear.shape[0], ACTION_HORIZON, ACTIVE_ACTION_DIM)
    )
    result = jnp.zeros(
        (ear.shape[0], ACTION_HORIZON, FULL_ACTION_DIM),
        dtype=jnp.float32,
    )
    return result.at[..., :ACTIVE_ACTION_DIM].set(active)


__all__ = [
    "ContextualPlanCompiler",
    "ContextualPlanCompilerMetadata",
    "ContextualPlanCompilerState",
    "load_contextual_plan_compiler",
    "predict_batch",
]
