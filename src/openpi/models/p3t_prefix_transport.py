"""Plan-conditioned predictive transport for an ACoT-VLA prefix KV cache.

The module is intentionally shape-static for the current LIBERO ACoT checkpoint:

* 18 Gemma layers, one KV head with width 256;
* 968 prefix slots;
* two valid 16x16 image-token grids (512 slots total);
* one masked 256-token image segment, copied without modification; and
* 200 language slots, corrected by a current-observation-conditioned low-rank
  residual.

The adapter consumes only inexpensive 64x64 anchor/current images and control
context.  It does not call SigLIP or Gemma.  Cached visual keys are first moved
back into content-key space with inverse RoPE, spatially warped, corrected, and
then rotated at their target positions.  Values are warped directly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
from typing import NamedTuple

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
KVCache = tuple[Array, Array]
_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_CONFIG_NAME = "config.json"
_CHECKPOINT_PARAMS_NAME = "params.npz"


@dataclasses.dataclass(frozen=True)
class P3TPrefixTransportConfig:
    """Compile-time dimensions for the fixed-shape P3T adapter."""

    num_layers: int = 18
    num_layer_groups: int = 6
    prefix_tokens: int = 968
    visual_tokens: int = 512
    dummy_tokens: int = 256
    language_tokens: int = 200
    num_views: int = 2
    image_size: int = 64
    token_grid_size: int = 16
    state_dim: int = 32
    action_dim: int = 32
    executed_action_horizon: int = 10
    ear_horizon: int = 15
    kv_dim: int = 256
    rank: int = 4
    hidden_dim: int = 64
    max_flow_tokens: float = 2.0
    rope_max_wavelength: float = 10_000.0

    def __post_init__(self) -> None:
        if self.num_layers != 18 or self.num_layer_groups != 6:
            raise ValueError("P3T currently requires 18 layers split into 6 groups.")
        if self.rank != 4:
            raise ValueError("P3T currently requires rank=4.")
        if self.num_layers % self.num_layer_groups:
            raise ValueError("num_layers must be divisible by num_layer_groups.")
        if self.num_views * self.token_grid_size**2 != self.visual_tokens:
            raise ValueError("visual_tokens must equal num_views * token_grid_size**2.")
        if self.visual_tokens + self.dummy_tokens + self.language_tokens != self.prefix_tokens:
            raise ValueError("visual, dummy, and language segments must cover the prefix.")
        if self.image_size % self.token_grid_size:
            raise ValueError("image_size must be divisible by token_grid_size.")
        if self.kv_dim % 2:
            raise ValueError("RoPE requires an even kv_dim.")


class P3TPrefixTransportOutput(NamedTuple):
    """Transported cache and predicted probability that a full refresh is needed."""

    kv_cache: KVCache
    risk: Array  # [B], in [0, 1]


def _zero_linear(
    in_features: int,
    out_features: int,
    *,
    rngs: nnx.Rngs,
    param_dtype: jnp.dtype,
) -> nnx.Linear:
    """Linear head whose initial output is exactly zero."""

    return nnx.Linear(
        in_features,
        out_features,
        rngs=rngs,
        param_dtype=param_dtype,
        kernel_init=jax.nn.initializers.zeros,
        bias_init=jax.nn.initializers.zeros,
    )


def _masked_mean(values: Array, mask: Array, *, axis: int) -> Array:
    """Mean over a fixed-length sequence without introducing a dynamic shape."""

    weights = mask.astype(values.dtype)[..., None]
    denominator = jnp.maximum(jnp.sum(weights, axis=axis), 1.0)
    return jnp.sum(values * weights, axis=axis) / denominator


def _apply_rope(x: Array, positions: Array, *, max_wavelength: float) -> Array:
    """Apply Gemma-compatible RoPE to ``x[B,T,H,D]`` at ``positions[B,T]``."""

    half_dim = x.shape[-1] // 2
    frequency_exponents = (2.0 / x.shape[-1]) * jnp.arange(half_dim, dtype=jnp.float32)
    timescale = jnp.asarray(max_wavelength, dtype=jnp.float32) ** frequency_exponents
    radians = positions.astype(jnp.float32)[..., None, None] / timescale[None, None, None, :]
    sine, cosine = jnp.sin(radians), jnp.cos(radians)
    x_first, x_second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate(
        [x_first * cosine - x_second * sine, x_second * cosine + x_first * sine],
        axis=-1,
    )


def _local_average_3x3(tokens: Array) -> Array:
    """Static 3x3 local mixing on ``[B,V,H,W,C]`` without a learned convolution."""

    height, width = tokens.shape[2:4]
    padded = jnp.pad(tokens, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    neighbours = [
        padded[:, :, dy : dy + height, dx : dx + width, :]
        for dy in range(3)
        for dx in range(3)
    ]
    return sum(neighbours) / 9.0


def _bilinear_warp_token_grid(values: Array, flow: Array) -> Array:
    """Warp ``values[B,L,V,H,W,D]`` by ``flow[B,V,H,W,(dy,dx)]``.

    Sampling coordinates are ``target_grid + flow`` and are clamped to the
    source grid.  Four-neighbour gathers retain a fully static output shape.
    """

    batch, layers, views, height, width, dim = values.shape
    grid_y, grid_x = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32),
        jnp.arange(width, dtype=jnp.float32),
        indexing="ij",
    )
    source_y = jnp.clip(grid_y[None, None, :, :] + flow[..., 0], 0.0, height - 1.0)
    source_x = jnp.clip(grid_x[None, None, :, :] + flow[..., 1], 0.0, width - 1.0)

    y0 = jnp.floor(source_y).astype(jnp.int32)
    x0 = jnp.floor(source_x).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, height - 1)
    x1 = jnp.minimum(x0 + 1, width - 1)

    wy = source_y - y0.astype(source_y.dtype)
    wx = source_x - x0.astype(source_x.dtype)
    flat_values = values.reshape(batch, layers, views, height * width, dim)

    def gather(y_index: Array, x_index: Array) -> Array:
        flat_index = (y_index * width + x_index).reshape(batch, views, height * width)
        # Non-gather axes broadcast across layers and the KV dimension.
        gather_index = flat_index[:, None, :, :, None]
        return jnp.take_along_axis(flat_values, gather_index, axis=3)

    value_00 = gather(y0, x0)
    value_01 = gather(y0, x1)
    value_10 = gather(y1, x0)
    value_11 = gather(y1, x1)

    wx = wx.reshape(batch, 1, views, height * width, 1)
    wy = wy.reshape(batch, 1, views, height * width, 1)
    warped = (
        (1.0 - wy) * (1.0 - wx) * value_00
        + (1.0 - wy) * wx * value_01
        + wy * (1.0 - wx) * value_10
        + wy * wx * value_11
    )
    return warped.reshape(batch, layers, views, height, width, dim)


class P3TPrefixTransport(nnx.Module):
    """Fixed-shape, plan-conditioned prefix KV transport adapter.

    Inputs:
        anchor_images: normalized float images ``[B,2,64,64,3]``.
        current_images: normalized float images ``[B,2,64,64,3]``.
        state_delta: current minus anchor padded robot state ``[B,32]``.
        executed_actions: fixed action history ``[B,10,32]``.
        executed_action_mask: valid history entries ``[B,10]``.
        anchor_ear: anchor explicit action reason ``[B,15,32]``.
        anchor_kv: Gemma cache pair, each ``[18,B,968,1,256]``.

    Returns:
        A :class:`P3TPrefixTransportOutput` containing cache tensors with the
        same shape/dtype as ``anchor_kv`` and one refresh risk per batch item.
    """

    def __init__(
        self,
        config: P3TPrefixTransportConfig = P3TPrefixTransportConfig(),
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        hidden = config.hidden_dim

        # A 4x4 average pool maps 64x64 pixels to the Gemma 16x16 visual-token
        # grid.  The per-patch input contains anchor, current, and RGB delta.
        self.image_in = nnx.Linear(9, hidden, rngs=rngs, param_dtype=param_dtype)
        self.image_mix = nnx.Linear(2 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)

        self.state_proj = nnx.Linear(config.state_dim, hidden, rngs=rngs, param_dtype=param_dtype)
        self.action_proj = nnx.Linear(config.action_dim, hidden, rngs=rngs, param_dtype=param_dtype)
        self.ear_proj = nnx.Linear(config.action_dim, hidden, rngs=rngs, param_dtype=param_dtype)
        self.plan_fusion = nnx.Linear(3 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.context_fusion = nnx.Linear(2 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.film = nnx.Linear(hidden, 2 * hidden, rngs=rngs, param_dtype=param_dtype)

        # Zero initialization makes the initial transport an identity warp with
        # zero learned KV correction.  Random bases keep coefficient gradients
        # non-zero on the first optimization step.
        self.flow_head = _zero_linear(2 * hidden, 2, rngs=rngs, param_dtype=param_dtype)
        self.spatial_coeff_head = _zero_linear(hidden, config.rank, rngs=rngs, param_dtype=param_dtype)
        self.residual_gain_head = _zero_linear(hidden, 1, rngs=rngs, param_dtype=param_dtype)
        self.group_coeff_head = _zero_linear(
            hidden,
            config.num_layer_groups * config.rank,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        basis_scale = jnp.asarray(0.02, dtype=param_dtype)
        basis_shape = (config.num_layer_groups, config.rank, config.kv_dim)
        self.visual_k_basis = nnx.Param(
            jax.random.normal(rngs.params(), basis_shape, dtype=param_dtype) * basis_scale
        )
        self.visual_v_basis = nnx.Param(
            jax.random.normal(rngs.params(), basis_shape, dtype=param_dtype) * basis_scale
        )

        self.language_token_embedding = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (config.language_tokens, hidden),
                dtype=param_dtype,
            )
            * basis_scale
        )
        self.language_coeff_head = _zero_linear(
            hidden,
            config.num_layer_groups * config.rank,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.language_k_basis = nnx.Param(
            jax.random.normal(rngs.params(), basis_shape, dtype=param_dtype) * basis_scale
        )
        self.language_v_basis = nnx.Param(
            jax.random.normal(rngs.params(), basis_shape, dtype=param_dtype) * basis_scale
        )

        self.risk_hidden = nnx.Linear(2 * hidden, hidden, rngs=rngs, param_dtype=param_dtype)
        self.risk_head = _zero_linear(hidden, 1, rngs=rngs, param_dtype=param_dtype)

    def __call__(
        self,
        anchor_images: Array,
        current_images: Array,
        state_delta: Array,
        executed_actions: Array,
        executed_action_mask: Array,
        anchor_ear: Array,
        anchor_kv: KVCache,
    ) -> P3TPrefixTransportOutput:
        config = self.config
        batch = anchor_images.shape[0]
        expected_image_shape = (
            batch,
            config.num_views,
            config.image_size,
            config.image_size,
            3,
        )
        self._check_shape("anchor_images", anchor_images, expected_image_shape)
        self._check_shape("current_images", current_images, expected_image_shape)
        self._check_shape("state_delta", state_delta, (batch, config.state_dim))
        self._check_shape(
            "executed_actions",
            executed_actions,
            (batch, config.executed_action_horizon, config.action_dim),
        )
        self._check_shape(
            "executed_action_mask",
            executed_action_mask,
            (batch, config.executed_action_horizon),
        )
        self._check_shape("anchor_ear", anchor_ear, (batch, config.ear_horizon, config.action_dim))

        anchor_k, anchor_v = anchor_kv
        expected_cache_shape = (
            config.num_layers,
            batch,
            config.prefix_tokens,
            1,
            config.kv_dim,
        )
        self._check_shape("anchor_k", anchor_k, expected_cache_shape)
        self._check_shape("anchor_v", anchor_v, expected_cache_shape)

        patch_features, image_summary = self._encode_images(anchor_images, current_images)
        context, patch_features = self._encode_context(
            patch_features,
            image_summary,
            state_delta,
            executed_actions,
            executed_action_mask,
            anchor_ear,
        )
        transported_k, transported_v = self._transport_cache(
            anchor_k,
            anchor_v,
            patch_features,
            image_summary,
            context,
        )
        risk_features = jnp.concatenate([context, image_summary], axis=-1)
        risk = nnx.sigmoid(self.risk_head(nnx.swish(self.risk_hidden(risk_features))))[..., 0]
        return P3TPrefixTransportOutput(kv_cache=(transported_k, transported_v), risk=risk)

    def _encode_images(self, anchor_images: Array, current_images: Array) -> tuple[Array, Array]:
        config = self.config
        pool = config.image_size // config.token_grid_size

        def pool_to_tokens(images: Array) -> Array:
            batch, views, _, _, channels = images.shape
            return images.astype(jnp.float32).reshape(
                batch,
                views,
                config.token_grid_size,
                pool,
                config.token_grid_size,
                pool,
                channels,
            ).mean(axis=(3, 5))

        anchor_tokens = pool_to_tokens(anchor_images)
        current_tokens = pool_to_tokens(current_images)
        image_input = jnp.concatenate(
            [anchor_tokens, current_tokens, current_tokens - anchor_tokens],
            axis=-1,
        )
        patch_features = nnx.swish(self.image_in(image_input))
        local_features = _local_average_3x3(patch_features)
        patch_features = nnx.swish(self.image_mix(jnp.concatenate([patch_features, local_features], axis=-1)))
        image_summary = jnp.mean(patch_features, axis=(1, 2, 3))
        return patch_features, image_summary

    def _encode_context(
        self,
        patch_features: Array,
        image_summary: Array,
        state_delta: Array,
        executed_actions: Array,
        executed_action_mask: Array,
        anchor_ear: Array,
    ) -> tuple[Array, Array]:
        state_feature = nnx.swish(self.state_proj(state_delta.astype(jnp.float32)))
        action_tokens = nnx.swish(self.action_proj(executed_actions.astype(jnp.float32)))
        action_feature = _masked_mean(action_tokens, executed_action_mask, axis=1)
        ear_tokens = nnx.swish(self.ear_proj(anchor_ear.astype(jnp.float32)))
        ear_feature = jnp.mean(ear_tokens, axis=1)
        plan_feature = nnx.swish(
            self.plan_fusion(jnp.concatenate([state_feature, action_feature, ear_feature], axis=-1))
        )
        context = nnx.swish(self.context_fusion(jnp.concatenate([plan_feature, image_summary], axis=-1)))

        film_scale, film_shift = jnp.split(self.film(context), 2, axis=-1)
        film_scale = 1.0 + 0.1 * jnp.tanh(film_scale)
        patch_features = (
            patch_features * film_scale[:, None, None, None, :]
            + film_shift[:, None, None, None, :]
        )
        return context, patch_features

    def _transport_cache(
        self,
        anchor_k: Array,
        anchor_v: Array,
        patch_features: Array,
        image_summary: Array,
        context: Array,
    ) -> KVCache:
        del image_summary  # Kept explicit in the caller to make the risk path clear.
        config = self.config
        batch = anchor_k.shape[1]
        grid = config.token_grid_size
        layers_per_group = config.num_layers // config.num_layer_groups

        # Flow sees both the locally FiLM-conditioned feature and the global
        # plan/current-observation context.
        context_grid = jnp.broadcast_to(
            context[:, None, None, None, :],
            (*patch_features.shape[:-1], context.shape[-1]),
        )
        flow_input = jnp.concatenate([patch_features, context_grid], axis=-1)
        flow = config.max_flow_tokens * jnp.tanh(self.flow_head(flow_input))

        spatial_coeff = self.spatial_coeff_head(patch_features)
        group_coeff = 1.0 + jnp.tanh(self.group_coeff_head(context))
        group_coeff = group_coeff.reshape(batch, config.num_layer_groups, config.rank)
        visual_coeff = spatial_coeff[:, None, ...] * group_coeff[:, :, None, None, None, :]
        residual_gain = 1.0 + jnp.tanh(self.residual_gain_head(patch_features))

        visual_k_residual = jnp.einsum(
            "bgvhwr,grd->bgvhwd",
            visual_coeff,
            self.visual_k_basis,
        )
        visual_v_residual = jnp.einsum(
            "bgvhwr,grd->bgvhwd",
            visual_coeff,
            self.visual_v_basis,
        )
        visual_k_residual = jnp.repeat(visual_k_residual, layers_per_group, axis=1)
        visual_v_residual = jnp.repeat(visual_v_residual, layers_per_group, axis=1)

        visual_k = jnp.transpose(anchor_k[:, :, : config.visual_tokens, 0, :], (1, 0, 2, 3))
        visual_v = jnp.transpose(anchor_v[:, :, : config.visual_tokens, 0, :], (1, 0, 2, 3))
        visual_k = visual_k.astype(jnp.float32)
        visual_v = visual_v.astype(jnp.float32)

        flat_positions = jnp.arange(config.visual_tokens, dtype=jnp.float32)
        rope_positions = jnp.broadcast_to(
            flat_positions[None, :],
            (batch * config.num_layers, config.visual_tokens),
        )
        visual_k_flat = visual_k.reshape(
            batch * config.num_layers,
            config.visual_tokens,
            1,
            config.kv_dim,
        )
        visual_k_content = _apply_rope(
            visual_k_flat,
            -rope_positions,
            max_wavelength=config.rope_max_wavelength,
        ).reshape(batch, config.num_layers, config.num_views, grid, grid, config.kv_dim)
        visual_v_grid = visual_v.reshape(
            batch,
            config.num_layers,
            config.num_views,
            grid,
            grid,
            config.kv_dim,
        )

        warped_k_content = _bilinear_warp_token_grid(visual_k_content, flow)
        warped_v = _bilinear_warp_token_grid(visual_v_grid, flow)
        gain = residual_gain[:, None, ...]
        current_k_content = warped_k_content + gain * visual_k_residual
        current_v = warped_v + gain * visual_v_residual

        current_k_flat = current_k_content.reshape(
            batch * config.num_layers,
            config.visual_tokens,
            1,
            config.kv_dim,
        )
        current_k = _apply_rope(
            current_k_flat,
            rope_positions,
            max_wavelength=config.rope_max_wavelength,
        ).reshape(batch, config.num_layers, config.visual_tokens, config.kv_dim)
        current_v = current_v.reshape(batch, config.num_layers, config.visual_tokens, config.kv_dim)

        language_context = nnx.swish(self.language_token_embedding[None, :, :] + context[:, None, :])
        language_coeff = self.language_coeff_head(language_context)
        language_coeff = language_coeff.reshape(
            batch,
            config.language_tokens,
            config.num_layer_groups,
            config.rank,
        ).transpose(0, 2, 1, 3)
        language_k_residual = jnp.einsum(
            "bgtr,grd->bgtd",
            language_coeff,
            self.language_k_basis,
        )
        language_v_residual = jnp.einsum(
            "bgtr,grd->bgtd",
            language_coeff,
            self.language_v_basis,
        )
        language_k_residual = jnp.repeat(language_k_residual, layers_per_group, axis=1)
        language_v_residual = jnp.repeat(language_v_residual, layers_per_group, axis=1)

        language_start = config.visual_tokens + config.dummy_tokens
        language_k = jnp.transpose(anchor_k[:, :, language_start:, 0, :], (1, 0, 2, 3)).astype(jnp.float32)
        language_v = jnp.transpose(anchor_v[:, :, language_start:, 0, :], (1, 0, 2, 3)).astype(jnp.float32)
        current_language_k = language_k + language_k_residual
        current_language_v = language_v + language_v_residual

        # Return to Gemma cache layout [L,B,T,1,D].  The masked image segment
        # is sliced directly from the anchor cache and is never transformed.
        current_k = jnp.transpose(current_k, (1, 0, 2, 3))[:, :, :, None, :].astype(anchor_k.dtype)
        current_v = jnp.transpose(current_v, (1, 0, 2, 3))[:, :, :, None, :].astype(anchor_v.dtype)
        current_language_k = jnp.transpose(current_language_k, (1, 0, 2, 3))[:, :, :, None, :].astype(
            anchor_k.dtype
        )
        current_language_v = jnp.transpose(current_language_v, (1, 0, 2, 3))[:, :, :, None, :].astype(
            anchor_v.dtype
        )
        dummy_start = config.visual_tokens
        dummy_end = dummy_start + config.dummy_tokens
        transported_k = jnp.concatenate(
            [current_k, anchor_k[:, :, dummy_start:dummy_end, :, :], current_language_k],
            axis=2,
        )
        transported_v = jnp.concatenate(
            [current_v, anchor_v[:, :, dummy_start:dummy_end, :, :], current_language_v],
            axis=2,
        )
        return transported_k, transported_v

    @staticmethod
    def _check_shape(name: str, value: Array, expected: tuple[int, ...]) -> None:
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")

    def save(self, checkpoint_dir: str | os.PathLike[str], *, overwrite: bool = False) -> pathlib.Path:
        """Save this adapter as an independent, versioned NPZ checkpoint."""

        return save_p3t_prefix_transport(self, checkpoint_dir, overwrite=overwrite)

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | os.PathLike[str],
        *,
        rngs: nnx.Rngs | None = None,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> "P3TPrefixTransport":
        """Construct an adapter and restore an independent NPZ checkpoint."""

        return load_p3t_prefix_transport(
            checkpoint_dir,
            rngs=rngs,
            param_dtype=param_dtype,
        )


def _path_identifier(path: tuple[object, ...]) -> str:
    """Stable JSON identifier for a nested NNX state path."""

    encoded = []
    for component in path:
        if isinstance(component, str):
            encoded.append(["str", component])
        elif isinstance(component, int):
            encoded.append(["int", component])
        else:
            # NNX state paths are normally strings/integers.  Retaining the
            # class name makes validation deterministic if another key type is
            # introduced by a future Flax release.
            encoded.append([type(component).__name__, str(component)])
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_p3t_prefix_transport(
    module: P3TPrefixTransport,
    checkpoint_dir: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> pathlib.Path:
    """Save P3T parameters and config without serializing the NNX graph.

    The checkpoint directory contains ``params.npz`` plus ``config.json``.
    ``config.json`` records the exact state paths, shapes, dtypes, and a SHA256
    checksum.  The JSON file is installed last and therefore acts as the commit
    marker for an atomic save.
    """

    target = pathlib.Path(checkpoint_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / _CHECKPOINT_CONFIG_NAME
    params_path = target / _CHECKPOINT_PARAMS_NAME
    if not overwrite and (config_path.exists() or params_path.exists()):
        raise FileExistsError(
            f"P3T checkpoint already exists in {target}; pass overwrite=True to replace it."
        )

    _, state = nnx.split(module)
    flat_state = traverse_util.flatten_dict(state.to_pure_dict())
    ordered_state = sorted(flat_state.items(), key=lambda item: _path_identifier(item[0]))

    arrays: dict[str, np.ndarray] = {}
    parameter_records: list[dict[str, object]] = []
    for index, (path, value) in enumerate(ordered_state):
        array_name = f"param_{index:05d}"
        host_value = np.asarray(jax.device_get(value))
        arrays[array_name] = host_value
        parameter_records.append(
            {
                "array_name": array_name,
                "path": _path_identifier(path),
                "shape": list(host_value.shape),
                "dtype": str(host_value.dtype),
            }
        )

    temporary_params = target / f".{_CHECKPOINT_PARAMS_NAME}.tmp-{os.getpid()}"
    temporary_config = target / f".{_CHECKPOINT_CONFIG_NAME}.tmp-{os.getpid()}"
    try:
        with temporary_params.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        metadata = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "module": "P3TPrefixTransport",
            "config": dataclasses.asdict(module.config),
            "params_file": _CHECKPOINT_PARAMS_NAME,
            "params_sha256": _sha256(temporary_params),
            "parameters": parameter_records,
        }
        with temporary_config.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        os.replace(temporary_params, params_path)
        os.replace(temporary_config, config_path)
    finally:
        temporary_params.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)
    return target


def load_p3t_prefix_transport(
    checkpoint_dir: str | os.PathLike[str],
    *,
    rngs: nnx.Rngs | None = None,
    param_dtype: jnp.dtype = jnp.float32,
) -> P3TPrefixTransport:
    """Load a standalone P3T checkpoint with strict path/shape validation."""

    target = pathlib.Path(checkpoint_dir).expanduser()
    config_path = target / _CHECKPOINT_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing P3T checkpoint metadata: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported P3T checkpoint schema version "
            f"{metadata.get('schema_version')!r}; expected {_CHECKPOINT_SCHEMA_VERSION}."
        )
    if metadata.get("module") != "P3TPrefixTransport":
        raise ValueError(f"Unexpected checkpoint module {metadata.get('module')!r}.")
    try:
        config = P3TPrefixTransportConfig(**metadata["config"])
        parameter_records = metadata["parameters"]
        params_file = metadata["params_file"]
        expected_checksum = metadata["params_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed P3T checkpoint metadata in {config_path}.") from error

    params_path = target / params_file
    if not params_path.is_file():
        raise FileNotFoundError(f"Missing P3T parameter archive: {params_path}")
    actual_checksum = _sha256(params_path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"P3T parameter checksum mismatch for {params_path}: "
            f"expected {expected_checksum}, got {actual_checksum}."
        )

    module = P3TPrefixTransport(
        config,
        rngs=nnx.Rngs(0) if rngs is None else rngs,
        param_dtype=param_dtype,
    )
    graphdef, state = nnx.split(module)
    expected_flat = traverse_util.flatten_dict(state.to_pure_dict())
    expected_by_path = {
        _path_identifier(path): (path, value)
        for path, value in expected_flat.items()
    }
    saved_paths = {record.get("path") for record in parameter_records}
    if saved_paths != set(expected_by_path):
        missing = sorted(set(expected_by_path) - saved_paths)
        unexpected = sorted(saved_paths - set(expected_by_path))
        raise ValueError(
            "P3T checkpoint parameter paths do not match the constructed module; "
            f"missing={missing}, unexpected={unexpected}."
        )

    restored_flat: dict[tuple[object, ...], Array] = {}
    with np.load(params_path, allow_pickle=False) as archive:
        expected_array_names = {record.get("array_name") for record in parameter_records}
        if expected_array_names != set(archive.files):
            raise ValueError(
                "P3T NPZ entries do not match metadata; "
                f"metadata={sorted(expected_array_names)}, archive={sorted(archive.files)}."
            )
        for record in parameter_records:
            path_id = record["path"]
            actual_path, expected_value = expected_by_path[path_id]
            host_value = archive[record["array_name"]]
            recorded_shape = tuple(record["shape"])
            if host_value.shape != recorded_shape or host_value.shape != expected_value.shape:
                raise ValueError(
                    f"Shape mismatch for P3T state {path_id}: metadata={recorded_shape}, "
                    f"archive={host_value.shape}, module={expected_value.shape}."
                )
            if str(host_value.dtype) != record["dtype"]:
                raise ValueError(
                    f"Dtype mismatch for P3T state {path_id}: "
                    f"metadata={record['dtype']}, archive={host_value.dtype}."
                )
            restored_flat[actual_path] = jnp.asarray(host_value, dtype=expected_value.dtype)

    restored = traverse_util.unflatten_dict(restored_flat)
    state.replace_by_pure_dict(restored)
    return nnx.merge(graphdef, state)


__all__ = [
    "P3TPrefixTransport",
    "P3TPrefixTransportConfig",
    "P3TPrefixTransportOutput",
    "load_p3t_prefix_transport",
    "save_p3t_prefix_transport",
]
