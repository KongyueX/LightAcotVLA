"""Strict runtime for the deployable MRR-ACoT visual-block selector.

This module intentionally contains no policy integration.  It restores the
standalone selector artifact produced by ``train_mrr_acot_block_selector.py``
and exposes a JAX-only fixed-shape selection path.  The runtime feature
boundary cannot accept a Gemma KV cache or a teacher outcome: it consumes only
current/anchor pre-Gemma visual tokens, 64x64 RGB, and cached control context.

Block ids partition the two 16x16 visual-token grids into 4x4 blocks.  Ids
``0..15`` cover ``base_0_rgb`` in row-major order and ids ``16..31`` cover
``left_wrist_0_rgb``.  Each selected id therefore refreshes exactly 16 visual
tokens.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, NamedTuple

import flax.nnx as nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib


Array = jax.Array
NUM_VIEWS = 2
VIEW_NAMES = ("base_0_rgb", "left_wrist_0_rgb")
TOKEN_GRID = 16
BLOCK_EDGE = 4
BLOCKS_PER_VIEW = 16
NUM_BLOCKS = 32
TOKENS_PER_BLOCK = 16
VISUAL_TOKENS = 512
TOKEN_EMBEDDING_DIM = 2048
IMAGE_SIZE = 64
STATE_DIM = 32
ACTION_DIM = 32
EXECUTED_ACTION_HORIZON = 10
EAR_HORIZON = 15
TOP_K = 4
PROJECTION_RANK = 8
FEATURE_DIM = 132
HIDDEN_DIM = 32
PARAMETER_COUNT = 4_289
PARAMETER_NAMESPACE = "mrr_acot_block_selector"
SELECTOR_CONFIG_NAME = "selector_config.json"
PREPROCESSING_NAME = "feature_preprocessing.npz"
PARAMS_NAME = "params"

_EXPECTED_CONFIG_KEYS = frozenset(
    {
        "feature_dim",
        "hidden_dim",
        "top_k",
        "feature_projection_seed",
        "projection_rank",
        "input_contract",
        "forbidden_inputs",
    }
)
_EXPECTED_INPUT_CONTRACT = (
    "current-vs-anchor pre-Gemma visual prefix token statistics and fixed projections",
    "64x64 RGB block deltas",
    "view and block coordinates",
    "normalized state delta",
    "executed-action summaries",
    "anchor-EAR summaries",
)
_EXPECTED_FORBIDDEN_INPUTS = ("fresh deep KV delta", "teacher outcome")


@dataclasses.dataclass(frozen=True)
class MRRBlockSelectorConfig:
    """Strict metadata mirrored from the standalone training artifact."""

    feature_dim: int
    hidden_dim: int
    top_k: int
    feature_projection_seed: int
    projection_rank: int
    input_contract: tuple[str, ...]
    forbidden_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.feature_dim != FEATURE_DIM:
            raise ValueError(f"MRR feature_dim must be {FEATURE_DIM}, got {self.feature_dim}.")
        if self.hidden_dim != HIDDEN_DIM:
            raise ValueError(f"MRR hidden_dim must be {HIDDEN_DIM}, got {self.hidden_dim}.")
        if self.top_k != TOP_K:
            raise ValueError(f"MRR top_k must be {TOP_K}, got {self.top_k}.")
        if self.projection_rank != PROJECTION_RANK:
            raise ValueError(
                f"MRR projection_rank must be {PROJECTION_RANK}, got {self.projection_rank}."
            )
        if self.feature_projection_seed < 0:
            raise ValueError("MRR feature_projection_seed must be non-negative.")
        if self.input_contract != _EXPECTED_INPUT_CONTRACT:
            raise ValueError(
                "MRR selector input contract does not match the deployable runtime contract: "
                f"{self.input_contract!r}."
            )
        if self.forbidden_inputs != _EXPECTED_FORBIDDEN_INPUTS:
            raise ValueError(
                "MRR forbidden-input declaration does not match the runtime boundary: "
                f"{self.forbidden_inputs!r}."
            )


class MRRBlockSelectorOutput(NamedTuple):
    """Top-4 selection for every batch item.

    Attributes:
        selected_block_ids: Descending-score unique ids, ``int32[B,4]``.
        selected_block_scores: Scores aligned with ids, ``float32[B,4]``.
        block_logits: Dense scores in block-id order, ``float32[B,32]``.
    """

    selected_block_ids: Array
    selected_block_scores: Array
    block_logits: Array


class MRRBlockSelectorScorer(nnx.Module):
    """The exact 4,289-parameter shared scorer used during training."""

    def __init__(self, feature_dim: int, hidden_dim: int, *, rngs: nnx.Rngs) -> None:
        self.input = nnx.Linear(feature_dim, hidden_dim, rngs=rngs, param_dtype=jnp.float32)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=jnp.float32)

    def __call__(self, features: Array) -> Array:
        return self.output(nnx.swish(self.input(features.astype(jnp.float32))))[..., 0]


@dataclasses.dataclass(frozen=True, eq=False)
class MRRBlockSelectorRuntime:
    """Loaded frozen scorer plus preprocessing needed for JAX inference.

    Capture :meth:`select` in an outer ``jax.jit``.  All work after loading,
    including feature construction, normalization, scoring, and top-k, stays in
    JAX.  Inputs carry a leading batch dimension and outputs are always
    ``[B,4]`` for top-4 ids/scores.
    """

    scorer_graphdef: Any
    scorer_state: nnx.State
    feature_mean: Array
    feature_std: Array
    config: MRRBlockSelectorConfig
    artifact_dir: pathlib.Path

    def select(
        self,
        anchor_visual_tokens: Array,
        current_visual_tokens: Array,
        anchor_rgb_64: Array,
        current_rgb_64: Array,
        state_delta: Array,
        executed_actions: Array,
        anchor_ear: Array,
    ) -> MRRBlockSelectorOutput:
        """Select top-4 blocks from strictly deployable, batched inputs."""

        features = construct_mrr_block_features(
            anchor_visual_tokens,
            current_visual_tokens,
            anchor_rgb_64,
            current_rgb_64,
            state_delta,
            executed_actions,
            anchor_ear,
            projection_seed=self.config.feature_projection_seed,
            projection_rank=self.config.projection_rank,
        )
        normalized = (features - self.feature_mean[None, None, :]) / self.feature_std[None, None, :]
        scorer = nnx.merge(self.scorer_graphdef, self.scorer_state)
        block_logits = scorer(normalized).astype(jnp.float32)
        selected_block_scores, selected_block_ids = jax.lax.top_k(
            block_logits,
            self.config.top_k,
        )
        return MRRBlockSelectorOutput(
            selected_block_ids=selected_block_ids.astype(jnp.int32),
            selected_block_scores=selected_block_scores.astype(jnp.float32),
            block_logits=block_logits,
        )

    __call__ = select


def _block_token_indices() -> np.ndarray:
    indices = np.empty((NUM_BLOCKS, TOKENS_PER_BLOCK), dtype=np.int32)
    block_id = 0
    for view in range(NUM_VIEWS):
        view_offset = view * TOKEN_GRID * TOKEN_GRID
        for block_row in range(TOKEN_GRID // BLOCK_EDGE):
            for block_col in range(TOKEN_GRID // BLOCK_EDGE):
                values = []
                for row in range(block_row * BLOCK_EDGE, (block_row + 1) * BLOCK_EDGE):
                    start = view_offset + row * TOKEN_GRID + block_col * BLOCK_EDGE
                    values.extend(range(start, start + BLOCK_EDGE))
                indices[block_id] = np.asarray(values, dtype=np.int32)
                block_id += 1
    if block_id != NUM_BLOCKS:
        raise AssertionError("Invalid MRR visual block partition.")
    if not np.array_equal(np.sort(indices.reshape(-1)), np.arange(VISUAL_TOKENS)):
        raise AssertionError("MRR blocks must partition all visual tokens exactly once.")
    return indices


def _coordinate_features() -> np.ndarray:
    rows = []
    for block_id in range(NUM_BLOCKS):
        view = block_id // BLOCKS_PER_VIEW
        local = block_id % BLOCKS_PER_VIEW
        row, col = local // 4, local % 4
        rows.append(
            np.concatenate(
                [
                    np.eye(2, dtype=np.float32)[view],
                    np.asarray([2.0 * row / 3.0 - 1.0, 2.0 * col / 3.0 - 1.0], dtype=np.float32),
                    np.eye(4, dtype=np.float32)[row],
                    np.eye(4, dtype=np.float32)[col],
                ]
            )
        )
    return np.stack(rows)


BLOCK_TOKEN_INDICES = jnp.asarray(_block_token_indices(), dtype=jnp.int32)
COORDINATE_FEATURES = jnp.asarray(_coordinate_features(), dtype=jnp.float32)


def _rademacher_projection(key: Array, input_dim: int, rank: int) -> Array:
    signs = 2.0 * jax.random.bernoulli(key, shape=(input_dim, rank)).astype(jnp.float32) - 1.0
    return signs / jnp.sqrt(jnp.asarray(input_dim, dtype=jnp.float32))


def _project_summaries(vectors: tuple[Array, ...], *, seed: int, rank: int) -> Array:
    projected = []
    norms = []
    root = jax.random.key(seed)
    for index, vector in enumerate(vectors):
        projection = _rademacher_projection(jax.random.fold_in(root, index), vector.shape[-1], rank)
        vector = vector.astype(jnp.float32)
        projected.append(vector @ projection)
        norms.append(jnp.sqrt(jnp.mean(jnp.square(vector)) + 1e-8)[None])
    return jnp.concatenate([*projected, *norms], axis=-1)


def _construct_features_single(
    anchor_visual_tokens: Array,
    current_visual_tokens: Array,
    anchor_rgb_64: Array,
    current_rgb_64: Array,
    state_delta: Array,
    executed_actions: Array,
    anchor_ear: Array,
    *,
    projection_seed: int,
    projection_rank: int,
) -> Array:
    """Mirror the training feature construction for one batch element."""

    anchor_tokens = anchor_visual_tokens.astype(jnp.float32)
    current_tokens = current_visual_tokens.astype(jnp.float32)
    anchor_blocks = anchor_tokens[BLOCK_TOKEN_INDICES]
    current_blocks = current_tokens[BLOCK_TOKEN_INDICES]
    delta = current_blocks - anchor_blocks
    delta_mean = jnp.mean(delta, axis=1)
    delta_abs_mean = jnp.mean(jnp.abs(delta), axis=1)
    delta_rms_by_dim = jnp.sqrt(jnp.mean(jnp.square(delta), axis=1) + 1e-8)
    embedding_dim = delta.shape[-1]
    root = jax.random.key(projection_seed)
    local_projection = jnp.concatenate(
        [
            delta_mean
            @ _rademacher_projection(jax.random.fold_in(root, 0), embedding_dim, projection_rank),
            delta_abs_mean
            @ _rademacher_projection(jax.random.fold_in(root, 1), embedding_dim, projection_rank),
            delta_rms_by_dim
            @ _rademacher_projection(jax.random.fold_in(root, 2), embedding_dim, projection_rank),
        ],
        axis=-1,
    )
    dot = jnp.sum(anchor_blocks * current_blocks, axis=(1, 2))
    denominator = jnp.sqrt(
        jnp.sum(jnp.square(anchor_blocks), axis=(1, 2))
        * jnp.sum(jnp.square(current_blocks), axis=(1, 2))
        + 1e-8
    )
    local_statistics = jnp.stack(
        [
            jnp.mean(jnp.abs(delta), axis=(1, 2)),
            jnp.sqrt(jnp.mean(jnp.square(delta), axis=(1, 2)) + 1e-8),
            jnp.max(jnp.abs(delta), axis=(1, 2)),
            jnp.sqrt(jnp.mean(jnp.square(anchor_blocks), axis=(1, 2)) + 1e-8),
            jnp.sqrt(jnp.mean(jnp.square(current_blocks), axis=(1, 2)) + 1e-8),
            dot / jnp.maximum(denominator, 1e-8),
        ],
        axis=-1,
    )

    def image_tokens(images: Array) -> Array:
        return images.reshape(NUM_VIEWS, TOKEN_GRID, 4, TOKEN_GRID, 4, 3).mean(axis=(2, 4)).reshape(
            VISUAL_TOKENS, 3
        )

    anchor_rgb = image_tokens(anchor_rgb_64)[BLOCK_TOKEN_INDICES]
    current_rgb = image_tokens(current_rgb_64)[BLOCK_TOKEN_INDICES]
    rgb_delta = current_rgb - anchor_rgb
    rgb_features = jnp.concatenate(
        [
            jnp.mean(rgb_delta, axis=1),
            jnp.mean(jnp.abs(rgb_delta), axis=1),
            jnp.mean(jnp.abs(rgb_delta), axis=(1, 2))[:, None],
            jnp.sqrt(jnp.mean(jnp.square(rgb_delta), axis=(1, 2)) + 1e-8)[:, None],
            jnp.max(jnp.abs(rgb_delta), axis=(1, 2))[:, None],
        ],
        axis=-1,
    )

    actions = executed_actions.astype(jnp.float32)
    ear = anchor_ear.astype(jnp.float32)
    global_vectors = (
        state_delta.astype(jnp.float32),
        jnp.mean(actions, axis=0),
        jnp.std(actions, axis=0),
        actions[0],
        actions[-1],
        jnp.mean(ear, axis=0),
        jnp.std(ear, axis=0),
        ear[0],
        ear[-1],
    )
    global_features = _project_summaries(
        global_vectors,
        seed=projection_seed + 10_000,
        rank=projection_rank,
    )
    global_features = jnp.broadcast_to(global_features[None, :], (NUM_BLOCKS, global_features.size))
    return jnp.concatenate(
        [local_projection, local_statistics, rgb_features, COORDINATE_FEATURES, global_features],
        axis=-1,
    )


def _check_batched_inputs(
    anchor_visual_tokens: Array,
    current_visual_tokens: Array,
    anchor_rgb_64: Array,
    current_rgb_64: Array,
    state_delta: Array,
    executed_actions: Array,
    anchor_ear: Array,
) -> int:
    if anchor_visual_tokens.ndim != 3:
        raise ValueError(
            "anchor_visual_tokens must have shape [B,512,2048], "
            f"got {anchor_visual_tokens.shape}."
        )
    batch = anchor_visual_tokens.shape[0]
    expected = {
        "anchor_visual_tokens": (batch, VISUAL_TOKENS, TOKEN_EMBEDDING_DIM),
        "current_visual_tokens": (batch, VISUAL_TOKENS, TOKEN_EMBEDDING_DIM),
        "anchor_rgb_64": (batch, NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 3),
        "current_rgb_64": (batch, NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 3),
        "state_delta": (batch, STATE_DIM),
        "executed_actions": (batch, EXECUTED_ACTION_HORIZON, ACTION_DIM),
        "anchor_ear": (batch, EAR_HORIZON, ACTION_DIM),
    }
    actual = {
        "anchor_visual_tokens": anchor_visual_tokens.shape,
        "current_visual_tokens": current_visual_tokens.shape,
        "anchor_rgb_64": anchor_rgb_64.shape,
        "current_rgb_64": current_rgb_64.shape,
        "state_delta": state_delta.shape,
        "executed_actions": executed_actions.shape,
        "anchor_ear": anchor_ear.shape,
    }
    mismatches = {name: (actual[name], shape) for name, shape in expected.items() if actual[name] != shape}
    if mismatches:
        raise ValueError(f"Invalid MRR selector input shapes (actual, expected): {mismatches}.")
    if batch <= 0:
        raise ValueError("MRR selector requires a positive batch size.")
    return batch


def construct_mrr_block_features(
    anchor_visual_tokens: Array,
    current_visual_tokens: Array,
    anchor_rgb_64: Array,
    current_rgb_64: Array,
    state_delta: Array,
    executed_actions: Array,
    anchor_ear: Array,
    *,
    projection_seed: int,
    projection_rank: int = PROJECTION_RANK,
) -> Array:
    """Build the exact normalized-input features before preprocessing.

    Args use shapes ``[B,512,2048]``, ``[B,2,64,64,3]``, ``[B,32]``,
    ``[B,10,32]``, and ``[B,15,32]`` respectively.  The returned tensor is
    float32 ``[B,32,132]`` in fixed block-id order.
    """

    _check_batched_inputs(
        anchor_visual_tokens,
        current_visual_tokens,
        anchor_rgb_64,
        current_rgb_64,
        state_delta,
        executed_actions,
        anchor_ear,
    )
    if projection_rank != PROJECTION_RANK:
        raise ValueError(f"MRR projection_rank must be {PROJECTION_RANK}, got {projection_rank}.")
    return jax.vmap(
        lambda anchor_tokens, current_tokens, anchor_rgb, current_rgb, delta, actions, ear: (
            _construct_features_single(
                anchor_tokens,
                current_tokens,
                anchor_rgb,
                current_rgb,
                delta,
                actions,
                ear,
                projection_seed=projection_seed,
                projection_rank=projection_rank,
            )
        )
    )(
        anchor_visual_tokens,
        current_visual_tokens,
        anchor_rgb_64,
        current_rgb_64,
        state_delta,
        executed_actions,
        anchor_ear,
    )


def _load_config(path: pathlib.Path) -> MRRBlockSelectorConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed MRR selector config: {path}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"MRR selector config must be a JSON object: {path}.")
    keys = frozenset(payload)
    if keys != _EXPECTED_CONFIG_KEYS:
        raise ValueError(
            "MRR selector config keys do not match the runtime schema; "
            f"missing={sorted(_EXPECTED_CONFIG_KEYS - keys)}, unexpected={sorted(keys - _EXPECTED_CONFIG_KEYS)}."
        )
    integer_fields = (
        "feature_dim",
        "hidden_dim",
        "top_k",
        "feature_projection_seed",
        "projection_rank",
    )
    if any(type(payload[name]) is not int for name in integer_fields):
        raise ValueError(f"MRR selector integer config fields have invalid types in {path}.")
    if not isinstance(payload["input_contract"], list) or not all(
        isinstance(value, str) for value in payload["input_contract"]
    ):
        raise ValueError(f"MRR input_contract must be a list of strings in {path}.")
    if not isinstance(payload["forbidden_inputs"], list) or not all(
        isinstance(value, str) for value in payload["forbidden_inputs"]
    ):
        raise ValueError(f"MRR forbidden_inputs must be a list of strings in {path}.")
    return MRRBlockSelectorConfig(
        feature_dim=payload["feature_dim"],
        hidden_dim=payload["hidden_dim"],
        top_k=payload["top_k"],
        feature_projection_seed=payload["feature_projection_seed"],
        projection_rank=payload["projection_rank"],
        input_contract=tuple(payload["input_contract"]),
        forbidden_inputs=tuple(payload["forbidden_inputs"]),
    )


def _resolve_artifact_dir(checkpoint_dir: str | pathlib.Path) -> pathlib.Path:
    requested = pathlib.Path(checkpoint_dir).expanduser().resolve()
    if not requested.is_dir():
        raise FileNotFoundError(f"MRR selector checkpoint directory does not exist: {requested}.")

    direct_markers = (
        requested / SELECTOR_CONFIG_NAME,
        requested / PREPROCESSING_NAME,
        requested / PARAMS_NAME,
    )
    final = requested / "final"
    final_markers = (
        final / SELECTOR_CONFIG_NAME,
        final / PREPROCESSING_NAME,
        final / PARAMS_NAME,
    )
    direct_complete = (
        direct_markers[0].is_file()
        and direct_markers[1].is_file()
        and direct_markers[2].is_dir()
    )
    final_complete = (
        final_markers[0].is_file()
        and final_markers[1].is_file()
        and final_markers[2].is_dir()
    )
    if direct_complete and final_complete:
        raise ValueError(
            f"Ambiguous MRR selector checkpoint: both {requested} and {final} contain complete artifacts."
        )
    if direct_complete:
        return requested
    if final_complete:
        return final
    missing_direct = [path.name for path in direct_markers if not path.exists()]
    missing_final = [path.name for path in final_markers if not path.exists()]
    raise FileNotFoundError(
        f"Incomplete MRR selector artifact at {requested}; "
        f"direct missing={missing_direct}, final missing={missing_final}."
    )


def _load_preprocessing(path: pathlib.Path) -> tuple[Array, Array]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"feature_mean", "feature_std"}:
            raise ValueError(
                f"MRR preprocessing entries must be feature_mean/feature_std, got {archive.files}."
            )
        feature_mean = np.asarray(archive["feature_mean"])
        feature_std = np.asarray(archive["feature_std"])
    for name, value in (("feature_mean", feature_mean), ("feature_std", feature_std)):
        if value.shape != (FEATURE_DIM,):
            raise ValueError(f"MRR {name} must have shape ({FEATURE_DIM},), got {value.shape}.")
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"MRR {name} must be float32, got {value.dtype}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"MRR {name} contains non-finite values.")
    if np.any(feature_std < 1e-4):
        raise ValueError("MRR feature_std violates the training floor of 1e-4.")
    return jnp.asarray(feature_mean), jnp.asarray(feature_std)


def _load_scorer(params_dir: pathlib.Path, config: MRRBlockSelectorConfig) -> tuple[Any, nnx.State]:
    scorer = MRRBlockSelectorScorer(
        config.feature_dim,
        config.hidden_dim,
        rngs=nnx.Rngs(0),
    )
    graphdef, state = nnx.split(scorer)
    loaded = model_lib.restore_params(params_dir, restore_type=np.ndarray)
    loaded = model_lib.convert_str_keys_to_int(loaded)
    if not isinstance(loaded, dict) or set(loaded) != {PARAMETER_NAMESPACE}:
        keys = sorted(loaded) if isinstance(loaded, dict) else type(loaded).__name__
        raise ValueError(
            f"MRR Orbax params must contain only namespace {PARAMETER_NAMESPACE!r}, got {keys!r}."
        )
    loaded = loaded[PARAMETER_NAMESPACE]
    expected_flat = traverse_util.flatten_dict(state.to_pure_dict())
    loaded_flat = traverse_util.flatten_dict(loaded)
    if set(loaded_flat) != set(expected_flat):
        missing = sorted(set(expected_flat) - set(loaded_flat))
        unexpected = sorted(set(loaded_flat) - set(expected_flat))
        raise ValueError(f"MRR scorer state paths mismatch; missing={missing}, unexpected={unexpected}.")

    count = 0
    for path, expected in expected_flat.items():
        value = np.asarray(loaded_flat[path])
        if value.shape != expected.shape:
            raise ValueError(
                f"MRR scorer shape mismatch at {path}: checkpoint={value.shape}, expected={expected.shape}."
            )
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"MRR scorer dtype mismatch at {path}: expected float32, got {value.dtype}.")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"MRR scorer parameter {path} contains non-finite values.")
        count += int(value.size)
    if count != PARAMETER_COUNT:
        raise ValueError(f"MRR scorer must contain {PARAMETER_COUNT} parameters, got {count}.")

    state.replace_by_pure_dict(jax.tree.map(lambda value: jnp.asarray(value, dtype=jnp.float32), loaded))
    return graphdef, state


def load_mrr_block_selector(checkpoint_dir: str | pathlib.Path) -> MRRBlockSelectorRuntime:
    """Strictly load an MRR artifact root or its ``final`` directory.

    Expected files under the resolved artifact directory are
    ``selector_config.json``, ``feature_preprocessing.npz``, and the Orbax
    ``params`` directory.  No partial or extra parameter tree is accepted.
    """

    artifact_dir = _resolve_artifact_dir(checkpoint_dir)
    config = _load_config(artifact_dir / SELECTOR_CONFIG_NAME)
    feature_mean, feature_std = _load_preprocessing(artifact_dir / PREPROCESSING_NAME)
    graphdef, state = _load_scorer(artifact_dir / PARAMS_NAME, config)
    return MRRBlockSelectorRuntime(
        scorer_graphdef=graphdef,
        scorer_state=state,
        feature_mean=feature_mean,
        feature_std=feature_std,
        config=config,
        artifact_dir=artifact_dir,
    )
