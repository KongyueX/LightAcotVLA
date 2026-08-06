"""GPU microbenchmark for fixed-shape MRR active-query Gemma replay.

This is an isolated feasibility benchmark.  It does not add a policy mode or
change normal inference.  One episode-disjoint Task8 anchor/current pair is
loaded with the same split used by the P3T and MRR probes.  The benchmark then
compares:

* the existing full-fresh 968-query prefix path;
* 264 active queries: 64 fixed visual tokens plus all 200 language slots; and
* 520 active queries: the same queries plus all 256 masked dummy-image slots.

Both active paths run the existing SigLIP module on the two valid current
views and the existing Gemma token embedder on the current language.  They run
all 18 Gemma layers on physically shorter query tensors.  At every layer, the
fixed inactive anchor K/V are gathered first and the new active K/V are
appended, keeping exactly 968 logical attention keys; the appended K/V are
then scattered back into a 968-token composite cache.  The active queries
retain their positions from the original full prefix.  No dense 968-query
masked surrogate or dynamic top-k is used.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from collections.abc import Callable
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import model as model_lib

try:
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("benchmark_mrr_active_gemma_replay")

NUM_LAYERS = 18
PREFIX_TOKENS = 968
PREFIX_WIDTH = 2048
KV_HEAD_DIM = 256
TOKENS_PER_IMAGE = 256
VALID_VISUAL_TOKENS = 2 * TOKENS_PER_IMAGE
DUMMY_TOKENS = TOKENS_PER_IMAGE
LANGUAGE_TOKENS = 200
FIXED_VISUAL_QUERIES = 64
TOKEN_GRID = 16
VALID_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
DUMMY_IMAGE_KEY = "right_wrist_0_rgb"


def _fixed_visual_indices() -> np.ndarray:
    """Choose 32 spatially uniform tokens per valid view with no runtime top-k."""

    indices: list[int] = []
    for view in range(len(VALID_IMAGE_KEYS)):
        view_offset = view * TOKENS_PER_IMAGE
        for row in range(0, TOKEN_GRID, 2):
            for col in range(0, TOKEN_GRID, 4):
                indices.append(view_offset + row * TOKEN_GRID + col)
    result = np.asarray(indices, dtype=np.int32)
    if result.shape != (FIXED_VISUAL_QUERIES,):
        raise AssertionError(f"Expected {FIXED_VISUAL_QUERIES} fixed visual indices, got {result.shape}.")
    if np.unique(result).size != result.size or np.any(result < 0) or np.any(result >= VALID_VISUAL_TOKENS):
        raise AssertionError("Fixed visual query indices must be unique and lie in the two valid views.")
    return result


FIXED_VISUAL_INDICES = _fixed_visual_indices()


def _active_indices(*, include_dummy: bool) -> np.ndarray:
    pieces = [FIXED_VISUAL_INDICES]
    if include_dummy:
        pieces.append(np.arange(VALID_VISUAL_TOKENS, VALID_VISUAL_TOKENS + DUMMY_TOKENS, dtype=np.int32))
    pieces.append(np.arange(VALID_VISUAL_TOKENS + DUMMY_TOKENS, PREFIX_TOKENS, dtype=np.int32))
    result = np.concatenate(pieces)
    expected = FIXED_VISUAL_QUERIES + LANGUAGE_TOKENS + (DUMMY_TOKENS if include_dummy else 0)
    if result.shape != (expected,) or np.unique(result).size != result.size:
        raise AssertionError("Active query indices do not have the requested fixed shape.")
    return result


ACTIVE_INDICES_264 = _active_indices(include_dummy=False)
ACTIVE_INDICES_520 = _active_indices(include_dummy=True)


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    output_dir: str
    endpoint_student_params: str | None = None
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    dataset_task_id: int = 6
    temporal_stride: int = 10
    seed: int = 7
    split_seed: int = 7
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    expected_heldout_pairs: int = 22
    pair_index: int = 0
    warmup: int = 30
    iterations: int = 200
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.dataset_task_id < 0 or args.seed < 0 or args.split_seed < 0:
        raise ValueError("Task id and seeds must be non-negative.")
    if args.temporal_stride != 10:
        raise ValueError("This benchmark reuses the fixed anchor-to-anchor+10 Task8 protocol.")
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("Validation/test fractions must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    if args.expected_heldout_pairs <= 0 or not 0 <= args.pair_index < args.expected_heldout_pairs:
        raise ValueError("pair_index must lie inside the expected held-out pair count.")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive.")


def _make_anchor_preparer(base_graphdef: Any) -> Callable[..., Any]:
    @jax.jit
    def prepare_anchor(base_state: nnx.State, observation: model_lib.Observation) -> dict[str, Any]:
        base_model = nnx.merge(base_graphdef, base_state)
        prefix = base_model._compute_prefix_state(observation)  # noqa: SLF001
        return {
            "prefix_tokens": prefix["prefix_tokens"],
            "prefix_mask": prefix["prefix_mask"],
            "prefix_out": prefix["prefix_out"],
            "kv_cache": prefix["kv_cache"],
        }

    return prepare_anchor


def _make_full_fresh_prefix(base_graphdef: Any) -> Callable[..., Any]:
    @jax.jit
    def full_fresh_prefix(
        base_state: nnx.State,
        observation: model_lib.Observation,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        prefix = base_model._compute_prefix_state(observation)  # noqa: SLF001
        cache_k, cache_v = prefix["kv_cache"]
        return cache_k, cache_v, prefix["prefix_out"], prefix["prefix_mask"]

    return full_fresh_prefix


def _make_active_replay(
    base_graphdef: Any,
    *,
    include_dummy: bool,
) -> Callable[..., Any]:
    active_indices_np = ACTIVE_INDICES_520 if include_dummy else ACTIVE_INDICES_264
    active_query_count = int(active_indices_np.size)
    inactive_indices_np = np.setdiff1d(
        np.arange(PREFIX_TOKENS, dtype=np.int32),
        active_indices_np,
        assume_unique=True,
    )
    inactive_query_count = int(inactive_indices_np.size)
    if inactive_query_count + active_query_count != PREFIX_TOKENS:
        raise AssertionError("Inactive and active fixed indices must partition the full prefix.")
    active_indices = jnp.asarray(active_indices_np, dtype=jnp.int32)
    inactive_indices = jnp.asarray(inactive_indices_np, dtype=jnp.int32)
    fixed_visual_indices = jnp.asarray(FIXED_VISUAL_INDICES, dtype=jnp.int32)

    @jax.jit
    def active_replay(
        base_state: nnx.State,
        current_observation: model_lib.Observation,
        anchor_cache_k: jax.Array,
        anchor_cache_v: jax.Array,
        anchor_prefix_out: jax.Array,
        anchor_dummy_tokens: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        observation = model_lib.preprocess_observation(None, current_observation, train=False)

        visual_tokens: list[jax.Array] = []
        visual_masks: list[jax.Array] = []
        for image_key in VALID_IMAGE_KEYS:
            image_tokens, _ = base_model.PaliGemma.img(observation.images[image_key], train=False)
            if image_tokens.shape[1] != TOKENS_PER_IMAGE:
                raise ValueError(
                    f"Expected {TOKENS_PER_IMAGE} SigLIP tokens for {image_key}, got {image_tokens.shape}."
                )
            visual_tokens.append(image_tokens)
            visual_masks.append(
                jnp.broadcast_to(observation.image_masks[image_key][:, None], image_tokens.shape[:2])
            )
        current_visual_tokens = jnp.concatenate(visual_tokens, axis=1)
        current_visual_mask = jnp.concatenate(visual_masks, axis=1)

        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("MRR replay requires the fixed 200-slot language prefix.")
        current_language_tokens = base_model.PaliGemma.llm(observation.tokenized_prompt, method="embed")
        if current_language_tokens.shape[1] != LANGUAGE_TOKENS:
            raise ValueError(
                f"Expected {LANGUAGE_TOKENS} language slots, got {current_language_tokens.shape}."
            )
        if anchor_dummy_tokens.shape[1] != DUMMY_TOKENS:
            raise ValueError(f"Expected {DUMMY_TOKENS} anchor dummy tokens, got {anchor_dummy_tokens.shape}.")

        dummy_mask = jnp.broadcast_to(
            observation.image_masks[DUMMY_IMAGE_KEY][:, None],
            (observation.state.shape[0], DUMMY_TOKENS),
        )
        full_prefix_mask = jnp.concatenate(
            [current_visual_mask, dummy_mask, observation.tokenized_prompt_mask],
            axis=1,
        )
        if full_prefix_mask.shape[1] != PREFIX_TOKENS:
            raise ValueError(f"Expected a {PREFIX_TOKENS}-slot full prefix, got {full_prefix_mask.shape}.")
        full_positions = jnp.cumsum(full_prefix_mask, axis=1) - 1

        selected_visual_tokens = jnp.take(current_visual_tokens, fixed_visual_indices, axis=1)
        token_pieces = [selected_visual_tokens]
        if include_dummy:
            # The third view is a deterministic zero/masked dummy image.  Its
            # pre-Gemma embedding is invariant, so reuse the anchor embedding
            # while still replaying all 256 dummy queries through every Gemma
            # layer.  Only the two valid current views run SigLIP.
            token_pieces.append(anchor_dummy_tokens)
        token_pieces.append(current_language_tokens)
        active_tokens = jnp.concatenate(token_pieces, axis=1)
        if active_tokens.shape[1] != active_query_count:
            raise ValueError(
                f"Expected {active_query_count} physical active queries, got {active_tokens.shape}."
            )

        active_mask = jnp.take(full_prefix_mask, active_indices, axis=1)
        active_positions = jnp.take(full_positions, active_indices, axis=1)

        # Remove stale copies of active slots before using Gemma's append API.
        # This keeps the physical key count at the original 968.  It is also
        # required for the existing all-false masked-query behavior: dummy and
        # padded queries take a uniform mean over physical keys, so retaining
        # duplicate stale active keys would silently change their semantics.
        inactive_anchor_k = jnp.take(anchor_cache_k, inactive_indices, axis=2)
        inactive_anchor_v = jnp.take(anchor_cache_v, inactive_indices, axis=2)
        inactive_key_mask = jnp.take(full_prefix_mask, inactive_indices, axis=1)
        composite_key_mask = jnp.concatenate([inactive_key_mask, active_mask], axis=1)
        replay_attention_mask = jnp.logical_and(
            active_mask[:, :, None],
            composite_key_mask[:, None, :],
        )

        (active_prefix_out, _, _), appended_cache = base_model.PaliGemma.llm(
            [active_tokens, None, None],
            positions=active_positions,
            mask=replay_attention_mask,
            kv_cache=(inactive_anchor_k, inactive_anchor_v),
        )
        appended_k, appended_v = appended_cache
        expected_appended_tokens = inactive_query_count + active_query_count
        if appended_k.shape[:3] != (NUM_LAYERS, observation.state.shape[0], expected_appended_tokens):
            raise ValueError(
                "Gemma did not return the expected append cache: "
                f"K shape={appended_k.shape}, expected [18,B,{expected_appended_tokens},1,256]."
            )
        if appended_v.shape != appended_k.shape:
            raise ValueError(f"K/V append cache shapes differ: {appended_k.shape} vs {appended_v.shape}.")

        fresh_active_k = appended_k[:, :, inactive_query_count:]
        fresh_active_v = appended_v[:, :, inactive_query_count:]
        composite_k = anchor_cache_k.at[:, :, active_indices, :, :].set(fresh_active_k)
        composite_v = anchor_cache_v.at[:, :, active_indices, :, :].set(fresh_active_v)
        composite_prefix_out = anchor_prefix_out.at[:, active_indices, :].set(active_prefix_out)
        return composite_k, composite_v, composite_prefix_out, full_prefix_mask

    return active_replay


def _block(result: Any) -> Any:
    return jax.block_until_ready(result)


def _time_callable(
    function: Callable[..., Any],
    call_args: tuple[Any, ...],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float | int], Any]:
    output: Any = None
    for _ in range(warmup):
        output = _block(function(*call_args))

    samples = np.empty((iterations,), dtype=np.float64)
    for index in range(iterations):
        started = time.perf_counter_ns()
        output = function(*call_args)
        _block(output)
        samples[index] = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }, output


def _tensor_checksum(value: jax.Array) -> dict[str, float]:
    value_f32 = value.astype(jnp.float32)
    total, square_total, absolute_total = jax.device_get(
        (
            jnp.sum(value_f32),
            jnp.sum(jnp.square(value_f32)),
            jnp.sum(jnp.abs(value_f32)),
        )
    )
    return {
        "sum_f32": float(total),
        "square_sum_f32": float(square_total),
        "absolute_sum_f32": float(absolute_total),
    }


def _summarize_composite(result: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> dict[str, Any]:
    cache_k, cache_v, prefix_out, prefix_mask = result
    expected_cache_shape = (NUM_LAYERS, 1, PREFIX_TOKENS, 1, KV_HEAD_DIM)
    expected_prefix_out_shape = (1, PREFIX_TOKENS, PREFIX_WIDTH)
    expected_mask_shape = (1, PREFIX_TOKENS)
    compatible = (
        cache_k.shape == expected_cache_shape
        and cache_v.shape == expected_cache_shape
        and prefix_out.shape == expected_prefix_out_shape
        and prefix_mask.shape == expected_mask_shape
    )
    if not compatible:
        raise ValueError(
            "Composite cannot be consumed by the existing downstream path: "
            f"K={cache_k.shape}, V={cache_v.shape}, out={prefix_out.shape}, mask={prefix_mask.shape}."
        )
    return {
        "downstream_compatible": True,
        "cache_k_shape": list(cache_k.shape),
        "cache_v_shape": list(cache_v.shape),
        "prefix_out_shape": list(prefix_out.shape),
        "prefix_mask_shape": list(prefix_mask.shape),
        "checksum": {
            "cache_k": _tensor_checksum(cache_k),
            "cache_v": _tensor_checksum(cache_v),
            "prefix_out": _tensor_checksum(prefix_out),
            "prefix_mask_true": int(jax.device_get(jnp.sum(prefix_mask))),
        },
    }


def _load_pair_and_model(
    args: Args,
) -> tuple[Any, nnx.State, dict[str, Any], dict[str, Any], p3t_trainer.PairIndices]:
    trainer_args = p3t_trainer.Args(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        endpoint_student_params=args.endpoint_student_params,
        config_name=args.config_name,
        dataset_task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        seed=args.seed,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    arrays = multirate_dataset.load_multirate_arrays(
        args.dataset,
        fields=("anchor_index", "task_id", "episode_id", "frame_id", "fresh_ear"),
    )
    base_graphdef, base_state, observation_dataset, raw_dataset, model_metadata = (
        p3t_trainer._load_model_and_dataset(trainer_args)  # noqa: SLF001
    )
    pairs = p3t_trainer._select_pairs(  # noqa: SLF001
        arrays,
        raw_dataset,
        task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        maximum_pairs=200,
        seed=args.seed,
    )
    _, _, test_indices = p3t_trainer._split_pairs(  # noqa: SLF001
        pairs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    heldout_pairs = pairs.take(test_indices)
    if len(heldout_pairs) != args.expected_heldout_pairs:
        raise ValueError(
            f"Expected {args.expected_heldout_pairs} held-out Task8 pairs, got {len(heldout_pairs)}. "
            "Keep this command aligned with the P3T/MRR split."
        )
    selected_pair = heldout_pairs.take(np.asarray([args.pair_index], dtype=np.int64))
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
        selected_pair,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    batch = p3t_trainer._batch(records, np.asarray([0], dtype=np.int64))  # noqa: SLF001
    return base_graphdef, base_state, batch, model_metadata, selected_pair


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Benchmark summary already exists at {summary_path}; pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_graphdef, base_state, batch, model_metadata, selected_pair = _load_pair_and_model(args)
    prepare_anchor = _make_anchor_preparer(base_graphdef)
    full_fresh_prefix = _make_full_fresh_prefix(base_graphdef)
    active_264 = _make_active_replay(base_graphdef, include_dummy=False)
    active_520 = _make_active_replay(base_graphdef, include_dummy=True)

    anchor_prefix = _block(prepare_anchor(base_state, batch["anchor_observation"]))
    anchor_cache_k, anchor_cache_v = anchor_prefix["kv_cache"]
    anchor_dummy_tokens = anchor_prefix["prefix_tokens"][:, VALID_VISUAL_TOKENS : VALID_VISUAL_TOKENS + DUMMY_TOKENS]
    if anchor_cache_k.shape != (NUM_LAYERS, 1, PREFIX_TOKENS, 1, KV_HEAD_DIM):
        raise ValueError(f"Unexpected anchor K cache shape {anchor_cache_k.shape}.")
    if anchor_cache_v.shape != anchor_cache_k.shape:
        raise ValueError(f"Unexpected anchor V cache shape {anchor_cache_v.shape}.")
    if anchor_prefix["prefix_out"].shape[:2] != (1, PREFIX_TOKENS):
        raise ValueError(f"Unexpected anchor prefix output shape {anchor_prefix['prefix_out'].shape}.")
    anchor_dummy_image = np.asarray(
        jax.device_get(batch["anchor_observation"].images[DUMMY_IMAGE_KEY])
    )
    current_dummy_image = np.asarray(
        jax.device_get(batch["current_observation"].images[DUMMY_IMAGE_KEY])
    )
    dummy_inputs_invariant = bool(np.array_equal(anchor_dummy_image, current_dummy_image))
    dummy_masks_false = bool(
        not np.any(
            np.asarray(jax.device_get(batch["anchor_observation"].image_masks[DUMMY_IMAGE_KEY]))
        )
        and not np.any(
            np.asarray(jax.device_get(batch["current_observation"].image_masks[DUMMY_IMAGE_KEY]))
        )
    )
    if not dummy_inputs_invariant or not dummy_masks_false:
        raise ValueError(
            "A520 may reuse anchor dummy pre-Gemma tokens only when the anchor/current dummy inputs are "
            "identical and both image masks are false."
        )

    shared_active_args = (
        base_state,
        batch["current_observation"],
        anchor_cache_k,
        anchor_cache_v,
        anchor_prefix["prefix_out"],
        anchor_dummy_tokens,
    )
    functions: dict[str, tuple[Callable[..., Any], tuple[Any, ...]]] = {
        "full_fresh_prefix": (full_fresh_prefix, (base_state, batch["current_observation"])),
        "active_264": (active_264, shared_active_args),
        "active_520": (active_520, shared_active_args),
    }

    # Compile every graph before any timed loop so compilation is excluded and
    # no variant benefits from another variant's timed warmup.
    for name, (function, call_args) in functions.items():
        LOGGER.info("Compiling %s", name)
        _block(function(*call_args))

    timings: dict[str, dict[str, float | int]] = {}
    outputs: dict[str, tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = {}
    for name, (function, call_args) in functions.items():
        LOGGER.info("Benchmarking %s: warmup=%d iterations=%d", name, args.warmup, args.iterations)
        timing, output = _time_callable(
            function,
            call_args,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        timings[name] = timing
        outputs[name] = output

    # Validate and reduce outputs only after all timed loops, so first-time
    # checksum compilation cannot perturb a later timing distribution.
    results: dict[str, Any] = {
        name: {
            "timing": timings[name],
            "composite": _summarize_composite(outputs[name]),
        }
        for name in functions
    }

    full_p50 = float(results["full_fresh_prefix"]["timing"]["p50_ms"])
    full_p95 = float(results["full_fresh_prefix"]["timing"]["p95_ms"])
    results["full_fresh_prefix"]["physical_queries"] = PREFIX_TOKENS
    results["full_fresh_prefix"]["physical_attention_keys"] = PREFIX_TOKENS
    for name, query_count in (("active_264", 264), ("active_520", 520)):
        results[name]["physical_queries"] = query_count
        results[name]["physical_attention_keys"] = PREFIX_TOKENS
        results[name]["p50_speedup_vs_full"] = full_p50 / float(results[name]["timing"]["p50_ms"])
        results[name]["p95_speedup_vs_full"] = full_p95 / float(results[name]["timing"]["p95_ms"])

    full_output = outputs["full_fresh_prefix"]
    active_264_output = outputs["active_264"]
    active_520_output = outputs["active_520"]
    prefix_masks_match = bool(
        np.array_equal(np.asarray(jax.device_get(full_output[3])), np.asarray(jax.device_get(active_264_output[3])))
        and np.array_equal(
            np.asarray(jax.device_get(full_output[3])),
            np.asarray(jax.device_get(active_520_output[3])),
        )
    )
    if not prefix_masks_match:
        raise ValueError("Full and active paths produced different current prefix masks.")
    anchor_current_prefix_masks_match = bool(
        np.array_equal(
            np.asarray(jax.device_get(anchor_prefix["prefix_mask"])),
            np.asarray(jax.device_get(full_output[3])),
        )
    )
    if not anchor_current_prefix_masks_match:
        raise ValueError("Selected Task8 anchor/current prefix masks differ; anchor KV positions are not reusable.")

    summary = {
        "method": "MRR fixed-shape active-query Gemma replay microbenchmark",
        "status": "gpu_microbenchmark_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "pair": {
            "partition": "P3T episode-disjoint Task8 test",
            "pair_index": args.pair_index,
            "anchor_index": int(selected_pair.anchor_indices[0]),
            "target_index": int(selected_pair.target_indices[0]),
            "episode_id": int(selected_pair.episode_ids[0]),
            "anchor_frame_id": int(selected_pair.frame_ids[0]),
            "temporal_stride": args.temporal_stride,
        },
        "fixed_query_definition": {
            "prefix_tokens": PREFIX_TOKENS,
            "valid_visual_tokens": VALID_VISUAL_TOKENS,
            "dummy_tokens": DUMMY_TOKENS,
            "language_slots": LANGUAGE_TOKENS,
            "fixed_visual_queries": FIXED_VISUAL_QUERIES,
            "fixed_visual_indices": FIXED_VISUAL_INDICES.tolist(),
            "active_264_indices": ACTIVE_INDICES_264.tolist(),
            "active_520_indices": ACTIVE_INDICES_520.tolist(),
            "active_264_inactive_anchor_tokens": PREFIX_TOKENS - 264,
            "active_520_inactive_anchor_tokens": PREFIX_TOKENS - 520,
            "selection": "static spatial grid: rows stride 2 and columns stride 4 in each 16x16 valid-view grid",
            "dynamic_top_k": False,
        },
        "replay_semantics": {
            "gemma_layers_replayed": NUM_LAYERS,
            "original_rope_positions_preserved": True,
            "anchor_active_kv_copies_removed_before_append": True,
            "new_active_kv_appended_then_scattered": True,
            "dense_968_query_surrogate": False,
            "prefix_masks_match_full": prefix_masks_match,
            "anchor_current_prefix_masks_match": anchor_current_prefix_masks_match,
            "dummy_inputs_invariant": dummy_inputs_invariant,
            "dummy_masks_false": dummy_masks_false,
            "active_264_attention_shape": [1, 1, 264, PREFIX_TOKENS],
            "active_520_attention_shape": [1, 1, 520, PREFIX_TOKENS],
        },
        "timing_scope": {
            "batch_size": 1,
            "synchronization": "jax.block_until_ready on every complete output pytree",
            "compilation": "all three graphs compiled before timed warmups",
            "full_fresh_prefix": (
                "existing preprocess + three-view SigLIP + language embedding + full 968-query 18-layer Gemma"
            ),
            "active_replay": (
                "existing preprocess + two valid current-view SigLIP + current language embedding + fixed-shape "
                "inactive-cache gather + 18-layer append-mask active replay + scatter to full 968-token cache"
            ),
            "active_520_dummy_embedding": (
                "reuses the exact invariant anchor pre-Gemma embedding for the zero/masked dummy image, then "
                "physically replays its 256 Gemma queries"
            ),
            "excluded": "IAR, EAR, final action suffix, policy/RPC, dynamic selector, data loading, and JIT compilation",
        },
        "results": results,
        "output": str(summary_path),
        "note": (
            "The existing Gemma append API is used without changing model code.  Fixed inactive-cache gather "
            "removes stale duplicates first, so both active variants attend exactly 968 physical keys."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
