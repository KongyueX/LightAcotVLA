"""GPU feasibility oracle for block-sparse MRR-ACoT prefix refresh.

The probe uses the episode-disjoint Task8 held-out pairs from the P3T pilot.
For each anchor/current pair it computes two frozen VLM prefix caches with the
same flow noise, partitions the 512 valid visual tokens into 32 blocks (two
views, sixteen 4x4-token blocks per view), and evaluates three top-4 selectors:

* low-resolution RGB difference;
* normalized full-fresh KV delta magnitude; and
* a causal oracle ranked by one-block action plus EAR gap improvement.

The 32 causal one-block interventions are evaluated as one batch.  To expose
fresh-language leakage, every selected visual set is evaluated both with the
anchor language cache and with the fresh language cache, while a separate
fresh-language-only baseline refreshes no visual block.  Dummy image tokens
always remain from the anchor for every sparse composite.  This script never
changes the default policy/evaluation path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.action_cot import multirate_dataset

try:
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("probe_mrr_acot_sparse_refresh_oracle")

NUM_LAYERS = 18
PREFIX_TOKENS = 968
VISUAL_TOKENS = 512
DUMMY_TOKENS = 256
LANGUAGE_TOKENS = 200
KV_DIM = 256
NUM_VIEWS = 2
TOKEN_GRID = 16
BLOCK_EDGE = 4
BLOCKS_PER_VIEW = (TOKEN_GRID // BLOCK_EDGE) ** 2
NUM_BLOCKS = NUM_VIEWS * BLOCKS_PER_VIEW
TOKENS_PER_BLOCK = BLOCK_EDGE**2
TOP_K = 4


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    endpoint_student_params: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    dataset_task_id: int = 6
    seed: int = 7
    split_seed: int = 7
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    expected_heldout_pairs: int = 22
    coarse_flow_steps: int = 1
    final_flow_steps: int = 1
    delta_floor: float = 1e-6
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.dataset_task_id < 0 or args.seed < 0 or args.split_seed < 0:
        raise ValueError("Task id and seeds must be non-negative.")
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("Validation/test fractions must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    if args.expected_heldout_pairs <= 0:
        raise ValueError("expected_heldout_pairs must be positive.")
    if args.coarse_flow_steps <= 0 or args.final_flow_steps <= 0:
        raise ValueError("Flow step counts must be positive.")
    if args.delta_floor <= 0.0:
        raise ValueError("delta_floor must be positive.")


def _block_mask() -> np.ndarray:
    mask = np.zeros((NUM_BLOCKS, VISUAL_TOKENS), dtype=np.bool_)
    block_id = 0
    for view in range(NUM_VIEWS):
        view_offset = view * TOKEN_GRID * TOKEN_GRID
        for block_row in range(TOKEN_GRID // BLOCK_EDGE):
            for block_col in range(TOKEN_GRID // BLOCK_EDGE):
                for row in range(block_row * BLOCK_EDGE, (block_row + 1) * BLOCK_EDGE):
                    start = view_offset + row * TOKEN_GRID + block_col * BLOCK_EDGE
                    mask[block_id, start : start + BLOCK_EDGE] = True
                block_id += 1
    if block_id != NUM_BLOCKS or not np.all(np.sum(mask, axis=1) == TOKENS_PER_BLOCK):
        raise AssertionError("Invalid fixed visual block partition.")
    if not np.all(np.sum(mask, axis=0) == 1):
        raise AssertionError("Visual blocks must partition every valid visual token exactly once.")
    return mask


BLOCK_MASK = jnp.asarray(_block_mask())


def _repeat_batch(value: jax.Array, repeats: int) -> jax.Array:
    return jnp.repeat(value, repeats, axis=0)


def _repeat_prefix_state(
    current_prefix: dict[str, Any],
    anchor_prefix_out: jax.Array,
    kv_cache: tuple[jax.Array, jax.Array],
    repeats: int,
) -> dict[str, Any]:
    return {
        "observation": jax.tree.map(lambda value: _repeat_batch(value, repeats), current_prefix["observation"]),
        "prefix_tokens": _repeat_batch(current_prefix["prefix_tokens"], repeats),
        "prefix_mask": _repeat_batch(current_prefix["prefix_mask"], repeats),
        "prefix_ar_mask": current_prefix["prefix_ar_mask"],
        "prefix_out": _repeat_batch(anchor_prefix_out, repeats),
        "kv_cache": kv_cache,
        "ref_action_noise": _repeat_batch(current_prefix["ref_action_noise"], repeats),
        "expert_action_noise": _repeat_batch(current_prefix["expert_action_noise"], repeats),
    }


def _composite_cache(
    anchor_kv: tuple[jax.Array, jax.Array],
    fresh_kv: tuple[jax.Array, jax.Array],
    visual_masks: jax.Array,
    fresh_language: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Build B sparse composites; dummy slots are always copied from anchor."""

    batch = visual_masks.shape[0]
    language_start = VISUAL_TOKENS + DUMMY_TOKENS
    visual_selector = visual_masks[None, :, :, None, None]
    language_selector = fresh_language[None, :, None, None, None]

    def combine(anchor: jax.Array, fresh: jax.Array) -> jax.Array:
        anchor_visual = jnp.repeat(anchor[:, :1, :VISUAL_TOKENS], batch, axis=1)
        fresh_visual = jnp.repeat(fresh[:, :1, :VISUAL_TOKENS], batch, axis=1)
        visual = jnp.where(visual_selector, fresh_visual, anchor_visual)
        dummy = jnp.repeat(anchor[:, :1, VISUAL_TOKENS:language_start], batch, axis=1)
        anchor_language = jnp.repeat(anchor[:, :1, language_start:], batch, axis=1)
        current_language = jnp.repeat(fresh[:, :1, language_start:], batch, axis=1)
        language = jnp.where(language_selector, current_language, anchor_language)
        return jnp.concatenate([visual, dummy, language], axis=2)

    return combine(anchor_kv[0], fresh_kv[0]), combine(anchor_kv[1], fresh_kv[1])


def _selected_visual_mask(block_ids: jax.Array) -> jax.Array:
    return jnp.any(BLOCK_MASK[block_ids], axis=0)


def _rgb_block_scores(anchor_images: jax.Array, current_images: jax.Array) -> jax.Array:
    """Return mean squared RGB delta per 4x4-token block."""

    def to_tokens(images: jax.Array) -> jax.Array:
        # 64x64 -> 16x16 patch grid, then flatten the two views in prefix order.
        return images.reshape(1, NUM_VIEWS, TOKEN_GRID, 4, TOKEN_GRID, 4, 3).mean(axis=(3, 5)).reshape(
            1, VISUAL_TOKENS, 3
        )

    delta = jnp.mean(jnp.square(to_tokens(current_images) - to_tokens(anchor_images)), axis=-1)[0]
    return jnp.sum(BLOCK_MASK.astype(delta.dtype) * delta[None, :], axis=1) / TOKENS_PER_BLOCK


def _kv_delta_block_scores(
    anchor_kv: tuple[jax.Array, jax.Array],
    fresh_kv: tuple[jax.Array, jax.Array],
    *,
    delta_floor: float,
) -> jax.Array:
    """Layer-normalized K/V delta energy per visual block."""

    def token_power(anchor: jax.Array, fresh: jax.Array) -> jax.Array:
        delta = fresh[:, 0, :VISUAL_TOKENS, 0].astype(jnp.float32) - anchor[:, 0, :VISUAL_TOKENS, 0].astype(
            jnp.float32
        )
        power = jnp.mean(jnp.square(delta), axis=-1)  # [L,T]
        normalized = power / jax.lax.stop_gradient(jnp.mean(power, axis=1, keepdims=True) + delta_floor)
        return jnp.mean(normalized, axis=0)

    score_by_token = 0.5 * (
        token_power(anchor_kv[0], fresh_kv[0]) + token_power(anchor_kv[1], fresh_kv[1])
    )
    return jnp.sum(BLOCK_MASK.astype(score_by_token.dtype) * score_by_token[None, :], axis=1) / TOKENS_PER_BLOCK


def _mse_per_example(predicted: jax.Array, target: jax.Array, *, last_dims: int | None = None) -> jax.Array:
    if last_dims is not None:
        predicted = predicted[..., :last_dims]
        target = target[..., :last_dims]
    return jnp.mean(
        jnp.square(predicted.astype(jnp.float32) - target.astype(jnp.float32)),
        axis=tuple(range(1, predicted.ndim)),
    )


def _downstream_metrics(
    iar: jax.Array,
    ear: jax.Array,
    actions: jax.Array,
    target_iar: jax.Array,
    target_ear: jax.Array,
    target_actions: jax.Array,
) -> jax.Array:
    action_mse = _mse_per_example(actions, target_actions, last_dims=7)
    ear_mse = _mse_per_example(ear, target_ear, last_dims=7)
    iar_mse = _mse_per_example(iar, target_iar)
    gripper_sign = jnp.mean(
        (actions[..., 6] >= 0.0) == (target_actions[..., 6] >= 0.0),
        axis=tuple(range(1, actions[..., 6].ndim)),
    )
    return jnp.stack([action_mse, ear_mse, iar_mse, gripper_sign], axis=-1)


def _make_pair_evaluator(base_graphdef: Any, args: Args):
    @jax.jit
    def evaluate_pair(base_state: nnx.State, batch: dict[str, Any], rng: jax.Array) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        # Identical RNG keys guarantee identical coarse/final flow noise.
        anchor_prefix = base_model.sample_actions_profile_prefix(rng, batch["anchor_observation"])
        fresh_prefix = base_model.sample_actions_profile_prefix(rng, batch["current_observation"])
        anchor_kv = anchor_prefix["kv_cache"]
        fresh_kv = fresh_prefix["kv_cache"]
        anchor_out = anchor_prefix["prefix_out"]

        # Baselines: stale, exact fresh, and the explicit language-leakage control.
        baseline_masks = jnp.zeros((3, VISUAL_TOKENS), dtype=jnp.bool_)
        baseline_languages = jnp.asarray([False, False, True], dtype=jnp.bool_)
        baseline_kv = _composite_cache(anchor_kv, fresh_kv, baseline_masks, baseline_languages)
        # Replace row one by the exact fresh cache; the sparse cache constructor
        # intentionally keeps dummy anchor tokens for every other row.
        baseline_kv = (
            baseline_kv[0].at[:, 1].set(fresh_kv[0][:, 0]),
            baseline_kv[1].at[:, 1].set(fresh_kv[1][:, 0]),
        )
        baseline_prefix = _repeat_prefix_state(fresh_prefix, anchor_out, baseline_kv, 3)
        baseline_iar, baseline_ear, baseline_actions = p3t_trainer._reason_and_act(  # noqa: SLF001
            base_model,
            baseline_prefix,
            coarse_flow_steps=args.coarse_flow_steps,
            final_flow_steps=args.final_flow_steps,
        )
        target_iar = baseline_iar[1:2]
        target_ear = baseline_ear[1:2]
        target_actions = baseline_actions[1:2]
        baseline_metrics = _downstream_metrics(
            baseline_iar,
            baseline_ear,
            baseline_actions,
            target_iar,
            target_ear,
            target_actions,
        )

        anchor_images = p3t_trainer._low_resolution_images(anchor_prefix, 64)  # noqa: SLF001
        current_images = p3t_trainer._low_resolution_images(fresh_prefix, 64)  # noqa: SLF001
        rgb_scores = _rgb_block_scores(anchor_images, current_images)
        kv_scores = _kv_delta_block_scores(anchor_kv, fresh_kv, delta_floor=args.delta_floor)

        # Causal block ranking is computed with anchor language.  This prevents
        # the oracle ranking itself from benefiting from full-fresh language KV.
        single_block_kv = _composite_cache(
            anchor_kv,
            fresh_kv,
            BLOCK_MASK,
            jnp.zeros((NUM_BLOCKS,), dtype=jnp.bool_),
        )
        single_prefix = _repeat_prefix_state(fresh_prefix, anchor_out, single_block_kv, NUM_BLOCKS)
        single_iar, single_ear, single_actions = p3t_trainer._reason_and_act(  # noqa: SLF001
            base_model,
            single_prefix,
            coarse_flow_steps=args.coarse_flow_steps,
            final_flow_steps=args.final_flow_steps,
        )
        single_metrics = _downstream_metrics(
            single_iar,
            single_ear,
            single_actions,
            target_iar,
            target_ear,
            target_actions,
        )
        stale_action_gap = jnp.maximum(baseline_metrics[0, 0], args.delta_floor)
        stale_ear_gap = jnp.maximum(baseline_metrics[0, 1], args.delta_floor)
        causal_scores = 0.5 * (
            (baseline_metrics[0, 0] - single_metrics[:, 0]) / stale_action_gap
            + (baseline_metrics[0, 1] - single_metrics[:, 1]) / stale_ear_gap
        )

        _, rgb_top4 = jax.lax.top_k(rgb_scores, TOP_K)
        _, kv_top4 = jax.lax.top_k(kv_scores, TOP_K)
        _, causal_top4 = jax.lax.top_k(causal_scores, TOP_K)
        selected_ids = jnp.stack([rgb_top4, kv_top4, causal_top4])
        selected_masks = jnp.stack([_selected_visual_mask(ids) for ids in selected_ids])

        # Six composites in one downstream batch:
        # [RGB anchor/fresh language, KV anchor/fresh, causal anchor/fresh].
        selected_masks = jnp.repeat(selected_masks, 2, axis=0)
        selected_languages = jnp.tile(jnp.asarray([False, True], dtype=jnp.bool_), 3)
        selected_kv = _composite_cache(anchor_kv, fresh_kv, selected_masks, selected_languages)
        selected_prefix = _repeat_prefix_state(fresh_prefix, anchor_out, selected_kv, 6)
        selected_iar, selected_ear, selected_actions = p3t_trainer._reason_and_act(  # noqa: SLF001
            base_model,
            selected_prefix,
            coarse_flow_steps=args.coarse_flow_steps,
            final_flow_steps=args.final_flow_steps,
        )
        selected_metrics = _downstream_metrics(
            selected_iar,
            selected_ear,
            selected_actions,
            target_iar,
            target_ear,
            target_actions,
        ).reshape(3, 2, 4)
        return {
            "baseline_metrics": baseline_metrics,
            "selected_metrics": selected_metrics,
            "selected_ids": selected_ids,
            "selector_scores": jnp.stack([rgb_scores, kv_scores, causal_scores]),
            "single_block_metrics": single_metrics,
        }

    return evaluate_pair


METRIC_NAMES = ("action_mse_7d", "ear_mse_7d", "iar_mse", "gripper_sign_accuracy")
SELECTOR_NAMES = ("rgb_diff_top4", "kv_delta_norm_top4", "causal_action_ear_top4")
LANGUAGE_VARIANTS = ("anchor_language_selected_visual", "fresh_language_selected_visual")
BASELINE_NAMES = ("stale", "fresh", "fresh_language_only")


def _metric_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(METRIC_NAMES, values, strict=True)}


def _closure(candidate: np.ndarray, stale: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for index, name in enumerate(METRIC_NAMES[:3]):
        denominator = float(stale[index])
        result[name] = None if denominator <= 1e-12 else float(1.0 - candidate[index] / denominator)
    return result


def _block_name(block_id: int) -> str:
    view = block_id // BLOCKS_PER_VIEW
    local = block_id % BLOCKS_PER_VIEW
    return f"view{view}_block_r{local // 4}_c{local % 4}"


def _aggregate(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    stale = np.stack([np.asarray(record["baseline_arrays"]["stale"]) for record in per_pair])
    summary: dict[str, Any] = {
        "num_pairs": len(per_pair),
        "baselines": {},
        "selectors": {},
    }
    for baseline_name in BASELINE_NAMES:
        values = np.stack([np.asarray(record["baseline_arrays"][baseline_name]) for record in per_pair])
        mean_values = np.mean(values, axis=0)
        summary["baselines"][baseline_name] = {
            "metrics": _metric_dict(mean_values),
            "global_gap_closure_vs_stale": _closure(mean_values, np.mean(stale, axis=0)),
            "mean_pair_gap_closure_vs_stale": {
                metric_name: float(np.mean(1.0 - values[:, metric_index] / np.maximum(stale[:, metric_index], 1e-12)))
                for metric_index, metric_name in enumerate(METRIC_NAMES[:3])
            },
        }

    language_only = np.stack(
        [np.asarray(record["baseline_arrays"]["fresh_language_only"]) for record in per_pair]
    )
    stale_mean = np.mean(stale, axis=0)
    language_only_mean = np.mean(language_only, axis=0)
    for selector_name in SELECTOR_NAMES:
        selector_summary: dict[str, Any] = {}
        variant_values: dict[str, np.ndarray] = {}
        for variant_name in LANGUAGE_VARIANTS:
            values = np.stack(
                [np.asarray(record["selected_arrays"][selector_name][variant_name]) for record in per_pair]
            )
            variant_values[variant_name] = values
            mean_values = np.mean(values, axis=0)
            selector_summary[variant_name] = {
                "metrics": _metric_dict(mean_values),
                "global_gap_closure_vs_stale": _closure(mean_values, stale_mean),
                "mean_pair_gap_closure_vs_stale": {
                    metric_name: float(
                        np.mean(1.0 - values[:, metric_index] / np.maximum(stale[:, metric_index], 1e-12))
                    )
                    for metric_index, metric_name in enumerate(METRIC_NAMES[:3])
                },
            }
        fresh_selected = variant_values["fresh_language_selected_visual"]
        fresh_selected_mean = np.mean(fresh_selected, axis=0)
        selector_summary["fresh_visual_increment_beyond_fresh_language_only"] = {
            "global_stale_gap_units": {
                metric_name: float(
                    (language_only_mean[metric_index] - fresh_selected_mean[metric_index])
                    / max(stale_mean[metric_index], 1e-12)
                )
                for metric_index, metric_name in enumerate(METRIC_NAMES[:3])
            },
            "closure_of_language_only_residual_gap": {
                metric_name: (
                    None
                    if language_only_mean[metric_index] <= 1e-12
                    else float(1.0 - fresh_selected_mean[metric_index] / language_only_mean[metric_index])
                )
                for metric_index, metric_name in enumerate(METRIC_NAMES[:3])
            },
        }
        selected = [set(record["selected_block_ids"][selector_name]) for record in per_pair]
        counts = np.zeros((NUM_BLOCKS,), dtype=np.int64)
        for ids in selected:
            for block_id in ids:
                counts[block_id] += 1
        selector_summary["block_selection_frequency"] = {
            _block_name(int(block_id)): int(count)
            for block_id, count in enumerate(counts)
            if count
        }
        summary["selectors"][selector_name] = selector_summary
    return summary


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    per_pair_path = output_dir / "per_pair.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (per_pair_path.exists() or summary_path.exists()):
        raise FileExistsError(f"MRR oracle output already exists in {output_dir}; pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = p3t_trainer.Args(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        endpoint_student_params=args.endpoint_student_params,
        config_name=args.config_name,
        dataset_task_id=args.dataset_task_id,
        temporal_stride=10,
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
        temporal_stride=10,
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
            "Keep the P3T split settings aligned."
        )
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
        heldout_pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=10,
    )
    evaluate_pair = _make_pair_evaluator(base_graphdef, args)
    per_pair: list[dict[str, Any]] = []
    started = time.monotonic()
    with per_pair_path.open("w", encoding="utf-8") as output_file:
        for index, pair in enumerate(
            zip(
                heldout_pairs.anchor_indices,
                heldout_pairs.target_indices,
                heldout_pairs.episode_ids,
                heldout_pairs.frame_ids,
                strict=True,
            )
        ):
            anchor, target, episode, frame = (int(value) for value in pair)
            batch = p3t_trainer._batch(records, np.asarray([index], dtype=np.int64))  # noqa: SLF001
            key = jax.random.fold_in(jax.random.key(args.seed), anchor)
            result = jax.device_get(evaluate_pair(base_state, batch, key))
            baseline_metrics = np.asarray(result["baseline_metrics"], dtype=np.float64)
            selected_metrics = np.asarray(result["selected_metrics"], dtype=np.float64)
            selected_ids = np.asarray(result["selected_ids"], dtype=np.int64)
            selector_scores = np.asarray(result["selector_scores"], dtype=np.float64)
            record: dict[str, Any] = {
                "pair_index": index,
                "anchor_index": anchor,
                "target_index": target,
                "episode_id": episode,
                "frame_id": frame,
                "baseline_arrays": {
                    name: baseline_metrics[position].tolist()
                    for position, name in enumerate(BASELINE_NAMES)
                },
                "baselines": {
                    name: _metric_dict(baseline_metrics[position])
                    for position, name in enumerate(BASELINE_NAMES)
                },
                "selected_arrays": {},
                "selectors": {},
                "selected_block_ids": {},
            }
            stale = baseline_metrics[0]
            language_only = baseline_metrics[2]
            for selector_index, selector_name in enumerate(SELECTOR_NAMES):
                ids = [int(value) for value in selected_ids[selector_index]]
                record["selected_block_ids"][selector_name] = ids
                record["selected_arrays"][selector_name] = {}
                record["selectors"][selector_name] = {
                    "selected_blocks": [
                        {
                            "id": block_id,
                            "name": _block_name(block_id),
                            "score": float(selector_scores[selector_index, block_id]),
                        }
                        for block_id in ids
                    ],
                    "variants": {},
                }
                for language_index, variant_name in enumerate(LANGUAGE_VARIANTS):
                    values = selected_metrics[selector_index, language_index]
                    record["selected_arrays"][selector_name][variant_name] = values.tolist()
                    record["selectors"][selector_name]["variants"][variant_name] = {
                        "metrics": _metric_dict(values),
                        "gap_closure_vs_stale": _closure(values, stale),
                    }
                fresh_selected = selected_metrics[selector_index, 1]
                record["selectors"][selector_name]["fresh_visual_increment_beyond_fresh_language_only"] = {
                    metric_name: float(
                        (language_only[metric_index] - fresh_selected[metric_index])
                        / max(stale[metric_index], 1e-12)
                    )
                    for metric_index, metric_name in enumerate(METRIC_NAMES[:3])
                }
            output_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output_file.flush()
            per_pair.append(record)
            LOGGER.info(
                "MRR sparse oracle pair %d/%d anchor=%d episode=%d elapsed=%.1fs",
                index + 1,
                len(records),
                anchor,
                episode,
                time.monotonic() - started,
            )

    summary = {
        "method": "MRR-ACoT block-sparse refresh oracle",
        "status": "offline_oracle_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "heldout": {
            "partition": "P3T episode-disjoint test",
            "pairs": len(heldout_pairs),
            "episodes": sorted(int(value) for value in np.unique(heldout_pairs.episode_ids)),
            "task": "Task8",
            "dataset_task_id": args.dataset_task_id,
        },
        "block_definition": {
            "valid_visual_tokens": VISUAL_TOKENS,
            "views": NUM_VIEWS,
            "token_grid_per_view": [TOKEN_GRID, TOKEN_GRID],
            "block_grid_per_view": [TOKEN_GRID // BLOCK_EDGE, TOKEN_GRID // BLOCK_EDGE],
            "tokens_per_block": TOKENS_PER_BLOCK,
            "blocks_per_view": BLOCKS_PER_VIEW,
            "total_blocks": NUM_BLOCKS,
            "selected_blocks": TOP_K,
            "selected_visual_token_fraction": TOP_K * TOKENS_PER_BLOCK / VISUAL_TOKENS,
            "dummy_policy": "always anchor",
        },
        "leakage_protocol": {
            "anchor_language_selected_visual": "anchor language plus selected fresh visual blocks",
            "fresh_language_selected_visual": "fresh language plus the same selected fresh visual blocks",
            "fresh_language_only": "fresh language, zero refreshed visual blocks",
            "causal_selector_language": "anchor language",
        },
        "results": _aggregate(per_pair),
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {"per_pair": str(per_pair_path), "summary": str(summary_path)},
        "note": "Oracle/open-loop gap closure only; this is not a Task8 success-rate result.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
