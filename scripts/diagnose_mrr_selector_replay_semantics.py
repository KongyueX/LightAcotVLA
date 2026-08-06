"""Diagnose learned MRR selector semantics on the fixed Task8 test22 split.

For every held-out pair and one shared policy RNG, this standalone GPU probe
compares five frozen downstream prefixes:

1. stale anchor KV at the current observation;
2. direct deep-KV splice for the learned top-4 visual blocks;
3. learned top-4 A264 physical active-query Gemma replay;
4. learned top-4 A520 replay including all masked dummy queries; and
5. exact full-fresh target.

The A264 path calls the opt-in online model entrypoints.  A520 mirrors the
existing benchmark's inactive-gather, physical append, and scatter semantics,
but uses learned dynamic blocks instead of the benchmark's fixed visual set.
Neither path constructs a dense 968-query masked surrogate.  Default policy
inference is untouched.
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
from openpi.models import acot_vla as acot_model
from openpi.models import mrr_block_selector

try:
    import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("diagnose_mrr_selector_replay_semantics")
METHOD_NAMES = ("stale", "direct_splice", "active_a264", "active_a520", "fresh")
NUM_LAYERS = 18
PREFIX_TOKENS = 968
VALID_VISUAL_TOKENS = 512
DUMMY_TOKENS = 256
LANGUAGE_TOKENS = 200
SELECTED_VISUAL_TOKENS = 64
ACTIVE_A264 = SELECTED_VISUAL_TOKENS + LANGUAGE_TOKENS
ACTIVE_A520 = SELECTED_VISUAL_TOKENS + DUMMY_TOKENS + LANGUAGE_TOKENS
INACTIVE_A520 = PREFIX_TOKENS - ACTIVE_A520
DUMMY_IMAGE_KEY = "right_wrist_0_rgb"


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    endpoint_student_params: str
    selector_checkpoint: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    dataset_task_id: int = 6
    temporal_stride: int = 10
    seed: int = 7
    split_seed: int = 7
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    expected_pairs: int = 188
    expected_train_pairs: int = 144
    expected_validation_pairs: int = 22
    expected_test_pairs: int = 22
    coarse_flow_steps: int = 1
    final_flow_steps: int = 1
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if any(value < 0 for value in (args.dataset_task_id, args.seed, args.split_seed)):
        raise ValueError("Task id and seeds must be non-negative.")
    if args.temporal_stride != 10:
        raise ValueError("MRR semantic diagnosis requires the fixed anchor-to-anchor+10 protocol.")
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("Validation/test fractions must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    expected = (
        args.expected_pairs,
        args.expected_train_pairs,
        args.expected_validation_pairs,
        args.expected_test_pairs,
    )
    if any(value <= 0 for value in expected):
        raise ValueError("Expected split sizes must be positive.")
    if sum(expected[1:]) != expected[0]:
        raise ValueError("Expected train/validation/test sizes must sum to expected_pairs.")
    if args.coarse_flow_steps <= 0 or args.final_flow_steps <= 0:
        raise ValueError("Flow step counts must be positive.")


def _stack_prefix_states(states: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Stack equal-batch prefix states into one downstream comparison batch."""

    first = states[0]
    if any(state["prefix_ar_mask"].shape != first["prefix_ar_mask"].shape for state in states[1:]):
        raise ValueError("MRR comparison prefixes have incompatible prefix_ar_mask shapes.")
    return {
        "observation": jax.tree.map(
            lambda *values: jnp.concatenate(values, axis=0),
            *(state["observation"] for state in states),
        ),
        "prefix_tokens": jnp.concatenate([state["prefix_tokens"] for state in states], axis=0),
        "prefix_mask": jnp.concatenate([state["prefix_mask"] for state in states], axis=0),
        "prefix_ar_mask": first["prefix_ar_mask"],
        "prefix_out": jnp.concatenate([state["prefix_out"] for state in states], axis=0),
        "kv_cache": (
            jnp.concatenate([state["kv_cache"][0] for state in states], axis=1),
            jnp.concatenate([state["kv_cache"][1] for state in states], axis=1),
        ),
        "ref_action_noise": jnp.concatenate([state["ref_action_noise"] for state in states], axis=0),
        "expert_action_noise": jnp.concatenate(
            [state["expert_action_noise"] for state in states],
            axis=0,
        ),
    }


def _direct_splice_prefix(
    anchor_prefix: dict[str, Any],
    fresh_prefix: dict[str, Any],
    selected_block_ids: jax.Array,
) -> dict[str, Any]:
    visual_mask = mrr_oracle._selected_visual_mask(selected_block_ids[0])  # noqa: SLF001
    composite_kv = mrr_oracle._composite_cache(  # noqa: SLF001
        anchor_prefix["kv_cache"],
        fresh_prefix["kv_cache"],
        visual_mask[None, :],
        jnp.zeros((1,), dtype=jnp.bool_),
    )
    result = dict(fresh_prefix)
    result["kv_cache"] = composite_kv
    # This is the exact anchor-language direct-splice semantics used by the
    # learned-selector test22 baseline (the reported 67%/62% reference).
    result["prefix_out"] = anchor_prefix["prefix_out"]
    return result


def _stale_prefix(
    anchor_prefix: dict[str, Any],
    fresh_prefix: dict[str, Any],
) -> dict[str, Any]:
    result = dict(fresh_prefix)
    result["kv_cache"] = anchor_prefix["kv_cache"]
    result["prefix_out"] = anchor_prefix["prefix_out"]
    return result


def _active_a520_replay(
    base_model: Any,
    anchor_prefix: dict[str, Any],
    current: dict[str, Any],
    selected_block_ids: jax.Array,
) -> dict[str, Any]:
    """Physical learned-block A520 replay, matching benchmark append semantics."""

    batch = selected_block_ids.shape[0]
    selected_visual_indices = acot_model._mrr_block_token_indices(selected_block_ids)  # noqa: SLF001
    if selected_visual_indices.shape != (batch, SELECTED_VISUAL_TOKENS):
        raise ValueError(
            "Learned MRR blocks must expand to [B,64] visual indices, got "
            f"{selected_visual_indices.shape}."
        )
    selected_block_mask = jnp.zeros(
        (batch, mrr_block_selector.NUM_BLOCKS),
        dtype=jnp.bool_,
    ).at[jnp.arange(batch)[:, None], selected_block_ids].set(True)
    _, inactive_block_ids = jax.lax.top_k(
        (~selected_block_mask).astype(jnp.int32),
        mrr_block_selector.NUM_BLOCKS - mrr_block_selector.TOP_K,
    )
    inactive_visual_indices = acot_model._mrr_block_token_indices(inactive_block_ids)  # noqa: SLF001
    if inactive_visual_indices.shape != (batch, INACTIVE_A520):
        raise ValueError(f"A520 expected [B,{INACTIVE_A520}] inactive visual ids.")

    dummy_indices = jnp.broadcast_to(
        jnp.arange(
            VALID_VISUAL_TOKENS,
            VALID_VISUAL_TOKENS + DUMMY_TOKENS,
            dtype=jnp.int32,
        )[None],
        (batch, DUMMY_TOKENS),
    )
    language_indices = jnp.broadcast_to(
        jnp.arange(
            VALID_VISUAL_TOKENS + DUMMY_TOKENS,
            PREFIX_TOKENS,
            dtype=jnp.int32,
        )[None],
        (batch, LANGUAGE_TOKENS),
    )
    active_indices = jnp.concatenate(
        [selected_visual_indices, dummy_indices, language_indices],
        axis=1,
    )
    if active_indices.shape != (batch, ACTIVE_A520):
        raise ValueError(f"A520 requires exactly {ACTIVE_A520} physical active queries.")

    selected_visual_tokens = acot_model._mrr_take_tokens(  # noqa: SLF001
        current["visual_tokens"],
        selected_visual_indices,
    )
    anchor_dummy_tokens = anchor_prefix["prefix_tokens"][
        :, VALID_VISUAL_TOKENS : VALID_VISUAL_TOKENS + DUMMY_TOKENS
    ]
    active_tokens = jnp.concatenate(
        [selected_visual_tokens, anchor_dummy_tokens, current["language_tokens"]],
        axis=1,
    )
    active_mask = acot_model._mrr_take_tokens(current["prefix_mask"], active_indices)  # noqa: SLF001
    active_positions = acot_model._mrr_take_tokens(current["positions"], active_indices)  # noqa: SLF001
    inactive_key_mask = acot_model._mrr_take_tokens(  # noqa: SLF001
        current["prefix_mask"],
        inactive_visual_indices,
    )
    replay_mask = jnp.logical_and(
        active_mask[:, :, None],
        jnp.concatenate([inactive_key_mask, active_mask], axis=1)[:, None, :],
    )
    anchor_k, anchor_v = anchor_prefix["kv_cache"]
    inactive_anchor_k = acot_model._mrr_take_cache(anchor_k, inactive_visual_indices)  # noqa: SLF001
    inactive_anchor_v = acot_model._mrr_take_cache(anchor_v, inactive_visual_indices)  # noqa: SLF001
    (active_prefix_out, _, _), appended_cache = base_model.PaliGemma.llm(
        [active_tokens, None, None],
        positions=active_positions,
        mask=replay_mask,
        kv_cache=(inactive_anchor_k, inactive_anchor_v),
    )
    appended_k, appended_v = appended_cache
    if appended_k.shape[:3] != (NUM_LAYERS, batch, PREFIX_TOKENS):
        raise ValueError(f"A520 Gemma append produced incompatible K {appended_k.shape}.")
    if appended_v.shape != appended_k.shape:
        raise ValueError(f"A520 appended K/V shapes differ: {appended_k.shape} vs {appended_v.shape}.")
    fresh_active_k = appended_k[:, :, INACTIVE_A520:]
    fresh_active_v = appended_v[:, :, INACTIVE_A520:]
    composite_k = acot_model._mrr_scatter_cache(  # noqa: SLF001
        anchor_k,
        fresh_active_k,
        active_indices,
    )
    composite_v = acot_model._mrr_scatter_cache(  # noqa: SLF001
        anchor_v,
        fresh_active_v,
        active_indices,
    )
    composite_prefix_out = acot_model._mrr_scatter_tokens(  # noqa: SLF001
        anchor_prefix["prefix_out"],
        active_prefix_out,
        active_indices,
    )
    composite_prefix_tokens = acot_model._mrr_scatter_tokens(  # noqa: SLF001
        anchor_prefix["prefix_tokens"],
        active_tokens,
        active_indices,
    )
    return {
        "observation": current["observation"],
        "prefix_tokens": composite_prefix_tokens,
        "prefix_mask": current["prefix_mask"],
        "prefix_ar_mask": anchor_prefix["prefix_ar_mask"],
        "prefix_out": composite_prefix_out,
        "kv_cache": (composite_k, composite_v),
        "ref_action_noise": current["ref_action_noise"],
        "expert_action_noise": current["expert_action_noise"],
        "mrr_selected_block_ids": selected_block_ids,
    }


def _make_evaluator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    args: Args,
):
    @jax.jit
    def evaluate(
        base_state: nnx.State,
        selector_state: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        feature_mean: jax.Array,
        feature_std: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, selector_state)
        # Same key keeps flow noise identical across stale, splice, active, and
        # exact-fresh prefixes.  MRR prepare_current reproduces the same split.
        anchor_prefix = base_model.sample_actions_profile_prefix(rng, batch["anchor_observation"])
        fresh_prefix = base_model.sample_actions_profile_prefix(rng, batch["current_observation"])
        current = base_model.sample_actions_profile_mrr_prepare_current(
            rng,
            batch["current_observation"],
        )

        features = mrr_block_selector.construct_mrr_block_features(
            anchor_prefix["prefix_tokens"][:, :VALID_VISUAL_TOKENS],
            current["visual_tokens"],
            p3t_trainer._low_resolution_images(anchor_prefix, 64),  # noqa: SLF001
            current["images_64"],
            current["observation"].state[:, :32] - anchor_prefix["observation"].state[:, :32],
            batch["executed_actions"],
            batch["anchor_ear"],
            projection_seed=selector_runtime.config.feature_projection_seed,
            projection_rank=selector_runtime.config.projection_rank,
        )
        normalized = (features - feature_mean[None, None]) / feature_std[None, None]
        block_logits = scorer(normalized).astype(jnp.float32)
        selected_block_scores, selected_block_ids = jax.lax.top_k(
            block_logits,
            mrr_block_selector.TOP_K,
        )

        stale = _stale_prefix(anchor_prefix, fresh_prefix)
        direct = _direct_splice_prefix(anchor_prefix, fresh_prefix, selected_block_ids)
        active_a264 = base_model.sample_actions_profile_mrr_replay(
            anchor_prefix,
            current,
            selected_block_ids,
        )
        active_a520 = _active_a520_replay(
            base_model,
            anchor_prefix,
            current,
            selected_block_ids,
        )
        comparison = _stack_prefix_states(
            (stale, direct, active_a264, active_a520, fresh_prefix)
        )
        iar, ear, actions = p3t_trainer._reason_and_act(  # noqa: SLF001
            base_model,
            comparison,
            coarse_flow_steps=args.coarse_flow_steps,
            final_flow_steps=args.final_flow_steps,
        )
        target_iar = iar[4:5]
        target_ear = ear[4:5]
        target_actions = actions[4:5]
        metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            iar,
            ear,
            actions,
            target_iar,
            target_ear,
            target_actions,
        )

        def action_rmse(left: int, right: int) -> jax.Array:
            delta = actions[left, ..., :7].astype(jnp.float32) - actions[right, ..., :7].astype(
                jnp.float32
            )
            return jnp.sqrt(jnp.mean(jnp.square(delta)))

        return {
            "metrics": metrics,
            "selected_block_ids": selected_block_ids[0],
            "selected_block_scores": selected_block_scores[0],
            "block_logits": block_logits[0],
            "active_a264_vs_splice_action_rmse_7d": action_rmse(2, 1),
            "active_a520_vs_splice_action_rmse_7d": action_rmse(3, 1),
            "active_a520_vs_a264_action_rmse_7d": action_rmse(3, 2),
            "current_visual_token_max_abs_error_vs_full": jnp.max(
                jnp.abs(
                    current["visual_tokens"].astype(jnp.float32)
                    - fresh_prefix["prefix_tokens"][:, :VALID_VISUAL_TOKENS].astype(jnp.float32)
                )
            ),
            "current_rgb64_max_abs_error_vs_full": jnp.max(
                jnp.abs(
                    current["images_64"]
                    - p3t_trainer._low_resolution_images(fresh_prefix, 64)  # noqa: SLF001
                )
            ),
            "ref_noise_max_abs_error": jnp.max(
                jnp.abs(current["ref_action_noise"] - fresh_prefix["ref_action_noise"])
            ),
            "expert_noise_max_abs_error": jnp.max(
                jnp.abs(current["expert_action_noise"] - fresh_prefix["expert_action_noise"])
            ),
        }

    return evaluate


def _validate_dummy_invariant(batch: dict[str, Any]) -> None:
    anchor = np.asarray(
        jax.device_get(batch["anchor_observation"].images[DUMMY_IMAGE_KEY])
    )
    current = np.asarray(
        jax.device_get(batch["current_observation"].images[DUMMY_IMAGE_KEY])
    )
    anchor_mask = np.asarray(
        jax.device_get(batch["anchor_observation"].image_masks[DUMMY_IMAGE_KEY])
    )
    current_mask = np.asarray(
        jax.device_get(batch["current_observation"].image_masks[DUMMY_IMAGE_KEY])
    )
    if not np.array_equal(anchor, current) or np.any(anchor_mask) or np.any(current_mask):
        raise ValueError(
            "A520 requires identical anchor/current dummy images and false dummy masks."
        )


def _summarize_metrics(values: np.ndarray, stale_mean: np.ndarray) -> dict[str, Any]:
    mean = np.mean(values, axis=0)
    closure = {
        name: float(1.0 - mean[index] / max(stale_mean[index], 1e-12))
        for index, name in enumerate(mrr_oracle.METRIC_NAMES[:3])
    }
    return {
        "metrics": mrr_oracle._metric_dict(mean),  # noqa: SLF001
        "global_gap_closure_vs_stale": closure,
    }


def _aggregate(pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = np.stack([record["metrics"] for record in pair_records])
    if metrics.shape[1:] != (len(METHOD_NAMES), len(mrr_oracle.METRIC_NAMES)):
        raise ValueError(f"Unexpected semantic diagnostic metric shape {metrics.shape}.")
    stale_mean = np.mean(metrics[:, 0], axis=0)
    methods = {
        name: _summarize_metrics(metrics[:, index], stale_mean)
        for index, name in enumerate(METHOD_NAMES)
    }
    direct_closure = methods["direct_splice"]["global_gap_closure_vs_stale"]

    def distribution(name: str) -> dict[str, float]:
        values = np.asarray([record[name] for record in pair_records], dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    semantic_comparison = {}
    for method_name in ("active_a264", "active_a520"):
        closure = methods[method_name]["global_gap_closure_vs_stale"]
        semantic_comparison[method_name] = {
            "action_closure_minus_direct_splice": closure["action_mse_7d"]
            - direct_closure["action_mse_7d"],
            "ear_closure_minus_direct_splice": closure["ear_mse_7d"]
            - direct_closure["ear_mse_7d"],
            "iar_closure_minus_direct_splice": closure["iar_mse"]
            - direct_closure["iar_mse"],
        }
    return {
        "num_pairs": len(pair_records),
        "methods": methods,
        "prior_direct_splice_reference_approx": {
            "action_gap_closure": 0.67,
            "ear_gap_closure": 0.62,
            "current_minus_reference": {
                "action": direct_closure["action_mse_7d"] - 0.67,
                "ear": direct_closure["ear_mse_7d"] - 0.62,
            },
        },
        "active_vs_splice_action_rmse_7d": {
            "active_a264": distribution("active_a264_vs_splice_action_rmse_7d"),
            "active_a520": distribution("active_a520_vs_splice_action_rmse_7d"),
            "a520_vs_a264": distribution("active_a520_vs_a264_action_rmse_7d"),
        },
        "semantic_closure_delta_vs_direct_splice": semantic_comparison,
        "preparation_equivalence": {
            "visual_token_max_abs_error": distribution(
                "current_visual_token_max_abs_error_vs_full"
            ),
            "rgb64_max_abs_error": distribution("current_rgb64_max_abs_error_vs_full"),
            "ref_noise_max_abs_error": distribution("ref_noise_max_abs_error"),
            "expert_noise_max_abs_error": distribution("expert_noise_max_abs_error"),
        },
        "interpretation": (
            "If direct_splice retains its prior closure while A264 collapses, the 2/5 result is an active-replay "
            "semantic mismatch rather than selector failure. If A520 materially recovers A264, masked dummy-query "
            "semantics are causal; if A264 and A520 both track direct splice, investigate closed-loop distribution "
            "shift."
        ),
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    pairs_path = output_dir / "pairs.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (pairs_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"MRR semantic diagnostic output already exists in {output_dir}; pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

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
    train_indices, validation_indices, test_indices = p3t_trainer._split_pairs(  # noqa: SLF001
        pairs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    actual_split = (len(pairs), train_indices.size, validation_indices.size, test_indices.size)
    expected_split = (
        args.expected_pairs,
        args.expected_train_pairs,
        args.expected_validation_pairs,
        args.expected_test_pairs,
    )
    if actual_split != expected_split:
        raise ValueError(f"Expected selector split {expected_split}, got {actual_split}.")
    test_pairs = pairs.take(test_indices)
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
        test_pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    selector_runtime = mrr_block_selector.load_mrr_block_selector(args.selector_checkpoint)
    evaluator = _make_evaluator(base_graphdef, selector_runtime, args)

    pair_records: list[dict[str, Any]] = []
    with pairs_path.open("w", encoding="utf-8") as pairs_file:
        for position in range(len(records)):
            batch = p3t_trainer._batch(records, np.asarray([position], dtype=np.int64))  # noqa: SLF001
            _validate_dummy_invariant(batch)
            anchor = int(test_pairs.anchor_indices[position])
            key = jax.random.fold_in(jax.random.key(args.seed), anchor)
            output = jax.device_get(
                evaluator(
                    base_state,
                    selector_runtime.scorer_state,
                    batch,
                    key,
                    selector_runtime.feature_mean,
                    selector_runtime.feature_std,
                )
            )
            metrics = np.asarray(output["metrics"], dtype=np.float64)
            record = {
                "test_position": position,
                "original_pair_index": int(test_indices[position]),
                "anchor_index": anchor,
                "target_index": int(test_pairs.target_indices[position]),
                "episode_id": int(test_pairs.episode_ids[position]),
                "frame_id": int(test_pairs.frame_ids[position]),
                "method_order": METHOD_NAMES,
                "metrics": metrics.tolist(),
                "method_metrics": {
                    name: mrr_oracle._metric_dict(metrics[index])  # noqa: SLF001
                    for index, name in enumerate(METHOD_NAMES)
                },
                "selected_block_ids": [
                    int(value) for value in np.asarray(output["selected_block_ids"])
                ],
                "selected_block_scores": np.asarray(
                    output["selected_block_scores"],
                    dtype=np.float64,
                ).tolist(),
                "block_logits": np.asarray(output["block_logits"], dtype=np.float64).tolist(),
                "active_a264_vs_splice_action_rmse_7d": float(
                    output["active_a264_vs_splice_action_rmse_7d"]
                ),
                "active_a520_vs_splice_action_rmse_7d": float(
                    output["active_a520_vs_splice_action_rmse_7d"]
                ),
                "active_a520_vs_a264_action_rmse_7d": float(
                    output["active_a520_vs_a264_action_rmse_7d"]
                ),
                "current_visual_token_max_abs_error_vs_full": float(
                    output["current_visual_token_max_abs_error_vs_full"]
                ),
                "current_rgb64_max_abs_error_vs_full": float(
                    output["current_rgb64_max_abs_error_vs_full"]
                ),
                "ref_noise_max_abs_error": float(output["ref_noise_max_abs_error"]),
                "expert_noise_max_abs_error": float(output["expert_noise_max_abs_error"]),
            }
            flat_numeric = [
                *metrics.reshape(-1),
                *np.asarray(output["selected_block_scores"]).reshape(-1),
                *np.asarray(output["block_logits"]).reshape(-1),
                record["active_a264_vs_splice_action_rmse_7d"],
                record["active_a520_vs_splice_action_rmse_7d"],
                record["active_a520_vs_a264_action_rmse_7d"],
            ]
            if not np.all(np.isfinite(np.asarray(flat_numeric, dtype=np.float64))):
                raise FloatingPointError(f"Non-finite MRR semantic diagnostic output at pair {position}.")
            pairs_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            pairs_file.flush()
            pair_records.append(record)
            LOGGER.info(
                "MRR semantic test %d/%d anchor=%d blocks=%s elapsed=%.1fs",
                position + 1,
                len(records),
                anchor,
                record["selected_block_ids"],
                time.monotonic() - started,
            )

    aggregate = _aggregate(pair_records)
    summary = {
        "method": "MRR learned-selector direct-splice vs active-replay semantic diagnosis",
        "status": "offline_test22_diagnostic_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "selector": {
            "artifact_dir": str(selector_runtime.artifact_dir),
            "feature_dim": selector_runtime.config.feature_dim,
            "hidden_dim": selector_runtime.config.hidden_dim,
            "parameter_count": mrr_block_selector.PARAMETER_COUNT,
            "top_k": selector_runtime.config.top_k,
        },
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
            "test_episodes": sorted(
                int(value) for value in np.unique(pairs.episode_ids[test_indices])
            ),
        },
        "physical_replay": {
            "direct_splice": "no Gemma replay; 64 fresh deep-KV visual slots, anchor language",
            "active_a264": {
                "queries": ACTIVE_A264,
                "inactive_anchor_keys": PREFIX_TOKENS - ACTIVE_A264,
                "active": "64 learned visual + 200 current language",
            },
            "active_a520": {
                "queries": ACTIVE_A520,
                "inactive_anchor_keys": INACTIVE_A520,
                "active": "64 learned visual + 256 invariant dummy + 200 current language",
            },
            "dense_968_query_surrogate": False,
        },
        "test": aggregate,
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "pairs": str(pairs_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
