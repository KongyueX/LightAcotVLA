"""Low-cost MRR-Compiler bottleneck oracle on the fixed Task8 split.

The learned MRR top-4 selector first constructs the existing anchor-language
direct deep-KV splice.  The frozen policy then supplies stale, composite, and
fresh IAR/EAR bottlenecks under identical flow noise.

Only the 144 training pairs fit two centered PCA bases: one for
``composite_IAR - stale_IAR`` and one for
``composite_EAR - stale_EAR``.  On test22, exact test deltas provide oracle
projection coefficients at ranks 4/8/16/32.  Reconstructed IAR-only,
EAR-only, and joint bottlenecks are passed to the frozen final expert while
the prefix remains strictly stale.  No action residual is fitted or injected.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import mrr_block_selector

try:
    import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("probe_mrr_compiler_bottleneck_oracle")
RANKS = (4, 8, 16, 32)
MAX_RANK = max(RANKS)
BASELINE_NAMES = ("stale", "direct_splice", "fresh")
EXACT_NAME = "exact_bottleneck"


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
    action_closure_gate: float = 0.65
    ear_closure_gate: float = 0.60
    gripper_accuracy_gate: float = 0.95
    overwrite: bool = False


class PCAFit(NamedTuple):
    mean: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    explained_variance: dict[int, float]
    original_shape: tuple[int, ...]


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if any(value < 0 for value in (args.dataset_task_id, args.seed, args.split_seed)):
        raise ValueError("Task id and seeds must be non-negative.")
    if args.temporal_stride != 10:
        raise ValueError("MRR-Compiler oracle requires the fixed anchor-to-anchor+10 protocol.")
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
    if any(value <= 0 for value in expected) or sum(expected[1:]) != expected[0]:
        raise ValueError("Expected split sizes must be positive and sum to expected_pairs.")
    if args.coarse_flow_steps <= 0 or args.final_flow_steps <= 0:
        raise ValueError("Flow step counts must be positive.")
    gates = (
        args.action_closure_gate,
        args.ear_closure_gate,
        args.gripper_accuracy_gate,
    )
    if any(not 0.0 <= value <= 1.0 for value in gates):
        raise ValueError("Compiler gates must lie in [0, 1].")


def _prefix_comparison(
    anchor_prefix: dict[str, Any],
    fresh_prefix: dict[str, Any],
    direct_kv: tuple[jax.Array, jax.Array],
) -> dict[str, Any]:
    kv_cache = (
        jnp.concatenate(
            [anchor_prefix["kv_cache"][0], direct_kv[0], fresh_prefix["kv_cache"][0]],
            axis=1,
        ),
        jnp.concatenate(
            [anchor_prefix["kv_cache"][1], direct_kv[1], fresh_prefix["kv_cache"][1]],
            axis=1,
        ),
    )
    comparison = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
        fresh_prefix,
        anchor_prefix["prefix_out"],
        kv_cache,
        3,
    )
    comparison["prefix_out"] = jnp.concatenate(
        [anchor_prefix["prefix_out"], anchor_prefix["prefix_out"], fresh_prefix["prefix_out"]],
        axis=0,
    )
    return comparison


def _final_actions(
    base_model: Any,
    prefix_state: dict[str, Any],
    ear: jax.Array,
    iar: jax.Array,
    *,
    final_flow_steps: int,
) -> jax.Array:
    if final_flow_steps == 1:
        return base_model.sample_actions_profile_direct_one_step_expert(
            prefix_state,
            ear,
            iar,
            0.0,
        )["actions"]
    return base_model.sample_actions_profile_expert(
        prefix_state,
        ear,
        iar,
        num_steps=final_flow_steps,
    )["actions"]


def _bottlenecks(
    base_model: Any,
    prefix_state: dict[str, Any],
    *,
    coarse_flow_steps: int,
) -> tuple[jax.Array, jax.Array]:
    iar = base_model.sample_actions_profile_implicit(prefix_state)["implicit_action_reason"]
    if iar is None:
        raise ValueError("MRR-Compiler requires the frozen implicit action reasoner.")
    if coarse_flow_steps == 1:
        ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            prefix_state,
            prefix_state["ref_action_noise"],
        )
    else:
        ear = base_model.sample_actions_profile_coarse(
            prefix_state,
            num_steps=coarse_flow_steps,
            action_cot_denoising_steps=coarse_flow_steps,
            dynamic_denoising_steps=False,
        )["explicit_action_reason"]
    if ear is None:
        raise ValueError("MRR-Compiler requires the frozen explicit action reasoner.")
    return iar, ear


def _learned_direct_context(
    base_model: Any,
    scorer: mrr_block_selector.MRRBlockSelectorScorer,
    batch: dict[str, Any],
    rng: jax.Array,
    feature_mean: jax.Array,
    feature_std: jax.Array,
    *,
    projection_seed: int,
    projection_rank: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], jax.Array, jax.Array]:
    anchor_prefix = base_model.sample_actions_profile_prefix(rng, batch["anchor_observation"])
    fresh_prefix = base_model.sample_actions_profile_prefix(rng, batch["current_observation"])
    features = mrr_block_selector.construct_mrr_block_features(
        anchor_prefix["prefix_tokens"][:, : mrr_oracle.VISUAL_TOKENS],
        fresh_prefix["prefix_tokens"][:, : mrr_oracle.VISUAL_TOKENS],
        p3t_trainer._low_resolution_images(anchor_prefix, 64),  # noqa: SLF001
        p3t_trainer._low_resolution_images(fresh_prefix, 64),  # noqa: SLF001
        fresh_prefix["observation"].state[:, :32] - anchor_prefix["observation"].state[:, :32],
        batch["executed_actions"],
        batch["anchor_ear"],
        projection_seed=projection_seed,
        projection_rank=projection_rank,
    )
    normalized = (features - feature_mean[None, None]) / feature_std[None, None]
    block_logits = scorer(normalized).astype(jnp.float32)
    _, selected_ids = jax.lax.top_k(block_logits, mrr_block_selector.TOP_K)
    selected_mask = mrr_oracle._selected_visual_mask(selected_ids[0])  # noqa: SLF001
    direct_kv = mrr_oracle._composite_cache(  # noqa: SLF001
        anchor_prefix["kv_cache"],
        fresh_prefix["kv_cache"],
        selected_mask[None],
        jnp.zeros((1,), dtype=jnp.bool_),
    )
    comparison = _prefix_comparison(anchor_prefix, fresh_prefix, direct_kv)
    return anchor_prefix, fresh_prefix, comparison, selected_ids, block_logits


def _make_delta_generator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    args: Args,
):
    @jax.jit
    def generate(
        base_state: nnx.State,
        selector_state: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        feature_mean: jax.Array,
        feature_std: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, selector_state)
        _, _, comparison, selected_ids, _ = _learned_direct_context(
            base_model,
            scorer,
            batch,
            rng,
            feature_mean,
            feature_std,
            projection_seed=selector_runtime.config.feature_projection_seed,
            projection_rank=selector_runtime.config.projection_rank,
        )
        iar, ear = _bottlenecks(
            base_model,
            comparison,
            coarse_flow_steps=args.coarse_flow_steps,
        )
        return {
            "delta_iar": iar[1] - iar[0],
            "delta_ear": ear[1] - ear[0],
            "selected_block_ids": selected_ids[0],
        }

    return generate


def _make_exact_evaluator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    args: Args,
):
    """Compile the cheap exact-bottleneck screen before fitting any PCA."""

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
        anchor_prefix, fresh_prefix, comparison, selected_ids, _ = _learned_direct_context(
            base_model,
            scorer,
            batch,
            rng,
            feature_mean,
            feature_std,
            projection_seed=selector_runtime.config.feature_projection_seed,
            projection_rank=selector_runtime.config.projection_rank,
        )
        baseline_iar, baseline_ear = _bottlenecks(
            base_model,
            comparison,
            coarse_flow_steps=args.coarse_flow_steps,
        )
        baseline_actions = _final_actions(
            base_model,
            comparison,
            baseline_ear,
            baseline_iar,
            final_flow_steps=args.final_flow_steps,
        )
        fresh_iar = baseline_iar[2:3]
        fresh_ear = baseline_ear[2:3]
        fresh_actions = baseline_actions[2:3]
        baseline_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            baseline_iar,
            baseline_ear,
            baseline_actions,
            fresh_iar,
            fresh_ear,
            fresh_actions,
        )
        stale_prefix = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
            fresh_prefix,
            anchor_prefix["prefix_out"],
            anchor_prefix["kv_cache"],
            1,
        )
        exact_actions = _final_actions(
            base_model,
            stale_prefix,
            baseline_ear[1:2],
            baseline_iar[1:2],
            final_flow_steps=args.final_flow_steps,
        )
        exact_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            baseline_iar[1:2],
            baseline_ear[1:2],
            exact_actions,
            fresh_iar,
            fresh_ear,
            fresh_actions,
        )[0]
        return {
            "baseline_metrics": baseline_metrics,
            "exact_metrics": exact_metrics,
            "selected_block_ids": selected_ids[0],
        }

    return evaluate


def _fit_pca(values: np.ndarray, *, name: str) -> PCAFit:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] <= MAX_RANK:
        raise ValueError(f"{name} PCA requires more than {MAX_RANK} training examples, got {values.shape}.")
    original_shape = tuple(int(value) for value in values.shape[1:])
    flat = values.reshape(values.shape[0], -1)
    mean = np.mean(flat, axis=0, dtype=np.float64).astype(np.float32)
    centered = flat - mean[None]
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    basis = np.asarray(right_vectors[:MAX_RANK], dtype=np.float32)
    if basis.shape != (MAX_RANK, flat.shape[1]):
        raise ValueError(f"{name} PCA returned incompatible basis {basis.shape}.")
    variance = np.square(singular_values)
    total = float(np.sum(variance))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} training deltas have no finite PCA variance.")
    explained = {
        rank: float(np.sum(variance[:rank]) / total)
        for rank in RANKS
    }
    return PCAFit(
        mean=mean,
        basis=basis,
        singular_values=singular_values.astype(np.float32),
        explained_variance=explained,
        original_shape=original_shape,
    )


def _jax_reconstruct_delta(
    delta: jax.Array,
    mean: jax.Array,
    basis: jax.Array,
    rank: int,
) -> jax.Array:
    flat = delta.reshape((delta.shape[0], -1)).astype(jnp.float32)
    active_basis = basis[:rank]
    coefficients = (flat - mean[None]) @ active_basis.T
    reconstructed = mean[None] + coefficients @ active_basis
    return reconstructed.reshape(delta.shape)


def _make_test_evaluator(
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
        iar_mean: jax.Array,
        iar_basis: jax.Array,
        ear_mean: jax.Array,
        ear_basis: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, selector_state)
        anchor_prefix, fresh_prefix, comparison, selected_ids, block_logits = _learned_direct_context(
            base_model,
            scorer,
            batch,
            rng,
            feature_mean,
            feature_std,
            projection_seed=selector_runtime.config.feature_projection_seed,
            projection_rank=selector_runtime.config.projection_rank,
        )
        baseline_iar, baseline_ear = _bottlenecks(
            base_model,
            comparison,
            coarse_flow_steps=args.coarse_flow_steps,
        )
        baseline_actions = _final_actions(
            base_model,
            comparison,
            baseline_ear,
            baseline_iar,
            final_flow_steps=args.final_flow_steps,
        )
        stale_iar, direct_iar, fresh_iar = baseline_iar[0:1], baseline_iar[1:2], baseline_iar[2:3]
        stale_ear, direct_ear, fresh_ear = baseline_ear[0:1], baseline_ear[1:2], baseline_ear[2:3]
        fresh_actions = baseline_actions[2:3]
        baseline_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            baseline_iar,
            baseline_ear,
            baseline_actions,
            fresh_iar,
            fresh_ear,
            fresh_actions,
        )

        delta_iar = direct_iar - stale_iar
        delta_ear = direct_ear - stale_ear
        injected_iars = [direct_iar]
        injected_ears = [direct_ear]
        for rank in RANKS:
            reconstructed_iar = stale_iar + _jax_reconstruct_delta(
                delta_iar,
                iar_mean,
                iar_basis,
                rank,
            )
            reconstructed_ear = stale_ear + _jax_reconstruct_delta(
                delta_ear,
                ear_mean,
                ear_basis,
                rank,
            )
            injected_iars.extend([reconstructed_iar, stale_iar, reconstructed_iar])
            injected_ears.extend([stale_ear, reconstructed_ear, reconstructed_ear])
        injected_iar = jnp.concatenate(injected_iars, axis=0)
        injected_ear = jnp.concatenate(injected_ears, axis=0)
        variant_count = 1 + 3 * len(RANKS)
        if injected_iar.shape[0] != variant_count or injected_ear.shape[0] != variant_count:
            raise AssertionError("MRR-Compiler produced an invalid static variant count.")

        stale_kv = (
            jnp.repeat(anchor_prefix["kv_cache"][0], variant_count, axis=1),
            jnp.repeat(anchor_prefix["kv_cache"][1], variant_count, axis=1),
        )
        stale_prefix = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
            fresh_prefix,
            anchor_prefix["prefix_out"],
            stale_kv,
            variant_count,
        )
        injected_actions = _final_actions(
            base_model,
            stale_prefix,
            injected_ear,
            injected_iar,
            final_flow_steps=args.final_flow_steps,
        )
        injected_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            injected_iar,
            injected_ear,
            injected_actions,
            fresh_iar,
            fresh_ear,
            fresh_actions,
        )
        return {
            "baseline_metrics": baseline_metrics,
            "injected_metrics": injected_metrics,
            "delta_iar": delta_iar[0],
            "delta_ear": delta_ear[0],
            "selected_block_ids": selected_ids[0],
            "block_logits": block_logits[0],
        }

    return evaluate


def _projection_statistics(values: np.ndarray, fit: PCAFit) -> dict[int, dict[str, float]]:
    flat = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    centered = flat - fit.mean[None]
    centered_energy = float(np.sum(np.square(centered, dtype=np.float64)))
    raw_energy = float(np.sum(np.square(flat, dtype=np.float64)))
    result = {}
    for rank in RANKS:
        basis = fit.basis[:rank]
        coefficients = centered @ basis.T
        reconstructed = fit.mean[None] + coefficients @ basis
        residual = flat - reconstructed
        residual_energy = float(np.sum(np.square(residual, dtype=np.float64)))
        result[rank] = {
            "explained_variance_around_train_mean": 1.0
            - residual_energy / max(centered_energy, 1e-12),
            "raw_delta_energy_recovery": 1.0 - residual_energy / max(raw_energy, 1e-12),
            "projection_rmse": float(np.sqrt(np.mean(np.square(residual, dtype=np.float64)))),
        }
    return result


def _variant_names() -> tuple[str, ...]:
    names = [EXACT_NAME]
    for rank in RANKS:
        names.extend(
            [
                f"rank{rank}_iar_only",
                f"rank{rank}_ear_only",
                f"rank{rank}_iar_ear",
            ]
        )
    return tuple(names)


def _summarize_method(values: np.ndarray, stale_mean: np.ndarray) -> dict[str, Any]:
    mean = np.mean(values, axis=0)
    closure = {
        name: float(1.0 - mean[index] / max(stale_mean[index], 1e-12))
        for index, name in enumerate(mrr_oracle.METRIC_NAMES[:3])
    }
    return {
        "metrics": mrr_oracle._metric_dict(mean),  # noqa: SLF001
        "global_gap_closure_vs_stale": closure,
    }


def _checks(method: dict[str, Any], args: Args) -> dict[str, bool]:
    closure = method["global_gap_closure_vs_stale"]
    metrics = method["metrics"]
    return {
        "action_gap_closure": closure["action_mse_7d"] >= args.action_closure_gate,
        "ear_gap_closure": closure["ear_mse_7d"] >= args.ear_closure_gate,
        "gripper_sign_accuracy": metrics["gripper_sign_accuracy"] >= args.gripper_accuracy_gate,
    }


def _aggregate_exact_screen(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    baselines = np.stack([record["baseline_metrics"] for record in records])
    exact = np.stack([record["exact_metrics"] for record in records])
    stale_mean = np.mean(baselines[:, 0], axis=0)
    baseline_summary = {
        name: _summarize_method(baselines[:, index], stale_mean)
        for index, name in enumerate(BASELINE_NAMES)
    }
    exact_summary = _summarize_method(exact, stale_mean)
    checks = _checks(exact_summary, args)
    return {
        "num_pairs": len(records),
        "baselines": baseline_summary,
        "exact_bottleneck": exact_summary,
        "checks": checks,
        "pass": all(checks.values()),
        "failure_decision": None if all(checks.values()) else "no_go_exact_bottleneck",
    }


def _aggregate(
    records: list[dict[str, Any]],
    args: Args,
) -> dict[str, Any]:
    baselines = np.stack([record["baseline_metrics"] for record in records])
    variants = np.stack([record["injected_metrics"] for record in records])
    variant_names = _variant_names()
    if baselines.shape[1:] != (len(BASELINE_NAMES), len(mrr_oracle.METRIC_NAMES)):
        raise ValueError(f"Unexpected compiler baseline metric shape {baselines.shape}.")
    if variants.shape[1:] != (len(variant_names), len(mrr_oracle.METRIC_NAMES)):
        raise ValueError(f"Unexpected compiler injected metric shape {variants.shape}.")
    stale_mean = np.mean(baselines[:, 0], axis=0)
    baseline_summary = {
        name: _summarize_method(baselines[:, index], stale_mean)
        for index, name in enumerate(BASELINE_NAMES)
    }
    variant_summary = {
        name: _summarize_method(variants[:, index], stale_mean)
        for index, name in enumerate(variant_names)
    }
    exact_checks = _checks(variant_summary[EXACT_NAME], args)
    rank_checks = {
        rank: _checks(variant_summary[f"rank{rank}_iar_ear"], args)
        for rank in RANKS
    }
    eligible = [rank for rank in RANKS if rank <= 16 and all(rank_checks[rank].values())]
    exact_pass = all(exact_checks.values())
    if not exact_pass:
        decision = "no_go_exact_bottleneck"
        minimum_rank = None
    elif eligible:
        decision = "continue"
        minimum_rank = min(eligible)
    else:
        decision = "no_go_rank_le_16"
        minimum_rank = None
    return {
        "num_pairs": len(records),
        "baselines": baseline_summary,
        "bottleneck_variants": variant_summary,
        "gate": {
            "thresholds": {
                "action_gap_closure": args.action_closure_gate,
                "ear_gap_closure": args.ear_closure_gate,
                "gripper_sign_accuracy": args.gripper_accuracy_gate,
                "eligible_rank_max": 16,
            },
            "exact_bottleneck_checks": exact_checks,
            "exact_bottleneck_pass": exact_pass,
            "joint_rank_checks": {str(rank): checks for rank, checks in rank_checks.items()},
            "minimum_passing_rank_le_16": minimum_rank,
            "decision": decision,
            "pass": decision == "continue",
        },
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    exact_pairs_path = output_dir / "exact_bottleneck_pairs.jsonl"
    train_labels_path = output_dir / "train_labels.jsonl"
    test_pairs_path = output_dir / "test_pairs.jsonl"
    pca_path = output_dir / "pca_basis.npz"
    summary_path = output_dir / "summary.json"
    targets = (exact_pairs_path, train_labels_path, test_pairs_path, pca_path, summary_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"MRR-Compiler oracle output already exists in {output_dir}; pass --overwrite to replace it."
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
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
        pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    selector_runtime = mrr_block_selector.load_mrr_block_selector(args.selector_checkpoint)

    # Exact composite IAR+EAR on a stale prefix is the compiler's two-bottleneck
    # upper bound.  Screen it first so a failed ceiling does not spend 144 GPU
    # prefix evaluations or fit meaningless PCA bases.
    exact_evaluator = _make_exact_evaluator(base_graphdef, selector_runtime, args)
    exact_records: list[dict[str, Any]] = []
    with exact_pairs_path.open("w", encoding="utf-8") as exact_file:
        for position, record_index in enumerate(test_indices):
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
            anchor = int(pairs.anchor_indices[record_index])
            key = jax.random.fold_in(jax.random.key(args.seed), anchor)
            output = jax.device_get(
                exact_evaluator(
                    base_state,
                    selector_runtime.scorer_state,
                    batch,
                    key,
                    selector_runtime.feature_mean,
                    selector_runtime.feature_std,
                )
            )
            baseline_metrics = np.asarray(output["baseline_metrics"], dtype=np.float64)
            exact_metrics = np.asarray(output["exact_metrics"], dtype=np.float64)
            if not np.all(np.isfinite(baseline_metrics)) or not np.all(np.isfinite(exact_metrics)):
                raise FloatingPointError(f"Non-finite exact bottleneck output at pair {record_index}.")
            exact_record = {
                "test_position": position,
                "pair_index": int(record_index),
                "anchor_index": anchor,
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [
                    int(value) for value in np.asarray(output["selected_block_ids"])
                ],
                "baseline_metrics": baseline_metrics.tolist(),
                "exact_metrics": exact_metrics.tolist(),
            }
            exact_file.write(json.dumps(exact_record, sort_keys=True, allow_nan=False) + "\n")
            exact_file.flush()
            exact_records.append(exact_record)
            LOGGER.info(
                "Compiler exact screen %d/%d anchor=%d elapsed=%.1fs",
                position + 1,
                test_indices.size,
                anchor,
                time.monotonic() - started,
            )
    exact_screen = _aggregate_exact_screen(exact_records, args)
    if not exact_screen["pass"]:
        summary = {
            "method": "MRR-Compiler exact and PCA bottleneck oracle",
            "status": "no_go_exact_bottleneck",
            "device": str(device),
            "args": dataclasses.asdict(args),
            "model": model_metadata,
            "selector": {
                "artifact_dir": str(selector_runtime.artifact_dir),
                "parameter_count": mrr_block_selector.PARAMETER_COUNT,
                "top_k": selector_runtime.config.top_k,
            },
            "split": {
                "pairs": len(pairs),
                "train_skipped": int(train_indices.size),
                "validation_unused": int(validation_indices.size),
                "test": int(test_indices.size),
            },
            "exact_screen": exact_screen,
            "pca": {"status": "skipped_because_exact_bottleneck_failed"},
            "constraints": {
                "exact_final_prefix": "stale anchor KV/prefix_out at current observation",
                "direct_action_residual": False,
            },
            "elapsed_seconds": time.monotonic() - started,
            "outputs": {
                "exact_pairs": str(exact_pairs_path),
                "summary": str(summary_path),
            },
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        return

    delta_generator = _make_delta_generator(base_graphdef, selector_runtime, args)

    train_delta_iar = []
    train_delta_ear = []
    with train_labels_path.open("w", encoding="utf-8") as train_file:
        for position, record_index in enumerate(train_indices):
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
            anchor = int(pairs.anchor_indices[record_index])
            key = jax.random.fold_in(jax.random.key(args.seed), anchor)
            output = jax.device_get(
                delta_generator(
                    base_state,
                    selector_runtime.scorer_state,
                    batch,
                    key,
                    selector_runtime.feature_mean,
                    selector_runtime.feature_std,
                )
            )
            delta_iar = np.asarray(output["delta_iar"], dtype=np.float32)
            delta_ear = np.asarray(output["delta_ear"], dtype=np.float32)
            if not np.all(np.isfinite(delta_iar)) or not np.all(np.isfinite(delta_ear)):
                raise FloatingPointError(f"Non-finite train bottleneck delta at pair {record_index}.")
            train_delta_iar.append(delta_iar)
            train_delta_ear.append(delta_ear)
            label_record = {
                "position": position,
                "pair_index": int(record_index),
                "anchor_index": anchor,
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [
                    int(value) for value in np.asarray(output["selected_block_ids"])
                ],
                "delta_iar_rms": float(np.sqrt(np.mean(np.square(delta_iar, dtype=np.float64)))),
                "delta_ear_rms": float(np.sqrt(np.mean(np.square(delta_ear, dtype=np.float64)))),
            }
            train_file.write(json.dumps(label_record, sort_keys=True, allow_nan=False) + "\n")
            train_file.flush()
            if position == 0 or (position + 1) % 10 == 0 or position + 1 == train_indices.size:
                LOGGER.info(
                    "Compiler train delta %d/%d anchor=%d elapsed=%.1fs",
                    position + 1,
                    train_indices.size,
                    anchor,
                    time.monotonic() - started,
                )

    train_delta_iar_array = np.stack(train_delta_iar)
    train_delta_ear_array = np.stack(train_delta_ear)
    iar_pca = _fit_pca(train_delta_iar_array, name="IAR")
    ear_pca = _fit_pca(train_delta_ear_array, name="EAR")
    np.savez_compressed(
        pca_path,
        iar_mean=iar_pca.mean,
        iar_basis=iar_pca.basis,
        iar_singular_values=iar_pca.singular_values,
        ear_mean=ear_pca.mean,
        ear_basis=ear_pca.basis,
        ear_singular_values=ear_pca.singular_values,
    )
    LOGGER.info("Fitted train144 IAR/EAR PCA bases; starting test22 oracle projection.")

    test_evaluator = _make_test_evaluator(base_graphdef, selector_runtime, args)
    test_records: list[dict[str, Any]] = []
    test_delta_iar = []
    test_delta_ear = []
    variant_names = _variant_names()
    with test_pairs_path.open("w", encoding="utf-8") as test_file:
        for position, record_index in enumerate(test_indices):
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
            anchor = int(pairs.anchor_indices[record_index])
            key = jax.random.fold_in(jax.random.key(args.seed), anchor)
            output = jax.device_get(
                test_evaluator(
                    base_state,
                    selector_runtime.scorer_state,
                    batch,
                    key,
                    selector_runtime.feature_mean,
                    selector_runtime.feature_std,
                    jnp.asarray(iar_pca.mean),
                    jnp.asarray(iar_pca.basis),
                    jnp.asarray(ear_pca.mean),
                    jnp.asarray(ear_pca.basis),
                )
            )
            baseline_metrics = np.asarray(output["baseline_metrics"], dtype=np.float64)
            injected_metrics = np.asarray(output["injected_metrics"], dtype=np.float64)
            delta_iar = np.asarray(output["delta_iar"], dtype=np.float32)
            delta_ear = np.asarray(output["delta_ear"], dtype=np.float32)
            numeric = np.concatenate(
                [
                    baseline_metrics.reshape(-1),
                    injected_metrics.reshape(-1),
                    delta_iar.reshape(-1),
                    delta_ear.reshape(-1),
                ]
            )
            if not np.all(np.isfinite(numeric)):
                raise FloatingPointError(f"Non-finite compiler test output at pair {record_index}.")
            test_delta_iar.append(delta_iar)
            test_delta_ear.append(delta_ear)
            pair_record = {
                "test_position": position,
                "pair_index": int(record_index),
                "anchor_index": anchor,
                "target_index": int(pairs.target_indices[record_index]),
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [
                    int(value) for value in np.asarray(output["selected_block_ids"])
                ],
                "baseline_metrics": baseline_metrics.tolist(),
                "baseline_method_metrics": {
                    name: mrr_oracle._metric_dict(baseline_metrics[index])  # noqa: SLF001
                    for index, name in enumerate(BASELINE_NAMES)
                },
                "injected_metrics": injected_metrics.tolist(),
                "injected_method_metrics": {
                    name: mrr_oracle._metric_dict(injected_metrics[index])  # noqa: SLF001
                    for index, name in enumerate(variant_names)
                },
            }
            test_file.write(json.dumps(pair_record, sort_keys=True, allow_nan=False) + "\n")
            test_file.flush()
            test_records.append(pair_record)
            LOGGER.info(
                "Compiler test oracle %d/%d anchor=%d elapsed=%.1fs",
                position + 1,
                test_indices.size,
                anchor,
                time.monotonic() - started,
            )

    test_delta_iar_array = np.stack(test_delta_iar)
    test_delta_ear_array = np.stack(test_delta_ear)
    evaluation = _aggregate(test_records, args)
    summary = {
        "method": "MRR-Compiler exact and PCA bottleneck oracle",
        "status": "offline_capacity_oracle_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "selector": {
            "artifact_dir": str(selector_runtime.artifact_dir),
            "parameter_count": mrr_block_selector.PARAMETER_COUNT,
            "top_k": selector_runtime.config.top_k,
        },
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation_unused_for_fit": int(validation_indices.size),
            "test": int(test_indices.size),
            "train_episodes": sorted(
                int(value) for value in np.unique(pairs.episode_ids[train_indices])
            ),
            "test_episodes": sorted(
                int(value) for value in np.unique(pairs.episode_ids[test_indices])
            ),
        },
        "pca": {
            "fit_partition": "train144 only",
            "test_coefficients": "oracle projection of exact held-out composite-minus-stale delta",
            "iar_shape": list(iar_pca.original_shape),
            "ear_shape": list(ear_pca.original_shape),
            "train_explained_variance": {
                "iar": {str(rank): value for rank, value in iar_pca.explained_variance.items()},
                "ear": {str(rank): value for rank, value in ear_pca.explained_variance.items()},
            },
            "test_projection": {
                "iar": {
                    str(rank): values
                    for rank, values in _projection_statistics(test_delta_iar_array, iar_pca).items()
                },
                "ear": {
                    str(rank): values
                    for rank, values in _projection_statistics(test_delta_ear_array, ear_pca).items()
                },
            },
        },
        "test": evaluation,
        "exact_screen_before_pca": exact_screen,
        "constraints": {
            "all_injected_final_prefixes": "stale anchor KV/prefix_out at current observation",
            "direct_action_residual": False,
            "validation_used_for_basis_or_selection": False,
        },
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "exact_pairs": str(exact_pairs_path),
            "train_labels": str(train_labels_path),
            "test_pairs": str(test_pairs_path),
            "pca_basis": str(pca_path),
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
