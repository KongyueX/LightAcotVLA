"""Exact dual-path counterfactual oracle on the fixed MRR Task8 test22 split.

This probe is the first structural screen for DCE/CDVA-ACoT.  It ranks visual
blocks with the learned MRR selector, retains top 4/8/16 blocks, and evaluates
four exact interventions under identical prefix/coarse/final flow noise:

* stale:       anchor prefix + stale IAR/EAR;
* plan-only:   anchor prefix + direct-splice IAR/EAR;
* direct-only: direct-splice prefix + stale IAR/EAR;
* joint:       direct-splice prefix + direct-splice IAR/EAR.

The exact full-fresh policy is the target.  The probe reports held-out action,
EAR, and IAR gap closure, gripper-sign agreement, and an inclusion-exclusion
decomposition into mediated, direct, and interaction gains.  It trains no
adapter and never changes the default policy/evaluation path.
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
from openpi.models import mrr_block_selector

try:
    import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    import probe_mrr_compiler_bottleneck_oracle as compiler_oracle
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    from scripts import probe_mrr_compiler_bottleneck_oracle as compiler_oracle
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("probe_dce_acot_dual_path_oracle")
METHOD_NAMES = ("stale", "plan_only", "direct_only", "joint", "fresh")
STALE_INDEX = 0
PLAN_INDEX = 1
DIRECT_INDEX = 2
JOINT_INDEX = 3
FRESH_INDEX = 4


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
    top_k: int = 4
    joint_action_closure_gate: float = 0.75
    conditional_direct_gain_gate: float = 0.25
    conditional_plan_gain_gate: float = 0.10
    plan_ear_closure_gate: float = 0.60
    joint_gripper_accuracy_gate: float = 0.95
    delta_floor: float = 1e-8
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if any(value < 0 for value in (args.dataset_task_id, args.seed, args.split_seed)):
        raise ValueError("Task id and seeds must be non-negative.")
    if args.temporal_stride != 10:
        raise ValueError("DCE oracle requires the fixed anchor-to-anchor+10 protocol.")
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
    if args.top_k not in (4, 8, 16):
        raise ValueError("DCE oracle --top-k must be one of 4, 8, or 16.")
    gates = (
        args.joint_action_closure_gate,
        args.conditional_direct_gain_gate,
        args.conditional_plan_gain_gate,
        args.plan_ear_closure_gate,
        args.joint_gripper_accuracy_gate,
    )
    if any(not 0.0 <= value <= 1.0 for value in gates):
        raise ValueError("DCE gates must lie in [0, 1].")
    if args.delta_floor <= 0.0:
        raise ValueError("delta_floor must be positive.")


def _counterfactual_prefix(
    anchor_prefix: dict[str, Any],
    fresh_prefix: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Build [anchor, anchor, direct, direct, fresh] final-prefix rows."""

    anchor_kv = anchor_prefix["kv_cache"]
    comparison_kv = comparison["kv_cache"]
    kv_cache = (
        jnp.concatenate(
            [
                anchor_kv[0],
                anchor_kv[0],
                comparison_kv[0][:, 1:2],
                comparison_kv[0][:, 1:2],
                fresh_prefix["kv_cache"][0],
            ],
            axis=1,
        ),
        jnp.concatenate(
            [
                anchor_kv[1],
                anchor_kv[1],
                comparison_kv[1][:, 1:2],
                comparison_kv[1][:, 1:2],
                fresh_prefix["kv_cache"][1],
            ],
            axis=1,
        ),
    )
    result = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
        fresh_prefix,
        anchor_prefix["prefix_out"],
        kv_cache,
        len(METHOD_NAMES),
    )
    result["prefix_out"] = jnp.concatenate(
        [
            jnp.repeat(anchor_prefix["prefix_out"], JOINT_INDEX + 1, axis=0),
            fresh_prefix["prefix_out"],
        ],
        axis=0,
    )
    return result


def _action_effect_statistics(actions: jax.Array, *, delta_floor: float) -> jax.Array:
    """Return RMS and target alignment for plan/direct/interaction/joint effects."""

    actions = actions[..., :7].astype(jnp.float32)
    stale = actions[STALE_INDEX]
    plan_effect = actions[PLAN_INDEX] - stale
    direct_effect = actions[DIRECT_INDEX] - stale
    interaction_effect = (
        actions[JOINT_INDEX] - actions[DIRECT_INDEX] - actions[PLAN_INDEX] + stale
    )
    joint_effect = actions[JOINT_INDEX] - stale
    target_delta = actions[FRESH_INDEX] - stale
    effects = jnp.stack([plan_effect, direct_effect, interaction_effect, joint_effect])
    target_flat = target_delta.reshape(-1)
    target_norm = jnp.linalg.norm(target_flat)

    def statistics(effect: jax.Array) -> jax.Array:
        flat = effect.reshape(-1)
        rms = jnp.sqrt(jnp.mean(jnp.square(flat)))
        cosine = jnp.vdot(flat, target_flat) / jnp.maximum(
            jnp.linalg.norm(flat) * target_norm,
            delta_floor,
        )
        return jnp.stack([rms, cosine])

    return jax.vmap(statistics)(effects)


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
        anchor_prefix, fresh_prefix, _, _, block_logits = (
            compiler_oracle._learned_direct_context(  # noqa: SLF001
                base_model,
                scorer,
                batch,
                rng,
                feature_mean,
                feature_std,
                projection_seed=selector_runtime.config.feature_projection_seed,
                projection_rank=selector_runtime.config.projection_rank,
            )
        )
        _, selected_ids = jax.lax.top_k(block_logits, args.top_k)
        selected_mask = mrr_oracle._selected_visual_mask(selected_ids[0])  # noqa: SLF001
        direct_kv = mrr_oracle._composite_cache(  # noqa: SLF001
            anchor_prefix["kv_cache"],
            fresh_prefix["kv_cache"],
            selected_mask[None],
            jnp.zeros((1,), dtype=jnp.bool_),
        )
        comparison = compiler_oracle._prefix_comparison(  # noqa: SLF001
            anchor_prefix,
            fresh_prefix,
            direct_kv,
        )
        comparison_iar, comparison_ear = compiler_oracle._bottlenecks(  # noqa: SLF001
            base_model,
            comparison,
            coarse_flow_steps=args.coarse_flow_steps,
        )
        stale_iar, direct_iar, fresh_iar = (
            comparison_iar[0:1],
            comparison_iar[1:2],
            comparison_iar[2:3],
        )
        stale_ear, direct_ear, fresh_ear = (
            comparison_ear[0:1],
            comparison_ear[1:2],
            comparison_ear[2:3],
        )
        variant_iar = jnp.concatenate(
            [stale_iar, direct_iar, stale_iar, direct_iar, fresh_iar],
            axis=0,
        )
        variant_ear = jnp.concatenate(
            [stale_ear, direct_ear, stale_ear, direct_ear, fresh_ear],
            axis=0,
        )
        variant_prefix = _counterfactual_prefix(anchor_prefix, fresh_prefix, comparison)
        actions = compiler_oracle._final_actions(  # noqa: SLF001
            base_model,
            variant_prefix,
            variant_ear,
            variant_iar,
            final_flow_steps=args.final_flow_steps,
        )
        metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            variant_iar,
            variant_ear,
            actions,
            fresh_iar,
            fresh_ear,
            actions[FRESH_INDEX : FRESH_INDEX + 1],
        )
        return {
            "metrics": metrics,
            "action_effect_statistics": _action_effect_statistics(
                actions,
                delta_floor=args.delta_floor,
            ),
            "selected_block_ids": selected_ids[0],
            "block_logits": block_logits[0],
        }

    return evaluate


def _closure(candidate: np.ndarray, stale: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for index, name in enumerate(mrr_oracle.METRIC_NAMES[:3]):
        denominator = float(stale[index])
        result[name] = None if denominator <= 1e-12 else float(1.0 - candidate[index] / denominator)
    return result


def _pair_decomposition(metrics: np.ndarray, *, delta_floor: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(mrr_oracle.METRIC_NAMES[:3]):
        stale = float(metrics[STALE_INDEX, index])
        denominator = max(stale, delta_floor)
        plan = float(metrics[PLAN_INDEX, index])
        direct = float(metrics[DIRECT_INDEX, index])
        joint = float(metrics[JOINT_INDEX, index])
        mediated_gain = (stale - plan) / denominator
        direct_gain = (stale - direct) / denominator
        interaction_gain = (plan + direct - stale - joint) / denominator
        result[name] = {
            "mediated_gain_stale_gap_units": mediated_gain,
            "direct_gain_stale_gap_units": direct_gain,
            "interaction_gain_stale_gap_units": interaction_gain,
            "joint_gain_stale_gap_units": (stale - joint) / denominator,
            "conditional_direct_gain_beyond_plan": (plan - joint) / denominator,
            "conditional_plan_gain_beyond_direct": (direct - joint) / denominator,
            "reconstruction_error": mediated_gain
            + direct_gain
            + interaction_gain
            - (stale - joint) / denominator,
        }
    return result


def _aggregate(records: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    metrics = np.stack([np.asarray(record["metric_arrays"], dtype=np.float64) for record in records])
    if metrics.shape[1:] != (len(METHOD_NAMES), len(mrr_oracle.METRIC_NAMES)):
        raise ValueError(f"Unexpected DCE metric shape {metrics.shape}.")
    means = np.mean(metrics, axis=0)
    stale_mean = means[STALE_INDEX]
    methods = {}
    for method_index, method_name in enumerate(METHOD_NAMES):
        values = metrics[:, method_index]
        methods[method_name] = {
            "metrics": mrr_oracle._metric_dict(means[method_index]),  # noqa: SLF001
            "global_gap_closure_vs_stale": _closure(means[method_index], stale_mean),
            "mean_pair_gap_closure_vs_stale": {
                metric_name: float(
                    np.mean(
                        1.0
                        - values[:, metric_index]
                        / np.maximum(metrics[:, STALE_INDEX, metric_index], args.delta_floor)
                    )
                )
                for metric_index, metric_name in enumerate(mrr_oracle.METRIC_NAMES[:3])
            },
        }

    decomposition = _pair_decomposition(means, delta_floor=args.delta_floor)
    action_decomposition = decomposition["action_mse_7d"]
    action_effects = np.stack(
        [np.asarray(record["action_effect_statistics"], dtype=np.float64) for record in records]
    )
    effect_names = ("mediated", "direct", "interaction", "joint")
    effect_summary = {
        name: {
            "mean_rms_7d": float(np.mean(action_effects[:, index, 0])),
            "mean_cosine_to_full_fresh_delta": float(np.mean(action_effects[:, index, 1])),
        }
        for index, name in enumerate(effect_names)
    }
    checks = {
        "joint_action_gap_closure": action_decomposition["joint_gain_stale_gap_units"]
        >= args.joint_action_closure_gate,
        "conditional_direct_gain_beyond_plan": action_decomposition[
            "conditional_direct_gain_beyond_plan"
        ]
        >= args.conditional_direct_gain_gate,
        "conditional_plan_gain_beyond_direct": action_decomposition[
            "conditional_plan_gain_beyond_direct"
        ]
        >= args.conditional_plan_gain_gate,
        "plan_ear_gap_closure": methods["plan_only"]["global_gap_closure_vs_stale"]["ear_mse_7d"]
        >= args.plan_ear_closure_gate,
        "joint_gripper_sign_accuracy": methods["joint"]["metrics"]["gripper_sign_accuracy"]
        >= args.joint_gripper_accuracy_gate,
    }
    if not checks["joint_action_gap_closure"]:
        decision = "no_go_mrr_top4_joint_action_ceiling"
    elif not checks["conditional_direct_gain_beyond_plan"]:
        decision = "no_go_direct_path_complementarity"
    elif not checks["conditional_plan_gain_beyond_direct"]:
        decision = "no_go_plan_path_complementarity"
    elif not checks["plan_ear_gap_closure"]:
        decision = "no_go_plan_quality"
    elif not checks["joint_gripper_sign_accuracy"]:
        decision = "no_go_gripper_quality"
    else:
        decision = "continue_to_learned_dual_path_adapter"

    return {
        "num_pairs": len(records),
        "methods": methods,
        "counterfactual_decomposition": decomposition,
        "action_effect_vectors": effect_summary,
        "gate": {
            "thresholds": {
                "joint_action_gap_closure": args.joint_action_closure_gate,
                "conditional_direct_gain_beyond_plan": args.conditional_direct_gain_gate,
                "conditional_plan_gain_beyond_direct": args.conditional_plan_gain_gate,
                "plan_ear_gap_closure": args.plan_ear_closure_gate,
                "joint_gripper_sign_accuracy": args.joint_gripper_accuracy_gate,
            },
            "checks": checks,
            "decision": decision,
            "pass": all(checks.values()),
        },
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    per_pair_path = output_dir / "per_pair.jsonl"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and (per_pair_path.exists() or summary_path.exists()):
        raise FileExistsError(f"DCE oracle output already exists in {output_dir}; pass --overwrite.")
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
        raise ValueError(f"Expected MRR split {expected_split}, got {actual_split}.")

    heldout_pairs = pairs.take(test_indices)
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
        heldout_pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    selector_runtime = mrr_block_selector.load_mrr_block_selector(args.selector_checkpoint)
    evaluator = _make_evaluator(base_graphdef, selector_runtime, args)

    per_pair: list[dict[str, Any]] = []
    with per_pair_path.open("w", encoding="utf-8") as output_file:
        for position, pair in enumerate(
            zip(
                heldout_pairs.anchor_indices,
                heldout_pairs.target_indices,
                heldout_pairs.episode_ids,
                heldout_pairs.frame_ids,
                strict=True,
            )
        ):
            anchor, target, episode, frame = (int(value) for value in pair)
            batch = p3t_trainer._batch(records, np.asarray([position], dtype=np.int64))  # noqa: SLF001
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
            action_effect_statistics = np.asarray(
                output["action_effect_statistics"],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(metrics)) or not np.all(np.isfinite(action_effect_statistics)):
                raise FloatingPointError(f"Non-finite DCE output at held-out pair {position}.")
            record = {
                "test_position": position,
                "anchor_index": anchor,
                "target_index": target,
                "episode_id": episode,
                "frame_id": frame,
                "selected_block_ids": [
                    int(value) for value in np.asarray(output["selected_block_ids"])
                ],
                "metric_arrays": metrics.tolist(),
                "methods": {
                    name: {
                        "metrics": mrr_oracle._metric_dict(metrics[index]),  # noqa: SLF001
                        "gap_closure_vs_stale": _closure(metrics[index], metrics[STALE_INDEX]),
                    }
                    for index, name in enumerate(METHOD_NAMES)
                },
                "counterfactual_decomposition": _pair_decomposition(
                    metrics,
                    delta_floor=args.delta_floor,
                ),
                "action_effect_statistics": action_effect_statistics.tolist(),
            }
            output_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output_file.flush()
            per_pair.append(record)
            LOGGER.info(
                "DCE exact oracle pair %d/%d anchor=%d episode=%d elapsed=%.1fs",
                position + 1,
                len(records),
                anchor,
                episode,
                time.monotonic() - started,
            )

    evaluation = _aggregate(per_pair, args)
    summary = {
        "method": "DCE/CDVA-ACoT exact dual-path counterfactual oracle",
        "status": "offline_capacity_oracle_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "selector": {
            "artifact_dir": str(selector_runtime.artifact_dir),
            "parameter_count": mrr_block_selector.PARAMETER_COUNT,
            "trained_top_k": selector_runtime.config.top_k,
            "oracle_top_k": args.top_k,
            "selected_visual_tokens": args.top_k * mrr_oracle.TOKENS_PER_BLOCK,
            "selected_visual_token_fraction": (
                args.top_k * mrr_oracle.TOKENS_PER_BLOCK / mrr_oracle.VISUAL_TOKENS
            ),
            "ranking": "same learned selector logits; only the retained prefix length changes",
        },
        "heldout": {
            "partition": "MRR/P3T episode-disjoint test22",
            "pairs": len(heldout_pairs),
            "episodes": sorted(int(value) for value in np.unique(heldout_pairs.episode_ids)),
            "task": "Task8",
            "dataset_task_id": args.dataset_task_id,
        },
        "interventions": {
            "stale": "anchor prefix plus stale IAR/EAR",
            "plan_only": "anchor prefix plus learned-direct-splice IAR/EAR",
            "direct_only": "learned direct-splice prefix plus stale IAR/EAR",
            "joint": "learned direct-splice prefix plus learned-direct-splice IAR/EAR",
            "fresh": "full-fresh prefix plus full-fresh IAR/EAR target",
            "noise": "identical prefix/coarse/final flow noise for every row within a pair",
        },
        "evaluation": evaluation,
        "constraints": {
            "adapter_training": False,
            "direct_action_residual": False,
            "default_inference_modified": False,
            "oracle_scope": "exact path decomposition; not deployable latency or Task8 success",
        },
        "interpretation": (
            "A failed joint ceiling rejects the learned MRR top-4 splice as the visual carrier; "
            "it does not by itself reject a denser low-cost dual-path visual adapter."
        ),
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {"per_pair": str(per_pair_path), "summary": str(summary_path)},
        "note": "Open-loop Test22 structure gate only; pass requires later learned adapter and closed-loop success.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
