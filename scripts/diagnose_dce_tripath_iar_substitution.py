"""Diagnose the learned IAR path by exact substitution on Task8 Test22.

This probe restores an already-trained three-path DCE artifact and performs no
optimization.  For every held-out pair it freezes the learned predicted EAR,
the learned final evidence hook, current evidence, anchor prefix, and flow
noise.  The only intervention is the IAR supplied to the final expert:

* ``stale_iar``: the anchor-prefix IAR;
* ``predicted_iar``: the learned IAR evidence adapter output;
* ``exact_direct_iar``: the learned-top8 deep-KV-splice IAR teacher.

The resulting action-gap closures discriminate a bad IAR predictor from a
downstream path that does not use IAR or an incompatible IAR target geometry.
It is an offline privileged diagnostic and makes no deployment claim.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import dce_iar_evidence_adapter
from openpi.models import dce_multilayer_evidence_adapter
from openpi.models import model as model_lib
from openpi.models import mrr_block_selector

try:
    import train_dce_evidence_acot_tri_path_oracle as tri
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import train_dce_evidence_acot_tri_path_oracle as tri


dce_base = tri.dce_base
ml_base = tri.ml_base
LOGGER = logging.getLogger("diagnose_dce_tripath_iar_substitution")
METHOD_NAMES = ("stale_iar", "predicted_iar", "exact_direct_iar")


@dataclasses.dataclass(frozen=True)
class Args(tri.Args):
    trained_dir: str = ""
    exact_rescue_gate: float = 0.10
    exact_action_closure_gate: float = 0.70
    mse_reduction_gate: float = 0.20
    equivalence_band: float = 0.05


def _validate_args(args: Args) -> None:
    tri._validate_args(args)  # noqa: SLF001
    if not args.trained_dir:
        raise ValueError("--trained-dir must point to an existing tri-path artifact.")
    for name in (
        "exact_rescue_gate",
        "exact_action_closure_gate",
        "mse_reduction_gate",
        "equivalence_band",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0,1], got {value}.")


def _restore_params(params: nnx.State, path: pathlib.Path, *, namespace: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter checkpoint: {path}")
    loaded_namespaces = model_lib.restore_params(path, restore_type=np.ndarray)
    loaded_namespaces = model_lib.convert_str_keys_to_int(loaded_namespaces)
    if not isinstance(loaded_namespaces, dict) or set(loaded_namespaces) != {namespace}:
        raise ValueError(f"Expected adapter namespace {namespace!r} at {path}.")
    loaded = loaded_namespaces[namespace]
    expected_flat = traverse_util.flatten_dict(params.to_pure_dict())
    loaded_flat = traverse_util.flatten_dict(loaded)
    if set(expected_flat) != set(loaded_flat):
        raise ValueError(f"Adapter parameter paths mismatch for {namespace!r}.")
    for key, expected in expected_flat.items():
        loaded_value = loaded_flat[key]
        if expected is None:
            if loaded_value is not None:
                raise ValueError(
                    f"Adapter parameter {namespace!r} at {key} must remain None."
                )
            continue
        value = np.asarray(loaded_value)
        if value.shape != expected.shape:
            raise ValueError(
                f"Adapter shape mismatch for {namespace!r} at {key}: "
                f"checkpoint={value.shape}, expected={expected.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite adapter parameter for {namespace!r} at {key}.")
    params.replace_by_pure_dict(
        jax.tree.map(
            lambda value: None if value is None else jnp.asarray(value),
            loaded,
            is_leaf=lambda value: value is None,
        )
    )


def _make_evaluator(
    base_graphdef: Any,
    selector_runtime: mrr_block_selector.MRRBlockSelectorRuntime,
    iar_graphdef: Any,
    ear_graphdef: Any,
    final_graphdef: Any,
    args: Args,
):
    @jax.jit
    def evaluate(
        base_state: nnx.State,
        scorer_state: nnx.State,
        iar_params: nnx.State,
        ear_params: nnx.State,
        final_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(selector_runtime.scorer_graphdef, scorer_state)
        iar_adapter = nnx.merge(iar_graphdef, iar_params)
        ear_adapter = nnx.merge(ear_graphdef, ear_params)
        final_adapter = nnx.merge(final_graphdef, final_params)
        context = dce_base._pair_context(  # noqa: SLF001
            base_model, scorer, selector_runtime, batch, rng, args
        )

        comparison = context["comparison"]
        teacher_iar = tri._frozen_iar(base_model, comparison)  # noqa: SLF001
        teacher_ear = base_model._one_step_coarse_endpoint(  # noqa: SLF001
            comparison,
            comparison["ref_action_noise"],
        )
        stale_iar, exact_direct_iar, fresh_iar = (
            teacher_iar[0:1],
            teacher_iar[1:2],
            teacher_iar[2:3],
        )
        fresh_ear = teacher_ear[2:3]
        predicted_iar = iar_adapter(stale_iar, context["evidence"])
        predicted_ear = ml_base._adapted_coarse_endpoint(  # noqa: SLF001
            base_model,
            context["stale_prefix"],
            ear_adapter,
            context["evidence"],
        )

        iar_variants = jnp.concatenate(
            [stale_iar, predicted_iar, exact_direct_iar], axis=0
        )
        fixed_ear = jnp.repeat(predicted_ear, len(METHOD_NAMES), axis=0)
        fixed_prefix = dce_base._repeat_stale_prefix(  # noqa: SLF001
            context["anchor_prefix"], context["fresh_prefix"], len(METHOD_NAMES)
        )
        actions = ml_base._adapted_final_endpoint(  # noqa: SLF001
            base_model,
            fixed_prefix,
            final_adapter,
            jnp.repeat(context["evidence"], len(METHOD_NAMES), axis=0),
            fixed_ear,
            iar_variants,
        )
        fresh_action = base_model._one_step_action_endpoint(  # noqa: SLF001
            context["fresh_prefix"],
            context["fresh_prefix"]["expert_action_noise"],
            fresh_ear,
            fresh_iar,
        )
        metrics = dce_base.mrr_oracle._downstream_metrics(  # noqa: SLF001
            iar_variants,
            fixed_ear,
            actions,
            fresh_iar,
            fresh_ear,
            fresh_action,
        )
        return {
            "metrics": metrics,
            "selected_ids": context["selected_ids"],
            "predicted_to_direct_iar_mse": tri._latent_mse(  # noqa: SLF001
                predicted_iar, exact_direct_iar
            ),
            "stale_to_direct_iar_mse": tri._latent_mse(  # noqa: SLF001
                stale_iar, exact_direct_iar
            ),
        }

    return evaluate


def _metric_dict(values: np.ndarray) -> dict[str, float]:
    return dce_base.mrr_oracle._metric_dict(values)  # noqa: SLF001


def _closure(candidate: np.ndarray, stale: np.ndarray) -> dict[str, float | None]:
    return dce_base._closure(candidate, stale)  # noqa: SLF001


def _decision(means: np.ndarray, args: Args) -> dict[str, Any]:
    stale_action_mse = float(means[0, 0])
    predicted_action_mse = float(means[1, 0])
    exact_action_mse = float(means[2, 0])
    predicted_closure = 1.0 - predicted_action_mse / max(stale_action_mse, args.delta_floor)
    exact_closure = 1.0 - exact_action_mse / max(stale_action_mse, args.delta_floor)
    rescue = exact_closure - predicted_closure
    mse_reduction = (predicted_action_mse - exact_action_mse) / max(
        predicted_action_mse, args.delta_floor
    )

    if rescue >= args.exact_rescue_gate and mse_reduction >= args.mse_reduction_gate:
        outcome = "learned_iar_predictor_or_representation_is_bottleneck"
    elif (
        abs(exact_closure - predicted_closure) < args.equivalence_band
        and abs(exact_closure) < args.equivalence_band
    ):
        outcome = "learned_ear_final_path_is_effectively_insensitive_to_iar"
    elif exact_closure >= args.equivalence_band and predicted_closure < 0.0:
        outcome = "predicted_iar_has_wrong_direction_or_token_correspondence"
    elif exact_closure < 0.0:
        outcome = "exact_direct_iar_geometry_is_incompatible_with_fixed_learned_downstream"
    else:
        outcome = "ambiguous_run_per_layer_cosine_and_fixed_hungarian_correspondence_probe"

    return {
        "outcome": outcome,
        "action_gap_closure": {
            "predicted_iar": predicted_closure,
            "exact_direct_iar": exact_closure,
            "exact_minus_predicted": rescue,
        },
        "exact_action_mse_reduction_vs_predicted": mse_reduction,
        "route_worthy_exact_ceiling": exact_closure >= args.exact_action_closure_gate,
        "thresholds": {
            "exact_minus_predicted": args.exact_rescue_gate,
            "exact_action_mse_reduction_vs_predicted": args.mse_reduction_gate,
            "route_worthy_exact_action_closure": args.exact_action_closure_gate,
            "equivalence_band": args.equivalence_band,
        },
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = dce_base.p3t_trainer._gpu()  # noqa: SLF001
    started = time.monotonic()
    output_dir = pathlib.Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    per_pair_path = output_dir / "test_pairs.jsonl"
    if not args.overwrite and (summary_path.exists() or per_pair_path.exists()):
        raise FileExistsError(f"Diagnostic output exists in {output_dir}; pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = dce_base.p3t_trainer.Args(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        endpoint_student_params=args.endpoint_student_params,
        output_dir=args.output_dir,
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
        dce_base.p3t_trainer._load_model_and_dataset(trainer_args)  # noqa: SLF001
    )
    pairs = dce_base.p3t_trainer._select_pairs(  # noqa: SLF001
        arrays,
        raw_dataset,
        task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        maximum_pairs=200,
        seed=args.seed,
    )
    train_indices, validation_indices, test_indices = dce_base.p3t_trainer._split_pairs(  # noqa: SLF001
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
        raise ValueError(f"Expected diagnostic split {expected_split}, got {actual_split}.")
    records = dce_base.p3t_trainer._materialize_pairs(  # noqa: SLF001
        pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=args.temporal_stride,
    )
    selector_runtime = mrr_block_selector.load_mrr_block_selector(args.selector_checkpoint)
    base_model = nnx.merge(base_graphdef, base_state)
    query_dim = int(base_model.action_in_proj.out_features)
    evidence_tokens = dce_base.EVIDENCE_TOKENS_BY_MODE[args.evidence_mode]
    evidence_dim = int(mrr_block_selector.TOKEN_EMBEDDING_DIM) * (
        2 if args.evidence_mode == "selected128_pair" else 1
    )
    iar_adapter = dce_iar_evidence_adapter.DCEIAREvidenceAdapter(
        dce_iar_evidence_adapter.DCEIAREvidenceAdapterConfig(
            query_dim=tri.IAR_DIM,
            query_tokens=tri.IAR_TOKENS,
            evidence_dim=evidence_dim,
            evidence_tokens=evidence_tokens,
            attention_dim=args.attention_dim,
            num_heads=args.attention_heads,
        ),
        rngs=nnx.Rngs(args.seed + 707),
    )
    multilayer_config = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapterConfig(
        evidence_dim=evidence_dim,
        evidence_tokens=evidence_tokens,
        expert_dim=query_dim,
        attention_dim=args.attention_dim,
        num_heads=args.attention_heads,
        expert_depth=ml_base.EXPERT_DEPTH,
        injection_layers=ml_base.INJECTION_LAYERS,
    )
    ear_adapter = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter(
        multilayer_config, rngs=nnx.Rngs(args.seed + 808)
    )
    final_adapter = dce_multilayer_evidence_adapter.DCEMultiLayerEvidenceAdapter(
        multilayer_config, rngs=nnx.Rngs(args.seed + 909)
    )
    iar_graphdef, iar_params = nnx.split(iar_adapter)
    ear_graphdef, ear_params = nnx.split(ear_adapter)
    final_graphdef, final_params = nnx.split(final_adapter)

    trained_dir = pathlib.Path(args.trained_dir).resolve()
    _restore_params(
        iar_params,
        trained_dir / "final" / "iar_adapter" / "params",
        namespace="dce_iar_evidence_adapter",
    )
    _restore_params(
        ear_params,
        trained_dir / "final" / "ear_adapter" / "params",
        namespace="dce_ear_multilayer_evidence_adapter",
    )
    _restore_params(
        final_params,
        trained_dir / "final" / "final_adapter" / "params",
        namespace="dce_final_multilayer_evidence_adapter",
    )

    evaluator = _make_evaluator(
        base_graphdef,
        selector_runtime,
        iar_graphdef,
        ear_graphdef,
        final_graphdef,
        args,
    )
    metric_records: list[np.ndarray] = []
    direct_mse_records: list[float] = []
    stale_direct_mse_records: list[float] = []
    with per_pair_path.open("w", encoding="utf-8") as output_file:
        for position, record_index_value in enumerate(test_indices):
            record_index = int(record_index_value)
            anchor = int(pairs.anchor_indices[record_index])
            batch = dce_base.p3t_trainer._batch(  # noqa: SLF001
                records, np.asarray([record_index], dtype=np.int64)
            )
            output = jax.device_get(
                evaluator(
                    base_state,
                    selector_runtime.scorer_state,
                    iar_params,
                    ear_params,
                    final_params,
                    batch,
                    jax.random.fold_in(jax.random.key(args.seed), anchor),
                )
            )
            metrics = np.asarray(output["metrics"], dtype=np.float64)
            if not np.all(np.isfinite(metrics)):
                raise FloatingPointError(f"Non-finite diagnostic output at pair {record_index}.")
            metric_records.append(metrics)
            direct_mse_records.append(float(output["predicted_to_direct_iar_mse"]))
            stale_direct_mse_records.append(float(output["stale_to_direct_iar_mse"]))
            record = {
                "test_position": position,
                "pair_index": record_index,
                "anchor_index": anchor,
                "target_index": int(pairs.target_indices[record_index]),
                "episode_id": int(pairs.episode_ids[record_index]),
                "selected_block_ids": [int(value) for value in np.asarray(output["selected_ids"])],
                "methods": {
                    name: {
                        "metrics": _metric_dict(metrics[index]),
                        "gap_closure_vs_fixed_stale_iar": _closure(metrics[index], metrics[0]),
                    }
                    for index, name in enumerate(METHOD_NAMES)
                },
                "predicted_to_direct_iar_mse": direct_mse_records[-1],
                "stale_to_direct_iar_mse": stale_direct_mse_records[-1],
            }
            output_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            LOGGER.info("IAR substitution %d/%d anchor=%d", position + 1, test_indices.size, anchor)

    means = np.mean(np.stack(metric_records), axis=0)
    summary = {
        "method": "DCE tri-path exact-IAR substitution diagnostic",
        "status": "offline_test22_privileged_diagnostic",
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
        },
        "controlled_intervention": {
            "fixed": [
                "learned predicted EAR",
                "learned final evidence hook",
                "current selected128_pair evidence",
                "anchor prefix",
                "flow noise",
            ],
            "varied": list(METHOD_NAMES),
            "exact_direct_iar": "learned-top8 deep-KV-splice teacher",
        },
        "methods": {
            name: {
                "metrics": _metric_dict(means[index]),
                "global_gap_closure_vs_fixed_stale_iar": _closure(means[index], means[0]),
            }
            for index, name in enumerate(METHOD_NAMES)
        },
        "iar_target_fit": {
            "predicted_to_direct_mse": float(np.mean(direct_mse_records)),
            "stale_to_direct_mse": float(np.mean(stale_direct_mse_records)),
        },
        "decision": _decision(means, args),
        "constraints": {
            "training": False,
            "full_fresh_used_as_action_target_only": True,
            "exact_direct_iar_is_privileged": True,
            "closed_loop_success_claim": False,
            "deployable_speed_claim": False,
        },
        "outputs": {
            "summary": str(summary_path),
            "per_pair": str(per_pair_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
