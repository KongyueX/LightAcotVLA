"""Audit EAR causality and export endpoint-distillation labels.

The exporter makes one clean policy call and one call per requested EAR
intervention. Every call for a frame uses the same policy seed, so observation,
IAR, and final-action flow noise are held fixed. The resulting HDF5 shards are
consumed by ``train_acot_endpoint_distillation.py``.

Typical teacher export:

    python scripts/audit_ear_interventions.py \
      --config-name acot_libero_action_cot_explicit_implicit_co_fusion \
      --checkpoint-dir /path/to/50999 \
      --output-dir /path/to/ir_acot/teacher_labels \
      --max-items 10000

To relabel final actions under a trained student EAR, first export that
student's one-step coarse trajectories with ``--endpoint-student-params``.
Then run the base teacher again with ``--coarse-overrides-from`` pointing at
the student's export.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

import h5py
import jax
import numpy as np

from openpi.action_cot import endpoint_dataset
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader

LOGGER = logging.getLogger("audit_ear_interventions")


def _status(message: str) -> None:
    print(f"[audit_ear_interventions] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="acot_libero_action_cot_explicit_implicit_co_fusion",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint-student-params", default=None)
    parser.add_argument(
        "--coarse-overrides-from",
        nargs="*",
        default=None,
        help="Endpoint shard directories/files whose clean_coarse_env should replace generated EAR.",
    )
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--selection", choices=("evenly_spaced", "random", "first"), default="evenly_spaced")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--records-per-shard", type=int, default=256)
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=endpoint_dataset.INTERVENTION_NAMES,
        default=list(endpoint_dataset.INTERVENTION_NAMES),
    )
    parser.add_argument(
        "--one-intervention-per-record",
        action="store_true",
        help="Cycle through one configured non-null intervention per frame to reduce teacher calls.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Export only clean EAR/endpoints. Intended for the intermediate student-EAR pass.",
    )
    parser.add_argument("--translation-magnitude", type=float, default=0.02)
    parser.add_argument("--rotation-magnitude", type=float, default=0.10)
    parser.add_argument("--gripper-shift", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=None)
    parser.add_argument("--profile-policy-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--min-semantic-response-l2", type=float, default=0.05)
    parser.add_argument("--min-semantic-null-ratio", type=float, default=3.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_items <= 0 or args.records_per_shard <= 0:
        raise ValueError("--max-items and --records-per-shard must be positive.")
    if args.translation_magnitude <= 0 or args.rotation_magnitude <= 0:
        raise ValueError("Translation and rotation magnitudes must be positive.")
    if args.gripper_shift == 0:
        raise ValueError("--gripper-shift must be non-zero.")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.action_cot_denoising_steps is not None and args.action_cot_denoising_steps <= 0:
        raise ValueError("--action-cot-denoising-steps must be positive.")
    if args.one_intervention_per_record and not any(name != "null" for name in args.interventions):
        raise ValueError("--one-intervention-per-record needs at least one non-null intervention.")


def _load_norm_stats(
    train_config: config_lib.TrainConfig,
    data_config: config_lib.DataConfig,
    checkpoint_dir: pathlib.Path,
) -> dict[str, Any]:
    if data_config.norm_stats is not None:
        return data_config.norm_stats
    if data_config.asset_id is None:
        raise ValueError("The data config needs asset_id to load checkpoint normalization stats.")
    return checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)


def _select_indices(length: int, max_items: int, selection: str, seed: int) -> np.ndarray:
    count = min(length, max_items)
    if count <= 0:
        raise ValueError("The policy dataset is empty.")
    if selection == "first" or count == length:
        return np.arange(count, dtype=np.int64)
    if selection == "random":
        return np.sort(np.random.default_rng(seed).choice(length, size=count, replace=False))
    return np.unique(np.linspace(0, length - 1, num=count, dtype=np.int64))


def _existing_indices(output_dir: pathlib.Path) -> set[int]:
    result: set[int] = set()
    for shard in sorted(output_dir.glob("shard-*.h5")):
        with h5py.File(shard, "r") as handle:
            result.update(int(value) for value in handle["dataset_index"][:])
    return result


def _accumulate_existing_responses(
    output_dir: pathlib.Path,
    response_values: dict[str, list[float]],
) -> int:
    """Restore audit statistics from completed shards before a resumed export."""

    shards = sorted(output_dir.glob("shard-*.h5"))
    if not shards:
        return 0
    arrays = endpoint_dataset.load_endpoint_arrays((output_dir,))
    ids = arrays["intervention_ids"]
    valid = arrays["intervention_valid"]
    responses = arrays["response_l2"]
    for name, intervention_id in endpoint_dataset.INTERVENTION_IDS.items():
        selected = valid & (ids == intervention_id) & np.isfinite(responses)
        response_values[name].extend(float(value) for value in responses[selected])
    return int(arrays["dataset_index"].shape[0])


def _resume_timing_values(output_dir: pathlib.Path) -> list[float]:
    """Recover the previous timing mean with its record count for weighted resume stats."""

    summary_path = output_dir / "audit_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mean = summary.get("mean_clean_policy_infer_ms")
    count = int(summary.get("processed_records", 0))
    if mean is None or count <= 0:
        return []
    return [float(mean)] * count


def _scalar(item: Any, names: tuple[str, ...], default: int) -> int:
    if not isinstance(item, dict):
        return default
    for name in names:
        if name not in item:
            continue
        value = np.asarray(item[name])
        if value.shape == () and np.issubdtype(value.dtype, np.number):
            return int(value.item())
    return default


def _pad_action_dim(values: np.ndarray, horizon: int, action_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != horizon:
        raise ValueError(f"Expected action tensor [{horizon}, D], got {values.shape}.")
    if values.shape[1] > action_dim:
        return values[:, :action_dim]
    if values.shape[1] < action_dim:
        values = np.pad(values, ((0, 0), (0, action_dim - values.shape[1])))
    return values


def _require_output(
    result: dict[str, Any],
    *,
    coarse_horizon: int,
    action_horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "coarse_actions",
        "actions",
        "execution_horizon_coarse_actions_normalized",
        "execution_horizon_final_actions_normalized",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError(f"Policy output is missing endpoint export fields: {missing}")
    coarse_env = np.asarray(result["coarse_actions"], dtype=np.float32)
    actions_env = np.asarray(result["actions"], dtype=np.float32)
    coarse = _pad_action_dim(
        result["execution_horizon_coarse_actions_normalized"],
        coarse_horizon,
        action_dim,
    )
    actions = _pad_action_dim(
        result["execution_horizon_final_actions_normalized"],
        action_horizon,
        action_dim,
    )
    return coarse_env, actions_env, coarse, actions


def _load_coarse_overrides(
    inputs: list[str] | None,
) -> dict[int, tuple[int, np.ndarray]]:
    if not inputs:
        return {}
    arrays = endpoint_dataset.load_endpoint_arrays(inputs)
    result: dict[int, tuple[int, np.ndarray]] = {}
    for index, seed, coarse in zip(
        arrays["dataset_index"],
        arrays["policy_seed"],
        arrays["clean_coarse_env"],
        strict=True,
    ):
        result[int(index)] = (int(seed), np.asarray(coarse, dtype=np.float32))
    return result


def _noise(seed: int, shape: endpoint_dataset.EndpointDatasetShape) -> tuple[np.ndarray, np.ndarray]:
    coarse_key, action_key = jax.random.split(jax.random.key(seed), 2)
    coarse = jax.random.normal(
        coarse_key,
        (shape.coarse_horizon, shape.action_dim),
        dtype=jax.numpy.float32,
    )
    action = jax.random.normal(
        action_key,
        (shape.action_horizon, shape.action_dim),
        dtype=jax.numpy.float32,
    )
    return np.asarray(coarse), np.asarray(action)


def _infer(
    policy: Any,
    item: dict[str, Any],
    *,
    seed: int,
    profile_policy_timing: bool,
    coarse_override: np.ndarray | None = None,
) -> dict[str, Any]:
    inputs = dict(item)
    inputs["policy_seed"] = np.asarray(seed, dtype=np.uint32)
    inputs["profile_policy_timing"] = np.asarray(profile_policy_timing)
    if coarse_override is not None:
        inputs["coarse_actions_override"] = np.asarray(coarse_override, dtype=np.float32)
    return policy.infer(inputs)


def _summary(
    response_values: dict[str, list[float]],
    timing_values: list[float],
    *,
    args: argparse.Namespace,
    processed: int,
    failed: int,
) -> dict[str, Any]:
    per_intervention = {}
    for name in endpoint_dataset.INTERVENTION_NAMES:
        values = np.asarray(response_values[name], dtype=np.float64)
        values = values[np.isfinite(values)]
        per_intervention[name] = {
            "count": int(values.size),
            "median_response_l2": float(np.median(values)) if values.size else None,
            "mean_response_l2": float(np.mean(values)) if values.size else None,
            "p90_response_l2": float(np.quantile(values, 0.9)) if values.size else None,
        }
    null_median = per_intervention["null"]["median_response_l2"]
    semantic_medians = [
        per_intervention[name]["median_response_l2"]
        for name in endpoint_dataset.INTERVENTION_NAMES
        if name != "null" and per_intervention[name]["median_response_l2"] is not None
    ]
    max_semantic = max(semantic_medians, default=0.0)
    semantic_null_ratio = (
        max_semantic / max(float(null_median), 1e-8)
        if null_median is not None and semantic_medians
        else None
    )
    audit_pass = (
        semantic_null_ratio is not None
        and max_semantic >= args.min_semantic_response_l2
        and semantic_null_ratio >= args.min_semantic_null_ratio
    )
    timing = np.asarray(timing_values, dtype=np.float64)
    timing = timing[np.isfinite(timing)]
    return {
        "processed_records": processed,
        "failed_records": failed,
        "interventions": per_intervention,
        "max_semantic_median_response_l2": max_semantic,
        "semantic_to_null_median_ratio": semantic_null_ratio,
        "audit_thresholds": {
            "min_semantic_response_l2": args.min_semantic_response_l2,
            "min_semantic_null_ratio": args.min_semantic_null_ratio,
        },
        "ear_causal_audit_pass": audit_pass,
        "mean_clean_policy_infer_ms": float(np.mean(timing)) if timing.size else None,
        "note": (
            "This is an open-loop causal sensitivity gate, not a success-rate result. "
            "Closed-loop LIBERO evaluation is still required."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = _parse_args()
    _validate_args(args)

    train_config = config_lib.get_config(args.config_name)
    model_config = train_config.model
    if not getattr(model_config, "adopt_explicit_action_reasoner", False):
        raise ValueError("The selected config does not enable explicit Action-CoT/EAR.")
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = _load_norm_stats(train_config, data_config, checkpoint_dir)

    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    policy_dataset = data_loader.TransformedDataset(
        raw_dataset,
        [*data_config.repack_transforms.inputs],
    )
    selected_indices = _select_indices(len(policy_dataset), args.max_items, args.selection, args.seed)
    overrides = _load_coarse_overrides(args.coarse_overrides_from)
    if overrides:
        missing = [int(index) for index in selected_indices if int(index) not in overrides]
        if missing:
            raise KeyError(
                f"Coarse override dataset is missing {len(missing)} selected indices; first={missing[:5]}"
            )

    default_steps = 1 if args.endpoint_student_params is not None else 10
    sample_kwargs = {
        "num_steps": args.num_steps or default_steps,
        "action_cot_denoising_steps": args.action_cot_denoising_steps or default_steps,
    }
    _status(
        f"Loading policy: checkpoint={checkpoint_dir}, "
        f"coarse_steps={sample_kwargs['action_cot_denoising_steps']}, "
        f"final_steps={sample_kwargs['num_steps']}"
    )
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
        sample_kwargs=sample_kwargs,
        acot_endpoint_student_params=args.endpoint_student_params,
    )

    shape = endpoint_dataset.EndpointDatasetShape(
        action_dim=model_config.action_dim,
        env_action_dim=7,
        coarse_horizon=model_config.coarse_action_horizon,
        action_horizon=model_config.action_horizon,
        num_interventions=len(endpoint_dataset.INTERVENTION_NAMES),
    )
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_indices(output_dir)
    metadata = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "endpoint_student_params": args.endpoint_student_params,
        "coarse_overrides_from": args.coarse_overrides_from,
        "num_steps": sample_kwargs["num_steps"],
        "action_cot_denoising_steps": sample_kwargs["action_cot_denoising_steps"],
        "selection": args.selection,
        "selection_seed": args.seed,
        "interventions": list(args.interventions),
        "one_intervention_per_record": args.one_intervention_per_record,
        "clean_only": args.clean_only,
        "translation_magnitude": args.translation_magnitude,
        "rotation_magnitude": args.rotation_magnitude,
        "gripper_shift": args.gripper_shift,
    }
    response_values = {name: [] for name in endpoint_dataset.INTERVENTION_NAMES}
    existing_record_count = _accumulate_existing_responses(output_dir, response_values)
    timing_values = _resume_timing_values(output_dir)
    processed = 0
    failed = 0

    with endpoint_dataset.ShardedEndpointWriter(
        output_dir,
        shape=shape,
        records_per_shard=args.records_per_shard,
        metadata=metadata,
    ) as writer:
        for position, dataset_index_value in enumerate(selected_indices):
            dataset_index = int(dataset_index_value)
            if dataset_index in existing:
                continue
            try:
                item = policy_dataset[dataset_index]
                raw_item = raw_dataset[dataset_index]
                if item is None:
                    raise ValueError("Dataset returned None.")
                seed = args.seed + dataset_index
                coarse_override = None
                if overrides:
                    seed, coarse_override = overrides[dataset_index]
                clean_result = _infer(
                    policy,
                    item,
                    seed=seed,
                    profile_policy_timing=args.profile_policy_timing,
                    coarse_override=coarse_override,
                )
                clean_coarse_env, _, clean_coarse, clean_actions = _require_output(
                    clean_result,
                    coarse_horizon=shape.coarse_horizon,
                    action_horizon=shape.action_horizon,
                    action_dim=shape.action_dim,
                )
                if clean_coarse_env.shape != (shape.coarse_horizon, shape.env_action_dim):
                    raise ValueError(
                        f"Expected environment EAR shape "
                        f"{(shape.coarse_horizon, shape.env_action_dim)}, got {clean_coarse_env.shape}."
                    )
                timing_values.append(
                    float(clean_result.get("policy_timing", {}).get("infer_ms", np.nan))
                )

                intervention_ids = np.arange(shape.num_interventions, dtype=np.uint8)
                intervention_valid = np.zeros((shape.num_interventions,), dtype=np.bool_)
                intervention_coarse = np.repeat(clean_coarse[None, ...], shape.num_interventions, axis=0)
                intervention_actions = np.repeat(clean_actions[None, ...], shape.num_interventions, axis=0)
                intervention_coarse_env = np.repeat(
                    clean_coarse_env[None, ...],
                    shape.num_interventions,
                    axis=0,
                )
                response_l2 = np.full((shape.num_interventions,), np.nan, dtype=np.float32)

                active_interventions = list(args.interventions)
                if args.clean_only:
                    active_interventions = []
                elif args.one_intervention_per_record:
                    semantic_names = [name for name in active_interventions if name != "null"]
                    active_interventions = [
                        semantic_names[(args.seed + dataset_index) % len(semantic_names)]
                    ]

                for name in active_interventions:
                    intervention_id = endpoint_dataset.INTERVENTION_IDS[name]
                    intervened_env = endpoint_dataset.apply_intervention(
                        clean_coarse_env,
                        name,
                        seed=seed + 104_729 * (intervention_id + 1),
                        translation_magnitude=args.translation_magnitude,
                        rotation_magnitude=args.rotation_magnitude,
                        gripper_shift=args.gripper_shift,
                    )
                    result = _infer(
                        policy,
                        item,
                        seed=seed,
                        profile_policy_timing=False,
                        coarse_override=intervened_env,
                    )
                    _, _, normalized_coarse, normalized_actions = _require_output(
                        result,
                        coarse_horizon=shape.coarse_horizon,
                        action_horizon=shape.action_horizon,
                        action_dim=shape.action_dim,
                    )
                    response = float(np.linalg.norm(normalized_actions - clean_actions))
                    intervention_valid[intervention_id] = True
                    intervention_coarse[intervention_id] = normalized_coarse
                    intervention_actions[intervention_id] = normalized_actions
                    intervention_coarse_env[intervention_id] = intervened_env
                    response_l2[intervention_id] = response
                    response_values[name].append(response)

                coarse_noise, action_noise = _noise(seed, shape)
                writer.append(
                    {
                        "dataset_index": dataset_index,
                        "task_id": _scalar(raw_item, ("task_index", "task_id"), -1),
                        "episode_id": _scalar(raw_item, ("episode_index", "episode_id"), -1),
                        "frame_id": _scalar(raw_item, ("frame_index", "frame_id", "index"), dataset_index),
                        "policy_seed": seed,
                        "coarse_noise": coarse_noise,
                        "action_noise": action_noise,
                        "clean_coarse": clean_coarse,
                        "clean_actions": clean_actions,
                        "clean_coarse_env": clean_coarse_env,
                        "intervention_ids": intervention_ids,
                        "intervention_valid": intervention_valid,
                        "intervention_coarse": intervention_coarse,
                        "intervention_actions": intervention_actions,
                        "intervention_coarse_env": intervention_coarse_env,
                        "response_l2": response_l2,
                    }
                )
                processed += 1
                if processed == 1 or processed % 10 == 0:
                    _status(
                        f"Exported {processed}/{len(selected_indices)} new records "
                        f"(dataset_index={dataset_index}, scan={position + 1})"
                    )
            except Exception as exc:
                failed += 1
                LOGGER.exception("Failed dataset_index=%s", dataset_index)
                if not args.continue_on_error:
                    raise
                _status(f"Skipping dataset_index={dataset_index}: {exc}")

    summary = _summary(
        response_values,
        timing_values,
        args=args,
        processed=existing_record_count + processed,
        failed=failed,
    )
    summary["new_records"] = processed
    summary["metadata"] = metadata
    summary["dataset_indices_requested"] = len(selected_indices)
    summary["dataset_indices_already_present"] = len(existing.intersection(map(int, selected_indices)))
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _status(
        f"Done: new={processed}, failed={failed}, "
        f"EAR audit pass={summary['ear_causal_audit_pass']}, output={output_dir}"
    )


if __name__ == "__main__":
    main()
