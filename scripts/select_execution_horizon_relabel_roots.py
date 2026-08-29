"""Select noisy or decision-critical roots for paired continuation relabeling."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any
import warnings

import h5py
import numpy as np

from openpi.execution_horizon import dataset as horizon_dataset

_FIELDS = (
    "candidate_horizons",
    "trial_valid",
    "trial_success",
    "trial_elapsed",
    "task_id",
    "episode_id",
    "decision_step",
    "root_seed",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--reference-horizon", type=int, default=10)
    parser.add_argument("--elapsed-variance-quantile", type=float, default=0.90)
    parser.add_argument("--target-trials", type=int, default=10)
    return parser


def _input_metadata(path: pathlib.Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json" if path.is_dir() else path.parent / "manifest.json"
    if manifest_path.exists():
        return dict(json.loads(manifest_path.read_text()).get("metadata", {}))
    shard = horizon_dataset.discover_shards((path,))[0]
    with h5py.File(shard, "r") as handle:
        return dict(json.loads(handle.attrs.get("metadata_json", "{}")))


def _load_fields(inputs: list[pathlib.Path]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {field: [] for field in _FIELDS}
    source_inputs: list[str] = []
    offset_cycles: list[int] = []
    expected_candidates: tuple[int, ...] | None = None
    for input_path in inputs:
        metadata = _input_metadata(input_path)
        offset_cycle = int(metadata.get("root_call_offset_cycle", 0))
        if offset_cycle <= 0:
            raise ValueError(f"Missing positive root_call_offset_cycle in {input_path} metadata.")
        for shard in horizon_dataset.discover_shards((input_path,)):
            with h5py.File(shard, "r") as handle:
                if int(handle.attrs["schema_version"]) != horizon_dataset.SCHEMA_VERSION:
                    raise ValueError(f"Relabel selection requires schema v2 shards: {shard}")
                missing = sorted(set(_FIELDS).difference(handle))
                if missing:
                    raise KeyError(f"{shard} is missing fields required for relabel selection: {missing}")
                shard_candidates = np.asarray(handle["candidate_horizons"][:], dtype=np.int64)
                if shard_candidates.ndim != 2 or not shard_candidates.shape[0]:
                    raise ValueError(f"Invalid candidate_horizons in {shard}.")
                candidate_tuple = tuple(int(value) for value in shard_candidates[0])
                if expected_candidates is None:
                    expected_candidates = candidate_tuple
                elif candidate_tuple != expected_candidates:
                    raise ValueError("All relabel-selection inputs must use identical candidate horizons.")
                if not np.all(shard_candidates == np.asarray(candidate_tuple)[None, :]):
                    raise ValueError(f"Candidate horizons vary within {shard}.")
                rows = int(shard_candidates.shape[0])
                for field in _FIELDS:
                    pieces[field].append(handle[field][:])
                source_inputs.extend([str(input_path.resolve())] * rows)
                offset_cycles.extend([offset_cycle] * rows)
    arrays = {name: np.concatenate(values, axis=0) for name, values in pieces.items()}
    return arrays, np.asarray(source_inputs, dtype=object), np.asarray(offset_cycles, dtype=np.int64)


def _split_names(arrays: dict[str, np.ndarray], manifest_path: pathlib.Path) -> np.ndarray:
    manifest = json.loads(manifest_path.read_text())
    groups = np.asarray(arrays["task_id"], dtype=np.uint64) * np.uint64(1_000_000_000)
    groups += np.asarray(arrays["episode_id"], dtype=np.uint64)
    result = np.full(groups.shape, "train", dtype=object)
    assigned = np.zeros(groups.shape, dtype=np.bool_)
    for split_name in ("train", "calibration", "validation"):
        selected_groups = np.asarray(manifest[f"{split_name}_group_ids"], dtype=np.uint64)
        selected = np.isin(groups, selected_groups)
        if np.any(assigned & selected):
            raise ValueError("Split manifest assigns at least one episode group more than once.")
        result[selected] = split_name
        assigned |= selected
    if not np.all(assigned):
        missing = np.unique(groups[~assigned]).tolist()
        raise ValueError(f"Split manifest does not assign all roots; missing groups include {missing[:5]}.")
    return result


def _metrics(
    arrays: dict[str, np.ndarray],
    *,
    reference_horizon: int,
    elapsed_variance_quantile: float,
) -> dict[str, np.ndarray | float | tuple[int, ...]]:
    candidate_rows = np.asarray(arrays["candidate_horizons"], dtype=np.int64)
    if not np.all(candidate_rows == candidate_rows[:1]):
        raise ValueError("candidate_horizons must be identical for every root.")
    candidates = tuple(int(value) for value in candidate_rows[0])
    if reference_horizon not in candidates:
        raise ValueError("reference_horizon is absent from candidate_horizons.")
    reference_index = candidates.index(reference_horizon)
    long_indices = [index for index, horizon in enumerate(candidates) if horizon > reference_horizon]
    if not long_indices:
        raise ValueError("At least one candidate must be longer than reference_horizon.")

    valid = np.asarray(arrays["trial_valid"], dtype=np.bool_)
    success = np.asarray(arrays["trial_success"], dtype=np.bool_)
    elapsed = np.asarray(arrays["trial_elapsed"], dtype=np.float64)
    paired = valid[:, reference_index : reference_index + 1] & valid[:, long_indices]
    reference_success = success[:, reference_index : reference_index + 1]
    danger = paired & reference_success & ~success[:, long_indices]
    rescue = paired & ~reference_success & success[:, long_indices]
    danger_count = np.sum(danger, axis=-1, dtype=np.int64)
    rescue_count = np.sum(rescue, axis=-1, dtype=np.int64)

    paired_delta = np.where(
        paired,
        elapsed[:, long_indices] - elapsed[:, reference_index : reference_index + 1],
        np.nan,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        elapsed_delta_variance = np.nanvar(paired_delta, axis=-1, ddof=1)
    maximum_variance = np.nanmax(elapsed_delta_variance, axis=-1)
    finite_variance = maximum_variance[np.isfinite(maximum_variance)]
    if not finite_variance.size:
        raise ValueError("No finite paired elapsed variance is available for relabel selection.")
    variance_threshold = float(np.quantile(finite_variance, elapsed_variance_quantile))
    any_danger = np.any(danger_count > 0, axis=-1)
    any_rescue = np.any(rescue_count > 0, axis=-1)
    high_variance = maximum_variance >= variance_threshold
    selected = any_danger | any_rescue | high_variance
    return {
        "candidates": candidates,
        "long_indices": np.asarray(long_indices, dtype=np.int64),
        "danger_count": danger_count,
        "rescue_count": rescue_count,
        "elapsed_delta_variance": elapsed_delta_variance,
        "maximum_elapsed_delta_variance": maximum_variance,
        "elapsed_variance_threshold": variance_threshold,
        "any_danger": any_danger,
        "any_rescue": any_rescue,
        "high_variance": high_variance,
        "selected": selected,
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=True)}


def main(args: argparse.Namespace) -> None:
    if not 0 < args.elapsed_variance_quantile < 1:
        raise ValueError("elapsed_variance_quantile must lie in (0, 1).")
    if args.target_trials <= 0:
        raise ValueError("target_trials must be positive.")
    input_paths = [pathlib.Path(value).resolve() for value in args.dataset]
    arrays, source_inputs, offset_cycles = _load_fields(input_paths)
    splits = _split_names(arrays, pathlib.Path(args.split_manifest).resolve())
    metrics = _metrics(
        arrays,
        reference_horizon=args.reference_horizon,
        elapsed_variance_quantile=args.elapsed_variance_quantile,
    )
    candidates = metrics["candidates"]
    assert isinstance(candidates, tuple)
    long_indices = np.asarray(metrics["long_indices"], dtype=np.int64)
    selected = np.asarray(metrics["selected"], dtype=np.bool_)
    danger = np.asarray(metrics["any_danger"], dtype=np.bool_)
    rescue = np.asarray(metrics["any_rescue"], dtype=np.bool_)
    high_variance = np.asarray(metrics["high_variance"], dtype=np.bool_)
    danger_count = np.asarray(metrics["danger_count"], dtype=np.int64)
    rescue_count = np.asarray(metrics["rescue_count"], dtype=np.int64)
    elapsed_delta_variance = np.asarray(metrics["elapsed_delta_variance"], dtype=np.float64)
    maximum_variance = np.asarray(metrics["maximum_elapsed_delta_variance"], dtype=np.float64)

    identity = np.stack(
        [
            np.asarray(arrays["task_id"], dtype=np.uint64),
            np.asarray(arrays["episode_id"], dtype=np.uint64),
            np.asarray(arrays["decision_step"], dtype=np.uint64),
            np.asarray(arrays["root_seed"], dtype=np.uint64),
        ],
        axis=-1,
    )
    if np.unique(identity, axis=0).shape[0] != identity.shape[0]:
        raise ValueError("Input data contains duplicate exact roots.")

    records = []
    for index in np.flatnonzero(selected):
        reasons = []
        if danger[index]:
            reasons.append("dangerous_long")
        if rescue[index]:
            reasons.append("long_rescue")
        if high_variance[index]:
            reasons.append("elapsed_high_variance")
        records.append(
            {
                "task_id": int(arrays["task_id"][index]),
                "episode_id": int(arrays["episode_id"][index]),
                "decision_step": int(arrays["decision_step"][index]),
                "root_seed": int(arrays["root_seed"][index]),
                "split": str(splits[index]),
                "source_input": str(source_inputs[index]),
                "root_call_offset_cycle": int(offset_cycles[index]),
                "reasons": reasons,
                "current_trials_by_h": np.sum(arrays["trial_valid"][index], axis=-1).astype(int).tolist(),
                "target_trials": int(args.target_trials),
                "danger_count_by_long_h": danger_count[index].astype(int).tolist(),
                "rescue_count_by_long_h": rescue_count[index].astype(int).tolist(),
                "elapsed_delta_variance_by_long_h": elapsed_delta_variance[index].tolist(),
                "maximum_elapsed_delta_variance": float(maximum_variance[index]),
            }
        )
    records.sort(key=lambda row: (row["task_id"], row["episode_id"], row["decision_step"], row["root_seed"]))
    selected_tasks = np.asarray([record["task_id"] for record in records], dtype=np.int64)
    selected_splits = np.asarray([record["split"] for record in records], dtype=object)
    payload = {
        "status": "complete",
        "semantics": (
            "Development relabel manifest. Selection uses observed paired outcomes and must not be treated as an "
            "untouched final test set."
        ),
        "dataset_inputs": [str(path) for path in input_paths],
        "split_manifest": str(pathlib.Path(args.split_manifest).resolve()),
        "candidate_horizons": list(candidates),
        "long_horizons": [int(candidates[index]) for index in long_indices],
        "reference_horizon": int(args.reference_horizon),
        "elapsed_variance_quantile": float(args.elapsed_variance_quantile),
        "elapsed_variance_threshold": float(metrics["elapsed_variance_threshold"]),
        "target_trials": int(args.target_trials),
        "num_input_roots": len(selected),
        "num_selected_roots": int(np.sum(selected)),
        "num_dangerous_roots": int(np.sum(danger)),
        "num_rescue_roots": int(np.sum(rescue)),
        "num_danger_and_rescue_roots": int(np.sum(danger & rescue)),
        "num_high_variance_roots": int(np.sum(high_variance)),
        "num_high_variance_only_roots": int(np.sum(high_variance & ~danger & ~rescue)),
        "selected_count_by_task": _counts(selected_tasks),
        "selected_count_by_split": _counts(selected_splits),
        "records": records,
    }
    output_path = pathlib.Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
