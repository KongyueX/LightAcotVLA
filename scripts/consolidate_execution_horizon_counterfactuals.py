"""Consolidate counterfactual roots and replace selected roots with denser relabels."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import h5py
import numpy as np

from openpi.execution_horizon import dataset as horizon_dataset

_IDENTITY_FIELDS = ("task_id", "episode_id", "decision_step", "root_seed")
_TRIAL_FIELDS = (
    "trial_success",
    "trial_timeout",
    "trial_remaining_steps",
    "trial_remaining_calls",
    "trial_elapsed",
    "trial_valid",
)
_STATIC_FLOAT_FIELDS = (
    "prefix_feature",
    "state",
    "coarse_actions",
    "final_actions",
    "previous_actions",
    "budget_balance",
    "episode_progress",
    "final_risk",
    "action_cot_risk",
    "fused_risk",
    "prefix_tokens",
)
_STATIC_EXACT_FIELDS = (
    "previous_h",
    "previous_valid",
    "event_mask",
    "risk_valid",
    "raw_h",
    "candidate_horizons",
    "prefix_token_mask",
)


@dataclasses.dataclass(frozen=True)
class RowRef:
    shard: pathlib.Path
    row: int
    shape: horizon_dataset.DatasetShape
    source_input: pathlib.Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="append", required=True)
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records-per-shard", type=int, default=10)
    parser.add_argument("--expected-roots", type=int, default=None)
    parser.add_argument("--expected-replacements", type=int, default=None)
    parser.add_argument("--allow-new-overlay-roots", action="store_true")
    parser.add_argument("--skip-overlap-verification", action="store_true")
    return parser


def _shape(handle: h5py.File, shard: pathlib.Path) -> horizon_dataset.DatasetShape:
    if int(handle.attrs["schema_version"]) != horizon_dataset.SCHEMA_VERSION:
        raise ValueError(f"Consolidation requires schema v2 shards: {shard}")
    values = json.loads(handle.attrs["shape_json"])
    return horizon_dataset.DatasetShape(**values)


def _root_key(handle: h5py.File, row: int) -> tuple[int, int, int, int]:
    return tuple(int(np.asarray(handle[name][row]).item()) for name in _IDENTITY_FIELDS)


def _compatible_shape(
    expected: horizon_dataset.DatasetShape | None,
    observed: horizon_dataset.DatasetShape,
) -> horizon_dataset.DatasetShape:
    if expected is None:
        return observed
    expected_values = dataclasses.asdict(expected)
    observed_values = dataclasses.asdict(observed)
    expected_values.pop("max_trials")
    observed_values.pop("max_trials")
    if expected_values != observed_values:
        raise ValueError(f"Counterfactual shapes differ beyond max_trials: {expected} vs {observed}.")
    return expected


def _index_inputs(inputs: list[pathlib.Path]) -> tuple[dict[tuple[int, int, int, int], RowRef], int]:
    rows: dict[tuple[int, int, int, int], RowRef] = {}
    maximum_trials = 0
    expected_shape: horizon_dataset.DatasetShape | None = None
    for input_path in inputs:
        for shard in horizon_dataset.discover_shards((input_path,)):
            with h5py.File(shard, "r") as handle:
                shape = _shape(handle, shard)
                expected_shape = _compatible_shape(expected_shape, shape)
                maximum_trials = max(maximum_trials, shape.max_trials)
                num_rows = int(handle["task_id"].shape[0])
                for row in range(num_rows):
                    key = _root_key(handle, row)
                    if key in rows:
                        raise ValueError(f"Duplicate exact root within one input tier: {key}.")
                    rows[key] = RowRef(shard=shard, row=row, shape=shape, source_input=input_path)
    if not rows:
        raise ValueError("No counterfactual roots were found.")
    return rows, maximum_trials


def _read_field(reference: RowRef, field: str) -> np.ndarray:
    with h5py.File(reference.shard, "r") as handle:
        return np.asarray(handle[field][reference.row])


def _read_record(reference: RowRef, target_shape: horizon_dataset.DatasetShape) -> dict[str, Any]:
    record: dict[str, Any] = {}
    target_specs = horizon_dataset._fixed_specs(target_shape)  # noqa: SLF001
    with h5py.File(reference.shard, "r") as handle:
        for field, (dtype, shape_names) in target_specs.items():
            if field not in handle:
                raise KeyError(f"{reference.shard} is missing required field {field!r}.")
            value = np.asarray(handle[field][reference.row])
            target_field_shape = horizon_dataset._shape_for(shape_names, target_shape)  # noqa: SLF001
            if field in _TRIAL_FIELDS and value.shape[-1] < target_field_shape[-1]:
                fill = np.nan if field == "trial_elapsed" else 0
                padded = np.full(target_field_shape, fill, dtype=dtype)
                padded[..., : value.shape[-1]] = value.astype(dtype, copy=False)
                value = padded
            if value.shape != target_field_shape:
                raise ValueError(
                    f"{field} in {reference.shard} row {reference.row} has shape {value.shape}; "
                    f"expected {target_field_shape}."
                )
            record[field] = value
        record["physics_state"] = np.asarray(handle["physics_state"][reference.row], dtype=np.float64)
    return record


def _verify_overlay(base: RowRef, overlay: RowRef) -> None:
    base_record = _read_record(base, base.shape)
    overlay_record = _read_record(overlay, overlay.shape)
    for field in _STATIC_FLOAT_FIELDS:
        if field not in base_record and field not in overlay_record:
            continue
        if field not in base_record or field not in overlay_record:
            raise ValueError(f"Overlay static field availability differs for {field!r}.")
        if not np.allclose(base_record[field], overlay_record[field], rtol=2e-3, atol=2e-3, equal_nan=True):
            raise ValueError(f"Overlay root does not reproduce static field {field!r} for {_root_identity(base)}.")
    for field in _STATIC_EXACT_FIELDS:
        if field not in base_record and field not in overlay_record:
            continue
        if field not in base_record or field not in overlay_record:
            raise ValueError(f"Overlay static field availability differs for {field!r}.")
        if not np.array_equal(base_record[field], overlay_record[field]):
            raise ValueError(f"Overlay root does not reproduce static field {field!r} for {_root_identity(base)}.")
    if not np.allclose(
        base_record["physics_state"],
        overlay_record["physics_state"],
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    ):
        raise ValueError(f"Overlay root does not reproduce physics_state for {_root_identity(base)}.")

    common = min(base.shape.max_trials, overlay.shape.max_trials)
    base_valid = np.asarray(base_record["trial_valid"])[..., :common]
    overlay_valid = np.asarray(overlay_record["trial_valid"])[..., :common]
    if not np.array_equal(base_valid, overlay_valid):
        raise ValueError(f"Overlay trial validity differs for {_root_identity(base)}.")
    for field in (
        "trial_success",
        "trial_timeout",
        "trial_remaining_steps",
        "trial_remaining_calls",
    ):
        base_values = np.asarray(base_record[field])[..., :common]
        overlay_values = np.asarray(overlay_record[field])[..., :common]
        if not np.array_equal(base_values[base_valid], overlay_values[overlay_valid]):
            raise ValueError(f"Overlay paired outcomes differ in {field!r} for {_root_identity(base)}.")


def _root_identity(reference: RowRef) -> tuple[int, int, int, int]:
    with h5py.File(reference.shard, "r") as handle:
        return _root_key(handle, reference.row)


def _trial_count_summary(output_dir: pathlib.Path) -> dict[str, Any]:
    minimum: int | None = None
    maximum = 0
    histogram: dict[int, int] = {}
    task_counts: dict[int, int] = {}
    for shard in horizon_dataset.discover_shards((output_dir,)):
        with h5py.File(shard, "r") as handle:
            counts = np.asarray(handle["trial_count"][:], dtype=np.int64)
            if not np.all(counts == counts[:, :1]):
                raise ValueError(f"Candidate trial counts are not paired within roots in {shard}.")
            root_counts = counts[:, 0]
            for value in root_counts:
                count = int(value)
                histogram[count] = histogram.get(count, 0) + 1
                minimum = count if minimum is None else min(minimum, count)
                maximum = max(maximum, count)
            tasks, task_frequency = np.unique(np.asarray(handle["task_id"][:], dtype=np.int64), return_counts=True)
            for task, frequency in zip(tasks, task_frequency, strict=True):
                task_counts[int(task)] = task_counts.get(int(task), 0) + int(frequency)
    return {
        "minimum_trials_per_candidate": minimum,
        "maximum_trials_per_candidate": maximum,
        "root_count_by_trial_count": {str(key): value for key, value in sorted(histogram.items())},
        "root_count_by_task": {str(key): value for key, value in sorted(task_counts.items())},
    }


def main(args: argparse.Namespace) -> None:
    if args.records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive.")
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    base_inputs = [pathlib.Path(value).resolve() for value in args.base]
    overlay_inputs = [pathlib.Path(value).resolve() for value in args.overlay]
    base_rows, base_max_trials = _index_inputs(base_inputs)
    overlay_rows: dict[tuple[int, int, int, int], RowRef] = {}
    overlay_max_trials = 0
    if overlay_inputs:
        overlay_rows, overlay_max_trials = _index_inputs(overlay_inputs)
    expected_shape = next(iter(base_rows.values())).shape
    for reference in list(base_rows.values()) + list(overlay_rows.values()):
        _compatible_shape(expected_shape, reference.shape)
    target_shape = dataclasses.replace(expected_shape, max_trials=max(base_max_trials, overlay_max_trials))

    replacements = 0
    for key, overlay in overlay_rows.items():
        base = base_rows.get(key)
        if base is None:
            if not args.allow_new_overlay_roots:
                raise ValueError(f"Overlay root is absent from base inputs: {key}.")
        else:
            replacements += 1
            if not args.skip_overlap_verification:
                _verify_overlay(base, overlay)
        base_rows[key] = overlay
    if args.expected_roots is not None and len(base_rows) != args.expected_roots:
        raise ValueError(f"Expected {args.expected_roots} consolidated roots, found {len(base_rows)}.")
    if args.expected_replacements is not None and replacements != args.expected_replacements:
        raise ValueError(f"Expected {args.expected_replacements} replacements, found {replacements}.")

    metadata = {
        "operation": "last-overlay-wins exact-root consolidation",
        "base_inputs": [str(path) for path in base_inputs],
        "overlay_inputs": [str(path) for path in overlay_inputs],
        "num_replacements": replacements,
        "overlap_verified": not args.skip_overlap_verification,
    }
    ordered = sorted(base_rows.items())
    with horizon_dataset.ShardedCounterfactualWriter(
        output_dir,
        shape=target_shape,
        records_per_shard=args.records_per_shard,
        metadata=metadata,
    ) as writer:
        for _, reference in ordered:
            writer.append(_read_record(reference, target_shape))

    report = {
        "status": "complete",
        "output_dir": str(output_dir),
        "num_roots": len(base_rows),
        "num_replacements": replacements,
        "target_shape": dataclasses.asdict(target_shape),
        "base_inputs": [str(path) for path in base_inputs],
        "overlay_inputs": [str(path) for path in overlay_inputs],
        **_trial_count_summary(output_dir),
    }
    (output_dir / "consolidation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(build_parser().parse_args())
