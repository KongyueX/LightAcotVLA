"""Compact sharded HDF5 storage for execution-horizon counterfactuals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class DatasetShape:
    prefix_feature_dim: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    coarse_horizon: int = 15
    action_horizon: int = 10
    candidate_horizons: tuple[int, ...] = tuple(range(1, 11))
    max_trials: int = 1
    prefix_token_count: int = 0

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_horizons)
        object.__setattr__(self, "candidate_horizons", candidates)
        if candidates != tuple(sorted(set(candidates))) or not candidates:
            raise ValueError("candidate_horizons must be non-empty, sorted, and unique.")
        if candidates[0] <= 0 or candidates[-1] > self.action_horizon:
            raise ValueError("candidate_horizons must lie within the stored final-action horizon.")
        for name in (
            "prefix_feature_dim",
            "state_dim",
            "action_dim",
            "coarse_horizon",
            "action_horizon",
            "max_trials",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.prefix_token_count < 0:
            raise ValueError("prefix_token_count must be non-negative.")

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_horizons)


DEFAULT_DATASET_SHAPE = DatasetShape()


_LEGACY_FIXED_SPECS: dict[str, tuple[np.dtype, tuple[str, ...]]] = {
    "prefix_feature": (np.dtype(np.float16), ("prefix_feature_dim",)),
    "state": (np.dtype(np.float16), ("state_dim",)),
    "coarse_actions": (np.dtype(np.float16), ("coarse_horizon", "action_dim")),
    "final_actions": (np.dtype(np.float16), ("action_horizon", "action_dim")),
    "previous_actions": (np.dtype(np.float16), ("action_horizon", "action_dim")),
    "previous_h": (np.dtype(np.uint8), ()),
    "previous_valid": (np.dtype(np.bool_), ()),
    "budget_balance": (np.dtype(np.float16), ()),
    "episode_progress": (np.dtype(np.float16), ()),
    "final_risk": (np.dtype(np.float16), ("action_horizon",)),
    "action_cot_risk": (np.dtype(np.float16), ("action_horizon",)),
    "fused_risk": (np.dtype(np.float16), ("action_horizon",)),
    "event_mask": (np.dtype(np.bool_), ("action_horizon",)),
    "risk_valid": (np.dtype(np.bool_), ("action_horizon",)),
    "raw_h": (np.dtype(np.uint8), ()),
    "branch_success": (np.dtype(np.bool_), ("action_horizon",)),
    "branch_timeout": (np.dtype(np.bool_), ("action_horizon",)),
    "remaining_steps": (np.dtype(np.uint16), ("action_horizon",)),
    "remaining_calls": (np.dtype(np.uint16), ("action_horizon",)),
    "branch_valid": (np.dtype(np.bool_), ("action_horizon",)),
    "task_id": (np.dtype(np.uint8), ()),
    "episode_id": (np.dtype(np.uint32), ()),
    "decision_step": (np.dtype(np.uint16), ()),
    "root_seed": (np.dtype(np.uint64), ()),
    "source_iteration": (np.dtype(np.uint8), ()),
}


def _fixed_specs(shape: DatasetShape) -> dict[str, tuple[np.dtype, tuple[str, ...]]]:
    specs: dict[str, tuple[np.dtype, tuple[str, ...]]] = {
        "prefix_feature": (np.dtype(np.float16), ("prefix_feature_dim",)),
        "state": (np.dtype(np.float16), ("state_dim",)),
        "coarse_actions": (np.dtype(np.float16), ("coarse_horizon", "action_dim")),
        "final_actions": (np.dtype(np.float16), ("action_horizon", "action_dim")),
        "previous_actions": (np.dtype(np.float16), ("action_horizon", "action_dim")),
        "previous_h": (np.dtype(np.uint8), ()),
        "previous_valid": (np.dtype(np.bool_), ()),
        "budget_balance": (np.dtype(np.float16), ()),
        "episode_progress": (np.dtype(np.float16), ()),
        "final_risk": (np.dtype(np.float16), ("action_horizon",)),
        "action_cot_risk": (np.dtype(np.float16), ("action_horizon",)),
        "fused_risk": (np.dtype(np.float16), ("action_horizon",)),
        "event_mask": (np.dtype(np.bool_), ("action_horizon",)),
        "risk_valid": (np.dtype(np.bool_), ("action_horizon",)),
        "hazard_event_count": (np.dtype(np.uint16), ("action_horizon",)),
        "hazard_at_risk_count": (np.dtype(np.uint16), ("action_horizon",)),
        "raw_h": (np.dtype(np.uint8), ()),
        "candidate_horizons": (np.dtype(np.uint8), ("num_candidates",)),
        # Repeat-0 projections remain for old analysis utilities. Count-aware
        # training uses the fields below instead.
        "branch_success": (np.dtype(np.bool_), ("num_candidates",)),
        "branch_timeout": (np.dtype(np.bool_), ("num_candidates",)),
        "remaining_steps": (np.dtype(np.uint16), ("num_candidates",)),
        "remaining_calls": (np.dtype(np.uint16), ("num_candidates",)),
        "branch_valid": (np.dtype(np.bool_), ("num_candidates",)),
        "success_count": (np.dtype(np.uint16), ("num_candidates",)),
        "timeout_count": (np.dtype(np.uint16), ("num_candidates",)),
        "trial_count": (np.dtype(np.uint16), ("num_candidates",)),
        "remaining_steps_mean": (np.dtype(np.float32), ("num_candidates",)),
        "remaining_steps_variance": (np.dtype(np.float32), ("num_candidates",)),
        "remaining_calls_mean": (np.dtype(np.float32), ("num_candidates",)),
        "remaining_calls_variance": (np.dtype(np.float32), ("num_candidates",)),
        "elapsed_mean": (np.dtype(np.float32), ("num_candidates",)),
        "elapsed_variance": (np.dtype(np.float32), ("num_candidates",)),
        "trial_success": (np.dtype(np.bool_), ("num_candidates", "max_trials")),
        "trial_timeout": (np.dtype(np.bool_), ("num_candidates", "max_trials")),
        "trial_remaining_steps": (np.dtype(np.uint16), ("num_candidates", "max_trials")),
        "trial_remaining_calls": (np.dtype(np.uint16), ("num_candidates", "max_trials")),
        "trial_elapsed": (np.dtype(np.float32), ("num_candidates", "max_trials")),
        "trial_valid": (np.dtype(np.bool_), ("num_candidates", "max_trials")),
        # H10-success/long-H-failure pair counts. Arrays use candidate width so
        # arbitrary long-H sets remain representable.
        "dangerous_long_count": (np.dtype(np.uint16), ("num_candidates",)),
        "paired_trial_count": (np.dtype(np.uint16), ("num_candidates",)),
        "task_id": (np.dtype(np.uint8), ()),
        "episode_id": (np.dtype(np.uint32), ()),
        "decision_step": (np.dtype(np.uint16), ()),
        "root_seed": (np.dtype(np.uint64), ()),
        "source_iteration": (np.dtype(np.uint8), ()),
    }
    if shape.prefix_token_count:
        specs.update(
            {
                "prefix_tokens": (
                    np.dtype(np.float16),
                    ("prefix_token_count", "prefix_feature_dim"),
                ),
                "prefix_token_mask": (np.dtype(np.bool_), ("prefix_token_count",)),
            }
        )
    return specs


def _shape_for(spec: tuple[str, ...], shape: DatasetShape) -> tuple[int, ...]:
    return tuple(int(getattr(shape, name)) for name in spec)


def _coerce_record(record: Mapping[str, Any], shape: DatasetShape) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    specs = _fixed_specs(shape)
    missing = sorted(set(specs).difference(record))
    if missing:
        raise KeyError(f"Counterfactual record is missing required fields: {missing}")
    for name, (dtype, shape_names) in specs.items():
        value = np.asarray(record[name], dtype=dtype)
        expected_shape = _shape_for(shape_names, shape)
        if value.shape != expected_shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected_shape}.")
        result[name] = value
    physics_state = np.asarray(record["physics_state"], dtype=np.float64).reshape((-1,))
    if not physics_state.size:
        raise ValueError("physics_state must not be empty.")
    result["physics_state"] = physics_state
    return result


class ShardedCounterfactualWriter:
    """Append-only writer; one record corresponds to one MuJoCo root state."""

    def __init__(
        self,
        output_dir: pathlib.Path | str,
        *,
        shape: DatasetShape = DEFAULT_DATASET_SHAPE,
        records_per_shard: int = 1024,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shape = shape
        self.records_per_shard = records_per_shard
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive.")
        self.metadata = dict(metadata or {})
        self._buffer: list[dict[str, np.ndarray]] = []
        existing = sorted(self.output_dir.glob("shard-*.h5"))
        self._next_shard = len(existing)
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if int(manifest["schema_version"]) != SCHEMA_VERSION:
                raise ValueError(f"Cannot append to schema version {manifest['schema_version']}.")
            existing_shape = DatasetShape(**manifest["shape"])
            if existing_shape != shape:
                raise ValueError(f"Dataset shape mismatch: existing={existing_shape}, requested={shape}.")

    def append(self, record: Mapping[str, Any]) -> None:
        self._buffer.append(_coerce_record(record, self.shape))
        if len(self._buffer) >= self.records_per_shard:
            self.flush()

    def flush(self) -> pathlib.Path | None:
        if not self._buffer:
            return None
        target = self.output_dir / f"shard-{self._next_shard:05d}.h5"
        temporary = target.with_suffix(".h5.tmp")
        records = self._buffer
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["shape_json"] = json.dumps(dataclasses.asdict(self.shape), sort_keys=True)
            handle.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
            for name, (dtype, _) in _fixed_specs(self.shape).items():
                values = np.stack([record[name] for record in records])
                handle.create_dataset(
                    name,
                    data=values.astype(dtype, copy=False),
                    compression="lzf",
                    shuffle=True,
                )
            variable_dtype = h5py.vlen_dtype(np.dtype(np.float64))
            physics = handle.create_dataset("physics_state", (len(records),), dtype=variable_dtype)
            for index, record in enumerate(records):
                physics[index] = record["physics_state"]
            handle.flush()
        temporary.replace(target)
        self._buffer = []
        self._next_shard += 1
        self._write_manifest()
        return target

    def _write_manifest(self) -> None:
        shards = sorted(self.output_dir.glob("shard-*.h5"))
        total_records = 0
        for shard in shards:
            with h5py.File(shard, "r") as handle:
                total_records += int(handle["task_id"].shape[0])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "shape": dataclasses.asdict(self.shape),
            "records_per_shard": self.records_per_shard,
            "num_shards": len(shards),
            "num_records": total_records,
            "metadata": self.metadata,
        }
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output_dir / "manifest.json")

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> ShardedCounterfactualWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()


def discover_shards(inputs: Sequence[pathlib.Path | str]) -> list[pathlib.Path]:
    shards: list[pathlib.Path] = []
    for item in inputs:
        path = pathlib.Path(item)
        if path.is_dir():
            shards.extend(sorted(path.glob("shard-*.h5")))
        elif path.suffix in {".h5", ".hdf5"}:
            shards.append(path)
        else:
            raise FileNotFoundError(f"Counterfactual input is not an HDF5 shard/directory: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in shards))
    if not unique:
        raise FileNotFoundError(f"No counterfactual HDF5 shards found under {inputs}.")
    return unique


def _legacy_arrays(handle: h5py.File) -> dict[str, np.ndarray]:
    arrays = {name: handle[name][:] for name in _LEGACY_FIXED_SPECS}
    num_records, num_candidates = arrays["branch_success"].shape
    valid = np.asarray(arrays["branch_valid"], dtype=np.bool_)
    success = np.asarray(arrays["branch_success"], dtype=np.bool_)
    timeout = np.asarray(arrays["branch_timeout"], dtype=np.bool_)
    arrays.update(
        {
            "candidate_horizons": np.broadcast_to(
                np.arange(1, num_candidates + 1, dtype=np.uint8),
                (num_records, num_candidates),
            ).copy(),
            "success_count": (success & valid).astype(np.uint16),
            "timeout_count": (timeout & valid).astype(np.uint16),
            "trial_count": valid.astype(np.uint16),
            "hazard_event_count": np.asarray(arrays["event_mask"], dtype=np.uint16),
            "hazard_at_risk_count": np.asarray(arrays["risk_valid"], dtype=np.uint16),
            "remaining_steps_mean": np.asarray(arrays["remaining_steps"], dtype=np.float32),
            "remaining_steps_variance": np.zeros((num_records, num_candidates), dtype=np.float32),
            "remaining_calls_mean": np.asarray(arrays["remaining_calls"], dtype=np.float32),
            "remaining_calls_variance": np.zeros((num_records, num_candidates), dtype=np.float32),
            "elapsed_mean": np.full((num_records, num_candidates), np.nan, dtype=np.float32),
            "elapsed_variance": np.full((num_records, num_candidates), np.nan, dtype=np.float32),
            "trial_success": success[..., None],
            "trial_timeout": timeout[..., None],
            "trial_remaining_steps": np.asarray(arrays["remaining_steps"], dtype=np.uint16)[..., None],
            "trial_remaining_calls": np.asarray(arrays["remaining_calls"], dtype=np.uint16)[..., None],
            "trial_elapsed": np.full((num_records, num_candidates, 1), np.nan, dtype=np.float32),
            "trial_valid": valid[..., None],
            "dangerous_long_count": np.zeros((num_records, num_candidates), dtype=np.uint16),
            "paired_trial_count": np.zeros((num_records, num_candidates), dtype=np.uint16),
        }
    )
    return arrays


def load_counterfactual_arrays(
    inputs: Sequence[pathlib.Path | str],
    *,
    include_physics: bool = False,
) -> dict[str, np.ndarray]:
    """Load fixed-size arrays, normalizing legacy v1 shards when requested alone."""

    shards = discover_shards(inputs)
    versions: set[int] = set()
    pieces: dict[str, list[np.ndarray]] = {}
    physics_rows: list[np.ndarray] = []
    expected_keys: set[str] | None = None
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with h5py.File(shard, "r") as handle:
            version = int(handle.attrs["schema_version"])
            versions.add(version)
            if version == LEGACY_SCHEMA_VERSION:
                shard_arrays = _legacy_arrays(handle)
            elif version == SCHEMA_VERSION:
                shard_arrays = {name: handle[name][:] for name in handle if name != "physics_state"}
            else:
                raise ValueError(f"Unsupported schema in {shard}: {version}")
            if expected_keys is None:
                expected_keys = set(shard_arrays)
                expected_shapes = {name: value.shape[1:] for name, value in shard_arrays.items()}
            elif set(shard_arrays) != expected_keys:
                raise ValueError("Counterfactual shards do not expose the same fixed fields.")
            for name, value in shard_arrays.items():
                if value.shape[1:] != expected_shapes[name]:
                    raise ValueError(
                        f"Counterfactual shard shape mismatch for {name}: "
                        f"expected {expected_shapes[name]}, got {value.shape[1:]}."
                    )
                pieces.setdefault(name, []).append(value)
            if include_physics:
                physics_rows.extend(np.asarray(row, dtype=np.float64) for row in handle["physics_state"])
    if len(versions) > 1:
        raise ValueError("Do not mix legacy v1 and count-aware v2 shards in one training/audit input.")
    result = {name: np.concatenate(values, axis=0) for name, values in pieces.items()}
    num_records = len(next(iter(result.values())))
    result["schema_version"] = np.full((num_records,), next(iter(versions)), dtype=np.uint8)
    if include_physics:
        result["physics_state"] = np.asarray(physics_rows, dtype=object)
    return result


def sampling_weights(
    arrays: Mapping[str, np.ndarray],
    *,
    focus_task_ids: Iterable[int] = (8, 9),
    focus_task_multiplier: float = 2.0,
    high_risk_multiplier: float = 2.0,
    gripper_multiplier: float = 1.5,
    failure_multiplier: float = 2.0,
    high_risk_quantile: float = 0.75,
) -> np.ndarray:
    """Task-balanced weights with deliberate high-value-state oversampling."""
    task_ids = np.asarray(arrays["task_id"], dtype=np.int64)
    counts = np.bincount(task_ids, minlength=int(task_ids.max(initial=0)) + 1)
    weights = 1.0 / np.maximum(counts[task_ids], 1)
    weights *= len(weights) / np.maximum(weights.sum(), 1e-12)

    focus = np.isin(task_ids, np.asarray(tuple(focus_task_ids), dtype=np.int64))
    weights *= np.where(focus, focus_task_multiplier, 1.0)
    fused_risk = np.max(np.asarray(arrays["fused_risk"], dtype=np.float32), axis=-1)
    risk_threshold = float(np.quantile(fused_risk, high_risk_quantile))
    weights *= np.where(fused_risk >= risk_threshold, high_risk_multiplier, 1.0)

    actions = np.asarray(arrays["final_actions"], dtype=np.float32)
    gripper_dim = min(6, actions.shape[-1] - 1)
    gripper_change = np.max(np.abs(np.diff(actions[..., gripper_dim], axis=1)), axis=-1)
    weights *= np.where(gripper_change >= np.quantile(gripper_change, 0.75), gripper_multiplier, 1.0)
    if "success_count" in arrays and "trial_count" in arrays:
        has_failed_branch = np.any(
            np.asarray(arrays["success_count"]) < np.asarray(arrays["trial_count"]),
            axis=-1,
        )
    else:
        has_failed_branch = ~np.all(np.asarray(arrays["branch_success"], dtype=np.bool_), axis=-1)
    weights *= np.where(has_failed_branch, failure_multiplier, 1.0)
    weights = np.asarray(weights, dtype=np.float64)
    return weights / weights.sum()
