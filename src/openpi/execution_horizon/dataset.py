"""Compact sharded HDF5 storage for execution-horizon counterfactuals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class DatasetShape:
    prefix_feature_dim: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    coarse_horizon: int = 15
    action_horizon: int = 10


_FIXED_SPECS: dict[str, tuple[np.dtype, tuple[str, ...]]] = {
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


def _shape_for(spec: tuple[str, ...], shape: DatasetShape) -> tuple[int, ...]:
    return tuple(int(getattr(shape, name)) for name in spec)


def _coerce_record(record: Mapping[str, Any], shape: DatasetShape) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    missing = sorted(set(_FIXED_SPECS).difference(record))
    if missing:
        raise KeyError(f"Counterfactual record is missing required fields: {missing}")
    for name, (dtype, shape_names) in _FIXED_SPECS.items():
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
        shape: DatasetShape = DatasetShape(),
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
            for name, (dtype, _) in _FIXED_SPECS.items():
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

    def __enter__(self) -> "ShardedCounterfactualWriter":
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


def load_counterfactual_arrays(
    inputs: Sequence[pathlib.Path | str],
    *,
    include_physics: bool = False,
) -> dict[str, np.ndarray]:
    """Load fixed-size training arrays; physics is excluded by default."""
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in _FIXED_SPECS}
    if include_physics:
        pieces["physics_state"] = []
    for shard in discover_shards(inputs):
        with h5py.File(shard, "r") as handle:
            if int(handle.attrs["schema_version"]) != SCHEMA_VERSION:
                raise ValueError(f"Unsupported schema in {shard}: {handle.attrs['schema_version']}")
            for name in pieces:
                if name == "physics_state":
                    pieces[name].extend(np.asarray(row, dtype=np.float64) for row in handle[name])
                else:
                    pieces[name].append(handle[name][:])
    result: dict[str, np.ndarray] = {}
    for name, values in pieces.items():
        if name == "physics_state":
            result[name] = np.asarray(values, dtype=object)
        else:
            result[name] = np.concatenate(values, axis=0)
    return result


def sampling_weights(
    arrays: Mapping[str, np.ndarray],
    *,
    focus_task_ids: Iterable[int] = (7, 8),
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
    has_failed_branch = ~np.all(np.asarray(arrays["branch_success"], dtype=np.bool_), axis=-1)
    weights *= np.where(has_failed_branch, failure_multiplier, 1.0)
    weights = np.asarray(weights, dtype=np.float64)
    return weights / weights.sum()
