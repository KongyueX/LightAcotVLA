"""Interruption-safe branched state labels for persistent Action-CoT updates.

Each record is one simulator root.  All branches start from the exact same
MuJoCo snapshot and use the same policy seed, so differences in fresh EAR/IAR
targets are attributable to the branch state instead of teacher sampling
noise.  Train/validation/test consumers must split by root (or episode), never
by individual branches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = 1

BRANCH_NAMES = (
    "nominal",
    "packet_fault",
    "action_scale_down",
    "action_scale_up",
    "translation_pulse",
    "gripper_fault",
)
BRANCH_IDS = {name: index for index, name in enumerate(BRANCH_NAMES)}


@dataclasses.dataclass(frozen=True)
class BranchedDatasetShape:
    num_branches: int = len(BRANCH_NAMES)
    num_cameras: int = 2
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    state_dim: int = 32
    coarse_horizon: int = 15
    action_horizon: int = 10
    action_dim: int = 32
    env_action_dim: int = 7
    iar_tokens: int = 18
    iar_dim: int = 1024
    max_executed_steps: int = 4

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field.name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive.")
        if self.num_branches != len(BRANCH_NAMES):
            raise ValueError(
                f"num_branches must match the canonical branch protocol ({len(BRANCH_NAMES)})."
            )


DEFAULT_DATASET_SHAPE = BranchedDatasetShape()

_SCALAR_SPECS: dict[str, np.dtype] = {
    "root_id": np.dtype(np.uint32),
    "task_id": np.dtype(np.int16),
    "episode_id": np.dtype(np.int32),
    "decision_step": np.dtype(np.int32),
    "policy_seed": np.dtype(np.uint32),
}

_ARRAY_SPECS: dict[str, tuple[np.dtype, tuple[str, ...]]] = {
    "branch_ids": (np.dtype(np.uint8), ("num_branches",)),
    "branch_steps": (np.dtype(np.uint8), ("num_branches",)),
    "branch_valid": (np.dtype(np.bool_), ("num_branches",)),
    "endpoint_done": (np.dtype(np.bool_), ("num_branches",)),
    "branch_strength": (np.dtype(np.float32), ("num_branches",)),
    "physics_delta_l2": (np.dtype(np.float32), ("num_branches",)),
    "anchor_images": (
        np.dtype(np.uint8),
        ("num_cameras", "image_height", "image_width", "image_channels"),
    ),
    "current_images": (
        np.dtype(np.uint8),
        ("num_branches", "num_cameras", "image_height", "image_width", "image_channels"),
    ),
    "anchor_state": (np.dtype(np.float16), ("state_dim",)),
    "current_state": (np.dtype(np.float16), ("num_branches", "state_dim")),
    "cached_ear": (np.dtype(np.float16), ("coarse_horizon", "action_dim")),
    "fresh_ear": (
        np.dtype(np.float16),
        ("num_branches", "coarse_horizon", "action_dim"),
    ),
    "cached_iar": (np.dtype(np.float16), ("iar_tokens", "iar_dim")),
    "fresh_iar": (
        np.dtype(np.float16),
        ("num_branches", "iar_tokens", "iar_dim"),
    ),
    "cached_actions": (np.dtype(np.float16), ("action_horizon", "action_dim")),
    "fresh_actions": (
        np.dtype(np.float16),
        ("num_branches", "action_horizon", "action_dim"),
    ),
    "cached_actions_env": (np.dtype(np.float32), ("action_horizon", "env_action_dim")),
    "fresh_actions_env": (
        np.dtype(np.float32),
        ("num_branches", "action_horizon", "env_action_dim"),
    ),
    "executed_actions": (
        np.dtype(np.float32),
        ("num_branches", "max_executed_steps", "env_action_dim"),
    ),
    "executed_valid": (
        np.dtype(np.bool_),
        ("num_branches", "max_executed_steps"),
    ),
}

FIELD_NAMES = (*_SCALAR_SPECS, *_ARRAY_SPECS)


def _shape_for(names: tuple[str, ...], shape: BranchedDatasetShape) -> tuple[int, ...]:
    return tuple(getattr(shape, name) for name in names)


def _normalise_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(metadata or {}), sort_keys=True))
    if not isinstance(payload, dict):
        raise TypeError("metadata must serialise to a JSON object.")
    return payload


def _coerce_record(
    record: Mapping[str, Any],
    shape: BranchedDatasetShape,
) -> dict[str, np.ndarray]:
    required = set(FIELD_NAMES)
    missing = sorted(required.difference(record))
    unexpected = sorted(set(record).difference(required))
    if missing or unexpected:
        raise KeyError(f"Invalid branched record fields: missing={missing}, unexpected={unexpected}.")

    result: dict[str, np.ndarray] = {}
    for name, dtype in _SCALAR_SPECS.items():
        value = np.asarray(record[name])
        if value.shape != () or value.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{name} must be an integer scalar.")
        integer = int(value.item())
        bounds = np.iinfo(dtype)
        if integer < bounds.min or integer > bounds.max:
            raise ValueError(f"{name}={integer} cannot be represented as {dtype}.")
        result[name] = np.asarray(integer, dtype=dtype)

    for name, (dtype, shape_names) in _ARRAY_SPECS.items():
        value = np.asarray(record[name])
        expected = _shape_for(shape_names, shape)
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected}.")
        if name in {"branch_valid", "endpoint_done", "executed_valid"}:
            if value.dtype != np.dtype(np.bool_):
                raise TypeError(f"{name} must have boolean dtype.")
        elif name in {"anchor_images", "current_images"}:
            if value.dtype.kind not in {"i", "u"} or np.any(value < 0) or np.any(value > 255):
                raise ValueError(f"{name} must contain integer pixels in [0, 255].")
        elif value.dtype.kind not in {"f", "i", "u"} or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite real values.")
        converted = np.asarray(value, dtype=dtype)
        if dtype.kind == "f" and not np.all(np.isfinite(converted)):
            raise ValueError(f"{name} cannot be represented as finite {dtype}.")
        result[name] = converted

    expected_ids = np.arange(shape.num_branches, dtype=np.uint8)
    if not np.array_equal(result["branch_ids"], expected_ids):
        raise ValueError(f"branch_ids must equal the canonical ordering {expected_ids.tolist()}.")
    if np.any(result["branch_steps"] > shape.max_executed_steps):
        raise ValueError("branch_steps exceeds max_executed_steps.")
    valid_counts = np.sum(result["executed_valid"], axis=1)
    if not np.array_equal(valid_counts, result["branch_steps"].astype(np.int64)):
        raise ValueError("executed_valid counts must equal branch_steps.")
    return result


def _manifest_payload(
    output_dir: pathlib.Path,
    shape: BranchedDatasetShape,
    records_per_shard: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    shards = sorted(output_dir.glob("shard-*.h5"))
    num_records = 0
    for shard in shards:
        with h5py.File(shard, "r") as handle:
            num_records += int(handle["root_id"].shape[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "shape": dataclasses.asdict(shape),
        "records_per_shard": records_per_shard,
        "num_shards": len(shards),
        "num_records": num_records,
        "branch_names": list(BRANCH_NAMES),
        "metadata": dict(metadata),
    }


class ShardedBranchedWriter:
    """Append complete roots to atomic HDF5 shards."""

    def __init__(
        self,
        output_dir: pathlib.Path | str,
        *,
        shape: BranchedDatasetShape = DEFAULT_DATASET_SHAPE,
        records_per_shard: int = 16,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(records_per_shard, bool) or not isinstance(records_per_shard, int):
            raise TypeError("records_per_shard must be an integer.")
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive.")
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shape = shape
        self.records_per_shard = records_per_shard
        self.metadata = _normalise_metadata(metadata)
        self._buffer: list[dict[str, np.ndarray]] = []

        existing = sorted(self.output_dir.glob("shard-*.h5"))
        indices = [int(path.stem.removeprefix("shard-")) for path in existing]
        self._next_shard = max(indices) + 1 if indices else 0
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError("Cannot append to a different branched schema version.")
            if BranchedDatasetShape(**manifest["shape"]) != shape:
                raise ValueError("Branched dataset shape differs from the existing manifest.")
            if manifest.get("metadata", {}) != self.metadata:
                raise ValueError("Branched export metadata differs; use a new output directory.")
            if int(manifest.get("records_per_shard", -1)) != records_per_shard:
                raise ValueError("records_per_shard differs from the existing manifest.")
        if existing:
            self._write_manifest()

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
            for name, dtype in _SCALAR_SPECS.items():
                values = np.stack([record[name] for record in records])
                handle.create_dataset(name, data=values.astype(dtype, copy=False), compression="lzf", shuffle=True)
            for name, (dtype, _) in _ARRAY_SPECS.items():
                values = np.stack([record[name] for record in records])
                handle.create_dataset(name, data=values.astype(dtype, copy=False), compression="lzf", shuffle=True)
            handle.flush()
        temporary.replace(target)
        self._buffer = []
        self._next_shard += 1
        self._write_manifest()
        return target

    def _write_manifest(self) -> None:
        payload = _manifest_payload(
            self.output_dir,
            self.shape,
            self.records_per_shard,
            self.metadata,
        )
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "manifest.json")

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> ShardedBranchedWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
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
            raise FileNotFoundError(f"Branched input is not a shard/directory: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in shards))
    if not unique:
        raise FileNotFoundError(f"No branched HDF5 shards found under {inputs}.")
    return unique


def load_branched_arrays(
    inputs: Sequence[pathlib.Path | str],
    *,
    fields: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    selected = tuple(FIELD_NAMES if fields is None else fields)
    unknown = sorted(set(selected).difference(FIELD_NAMES))
    if unknown:
        raise KeyError(f"Unknown branched fields: {unknown}.")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in selected}
    expected_shape: BranchedDatasetShape | None = None
    expected_metadata: dict[str, Any] | None = None
    for shard in discover_shards(inputs):
        with h5py.File(shard, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"Unsupported branched schema in {shard}.")
            current_shape = BranchedDatasetShape(**json.loads(handle.attrs["shape_json"]))
            current_metadata = json.loads(handle.attrs["metadata_json"])
            if expected_shape is None:
                expected_shape = current_shape
                expected_metadata = current_metadata
            elif current_shape != expected_shape or current_metadata != expected_metadata:
                raise ValueError(f"Branched shard protocol mismatch in {shard}.")
            for name in selected:
                pieces[name].append(handle[name][:])
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}
