"""Sharded four-frame windows for the fixed-ratio multi-rate ACoT pilot.

Each record contains one four-frame window.  A complete HDF5 shard is first
written to a temporary file and then atomically renamed, so readers never see
partially written shards.  Existing shards are never modified when an export
is resumed.
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


@dataclasses.dataclass(frozen=True)
class MultirateDatasetShape:
    window_size: int = 4
    num_cameras: int = 2
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    state_dim: int = 32
    coarse_horizon: int = 15
    action_dim: int = 32
    iar_tokens: int = 18
    iar_dim: int = 1024

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field.name} must be an integer, got {type(value).__name__}.")
            if value <= 0:
                raise ValueError(f"{field.name} must be positive, got {value}.")


DEFAULT_DATASET_SHAPE = MultirateDatasetShape()


_SCALAR_SPECS: dict[str, np.dtype] = {
    "anchor_index": np.dtype(np.uint32),
    "task_id": np.dtype(np.int16),
    "episode_id": np.dtype(np.int32),
    "frame_id": np.dtype(np.int32),
    "policy_seed": np.dtype(np.uint32),
}

_ARRAY_SPECS: dict[str, tuple[np.dtype, tuple[str, ...]]] = {
    "images": (
        np.dtype(np.uint8),
        ("window_size", "num_cameras", "image_height", "image_width", "image_channels"),
    ),
    "states": (np.dtype(np.float16), ("window_size", "state_dim")),
    "fresh_ear": (
        np.dtype(np.float16),
        ("window_size", "coarse_horizon", "action_dim"),
    ),
    "fresh_iar": (
        np.dtype(np.float16),
        ("window_size", "iar_tokens", "iar_dim"),
    ),
    "teacher_actions": (np.dtype(np.float16), ("window_size", "action_dim")),
    "b6_actions": (np.dtype(np.float16), ("window_size", "action_dim")),
    "hold_actions": (np.dtype(np.float16), ("window_size", "action_dim")),
    "event_mask": (np.dtype(np.bool_), ("window_size",)),
}

_FIELD_NAMES = (*_SCALAR_SPECS, *_ARRAY_SPECS)


def _shape_for(names: tuple[str, ...], shape: MultirateDatasetShape) -> tuple[int, ...]:
    return tuple(int(getattr(shape, name)) for name in names)


def _normalise_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    serialised = json.dumps(dict(metadata or {}), sort_keys=True)
    normalised = json.loads(serialised)
    if not isinstance(normalised, dict):
        raise TypeError("metadata must serialise to a JSON object.")
    return normalised


def _coerce_scalar(name: str, value: Any, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} has shape {array.shape}; expected a scalar.")
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must be an integer scalar, got dtype {array.dtype}.")

    integer = int(array.item())
    bounds = np.iinfo(dtype)
    if integer < 0 or integer < bounds.min or integer > bounds.max:
        raise ValueError(f"{name}={integer} is outside the supported range [0, {bounds.max}].")
    return np.asarray(integer, dtype=dtype)


def _coerce_array(
    name: str,
    value: Any,
    dtype: np.dtype,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {expected_shape}.")

    if name == "images":
        if array.dtype.kind not in {"i", "u"}:
            raise TypeError(f"images must contain integer pixels, got dtype {array.dtype}.")
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("images contains values outside the uint8 range [0, 255].")
    elif name == "event_mask":
        if array.dtype != np.dtype(np.bool_):
            raise TypeError(f"event_mask must have boolean dtype, got {array.dtype}.")
    else:
        if array.dtype.kind not in {"f", "i", "u"}:
            raise TypeError(f"{name} must contain real numeric values, got dtype {array.dtype}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")

    converted = np.asarray(array, dtype=dtype)
    if dtype.kind == "f" and not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} cannot be represented as finite {dtype} values.")
    return converted


def _coerce_record(
    record: Mapping[str, Any],
    shape: MultirateDatasetShape,
) -> dict[str, np.ndarray]:
    required = set(_FIELD_NAMES)
    missing = sorted(required.difference(record))
    if missing:
        raise KeyError(f"Multirate record is missing required fields: {missing}")
    unexpected = sorted(set(record).difference(required))
    if unexpected:
        raise KeyError(f"Multirate record has unexpected fields: {unexpected}")

    result: dict[str, np.ndarray] = {}
    for name, dtype in _SCALAR_SPECS.items():
        result[name] = _coerce_scalar(name, record[name], dtype)
    for name, (dtype, shape_names) in _ARRAY_SPECS.items():
        result[name] = _coerce_array(name, record[name], dtype, _shape_for(shape_names, shape))
    return result


def _read_json_attribute(handle: h5py.File, name: str, shard: pathlib.Path) -> Any:
    try:
        value = handle.attrs[name]
    except KeyError as error:
        raise ValueError(f"Missing {name!r} attribute in multirate shard {shard}.") from error
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON in {name!r} attribute of multirate shard {shard}.") from error


def _validate_shard(
    shard: pathlib.Path,
) -> tuple[MultirateDatasetShape, dict[str, Any], int]:
    try:
        handle = h5py.File(shard, "r")
    except OSError as error:
        raise ValueError(f"Cannot open multirate HDF5 shard {shard}.") from error

    with handle:
        try:
            schema_version = int(handle.attrs["schema_version"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid schema_version in multirate shard {shard}.") from error
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported multirate schema in {shard}: {schema_version}.")

        shape_payload = _read_json_attribute(handle, "shape_json", shard)
        metadata = _read_json_attribute(handle, "metadata_json", shard)
        try:
            shape = MultirateDatasetShape(**shape_payload)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid shape metadata in multirate shard {shard}: {shape_payload!r}.") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata_json in multirate shard {shard} must contain a JSON object.")

        actual_fields = set(handle.keys())
        required_fields = set(_FIELD_NAMES)
        missing = sorted(required_fields.difference(actual_fields))
        unexpected = sorted(actual_fields.difference(required_fields))
        if missing or unexpected:
            raise ValueError(
                f"Invalid datasets in multirate shard {shard}: missing={missing}, unexpected={unexpected}."
            )

        num_records: int | None = None
        for name, dtype in _SCALAR_SPECS.items():
            dataset = handle[name]
            if dataset.ndim != 1:
                raise ValueError(f"{name} in {shard} has shape {dataset.shape}; expected [records].")
            if np.dtype(dataset.dtype) != dtype:
                raise ValueError(f"{name} in {shard} has dtype {dataset.dtype}; expected {dtype}.")
            if num_records is None:
                num_records = int(dataset.shape[0])
            elif dataset.shape[0] != num_records:
                raise ValueError(f"{name} in {shard} has {dataset.shape[0]} records; expected {num_records}.")

        for name, (dtype, shape_names) in _ARRAY_SPECS.items():
            dataset = handle[name]
            expected_tail = _shape_for(shape_names, shape)
            if dataset.shape[1:] != expected_tail:
                raise ValueError(f"{name} in {shard} has shape {dataset.shape}; expected [records, {expected_tail}].")
            if np.dtype(dataset.dtype) != dtype:
                raise ValueError(f"{name} in {shard} has dtype {dataset.dtype}; expected {dtype}.")
            if num_records is None:
                num_records = int(dataset.shape[0])
            elif dataset.shape[0] != num_records:
                raise ValueError(f"{name} in {shard} has {dataset.shape[0]} records; expected {num_records}.")

        if not num_records:
            raise ValueError(f"Multirate shard {shard} contains no records.")
        return shape, metadata, num_records


class ShardedMultirateWriter:
    """Append-only writer that commits only complete HDF5 shards."""

    def __init__(
        self,
        output_dir: pathlib.Path | str,
        *,
        shape: MultirateDatasetShape = DEFAULT_DATASET_SHAPE,
        records_per_shard: int = 256,
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
        self._closed = False

        existing = sorted(self.output_dir.glob("shard-*.h5"))
        indices: list[int] = []
        for path in existing:
            try:
                indices.append(int(path.stem.removeprefix("shard-")))
            except ValueError as error:
                raise ValueError(f"Malformed multirate shard name: {path.name}.") from error
        self._next_shard = max(indices) + 1 if indices else 0

        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid multirate manifest: {manifest_path}.") from error
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"Cannot append to schema version {manifest.get('schema_version')}.")
            try:
                existing_shape = MultirateDatasetShape(**manifest["shape"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid shape in multirate manifest: {manifest_path}.") from error
            if existing_shape != shape:
                raise ValueError(f"Dataset shape mismatch: existing={existing_shape}, requested={shape}.")
            if manifest.get("metadata", {}) != self.metadata:
                raise ValueError(
                    "Multirate export metadata differs from the existing dataset. "
                    "Use a new output directory instead of mixing label protocols."
                )
            if int(manifest.get("records_per_shard", -1)) != records_per_shard:
                raise ValueError(
                    "records_per_shard differs from the existing multirate dataset: "
                    f"{manifest.get('records_per_shard')} != {records_per_shard}."
                )

        for shard in existing:
            existing_shape, existing_metadata, _ = _validate_shard(shard)
            if existing_shape != shape:
                raise ValueError(f"Dataset shape mismatch in {shard}: {existing_shape} != {shape}.")
            if existing_metadata != self.metadata:
                raise ValueError(f"Multirate export metadata mismatch in existing shard {shard}.")

        # Rebuild a missing or stale manifest from committed shards.  This
        # covers interruption after the atomic shard rename but before the
        # manifest rename.
        if existing:
            self._write_manifest()

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed multirate writer.")
        self._buffer.append(_coerce_record(record, self.shape))
        if len(self._buffer) >= self.records_per_shard:
            self.flush()

    def flush(self) -> pathlib.Path | None:
        if self._closed:
            raise RuntimeError("Cannot flush a closed multirate writer.")
        if not self._buffer:
            return None

        target = self.output_dir / f"shard-{self._next_shard:05d}.h5"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing multirate shard {target}.")
        temporary = target.with_suffix(".h5.tmp")
        records = self._buffer

        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["shape_json"] = json.dumps(dataclasses.asdict(self.shape), sort_keys=True)
            handle.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
            for name, dtype in _SCALAR_SPECS.items():
                values = np.stack([record[name] for record in records])
                handle.create_dataset(
                    name,
                    data=values.astype(dtype, copy=False),
                    compression="lzf",
                    shuffle=True,
                )
            for name, (dtype, _) in _ARRAY_SPECS.items():
                values = np.stack([record[name] for record in records])
                handle.create_dataset(
                    name,
                    data=values.astype(dtype, copy=False),
                    compression="lzf",
                    shuffle=True,
                )
            handle.flush()

        temporary.replace(target)
        self._buffer = []
        self._next_shard += 1
        self._write_manifest()
        return target

    def _write_manifest(self) -> None:
        shard_entries: list[dict[str, Any]] = []
        total_records = 0
        for shard in sorted(self.output_dir.glob("shard-*.h5")):
            with h5py.File(shard, "r") as handle:
                records = int(handle["anchor_index"].shape[0])
            total_records += records
            shard_entries.append({"file": shard.name, "num_records": records})

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "shape": dataclasses.asdict(self.shape),
            "records_per_shard": self.records_per_shard,
            "num_shards": len(shard_entries),
            "num_records": total_records,
            "metadata": self.metadata,
            "shards": shard_entries,
        }
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "manifest.json")

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self) -> ShardedMultirateWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # append() validates a complete record before adding it to the buffer,
        # so completed records remain safe to commit if a later sample fails.
        self.close()


def discover_shards(inputs: Sequence[pathlib.Path | str]) -> list[pathlib.Path]:
    shards: list[pathlib.Path] = []
    for item in inputs:
        path = pathlib.Path(item)
        if path.is_dir():
            shards.extend(sorted(path.glob("shard-*.h5")))
        elif path.is_file() and path.suffix in {".h5", ".hdf5"}:
            shards.append(path)
        else:
            raise FileNotFoundError(f"Multirate input is not an HDF5 shard/directory: {path}")

    unique = list(dict.fromkeys(path.resolve() for path in shards))
    if not unique:
        raise FileNotFoundError(f"No multirate HDF5 shards found under {inputs}.")
    return unique


def load_multirate_arrays(
    inputs: Sequence[pathlib.Path | str],
    fields: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    selected_fields = _FIELD_NAMES if fields is None else tuple(fields)
    if not selected_fields:
        raise ValueError("fields must contain at least one dataset name.")
    if len(set(selected_fields)) != len(selected_fields):
        raise ValueError(f"fields contains duplicate dataset names: {selected_fields}.")
    unknown = sorted(set(selected_fields).difference(_FIELD_NAMES))
    if unknown:
        raise KeyError(f"Unknown multirate dataset fields: {unknown}")

    pieces: dict[str, list[np.ndarray]] = {name: [] for name in selected_fields}
    expected_shape: MultirateDatasetShape | None = None
    expected_metadata: dict[str, Any] | None = None

    for shard in discover_shards(inputs):
        current_shape, current_metadata, _ = _validate_shard(shard)
        if expected_shape is None:
            expected_shape = current_shape
            expected_metadata = current_metadata
        elif current_shape != expected_shape:
            raise ValueError(f"Multirate shape mismatch in {shard}: {current_shape} != {expected_shape}.")
        elif current_metadata != expected_metadata:
            raise ValueError(f"Multirate metadata mismatch in {shard}.")

        with h5py.File(shard, "r") as handle:
            for name, destination in pieces.items():
                destination.append(handle[name][:])

    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}
