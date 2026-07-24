"""Sharded endpoint and intervention labels for the IR-ACoT pilot.

The dataset intentionally stores only compact action-space tensors and the
index of the corresponding frame in the original LeRobot dataset. Images and
tokenized prompts are reconstructed from that source dataset during training.
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

INTERVENTION_NAMES = (
    "null",
    "translate",
    "rotate",
    "gripper_shift",
    "swap",
)
INTERVENTION_IDS = {name: index for index, name in enumerate(INTERVENTION_NAMES)}


@dataclasses.dataclass(frozen=True)
class EndpointDatasetShape:
    action_dim: int = 32
    env_action_dim: int = 7
    coarse_horizon: int = 15
    action_horizon: int = 10
    num_interventions: int = len(INTERVENTION_NAMES)


DEFAULT_DATASET_SHAPE = EndpointDatasetShape()


_SCALAR_SPECS: dict[str, np.dtype] = {
    "dataset_index": np.dtype(np.uint32),
    "task_id": np.dtype(np.int16),
    "episode_id": np.dtype(np.int32),
    "frame_id": np.dtype(np.int32),
    "policy_seed": np.dtype(np.uint32),
}

_ARRAY_SPECS: dict[str, tuple[np.dtype, tuple[str, ...]]] = {
    "coarse_noise": (np.dtype(np.float16), ("coarse_horizon", "action_dim")),
    "action_noise": (np.dtype(np.float16), ("action_horizon", "action_dim")),
    "clean_coarse": (np.dtype(np.float16), ("coarse_horizon", "action_dim")),
    "clean_actions": (np.dtype(np.float16), ("action_horizon", "action_dim")),
    "clean_coarse_env": (np.dtype(np.float32), ("coarse_horizon", "env_action_dim")),
    "intervention_ids": (np.dtype(np.uint8), ("num_interventions",)),
    "intervention_valid": (np.dtype(np.bool_), ("num_interventions",)),
    "intervention_coarse": (
        np.dtype(np.float16),
        ("num_interventions", "coarse_horizon", "action_dim"),
    ),
    "intervention_actions": (
        np.dtype(np.float16),
        ("num_interventions", "action_horizon", "action_dim"),
    ),
    "intervention_coarse_env": (
        np.dtype(np.float32),
        ("num_interventions", "coarse_horizon", "env_action_dim"),
    ),
    "response_l2": (np.dtype(np.float32), ("num_interventions",)),
}


def _shape_for(names: tuple[str, ...], shape: EndpointDatasetShape) -> tuple[int, ...]:
    return tuple(int(getattr(shape, name)) for name in names)


def _coerce_record(record: Mapping[str, Any], shape: EndpointDatasetShape) -> dict[str, np.ndarray]:
    required = set(_SCALAR_SPECS) | set(_ARRAY_SPECS)
    missing = sorted(required.difference(record))
    if missing:
        raise KeyError(f"Endpoint record is missing required fields: {missing}")

    result: dict[str, np.ndarray] = {}
    for name, dtype in _SCALAR_SPECS.items():
        value = np.asarray(record[name], dtype=dtype)
        if value.shape != ():
            raise ValueError(f"{name} has shape {value.shape}; expected a scalar.")
        result[name] = value

    for name, (dtype, shape_names) in _ARRAY_SPECS.items():
        value = np.asarray(record[name], dtype=dtype)
        expected_shape = _shape_for(shape_names, shape)
        if value.shape != expected_shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected_shape}.")
        result[name] = value
    return result


class ShardedEndpointWriter:
    """Append-only, interruption-safe HDF5 writer."""

    def __init__(
        self,
        output_dir: pathlib.Path | str,
        *,
        shape: EndpointDatasetShape = DEFAULT_DATASET_SHAPE,
        records_per_shard: int = 256,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if records_per_shard <= 0:
            raise ValueError("records_per_shard must be positive.")
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shape = shape
        self.records_per_shard = records_per_shard
        self.metadata = dict(metadata or {})
        self._buffer: list[dict[str, np.ndarray]] = []

        existing = sorted(self.output_dir.glob("shard-*.h5"))
        self._next_shard = (
            max(int(path.stem.removeprefix("shard-")) for path in existing) + 1
            if existing
            else 0
        )
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest["schema_version"]) != SCHEMA_VERSION:
                raise ValueError(f"Cannot append to schema version {manifest['schema_version']}.")
            existing_shape = EndpointDatasetShape(**manifest["shape"])
            if existing_shape != shape:
                raise ValueError(f"Dataset shape mismatch: existing={existing_shape}, requested={shape}.")
            if manifest.get("metadata", {}) != self.metadata:
                raise ValueError(
                    "Endpoint export metadata differs from the existing dataset. "
                    "Use a new output directory instead of mixing label protocols."
                )

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
        shards = sorted(self.output_dir.glob("shard-*.h5"))
        total_records = 0
        for shard in shards:
            with h5py.File(shard, "r") as handle:
                total_records += int(handle["dataset_index"].shape[0])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "shape": dataclasses.asdict(self.shape),
            "records_per_shard": self.records_per_shard,
            "num_shards": len(shards),
            "num_records": total_records,
            "metadata": self.metadata,
        }
        temporary = self.output_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "manifest.json")

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> ShardedEndpointWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Records enter the buffer only after a sample is complete, so they
        # remain safe to commit even when a later sample raises or the user
        # interrupts the exporter.
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
            raise FileNotFoundError(f"Endpoint input is not an HDF5 shard/directory: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in shards))
    if not unique:
        raise FileNotFoundError(f"No endpoint HDF5 shards found under {inputs}.")
    return unique


def load_endpoint_arrays(inputs: Sequence[pathlib.Path | str]) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in (*_SCALAR_SPECS, *_ARRAY_SPECS)
    }
    expected_shape: EndpointDatasetShape | None = None
    for shard in discover_shards(inputs):
        with h5py.File(shard, "r") as handle:
            if int(handle.attrs["schema_version"]) != SCHEMA_VERSION:
                raise ValueError(f"Unsupported endpoint schema in {shard}: {handle.attrs['schema_version']}")
            current_shape = EndpointDatasetShape(**json.loads(handle.attrs["shape_json"]))
            if expected_shape is None:
                expected_shape = current_shape
            elif current_shape != expected_shape:
                raise ValueError(f"Endpoint shape mismatch in {shard}: {current_shape} != {expected_shape}.")
            for name, destination in pieces.items():
                destination.append(handle[name][:])
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def apply_intervention(
    coarse_actions: np.ndarray,
    intervention: str,
    *,
    seed: int = 0,
    translation_magnitude: float = 0.02,
    rotation_magnitude: float = 0.10,
    gripper_shift: int = 1,
) -> np.ndarray:
    """Apply a deterministic, physically interpretable intervention.

    The input is expected in environment action space. For LIBERO this is a
    seven-dimensional pose delta plus gripper command.
    """

    if intervention not in INTERVENTION_IDS:
        raise ValueError(f"Unknown intervention {intervention!r}; expected one of {INTERVENTION_NAMES}.")
    values = np.array(coarse_actions, dtype=np.float32, copy=True)
    if values.ndim != 2:
        raise ValueError(f"coarse_actions must be rank 2, got {values.shape}.")
    horizon, action_dim = values.shape
    if horizon < 2:
        raise ValueError("coarse_actions must contain at least two frames.")
    if intervention == "null":
        return values

    rng = np.random.default_rng(seed)
    start = max(0, horizon // 3)
    stop = max(start + 1, min(horizon, (2 * horizon + 2) // 3))

    if intervention == "translate":
        if action_dim < 3:
            raise ValueError("translate intervention requires at least three action dimensions.")
        axis = int(rng.integers(0, 3))
        sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        values[start:stop, axis] += sign * translation_magnitude
        return values

    if intervention == "rotate":
        if action_dim < 6:
            raise ValueError("rotate intervention requires at least six action dimensions.")
        axis = 3 + int(rng.integers(0, 3))
        sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        values[start:stop, axis] += sign * rotation_magnitude
        return values

    if intervention == "gripper_shift":
        if gripper_shift == 0:
            raise ValueError("gripper_shift must be non-zero.")
        source = values[:, -1].copy()
        shift = int(np.sign(gripper_shift))
        if shift > 0:
            values[1:, -1] = source[:-1]
            values[0, -1] = source[0]
        else:
            values[:-1, -1] = source[1:]
            values[-1, -1] = source[-1]
        return values

    left = max(0, min(horizon - 2, horizon // 2 - 1))
    values[[left, left + 1]] = values[[left + 1, left]]
    return values
