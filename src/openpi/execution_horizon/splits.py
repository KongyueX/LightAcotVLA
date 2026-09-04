"""Validated episode-group split manifests for execution-horizon experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import pathlib
import re
from typing import Any

import numpy as np

GROUP_ID_TASK_MULTIPLIER = 1_000_000_000
FOUR_WAY_SPLIT_SCHEMA_VERSION = 2
FOUR_WAY_SPLIT_NAMES = ("train", "early_stop", "calibration", "dev_audit")
FOUR_WAY_SPLIT_ROLES = {
    "train": "train",
    "early_stop": "early_stop",
    "calibration": "calibration",
    "dev_audit": "development_audit",
}
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def episode_group_ids(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return collision-free task-qualified episode IDs for supported inputs."""

    task = np.asarray(arrays["task_id"])
    episode = np.asarray(arrays["episode_id"])
    if task.ndim != 1 or episode.shape != task.shape:
        raise ValueError("task_id and episode_id must be same-length one-dimensional arrays.")
    if not np.issubdtype(task.dtype, np.integer) or not np.issubdtype(episode.dtype, np.integer):
        raise ValueError("task_id and episode_id must contain integers.")
    if np.any(task < 0) or np.any(episode < 0):
        raise ValueError("task_id and episode_id must be non-negative.")
    if np.any(episode >= GROUP_ID_TASK_MULTIPLIER):
        raise ValueError(f"episode_id must be below {GROUP_ID_TASK_MULTIPLIER} for task-qualified group IDs.")
    task_u64 = task.astype(np.uint64)
    maximum_task = np.iinfo(np.uint64).max // np.uint64(GROUP_ID_TASK_MULTIPLIER)
    if np.any(task_u64 > maximum_task):
        raise ValueError("task_id is too large for a uint64 task-qualified group ID.")
    return task_u64 * np.uint64(GROUP_ID_TASK_MULTIPLIER) + episode.astype(np.uint64)


def _group_list(name: str, values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"split_manifest {name!r} must contain a one-dimensional group list.")
    if not raw.size:
        return np.empty((0,), dtype=np.uint64)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"split_manifest {name!r} must contain integer group IDs.")
    if np.any(raw < 0):
        raise ValueError(f"split_manifest {name!r} must contain non-negative group IDs.")
    groups = raw.astype(np.uint64)
    if np.unique(groups).size != groups.size:
        raise ValueError(f"split_manifest {name!r} contains duplicate groups.")
    return groups


def validate_manifest_group_lists(manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate every declared ``*_group_ids`` list and their mutual exclusion."""

    declared = {
        key.removesuffix("_group_ids"): _group_list(key, values)
        for key, values in manifest.items()
        if key.endswith("_group_ids")
    }
    if not declared:
        raise ValueError("split_manifest declares no *_group_ids partitions.")
    names = sorted(declared)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = np.intersect1d(declared[left], declared[right])
            if overlap.size:
                raise ValueError(
                    f"split_manifest groups overlap between {left!r} and {right!r}: "
                    f"{overlap[:10].tolist()}."
                )
    return declared


def is_four_way_manifest(manifest: Mapping[str, Any]) -> bool:
    return int(manifest.get("split_schema_version", 0)) == FOUR_WAY_SPLIT_SCHEMA_VERSION


def validate_four_way_schema(
    manifest: Mapping[str, Any],
    groups_by_split: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Require the explicit train/early-stop/calibration/dev-audit schema."""

    version = manifest.get("split_schema_version")
    if version != FOUR_WAY_SPLIT_SCHEMA_VERSION:
        raise ValueError(
            "Explicit four-way splitting requires "
            f"split_schema_version={FOUR_WAY_SPLIT_SCHEMA_VERSION}; got {version!r}."
        )
    groups = validate_manifest_group_lists(manifest) if groups_by_split is None else groups_by_split
    if set(groups) != set(FOUR_WAY_SPLIT_NAMES):
        raise ValueError(
            "Four-way split_manifest must declare exactly "
            f"{list(FOUR_WAY_SPLIT_NAMES)}; got {sorted(groups)}."
        )
    empty = [name for name in FOUR_WAY_SPLIT_NAMES if not groups[name].size]
    if empty:
        raise ValueError(f"Four-way split_manifest partitions must be non-empty: {empty}.")
    roles = manifest.get("split_roles")
    if not isinstance(roles, Mapping):
        raise ValueError("Four-way split_manifest requires a split_roles mapping.")
    normalized_roles = {str(name): str(role) for name, role in roles.items()}
    if normalized_roles != FOUR_WAY_SPLIT_ROLES:
        raise ValueError(
            "Four-way split_manifest roles must be exactly "
            f"{FOUR_WAY_SPLIT_ROLES}; got {normalized_roles}."
        )
    bank_path = manifest.get("development_initial_state_bank")
    bank_sha256 = manifest.get("development_initial_state_bank_sha256")
    if (bank_path is None) != (bank_sha256 is None):
        raise ValueError("Development initial-state bank path and SHA-256 must be provided together.")
    if bank_path is not None:
        if not isinstance(bank_path, str) or not bank_path.strip():
            raise ValueError("development_initial_state_bank must be a non-empty path string.")
        if not isinstance(bank_sha256, str) or _SHA256.fullmatch(bank_sha256) is None:
            raise ValueError("development_initial_state_bank_sha256 must be a 64-character hexadecimal digest.")


def validate_dataset_coverage(
    arrays: Mapping[str, np.ndarray],
    groups_by_split: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Prove every episode group in the loaded dataset has exactly one split role."""

    row_groups = episode_group_ids(arrays)
    observed = np.unique(row_groups)
    declared = np.unique(np.concatenate([np.asarray(values, dtype=np.uint64) for values in groups_by_split.values()]))
    undeclared = np.setdiff1d(observed, declared)
    missing = np.setdiff1d(declared, observed)
    if undeclared.size or missing.size:
        raise ValueError(
            "split_manifest must exactly cover loaded episode groups: "
            f"undeclared={undeclared[:10].tolist()}, missing={missing[:10].tolist()}."
        )
    return row_groups


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    require_four_way: bool = False,
) -> dict[str, np.ndarray]:
    """Validate split lists, optional four-way roles, and dataset coverage."""

    groups_by_split = validate_manifest_group_lists(manifest)
    if require_four_way or is_four_way_manifest(manifest):
        validate_four_way_schema(manifest, groups_by_split)
    elif "early_stop" in groups_by_split or "dev_audit" in groups_by_split:
        raise ValueError("Four-way split names require split_schema_version=2 and explicit split_roles.")
    if arrays is not None:
        validate_dataset_coverage(arrays, groups_by_split)
    return groups_by_split


def load_manifest(
    path: pathlib.Path | str,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    require_four_way: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = pathlib.Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("split_manifest must contain a JSON object.")
    return manifest, validate_manifest(manifest, arrays=arrays, require_four_way=require_four_way)


def indices_for_split(
    row_groups: np.ndarray,
    groups_by_split: Mapping[str, Sequence[int] | np.ndarray],
    split_name: str,
) -> np.ndarray:
    if split_name not in groups_by_split:
        raise KeyError(f"split_manifest has no {split_name!r} group list.")
    indices = np.flatnonzero(np.isin(np.asarray(row_groups, dtype=np.uint64), groups_by_split[split_name]))
    if not indices.size:
        raise ValueError(f"No dataset roots match split {split_name!r}.")
    return indices
