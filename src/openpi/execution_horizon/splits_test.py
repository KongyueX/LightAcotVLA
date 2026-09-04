from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.execution_horizon import splits


def _arrays() -> dict[str, np.ndarray]:
    return {
        "task_id": np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        "episode_id": np.asarray([100, 100, 101, 102, 100, 101, 102, 103], dtype=np.int64),
    }


def _four_way_manifest() -> dict[str, object]:
    return {
        "split_schema_version": 2,
        "split_roles": {
            "train": "train",
            "early_stop": "early_stop",
            "calibration": "calibration",
            "dev_audit": "development_audit",
        },
        "train_group_ids": [100, 1_000_000_100],
        "early_stop_group_ids": [101, 1_000_000_101],
        "calibration_group_ids": [102, 1_000_000_102],
        "dev_audit_group_ids": [1_000_000_103],
    }


def test_four_way_manifest_is_disjoint_episode_atomic_and_exact() -> None:
    arrays = _arrays()
    groups = splits.validate_manifest(_four_way_manifest(), arrays=arrays, require_four_way=True)
    row_groups = splits.episode_group_ids(arrays)

    train = splits.indices_for_split(row_groups, groups, "train")
    assert train.tolist() == [0, 1, 4]
    for group in np.unique(row_groups):
        memberships = [group in values for values in groups.values()]
        assert sum(memberships) == 1


def test_four_way_manifest_rejects_overlap_or_incomplete_dataset_coverage() -> None:
    overlapping = _four_way_manifest()
    overlapping["dev_audit_group_ids"] = [102]
    with pytest.raises(ValueError, match="overlap"):
        splits.validate_manifest(overlapping, require_four_way=True)

    missing = _four_way_manifest()
    missing["dev_audit_group_ids"] = [1_000_000_104]
    with pytest.raises(ValueError, match="exactly cover"):
        splits.validate_manifest(missing, arrays=_arrays(), require_four_way=True)


def test_four_way_manifest_preserves_optional_bank_identity(tmp_path) -> None:
    manifest = {
        **_four_way_manifest(),
        "development_initial_state_bank": "/tmp/frozen-bank",
        "development_initial_state_bank_sha256": "a" * 64,
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))

    loaded, _ = splits.load_manifest(path, arrays=_arrays(), require_four_way=True)
    assert loaded["development_initial_state_bank"] == "/tmp/frozen-bank"
    assert loaded["development_initial_state_bank_sha256"] == "a" * 64

    bad = {**manifest, "development_initial_state_bank_sha256": "not-a-digest"}
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        splits.validate_manifest(bad, require_four_way=True)


def test_legacy_manifest_allows_unused_empty_calibration() -> None:
    groups = splits.validate_manifest(
        {
            "train_group_ids": [1],
            "validation_group_ids": [2],
            "calibration_group_ids": [],
        }
    )
    assert groups["calibration"].size == 0


def test_four_way_manifest_rejects_empty_partition() -> None:
    manifest = _four_way_manifest()
    manifest["dev_audit_group_ids"] = []
    with pytest.raises(ValueError, match="partitions must be non-empty"):
        splits.validate_manifest(manifest)
