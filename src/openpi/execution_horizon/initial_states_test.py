from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

from openpi.execution_horizon import initial_states


def _make_bank(path, *, duplicate_pose=False):
    path.mkdir()
    states = np.asarray([[0, 1, 2, 0, 0], [0, 2, 3, 0, 0], [0, 3, 4, 0, 0]], dtype=np.float64)
    if duplicate_pose:
        states[2] = states[0]
        states[2, 0] = 20  # Different timestamp cannot disguise a reused pose.
        states[2, -1] = 5
    target = path / "task00.npz"
    np.savez_compressed(target, states=states, episode_ids=np.arange(3), generation_seeds=np.asarray([-1, -1, 100]))
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "fingerprint_method": initial_states.FINGERPRINT_METHOD,
        "task_suite": "libero_10",
        "max_tasks": 1,
        "generated_per_task": 1,
        "tasks": [
            {
                "task_id": 0,
                "file": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "preset_count": 2,
                "state_dim": 5,
                "nq": 2,
                "fingerprints": [initial_states.fingerprint(state, 2) for state in states],
            }
        ],
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return states


def _write_groups(path, episode, metadata):
    path.mkdir()
    with h5py.File(path / "shard-00000.h5", "w") as handle:
        handle.attrs["metadata_json"] = json.dumps({"task_suite": "libero_10", **metadata})
        handle.create_dataset("task_id", data=[0])
        handle.create_dataset("episode_id", data=[episode])


def test_bank_explicit_lookup_and_presets(tmp_path):
    states = _make_bank(tmp_path / "bank")
    bank = initial_states.InitialStateBank(tmp_path / "bank")
    bank.validate_presets(0, states[:2])
    np.testing.assert_array_equal(bank.state(0, 2), states[2])
    copied = bank.state(0, 2)
    copied[:] = 0
    np.testing.assert_array_equal(bank.state(0, 2), states[2])
    with pytest.raises(ValueError, match="modulo"):
        bank.state(0, 3)
    with pytest.raises(ValueError, match="preset states differ"):
        bank.validate_presets(0, states[:2] + 1)


def test_time_and_velocity_do_not_make_a_fresh_initial_pose(tmp_path):
    _make_bank(tmp_path / "bank", duplicate_pose=True)
    with pytest.raises(ValueError, match="Duplicate"):
        initial_states.InitialStateBank(tmp_path / "bank")


def test_bank_checksum_rejects_changed_state_file(tmp_path):
    _make_bank(tmp_path / "bank")
    (tmp_path / "bank" / "task00.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        initial_states.InitialStateBank(tmp_path / "bank")


def test_physical_partitions_and_dataset_provenance(tmp_path):
    _make_bank(tmp_path / "bank")
    bank = initial_states.InitialStateBank(tmp_path / "bank")
    _write_groups(tmp_path / "legacy", 0, {})
    _write_groups(tmp_path / "fresh", 2, bank.metadata())
    base = initial_states.dataset_groups([str(tmp_path / "legacy")], bank, allow_legacy_presets=True)
    fresh = initial_states.dataset_groups([str(tmp_path / "fresh")], bank)
    audit = bank.audit_partitions({"base": base, "final": fresh})
    assert audit["pairwise_initial_state_overlap"] == 0
    assert audit["partition_group_counts"] == {"base": 1, "final": 1}
    with pytest.raises(ValueError, match="overlap"):
        bank.audit_partitions({"base": base, "final": base})
    with pytest.raises(ValueError, match="lacks explicit"):
        initial_states.dataset_groups([str(tmp_path / "legacy")], bank)


def test_legacy_episode_number_cannot_disguise_modulo_reuse(tmp_path):
    _make_bank(tmp_path / "bank")
    bank = initial_states.InitialStateBank(tmp_path / "bank")
    _write_groups(tmp_path / "alias", 2, {})
    with pytest.raises(ValueError, match="alias a preset"):
        initial_states.dataset_groups([str(tmp_path / "alias")], bank, allow_legacy_presets=True)
