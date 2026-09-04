from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

from openpi.execution_horizon import initial_states

_SCRIPT = pathlib.Path(__file__).with_name("finalize_execution_horizon_aggregate_gate.py")
sys.path.insert(0, str(_SCRIPT.parent))
final_gate = importlib.import_module("finalize_execution_horizon_aggregate_gate")


def _make_bank(path: pathlib.Path, *, generated_states: int) -> initial_states.InitialStateBank:
    path.mkdir()
    states = np.asarray(
        [[0, episode + 1, episode + 2, 0, 0] for episode in range(2 + generated_states)],
        dtype=np.float64,
    )
    seeds = np.asarray([-1, -1, *range(100, 100 + generated_states)], dtype=np.int64)
    target = path / "task00.npz"
    np.savez_compressed(target, states=states, episode_ids=np.arange(len(states)), generation_seeds=seeds)
    manifest = {
        "status": "complete",
        "schema_version": 1,
        "fingerprint_method": initial_states.FINGERPRINT_METHOD,
        "task_suite": "libero_10",
        "max_tasks": 1,
        "generated_per_task": generated_states,
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
    return initial_states.InitialStateBank(path)


def _passing_development_audit() -> dict[str, object]:
    return {
        "status": "complete",
        "offline_engineering_gate": True,
        "provenance_verified": True,
        "aggregate_calibration_gate_passed": True,
        "rule_frozen_before_audit": True,
        "audit_split_used_for_threshold_fit": False,
        "split_name": "validation",
        "split_role": "validation",
    }


def test_development_gate_accepts_four_way_audit_and_legacy_validation() -> None:
    final_gate._require_development_gate(_passing_development_audit())  # noqa: SLF001
    final_gate._require_development_gate(  # noqa: SLF001
        {
            **_passing_development_audit(),
            "split_name": "dev_audit",
            "split_role": "development_audit",
        }
    )

    with pytest.raises(ValueError, match="recognized_independent_audit_split"):
        final_gate._require_development_gate(  # noqa: SLF001
            {
                **_passing_development_audit(),
                "split_name": "dev_audit",
                "split_role": "validation",
            }
        )


def test_manifest_audit_groups_follow_frozen_audit_split_name(tmp_path: pathlib.Path) -> None:
    legacy_manifest = tmp_path / "legacy.json"
    legacy_manifest.write_text(
        json.dumps(
            {
                "train_group_ids": [1],
                "validation_group_ids": [2],
                "calibration_group_ids": [3],
            }
        )
    )
    assert final_gate._manifest_audit_groups(  # noqa: SLF001
        _passing_development_audit(),
        legacy_manifest,
    ) == (2,)

    four_way_manifest = tmp_path / "four_way.json"
    four_way_manifest.write_text(
        json.dumps(
            {
                "split_schema_version": 2,
                "split_roles": {
                    "train": "train",
                    "early_stop": "early_stop",
                    "calibration": "calibration",
                    "dev_audit": "development_audit",
                },
                "train_group_ids": [1],
                "early_stop_group_ids": [2],
                "calibration_group_ids": [3],
                "dev_audit_group_ids": [4],
            }
        )
    )
    assert final_gate._manifest_audit_groups(  # noqa: SLF001
        {
            **_passing_development_audit(),
            "split_name": "dev_audit",
            "split_role": "development_audit",
        },
        four_way_manifest,
    ) == (4,)


def test_final_data_is_not_opened_before_development_gate(tmp_path: pathlib.Path, monkeypatch) -> None:
    development = tmp_path / "development_audit.json"
    development.write_text(json.dumps({**_passing_development_audit(), "offline_engineering_gate": False}))
    final_summary = tmp_path / "fresh_final_e90_99" / "summary.json"
    accessed: list[pathlib.Path] = []
    original = final_gate._json_snapshot  # noqa: SLF001

    def record(path: pathlib.Path):
        accessed.append(path)
        return original(path)

    monkeypatch.setattr(final_gate, "_json_snapshot", record)
    args = argparse.Namespace(
        development_audit_json=str(development),
        fresh_final_summary=str(final_summary),
        checkpoint_dir=str(tmp_path / "checkpoint"),
        output_dir=str(tmp_path / "gate"),
        initial_state_isolation_json=None,
        batch_size=1,
        inference_initialization_seed=None,
        expected_final_roots=100,
        expected_task_start=0,
        expected_max_tasks=10,
        expected_episode_start=90,
        expected_episode_end=99,
        expected_branch_repeats=3,
    )

    with pytest.raises(ValueError, match="not eligible"):
        final_gate.main(args)

    assert accessed == [development.resolve()]
    assert not final_summary.parent.exists()
    assert not (tmp_path / "gate.one_shot_claim.json").exists()


def test_fresh_final_summary_must_match_preregistered_grid() -> None:
    summary = {
        "status": "complete",
        "num_roots": 100,
        "task_start": 0,
        "max_tasks": 10,
        "episode_start": 90,
        "episode_end": 99,
        "branch_repeats": 3,
        "data_dirs": ["/tmp/fresh_final/task00/data"],
    }

    assert final_gate._validate_fresh_final_summary(  # noqa: SLF001
        summary,
        expected_roots=100,
        expected_task_start=0,
        expected_max_tasks=10,
        expected_episode_start=90,
        expected_episode_end=99,
        expected_branch_repeats=3,
    ) == (str(pathlib.Path("/tmp/fresh_final/task00/data").resolve()),)

    with pytest.raises(ValueError, match="pre-registered contract"):
        final_gate._validate_fresh_final_summary(  # noqa: SLF001
            {**summary, "branch_repeats": 10},
            expected_roots=100,
            expected_task_start=0,
            expected_max_tasks=10,
            expected_episode_start=90,
            expected_episode_end=99,
            expected_branch_repeats=3,
        )


def test_final_groups_require_exactly_one_root_per_task_episode() -> None:
    task = np.repeat(np.arange(2, dtype=np.uint32), 2)
    episode = np.tile(np.arange(90, 92, dtype=np.uint32), 2)
    arrays = {
        "task_id": task,
        "episode_id": episode,
        "decision_step": np.full((4,), 10, dtype=np.uint32),
        "root_seed": np.arange(4, dtype=np.uint64),
    }

    indices, groups, observed = final_gate._final_groups_and_indices(  # noqa: SLF001
        arrays,
        expected_task_start=0,
        expected_max_tasks=2,
        expected_episode_start=90,
        expected_episode_end=91,
    )

    assert indices.tolist() == [0, 1, 2, 3]
    assert len(np.unique(groups)) == 4
    assert observed == {(0, 90), (0, 91), (1, 90), (1, 91)}


def test_four_way_manifest_binds_development_superset_bank(tmp_path: pathlib.Path) -> None:
    child = _make_bank(tmp_path / "child", generated_states=3)
    manifest = {
        "split_schema_version": 2,
        "development_initial_state_bank": str(child.directory),
        "development_initial_state_bank_sha256": child.sha256,
    }

    loaded = final_gate._development_bank_from_manifest(manifest, {(0, 4)})  # noqa: SLF001
    assert loaded is not None
    assert loaded.sha256 == child.sha256

    with pytest.raises(ValueError, match="requires a frozen development"):
        final_gate._development_bank_from_manifest(  # noqa: SLF001
            {"split_schema_version": 2},
            {(0, 4)},
        )
    with pytest.raises(ValueError, match="changed after the split manifest"):
        final_gate._development_bank_from_manifest(  # noqa: SLF001
            {**manifest, "development_initial_state_bank_sha256": "0" * 64},
            {(0, 4)},
        )


def test_legacy_manifest_keeps_single_bank_path() -> None:
    assert final_gate._development_bank_from_manifest({}, {(0, 1)}) is None  # noqa: SLF001


def test_finalizer_audits_parent_child_lineage_and_cross_bank_isolation(tmp_path: pathlib.Path) -> None:
    parent = _make_bank(tmp_path / "parent", generated_states=1)
    child = _make_bank(tmp_path / "child", generated_states=3)

    isolation = final_gate._audit_initial_state_isolation(  # noqa: SLF001
        final_bank=parent,
        development_bank=child,
        development_groups={(0, 4)},
        final_groups={(0, 2)},
    )
    assert isolation["schema_version"] == 2
    assert isolation["bank_lineage"]["parent"]["initial_state_bank_sha256"] == parent.sha256
    assert isolation["bank_lineage"]["child"]["initial_state_bank_sha256"] == child.sha256
    assert isolation["partition_group_counts"] == {"development": 1, "final": 1}

    with pytest.raises(ValueError, match="overlap"):
        final_gate._audit_initial_state_isolation(  # noqa: SLF001
            final_bank=parent,
            development_bank=child,
            development_groups={(0, 2)},
            final_groups={(0, 2)},
        )


def test_finalizer_legacy_single_bank_isolation_is_unchanged(tmp_path: pathlib.Path) -> None:
    bank = _make_bank(tmp_path / "bank", generated_states=2)
    isolation = final_gate._audit_initial_state_isolation(  # noqa: SLF001
        final_bank=bank,
        development_bank=None,
        development_groups={(0, 2)},
        final_groups={(0, 3)},
    )
    assert "bank_lineage" not in isolation
    assert isolation["initial_state_bank_sha256"] == bank.sha256
