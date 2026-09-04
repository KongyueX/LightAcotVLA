from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).with_name("finalize_execution_horizon_aggregate_gate.py")
sys.path.insert(0, str(_SCRIPT.parent))
final_gate = importlib.import_module("finalize_execution_horizon_aggregate_gate")


def _passing_development_audit() -> dict[str, object]:
    return {
        "status": "complete",
        "offline_engineering_gate": True,
        "provenance_verified": True,
        "aggregate_calibration_gate_passed": True,
        "rule_frozen_before_audit": True,
        "audit_split_used_for_threshold_fit": False,
        "split_role": "validation",
    }


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
