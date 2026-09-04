from __future__ import annotations

import importlib
from itertools import pairwise
import json
import pathlib
import sys

import pytest

from openpi.execution_horizon import splits as horizon_splits

_SCRIPT = pathlib.Path(__file__).with_name("run_execution_horizon_round3_collection.py")
sys.path.insert(0, str(_SCRIPT.parent))
round3 = importlib.import_module("run_execution_horizon_round3_collection")


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_selection_is_deterministic_disjoint_and_balanced() -> None:
    first = round3.build_selection()
    second = round3.build_selection()
    assert first == second
    assert first["total_target_roots"] == 600
    assert first["root_seed_scheme"] == "affine_task_episode_repeat_schedule_continuation_lanes_uint32_v2"
    assert first["root_seed_branch_schedule_offset"] == 5_000_000
    assert first["root_seed_branch_continuation_offset"] == 10_000_000
    assert first["root_seed_strict_namespace_validation"] is True

    for task in round3.TASK_IDS:
        task_plan = first["tasks"][str(task)]["roles"]
        assigned: set[int] = set()
        for role, target in round3.ROLE_TARGETS.items():
            primary = task_plan[role]["primary"]
            fallback = task_plan[role]["fallback"]
            assert len(primary) == target
            assert not assigned.intersection(primary)
            assert not assigned.intersection(fallback)
            assigned.update(primary)
            assigned.update(fallback)
            assert set(primary).isdisjoint(fallback)
            assert all(3 <= episode % 12 <= 10 for episode in (*primary, *fallback))
            primary_offset_counts = {offset: sum(episode % 12 == offset for episode in primary) for offset in range(3, 11)}
            assert max(primary_offset_counts.values()) - min(primary_offset_counts.values()) <= 1
        assert assigned == set(round3.eligible_episodes())


def test_requested_task_seed_stride_is_uint32_safe_and_disjoint() -> None:
    assert round3.MAXIMUM_EPISODE_STEP_FOR_SEED_AUDIT == 1_570
    intervals = round3.validate_root_seed_namespace()
    assert len(intervals) == 10
    assert all(left[1] < right[0] for left, right in pairwise(intervals))
    assert intervals[-1][1] <= round3.MAX_POLICY_SEED

    with pytest.raises(ValueError, match="namespace"):
        round3.validate_root_seed_namespace(task_stride=10_000_000)


def test_python_executable_keeps_virtualenv_symlink(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "python-real"
    target.write_text("binary")
    virtualenv_python = tmp_path / "venv-python"
    virtualenv_python.symlink_to(target)

    assert round3._executable_path(str(virtualenv_python)) == virtualenv_python.absolute()  # noqa: SLF001
    assert round3._executable_path(str(virtualenv_python)) != target.resolve()  # noqa: SLF001


def test_fallback_batch_never_crosses_role_or_reuses_episode() -> None:
    selection = round3.build_selection()
    task = 3
    role = "dev_audit"
    plan = selection["tasks"][str(task)]["roles"][role]
    attempted = set(plan["primary"]) | set(plan["fallback"][:3])

    batch = round3.next_fallback_batch(
        selection,
        task_id=task,
        role=role,
        attempted_episodes=attempted,
        deficit=7,
        maximum_batch=4,
    )

    assert len(batch) == 4
    assert set(batch).isdisjoint(attempted)
    assert all(round3.split_for_episode(selection, task, episode) == role for episode in batch)


def test_collector_command_encodes_round3_protocol_without_final_input(tmp_path) -> None:
    command = round3.build_collector_command(
        python=tmp_path / "python",
        collector_script=tmp_path / "collector.py",
        data_dir=tmp_path / "data",
        initial_state_bank=tmp_path / "development_bank",
        aggregate_calibration_json=tmp_path / "seed7_aggregate.json",
        host="127.0.0.1",
        port=8040,
        task_id=2,
        episodes=(100, 101),
    )

    assert _flag_value(command, "--source-policy") == "current_student"
    assert _flag_value(command, "--continuation-policy") == "fixed_h"
    assert _flag_value(command, "--fixed-continuation-horizon") == "5"
    assert _flag_value(command, "--branch-repeats") == "3"
    assert _flag_value(command, "--source-iteration") == "3"
    assert _flag_value(command, "--root-seed-task-stride") == "250000000"
    assert _flag_value(command, "--root-call-offset-cycle") == "12"
    assert _flag_value(command, "--student-mode") == "hierarchical_transformer"
    assert "--hierarchical-aggregate-calibration-json" in command
    assert not any(
        value.startswith("--") and ("final" in value.lower() or "holdout" in value.lower()) for value in command
    )


def test_combined_manifest_preserves_old_roles_and_is_four_way(tmp_path) -> None:
    base = {
        "train_group_ids": [0, 1_000_000_000],
        "validation_group_ids": [1, 1_000_000_001],
        "calibration_group_ids": [2, 1_000_000_002],
    }
    selection = round3.build_selection()
    realized = {role: [] for role in round3.ROLE_TARGETS}
    for task in round3.TASK_IDS:
        for role in round3.ROLE_TARGETS:
            realized[role].extend(
                task * horizon_splits.GROUP_ID_TASK_MULTIPLIER + episode
                for episode in selection["tasks"][str(task)]["roles"][role]["primary"]
            )
    manifest = round3.build_four_way_split_manifest(
        base,
        realized_groups=realized,
        initial_state_bank=tmp_path / "bank",
        initial_state_bank_sha256="a" * 64,
        bank_lineage_path=tmp_path / "bank_lineage.json",
        bank_lineage_sha256="c" * 64,
        selection_path=tmp_path / "selection.json",
        selection_sha256="b" * 64,
    )

    horizon_splits.validate_manifest(manifest, require_four_way=True)
    assert set(base["train_group_ids"]).issubset(manifest["train_group_ids"])
    assert set(base["calibration_group_ids"]).issubset(manifest["train_group_ids"])
    assert manifest["early_stop_group_ids"] == base["validation_group_ids"]
    assert set(base["calibration_group_ids"]).isdisjoint(manifest["calibration_group_ids"])
    assert len(manifest["train_group_ids"]) == 104
    assert len(manifest["calibration_group_ids"]) == 100
    assert len(manifest["dev_audit_group_ids"]) == 400


def test_immutable_json_refuses_protocol_change(tmp_path) -> None:
    path = tmp_path / "selection.json"
    round3._write_once_json(path, {"status": "complete", "seed": 7})  # noqa: SLF001
    original = path.read_bytes()
    round3._write_once_json(path, {"seed": 7, "status": "complete"})  # noqa: SLF001
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="differs"):
        round3._write_once_json(path, {"status": "complete", "seed": 8})  # noqa: SLF001
    assert json.loads(path.read_text())["seed"] == 7
