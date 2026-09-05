from __future__ import annotations

import json
import pathlib

import pytest

import run_execution_horizon_round4_dynamic_collection as round4


def test_role_preserving_selection_uses_realized_groups_and_unattempted_fallback(tmp_path: pathlib.Path) -> None:
    eligible = [episode for episode in range(100, 300) if episode % 12 in round4.ALLOWED_OFFSETS]
    train_pool, calibration_pool, dev_pool = eligible[:30], eligible[30:70], eligible[70:]
    breadth = {"status": "complete", "purpose": "breadth_first_warm_start", "realized_by_task": {}}
    selection = {"tasks": {}}
    validation_groups = []
    for task in round4.TASK_IDS:
        breadth["realized_by_task"][str(task)] = {
            "train": {"episodes": train_pool[:10]},
            "calibration": {"episodes": calibration_pool[:10]},
            "dev_audit": {"episodes": dev_pool[:10]},
        }
        selection["tasks"][str(task)] = {
            "roles": {
                "train": {"primary": train_pool[:10], "fallback": train_pool[10:]},
                "calibration": {"primary": calibration_pool[:10], "fallback": calibration_pool[10:]},
                "dev_audit": {"primary": dev_pool[:40], "fallback": dev_pool[40:]},
            }
        }
        validation_groups.extend(task * round4.GROUP_ID_TASK_MULTIPLIER + episode for episode in range(3, 11))
        for role, episodes in (
            ("train", [*train_pool[:10], train_pool[10]]),
            ("calibration", calibration_pool[:10]),
            ("dev_audit", dev_pool[:10]),
        ):
            attempt = tmp_path / f"task{task:02d}/{role}/attempt-0000"
            attempt.mkdir(parents=True)
            (attempt / "run_config.json").write_text(json.dumps({"episodes": episodes}))

    result = round4.build_selection(
        breadth,
        selection,
        {"validation_group_ids": validation_groups},
        tmp_path,
    )

    for task in round4.TASK_IDS:
        roles = result["tasks"][str(task)]["roles"]
        assert {role: len(plan["primary"]) for role, plan in roles.items()} == round4.ROLE_TARGETS
        assert train_pool[10] not in roles["train"]["fallback"]
        assert set(roles["train"]["fallback"]).issubset(set(train_pool))
        assert set(roles["calibration"]["primary"] + roles["calibration"]["fallback"]).issubset(
            set(calibration_pool)
        )
        assert set(roles["dev_audit"]["primary"] + roles["dev_audit"]["fallback"]).issubset(set(dev_pool))
        assert all(episode < 100 for episode in roles["early_stop"]["primary"])


def test_collector_command_uses_ordered_dynamic_source_and_continuation(tmp_path: pathlib.Path) -> None:
    command = round4.build_collector_command(
        python=pathlib.Path("/venv/python"),
        collector=pathlib.Path("/code/collector.py"),
        output=tmp_path / "data",
        bank=pathlib.Path("/bank"),
        host="127.0.0.1",
        port=8040,
        task=2,
        episodes=[103, 104],
    )

    assert command[command.index("--source-policy") + 1] == "current_student"
    assert command[command.index("--continuation-policy") + 1] == "current_student"
    assert command[command.index("--student-mode") + 1] == "ordered_transformer"
    assert command[command.index("--source-iteration") + 1] == "4"
    assert command[command.index("--seed") + 1] == "47007"
    assert "--root-sampling" not in command
    assert command[command.index("--branch-repeats") + 1] == "3"
    assert "--fixed-continuation-horizon" not in command
    assert "--hierarchical-calibration-json" not in command
    assert "--hierarchical-aggregate-calibration-json" not in command


def test_cli_preserves_defaults_and_reuse_omits_legacy_selection_inputs() -> None:
    required = [
        "--code-dir", "/code", "--code-commit", "abc123", "--output-dir", "/output",
        "--initial-state-bank", "/bank", "--predictor-dir", "/predictor",
    ]
    default = round4.build_parser().parse_args([
        *required, "--round3-breadth-summary", "/breadth.json", "--round3-selection", "/selection.json",
        "--base-split-manifest", "/split.json",
    ])
    assert (default.seed, default.source_iteration, default.root_sampling) == (47007, 4, "call_offset")
    assert default.reuse_collection_summary is None

    reused = round4.build_parser().parse_args([
        *required, "--reuse-collection-summary", "/round4/collection/summary.json", "--seed", "57007",
        "--source-iteration", "5", "--root-sampling", "trajectory_reservoir",
    ])
    assert reused.round3_breadth_summary is reused.round3_selection is reused.base_split_manifest is None
    command = round4.build_collector_command(
        python=pathlib.Path("/python"), collector=pathlib.Path("/collector.py"),
        output=pathlib.Path("/data"), bank=pathlib.Path("/bank"), host="127.0.0.1", port=8040,
        task=0, episodes=[100], seed=reused.seed, source_iteration=reused.source_iteration,
        root_sampling=reused.root_sampling,
    )
    assert command[command.index("--seed") + 1] == "57007"
    assert command[command.index("--source-iteration") + 1] == "5"
    assert command[command.index("--root-sampling") + 1] == "trajectory_reservoir"
    assert command[command.index("--root-seed-task-stride") + 1] == "250000000"
    assert command[command.index("--branch-repeat-seed-stride") + 1] == "20000000"
    assert command[command.index("--branch-repeats") + 1] == "3"


def test_reuse_selection_uses_actual_roles_deduplicates_and_counts_realized_groups() -> None:
    previous = {"status": "complete", "num_roots": 180, "realized_by_task": {}}
    selection = {"tasks": {}}
    for task in round4.TASK_IDS:
        previous["realized_by_task"][str(task)] = {}
        selection["tasks"][str(task)] = {"roles": {}}
        for index, (role, target) in enumerate(round4.ROLE_TARGETS.items()):
            start = index * 100
            primary = list(range(start, start + target))
            actual = [*primary[:-1], start + target + 2]
            previous["realized_by_task"][str(task)][role] = {"episodes": actual}
            selection["tasks"][str(task)]["roles"][role] = {
                "primary": primary,
                "fallback": [primary[0], start + target + 2, start + target + 3, start + target + 3],
            }

    result = round4.build_reuse_selection(
        previous, selection, source_iteration=5, seed=57007, root_sampling="trajectory_reservoir",
        reuse_collection_summary=pathlib.Path("/round4/collection/summary.json"),
    )
    assert result["protocol"] == "h25_ordered_dynamic_continuation_round5_v1"
    assert result["primary_reused_groups"] == 180
    for task in round4.TASK_IDS:
        for index, (role, target) in enumerate(round4.ROLE_TARGETS.items()):
            plan = result["tasks"][str(task)]["roles"][role]
            assert plan["primary"] == previous["realized_by_task"][str(task)][role]["episodes"]
            assert plan["fallback"] == [index * 100 + target - 1, index * 100 + target + 3]
            assert not set(plan["primary"]) & set(plan["fallback"])

    realized = json.loads(json.dumps(previous["realized_by_task"]))
    realized["0"]["train"]["episodes"][0] = result["tasks"]["0"]["roles"]["train"]["fallback"][0]
    counts = round4._group_reuse_counts(realized, previous)
    assert counts["reused_groups"] == 179
    assert counts["new_groups"] == 1
    assert counts["by_role"]["train"] == {"reused_groups": 99, "new_groups": 1}

    selection["tasks"]["0"]["roles"]["calibration"]["fallback"].append(0)
    with pytest.raises(ValueError, match="crosses roles"):
        round4.build_reuse_selection(
            previous, selection, source_iteration=5, seed=57007, root_sampling="trajectory_reservoir",
            reuse_collection_summary=pathlib.Path("/round4/collection/summary.json"),
        )
