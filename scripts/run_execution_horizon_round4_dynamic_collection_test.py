from __future__ import annotations

import json
import pathlib

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
    assert command[command.index("--branch-repeats") + 1] == "3"
    assert "--fixed-continuation-horizon" not in command
    assert "--hierarchical-calibration-json" not in command
    assert "--hierarchical-aggregate-calibration-json" not in command
