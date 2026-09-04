from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import run_execution_horizon_round3_breadth_resume as breadth


class _FakeController:
    TASK_IDS = (0, 1)
    ROLE_TARGETS = {"train": 10, "calibration": 10, "dev_audit": 40}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def next_fallback_batch(*args, **kwargs):  # pragma: no cover - primary pools suffice here.
        del args, kwargs
        raise AssertionError("unexpected fallback")

    def _run_attempt(self, **kwargs):
        self.calls.append(kwargs)
        task = int(kwargs["task_id"])
        return {(task, int(episode), 50, task * 1_000_000 + int(episode)) for episode in kwargs["episodes"]}


def _state(tmp_path: pathlib.Path) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for task in (0, 1):
        result[task] = {}
        for role in ("train", "calibration", "dev_audit"):
            roots = (
                {(task, 100 + index, 50, task * 1_000_000 + 100 + index) for index in range(10)}
                if task == 0
                else set()
            )
            result[task][role] = {
                "role_dir": tmp_path / f"task{task:02d}" / role,
                "attempted": {root[1] for root in roots},
                "roots": roots,
                "maximum_attempt": 0 if roots else -1,
            }
    return result


def test_breadth_resume_skips_existing_coverage_and_reuses_frozen_identity(tmp_path: pathlib.Path) -> None:
    selection = {
        "tasks": {
            str(task): {
                "roles": {
                    role: {"primary": list(range(200 + role_index * 20, 210 + role_index * 20))}
                    for role_index, role in enumerate(("train", "calibration", "dev_audit"))
                }
            }
            for task in (0, 1)
        }
    }
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    protocol = {
        "collector_script": "/frozen/2fe308c/collector.py",
        "initial_state_bank": "/frozen/bank",
        "aggregate_calibration_json": "/frozen/aggregate.json",
        "host": "127.0.0.1",
        "port": 8040,
        "code_commit": "2fe308c",
    }
    controller = _FakeController()
    state = _state(tmp_path)

    breadth._collect_to_minimum(
        controller,
        output=tmp_path,
        protocol=protocol,
        state=state,
        python=pathlib.Path("/frozen/python"),
        episodes_per_attempt=10,
    )

    assert [(call["task_id"], call["role"]) for call in controller.calls] == [
        (1, "train"),
        (1, "calibration"),
        (1, "dev_audit"),
    ]
    assert all(call["protocol_identity"] is protocol for call in controller.calls)
    assert all(len(state[task][role]["roots"]) >= 10 for task in (0, 1) for role in controller.ROLE_TARGETS)
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "split_manifest.json").exists()


def test_snapshot_name_is_separate_from_full_round3_outputs() -> None:
    assert breadth.SNAPSHOT_DIRECTORY_NAME == "breadth_first_min10"
    assert breadth.MINIMUM_ROOTS_PER_TASK_ROLE == 10
