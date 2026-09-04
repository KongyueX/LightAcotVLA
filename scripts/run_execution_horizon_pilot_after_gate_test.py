from __future__ import annotations

import importlib
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).with_name("run_execution_horizon_pilot_after_gate.py")
sys.path.insert(0, str(_SCRIPT.parent))
pilot = importlib.import_module("run_execution_horizon_pilot_after_gate")


def test_aggregate_artifact_cannot_be_overridden_from_pilot_cli() -> None:
    args = pilot.build_parser().parse_args(
        [
            "--post-summary",
            "/tmp/post.json",
            "--output-dir",
            "/tmp/controller",
            "--checkpoint-dir",
            "/tmp/checkpoint",
            "--pilot-output-dir",
            "/tmp/pilot",
        ]
    )

    assert not hasattr(args, "aggregate_calibration_json")


def test_eval_resume_config_is_bound_to_aggregate_and_pilot_shape(tmp_path: pathlib.Path) -> None:
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text("{}")
    config_path = tmp_path / "run_config.json"
    config = {
        "modes": ["hierarchical_transformer"],
        "hierarchical_aggregate_calibration_json": str(aggregate),
        "hierarchical_calibration_json": None,
        "seed": 42,
        "num_trials_per_task": 20,
        "model_action_horizon": 25,
        "max_tasks": 10,
        "task_suite_name": "libero_10",
    }
    config_path.write_text(json.dumps(config))

    assert pilot._verify_eval_run_config(  # noqa: SLF001
        config_path,
        aggregate_path=aggregate,
        seed=42,
        num_trials_per_task=20,
    ) == config

    config["seed"] = 7
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="seed"):
        pilot._verify_eval_run_config(  # noqa: SLF001
            config_path,
            aggregate_path=aggregate,
            seed=42,
            num_trials_per_task=20,
        )


def test_pilot_binding_is_immutable(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pilot_binding.json"
    expected = {"schema_version": 1, "artifact": "abc"}

    pilot._ensure_pilot_binding(path, expected)  # noqa: SLF001
    pilot._ensure_pilot_binding(path, expected)  # noqa: SLF001

    with pytest.raises(ValueError, match="binding differs"):
        pilot._ensure_pilot_binding(path, {**expected, "artifact": "changed"})  # noqa: SLF001
