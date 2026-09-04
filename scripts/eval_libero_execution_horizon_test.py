from __future__ import annotations

import importlib
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).with_name("eval_libero_execution_horizon.py")
sys.path.insert(0, str(_SCRIPT.parent))
evaluator = importlib.import_module("eval_libero_execution_horizon")


def test_default_eval_modes_and_horizon_remain_legacy() -> None:
    args = evaluator.build_parser().parse_args(["--output-dir", "/tmp/eval"])

    assert args.modes == list(evaluator.LEGACY_MODES)
    assert args.model_action_horizon == 10
    assert args.fixed_horizon == 9
    assert args.hierarchical_calibration_json is None
    assert args.hierarchical_aggregate_calibration_json is None


def test_disabled_aggregate_calibration_preserves_legacy_resume_signature() -> None:
    args = evaluator.build_parser().parse_args(["--output-dir", "/tmp/eval"])

    signature = evaluator._run_signature(args)  # noqa: SLF001

    assert "hierarchical_aggregate_calibration_json" not in signature


def test_enabled_aggregate_calibration_is_bound_into_resume_signature() -> None:
    args = evaluator.build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/eval",
            "--modes",
            "hierarchical_transformer",
            "--hierarchical-aggregate-calibration-json",
            "/tmp/aggregate.json",
        ]
    )

    signature = evaluator._run_signature(args)  # noqa: SLF001

    assert signature["hierarchical_aggregate_calibration_json"] == "/tmp/aggregate.json"
