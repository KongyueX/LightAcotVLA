from __future__ import annotations

import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_ordered_transformer_selects_model_output_without_calibration() -> None:
    args = evaluator.build_parser().parse_args(
        [
            "--output-dir",
            "/tmp/eval",
            "--modes",
            "ordered_transformer",
            "--model-action-horizon",
            "25",
        ]
    )
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(20, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25], dtype=np.int32),
    }

    selected, info = evaluator._select_horizon(  # noqa: SLF001
        evaluator.ORDERED_MODE,
        result,
        args=args,
        budget_state=SimpleNamespace(),
    )

    assert selected == 20
    assert info == {
        "raw_horizon": 20,
        "budget_limited": 0.0,
        "selector_policy": "ordered_transformer",
    }
    assert args.hierarchical_calibration_json is None
    assert args.hierarchical_aggregate_calibration_json is None


@pytest.mark.parametrize(
    ("selected", "candidates", "model_horizon", "message"),
    [
        (20, [5, 10, 15, 25], 25, "not in candidate_horizons"),
        (25, [5, 10, 15, 20, 25], 20, "model_action_horizon=20"),
    ],
)
def test_ordered_transformer_rejects_invalid_selected_horizon(
    selected: int,
    candidates: list[int],
    model_horizon: int,
    message: str,
) -> None:
    args = SimpleNamespace(model_action_horizon=model_horizon)
    result = {
        "execution_horizon_ordered_selected_h": np.asarray(selected, dtype=np.int32),
        "execution_horizon_candidate_horizons": np.asarray(candidates, dtype=np.int32),
    }

    with pytest.raises(ValueError, match=message):
        evaluator._select_horizon(  # noqa: SLF001
            evaluator.ORDERED_MODE,
            result,
            args=args,
            budget_state=SimpleNamespace(),
        )
