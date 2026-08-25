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
