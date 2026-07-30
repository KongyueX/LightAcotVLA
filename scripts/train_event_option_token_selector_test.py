# ruff: noqa: SLF001

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np

_SCRIPT = pathlib.Path(__file__).with_name("train_event_option_token_selector.py")
sys.path.insert(0, str(_SCRIPT.parent))
selector = importlib.import_module("train_event_option_token_selector")


def test_discrete_oracle_and_global_token() -> None:
    cached = np.zeros((2, 10, 7), dtype=np.float32)
    cached[:, :, :6] = np.arange(10, dtype=np.float32)[None, :, None]
    fresh = np.zeros((2, 10, 7), dtype=np.float32)
    fresh[0, 0, :6] = 2.0
    fresh[1, 0, :6] = 8.0
    np.testing.assert_array_equal(
        selector._discrete_oracle_token(cached, fresh),
        np.asarray([2, 8]),
    )
    assert selector._calibrate_global_token(cached, fresh) == 5


def test_continuous_calibration_and_fixed_gripper() -> None:
    cached = np.zeros((2, 10, 7), dtype=np.float32)
    cached[:, :, :6] = np.arange(10, dtype=np.float32)[None, :, None]
    cached[:, :, 6] = np.arange(10, dtype=np.float32)[None]
    fresh = np.zeros((2, 10, 7), dtype=np.float32)
    fresh[:, 0, :6] = 2.5
    phase = selector._calibrate_global_continuous(cached, fresh, grid_size=19)
    assert phase == 2.5
    actions = selector._action_from_hard_tokens(
        cached,
        np.asarray([2, 8]),
        fixed_gripper_age=4,
    )
    np.testing.assert_allclose(actions[:, :6], np.asarray([[2.0] * 6, [8.0] * 6]))
    np.testing.assert_allclose(actions[:, 6], 4.0)


def test_model_budget_matches_221k_family() -> None:
    parameters = selector.estimate_parameter_count(
        image_views=2,
        image_channels=3,
        state_dim=32,
        action_dim=32,
        env_action_dim=7,
        max_executed_steps=4,
        action_horizon=10,
    )
    assert 200_000 < parameters < 250_000
