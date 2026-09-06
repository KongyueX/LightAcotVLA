from __future__ import annotations

import numpy as np
import pytest

from openpi.execution_horizon import ordered


def test_selected_horizon_accepts_candidate_within_model_chunk() -> None:
    result = {
        ordered.ORDERED_SELECTED_H_KEY: np.asarray(20, dtype=np.int32),
        ordered.CANDIDATE_HORIZONS_KEY: np.asarray([5, 10, 15, 20, 25], dtype=np.int32),
    }

    assert ordered.selected_horizon(result, model_action_horizon=25) == 20


@pytest.mark.parametrize(
    ("selected", "candidates", "model_horizon", "message"),
    [
        (20, [5, 10, 15, 25], 25, "not in candidate_horizons"),
        (25, [5, 10, 15, 20, 25], 20, "model_action_horizon=20"),
    ],
)
def test_selected_horizon_rejects_invalid_output(
    selected: int,
    candidates: list[int],
    model_horizon: int,
    message: str,
) -> None:
    result = {
        ordered.ORDERED_SELECTED_H_KEY: np.asarray(selected, dtype=np.int32),
        ordered.CANDIDATE_HORIZONS_KEY: np.asarray(candidates, dtype=np.int32),
    }

    with pytest.raises(ValueError, match=message):
        ordered.selected_horizon(result, model_action_horizon=model_horizon)


def _probability_result(probabilities: list[float]) -> dict[str, np.ndarray]:
    candidates = np.asarray([5, 10, 15, 20, 25], dtype=np.int32)
    return {
        ordered.ORDERED_SELECTED_H_KEY: candidates[np.argmax(probabilities)],
        ordered.CANDIDATE_HORIZONS_KEY: candidates,
        ordered.ORDERED_PROBABILITY_KEY: np.asarray(probabilities),
    }


@pytest.mark.parametrize("probabilities", [[0.22, 0.21, 0.20, 0.19, 0.18], [0.21, 0.22, 0.20, 0.19, 0.18]])
def test_h10_hysteresis_releases_short_choices_immediately(probabilities) -> None:
    result = _probability_result(probabilities)
    selected, info = ordered.select_h10_hysteresis(result, model_action_horizon=25, previous_horizon=25)
    assert selected == result[ordered.ORDERED_SELECTED_H_KEY]
    assert info["hysteresis_reason"] == "short_immediate"
    assert info["hysteresis_changed"] is False


def test_h10_hysteresis_anchors_weak_long_choice_without_changing_raw_helper() -> None:
    result = _probability_result([0.18, 0.19, 0.20, 0.21, 0.22])
    selected, info = ordered.select_h10_hysteresis(result, model_action_horizon=25, previous_horizon=10)
    assert selected == 10
    assert info["raw_horizon"] == 25
    assert info["hysteresis_reason"] == "h10_anchor"
    assert info["hysteresis_changed"] is True
    assert ordered.selected_horizon(result, model_action_horizon=25) == 25


def test_h10_hysteresis_uses_different_enter_and_hold_margins() -> None:
    result = _probability_result([0.13, 0.15, 0.17, 0.24, 0.31])
    enter, enter_info = ordered.select_h10_hysteresis(result, model_action_horizon=25, previous_horizon=20)
    hold, hold_info = ordered.select_h10_hysteresis(result, model_action_horizon=25, previous_horizon=25)
    assert enter == 10
    assert enter_info["hysteresis_threshold"] == 0.10
    assert hold == 25
    assert hold_info["hysteresis_threshold"] == 0.05
    assert hold_info["hysteresis_reason"] == "long_hold"


def test_h10_hysteresis_accepts_strong_long_choice_after_truncated_previous_h() -> None:
    result = _probability_result([0.10, 0.15, 0.20, 0.15, 0.40])
    selected, info = ordered.select_h10_hysteresis(result, model_action_horizon=25, previous_horizon=13)
    assert selected == 25
    assert info["hysteresis_reason"] == "long_enter"
    assert info["hysteresis_changed"] is False
    assert info["selector_policy"] == "ordered_h10_hysteresis"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (ordered.CANDIDATE_HORIZONS_KEY, [5, 7, 15, 20, 25], "requires H10"),
        (ordered.ORDERED_PROBABILITY_KEY, [0.25, 0.25, 0.50], "match candidate_horizons"),
        (ordered.ORDERED_PROBABILITY_KEY, [0.1, 0.1, 0.1, 0.1, float("nan")], "finite"),
        (ordered.ORDERED_PROBABILITY_KEY, [-0.1, 0.1, 0.2, 0.3, 0.5], "non-negative"),
        (ordered.ORDERED_PROBABILITY_KEY, [0.1, 0.1, 0.1, 0.1, 0.5], "sum to one"),
        (ordered.ORDERED_SELECTED_H_KEY, 20, "probability argmax"),
    ],
)
def test_h10_hysteresis_rejects_invalid_probability_or_candidate_inputs(key, value, message) -> None:
    result = _probability_result([0.10, 0.15, 0.20, 0.15, 0.40])
    result[key] = np.asarray(value)
    with pytest.raises(ValueError, match=message):
        ordered.select_h10_hysteresis(result, model_action_horizon=25, previous_horizon=10)


@pytest.mark.parametrize("enter, hold", [(0.1, 0.2), (1.1, 0.05), (0.1, -0.05), (float("inf"), 0.05)])
def test_h10_hysteresis_rejects_invalid_margins(enter, hold) -> None:
    result = _probability_result([0.10, 0.15, 0.20, 0.15, 0.40])
    with pytest.raises(ValueError, match="Hysteresis margins"):
        ordered.select_h10_hysteresis(
            result, model_action_horizon=25, previous_horizon=10, enter_margin=enter, hold_margin=hold
        )
