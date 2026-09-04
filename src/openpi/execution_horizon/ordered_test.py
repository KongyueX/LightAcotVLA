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
