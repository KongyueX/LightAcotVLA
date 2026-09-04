"""Deployment helpers for the ordered execution-horizon head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ORDERED_SELECTED_H_KEY = "execution_horizon_ordered_selected_h"
CANDIDATE_HORIZONS_KEY = "execution_horizon_candidate_horizons"


def selected_horizon(result: Mapping[str, Any], *, model_action_horizon: int) -> int:
    """Read and validate the ordered head's selected execution horizon."""

    if ORDERED_SELECTED_H_KEY not in result:
        raise KeyError(f"Policy response is missing {ORDERED_SELECTED_H_KEY!r}.")
    if CANDIDATE_HORIZONS_KEY not in result:
        raise KeyError(f"Policy response is missing {CANDIDATE_HORIZONS_KEY!r}.")

    raw_selected = np.asarray(result[ORDERED_SELECTED_H_KEY])
    if raw_selected.size != 1:
        raise ValueError(
            f"{ORDERED_SELECTED_H_KEY} must contain one value, got shape {raw_selected.shape}."
        )
    selected_value = raw_selected.item()
    selected = int(selected_value)
    if float(selected) != float(selected_value):
        raise ValueError(f"Ordered execution horizon must be an integer, got {selected_value!r}.")

    raw_candidates = np.asarray(result[CANDIDATE_HORIZONS_KEY])
    if raw_candidates.ndim != 1 or raw_candidates.size == 0:
        raise ValueError(
            f"{CANDIDATE_HORIZONS_KEY} must be a non-empty vector, got shape {raw_candidates.shape}."
        )
    candidates = tuple(int(value) for value in raw_candidates.tolist())
    if selected not in candidates:
        raise ValueError(f"Ordered execution horizon H{selected} is not in candidate_horizons={candidates}.")
    if selected <= 0 or selected > model_action_horizon:
        raise ValueError(
            f"Ordered execution horizon H{selected} must lie within model_action_horizon={model_action_horizon}."
        )
    return selected
