"""Deployment helpers for the ordered execution-horizon head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ORDERED_SELECTED_H_KEY = "execution_horizon_ordered_selected_h"
CANDIDATE_HORIZONS_KEY = "execution_horizon_candidate_horizons"
ORDERED_PROBABILITY_KEY = "execution_horizon_ordered_horizon_probability"


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


def select_h10_hysteresis(
    result: Mapping[str, Any],
    *,
    model_action_horizon: int,
    previous_horizon: int,
    enter_margin: float = 0.10,
    hold_margin: float = 0.05,
) -> tuple[int, dict[str, Any]]:
    """Anchor uncertain long-H choices to H10 using ordered probability margins."""
    if not (
        np.isfinite(enter_margin)
        and np.isfinite(hold_margin)
        and 0 <= hold_margin <= enter_margin <= 1
    ):
        raise ValueError("Hysteresis margins must be finite and satisfy 0 <= hold_margin <= enter_margin <= 1.")
    raw = selected_horizon(result, model_action_horizon=model_action_horizon)
    candidates = np.asarray(result[CANDIDATE_HORIZONS_KEY], dtype=np.int64)
    if 10 not in candidates:
        raise ValueError("H10 hysteresis requires H10 in candidate_horizons.")
    if ORDERED_PROBABILITY_KEY not in result:
        raise KeyError(f"Policy response is missing {ORDERED_PROBABILITY_KEY!r}.")
    probabilities = np.asarray(result[ORDERED_PROBABILITY_KEY], dtype=np.float64).reshape(-1)
    if probabilities.size != candidates.size or probabilities.size < 2:
        raise ValueError("Ordered probabilities must match candidate_horizons and contain at least two choices.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Ordered probabilities must be finite and non-negative.")
    if not np.isclose(probabilities.sum(), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Ordered probabilities must sum to one.")
    if int(candidates[np.argmax(probabilities)]) != raw:
        raise ValueError("Ordered selected horizon must agree with the probability argmax.")

    top_two = np.sort(probabilities)[-2:]
    margin = float(top_two[1] - top_two[0])
    if raw <= 10:
        selected, threshold, reason = raw, 0.0, "short_immediate"
    else:
        holding = raw == previous_horizon
        threshold = float(hold_margin if holding else enter_margin)
        if margin >= threshold:
            selected = raw
            reason = "long_hold" if holding else "long_enter"
        else:
            selected, reason = 10, "h10_anchor"
    return selected, {
        "raw_horizon": raw,
        "selector_policy": "ordered_h10_hysteresis",
        "hysteresis_margin": margin,
        "hysteresis_threshold": threshold,
        "hysteresis_reason": reason,
        "hysteresis_changed": selected != raw,
    }
