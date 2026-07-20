"""Counterfactual execution-horizon headroom and failure-fixability audits.

Each input row is one simulator root state with outcomes for every candidate
execution horizon.  The utilities in this module compute optimistic, root-level
upper bounds.  They do not model the state distribution induced by repeatedly
applying an oracle in a closed-loop episode.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np


def _validated_branch_arrays(
    branch_success: np.ndarray,
    branch_timeout: np.ndarray,
    remaining_calls: np.ndarray,
    branch_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    success = np.asarray(branch_success, dtype=np.bool_)
    timeout = np.asarray(branch_timeout, dtype=np.bool_)
    calls = np.asarray(remaining_calls, dtype=np.float64)
    valid = np.asarray(branch_valid, dtype=np.bool_)
    if success.ndim != 2 or success.shape[0] == 0 or success.shape[1] == 0:
        raise ValueError("branch_success must have non-empty shape [num_roots, num_horizons].")
    if timeout.shape != success.shape or calls.shape != success.shape or valid.shape != success.shape:
        raise ValueError("All branch arrays must have the same [num_roots, num_horizons] shape.")
    if np.any(~np.any(valid, axis=1)):
        raise ValueError("Every root must have at least one valid counterfactual branch.")
    if np.any(~np.isfinite(calls[valid])) or np.any(calls[valid] < 0):
        raise ValueError("Valid remaining_calls entries must be finite and non-negative.")
    return success, timeout, calls, valid


def _validate_horizons(horizons: np.ndarray, valid: np.ndarray) -> np.ndarray:
    selected = np.asarray(horizons, dtype=np.int64)
    if selected.shape != (valid.shape[0],):
        raise ValueError(f"Expected one selected horizon per root; got {selected.shape}.")
    if np.any((selected < 1) | (selected > valid.shape[1])):
        raise ValueError("Selected horizons fall outside the stored counterfactual range.")
    row_indices = np.arange(valid.shape[0])
    if np.any(~valid[row_indices, selected - 1]):
        raise ValueError("A selected horizon refers to an invalid counterfactual branch.")
    return selected


def _distribution(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _lowest_cost_horizons(remaining_calls: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Select the lowest-cost eligible branch, preferring larger H on cost ties."""
    if remaining_calls.shape != eligible.shape:
        raise ValueError("remaining_calls and eligible must have the same shape.")
    has_eligible = np.any(eligible, axis=1)
    if np.any(~has_eligible):
        raise ValueError("Every root must have at least one eligible branch.")
    masked_calls = np.where(eligible, remaining_calls, np.inf)
    minimum_calls = np.min(masked_calls, axis=1, keepdims=True)
    tied = eligible & np.isclose(masked_calls, minimum_calls, rtol=0.0, atol=1e-9)
    horizon_values = np.arange(1, remaining_calls.shape[1] + 1, dtype=np.int64)
    return np.max(np.where(tied, horizon_values[None, :], 0), axis=1)


def selection_metrics(
    horizons: np.ndarray,
    branch_success: np.ndarray,
    branch_timeout: np.ndarray,
    remaining_calls: np.ndarray,
    branch_valid: np.ndarray,
    *,
    remaining_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one selected counterfactual horizon per root."""
    success, timeout, calls, valid = _validated_branch_arrays(
        branch_success,
        branch_timeout,
        remaining_calls,
        branch_valid,
    )
    selected = _validate_horizons(horizons, valid)
    row_indices = np.arange(success.shape[0])
    selected_success = success[row_indices, selected - 1]
    selected_timeout = timeout[row_indices, selected - 1]
    selected_calls = calls[row_indices, selected - 1]
    result: dict[str, Any] = {
        "num_roots": int(success.shape[0]),
        "success_count": int(np.sum(selected_success)),
        "success_rate": float(np.mean(selected_success)),
        "timeout_count": int(np.sum(selected_timeout)),
        "timeout_rate": float(np.mean(selected_timeout)),
        "average_remaining_calls": float(np.mean(selected_calls)),
        "average_horizon": float(np.mean(selected)),
        "h_distribution": _distribution(selected),
    }
    if remaining_steps is not None:
        steps = np.asarray(remaining_steps, dtype=np.float64)
        if steps.shape != success.shape:
            raise ValueError("remaining_steps must have the same shape as branch_success.")
        if np.any(~np.isfinite(steps[valid])) or np.any(steps[valid] < 0):
            raise ValueError("Valid remaining_steps entries must be finite and non-negative.")
        result["average_remaining_steps"] = float(np.mean(steps[row_indices, selected - 1]))
    return result


def reference_headroom(
    reference_horizons: np.ndarray,
    branch_success: np.ndarray,
    branch_timeout: np.ndarray,
    remaining_calls: np.ndarray,
    branch_valid: np.ndarray,
    *,
    remaining_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure how many reference failures have a successful alternative H."""
    success, timeout, calls, valid = _validated_branch_arrays(
        branch_success,
        branch_timeout,
        remaining_calls,
        branch_valid,
    )
    reference = _validate_horizons(reference_horizons, valid)
    row_indices = np.arange(success.shape[0])
    reference_success = success[row_indices, reference - 1]
    any_success = np.any(success & valid, axis=1)
    fixable = ~reference_success & any_success
    unfixable = ~reference_success & ~any_success

    cheapest_valid = _lowest_cost_horizons(calls, valid)
    cheapest_success = np.zeros(success.shape[0], dtype=np.int64)
    if np.any(any_success):
        cheapest_success[any_success] = _lowest_cost_horizons(
            calls[any_success],
            (success & valid)[any_success],
        )
    oracle_horizons = np.where(any_success, cheapest_success, cheapest_valid)
    reference_metrics = selection_metrics(
        reference,
        success,
        timeout,
        calls,
        valid,
        remaining_steps=remaining_steps,
    )
    oracle_metrics = selection_metrics(
        oracle_horizons,
        success,
        timeout,
        calls,
        valid,
        remaining_steps=remaining_steps,
    )
    reference_calls = calls[row_indices, reference - 1]
    oracle_calls = calls[row_indices, oracle_horizons - 1]
    failure_count = int(np.sum(~reference_success))
    return {
        "reference": reference_metrics,
        "success_first_oracle": oracle_metrics,
        "reference_failure_count": failure_count,
        "fixable_failure_count": int(np.sum(fixable)),
        "unfixable_failure_count": int(np.sum(unfixable)),
        "fixable_fraction_of_reference_failures": (
            float(np.mean(fixable[~reference_success])) if failure_count else 0.0
        ),
        "success_upper_bound": float(np.mean(any_success)),
        "average_remaining_calls_delta_vs_reference": float(np.mean(oracle_calls - reference_calls)),
        "average_rescue_calls_delta_vs_reference": (
            float(np.mean(oracle_calls[fixable] - reference_calls[fixable])) if np.any(fixable) else None
        ),
        "oracle_h_distribution": _distribution(oracle_horizons),
    }


def _cheapest_base_and_upgrades(
    branch_success: np.ndarray,
    remaining_calls: np.ndarray,
    branch_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    success = np.asarray(branch_success, dtype=np.bool_)
    calls = np.asarray(remaining_calls, dtype=np.float64)
    valid = np.asarray(branch_valid, dtype=np.bool_)
    if success.ndim != 2 or calls.shape != success.shape or valid.shape != success.shape:
        raise ValueError("branch_success, remaining_calls and branch_valid must share a 2D shape.")
    if np.any(~np.any(valid, axis=1)):
        raise ValueError("Every root must have at least one valid counterfactual branch.")
    cheapest = _lowest_cost_horizons(calls, valid)
    row_indices = np.arange(success.shape[0])
    cheapest_success_mask = success[row_indices, cheapest - 1]
    any_success = np.any(success & valid, axis=1)
    successful_horizons = np.zeros(success.shape[0], dtype=np.int64)
    if np.any(any_success):
        successful_horizons[any_success] = _lowest_cost_horizons(
            calls[any_success],
            (success & valid)[any_success],
        )
    candidate = ~cheapest_success_mask & any_success
    incremental_cost = np.full(success.shape[0], np.inf, dtype=np.float64)
    incremental_cost[candidate] = (
        calls[row_indices[candidate], successful_horizons[candidate] - 1]
        - calls[row_indices[candidate], cheapest[candidate] - 1]
    )
    if np.any(incremental_cost[candidate] < -1e-9):
        raise ValueError("Successful upgrade cost cannot be lower than the cheapest valid branch cost.")
    incremental_cost[candidate] = np.maximum(incremental_cost[candidate], 0.0)
    return cheapest, cheapest_success_mask, successful_horizons, candidate, incremental_cost


def budgeted_success_oracle(
    average_call_budget: float,
    branch_success: np.ndarray,
    branch_timeout: np.ndarray,
    remaining_calls: np.ndarray,
    branch_valid: np.ndarray,
    *,
    remaining_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Maximize root success under an average remaining-call budget.

    Starting from the cheapest valid branch at every root, each successful
    upgrade has unit benefit.  Sorting upgrades by incremental call cost is
    therefore the exact finite-label optimum for this root-independent audit.
    """
    success, timeout, calls, valid = _validated_branch_arrays(
        branch_success,
        branch_timeout,
        remaining_calls,
        branch_valid,
    )
    if not math.isfinite(average_call_budget) or average_call_budget < 0:
        raise ValueError("average_call_budget must be finite and non-negative.")
    cheapest, _, successful_horizons, candidate, incremental_cost = _cheapest_base_and_upgrades(
        success,
        calls,
        valid,
    )
    row_indices = np.arange(success.shape[0])
    base_total = float(np.sum(calls[row_indices, cheapest - 1]))
    allowed_total = float(average_call_budget * success.shape[0])
    if allowed_total + 1e-9 < base_total:
        return {
            "feasible": False,
            "average_call_budget": float(average_call_budget),
            "minimum_achievable_average_calls": base_total / success.shape[0],
        }

    selected = cheapest.copy()
    spent = base_total
    upgrade_indices = np.flatnonzero(candidate)
    order = upgrade_indices[np.argsort(incremental_cost[upgrade_indices], kind="stable")]
    selected_upgrades = 0
    for root_index in order:
        cost = float(incremental_cost[root_index])
        if spent + cost > allowed_total + 1e-9:
            continue
        selected[root_index] = successful_horizons[root_index]
        spent += cost
        selected_upgrades += 1
    metrics = selection_metrics(
        selected,
        success,
        timeout,
        calls,
        valid,
        remaining_steps=remaining_steps,
    )
    return {
        "feasible": True,
        "average_call_budget": float(average_call_budget),
        "minimum_achievable_average_calls": base_total / success.shape[0],
        "unused_average_call_budget": max(allowed_total - spent, 0.0) / success.shape[0],
        "selected_successful_upgrades": selected_upgrades,
        "metrics": metrics,
    }


def target_success_requirements(
    target_success_rates: Sequence[float],
    reference_horizons: np.ndarray,
    branch_success: np.ndarray,
    remaining_calls: np.ndarray,
    branch_valid: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute the minimum branch-label cost needed to reach each root-success target."""
    success = np.asarray(branch_success, dtype=np.bool_)
    calls = np.asarray(remaining_calls, dtype=np.float64)
    valid = np.asarray(branch_valid, dtype=np.bool_)
    cheapest, cheapest_success, _, candidate, incremental_cost = _cheapest_base_and_upgrades(
        success,
        calls,
        valid,
    )
    reference = _validate_horizons(reference_horizons, valid)
    row_indices = np.arange(success.shape[0])
    reference_success_count = int(np.sum(success[row_indices, reference - 1]))
    base_success_count = int(np.sum(cheapest_success))
    base_total = float(np.sum(calls[row_indices, cheapest - 1]))
    sorted_incremental = np.sort(incremental_cost[candidate])
    maximum_success_count = base_success_count + len(sorted_incremental)
    result = []
    for target_rate in target_success_rates:
        if not math.isfinite(target_rate) or not 0 <= target_rate <= 1:
            raise ValueError("Target success rates must be finite and in [0, 1].")
        target_count = int(math.ceil(target_rate * success.shape[0] - 1e-12))
        upgrades_needed = max(target_count - base_success_count, 0)
        achievable = target_count <= maximum_success_count
        if achievable:
            minimum_total = base_total + float(np.sum(sorted_incremental[:upgrades_needed]))
            minimum_average_calls: float | None = minimum_total / success.shape[0]
        else:
            minimum_average_calls = None
        result.append(
            {
                "target_success_rate": float(target_rate),
                "target_success_count": target_count,
                "additional_successes_needed_vs_reference": max(target_count - reference_success_count, 0),
                "achievable_in_root_labels": achievable,
                "minimum_average_remaining_calls": minimum_average_calls,
            }
        )
    return result
