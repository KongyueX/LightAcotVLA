"""Audit counterfactual H1-H10 labels for execution-horizon headroom.

The report is deliberately an optimistic root-state audit.  It answers whether
an H-only selector has label-level rescue opportunities and whether those
opportunities fit within a call budget.  It does not replace closed-loop LIBERO
evaluation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from openpi.execution_horizon import dataset as horizon_dataset
from openpi.execution_horizon import headroom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="Counterfactual HDF5 shard or directory.")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--reference",
        action="append",
        default=None,
        help="Reference selection: raw_h or h1 through h10. May be repeated. Default: raw_h, h9, h10.",
    )
    parser.add_argument(
        "--target-success-rate",
        action="append",
        type=float,
        default=None,
        help="Root-level success target. May be repeated. Defaults: 0.96, 0.966, 0.97.",
    )
    parser.add_argument(
        "--call-budget",
        action="append",
        type=float,
        default=None,
        help="Average remaining-call budget. May be repeated.",
    )
    return parser


def _reference_horizons(name: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    normalized = name.strip().lower()
    if normalized == "raw_h":
        return np.asarray(arrays["raw_h"], dtype=np.int64)
    if normalized.startswith("h") and normalized[1:].isdigit():
        horizon = int(normalized[1:])
        num_horizons = int(np.asarray(arrays["branch_success"]).shape[1])
        if 1 <= horizon <= num_horizons:
            return np.full(len(arrays["task_id"]), horizon, dtype=np.int64)
    raise ValueError(f"Unknown reference {name!r}; expected raw_h or h1 through h10.")


def _selection_metrics(
    horizons: np.ndarray,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    return headroom.selection_metrics(
        horizons[indices],
        arrays["branch_success"][indices],
        arrays["branch_timeout"][indices],
        arrays["remaining_calls"][indices],
        arrays["branch_valid"][indices],
        remaining_steps=arrays["remaining_steps"][indices],
    )


def _subset_report(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    references: dict[str, np.ndarray],
    target_success_rates: tuple[float, ...],
    requested_call_budgets: tuple[float, ...],
) -> dict[str, Any]:
    success = arrays["branch_success"][indices]
    timeout = arrays["branch_timeout"][indices]
    calls = arrays["remaining_calls"][indices]
    valid = arrays["branch_valid"][indices]
    steps = arrays["remaining_steps"][indices]
    num_horizons = int(success.shape[1])
    fixed_metrics = {
        f"h{horizon}": _selection_metrics(
            np.full(len(arrays["task_id"]), horizon, dtype=np.int64),
            arrays,
            indices,
        )
        for horizon in range(1, num_horizons + 1)
    }

    reference_reports: dict[str, Any] = {}
    reference_call_budgets = []
    for name, horizons in references.items():
        report = headroom.reference_headroom(
            horizons[indices],
            success,
            timeout,
            calls,
            valid,
            remaining_steps=steps,
        )
        report["targets"] = headroom.target_success_requirements(
            target_success_rates,
            horizons[indices],
            success,
            calls,
            valid,
        )
        reference_reports[name] = report
        reference_call_budgets.append(float(report["reference"]["average_remaining_calls"]))

    default_budget_names = ("h5", "h9", "h10")
    budgets = list(requested_call_budgets)
    budgets.extend(
        float(fixed_metrics[name]["average_remaining_calls"])
        for name in default_budget_names
        if name in fixed_metrics
    )
    budgets.extend(reference_call_budgets)
    unique_budgets = sorted({round(value, 9) for value in budgets})
    budgeted_oracles = [
        headroom.budgeted_success_oracle(
            budget,
            success,
            timeout,
            calls,
            valid,
            remaining_steps=steps,
        )
        for budget in unique_budgets
    ]
    return {
        "num_roots": int(len(indices)),
        "fixed_horizons": fixed_metrics,
        "references": reference_reports,
        "budgeted_success_oracles": budgeted_oracles,
    }


def main(args: argparse.Namespace) -> None:
    arrays = horizon_dataset.load_counterfactual_arrays(args.dataset)
    references_requested = tuple(args.reference or ("raw_h", "h9", "h10"))
    target_success_rates = tuple(args.target_success_rate or (0.96, 0.966, 0.97))
    requested_call_budgets = tuple(args.call_budget or ())
    references = {name: _reference_horizons(name, arrays) for name in references_requested}
    task_ids = np.asarray(arrays["task_id"], dtype=np.int64)
    subsets: dict[str, np.ndarray] = {"overall": np.arange(len(task_ids))}
    for task_id in sorted(np.unique(task_ids).tolist()):
        subsets[f"task{task_id}"] = np.flatnonzero(task_ids == task_id)
    hard_indices = np.flatnonzero(np.isin(task_ids, (8, 9)))
    if hard_indices.size:
        subsets["hard_tasks_8_9"] = hard_indices

    report = {
        "status": "complete",
        "semantics": (
            "Optimistic root-state counterfactual upper bound under the stored continuation policy. "
            "Rows are decision roots, not independent full episodes; this audit cannot establish "
            "closed-loop LIBERO success or prove rescue of the formal failed episodes."
        ),
        "dataset_inputs": list(args.dataset),
        "num_records": int(len(task_ids)),
        "references": list(references),
        "target_success_rates": list(target_success_rates),
        "subsets": {
            name: _subset_report(
                arrays,
                indices,
                references,
                target_success_rates,
                requested_call_budgets,
            )
            for name, indices in subsets.items()
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        output_path = pathlib.Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
