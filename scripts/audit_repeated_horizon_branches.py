"""Audit repeated live horizon branches with paired rescue/regression metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="repeated_branch_outcomes.jsonl file or collector output directory.",
    )
    parser.add_argument("--reference-horizon", type=int, default=10)
    parser.add_argument(
        "--candidate-horizons",
        nargs="+",
        type=int,
        default=None,
        help="Repeated horizons to compare. Defaults to the intersection stored in all rows.",
    )
    parser.add_argument("--output-json")
    return parser


def _resolve_input(path: str) -> pathlib.Path:
    resolved = pathlib.Path(path)
    if resolved.is_dir():
        resolved = resolved / "repeated_branch_outcomes.jsonl"
    if not resolved.is_file():
        raise FileNotFoundError(f"Repeated-branch input does not exist: {resolved}")
    return resolved


def _load_rows(paths: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    resolved_inputs = []
    seen_roots: set[tuple[int, int, int, int]] = set()
    for path in paths:
        resolved = _resolve_input(path)
        resolved_inputs.append(str(resolved))
        with resolved.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                root = (
                    int(row["task_id"]),
                    int(row["episode_id"]),
                    int(row["decision_step"]),
                    int(row["root_seed"]),
                )
                if root in seen_roots:
                    raise ValueError(f"Duplicate root {root} at {resolved}:{line_number}.")
                seen_roots.add(root)
                rows.append(row)
    if not rows:
        raise ValueError("No repeated branch rows were found.")
    return rows, resolved_inputs


def _indexed_outcomes(row: dict[str, Any], horizon: int) -> dict[int, dict[str, Any]]:
    try:
        outcomes = row["outcomes_by_h"][str(horizon)]
    except KeyError as exc:
        raise ValueError(f"Root {row['root_seed']} has no H{horizon} outcomes.") from exc
    indexed = {int(outcome["repeat_index"]): outcome for outcome in outcomes}
    if len(indexed) != len(outcomes):
        raise ValueError(f"Root {row['root_seed']} H{horizon} has duplicate repeat indices.")
    return indexed


def _candidate_horizons(rows: list[dict[str, Any]], requested: list[int] | None) -> list[int]:
    available = {int(horizon) for horizon in rows[0]["repeated_horizons"]}
    for row in rows[1:]:
        available &= {int(horizon) for horizon in row["repeated_horizons"]}
    candidates = sorted(available if requested is None else set(requested))
    if not candidates or any(horizon not in available for horizon in candidates):
        raise ValueError(f"Candidate horizons must be present as repeats in every root; available={sorted(available)}.")
    return candidates


def _root_report(
    row: dict[str, Any],
    candidates: list[int],
    reference_horizon: int,
) -> dict[str, Any]:
    outcomes_by_h = {horizon: _indexed_outcomes(row, horizon) for horizon in candidates}
    repeat_indices = set(outcomes_by_h[reference_horizon])
    for horizon, outcomes in outcomes_by_h.items():
        if set(outcomes) != repeat_indices:
            raise ValueError(f"Root {row['root_seed']} H{horizon} does not share the reference repeat indices.")
        for repeat_index in repeat_indices:
            if int(outcomes[repeat_index]["policy_seed"]) != int(
                outcomes_by_h[reference_horizon][repeat_index]["policy_seed"]
            ):
                raise ValueError(
                    f"Root {row['root_seed']} repeat {repeat_index} does not use paired policy seeds."
                )

    horizon_reports: dict[str, Any] = {}
    reference = outcomes_by_h[reference_horizon]
    for horizon, outcomes in outcomes_by_h.items():
        successes = [bool(outcomes[index]["success"]) for index in sorted(repeat_indices)]
        reference_successes = [bool(reference[index]["success"]) for index in sorted(repeat_indices)]
        calls = [int(outcomes[index]["remaining_calls"]) for index in sorted(repeat_indices)]
        rescue_count = sum(success and not ref for success, ref in zip(successes, reference_successes, strict=True))
        regression_count = sum(
            not success and ref for success, ref in zip(successes, reference_successes, strict=True)
        )
        horizon_reports[str(horizon)] = {
            "success_count": sum(successes),
            "success_rate": sum(successes) / len(successes),
            "average_remaining_calls": sum(calls) / len(calls),
            "paired_rescue_count": rescue_count,
            "paired_regression_count": regression_count,
            "paired_net_rescue": rescue_count - regression_count,
        }

    selected_horizon = max(
        candidates,
        key=lambda horizon: (
            horizon_reports[str(horizon)]["success_count"],
            -horizon_reports[str(horizon)]["average_remaining_calls"],
            horizon,
        ),
    )
    hindsight_successes = sum(
        any(bool(outcomes_by_h[horizon][index]["success"]) for horizon in candidates)
        for index in repeat_indices
    )
    selected = horizon_reports[str(selected_horizon)]
    reference_report = horizon_reports[str(reference_horizon)]
    return {
        "task_id": int(row["task_id"]),
        "episode_id": int(row["episode_id"]),
        "decision_step": int(row["decision_step"]),
        "root_seed": int(row["root_seed"]),
        "num_repeats": len(repeat_indices),
        "reference_horizon": reference_horizon,
        "selected_empirical_best_horizon": selected_horizon,
        "selected_has_success_margin": selected["success_count"] > reference_report["success_count"],
        "empirical_best_success_count": selected["success_count"],
        "empirical_best_average_remaining_calls": selected["average_remaining_calls"],
        "hindsight_any_h_success_count": hindsight_successes,
        "horizons": horizon_reports,
    }


def _aggregate(root_reports: list[dict[str, Any]], candidates: list[int], reference_horizon: int) -> dict[str, Any]:
    total_trials = sum(int(root["num_repeats"]) for root in root_reports)
    fixed_horizons = {}
    paired_vs_reference = {}
    for horizon in candidates:
        reports = [root["horizons"][str(horizon)] for root in root_reports]
        success_count = sum(int(report["success_count"]) for report in reports)
        fixed_horizons[str(horizon)] = {
            "success_count": success_count,
            "success_rate": success_count / total_trials,
            "average_remaining_calls": sum(
                float(report["average_remaining_calls"]) * int(root["num_repeats"])
                for root, report in zip(root_reports, reports, strict=True)
            )
            / total_trials,
        }
        rescue_count = sum(int(report["paired_rescue_count"]) for report in reports)
        regression_count = sum(int(report["paired_regression_count"]) for report in reports)
        paired_vs_reference[str(horizon)] = {
            "rescue_count": rescue_count,
            "regression_count": regression_count,
            "net_rescue": rescue_count - regression_count,
        }

    empirical_best_successes = sum(int(root["empirical_best_success_count"]) for root in root_reports)
    empirical_best_calls = sum(
        float(root["empirical_best_average_remaining_calls"]) * int(root["num_repeats"])
        for root in root_reports
    )
    hindsight_successes = sum(int(root["hindsight_any_h_success_count"]) for root in root_reports)
    return {
        "num_roots": len(root_reports),
        "num_paired_trials": total_trials,
        "reference_horizon": reference_horizon,
        "fixed_horizons": fixed_horizons,
        "paired_vs_reference": paired_vs_reference,
        "roots_with_empirical_success_margin": sum(
            bool(root["selected_has_success_margin"]) for root in root_reports
        ),
        "empirical_root_level_best_h": {
            "success_count": empirical_best_successes,
            "success_rate": empirical_best_successes / total_trials,
            "average_remaining_calls": empirical_best_calls / total_trials,
        },
        "per_repeat_hindsight_any_h": {
            "success_count": hindsight_successes,
            "success_rate": hindsight_successes / total_trials,
        },
    }


def main(args: argparse.Namespace) -> None:
    rows, resolved_inputs = _load_rows(args.input)
    candidates = _candidate_horizons(rows, args.candidate_horizons)
    if args.reference_horizon not in candidates:
        raise ValueError("reference_horizon must be one of the repeated candidate horizons.")
    root_reports = [_root_report(row, candidates, args.reference_horizon) for row in rows]
    report = {
        "status": "complete",
        "semantics": (
            "Repeated live branches share one root snapshot and primary action, and pair horizons by continuation "
            "policy seed. The empirical root-level best-H result is an in-sample diagnostic, not a deployable "
            "selector or closed-loop success estimate. Per-repeat any-H is a stronger hindsight upper bound."
        ),
        "inputs": resolved_inputs,
        "candidate_horizons": candidates,
        "aggregate": _aggregate(root_reports, candidates, args.reference_horizon),
        "roots": root_reports,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        output_path = pathlib.Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
