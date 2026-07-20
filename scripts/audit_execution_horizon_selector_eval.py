"""Create a paired success/efficiency audit from a selector evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--reference-mode", default="fixed_h9")
    parser.add_argument("--output-json", default=None)
    return parser


def _mean(rows: list[dict[str, str]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.mean(values))


def main(args: argparse.Namespace) -> None:
    eval_dir = pathlib.Path(args.eval_dir)
    with (eval_dir / "rollout_rows.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_mode: dict[str, dict[tuple[int, int, int], dict[str, str]]] = {}
    for row in rows:
        key = (int(row["task_id"]), int(row["episode"]), int(row["initial_state_id"]))
        by_mode.setdefault(row["mode"], {})[key] = row
    if args.reference_mode not in by_mode:
        raise ValueError(f"Reference mode {args.reference_mode!r} was not found.")
    reference = by_mode[args.reference_mode]
    modes: dict[str, Any] = {}
    for mode, indexed in by_mode.items():
        common = sorted(set(reference).intersection(indexed))
        reference_success = np.asarray([int(reference[key]["success"]) for key in common], dtype=np.int8)
        candidate_success = np.asarray([int(indexed[key]["success"]) for key in common], dtype=np.int8)
        selected_rows = [indexed[key] for key in common]
        rescues = int(np.sum((reference_success == 0) & (candidate_success == 1)))
        regressions = int(np.sum((reference_success == 1) & (candidate_success == 0)))
        modes[mode] = {
            "paired_episodes": len(common),
            "success_count": int(np.sum(candidate_success)),
            "success_rate": float(np.mean(candidate_success)),
            "average_policy_calls": _mean(selected_rows, "policy_calls"),
            "average_h": _mean(selected_rows, "avg_h"),
            "average_policy_rpc_wall_ms": _mean(selected_rows, "policy_rpc_wall_total_ms"),
            "average_episode_elapsed_ms": _mean(selected_rows, "actual_episode_elapsed_total_ms"),
            "rescue_count_vs_reference": rescues,
            "regression_count_vs_reference": regressions,
            "net_rescue_vs_reference": rescues - regressions,
        }
    result = {
        "status": "complete",
        "eval_dir": str(eval_dir.resolve()),
        "reference_mode": args.reference_mode,
        "paired_key": ["task_id", "episode", "initial_state_id"],
        "modes": modes,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    output_path = pathlib.Path(args.output_json) if args.output_json else eval_dir / "paired_audit.json"
    output_path.write_text(payload + "\n")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
