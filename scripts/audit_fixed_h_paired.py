"""Audit strictly paired Fixed-H LIBERO rollout CSVs.

The audit is intentionally independent of an evaluator's aggregate summary. It
first proves that every run contains the same ``(task_id, episode)`` keys and
the same ``initial_state_id`` for each key, then computes paired success and
latency comparisons. Continuous metrics use a hierarchical bootstrap that
resamples tasks and, within each sampled task, task-by-initial-state clusters.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
from typing import Any

import numpy as np

_KEY_FIELDS = ("task_id", "episode")
_METRICS = (
    "policy_calls",
    "actual_policy_total_ms",
    "policy_rpc_wall_total_ms",
    "actual_episode_elapsed_total_ms",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=ROLLOUT_ROWS_CSV",
        help="Named rollout CSV. Repeat for every Fixed-H run.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="REFERENCE:CANDIDATE",
        help="Directed comparison. Defaults to the first run against every later run.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success-noninferiority-margin", type=float, default=0.01)
    return parser


def _parse_assignment(value: str, *, separator: str, kind: str) -> tuple[str, str]:
    if separator not in value:
        raise ValueError(f"{kind} must have the form NAME{separator}VALUE, got {value!r}.")
    left, right = value.split(separator, 1)
    if not left or not right:
        raise ValueError(f"{kind} must have non-empty fields, got {value!r}.")
    return left, right


def _load_rows(path: pathlib.Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rollout rows found in {path}.")
    required = {*_KEY_FIELDS, "initial_state_id", "success", "timeout", *_METRICS}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")
    indexed: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = (int(row["task_id"]), int(row["episode"]))
        if key in indexed:
            raise ValueError(f"Duplicate paired key {key} in {path}.")
        indexed[key] = row
    return indexed


def _validate_pairing(runs: dict[str, dict[tuple[int, int], dict[str, str]]]) -> list[tuple[int, int]]:
    labels = list(runs)
    reference_label = labels[0]
    reference = runs[reference_label]
    reference_keys = set(reference)
    for label in labels[1:]:
        candidate_keys = set(runs[label])
        if candidate_keys != reference_keys:
            missing = sorted(reference_keys - candidate_keys)[:10]
            extra = sorted(candidate_keys - reference_keys)[:10]
            raise ValueError(
                f"Run {label!r} is not strictly paired with {reference_label!r}: missing={missing}, extra={extra}."
            )
        mismatched_states = [
            key
            for key in reference_keys
            if int(runs[label][key]["initial_state_id"]) != int(reference[key]["initial_state_id"])
        ]
        if mismatched_states:
            raise ValueError(f"Run {label!r} has mismatched initial_state_id values at keys {mismatched_states[:10]}.")
    return sorted(reference_keys)


def _values(rows: dict[tuple[int, int], dict[str, str]], keys: list[tuple[int, int]], field: str) -> np.ndarray:
    return np.asarray([float(rows[key][field]) for key in keys], dtype=np.float64)


def _exact_mcnemar_pvalue(rescues: int, regressions: int) -> float:
    discordant = rescues + regressions
    if discordant == 0:
        return 1.0
    tail_end = min(rescues, regressions)
    tail_numerator = sum(math.comb(discordant, index) for index in range(tail_end + 1))
    return float(min(1.0, 2.0 * tail_numerator / (1 << discordant)))


def _hierarchical_cluster_bootstrap(
    differences: np.ndarray,
    *,
    task_ids: np.ndarray,
    initial_state_ids: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    task_ids = np.asarray(task_ids, dtype=np.int64)
    initial_state_ids = np.asarray(initial_state_ids, dtype=np.int64)
    valid = np.isfinite(differences)
    differences = differences[valid]
    task_ids = task_ids[valid]
    initial_state_ids = initial_state_ids[valid]
    if differences.size == 0:
        return {
            "mean": None,
            "ci95": [None, None],
            "lcb95": None,
            "ucb95": None,
            "num_tasks": 0,
            "num_clusters": 0,
            "num_rows": 0,
        }
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")

    cluster_values: dict[int, list[float]] = collections.defaultdict(list)
    for task_id in np.unique(task_ids):
        task_member = task_ids == task_id
        for initial_state_id in np.unique(initial_state_ids[task_member]):
            member = task_member & (initial_state_ids == initial_state_id)
            cluster_values[int(task_id)].append(float(np.mean(differences[member])))

    tasks = np.asarray(sorted(cluster_values), dtype=np.int64)
    per_task_clusters = {task_id: np.asarray(cluster_values[int(task_id)], dtype=np.float64) for task_id in tasks}
    point = float(np.mean([np.mean(per_task_clusters[task_id]) for task_id in tasks]))
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((samples,), dtype=np.float64)
    for sample_index in range(samples):
        sampled_tasks = rng.choice(tasks, size=tasks.size, replace=True)
        sampled_task_means = []
        for task_id in sampled_tasks:
            values = per_task_clusters[task_id]
            sampled_task_means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
        bootstrap_means[sample_index] = float(np.mean(sampled_task_means))
    quantiles = np.quantile(bootstrap_means, (0.025, 0.05, 0.95, 0.975))
    return {
        "mean": point,
        "ci95": [float(quantiles[0]), float(quantiles[3])],
        "lcb95": float(quantiles[1]),
        "ucb95": float(quantiles[2]),
        "num_tasks": int(tasks.size),
        "num_clusters": int(sum(values.size for values in per_task_clusters.values())),
        "num_rows": int(differences.size),
    }


def _run_summary(rows: dict[tuple[int, int], dict[str, str]], keys: list[tuple[int, int]]) -> dict[str, Any]:
    success = _values(rows, keys, "success")
    timeout = _values(rows, keys, "timeout")
    task_ids = np.asarray([key[0] for key in keys], dtype=np.int64)
    per_task = {}
    for task_id in np.unique(task_ids):
        member = task_ids == task_id
        per_task[str(int(task_id))] = {
            "episodes": int(np.sum(member)),
            "success_count": int(np.sum(success[member])),
            "success_rate": float(np.mean(success[member])),
        }
    return {
        "episodes": len(keys),
        "success_count": int(np.sum(success)),
        "success_rate": float(np.mean(success)),
        "timeout_count": int(np.sum(timeout)),
        "timeout_rate": float(np.mean(timeout)),
        "means": {field: float(np.mean(_values(rows, keys, field))) for field in _METRICS},
        "per_task_success": per_task,
    }


def _pairwise_audit(
    reference: dict[tuple[int, int], dict[str, str]],
    candidate: dict[tuple[int, int], dict[str, str]],
    keys: list[tuple[int, int]],
    *,
    samples: int,
    seed: int,
    noninferiority_margin: float,
) -> dict[str, Any]:
    reference_success = _values(reference, keys, "success").astype(np.bool_)
    candidate_success = _values(candidate, keys, "success").astype(np.bool_)
    task_ids = np.asarray([key[0] for key in keys], dtype=np.int64)
    initial_state_ids = np.asarray([int(reference[key]["initial_state_id"]) for key in keys], dtype=np.int64)
    rescues = int(np.sum(~reference_success & candidate_success))
    regressions = int(np.sum(reference_success & ~candidate_success))
    success_difference = candidate_success.astype(np.float64) - reference_success.astype(np.float64)
    success_bootstrap = _hierarchical_cluster_bootstrap(
        success_difference,
        task_ids=task_ids,
        initial_state_ids=initial_state_ids,
        samples=samples,
        seed=seed,
    )

    metric_differences = {}
    speedups = {}
    for metric_index, field in enumerate(_METRICS, start=1):
        reference_values = _values(reference, keys, field)
        candidate_values = _values(candidate, keys, field)
        difference = candidate_values - reference_values
        metric_differences[field] = _hierarchical_cluster_bootstrap(
            difference,
            task_ids=task_ids,
            initial_state_ids=initial_state_ids,
            samples=samples,
            seed=seed + metric_index,
        )
        reference_mean = float(np.mean(reference_values))
        candidate_mean = float(np.mean(candidate_values))
        speedups[field] = {
            "reference_mean": reference_mean,
            "candidate_mean": candidate_mean,
            "reference_over_candidate": reference_mean / candidate_mean if candidate_mean > 0 else None,
            "candidate_reduction_fraction": (
                (reference_mean - candidate_mean) / reference_mean if reference_mean > 0 else None
            ),
        }

    per_task = {}
    for task_id in np.unique(task_ids):
        member = task_ids == task_id
        per_task[str(int(task_id))] = {
            "episodes": int(np.sum(member)),
            "success_delta": float(np.mean(success_difference[member])),
            "rescues": int(np.sum(~reference_success[member] & candidate_success[member])),
            "regressions": int(np.sum(reference_success[member] & ~candidate_success[member])),
            "policy_calls_delta": float(
                np.mean(
                    _values(candidate, keys, "policy_calls")[member] - _values(reference, keys, "policy_calls")[member]
                )
            ),
            "policy_ms_delta": float(
                np.mean(
                    _values(candidate, keys, "actual_policy_total_ms")[member]
                    - _values(reference, keys, "actual_policy_total_ms")[member]
                )
            ),
            "rpc_ms_delta": float(
                np.mean(
                    _values(candidate, keys, "policy_rpc_wall_total_ms")[member]
                    - _values(reference, keys, "policy_rpc_wall_total_ms")[member]
                )
            ),
            "elapsed_ms_delta": float(
                np.mean(
                    _values(candidate, keys, "actual_episode_elapsed_total_ms")[member]
                    - _values(reference, keys, "actual_episode_elapsed_total_ms")[member]
                )
            ),
        }

    gates = {
        "success_noninferior_lcb95": bool(success_bootstrap["lcb95"] >= -noninferiority_margin),
        "policy_time_reduced_ucb95": bool(metric_differences["actual_policy_total_ms"]["ucb95"] < 0.0),
        "rpc_time_reduced_ucb95": bool(metric_differences["policy_rpc_wall_total_ms"]["ucb95"] < 0.0),
        "full_elapsed_reduced_ucb95": bool(metric_differences["actual_episode_elapsed_total_ms"]["ucb95"] < 0.0),
    }
    gates["strict_engineering_go"] = bool(all(gates.values()))
    return {
        "paired_episodes": len(keys),
        "both_success": int(np.sum(reference_success & candidate_success)),
        "both_failure": int(np.sum(~reference_success & ~candidate_success)),
        "rescues": rescues,
        "regressions": regressions,
        "net_rescues": rescues - regressions,
        "success_rate_delta": float(np.mean(success_difference)),
        "exact_mcnemar_two_sided_p": _exact_mcnemar_pvalue(rescues, regressions),
        "success_delta_cluster_bootstrap": success_bootstrap,
        "metric_delta_candidate_minus_reference": metric_differences,
        "speedups": speedups,
        "per_task": per_task,
        "gates": gates,
    }


def main(args: argparse.Namespace) -> None:
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if not 0 <= args.success_noninferiority_margin < 1:
        raise ValueError("success_noninferiority_margin must lie in [0, 1).")
    run_paths: dict[str, pathlib.Path] = {}
    for assignment in args.run:
        label, raw_path = _parse_assignment(assignment, separator="=", kind="run")
        if label in run_paths:
            raise ValueError(f"Duplicate run label {label!r}.")
        run_paths[label] = pathlib.Path(raw_path).resolve()
    if len(run_paths) < 2:
        raise ValueError("At least two --run values are required.")
    runs = {label: _load_rows(path) for label, path in run_paths.items()}
    keys = _validate_pairing(runs)

    if args.pair:
        pairs = [_parse_assignment(value, separator=":", kind="pair") for value in args.pair]
    else:
        labels = list(runs)
        pairs = [(labels[0], candidate) for candidate in labels[1:]]
    for reference, candidate in pairs:
        if reference not in runs or candidate not in runs:
            raise ValueError(f"Unknown pair {reference}:{candidate}; known runs are {sorted(runs)}.")
        if reference == candidate:
            raise ValueError("A pair must compare two distinct runs.")

    pairwise = {}
    for pair_index, (reference, candidate) in enumerate(pairs):
        pairwise[f"{candidate}_vs_{reference}"] = _pairwise_audit(
            runs[reference],
            runs[candidate],
            keys,
            samples=args.bootstrap_samples,
            seed=args.seed + pair_index * 100,
            noninferiority_margin=args.success_noninferiority_margin,
        )
    result = {
        "status": "complete",
        "strict_pairing_verified": True,
        "paired_key": list(_KEY_FIELDS),
        "initial_state_id_verified": True,
        "num_paired_episodes": len(keys),
        "run_paths": {label: str(path) for label, path in run_paths.items()},
        "runs": {label: _run_summary(rows, keys) for label, rows in runs.items()},
        "pairwise": pairwise,
        "bootstrap": {
            "method": "equal-weight tasks with task-by-initial-state cluster resampling",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "success_noninferiority_margin": args.success_noninferiority_margin,
    }
    output_path = pathlib.Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
