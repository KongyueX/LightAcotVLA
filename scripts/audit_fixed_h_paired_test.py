from __future__ import annotations

import csv
import importlib
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).with_name("audit_fixed_h_paired.py")
sys.path.insert(0, str(_SCRIPT.parent))
audit = importlib.import_module("audit_fixed_h_paired")


_FIELDS = (
    "task_id",
    "episode",
    "initial_state_id",
    "success",
    "timeout",
    "policy_calls",
    "actual_policy_total_ms",
    "policy_rpc_wall_total_ms",
    "actual_episode_elapsed_total_ms",
)


def _write(path: pathlib.Path, rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(_FIELDS)
        writer.writerows(rows)


def test_pairwise_audit_reports_success_flips_and_faster_metrics(tmp_path: pathlib.Path) -> None:
    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    reference_rows = [
        (0, 0, 0, 1, 0, 10, 100, 120, 1000),
        (0, 1, 1, 0, 1, 12, 120, 140, 1100),
        (1, 0, 0, 1, 0, 11, 110, 130, 1050),
        (1, 1, 1, 0, 1, 13, 130, 150, 1150),
    ]
    candidate_rows = [
        (0, 0, 0, 0, 1, 8, 80, 100, 900),
        (0, 1, 1, 1, 0, 9, 90, 110, 950),
        (1, 0, 0, 1, 0, 8, 85, 105, 925),
        (1, 1, 1, 0, 1, 10, 95, 115, 975),
    ]
    _write(reference_path, reference_rows)
    _write(candidate_path, candidate_rows)
    reference = audit._load_rows(reference_path)  # noqa: SLF001
    candidate = audit._load_rows(candidate_path)  # noqa: SLF001
    keys = audit._validate_pairing({"reference": reference, "candidate": candidate})  # noqa: SLF001
    result = audit._pairwise_audit(  # noqa: SLF001
        reference,
        candidate,
        keys,
        samples=1_000,
        seed=7,
        noninferiority_margin=0.01,
    )

    assert result["rescues"] == 1
    assert result["regressions"] == 1
    assert result["exact_mcnemar_two_sided_p"] == 1.0
    assert result["success_rate_delta"] == 0.0
    assert result["speedups"]["actual_policy_total_ms"]["reference_over_candidate"] > 1.0
    assert result["metric_delta_candidate_minus_reference"]["actual_policy_total_ms"]["ucb95"] < 0
    assert result["metric_delta_candidate_minus_reference"]["policy_rpc_wall_total_ms"]["ucb95"] < 0


def test_pairing_rejects_mismatched_initial_state(tmp_path: pathlib.Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write(first_path, [(0, 0, 0, 1, 0, 10, 100, 120, 1000)])
    _write(second_path, [(0, 0, 1, 1, 0, 10, 100, 120, 1000)])

    with pytest.raises(ValueError, match="mismatched initial_state_id"):
        audit._validate_pairing(  # noqa: SLF001
            {
                "first": audit._load_rows(first_path),  # noqa: SLF001
                "second": audit._load_rows(second_path),  # noqa: SLF001
            }
        )


def test_hierarchical_bootstrap_is_deterministic() -> None:
    differences = np.asarray([-1.0, -2.0, -3.0, -4.0])
    task_ids = np.asarray([0, 0, 1, 1])
    initial_state_ids = np.asarray([0, 1, 0, 1])
    first = audit._hierarchical_cluster_bootstrap(  # noqa: SLF001
        differences,
        task_ids=task_ids,
        initial_state_ids=initial_state_ids,
        samples=1_000,
        seed=42,
    )
    second = audit._hierarchical_cluster_bootstrap(  # noqa: SLF001
        differences,
        task_ids=task_ids,
        initial_state_ids=initial_state_ids,
        samples=1_000,
        seed=42,
    )

    assert first == second
    assert first["mean"] == -2.5
    assert first["ucb95"] < 0
