import numpy as np

from openpi.execution_horizon import headroom


def _example_arrays():
    success = np.asarray(
        [
            [True, True, False],
            [False, False, False],
            [True, False, True],
            [False, True, True],
        ],
        dtype=np.bool_,
    )
    timeout = ~success
    calls = np.asarray(
        [
            [5, 3, 2],
            [3, 2, 1],
            [4, 2, 3],
            [2, 4, 3],
        ],
        dtype=np.float64,
    )
    valid = np.ones_like(success)
    return success, timeout, calls, valid


def test_reference_headroom_counts_fixable_and_unfixable_failures():
    success, timeout, calls, valid = _example_arrays()

    report = headroom.reference_headroom(
        np.full(4, 3),
        success,
        timeout,
        calls,
        valid,
    )

    assert report["reference"]["success_count"] == 2
    assert report["reference_failure_count"] == 2
    assert report["fixable_failure_count"] == 1
    assert report["unfixable_failure_count"] == 1
    assert report["success_first_oracle"]["success_count"] == 3
    assert report["success_upper_bound"] == 0.75
    assert report["oracle_h_distribution"] == {"2": 1, "3": 3}


def test_budgeted_oracle_uses_cheapest_successful_upgrades():
    success, timeout, calls, valid = _example_arrays()

    report = headroom.budgeted_success_oracle(
        2.25,
        success,
        timeout,
        calls,
        valid,
    )

    assert report["feasible"]
    assert report["selected_successful_upgrades"] == 2
    assert report["metrics"]["success_count"] == 2
    assert report["metrics"]["average_remaining_calls"] == 2.25


def test_budgeted_oracle_reports_infeasible_budget():
    success, timeout, calls, valid = _example_arrays()

    report = headroom.budgeted_success_oracle(
        1.5,
        success,
        timeout,
        calls,
        valid,
    )

    assert not report["feasible"]
    assert report["minimum_achievable_average_calls"] == 1.75


def test_target_requirements_report_cost_and_unachievable_target():
    success, _, calls, valid = _example_arrays()

    targets = headroom.target_success_requirements(
        (0.5, 0.75, 1.0),
        np.full(4, 3),
        success,
        calls,
        valid,
    )

    assert targets[0]["additional_successes_needed_vs_reference"] == 0
    assert targets[0]["minimum_average_remaining_calls"] == 2.25
    assert targets[1]["additional_successes_needed_vs_reference"] == 1
    assert targets[1]["minimum_average_remaining_calls"] == 2.5
    assert targets[2]["additional_successes_needed_vs_reference"] == 2
    assert not targets[2]["achievable_in_root_labels"]
    assert targets[2]["minimum_average_remaining_calls"] is None


def test_lowest_cost_tie_prefers_larger_horizon():
    success = np.asarray([[True, True, False]], dtype=np.bool_)
    timeout = ~success
    calls = np.asarray([[2.0, 2.0, 1.0]])
    valid = np.ones_like(success)

    report = headroom.reference_headroom(
        np.asarray([3]),
        success,
        timeout,
        calls,
        valid,
    )

    assert report["success_first_oracle"]["h_distribution"] == {"2": 1}
