from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np

_SCRIPT = pathlib.Path(__file__).with_name("audit_hierarchical_execution_horizon.py")
sys.path.insert(0, str(_SCRIPT.parent))
auditor = importlib.import_module("audit_hierarchical_execution_horizon")


def test_cluster_bootstrap_weights_initial_states_not_root_count() -> None:
    # The first initial state contributes three roots; the second contributes
    # one. Equal cluster weighting reports (1 + 3) / 2 = 2, not 1.5.
    interval = auditor._cluster_bootstrap_interval(  # noqa: SLF001
        np.asarray([1.0, 1.0, 1.0, 3.0]),
        np.asarray([10, 10, 10, 20], dtype=np.uint64),
        seed=7,
        samples=2_000,
    )

    assert interval["mean"] == 2.0
    assert interval["num_roots"] == 4
    assert interval["num_clusters"] == 2
