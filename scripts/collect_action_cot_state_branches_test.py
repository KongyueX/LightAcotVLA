from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = pathlib.Path(__file__).with_name("collect_action_cot_state_branches.py")
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("collect_action_cot_state_branches", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
collector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector)


def test_canonical_seed_depends_on_physical_root() -> None:
    seed = collector.canonical_policy_seed(7, 2, 3, 41)
    assert seed == 2_030_048
    assert seed == collector.canonical_policy_seed(7, 2, 3, 41)
    assert seed != collector.canonical_policy_seed(7, 2, 3, 42)
    with pytest.raises(ValueError, match="non-negative"):
        collector.canonical_policy_seed(7, 2, -1, 41)


def test_branch_actions_preserve_protocol_and_shape() -> None:
    primary = np.linspace(-0.8, 0.8, 10 * 7, dtype=np.float32).reshape(10, 7)
    branches, strengths = collector.make_branch_actions(primary)
    assert [len(values) for values in branches] == [4, 4, 4, 4, 4, 4]
    assert strengths.tolist() == [1.0, 0.0, 0.5, 1.25, 0.25, 1.0]
    np.testing.assert_allclose(branches[2][:, :6], primary[:4, :6] * 0.5)
    assert np.all(np.abs(branches[3][:, :6]) <= 1.0)
    np.testing.assert_allclose(branches[1][1, :6], 0.0)
    assert branches[4][1, 2] == pytest.approx(np.clip(primary[1, 2] + 0.25, -1.0, 1.0))
    np.testing.assert_allclose(branches[5][1:, 6], primary[:3, 6])
