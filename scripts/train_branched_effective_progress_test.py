# ruff: noqa: SLF001

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np

_SCRIPT = pathlib.Path(__file__).with_name("train_branched_effective_progress.py")
sys.path.insert(0, str(_SCRIPT.parent))
probe = importlib.import_module("train_branched_effective_progress")


def test_split_is_episode_disjoint_and_uses_each_three_episode_task() -> None:
    arrays = {
        "task_id": np.repeat(np.arange(2), 6),
        "episode_id": np.tile(np.repeat(np.arange(3), 2), 2),
    }
    partitions = probe._split_roots(
        arrays,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    selected_pairs: list[set[tuple[int, int]]] = []
    for roots in partitions:
        pairs = set(
            zip(
                arrays["task_id"][roots].tolist(),
                arrays["episode_id"][roots].tolist(),
                strict=True,
            )
        )
        selected_pairs.append(pairs)
        assert len(pairs) == 2
        assert {task for task, _ in pairs} == {0, 1}
    assert selected_pairs[0].isdisjoint(selected_pairs[1])
    assert selected_pairs[0].isdisjoint(selected_pairs[2])
    assert selected_pairs[1].isdisjoint(selected_pairs[2])


def test_dense_phase_oracle_recovers_known_continuous_shift() -> None:
    horizon = 8
    token = np.arange(horizon, dtype=np.float32)
    cached = np.zeros((3, horizon, 7), dtype=np.float32)
    cached[..., :6] = token[None, :, None] * np.arange(1, 7, dtype=np.float32)[None, None]
    expected_phase = np.asarray([0.5, 1.375, 2.75], dtype=np.float32)
    fresh = probe._transport_ear_numpy(cached, expected_phase)
    predicted_phase, error = probe._dense_phase_oracle(
        cached,
        fresh,
        max_phase=4.0,
        grid_size=33,
        chunk_size=2,
    )
    np.testing.assert_allclose(predicted_phase, expected_phase, atol=1e-6)
    np.testing.assert_allclose(error, 0.0, atol=1e-7)


def test_flatten_valid_branches_and_model_budget() -> None:
    arrays = {
        "branch_valid": np.asarray(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=np.bool_,
        )
    }
    indices = probe._flatten_valid_branches(arrays, np.asarray([0, 1]))
    assert indices.roots.tolist() == [0, 0, 1]
    assert indices.branches.tolist() == [0, 2, 1]
    assert (
        probe.estimate_parameter_count(
            image_views=2,
            image_channels=3,
            state_dim=32,
            action_dim=32,
            env_action_dim=7,
            max_executed_steps=4,
        )
        < 1_000_000
    )
