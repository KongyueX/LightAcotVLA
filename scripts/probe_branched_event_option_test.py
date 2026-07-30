from __future__ import annotations

# ruff: noqa: SLF001
import importlib.util
import pathlib
import sys

import numpy as np

_MODULE_PATH = pathlib.Path(__file__).with_name("probe_branched_event_option.py")
_SPEC = importlib.util.spec_from_file_location("probe_branched_event_option", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def test_split_roots_is_task_stratified_and_episode_disjoint() -> None:
    arrays = {
        "task_id": np.repeat(np.arange(2), 6),
        "episode_id": np.tile(np.repeat(np.arange(3), 2), 2),
    }
    train, validation, test = probe._split_roots(
        arrays,
        validation_fraction=0.1,
        test_fraction=0.1,
        seed=7,
    )
    for task_id in range(2):
        task = arrays["task_id"] == task_id
        episode_sets = [
            set(arrays["episode_id"][partition][arrays["task_id"][partition] == task_id])
            for partition in (train, validation, test)
        ]
        assert all(values for values in episode_sets)
        assert not (episode_sets[0] & episode_sets[1])
        assert not (episode_sets[0] & episode_sets[2])
        assert not (episode_sets[1] & episode_sets[2])
        assert sum(np.sum(task[partition]) for partition in (train, validation, test)) == np.sum(task)


def test_discrete_action_oracle_recovers_exact_cached_index() -> None:
    cached = np.zeros((2, 5, 7), dtype=np.float32)
    cached[0, :, 0] = np.arange(5)
    cached[1, :, 1] = np.arange(5)
    target = np.stack((cached[0, 3], cached[1, 1]))
    indices, errors = probe._discrete_action_oracle(cached, target)
    np.testing.assert_array_equal(indices, np.asarray([3, 1]))
    np.testing.assert_allclose(errors, 0.0)


def test_selector_actions_decouples_progress_from_gripper_mode() -> None:
    cached = np.zeros((2, 5, 7), dtype=np.float32)
    cached[:, :, 0] = np.arange(5)
    cached[:, :, 6] = np.asarray([-1.0, -1.0, 1.0, 1.0, 1.0])
    fixed = cached[:, 1].copy()
    selected = probe._selector_actions(
        cached,
        np.asarray([3.0, 2.5], dtype=np.float32),
        fixed,
    )
    np.testing.assert_allclose(selected[:, 0], np.asarray([3.0, 2.5]))
    np.testing.assert_allclose(selected[:, 6], -1.0)


def test_global_index_calibration_minimizes_train_action_error() -> None:
    cached = np.zeros((3, 5, 7), dtype=np.float32)
    cached[:, :, 0] = np.arange(5)
    target = cached[:, 3].copy()
    selected = probe._calibrate_continuous_index(cached, target, grid_size=5)
    assert selected == 3.0


def test_option_compiler_covers_trajectory_and_maps_action_phase() -> None:
    ear = np.zeros((1, 15, 7), dtype=np.float32)
    ear[0, :, 0] = np.linspace(0.0, 1.0, 15)
    ear[0, 8:, 6] = 1.0
    descriptors, boundaries, valid = probe.compile_options(
        ear,
        max_options=4,
        min_length=2,
        max_length=6,
        gripper_weight=4.0,
    )
    assert descriptors.shape == (1, 4, 35)
    selected = boundaries[0, valid[0]]
    assert selected[0, 0] == 0
    assert selected[-1, 1] == 15
    np.testing.assert_array_equal(selected[1:, 0], selected[:-1, 1])
    option = probe._option_ids(np.asarray([8]), boundaries, valid)
    start, end = boundaries[0, option[0]]
    assert start <= 4 < end


def test_ridge_model_fits_multioutput_affine_mapping() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(100, 5))
    weights = rng.normal(size=(5, 3))
    bias = rng.normal(size=(3,))
    targets = features @ weights + bias
    model = probe._fit_ridge(features, targets, regularization=1e-8)
    prediction = probe._predict_ridge(model, features)
    np.testing.assert_allclose(prediction, targets, atol=1e-5)
