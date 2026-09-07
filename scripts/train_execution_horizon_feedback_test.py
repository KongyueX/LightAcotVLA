from __future__ import annotations

import importlib
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.execution_horizon.feedback import FEATURE_DIM
from openpi.execution_horizon.feedback import FeedbackSelector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
trainer = importlib.import_module("train_execution_horizon_feedback")


def _record() -> dict[str, np.ndarray]:
    return {
        "temporal_feature": np.ones(256, dtype=np.float32),
        "prefix_feature": np.ones(2048, dtype=np.float32),
        "state": np.ones(32, dtype=np.float32),
        "previous_prefix_feature": np.zeros(2048, dtype=np.float32),
        "previous_state": np.zeros(32, dtype=np.float32),
        "previous_h": np.asarray(25), "elapsed_steps": np.asarray(25),
        "history_valid": np.asarray(1, dtype=np.bool_),
        "continuation_logits": np.asarray([10, -10, 0, 0], dtype=np.float32),
        "trial_success": np.asarray([[0, 1], [1, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32),
        "trial_rpc": np.asarray([[1, 2], [2, 2], [3, 3], [1, 1], [0.5, 0.5]], dtype=np.float32),
        "trial_elapsed": np.ones((5, 2), dtype=np.float32) * 10,
        "trial_calls": np.ones((5, 2), dtype=np.int32) * 10,
        "trial_valid": np.ones((5, 2), dtype=np.bool_), "selected_h": np.asarray(10),
        "task_id": np.asarray(0), "episode_id": np.asarray(100),
        "root_step": np.asarray(35), "source_success": np.asarray(1, dtype=np.bool_),
    }


def _selector() -> FeedbackSelector:
    return FeedbackSelector.initialize(np.zeros((2048, 256)), np.zeros(256))


def test_paired_advantage_preserves_success_and_rpc_units_and_excludes_invalid_repeats() -> None:
    record = _record()
    labels = trainer.paired_advantages(record)
    np.testing.assert_allclose(labels["success_delta"], [-0.5, 0, 0, -0.5, -1])
    np.testing.assert_allclose(labels["rpc_delta_seconds"], [-0.5, 0, 1, -1, -1.5])
    np.testing.assert_allclose(labels["advantage"], [-0.49, 0, -0.02, -0.48, -0.97])
    record["trial_valid"][1, 1] = False
    record["trial_rpc"][1, 1] = np.nan
    labels = trainer.paired_advantages(record)
    np.testing.assert_array_equal(labels["paired_count"], np.ones(5))
    np.testing.assert_allclose(labels["success_delta"], [-1, 0, 0, 0, -1])
    record["trial_valid"][0, 0] = False
    with pytest.raises(ValueError, match="paired with source A"):
        trainer.paired_advantages(record)


def test_numpy_inference_matches_jax_with_nonzero_residual_and_history_gate() -> None:
    selector = _selector()
    rng = np.random.default_rng(13)
    selector.params["output_w"][:] = rng.normal(0, 0.05, (64, 4))
    selector.params["output_b"][:] = [0.2, -0.1, 0.3, -0.4]
    selector.feature_mean[:] = rng.normal(0, 0.1, FEATURE_DIM)
    selector.feature_std[:] = rng.uniform(0.5, 1.5, FEATURE_DIM)
    features = rng.normal(0, 1, (2, FEATURE_DIM)).astype(np.float32)
    anchors = rng.normal(0, 1, (2, 4)).astype(np.float32)
    valid = np.asarray([True, False])
    normalized = (features - selector.feature_mean) / selector.feature_std
    outputs = trainer.forward(
        {name: jnp.asarray(value) for name, value in selector.params.items()},
        jnp.asarray(normalized), jnp.asarray(anchors), jnp.asarray(valid, dtype=jnp.float32),
    )
    for index in range(2):
        numpy_output = selector.forward(features[index], anchors[index], history_valid=bool(valid[index]))
        for name in ("continuation_logits", "probabilities", "log_probabilities"):
            np.testing.assert_allclose(np.asarray(outputs[name])[index], numpy_output[name], atol=2e-6)
    np.testing.assert_array_equal(np.asarray(outputs["continuation_logits"])[1], anchors[1])


def test_paired_objective_updates_residual_and_uses_forward_anchor_kl() -> None:
    selector = _selector()
    params = {name: jnp.asarray(value) for name, value in selector.params.items()}
    labels = trainer.paired_advantages(_record())
    batch = {
        "feature": jnp.ones((1, FEATURE_DIM)), "anchor_logits": jnp.zeros((1, 4)),
        "history_valid": jnp.ones(1),
        **{name: jnp.asarray(value[None]) for name, value in labels.items()},
    }
    before, metrics = trainer.loss(params, batch)
    assert float(metrics["anchor_kl"]) == pytest.approx(0, abs=1e-7)
    gradients = jax.grad(lambda parameters: trainer.loss(parameters, batch)[0])(params)
    assert np.any(np.asarray(gradients["output_w"]) != 0)
    optimizer = optax.adam(1e-4)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    changed = optax.apply_updates(params, updates)
    after, changed_metrics = trainer.loss(changed, batch)
    assert float(after) < float(before)
    assert float(changed_metrics["anchor_kl"]) >= -1e-7
    probabilities = np.asarray(trainer.forward(
        changed, batch["feature"], batch["anchor_logits"], batch["history_valid"]
    )["probabilities"])
    anchor = np.asarray([0.5, 0.25, 0.125, 0.0625, 0.0625])
    expected_kl = np.sum(anchor * (np.log(anchor) - np.log(probabilities[0])))
    assert float(changed_metrics["anchor_kl"]) == pytest.approx(expected_kl, abs=1e-6)
    assert float(after) == pytest.approx(
        -float(changed_metrics["expected_paired_advantage"]) + 0.05 * float(changed_metrics["anchor_kl"])
    )
    np.testing.assert_array_equal(selector.prefix_kernel, np.zeros((2048, 256)))


def test_loader_reads_root_files_and_rejects_duplicate_observations(tmp_path) -> None:
    record = _record()
    directory = tmp_path / "nested"
    directory.mkdir()
    np.savez_compressed(directory / "root.npz", **record)
    data, summary = trainer.load_roots(tmp_path, _selector())
    assert data["feature"].shape == (1, FEATURE_DIM)
    assert summary["episode_groups"] == [[0, 100]]
    assert summary["history_valid_roots"] == 1
    np.savez_compressed(directory / "duplicate.npz", **record)
    with pytest.raises(ValueError, match="Duplicate"):
        trainer.load_roots(tmp_path, _selector())


def test_training_keeps_step_zero_and_fits_normalization_only_on_training(tmp_path, monkeypatch) -> None:
    training = tmp_path / "train"
    validation = tmp_path / "validation"
    training.mkdir()
    validation.mkdir()
    train_record = _record()
    train_record["history_valid"] = np.asarray(0, dtype=np.bool_)
    validation_record = {name: value.copy() for name, value in train_record.items()}
    validation_record["episode_id"] = np.asarray(200)
    validation_record["temporal_feature"][:] = 12
    np.savez_compressed(training / "root.npz", **train_record)
    np.savez_compressed(validation / "root.npz", **validation_record)
    monkeypatch.setattr(FeedbackSelector, "initialize_from_predictor", lambda *args, **kwargs: _selector())
    args = trainer.Args(
        str(training), str(validation), "/unused/A", str(tmp_path / "trained"),
        max_updates=1, log_every=1, patience=1,
    )
    summary = trainer.train(args)
    assert summary["best_step"] == 0
    assert summary["residual_changed"] is False
    loaded = FeedbackSelector.load(summary["checkpoint"])
    np.testing.assert_array_equal(loaded.feature_mean[:256], np.ones(256))
    np.testing.assert_array_equal(loaded.feature_std, np.ones(FEATURE_DIM))
    np.testing.assert_array_equal(loaded.params["output_w"], np.zeros((64, 4)))
