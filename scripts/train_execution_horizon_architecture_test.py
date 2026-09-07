"""Architecture adapters preserve A while using cached inference inputs only."""

from __future__ import annotations

import copy
from fractions import Fraction
import json

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import train_execution_horizon_architecture as trainer

from openpi.models import execution_horizon_predictor as predictor_lib
from openpi.models import model as model_lib


def _config():
    return predictor_lib.ExecutionHorizonPredictorConfig(
        prefix_feature_dim=16, state_dim=4, action_dim=7, physical_action_dim=7,
        coarse_horizon=15, action_horizon=25, hidden_dim=32, temporal_layers=2,
        temporal_backbone="transformer", num_heads=4, candidate_horizons=(5, 10, 15, 20, 25),
        reference_horizon=10, visual_num_queries=4, ordered_continuation_head=True,
    )


@pytest.fixture
def anchor(tmp_path):
    config = _config()
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    graphdef, params, frozen = nnx.split(module, nnx.Param, ...)
    directory = tmp_path / "anchor"
    trainer.save_sidecar(directory, config, graphdef, params, frozen)
    return directory, module


def _record(module, *, prefix_length=6, episode=300, constant=False):
    rng = np.random.default_rng(12)

    def feature(shape):
        return np.zeros(shape, dtype=np.float32) if constant else rng.normal(size=shape).astype(np.float32)

    inputs = {
        "prefix_feature": feature((16,)), "state": feature((4,)),
        "coarse_actions": feature((15, 7)), "final_actions": feature((25, 7)),
        "previous_actions": feature((25, 7)), "previous_h": np.asarray(10, dtype=np.int32),
        "budget_balance": np.asarray(0.5, dtype=np.float32),
        "episode_progress": np.asarray(0.25, dtype=np.float32), "previous_valid": np.asarray(1, dtype=np.bool_),
        "prefix_tokens": feature((prefix_length, 16)), "prefix_mask": np.ones(prefix_length, dtype=np.bool_),
        "expert_hidden": feature((25, 1024)),
    }
    prediction = module(**{name: jnp.asarray(value[None]) for name, value in inputs.items()})
    record = {
        "architecture_cache_schema": np.asarray(1),
        **{"input_" + name: value for name, value in inputs.items()},
        "continuation_logits": np.asarray(prediction["ordered_continuation_logits"][0]),
        "selected_h": np.asarray(prediction["ordered_selected_h"][0]),
        "trial_success": np.asarray([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.bool_),
        "trial_rpc": np.tile(np.asarray([1.0, 2.0], dtype=np.float32), (5, 1)),
        "trial_elapsed": np.full((5, 2), 10.0, dtype=np.float32),
        "trial_calls": np.full((5, 2), 10, dtype=np.int32), "trial_valid": np.ones((5, 2), dtype=np.bool_),
        "task_id": np.asarray(0), "episode_id": np.asarray(episode), "root_step": np.asarray(35),
        "source_success": np.asarray(1, dtype=np.bool_),
    }
    record.update(trainer.feedback_train.paired_advantages(record))
    return record


def _jax_batch(records, prefix_length):
    return jax.tree.map(jnp.asarray, trainer.batch_records(records, prefix_length))


@pytest.mark.parametrize("variant", ["visual_query", "expert_hidden"])
def test_split_updates_only_new_modules_and_keeps_original_a_at_step_zero(anchor, variant):
    directory, original = anchor
    _, graphdef, params, frozen = trainer.initialize_model(str(directory), variant)
    roots = set(trainer.VARIANT_ROOTS[variant])
    trainable_paths = set(params.flat_state())
    frozen_paths = set(frozen.flat_state())
    assert {path[0] for path in trainable_paths} == roots
    assert not trainable_paths.intersection(frozen_paths)
    original_state = nnx.state(original, nnx.Param).flat_state()
    assert frozen_paths == set(original_state)
    for path, value in original_state.items():
        np.testing.assert_array_equal(frozen.flat_state()[path].value, value.value)
    batch = _jax_batch([_record(original)], 6)
    step_zero = trainer.apply_model(graphdef, params, frozen, batch, variant)
    np.testing.assert_allclose(step_zero["ordered_continuation_logits"], batch["continuation_logits"], atol=1e-6)
    original_h = original(**{
        name: batch[name] for name in trainer.INPUT_NAMES if name != "expert_hidden"
    })["ordered_selected_h"]
    np.testing.assert_array_equal(step_zero["ordered_selected_h"], original_h)
    frozen_before = [np.asarray(value).copy() for value in jax.tree.leaves(frozen)]

    def objective(current):
        prediction = trainer.apply_model(graphdef, current, frozen, batch, variant)
        return trainer.objective(prediction, batch)[0]

    gradients = jax.grad(objective)(params)
    assert {path[0] for path in gradients.flat_state()} == roots
    assert any(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(gradients))
    optimizer = optax.adam(1e-4)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    changed = optax.apply_updates(params, updates)
    assert any(not np.array_equal(left, right) for left, right in zip(
        jax.tree.leaves(params), jax.tree.leaves(changed), strict=True,
    ))
    for before, after in zip(frozen_before, jax.tree.leaves(frozen), strict=True):
        np.testing.assert_array_equal(before, after)


def test_padded_full_prefix_has_no_label_inputs_and_expert_hidden_reaches_model(anchor):
    directory, original = anchor
    long_record = _record(original, prefix_length=600)
    short_record = _record(original, prefix_length=587, episode=301)
    batch = _jax_batch([long_record, short_record], 600)
    assert batch["prefix_tokens"].shape == (2, 600, 16)
    np.testing.assert_array_equal(batch["prefix_mask"][1, 587:], np.zeros(13, dtype=bool))
    np.testing.assert_array_equal(batch["prefix_tokens"][1, 587:], np.zeros((13, 16)))
    _, graphdef, params, frozen = trainer.initialize_model(str(directory), "expert_hidden")
    module = nnx.merge(graphdef, params, frozen)
    module.expert_feature_out.kernel.value = jnp.full_like(module.expert_feature_out.kernel.value, 0.1)
    graphdef, params, frozen = nnx.split(
        module, nnx.All(nnx.Param, lambda path, value: path[0] in trainer.VARIANT_ROOTS["expert_hidden"]), ...
    )
    prediction = trainer.apply_model(graphdef, params, frozen, batch, "expert_hidden")
    padding_changed = trainer.apply_model(graphdef, params, frozen, {
        **batch, "prefix_tokens": batch["prefix_tokens"].at[1, 587:].set(1000.0),
    }, "expert_hidden")
    np.testing.assert_array_equal(
        padding_changed["ordered_continuation_logits"], prediction["ordered_continuation_logits"]
    )
    altered_labels = {
        **batch, "advantage": batch["advantage"] + 100,
        "success_delta": batch["success_delta"] - 100,
        "rpc_delta_seconds": batch["rpc_delta_seconds"] + 1000,
        "continuation_logits": batch["continuation_logits"] + 40,
        "trial_success": jnp.ones((2, 5, 2)), "future_observation": jnp.ones((2, 16)),
    }
    same = trainer.apply_model(graphdef, params, frozen, altered_labels, "expert_hidden")
    for name in prediction:
        np.testing.assert_array_equal(same[name], prediction[name])
    hidden_changed = trainer.apply_model(
        graphdef, params, frozen, {**batch, "expert_hidden": batch["expert_hidden"] + 2.0}, "expert_hidden"
    )
    assert not np.allclose(prediction["ordered_continuation_logits"], hidden_changed["ordered_continuation_logits"])


@pytest.mark.parametrize("variant", ["visual_query", "expert_hidden"])
def test_saved_sidecar_restores_through_model_loader_with_identical_outputs(anchor, tmp_path, variant):
    directory, original = anchor
    config, graphdef, params, frozen = trainer.initialize_model(str(directory), variant)
    module = nnx.merge(graphdef, params, frozen)
    if variant == "visual_query":
        module.visual_query_conditioner.kernel.value = 0.2 * jnp.eye(config.hidden_dim)
    else:
        module.expert_feature_out.kernel.value = jnp.full_like(module.expert_feature_out.kernel.value, 0.02)
    graphdef, params, frozen = nnx.split(
        module, nnx.All(nnx.Param, lambda path, value: path[0] in trainer.VARIANT_ROOTS[variant]), ...
    )
    batch = _jax_batch([_record(original)], 6)
    expected = trainer.apply_model(graphdef, params, frozen, batch, variant)
    destination = tmp_path / "saved"
    trainer.save_sidecar(destination, config, graphdef, params, frozen)
    restored_config = predictor_lib.ExecutionHorizonPredictorConfig(
        **json.loads((destination / "predictor_config.json").read_text())
    )
    restored = predictor_lib.ExecutionHorizonPredictor(restored_config, rngs=nnx.Rngs(999))
    loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(destination / "params", dtype=jnp.float32))
    state = nnx.state(restored)
    state.replace_by_pure_dict(loaded["execution_horizon_predictor"])
    restored = nnx.merge(nnx.graphdef(restored), state)
    actual = restored(**{
        name: batch[name] for name in trainer.INPUT_NAMES if name != "expert_hidden" or variant == "expert_hidden"
    })
    assert restored_config == config
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    saved_paths = traverse_util.flatten_dict(loaded["execution_horizon_predictor"])
    assert any(path[0] in trainer.VARIANT_ROOTS[variant] for path in saved_paths)


def test_greedy_selection_uses_exact_root_mean_success_then_rpc_and_preserves_ties():
    probabilities = np.asarray([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32)
    data = {
        "paired_count": np.asarray([[3] * 5, [5] * 5]),
        "success_delta": np.asarray([[0, 1 / 3, 0, 0, 0], [0, 0, -1 / 5, 0, 0]], dtype=np.float32),
        "rpc_delta_seconds": np.asarray([[0, 2, 0, 0, 0], [0, 0, -1, 0, 0]], dtype=np.float32),
    }
    score, metrics = trainer.greedy_metrics(probabilities, data)
    assert score == (Fraction(1, 15), -0.5)
    assert metrics["success_delta_fraction"] == {"numerator": 1, "denominator": 15}
    assert metrics["paired_net_success_count"] == 0
    assert metrics["selected_h"] == [10, 15]
    faster = copy.deepcopy(data)
    faster["rpc_delta_seconds"][:] -= 1
    assert trainer.greedy_metrics(probabilities, faster)[0] > score
    more_success = copy.deepcopy(data)
    more_success["success_delta"][1, 2] = 0
    more_success["rpc_delta_seconds"][:] = 1000
    assert trainer.greedy_metrics(probabilities, more_success)[0] > score
    assert not trainer.greedy_metrics(probabilities.copy(), data)[0] > score
    with pytest.raises(ValueError, match="Non-finite"):
        trainer.greedy_metrics(np.full((2, 5), np.nan), data)


def test_training_runs_fixed_update_budget_and_keeps_step_zero_when_greedy_ties(anchor, tmp_path):
    directory, original = anchor
    train_dir, validation_dir = tmp_path / "train", tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    train_record = _record(original, constant=True)
    validation_record = _record(original, constant=True, episode=330)
    np.savez_compressed(train_dir / "task00_ep000300.npz", **train_record)
    np.savez_compressed(validation_dir / "task00_ep000330.npz", **validation_record)
    output = tmp_path / "trained"
    trainer.main(trainer.Args(
        str(train_dir), str(validation_dir), str(directory), str(output), "visual_query",
        max_updates=2, log_every=1, batch_size=1,
    ))
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "training_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert summary["updates"] == 2
    assert summary["best_step"] == 0
    assert summary["frozen_A_unchanged"] is True
    assert summary["step0_included"] is True
    assert (output / "params").is_dir()
    assert (output / "last/params").is_dir()
    loaded = model_lib.restore_params(output / "params", dtype=jnp.float32)["execution_horizon_predictor"]
    np.testing.assert_array_equal(loaded["visual_query_conditioner"]["kernel"], np.zeros((32, 32)))
