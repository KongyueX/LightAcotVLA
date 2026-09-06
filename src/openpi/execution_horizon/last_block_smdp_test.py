from __future__ import annotations

import dataclasses
import json

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.execution_horizon import last_block_smdp
from openpi.models import execution_horizon_predictor as predictor_lib


@pytest.fixture
def anchor(tmp_path):
    config = predictor_lib.ExecutionHorizonPredictorConfig(
        prefix_feature_dim=16, state_dim=4, action_dim=7, physical_action_dim=7,
        coarse_horizon=15, action_horizon=25, hidden_dim=32, temporal_layers=2,
        temporal_backbone="transformer", num_heads=4, visual_num_queries=4,
        candidate_horizons=(5, 10, 15, 20, 25), reference_horizon=10,
        ordered_continuation_head=True, ordered_readout="global",
    )
    predictor = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(11))
    directory = tmp_path / "anchor"
    directory.mkdir()
    (directory / "predictor_config.json").write_text(json.dumps(dataclasses.asdict(config)))
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            directory / "params", {"params": {"execution_horizon_predictor": nnx.state(predictor).to_pure_dict()}}
        )
    tokens = jax.random.normal(jax.random.key(2), (1, 29, 32))
    context = jax.random.normal(jax.random.key(3), (1, 32))
    return directory, predictor, tokens, context


def test_last_block_initialization_reproduces_anchor_actor(anchor):
    directory, predictor, tokens, context = anchor
    selector = last_block_smdp.LastBlockHorizonSelector.initialize_from_predictor(directory)
    expected_tokens = predictor.temporal_layers[-1](tokens)
    expected_summary = nnx.swish(predictor.summary_proj(jnp.concatenate(
        [jnp.mean(expected_tokens[:, -25:], axis=1), context], axis=-1
    )))
    expected_logits = predictor.raw_h_ordinal_head(expected_summary)
    expected_log_prob, expected_prob = predictor_lib.ordered_continuation_distribution(expected_logits)
    output = selector.forward(tokens, context)
    np.testing.assert_allclose(output["summary"], expected_summary, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output["continuation_logits"], expected_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output["log_probabilities"], expected_log_prob, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output["probabilities"], expected_prob, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(output["success_value"], [0.9], atol=1e-6)
    np.testing.assert_allclose(output["cost_value"], [3.0], atol=1e-6)
    assert set(selector.params) == set(last_block_smdp.ACTOR_ROOTS + last_block_smdp.CRITIC_ROOTS)
    assert selector.metadata["cost_multiplier"] == 0.02


def test_critic_gradient_does_not_update_actor_and_actor_loss_reaches_last_block(anchor):
    directory, _, tokens, context = anchor
    selector = last_block_smdp.LastBlockHorizonSelector.initialize_from_predictor(directory)
    state = selector.params.to_pure_dict()
    state["critic_out"]["kernel"] = jnp.full((64, 2), 0.1)
    selector.params.replace_by_pure_dict(state)

    def critic_loss(params):
        return last_block_smdp.apply_last_block(selector.graphdef, params, tokens, context)["success_logits"].sum()

    critic_gradients = jax.grad(critic_loss)(selector.params)
    for root in last_block_smdp.ACTOR_ROOTS:
        assert all(np.all(np.asarray(value) == 0) for value in jax.tree.leaves(critic_gradients[root]))
    assert any(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(critic_gradients["critic_in"]))

    def actor_loss(params):
        return -last_block_smdp.apply_last_block(selector.graphdef, params, tokens, context)["log_probabilities"][0, 0]

    actor_gradients = jax.grad(actor_loss)(selector.params)
    for root in last_block_smdp.ACTOR_ROOTS:
        assert any(np.any(np.asarray(value) != 0) for value in jax.tree.leaves(actor_gradients[root]))
    for root in last_block_smdp.CRITIC_ROOTS:
        assert all(np.all(np.asarray(value) == 0) for value in jax.tree.leaves(actor_gradients[root]))
    full = last_block_smdp.apply_last_block(selector.graphdef, selector.params, tokens, context)
    values = last_block_smdp.apply_critic(selector.graphdef, selector.params, full["summary"])
    for key in ("success_logits", "success_value", "cost_value"):
        np.testing.assert_allclose(values[key], full[key], rtol=1e-6)


def test_last_block_checkpoint_roundtrip_sampling_and_current_params(anchor, tmp_path):
    directory, _, tokens, context = anchor
    selector = last_block_smdp.LastBlockHorizonSelector.initialize_from_predictor(directory)
    anchor_logits = selector.forward(tokens, context)["continuation_logits"]
    state = selector.params.to_pure_dict()
    state["raw_h_ordinal_head"]["bias"] = state["raw_h_ordinal_head"]["bias"] + 0.2
    selector.params.replace_by_pure_dict(state)
    selector.metadata.update(actor_changed=True, cost_multiplier=0.03)
    checkpoint = tmp_path / "checkpoint"
    selector.save(checkpoint)
    loaded = last_block_smdp.LastBlockHorizonSelector.load(checkpoint)
    assert loaded.metadata == selector.metadata
    original = selector.forward(tokens, context)
    restored = loaded.forward(tokens, context)
    for key, value in original.items():
        np.testing.assert_array_equal(restored[key], value)
    outputs = {
        "execution_horizon_last_block_input": tokens,
        "execution_horizon_last_block_context": context,
        "execution_horizon_ordered_continuation_logits": anchor_logits,
        "execution_horizon_candidate_horizons": np.asarray(selector.candidates),
    }
    horizon, info = loaded.decide(outputs, sample=True, rng=np.random.default_rng(7))
    assert horizon == loaded.candidates[info["smdp_action_index"]]
    assert info["smdp_old_log_prob"] == pytest.approx(np.log(info["smdp_probabilities"][info["smdp_action_index"]]))
    assert np.asarray(info["smdp_last_block_input"]).shape == (29, 32)
    assert np.asarray(info["smdp_last_block_context"]).shape == (32,)
    assert len(info["smdp_feature"]) == 32
    anchor_probability = predictor_lib.ordered_continuation_distribution(anchor_logits)[1][0]
    assert info["raw_horizon"] == selector.candidates[int(np.argmax(anchor_probability))]
    with pytest.raises(FileExistsError):
        loaded.save(checkpoint)


def test_last_block_rejects_wrong_cache_shapes(anchor):
    directory, _, tokens, context = anchor
    selector = last_block_smdp.LastBlockHorizonSelector.initialize_from_predictor(directory)
    with pytest.raises(ValueError, match="dimensions"):
        selector.forward(tokens[:, :-1], context)
    with pytest.raises(ValueError, match="dimensions"):
        selector.forward(tokens, context[:, :-1])
