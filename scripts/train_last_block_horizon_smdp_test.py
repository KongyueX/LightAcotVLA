from __future__ import annotations

import dataclasses
import importlib
import io
import pathlib
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.execution_horizon import last_block_smdp as core
from openpi.models.execution_horizon_predictor import ExecutionHorizonPredictorConfig

sys.path.insert(0, str(pathlib.Path(__file__).parent))
trainer = importlib.import_module("train_last_block_horizon_smdp")


def _selector():
    config = ExecutionHorizonPredictorConfig(
        prefix_feature_dim=8, state_dim=4, action_dim=7, action_horizon=25,
        hidden_dim=8, temporal_layers=2, temporal_backbone="transformer", num_heads=2,
        candidate_horizons=(5, 10, 15, 20, 25), visual_num_queries=4, ordered_continuation_head=True,
    )
    model = core.LastBlockHorizonModel(config, rngs=nnx.Rngs(7))
    graphdef, params = nnx.split(model)
    return core.LastBlockHorizonSelector(graphdef, params, {
        "predictor_config": dataclasses.asdict(config), "candidate_horizons": list(config.candidate_horizons),
        "seed": 7, "cost_multiplier": 0.02,
    })


def test_mc_critic_targets_use_terminal_outcome_and_actual_cumulative_rpc():
    durations = np.asarray([5, 3])
    costs = np.asarray([0.2, 0.3])
    success, cost = trainer._mc_returns(durations, costs, 1, 1.0)  # noqa: SLF001
    np.testing.assert_array_equal(success, [1.0, 1.0])
    np.testing.assert_allclose(cost, [0.5, 0.3])
    failure, _ = trainer._mc_returns(durations, costs, 0, 1.0)  # noqa: SLF001
    np.testing.assert_array_equal(failure, [0.0, 0.0])
    discounted_success, discounted_cost = trainer._mc_returns(durations, costs, 1, 0.9)  # noqa: SLF001
    np.testing.assert_allclose(discounted_success, [0.9**7, 0.9**2])
    np.testing.assert_allclose(discounted_cost, [0.2 + 0.9**5 * 0.3, 0.3])


def _group_data():
    return {
        "episode_index": np.repeat(np.arange(20), 2),
        "task_id": np.repeat(np.arange(2), 20),
        "success_mc": np.repeat((np.arange(20) % 2).astype(np.float32), 2),
        "cost_mc": np.tile(np.asarray([1.0, 0.2], dtype=np.float32), 20),
    }


def test_critic_holdout_is_deterministic_task_stratified_and_episode_disjoint():
    data = _group_data()
    train, heldout = trainer._critic_split(data, 7)  # noqa: SLF001
    train_again, heldout_again = trainer._critic_split(data, 7)  # noqa: SLF001
    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(heldout, heldout_again)
    assert len(train) == 32 and len(heldout) == 8
    assert not set(data["episode_index"][train]) & set(data["episode_index"][heldout])
    for task in [0, 1]:
        assert np.sum(data["task_id"][train] == task) == 16
        assert np.sum(data["task_id"][heldout] == task) == 4


def test_mc_critic_warmup_preserves_actor_and_selects_no_worse_than_step_zero():
    selector = _selector()
    data = _group_data()
    frozen_summary = np.random.default_rng(7).normal(size=(40, 8)).astype(np.float32)
    args = trainer.Args("/unused", "/unused", "/unused", critic_warmup_epochs=2, batch_size=16)
    params, summary = trainer._warm_critic(  # noqa: SLF001
        selector, selector.params, data, frozen_summary, args, io.StringIO()
    )
    assert trainer._actor_changed(selector.params, params) is False  # noqa: SLF001
    assert summary["num_train_episodes"] == 16
    assert summary["num_heldout_episodes"] == 4
    assert summary["selected_heldout"]["mc_objective"] <= summary["initial_heldout"]["mc_objective"]
    assert 0 <= summary["selected_epoch"] <= 2
    assert np.isfinite(summary["constant_heldout"]["mc_success_bce"])
    assert np.isfinite(summary["constant_heldout"]["mc_cost_huber"])


def test_ppo_actor_gradient_is_separate_from_true_mc_critic_supervision():
    selector = _selector()
    tokens = jax.random.normal(jax.random.key(1), (2, 29, 8))
    context = jax.random.normal(jax.random.key(2), (2, 8))
    initial = core.apply_last_block(selector.graphdef, selector.params, tokens, context)
    actions = jnp.asarray([0, 4])
    batch = {
        "tokens": tokens, "context": context, "anchor_logits": initial["continuation_logits"],
        "old_probabilities": initial["probabilities"],
        "old_log_prob": initial["log_probabilities"][jnp.arange(2), actions], "action": actions,
        "advantage": jnp.zeros(2), "success_mc": jnp.asarray([1.0, 0.0]), "cost_mc": jnp.asarray([0.2, 0.5]),
    }
    args = trainer.Args("/unused", "/unused", "/unused", anchor_kl_weight=0.0)

    def loss(params):
        return trainer._ppo_loss(selector.graphdef, params, batch, args)  # noqa: SLF001

    (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(selector.params)
    actor_gradients = trainer._partition(gradients, core.ACTOR_ROOTS)  # noqa: SLF001
    assert all(np.all(np.asarray(leaf) == 0) for leaf in jax.tree.leaves(actor_gradients))
    critic_gradients = trainer._partition(gradients, core.CRITIC_ROOTS)  # noqa: SLF001
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(critic_gradients))
    np.testing.assert_allclose(metrics["ratio_mean"], 1.0, atol=1e-6)
    batch["advantage"] = jnp.asarray([1.0, -1.0])
    (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(selector.params)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradients["last_block"]))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradients))
    assert all(np.isfinite(float(value)) for value in metrics.values())
