from __future__ import annotations

import csv
import dataclasses
import importlib
import json
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.execution_horizon.ordered_smdp import OrderedSMDPSelector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
trainer = importlib.import_module("train_ordered_horizon_smdp")


def test_production_csv_reader_accepts_real_last_block_cache_and_restores_limit(tmp_path):
    tokens = np.full((29, 256), 0.12345679, dtype=np.float32)
    context = np.full(256, 0.12345679, dtype=np.float32)
    payload = json.dumps({
        "smdp_last_block_input": tokens.tolist(), "smdp_last_block_context": context.tolist(),
    })
    assert len(payload) > 131072
    path = tmp_path / "decisions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "selector_json"])
        writer.writeheader()
        writer.writerow({"mode": "ordered_smdp_last_block", "selector_json": payload})
        writer.writerow({"mode": "ordered_smdp", "selector_json": "{}"})
    previous_limit = csv.field_size_limit()

    rows = trainer._read_csv(path)  # noqa: SLF001

    np.testing.assert_array_equal(json.loads(rows[0]["selector_json"])["smdp_last_block_input"], tokens)
    assert rows[1] == {"mode": "ordered_smdp", "selector_json": "{}"}
    assert csv.field_size_limit() == previous_limit


def test_ordered_ppo_initial_ratio_clipping_and_anchor_kl():
    selector = OrderedSMDPSelector.initialize(feature_dim=3)
    args = trainer.Args("/unused/rollout", "/unused/input.npz", "/unused/output")
    params = {name: jnp.asarray(value) for name, value in selector.params.items()}
    features = jnp.zeros((2, 3))
    anchors = jnp.zeros((2, 4))
    outputs = trainer._forward(params, features, anchors)  # noqa: SLF001
    actions = jnp.asarray([0, 4])
    batch = {
        "feature": features, "anchor_logits": anchors, "action": actions,
        "old_probabilities": jnp.exp(outputs["log_probabilities"]),
        "old_log_prob": outputs["log_probabilities"][jnp.arange(2), actions],
        "advantage": jnp.asarray([1.0, -1.0]),
        "success_return": outputs["success_value"], "cost_return": outputs["cost_value"],
    }
    loss, metrics = trainer._loss(params, batch, args)  # noqa: SLF001
    np.testing.assert_allclose(loss, 0.0, atol=1e-6)
    np.testing.assert_allclose(metrics["ratio_mean"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["anchor_kl"], 0.0, atol=1e-6)
    changed = {**params, "actor_b": jnp.asarray([-2.0, 0.0, 0.0, 0.0])}
    _, metrics = trainer._loss(changed, batch, args)  # noqa: SLF001
    np.testing.assert_allclose(metrics["policy_loss"], -0.1, atol=1e-6)
    np.testing.assert_allclose(metrics["clip_fraction"], 1.0)
    assert float(metrics["anchor_kl"]) > 0
    assert all(np.isfinite(float(value)) for value in metrics.values())
    np.testing.assert_allclose(
        np.exp(np.asarray(outputs["log_probabilities"])[0]),
        selector.forward(np.zeros(3), np.zeros(4))["probabilities"], atol=1e-6,
    )


def test_observed_durations_discount_early_terminal_success_and_do_not_bootstrap_timeout():
    args = trainer.Args("/unused", "/unused", "/unused", gamma=0.9, gae_lambda=1.0)
    decisions = [
        {"environment_step": "10", "execution_horizon": "25", "wall_ms": "100",
         "selector_json": json.dumps({"smdp_success_value": 0.7, "smdp_cost_value": 1.5})},
        {"environment_step": "35", "execution_horizon": "25", "wall_ms": "200",
         "selector_json": json.dumps({"smdp_success_value": 0.8, "smdp_cost_value": 1.0})},
    ]
    durations, _, _, success_return, cost_return = trainer._episode_targets(  # noqa: SLF001
        decisions, {"steps": "38", "success": "1"}, args
    )
    np.testing.assert_array_equal(durations, [25, 3])
    np.testing.assert_allclose(success_return, [0.9**27, 0.9**2], atol=1e-6)
    np.testing.assert_allclose(cost_return, [0.1 + 0.9**25 * 0.2, 0.2], atol=1e-6)
    _, _, _, timeout_return, _ = trainer._episode_targets(  # noqa: SLF001
        decisions, {"steps": "38", "success": "0"}, args
    )
    np.testing.assert_allclose(timeout_return, 0.0, atol=1e-6)


@pytest.fixture
def onpolicy_batch(tmp_path):
    selector = OrderedSMDPSelector.initialize(feature_dim=3)
    checkpoint = tmp_path / "input.npz"
    selector.save(checkpoint)
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    config = {"ordered_smdp_sample": True, "ordered_smdp_params": str(checkpoint)}
    (rollout / "run_config.json").write_text(json.dumps(config))
    (rollout / "summary.json").write_text(json.dumps({"status": "complete"}))
    decisions = []
    for step, cost, action in [(10, 100, 0), (15, 200, 4), (40, 300, 4)]:
        feature = np.asarray([step / 10, 0.5, -0.25], dtype=np.float32)
        anchors = np.zeros(4, dtype=np.float32)
        prediction = selector.forward(feature, anchors)
        info = {
            "smdp_action_index": action, "smdp_old_log_prob": float(prediction["log_probabilities"][action]),
            "smdp_success_value": prediction["success_value"], "smdp_cost_value": prediction["cost_value"],
            "smdp_feature": feature.tolist(), "smdp_anchor_logits": anchors.tolist(),
            "smdp_probabilities": prediction["probabilities"].tolist(),
        }
        decisions.append({
            "mode": "ordered_smdp", "task_id": 0, "episode": 111, "environment_step": step,
            "selected_horizon": selector.candidates[action], "execution_horizon": 25,
            "wall_ms": cost, "selector_json": json.dumps(info),
        })
    outcome = {
        "mode": "ordered_smdp", "task_id": 0, "episode": 111,
        "success": 1, "timeout": 0, "steps": 43, "policy_calls": 3, "policy_rpc_wall_total_ms": 600,
    }
    for name, records in [("decisions.csv", decisions), ("rollout_rows.csv", [outcome])]:
        with (rollout / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    args = trainer.Args(str(rollout), str(checkpoint), str(tmp_path / "output"), rpc_budget_seconds=0.1)
    return args, selector, config


def test_onpolicy_batch_binds_sampling_checkpoint_and_normalizes_combined_advantage_once(onpolicy_batch):
    args, selector, config = onpolicy_batch
    data, statistics = trainer._load_rollouts(args, selector)  # noqa: SLF001
    np.testing.assert_array_equal(data["duration"], [5, 25, 3])
    np.testing.assert_allclose(statistics["cost_multiplier_after"], 0.025)
    combined = data["success_advantage"] - 0.025 * data["cost_advantage"]
    np.testing.assert_allclose(data["advantage"], (combined - combined.mean()) / combined.std(), atol=1e-6)
    config_path = pathlib.Path(args.rollout_dir) / "run_config.json"
    config_path.write_text(json.dumps({**config, "ordered_smdp_sample": False}))
    with pytest.raises(ValueError, match="ordered_smdp_sample=true"):
        trainer._load_rollouts(args, selector)  # noqa: SLF001
    config_path.write_text(json.dumps({**config, "ordered_smdp_params": "/another/selector.npz"}))
    with pytest.raises(ValueError, match="does not match"):
        trainer._load_rollouts(args, selector)  # noqa: SLF001


def test_kl_rejection_saves_the_previous_accepted_actor_and_critic(onpolicy_batch):
    args, selector, _ = onpolicy_batch
    args = dataclasses.replace(args, epochs=2, learning_rate=0.5, target_kl=1e-10)
    trainer.main(args)
    output = pathlib.Path(args.output_dir)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accepted_epochs"] == 0
    assert summary["rejected_epoch"] == 1
    assert summary["rolled_back"] is True
    assert summary["update_accepted"] is False
    assert summary["stop_reason"] == "target_kl_rollback"
    restored = OrderedSMDPSelector.load(output / "selector.npz")
    for name, value in selector.params.items():
        np.testing.assert_array_equal(restored.params[name], value)
    np.testing.assert_array_equal(restored.feature_mean, selector.feature_mean)
    np.testing.assert_array_equal(restored.feature_std, selector.feature_std)
