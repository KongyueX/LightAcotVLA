from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import execution_horizon_predictor as predictor_lib


def _transformer_config() -> predictor_lib.ExecutionHorizonPredictorConfig:
    return predictor_lib.ExecutionHorizonPredictorConfig(
        prefix_feature_dim=16,
        state_dim=4,
        action_dim=7,
        physical_action_dim=7,
        coarse_horizon=15,
        action_horizon=25,
        hidden_dim=32,
        temporal_layers=2,
        temporal_backbone="transformer",
        num_heads=4,
        candidate_horizons=(5, 10, 15, 20, 25),
        reference_horizon=10,
        coarse_stride=2,
        final_stride=1,
        visual_num_queries=4,
    )


def _inputs(batch_size: int = 2) -> dict[str, jax.Array]:
    return {
        "prefix_feature": jnp.zeros((batch_size, 16), dtype=jnp.float32),
        "prefix_tokens": jnp.zeros((batch_size, 12, 16), dtype=jnp.float32),
        "prefix_mask": jnp.asarray([[True] * 10 + [False] * 2] * batch_size),
        "state": jnp.zeros((batch_size, 4), dtype=jnp.float32),
        "coarse_actions": jnp.zeros((batch_size, 15, 7), dtype=jnp.float32),
        "final_actions": jnp.zeros((batch_size, 25, 7), dtype=jnp.float32),
        "previous_actions": jnp.zeros((batch_size, 25, 7), dtype=jnp.float32),
        "previous_h": jnp.full((batch_size,), 10, dtype=jnp.int32),
        "budget_balance": jnp.full((batch_size,), 0.5, dtype=jnp.float32),
        "episode_progress": jnp.full((batch_size,), 0.25, dtype=jnp.float32),
        "previous_valid": jnp.ones((batch_size,), dtype=jnp.bool_),
    }


def test_predictor_config_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="temporal_backbone"):
        predictor_lib.ExecutionHorizonPredictorConfig(temporal_backbone="unknown")


def test_predictor_config_rejects_paired_heads_on_legacy_backbone():
    with pytest.raises(ValueError, match="paired_advantage_heads"):
        predictor_lib.ExecutionHorizonPredictorConfig(paired_advantage_heads=True)


def test_transformer_outputs_hierarchical_shapes_and_monotonic_survival():
    config = _transformer_config()
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    outputs = module(**_inputs())

    assert outputs["hazard"].shape == (2, 25)
    assert outputs["survival"].shape == (2, 25)
    assert outputs["success_advantage"].shape == (2, 3)
    assert outputs["elapsed_advantage"].shape == (2, 3)
    assert outputs["calls_advantage"].shape == (2, 3)
    assert outputs["raw_h_logits"].shape == (2, 5)
    assert outputs["candidate_horizons"].shape == (2, 5)
    np.testing.assert_array_equal(np.asarray(outputs["reference_horizon"]), 10)
    assert np.all(np.diff(np.asarray(outputs["survival"]), axis=-1) <= 1e-6)


def test_transformer_paired_heads_define_success_treatment_effect():
    config = dataclasses.replace(_transformer_config(), paired_advantage_heads=True)
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    outputs = module(**_inputs())

    assert outputs["danger_logits"].shape == (2, 3)
    assert outputs["rescue_logits"].shape == (2, 3)
    assert outputs["faster_long_logits"].shape == (2, 3)
    np.testing.assert_allclose(
        np.asarray(outputs["success_advantage"]),
        np.asarray(outputs["rescue_probability"] - outputs["danger_probability"]),
    )


def test_transformer_coarse_alignment_uses_physical_stride():
    config = _transformer_config()
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    coarse = jnp.arange(15, dtype=jnp.float32)[None, :, None]
    aligned = np.asarray(module._align_coarse(coarse))[0, :, 0]  # noqa: SLF001

    np.testing.assert_allclose(aligned, np.arange(25, dtype=np.float32) / 2.0)


def test_transformer_masks_previous_disagreement_without_overlap():
    config = _transformer_config()
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    final_actions = jnp.ones((1, 25, 7), dtype=jnp.float32)
    aligned_previous, overlap_valid, consistency = module._previous_overlap(  # noqa: SLF001
        final_actions,
        jnp.zeros_like(final_actions),
        jnp.asarray([10]),
        jnp.asarray([False]),
    )
    features = module._transformer_action_features(  # noqa: SLF001
        final_actions,
        final_actions,
        aligned_previous,
        overlap_valid,
        consistency,
    )

    action_dim = config.action_dim
    np.testing.assert_array_equal(np.asarray(features[..., 4 * action_dim : 5 * action_dim]), 0.0)
    np.testing.assert_array_equal(np.asarray(features[..., 8 * action_dim + 1]), 0.0)


def test_transformer_count_survival_and_advantage_loss_is_finite():
    config = dataclasses.replace(_transformer_config(), paired_advantage_heads=True)
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    predictions = module(**_inputs())
    success_count = jnp.asarray([[3, 3, 2, 1, 1], [2, 2, 2, 2, 2]], dtype=jnp.float32)
    trial_count = jnp.full((2, 5), 3, dtype=jnp.float32)
    labels = {
        "branch_success": success_count > 0,
        "branch_timeout": success_count == 0,
        "success_count": success_count,
        "timeout_count": trial_count - success_count,
        "trial_count": trial_count,
        "remaining_calls": jnp.ones((2, 5), dtype=jnp.float32),
        "remaining_steps": jnp.ones((2, 5), dtype=jnp.float32),
        "remaining_calls_mean": jnp.ones((2, 5), dtype=jnp.float32),
        "remaining_calls_variance": jnp.full((2, 5), 0.25, dtype=jnp.float32),
        "remaining_steps_mean": jnp.ones((2, 5), dtype=jnp.float32),
        "elapsed_mean": jnp.asarray([[10, 9, 8, 7, 6], [10, 9, 8, 7, 6]], dtype=jnp.float32),
        "elapsed_variance": jnp.full((2, 5), 0.25, dtype=jnp.float32),
        "branch_valid": jnp.ones((2, 5), dtype=jnp.bool_),
        "final_risk": jnp.zeros((2, 25), dtype=jnp.float32),
        "action_cot_risk": jnp.zeros((2, 25), dtype=jnp.float32),
        "fused_risk": jnp.zeros((2, 25), dtype=jnp.float32),
        "event_mask": jnp.zeros((2, 25), dtype=jnp.bool_),
        "risk_valid": jnp.ones((2, 25), dtype=jnp.bool_),
        "hazard_event_count": jnp.zeros((2, 25), dtype=jnp.float32),
        "hazard_at_risk_count": jnp.full((2, 25), 3, dtype=jnp.float32),
        "raw_h": jnp.asarray([10, 15], dtype=jnp.int32),
        "dangerous_long_count": jnp.asarray([[1, 2, 2], [0, 0, 0]], dtype=jnp.float32),
        "paired_trial_count": jnp.full((2, 3), 3, dtype=jnp.float32),
        "trial_success": jnp.asarray(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],
                [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],
            ],
            dtype=jnp.bool_,
        ),
        "trial_elapsed": jnp.asarray(
            [
                [[10, 9, 11], [9, 8, 10], [8, 7, 9], [7, 6, 8], [6, 5, 7]],
                [[10, 9, 11], [9, 8, 10], [8, 7, 9], [7, 6, 8], [6, 5, 7]],
            ],
            dtype=jnp.float32,
        ),
        "trial_valid": jnp.ones((2, 5, 3), dtype=jnp.bool_),
    }
    weights = predictor_lib.ExecutionHorizonLossWeights(
        success=1.0,
        timeout=0.5,
        remaining_calls=0.0,
        remaining_steps=0.0,
        final_risk=0.0,
        action_cot_risk=0.0,
        fused_risk=0.0,
        event=0.0,
        raw_h_classification=0.1,
        raw_h_ordinal=0.1,
        survival=1.0,
        success_advantage=1.0,
        elapsed_advantage=0.1,
        calls_advantage=0.1,
        false_long=2.0,
        danger_rescue=1.0,
        paired_elapsed=1.0,
        faster_long=1.0,
    )
    loss, metrics = predictor_lib.execution_horizon_loss(
        predictions,
        labels,
        weights=weights,
        candidate_horizons=config.candidate_horizons,
        reference_horizon=config.reference_horizon,
    )

    assert np.isfinite(np.asarray(loss))
    assert np.isfinite(np.asarray(metrics["success_advantage_nll"]))
    assert np.isfinite(np.asarray(metrics["elapsed_advantage_nll"]))
    assert np.isfinite(np.asarray(metrics["calls_advantage_nll"]))
    assert np.isfinite(np.asarray(metrics["danger_rescue_binomial"]))
    assert np.isfinite(np.asarray(metrics["paired_elapsed_huber"]))
    assert np.isfinite(np.asarray(metrics["faster_long_binomial"]))


def test_legacy_local_mlp_default_shapes_are_unchanged():
    config = predictor_lib.ExecutionHorizonPredictorConfig(
        prefix_feature_dim=16,
        state_dim=4,
        action_dim=7,
        hidden_dim=32,
    )
    module = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(7))
    inputs = _inputs()
    inputs.pop("prefix_tokens")
    inputs.pop("prefix_mask")
    inputs["final_actions"] = inputs["final_actions"][:, :10]
    inputs["previous_actions"] = inputs["previous_actions"][:, :10]
    outputs = module(**inputs)

    assert outputs["raw_h_logits"].shape == (2, 10)
    assert outputs["success_logits"].shape == (2, 10)
    assert "hazard_logits" not in outputs
