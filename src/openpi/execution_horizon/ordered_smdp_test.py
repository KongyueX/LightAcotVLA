from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpi.execution_horizon import ordered_smdp


def _policy_outputs(feature_dim: int = 8) -> dict[str, np.ndarray]:
    continuation = np.asarray([0.9, 7.0 / 9.0, 4.0 / 7.0, 0.625])
    return {
        "execution_horizon_temporal_feature": np.linspace(-1.0, 1.0, feature_dim, dtype=np.float32),
        "execution_horizon_ordered_continuation_logits": np.log(continuation / (1.0 - continuation)).astype(np.float32),
        "execution_horizon_candidate_horizons": np.asarray([5, 10, 15, 20, 25]),
    }


def test_initial_selector_preserves_anchor_distribution_and_greedy_choice() -> None:
    selector = ordered_smdp.OrderedSMDPSelector.initialize(feature_dim=8)
    outputs = _policy_outputs()
    result = selector.forward(
        outputs["execution_horizon_temporal_feature"], outputs["execution_horizon_ordered_continuation_logits"]
    )
    np.testing.assert_array_equal(
        result["continuation_logits"], outputs["execution_horizon_ordered_continuation_logits"]
    )
    np.testing.assert_allclose(result["probabilities"], [0.1, 0.2, 0.3, 0.15, 0.25], atol=1e-7)
    assert result["success_value"] == pytest.approx(0.9, abs=1e-7)
    assert result["cost_value"] == pytest.approx(3.0, abs=1e-7)
    horizon, info = selector.decide(outputs)
    assert horizon == info["raw_horizon"] == 15
    assert info["smdp_action_index"] == 2
    assert info["selector_policy"] == "ordered_smdp"
    np.testing.assert_array_equal(selector.feature_mean, np.zeros(8))
    np.testing.assert_array_equal(selector.feature_std, np.ones(8))


def test_sampling_records_the_actual_distribution_log_probability_and_inputs() -> None:
    selector = ordered_smdp.OrderedSMDPSelector.initialize(feature_dim=8)
    outputs = _policy_outputs()
    result = selector.forward(
        outputs["execution_horizon_temporal_feature"], outputs["execution_horizon_ordered_continuation_logits"]
    )
    expected_index = np.random.default_rng(12).choice(5, p=result["probabilities"])
    horizon, info = selector.decide(outputs, sample=True, rng=np.random.default_rng(12))
    assert horizon == selector.candidates[expected_index]
    assert info["smdp_old_log_prob"] == pytest.approx(result["log_probabilities"][expected_index])
    np.testing.assert_allclose(info["smdp_probabilities"], result["probabilities"])
    np.testing.assert_array_equal(info["smdp_feature"], outputs["execution_horizon_temporal_feature"])
    np.testing.assert_array_equal(info["smdp_anchor_logits"], outputs["execution_horizon_ordered_continuation_logits"])


def test_all_horizons_including_h5_remain_selectable() -> None:
    selector = ordered_smdp.OrderedSMDPSelector.initialize(feature_dim=8)
    outputs = _policy_outputs()
    outputs["execution_horizon_ordered_continuation_logits"] = np.zeros(4, dtype=np.float32)
    horizon, info = selector.decide(outputs)
    assert horizon == 5
    assert all(probability > 0 for probability in info["smdp_probabilities"])
    selector.params["actor_b"][:] = 10.0
    horizon, info = selector.decide(outputs)
    assert horizon == 25
    assert info["raw_horizon"] == 5


def test_selector_npz_roundtrip_preserves_parameters_and_metadata(tmp_path) -> None:
    selector = ordered_smdp.OrderedSMDPSelector.initialize(feature_dim=8, seed=19)
    selector.params["actor_w"][0, 1] = 0.25
    selector.metadata["cost_multiplier"] = 0.03
    selector.metadata["source_selector_path"] = "/experiment/initial_selector.npz"
    path = tmp_path / "selector.npz"
    selector.save(path)
    loaded = ordered_smdp.OrderedSMDPSelector.load(path)
    assert loaded.candidates == tuple(loaded.metadata["candidate_horizons"]) == selector.candidates
    assert loaded.metadata == selector.metadata
    for key, value in selector.params.items():
        np.testing.assert_array_equal(loaded.params[key], value)
    assert loaded.decide(_policy_outputs()) == selector.decide(_policy_outputs())


def test_ordered_distribution_is_stable_for_large_finite_logits() -> None:
    log_probabilities = ordered_smdp.ordered_log_probabilities(np.asarray([1000.0, -1000.0, 1000.0, -1000.0]))
    assert np.all(np.isfinite(log_probabilities))
    assert np.exp(log_probabilities).sum() == pytest.approx(1.0)


def test_selector_rejects_mismatched_or_nonfinite_inputs() -> None:
    selector = ordered_smdp.OrderedSMDPSelector.initialize(feature_dim=8)
    with pytest.raises(ValueError, match="feature"):
        selector.forward(np.zeros(7), np.zeros(4))
    with pytest.raises(ValueError, match="anchor_logits"):
        selector.forward(np.zeros(8), np.asarray([0.0, 0.0, np.nan, 0.0]))
    outputs = _policy_outputs()
    outputs["execution_horizon_candidate_horizons"] = np.asarray([5, 10, 15, 20])
    with pytest.raises(ValueError, match="candidate_horizons"):
        selector.decide(outputs)
    with pytest.raises(ValueError, match="positive standard deviations"):
        dataclasses.replace(selector, feature_std=np.zeros(8))


def test_smdp_gae_single_step_durations_matches_standard_recursion() -> None:
    advantages, returns = ordered_smdp.compute_smdp_gae(
        [0.1, 0.0, 1.0], [0.2, 0.3, 0.4], [1, 1, 1], gamma=0.9, gae_lambda=0.8
    )
    np.testing.assert_allclose(advantages, [0.52424, 0.492, 0.6])
    np.testing.assert_allclose(returns, [0.72424, 0.792, 1.0])


def test_smdp_gae_uses_duration_for_both_discount_and_lambda() -> None:
    advantages, returns = ordered_smdp.compute_smdp_gae(
        [0.0, 1.0], [0.2, 0.4], [2, 3], gamma=0.9, gae_lambda=0.8
    )
    np.testing.assert_allclose(advantages, [0.43504, 0.6])
    np.testing.assert_allclose(returns, [0.63504, 1.0])


def test_smdp_lambda_one_returns_match_environment_step_discounted_rewards() -> None:
    rewards, values, durations = [0.1, 0.2, 0.3], [0.7, 0.4, 0.2], [2, 3, 4]
    advantages, returns = ordered_smdp.compute_smdp_gae(rewards, values, durations, gamma=0.9, gae_lambda=1.0)
    expected = [0.1 + 0.9**2 * 0.2 + 0.9**5 * 0.3, 0.2 + 0.9**3 * 0.3, 0.3]
    np.testing.assert_allclose(returns, expected)
    np.testing.assert_allclose(advantages, np.asarray(expected) - values)
    _, terminal_return = ordered_smdp.compute_smdp_gae([0.9**2], [0.4], [3], gamma=0.9, gae_lambda=1.0)
    np.testing.assert_allclose(terminal_return, [0.9**2])


@pytest.mark.parametrize("durations", [[0, 2], [1.5, 2], [1], [np.inf, 2]])
def test_smdp_gae_rejects_invalid_actual_durations(durations) -> None:
    with pytest.raises(ValueError):
        ordered_smdp.compute_smdp_gae([0.0, 1.0], [0.2, 0.3], durations)


def test_smdp_gae_rejects_invalid_discount_or_reward() -> None:
    with pytest.raises(ValueError, match="gamma and gae_lambda"):
        ordered_smdp.compute_smdp_gae([1.0], [0.0], [1], gae_lambda=1.1)
    with pytest.raises(ValueError, match="finite"):
        ordered_smdp.compute_smdp_gae([np.nan], [0.0], [1])
