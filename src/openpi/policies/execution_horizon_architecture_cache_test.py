# ruff: noqa: SLF001
from __future__ import annotations

import collections

import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.policies import policy as policy_lib


class _ProfileModel:
    action_dim = 32
    action_horizon = 25
    coarse_action_horizon = 15
    execution_horizon_predictor_enabled = True

    def __init__(self, expert_dim=0):
        self.execution_horizon_expert_feature_dim = expert_dim
        self.calls = []
        self.predictor_inputs = []

    def sample_actions(self, rng, observation, **kwargs):
        raise AssertionError("The architecture test should use profiled stage entrypoints.")

    def sample_actions_profile_prefix(self, rng, observation):
        self.calls.append("prefix")
        tokens = jnp.arange(6 * 2048, dtype=jnp.float32).reshape((1, 6, 2048)) / 1000.0
        mask = jnp.ones((1, 6), dtype=jnp.bool_)
        return {
            "observation": observation, "prefix_out": tokens, "prefix_mask": mask,
            "execution_horizon_prefix_tokens": tokens, "execution_horizon_prefix_mask": mask,
            "execution_horizon_prefix_feature": tokens.mean(axis=1),
        }

    def sample_actions_profile_implicit(self, prefix):
        self.calls.append("implicit")
        return {"implicit_action_reason": jnp.zeros((1, 15, 4))}

    def sample_actions_profile_coarse(self, prefix, **kwargs):
        self.calls.append("coarse")
        return {"explicit_action_reason": jnp.full((1, 15, 32), 0.5),
                "action_cot_denoising_steps": jnp.asarray([10], dtype=jnp.int32)}

    def sample_actions_profile_expert(self, prefix, explicit, implicit, *, return_expert_hidden=False, **kwargs):
        self.calls.append(("expert", return_expert_hidden))
        result = {"actions": jnp.arange(25 * 32, dtype=jnp.float32).reshape((1, 25, 32)) / 1000.0}
        if return_expert_hidden:
            result["execution_horizon_expert_hidden"] = jnp.full((1, 25, 4), 0.1)
        return result

    def predict_execution_horizon(self, **inputs):
        self.calls.append("predictor")
        self.predictor_inputs.append(inputs)
        if self.execution_horizon_expert_feature_dim:
            assert inputs["expert_hidden"].shape == (1, 25, self.execution_horizon_expert_feature_dim)
        logits = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
        return {"ordered_continuation_logits": logits, "ordered_selected_h": jnp.asarray([25]),
                "candidate_horizons": jnp.asarray([[5, 10, 15, 20, 25]])}


def _make_policy(monkeypatch, *, expert_dim=0):
    monkeypatch.setattr(policy_lib.nnx_utils, "module_jit", lambda function, *args, **kwargs: function)
    model = _ProfileModel(expert_dim=expert_dim)

    def output_transform(outputs):
        outputs["actions"] = outputs["actions"][..., :7] + 100.0
        return outputs

    instance = policy_lib.Policy(
        model, output_transforms=[output_transform],
        norm_stats={"actions": transforms.NormStats(mean=np.ones(7), std=np.full(7, 2.0))},
    )
    return instance, model


def _request():
    return {
        "image": {}, "image_mask": {}, "state": np.arange(32, dtype=np.float32),
        "policy_seed": 7, "profile_policy_timing": True, "run_execution_horizon_predictor": True,
        "action_cot_denoising_steps": 10, "action_cot_final_denoising_steps": 10,
        "execution_horizon_previous_actions": np.full((25, 7), 3.0, dtype=np.float32),
        "execution_horizon_previous_h": 15, "execution_horizon_previous_valid": True,
        "execution_horizon_budget_balance": 0.35, "execution_horizon_episode_progress": 0.25,
    }


def test_architecture_cache_preserves_a_outputs_and_exports_exact_predictor_inputs(monkeypatch):
    instance, model = _make_policy(monkeypatch)
    normal = instance.infer(_request())
    cached = instance.infer({**_request(), "execution_horizon_export_architecture_cache": True})
    np.testing.assert_array_equal(normal["actions"], cached["actions"])
    np.testing.assert_array_equal(normal["execution_horizon_ordered_continuation_logits"], cached["execution_horizon_ordered_continuation_logits"])
    assert "execution_horizon_expert_hidden" not in normal
    assert "execution_horizon_prefix_tokens" not in normal
    assert cached["execution_horizon_expert_hidden"].shape == (25, 4)
    assert cached["execution_horizon_prefix_tokens"].shape == (6, 2048)
    assert cached["execution_horizon_prefix_mask"].shape == (6,)
    inputs = model.predictor_inputs[-1]
    mapping = {
        "prefix_feature": "prefix_feature", "proprioception": "state_normalized",
        "final_actions": "final_actions_normalized", "coarse_actions": "coarse_actions_normalized",
        "previous_actions": "previous_actions_normalized", "previous_h": "previous_h",
        "previous_valid": "previous_valid", "budget_balance": "budget_balance", "episode_progress": "episode_progress",
        "prefix_tokens": "prefix_tokens", "prefix_mask": "prefix_mask",
    }
    for source, suffix in mapping.items():
        np.testing.assert_array_equal(np.asarray(inputs[source])[0], cached[f"execution_horizon_{suffix}"])
    assert "expert_hidden" not in inputs
    np.testing.assert_allclose(cached["execution_horizon_previous_actions_normalized"][:, :7], 1.0, atol=1e-6)
    np.testing.assert_array_equal(cached["execution_horizon_previous_actions_normalized"][:, 7:], 0.0)
    assert collections.Counter(str(call) for call in model.calls) == collections.Counter({
        "prefix": 2, "implicit": 2, "coarse": 2, "predictor": 2, "('expert', False)": 1, "('expert', True)": 1,
    })
    assert {"infer_ms", "action_expert_ms", "execution_horizon_predictor_ms"}.issubset(cached["policy_timing"])


def test_expert_sidecar_receives_hidden_automatically_without_rpc_export(monkeypatch):
    instance, model = _make_policy(monkeypatch, expert_dim=4)
    result = instance.infer(_request())
    assert ("expert", True) in model.calls
    assert model.predictor_inputs[-1]["expert_hidden"].shape == (1, 25, 4)
    assert "execution_horizon_expert_hidden" not in result
    assert "execution_horizon_prefix_tokens" not in result


@pytest.mark.parametrize("options", [
    {"num_steps": 1}, {"final_denoising_steps": 1}, {"ofp_interval_flow": True},
    {"final_hybrid_mode": "control_nfe2"}, {"token_time_warp_alpha": 0.1},
])
def test_architecture_features_reject_unsupported_expert_paths(monkeypatch, options):
    instance, _ = _make_policy(monkeypatch)
    with pytest.raises(ValueError, match="architecture features"):
        instance._validate_execution_horizon_architecture_path(
            options, profile_policy_timing=True, joint_coupled_sampler=False,
            batched_mc_samples=0, temporal_prefix_reuse_period=0,
        )


def test_architecture_cache_requires_profiled_path_before_any_vla_stage(monkeypatch):
    instance, model = _make_policy(monkeypatch)
    with pytest.raises(ValueError, match="profile_policy_timing"):
        instance.infer({**_request(), "profile_policy_timing": False,
                        "execution_horizon_export_architecture_cache": True})
    assert model.calls == []
