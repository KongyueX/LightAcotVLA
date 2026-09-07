from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import acot_vla
from openpi.models import model as model_lib
from openpi.shared import nnx_utils


class _PassThroughLLM(nnx.Module):
    def __call__(self, inputs, **kwargs):
        return (None, None, inputs[2]), None


class _FirstFeatureProjection(nnx.Module):
    def __call__(self, hidden):
        return hidden[..., :1]


class _TinyExpert(nnx.Module):
    def __init__(self):
        self.action_horizon = 2
        self.execution_horizon_expert_width = 4
        self.PaliGemma = nnx.Dict(llm=_PassThroughLLM())
        self.action_out_proj = _FirstFeatureProjection()

    def embed_suffix(self, observation, actions, times, **kwargs):
        tokens = jnp.broadcast_to(times[:, None, None], (1, 2, 4)).astype(jnp.bfloat16)
        return tokens, jnp.ones((1, 2), dtype=jnp.bool_), jnp.zeros((1, 2), dtype=jnp.bool_), None

    def sample_actions_profile_expert(self, *args, **kwargs):
        return acot_vla.ACOT_VLA.sample_actions_profile_expert(self, *args, **kwargs)


def test_expert_cache_is_the_last_executed_suffix_and_keeps_actions_identical():
    expert = _TinyExpert()
    sampler = nnx_utils.module_jit(expert.sample_actions_profile_expert, static_argnames=("return_expert_hidden",))
    prefix = {
        "observation": model_lib.Observation(images={}, image_masks={}, state=jnp.zeros((1, 1))),
        "prefix_tokens": jnp.zeros((1, 3, 4)), "prefix_mask": jnp.ones((1, 3), dtype=jnp.bool_),
        "kv_cache": None, "expert_action_noise": jnp.zeros((1, 2, 1)),
    }
    normal = sampler(prefix, None, None, num_steps=4)
    cached = sampler(prefix, None, None, num_steps=4, return_expert_hidden=True)
    normal_again = sampler(prefix, None, None, num_steps=4)
    assert set(normal) == {"actions"}
    assert set(cached) == {"actions", "execution_horizon_expert_hidden"}
    np.testing.assert_array_equal(cached["actions"], normal["actions"])
    np.testing.assert_array_equal(normal_again["actions"], normal["actions"])
    np.testing.assert_array_equal(normal["actions"], np.full((1, 2, 1), -0.625))
    # Four executed evaluations use t=1,.75,.5,.25. An extra t=0 evaluation
    # would change this tensor to zero even if final actions stayed unchanged.
    np.testing.assert_array_equal(cached["execution_horizon_expert_hidden"], np.full((1, 2, 4), 0.25))
    assert cached["execution_horizon_expert_hidden"].dtype == jnp.float32


def test_acot_config_accepts_real_expert_width_and_rejects_mismatched_sidecar():
    options = {
        "action_horizon": 25, "coarse_action_horizon": 15, "execution_horizon_predictor": True,
        "execution_horizon_temporal_backbone": "transformer", "execution_horizon_temporal_layers": 2,
        "execution_horizon_candidate_horizons": (5, 10, 15, 20, 25),
        "execution_horizon_visual_num_queries": 4,
        "execution_horizon_ordered_continuation_head": True,
    }
    original = acot_vla.ACOTConfig(**options)
    assert original.execution_horizon_visual_query_conditioning is False
    assert original.execution_horizon_expert_feature_dim == 0
    query = acot_vla.ACOTConfig(**options, execution_horizon_visual_query_conditioning=True)
    expert = acot_vla.ACOTConfig(**options, execution_horizon_expert_feature_dim=1024)
    assert query.execution_horizon_visual_query_conditioning is True
    assert expert.execution_horizon_expert_feature_dim == 1024
    with pytest.raises(ValueError, match="final action expert width"):
        acot_vla.ACOTConfig(**options, execution_horizon_expert_feature_dim=256)
