# ruff: noqa: SLF001

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

import train_branched_action_corrector as corrector


def _inputs(batch_size: int = 2) -> dict[str, jnp.ndarray]:
    return {
        "anchor_images": jnp.zeros((batch_size, 2, 64, 64, 3)),
        "current_images": jnp.ones((batch_size, 2, 64, 64, 3)),
        "anchor_state": jnp.zeros((batch_size, 32)),
        "current_state": jnp.ones((batch_size, 32)),
        "cached_plan_tokens": jnp.zeros((batch_size, 25, 32)),
        "cached_iar": jnp.zeros((batch_size, 18, 1024)),
        "intended_prefix": jnp.zeros((batch_size, 4, 7)),
        "intended_valid": jnp.ones((batch_size, 4), dtype=jnp.bool_),
        "transported_ear": jnp.zeros((batch_size, 15, 7)),
        "base_actions": jnp.zeros((batch_size, 4, 7)),
    }


def _model(mode: corrector.Mode) -> corrector.BranchedActionCorrector:
    return corrector.BranchedActionCorrector(
        mode=mode,
        image_views=2,
        image_channels=3,
        state_dim=32,
        plan_dim=32,
        iar_dim=1024,
        env_action_dim=7,
        max_executed_steps=4,
        ear_horizon=15,
        rollout_horizon=4,
        hidden_dim=128,
        action_residual_scale=(0.5,) * 7,
        plan_residual_scale=(0.5,) * 7,
        gripper_logit_scale=8.0,
        rngs=nnx.Rngs(0),
    )


def test_direct_zero_initialization_preserves_continuous_cached_actions() -> None:
    inputs = _inputs()
    inputs["base_actions"] = jnp.asarray(
        np.random.default_rng(0).normal(size=(2, 4, 7)),
        dtype=jnp.float32,
    )
    output = _model("direct")(**inputs)
    np.testing.assert_allclose(output.actions[..., :6], inputs["base_actions"][..., :6], atol=1e-6)
    assert output.actions.shape == (2, 4, 7)
    assert output.revised_ear.shape == (2, 15, 7)


def test_plan_zero_initialization_has_no_observation_to_action_bypass() -> None:
    model = _model("plan")
    inputs = _inputs()
    first = model(**inputs).actions
    inputs["current_images"] = jnp.full_like(inputs["current_images"], 7.0)
    inputs["current_state"] = jnp.full_like(inputs["current_state"], -3.0)
    second = model(**inputs).actions
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_transport_ear_advances_the_cached_plan() -> None:
    cached = np.arange(2 * 15 * 7, dtype=np.float32).reshape((2, 15, 7))
    transported = corrector._transport_ear(cached, 2.0)
    np.testing.assert_allclose(transported[:, 0], cached[:, 2])
    np.testing.assert_allclose(transported[:, -1], cached[:, -1])
