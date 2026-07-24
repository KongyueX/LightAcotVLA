import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import transported_action_cot
import pytest


def _inputs(
    config: transported_action_cot.TransportedActionCoTConfig,
    *,
    batch_size: int = 2,
    iar_tokens: int = 18,
) -> dict[str, jax.Array]:
    keys = jax.random.split(jax.random.key(23), 5)
    anchor_images = jax.random.normal(
        keys[0],
        (
            batch_size,
            config.image_views,
            config.image_size,
            config.image_size,
            config.image_channels,
        ),
    )
    return {
        "anchor_images": anchor_images,
        "current_images": jax.random.normal(keys[1], anchor_images.shape),
        "anchor_state": jax.random.normal(keys[2], (batch_size, config.state_dim)),
        "current_state": jax.random.normal(keys[3], (batch_size, config.state_dim)),
        "cached_ear": jax.random.normal(
            keys[4],
            (batch_size, config.ear_horizon, config.action_dim),
        ),
        "cached_iar": jax.random.normal(
            jax.random.key(29),
            (batch_size, iar_tokens, config.iar_dim),
        ),
        "cache_age": jnp.arange(batch_size, dtype=jnp.float32),
    }


def _parameter_count(model: nnx.Module) -> int:
    parameter_state = nnx.state(model, nnx.Param)
    return sum(int(np.prod(value.shape)) for value in jax.tree.leaves(parameter_state))


def _activate_observation_phase_path(
    model: transported_action_cot.TransportedActionCoTExecutor,
) -> None:
    kernel = jnp.zeros_like(model.phase_out.kernel.value)
    model.phase_out.kernel.value = kernel.at[:, 0].set(
        jnp.linspace(-0.02, 0.02, kernel.shape[0], dtype=kernel.dtype)
    )


def test_default_configuration_is_below_parameter_budget() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))

    estimated = transported_action_cot.estimate_parameter_count(config)
    actual = _parameter_count(model)

    assert actual == estimated
    assert actual < 5_000_000


def test_zero_initialized_phase_is_nominal_monotonic_transport() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)

    action, transported_ear, phase = model.forward_with_aux(**inputs)

    token_positions = jnp.arange(config.ear_horizon, dtype=jnp.float32)
    expected_phase = jnp.clip(
        inputs["cache_age"][:, None] / config.coarse_time_stride + token_positions[None, :],
        0.0,
        config.max_phase,
    )
    expected_transport = transported_action_cot.interpolate_ear(inputs["cached_ear"], expected_phase)

    np.testing.assert_allclose(phase, expected_phase, atol=1e-6)
    np.testing.assert_allclose(transported_ear, expected_transport, atol=1e-6)
    np.testing.assert_allclose(action, transported_ear[:, 0], atol=1e-6)
    assert bool(jnp.all(jnp.diff(phase, axis=1) >= 0.0))
    assert action.shape == (2, config.action_dim)
    assert transported_ear.shape == (2, config.ear_horizon, config.action_dim)
    assert phase.shape == (2, config.ear_horizon)


def test_observation_change_can_affect_phase() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    _activate_observation_phase_path(model)
    inputs = _inputs(config, batch_size=1)
    inputs["cache_age"] = jnp.zeros((1,), dtype=jnp.float32)

    matching_inputs = dict(inputs)
    matching_inputs["current_images"] = inputs["anchor_images"]
    changed_inputs = dict(inputs)
    changed_inputs["current_images"] = -inputs["anchor_images"]
    _, _, matching_phase = model.forward_with_aux(**matching_inputs)
    _, _, changed_phase = model.forward_with_aux(**changed_inputs)

    assert float(jnp.max(jnp.abs(matching_phase - changed_phase))) > 1e-5
    assert bool(jnp.all(jnp.diff(changed_phase, axis=1) >= -1e-6))
    assert float(jnp.min(changed_phase)) >= 0.0
    assert float(jnp.max(changed_phase)) <= config.max_phase


def test_action_has_no_direct_observation_path() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    _activate_observation_phase_path(model)
    inputs = _inputs(config, batch_size=1)
    constant_token = jnp.linspace(-1.0, 1.0, config.action_dim, dtype=jnp.float32)
    inputs["cached_ear"] = jnp.broadcast_to(
        constant_token,
        (1, config.ear_horizon, config.action_dim),
    )

    first_inputs = dict(inputs)
    first_inputs["current_images"] = inputs["anchor_images"]
    second_inputs = dict(inputs)
    second_inputs["current_images"] = -inputs["anchor_images"]
    first_action, _, first_phase = model.forward_with_aux(**first_inputs)
    second_action, _, second_phase = model.forward_with_aux(**second_inputs)

    assert float(jnp.max(jnp.abs(first_phase - second_phase))) > 1e-5
    np.testing.assert_allclose(first_action, second_action, atol=1e-6)


def test_parameter_gradients_are_finite() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    inputs = _inputs(config)

    def loss_function(candidate_params: nnx.State) -> jax.Array:
        candidate = nnx.merge(graphdef, candidate_params)
        action, transported_ear, phase = candidate.forward_with_aux(**inputs)
        phase_target = jax.lax.stop_gradient(jnp.clip(phase + 0.25, 0.0, config.max_phase))
        return (
            jnp.mean(jnp.square(action))
            + 0.01 * jnp.mean(jnp.square(transported_ear))
            + jnp.mean(jnp.square(phase - phase_target))
        )

    loss, gradients = jax.value_and_grad(loss_function)(params)

    assert bool(jnp.isfinite(loss))
    assert jax.tree.leaves(gradients)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradients))


@pytest.mark.parametrize("iar_tokens", [1, 5, 18])
def test_optional_and_variable_length_iar(iar_tokens: int) -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config, iar_tokens=iar_tokens)

    assert model(**inputs).shape == (2, config.action_dim)
    inputs["cached_iar"] = None
    assert model(**inputs).shape == (2, config.action_dim)


def test_invalid_shapes_are_rejected() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)
    inputs["anchor_images"] = inputs["anchor_images"][:, :, :-1]

    with pytest.raises(ValueError, match="anchor_images axis 2"):
        model(**inputs)
