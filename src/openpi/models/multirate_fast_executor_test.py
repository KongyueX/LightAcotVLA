import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import multirate_fast_executor as fast_executor
import pytest


def _inputs(
    config: fast_executor.MultiRateFastExecutorConfig,
    *,
    batch_size: int = 2,
    iar_tokens: int = 18,
) -> dict[str, jax.Array]:
    keys = jax.random.split(jax.random.key(11), 4)
    return {
        "current_images": jax.random.normal(
            keys[0],
            (
                batch_size,
                config.image_views,
                config.image_size,
                config.image_size,
                config.image_channels,
            ),
        ),
        "state": jax.random.normal(keys[1], (batch_size, config.state_dim)),
        "cached_ear": jax.random.normal(
            keys[2],
            (batch_size, config.ear_horizon, config.action_dim),
        ),
        "cached_iar": jax.random.normal(
            keys[3],
            (batch_size, iar_tokens, config.iar_dim),
        ),
        "cache_age": jnp.arange(batch_size, dtype=jnp.int32),
    }


def _parameter_count(model: nnx.Module) -> int:
    parameter_state = nnx.state(model, nnx.Param)
    return sum(int(np.prod(value.shape)) for value in jax.tree.leaves(parameter_state))


def _ear_necessity_gap(
    model: fast_executor.MultiRateFastExecutor,
    inputs: dict[str, jax.Array],
) -> float:
    baseline = model(**inputs)
    ablated_inputs = dict(inputs)
    ablated_inputs["cached_ear"] = jnp.zeros_like(inputs["cached_ear"])
    ablated = model(**ablated_inputs)
    return float(jnp.mean(jnp.linalg.norm(baseline - ablated, axis=-1)))


def test_default_configuration_is_below_parameter_budget() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))

    estimated = fast_executor.estimate_parameter_count(config)
    actual = _parameter_count(model)

    assert actual == estimated
    assert actual < 10_000_000
    assert actual < 2_000_000


def test_phase_aligned_ear_token_interpolates_raw_time() -> None:
    cached_ear = jnp.arange(5 * 4, dtype=jnp.float32).reshape((1, 5, 4))
    cached_ear = jnp.repeat(cached_ear, 4, axis=0)
    cache_age = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

    selected = fast_executor.phase_aligned_ear_token(
        cached_ear,
        cache_age,
        max_cache_age=3,
        coarse_time_stride=2,
    )

    np.testing.assert_array_equal(selected[:, 0], np.asarray([0.0, 2.0, 4.0, 6.0]))
    np.testing.assert_array_equal(selected[1], 0.5 * (cached_ear[1, 0] + cached_ear[1, 1]))
    np.testing.assert_array_equal(selected[3], 0.5 * (cached_ear[3, 1] + cached_ear[3, 2]))


def test_phase_aligned_ear_token_clips_invalid_age() -> None:
    cached_ear = jnp.arange(5 * 4, dtype=jnp.float32).reshape((1, 5, 4))
    cached_ear = jnp.repeat(cached_ear, 2, axis=0)

    selected = fast_executor.phase_aligned_ear_token(
        cached_ear,
        jnp.asarray([-1, 9], dtype=jnp.int32),
        max_cache_age=3,
        coarse_time_stride=2,
    )

    np.testing.assert_array_equal(selected[:, 0], np.asarray([0.0, 6.0]))


def test_executor_output_shape_and_zero_initialized_action_and_refresh() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)

    predicted, predicted_refresh = model.forward_with_aux(**inputs)
    expected_base = fast_executor.phase_aligned_ear_token(
        inputs["cached_ear"],
        inputs["cache_age"],
        max_cache_age=config.max_cache_age,
        coarse_time_stride=config.coarse_time_stride,
    )

    assert predicted.shape == (2, config.action_dim)
    assert predicted_refresh.shape == (2, config.action_dim)
    np.testing.assert_allclose(predicted, expected_base, atol=1e-6)
    np.testing.assert_allclose(predicted_refresh, expected_base, atol=1e-6)


def test_executor_supports_split_merge_jit() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    inputs = _inputs(config)

    @jax.jit
    def apply(current_params: nnx.State) -> jax.Array:
        candidate = nnx.merge(graphdef, current_params)
        return candidate(**inputs)

    predicted = apply(params)

    assert predicted.shape == (2, config.action_dim)
    assert bool(jnp.all(jnp.isfinite(predicted)))


def test_executor_auxiliary_output_supports_split_merge_jit() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    inputs = _inputs(config)

    @jax.jit
    def apply(current_params: nnx.State) -> tuple[jax.Array, jax.Array]:
        candidate = nnx.merge(graphdef, current_params)
        return candidate.forward_with_aux(**inputs)

    predicted, predicted_refresh = apply(params)

    assert predicted.shape == (2, config.action_dim)
    assert predicted_refresh.shape == (2, config.action_dim)
    assert bool(jnp.all(jnp.isfinite(predicted_refresh)))


@pytest.mark.parametrize("iar_tokens", [1, 5, 18])
def test_executor_accepts_variable_iar_token_length(iar_tokens: int) -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config, iar_tokens=iar_tokens)

    assert model(**inputs).shape == (2, config.action_dim)


def test_executor_zero_fills_missing_iar() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)
    inputs["cached_iar"] = None

    predicted = model(**inputs)

    assert predicted.shape == (2, config.action_dim)
    assert bool(jnp.all(jnp.isfinite(predicted)))


def test_ear_is_necessary_for_initial_base_action() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))

    assert _ear_necessity_gap(model, _inputs(config)) > 0.1


def test_gripper_weighted_huber_loss() -> None:
    config = fast_executor.FastExecutorLossConfig(
        action_dim=4,
        huber_delta=1.0,
        gripper_indices=(2,),
        gripper_weight=5.0,
    )
    target = jnp.zeros((1, 4), dtype=jnp.float32)
    regular_error = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    gripper_error = jnp.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=jnp.float32)

    regular_loss, _ = fast_executor.multirate_fast_executor_loss(
        regular_error,
        target,
        config=config,
    )
    gripper_loss, gripper_metrics = fast_executor.multirate_fast_executor_loss(
        gripper_error,
        target,
        config=config,
    )

    np.testing.assert_allclose(regular_loss, 0.5 / 8.0)
    np.testing.assert_allclose(gripper_loss, 2.5 / 8.0)
    np.testing.assert_allclose(gripper_metrics["gripper_huber"], 0.5)
    assert float(gripper_loss) == pytest.approx(5.0 * float(regular_loss))


def test_loss_supports_sample_mask_and_gradients() -> None:
    config = fast_executor.FastExecutorLossConfig(action_dim=4, gripper_indices=(2,))
    predicted = jnp.ones((2, 4), dtype=jnp.float32)
    target = jnp.zeros_like(predicted)

    def loss_function(actions: jax.Array) -> jax.Array:
        loss, _ = fast_executor.multirate_fast_executor_loss(
            actions,
            target,
            config=config,
            valid_mask=jnp.asarray([1.0, 0.0]),
        )
        return loss

    gradient = jax.grad(loss_function)(predicted)

    assert gradient.shape == predicted.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_array_equal(gradient[1], jnp.zeros((4,), dtype=jnp.float32))


def test_invalid_shapes_are_rejected() -> None:
    config = fast_executor.MultiRateFastExecutorConfig()
    model = fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)
    inputs["current_images"] = inputs["current_images"][:, :, :-1]

    with pytest.raises(ValueError, match="current_images axis 2"):
        model(**inputs)
