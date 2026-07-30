import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import transported_action_cot


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
    model.phase_out.kernel.value = kernel.at[:, 0].set(jnp.linspace(-0.02, 0.02, kernel.shape[0], dtype=kernel.dtype))


def _activate_plan_geometry_path(
    model: transported_action_cot.TransportedActionCoTExecutor,
) -> None:
    kernel = jnp.zeros_like(model.geometry_coefficients.kernel.value)
    input_pattern = jnp.linspace(-0.02, 0.02, kernel.shape[0], dtype=kernel.dtype)
    output_pattern = jnp.linspace(0.25, 1.0, kernel.shape[1], dtype=kernel.dtype)
    model.geometry_coefficients.kernel.value = input_pattern[:, None] * output_pattern[None, :]


def _activate_direct_action_path(
    model: transported_action_cot.TransportedActionCoTExecutor,
) -> None:
    kernel = jnp.zeros_like(model.direct_action_out.kernel.value)
    input_pattern = jnp.linspace(-0.02, 0.02, kernel.shape[0], dtype=kernel.dtype)
    output_pattern = jnp.linspace(0.25, 1.0, kernel.shape[1], dtype=kernel.dtype)
    model.direct_action_out.kernel.value = input_pattern[:, None] * output_pattern[None, :]


def test_default_configuration_is_below_parameter_budget() -> None:
    config = transported_action_cot.TransportedActionCoTConfig()
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))

    estimated = transported_action_cot.estimate_parameter_count(config)
    actual = _parameter_count(model)

    assert actual == estimated
    assert actual == 474_530
    assert actual < 5_000_000
    assert config.correction_mode == "phase"
    assert not hasattr(model, "plan_temporal_basis")
    assert not hasattr(model, "direct_action_out")


def test_plan_and_direct_corrections_have_matched_default_parameter_counts() -> None:
    phase_config = transported_action_cot.TransportedActionCoTConfig()
    plan_config = dataclasses.replace(phase_config, correction_mode="plan")
    direct_config = dataclasses.replace(phase_config, correction_mode="direct")
    phase_model = transported_action_cot.TransportedActionCoTExecutor(
        phase_config,
        rngs=nnx.Rngs(0),
    )
    plan_model = transported_action_cot.TransportedActionCoTExecutor(
        plan_config,
        rngs=nnx.Rngs(0),
    )
    direct_model = transported_action_cot.TransportedActionCoTExecutor(
        direct_config,
        rngs=nnx.Rngs(0),
    )

    phase_count = _parameter_count(phase_model)
    plan_count = _parameter_count(plan_model)
    direct_count = _parameter_count(direct_model)

    assert plan_count == transported_action_cot.estimate_parameter_count(plan_config)
    assert direct_count == transported_action_cot.estimate_parameter_count(direct_config)
    assert plan_count - phase_count == 4_128
    assert direct_count == plan_count


def test_event_correction_adds_only_one_scalar_head() -> None:
    phase_config = transported_action_cot.TransportedActionCoTConfig()
    event_config = dataclasses.replace(phase_config, correction_mode="event")
    phase_model = transported_action_cot.TransportedActionCoTExecutor(
        phase_config,
        rngs=nnx.Rngs(0),
    )
    event_model = transported_action_cot.TransportedActionCoTExecutor(
        event_config,
        rngs=nnx.Rngs(0),
    )

    phase_count = _parameter_count(phase_model)
    event_count = _parameter_count(event_model)

    assert event_count == transported_action_cot.estimate_parameter_count(event_config)
    assert event_count - phase_count == phase_config.hidden_dim + 1


def test_update_confidence_is_a_zero_initialized_scalar_sidecar() -> None:
    context_dim = 128
    router = transported_action_cot.ActionCoTUpdateConfidence(
        context_dim,
        rngs=nnx.Rngs(0),
    )
    context = jax.random.normal(jax.random.key(31), (3, context_dim))

    logits = router(context)

    assert logits.shape == (3,)
    np.testing.assert_allclose(logits, 0.0, atol=0.0)
    assert _parameter_count(router) == context_dim + 1
    assert _parameter_count(router) == transported_action_cot.estimate_update_confidence_parameter_count(context_dim)


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


@pytest.mark.parametrize("correction_mode", ["phase", "plan", "direct", "event"])
def test_forward_with_details_has_stable_shapes(correction_mode: str) -> None:
    config = transported_action_cot.TransportedActionCoTConfig(
        correction_mode=correction_mode,  # type: ignore[arg-type]
    )
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)
    bounded_gripper = jnp.linspace(-0.8, 0.8, config.ear_horizon)
    inputs["cached_ear"] = inputs["cached_ear"].at[..., 6].set(bounded_gripper[None, :])

    output = model.forward_with_details(**inputs)

    assert output.action.shape == (2, config.action_dim)
    assert output.base_action.shape == output.action.shape
    assert output.transported_ear.shape == (2, config.ear_horizon, config.action_dim)
    assert output.revised_ear.shape == output.transported_ear.shape
    assert output.phase.shape == (2, config.ear_horizon)
    assert output.event_phase_offset.shape == (2,)
    assert output.geometry_residual.shape == (2, config.ear_horizon, 6)
    assert output.gripper_logits.shape == (2, config.ear_horizon)
    assert output.event_prob.shape == (2, config.ear_horizon - 1)
    assert output.direct_action_residual.shape == output.action.shape
    assert output.update_context.shape == (2, config.hidden_dim)
    assert bool(jnp.all((output.event_prob >= 0.0) & (output.event_prob <= 1.0)))
    if correction_mode != "direct":
        np.testing.assert_allclose(output.direct_action_residual, 0.0, atol=0.0)
    if correction_mode != "plan":
        np.testing.assert_allclose(output.geometry_residual, 0.0, atol=0.0)
    if correction_mode not in ("plan", "event"):
        np.testing.assert_allclose(output.revised_ear, output.transported_ear, atol=0.0)
    if correction_mode != "event":
        np.testing.assert_allclose(output.event_phase_offset, 0.0, atol=0.0)
    if correction_mode == "phase":
        np.testing.assert_allclose(output.action, output.base_action, atol=0.0)


def test_zero_initialized_plan_heads_start_from_transported_plan() -> None:
    config = transported_action_cot.TransportedActionCoTConfig(correction_mode="plan")
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    inputs = _inputs(config)
    bounded_gripper = jnp.linspace(-0.8, 0.8, config.ear_horizon)
    inputs["cached_ear"] = inputs["cached_ear"].at[..., 6].set(bounded_gripper[None, :])

    output = model.forward_with_details(**inputs)

    np.testing.assert_allclose(output.geometry_residual, 0.0, atol=0.0)
    np.testing.assert_allclose(output.revised_ear, output.transported_ear, atol=1e-6)
    np.testing.assert_allclose(
        output.action[:, 6],
        output.revised_ear[:, 0, 6],
        atol=1e-6,
    )


def test_zero_initialized_event_mode_is_exactly_matched_to_phase() -> None:
    phase_config = transported_action_cot.TransportedActionCoTConfig()
    event_config = dataclasses.replace(
        phase_config,
        correction_mode="event",
        isolate_event_gradients=True,
    )
    phase_model = transported_action_cot.TransportedActionCoTExecutor(
        phase_config,
        rngs=nnx.Rngs(0),
    )
    event_model = transported_action_cot.TransportedActionCoTExecutor(
        event_config,
        rngs=nnx.Rngs(0),
    )
    inputs = _inputs(phase_config)
    inputs["cached_ear"] = (
        inputs["cached_ear"].at[..., 6].set(jnp.linspace(-0.8, 0.8, phase_config.ear_horizon)[None, :])
    )

    phase_output = phase_model.forward_with_details(**inputs)
    event_output = event_model.forward_with_details(**inputs)

    np.testing.assert_allclose(event_output.event_phase_offset, 0.0, atol=0.0)
    np.testing.assert_allclose(event_output.phase, phase_output.phase, atol=1e-6)
    np.testing.assert_allclose(event_output.revised_ear, phase_output.revised_ear, atol=1e-6)
    np.testing.assert_allclose(event_output.action, phase_output.action, atol=1e-6)


def test_event_shift_changes_only_gripper_timing() -> None:
    config = transported_action_cot.TransportedActionCoTConfig(
        correction_mode="event",
        isolate_event_gradients=True,
    )
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    model.event_scalar_out.bias.value = jnp.ones_like(model.event_scalar_out.bias.value)
    inputs = _inputs(config, batch_size=1)
    inputs["cached_ear"] = inputs["cached_ear"].at[..., 6].set(jnp.linspace(-0.9, 0.9, config.ear_horizon)[None, :])

    output = model.forward_with_details(**inputs)
    base_action = model._decode_action(output.transported_ear)  # noqa: SLF001

    assert float(output.event_phase_offset[0]) > 0.0
    np.testing.assert_allclose(
        output.revised_ear[..., :6],
        output.transported_ear[..., :6],
        atol=0.0,
    )
    np.testing.assert_allclose(output.action[..., :6], base_action[..., :6], atol=0.0)
    np.testing.assert_allclose(
        output.action[..., 6] - base_action[..., 6],
        output.revised_ear[:, 0, 6] - output.transported_ear[:, 0, 6],
        atol=1e-6,
    )


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


def test_plan_mode_observation_correction_is_mediated_by_revised_ear() -> None:
    config = transported_action_cot.TransportedActionCoTConfig(correction_mode="plan")
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    _activate_plan_geometry_path(model)
    inputs = _inputs(config, batch_size=1)
    constant_token = jnp.linspace(-0.8, 0.8, config.action_dim, dtype=jnp.float32)
    inputs["cached_ear"] = jnp.broadcast_to(
        constant_token,
        (1, config.ear_horizon, config.action_dim),
    )
    inputs["cache_age"] = jnp.zeros((1,), dtype=jnp.float32)

    first_inputs = dict(inputs)
    first_inputs["current_images"] = inputs["anchor_images"]
    second_inputs = dict(inputs)
    second_inputs["current_images"] = -inputs["anchor_images"]
    first = model.forward_with_details(**first_inputs)
    second = model.forward_with_details(**second_inputs)

    np.testing.assert_allclose(first.phase, second.phase, atol=1e-6)
    np.testing.assert_allclose(first.transported_ear, second.transported_ear, atol=1e-6)
    assert float(jnp.max(jnp.abs(first.geometry_residual - second.geometry_residual))) > 1e-6
    assert float(jnp.max(jnp.abs(first.revised_ear - second.revised_ear))) > 1e-6
    assert float(jnp.max(jnp.abs(first.action - second.action))) > 1e-6
    np.testing.assert_allclose(first.direct_action_residual, 0.0, atol=0.0)
    np.testing.assert_allclose(second.direct_action_residual, 0.0, atol=0.0)


def test_direct_mode_changes_action_without_revising_plan() -> None:
    config = transported_action_cot.TransportedActionCoTConfig(correction_mode="direct")
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    _activate_direct_action_path(model)
    inputs = _inputs(config, batch_size=1)
    constant_token = jnp.linspace(-0.8, 0.8, config.action_dim, dtype=jnp.float32)
    inputs["cached_ear"] = jnp.broadcast_to(
        constant_token,
        (1, config.ear_horizon, config.action_dim),
    )
    inputs["cache_age"] = jnp.zeros((1,), dtype=jnp.float32)

    first_inputs = dict(inputs)
    first_inputs["current_images"] = inputs["anchor_images"]
    second_inputs = dict(inputs)
    second_inputs["current_images"] = -inputs["anchor_images"]
    first = model.forward_with_details(**first_inputs)
    second = model.forward_with_details(**second_inputs)

    np.testing.assert_allclose(first.phase, second.phase, atol=1e-6)
    np.testing.assert_allclose(first.revised_ear, second.revised_ear, atol=1e-6)
    assert float(jnp.max(jnp.abs(first.direct_action_residual - second.direct_action_residual))) > 1e-6
    assert float(jnp.max(jnp.abs(first.action - second.action))) > 1e-6
    np.testing.assert_allclose(first.geometry_residual, 0.0, atol=0.0)
    np.testing.assert_allclose(second.geometry_residual, 0.0, atol=0.0)


def test_plan_gripper_logits_define_revised_gripper_and_events() -> None:
    config = transported_action_cot.TransportedActionCoTConfig(correction_mode="plan")
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    model.gripper_coefficients.bias.value = jnp.linspace(
        -0.5,
        0.5,
        config.geometry_rank,
    )
    inputs = _inputs(config, batch_size=1)
    inputs["cached_ear"] = inputs["cached_ear"].at[..., 6].set(jnp.linspace(-0.8, 0.8, config.ear_horizon)[None, :])

    output = model.forward_with_details(**inputs)
    probability = jax.nn.sigmoid(2.0 * output.gripper_logits)
    expected_event_probability = (
        probability[:, 1:] * (1.0 - probability[:, :-1]) + (1.0 - probability[:, 1:]) * probability[:, :-1]
    )

    np.testing.assert_allclose(
        output.revised_ear[..., 6],
        jnp.tanh(output.gripper_logits),
        atol=1e-6,
    )
    np.testing.assert_allclose(output.event_prob, expected_event_probability, atol=1e-6)
    np.testing.assert_allclose(output.action[:, 6], output.revised_ear[:, 0, 6], atol=1e-6)


def test_isolated_event_only_mode_preserves_continuous_plan() -> None:
    config = transported_action_cot.TransportedActionCoTConfig(
        correction_mode="plan",
        enable_geometry_correction=False,
        isolate_event_gradients=True,
    )
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    model.plan_temporal_basis.kernel.value = jnp.zeros_like(model.plan_temporal_basis.kernel.value)
    model.plan_temporal_basis.bias.value = jnp.ones_like(model.plan_temporal_basis.bias.value)
    model.gripper_coefficients.kernel.value = jnp.zeros_like(model.gripper_coefficients.kernel.value)
    model.gripper_coefficients.bias.value = jnp.ones_like(model.gripper_coefficients.bias.value)
    inputs = _inputs(config, batch_size=1)
    inputs["cached_ear"] = inputs["cached_ear"].at[..., 6].set(0.0)

    output = model.forward_with_details(**inputs)

    np.testing.assert_allclose(output.geometry_residual, 0.0, atol=0.0)
    np.testing.assert_allclose(
        output.revised_ear[..., :6],
        output.transported_ear[..., :6],
        atol=0.0,
    )
    assert float(jnp.min(output.revised_ear[..., 6])) > 0.0
    np.testing.assert_allclose(output.action[:, 6], output.revised_ear[:, 0, 6], atol=1e-6)


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


@pytest.mark.parametrize("correction_mode", ["plan", "direct", "event"])
def test_correction_parameter_gradients_are_finite(correction_mode: str) -> None:
    config = transported_action_cot.TransportedActionCoTConfig(
        correction_mode=correction_mode,  # type: ignore[arg-type]
    )
    model = transported_action_cot.TransportedActionCoTExecutor(config, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    inputs = _inputs(config)

    def loss_function(candidate_params: nnx.State) -> jax.Array:
        candidate = nnx.merge(graphdef, candidate_params)
        output = candidate.forward_with_details(**inputs)
        return (
            jnp.mean(jnp.square(output.action))
            + jnp.mean(jnp.square(output.revised_ear[..., :7]))
            + 0.1 * jnp.mean(jnp.square(output.event_prob))
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


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"correction_mode": "unknown"}, "correction_mode"),
        ({"geometry_rank": 3}, "geometry_rank"),
        ({"geometry_scale": (0.5,) * 5}, "geometry_scale"),
        ({"geometry_scale": (0.5, 0.5, 0.5, 0.5, 0.5, 0.0)}, "geometry_scale"),
        ({"direct_residual_scale": 0.0}, "direct_residual_scale"),
        ({"max_event_phase_offset": 0.0}, "max_event_phase_offset"),
    ],
)
def test_invalid_correction_configuration_is_rejected(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        transported_action_cot.TransportedActionCoTConfig(**updates)  # type: ignore[arg-type]
