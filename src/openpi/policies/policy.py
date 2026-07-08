from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias
import copy
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _as_bool(value: Any) -> bool:
    return bool(np.asarray(value).item())


def _block_until_ready(value: Any) -> None:
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._sample_actions_profile_prefix = None
        self._sample_actions_profile_implicit = None
        self._sample_actions_profile_coarse = None
        self._sample_actions_profile_expert = None
        if all(
            hasattr(model, name)
            for name in (
                "sample_actions_profile_prefix",
                "sample_actions_profile_implicit",
                "sample_actions_profile_coarse",
                "sample_actions_profile_expert",
            )
        ):
            self._sample_actions_profile_prefix = nnx_utils.module_jit(model.sample_actions_profile_prefix)
            self._sample_actions_profile_implicit = nnx_utils.module_jit(model.sample_actions_profile_implicit)
            self._sample_actions_profile_coarse = nnx_utils.module_jit(model.sample_actions_profile_coarse)
            self._sample_actions_profile_expert = nnx_utils.module_jit(model.sample_actions_profile_expert)

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        policy_seed = inputs.pop("policy_seed", None)
        coarse_actions_override = inputs.pop("coarse_actions_override", None)
        action_cot_skip_segment = inputs.pop("action_cot_skip_segment", None)
        profile_policy_timing = _as_bool(inputs.pop("profile_policy_timing", False))
        action_cot_coarse_num_steps = inputs.pop("action_cot_coarse_num_steps", None)
        if action_cot_coarse_num_steps is None:
            action_cot_coarse_num_steps = inputs.pop("coarse_num_steps", None)
        else:
            inputs.pop("coarse_num_steps", None)
        action_cot_dynamic_coarse_steps = inputs.pop("action_cot_dynamic_coarse_steps", None)
        if action_cot_dynamic_coarse_steps is None:
            action_cot_dynamic_coarse_steps = inputs.pop("dynamic_coarse_steps", None)
        else:
            inputs.pop("dynamic_coarse_steps", None)
        transformed_coarse_actions_override = None
        if coarse_actions_override is not None:
            override_inputs = jax.tree.map(lambda x: x, obs)
            override_inputs.pop("coarse_actions_override", None)
            override_inputs.pop("policy_seed", None)
            override_inputs.pop("action_cot_skip_segment", None)
            override_inputs.pop("profile_policy_timing", None)
            override_inputs.pop("action_cot_coarse_num_steps", None)
            override_inputs.pop("coarse_num_steps", None)
            override_inputs.pop("action_cot_dynamic_coarse_steps", None)
            override_inputs.pop("dynamic_coarse_steps", None)
            # Avoid data transforms regenerating coarse_actions from expert actions.
            override_inputs.pop("actions", None)
            override_inputs["coarse_actions"] = coarse_actions_override
            override_inputs = self._input_transform(override_inputs)
            if "coarse_actions" not in override_inputs:
                raise KeyError("Input transforms did not preserve coarse_actions_override as coarse_actions.")
            transformed_coarse_actions_override = override_inputs["coarse_actions"]

        inputs = self._input_transform(inputs)
        if transformed_coarse_actions_override is not None:
            inputs["coarse_actions_override"] = transformed_coarse_actions_override

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        if policy_seed is None:
            self._rng, sample_rng = jax.random.split(self._rng)
        else:
            sample_rng = jax.random.key(int(np.asarray(policy_seed).item()))
        outputs = {
            "state": inputs["state"]
        }
        sample_kwargs = self._sample_kwargs
        if "coarse_actions_override" in inputs:
            sample_kwargs = {
                **sample_kwargs,
                "explicit_action_reason_override": inputs.pop("coarse_actions_override"),
            }
        if action_cot_skip_segment is not None:
            sample_kwargs = {
                **sample_kwargs,
                "explicit_action_skip_segment": np.asarray(action_cot_skip_segment, dtype=np.int32).reshape(()),
            }
        if action_cot_coarse_num_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "coarse_num_steps": np.asarray(action_cot_coarse_num_steps, dtype=np.int32).reshape(()),
            }
        if action_cot_dynamic_coarse_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "dynamic_coarse_steps": bool(np.asarray(action_cot_dynamic_coarse_steps).item()),
            }
        observation = _model.Observation.from_dict(inputs)
        detailed_timing = {}
        if profile_policy_timing and self._can_profile_sample_actions():
            result, detailed_timing = self._profile_sample_actions(sample_rng, observation, sample_kwargs)
        else:
            result = self._sample_actions(sample_rng, observation, **sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)
        else:
            outputs["actions"] = result
        # outputs["actions"] = inputs["actions"]

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time
        if detailed_timing:
            stage_total_ms = sum(
                detailed_timing.get(key, 0.0)
                for key in (
                    "vlm_ms",
                    "implicit_action_reasoner_ms",
                    "coarse_action_expert_ms",
                    "action_expert_ms",
                )
            )
            detailed_timing["profile_overhead_ms"] = max(0.0, model_time * 1000 - stage_total_ms)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            **detailed_timing,
        }
        return self.post_process(obs, outputs)

    def _can_profile_sample_actions(self) -> bool:
        return (
            self._sample_actions_profile_prefix is not None
            and self._sample_actions_profile_implicit is not None
            and self._sample_actions_profile_coarse is not None
            and self._sample_actions_profile_expert is not None
        )

    def _profile_sample_actions(
        self,
        sample_rng: at.KeyArrayLike,
        observation: _model.Observation,
        sample_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        assert self._sample_actions_profile_prefix is not None
        assert self._sample_actions_profile_implicit is not None
        assert self._sample_actions_profile_coarse is not None
        assert self._sample_actions_profile_expert is not None

        timing: dict[str, float] = {}

        stage_start = time.monotonic()
        prefix_state = self._sample_actions_profile_prefix(sample_rng, observation)
        _block_until_ready(prefix_state)
        timing["vlm_ms"] = (time.monotonic() - stage_start) * 1000

        stage_start = time.monotonic()
        implicit_outputs = self._sample_actions_profile_implicit(prefix_state)
        _block_until_ready(implicit_outputs)
        timing["implicit_action_reasoner_ms"] = (time.monotonic() - stage_start) * 1000

        coarse_kwargs = {
            key: sample_kwargs[key]
            for key in (
                "coarse_num_steps",
                "dynamic_coarse_steps",
                "explicit_action_reason_override",
                "explicit_action_skip_segment",
            )
            if key in sample_kwargs
        }
        coarse_kwargs["num_steps"] = sample_kwargs.get("num_steps", 10)
        stage_start = time.monotonic()
        coarse_outputs = self._sample_actions_profile_coarse(prefix_state, **coarse_kwargs)
        _block_until_ready(coarse_outputs)
        timing["coarse_action_expert_ms"] = (time.monotonic() - stage_start) * 1000

        stage_start = time.monotonic()
        expert_outputs = self._sample_actions_profile_expert(
            prefix_state,
            coarse_outputs["explicit_action_reason"],
            implicit_outputs["implicit_action_reason"],
            num_steps=sample_kwargs.get("num_steps", 10),
        )
        _block_until_ready(expert_outputs)
        timing["action_expert_ms"] = (time.monotonic() - stage_start) * 1000

        result = dict(expert_outputs)
        if coarse_outputs.get("explicit_action_reason") is not None:
            result["coarse_actions"] = coarse_outputs["explicit_action_reason"]
            result["coarse_num_steps"] = coarse_outputs["coarse_num_steps"]
        return result, timing

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous"]
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)

        if task_name is None:
            return outputs

        print(f"Policy infering for task: {task_name}, with inference time: {outputs['policy_timing']['infer_ms']:.3f} ms")
        if task_name not in task_name_requiring_waist:
            # cut off waist actions for tasks that don't require it
            outputs["actions"] = outputs["actions"][:, :16]

        else:
            raw_state = jax.tree.map(lambda x: x, obs).get("state", None)
            assert raw_state is not None, "State is required for post-processing waist actions"
            # freeze four waist actions to the current state, utilizing only the last action for policy output
            outputs["actions"][:, 16:20] = raw_state[16:20]

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
