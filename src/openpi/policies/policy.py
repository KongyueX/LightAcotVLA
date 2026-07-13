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
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantile_norm: bool = False,
        action_dim: int | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._norm_stats = norm_stats
        self._use_quantile_norm = use_quantile_norm
        self._action_dim = action_dim or model.action_dim
        self._sample_actions_profile_prefix = None
        self._sample_actions_profile_implicit = None
        self._sample_actions_profile_coarse = None
        self._sample_actions_profile_expert = None
        self._sample_actions_batched_mc = None
        self._predict_execution_horizon = None
        if hasattr(model, "sample_actions_batched_mc"):
            self._sample_actions_batched_mc = nnx_utils.module_jit(model.sample_actions_batched_mc)
        if getattr(model, "execution_horizon_predictor_enabled", False):
            self._predict_execution_horizon = nnx_utils.module_jit(model.predict_execution_horizon)
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
        action_cot_denoising_steps = inputs.pop("action_cot_denoising_steps", None)
        action_cot_dynamic_denoising_steps = inputs.pop("action_cot_dynamic_denoising_steps", None)
        batched_mc_samples = int(np.asarray(inputs.pop("batched_mc_samples", 0)).item())
        run_execution_horizon_predictor = _as_bool(
            inputs.pop("run_execution_horizon_predictor", False)
        )
        previous_actions = inputs.pop("execution_horizon_previous_actions", None)
        previous_h = inputs.pop("execution_horizon_previous_h", 1)
        budget_balance = inputs.pop("execution_horizon_budget_balance", 0.0)
        episode_progress = inputs.pop("execution_horizon_episode_progress", 0.0)
        previous_valid = inputs.pop("execution_horizon_previous_valid", previous_actions is not None)
        transformed_coarse_actions_override = None
        if coarse_actions_override is not None:
            override_inputs = jax.tree.map(lambda x: x, obs)
            override_inputs.pop("coarse_actions_override", None)
            override_inputs.pop("policy_seed", None)
            override_inputs.pop("action_cot_skip_segment", None)
            override_inputs.pop("profile_policy_timing", None)
            override_inputs.pop("action_cot_denoising_steps", None)
            override_inputs.pop("action_cot_dynamic_denoising_steps", None)
            override_inputs.pop("batched_mc_samples", None)
            override_inputs.pop("run_execution_horizon_predictor", None)
            override_inputs.pop("execution_horizon_previous_actions", None)
            override_inputs.pop("execution_horizon_previous_h", None)
            override_inputs.pop("execution_horizon_budget_balance", None)
            override_inputs.pop("execution_horizon_episode_progress", None)
            override_inputs.pop("execution_horizon_previous_valid", None)
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
            sample_rngs = jax.random.split(sample_rng, batched_mc_samples) if batched_mc_samples else None
        else:
            policy_seed_int = int(np.asarray(policy_seed).item())
            sample_rng = jax.random.key(policy_seed_int)
            sample_rngs = (
                jax.random.key(
                    np.arange(policy_seed_int, policy_seed_int + batched_mc_samples, dtype=np.uint32)
                )
                if batched_mc_samples
                else None
            )
        outputs = {
            "state": inputs["state"],
            # This key is intentionally not present in norm_stats, so output
            # transforms preserve the exact normalized predictor input.
            "execution_horizon_state_normalized": inputs["state"],
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
        if action_cot_denoising_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "action_cot_denoising_steps": np.asarray(action_cot_denoising_steps, dtype=np.int32).reshape(()),
            }
        if action_cot_dynamic_denoising_steps is not None:
            sample_kwargs = {
                **sample_kwargs,
                "dynamic_denoising_steps": bool(np.asarray(action_cot_dynamic_denoising_steps).item()),
            }
        observation = _model.Observation.from_dict(inputs)
        detailed_timing = {}
        if batched_mc_samples:
            if self._sample_actions_batched_mc is None:
                raise ValueError("The loaded policy does not implement a batched MC teacher.")
            if batched_mc_samples not in (10, 20, 32):
                raise ValueError("batched_mc_samples must be one of 10, 20, or 32.")
            teacher_start = time.monotonic()
            result = self._sample_actions_batched_mc(
                sample_rngs,
                observation,
                num_steps=sample_kwargs.get("num_steps", 10),
                action_cot_denoising_steps=sample_kwargs.get("action_cot_denoising_steps", 10),
            )
            _block_until_ready(result)
            detailed_timing["batched_mc_teacher_ms"] = (time.monotonic() - teacher_start) * 1000
        elif profile_policy_timing and self._can_profile_sample_actions():
            result, detailed_timing = self._profile_sample_actions(sample_rng, observation, sample_kwargs)
        else:
            result = self._sample_actions(sample_rng, observation, **sample_kwargs)

        if isinstance(result, dict):
            if "actions" in result:
                result["execution_horizon_final_actions_normalized"] = result["actions"]
            if "coarse_actions" in result:
                result["execution_horizon_coarse_actions_normalized"] = result["coarse_actions"]

        if run_execution_horizon_predictor:
            if self._predict_execution_horizon is None:
                raise ValueError(
                    "run_execution_horizon_predictor=True requires a V2-P predictor sidecar checkpoint."
                )
            if not isinstance(result, dict):
                raise TypeError("Execution-horizon prediction requires structured action outputs.")
            if "execution_horizon_prefix_feature" not in result:
                raise KeyError("Policy result did not expose the shared prefix feature.")
            normalized_previous_actions = self._normalize_previous_actions(previous_actions)
            predictor_start = time.monotonic()
            predictor_outputs = self._predict_execution_horizon(
                prefix_feature=result["execution_horizon_prefix_feature"],
                state=inputs["state"],
                coarse_actions=result["coarse_actions"],
                final_actions=result["actions"],
                previous_actions=jnp.asarray(normalized_previous_actions)[None, ...],
                previous_h=jnp.asarray(previous_h, dtype=jnp.float32).reshape((1,)),
                budget_balance=jnp.asarray(budget_balance, dtype=jnp.float32).reshape((1,)),
                episode_progress=jnp.asarray(episode_progress, dtype=jnp.float32).reshape((1,)),
                previous_valid=jnp.asarray(previous_valid, dtype=jnp.bool_).reshape((1,)),
            )
            _block_until_ready(predictor_outputs)
            detailed_timing["execution_horizon_predictor_ms"] = (
                time.monotonic() - predictor_start
            ) * 1000
            result.update({f"execution_horizon_{key}": value for key, value in predictor_outputs.items()})

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
                    "batched_mc_teacher_ms",
                    "execution_horizon_predictor_ms",
                )
            )
            detailed_timing["profile_overhead_ms"] = max(0.0, model_time * 1000 - stage_total_ms)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            **detailed_timing,
        }
        return self.post_process(obs, outputs)

    def _normalize_previous_actions(self, previous_actions: Any) -> np.ndarray:
        if previous_actions is None:
            return np.zeros((10, self._action_dim), dtype=np.float32)
        actions = np.asarray(previous_actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"execution_horizon_previous_actions must be rank 2, got {actions.shape}.")
        if self._norm_stats is not None and "actions" in self._norm_stats:
            stats = self._norm_stats["actions"]
            dim = min(actions.shape[-1], np.asarray(stats.mean).shape[-1])
            if self._use_quantile_norm:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError("Quantile normalization requested but action q01/q99 are unavailable.")
                q01 = np.asarray(stats.q01)[..., :dim]
                q99 = np.asarray(stats.q99)[..., :dim]
                actions[..., :dim] = (actions[..., :dim] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                mean = np.asarray(stats.mean)[..., :dim]
                std = np.asarray(stats.std)[..., :dim]
                actions[..., :dim] = (actions[..., :dim] - mean) / (std + 1e-6)
        actions = actions[:10]
        if actions.shape[0] < 10:
            actions = np.pad(actions, ((0, 10 - actions.shape[0]), (0, 0)))
        if actions.shape[-1] < self._action_dim:
            actions = np.pad(actions, ((0, 0), (0, self._action_dim - actions.shape[-1])))
        return actions[:, : self._action_dim]

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
                "action_cot_denoising_steps",
                "dynamic_denoising_steps",
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
            result["action_cot_denoising_steps"] = coarse_outputs["action_cot_denoising_steps"]
        if "execution_horizon_prefix_feature" in prefix_state:
            result["execution_horizon_prefix_feature"] = prefix_state[
                "execution_horizon_prefix_feature"
            ]
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
