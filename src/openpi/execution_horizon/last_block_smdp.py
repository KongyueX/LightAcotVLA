"""Trainable final predictor block with a detached dual-value critic."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.execution_horizon.ordered_smdp import DEFAULT_CANDIDATES
from openpi.execution_horizon.ordered_smdp import ordered_log_probabilities
from openpi.models import execution_horizon_predictor as predictor_lib
from openpi.models import model as model_lib


ACTOR_ROOTS = ("last_block", "summary_proj", "raw_h_ordinal_head")
CRITIC_ROOTS = ("critic_in", "critic_out")


class LastBlockHorizonModel(nnx.Module):
    def __init__(self, config: predictor_lib.ExecutionHorizonPredictorConfig, *, rngs: nnx.Rngs) -> None:
        if config.temporal_backbone != "transformer" or not config.ordered_continuation_head:
            raise ValueError("Last-block optimization requires an ordered Transformer predictor.")
        if config.ordered_readout != "global" or config.candidate_horizons != DEFAULT_CANDIDATES:
            raise ValueError("Last-block optimization requires the global H5/10/15/20/25 predictor.")
        self.config = config
        h = config.hidden_dim
        self.last_block = predictor_lib._TransformerBlock(  # noqa: SLF001
            h, config.num_heads, config.feed_forward_multiplier, rngs=rngs, param_dtype=jnp.float32
        )
        self.summary_proj = nnx.Linear(2 * h, h, rngs=rngs, param_dtype=jnp.float32)
        self.raw_h_ordinal_head = nnx.Linear(h, config.num_candidates - 1, rngs=rngs, param_dtype=jnp.float32)
        self.critic_in = nnx.Linear(
            h, 64, rngs=rngs, param_dtype=jnp.float32, kernel_init=jax.nn.initializers.normal(0.02)
        )
        self.critic_out = nnx.Linear(
            64, 2, rngs=rngs, param_dtype=jnp.float32, kernel_init=jax.nn.initializers.zeros
        )
        self.critic_out.bias.value = jnp.asarray([np.log(9.0), np.log(np.expm1(3.0))], dtype=jnp.float32)

    def value_outputs(self, summary: jax.Array) -> dict[str, jax.Array]:
        hidden = jnp.tanh(self.critic_in(jax.lax.stop_gradient(summary)))
        values = self.critic_out(hidden)
        return {
            "success_logits": values[..., 0],
            "success_value": jax.nn.sigmoid(values[..., 0]),
            "cost_value": jax.nn.softplus(values[..., 1]),
        }

    def __call__(self, tokens: jax.Array, context: jax.Array) -> dict[str, jax.Array]:
        tokens = self.last_block(jnp.asarray(tokens, dtype=jnp.float32))
        context = jnp.asarray(context, dtype=jnp.float32)
        temporal_summary = jnp.mean(tokens[:, -self.config.action_horizon :], axis=1)
        summary = nnx.swish(self.summary_proj(jnp.concatenate([temporal_summary, context], axis=-1)))
        logits = self.raw_h_ordinal_head(summary)
        log_probabilities, probabilities = predictor_lib.ordered_continuation_distribution(logits)
        return {
            "continuation_logits": logits,
            "log_probabilities": log_probabilities,
            "probabilities": probabilities,
            "summary": summary,
            **self.value_outputs(summary),
        }


def apply_last_block(
    graphdef: Any, params: nnx.State, tokens: jax.Array, context: jax.Array
) -> dict[str, jax.Array]:
    return nnx.merge(graphdef, params)(tokens, context)


def apply_critic(graphdef: Any, params: nnx.State, summary: jax.Array) -> dict[str, jax.Array]:
    return nnx.merge(graphdef, params).value_outputs(summary)


def _replace_state(module: nnx.Module, loaded: dict[str, Any]) -> nnx.State:
    state = nnx.state(module)
    expected = traverse_util.flatten_dict(state.to_pure_dict())
    actual = traverse_util.flatten_dict(loaded)
    if expected.keys() != actual.keys() or any(
        np.shape(actual[key]) != np.shape(value) for key, value in expected.items()
    ):
        raise ValueError("Checkpoint parameter tree or shapes do not match the predictor module.")
    state.replace_by_pure_dict(loaded)
    return state


@dataclasses.dataclass
class LastBlockHorizonSelector:
    graphdef: Any
    params: nnx.State
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.metadata = json.loads(json.dumps(self.metadata))
        self.candidates = tuple(self.metadata["candidate_horizons"])
        self.config = predictor_lib.ExecutionHorizonPredictorConfig(**self.metadata["predictor_config"])
        if self.candidates != self.config.candidate_horizons:
            raise ValueError("Checkpoint candidates and predictor config differ.")
        graphdef = self.graphdef
        self._inference = jax.jit(
            lambda params, tokens, context: apply_last_block(graphdef, params, tokens, context), backend="cpu"
        )

    @classmethod
    def initialize_from_predictor(cls, anchor_dir: str | pathlib.Path, seed: int = 7) -> LastBlockHorizonSelector:
        anchor_dir = pathlib.Path(anchor_dir).resolve()
        config = predictor_lib.ExecutionHorizonPredictorConfig(
            **json.loads((anchor_dir / "predictor_config.json").read_text())
        )
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            partial_model = LastBlockHorizonModel(config, rngs=nnx.Rngs(seed))
            predictor = predictor_lib.ExecutionHorizonPredictor(config, rngs=nnx.Rngs(seed))
            loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(
                anchor_dir / "params", dtype=jnp.float32, sharding=jax.sharding.SingleDeviceSharding(cpu)
            ))
            loaded = loaded.get("execution_horizon_predictor", loaded)
            predictor = nnx.merge(nnx.graphdef(predictor), _replace_state(predictor, loaded))
            state = nnx.state(partial_model).to_pure_dict()
            state["last_block"] = nnx.state(predictor.temporal_layers[-1]).to_pure_dict()
            state["summary_proj"] = nnx.state(predictor.summary_proj).to_pure_dict()
            state["raw_h_ordinal_head"] = nnx.state(predictor.raw_h_ordinal_head).to_pure_dict()
            params = _replace_state(partial_model, state)
        return cls(
            nnx.graphdef(partial_model), params,
            {
                "candidate_horizons": list(config.candidate_horizons),
                "predictor_config": dataclasses.asdict(config),
                "anchor_predictor_dir": str(anchor_dir), "seed": seed, "cost_multiplier": 0.02,
                "actor_roots": list(ACTOR_ROOTS), "critic_roots": list(CRITIC_ROOTS),
                "actor_changed": False,
            },
        )

    def forward(self, tokens: np.ndarray, context: np.ndarray) -> dict[str, jax.Array]:
        tokens = np.asarray(tokens, dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)
        if tokens.ndim == 2:
            tokens = tokens[None]
        if context.ndim == 1:
            context = context[None]
        expected_sequence = self.config.action_horizon + self.config.visual_num_queries
        if (
            tokens.ndim != 3 or tokens.shape[1:] != (expected_sequence, self.config.hidden_dim)
            or context.shape != (tokens.shape[0], self.config.hidden_dim)
            or not np.all(np.isfinite(tokens)) or not np.all(np.isfinite(context))
        ):
            raise ValueError("Last-block tokens/context must have finite matching batch, sequence, and hidden dimensions.")
        return self._inference(self.params, tokens, context)

    def decide(
        self, policy_outputs: Mapping[str, Any], *, sample: bool = False, rng: np.random.Generator | None = None
    ) -> tuple[int, dict[str, Any]]:
        candidates = np.asarray(policy_outputs["execution_horizon_candidate_horizons"]).reshape(-1)
        if not np.array_equal(candidates, self.candidates):
            raise ValueError("Policy candidates differ from the last-block selector checkpoint.")
        tokens = np.asarray(policy_outputs["execution_horizon_last_block_input"], dtype=np.float32)
        context = np.asarray(policy_outputs["execution_horizon_last_block_context"], dtype=np.float32)
        prediction = jax.device_get(self.forward(tokens, context))
        if prediction["probabilities"].shape[0] != 1:
            raise ValueError("Selector decisions require exactly one policy observation.")
        probabilities = np.asarray(prediction["probabilities"][0], dtype=np.float64)
        probabilities /= probabilities.sum()
        if sample:
            rng = rng if rng is not None else np.random.default_rng()
            index = int(rng.choice(len(self.candidates), p=probabilities))
        else:
            index = int(np.argmax(probabilities))
        anchor_logits = np.asarray(
            policy_outputs["execution_horizon_ordered_continuation_logits"], dtype=np.float32
        ).reshape(-1)
        if anchor_logits.shape != (len(self.candidates) - 1,):
            raise ValueError("Anchor logits must have one value per candidate transition.")
        anchor_index = int(np.argmax(ordered_log_probabilities(anchor_logits)))
        return self.candidates[index], {
            "smdp_action_index": index,
            "smdp_old_log_prob": float(np.log(probabilities[index])),
            "smdp_probabilities": probabilities.tolist(),
            "smdp_success_value": float(prediction["success_value"][0]),
            "smdp_cost_value": float(prediction["cost_value"][0]),
            "smdp_anchor_logits": anchor_logits.tolist(),
            "smdp_feature": prediction["summary"][0].tolist(),
            "smdp_last_block_input": tokens.reshape((-1, self.config.hidden_dim)).tolist(),
            "smdp_last_block_context": context.reshape(-1).tolist(),
            "raw_horizon": self.candidates[anchor_index],
            "selector_policy": "ordered_smdp_last_block",
        }

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path).resolve()
        if (path / "params").exists() or (path / "metadata.json").exists():
            raise FileExistsError(f"Last-block checkpoint already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(path / "params", {"params": self.params.to_pure_dict()})
        (path / "metadata.json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> LastBlockHorizonSelector:
        path = pathlib.Path(path).resolve()
        metadata = json.loads((path / "metadata.json").read_text())
        config = predictor_lib.ExecutionHorizonPredictorConfig(**metadata["predictor_config"])
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            module = LastBlockHorizonModel(config, rngs=nnx.Rngs(int(metadata["seed"])))
            loaded = model_lib.convert_str_keys_to_int(model_lib.restore_params(
                path / "params", dtype=jnp.float32, sharding=jax.sharding.SingleDeviceSharding(cpu)
            ))
            params = _replace_state(module, loaded)
        return cls(nnx.graphdef(module), params, metadata)
