"""Shared gated evidence cross-attention for selected frozen Gemma layers."""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class DCEMultiLayerEvidenceAdapterConfig:
    """Shape and layer contract for one shared four-layer residual adapter."""

    evidence_dim: int = 4096
    evidence_tokens: int = 128
    expert_dim: int = 1024
    attention_dim: int = 128
    num_heads: int = 4
    expert_depth: int = 18
    injection_layers: tuple[int, ...] = (0, 5, 11, 17)

    def __post_init__(self) -> None:
        dimensions = {
            "evidence_dim": self.evidence_dim,
            "evidence_tokens": self.evidence_tokens,
            "expert_dim": self.expert_dim,
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
            "expert_depth": self.expert_depth,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.attention_dim % self.num_heads:
            raise ValueError(
                "attention_dim must be divisible by num_heads, got "
                f"{self.attention_dim} and {self.num_heads}."
            )
        if len(self.injection_layers) != 4:
            raise ValueError("Exactly four injection layers are required.")
        if tuple(sorted(set(self.injection_layers))) != self.injection_layers:
            raise ValueError("injection_layers must be sorted and unique.")
        if self.injection_layers[0] < 0 or self.injection_layers[-1] >= self.expert_depth:
            raise ValueError(
                f"injection_layers must lie in [0, {self.expert_depth}), "
                f"got {self.injection_layers}."
            )


def _scaled_normal(
    rng: jax.Array,
    shape: tuple[int, ...],
    fan_in: int,
    dtype: jnp.dtype,
) -> jax.Array:
    scale = jnp.asarray(fan_in**-0.5, dtype=dtype)
    return jax.random.normal(rng, shape, dtype=dtype) * scale


class DCEMultiLayerEvidenceAdapter(nnx.Module):
    """Own shared CA kernels and four independent zero-initialized gates."""

    def __init__(
        self,
        config: DCEMultiLayerEvidenceAdapterConfig = DCEMultiLayerEvidenceAdapterConfig(),
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        head_dim = config.attention_dim // config.num_heads
        self.query_kernel = nnx.Param(
            _scaled_normal(
                rngs.params(),
                (config.expert_dim, config.num_heads, head_dim),
                config.expert_dim,
                param_dtype,
            )
        )
        self.key_kernel = nnx.Param(
            _scaled_normal(
                rngs.params(),
                (config.evidence_dim, config.num_heads, head_dim),
                config.evidence_dim,
                param_dtype,
            )
        )
        self.value_kernel = nnx.Param(
            _scaled_normal(
                rngs.params(),
                (config.evidence_dim, config.num_heads, head_dim),
                config.evidence_dim,
                param_dtype,
            )
        )
        self.output_kernel = nnx.Param(
            _scaled_normal(
                rngs.params(),
                (config.num_heads, head_dim, config.expert_dim),
                config.attention_dim,
                param_dtype,
            )
        )
        self.layer_gates = nnx.Param(
            jnp.zeros((len(config.injection_layers),), dtype=param_dtype)
        )

    def hook_payload(self, evidence: jax.Array) -> dict[str, jax.Array]:
        """Return the external payload consumed by the opt-in Gemma scan hook."""

        if evidence.ndim != 3:
            raise ValueError(f"evidence must have shape [B,M,D], got {evidence.shape}.")
        if evidence.shape[1] != self.config.evidence_tokens:
            raise ValueError(
                f"evidence token count must be {self.config.evidence_tokens}, "
                f"got {evidence.shape[1]}."
            )
        if evidence.shape[-1] != self.config.evidence_dim:
            raise ValueError(
                f"evidence width must be {self.config.evidence_dim}, "
                f"got {evidence.shape[-1]}."
            )
        layer_ids = jnp.asarray(self.config.injection_layers, dtype=jnp.int32)
        layer_gates = jnp.zeros((self.config.expert_depth,), dtype=jnp.float32)
        layer_gates = layer_gates.at[layer_ids].set(
            jnp.tanh(self.layer_gates.value.astype(jnp.float32))
        )
        layer_active = jnp.zeros((self.config.expert_depth,), dtype=jnp.bool_)
        layer_active = layer_active.at[layer_ids].set(True)
        return {
            "evidence": evidence,
            "query_kernel": self.query_kernel.value,
            "key_kernel": self.key_kernel.value,
            "value_kernel": self.value_kernel.value,
            "output_kernel": self.output_kernel.value,
            "layer_gates": layer_gates,
            "layer_active": layer_active,
        }


__all__ = [
    "DCEMultiLayerEvidenceAdapter",
    "DCEMultiLayerEvidenceAdapterConfig",
]
