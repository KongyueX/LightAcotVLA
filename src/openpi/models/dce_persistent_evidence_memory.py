"""Persistent evidence tokens for frozen ACoT action experts.

The module compresses privileged visual evidence into a small set of learned
memory tokens.  A trainer can prepend those tokens to an EAR or final-action
suffix so the unchanged frozen Gemma expert can read and update the memory at
every transformer layer.  This module contains no action head or post-expert
residual path.
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class DCEPersistentEvidenceMemoryConfig:
    """Shape contract for one low-rank evidence-to-memory projector."""

    evidence_dim: int = 4096
    evidence_tokens: int = 128
    expert_dim: int = 1024
    memory_tokens: int = 8
    attention_dim: int = 128
    num_heads: int = 4

    def __post_init__(self) -> None:
        dimensions = {
            "evidence_dim": self.evidence_dim,
            "evidence_tokens": self.evidence_tokens,
            "expert_dim": self.expert_dim,
            "memory_tokens": self.memory_tokens,
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.attention_dim % self.num_heads:
            raise ValueError(
                "attention_dim must be divisible by num_heads, got "
                f"{self.attention_dim} and {self.num_heads}."
            )


class DCEPersistentEvidenceMemory(nnx.Module):
    """Convert evidence into gated memory tokens with low-rank cross-attention."""

    def __init__(
        self,
        config: DCEPersistentEvidenceMemoryConfig = DCEPersistentEvidenceMemoryConfig(),
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        query_scale = jnp.asarray(config.attention_dim**-0.5, dtype=param_dtype)
        self.memory_queries = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (config.memory_tokens, config.attention_dim),
                dtype=param_dtype,
            )
            * query_scale
        )
        kernel_init = jax.nn.initializers.lecun_normal()
        self.key_proj = nnx.Linear(
            config.evidence_dim,
            config.attention_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.value_proj = nnx.Linear(
            config.evidence_dim,
            config.attention_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.out_proj = nnx.Linear(
            config.attention_dim,
            config.expert_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.gate = nnx.Param(jnp.zeros((), dtype=param_dtype))

    def __call__(self, evidence: jax.Array) -> jax.Array:
        """Return ``[B, memory_tokens, expert_dim]`` persistent suffix tokens."""

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

        batch_size = evidence.shape[0]
        head_dim = self.config.attention_dim // self.config.num_heads
        query = jnp.broadcast_to(
            self.memory_queries.value[None],
            (batch_size, self.config.memory_tokens, self.config.attention_dim),
        ).reshape(
            batch_size,
            self.config.memory_tokens,
            self.config.num_heads,
            head_dim,
        )
        key = self.key_proj(evidence).reshape(
            batch_size,
            self.config.evidence_tokens,
            self.config.num_heads,
            head_dim,
        )
        value = self.value_proj(evidence).reshape(
            batch_size,
            self.config.evidence_tokens,
            self.config.num_heads,
            head_dim,
        )
        logits = jnp.einsum(
            "bqhd,bmhd->bhqm",
            query,
            key,
            preferred_element_type=jnp.float32,
        )
        logits = logits * jnp.asarray(head_dim**-0.5, dtype=logits.dtype)
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqm,bmhd->bqhd", weights, value).reshape(
            batch_size,
            self.config.memory_tokens,
            self.config.attention_dim,
        )
        tokens = self.out_proj(attended).astype(evidence.dtype)
        gate = jnp.tanh(jnp.asarray(self.gate.value, dtype=tokens.dtype))
        return gate * tokens


__all__ = [
    "DCEPersistentEvidenceMemory",
    "DCEPersistentEvidenceMemoryConfig",
]
