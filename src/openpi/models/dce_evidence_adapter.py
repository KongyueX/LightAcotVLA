"""ReZero-gated current-evidence adapters for dual-path ACoT reasoning.

This module deliberately contains no action head and has no dependency on the
ACoT-VLA backbone.  A trainer can place the EAR and final adapters before the
corresponding frozen action-expert stacks and optimize only these parameters.
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class DCEEvidenceAdapterConfig:
    """Fixed-shape contract for one evidence cross-attention adapter."""

    query_dim: int = 1024
    evidence_dim: int = 2048
    evidence_tokens: int = 16
    attention_dim: int = 128
    num_heads: int = 4

    def __post_init__(self) -> None:
        dimensions = {
            "query_dim": self.query_dim,
            "evidence_dim": self.evidence_dim,
            "evidence_tokens": self.evidence_tokens,
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


class DCEEvidenceAdapter(nnx.Module):
    """Fuse fixed current-evidence tokens into plan-conditioned query tokens.

    The Q/K/V/O kernels use their ordinary non-zero initialization.  Only the
    scalar ReZero gate starts at zero, so construction exactly preserves the
    input query while still giving the gate a gradient on the first update.
    """

    def __init__(
        self,
        config: DCEEvidenceAdapterConfig = DCEEvidenceAdapterConfig(),
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        kernel_init = jax.nn.initializers.lecun_normal()
        self.q_proj = nnx.Linear(
            config.query_dim,
            config.attention_dim,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.k_proj = nnx.Linear(
            config.evidence_dim,
            config.attention_dim,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.v_proj = nnx.Linear(
            config.evidence_dim,
            config.attention_dim,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.out_proj = nnx.Linear(
            config.attention_dim,
            config.query_dim,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.gate = nnx.Param(jnp.zeros((), dtype=param_dtype))

    def __call__(self, query: jax.Array, evidence: jax.Array) -> jax.Array:
        """Return ``query + tanh(gate) * CrossAttention(query, evidence)``."""

        if query.ndim != 3:
            raise ValueError(f"query must have shape [B,T,D], got {query.shape}.")
        if evidence.ndim != 3:
            raise ValueError(f"evidence must have shape [B,M,D], got {evidence.shape}.")
        if query.shape[0] != evidence.shape[0]:
            raise ValueError(
                "query and evidence batch dimensions must match, got "
                f"{query.shape[0]} and {evidence.shape[0]}."
            )
        if query.shape[-1] != self.config.query_dim:
            raise ValueError(
                f"query width must be {self.config.query_dim}, got {query.shape[-1]}."
            )
        if evidence.shape[1] != self.config.evidence_tokens:
            raise ValueError(
                "evidence token count must be "
                f"{self.config.evidence_tokens}, got {evidence.shape[1]}."
            )
        if evidence.shape[-1] != self.config.evidence_dim:
            raise ValueError(
                f"evidence width must be {self.config.evidence_dim}, got {evidence.shape[-1]}."
            )

        batch_size, query_tokens = query.shape[:2]
        evidence_tokens = evidence.shape[1]
        head_dim = self.config.attention_dim // self.config.num_heads

        projected_query = self.q_proj(query).reshape(
            batch_size,
            query_tokens,
            self.config.num_heads,
            head_dim,
        )
        projected_key = self.k_proj(evidence).reshape(
            batch_size,
            evidence_tokens,
            self.config.num_heads,
            head_dim,
        )
        projected_value = self.v_proj(evidence).reshape(
            batch_size,
            evidence_tokens,
            self.config.num_heads,
            head_dim,
        )

        logits = jnp.einsum(
            "bthd,bmhd->bhtm",
            projected_query,
            projected_key,
            preferred_element_type=jnp.float32,
        )
        logits = logits * jnp.asarray(head_dim**-0.5, dtype=logits.dtype)
        weights = jax.nn.softmax(logits, axis=-1).astype(projected_value.dtype)
        attended = jnp.einsum("bhtm,bmhd->bthd", weights, projected_value)
        attended = attended.reshape(batch_size, query_tokens, self.config.attention_dim)
        delta = self.out_proj(attended).astype(query.dtype)
        gate = jnp.tanh(jnp.asarray(self.gate.value, dtype=query.dtype))
        return query + gate * delta


class DCEDualPathEvidenceAdapters(nnx.Module):
    """Independent EAR and final adapters under one trainer-friendly tree."""

    def __init__(
        self,
        config: DCEEvidenceAdapterConfig = DCEEvidenceAdapterConfig(),
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.ear = DCEEvidenceAdapter(config, rngs=rngs, param_dtype=param_dtype)
        self.final = DCEEvidenceAdapter(config, rngs=rngs, param_dtype=param_dtype)


__all__ = [
    "DCEDualPathEvidenceAdapters",
    "DCEEvidenceAdapter",
    "DCEEvidenceAdapterConfig",
]
