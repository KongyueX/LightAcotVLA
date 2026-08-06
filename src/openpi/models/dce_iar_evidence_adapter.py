"""Evidence-conditioned residual adapter for frozen ACoT IAR tokens.

The adapter is deliberately independent of the ACoT-VLA backbone and contains
no action head.  It refines the frozen implicit-action-reasoner output with one
shared low-rank cross-attention and one zero-initialized ReZero gate per Gemma
layer.  Bias-free projections make zero evidence an exact no-op even after the
gates have been trained.
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class DCEIAREvidenceAdapterConfig:
    """Fixed-shape contract for the hierarchical IAR evidence adapter."""

    query_dim: int = 1024
    query_tokens: int = 18
    evidence_dim: int = 4096
    evidence_tokens: int = 128
    attention_dim: int = 128
    num_heads: int = 4

    def __post_init__(self) -> None:
        dimensions = {
            "query_dim": self.query_dim,
            "query_tokens": self.query_tokens,
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


class DCEIAREvidenceAdapter(nnx.Module):
    """Inject current visual evidence into the 18 frozen IAR layer tokens.

    Inputs follow the deployed downsample-extractor contract:

    * ``query``: ``[B, 18, 1024]`` frozen IAR tokens;
    * ``evidence``: ``[B, 128, 4096]`` selected current/delta evidence.

    Q/K/V/O parameters are shared across all IAR layers.  The residual strength
    is layer-specific, so each of the 18 output tokens receives its own scalar
    ``tanh``-bounded ReZero gate.
    """

    def __init__(
        self,
        config: DCEIAREvidenceAdapterConfig = DCEIAREvidenceAdapterConfig(),
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        kernel_init = jax.nn.initializers.lecun_normal()
        self.q_proj = nnx.Linear(
            config.query_dim,
            config.attention_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.k_proj = nnx.Linear(
            config.evidence_dim,
            config.attention_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.v_proj = nnx.Linear(
            config.evidence_dim,
            config.attention_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.out_proj = nnx.Linear(
            config.attention_dim,
            config.query_dim,
            use_bias=False,
            kernel_init=kernel_init,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.gates = nnx.Param(
            jnp.zeros((config.query_tokens,), dtype=param_dtype)
        )

    def __call__(self, query: jax.Array, evidence: jax.Array) -> jax.Array:
        """Return evidence-refined IAR tokens with the same shape as ``query``."""

        if query.ndim != 3:
            raise ValueError(f"query must have shape [B,T,D], got {query.shape}.")
        if evidence.ndim != 3:
            raise ValueError(f"evidence must have shape [B,M,D], got {evidence.shape}.")
        if query.shape[0] != evidence.shape[0]:
            raise ValueError(
                "query and evidence batch dimensions must match, got "
                f"{query.shape[0]} and {evidence.shape[0]}."
            )
        expected_query_shape = (self.config.query_tokens, self.config.query_dim)
        if query.shape[1:] != expected_query_shape:
            raise ValueError(
                "query trailing shape must be "
                f"{expected_query_shape}, got {query.shape[1:]}."
            )
        expected_evidence_shape = (
            self.config.evidence_tokens,
            self.config.evidence_dim,
        )
        if evidence.shape[1:] != expected_evidence_shape:
            raise ValueError(
                "evidence trailing shape must be "
                f"{expected_evidence_shape}, got {evidence.shape[1:]}."
            )

        batch_size = query.shape[0]
        head_dim = self.config.attention_dim // self.config.num_heads
        projected_query = self.q_proj(query).reshape(
            batch_size,
            self.config.query_tokens,
            self.config.num_heads,
            head_dim,
        )
        projected_key = self.k_proj(evidence).reshape(
            batch_size,
            self.config.evidence_tokens,
            self.config.num_heads,
            head_dim,
        )
        projected_value = self.v_proj(evidence).reshape(
            batch_size,
            self.config.evidence_tokens,
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
        attended = attended.reshape(
            batch_size,
            self.config.query_tokens,
            self.config.attention_dim,
        )
        delta = self.out_proj(attended).astype(query.dtype)
        gates = jnp.tanh(jnp.asarray(self.gates.value, dtype=query.dtype))
        return query + gates[None, :, None] * delta


__all__ = ["DCEIAREvidenceAdapter", "DCEIAREvidenceAdapterConfig"]
