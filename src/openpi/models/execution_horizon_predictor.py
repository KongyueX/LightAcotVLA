"""Single-call execution-horizon predictor used by Budgeted Event V2-P.

This module is intentionally independent from ``action_cot_step_head``. The
latter predicts flow-denoising iterations, while this module predicts how many
environment actions should be executed before the next policy call. The legacy
H1-H10 local MLP remains checkpoint-compatible; the opt-in transformer path
supports an arbitrary ordered candidate set within the generated action chunk.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.scipy as jsp


@dataclasses.dataclass(frozen=True)
class ExecutionHorizonPredictorConfig:
    prefix_feature_dim: int = 2048
    state_dim: int = 32
    action_dim: int = 32
    physical_action_dim: int = 7
    coarse_horizon: int = 15
    action_horizon: int = 10
    hidden_dim: int = 256
    temporal_layers: int = 3
    temporal_backbone: Literal["local_mlp", "transformer"] = "local_mlp"
    num_heads: int = 4
    feed_forward_multiplier: int = 4
    candidate_horizons: tuple[int, ...] = tuple(range(1, 11))
    reference_horizon: int = 10
    coarse_stride: int = 2
    final_stride: int = 1
    visual_num_queries: int = 0
    paired_advantage_heads: bool = False
    paired_distribution_heads: bool = False
    ordered_continuation_head: bool = False
    remaining_calls_scale: float = 64.0
    remaining_steps_scale: float = 512.0
    elapsed_advantage_scale: float = 1.0
    calls_advantage_scale: float = 1.0

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_horizons)
        object.__setattr__(self, "candidate_horizons", candidates)
        if self.temporal_backbone not in {"local_mlp", "transformer"}:
            raise ValueError("temporal_backbone must be local_mlp or transformer.")
        for name in (
            "prefix_feature_dim",
            "state_dim",
            "action_dim",
            "coarse_horizon",
            "action_horizon",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not candidates or candidates != tuple(sorted(set(candidates))):
            raise ValueError("candidate_horizons must be non-empty, sorted, and unique.")
        if len(candidates) < 2:
            raise ValueError("candidate_horizons must contain at least two choices.")
        if candidates[0] <= 0 or candidates[-1] > self.action_horizon:
            raise ValueError(
                "candidate_horizons must be positive and no larger than action_horizon; "
                f"got candidates={candidates}, action_horizon={self.action_horizon}."
            )
        if self.reference_horizon not in candidates:
            raise ValueError("reference_horizon must be present in candidate_horizons.")
        if self.temporal_backbone == "local_mlp":
            if self.action_horizon != 10:
                raise ValueError("Legacy local_mlp checkpoints require action_horizon=10.")
            if self.coarse_horizon < self.action_horizon:
                raise ValueError("Legacy local_mlp requires coarse_horizon >= action_horizon.")
            if candidates != tuple(range(1, self.action_horizon + 1)):
                raise ValueError("Legacy local_mlp requires contiguous H1-H10 candidates.")
            if self.visual_num_queries:
                raise ValueError("Learned visual queries are supported only by the transformer backbone.")
        if self.hidden_dim <= 0 or self.temporal_layers <= 0:
            raise ValueError("hidden_dim and temporal_layers must be positive.")
        if not 1 <= self.physical_action_dim <= self.action_dim:
            raise ValueError("physical_action_dim must lie in [1, action_dim].")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim.")
        if self.feed_forward_multiplier <= 0:
            raise ValueError("feed_forward_multiplier must be positive.")
        if self.coarse_stride <= 0 or self.final_stride <= 0:
            raise ValueError("coarse_stride and final_stride must be positive.")
        if self.visual_num_queries < 0:
            raise ValueError("visual_num_queries must be non-negative.")
        if self.visual_num_queries and not 4 <= self.visual_num_queries <= 8:
            raise ValueError("visual_num_queries must be zero (disabled) or lie in [4, 8].")
        if self.paired_advantage_heads and self.temporal_backbone != "transformer":
            raise ValueError("paired_advantage_heads are supported only by the transformer backbone.")
        if self.paired_distribution_heads and self.temporal_backbone != "transformer":
            raise ValueError("paired_distribution_heads are supported only by the transformer backbone.")
        if self.ordered_continuation_head and self.temporal_backbone != "transformer":
            raise ValueError("ordered_continuation_head is supported only by the transformer backbone.")
        if self.paired_advantage_heads and self.paired_distribution_heads:
            raise ValueError("paired_advantage_heads and paired_distribution_heads are mutually exclusive.")
        if not math.isfinite(self.elapsed_advantage_scale) or self.elapsed_advantage_scale <= 0:
            raise ValueError("elapsed_advantage_scale must be finite and positive.")
        if not math.isfinite(self.calls_advantage_scale) or self.calls_advantage_scale <= 0:
            raise ValueError("calls_advantage_scale must be finite and positive.")
        if self.remaining_calls_scale <= 0 or self.remaining_steps_scale <= 0:
            raise ValueError("remaining_calls_scale and remaining_steps_scale must be positive.")
        if self.temporal_backbone == "transformer" and not self.long_horizons:
            raise ValueError("Transformer hierarchy requires at least one horizon above reference_horizon.")

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_horizons)

    @property
    def long_horizons(self) -> tuple[int, ...]:
        return tuple(value for value in self.candidate_horizons if value > self.reference_horizon)


@dataclasses.dataclass(frozen=True)
class ExecutionHorizonLossWeights:
    success: float = 1.0
    timeout: float = 0.5
    remaining_calls: float = 0.25
    remaining_steps: float = 0.25
    final_risk: float = 0.5
    action_cot_risk: float = 0.5
    fused_risk: float = 1.0
    event: float = 0.5
    raw_h_classification: float = 0.5
    raw_h_ordinal: float = 0.25
    survival: float = 0.0
    success_advantage: float = 0.0
    elapsed_advantage: float = 0.0
    calls_advantage: float = 0.0
    false_long: float = 0.0
    danger_rescue: float = 0.0
    paired_elapsed: float = 0.0
    faster_long: float = 0.0
    ordered_listwise: float = 0.0


DEFAULT_LOSS_WEIGHTS = ExecutionHorizonLossWeights()


@dataclasses.dataclass(frozen=True)
class ExecutionHorizonLabelWeights:
    """Class/region weights for imbalanced counterfactual supervision.

    These multipliers are deliberately separate from the task-level loss
    coefficients above: changing them rebalances labels *within* a head while
    preserving the configured contribution of that head to the total loss.
    """

    success_failure: float = 1.0
    timeout_positive: float = 1.0
    event_positive: float = 1.0
    risk_event: float = 1.0

    def __post_init__(self) -> None:
        for name, value in dataclasses.asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


DEFAULT_LABEL_WEIGHTS = ExecutionHorizonLabelWeights()


def _bce_with_logits(logits: jax.Array, labels: jax.Array) -> jax.Array:
    labels = labels.astype(logits.dtype)
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _masked_mean(values: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    if mask is None:
        return jnp.mean(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _huber(values: jax.Array, delta: float = 1.0) -> jax.Array:
    absolute = jnp.abs(values)
    quadratic = jnp.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _binomial_nll(logits: jax.Array, success_count: jax.Array, trial_count: jax.Array) -> jax.Array:
    success_count = jnp.asarray(success_count, dtype=logits.dtype)
    trial_count = jnp.asarray(trial_count, dtype=logits.dtype)
    return trial_count * jax.nn.softplus(logits) - success_count * logits


def ordered_continuation_distribution(
    continuation_logits: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Convert ordered continue decisions into a categorical horizon distribution.

    For ``M`` ordered horizon candidates, logit ``i`` decides whether execution
    continues from candidate ``i`` to candidate ``i + 1``.  The resulting
    categorical distribution therefore respects the shared-prefix structure of
    action chunks instead of treating candidate horizons as unrelated classes.
    """

    continuation_logits = jnp.asarray(continuation_logits)
    if continuation_logits.shape[-1] < 1:
        raise ValueError("ordered continuation requires at least two horizon candidates.")
    log_continue = -jax.nn.softplus(-continuation_logits)
    log_stop = -jax.nn.softplus(continuation_logits)
    log_prefix = jnp.concatenate(
        [
            jnp.zeros_like(continuation_logits[..., :1]),
            jnp.cumsum(log_continue, axis=-1),
        ],
        axis=-1,
    )
    log_probability = jnp.concatenate(
        [log_prefix[..., :-1] + log_stop, log_prefix[..., -1:]],
        axis=-1,
    )
    log_probability -= jsp.special.logsumexp(log_probability, axis=-1, keepdims=True)
    return log_probability, jnp.exp(log_probability)


def success_first_listwise_target(
    success_count: jax.Array,
    trial_count: jax.Array,
    elapsed_mean: jax.Array,
    valid: jax.Array,
    *,
    elapsed_temperature: float,
) -> tuple[jax.Array, jax.Array]:
    """Build a lexicographic target: best success rate, then lower elapsed time.

    Candidates below the root's best observed success rate receive no target
    mass.  Among candidates tied at the best success rate, elapsed time defines
    a soft listwise target after root-local range normalization.
    """

    if not math.isfinite(elapsed_temperature) or elapsed_temperature <= 0:
        raise ValueError("elapsed_temperature must be finite and positive.")
    success_count = jnp.asarray(success_count, dtype=jnp.float32)
    trial_count = jnp.asarray(trial_count, dtype=jnp.float32)
    elapsed_mean = jnp.asarray(elapsed_mean, dtype=jnp.float32)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    if success_count.shape != trial_count.shape or success_count.shape != elapsed_mean.shape:
        raise ValueError("success_count, trial_count, and elapsed_mean must have identical shapes.")
    valid = jnp.broadcast_to(valid, success_count.shape)
    valid &= trial_count > 0
    valid &= jnp.isfinite(success_count) & jnp.isfinite(trial_count) & jnp.isfinite(elapsed_mean)
    success_rate = success_count / jnp.maximum(trial_count, 1.0)
    best_success = jnp.max(jnp.where(valid, success_rate, -jnp.inf), axis=-1, keepdims=True)
    best_mask = valid & jnp.isclose(success_rate, best_success, rtol=0.0, atol=1e-6)
    root_valid = jnp.any(best_mask, axis=-1)

    best_elapsed_min = jnp.min(jnp.where(best_mask, elapsed_mean, jnp.inf), axis=-1, keepdims=True)
    best_elapsed_max = jnp.max(jnp.where(best_mask, elapsed_mean, -jnp.inf), axis=-1, keepdims=True)
    best_elapsed_min = jnp.where(root_valid[..., None], best_elapsed_min, 0.0)
    best_elapsed_max = jnp.where(root_valid[..., None], best_elapsed_max, best_elapsed_min)
    elapsed_range = best_elapsed_max - best_elapsed_min
    normalized_elapsed = jnp.where(
        elapsed_range > 1e-6,
        (elapsed_mean - best_elapsed_min) / jnp.maximum(elapsed_range, 1e-6),
        0.0,
    )
    target_logits = jnp.where(best_mask, -normalized_elapsed / elapsed_temperature, -1e30)
    target = jax.nn.softmax(target_logits, axis=-1)
    target = jnp.where(root_valid[..., None], target, 0.0)
    return target, root_valid


def _gaussian_nll(
    error: jax.Array,
    log_scale: jax.Array,
    observation_variance: jax.Array | None = None,
) -> jax.Array:
    log_scale = jnp.clip(jnp.asarray(log_scale, dtype=error.dtype), -7.0, 7.0)
    variance = jnp.exp(2.0 * log_scale)
    if observation_variance is not None:
        variance += jnp.maximum(jnp.asarray(observation_variance, dtype=error.dtype), 0.0)
    return 0.5 * jnp.square(error) / jnp.maximum(variance, 1e-12) + 0.5 * jnp.log(jnp.maximum(variance, 1e-12))


def _student_t_nll(
    error: jax.Array,
    scale: jax.Array,
    *,
    degrees_of_freedom: float = 4.0,
) -> jax.Array:
    """Negative log likelihood for a location-scale Student-t distribution."""

    scale = jnp.maximum(jnp.asarray(scale, dtype=error.dtype), 1e-6)
    degrees_of_freedom = jnp.asarray(degrees_of_freedom, dtype=error.dtype)
    normalized = error / scale
    normalizer = (
        jnp.log(scale)
        + 0.5 * jnp.log(degrees_of_freedom * jnp.asarray(jnp.pi, dtype=error.dtype))
        + jsp.special.gammaln(0.5 * degrees_of_freedom)
        - jsp.special.gammaln(0.5 * (degrees_of_freedom + 1.0))
    )
    return normalizer + 0.5 * (degrees_of_freedom + 1.0) * jnp.log1p(jnp.square(normalized) / degrees_of_freedom)


def _root_equal_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Average all valid observations within a root, then average roots equally."""

    values = jnp.asarray(values)
    valid = jnp.broadcast_to(jnp.asarray(mask, dtype=jnp.bool_), values.shape)
    mask = valid.astype(values.dtype)
    safe_values = jnp.where(valid, values, 0.0)
    reduction_axes = tuple(range(1, values.ndim))
    count = jnp.sum(mask, axis=reduction_axes)
    per_root = jnp.sum(safe_values, axis=reduction_axes) / jnp.maximum(count, 1.0)
    root_valid = count > 0
    return jnp.sum(jnp.where(root_valid, per_root, 0.0)) / jnp.maximum(jnp.sum(root_valid), 1.0)


class _TransformerBlock(nnx.Module):
    """Small pre-norm transformer block without dropout."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feed_forward_multiplier: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype,
    ) -> None:
        self.attention_norm = nnx.LayerNorm(hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.attention = nnx.MultiHeadAttention(
            in_features=hidden_dim,
            num_heads=num_heads,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.feed_forward_norm = nnx.LayerNorm(hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.feed_forward_in = nnx.Linear(
            hidden_dim,
            hidden_dim * feed_forward_multiplier,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.feed_forward_out = nnx.Linear(
            hidden_dim * feed_forward_multiplier,
            hidden_dim,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(
            normalized,
            normalized,
            normalized,
            deterministic=True,
            decode=False,
        )
        normalized = self.feed_forward_norm(tokens)
        return tokens + self.feed_forward_out(nnx.gelu(self.feed_forward_in(normalized)))


class _LearnedQueryPool(nnx.Module):
    """Attention-pool already-computed VLA prefix tokens into visual slots."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_queries: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.key_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        self.value_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs, param_dtype=param_dtype)
        queries = jax.random.normal(rngs.params(), (num_queries, hidden_dim), dtype=param_dtype)
        self.queries = nnx.Param(queries / jnp.sqrt(float(hidden_dim)))

    def __call__(self, prefix_tokens: jax.Array, prefix_mask: jax.Array) -> jax.Array:
        keys = self.key_proj(jnp.asarray(prefix_tokens, dtype=jnp.float32))
        values = self.value_proj(jnp.asarray(prefix_tokens, dtype=jnp.float32))
        logits = jnp.einsum("qd,btd->bqt", self.queries.value, keys) / jnp.sqrt(float(self.hidden_dim))
        mask = jnp.asarray(prefix_mask, dtype=jnp.bool_)[:, None, :]
        masked_logits = jnp.where(mask, logits, -1e30)
        weights = jax.nn.softmax(masked_logits, axis=-1)
        weights = jnp.where(mask, weights, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-6)
        return jnp.einsum("bqt,btd->bqd", weights, values)


class ExecutionHorizonPredictor(nnx.Module):
    """Shared temporal encoder with entropy/event and counterfactual-Q heads."""

    def __init__(
        self,
        config: ExecutionHorizonPredictorConfig,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.config = config
        h = config.hidden_dim
        self.prefix_proj = nnx.Linear(config.prefix_feature_dim, h, rngs=rngs, param_dtype=param_dtype)
        self.state_proj = nnx.Linear(config.state_dim, h, rngs=rngs, param_dtype=param_dtype)
        self.controller_proj = nnx.Linear(4, h, rngs=rngs, param_dtype=param_dtype)
        if config.temporal_backbone == "local_mlp":
            # Preserve the exact legacy parameter tree and input width.
            self.action_proj = nnx.Linear(4 * config.action_dim + 2, h, rngs=rngs, param_dtype=param_dtype)
            self.temporal_layers = [
                nnx.Linear(3 * h, h, rngs=rngs, param_dtype=param_dtype) for _ in range(config.temporal_layers)
            ]
        else:
            # Eight vector channels: final/coarse/coarse-delta/previous/
            # previous-delta/velocity/acceleration/jerk; eight scalar event
            # channels are appended below.
            self.action_proj = nnx.Linear(8 * config.action_dim + 8, h, rngs=rngs, param_dtype=param_dtype)
            self.temporal_layers = [
                _TransformerBlock(
                    h,
                    config.num_heads,
                    config.feed_forward_multiplier,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
                for _ in range(config.temporal_layers)
            ]
            position_embedding = jax.random.normal(
                rngs.params(),
                (config.action_horizon, h),
                dtype=param_dtype,
            )
            self.position_embedding = nnx.Param(position_embedding / jnp.sqrt(float(h)))
            if config.visual_num_queries:
                self.visual_pool = _LearnedQueryPool(
                    config.prefix_feature_dim,
                    h,
                    config.visual_num_queries,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
        self.summary_proj = nnx.Linear(2 * h, h, rngs=rngs, param_dtype=param_dtype)

        self.final_risk_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
        self.action_cot_risk_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
        self.fused_risk_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
        self.event_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)

        head_width = config.action_horizon if config.temporal_backbone == "local_mlp" else config.num_candidates
        self.raw_h_logits_head = nnx.Linear(h, head_width, rngs=rngs, param_dtype=param_dtype)
        self.raw_h_ordinal_head = nnx.Linear(h, head_width - 1, rngs=rngs, param_dtype=param_dtype)
        self.success_head = nnx.Linear(h, head_width, rngs=rngs, param_dtype=param_dtype)
        self.timeout_head = nnx.Linear(h, head_width, rngs=rngs, param_dtype=param_dtype)
        self.remaining_calls_head = nnx.Linear(h, head_width, rngs=rngs, param_dtype=param_dtype)
        self.remaining_steps_head = nnx.Linear(h, head_width, rngs=rngs, param_dtype=param_dtype)
        if config.temporal_backbone == "transformer":
            long_width = len(config.long_horizons)
            self.hazard_head = nnx.Linear(h, 1, rngs=rngs, param_dtype=param_dtype)
            if config.paired_distribution_heads:
                self.paired_outcome_logits_head = nnx.Linear(
                    h,
                    3 * long_width,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
                self.faster_long_logits_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
            elif config.paired_advantage_heads:
                self.danger_logits_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
                self.rescue_logits_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
                self.faster_long_logits_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
            else:
                self.success_advantage_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
            if not config.paired_distribution_heads:
                self.success_advantage_log_scale_head = nnx.Linear(
                    h,
                    long_width,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
            self.elapsed_advantage_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
            self.elapsed_advantage_log_scale_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
            self.calls_advantage_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)
            self.calls_advantage_log_scale_head = nnx.Linear(h, long_width, rngs=rngs, param_dtype=param_dtype)

    def _align_coarse(self, coarse_actions: jax.Array) -> jax.Array:
        if self.config.temporal_backbone == "local_mlp":
            indices = jnp.rint(jnp.linspace(0, self.config.coarse_horizon - 1, self.config.action_horizon)).astype(
                jnp.int32
            )
            return jnp.take(coarse_actions, indices, axis=1)

        # Coarse token i describes physical time i * coarse_stride. Interpolate
        # that trajectory at final-token times instead of stretching both
        # sequences to share endpoints.
        final_time = jnp.arange(self.config.action_horizon, dtype=jnp.float32) * self.config.final_stride
        coarse_position = final_time / float(self.config.coarse_stride)
        left = jnp.floor(coarse_position).astype(jnp.int32)
        left = jnp.clip(left, 0, self.config.coarse_horizon - 1)
        right = jnp.minimum(left + 1, self.config.coarse_horizon - 1)
        weight = jnp.clip(coarse_position - left.astype(jnp.float32), 0.0, 1.0)
        left_value = jnp.take(coarse_actions, left, axis=1)
        right_value = jnp.take(coarse_actions, right, axis=1)
        return left_value + weight[None, :, None] * (right_value - left_value)

    @staticmethod
    def _temporal_derivatives(actions: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        def difference(values: jax.Array) -> jax.Array:
            return jnp.concatenate([jnp.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]], axis=1)

        velocity = difference(actions)
        acceleration = difference(velocity)
        jerk = difference(acceleration)
        return velocity, acceleration, jerk

    def _transformer_action_features(
        self,
        final_actions: jax.Array,
        aligned_coarse: jax.Array,
        aligned_previous: jax.Array,
        overlap_valid: jax.Array,
        consistency: jax.Array,
    ) -> jax.Array:
        velocity, acceleration, jerk = self._temporal_derivatives(final_actions)
        coarse_delta = aligned_coarse - final_actions
        # No overlap is evidence absence, not disagreement. Mask both the
        # previous-chunk delta and its scalar consistency outside the
        # unexecuted suffix so first decisions/H20 chunks are not assigned an
        # artificial high inconsistency equal to action magnitude.
        previous_delta = (final_actions - aligned_previous) * overlap_valid
        consistency = consistency * overlap_valid
        physical_dim = self.config.physical_action_dim
        gripper_dim = min(6, physical_dim - 1)
        gripper_transition = jnp.abs(velocity[..., gripper_dim : gripper_dim + 1])

        def magnitude(values: jax.Array) -> jax.Array:
            return jnp.sqrt(jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-8)

        physical_final = final_actions[..., :physical_dim]
        physical_coarse = aligned_coarse[..., :physical_dim]
        final_norm = jnp.linalg.norm(physical_final, axis=-1, keepdims=True)
        coarse_norm = jnp.linalg.norm(physical_coarse, axis=-1, keepdims=True)
        cosine = jnp.sum(physical_final * physical_coarse, axis=-1, keepdims=True) / jnp.maximum(
            final_norm * coarse_norm,
            1e-6,
        )
        direction_disagreement = jnp.where(
            (final_norm * coarse_norm) > 1e-6,
            1.0 - jnp.clip(cosine, -1.0, 1.0),
            0.0,
        )
        return jnp.concatenate(
            [
                final_actions,
                aligned_coarse,
                coarse_delta,
                aligned_previous,
                previous_delta,
                velocity,
                acceleration,
                jerk,
                overlap_valid,
                consistency,
                magnitude(velocity[..., :physical_dim]),
                magnitude(acceleration[..., :physical_dim]),
                magnitude(jerk[..., :physical_dim]),
                gripper_transition,
                magnitude(coarse_delta[..., :physical_dim]),
                direction_disagreement,
            ],
            axis=-1,
        )

    def _previous_overlap(
        self,
        final_actions: jax.Array,
        previous_actions: jax.Array,
        previous_h: jax.Array,
        previous_valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        horizon = self.config.action_horizon
        previous_h = jnp.clip(jnp.asarray(previous_h, dtype=jnp.int32).reshape((-1,)), 1, horizon)
        indices = previous_h[:, None] + jnp.arange(horizon, dtype=jnp.int32)[None, :]
        valid = (indices < horizon) & jnp.asarray(previous_valid, dtype=jnp.bool_).reshape((-1, 1))
        clipped = jnp.minimum(indices, horizon - 1)
        aligned = jnp.take_along_axis(previous_actions, clipped[..., None], axis=1)
        aligned = jnp.where(valid[..., None], aligned, 0.0)
        delta = final_actions - aligned
        consistency = jnp.mean(jnp.abs(delta), axis=-1, keepdims=True)
        return aligned, valid[..., None].astype(final_actions.dtype), consistency

    def __call__(
        self,
        *,
        prefix_feature: jax.Array,
        state: jax.Array,
        coarse_actions: jax.Array,
        final_actions: jax.Array,
        previous_actions: jax.Array,
        previous_h: jax.Array,
        budget_balance: jax.Array,
        episode_progress: jax.Array,
        previous_valid: jax.Array,
        prefix_tokens: jax.Array | None = None,
        prefix_mask: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        cfg = self.config
        prefix_feature = jnp.asarray(prefix_feature, dtype=jnp.float32)
        state = jnp.asarray(state, dtype=jnp.float32)[..., : cfg.state_dim]
        coarse_actions = jnp.asarray(coarse_actions, dtype=jnp.float32)[..., : cfg.coarse_horizon, : cfg.action_dim]
        final_actions = jnp.asarray(final_actions, dtype=jnp.float32)[..., : cfg.action_horizon, : cfg.action_dim]
        previous_actions = jnp.asarray(previous_actions, dtype=jnp.float32)[..., : cfg.action_horizon, : cfg.action_dim]

        aligned_coarse = self._align_coarse(coarse_actions)
        aligned_previous, overlap_valid, consistency = self._previous_overlap(
            final_actions, previous_actions, previous_h, previous_valid
        )
        if cfg.temporal_backbone == "local_mlp":
            action_features = jnp.concatenate(
                [
                    final_actions,
                    aligned_coarse,
                    aligned_previous,
                    final_actions - aligned_previous,
                    overlap_valid,
                    consistency,
                ],
                axis=-1,
            )
        else:
            action_features = self._transformer_action_features(
                final_actions,
                aligned_coarse,
                aligned_previous,
                overlap_valid,
                consistency,
            )

        previous_h_float = jnp.asarray(previous_h, dtype=jnp.float32).reshape((-1, 1)) / cfg.action_horizon
        controller = jnp.concatenate(
            [
                previous_h_float,
                jnp.asarray(budget_balance, dtype=jnp.float32).reshape((-1, 1)),
                jnp.asarray(episode_progress, dtype=jnp.float32).reshape((-1, 1)),
                jnp.asarray(previous_valid, dtype=jnp.float32).reshape((-1, 1)),
            ],
            axis=-1,
        )
        context = nnx.swish(self.prefix_proj(prefix_feature))
        context = context + nnx.swish(self.state_proj(state)) + nnx.swish(self.controller_proj(controller))
        tokens = nnx.swish(self.action_proj(action_features)) + context[:, None, :]

        if cfg.temporal_backbone == "local_mlp":
            for layer in self.temporal_layers:
                left = jnp.concatenate([tokens[:, :1], tokens[:, :-1]], axis=1)
                right = jnp.concatenate([tokens[:, 1:], tokens[:, -1:]], axis=1)
                tokens = tokens + nnx.swish(layer(jnp.concatenate([left, tokens, right], axis=-1)))
        else:
            tokens = tokens + self.position_embedding.value[None, :, :]
            sequence = tokens
            if cfg.visual_num_queries:
                if prefix_tokens is None or prefix_mask is None:
                    raise ValueError(
                        "Transformer visual query pooling requires prefix_tokens and prefix_mask from the existing "
                        "VLA prefix forward."
                    )
                visual_tokens = self.visual_pool(prefix_tokens, prefix_mask) + context[:, None, :]
                sequence = jnp.concatenate([visual_tokens, sequence], axis=1)
            for layer in self.temporal_layers:
                sequence = layer(sequence)
            tokens = sequence[:, -cfg.action_horizon :]

        temporal_summary = jnp.mean(tokens, axis=1)
        summary = nnx.swish(self.summary_proj(jnp.concatenate([temporal_summary, context], axis=-1)))
        remaining_calls = nnx.softplus(self.remaining_calls_head(summary)) * cfg.remaining_calls_scale
        remaining_steps = nnx.softplus(self.remaining_steps_head(summary)) * cfg.remaining_steps_scale
        raw_h_ordinal_logits = self.raw_h_ordinal_head(summary)
        result = {
            "final_risk": nnx.softplus(self.final_risk_head(tokens)[..., 0]),
            "action_cot_risk": nnx.softplus(self.action_cot_risk_head(tokens)[..., 0]),
            "fused_risk": nnx.softplus(self.fused_risk_head(tokens)[..., 0]),
            "event_logits": self.event_head(tokens)[..., 0],
            "raw_h_logits": self.raw_h_logits_head(summary),
            "raw_h_ordinal_logits": raw_h_ordinal_logits,
            "success_logits": self.success_head(summary),
            "timeout_logits": self.timeout_head(summary),
            "remaining_calls": remaining_calls,
            "remaining_steps": remaining_steps,
            "temporal_feature": summary,
            "overlap_consistency": consistency[..., 0],
        }
        if cfg.ordered_continuation_head:
            ordered_log_probability, ordered_probability = ordered_continuation_distribution(
                raw_h_ordinal_logits
            )
            ordered_index = jnp.argmax(ordered_probability, axis=-1)
            ordered_selected_h = jnp.asarray(cfg.candidate_horizons, dtype=jnp.int32)[ordered_index]
            result.update(
                {
                    "ordered_continuation_logits": raw_h_ordinal_logits,
                    "ordered_continuation_probability": jax.nn.sigmoid(raw_h_ordinal_logits),
                    "ordered_horizon_log_probability": ordered_log_probability,
                    "ordered_horizon_probability": ordered_probability,
                    "ordered_selected_h": ordered_selected_h,
                }
            )
        if cfg.temporal_backbone == "transformer":
            hazard_logits = self.hazard_head(tokens)[..., 0]
            hazard = jax.nn.sigmoid(hazard_logits)
            paired_outputs: dict[str, jax.Array] = {}
            if cfg.paired_distribution_heads:
                paired_outcome_logits = self.paired_outcome_logits_head(summary).reshape(
                    summary.shape[0],
                    len(cfg.long_horizons),
                    3,
                )
                # Outcome order is danger / tie / rescue.  A single softmax
                # makes these paired outcomes mutually exclusive, unlike the
                # legacy two-independent-sigmoid parameterization.
                paired_outcome_probability = jax.nn.softmax(paired_outcome_logits, axis=-1)
                danger_probability = paired_outcome_probability[..., 0]
                tie_probability = paired_outcome_probability[..., 1]
                rescue_probability = paired_outcome_probability[..., 2]
                success_advantage = rescue_probability - danger_probability
                success_advantage_variance = rescue_probability + danger_probability - jnp.square(success_advantage)
                success_advantage_std = jnp.sqrt(jnp.maximum(success_advantage_variance, 1e-8))
                faster_long_logits = self.faster_long_logits_head(summary)
                paired_outputs = {
                    "paired_outcome_logits": paired_outcome_logits,
                    "paired_outcome_probability": paired_outcome_probability,
                    # Preserve the legacy selector-facing keys while changing
                    # their probabilities to a coherent categorical model.
                    "danger_logits": paired_outcome_logits[..., 0],
                    "danger_probability": danger_probability,
                    "tie_logits": paired_outcome_logits[..., 1],
                    "tie_probability": tie_probability,
                    "rescue_logits": paired_outcome_logits[..., 2],
                    "rescue_probability": rescue_probability,
                    "faster_long_logits": faster_long_logits,
                    "faster_long_probability": jax.nn.sigmoid(faster_long_logits),
                }
            elif cfg.paired_advantage_heads:
                danger_logits = self.danger_logits_head(summary)
                rescue_logits = self.rescue_logits_head(summary)
                danger_probability = jax.nn.sigmoid(danger_logits)
                rescue_probability = jax.nn.sigmoid(rescue_logits)
                # For paired continuations, the success-rate treatment effect
                # is exactly P(rescue) - P(regression).  Predicting these two
                # rare events directly retains the causal supervision that is
                # lost when three seed outcomes are collapsed to one noisy
                # empirical rate difference.
                success_advantage = rescue_probability - danger_probability
                faster_long_logits = self.faster_long_logits_head(summary)
                paired_outputs = {
                    "danger_logits": danger_logits,
                    "danger_probability": danger_probability,
                    "rescue_logits": rescue_logits,
                    "rescue_probability": rescue_probability,
                    "faster_long_logits": faster_long_logits,
                    "faster_long_probability": jax.nn.sigmoid(faster_long_logits),
                }
            else:
                success_advantage = jnp.tanh(self.success_advantage_head(summary))
            if cfg.paired_distribution_heads:
                success_advantage_log_scale = jnp.log(success_advantage_std)
                paired_elapsed_raw_scale = self.elapsed_advantage_log_scale_head(summary)
                paired_elapsed_scale = jax.nn.softplus(paired_elapsed_raw_scale) * cfg.elapsed_advantage_scale + 1e-3
                # With nu=4, the Student-t standard deviation is
                # scale*sqrt(nu/(nu-2)). Keep it separate from the likelihood
                # scale so selector confidence bounds use the correct units.
                elapsed_advantage_std = paired_elapsed_scale * jnp.sqrt(jnp.asarray(2.0, dtype=summary.dtype))
                elapsed_advantage_log_scale = jnp.log(elapsed_advantage_std)
                paired_outputs.update(
                    {
                        "paired_elapsed_raw_scale": paired_elapsed_raw_scale,
                        "paired_elapsed_scale": paired_elapsed_scale,
                    }
                )
            else:
                success_advantage_log_scale = jnp.clip(
                    self.success_advantage_log_scale_head(summary),
                    -7.0,
                    2.0,
                )
                success_advantage_std = jnp.exp(success_advantage_log_scale)
                elapsed_advantage_log_scale = jnp.clip(
                    self.elapsed_advantage_log_scale_head(summary)
                    + jnp.log(jnp.asarray(cfg.elapsed_advantage_scale, dtype=summary.dtype)),
                    -7.0,
                    7.0,
                )
                elapsed_advantage_std = jnp.exp(elapsed_advantage_log_scale)
            calls_advantage_log_scale = jnp.clip(
                self.calls_advantage_log_scale_head(summary)
                + jnp.log(jnp.asarray(cfg.calls_advantage_scale, dtype=summary.dtype)),
                -7.0,
                7.0,
            )
            reference_index = cfg.candidate_horizons.index(cfg.reference_horizon)
            reference_success_probability = jax.nn.sigmoid(result["success_logits"][:, reference_index])
            result.update(
                {
                    "hazard_logits": hazard_logits,
                    "hazard": hazard,
                    "survival": jnp.cumprod(1.0 - hazard, axis=-1),
                    "reference_success_probability": reference_success_probability,
                    "success_advantage": success_advantage,
                    "success_advantage_log_scale": success_advantage_log_scale,
                    "success_advantage_std": success_advantage_std,
                    "long_success_probability": jnp.clip(
                        reference_success_probability[:, None] + success_advantage,
                        1e-5,
                        1.0 - 1e-5,
                    ),
                    "elapsed_advantage": self.elapsed_advantage_head(summary) * cfg.elapsed_advantage_scale,
                    "elapsed_advantage_log_scale": elapsed_advantage_log_scale,
                    "elapsed_advantage_std": elapsed_advantage_std,
                    "calls_advantage": self.calls_advantage_head(summary) * cfg.calls_advantage_scale,
                    "calls_advantage_log_scale": calls_advantage_log_scale,
                    "calls_advantage_std": jnp.exp(calls_advantage_log_scale),
                    "candidate_horizons": jnp.broadcast_to(
                        jnp.asarray(cfg.candidate_horizons, dtype=jnp.int32),
                        (summary.shape[0], cfg.num_candidates),
                    ),
                    "reference_horizon": jnp.full(
                        (summary.shape[0],),
                        cfg.reference_horizon,
                        dtype=jnp.int32,
                    ),
                    "long_horizons": jnp.broadcast_to(
                        jnp.asarray(cfg.long_horizons, dtype=jnp.int32),
                        (summary.shape[0], len(cfg.long_horizons)),
                    ),
                    **paired_outputs,
                }
            )
        return result


def execution_horizon_loss(
    predictions: Mapping[str, jax.Array],
    labels: Mapping[str, jax.Array],
    *,
    weights: ExecutionHorizonLossWeights = DEFAULT_LOSS_WEIGHTS,
    label_weights: ExecutionHorizonLabelWeights = DEFAULT_LABEL_WEIGHTS,
    remaining_calls_scale: float = 64.0,
    remaining_steps_scale: float = 512.0,
    elapsed_advantage_scale: float = 1.0,
    ordered_listwise_elapsed_temperature: float = 0.25,
    candidate_horizons: tuple[int, ...] | None = None,
    reference_horizon: int = 10,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the configurable SFT objective over all counterfactual H values."""

    branch_mask = jnp.asarray(labels.get("branch_valid", jnp.ones_like(labels["branch_success"])))
    risk_mask = jnp.asarray(labels.get("risk_valid", jnp.ones_like(labels["final_risk"])))
    if "success_count" in labels and "trial_count" in labels:
        count_trials = jnp.asarray(labels["trial_count"], dtype=jnp.float32)
        success_label = jnp.asarray(labels["success_count"], dtype=jnp.float32) >= 0.5 * jnp.maximum(count_trials, 1.0)
        timeout_label = jnp.asarray(labels.get("timeout_count", 0), dtype=jnp.float32) >= 0.5 * jnp.maximum(
            count_trials, 1.0
        )
    else:
        success_label = jnp.asarray(labels["branch_success"], dtype=jnp.bool_)
        timeout_label = jnp.asarray(labels["branch_timeout"], dtype=jnp.bool_)
    event_label = jnp.asarray(labels["event_mask"], dtype=jnp.bool_)
    success_mask = branch_mask * jnp.where(success_label, 1.0, label_weights.success_failure)
    timeout_mask = branch_mask * jnp.where(timeout_label, label_weights.timeout_positive, 1.0)
    event_mask = risk_mask * jnp.where(event_label, label_weights.event_positive, 1.0)
    risk_regression_mask = risk_mask * jnp.where(event_label, label_weights.risk_event, 1.0)
    if "success_count" in labels and "trial_count" in labels:
        trial_count = jnp.asarray(labels["trial_count"], dtype=jnp.float32)
        success_loss = _masked_mean(
            _binomial_nll(predictions["success_logits"], labels["success_count"], trial_count)
            / jnp.maximum(trial_count, 1.0),
            success_mask * (trial_count > 0),
        )
    else:
        trial_count = branch_mask
        success_loss = _masked_mean(
            _bce_with_logits(predictions["success_logits"], success_label),
            success_mask,
        )
    if "timeout_count" in labels and "trial_count" in labels:
        timeout_loss = _masked_mean(
            _binomial_nll(predictions["timeout_logits"], labels["timeout_count"], trial_count)
            / jnp.maximum(trial_count, 1.0),
            timeout_mask * (trial_count > 0),
        )
    else:
        timeout_loss = _masked_mean(
            _bce_with_logits(predictions["timeout_logits"], timeout_label),
            timeout_mask,
        )
    remaining_calls_label = labels.get("remaining_calls_mean", labels["remaining_calls"])
    remaining_steps_label = labels.get("remaining_steps_mean", labels["remaining_steps"])
    calls_loss = _masked_mean(
        _huber((predictions["remaining_calls"] - remaining_calls_label) / remaining_calls_scale),
        branch_mask,
    )
    steps_loss = _masked_mean(
        _huber((predictions["remaining_steps"] - remaining_steps_label) / remaining_steps_scale),
        branch_mask,
    )
    final_risk_loss = _masked_mean(
        _huber(predictions["final_risk"] - labels["final_risk"]),
        risk_regression_mask,
    )
    cot_risk_loss = _masked_mean(
        _huber(predictions["action_cot_risk"] - labels["action_cot_risk"]),
        risk_regression_mask,
    )
    fused_risk_loss = _masked_mean(
        _huber(predictions["fused_risk"] - labels["fused_risk"]),
        risk_regression_mask,
    )
    event_loss = _masked_mean(_bce_with_logits(predictions["event_logits"], event_label), event_mask)

    if candidate_horizons is None:
        candidate_horizons = tuple(range(1, predictions["raw_h_logits"].shape[-1] + 1))
    if len(candidate_horizons) != predictions["raw_h_logits"].shape[-1]:
        raise ValueError(
            "candidate_horizons length must match raw_h_logits width; "
            f"got {candidate_horizons} and {predictions['raw_h_logits'].shape[-1]}."
        )
    candidate_array = jnp.asarray(candidate_horizons, dtype=jnp.int32)
    raw_h = jnp.asarray(labels["raw_h"], dtype=jnp.int32).reshape((-1,))
    raw_h_index = jnp.argmin(jnp.abs(raw_h[:, None] - candidate_array[None, :]), axis=-1)
    raw_h_classification_loss = jnp.mean(
        -jnp.take_along_axis(
            jax.nn.log_softmax(predictions["raw_h_logits"], axis=-1),
            raw_h_index[:, None],
            axis=-1,
        )
    )
    ordinal_targets = raw_h[:, None] > candidate_array[:-1][None, :]
    raw_h_ordinal_loss = jnp.mean(_bce_with_logits(predictions["raw_h_ordinal_logits"], ordinal_targets))

    ordered_listwise_loss = jnp.asarray(0.0, dtype=jnp.float32)
    ordered_listwise_target_entropy = jnp.asarray(0.0, dtype=jnp.float32)
    ordered_listwise_target_horizon = jnp.asarray(0.0, dtype=jnp.float32)
    ordered_listwise_valid_fraction = jnp.asarray(0.0, dtype=jnp.float32)
    if weights.ordered_listwise > 0:
        required = ("success_count", "trial_count", "elapsed_mean")
        missing = [name for name in required if name not in labels]
        if missing:
            raise ValueError(f"ordered listwise loss requires count and elapsed labels: {missing}.")
        if "ordered_horizon_log_probability" not in predictions:
            raise ValueError("ordered listwise loss requires ordered_continuation_head predictor outputs.")
        ordered_log_probability = jnp.asarray(predictions["ordered_horizon_log_probability"])
        if ordered_log_probability.shape[-1] != len(candidate_horizons):
            raise ValueError(
                "ordered horizon probability width must match candidate_horizons; "
                f"got {ordered_log_probability.shape[-1]} and {candidate_horizons}."
            )
        ordered_target, ordered_root_valid = success_first_listwise_target(
            labels["success_count"],
            labels["trial_count"],
            labels["elapsed_mean"],
            branch_mask,
            elapsed_temperature=ordered_listwise_elapsed_temperature,
        )
        ordered_listwise_loss = _masked_mean(
            -jnp.sum(ordered_target * ordered_log_probability, axis=-1),
            ordered_root_valid,
        )
        ordered_listwise_target_entropy = _masked_mean(
            -jnp.sum(ordered_target * jnp.log(jnp.maximum(ordered_target, 1e-12)), axis=-1),
            ordered_root_valid,
        )
        ordered_listwise_target_horizon = _masked_mean(
            jnp.sum(ordered_target * candidate_array[None, :], axis=-1),
            ordered_root_valid,
        )
        ordered_listwise_valid_fraction = jnp.mean(ordered_root_valid)

    survival_loss = jnp.asarray(0.0, dtype=jnp.float32)
    if weights.survival > 0:
        if "hazard_logits" not in predictions:
            raise ValueError("survival loss requires transformer hazard_logits.")
        if "hazard_event_count" in labels and "hazard_at_risk_count" in labels:
            hazard_event_count = jnp.asarray(labels["hazard_event_count"], dtype=jnp.float32)
            hazard_at_risk_count = jnp.asarray(labels["hazard_at_risk_count"], dtype=jnp.float32)
            survival_loss = _masked_mean(
                _binomial_nll(
                    predictions["hazard_logits"],
                    hazard_event_count,
                    hazard_at_risk_count,
                )
                / jnp.maximum(hazard_at_risk_count, 1.0),
                hazard_at_risk_count > 0,
            )
        else:
            hazard_event = jnp.asarray(labels.get("hazard_event", labels["event_mask"]), dtype=jnp.bool_)
            hazard_valid = jnp.asarray(labels.get("hazard_valid", risk_mask), dtype=jnp.float32)
            prior_event = jnp.concatenate(
                [jnp.zeros_like(hazard_event[:, :1]), jnp.cumsum(hazard_event[:, :-1], axis=-1)],
                axis=-1,
            )
            at_risk = (prior_event == 0).astype(jnp.float32) * hazard_valid
            survival_loss = _masked_mean(
                _bce_with_logits(predictions["hazard_logits"], hazard_event),
                at_risk,
            )

    success_advantage_loss = jnp.asarray(0.0, dtype=jnp.float32)
    elapsed_advantage_loss = jnp.asarray(0.0, dtype=jnp.float32)
    calls_advantage_loss = jnp.asarray(0.0, dtype=jnp.float32)
    false_long_loss = jnp.asarray(0.0, dtype=jnp.float32)
    danger_rescue_loss = jnp.asarray(0.0, dtype=jnp.float32)
    danger_binomial_loss = jnp.asarray(0.0, dtype=jnp.float32)
    rescue_binomial_loss = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_multinomial_loss = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_danger_nll = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_tie_nll = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_rescue_nll = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_danger_rate = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_tie_rate = jnp.asarray(0.0, dtype=jnp.float32)
    paired_outcome_rescue_rate = jnp.asarray(0.0, dtype=jnp.float32)
    paired_elapsed_loss = jnp.asarray(0.0, dtype=jnp.float32)
    paired_elapsed_student_t_loss = jnp.asarray(0.0, dtype=jnp.float32)
    paired_elapsed_scale_mean = jnp.asarray(0.0, dtype=jnp.float32)
    paired_elapsed_covariance = jnp.asarray(0.0, dtype=jnp.float32)
    paired_elapsed_delta_variance = jnp.asarray(0.0, dtype=jnp.float32)
    faster_long_loss = jnp.asarray(0.0, dtype=jnp.float32)
    paired_distribution_mode = "paired_outcome_logits" in predictions
    long_indices = tuple(index for index, value in enumerate(candidate_horizons) if value > reference_horizon)
    if (
        weights.success_advantage > 0
        or weights.elapsed_advantage > 0
        or weights.calls_advantage > 0
        or weights.false_long > 0
        or weights.danger_rescue > 0
        or weights.paired_elapsed > 0
        or weights.faster_long > 0
    ):
        if reference_horizon not in candidate_horizons:
            raise ValueError("reference_horizon must be included in candidate_horizons.")
        if not long_indices:
            raise ValueError("Hierarchical advantage losses require at least one candidate above reference_horizon.")
        reference_index = candidate_horizons.index(reference_horizon)
        long_index_array = jnp.asarray(long_indices, dtype=jnp.int32)
        reference_trials = trial_count[:, reference_index : reference_index + 1]
        long_trials = jnp.take(trial_count, long_index_array, axis=1)
        advantage_valid = (reference_trials > 0) & (long_trials > 0)

        pair_valid = None
        dangerous_pair = None
        rescue_pair = None
        paired_elapsed_delta = None
        if weights.danger_rescue > 0 or weights.paired_elapsed > 0 or weights.faster_long > 0:
            required = ("trial_success", "trial_elapsed", "trial_valid")
            missing = [name for name in required if name not in labels]
            if missing:
                raise ValueError(f"Paired advantage losses require raw continuation labels: {missing}.")
            trial_valid_raw = jnp.asarray(labels["trial_valid"], dtype=jnp.bool_)
            trial_success_raw = jnp.asarray(labels["trial_success"], dtype=jnp.bool_)
            trial_elapsed_raw = jnp.asarray(labels["trial_elapsed"], dtype=jnp.float32)
            reference_valid_raw = trial_valid_raw[:, reference_index : reference_index + 1, :]
            reference_success_raw = trial_success_raw[:, reference_index : reference_index + 1, :]
            reference_elapsed_raw = trial_elapsed_raw[:, reference_index : reference_index + 1, :]
            long_valid_raw = jnp.take(trial_valid_raw, long_index_array, axis=1)
            long_success_raw = jnp.take(trial_success_raw, long_index_array, axis=1)
            long_elapsed_raw = jnp.take(trial_elapsed_raw, long_index_array, axis=1)
            pair_valid = reference_valid_raw & long_valid_raw
            pair_valid &= jnp.isfinite(reference_elapsed_raw) & jnp.isfinite(long_elapsed_raw)
            dangerous_pair = pair_valid & reference_success_raw & ~long_success_raw
            rescue_pair = pair_valid & ~reference_success_raw & long_success_raw
            paired_elapsed_delta = long_elapsed_raw - reference_elapsed_raw

        if weights.success_advantage > 0 and not paired_distribution_mode:
            if "success_advantage" not in predictions or "success_count" not in labels:
                raise ValueError("success advantage loss requires transformer predictions and count labels.")
            success_count = jnp.asarray(labels["success_count"], dtype=jnp.float32)
            success_rate = success_count / jnp.maximum(trial_count, 1.0)
            success_target = (
                jnp.take(success_rate, long_index_array, axis=1)
                - success_rate[:, reference_index : reference_index + 1]
            )
            # Jeffreys-smoothed binomial uncertainty prevents the learned
            # epistemic scale from absorbing finite continuation-seed noise.
            smoothed_rate = (success_count + 0.5) / (trial_count + 1.0)
            rate_variance = smoothed_rate * (1.0 - smoothed_rate) / jnp.maximum(trial_count + 2.0, 1.0)
            success_observation_variance = (
                jnp.take(rate_variance, long_index_array, axis=1)
                + rate_variance[:, reference_index : reference_index + 1]
            )
            success_error = predictions["success_advantage"] - success_target
            success_advantage_loss = _masked_mean(
                _gaussian_nll(
                    success_error,
                    predictions["success_advantage_log_scale"],
                    success_observation_variance,
                ),
                advantage_valid,
            )

        if weights.elapsed_advantage > 0 and not paired_distribution_mode:
            if "elapsed_advantage" not in predictions or "elapsed_mean" not in labels:
                raise ValueError("elapsed advantage loss requires transformer predictions and elapsed_mean labels.")
            elapsed_mean = jnp.asarray(labels["elapsed_mean"], dtype=jnp.float32)
            elapsed_target = (
                jnp.take(elapsed_mean, long_index_array, axis=1)
                - elapsed_mean[:, reference_index : reference_index + 1]
            )
            elapsed_valid = advantage_valid & jnp.isfinite(elapsed_target)
            elapsed_error = predictions["elapsed_advantage"] - jnp.nan_to_num(elapsed_target)
            elapsed_observation_variance = None
            if "elapsed_variance" in labels:
                elapsed_variance = jnp.asarray(labels["elapsed_variance"], dtype=jnp.float32)
                elapsed_observation_variance = jnp.take(
                    elapsed_variance / jnp.maximum(trial_count, 1.0),
                    long_index_array,
                    axis=1,
                ) + elapsed_variance[:, reference_index : reference_index + 1] / jnp.maximum(reference_trials, 1.0)
            elapsed_advantage_loss = _masked_mean(
                _gaussian_nll(
                    elapsed_error,
                    predictions["elapsed_advantage_log_scale"],
                    elapsed_observation_variance,
                ),
                elapsed_valid,
            )

        if weights.calls_advantage > 0:
            if "calls_advantage" not in predictions or "remaining_calls_mean" not in labels:
                raise ValueError(
                    "calls advantage loss requires transformer predictions and remaining_calls_mean labels."
                )
            calls_mean = jnp.asarray(labels["remaining_calls_mean"], dtype=jnp.float32)
            calls_target = (
                jnp.take(calls_mean, long_index_array, axis=1) - calls_mean[:, reference_index : reference_index + 1]
            )
            calls_error = predictions["calls_advantage"] - calls_target
            calls_observation_variance = None
            if "remaining_calls_variance" in labels:
                calls_variance = jnp.asarray(labels["remaining_calls_variance"], dtype=jnp.float32)
                calls_observation_variance = jnp.take(
                    calls_variance / jnp.maximum(trial_count, 1.0),
                    long_index_array,
                    axis=1,
                ) + calls_variance[:, reference_index : reference_index + 1] / jnp.maximum(reference_trials, 1.0)
            calls_advantage_loss = _masked_mean(
                _gaussian_nll(
                    calls_error,
                    predictions["calls_advantage_log_scale"],
                    calls_observation_variance,
                ),
                advantage_valid,
            )

        if weights.false_long > 0:
            if "dangerous_long_count" not in labels or "paired_trial_count" not in labels:
                raise ValueError("false-long loss requires paired dangerous_long_count labels.")
            dangerous_count = jnp.asarray(labels["dangerous_long_count"], dtype=jnp.float32)
            paired_count = jnp.asarray(labels["paired_trial_count"], dtype=jnp.float32)
            if dangerous_count.shape[-1] == len(candidate_horizons):
                dangerous_count = jnp.take(dangerous_count, long_index_array, axis=1)
                paired_count = jnp.take(paired_count, long_index_array, axis=1)
            dangerous_rate = dangerous_count / jnp.maximum(paired_count, 1.0)
            false_long_loss = _masked_mean(
                dangerous_rate * jax.nn.softplus(predictions["success_advantage"] / 0.02) * 0.02,
                paired_count > 0,
            )

        if weights.danger_rescue > 0:
            paired_count_raw = jnp.sum(pair_valid, axis=-1)
            dangerous_count_raw = jnp.sum(dangerous_pair, axis=-1)
            rescue_count_raw = jnp.sum(rescue_pair, axis=-1)
            if paired_distribution_mode:
                if "paired_outcome_logits" not in predictions:
                    raise ValueError("paired outcome loss requires paired_distribution_heads predictor outputs.")
                tie_count_raw = paired_count_raw - dangerous_count_raw - rescue_count_raw
                outcome_count = jnp.stack(
                    [dangerous_count_raw, tie_count_raw, rescue_count_raw],
                    axis=-1,
                )
                outcome_log_probability = jax.nn.log_softmax(predictions["paired_outcome_logits"], axis=-1)
                outcome_contribution = (
                    -outcome_count * outcome_log_probability / jnp.maximum(paired_count_raw[..., None], 1.0)
                )
                outcome_valid = paired_count_raw > 0
                paired_outcome_danger_nll = _root_equal_mean(outcome_contribution[..., 0], outcome_valid)
                paired_outcome_tie_nll = _root_equal_mean(outcome_contribution[..., 1], outcome_valid)
                paired_outcome_rescue_nll = _root_equal_mean(outcome_contribution[..., 2], outcome_valid)
                paired_outcome_multinomial_loss = (
                    paired_outcome_danger_nll + paired_outcome_tie_nll + paired_outcome_rescue_nll
                )
                paired_outcome_danger_rate = _root_equal_mean(
                    dangerous_count_raw / jnp.maximum(paired_count_raw, 1.0),
                    outcome_valid,
                )
                paired_outcome_tie_rate = _root_equal_mean(
                    tie_count_raw / jnp.maximum(paired_count_raw, 1.0),
                    outcome_valid,
                )
                paired_outcome_rescue_rate = _root_equal_mean(
                    rescue_count_raw / jnp.maximum(paired_count_raw, 1.0),
                    outcome_valid,
                )
            else:
                if "danger_logits" not in predictions or "rescue_logits" not in predictions:
                    raise ValueError("danger/rescue loss requires paired_advantage_heads predictor outputs.")
                danger_binomial_loss = _masked_mean(
                    _binomial_nll(predictions["danger_logits"], dangerous_count_raw, paired_count_raw)
                    / jnp.maximum(paired_count_raw, 1.0),
                    paired_count_raw > 0,
                )
                rescue_binomial_loss = _masked_mean(
                    _binomial_nll(predictions["rescue_logits"], rescue_count_raw, paired_count_raw)
                    / jnp.maximum(paired_count_raw, 1.0),
                    paired_count_raw > 0,
                )
                danger_rescue_loss = 0.5 * (danger_binomial_loss + rescue_binomial_loss)

        if weights.paired_elapsed > 0:
            paired_elapsed_error = predictions["elapsed_advantage"][..., None] - jnp.nan_to_num(paired_elapsed_delta)
            if paired_distribution_mode:
                if "paired_elapsed_scale" not in predictions:
                    raise ValueError("paired elapsed likelihood requires paired_distribution_heads outputs.")
                # Subtracting same-seed raw trials before evaluating the
                # likelihood retains paired covariance:
                # Var(T_long - T_ref) = Var(T_long) + Var(T_ref) - 2 Cov.
                paired_elapsed_student_t_loss = _root_equal_mean(
                    _student_t_nll(
                        paired_elapsed_error,
                        predictions["paired_elapsed_scale"][..., None],
                        degrees_of_freedom=4.0,
                    ),
                    pair_valid,
                )
                paired_count_raw = jnp.sum(pair_valid, axis=-1)
                paired_elapsed_scale_mean = _root_equal_mean(
                    predictions["paired_elapsed_scale"],
                    paired_count_raw > 0,
                )

                valid_float = pair_valid.astype(jnp.float32)
                safe_count = jnp.maximum(paired_count_raw, 1.0)
                reference_elapsed_broadcast = jnp.broadcast_to(reference_elapsed_raw, long_elapsed_raw.shape)
                reference_mean = (
                    jnp.sum(jnp.nan_to_num(reference_elapsed_broadcast) * valid_float, axis=-1) / safe_count
                )
                long_mean = jnp.sum(jnp.nan_to_num(long_elapsed_raw) * valid_float, axis=-1) / safe_count
                reference_centered = jnp.nan_to_num(reference_elapsed_broadcast) - reference_mean[..., None]
                long_centered = jnp.nan_to_num(long_elapsed_raw) - long_mean[..., None]
                covariance = jnp.sum(reference_centered * long_centered * valid_float, axis=-1) / jnp.maximum(
                    paired_count_raw - 1.0,
                    1.0,
                )
                delta_mean = jnp.sum(jnp.nan_to_num(paired_elapsed_delta) * valid_float, axis=-1) / safe_count
                delta_centered = jnp.nan_to_num(paired_elapsed_delta) - delta_mean[..., None]
                delta_variance = jnp.sum(jnp.square(delta_centered) * valid_float, axis=-1) / jnp.maximum(
                    paired_count_raw - 1.0,
                    1.0,
                )
                covariance_valid = paired_count_raw > 1
                paired_elapsed_covariance = _root_equal_mean(covariance, covariance_valid)
                paired_elapsed_delta_variance = _root_equal_mean(delta_variance, covariance_valid)
            else:
                scale = jnp.asarray(max(float(elapsed_advantage_scale), 1e-3), dtype=jnp.float32)
                paired_elapsed_loss = _masked_mean(_huber(paired_elapsed_error / scale), pair_valid)

        if weights.faster_long > 0:
            if "faster_long_logits" not in predictions:
                raise ValueError("faster-long loss requires paired_advantage_heads predictor outputs.")
            paired_count_raw = jnp.sum(pair_valid, axis=-1)
            faster_count = jnp.sum(pair_valid & (paired_elapsed_delta < 0.0), axis=-1)
            faster_long_loss = _masked_mean(
                _binomial_nll(predictions["faster_long_logits"], faster_count, paired_count_raw)
                / jnp.maximum(paired_count_raw, 1.0),
                paired_count_raw > 0,
            )

    metrics = {
        "success_bce": success_loss,
        "timeout_bce": timeout_loss,
        "remaining_calls_huber": calls_loss,
        "remaining_steps_huber": steps_loss,
        "final_risk_huber": final_risk_loss,
        "action_cot_risk_huber": cot_risk_loss,
        "fused_risk_huber": fused_risk_loss,
        "event_bce": event_loss,
        "raw_h_classification": raw_h_classification_loss,
        "raw_h_ordinal": raw_h_ordinal_loss,
        "ordered_listwise_nll": ordered_listwise_loss,
        "ordered_listwise_target_entropy": ordered_listwise_target_entropy,
        "ordered_listwise_target_horizon": ordered_listwise_target_horizon,
        "ordered_listwise_valid_fraction": ordered_listwise_valid_fraction,
        "survival_nll": survival_loss,
        "success_advantage_nll": success_advantage_loss,
        "elapsed_advantage_nll": elapsed_advantage_loss,
        "calls_advantage_nll": calls_advantage_loss,
        "false_long_penalty": false_long_loss,
        "danger_rescue_binomial": danger_rescue_loss,
        "danger_binomial": danger_binomial_loss,
        "rescue_binomial": rescue_binomial_loss,
        "paired_outcome_multinomial_nll": paired_outcome_multinomial_loss,
        "paired_outcome_danger_nll": paired_outcome_danger_nll,
        "paired_outcome_tie_nll": paired_outcome_tie_nll,
        "paired_outcome_rescue_nll": paired_outcome_rescue_nll,
        "paired_outcome_danger_rate": paired_outcome_danger_rate,
        "paired_outcome_tie_rate": paired_outcome_tie_rate,
        "paired_outcome_rescue_rate": paired_outcome_rescue_rate,
        "paired_elapsed_huber": paired_elapsed_loss,
        "paired_elapsed_student_t_nll": paired_elapsed_student_t_loss,
        "paired_elapsed_scale_mean": paired_elapsed_scale_mean,
        "paired_elapsed_covariance": paired_elapsed_covariance,
        "paired_elapsed_delta_variance": paired_elapsed_delta_variance,
        "faster_long_binomial": faster_long_loss,
    }
    total = (
        weights.success * success_loss
        + weights.timeout * timeout_loss
        + weights.remaining_calls * calls_loss
        + weights.remaining_steps * steps_loss
        + weights.final_risk * final_risk_loss
        + weights.action_cot_risk * cot_risk_loss
        + weights.fused_risk * fused_risk_loss
        + weights.event * event_loss
        + weights.raw_h_classification * raw_h_classification_loss
        + weights.raw_h_ordinal * raw_h_ordinal_loss
        + weights.ordered_listwise * ordered_listwise_loss
        + weights.survival * survival_loss
        + weights.success_advantage * success_advantage_loss
        + weights.elapsed_advantage * elapsed_advantage_loss
        + weights.calls_advantage * calls_advantage_loss
        + weights.false_long * false_long_loss
        + weights.danger_rescue * (paired_outcome_multinomial_loss if paired_distribution_mode else danger_rescue_loss)
        + weights.paired_elapsed * (paired_elapsed_student_t_loss if paired_distribution_mode else paired_elapsed_loss)
        + weights.faster_long * faster_long_loss
    )
    metrics["loss"] = total
    return total, metrics
