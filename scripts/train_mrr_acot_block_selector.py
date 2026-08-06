"""Train and test a deployable low-capacity MRR-ACoT block selector.

The frozen base+IR policy generates causal utility labels online for the 32
visual prefix blocks.  A label is the normalized action+EAR gap improvement
from refreshing exactly one visual block while retaining anchor language.

Selector inputs are deliberately separated from those privileged labels.  The
scorer sees only deployable pre-Gemma visual-token differences, low-resolution
RGB differences, block coordinates, normalized state delta, executed-action
summaries, and anchor-EAR summaries.  Fresh deep KV deltas and task outcomes
are never included in its feature tensor.

After listwise+pairwise training on the episode-disjoint 144/22 train/validation
split, the held-out 22-pair test reconstructs real top-4 composite KV caches and
reruns frozen IAR, EAR, and final action branches.  This remains a standalone
offline feasibility trainer and does not modify default inference.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import multirate_dataset

try:
    import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    import train_p3t_prefix_transport as p3t_trainer
except ImportError:  # pragma: no cover - supports python -m scripts...
    from scripts import probe_mrr_acot_sparse_refresh_oracle as mrr_oracle
    from scripts import train_p3t_prefix_transport as p3t_trainer


LOGGER = logging.getLogger("train_mrr_acot_block_selector")
TOP_K = 4
PROJECTION_RANK = 8


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    endpoint_student_params: str
    output_dir: str
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    dataset_task_id: int = 6
    seed: int = 7
    split_seed: int = 7
    feature_projection_seed: int = 2026
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    expected_pairs: int = 188
    expected_train_pairs: int = 144
    expected_validation_pairs: int = 22
    expected_test_pairs: int = 22
    coarse_flow_steps: int = 1
    final_flow_steps: int = 1
    delta_floor: float = 1e-6
    hidden_dim: int = 32
    train_steps: int = 1_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    listwise_temperature: float = 0.5
    listwise_loss_weight: float = 1.0
    pairwise_loss_weight: float = 1.0
    log_interval: int = 50
    early_stopping_patience_logs: int = 6
    early_stopping_min_delta: float = 1e-5
    action_closure_gate: float = 0.60
    ear_closure_gate: float = 0.50
    gripper_accuracy_gate: float = 0.95
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    integer_values = (
        args.seed,
        args.split_seed,
        args.feature_projection_seed,
        args.dataset_task_id,
    )
    if any(value < 0 for value in integer_values):
        raise ValueError("Task id and seeds must be non-negative.")
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("Validation/test fractions must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    expected = (
        args.expected_pairs,
        args.expected_train_pairs,
        args.expected_validation_pairs,
        args.expected_test_pairs,
    )
    if any(value <= 0 for value in expected):
        raise ValueError("Expected split sizes must be positive.")
    if args.expected_train_pairs + args.expected_validation_pairs + args.expected_test_pairs != args.expected_pairs:
        raise ValueError("Expected train/validation/test sizes must sum to expected_pairs.")
    if args.coarse_flow_steps <= 0 or args.final_flow_steps <= 0 or args.delta_floor <= 0.0:
        raise ValueError("Flow steps and delta_floor must be positive.")
    if args.hidden_dim <= 0 or args.train_steps <= 0 or args.batch_size <= 0:
        raise ValueError("Model/training sizes must be positive.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.gradient_clip_norm <= 0.0:
        raise ValueError("Optimizer scales are invalid.")
    if args.listwise_temperature <= 0.0:
        raise ValueError("listwise_temperature must be positive.")
    if args.listwise_loss_weight < 0.0 or args.pairwise_loss_weight < 0.0:
        raise ValueError("Ranking loss weights must be non-negative.")
    if args.listwise_loss_weight + args.pairwise_loss_weight <= 0.0:
        raise ValueError("At least one ranking loss weight must be positive.")
    if args.log_interval <= 0 or args.early_stopping_patience_logs < 0:
        raise ValueError("Logging/early-stopping settings are invalid.")
    if any(not 0.0 <= value <= 1.0 for value in (args.action_closure_gate, args.ear_closure_gate)):
        raise ValueError("Gap-closure gates must lie in [0, 1].")
    if not 0.0 <= args.gripper_accuracy_gate <= 1.0:
        raise ValueError("gripper_accuracy_gate must lie in [0, 1].")


class BlockSelectorScorer(nnx.Module):
    """Shared two-layer scorer applied independently to each of 32 blocks."""

    def __init__(self, feature_dim: int, hidden_dim: int, *, rngs: nnx.Rngs) -> None:
        self.input = nnx.Linear(feature_dim, hidden_dim, rngs=rngs, param_dtype=jnp.float32)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=jnp.float32)

    def __call__(self, features: jax.Array) -> jax.Array:
        return self.output(nnx.swish(self.input(features.astype(jnp.float32))))[..., 0]


def _rademacher_projection(key: jax.Array, input_dim: int, rank: int = PROJECTION_RANK) -> jax.Array:
    signs = 2.0 * jax.random.bernoulli(key, shape=(input_dim, rank)).astype(jnp.float32) - 1.0
    return signs / jnp.sqrt(jnp.asarray(input_dim, dtype=jnp.float32))


BLOCK_TOKEN_INDICES = jnp.asarray(
    np.stack([np.flatnonzero(np.asarray(mrr_oracle.BLOCK_MASK[block])) for block in range(mrr_oracle.NUM_BLOCKS)]),
    dtype=jnp.int32,
)


def _coordinate_features() -> jax.Array:
    rows = []
    for block_id in range(mrr_oracle.NUM_BLOCKS):
        view = block_id // mrr_oracle.BLOCKS_PER_VIEW
        local = block_id % mrr_oracle.BLOCKS_PER_VIEW
        row, col = local // 4, local % 4
        view_one_hot = np.eye(2, dtype=np.float32)[view]
        row_one_hot = np.eye(4, dtype=np.float32)[row]
        col_one_hot = np.eye(4, dtype=np.float32)[col]
        rows.append(
            np.concatenate(
                [
                    view_one_hot,
                    np.asarray([2.0 * row / 3.0 - 1.0, 2.0 * col / 3.0 - 1.0], dtype=np.float32),
                    row_one_hot,
                    col_one_hot,
                ]
            )
        )
    return jnp.asarray(np.stack(rows))


COORDINATE_FEATURES = _coordinate_features()


def _project_summaries(vectors: list[jax.Array], *, seed: int) -> jax.Array:
    projected = []
    norms = []
    root = jax.random.key(seed)
    for index, vector in enumerate(vectors):
        projection = _rademacher_projection(jax.random.fold_in(root, index), vector.shape[-1])
        projected.append(vector.astype(jnp.float32) @ projection)
        norms.append(jnp.sqrt(jnp.mean(jnp.square(vector.astype(jnp.float32))) + 1e-8)[None])
    return jnp.concatenate([*projected, *norms], axis=-1)


def _selector_features(
    anchor_visual_tokens: jax.Array,
    current_visual_tokens: jax.Array,
    anchor_images: jax.Array,
    current_images: jax.Array,
    state_delta: jax.Array,
    executed_actions: jax.Array,
    anchor_ear: jax.Array,
    *,
    projection_seed: int,
) -> jax.Array:
    """Construct features from an explicitly KV-free deployable boundary."""

    anchor_tokens = anchor_visual_tokens.astype(jnp.float32)
    current_tokens = current_visual_tokens.astype(jnp.float32)
    anchor_blocks = anchor_tokens[BLOCK_TOKEN_INDICES]
    current_blocks = current_tokens[BLOCK_TOKEN_INDICES]
    delta = current_blocks - anchor_blocks
    delta_mean = jnp.mean(delta, axis=1)
    delta_abs_mean = jnp.mean(jnp.abs(delta), axis=1)
    delta_rms_by_dim = jnp.sqrt(jnp.mean(jnp.square(delta), axis=1) + 1e-8)
    embedding_dim = delta.shape[-1]
    root = jax.random.key(projection_seed)
    local_projection = jnp.concatenate(
        [
            delta_mean @ _rademacher_projection(jax.random.fold_in(root, 0), embedding_dim),
            delta_abs_mean @ _rademacher_projection(jax.random.fold_in(root, 1), embedding_dim),
            delta_rms_by_dim @ _rademacher_projection(jax.random.fold_in(root, 2), embedding_dim),
        ],
        axis=-1,
    )
    dot = jnp.sum(anchor_blocks * current_blocks, axis=(1, 2))
    denominator = jnp.sqrt(
        jnp.sum(jnp.square(anchor_blocks), axis=(1, 2))
        * jnp.sum(jnp.square(current_blocks), axis=(1, 2))
        + 1e-8
    )
    local_statistics = jnp.stack(
        [
            jnp.mean(jnp.abs(delta), axis=(1, 2)),
            jnp.sqrt(jnp.mean(jnp.square(delta), axis=(1, 2)) + 1e-8),
            jnp.max(jnp.abs(delta), axis=(1, 2)),
            jnp.sqrt(jnp.mean(jnp.square(anchor_blocks), axis=(1, 2)) + 1e-8),
            jnp.sqrt(jnp.mean(jnp.square(current_blocks), axis=(1, 2)) + 1e-8),
            dot / jnp.maximum(denominator, 1e-8),
        ],
        axis=-1,
    )

    def image_tokens(images: jax.Array) -> jax.Array:
        return images.reshape(1, 2, 16, 4, 16, 4, 3).mean(axis=(3, 5)).reshape(
            mrr_oracle.VISUAL_TOKENS, 3
        )

    anchor_rgb = image_tokens(anchor_images)[BLOCK_TOKEN_INDICES]
    current_rgb = image_tokens(current_images)[BLOCK_TOKEN_INDICES]
    rgb_delta = current_rgb - anchor_rgb
    rgb_features = jnp.concatenate(
        [
            jnp.mean(rgb_delta, axis=1),
            jnp.mean(jnp.abs(rgb_delta), axis=1),
            jnp.mean(jnp.abs(rgb_delta), axis=(1, 2))[:, None],
            jnp.sqrt(jnp.mean(jnp.square(rgb_delta), axis=(1, 2)) + 1e-8)[:, None],
            jnp.max(jnp.abs(rgb_delta), axis=(1, 2))[:, None],
        ],
        axis=-1,
    )

    state_delta = state_delta.astype(jnp.float32)
    actions = executed_actions.astype(jnp.float32)
    anchor_ear = anchor_ear.astype(jnp.float32)
    global_vectors = [
        state_delta,
        jnp.mean(actions, axis=0),
        jnp.std(actions, axis=0),
        actions[0],
        actions[-1],
        jnp.mean(anchor_ear, axis=0),
        jnp.std(anchor_ear, axis=0),
        anchor_ear[0],
        anchor_ear[-1],
    ]
    global_features = _project_summaries(global_vectors, seed=projection_seed + 10_000)
    global_features = jnp.broadcast_to(global_features[None, :], (mrr_oracle.NUM_BLOCKS, global_features.size))
    return jnp.concatenate(
        [local_projection, local_statistics, rgb_features, COORDINATE_FEATURES, global_features],
        axis=-1,
    )


def _pair_context(base_model: Any, batch: dict[str, Any], rng: jax.Array, args: Args) -> dict[str, Any]:
    anchor_prefix = base_model.sample_actions_profile_prefix(rng, batch["anchor_observation"])
    fresh_prefix = base_model.sample_actions_profile_prefix(rng, batch["current_observation"])
    anchor_kv = anchor_prefix["kv_cache"]
    fresh_kv = fresh_prefix["kv_cache"]
    anchor_out = anchor_prefix["prefix_out"]

    baseline_kv = (
        jnp.concatenate([anchor_kv[0], fresh_kv[0]], axis=1),
        jnp.concatenate([anchor_kv[1], fresh_kv[1]], axis=1),
    )
    baseline_prefix = mrr_oracle._repeat_prefix_state(fresh_prefix, anchor_out, baseline_kv, 2)  # noqa: SLF001
    baseline_iar, baseline_ear, baseline_actions = p3t_trainer._reason_and_act(  # noqa: SLF001
        base_model,
        baseline_prefix,
        coarse_flow_steps=args.coarse_flow_steps,
        final_flow_steps=args.final_flow_steps,
    )
    target_iar = baseline_iar[1:2]
    target_ear = baseline_ear[1:2]
    target_actions = baseline_actions[1:2]
    baseline_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
        baseline_iar,
        baseline_ear,
        baseline_actions,
        target_iar,
        target_ear,
        target_actions,
    )

    single_kv = mrr_oracle._composite_cache(  # noqa: SLF001
        anchor_kv,
        fresh_kv,
        mrr_oracle.BLOCK_MASK,
        jnp.zeros((mrr_oracle.NUM_BLOCKS,), dtype=jnp.bool_),
    )
    single_prefix = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
        fresh_prefix,
        anchor_out,
        single_kv,
        mrr_oracle.NUM_BLOCKS,
    )
    single_iar, single_ear, single_actions = p3t_trainer._reason_and_act(  # noqa: SLF001
        base_model,
        single_prefix,
        coarse_flow_steps=args.coarse_flow_steps,
        final_flow_steps=args.final_flow_steps,
    )
    single_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
        single_iar,
        single_ear,
        single_actions,
        target_iar,
        target_ear,
        target_actions,
    )
    stale_action = jnp.maximum(baseline_metrics[0, 0], args.delta_floor)
    stale_ear = jnp.maximum(baseline_metrics[0, 1], args.delta_floor)
    utility = 0.5 * (
        (baseline_metrics[0, 0] - single_metrics[:, 0]) / stale_action
        + (baseline_metrics[0, 1] - single_metrics[:, 1]) / stale_ear
    )
    # This is the only selector-input boundary.  Deep fresh KV and all
    # downstream teacher outcomes stay in the privileged label branch above.
    features = _selector_features(
        anchor_prefix["prefix_tokens"][0, : mrr_oracle.VISUAL_TOKENS],
        fresh_prefix["prefix_tokens"][0, : mrr_oracle.VISUAL_TOKENS],
        p3t_trainer._low_resolution_images(anchor_prefix, 64),  # noqa: SLF001
        p3t_trainer._low_resolution_images(fresh_prefix, 64),  # noqa: SLF001
        fresh_prefix["observation"].state[0, :32] - anchor_prefix["observation"].state[0, :32],
        batch["executed_actions"][0],
        batch["anchor_ear"][0],
        projection_seed=args.feature_projection_seed,
    )
    return {
        "features": features,
        "utility": utility,
        "baseline_metrics": baseline_metrics,
        "anchor_prefix": anchor_prefix,
        "fresh_prefix": fresh_prefix,
        "anchor_kv": anchor_kv,
        "fresh_kv": fresh_kv,
        "target_iar": target_iar,
        "target_ear": target_ear,
        "target_actions": target_actions,
    }


def _make_label_generator(base_graphdef: Any, args: Args):
    @jax.jit
    def generate(base_state: nnx.State, batch: dict[str, Any], rng: jax.Array) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        context = _pair_context(base_model, batch, rng, args)
        return {
            "features": context["features"],
            "utility": context["utility"],
            "baseline_metrics": context["baseline_metrics"],
        }

    return generate


def _ranking_metrics(logits: jax.Array, utility: jax.Array) -> dict[str, jax.Array]:
    _, predicted = jax.lax.top_k(logits, TOP_K)
    _, oracle = jax.lax.top_k(utility, TOP_K)
    predicted_mask = jnp.sum(jax.nn.one_hot(predicted, mrr_oracle.NUM_BLOCKS), axis=1) > 0
    oracle_mask = jnp.sum(jax.nn.one_hot(oracle, mrr_oracle.NUM_BLOCKS), axis=1) > 0
    overlap = jnp.sum(predicted_mask & oracle_mask, axis=1) / TOP_K
    predicted_utility = jnp.take_along_axis(utility, predicted, axis=1).sum(axis=1)
    oracle_utility = jnp.take_along_axis(utility, oracle, axis=1).sum(axis=1)
    random_reference = TOP_K * jnp.mean(utility, axis=1)
    recovery = (predicted_utility - random_reference) / jnp.maximum(
        oracle_utility - random_reference,
        1e-6,
    )
    return {
        "top4_overlap": jnp.mean(overlap),
        "utility_gain_recovery": jnp.mean(recovery),
        "predicted_top4_utility": jnp.mean(predicted_utility),
        "oracle_top4_utility": jnp.mean(oracle_utility),
    }


def _selector_loss(
    scorer: BlockSelectorScorer,
    features: jax.Array,
    utility: jax.Array,
    *,
    args: Args,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    logits = scorer(features)
    centered = utility - jnp.mean(utility, axis=1, keepdims=True)
    standardized = centered / jnp.maximum(jnp.std(centered, axis=1, keepdims=True), 1e-4)
    target_distribution = jax.nn.softmax(standardized / args.listwise_temperature, axis=1)
    listwise = -jnp.mean(jnp.sum(target_distribution * jax.nn.log_softmax(logits, axis=1), axis=1))

    _, oracle = jax.lax.top_k(utility, TOP_K)
    positive_logits = jnp.take_along_axis(logits, oracle, axis=1)
    oracle_mask = jnp.sum(jax.nn.one_hot(oracle, mrr_oracle.NUM_BLOCKS), axis=1) > 0
    differences = positive_logits[:, :, None] - logits[:, None, :]
    negative_mask = (~oracle_mask)[:, None, :].astype(jnp.float32)
    pairwise = jnp.sum(jax.nn.softplus(-differences) * negative_mask) / jnp.maximum(
        jnp.sum(negative_mask) * TOP_K,
        1.0,
    )
    loss = args.listwise_loss_weight * listwise + args.pairwise_loss_weight * pairwise
    return loss, {
        "loss": loss,
        "listwise_loss": listwise,
        "pairwise_loss": pairwise,
        **_ranking_metrics(logits, utility),
    }


def _make_train_steps(adapter_graphdef: Any, optimizer: optax.GradientTransformation, args: Args):
    @jax.jit
    def train_step(
        params: nnx.State,
        optimizer_state: optax.OptState,
        features: jax.Array,
        utility: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        scorer = nnx.merge(adapter_graphdef, params)

        def loss_fn(candidate: BlockSelectorScorer):
            return _selector_loss(candidate, features, utility, args=args)

        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(scorer)
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_optimizer_state, {
            **metrics,
            "loss": loss,
            "gradient_norm": optax.global_norm(gradients),
        }

    @jax.jit
    def validation_step(params: nnx.State, features: jax.Array, utility: jax.Array) -> dict[str, jax.Array]:
        scorer = nnx.merge(adapter_graphdef, params)
        _, metrics = _selector_loss(scorer, features, utility, args=args)
        return metrics

    return train_step, validation_step


def _make_test_evaluator(base_graphdef: Any, scorer_graphdef: Any, args: Args):
    @jax.jit
    def evaluate(
        base_state: nnx.State,
        scorer_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
        feature_mean: jax.Array,
        feature_std: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        scorer = nnx.merge(scorer_graphdef, scorer_params)
        context = _pair_context(base_model, batch, rng, args)
        normalized = (context["features"] - feature_mean) / feature_std
        logits = scorer(normalized[None])[0]
        _, learned_ids = jax.lax.top_k(logits, TOP_K)
        _, oracle_ids = jax.lax.top_k(context["utility"], TOP_K)
        selected_masks = jnp.stack(
            [
                mrr_oracle._selected_visual_mask(learned_ids),  # noqa: SLF001
                mrr_oracle._selected_visual_mask(oracle_ids),  # noqa: SLF001
            ]
        )
        selected_kv = mrr_oracle._composite_cache(  # noqa: SLF001
            context["anchor_kv"],
            context["fresh_kv"],
            selected_masks,
            jnp.zeros((2,), dtype=jnp.bool_),
        )
        selected_prefix = mrr_oracle._repeat_prefix_state(  # noqa: SLF001
            context["fresh_prefix"],
            context["anchor_prefix"]["prefix_out"],
            selected_kv,
            2,
        )
        selected_iar, selected_ear, selected_actions = p3t_trainer._reason_and_act(  # noqa: SLF001
            base_model,
            selected_prefix,
            coarse_flow_steps=args.coarse_flow_steps,
            final_flow_steps=args.final_flow_steps,
        )
        selected_metrics = mrr_oracle._downstream_metrics(  # noqa: SLF001
            selected_iar,
            selected_ear,
            selected_actions,
            context["target_iar"],
            context["target_ear"],
            context["target_actions"],
        )
        return {
            "baseline_metrics": context["baseline_metrics"],
            "selected_metrics": selected_metrics,
            "utility": context["utility"],
            "logits": logits,
            "learned_ids": learned_ids,
            "oracle_ids": oracle_ids,
        }

    return evaluate


def _generate_partition(
    generator: Any,
    base_state: nnx.State,
    records: list[p3t_trainer.MaterializedPair],
    indices: np.ndarray,
    pairs: p3t_trainer.PairIndices,
    *,
    seed: int,
    label: str,
    started: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    utilities = []
    for position, record_index in enumerate(indices):
        batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
        anchor = int(pairs.anchor_indices[record_index])
        key = jax.random.fold_in(jax.random.key(seed), anchor)
        output = jax.device_get(generator(base_state, batch, key))
        features.append(np.asarray(output["features"], dtype=np.float32))
        utilities.append(np.asarray(output["utility"], dtype=np.float32))
        if position == 0 or (position + 1) % 10 == 0 or position + 1 == indices.size:
            LOGGER.info(
                "Generated %s causal labels %d/%d elapsed=%.1fs",
                label,
                position + 1,
                indices.size,
                time.monotonic() - started,
            )
    return np.stack(features), np.stack(utilities)


def _ranking_summary(logits: np.ndarray, utility: np.ndarray) -> dict[str, float]:
    learned_ids = np.argpartition(logits, -TOP_K, axis=1)[:, -TOP_K:]
    oracle_ids = np.argpartition(utility, -TOP_K, axis=1)[:, -TOP_K:]
    overlap = []
    learned_utility = []
    oracle_utility = []
    recovery = []
    for row in range(utility.shape[0]):
        learned = set(int(value) for value in learned_ids[row])
        oracle = set(int(value) for value in oracle_ids[row])
        overlap.append(len(learned & oracle) / TOP_K)
        learned_sum = float(np.sum(utility[row, list(learned)]))
        oracle_sum = float(np.sum(utility[row, list(oracle)]))
        reference = TOP_K * float(np.mean(utility[row]))
        learned_utility.append(learned_sum)
        oracle_utility.append(oracle_sum)
        recovery.append((learned_sum - reference) / max(oracle_sum - reference, 1e-6))
    return {
        "top4_overlap": float(np.mean(overlap)),
        "utility_gain_recovery": float(np.mean(recovery)),
        "learned_top4_utility": float(np.mean(learned_utility)),
        "oracle_top4_utility": float(np.mean(oracle_utility)),
    }


def _save_params(params: nnx.State, target: pathlib.Path, *, overwrite: bool) -> pathlib.Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {"mrr_acot_block_selector": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target.resolve(), item, force=overwrite)
    return target.resolve()


def _aggregate_test(per_pair: list[dict[str, Any]], args: Args) -> dict[str, Any]:
    stale = np.stack([record["baseline_metrics"][0] for record in per_pair])
    fresh = np.stack([record["baseline_metrics"][1] for record in per_pair])
    learned = np.stack([record["selected_metrics"][0] for record in per_pair])
    oracle = np.stack([record["selected_metrics"][1] for record in per_pair])

    def summarize(values: np.ndarray) -> dict[str, Any]:
        mean = np.mean(values, axis=0)
        closure = {
            name: float(1.0 - mean[index] / max(np.mean(stale, axis=0)[index], 1e-12))
            for index, name in enumerate(mrr_oracle.METRIC_NAMES[:3])
        }
        return {"metrics": mrr_oracle._metric_dict(mean), "global_gap_closure_vs_stale": closure}  # noqa: SLF001

    learned_summary = summarize(learned)
    oracle_summary = summarize(oracle)
    utility = np.stack([record["utility"] for record in per_pair])
    logits = np.stack([record["logits"] for record in per_pair])
    ranking = _ranking_summary(logits, utility)
    learned_closure = learned_summary["global_gap_closure_vs_stale"]
    oracle_closure = oracle_summary["global_gap_closure_vs_stale"]
    oracle_recovery = {
        "top4_overlap": ranking["top4_overlap"],
        "utility_gain_recovery": ranking["utility_gain_recovery"],
        "action_closure_recovery": learned_closure["action_mse_7d"]
        / max(oracle_closure["action_mse_7d"], 1e-12),
        "ear_closure_recovery": learned_closure["ear_mse_7d"] / max(oracle_closure["ear_mse_7d"], 1e-12),
        "iar_closure_recovery": learned_closure["iar_mse"] / max(oracle_closure["iar_mse"], 1e-12),
    }
    checks = {
        "action_gap_closure": learned_closure["action_mse_7d"] >= args.action_closure_gate,
        "ear_gap_closure": learned_closure["ear_mse_7d"] >= args.ear_closure_gate,
        "gripper_sign_accuracy": learned_summary["metrics"]["gripper_sign_accuracy"]
        >= args.gripper_accuracy_gate,
    }
    return {
        "num_pairs": len(per_pair),
        "stale": summarize(stale),
        "fresh": summarize(fresh),
        "learned_top4": learned_summary,
        "causal_oracle_top4": oracle_summary,
        "ranking": ranking,
        "oracle_recovery": oracle_recovery,
        "gate": {
            "thresholds": {
                "action_gap_closure": args.action_closure_gate,
                "ear_gap_closure": args.ear_closure_gate,
                "gripper_sign_accuracy": args.gripper_accuracy_gate,
            },
            "checks": checks,
            "pass": all(checks.values()),
        },
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = p3t_trainer._gpu()  # noqa: SLF001
    output_dir = pathlib.Path(args.output_dir).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    test_pairs_path = output_dir / "test_pairs.jsonl"
    summary_path = output_dir / "summary.json"
    params_path = output_dir / "final" / "params"
    preprocessing_path = output_dir / "final" / "feature_preprocessing.npz"
    selector_config_path = output_dir / "final" / "selector_config.json"
    targets = (metrics_path, test_pairs_path, summary_path, params_path, preprocessing_path, selector_config_path)
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"MRR selector output already exists in {output_dir}; pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    trainer_args = p3t_trainer.Args(
        dataset=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        endpoint_student_params=args.endpoint_student_params,
        config_name=args.config_name,
        dataset_task_id=args.dataset_task_id,
        temporal_stride=10,
        seed=args.seed,
        split_seed=args.split_seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    arrays = multirate_dataset.load_multirate_arrays(
        args.dataset,
        fields=("anchor_index", "task_id", "episode_id", "frame_id", "fresh_ear"),
    )
    base_graphdef, base_state, observation_dataset, raw_dataset, model_metadata = (
        p3t_trainer._load_model_and_dataset(trainer_args)  # noqa: SLF001
    )
    pairs = p3t_trainer._select_pairs(  # noqa: SLF001
        arrays,
        raw_dataset,
        task_id=args.dataset_task_id,
        temporal_stride=10,
        maximum_pairs=200,
        seed=args.seed,
    )
    train_indices, validation_indices, test_indices = p3t_trainer._split_pairs(  # noqa: SLF001
        pairs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    actual_sizes = (len(pairs), train_indices.size, validation_indices.size, test_indices.size)
    expected_sizes = (
        args.expected_pairs,
        args.expected_train_pairs,
        args.expected_validation_pairs,
        args.expected_test_pairs,
    )
    if actual_sizes != expected_sizes:
        raise ValueError(f"Expected P3T split {expected_sizes}, got {actual_sizes}.")
    records = p3t_trainer._materialize_pairs(  # noqa: SLF001
        pairs,
        arrays,
        observation_dataset,
        action_dim=int(model_metadata["action_dim"]),
        temporal_stride=10,
    )

    generator = _make_label_generator(base_graphdef, args)
    train_features, train_utility = _generate_partition(
        generator,
        base_state,
        records,
        train_indices,
        pairs,
        seed=args.seed,
        label="train",
        started=started,
    )
    validation_features, validation_utility = _generate_partition(
        generator,
        base_state,
        records,
        validation_indices,
        pairs,
        seed=args.seed,
        label="validation",
        started=started,
    )
    feature_mean = np.mean(train_features, axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.std(train_features, axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = np.maximum(feature_std, 1e-4)
    train_features = (train_features - feature_mean[None, None]) / feature_std[None, None]
    validation_features = (validation_features - feature_mean[None, None]) / feature_std[None, None]
    feature_dim = int(train_features.shape[-1])

    scorer = BlockSelectorScorer(feature_dim, args.hidden_dim, rngs=nnx.Rngs(args.seed))
    scorer_graphdef, params = nnx.split(scorer)
    schedule = optax.cosine_decay_schedule(args.learning_rate, args.train_steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(params)
    train_step, validation_step = _make_train_steps(scorer_graphdef, optimizer, args)
    rng = np.random.default_rng(args.seed)
    validation_batch = (
        jnp.asarray(validation_features),
        jnp.asarray(validation_utility),
    )
    best_params: nnx.State | None = None
    best_validation_loss = float("inf")
    best_step = 0
    stale_logs = 0
    completed_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            selected = rng.choice(
                train_indices.size,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            params, optimizer_state, train_metrics = train_step(
                params,
                optimizer_state,
                jnp.asarray(train_features[selected]),
                jnp.asarray(train_utility[selected]),
            )
            completed_steps = step
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_metrics = validation_step(params, *validation_batch)
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{
                        f"train/{name}": float(value)
                        for name, value in jax.device_get(train_metrics).items()
                    },
                    **{
                        f"validation/{name}": float(value)
                        for name, value in jax.device_get(validation_metrics).items()
                    },
                }
                if any(isinstance(value, float) and not np.isfinite(value) for value in record.values()):
                    raise FloatingPointError(f"Non-finite MRR selector metrics: {record}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                validation_loss = record["validation/loss"]
                if validation_loss < best_validation_loss - args.early_stopping_min_delta:
                    best_validation_loss = validation_loss
                    best_step = step
                    best_params = params
                    stale_logs = 0
                else:
                    stale_logs += 1
                if args.early_stopping_patience_logs and stale_logs >= args.early_stopping_patience_logs:
                    LOGGER.info("Early stopping at step %d; best step=%d", step, best_step)
                    break
    selected_params = params if best_params is None else best_params
    _save_params(selected_params, params_path, overwrite=args.overwrite)
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preprocessing_path, feature_mean=feature_mean, feature_std=feature_std)
    selector_config = {
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "top_k": TOP_K,
        "feature_projection_seed": args.feature_projection_seed,
        "projection_rank": PROJECTION_RANK,
        "input_contract": [
            "current-vs-anchor pre-Gemma visual prefix token statistics and fixed projections",
            "64x64 RGB block deltas",
            "view and block coordinates",
            "normalized state delta",
            "executed-action summaries",
            "anchor-EAR summaries",
        ],
        "forbidden_inputs": ["fresh deep KV delta", "teacher outcome"],
    }
    selector_config_path.write_text(json.dumps(selector_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    scorer_model = nnx.merge(scorer_graphdef, selected_params)
    train_logits = np.asarray(scorer_model(jnp.asarray(train_features)))
    validation_logits = np.asarray(scorer_model(jnp.asarray(validation_features)))
    test_evaluator = _make_test_evaluator(base_graphdef, scorer_graphdef, args)
    per_pair: list[dict[str, Any]] = []
    with test_pairs_path.open("w", encoding="utf-8") as test_file:
        for position, record_index in enumerate(test_indices):
            batch = p3t_trainer._batch(records, np.asarray([record_index], dtype=np.int64))  # noqa: SLF001
            anchor = int(pairs.anchor_indices[record_index])
            key = jax.random.fold_in(jax.random.key(args.seed), anchor)
            output = jax.device_get(
                test_evaluator(
                    base_state,
                    selected_params,
                    batch,
                    key,
                    jnp.asarray(feature_mean),
                    jnp.asarray(feature_std),
                )
            )
            baseline = np.asarray(output["baseline_metrics"], dtype=np.float64)
            selected = np.asarray(output["selected_metrics"], dtype=np.float64)
            utility = np.asarray(output["utility"], dtype=np.float64)
            logits = np.asarray(output["logits"], dtype=np.float64)
            learned_ids = [int(value) for value in np.asarray(output["learned_ids"])]
            oracle_ids = [int(value) for value in np.asarray(output["oracle_ids"])]
            pair_record = {
                "pair_index": int(record_index),
                "anchor_index": anchor,
                "target_index": int(pairs.target_indices[record_index]),
                "episode_id": int(pairs.episode_ids[record_index]),
                "frame_id": int(pairs.frame_ids[record_index]),
                "baseline_metrics": baseline.tolist(),
                "selected_metrics": selected.tolist(),
                "stale": mrr_oracle._metric_dict(baseline[0]),  # noqa: SLF001
                "fresh": mrr_oracle._metric_dict(baseline[1]),  # noqa: SLF001
                "learned_top4": {
                    "ids": learned_ids,
                    "names": [mrr_oracle._block_name(block_id) for block_id in learned_ids],  # noqa: SLF001
                    "metrics": mrr_oracle._metric_dict(selected[0]),  # noqa: SLF001
                },
                "causal_oracle_top4": {
                    "ids": oracle_ids,
                    "names": [mrr_oracle._block_name(block_id) for block_id in oracle_ids],  # noqa: SLF001
                    "metrics": mrr_oracle._metric_dict(selected[1]),  # noqa: SLF001
                },
                "utility": utility.tolist(),
                "logits": logits.tolist(),
            }
            test_file.write(json.dumps(pair_record, sort_keys=True, allow_nan=False) + "\n")
            test_file.flush()
            per_pair.append(pair_record)
            LOGGER.info(
                "Test composite %d/%d anchor=%d elapsed=%.1fs",
                position + 1,
                test_indices.size,
                anchor,
                time.monotonic() - started,
            )

    parameter_count = int(sum(np.prod(value.shape) for value in jax.tree.leaves(selected_params)))
    summary = {
        "method": "MRR-ACoT deployable learned top-4 block selector",
        "status": "offline_feasibility_only",
        "device": str(device),
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "split": {
            "pairs": len(pairs),
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
            "train_episodes": sorted(int(value) for value in np.unique(pairs.episode_ids[train_indices])),
            "validation_episodes": sorted(
                int(value) for value in np.unique(pairs.episode_ids[validation_indices])
            ),
            "test_episodes": sorted(int(value) for value in np.unique(pairs.episode_ids[test_indices])),
        },
        "feature_contract": selector_config,
        "training": {
            "completed_steps": completed_steps,
            "best_step": best_step,
            "best_validation_loss": best_validation_loss,
            "trainable_parameter_count": parameter_count,
            "train_ranking": _ranking_summary(train_logits, train_utility),
            "validation_ranking": _ranking_summary(validation_logits, validation_utility),
        },
        "test": _aggregate_test(per_pair, args),
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "params": str(params_path),
            "preprocessing": str(preprocessing_path),
            "selector_config": str(selector_config_path),
            "metrics": str(metrics_path),
            "test_pairs": str(test_pairs_path),
            "summary": str(summary_path),
        },
        "note": "Open-loop causal-gap feasibility only; passing the gate still requires Task8 closed-loop success.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
