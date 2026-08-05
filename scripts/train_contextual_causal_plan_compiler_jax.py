#!/usr/bin/env python3
"""Train a GPU-native contextual causal plan compiler.

The compiler replaces the one-step final 300M action-expert suffix while
retaining an explicit Action-CoT bottleneck.  It consumes only tensors that
already exist before the final suffix at deployment time:

* the one-step EAR trajectory and final-action noise;
* current-observation IAR tokens extracted from the shared prefix KV cache;
* a pooled current-observation VLM prefix feature; and
* normalized current proprioception.

Context cannot create an additive action path on its own.  A train-only PCA
context code and the EAR enter a low-rank bilinear interaction which FiLM
modulates an EAR/noise transport.  The prediction is a bounded residual over a
frozen train-only ridge anchor.  Semantic EAR interventions share context and
noise with the corresponding clean record, so response alignment measures the
functional role of the explicit plan rather than observation leakage.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import math
import os
import pathlib
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import h5py
import numpy as np
import optax
import tyro

from openpi.action_cot import endpoint_dataset


@dataclasses.dataclass(frozen=True)
class Args:
    target_dataset: tuple[str, ...]
    context_dataset: tuple[str, ...]
    output_dir: str
    branch_dataset: tuple[str, ...] = ()
    branch_pretrain_steps: int = 500
    branch_validation_fraction: float = 0.1
    branch_learning_rate: float = 1e-4
    seed: int = 7
    validation_fraction: float = 0.1
    active_action_dim: int = 7
    steps: int = 1_500
    batch_size: int = 256
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    warmup_steps: int = 50
    weight_decay: float = 1e-3
    gradient_clip_norm: float = 1.0
    log_interval: int = 25
    ridge_lambda: float = 0.01
    pca_dim: int = 128
    pca_oversample: int = 16
    pca_power_iterations: int = 1
    interaction_rank: int = 32
    hidden_dim: int = 128
    residual_scale: float = 0.25
    clean_loss_weight: float = 1.0
    intervention_loss_weight: float = 1.0
    response_loss_weight: float = 1.0
    response_cosine_loss_weight: float = 0.1
    context_dropout: float = 0.05
    clean_nondegradation_tolerance: float = 0.05
    intervention_nondegradation_tolerance: float = 0.05
    required_response_improvement: float = 0.10
    required_response_cosine: float = 0.40
    required_context_shuffle_ratio: float = 1.05
    required_ear_shuffle_ratio: float = 5.0
    max_latency_p95_ms: float = 1.0
    latency_warmup: int = 25
    latency_runs: int = 200
    allow_cpu: bool = False
    overwrite: bool = False


def _validate_args(args: Args) -> None:
    for name in (
        "steps",
        "batch_size",
        "warmup_steps",
        "log_interval",
        "pca_dim",
        "pca_oversample",
        "interaction_rank",
        "hidden_dim",
        "latency_warmup",
        "latency_runs",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.pca_power_iterations < 0:
        raise ValueError("--pca-power-iterations must be non-negative.")
    if args.branch_pretrain_steps < 0:
        raise ValueError("--branch-pretrain-steps must be non-negative.")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if not 0.0 < args.branch_validation_fraction < 0.5:
        raise ValueError("--branch-validation-fraction must be in (0, 0.5).")
    if args.active_action_dim <= 0:
        raise ValueError("--active-action-dim must be positive.")
    if args.learning_rate <= 0 or args.final_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and the peak must be positive.")
    if args.branch_learning_rate <= 0:
        raise ValueError("--branch-learning-rate must be positive.")
    if args.ridge_lambda < 0 or args.weight_decay < 0:
        raise ValueError("Ridge lambda and weight decay must be non-negative.")
    if args.gradient_clip_norm <= 0 or args.residual_scale <= 0:
        raise ValueError("Gradient clipping and residual scale must be positive.")
    if not 0.0 <= args.context_dropout < 1.0:
        raise ValueError("--context-dropout must be in [0, 1).")
    for name in (
        "clean_loss_weight",
        "intervention_loss_weight",
        "response_loss_weight",
        "response_cosine_loss_weight",
        "clean_nondegradation_tolerance",
        "intervention_nondegradation_tolerance",
        "required_response_improvement",
        "required_context_shuffle_ratio",
        "required_ear_shuffle_ratio",
        "max_latency_p95_ms",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if not -1.0 <= args.required_response_cosine <= 1.0:
        raise ValueError("--required-response-cosine must be in [-1, 1].")


def _select_device(*, allow_cpu: bool) -> jax.Device:
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if gpu_devices:
        return gpu_devices[0]
    if not allow_cpu:
        raise RuntimeError(
            "No JAX GPU is visible. This trainer refuses silent CPU fallback; "
            "pass --allow-cpu only for an explicit diagnostic."
        )
    return jax.devices()[0]


def _record_keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[int, int]]:
    return list(
        zip(
            np.asarray(arrays["dataset_index"], dtype=np.int64).tolist(),
            np.asarray(arrays["policy_seed"], dtype=np.int64).tolist(),
            strict=True,
        )
    )


def _join_target_and_context(
    target: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_keys = _record_keys(target)
    context_keys = _record_keys(context)
    if len(set(target_keys)) != len(target_keys):
        raise ValueError("Target dataset has duplicate (dataset_index, policy_seed) keys.")
    if len(set(context_keys)) != len(context_keys):
        raise ValueError("Context dataset has duplicate (dataset_index, policy_seed) keys.")
    target_set = set(target_keys)
    context_set = set(context_keys)
    if target_set != context_set:
        missing = sorted(target_set - context_set)[:10]
        extra = sorted(context_set - target_set)[:10]
        raise ValueError(
            "Target/context key sets must match exactly; "
            f"missing_context={missing}, extra_context={extra}, "
            f"target_count={len(target_set)}, context_count={len(context_set)}."
        )
    context_lookup = {key: index for index, key in enumerate(context_keys)}
    order = np.asarray([context_lookup[key] for key in target_keys], dtype=np.int64)
    for name in ("task_id", "episode_id", "frame_id"):
        left = np.asarray(target[name])
        right = np.asarray(context[name])[order]
        mismatch = np.flatnonzero(left != right)
        if mismatch.size:
            raise ValueError(
                f"Target/context {name} mismatch after exact key join at rows {mismatch[:10].tolist()}."
            )
    joined = {
        "deployment_prefix_feature": np.asarray(
            context["deployment_prefix_feature"][order], dtype=np.float32
        ),
        "deployment_iar": np.asarray(context["deployment_iar"][order], dtype=np.float32),
        "deployment_state": np.asarray(context["deployment_state"][order], dtype=np.float32),
    }
    return joined, {
        "join_key": ["dataset_index", "policy_seed"],
        "target_records": len(target_keys),
        "context_records": len(context_keys),
        "matched_records": len(order),
        "exact_key_set_match": True,
        "metadata_fields_verified": ["task_id", "episode_id", "frame_id"],
    }


def _semantic_mask(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    null_id = endpoint_dataset.INTERVENTION_IDS["null"]
    return np.asarray(arrays["intervention_valid"], dtype=np.bool_) & (
        np.asarray(arrays["intervention_ids"]) != null_id
    )


def _split_indices(
    arrays: Mapping[str, np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.any(_semantic_mask(arrays), axis=1)
    indices = np.flatnonzero(eligible)
    if indices.size < 2:
        raise ValueError("Fewer than two records contain a semantic intervention.")
    task = np.asarray(arrays["task_id"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    groups = task * np.int64(1_000_000_000) + episode
    unique_groups = np.unique(groups[indices])
    rng = np.random.default_rng(seed)
    if unique_groups.size < 2:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        count = min(max(1, round(shuffled.size * validation_fraction)), shuffled.size - 1)
        return np.sort(shuffled[count:]), np.sort(shuffled[:count])
    rng.shuffle(unique_groups)
    count = min(max(1, round(unique_groups.size * validation_fraction)), unique_groups.size - 1)
    validation_groups = unique_groups[:count]
    validation_mask = eligible & np.isin(groups, validation_groups)
    train = np.flatnonzero(eligible & ~validation_mask)
    validation = np.flatnonzero(validation_mask)
    if not train.size or not validation.size:
        raise ValueError("Episode-grouped split produced an empty partition.")
    return train, validation


def _all_pairs(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    semantic = _semantic_mask(arrays)
    pair_rows: list[int] = []
    pair_slots: list[int] = []
    pair_ids: list[int] = []
    intervention_ids = np.asarray(arrays["intervention_ids"])
    for row in rows:
        for slot in np.flatnonzero(semantic[int(row)]):
            pair_rows.append(int(row))
            pair_slots.append(int(slot))
            pair_ids.append(int(intervention_ids[int(row), int(slot)]))
    if not pair_rows:
        raise ValueError("The selected split contains no semantic intervention pairs.")
    return (
        np.asarray(pair_rows, dtype=np.int32),
        np.asarray(pair_slots, dtype=np.int32),
        np.asarray(pair_ids, dtype=np.int32),
    )


def _flatten_active(values: np.ndarray, active_dim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] < active_dim:
        raise ValueError(f"Tensor last dimension {values.shape[-1]} is below active_dim={active_dim}.")
    return np.ascontiguousarray(values[..., :active_dim].reshape((*values.shape[:-2], -1)))


def _fit_ridge(
    arrays: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
    train_pair_rows: np.ndarray,
    train_pair_slots: np.ndarray,
    *,
    active_dim: int,
    ridge_lambda: float,
) -> dict[str, np.ndarray]:
    clean_plan = _flatten_active(np.asarray(arrays["clean_coarse"])[train_rows], active_dim)
    clean_noise = _flatten_active(np.asarray(arrays["action_noise"])[train_rows], active_dim)
    clean_target = _flatten_active(np.asarray(arrays["clean_actions"])[train_rows], active_dim)
    intervention_plan = _flatten_active(
        np.asarray(arrays["intervention_coarse"])[train_pair_rows, train_pair_slots],
        active_dim,
    )
    intervention_noise = _flatten_active(
        np.asarray(arrays["action_noise"])[train_pair_rows], active_dim
    )
    intervention_target = _flatten_active(
        np.asarray(arrays["intervention_actions"])[train_pair_rows, train_pair_slots],
        active_dim,
    )
    features = np.concatenate(
        (
            np.concatenate((clean_plan, clean_noise), axis=-1),
            np.concatenate((intervention_plan, intervention_noise), axis=-1),
        ),
        axis=0,
    ).astype(np.float64)
    targets = np.concatenate((clean_target, intervention_target), axis=0).astype(np.float64)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    target_mean = targets.mean(axis=0)
    normalized = (features - feature_mean) / feature_std
    centered_target = targets - target_mean
    count = float(normalized.shape[0])
    gram = normalized.T @ normalized / count
    rhs = normalized.T @ centered_target / count
    weights = np.linalg.solve(
        gram + ridge_lambda * np.eye(gram.shape[0], dtype=np.float64), rhs
    )
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "weights": weights.astype(np.float32),
    }


def _raw_context(
    context: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    prefix = np.asarray(context["deployment_prefix_feature"], dtype=np.float32)
    iar = np.asarray(context["deployment_iar"], dtype=np.float32).reshape((prefix.shape[0], -1))
    state = np.asarray(context["deployment_state"], dtype=np.float32)
    if iar.shape[0] != prefix.shape[0] or state.shape[0] != prefix.shape[0]:
        raise ValueError("Deployment context fields have inconsistent record counts.")
    prefix_end = prefix.shape[1]
    iar_end = prefix_end + iar.shape[1]
    state_end = iar_end + state.shape[1]
    values = np.ascontiguousarray(np.concatenate((prefix, iar, state), axis=-1))
    return values, {
        "prefix": (0, prefix_end),
        "iar": (prefix_end, iar_end),
        "state": (iar_end, state_end),
    }


def _discover_branch_shards(inputs: tuple[str, ...]) -> list[pathlib.Path]:
    shards: list[pathlib.Path] = []
    for item in inputs:
        path = pathlib.Path(item)
        if path.is_dir():
            shards.extend(sorted(path.glob("shard-*.h5")))
        elif path.suffix in {".h5", ".hdf5"}:
            shards.append(path)
        else:
            raise FileNotFoundError(f"Branch input is not an HDF5 shard/directory: {path}")
    unique = list(dict.fromkeys(path.resolve() for path in shards))
    if not unique:
        raise FileNotFoundError(f"No branch HDF5 shards found under {inputs}.")
    return unique


def _load_branch_arrays(inputs: tuple[str, ...]) -> dict[str, np.ndarray]:
    fields = (
        "task_id",
        "episode_id",
        "root_id",
        "policy_seed",
        "branch_ids",
        "branch_valid",
        "current_state",
        "fresh_ear",
        "fresh_iar",
        "fresh_actions",
    )
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    for shard in _discover_branch_shards(inputs):
        with h5py.File(shard, "r") as handle:
            missing = [name for name in fields if name not in handle]
            if missing:
                raise KeyError(f"Branch shard {shard} is missing fields {missing}.")
            for name in fields:
                pieces[name].append(handle[name][:])
    return {name: np.concatenate(values, axis=0) for name, values in pieces.items()}


def _branch_root_split(
    arrays: Mapping[str, np.ndarray],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = list(
        zip(
            np.asarray(arrays["task_id"], dtype=np.int64).tolist(),
            np.asarray(arrays["episode_id"], dtype=np.int64).tolist(),
            np.asarray(arrays["root_id"], dtype=np.int64).tolist(),
            strict=True,
        )
    )
    unique = list(dict.fromkeys(keys))
    if len(unique) < 2:
        raise ValueError("Branch augmentation has fewer than two root groups.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = min(max(1, round(len(unique) * validation_fraction)), len(unique) - 1)
    validation_keys = set(unique[:count])
    validation = np.asarray(
        [index for index, key in enumerate(keys) if key in validation_keys], dtype=np.int32
    )
    train = np.asarray(
        [index for index, key in enumerate(keys) if key not in validation_keys], dtype=np.int32
    )
    return train, validation


def _action_noise_from_seeds(
    seeds: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    device: jax.Device,
) -> np.ndarray:
    seeds_device = jax.device_put(np.asarray(seeds, dtype=np.uint32), device)

    @jax.jit
    def generate(values: jax.Array) -> jax.Array:
        def one(seed: jax.Array) -> jax.Array:
            _, action_key = jax.random.split(jax.random.PRNGKey(seed), 2)
            return jax.random.normal(
                action_key, (action_horizon, action_dim), dtype=jnp.float32
            )

        return jax.vmap(one)(values)

    return np.asarray(generate(seeds_device), dtype=np.float32)


def _flatten_branch_examples(
    arrays: Mapping[str, np.ndarray],
    root_rows: np.ndarray,
    *,
    active_dim: int,
    raw_context_mean: np.ndarray,
    group_slices: Mapping[str, tuple[int, int]],
    pca: Mapping[str, np.ndarray],
    device: jax.Device,
) -> dict[str, np.ndarray]:
    valid = np.asarray(arrays["branch_valid"], dtype=np.bool_)[root_rows]
    local_rows, slots = np.nonzero(valid)
    global_rows = np.asarray(root_rows, dtype=np.int32)[local_rows]
    if not global_rows.size:
        raise ValueError("Selected branch roots contain no valid branch examples.")
    ear = np.asarray(arrays["fresh_ear"])[global_rows, slots]
    actions = np.asarray(arrays["fresh_actions"])[global_rows, slots]
    iar = np.asarray(arrays["fresh_iar"], dtype=np.float32)[global_rows, slots]
    state = np.asarray(arrays["current_state"], dtype=np.float32)[global_rows, slots]
    policy_seed = np.asarray(arrays["policy_seed"], dtype=np.uint32)[global_rows]
    action_dim = int(actions.shape[-1])
    noise = _action_noise_from_seeds(
        policy_seed,
        action_horizon=int(actions.shape[-2]),
        action_dim=action_dim,
        device=device,
    )
    raw = np.broadcast_to(
        np.asarray(raw_context_mean, dtype=np.float32)[None, :],
        (global_rows.size, raw_context_mean.size),
    ).copy()
    iar_start, iar_end = group_slices["iar"]
    state_start, state_end = group_slices["state"]
    flat_iar = iar.reshape((iar.shape[0], -1))
    if flat_iar.shape[1] != iar_end - iar_start:
        raise ValueError(
            f"Branch IAR width {flat_iar.shape[1]} does not match main context width {iar_end - iar_start}."
        )
    if state.shape[1] != state_end - state_start:
        raise ValueError(
            f"Branch state width {state.shape[1]} does not match main context width {state_end - state_start}."
        )
    raw[:, iar_start:iar_end] = flat_iar
    raw[:, state_start:state_end] = state
    return {
        "plan": _flatten_active(ear, active_dim),
        "noise": _flatten_active(noise, active_dim),
        "target": _flatten_active(actions, active_dim),
        "context": _transform_context(raw, pca, device=device),
        "root_row": global_rows.astype(np.int32),
        "branch_slot": slots.astype(np.int32),
    }


def _fit_randomized_pca(
    raw_context: np.ndarray,
    train_rows: np.ndarray,
    *,
    pca_dim: int,
    oversample: int,
    power_iterations: int,
    seed: int,
    device: jax.Device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    training = np.asarray(raw_context[train_rows], dtype=np.float32)
    mean = training.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = training.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-5, std, 1.0).astype(np.float32)
    standardized = np.ascontiguousarray((training - mean) / std, dtype=np.float32)
    max_rank = min(standardized.shape)
    rank = min(int(pca_dim), max_rank)
    projection_rank = min(max_rank, rank + int(oversample))
    x = jax.device_put(standardized, device)
    key = jax.random.PRNGKey(seed)
    omega = jax.random.normal(
        key,
        (standardized.shape[1], projection_rank),
        dtype=jnp.float32,
    ) / math.sqrt(float(projection_rank))
    y = x @ omega
    for _ in range(power_iterations):
        q, _ = jnp.linalg.qr(y, mode="reduced")
        z, _ = jnp.linalg.qr(x.T @ q, mode="reduced")
        y = x @ z
    q, _ = jnp.linalg.qr(y, mode="reduced")
    small = q.T @ x
    _, singular_values, right = jnp.linalg.svd(small, full_matrices=False)
    components = np.asarray(right[:rank], dtype=np.float32)
    selected_singular_values = np.asarray(singular_values[:rank], dtype=np.float64)
    total_energy = float(np.sum(np.square(standardized, dtype=np.float64)))
    explained = float(np.sum(np.square(selected_singular_values)) / max(total_energy, 1e-12))
    return {
        "mean": mean,
        "std": std,
        "components": components,
    }, {
        "method": "train-only standardized randomized PCA on JAX device",
        "raw_dim": int(standardized.shape[1]),
        "requested_dim": int(pca_dim),
        "retained_dim": int(rank),
        "oversample": int(oversample),
        "power_iterations": int(power_iterations),
        "approximate_explained_energy_ratio": explained,
        "fit_records": int(training.shape[0]),
    }


def _transform_context(
    raw_context: np.ndarray,
    pca: Mapping[str, np.ndarray],
    *,
    device: jax.Device,
    ablate_slice: tuple[int, int] | None = None,
) -> np.ndarray:
    mean = np.asarray(pca["mean"], dtype=np.float32)
    std = np.asarray(pca["std"], dtype=np.float32)
    standardized = np.asarray((raw_context - mean) / std, dtype=np.float32)
    if ablate_slice is not None:
        start, end = ablate_slice
        standardized[:, start:end] = 0.0
    scores = jax.device_put(standardized, device) @ jax.device_put(
        np.asarray(pca["components"], dtype=np.float32).T, device
    )
    return np.asarray(scores, dtype=np.float32)


def _init_params(
    key: jax.Array,
    *,
    plan_dim: int,
    feature_dim: int,
    context_dim: int,
    rank: int,
    hidden_dim: int,
    target_dim: int,
) -> dict[str, jax.Array]:
    keys = jax.random.split(key, 6)

    def normal(current: jax.Array, shape: tuple[int, ...], fan_in: int) -> jax.Array:
        return jax.random.normal(current, shape, dtype=jnp.float32) / math.sqrt(float(fan_in))

    return {
        "plan_projection": normal(keys[0], (plan_dim, rank), plan_dim),
        "context_projection": normal(keys[1], (context_dim, rank), context_dim),
        "base_projection": normal(keys[2], (feature_dim, hidden_dim), feature_dim),
        "base_bias": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "gamma_projection": normal(keys[3], (rank, hidden_dim), rank),
        "beta_projection": normal(keys[4], (rank, hidden_dim), rank),
        # Exact zeros make the initial candidate identical to the frozen ridge
        # anchor. The output layer learns first, then gradients reach the
        # upstream bilinear factors.
        "output_projection": jnp.zeros((hidden_dim, target_dim), dtype=jnp.float32),
    }


def _ridge_predict(
    ridge: Mapping[str, jax.Array], plan: jax.Array, noise: jax.Array
) -> tuple[jax.Array, jax.Array]:
    features = jnp.concatenate((plan, noise), axis=-1)
    normalized = (features - ridge["feature_mean"]) / ridge["feature_std"]
    prediction = normalized @ ridge["weights"] + ridge["target_mean"]
    return prediction, normalized


def _predict(
    params: Mapping[str, jax.Array],
    ridge: Mapping[str, jax.Array],
    plan: jax.Array,
    noise: jax.Array,
    context: jax.Array,
    *,
    residual_scale: float,
) -> jax.Array:
    anchor, normalized_features = _ridge_predict(ridge, plan, noise)
    plan_factor = jnp.tanh(plan @ params["plan_projection"])
    context_factor = jnp.tanh(context @ params["context_projection"])
    interaction = plan_factor * context_factor
    base_hidden = jax.nn.gelu(
        normalized_features @ params["base_projection"] + params["base_bias"],
        approximate=True,
    )
    gamma = jnp.tanh(interaction @ params["gamma_projection"])
    beta = interaction @ params["beta_projection"]
    delta_hidden = base_hidden * gamma + beta
    residual = jax.nn.gelu(delta_hidden, approximate=True) @ params["output_projection"]
    return anchor + residual_scale * jnp.tanh(residual)


def _response_scales(
    arrays: Mapping[str, np.ndarray],
    train_pair_rows: np.ndarray,
    train_pair_slots: np.ndarray,
    train_pair_ids: np.ndarray,
    *,
    active_dim: int,
) -> np.ndarray:
    clean = _flatten_active(np.asarray(arrays["clean_actions"])[train_pair_rows], active_dim)
    intervention = _flatten_active(
        np.asarray(arrays["intervention_actions"])[train_pair_rows, train_pair_slots],
        active_dim,
    )
    per_pair = np.mean(np.square(intervention - clean), axis=-1)
    overall = float(np.mean(per_pair))
    floor = max(overall * 0.1, 1e-6)
    scales = np.full((len(endpoint_dataset.INTERVENTION_NAMES),), max(overall, floor), np.float32)
    for intervention_id in np.unique(train_pair_ids):
        selected = per_pair[train_pair_ids == intervention_id]
        scales[int(intervention_id)] = max(float(np.mean(selected)), floor)
    return scales


def _loss(
    params: Mapping[str, jax.Array],
    ridge: Mapping[str, jax.Array],
    plan: jax.Array,
    noise: jax.Array,
    context: jax.Array,
    clean_target: jax.Array,
    intervention_plan: jax.Array,
    intervention_target: jax.Array,
    intervention_ids: jax.Array,
    response_scales: jax.Array,
    key: jax.Array,
    *,
    residual_scale: float,
    context_dropout: float,
    clean_weight: float,
    intervention_weight: float,
    response_weight: float,
    cosine_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if context_dropout > 0:
        keep = jax.random.bernoulli(
            key, 1.0 - context_dropout, (context.shape[0], 1)
        ).astype(context.dtype)
        context = context * keep / (1.0 - context_dropout)
    clean_prediction = _predict(
        params, ridge, plan, noise, context, residual_scale=residual_scale
    )
    intervention_prediction = _predict(
        params,
        ridge,
        intervention_plan,
        noise,
        context,
        residual_scale=residual_scale,
    )
    clean_mse = jnp.mean(jnp.square(clean_prediction - clean_target))
    intervention_mse = jnp.mean(
        jnp.square(intervention_prediction - intervention_target)
    )
    predicted_response = intervention_prediction - clean_prediction
    teacher_response = intervention_target - clean_target
    response_error = jnp.mean(
        jnp.square(predicted_response - teacher_response), axis=-1
    )
    response_mse = jnp.mean(response_error)
    normalized_response_mse = jnp.mean(
        response_error / response_scales[intervention_ids]
    )
    numerator = jnp.sum(predicted_response * teacher_response, axis=-1)
    predicted_norm = jnp.sqrt(jnp.sum(jnp.square(predicted_response), axis=-1) + 1e-12)
    teacher_norm = jnp.sqrt(jnp.sum(jnp.square(teacher_response), axis=-1) + 1e-12)
    cosine = numerator / jnp.maximum(predicted_norm * teacher_norm, 1e-8)
    valid_cosine = teacher_norm > 1e-4
    cosine_mean = jnp.sum(jnp.where(valid_cosine, cosine, 0.0)) / jnp.maximum(
        jnp.sum(valid_cosine), 1
    )
    cosine_loss = 1.0 - cosine_mean
    total = (
        clean_weight * clean_mse
        + intervention_weight * intervention_mse
        + response_weight * normalized_response_mse
        + cosine_weight * cosine_loss
    )
    return total, {
        "loss": total,
        "clean_action_mse_active7": clean_mse,
        "intervention_action_mse_active7": intervention_mse,
        "response_mse_active7": response_mse,
        "normalized_response_mse": normalized_response_mse,
        "response_cosine_active7": cosine_mean,
    }


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction, np.float64) - np.asarray(target, np.float64))))


def _response_metrics(
    clean_prediction: np.ndarray,
    clean_target: np.ndarray,
    intervention_prediction: np.ndarray,
    intervention_target: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(intervention_prediction, np.float64) - np.asarray(
        clean_prediction, np.float64
    )
    teacher = np.asarray(intervention_target, np.float64) - np.asarray(clean_target, np.float64)
    numerator = np.sum(predicted * teacher, axis=-1)
    predicted_norm = np.sqrt(np.sum(np.square(predicted), axis=-1) + 1e-12)
    teacher_norm = np.sqrt(np.sum(np.square(teacher), axis=-1) + 1e-12)
    valid = teacher_norm > 1e-4
    cosine = numerator / np.maximum(predicted_norm * teacher_norm, 1e-8)
    return {
        "response_mse_active7": _mse(predicted, teacher),
        "response_rmse_active7": math.sqrt(_mse(predicted, teacher)),
        "response_cosine_active7": float(np.mean(cosine[valid])) if np.any(valid) else 0.0,
        "student_response_l2": float(np.mean(predicted_norm)),
        "teacher_response_l2": float(np.mean(teacher_norm)),
    }


def _aggregate(
    clean_prediction: np.ndarray,
    clean_target: np.ndarray,
    pair_clean_prediction: np.ndarray,
    pair_clean_target: np.ndarray,
    intervention_prediction: np.ndarray,
    intervention_target: np.ndarray,
) -> dict[str, float]:
    clean_mse = _mse(clean_prediction, clean_target)
    intervention_mse = _mse(intervention_prediction, intervention_target)
    return {
        "clean_action_mse_active7": clean_mse,
        "clean_action_rmse_active7": math.sqrt(clean_mse),
        "intervention_action_mse_active7": intervention_mse,
        "intervention_action_rmse_active7": math.sqrt(intervention_mse),
        **_response_metrics(
            pair_clean_prediction,
            pair_clean_target,
            intervention_prediction,
            intervention_target,
        ),
    }


def _same_task_permutation(
    rows: np.ndarray,
    tasks: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    result = np.arange(rows.size, dtype=np.int64)
    row_tasks = np.asarray(tasks)[rows]
    for task in np.unique(row_tasks):
        positions = np.flatnonzero(row_tasks == task)
        if positions.size < 2:
            continue
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        if np.any(shuffled == positions):
            shuffled = np.roll(positions, 1)
        result[positions] = shuffled
    return result, int(np.count_nonzero(result != np.arange(rows.size)))


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree.map(lambda value: np.asarray(value, dtype=np.float32), tree)


def _tree_to_device(tree: Any, device: jax.Device) -> Any:
    return jax.tree.map(lambda value: jax.device_put(jnp.asarray(value), device), tree)


def _predict_batches(
    predict_step: Any,
    params: Any,
    plan: np.ndarray,
    noise: np.ndarray,
    context: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, plan.shape[0], batch_size):
        end = min(start + batch_size, plan.shape[0])
        prediction = predict_step(
            params,
            jnp.asarray(plan[start:end]),
            jnp.asarray(noise[start:end]),
            jnp.asarray(context[start:end]),
        )
        outputs.append(np.asarray(prediction, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _ridge_predict_numpy(
    ridge: Mapping[str, np.ndarray], plan: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    features = np.concatenate((plan, noise), axis=-1)
    normalized = (features - ridge["feature_mean"]) / ridge["feature_std"]
    return normalized @ ridge["weights"] + ridge["target_mean"]


def _evaluation_arrays(
    arrays: Mapping[str, np.ndarray],
    rows: np.ndarray,
    pair_rows: np.ndarray,
    pair_slots: np.ndarray,
    *,
    active_dim: int,
) -> dict[str, np.ndarray]:
    row_to_offset = {int(row): offset for offset, row in enumerate(rows)}
    return {
        "clean_plan": _flatten_active(np.asarray(arrays["clean_coarse"])[rows], active_dim),
        "clean_noise": _flatten_active(np.asarray(arrays["action_noise"])[rows], active_dim),
        "clean_target": _flatten_active(np.asarray(arrays["clean_actions"])[rows], active_dim),
        "intervention_plan": _flatten_active(
            np.asarray(arrays["intervention_coarse"])[pair_rows, pair_slots], active_dim
        ),
        "intervention_noise": _flatten_active(
            np.asarray(arrays["action_noise"])[pair_rows], active_dim
        ),
        "intervention_target": _flatten_active(
            np.asarray(arrays["intervention_actions"])[pair_rows, pair_slots], active_dim
        ),
        "pair_clean_offsets": np.asarray(
            [row_to_offset[int(row)] for row in pair_rows], dtype=np.int32
        ),
    }


def _evaluate_predictions(
    predict_step: Any,
    params: Any,
    values: Mapping[str, np.ndarray],
    context: np.ndarray,
    pair_rows: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    row_to_offset = {int(row): offset for offset, row in enumerate(rows)}
    pair_context = context[
        np.asarray([row_to_offset[int(row)] for row in pair_rows], dtype=np.int32)
    ]
    clean_prediction = _predict_batches(
        predict_step,
        params,
        values["clean_plan"],
        values["clean_noise"],
        context,
        batch_size=batch_size,
    )
    intervention_prediction = _predict_batches(
        predict_step,
        params,
        values["intervention_plan"],
        values["intervention_noise"],
        pair_context,
        batch_size=batch_size,
    )
    offsets = values["pair_clean_offsets"]
    metrics = _aggregate(
        clean_prediction,
        values["clean_target"],
        clean_prediction[offsets],
        values["clean_target"][offsets],
        intervention_prediction,
        values["intervention_target"],
    )
    return metrics, {
        "clean_prediction": clean_prediction,
        "intervention_prediction": intervention_prediction,
    }


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: pathlib.Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def _flat_save_values(prefix: str, tree: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}/{name}": np.asarray(value) for name, value in tree.items()}


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    if not args.overwrite and (summary_path.exists() or (metrics_path.exists() and metrics_path.stat().st_size)):
        raise FileExistsError(
            f"Output already exists in {output_dir}; choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and metrics_path.exists():
        metrics_path.unlink()

    device = _select_device(allow_cpu=args.allow_cpu)
    started = time.perf_counter()
    target_arrays = endpoint_dataset.load_endpoint_arrays(args.target_dataset)
    context_arrays = endpoint_dataset.load_endpoint_arrays(
        args.context_dataset, include_deployment_context=True
    )
    context, join_audit = _join_target_and_context(target_arrays, context_arrays)
    if np.asarray(target_arrays["clean_coarse"]).shape[-1] < args.active_action_dim:
        raise ValueError("Target action dimension is smaller than --active-action-dim.")

    train_rows, validation_rows = _split_indices(
        target_arrays,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_pair_rows, train_pair_slots, train_pair_ids = _all_pairs(target_arrays, train_rows)
    validation_pair_rows, validation_pair_slots, validation_pair_ids = _all_pairs(
        target_arrays, validation_rows
    )
    ridge_np = _fit_ridge(
        target_arrays,
        train_rows,
        train_pair_rows,
        train_pair_slots,
        active_dim=args.active_action_dim,
        ridge_lambda=args.ridge_lambda,
    )
    raw_context, group_slices = _raw_context(context)
    pca_np, pca_audit = _fit_randomized_pca(
        raw_context,
        train_rows,
        pca_dim=args.pca_dim,
        oversample=args.pca_oversample,
        power_iterations=args.pca_power_iterations,
        seed=args.seed,
        device=device,
    )
    context_scores = _transform_context(raw_context, pca_np, device=device)
    prefix_ablated_scores = _transform_context(
        raw_context, pca_np, device=device, ablate_slice=group_slices["prefix"]
    )
    iar_ablated_scores = _transform_context(
        raw_context, pca_np, device=device, ablate_slice=group_slices["iar"]
    )
    state_ablated_scores = _transform_context(
        raw_context, pca_np, device=device, ablate_slice=group_slices["state"]
    )
    branch_train: dict[str, np.ndarray] | None = None
    branch_validation: dict[str, np.ndarray] | None = None
    branch_audit: dict[str, Any] = {
        "enabled": False,
        "note": "No optional same-root branch pretraining dataset was supplied.",
    }
    if args.branch_dataset:
        branch_arrays = _load_branch_arrays(args.branch_dataset)
        branch_train_roots, branch_validation_roots = _branch_root_split(
            branch_arrays,
            validation_fraction=args.branch_validation_fraction,
            seed=args.seed,
        )
        branch_train = _flatten_branch_examples(
            branch_arrays,
            branch_train_roots,
            active_dim=args.active_action_dim,
            raw_context_mean=pca_np["mean"],
            group_slices=group_slices,
            pca=pca_np,
            device=device,
        )
        branch_validation = _flatten_branch_examples(
            branch_arrays,
            branch_validation_roots,
            active_dim=args.active_action_dim,
            raw_context_mean=pca_np["mean"],
            group_slices=group_slices,
            pca=pca_np,
            device=device,
        )
        branch_audit = {
            "enabled": True,
            "inputs": list(args.branch_dataset),
            "split_unit": ["task_id", "episode_id", "root_id"],
            "branch_is_never_a_split_unit": True,
            "train_roots": int(branch_train_roots.size),
            "validation_roots": int(branch_validation_roots.size),
            "train_branch_examples": int(branch_train["plan"].shape[0]),
            "validation_branch_examples": int(branch_validation["plan"].shape[0]),
            "prefix_feature_policy": (
                "unavailable in canonical800; filled with the main-train prefix mean so only IAR/state "
                "contribute during optional pretraining"
            ),
            "noise_reconstruction": "exact policy_seed -> split(PRNGKey(seed), 2)[1]",
            "use": "clean endpoint pretraining only; never used for main held-out intervention selection",
        }

    clean_plan_all = _flatten_active(target_arrays["clean_coarse"], args.active_action_dim)
    action_noise_all = _flatten_active(target_arrays["action_noise"], args.active_action_dim)
    clean_target_all = _flatten_active(target_arrays["clean_actions"], args.active_action_dim)
    intervention_plan_all = _flatten_active(
        target_arrays["intervention_coarse"], args.active_action_dim
    )
    intervention_target_all = _flatten_active(
        target_arrays["intervention_actions"], args.active_action_dim
    )
    response_scales_np = _response_scales(
        target_arrays,
        train_pair_rows,
        train_pair_slots,
        train_pair_ids,
        active_dim=args.active_action_dim,
    )

    ridge = _tree_to_device(ridge_np, device)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = _init_params(
        init_key,
        plan_dim=clean_plan_all.shape[-1],
        feature_dim=clean_plan_all.shape[-1] + action_noise_all.shape[-1],
        context_dim=context_scores.shape[-1],
        rank=args.interaction_rank,
        hidden_dim=args.hidden_dim,
        target_dim=clean_target_all.shape[-1],
    )
    params = _tree_to_device(params, device)
    parameter_count = int(sum(np.prod(value.shape) for value in jax.tree.leaves(params)))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=max(args.steps, args.warmup_steps + 1),
        end_value=args.final_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(params)

    plan_dev = jax.device_put(clean_plan_all, device)
    noise_dev = jax.device_put(action_noise_all, device)
    target_dev = jax.device_put(clean_target_all, device)
    intervention_plan_dev = jax.device_put(intervention_plan_all, device)
    intervention_target_dev = jax.device_put(intervention_target_all, device)
    context_dev = jax.device_put(context_scores, device)
    train_pair_rows_dev = jax.device_put(train_pair_rows, device)
    train_pair_slots_dev = jax.device_put(train_pair_slots, device)
    train_pair_ids_dev = jax.device_put(train_pair_ids, device)
    response_scales_dev = jax.device_put(response_scales_np, device)

    @jax.jit
    def train_step(
        current_params: Any,
        current_optimizer_state: Any,
        pair_indices: jax.Array,
        step_key: jax.Array,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        rows = train_pair_rows_dev[pair_indices]
        slots = train_pair_slots_dev[pair_indices]
        ids = train_pair_ids_dev[pair_indices]

        def candidate_loss(candidate: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            return _loss(
                candidate,
                ridge,
                plan_dev[rows],
                noise_dev[rows],
                context_dev[rows],
                target_dev[rows],
                intervention_plan_dev[rows, slots],
                intervention_target_dev[rows, slots],
                ids,
                response_scales_dev,
                step_key,
                residual_scale=args.residual_scale,
                context_dropout=args.context_dropout,
                clean_weight=args.clean_loss_weight,
                intervention_weight=args.intervention_loss_weight,
                response_weight=args.response_loss_weight,
                cosine_weight=args.response_cosine_loss_weight,
            )

        (_, values), gradients = jax.value_and_grad(candidate_loss, has_aux=True)(current_params)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_optimizer_state, {
            **values,
            "gradient_norm": optax.global_norm(gradients),
        }

    @jax.jit
    def predict_step(
        current_params: Any, plan: jax.Array, noise: jax.Array, context_code: jax.Array
    ) -> jax.Array:
        return _predict(
            current_params,
            ridge,
            plan,
            noise,
            context_code,
            residual_scale=args.residual_scale,
        )

    validation_values = _evaluation_arrays(
        target_arrays,
        validation_rows,
        validation_pair_rows,
        validation_pair_slots,
        active_dim=args.active_action_dim,
    )
    validation_context = context_scores[validation_rows]
    ridge_clean = _ridge_predict_numpy(
        ridge_np, validation_values["clean_plan"], validation_values["clean_noise"]
    )
    ridge_intervention = _ridge_predict_numpy(
        ridge_np,
        validation_values["intervention_plan"],
        validation_values["intervention_noise"],
    )
    ridge_metrics = _aggregate(
        ridge_clean,
        validation_values["clean_target"],
        ridge_clean[validation_values["pair_clean_offsets"]],
        validation_values["clean_target"][validation_values["pair_clean_offsets"]],
        ridge_intervention,
        validation_values["intervention_target"],
    )

    def candidate_rank(metrics: Mapping[str, float]) -> tuple[float, float, float]:
        clean_ratio = metrics["clean_action_mse_active7"] / max(
            ridge_metrics["clean_action_mse_active7"], 1e-12
        )
        intervention_ratio = metrics["intervention_action_mse_active7"] / max(
            ridge_metrics["intervention_action_mse_active7"], 1e-12
        )
        feasible = (
            clean_ratio <= 1.0 + args.clean_nondegradation_tolerance
            and intervention_ratio <= 1.0 + args.intervention_nondegradation_tolerance
        )
        if feasible:
            return (0.0, metrics["response_mse_active7"], metrics["clean_action_mse_active7"])
        return (
            1.0,
            max(clean_ratio, intervention_ratio),
            metrics["response_mse_active7"],
        )

    initial_metrics, _ = _evaluate_predictions(
        predict_step,
        params,
        validation_values,
        validation_context,
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )
    best_params = _tree_to_numpy(params)
    best_metrics = initial_metrics
    best_rank = candidate_rank(initial_metrics)
    best_step = 0
    initial_record = {
        "phase": "validation",
        "step": 0,
        "elapsed_seconds": time.perf_counter() - started,
        **{f"validation/{name}": value for name, value in initial_metrics.items()},
        "selected_as_best_checkpoint": True,
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(initial_record, sort_keys=True) + "\n")

    rng = np.random.default_rng(args.seed)
    last_train_metrics: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        selected = rng.integers(
            0, train_pair_rows.size, size=(args.batch_size,), dtype=np.int32
        )
        key, step_key = jax.random.split(key)
        params, optimizer_state, train_metrics = train_step(
            params,
            optimizer_state,
            jax.device_put(selected, device),
            step_key,
        )
        if step == 1:
            jax.block_until_ready(train_metrics)
        last_train_metrics = {
            name: float(np.asarray(value)) for name, value in train_metrics.items()
        }
        if step % args.log_interval != 0 and step != args.steps:
            continue
        validation_metrics, _ = _evaluate_predictions(
            predict_step,
            params,
            validation_values,
            validation_context,
            validation_pair_rows,
            validation_rows,
            batch_size=max(args.batch_size, 256),
        )
        rank = candidate_rank(validation_metrics)
        selected_as_best = rank < best_rank
        if selected_as_best:
            best_rank = rank
            best_params = _tree_to_numpy(params)
            best_metrics = validation_metrics
            best_step = step
        record = {
            "phase": "train",
            "step": step,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": float(np.asarray(schedule(step))),
            **{f"train/{name}": value for name, value in last_train_metrics.items()},
            **{f"validation/{name}": value for name, value in validation_metrics.items()},
            "selected_as_best_checkpoint": selected_as_best,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    best_params_device = _tree_to_device(best_params, device)
    final_metrics, predictions = _evaluate_predictions(
        predict_step,
        best_params_device,
        validation_values,
        validation_context,
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )

    validation_row_to_offset = {
        int(row): offset for offset, row in enumerate(validation_rows)
    }
    context_permutation, context_moved = _same_task_permutation(
        validation_rows,
        np.asarray(target_arrays["task_id"]),
        seed=args.seed + 101,
    )
    shuffled_context = validation_context[context_permutation]
    context_shuffle_metrics, _ = _evaluate_predictions(
        predict_step,
        best_params_device,
        validation_values,
        shuffled_context,
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )
    zero_context_metrics, _ = _evaluate_predictions(
        predict_step,
        best_params_device,
        validation_values,
        np.zeros_like(validation_context),
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )
    prefix_ablation_metrics, _ = _evaluate_predictions(
        predict_step,
        best_params_device,
        validation_values,
        prefix_ablated_scores[validation_rows],
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )
    iar_ablation_metrics, _ = _evaluate_predictions(
        predict_step,
        best_params_device,
        validation_values,
        iar_ablated_scores[validation_rows],
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )
    state_ablation_metrics, _ = _evaluate_predictions(
        predict_step,
        best_params_device,
        validation_values,
        state_ablated_scores[validation_rows],
        validation_pair_rows,
        validation_rows,
        batch_size=max(args.batch_size, 256),
    )

    ear_permutation, ear_moved = _same_task_permutation(
        validation_rows,
        np.asarray(target_arrays["task_id"]),
        seed=args.seed + 202,
    )
    shuffled_ear_prediction = _predict_batches(
        predict_step,
        best_params_device,
        validation_values["clean_plan"][ear_permutation],
        validation_values["clean_noise"],
        validation_context,
        batch_size=max(args.batch_size, 256),
    )
    shuffled_ear_clean_mse = _mse(
        shuffled_ear_prediction, validation_values["clean_target"]
    )

    per_intervention: dict[str, Any] = {}
    clean_offsets = validation_values["pair_clean_offsets"]
    for intervention_id in np.unique(validation_pair_ids):
        selected = np.flatnonzero(validation_pair_ids == intervention_id)
        name = endpoint_dataset.INTERVENTION_NAMES[int(intervention_id)]
        per_intervention[name] = {
            "semantic_pairs": int(selected.size),
            "intervention_id": int(intervention_id),
            **_aggregate(
                predictions["clean_prediction"],
                validation_values["clean_target"],
                predictions["clean_prediction"][clean_offsets[selected]],
                validation_values["clean_target"][clean_offsets[selected]],
                predictions["intervention_prediction"][selected],
                validation_values["intervention_target"][selected],
            ),
        }

    per_task: dict[str, Any] = {}
    validation_tasks = np.asarray(target_arrays["task_id"])[validation_rows]
    pair_tasks = np.asarray(target_arrays["task_id"])[validation_pair_rows]
    for task_id in np.unique(validation_tasks):
        clean_selected = np.flatnonzero(validation_tasks == task_id)
        pair_selected = np.flatnonzero(pair_tasks == task_id)
        if not pair_selected.size:
            continue
        task_offset_lookup = {
            int(offset): local for local, offset in enumerate(clean_selected.tolist())
        }
        task_pair_offsets = np.asarray(
            [task_offset_lookup[int(clean_offsets[index])] for index in pair_selected],
            dtype=np.int32,
        )
        per_task[str(int(task_id))] = {
            "validation_records": int(clean_selected.size),
            "semantic_pairs": int(pair_selected.size),
            **_aggregate(
                predictions["clean_prediction"][clean_selected],
                validation_values["clean_target"][clean_selected],
                predictions["clean_prediction"][clean_selected][task_pair_offsets],
                validation_values["clean_target"][clean_selected][task_pair_offsets],
                predictions["intervention_prediction"][pair_selected],
                validation_values["intervention_target"][pair_selected],
            ),
        }

    pca_device = _tree_to_device(pca_np, device)

    @jax.jit
    def raw_context_predict(
        current_params: Any,
        plan: jax.Array,
        noise: jax.Array,
        raw: jax.Array,
    ) -> jax.Array:
        standardized = (raw - pca_device["mean"]) / pca_device["std"]
        context_code = standardized @ pca_device["components"].T
        return _predict(
            current_params,
            ridge,
            plan,
            noise,
            context_code,
            residual_scale=args.residual_scale,
        )

    latency_row = int(validation_rows[0])
    latency_inputs = (
        jax.device_put(clean_plan_all[latency_row : latency_row + 1], device),
        jax.device_put(action_noise_all[latency_row : latency_row + 1], device),
        jax.device_put(raw_context[latency_row : latency_row + 1], device),
    )
    for _ in range(args.latency_warmup):
        jax.block_until_ready(raw_context_predict(best_params_device, *latency_inputs))
    latency_values = []
    for _ in range(args.latency_runs):
        latency_start = time.perf_counter()
        jax.block_until_ready(raw_context_predict(best_params_device, *latency_inputs))
        latency_values.append((time.perf_counter() - latency_start) * 1000.0)
    latency = {
        "backend": "jax_jit",
        "includes_train_fitted_standardization_and_pca": True,
        "excludes_already_paid_prefix_and_iar_extraction": True,
        "batch_size": 1,
        "warmup_runs": args.latency_warmup,
        "timed_runs": args.latency_runs,
        "mean_ms": float(np.mean(latency_values)),
        "median_ms": float(np.median(latency_values)),
        "p95_ms": float(np.percentile(latency_values, 95)),
    }

    context_shuffle_ratio = context_shuffle_metrics["clean_action_mse_active7"] / max(
        final_metrics["clean_action_mse_active7"], 1e-12
    )
    ear_shuffle_ratio = shuffled_ear_clean_mse / max(
        final_metrics["clean_action_mse_active7"], 1e-12
    )
    response_improvement = 1.0 - final_metrics["response_mse_active7"] / max(
        ridge_metrics["response_mse_active7"], 1e-12
    )
    gates = {
        "clean_nondegradation_vs_ridge": final_metrics["clean_action_mse_active7"]
        <= ridge_metrics["clean_action_mse_active7"]
        * (1.0 + args.clean_nondegradation_tolerance),
        "intervention_nondegradation_vs_ridge": final_metrics[
            "intervention_action_mse_active7"
        ]
        <= ridge_metrics["intervention_action_mse_active7"]
        * (1.0 + args.intervention_nondegradation_tolerance),
        "response_improvement_vs_ridge": response_improvement
        >= args.required_response_improvement,
        "response_cosine": final_metrics["response_cosine_active7"]
        >= args.required_response_cosine,
        "context_shuffle_faithfulness": context_shuffle_ratio
        >= args.required_context_shuffle_ratio,
        "ear_shuffle_load_bearing": ear_shuffle_ratio >= args.required_ear_shuffle_ratio,
        "standalone_latency": latency["p95_ms"] <= args.max_latency_p95_ms,
    }
    go = all(gates.values())

    model_save = {
        **_flat_save_values("model", best_params),
        **_flat_save_values("ridge", ridge_np),
        **_flat_save_values("context_pca", pca_np),
        "response_scales": response_scales_np,
        "group_slices": np.asarray(
            [group_slices[name] for name in ("prefix", "iar", "state")], dtype=np.int32
        ),
    }
    _atomic_npz(output_dir / "model_params.npz", **model_save)
    _atomic_npz(
        output_dir / "heldout_predictions.npz",
        validation_rows=validation_rows,
        validation_pair_rows=validation_pair_rows,
        validation_pair_slots=validation_pair_slots,
        validation_pair_ids=validation_pair_ids,
        clean_prediction_active7=predictions["clean_prediction"],
        clean_target_active7=validation_values["clean_target"],
        intervention_prediction_active7=predictions["intervention_prediction"],
        intervention_target_active7=validation_values["intervention_target"],
        context_scores=context_scores[validation_rows],
        context_permutation=context_permutation,
        ear_permutation=ear_permutation,
    )
    summary = {
        "status": "complete",
        "go": go,
        "gates": gates,
        "args": dataclasses.asdict(args),
        "device": {
            "platform": device.platform,
            "kind": device.device_kind,
            "id": int(device.id),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": {
            "target_inputs": list(args.target_dataset),
            "context_inputs": list(args.context_dataset),
            "join_audit": join_audit,
            "train_records": int(train_rows.size),
            "validation_records": int(validation_rows.size),
            "train_semantic_pairs": int(train_pair_rows.size),
            "validation_semantic_pairs": int(validation_pair_rows.size),
            "episode_grouped_split": True,
            "split_seed": args.seed,
            "forbidden_inputs": [
                "task_id",
                "episode_id",
                "frame_id",
                "future_state",
                "outcome",
                "privileged_simulator_state",
                "intervention_id_as_model_input",
            ],
        },
        "input_contract": {
            "deployment_available_before_final_suffix": [
                "one-step EAR",
                "shared final action noise",
                "current-observation pooled VLM prefix",
                "current-observation IAR from existing prefix KV cache",
                "normalized current proprioception",
            ],
            "additional_large_model_calls": 0,
            "causal_structure": (
                "context interacts multiplicatively with EAR and can only FiLM the EAR/noise "
                "transport; prediction is a bounded residual over a frozen EAR/noise ridge anchor"
            ),
            "clean_and_intervention_share_context_and_noise": True,
            "active_action_dimensions": args.active_action_dim,
        },
        "architecture": {
            "name": "contextual causal ridge-residual plan compiler",
            "parameter_count": parameter_count,
            "context_compression": pca_audit,
            "context_group_slices": group_slices,
            "interaction_rank": args.interaction_rank,
            "hidden_dim": args.hidden_dim,
            "residual_scale": args.residual_scale,
            "ridge_lambda": args.ridge_lambda,
            "runtime": "JAX/XLA JIT; GPU required unless explicitly overridden",
        },
        "checkpoint_selection": {
            "best_step": best_step,
            "rule": (
                "minimum held-out response MSE among candidates within configured clean and "
                "intervention ridge non-degradation bounds; otherwise minimum worst normalized regression"
            ),
            "best_validation_metrics": best_metrics,
        },
        "ridge_baseline_metrics": ridge_metrics,
        "full_validation_metrics": final_metrics,
        "ablations": {
            "zero_context": zero_context_metrics,
            "same_task_context_shuffle": {
                **context_shuffle_metrics,
                "moved_records": context_moved,
                "clean_mse_ratio_vs_full": context_shuffle_ratio,
            },
            "prefix_mean_ablation": prefix_ablation_metrics,
            "iar_ablation": iar_ablation_metrics,
            "state_ablation": state_ablation_metrics,
            "same_task_ear_shuffle": {
                "moved_records": ear_moved,
                "clean_action_mse_active7": shuffled_ear_clean_mse,
                "clean_mse_ratio_vs_full": ear_shuffle_ratio,
            },
        },
        "response_improvement_vs_ridge": response_improvement,
        "per_intervention": per_intervention,
        "per_task": per_task,
        "latency": latency,
        "artifacts": {
            "model_params": str(output_dir / "model_params.npz"),
            "heldout_predictions": str(output_dir / "heldout_predictions.npz"),
            "metrics": str(metrics_path),
        },
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
