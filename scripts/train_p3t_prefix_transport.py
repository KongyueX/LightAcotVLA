"""Train the P3T plan-conditioned predictive prefix transport adapter.

This pilot deliberately keeps the complete ACoT-VLA policy frozen.  The
``windows2k`` export is used only as a compact index/anchor-plan manifest:
``anchor_index`` recovers the exact canonical observation from the original
LeRobot/Hugging Face cache, while the current observation is the same episode
at ``anchor_index + 10``.  The ten intervening normalized demonstration
actions are the causal executed-action history.

For every minibatch the frozen policy computes anchor and current prefix KV
caches online.  P3T transports the anchor cache, and the transported cache is
then consumed by the frozen IAR, EAR, and final action branches with exactly
the same flow noise as the fresh-current teacher.  No full KV cache is ever
written to disk.

The default dataset task id is six because the canonical LeRobot LIBERO
dataset maps the evaluation suite's Task8 instruction to ``task_index == 6``.
Splits are episode-disjoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from openpi.action_cot import multirate_dataset
from openpi.models import model as model_lib
from openpi.models import p3t_prefix_transport
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader


LOGGER = logging.getLogger("train_p3t_prefix_transport")


@dataclasses.dataclass(frozen=True)
class Args:
    dataset: tuple[str, ...]
    checkpoint_dir: str
    output_dir: str
    endpoint_student_params: str | None = None
    config_name: str = "acot_libero_action_cot_explicit_implicit_co_fusion"
    dataset_task_id: int = 6
    temporal_stride: int = 10
    seed: int = 7
    split_seed: int = 7
    train_steps: int = 500
    batch_size: int = 1
    eval_batch_size: int = 1
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    validation_examples: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 25
    log_interval: int = 25
    early_stopping_patience_logs: int = 6
    early_stopping_min_delta: float = 1e-5
    coarse_flow_steps: int = 1
    final_flow_steps: int = 1
    kv_loss_weight: float = 1.0
    iar_loss_weight: float = 0.1
    ear_loss_weight: float = 1.0
    action_loss_weight: float = 2.0
    risk_loss_weight: float = 0.1
    risk_action_mse_threshold: float = 0.05
    kv_delta_floor: float = 1e-4
    maximum_pairs: int = 200
    profile_warmup: int = 20
    profile_iterations: int = 200
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class PairIndices:
    row_indices: np.ndarray
    anchor_indices: np.ndarray
    target_indices: np.ndarray
    episode_ids: np.ndarray
    frame_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.row_indices,
            self.anchor_indices,
            self.target_indices,
            self.episode_ids,
            self.frame_ids,
        )
        if any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("P3T pair metadata must contain rank-one arrays.")
        if len({np.asarray(value).shape for value in arrays}) != 1:
            raise ValueError("P3T pair metadata arrays must have matching shapes.")
        if np.any(self.target_indices - self.anchor_indices != 10):
            raise ValueError("The first P3T pilot requires anchor-to-anchor+10 pairs.")

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def take(self, indices: np.ndarray | slice) -> "PairIndices":
        return PairIndices(
            row_indices=self.row_indices[indices],
            anchor_indices=self.anchor_indices[indices],
            target_indices=self.target_indices[indices],
            episode_ids=self.episode_ids[indices],
            frame_ids=self.frame_ids[indices],
        )


class MaterializedPair(NamedTuple):
    anchor_observation: dict[str, Any]
    current_observation: dict[str, Any]
    executed_actions: np.ndarray
    anchor_ear: np.ndarray


def _validate_args(args: Args) -> None:
    if not args.dataset:
        raise ValueError("At least one --dataset path is required.")
    if args.dataset_task_id < 0:
        raise ValueError("dataset_task_id must be non-negative.")
    if args.temporal_stride != 10:
        raise ValueError("The fixed first P3T pilot requires --temporal-stride=10.")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("Seeds must be non-negative.")
    if args.train_steps <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Training steps and batch sizes must be positive.")
    if args.validation_examples <= 0:
        raise ValueError("validation_examples must be positive.")
    if not 0.0 < args.validation_fraction < 0.5 or not 0.0 < args.test_fraction < 0.5:
        raise ValueError("Validation and test fractions must lie in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("validation_fraction + test_fraction must be below 0.5.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0 or args.gradient_clip_norm <= 0.0:
        raise ValueError("Optimizer scales must be positive, with non-negative weight decay.")
    if args.warmup_steps < 0 or args.log_interval <= 0:
        raise ValueError("warmup_steps must be non-negative and log_interval positive.")
    if args.early_stopping_patience_logs < 0 or args.early_stopping_min_delta < 0.0:
        raise ValueError("Early-stopping settings must be non-negative.")
    if args.coarse_flow_steps <= 0 or args.final_flow_steps <= 0:
        raise ValueError("Flow step counts must be positive.")
    weights = (
        args.kv_loss_weight,
        args.iar_loss_weight,
        args.ear_loss_weight,
        args.action_loss_weight,
        args.risk_loss_weight,
    )
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Loss weights must be non-negative and at least one must be positive.")
    if args.risk_action_mse_threshold <= 0.0:
        raise ValueError("risk_action_mse_threshold must be positive.")
    if args.kv_delta_floor <= 0.0:
        raise ValueError("kv_delta_floor must be positive.")
    if args.maximum_pairs < 0:
        raise ValueError("maximum_pairs must be non-negative.")
    if args.profile_warmup < 0 or args.profile_iterations <= 0:
        raise ValueError("Profile warmup must be non-negative and iterations positive.")


def _gpu() -> jax.Device:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("P3T training is GPU-only; no JAX GPU device was found.")
    LOGGER.info("Using JAX device %s", devices[0])
    return devices[0]


def _load_norm_stats(
    train_config: config_lib.TrainConfig,
    data_config: config_lib.DataConfig,
    checkpoint_dir: pathlib.Path,
) -> dict[str, Any]:
    if data_config.norm_stats is not None:
        return data_config.norm_stats
    if data_config.asset_id is None:
        raise ValueError("The data config needs asset_id to load checkpoint normalization stats.")
    return checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)


def _with_norm_stats(
    data_config: config_lib.DataConfig,
    norm_stats: dict[str, Any],
) -> config_lib.DataConfig:
    """Replace norm stats while retaining runtime fields attached by factories."""

    updated = dataclasses.replace(data_config, norm_stats=norm_stats)
    declared = {field.name for field in dataclasses.fields(data_config)}
    for name, value in vars(data_config).items():
        if name not in declared:
            object.__setattr__(updated, name, value)
    return updated


def _unwrap_lerobot_dataset(dataset: Any) -> Any:
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset  # noqa: SLF001
    if not hasattr(current, "hf_dataset"):
        raise TypeError(f"Expected a LeRobotDataset with hf_dataset, got {type(current)!r}.")
    return current


def _numpy_column(dataset: Any, name: str) -> np.ndarray:
    values = dataset.hf_dataset.with_format("numpy", columns=[name])[name]
    result = np.asarray(values)
    if result.dtype == object:
        result = np.stack([np.asarray(value) for value in values])
    return result


def _select_pairs(
    arrays: dict[str, np.ndarray],
    raw_dataset: Any,
    *,
    task_id: int,
    temporal_stride: int,
    maximum_pairs: int,
    seed: int,
) -> PairIndices:
    episodes = _numpy_column(raw_dataset, "episode_index").astype(np.int64)
    frames = _numpy_column(raw_dataset, "frame_index").astype(np.int64)
    tasks = _numpy_column(raw_dataset, "task_index").astype(np.int64)
    selected_rows = np.flatnonzero(np.asarray(arrays["task_id"], dtype=np.int64) == task_id)
    if not selected_rows.size:
        available = sorted(int(value) for value in np.unique(arrays["task_id"]))
        raise ValueError(f"No windows for dataset task id {task_id}; available ids are {available}.")

    rows: list[int] = []
    anchors: list[int] = []
    targets: list[int] = []
    pair_episodes: list[int] = []
    pair_frames: list[int] = []
    seen_anchors: set[int] = set()
    for row in selected_rows:
        anchor = int(arrays["anchor_index"][row])
        target = anchor + temporal_stride
        if anchor in seen_anchors or target >= episodes.size:
            continue
        expected_episode = int(arrays["episode_id"][row])
        expected_frame = int(arrays["frame_id"][row])
        valid = (
            int(tasks[anchor]) == task_id
            and int(tasks[target]) == task_id
            and int(episodes[anchor]) == expected_episode
            and int(episodes[target]) == expected_episode
            and int(frames[anchor]) == expected_frame
            and int(frames[target]) == expected_frame + temporal_stride
        )
        if not valid:
            continue
        # Validate every intervening frame, not merely the endpoints.  This
        # makes the action history a true contiguous decision interval.
        offsets = np.arange(temporal_stride + 1, dtype=np.int64)
        interval = anchor + offsets
        if not (
            np.all(episodes[interval] == expected_episode)
            and np.all(tasks[interval] == task_id)
            and np.array_equal(frames[interval], expected_frame + offsets)
        ):
            continue
        seen_anchors.add(anchor)
        rows.append(int(row))
        anchors.append(anchor)
        targets.append(target)
        pair_episodes.append(expected_episode)
        pair_frames.append(expected_frame)

    if not rows:
        raise ValueError("No contiguous anchor-to-anchor+10 P3T pairs survived validation.")
    order = np.arange(len(rows), dtype=np.int64)
    if maximum_pairs and order.size > maximum_pairs:
        rng = np.random.default_rng(seed)
        order = np.sort(rng.choice(order, size=maximum_pairs, replace=False))
    return PairIndices(
        row_indices=np.asarray(rows, dtype=np.int64)[order],
        anchor_indices=np.asarray(anchors, dtype=np.int64)[order],
        target_indices=np.asarray(targets, dtype=np.int64)[order],
        episode_ids=np.asarray(pair_episodes, dtype=np.int64)[order],
        frame_ids=np.asarray(pair_frames, dtype=np.int64)[order],
    )


def _split_pairs(
    pairs: PairIndices,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = np.unique(pairs.episode_ids)
    if episodes.size < 3:
        raise ValueError("P3T requires at least three Task8 episodes for episode-disjoint splits.")
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    test_count = max(1, round(episodes.size * test_fraction))
    validation_count = max(1, round(episodes.size * validation_fraction))
    if test_count + validation_count >= episodes.size:
        test_count = 1
        validation_count = 1
    test_episodes = episodes[:test_count]
    validation_episodes = episodes[test_count : test_count + validation_count]
    test = np.flatnonzero(np.isin(pairs.episode_ids, test_episodes))
    validation = np.flatnonzero(np.isin(pairs.episode_ids, validation_episodes))
    train = np.flatnonzero(~np.isin(pairs.episode_ids, np.concatenate([test_episodes, validation_episodes])))
    if any(not partition.size for partition in (train, validation, test)):
        raise ValueError("Episode-disjoint split produced an empty P3T partition.")
    return train, validation, test


_OBSERVATION_KEYS = (
    "image",
    "image_mask",
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "token_ar_mask",
    "token_loss_mask",
)


def _observation_payload(item: dict[str, Any]) -> dict[str, Any]:
    required = {"image", "image_mask", "state"}
    missing = sorted(required.difference(item))
    if missing:
        raise KeyError(f"Canonical transformed record is missing observation fields: {missing}.")
    return {name: item[name] for name in _OBSERVATION_KEYS if name in item}


def _first_normalized_action(item: dict[str, Any], action_dim: int) -> np.ndarray:
    actions = np.asarray(item.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or not actions.shape[0]:
        raise ValueError(f"Expected normalized action chunk [T,D], got {actions.shape}.")
    result = np.zeros((action_dim,), dtype=np.float32)
    copied = min(action_dim, actions.shape[-1])
    result[:copied] = actions[0, :copied]
    if not np.all(np.isfinite(result)):
        raise ValueError("Normalized executed action contains non-finite values.")
    return result


def _materialize_pairs(
    pairs: PairIndices,
    arrays: dict[str, np.ndarray],
    observation_dataset: data_loader.Dataset,
    *,
    action_dim: int,
    temporal_stride: int,
) -> list[MaterializedPair]:
    """Decode exact observations once and retain no KV tensors."""

    records: list[MaterializedPair] = []
    endpoint_cache: dict[int, dict[str, Any]] = {}
    action_cache: dict[int, np.ndarray] = {}

    def item(index: int) -> dict[str, Any]:
        value = observation_dataset[index]
        if value is None:
            raise ValueError(f"Canonical observation dataset returned None at index {index}.")
        return value

    for position, (row, anchor, target) in enumerate(
        zip(pairs.row_indices, pairs.anchor_indices, pairs.target_indices, strict=True)
    ):
        anchor = int(anchor)
        target = int(target)
        if anchor not in endpoint_cache:
            endpoint_cache[anchor] = _observation_payload(item(anchor))
        if target not in endpoint_cache:
            endpoint_cache[target] = _observation_payload(item(target))

        executed = np.empty((temporal_stride, action_dim), dtype=np.float32)
        for offset in range(temporal_stride):
            index = anchor + offset
            if index not in action_cache:
                # Endpoint records can be reused, but action-only intermediate
                # records are discarded immediately so decoded images do not
                # accumulate in host memory.
                transformed = item(index)
                action_cache[index] = _first_normalized_action(transformed, action_dim)
            executed[offset] = action_cache[index]
        anchor_ear = np.asarray(arrays["fresh_ear"][int(row), 0], dtype=np.float32)
        records.append(
            MaterializedPair(
                anchor_observation=endpoint_cache[anchor],
                current_observation=endpoint_cache[target],
                executed_actions=executed,
                anchor_ear=anchor_ear,
            )
        )
        if position == 0 or (position + 1) % 20 == 0 or position + 1 == len(pairs):
            LOGGER.info("Materialized exact Task8 pairs: %d/%d", position + 1, len(pairs))
    return records


def _batch(records: Sequence[MaterializedPair], indices: np.ndarray) -> dict[str, Any]:
    selected = [records[int(index)] for index in np.asarray(indices, dtype=np.int64)]
    anchor_data = data_loader._collate_fn([record.anchor_observation for record in selected])  # noqa: SLF001
    current_data = data_loader._collate_fn([record.current_observation for record in selected])  # noqa: SLF001
    anchor_data = jax.tree.map(jnp.asarray, anchor_data)
    current_data = jax.tree.map(jnp.asarray, current_data)
    return {
        "anchor_observation": model_lib.Observation.from_dict(anchor_data),
        "current_observation": model_lib.Observation.from_dict(current_data),
        "executed_actions": jnp.asarray(np.stack([record.executed_actions for record in selected])),
        "executed_action_mask": jnp.ones((len(selected), 10), dtype=jnp.bool_),
        "anchor_ear": jnp.asarray(np.stack([record.anchor_ear for record in selected])),
    }


def _low_resolution_images(prefix_state: dict[str, Any], image_size: int) -> jax.Array:
    observation = prefix_state["observation"]
    images = jnp.stack(
        [observation.images["base_0_rgb"], observation.images["left_wrist_0_rgb"]],
        axis=1,
    )
    target_shape = (images.shape[0], 2, image_size, image_size, 3)
    if images.shape != target_shape:
        images = jax.image.resize(images, target_shape, method="linear")
    return jnp.asarray(images, dtype=jnp.float32)


def _active_cache_errors_per_example(
    predicted: tuple[jax.Array, jax.Array],
    target: tuple[jax.Array, jax.Array],
    anchor: tuple[jax.Array, jax.Array],
    config: p3t_prefix_transport.P3TPrefixTransportConfig,
    *,
    delta_floor: float,
) -> tuple[jax.Array, jax.Array]:
    language_start = config.visual_tokens + config.dummy_tokens

    def segment_error(
        left: jax.Array,
        right: jax.Array,
        base: jax.Array,
        token_slice: slice,
    ) -> tuple[jax.Array, jax.Array]:
        error = left[:, :, token_slice].astype(jnp.float32) - right[:, :, token_slice].astype(jnp.float32)
        target_delta = right[:, :, token_slice].astype(jnp.float32) - base[:, :, token_slice].astype(jnp.float32)
        # Normalize each layer/example by how much a genuinely fresh prefix
        # moved from the anchor.  Otherwise the large identity component of KV
        # makes copying the cache appear deceptively good.
        error_power = jnp.mean(jnp.square(error), axis=(2, 3, 4))
        delta_power = jnp.mean(jnp.square(target_delta), axis=(2, 3, 4))
        normalized = error_power / jax.lax.stop_gradient(delta_power + delta_floor)
        return jnp.mean(error_power, axis=0), jnp.mean(normalized, axis=0)

    def tensor_error(left: jax.Array, right: jax.Array, base: jax.Array) -> tuple[jax.Array, jax.Array]:
        visual_raw, visual_normalized = segment_error(
            left,
            right,
            base,
            slice(0, config.visual_tokens),
        )
        language_raw, language_normalized = segment_error(
            left,
            right,
            base,
            slice(language_start, None),
        )
        active_tokens = config.visual_tokens + config.language_tokens
        raw = (config.visual_tokens * visual_raw + config.language_tokens * language_raw) / active_tokens
        normalized = (
            config.visual_tokens * visual_normalized + config.language_tokens * language_normalized
        ) / active_tokens
        return raw, normalized

    k_raw, k_normalized = tensor_error(predicted[0], target[0], anchor[0])
    v_raw, v_normalized = tensor_error(predicted[1], target[1], anchor[1])
    return 0.5 * (k_raw + v_raw), 0.5 * (k_normalized + v_normalized)


def _mse_per_example(predicted: jax.Array, target: jax.Array, *, last_dims: int | None = None) -> jax.Array:
    if last_dims is not None:
        predicted = predicted[..., :last_dims]
        target = target[..., :last_dims]
    axes = tuple(range(1, predicted.ndim))
    return jnp.mean(jnp.square(predicted.astype(jnp.float32) - target.astype(jnp.float32)), axis=axes)


def _reason_and_act(
    base_model: Any,
    prefix_state: dict[str, Any],
    *,
    coarse_flow_steps: int,
    final_flow_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    iar = base_model.sample_actions_profile_implicit(prefix_state)["implicit_action_reason"]
    if iar is None:
        raise ValueError("P3T requires the frozen implicit action reasoner.")
    ear = base_model.sample_actions_profile_coarse(
        prefix_state,
        num_steps=coarse_flow_steps,
        action_cot_denoising_steps=coarse_flow_steps,
        dynamic_denoising_steps=False,
    )["explicit_action_reason"]
    if ear is None:
        raise ValueError("P3T requires the frozen explicit action reasoner.")
    actions = base_model.sample_actions_profile_expert(
        prefix_state,
        ear,
        iar,
        num_steps=final_flow_steps,
    )["actions"]
    return iar, ear, actions


def _forward_losses(
    adapter: p3t_prefix_transport.P3TPrefixTransport,
    base_model: Any,
    batch: dict[str, Any],
    rng: jax.Array,
    *,
    args: Args,
    include_stale: bool,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    # Reusing the exact same key makes both prefix states carry identical
    # coarse/final flow noise.  Only the observation/prefix cache differs.
    anchor_prefix = base_model.sample_actions_profile_prefix(rng, batch["anchor_observation"])
    fresh_prefix = base_model.sample_actions_profile_prefix(rng, batch["current_observation"])
    anchor_kv = jax.tree.map(jax.lax.stop_gradient, anchor_prefix["kv_cache"])
    fresh_kv = jax.tree.map(jax.lax.stop_gradient, fresh_prefix["kv_cache"])
    anchor_images = _low_resolution_images(anchor_prefix, adapter.config.image_size)
    current_images = _low_resolution_images(fresh_prefix, adapter.config.image_size)
    state_delta = (
        fresh_prefix["observation"].state[..., : adapter.config.state_dim]
        - anchor_prefix["observation"].state[..., : adapter.config.state_dim]
    )
    output = adapter(
        anchor_images,
        current_images,
        state_delta,
        batch["executed_actions"],
        batch["executed_action_mask"],
        batch["anchor_ear"],
        anchor_kv,
    )

    transported_prefix = dict(fresh_prefix)
    transported_prefix["kv_cache"] = output.kv_cache
    # The current P3T module predicts KV only.  Existing suffix paths do not
    # consume prefix_out under fixed denoising settings, so retain the causal
    # anchor value rather than leaking the fresh teacher representation.
    transported_prefix["prefix_out"] = jax.lax.stop_gradient(anchor_prefix["prefix_out"])

    target_iar, target_ear, target_actions = _reason_and_act(
        base_model,
        fresh_prefix,
        coarse_flow_steps=args.coarse_flow_steps,
        final_flow_steps=args.final_flow_steps,
    )
    target_iar, target_ear, target_actions = jax.tree.map(
        jax.lax.stop_gradient,
        (target_iar, target_ear, target_actions),
    )
    predicted_iar, predicted_ear, predicted_actions = _reason_and_act(
        base_model,
        transported_prefix,
        coarse_flow_steps=args.coarse_flow_steps,
        final_flow_steps=args.final_flow_steps,
    )

    kv_mse, kv_delta_normalized_mse = _active_cache_errors_per_example(
        output.kv_cache,
        fresh_kv,
        anchor_kv,
        adapter.config,
        delta_floor=args.kv_delta_floor,
    )
    stale_kv_mse, stale_kv_delta_normalized_mse = _active_cache_errors_per_example(
        anchor_kv,
        fresh_kv,
        anchor_kv,
        adapter.config,
        delta_floor=args.kv_delta_floor,
    )
    iar_mse = _mse_per_example(predicted_iar, target_iar)
    ear_mse = _mse_per_example(predicted_ear, target_ear, last_dims=7)
    action_mse = _mse_per_example(predicted_actions, target_actions, last_dims=7)
    risk_target = jax.lax.stop_gradient((action_mse > args.risk_action_mse_threshold).astype(jnp.float32))
    risk_probability = jnp.clip(output.risk.astype(jnp.float32), 1e-6, 1.0 - 1e-6)
    risk_bce = -risk_target * jnp.log(risk_probability) - (1.0 - risk_target) * jnp.log(1.0 - risk_probability)
    total_per_example = (
        args.kv_loss_weight * kv_delta_normalized_mse
        + args.iar_loss_weight * iar_mse
        + args.ear_loss_weight * ear_mse
        + args.action_loss_weight * action_mse
        + args.risk_loss_weight * risk_bce
    )
    metrics: dict[str, jax.Array] = {
        "loss": total_per_example,
        "kv_mse": kv_mse,
        "kv_delta_normalized_mse": kv_delta_normalized_mse,
        "stale_kv_mse": stale_kv_mse,
        "stale_kv_delta_normalized_mse": stale_kv_delta_normalized_mse,
        "iar_mse": iar_mse,
        "ear_mse_7d": ear_mse,
        "action_mse_7d": action_mse,
        "risk_bce": risk_bce,
        "risk_probability": output.risk,
        "risk_target": risk_target,
    }

    if include_stale:
        stale_prefix = dict(fresh_prefix)
        stale_prefix["kv_cache"] = anchor_kv
        stale_prefix["prefix_out"] = jax.lax.stop_gradient(anchor_prefix["prefix_out"])
        stale_iar, stale_ear, stale_actions = _reason_and_act(
            base_model,
            stale_prefix,
            coarse_flow_steps=args.coarse_flow_steps,
            final_flow_steps=args.final_flow_steps,
        )
        metrics.update(
            {
                "stale_iar_mse": _mse_per_example(stale_iar, target_iar),
                "stale_ear_mse_7d": _mse_per_example(stale_ear, target_ear, last_dims=7),
                "stale_action_mse_7d": _mse_per_example(stale_actions, target_actions, last_dims=7),
            }
        )
    return jnp.mean(total_per_example), metrics


def _make_steps(
    base_graphdef: Any,
    adapter_graphdef: Any,
    optimizer: optax.GradientTransformation,
    args: Args,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    @jax.jit
    def train_step(
        base_state: nnx.State,
        adapter_params: nnx.State,
        optimizer_state: optax.OptState,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> tuple[nnx.State, optax.OptState, dict[str, jax.Array]]:
        base_model = nnx.merge(base_graphdef, base_state)

        def loss_fn(candidate: p3t_prefix_transport.P3TPrefixTransport):
            loss, per_example = _forward_losses(
                candidate,
                base_model,
                batch,
                rng,
                args=args,
                include_stale=False,
            )
            return loss, {name: jnp.mean(value) for name, value in per_example.items()}

        adapter = nnx.merge(adapter_graphdef, adapter_params)
        (loss, metrics), gradients = nnx.value_and_grad(loss_fn, has_aux=True)(adapter)
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, adapter_params)
        next_params = optax.apply_updates(adapter_params, updates)
        return next_params, next_optimizer_state, {
            **metrics,
            "loss": loss,
            "gradient_norm": optax.global_norm(gradients),
        }

    @jax.jit
    def eval_step(
        base_state: nnx.State,
        adapter_params: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> dict[str, jax.Array]:
        base_model = nnx.merge(base_graphdef, base_state)
        adapter = nnx.merge(adapter_graphdef, adapter_params)
        _, per_example = _forward_losses(
            adapter,
            base_model,
            batch,
            rng,
            args=args,
            include_stale=True,
        )
        return per_example

    @jax.jit
    def prepare_transport_inputs(
        base_state: nnx.State,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> dict[str, Any]:
        base_model = nnx.merge(base_graphdef, base_state)
        anchor_prefix = base_model.sample_actions_profile_prefix(rng, batch["anchor_observation"])
        current_prefix = base_model.sample_actions_profile_prefix(rng, batch["current_observation"])
        return {
            "anchor_images": _low_resolution_images(anchor_prefix, 64),
            "current_images": _low_resolution_images(current_prefix, 64),
            "state_delta": current_prefix["observation"].state[..., :32]
            - anchor_prefix["observation"].state[..., :32],
            "executed_actions": batch["executed_actions"],
            "executed_action_mask": batch["executed_action_mask"],
            "anchor_ear": batch["anchor_ear"],
            "anchor_kv": anchor_prefix["kv_cache"],
        }

    @jax.jit
    def apply_adapter(adapter_params: nnx.State, inputs: dict[str, Any]):
        adapter = nnx.merge(adapter_graphdef, adapter_params)
        return adapter(
            inputs["anchor_images"],
            inputs["current_images"],
            inputs["state_delta"],
            inputs["executed_actions"],
            inputs["executed_action_mask"],
            inputs["anchor_ear"],
            inputs["anchor_kv"],
        )

    return train_step, eval_step, prepare_transport_inputs, apply_adapter


def _evaluate(
    eval_step: Callable[..., dict[str, jax.Array]],
    base_state: nnx.State,
    adapter_params: nnx.State,
    records: Sequence[MaterializedPair],
    indices: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, indices.size, batch_size):
        selected = indices[start : start + batch_size]
        batch = _batch(records, selected)
        key = jax.random.fold_in(jax.random.key(seed), start)
        result = jax.device_get(eval_step(base_state, adapter_params, batch, key))
        for name, value in result.items():
            pieces.setdefault(name, []).append(np.asarray(value, dtype=np.float64).reshape(-1))
    values = {name: np.concatenate(parts) for name, parts in pieces.items()}
    non_finite = {name: int(np.sum(~np.isfinite(value))) for name, value in values.items()}
    if any(non_finite.values()):
        raise FloatingPointError(f"Non-finite P3T evaluation values: {non_finite}.")
    means = {name: float(np.mean(value)) for name, value in values.items()}

    def ratio(numerator: str, denominator: str) -> float:
        return means[numerator] / max(means[denominator], 1e-12)

    risk_target = values["risk_target"].astype(np.bool_)
    risk_prediction = values["risk_probability"] >= 0.5
    risk_auc = _binary_auc(values["risk_probability"], risk_target)
    ratios = {
        "kv_over_stale": ratio("kv_mse", "stale_kv_mse"),
        "iar_over_stale": ratio("iar_mse", "stale_iar_mse"),
        "ear_over_stale": ratio("ear_mse_7d", "stale_ear_mse_7d"),
        "action_over_stale": ratio("action_mse_7d", "stale_action_mse_7d"),
    }
    checks = {
        "kv_improves_20pct": ratios["kv_over_stale"] <= 0.80,
        "iar_improves_10pct": ratios["iar_over_stale"] <= 0.90,
        "ear_improves_10pct": ratios["ear_over_stale"] <= 0.90,
        "action_improves_10pct": ratios["action_over_stale"] <= 0.90,
    }
    summary = {
        "count": int(indices.size),
        "means": means,
        "ratios": ratios,
        "risk": {
            "accuracy_at_0.5": float(np.mean(risk_prediction == risk_target)),
            "target_refresh_rate": float(np.mean(risk_target)),
            "predicted_refresh_rate": float(np.mean(risk_prediction)),
            "auc": risk_auc,
        },
        "offline_gate": {
            "checks": checks,
            "pass": all(checks.values()),
            "note": "Offline feasibility gate only; Task8 closed-loop success remains mandatory.",
        },
    }
    return summary, values


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    positive = int(np.sum(labels))
    negative = int(labels.size - positive)
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[labels]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _save_params(params: nnx.State, target: pathlib.Path, *, overwrite: bool) -> pathlib.Path:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    item = {"params": {"p3t_prefix_transport": params.to_pure_dict()}}
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, item, force=overwrite)
    return target


def _profile_adapter(
    prepare_transport_inputs: Callable[..., dict[str, Any]],
    apply_adapter: Callable[..., Any],
    base_state: nnx.State,
    adapter_params: nnx.State,
    batch: dict[str, Any],
    *,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    batch_one = jax.tree.map(lambda value: value[:1], batch)
    inputs = prepare_transport_inputs(base_state, batch_one, jax.random.key(seed))
    jax.block_until_ready(inputs["anchor_kv"][0])
    for _ in range(warmup):
        output = apply_adapter(adapter_params, inputs)
        jax.block_until_ready(output.risk)
    samples = np.empty((iterations,), dtype=np.float64)
    for index in range(iterations):
        started = time.perf_counter()
        output = apply_adapter(adapter_params, inputs)
        jax.block_until_ready(output.risk)
        samples[index] = (time.perf_counter() - started) * 1_000.0
    cache_bytes = sum(int(value.size * value.dtype.itemsize) for value in inputs["anchor_kv"])
    return {
        "device": str(jax.devices()[0]),
        "batch_size": 1,
        "warmup": warmup,
        "iterations": iterations,
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "cache_input_bytes": cache_bytes,
        "p95_below_2ms": bool(np.percentile(samples, 95) < 2.0),
        "scope": "P3T adapter only with precomputed 64x64 images and anchor KV; excludes VLM, suffixes, data, and RPC.",
    }


def _load_model_and_dataset(
    args: Args,
) -> tuple[Any, nnx.State, data_loader.Dataset, Any, dict[str, Any]]:
    train_config = config_lib.get_config(args.config_name)
    model_config = train_config.model
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    base_params_path = checkpoint_dir / "params"
    base_params = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(base_params_path, dtype=jnp.bfloat16)
    )
    sidecar_path: pathlib.Path | None = None
    if args.endpoint_student_params is not None:
        sidecar_path = pathlib.Path(download.maybe_download(args.endpoint_student_params))
        sidecar_params = model_lib.convert_str_keys_to_int(
            model_lib.restore_params(sidecar_path, dtype=jnp.bfloat16)
        )
        base_params = policy_config.merge_acot_endpoint_student_params(base_params, sidecar_params)
    base_model = model_config.load(base_params)
    base_model.eval()
    base_graphdef, base_state = nnx.split(base_model)

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = _load_norm_stats(train_config, data_config, checkpoint_dir)
    data_config = _with_norm_stats(data_config, norm_stats)
    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    observation_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    raw_lerobot = _unwrap_lerobot_dataset(raw_dataset)
    metadata = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "base_params": str(base_params_path),
        "endpoint_student_params": str(sidecar_path) if sidecar_path is not None else None,
        "action_dim": int(model_config.action_dim),
        "coarse_horizon": int(model_config.coarse_action_horizon),
        "action_horizon": int(model_config.action_horizon),
    }
    return base_graphdef, base_state, observation_dataset, raw_lerobot, metadata


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _validate_args(args)
    device = _gpu()
    output_dir = pathlib.Path(args.output_dir).resolve()
    metrics_path = output_dir / "metrics.jsonl"
    final_params_path = output_dir / "final" / "params"
    portable_sidecar_path = output_dir / "final" / "p3t_prefix_transport"
    summary_path = output_dir / "summary.json"
    if not args.overwrite and any(
        path.exists() for path in (metrics_path, final_params_path, portable_sidecar_path, summary_path)
    ):
        raise FileExistsError(f"P3T output already exists in {output_dir}; pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = multirate_dataset.load_multirate_arrays(
        args.dataset,
        fields=("anchor_index", "task_id", "episode_id", "frame_id", "fresh_ear"),
    )
    base_graphdef, base_state, observation_dataset, raw_dataset, model_metadata = _load_model_and_dataset(args)
    pairs = _select_pairs(
        arrays,
        raw_dataset,
        task_id=args.dataset_task_id,
        temporal_stride=args.temporal_stride,
        maximum_pairs=args.maximum_pairs,
        seed=args.seed,
    )
    train_indices, validation_indices, test_indices = _split_pairs(
        pairs,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    action_dim = int(model_metadata["action_dim"])
    config = p3t_prefix_transport.P3TPrefixTransportConfig(
        state_dim=action_dim,
        action_dim=action_dim,
        executed_action_horizon=args.temporal_stride,
        ear_horizon=int(model_metadata["coarse_horizon"]),
    )
    records = _materialize_pairs(
        pairs,
        arrays,
        observation_dataset,
        action_dim=action_dim,
        temporal_stride=args.temporal_stride,
    )
    adapter = p3t_prefix_transport.P3TPrefixTransport(
        config,
        rngs=nnx.Rngs(args.seed),
        param_dtype=jnp.float32,
    )
    adapter_graphdef, adapter_params = nnx.split(adapter)
    decay_steps = max(args.train_steps - args.warmup_steps, 1)
    if args.warmup_steps:
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, args.learning_rate, args.warmup_steps),
                optax.cosine_decay_schedule(args.learning_rate, decay_steps, alpha=0.1),
            ],
            [args.warmup_steps],
        )
    else:
        schedule = optax.cosine_decay_schedule(args.learning_rate, args.train_steps, alpha=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.gradient_clip_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    optimizer_state = optimizer.init(adapter_params)
    train_step, eval_step, prepare_transport_inputs, apply_adapter = _make_steps(
        base_graphdef,
        adapter_graphdef,
        optimizer,
        args,
    )

    LOGGER.info(
        "P3T Task8 pilot: pairs=%d train=%d validation=%d test=%d steps=%d batch=%d",
        len(pairs),
        train_indices.size,
        validation_indices.size,
        test_indices.size,
        args.train_steps,
        args.batch_size,
    )
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(args.split_seed + 1)
    validation_probe = validation_rng.choice(
        validation_indices,
        size=min(args.validation_examples, validation_indices.size),
        replace=False,
    )
    best_params: nnx.State | None = None
    best_validation_loss = float("inf")
    best_step = 0
    stale_logs = 0
    completed_steps = 0
    started = time.monotonic()

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, args.train_steps + 1):
            sampled = rng.choice(
                train_indices,
                size=args.batch_size,
                replace=train_indices.size < args.batch_size,
            )
            batch = _batch(records, sampled)
            step_key = jax.random.fold_in(jax.random.key(args.seed), step)
            adapter_params, optimizer_state, train_metrics = train_step(
                base_state,
                adapter_params,
                optimizer_state,
                batch,
                step_key,
            )
            completed_steps = step
            if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
                validation_summary, _ = _evaluate(
                    eval_step,
                    base_state,
                    adapter_params,
                    records,
                    validation_probe,
                    batch_size=args.eval_batch_size,
                    seed=args.seed + 100_000,
                )
                train_host = {name: float(value) for name, value in jax.device_get(train_metrics).items()}
                record = {
                    "step": step,
                    "elapsed_seconds": time.monotonic() - started,
                    **{f"train/{name}": value for name, value in train_host.items()},
                    **{f"validation/{name}": value for name, value in validation_summary["means"].items()},
                    **{f"validation_ratio/{name}": value for name, value in validation_summary["ratios"].items()},
                }
                non_finite = {
                    name: value
                    for name, value in record.items()
                    if isinstance(value, float) and not np.isfinite(value)
                }
                if non_finite:
                    raise FloatingPointError(f"Non-finite P3T training metrics: {non_finite}.")
                metrics_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                validation_loss = validation_summary["means"]["loss"]
                if validation_loss < best_validation_loss - args.early_stopping_min_delta:
                    best_validation_loss = validation_loss
                    best_step = step
                    best_params = adapter_params
                    stale_logs = 0
                else:
                    stale_logs += 1
                if args.early_stopping_patience_logs and stale_logs >= args.early_stopping_patience_logs:
                    LOGGER.info("Early stopping at step %d; best validation step is %d.", step, best_step)
                    break

    selected_params = adapter_params if best_params is None else best_params
    _save_params(selected_params, final_params_path, overwrite=args.overwrite)
    selected_adapter = nnx.merge(adapter_graphdef, selected_params)
    selected_adapter.save(portable_sidecar_path, overwrite=args.overwrite)
    validation_summary, _ = _evaluate(
        eval_step,
        base_state,
        selected_params,
        records,
        validation_indices,
        batch_size=args.eval_batch_size,
        seed=args.seed + 200_000,
    )
    test_summary, _ = _evaluate(
        eval_step,
        base_state,
        selected_params,
        records,
        test_indices,
        batch_size=args.eval_batch_size,
        seed=args.seed + 300_000,
    )
    profile_source = test_indices[:1] if test_indices.size else validation_indices[:1]
    profile = _profile_adapter(
        prepare_transport_inputs,
        apply_adapter,
        base_state,
        selected_params,
        _batch(records, profile_source),
        seed=args.seed + 400_000,
        warmup=args.profile_warmup,
        iterations=args.profile_iterations,
    )
    parameter_count = int(sum(np.prod(value.shape) for value in jax.tree.leaves(selected_params)))
    split_metadata = {
        "train_pairs": int(train_indices.size),
        "validation_pairs": int(validation_indices.size),
        "test_pairs": int(test_indices.size),
        "train_episodes": sorted(int(value) for value in np.unique(pairs.episode_ids[train_indices])),
        "validation_episodes": sorted(int(value) for value in np.unique(pairs.episode_ids[validation_indices])),
        "test_episodes": sorted(int(value) for value in np.unique(pairs.episode_ids[test_indices])),
    }
    summary = {
        "method": "P3T-ACoT plan-conditioned predictive prefix transport",
        "status": "offline_feasibility_only",
        "args": dataclasses.asdict(args),
        "model": model_metadata,
        "adapter_config": dataclasses.asdict(config),
        "device": str(device),
        "pair_definition": {
            "evaluation_task": "Task8",
            "dataset_task_id": args.dataset_task_id,
            "target": "anchor_index + 10 in the same episode",
            "executed_actions": "ten canonical normalized demonstration actions at offsets 0..9",
            "kv_generation": "online frozen anchor/current VLM; no persisted KV dataset",
            "same_flow_noise": True,
        },
        "split": split_metadata,
        "training": {
            "completed_steps": completed_steps,
            "requested_steps": args.train_steps,
            "best_validation_step": best_step,
            "best_validation_loss": best_validation_loss,
            "elapsed_seconds": time.monotonic() - started,
            "trainable_parameter_count": parameter_count,
            "sidecar_params": str(portable_sidecar_path),
            "orbax_sidecar_params": str(final_params_path),
        },
        "validation": validation_summary,
        "test": test_summary,
        "batch1_latency": profile,
        "next_required_evaluation": "Task8 20-trial closed-loop success and end-to-end RPC timing if the offline gate passes.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
