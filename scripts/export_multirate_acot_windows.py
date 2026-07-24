"""Export continuous four-frame windows for the fixed 1:4 ACoT pilot.

Each record contains fresh-observation teacher actions and Action-CoT caches
for four consecutive frames.  Training uses the cache from frame zero for
ages 0..3; the per-frame fresh caches are retained for the 1:1 oracle and
shuffled-plan diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

from openpi.action_cot import multirate_dataset
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader

LOGGER = logging.getLogger("export_multirate_acot_windows")


def _status(message: str) -> None:
    print(f"[export_multirate_acot_windows] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="acot_libero_action_cot_explicit_implicit_co_fusion",
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--b6-sidecar-params",
        required=True,
        help="Complete coarse+final B6 endpoint sidecar used for the matched 1/1 baseline.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows-per-task", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--event-fraction", type=float, default=0.4)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--policy-seed", type=int, default=7)
    parser.add_argument("--records-per-shard", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--action-cot-denoising-steps", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.windows_per_task <= 0 or args.records_per_shard <= 0:
        raise ValueError("--windows-per-task and --records-per-shard must be positive.")
    if args.window_size != 4:
        raise ValueError("The fixed 1:4 pilot requires --window-size=4.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if not 0.0 <= args.event_fraction <= 1.0:
        raise ValueError("--event-fraction must be in [0, 1].")
    if args.num_steps <= 0 or args.action_cot_denoising_steps <= 0:
        raise ValueError("Teacher denoising steps must be positive.")


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


def _unwrap_lerobot_dataset(dataset: Any) -> Any:
    current = dataset
    while hasattr(current, "_dataset"):
        current = getattr(current, "_dataset")
    if not hasattr(current, "hf_dataset"):
        raise TypeError(f"Expected a LeRobotDataset with hf_dataset, got {type(current)!r}.")
    return current


def _numpy_column(dataset: Any, name: str) -> np.ndarray:
    values = dataset.hf_dataset.with_format("numpy", columns=[name])[name]
    result = np.asarray(values)
    if result.dtype == object:
        result = np.stack([np.asarray(value) for value in values])
    return result


def _valid_anchor_indices(
    episodes: np.ndarray,
    frames: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    if episodes.ndim != 1 or frames.ndim != 1 or episodes.shape != frames.shape:
        raise ValueError("episode_index and frame_index must be matching rank-one arrays.")
    if episodes.size < window_size:
        return np.empty((0,), dtype=np.int64)
    starts = np.arange(episodes.size - window_size + 1, dtype=np.int64)
    valid = np.ones(starts.shape, dtype=np.bool_)
    for age in range(1, window_size):
        valid &= episodes[starts + age] == episodes[starts]
        valid &= frames[starts + age] == frames[starts] + age
    return starts[valid]


def _event_scores(actions: np.ndarray, anchors: np.ndarray, window_size: int) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[-1] < 7:
        raise ValueError(f"Expected per-frame actions [N,D>=7], got {actions.shape}.")
    scores = np.empty((anchors.size,), dtype=np.float32)
    for position, anchor in enumerate(anchors):
        window = actions[anchor : anchor + window_size, :7]
        delta = np.diff(window, axis=0)
        motion = float(np.max(np.linalg.norm(delta[:, :6], axis=-1), initial=0.0))
        gripper = float(np.max(np.abs(delta[:, 6]), initial=0.0))
        scores[position] = motion + 10.0 * gripper
    return scores


def _select_anchors(
    episodes: np.ndarray,
    frames: np.ndarray,
    tasks: np.ndarray,
    actions: np.ndarray,
    *,
    window_size: int,
    windows_per_task: int,
    event_fraction: float,
    seed: int,
) -> np.ndarray:
    candidates = _valid_anchor_indices(episodes, frames, window_size=window_size)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for task_id in sorted(int(value) for value in np.unique(tasks[candidates])):
        task_candidates = candidates[tasks[candidates] == task_id]
        if task_candidates.size < windows_per_task:
            raise ValueError(
                f"Task {task_id} has only {task_candidates.size} valid windows; "
                f"requested {windows_per_task}."
            )
        scores = _event_scores(actions, task_candidates, window_size)
        event_count = min(round(windows_per_task * event_fraction), windows_per_task)
        # Sample from the most dynamic half instead of deterministically taking
        # only near-identical gripper transitions from a few episodes.
        order = np.argsort(scores, kind="stable")
        event_pool = task_candidates[order[-max(event_count * 2, event_count) :]]
        event_selected = (
            rng.choice(event_pool, size=event_count, replace=False)
            if event_count
            else np.empty((0,), dtype=np.int64)
        )
        remaining_pool = np.setdiff1d(task_candidates, event_selected, assume_unique=False)
        uniform_count = windows_per_task - event_count
        uniform_selected = rng.choice(remaining_pool, size=uniform_count, replace=False)
        selected.append(np.concatenate([event_selected, uniform_selected]).astype(np.int64))
    if not selected:
        raise ValueError("No task windows were selected.")
    return np.sort(np.concatenate(selected))


def _resize_image(value: Any, image_size: int) -> np.ndarray:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected rank-three image, got {tuple(tensor.shape)}.")
    if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] not in (1, 3, 4):
        raise ValueError(f"Cannot infer image channels from {tuple(tensor.shape)}.")
    tensor = tensor[:3]
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    resized = torch_functional.interpolate(
        tensor[None],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    resized = torch.clamp(torch.round(resized * 255.0), 0, 255).to(torch.uint8)
    return resized.permute(1, 2, 0).cpu().numpy()


def _pad_last_dim(values: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != len(expected_shape):
        raise ValueError(f"Expected rank {len(expected_shape)}, got {array.shape}.")
    if array.shape[:-1] != expected_shape[:-1]:
        raise ValueError(f"Expected leading shape {expected_shape[:-1]}, got {array.shape[:-1]}.")
    target_dim = expected_shape[-1]
    if array.shape[-1] < target_dim:
        array = np.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, target_dim - array.shape[-1])])
    return array[..., :target_dim]


def _existing_anchors(output_dir: pathlib.Path) -> set[int]:
    if not output_dir.exists():
        return set()
    try:
        arrays = multirate_dataset.load_multirate_arrays((output_dir,), fields=("anchor_index",))
    except FileNotFoundError:
        return set()
    return {int(value) for value in arrays["anchor_index"]}


def _teacher_infer(
    policy: Any,
    item: dict[str, Any],
    *,
    seed: int,
    export_acot_cache: bool,
) -> dict[str, Any]:
    inputs = dict(item)
    inputs["policy_seed"] = np.asarray(seed, dtype=np.uint32)
    inputs["profile_policy_timing"] = np.ones((), dtype=np.bool_)
    if export_acot_cache:
        inputs["export_acot_cache"] = np.ones((), dtype=np.bool_)
    return policy.infer(inputs)


def _make_record(
    *,
    policy: Any,
    b6_policy: Any,
    policy_dataset: Any,
    raw_dataset: Any,
    anchor: int,
    args: argparse.Namespace,
    shape: multirate_dataset.MultirateDatasetShape,
    action_horizon: int,
) -> tuple[dict[str, Any], list[float]]:
    images = np.empty(
        (
            shape.window_size,
            shape.num_cameras,
            shape.image_height,
            shape.image_width,
            shape.image_channels,
        ),
        dtype=np.uint8,
    )
    states = np.empty((shape.window_size, shape.state_dim), dtype=np.float32)
    fresh_ear = np.empty(
        (shape.window_size, shape.coarse_horizon, shape.action_dim),
        dtype=np.float32,
    )
    fresh_iar = np.empty(
        (shape.window_size, shape.iar_tokens, shape.iar_dim),
        dtype=np.float32,
    )
    teacher_actions = np.empty((shape.window_size, shape.action_dim), dtype=np.float32)
    b6_actions = np.empty((shape.window_size, shape.action_dim), dtype=np.float32)
    fresh_chunks: list[np.ndarray] = []
    timings: list[float] = []
    window_seed = args.policy_seed + anchor
    anchor_raw = raw_dataset[anchor]
    expected_episode = int(np.asarray(anchor_raw["episode_index"]).item())
    anchor_frame = int(np.asarray(anchor_raw["frame_index"]).item())

    for age in range(shape.window_size):
        index = anchor + age
        policy_item = policy_dataset[index]
        if policy_item is None:
            raise ValueError(f"Policy dataset returned None at index {index}.")
        raw_item = raw_dataset[index]
        result = _teacher_infer(
            policy,
            policy_item,
            seed=window_seed,
            export_acot_cache=True,
        )
        b6_result = _teacher_infer(
            b6_policy,
            policy_item,
            seed=window_seed,
            export_acot_cache=False,
        )
        if shape.image_height != shape.image_width:
            raise ValueError("The pilot image resizer currently requires square images.")
        images[age, 0] = _resize_image(
            policy_item["observation/image"],
            shape.image_height,
        )
        images[age, 1] = _resize_image(
            policy_item["observation/wrist_image"],
            shape.image_height,
        )
        states[age] = _pad_last_dim(
            np.asarray(result["execution_horizon_state_normalized"])[None],
            (1, shape.state_dim),
        )[0]
        fresh_ear[age] = _pad_last_dim(
            result["execution_horizon_coarse_actions_normalized"],
            (shape.coarse_horizon, shape.action_dim),
        )
        iar = np.asarray(result["acot_iar_tokens"], dtype=np.float32)
        if iar.shape != (shape.iar_tokens, shape.iar_dim):
            raise ValueError(
                f"Expected IAR shape {(shape.iar_tokens, shape.iar_dim)}, got {iar.shape}."
            )
        fresh_iar[age] = iar
        chunk = _pad_last_dim(
            result["execution_horizon_final_actions_normalized"],
            (action_horizon, shape.action_dim),
        )
        fresh_chunks.append(chunk)
        teacher_actions[age] = chunk[0]
        timings.append(float(result.get("policy_timing", {}).get("infer_ms", np.nan)))
        b6_chunk = _pad_last_dim(
            b6_result["execution_horizon_final_actions_normalized"],
            (action_horizon, shape.action_dim),
        )
        b6_actions[age] = b6_chunk[0]

        current_episode = int(np.asarray(raw_item["episode_index"]).item())
        expected_frame = anchor_frame + age
        current_frame = int(np.asarray(raw_item["frame_index"]).item())
        if current_episode != expected_episode or current_frame != expected_frame:
            raise ValueError(
                f"Window continuity changed while decoding anchor {anchor}: "
                f"age={age}, episode={current_episode}, frame={current_frame}."
            )

    hold_actions = np.stack([fresh_chunks[0][age] for age in range(shape.window_size)])
    action_delta = np.zeros_like(teacher_actions[:, 0], dtype=np.float32)
    action_delta[1:] = np.linalg.norm(
        teacher_actions[1:, :6] - teacher_actions[:-1, :6],
        axis=-1,
    )
    gripper_delta = np.zeros_like(action_delta)
    gripper_delta[1:] = np.abs(teacher_actions[1:, 6] - teacher_actions[:-1, 6])
    event_mask = (action_delta > 0.25) | (gripper_delta > 0.5)
    return (
        {
            "anchor_index": anchor,
            "task_id": int(np.asarray(anchor_raw["task_index"]).item()),
            "episode_id": int(np.asarray(anchor_raw["episode_index"]).item()),
            "frame_id": int(np.asarray(anchor_raw["frame_index"]).item()),
            "policy_seed": window_seed,
            "images": images,
            "states": states,
            "fresh_ear": fresh_ear,
            "fresh_iar": fresh_iar,
            "teacher_actions": teacher_actions,
            "b6_actions": b6_actions,
            "hold_actions": hold_actions,
            "event_mask": event_mask,
        },
        timings,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = _parse_args()
    _validate_args(args)
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    train_config = config_lib.get_config(args.config_name)
    model_config = train_config.model
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats = _load_norm_stats(train_config, data_config, checkpoint_dir)

    raw_dataset = data_loader.create_torch_dataset(data_config, model_config)
    policy_dataset = data_loader.TransformedDataset(
        raw_dataset,
        [*data_config.repack_transforms.inputs],
    )
    base_dataset = _unwrap_lerobot_dataset(raw_dataset)
    episodes = _numpy_column(base_dataset, "episode_index").astype(np.int64)
    frames = _numpy_column(base_dataset, "frame_index").astype(np.int64)
    tasks = _numpy_column(base_dataset, "task_index").astype(np.int64)
    actions = _numpy_column(base_dataset, "actions").astype(np.float32)
    anchors = _select_anchors(
        episodes,
        frames,
        tasks,
        actions,
        window_size=args.window_size,
        windows_per_task=args.windows_per_task,
        event_fraction=args.event_fraction,
        seed=args.selection_seed,
    )

    _status(f"Selected {anchors.size} windows across {np.unique(tasks[anchors]).size} tasks.")
    sample_kwargs = {
        "num_steps": args.num_steps,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
    }
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        sample_kwargs=sample_kwargs,
    )
    b6_policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        sample_kwargs={
            "num_steps": 1,
            "action_cot_denoising_steps": 1,
        },
        acot_endpoint_student_params=args.b6_sidecar_params,
    )
    shape = multirate_dataset.MultirateDatasetShape(
        window_size=args.window_size,
        num_cameras=2,
        image_height=args.image_size,
        image_width=args.image_size,
        image_channels=3,
        state_dim=model_config.action_dim,
        action_dim=model_config.action_dim,
        coarse_horizon=model_config.coarse_action_horizon,
        iar_tokens=18,
        iar_dim=1024,
    )
    output_dir = pathlib.Path(args.output_dir)
    metadata = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "b6_sidecar_params": args.b6_sidecar_params,
        "windows_per_task": args.windows_per_task,
        "event_fraction": args.event_fraction,
        "selection_seed": args.selection_seed,
        "policy_seed": args.policy_seed,
        "num_steps": args.num_steps,
        "action_cot_denoising_steps": args.action_cot_denoising_steps,
        "definition": "fixed 1:4 cache; all four frames have fresh teacher/cache targets",
    }
    existing = _existing_anchors(output_dir)
    timing_values: list[float] = []
    processed = 0
    failed = 0

    with multirate_dataset.ShardedMultirateWriter(
        output_dir,
        shape=shape,
        records_per_shard=args.records_per_shard,
        metadata=metadata,
    ) as writer:
        for position, anchor_value in enumerate(anchors):
            anchor = int(anchor_value)
            if anchor in existing:
                continue
            try:
                record, timings = _make_record(
                    policy=policy,
                    b6_policy=b6_policy,
                    policy_dataset=policy_dataset,
                    raw_dataset=raw_dataset,
                    anchor=anchor,
                    args=args,
                    shape=shape,
                    action_horizon=model_config.action_horizon,
                )
                writer.append(record)
                timing_values.extend(timings)
                processed += 1
                if processed == 1 or processed % 10 == 0:
                    finite = np.asarray(timing_values, dtype=np.float64)
                    finite = finite[np.isfinite(finite)]
                    mean_ms = float(np.mean(finite)) if finite.size else float("nan")
                    _status(
                        f"{position + 1}/{anchors.size}: wrote={processed}, failed={failed}, "
                        f"teacher_mean_ms={mean_ms:.3f}"
                    )
            except Exception:
                failed += 1
                LOGGER.exception("Failed anchor %d", anchor)
                if not args.continue_on_error:
                    raise

    finite = np.asarray(timing_values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    summary = {
        "selected_windows": int(anchors.size),
        "newly_processed_windows": processed,
        "failed_windows": failed,
        "existing_windows_skipped": len(existing.intersection(int(value) for value in anchors)),
        "teacher_calls_per_window": shape.window_size,
        "mean_teacher_infer_ms": float(np.mean(finite)) if finite.size else None,
        "p95_teacher_infer_ms": float(np.quantile(finite, 0.95)) if finite.size else None,
        "metadata": metadata,
        "note": "Teacher/open-loop export only; this is not a success-rate result.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _status(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
