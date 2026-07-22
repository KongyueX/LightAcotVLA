"""Frame samplers used by the LeRobot training pipeline."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import random
from typing import Any

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import torch


def get_base_dataset(dataset: Any) -> Any:
    """Remove local transform wrappers without unwrapping MultiLeRobotDataset."""
    if hasattr(dataset, "_dataset"):
        return get_base_dataset(dataset._dataset)  # noqa: SLF001
    return dataset


def _subdatasets(dataset: Any) -> list[tuple[Any, int]]:
    """Return unwrapped LeRobot datasets and their global frame offsets."""
    base_dataset = get_base_dataset(dataset)
    if isinstance(base_dataset, lerobot_dataset.MultiLeRobotDataset):
        result = []
        global_offset = 0
        for wrapped_dataset in base_dataset._datasets:  # noqa: SLF001
            result.append((get_base_dataset(wrapped_dataset), global_offset))
            global_offset += len(wrapped_dataset)
        return result
    return [(base_dataset, 0)]


def _episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int]:
    episode_data_index = dataset.episode_data_index
    if episode_index < 0 or episode_index >= len(episode_data_index["from"]):
        raise IndexError(f"Episode index {episode_index} is outside this dataset")
    start = int(episode_data_index["from"][episode_index].item())
    end = int(episode_data_index["to"][episode_index].item())
    return start, end


def _episode_tasks(dataset: Any, episode_index: int) -> tuple[str, ...]:
    episodes = dataset.meta.episodes
    episode = episodes.get(episode_index, episodes.get(str(episode_index)))
    if episode is None:
        raise KeyError(f"Episode {episode_index} is missing from LeRobot metadata")
    tasks = episode.get("tasks", ())
    return tuple(str(task) for task in tasks)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _matches_target_task(tasks: Sequence[str], patterns: Sequence[str]) -> bool:
    normalized_tasks = tuple(_normalize_text(task) for task in tasks)
    normalized_patterns = tuple(_normalize_text(pattern) for pattern in patterns)
    return any(pattern in task for task in normalized_tasks for pattern in normalized_patterns)


def sample_subtask(dataset: Any) -> list[tuple[int, int]]:
    """Collect legacy high-level instruction intervals."""
    valid_intervals = []
    total_episodes_processed = 0
    subdatasets = _subdatasets(dataset)
    print(f"Processing {len(subdatasets)} sub-datasets...")

    for inner_dataset, global_offset in subdatasets:
        instruction_segments = inner_dataset.meta.info.get("instruction_segments", {})
        num_episodes = len(inner_dataset.episode_data_index["from"])

        for episode_index in range(num_episodes):
            episode_start, _ = _episode_bounds(inner_dataset, episode_index)
            tasks = instruction_segments.get(str(episode_index))
            if tasks is None:
                continue

            for subtask in tasks:
                local_start = int(subtask["start_frame_index"]) + episode_start
                local_end = int(subtask["success_frame_index"]) + episode_start
                instruction = str(subtask["instruction"]).lower()
                is_reset = any(keyword in instruction for keyword in ("reset", "return", "default"))
                if is_reset and local_end - local_start > 90:
                    local_end = local_start + 45
                valid_intervals.append((local_start + global_offset, local_end + global_offset))

        total_episodes_processed += num_episodes

    print(f"Total {len(valid_intervals)} valid intervals from {total_episodes_processed} episodes.")
    return valid_intervals


def _target_frame_indices(dataset: Any, target_tasks: Sequence[str]) -> list[int]:
    indices = []
    for inner_dataset, global_offset in _subdatasets(dataset):
        num_episodes = len(inner_dataset.episode_data_index["from"])
        for episode_index in range(num_episodes):
            if not _matches_target_task(_episode_tasks(inner_dataset, episode_index), target_tasks):
                continue
            start, end = _episode_bounds(inner_dataset, episode_index)
            indices.extend(range(global_offset + start, global_offset + end))
    return indices


def _manifest_frame_weights(
    dataset: Any,
    manifest_path: str | None,
    manifest_split: str,
) -> dict[int, float]:
    if manifest_path is None:
        return {}

    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Sampler manifest does not exist: {path}")

    subdatasets = _subdatasets(dataset)
    by_repo_id: dict[str, tuple[Any, int]] = {}
    for inner_dataset, global_offset in subdatasets:
        repo_id = str(getattr(inner_dataset, "repo_id", ""))
        if repo_id:
            if repo_id in by_repo_id:
                raise ValueError(f"Duplicate repo_id in MultiLeRobotDataset: {repo_id}")
            by_repo_id[repo_id] = (inner_dataset, global_offset)

    frame_weights: dict[int, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error

            entry_split = str(entry.get("split", manifest_split))
            if entry_split != manifest_split:
                continue

            repo_id = entry.get("repo_id")
            if repo_id is None:
                if len(subdatasets) != 1:
                    raise ValueError(f"Manifest entry at {path}:{line_number} needs repo_id for a multi-dataset")
                inner_dataset, global_offset = subdatasets[0]
            else:
                try:
                    inner_dataset, global_offset = by_repo_id[str(repo_id)]
                except KeyError as error:
                    raise ValueError(
                        f"Manifest entry at {path}:{line_number} references unknown repo_id {repo_id!r}"
                    ) from error

            try:
                episode_index = int(entry["episode_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Manifest entry at {path}:{line_number} needs an integer episode_index") from error

            episode_start, episode_end = _episode_bounds(inner_dataset, episode_index)
            episode_length = episode_end - episode_start
            start_frame = int(entry.get("start_frame", entry.get("frame_index", 0)))
            end_frame = int(entry.get("end_frame", entry.get("frame_index", start_frame)))
            if start_frame < 0 or end_frame < start_frame or end_frame >= episode_length:
                raise ValueError(
                    f"Manifest range [{start_frame}, {end_frame}] at {path}:{line_number} "
                    f"is invalid for episode length {episode_length}"
                )

            weight = float(entry.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"Manifest weight must be positive at {path}:{line_number}")

            for frame_index in range(start_frame, end_frame + 1):
                global_index = global_offset + episode_start + frame_index
                # Overlapping phase windows should not accidentally multiply a
                # frame's probability. An explicit larger weight still wins.
                frame_weights[global_index] = max(frame_weights.get(global_index, 0.0), weight)

    return frame_weights


def _mixture_weights(
    dataset: Any,
    *,
    manifest_path: str | None,
    manifest_split: str,
    target_tasks: Sequence[str],
    general_fraction: float,
    target_fraction: float,
    manifest_fraction: float,
) -> torch.Tensor:
    component_fractions = (general_fraction, target_fraction, manifest_fraction)
    if any(fraction < 0 for fraction in component_fractions):
        raise ValueError(f"Sampler fractions must be non-negative, got {component_fractions}")
    fraction_sum = sum(component_fractions)
    if fraction_sum <= 0:
        raise ValueError("At least one sampler fraction must be positive")
    general_fraction, target_fraction, manifest_fraction = (fraction / fraction_sum for fraction in component_fractions)

    dataset_size = len(dataset)
    weights = torch.zeros(dataset_size, dtype=torch.double)
    if general_fraction:
        weights += general_fraction / dataset_size

    target_indices: list[int] = []
    if target_fraction:
        if not target_tasks:
            raise ValueError("target_fraction is positive but sampler_target_tasks is empty")
        target_indices = _target_frame_indices(dataset, target_tasks)
        if not target_indices:
            raise ValueError(f"No dataset frames matched target tasks: {tuple(target_tasks)!r}")
        weights[target_indices] += target_fraction / len(target_indices)

    manifest_weights: dict[int, float] = {}
    if manifest_fraction:
        manifest_weights = _manifest_frame_weights(dataset, manifest_path, manifest_split)
        if not manifest_weights:
            raise ValueError(f"No manifest frames with split {manifest_split!r} were loaded from {manifest_path!r}")
        manifest_total = sum(manifest_weights.values())
        manifest_indices = list(manifest_weights)
        manifest_values = torch.tensor([manifest_weights[index] for index in manifest_indices], dtype=torch.double)
        weights[manifest_indices] += manifest_fraction * manifest_values / manifest_total

    print(
        "Mixture sampler: "
        f"frames={dataset_size}, target_frames={len(target_indices)}, "
        f"manifest_frames={len(manifest_weights)}, "
        f"fractions=({general_fraction:.3f}, {target_fraction:.3f}, {manifest_fraction:.3f})"
    )
    return weights


class FrameSampler(torch.utils.data.Sampler[int]):
    """Legacy interval sampler plus reproducible targeted-mixture sampling."""

    def __init__(
        self,
        dataset: Any,
        sampler_type: str,
        *,
        seed: int = 0,
        manifest_path: str | None = None,
        manifest_split: str = "train",
        target_tasks: Sequence[str] = (),
        general_fraction: float = 1.0,
        target_fraction: float = 0.0,
        manifest_fraction: float = 0.0,
        num_samples: int | None = None,
    ) -> None:
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)
        self._weights: torch.Tensor | None = None

        if sampler_type == "subtask":
            valid_intervals = sample_subtask(dataset)
            self._valid_indices = self._sample_frames(valid_intervals, len(dataset), seed)
            self._num_samples = len(self._valid_indices)
        elif sampler_type == "mixture":
            self._valid_indices = []
            self._weights = _mixture_weights(
                dataset,
                manifest_path=manifest_path,
                manifest_split=manifest_split,
                target_tasks=target_tasks,
                general_fraction=general_fraction,
                target_fraction=target_fraction,
                manifest_fraction=manifest_fraction,
            )
            self._num_samples = num_samples or len(dataset)
            if self._num_samples <= 0:
                raise ValueError(f"sampler_num_samples must be positive, got {self._num_samples}")
        else:
            raise ValueError(f"Invalid sampler type: {sampler_type}")

    @staticmethod
    def _sample_frames(intervals: Sequence[tuple[int, int]], dataset_size: int, seed: int) -> list[int]:
        valid_indices = []
        for start_index, end_index in intervals:
            bounded_start = max(0, start_index)
            bounded_end = min(dataset_size - 1, end_index)
            valid_indices.extend(range(bounded_start, bounded_end + 1))

        valid_indices = sorted(set(valid_indices))
        random.Random(seed).shuffle(valid_indices)
        print(f"Total {len(valid_indices)} valid indices, original: {dataset_size}")
        return valid_indices

    def __iter__(self):
        if self._weights is None:
            return iter(self._valid_indices)
        sampled = torch.multinomial(
            self._weights,
            self._num_samples,
            replacement=True,
            generator=self._generator,
        )
        return iter(sampled.tolist())

    def __len__(self) -> int:
        return self._num_samples
