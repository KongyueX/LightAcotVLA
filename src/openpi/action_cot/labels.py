"""Action-CoT Stage C label loading utilities."""

from __future__ import annotations

import functools
import pathlib
from typing import Any

import numpy as np


def cap_skip_mask(
    skip_mask: np.ndarray,
    entropy: np.ndarray,
    max_skip_segments: int | None,
) -> np.ndarray:
    """Limit a segment skip mask to the lowest-entropy skipped segments."""
    mask = np.asarray(skip_mask, dtype=np.int8).copy()
    if max_skip_segments is None:
        return mask
    if max_skip_segments < 0:
        raise ValueError(f"max_skip_segments must be non-negative, got {max_skip_segments}.")

    skipped = np.flatnonzero(mask > 0)
    if skipped.size <= max_skip_segments:
        return mask

    entropy = np.asarray(entropy, dtype=np.float64)
    if entropy.shape[0] != mask.shape[0]:
        raise ValueError(f"entropy length {entropy.shape[0]} does not match skip_mask length {mask.shape[0]}.")

    sortable = np.nan_to_num(entropy[skipped], nan=np.inf, posinf=np.inf, neginf=-np.inf)
    kept_skips = skipped[np.argsort(sortable, kind="stable")[:max_skip_segments]]
    capped = np.zeros_like(mask)
    capped[kept_skips] = 1
    return capped


def pad_segment_labels(
    skip_mask: np.ndarray,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a variable-length segment skip mask and emit a valid-segment mask."""
    mask = np.asarray(skip_mask, dtype=np.float32)
    if mask.ndim != 1:
        raise ValueError(f"skip_mask must have shape [N], got {mask.shape}.")
    if max_segments <= 0:
        raise ValueError(f"max_segments must be positive, got {max_segments}.")
    if mask.shape[0] > max_segments:
        raise ValueError(f"skip_mask has {mask.shape[0]} segments, larger than max_segments={max_segments}.")

    padded = np.zeros((max_segments,), dtype=np.float32)
    valid = np.zeros((max_segments,), dtype=np.float32)
    padded[: mask.shape[0]] = mask
    valid[: mask.shape[0]] = 1.0
    return padded, valid


class ActionCotLabelLoader:
    """Loads Stage A per-sample labels by dataset index."""

    def __init__(
        self,
        entropy_dir: str | pathlib.Path,
        *,
        max_segments: int,
        max_skip_segments: int | None = None,
        cache_size: int = 8192,
    ):
        self.entropy_dir = pathlib.Path(entropy_dir)
        self.max_segments = max_segments
        self.max_skip_segments = max_skip_segments
        self._load_cached = functools.lru_cache(maxsize=cache_size)(self._load_uncached)

    def load(self, index: int) -> dict[str, np.ndarray]:
        return self._load_cached(int(index))

    def _load_uncached(self, index: int) -> dict[str, np.ndarray]:
        path = self.entropy_dir / f"sample_{index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Action-CoT label file not found: {path}")

        with np.load(path, allow_pickle=True) as data:
            if "skip_mask" not in data:
                raise KeyError(f"{path} does not contain skip_mask.")
            if "entropy" not in data:
                raise KeyError(f"{path} does not contain entropy.")
            skip_mask = np.asarray(data["skip_mask"], dtype=np.int8)
            entropy = np.asarray(data["entropy"], dtype=np.float64)

        skip_mask = cap_skip_mask(skip_mask, entropy, self.max_skip_segments)
        padded, valid = pad_segment_labels(skip_mask, self.max_segments)
        return {
            "action_cot_skip_mask": padded,
            "action_cot_skip_valid_mask": valid,
        }


class ActionCotLabelDataset:
    """Adds Action-CoT Stage C labels to a random-access dataset."""

    def __init__(
        self,
        dataset: Any,
        entropy_dir: str | pathlib.Path,
        *,
        max_segments: int,
        max_skip_segments: int | None = None,
    ):
        self.dataset = dataset
        self.labels = ActionCotLabelLoader(
            entropy_dir,
            max_segments=max_segments,
            max_skip_segments=max_skip_segments,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if item is None:
            return None
        item = dict(item)
        item.update(self.labels.load(int(index)))
        return item
