"""Action-CoT Stage C label loading utilities."""

from __future__ import annotations

import collections
from collections.abc import Sequence
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


def entropy_score(entropy: np.ndarray) -> float:
    values = np.asarray(entropy, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.mean(finite))


def step_label_from_entropy_score(score: float, thresholds: Sequence[float]) -> int:
    label = 0
    for threshold in thresholds:
        if score > threshold:
            label += 1
    return label


def estimate_step_thresholds(
    entropy_dir: str | pathlib.Path,
    quantiles: Sequence[float],
) -> tuple[float, ...]:
    paths = sorted(pathlib.Path(entropy_dir).glob("sample_*.npz"))
    if not paths:
        paths = sorted(pathlib.Path(entropy_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz entropy files found in {entropy_dir}.")

    scores = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if "entropy" not in data:
                continue
            scores.append(entropy_score(np.asarray(data["entropy"], dtype=np.float64)))
    if not scores:
        raise ValueError(f"No usable entropy arrays found in {entropy_dir}.")

    quantiles = tuple(float(q) for q in quantiles)
    if any(q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError(f"step quantiles must be in (0, 1), got {quantiles}.")
    if any(left >= right for left, right in zip(quantiles, quantiles[1:])):
        raise ValueError(f"step quantiles must be strictly increasing, got {quantiles}.")
    return tuple(float(value) for value in np.quantile(np.asarray(scores, dtype=np.float64), quantiles))


class ActionCotLabelLoader:
    """Loads Stage A per-sample labels by dataset index."""

    def __init__(
        self,
        entropy_dir: str | pathlib.Path,
        *,
        max_segments: int,
        max_skip_segments: int | None = None,
        step_values: Sequence[int] | None = None,
        step_quantiles: Sequence[float] | None = None,
        step_thresholds: Sequence[float] | None = None,
        cache_size: int = 8192,
    ):
        self.entropy_dir = pathlib.Path(entropy_dir)
        self.max_segments = max_segments
        self.max_skip_segments = max_skip_segments
        self.step_values = tuple(int(value) for value in step_values) if step_values is not None else None
        if self.step_values is not None and len(self.step_values) < 2:
            raise ValueError("step_values must contain at least two classes when provided.")
        if self.step_values is not None and any(value <= 0 for value in self.step_values):
            raise ValueError(f"step_values must be positive, got {self.step_values}.")
        if step_thresholds is not None:
            self.step_thresholds = tuple(float(value) for value in step_thresholds)
        elif self.step_values is not None:
            quantiles = step_quantiles
            if quantiles is None:
                quantiles = tuple((idx + 1) / len(self.step_values) for idx in range(len(self.step_values) - 1))
            self.step_thresholds = estimate_step_thresholds(self.entropy_dir, quantiles)
        else:
            self.step_thresholds = None
        if self.step_values is not None and len(self.step_thresholds or ()) != len(self.step_values) - 1:
            raise ValueError(
                "step_thresholds length must equal len(step_values) - 1, "
                f"got thresholds={self.step_thresholds}, step_values={self.step_values}."
            )
        self.cache_size = cache_size
        self._cache: collections.OrderedDict[int, dict[str, np.ndarray]] = collections.OrderedDict()

    def load(self, index: int) -> dict[str, np.ndarray]:
        index = int(index)
        if self.cache_size <= 0:
            return self._load_uncached(index)

        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        item = self._load_uncached(index)
        self._cache[index] = item
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item

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
        item = {
            "action_cot_skip_mask": padded,
            "action_cot_skip_valid_mask": valid,
        }
        if self.step_values is not None:
            label = step_label_from_entropy_score(entropy_score(entropy), self.step_thresholds or ())
            label = min(max(label, 0), len(self.step_values) - 1)
            item["action_cot_step_label"] = np.asarray(label, dtype=np.int32)
            item["action_cot_step_value"] = np.asarray(self.step_values[label], dtype=np.int32)
        return item


class ActionCotLabelDataset:
    """Adds Action-CoT Stage C labels to a random-access dataset."""

    def __init__(
        self,
        dataset: Any,
        entropy_dir: str | pathlib.Path,
        *,
        max_segments: int,
        max_skip_segments: int | None = None,
        step_values: Sequence[int] | None = None,
        step_quantiles: Sequence[float] | None = None,
        step_thresholds: Sequence[float] | None = None,
    ):
        self.dataset = dataset
        self.labels = ActionCotLabelLoader(
            entropy_dir,
            max_segments=max_segments,
            max_skip_segments=max_skip_segments,
            step_values=step_values,
            step_quantiles=step_quantiles,
            step_thresholds=step_thresholds,
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
