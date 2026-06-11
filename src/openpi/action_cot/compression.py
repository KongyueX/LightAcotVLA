"""Offline Action-CoT segmentation and entropy-label helpers."""

from collections.abc import Sequence

import numpy as np

Segment = tuple[int, int]


def _as_trajectory(coarse_actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(coarse_actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"coarse_actions must have shape [T, D], got {actions.shape}.")
    return actions


def _as_samples(coarse_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(coarse_samples, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError(f"coarse_samples must have shape [K, T, D], got {samples.shape}.")
    return samples


def _segments_cover(segments: Sequence[Segment], t_len: int) -> bool:
    cursor = 0
    for start, end in segments:
        if start != cursor or end <= start or end > t_len:
            return False
        cursor = end
    return cursor == t_len


def segment_fixed(coarse_actions: np.ndarray, chunk_size: int) -> list[Segment]:
    """Split a coarse action trajectory into fixed-size contiguous chunks."""
    actions = _as_trajectory(coarse_actions)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    t_len = actions.shape[0]
    return [(start, min(start + chunk_size, t_len)) for start in range(0, t_len, chunk_size)]


def _boundary_scores(
    actions: np.ndarray,
    gripper_indices: Sequence[int] | None,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    t_len, action_dim = actions.shape
    scores = np.zeros(t_len, dtype=np.float64)

    if t_len >= 2:
        deltas = actions[1:] - actions[:-1]
        scores[1:] += alpha * np.linalg.norm(deltas, axis=-1)

        if gripper_indices is not None:
            indices = np.asarray(gripper_indices, dtype=np.int64)
            if indices.size:
                indices = np.where(indices < 0, indices + action_dim, indices)
                if np.any((indices < 0) | (indices >= action_dim)):
                    raise ValueError(f"gripper_indices must be within [0, {action_dim}), got {gripper_indices}.")
                gripper = actions[:, indices]
                scores[1:] += gamma * np.linalg.norm(gripper[1:] - gripper[:-1], axis=-1)

    if t_len >= 3:
        deltas = actions[1:] - actions[:-1]
        curvature = deltas[1:] - deltas[:-1]
        scores[2:] += beta * np.linalg.norm(curvature, axis=-1)

    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def _local_maxima(scores: np.ndarray) -> list[int]:
    candidates: list[int] = []
    # Boundary index t means a new segment starts at frame t, so valid boundaries are in [1, T).
    for idx in range(1, scores.shape[0]):
        left = scores[idx - 1]
        right = scores[idx + 1] if idx + 1 < scores.shape[0] else -np.inf
        value = scores[idx]
        if value >= left and value >= right and (value > left or value > right):
            candidates.append(idx)
    return candidates


def _choose_boundary(
    start: int,
    t_len: int,
    scores: np.ndarray,
    candidates: Sequence[int],
    min_len: int,
    max_len: int,
    max_segments: int,
    num_segments: int,
) -> int:
    remaining = t_len - start
    if remaining <= max_len:
        return t_len

    remaining_slots_after = max_segments - num_segments - 1
    if remaining_slots_after <= 0:
        return t_len

    lower = start + min_len
    upper = min(start + max_len, t_len - min_len)
    lower = max(lower, t_len - remaining_slots_after * max_len)
    if lower > upper:
        return -1

    window = [idx for idx in candidates if lower <= idx <= upper]
    if window:
        return max(window, key=lambda idx: (scores[idx], -abs(idx - (start + max_len)), -idx))

    forced = min(start + max_len, t_len - min_len)
    return forced if lower <= forced <= upper else -1


def _valid_adaptive_segments(segments: Sequence[Segment], t_len: int, min_len: int, max_len: int) -> bool:
    if not _segments_cover(segments, t_len):
        return False
    if t_len == 0:
        return len(segments) == 0
    if t_len < min_len:
        return len(segments) == 1 and segments[0] == (0, t_len)
    return all(min_len <= end - start <= max_len for start, end in segments)


def segment_adaptive(
    coarse_actions: np.ndarray,
    min_len: int = 3,
    max_len: int = 8,
    max_segments: int = 8,
    gripper_indices: Sequence[int] | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
) -> list[Segment]:
    """Segment normalized coarse actions using local action-change maxima.

    Falls back to fixed chunks of size 5 when the adaptive constraints cannot produce
    a contiguous, non-overlapping segmentation.
    """
    actions = _as_trajectory(coarse_actions)
    if min_len <= 0:
        raise ValueError(f"min_len must be positive, got {min_len}.")
    if max_len < min_len:
        raise ValueError(f"max_len must be >= min_len, got min_len={min_len}, max_len={max_len}.")
    if max_segments <= 0:
        raise ValueError(f"max_segments must be positive, got {max_segments}.")

    t_len = actions.shape[0]
    if t_len == 0:
        return []
    if t_len <= max_len:
        return [(0, t_len)]
    if t_len > max_segments * max_len:
        return segment_fixed(actions, chunk_size=5)

    scores = _boundary_scores(actions, gripper_indices, alpha, beta, gamma)
    candidates = _local_maxima(scores)
    if not candidates:
        candidates = list(range(1, t_len))
    candidates = sorted(candidates, key=lambda idx: (scores[idx], -idx), reverse=True)

    segments: list[Segment] = []
    start = 0
    while start < t_len:
        end = _choose_boundary(start, t_len, scores, candidates, min_len, max_len, max_segments, len(segments))
        if end <= start:
            return segment_fixed(actions, chunk_size=5)
        segments.append((start, end))
        start = end

        if len(segments) > max_segments:
            return segment_fixed(actions, chunk_size=5)

    if not _valid_adaptive_segments(segments, t_len, min_len, max_len):
        return segment_fixed(actions, chunk_size=5)
    return segments


def compute_mc_predictive_entropy(
    coarse_samples: np.ndarray,
    segments: Sequence[Segment],
    eps: float = 1e-6,
    dim_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a length-normalized MC predictive entropy proxy per segment."""
    samples = _as_samples(coarse_samples)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if not _segments_cover(segments, samples.shape[1]):
        raise ValueError("segments must be contiguous and cover the sample time dimension.")

    weights = None
    if dim_weights is not None:
        weights = np.asarray(dim_weights, dtype=np.float64)
        if weights.shape != (samples.shape[2],):
            raise ValueError(f"dim_weights must have shape [{samples.shape[2]}], got {weights.shape}.")
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise ValueError("dim_weights must have a positive sum.")
        weights = weights / weight_sum

    entropy = []
    for start, end in segments:
        var = np.var(samples[:, start:end, :], axis=0)
        log_var = np.log(var + eps)
        if weights is None:
            entropy.append(float(np.mean(log_var)))
        else:
            entropy.append(float(np.mean(np.sum(log_var * weights[None, :], axis=-1))))
    return np.asarray(entropy, dtype=np.float64)


def make_skip_mask(
    entropy: np.ndarray,
    prune_ratio: float | None = None,
    threshold: float | None = None,
    min_keep: int = 1,
    max_skip_ratio: float = 0.7,
) -> np.ndarray:
    """Create a segment-level skip mask where 1 means skip and 0 means keep."""
    entropy = np.asarray(entropy, dtype=np.float64)
    if entropy.ndim != 1:
        raise ValueError(f"entropy must have shape [N], got {entropy.shape}.")
    if min_keep < 0:
        raise ValueError(f"min_keep must be non-negative, got {min_keep}.")
    if not 0.0 <= max_skip_ratio <= 1.0:
        raise ValueError(f"max_skip_ratio must be in [0, 1], got {max_skip_ratio}.")
    if prune_ratio is not None and not 0.0 <= prune_ratio <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}.")

    num_segments = entropy.shape[0]
    skip_mask = np.zeros(num_segments, dtype=np.int8)
    if num_segments == 0:
        return skip_mask

    keep_count = min(max(min_keep, 0), num_segments)
    sortable_entropy = np.nan_to_num(entropy, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    protected = set(np.argsort(-sortable_entropy, kind="stable")[:keep_count].tolist())

    max_skips = int(np.floor(num_segments * max_skip_ratio))
    max_skips = min(max_skips, num_segments - keep_count, num_segments - 1)
    max_skips = max(max_skips, 0)
    if max_skips == 0:
        return skip_mask

    if prune_ratio is not None:
        requested = int(np.floor(num_segments * prune_ratio + 0.5))
        candidate_indices = np.argsort(sortable_entropy, kind="stable").tolist()
    elif threshold is not None:
        requested = num_segments
        candidate_indices = np.where(sortable_entropy <= threshold)[0].tolist()
        candidate_indices = sorted(candidate_indices, key=lambda idx: (sortable_entropy[idx], idx))
    else:
        return skip_mask

    requested = min(max(requested, 0), max_skips)
    selected = []
    for idx in candidate_indices:
        if idx in protected:
            continue
        selected.append(idx)
        if len(selected) >= requested:
            break

    skip_mask[selected] = 1
    if np.all(skip_mask == 1):
        skip_mask[np.argmax(sortable_entropy)] = 0
    return skip_mask


def expand_segment_mask(skip_mask: np.ndarray, segments: Sequence[Segment], t_len: int) -> np.ndarray:
    """Expand a segment-level skip mask into a frame-level mask."""
    if t_len < 0:
        raise ValueError(f"t_len must be non-negative, got {t_len}.")
    mask = np.asarray(skip_mask, dtype=np.int8)
    if mask.ndim != 1:
        raise ValueError(f"skip_mask must have shape [N], got {mask.shape}.")
    if len(mask) != len(segments):
        raise ValueError(f"skip_mask length {len(mask)} does not match {len(segments)} segments.")
    if not _segments_cover(segments, t_len):
        raise ValueError("segments must be contiguous and cover [0, T).")

    frame_mask = np.zeros(t_len, dtype=np.int8)
    for value, (start, end) in zip(mask, segments, strict=True):
        frame_mask[start:end] = value
    return frame_mask
