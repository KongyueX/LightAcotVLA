"""Train and evaluate a leakage-resistant terminal Action-CoT refresh gate.

This script consumes exact stale-vs-refresh terminal pairs collected by
``collect_libero_terminal_refresh_pairs.py``.  Model and threshold selection use
only episodes 0-9 and 30-39.  Episodes 40-49 are opened once as a held-out dev
set after the threshold has been frozen from episode-grouped train OOF scores.

The primary model never receives branch/root/step metadata, an injected or
executed action prefix, fresh Action-CoT outputs, or privileged task progress.
Those fields are used only for grouping, labels, controls, and reporting.  The
terminal preference label is lexicographic: terminal success first; if both
arms succeed, fewer decision-to-terminal policy calls and then fewer executed
steps; if both fail, privileged terminal progress.  H6 differences are emitted
only as auxiliary diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import pathlib
from typing import Any, Iterable, Sequence

import numpy as np

import train_verified_acot_gate as base


DEFAULT_TRAIN_EPISODES = (*range(0, 10), *range(30, 40))
DEFAULT_DEV_EPISODES = tuple(range(40, 50))
FORMAL_EPISODES = tuple(range(10, 30))
VARIANTS = ("real", "anchor_current", "shuffled_current", "tail_shuffle")
BASE_VARIANTS = {
    "real": "real",
    "anchor_current": "current_equals_anchor",
    "shuffled_current": "same_root_shuffled_current",
    "tail_shuffle": "tail_shuffle",
}

FORBIDDEN_PRIMARY_INPUTS = (
    "branch_id",
    "branch_name",
    "branch_canonical_name",
    "branch_strength",
    "root_index",
    "*_step",
    "policy_seed",
    "continuation_seed",
    "actual_prefix_env",
    "actual_prefix_executed_env",
    "actual_prefix_executed_valid",
    "current_state (collector field is returned by the fresh endpoint RPC)",
    "fresh_ear",
    "fresh_iar",
    "fresh_final_actions",
    "fresh_env_actions",
    "terminal_privileged_progress_difference",
    "stale_*_progress",
    "fresh_*_progress",
    "fresh_minus_stale_*",
    "terminal outcomes and costs",
)


@dataclasses.dataclass(frozen=True)
class TerminalExample:
    path: str
    task_id: int
    episode_id: int
    anchor_id: int
    root_index: int
    branch_id: int
    branch_name: str
    branch_canonical_name: str
    is_nominal: bool
    anchor_images: np.ndarray
    current_images: np.ndarray
    anchor_state: np.ndarray
    current_state: np.ndarray
    cached_ear: np.ndarray
    cached_iar: np.ndarray
    cached_final: np.ndarray
    cached_env: np.ndarray
    intended_prefix: np.ndarray
    # A synthetic no-error placeholder required by the shared feature builder.
    # No actual/injected/executed prefix field is read from disk.
    executed_prefix: np.ndarray
    stale_terminal_success: bool
    fresh_terminal_success: bool
    terminal_success_difference: int
    terminal_progress_difference: float
    terminal_satisfied_difference: int
    stale_policy_calls: int
    fresh_policy_calls: int
    stale_policy_wall_ms: float
    fresh_policy_wall_ms: float
    common_sunk_anchor_calls: int
    common_sunk_anchor_wall_ms: float
    fresh_refresh_request_calls: int
    fresh_refresh_request_wall_ms: float
    stale_terminal_steps: int
    fresh_terminal_steps: int
    h6_score_difference: float
    h6_success_difference: int

    @property
    def root_key(self) -> tuple[int, int, int, int]:
        return (self.task_id, self.episode_id, self.anchor_id, self.root_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collector-roots",
        "--input",
        nargs="+",
        required=True,
        help="Terminal-pair NPZ files, directories, or glob patterns.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument(
        "--train-episodes", nargs="+", type=int, default=list(DEFAULT_TRAIN_EPISODES)
    )
    parser.add_argument(
        "--dev-episodes", nargs="+", type=int, default=list(DEFAULT_DEV_EPISODES)
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=12)
    parser.add_argument("--pca-fit-tokens", type=int, default=65_536)
    parser.add_argument("--expected-transition-ridge", type=float, default=10.0)
    parser.add_argument("--gate-ridge", type=float, default=10.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-episode", type=int, default=32)
    parser.add_argument("--iar-pool-bins", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resnet-weights", default="default")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--progress-tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--expected-dev-examples", type=int, default=50)
    parser.add_argument("--expected-dev-min-refreshes", type=int, default=5)
    parser.add_argument("--expected-dev-max-refreshes", type=int, default=20)
    parser.add_argument("--max-train-oof-regression-episodes", type=int, default=0)
    parser.add_argument("--max-dev-regression-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    train = set(args.train_episodes)
    dev = set(args.dev_episodes)
    if not train or not dev:
        raise ValueError("Train and dev episode sets must both be non-empty.")
    if train & dev:
        raise ValueError(f"Train/dev episodes overlap: {sorted(train & dev)}")
    forbidden = sorted((train | dev) & set(FORMAL_EPISODES))
    if forbidden:
        raise ValueError(f"Formal episodes 10-29 are forbidden: {forbidden}")
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least two.")
    if not 0 < args.pca_dim <= 256:
        raise ValueError("--pca-dim must be in [1, 256].")
    if args.pca_fit_tokens <= 0 or args.iar_pool_bins <= 0:
        raise ValueError("PCA and IAR pooling limits must be positive.")
    if args.expected_transition_ridge <= 0 or args.gate_ridge <= 0:
        raise ValueError("Ridge penalties must be positive.")
    if args.pairwise_weight < 0 or args.max_pairs_per_episode <= 0:
        raise ValueError("Pairwise settings are invalid.")
    if args.progress_tie_tolerance < 0:
        raise ValueError("--progress-tie-tolerance must be non-negative.")
    if args.expected_dev_examples <= 0:
        raise ValueError("--expected-dev-examples must be positive.")
    if not (
        0 <= args.expected_dev_min_refreshes
        <= args.expected_dev_max_refreshes
        <= args.expected_dev_examples
    ):
        raise ValueError("Expected dev refresh bounds are invalid.")
    if (
        args.max_train_oof_regression_episodes < 0
        or args.max_dev_regression_episodes < 0
    ):
        raise ValueError("Regression episode limits must be non-negative.")


def _load_example(
    path: pathlib.Path,
    *,
    task_id: int,
    allowed_episodes: set[int],
) -> TerminalExample | None:
    with np.load(path, allow_pickle=False) as data:
        valid = base._scalar(data, ("valid",), bool, required=False, default=True)
        current_task = base._scalar(data, ("task_id",), int)
        episode_id = base._scalar(data, ("episode_id",), int)
        if not valid or current_task != task_id or episode_id not in allowed_episodes:
            return None
        intended = base._array(data, ("intended_prefix_env",), dtype=np.float64)
        anchor_state = base._array(data, ("anchor_state",), dtype=np.float64)
        branch_id = base._scalar(data, ("branch_id",), int)
        canonical_name = base._scalar(
            data, ("branch_canonical_name", "branch_name"), str
        )
        stale_success = base._scalar(data, ("stale_terminal_success",), bool)
        fresh_success = base._scalar(data, ("fresh_terminal_success",), bool)
        success_difference = base._scalar(
            data, ("terminal_success_difference",), int
        )
        if success_difference != int(fresh_success) - int(stale_success):
            raise ValueError(f"Inconsistent terminal success labels in {path}.")
        return TerminalExample(
            path=str(path),
            task_id=current_task,
            episode_id=episode_id,
            anchor_id=base._scalar(
                data, ("initial_state_id", "anchor_id"), int, required=False, default=0
            ),
            root_index=base._scalar(data, ("root_index",), int),
            branch_id=branch_id,
            branch_name=base._scalar(data, ("branch_name",), str),
            branch_canonical_name=canonical_name,
            is_nominal=canonical_name == "nominal" or branch_id == 0,
            anchor_images=base._array(data, ("anchor_images",), dtype=np.uint8),
            current_images=base._array(data, ("current_images",), dtype=np.uint8),
            anchor_state=anchor_state,
            # The serialized current_state is returned by the fresh RPC.  A gate
            # deciding whether to issue that RPC cannot use it, so the shared
            # feature builder receives a non-informative anchor copy instead.
            current_state=anchor_state.copy(),
            cached_ear=base._array(data, ("cached_ear",), dtype=np.float64),
            cached_iar=base._array(data, ("cached_iar",), dtype=np.float64),
            cached_final=base._array(
                data, ("cached_final_actions",), dtype=np.float64
            ),
            cached_env=base._array(data, ("cached_env_actions",), dtype=np.float64),
            intended_prefix=intended,
            executed_prefix=intended.copy(),
            stale_terminal_success=stale_success,
            fresh_terminal_success=fresh_success,
            terminal_success_difference=success_difference,
            terminal_progress_difference=base._scalar(
                data, ("terminal_privileged_progress_difference",), float
            ),
            terminal_satisfied_difference=base._scalar(
                data, ("fresh_minus_stale_terminal_satisfied_count",), int
            ),
            stale_policy_calls=base._scalar(
                data, ("stale_decision_to_terminal_policy_calls",), int
            ),
            fresh_policy_calls=base._scalar(
                data, ("fresh_decision_to_terminal_policy_calls",), int
            ),
            stale_policy_wall_ms=base._scalar(
                data, ("stale_decision_to_terminal_policy_wall_ms",), float
            ),
            fresh_policy_wall_ms=base._scalar(
                data, ("fresh_decision_to_terminal_policy_wall_ms",), float
            ),
            common_sunk_anchor_calls=base._scalar(
                data, ("common_sunk_anchor_request_calls",), int
            ),
            common_sunk_anchor_wall_ms=base._scalar(
                data, ("common_sunk_anchor_request_wall_ms",), float
            ),
            fresh_refresh_request_calls=base._scalar(
                data, ("fresh_decision_refresh_request_calls",), int
            ),
            fresh_refresh_request_wall_ms=base._scalar(
                data, ("fresh_decision_refresh_request_wall_ms",), float
            ),
            stale_terminal_steps=base._scalar(
                data, ("stale_endpoint_to_terminal_steps",), int
            ),
            fresh_terminal_steps=base._scalar(
                data, ("fresh_endpoint_to_terminal_steps",), int
            ),
            h6_score_difference=float(base._benefit(data, "h6", dtype=float)),
            h6_success_difference=base._scalar(
                data, ("fresh_minus_stale_h6_success", "advantage_h6_success"), int
            ),
        )


def _load_examples(
    paths: Sequence[pathlib.Path],
    *,
    task_id: int,
    train_episodes: set[int],
    dev_episodes: set[int],
) -> tuple[list[TerminalExample], dict[str, int]]:
    allowed = train_episodes | dev_episodes
    examples: list[TerminalExample] = []
    skipped_invalid = 0
    skipped_outside_split = 0
    seen: dict[tuple[int, int, int, int], str] = {}
    for path in paths:
        example = _load_example(path, task_id=task_id, allowed_episodes=allowed)
        if example is None:
            with np.load(path, allow_pickle=False) as data:
                valid = base._scalar(
                    data, ("valid",), bool, required=False, default=True
                )
                current_task = base._scalar(
                    data, ("task_id",), int, required=False, default=task_id
                )
                episode_id = base._scalar(
                    data, ("episode_id",), int, required=False, default=-1
                )
            if current_task == task_id and episode_id in allowed and not valid:
                skipped_invalid += 1
            else:
                skipped_outside_split += 1
            continue
        key = (
            example.task_id,
            example.episode_id,
            example.root_index,
            example.branch_id,
        )
        if key in seen:
            raise ValueError(f"Duplicate terminal pair {key}: {seen[key]} and {path}")
        seen[key] = str(path)
        examples.append(example)
    if not examples:
        raise ValueError("No compatible valid terminal refresh pairs were loaded.")
    base._validate_shapes(examples)
    train_indices = [i for i, item in enumerate(examples) if item.episode_id in train_episodes]
    dev_indices = [i for i, item in enumerate(examples) if item.episode_id in dev_episodes]
    if not train_indices or not dev_indices:
        raise ValueError("Both train and dev require at least one valid example.")
    if not any(examples[i].is_nominal for i in train_indices):
        raise ValueError("Train split has no valid nominal branch.")
    return examples, {
        "files_considered": len(paths),
        "valid_examples": len(examples),
        "skipped_invalid": skipped_invalid,
        "skipped_outside_split": skipped_outside_split,
    }


def _terminal_label(
    example: TerminalExample,
    *,
    progress_tolerance: float,
) -> tuple[int, str, float]:
    if example.terminal_success_difference > 0:
        return 1, "rescue", 4.0
    if example.terminal_success_difference < 0:
        return -1, "regression", 4.0
    if example.stale_terminal_success and example.fresh_terminal_success:
        if example.fresh_policy_calls < example.stale_policy_calls:
            return 1, "both_success_fewer_calls", 2.0
        if example.fresh_policy_calls > example.stale_policy_calls:
            return -1, "both_success_more_calls", 2.0
        if example.fresh_terminal_steps < example.stale_terminal_steps:
            return 1, "both_success_fewer_steps", 2.0
        if example.fresh_terminal_steps > example.stale_terminal_steps:
            return -1, "both_success_more_steps", 2.0
        return 0, "both_success_cost_tie", 0.5
    progress = example.terminal_progress_difference
    if progress > progress_tolerance:
        return 1, "both_fail_progress_gain", 1.0
    if progress < -progress_tolerance:
        return -1, "both_fail_progress_loss", 1.0
    return 0, "both_fail_progress_tie", 0.5


def _indices_for_episodes(
    examples: Sequence[TerminalExample], episode_ids: Iterable[int]
) -> np.ndarray:
    selected = set(episode_ids)
    return np.asarray(
        [i for i, item in enumerate(examples) if item.episode_id in selected],
        dtype=np.int64,
    )


def _episode_folds(
    examples: Sequence[TerminalExample],
    indices: np.ndarray,
    folds: int,
    seed: int,
) -> list[list[int]]:
    episodes = np.asarray(
        sorted({examples[int(i)].episode_id for i in indices}), dtype=np.int64
    )
    if episodes.size < folds:
        raise ValueError(f"Cannot split {episodes.size} train episodes into {folds} folds.")
    rng = np.random.default_rng(seed)
    shuffled = episodes[rng.permutation(episodes.size)]
    return [sorted(chunk.tolist()) for chunk in np.array_split(shuffled, folds)]


def _episode_balanced_weights(
    examples: Sequence[TerminalExample],
    indices: np.ndarray,
    priorities: np.ndarray,
) -> np.ndarray:
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for local, global_index in enumerate(indices):
        groups[examples[int(global_index)].episode_id].append(local)
    weights = np.zeros(len(indices), dtype=np.float64)
    total_scale = len(indices) / len(groups)
    for members in groups.values():
        raw = np.asarray([priorities[int(indices[local])] for local in members])
        raw_sum = float(np.sum(raw))
        if raw_sum <= 0:
            raw = np.ones(len(members), dtype=np.float64)
            raw_sum = float(len(members))
        weights[members] = total_scale * raw / raw_sum
    return weights


def _fit_gate_model(
    features: np.ndarray,
    labels: np.ndarray,
    priorities: np.ndarray,
    indices: np.ndarray,
    *,
    examples: Sequence[TerminalExample],
    ridge_lambda: float,
    pairwise_weight: float,
    max_pairs_per_episode: int,
    seed: int,
) -> tuple[base.LinearModel, dict[str, float | int]]:
    values = np.asarray(features, dtype=np.float64)
    target = labels[indices].astype(np.float64)
    absolute_weights = _episode_balanced_weights(examples, indices, priorities)
    weight_sum = float(np.sum(absolute_weights))
    mean = np.sum(values * absolute_weights[:, None], axis=0) / weight_sum
    variance = (
        np.sum(np.square(values - mean) * absolute_weights[:, None], axis=0)
        / weight_sum
    )
    scale = np.where(variance >= 1e-16, np.sqrt(variance), 1.0)
    normalized = (values - mean) / scale
    intercept = np.asarray(
        [float(np.sum(target * absolute_weights) / weight_sum)], dtype=np.float64
    )
    design_rows = [normalized * np.sqrt(absolute_weights)[:, None]]
    target_rows = [
        (target - intercept[0])[:, None] * np.sqrt(absolute_weights)[:, None]
    ]

    local_by_global = {int(global_index): local for local, global_index in enumerate(indices)}
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for global_index in indices:
        groups[examples[int(global_index)].episode_id].append(int(global_index))
    pair_count = 0
    rng = np.random.default_rng(seed)
    pair_budget_per_episode = pairwise_weight * len(indices) / len(groups)
    for members in groups.values():
        pairs = [
            (left, right)
            for position, left in enumerate(members)
            for right in members[position + 1 :]
            if labels[left] != labels[right]
        ]
        if len(pairs) > max_pairs_per_episode:
            chosen = rng.choice(len(pairs), size=max_pairs_per_episode, replace=False)
            pairs = [pairs[int(item)] for item in chosen]
        if not pairs or pairwise_weight <= 0:
            continue
        pair_scale = np.sqrt(pair_budget_per_episode / len(pairs))
        rows = []
        outputs = []
        for left, right in pairs:
            rows.append(
                pair_scale
                * (
                    normalized[local_by_global[left]]
                    - normalized[local_by_global[right]]
                )
            )
            outputs.append(pair_scale * float(labels[left] - labels[right]))
        design_rows.append(np.stack(rows))
        target_rows.append(np.asarray(outputs, dtype=np.float64)[:, None])
        pair_count += len(pairs)

    design = np.concatenate(design_rows, axis=0)
    regression_target = np.concatenate(target_rows, axis=0)
    coefficient = base._ridge_solve(design, regression_target, ridge_lambda)
    model = base.LinearModel(mean, scale, coefficient, intercept)
    prediction = model.predict(values).reshape(-1)
    weighted_rmse = np.sqrt(
        np.sum(absolute_weights * np.square(prediction - target)) / weight_sum
    )
    return model, {
        "absolute_rows": int(len(indices)),
        "pairwise_rows": int(pair_count),
        "effective_episode_clusters": int(len(groups)),
        "episode_balanced_weight_sum": weight_sum,
        "weighted_train_rmse": float(weighted_rmse),
    }


def _variant_features(
    variant: str,
    indices: np.ndarray,
    *,
    examples: Sequence[TerminalExample],
    context: base.FoldContext,
    anchor_maps: np.ndarray,
    current_maps: np.ndarray,
    anchor_state: np.ndarray,
    current_state: np.ndarray,
    transition_static: np.ndarray,
    plan_features: np.ndarray,
    current_shuffle: np.ndarray,
    tail_shuffle: np.ndarray,
) -> np.ndarray:
    return base._build_features(
        BASE_VARIANTS[variant],
        indices,
        examples=examples,
        context=context,
        anchor_maps=anchor_maps,
        current_maps=current_maps,
        anchor_state=anchor_state,
        current_state=current_state,
        transition_static=transition_static,
        plan_features=plan_features,
        current_shuffle=current_shuffle,
        tail_shuffle=tail_shuffle,
        include_prefix_telemetry=False,
    )


def _episode_pairwise_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    examples: Sequence[TerminalExample],
) -> dict[str, Any]:
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for local, global_index in enumerate(indices):
        groups[examples[int(global_index)].episode_id].append(local)
    episode_accuracies: list[float] = []
    correct_total = 0.0
    pair_total = 0
    for members in groups.values():
        correct = 0.0
        count = 0
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                true_delta = labels[left] - labels[right]
                if true_delta == 0:
                    continue
                predicted_delta = scores[left] - scores[right]
                count += 1
                if abs(predicted_delta) <= 1e-12:
                    correct += 0.5
                elif np.sign(predicted_delta) == np.sign(true_delta):
                    correct += 1.0
        if count:
            episode_accuracies.append(correct / count)
            correct_total += correct
            pair_total += count
    return {
        "pairs": int(pair_total),
        "micro_accuracy": float(correct_total / pair_total) if pair_total else None,
        "episode_equal_accuracy": (
            float(np.mean(episode_accuracies)) if episode_accuracies else None
        ),
        "episodes_with_pairs": len(episode_accuracies),
    }


def _ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    examples: Sequence[TerminalExample],
) -> dict[str, Any]:
    return {
        "spearman": base._safe_spearman(scores, labels),
        "pairwise": _episode_pairwise_accuracy(scores, labels, indices, examples),
    }


def _policy_summary(
    selected: np.ndarray,
    indices: np.ndarray,
    *,
    examples: Sequence[TerminalExample],
    labels: np.ndarray,
) -> dict[str, Any]:
    chosen = [examples[int(index)] for index in indices]
    mask = np.asarray(selected, dtype=np.bool_)
    if mask.shape != (len(chosen),):
        raise ValueError(f"Selection mask has shape {mask.shape}, expected {(len(chosen),)}")
    local_labels = labels[indices]
    success_delta = np.asarray(
        [item.terminal_success_difference for item in chosen], dtype=np.int64
    )
    stale_success = np.asarray(
        [item.stale_terminal_success for item in chosen], dtype=np.bool_
    )
    fresh_success = np.asarray(
        [item.fresh_terminal_success for item in chosen], dtype=np.bool_
    )
    stale_calls = np.asarray([item.stale_policy_calls for item in chosen], dtype=np.int64)
    fresh_calls = np.asarray([item.fresh_policy_calls for item in chosen], dtype=np.int64)
    common_calls = np.asarray(
        [item.common_sunk_anchor_calls for item in chosen], dtype=np.int64
    )
    refresh_calls = np.asarray(
        [item.fresh_refresh_request_calls for item in chosen], dtype=np.int64
    )
    stale_wall = np.asarray(
        [item.stale_policy_wall_ms for item in chosen], dtype=np.float64
    )
    fresh_wall = np.asarray(
        [item.fresh_policy_wall_ms for item in chosen], dtype=np.float64
    )
    common_wall = np.asarray(
        [item.common_sunk_anchor_wall_ms for item in chosen], dtype=np.float64
    )
    stale_steps = np.asarray(
        [item.stale_terminal_steps for item in chosen], dtype=np.int64
    )
    fresh_steps = np.asarray(
        [item.fresh_terminal_steps for item in chosen], dtype=np.int64
    )
    chosen_success = np.where(mask, fresh_success, stale_success)
    chosen_calls = np.where(mask, fresh_calls, stale_calls)
    chosen_wall = np.where(mask, fresh_wall, stale_wall)
    chosen_steps = np.where(mask, fresh_steps, stale_steps)
    selected_rescue = mask & (success_delta > 0)
    selected_regression = mask & (success_delta < 0)

    episode_rows: list[dict[str, Any]] = []
    episode_groups: dict[int, list[int]] = collections.defaultdict(list)
    for local, item in enumerate(chosen):
        episode_groups[item.episode_id].append(local)
    for episode_id, members in sorted(episode_groups.items()):
        member = np.asarray(members, dtype=np.int64)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "examples": len(members),
                "selected": int(np.sum(mask[member])),
                "selection_rate": float(np.mean(mask[member])),
                "rescues": int(np.sum(selected_rescue[member])),
                "regressions": int(np.sum(selected_regression[member])),
                "terminal_success_delta": int(np.sum(success_delta[member] * mask[member])),
                "lexicographic_net": int(
                    np.sum((local_labels[member] > 0) & mask[member])
                    - np.sum((local_labels[member] < 0) & mask[member])
                ),
                "decision_to_terminal_calls": int(np.sum(chosen_calls[member])),
            }
        )

    branch_groups: dict[str, list[int]] = collections.defaultdict(list)
    for local, item in enumerate(chosen):
        branch_groups[item.branch_canonical_name].append(local)
    branch_coverage = {}
    for name, members in sorted(branch_groups.items()):
        member = np.asarray(members, dtype=np.int64)
        branch_coverage[name] = {
            "examples": len(members),
            "selected": int(np.sum(mask[member])),
            "selection_rate": float(np.mean(mask[member])),
            "rescues": int(np.sum(selected_rescue[member])),
            "regressions": int(np.sum(selected_regression[member])),
        }

    selected_count = int(np.sum(mask))
    episode_selection_rates = [row["selection_rate"] for row in episode_rows]
    episode_success_deltas = [
        row["terminal_success_delta"] / row["examples"] for row in episode_rows
    ]
    h6 = np.asarray([item.h6_score_difference for item in chosen], dtype=np.float64)
    return {
        "examples": len(chosen),
        "selected": selected_count,
        "selection_rate": float(np.mean(mask)),
        "rescues": int(np.sum(selected_rescue)),
        "regressions": int(np.sum(selected_regression)),
        "net_terminal_successes": int(np.sum(success_delta * mask)),
        "lexicographic_wins": int(np.sum(mask & (local_labels > 0))),
        "lexicographic_losses": int(np.sum(mask & (local_labels < 0))),
        "lexicographic_ties": int(np.sum(mask & (local_labels == 0))),
        "policy_terminal_successes": int(np.sum(chosen_success)),
        "always_stale_terminal_successes": int(np.sum(stale_success)),
        "always_fresh_terminal_successes": int(np.sum(fresh_success)),
        "episode_coverage": {
            "episodes": len(episode_rows),
            "episodes_selected": int(sum(row["selected"] > 0 for row in episode_rows)),
            "episodes_with_rescue": int(sum(row["rescues"] > 0 for row in episode_rows)),
            "episodes_with_regression": int(
                sum(row["regressions"] > 0 for row in episode_rows)
            ),
            "episode_equal_selection_rate": float(np.mean(episode_selection_rates)),
            "episode_equal_terminal_success_delta_per_branch": float(
                np.mean(episode_success_deltas)
            ),
        },
        "branch_coverage": branch_coverage,
        "calls": {
            "selected_refresh_request_calls": int(np.sum(refresh_calls * mask)),
            "common_sunk_anchor_calls": int(np.sum(common_calls)),
            "decision_to_terminal_policy_calls": int(np.sum(chosen_calls)),
            "total_policy_calls_including_common_sunk": int(
                np.sum(common_calls + chosen_calls)
            ),
            "always_stale_decision_to_terminal_calls": int(np.sum(stale_calls)),
            "always_fresh_decision_to_terminal_calls": int(np.sum(fresh_calls)),
            "delta_vs_always_stale": int(np.sum(chosen_calls - stale_calls)),
            "saved_vs_always_fresh": int(np.sum(fresh_calls - chosen_calls)),
        },
        "wall_ms": {
            "decision_to_terminal": float(np.sum(chosen_wall)),
            "total_including_common_sunk": float(np.sum(common_wall + chosen_wall)),
            "delta_vs_always_stale": float(np.sum(chosen_wall - stale_wall)),
            "saved_vs_always_fresh": float(np.sum(fresh_wall - chosen_wall)),
        },
        "endpoint_to_terminal_steps": {
            "policy": int(np.sum(chosen_steps)),
            "always_stale": int(np.sum(stale_steps)),
            "always_fresh": int(np.sum(fresh_steps)),
        },
        "h6_auxiliary_selected": {
            "mean_score_difference": (
                float(np.mean(h6[mask])) if selected_count else None
            ),
            "positive": int(np.sum(mask & (h6 > 0))),
            "negative": int(np.sum(mask & (h6 < 0))),
            "ties": int(np.sum(mask & (h6 == 0))),
        },
        "episode_rows": episode_rows,
    }


def _threshold_candidates(
    scores: np.ndarray,
    train_indices: np.ndarray,
    *,
    examples: Sequence[TerminalExample],
    labels: np.ndarray,
    expected_dev_examples: int,
    min_refreshes: int,
    max_refreshes: int,
    max_regression_episodes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    rows: list[dict[str, Any]] = []
    min_rate = min_refreshes / expected_dev_examples
    max_rate = max_refreshes / expected_dev_examples
    for threshold in unique:
        selected = scores > threshold
        selection_rate = float(np.mean(selected))
        if selection_rate < min_rate - 1e-12 or selection_rate > max_rate + 1e-12:
            continue
        summary = _policy_summary(
            selected, train_indices, examples=examples, labels=labels
        )
        regression_episodes = summary["episode_coverage"]["episodes_with_regression"]
        rescue_episodes = summary["episode_coverage"]["episodes_with_rescue"]
        row = {
            "threshold": float(threshold),
            "selected": summary["selected"],
            "selection_rate": summary["selection_rate"],
            "expected_dev_selected": summary["selection_rate"] * expected_dev_examples,
            "rescues": summary["rescues"],
            "regressions": summary["regressions"],
            "rescue_episodes": rescue_episodes,
            "regression_episodes": regression_episodes,
            "net_terminal_successes": summary["net_terminal_successes"],
            "lexicographic_net": (
                summary["lexicographic_wins"] - summary["lexicographic_losses"]
            ),
            "calls_delta_vs_always_stale": summary["calls"]["delta_vs_always_stale"],
            "risk_constraint_met": regression_episodes <= max_regression_episodes,
        }
        rows.append(row)
    if not rows:
        raise ValueError(
            "No distinct train-OOF threshold satisfies the requested expected dev "
            "refresh-rate interval."
        )

    def rank(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            bool(row["risk_constraint_met"]),
            -int(row["regression_episodes"]),
            int(row["rescue_episodes"]),
            int(row["rescues"]),
            int(row["net_terminal_successes"]),
            int(row["lexicographic_net"]),
            -int(row["calls_delta_vs_always_stale"]),
            -int(row["selected"]),
        )

    chosen = max(rows, key=rank)
    return chosen, rows


def _split_results(
    predictions: dict[str, np.ndarray],
    indices: np.ndarray,
    threshold: float,
    *,
    examples: Sequence[TerminalExample],
    labels: np.ndarray,
) -> dict[str, Any]:
    variants = {}
    for name in VARIANTS:
        scores = predictions[name][indices]
        variants[name] = {
            "ranking": _ranking_metrics(scores, labels[indices], indices, examples),
            "selection": _policy_summary(
                scores > threshold,
                indices,
                examples=examples,
                labels=labels,
            ),
        }
    return {
        "variants": variants,
        "always_fresh_upper_bound": _policy_summary(
            np.ones(len(indices), dtype=np.bool_),
            indices,
            examples=examples,
            labels=labels,
        ),
        "always_stale_reference": _policy_summary(
            np.zeros(len(indices), dtype=np.bool_),
            indices,
            examples=examples,
            labels=labels,
        ),
    }


def _write_csv(path: pathlib.Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    artifacts = (
        output_dir / "summary.json",
        output_dir / "predictions.csv",
        output_dir / "threshold_candidates.csv",
        output_dir / "episode_metrics.csv",
    )
    if any(path.exists() for path in artifacts):
        raise FileExistsError(
            f"Output already exists under {output_dir}; dev is intentionally single-look."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = base._expand_inputs(args.collector_roots)
    train_episode_set = set(args.train_episodes)
    dev_episode_set = set(args.dev_episodes)
    examples, load_summary = _load_examples(
        paths,
        task_id=args.task_id,
        train_episodes=train_episode_set,
        dev_episodes=dev_episode_set,
    )
    train_indices = _indices_for_episodes(examples, train_episode_set)
    dev_indices = _indices_for_episodes(examples, dev_episode_set)
    labels_and_reasons = [
        _terminal_label(item, progress_tolerance=args.progress_tie_tolerance)
        for item in examples
    ]
    labels = np.asarray([item[0] for item in labels_and_reasons], dtype=np.int8)
    reasons = [item[1] for item in labels_and_reasons]
    priorities = np.asarray([item[2] for item in labels_and_reasons], dtype=np.float64)

    anchor_state = base._stack(examples, "anchor_state")
    current_state = base._stack(examples, "current_state")
    anchor_maps, current_maps, spatial_metadata = base._extract_spatial_maps(
        base._stack(examples, "anchor_images"),
        base._stack(examples, "current_images"),
        weights_spec=args.resnet_weights,
        device_name=args.device,
        batch_size=args.image_batch_size,
        torch_threads=args.torch_threads,
    )
    plan_features = np.stack(
        [
            base._cached_plan_features(item, iar_pool_bins=args.iar_pool_bins)
            for item in examples
        ]
    )
    transition_static = base._transition_static_features(
        examples, iar_pool_bins=args.iar_pool_bins
    )
    predictions = {
        name: np.full(len(examples), np.nan, dtype=np.float64) for name in VARIANTS
    }
    identity = np.arange(len(examples), dtype=np.int64)
    folds = _episode_folds(examples, train_indices, args.outer_folds, args.seed)
    fold_records = []
    for fold_id, held_out_episodes in enumerate(folds):
        held_out_indices = _indices_for_episodes(examples, held_out_episodes)
        fold_train_indices = np.setdiff1d(
            train_indices, held_out_indices, assume_unique=False
        )
        context = base._fit_fold_context(
            fold_train_indices,
            fold_id,
            examples=examples,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=anchor_state,
            current_state=current_state,
            transition_static=transition_static,
            pca_dim=args.pca_dim,
            pca_fit_tokens=args.pca_fit_tokens,
            expected_transition_ridge=args.expected_transition_ridge,
            seed=args.seed,
        )
        train_features = _variant_features(
            "real",
            fold_train_indices,
            examples=examples,
            context=context,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=anchor_state,
            current_state=current_state,
            transition_static=transition_static,
            plan_features=plan_features,
            current_shuffle=identity,
            tail_shuffle=identity,
        )
        model, training_record = _fit_gate_model(
            train_features,
            labels,
            priorities,
            fold_train_indices,
            examples=examples,
            ridge_lambda=args.gate_ridge,
            pairwise_weight=args.pairwise_weight,
            max_pairs_per_episode=args.max_pairs_per_episode,
            seed=args.seed + fold_id,
        )
        current_shuffle = base._same_root_current_shuffle(examples, held_out_indices)
        tail_shuffle = base._same_phase_tail_shuffle(examples, held_out_indices)
        for name in VARIANTS:
            features = _variant_features(
                name,
                held_out_indices,
                examples=examples,
                context=context,
                anchor_maps=anchor_maps,
                current_maps=current_maps,
                anchor_state=anchor_state,
                current_state=current_state,
                transition_static=transition_static,
                plan_features=plan_features,
                current_shuffle=current_shuffle,
                tail_shuffle=tail_shuffle,
            )
            predictions[name][held_out_indices] = model.predict(features).reshape(-1)
        fold_records.append(
            {
                "fold": fold_id,
                "held_out_episodes": held_out_episodes,
                "train_examples": int(len(fold_train_indices)),
                "held_out_examples": int(len(held_out_indices)),
                "training": training_record,
            }
        )

    for name in VARIANTS:
        if not np.all(np.isfinite(predictions[name][train_indices])):
            raise RuntimeError(f"Variant {name} has missing train OOF predictions.")

    chosen_threshold, candidate_rows = _threshold_candidates(
        predictions["real"][train_indices],
        train_indices,
        examples=examples,
        labels=labels,
        expected_dev_examples=args.expected_dev_examples,
        min_refreshes=args.expected_dev_min_refreshes,
        max_refreshes=args.expected_dev_max_refreshes,
        max_regression_episodes=args.max_train_oof_regression_episodes,
    )
    threshold = float(chosen_threshold["threshold"])

    # All choices above are now frozen.  Dev examples are scored exactly once.
    final_context = base._fit_fold_context(
        train_indices,
        10_000,
        examples=examples,
        anchor_maps=anchor_maps,
        current_maps=current_maps,
        anchor_state=anchor_state,
        current_state=current_state,
        transition_static=transition_static,
        pca_dim=args.pca_dim,
        pca_fit_tokens=args.pca_fit_tokens,
        expected_transition_ridge=args.expected_transition_ridge,
        seed=args.seed,
    )
    final_train_features = _variant_features(
        "real",
        train_indices,
        examples=examples,
        context=final_context,
        anchor_maps=anchor_maps,
        current_maps=current_maps,
        anchor_state=anchor_state,
        current_state=current_state,
        transition_static=transition_static,
        plan_features=plan_features,
        current_shuffle=identity,
        tail_shuffle=identity,
    )
    final_model, final_training_record = _fit_gate_model(
        final_train_features,
        labels,
        priorities,
        train_indices,
        examples=examples,
        ridge_lambda=args.gate_ridge,
        pairwise_weight=args.pairwise_weight,
        max_pairs_per_episode=args.max_pairs_per_episode,
        seed=args.seed + 10_000,
    )
    dev_current_shuffle = base._same_root_current_shuffle(examples, dev_indices)
    dev_tail_shuffle = base._same_phase_tail_shuffle(examples, dev_indices)
    for name in VARIANTS:
        features = _variant_features(
            name,
            dev_indices,
            examples=examples,
            context=final_context,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=anchor_state,
            current_state=current_state,
            transition_static=transition_static,
            plan_features=plan_features,
            current_shuffle=dev_current_shuffle,
            tail_shuffle=dev_tail_shuffle,
        )
        predictions[name][dev_indices] = final_model.predict(features).reshape(-1)

    train_results = _split_results(
        predictions,
        train_indices,
        threshold,
        examples=examples,
        labels=labels,
    )
    dev_results = _split_results(
        predictions,
        dev_indices,
        threshold,
        examples=examples,
        labels=labels,
    )
    dev_real = dev_results["variants"]["real"]["selection"]
    dev_expected_at_50 = dev_real["selection_rate"] * args.expected_dev_examples
    go_checks = {
        "train_oof_regression_risk_constraint_met": bool(
            chosen_threshold["risk_constraint_met"]
        ),
        "dev_expected_refreshes_in_5_to_20_of_50": (
            args.expected_dev_min_refreshes
            <= dev_expected_at_50
            <= args.expected_dev_max_refreshes
        ),
        "dev_regression_episode_limit": (
            dev_real["episode_coverage"]["episodes_with_regression"]
            <= args.max_dev_regression_episodes
        ),
        "dev_has_rescue": dev_real["rescues"] > 0,
        "dev_positive_net_terminal_successes": dev_real["net_terminal_successes"] > 0,
        "dev_positive_lexicographic_net": (
            dev_real["lexicographic_wins"] > dev_real["lexicographic_losses"]
        ),
        "dev_saves_calls_vs_always_fresh": (
            dev_real["calls"]["saved_vs_always_fresh"] > 0
        ),
    }

    available_train_episodes = sorted(
        {examples[int(i)].episode_id for i in train_indices}
    )
    available_dev_episodes = sorted({examples[int(i)].episode_id for i in dev_indices})
    label_counts = collections.Counter(reasons[int(i)] for i in train_indices)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "probe_only": True,
        "deployable_checkpoint_saved": False,
        "go": bool(all(go_checks.values())),
        "go_checks": go_checks,
        "protocol": {
            "task_id": args.task_id,
            "train_episodes_requested": list(args.train_episodes),
            "dev_episodes_requested": list(args.dev_episodes),
            "formal_episodes_forbidden": list(FORMAL_EPISODES),
            "train_oof_split": "episode-grouped",
            "outer_folds": args.outer_folds,
            "folds": folds,
            "dev_single_look": True,
            "dev_used_for_model_or_threshold_selection": False,
            "threshold_source": "real train OOF predictions only",
            "selection_rule": "score > frozen_threshold",
            "expected_dev_refresh_interval": {
                "examples": args.expected_dev_examples,
                "min_refreshes": args.expected_dev_min_refreshes,
                "max_refreshes": args.expected_dev_max_refreshes,
                "min_rate": args.expected_dev_min_refreshes / args.expected_dev_examples,
                "max_rate": args.expected_dev_max_refreshes / args.expected_dev_examples,
            },
            "unit_of_independence": "episode",
            "episode_balance": (
                "each episode has equal total absolute loss weight; pairwise rows are "
                "formed within episode and each episode has equal pairwise mass"
            ),
            "lexicographic_label": [
                "terminal success difference",
                "if both succeed: fewer decision-to-terminal policy calls",
                "if calls tie: fewer endpoint-to-terminal executed steps",
                "if both fail: privileged terminal progress difference",
            ],
            "lexicographic_training_priority_weights": {
                "success_difference": 4.0,
                "both_success_efficiency": 2.0,
                "both_failure_progress": 1.0,
                "tie": 0.5,
            },
            "h6_difference_role": "auxiliary diagnostics only",
            "primary_inputs": [
                "anchor/current visual innovation",
                "cached EAR",
                "cached IAR",
                "cached final/environment action tail",
                "policy-issued intended prefix context",
            ],
            "forbidden_primary_inputs": list(FORBIDDEN_PRIMARY_INPUTS),
            "actual_prefix_fields_loaded": False,
            "serialized_current_state_loaded": False,
            "current_state_reason": (
                "collector current_state is returned by the fresh endpoint RPC and is "
                "therefore unavailable before the refresh decision"
            ),
            "fresh_action_cot_outputs_loaded": False,
            "privileged_progress_used_as_primary_feature": False,
            "controls_share_real_model_and_frozen_threshold": True,
            "call_accounting_unit": (
                "counterfactual-pair aggregate: each branch is treated as one possible "
                "deployment encounter; common anchor calls are not collector physical RPC totals"
            ),
        },
        "data": {
            **load_summary,
            "train_examples": int(len(train_indices)),
            "dev_examples": int(len(dev_indices)),
            "train_episodes_available": available_train_episodes,
            "dev_episodes_available": available_dev_episodes,
            "missing_train_episodes": sorted(train_episode_set - set(available_train_episodes)),
            "missing_dev_episodes": sorted(dev_episode_set - set(available_dev_episodes)),
            "train_label_reason_counts": dict(sorted(label_counts.items())),
        },
        "spatial_encoder": spatial_metadata,
        "fold_records": fold_records,
        "final_training": final_training_record,
        "frozen_threshold": chosen_threshold,
        "train_oof": train_results,
        "dev": {
            **dev_results,
            "real_expected_selected_at_50": dev_expected_at_50,
        },
        "artifacts": {
            "summary_json": "summary.json",
            "predictions_csv": "predictions.csv",
            "threshold_candidates_csv": "threshold_candidates.csv",
            "episode_metrics_csv": "episode_metrics.csv",
        },
    }

    prediction_rows = []
    for split, indices in (("train_oof", train_indices), ("dev", dev_indices)):
        for global_index in indices:
            item = examples[int(global_index)]
            row = {
                "split": split,
                "task_id": item.task_id,
                "episode_id": item.episode_id,
                "root_index_report_only": item.root_index,
                "branch_id_report_only": item.branch_id,
                "branch_canonical_name_report_only": item.branch_canonical_name,
                "lexicographic_label": int(labels[global_index]),
                "label_reason": reasons[int(global_index)],
                "terminal_success_difference": item.terminal_success_difference,
                "terminal_privileged_progress_difference_label_only": (
                    item.terminal_progress_difference
                ),
                "stale_policy_calls": item.stale_policy_calls,
                "fresh_policy_calls": item.fresh_policy_calls,
                "stale_terminal_steps": item.stale_terminal_steps,
                "fresh_terminal_steps": item.fresh_terminal_steps,
                "h6_score_difference_diagnostic_only": item.h6_score_difference,
            }
            for name in VARIANTS:
                score = float(predictions[name][global_index])
                row[f"score_{name}"] = score
                row[f"selected_{name}"] = int(score > threshold)
            prediction_rows.append(row)

    flat_candidate_rows = [
        {**row, "chosen": int(row is chosen_threshold)} for row in candidate_rows
    ]
    episode_metric_rows = []
    for split_name, results in (("train_oof", train_results), ("dev", dev_results)):
        policies = {
            **{
                name: results["variants"][name]["selection"] for name in VARIANTS
            },
            "always_fresh": results["always_fresh_upper_bound"],
            "always_stale": results["always_stale_reference"],
        }
        for variant, policy in policies.items():
            for row in policy["episode_rows"]:
                episode_metric_rows.append(
                    {"split": split_name, "variant": variant, **row}
                )

    _write_csv(output_dir / "predictions.csv", prediction_rows)
    _write_csv(output_dir / "threshold_candidates.csv", flat_candidate_rows)
    _write_csv(output_dir / "episode_metrics.csv", episode_metric_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
