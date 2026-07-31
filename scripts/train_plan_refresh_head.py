"""Train a conservative Task8 plan-refresh head from per-root counterfactuals.

Every input NPZ is one collector root.  The supervised target is the
same-snapshot, same-continuation-seed normalized progress advantage of
``fresh H6 + teacher H9`` over ``stale H6 + teacher H9``.  Predicate-count
gain is never used as an input; it is a privileged validation guard that
rejects thresholds causing logical-goal regressions.

Three selectors are reported:

* a train-calibrated fixed phase (non-deployable timing baseline);
* ridge regression over the collector's frozen temporal feature [256];
* ridge regression over temporal plus state/plan summaries (diagnostic only).

The validation threshold allows at most one refresh per episode.  Only the
pure temporal model is serialized to ``head.npz``; the artifact deliberately
contains no root index or privileged outcome field.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import pathlib
from typing import Any, Iterable, Sequence

import numpy as np


FEATURE_TEMPORAL = "temporal"
FEATURE_TEMPORAL_STATE_PLAN = "temporal_state_plan_diagnostic"


@dataclasses.dataclass(frozen=True)
class RootExample:
    path: str
    task_id: int
    episode_id: int
    root_index: int
    temporal_feature: np.ndarray
    anchor_state: np.ndarray
    current_state: np.ndarray
    coarse_actions: np.ndarray
    final_actions: np.ndarray
    actions_env: np.ndarray
    benefit: float
    predicate_gain: int
    h6_benefit: float | None
    h6_predicate_gain: int | None


@dataclasses.dataclass(frozen=True)
class RidgeHead:
    feature_variant: str
    ridge_lambda: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        return ((values - self.mean) / self.scale) @ self.coef + self.intercept


@dataclasses.dataclass(frozen=True)
class SelectionOutcome:
    episode_ids: np.ndarray
    selected_root_indices: np.ndarray
    benefits: np.ndarray
    predicate_gains: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collector-roots",
        "--input",
        "--input-dir",
        dest="collector_roots",
        nargs="+",
        required=True,
        help="Per-root NPZ files, directories, or shell-style glob patterns.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument(
        "--train-episodes",
        nargs="+",
        type=int,
        default=[*range(0, 10), *range(30, 40)],
    )
    parser.add_argument(
        "--validation-episodes",
        nargs="+",
        type=int,
        default=list(range(40, 50)),
    )
    parser.add_argument(
        "--ridge-lambdas",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--fixed-root-index", type=int, default=10)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    train = list(dict.fromkeys(args.train_episodes))
    validation = list(dict.fromkeys(args.validation_episodes))
    if not train or not validation:
        raise ValueError("Train and validation episode sets must both be non-empty.")
    overlap = sorted(set(train) & set(validation))
    if overlap:
        raise ValueError(f"Train/validation episodes overlap: {overlap}.")
    if any(episode < 0 for episode in (*train, *validation)):
        raise ValueError("Episode IDs must be non-negative.")
    if args.task_id < 0 or args.fixed_root_index < 0:
        raise ValueError("task-id and fixed-root-index must be non-negative.")
    if not args.ridge_lambdas or any(value <= 0 for value in args.ridge_lambdas):
        raise ValueError("ridge-lambdas must contain positive values.")
    if args.tie_tolerance < 0:
        raise ValueError("tie-tolerance must be non-negative.")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")


def _expand_inputs(values: Sequence[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for value in values:
        matches = [pathlib.Path(item) for item in glob.glob(value)]
        if not matches:
            matches = [pathlib.Path(value)]
        for match in matches:
            if match.is_dir():
                root_named = sorted(match.rglob("root*.npz"))
                paths.extend(root_named if root_named else sorted(match.rglob("*.npz")))
            elif match.is_file():
                paths.append(match)
            else:
                raise FileNotFoundError(f"Collector input does not exist: {match}")
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No collector NPZ files were found.")
    return unique


def _first_key(data: Any, names: Sequence[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in data.files:
            return name
    if required:
        raise KeyError(f"Missing all compatible fields {tuple(names)}; available={sorted(data.files)}")
    return None


def _scalar(data: Any, names: Sequence[str], *, dtype: type = float) -> Any:
    name = _first_key(data, names)
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"Field {name!r} must be scalar, got {value.shape}.")
    return dtype(value.reshape(()).item())


def _optional_scalar(
    data: Any,
    names: Sequence[str],
    *,
    dtype: type = float,
) -> Any | None:
    name = _first_key(data, names, required=False)
    if name is None:
        return None
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"Field {name!r} must be scalar, got {value.shape}.")
    return dtype(value.reshape(()).item())


def _array(
    data: Any,
    names: Sequence[str],
    shape: tuple[int, ...],
) -> np.ndarray:
    name = _first_key(data, names)
    value = np.asarray(data[name], dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"Field {name!r} must have shape {shape}, got {value.shape}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Field {name!r} contains non-finite values.")
    return value


def _derived_difference(
    data: Any,
    *,
    direct_names: Sequence[str],
    fresh_names: Sequence[str],
    stale_names: Sequence[str],
    dtype: type,
) -> Any:
    direct = _optional_scalar(data, direct_names, dtype=dtype)
    fresh = _optional_scalar(data, fresh_names, dtype=dtype)
    stale = _optional_scalar(data, stale_names, dtype=dtype)
    if direct is None:
        if fresh is None or stale is None:
            raise KeyError(
                f"Need one of {tuple(direct_names)} or both fresh={tuple(fresh_names)} "
                f"and stale={tuple(stale_names)}."
            )
        return dtype(fresh - stale)
    if fresh is not None and stale is not None:
        derived = dtype(fresh - stale)
        tolerance = 1e-6 if dtype is float else 0
        if abs(direct - derived) > tolerance:
            raise ValueError(
                f"Stored advantage {direct} disagrees with fresh-minus-stale {derived}."
            )
    return direct


def _load_root(path: pathlib.Path, *, task_id: int) -> RootExample | None:
    with np.load(path, allow_pickle=False) as data:
        valid = _optional_scalar(data, ("valid",), dtype=bool)
        if valid is False:
            return None
        current_task = _scalar(data, ("task_id",), dtype=int)
        if current_task != task_id:
            return None
        benefit = _derived_difference(
            data,
            direct_names=(
                "fresh_minus_stale_handoff_normalized_score",
                "advantage_handoff_normalized_score",
            ),
            fresh_names=("fresh_handoff_normalized_score",),
            stale_names=("stale_handoff_normalized_score",),
            dtype=float,
        )
        predicate_gain = _derived_difference(
            data,
            direct_names=(
                "fresh_minus_stale_handoff_satisfied_count",
                "advantage_handoff_satisfied_count",
            ),
            fresh_names=("fresh_handoff_satisfied_count",),
            stale_names=("stale_handoff_satisfied_count",),
            dtype=int,
        )
        h6_benefit = None
        try:
            h6_benefit = _derived_difference(
                data,
                direct_names=(
                    "fresh_minus_stale_h6_normalized_score",
                    "advantage_h6_normalized_score",
                ),
                fresh_names=("fresh_h6_normalized_score",),
                stale_names=("stale_h6_normalized_score",),
                dtype=float,
            )
        except KeyError:
            pass
        h6_predicate_gain = None
        try:
            h6_predicate_gain = _derived_difference(
                data,
                direct_names=(
                    "fresh_minus_stale_h6_satisfied_count",
                    "advantage_h6_satisfied_count",
                ),
                fresh_names=("fresh_h6_satisfied_count",),
                stale_names=("stale_h6_satisfied_count",),
                dtype=int,
            )
        except KeyError:
            pass
        if not np.isfinite(benefit):
            raise ValueError(f"Non-finite handoff advantage in {path}.")
        return RootExample(
            path=str(path),
            task_id=current_task,
            episode_id=_scalar(data, ("episode_id", "episode_idx", "trial_id"), dtype=int),
            root_index=_scalar(data, ("root_index",), dtype=int),
            temporal_feature=_array(
                data,
                ("temporal_feature", "temporal_feature_256", "temporal256"),
                (256,),
            ),
            anchor_state=_array(data, ("anchor_state",), (32,)),
            current_state=_array(data, ("current_state",), (32,)),
            coarse_actions=_array(
                data,
                ("cached_coarse_actions", "coarse_actions_normalized"),
                (15, 32),
            ),
            final_actions=_array(
                data,
                ("cached_final_actions", "final_actions_normalized"),
                (10, 32),
            ),
            actions_env=_array(
                data,
                ("cached_env_actions", "actions_env"),
                (10, 7),
            ),
            benefit=float(benefit),
            predicate_gain=int(predicate_gain),
            h6_benefit=None if h6_benefit is None else float(h6_benefit),
            h6_predicate_gain=(
                None if h6_predicate_gain is None else int(h6_predicate_gain)
            ),
        )


def _load_examples(paths: Sequence[pathlib.Path], *, task_id: int) -> tuple[list[RootExample], int]:
    examples: list[RootExample] = []
    skipped = 0
    seen: dict[tuple[int, int], str] = {}
    for path in paths:
        example = _load_root(path, task_id=task_id)
        if example is None:
            skipped += 1
            continue
        key = (example.episode_id, example.root_index)
        if key in seen:
            raise ValueError(
                f"Duplicate Task{task_id} episode/root {key}: {seen[key]} and {path}."
            )
        seen[key] = str(path)
        examples.append(example)
    if not examples:
        raise ValueError(f"No valid Task{task_id} root examples were loaded.")
    return examples, skipped


def _sequence_summary(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Plan sequence must be non-empty rank two, got {values.shape}.")
    return np.concatenate(
        (
            np.mean(values, axis=0),
            np.std(values, axis=0),
            values[0],
            values[-1],
        )
    )


def _features(examples: Sequence[RootExample], variant: str) -> np.ndarray:
    rows = []
    for example in examples:
        if variant == FEATURE_TEMPORAL:
            row = example.temporal_feature
        elif variant == FEATURE_TEMPORAL_STATE_PLAN:
            state_summary = np.concatenate(
                (
                    example.anchor_state,
                    example.current_state,
                    example.current_state - example.anchor_state,
                    np.abs(example.current_state - example.anchor_state),
                )
            )
            plan_summary = np.concatenate(
                (
                    _sequence_summary(example.coarse_actions),
                    _sequence_summary(example.final_actions),
                    _sequence_summary(example.actions_env),
                )
            )
            row = np.concatenate((example.temporal_feature, state_summary, plan_summary))
        else:
            raise ValueError(f"Unknown feature variant {variant!r}.")
        rows.append(np.asarray(row, dtype=np.float64))
    result = np.stack(rows)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite values in {variant} features.")
    return result


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    ridge_lambda: float,
    feature_variant: str,
) -> RidgeHead:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean = np.mean(features, axis=0)
    raw_scale = np.std(features, axis=0)
    scale = np.where(raw_scale >= 1e-8, raw_scale, 1.0)
    normalized = (features - mean) / scale
    intercept = float(np.mean(target))
    centered_target = target - intercept
    if normalized.shape[1] <= normalized.shape[0]:
        system = normalized.T @ normalized + ridge_lambda * np.eye(normalized.shape[1])
        coef = np.linalg.solve(system, normalized.T @ centered_target)
    else:
        system = normalized @ normalized.T + ridge_lambda * np.eye(normalized.shape[0])
        coef = normalized.T @ np.linalg.solve(system, centered_target)
    return RidgeHead(
        feature_variant=feature_variant,
        ridge_lambda=float(ridge_lambda),
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
    )


def _empty_outcome(episode_ids: Sequence[int]) -> SelectionOutcome:
    count = len(episode_ids)
    return SelectionOutcome(
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        selected_root_indices=np.full((count,), -1, dtype=np.int64),
        benefits=np.zeros((count,), dtype=np.float64),
        predicate_gains=np.zeros((count,), dtype=np.int64),
    )


def _fixed_root_outcome(
    examples: Sequence[RootExample],
    episode_ids: Sequence[int],
    *,
    root_index: int | None,
) -> SelectionOutcome:
    outcome = _empty_outcome(episode_ids)
    if root_index is None:
        return outcome
    lookup = {(item.episode_id, item.root_index): item for item in examples}
    roots = outcome.selected_root_indices.copy()
    benefits = outcome.benefits.copy()
    predicates = outcome.predicate_gains.copy()
    for index, episode_id in enumerate(episode_ids):
        example = lookup.get((episode_id, root_index))
        if example is None:
            continue
        roots[index] = root_index
        benefits[index] = example.benefit
        predicates[index] = example.predicate_gain
    return SelectionOutcome(outcome.episode_ids, roots, benefits, predicates)


def _scored_outcome(
    examples: Sequence[RootExample],
    scores: np.ndarray,
    episode_ids: Sequence[int],
    *,
    threshold: float,
) -> SelectionOutcome:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(examples),):
        raise ValueError(f"Expected {len(examples)} scores, got {scores.shape}.")
    by_episode: dict[int, list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_episode[example.episode_id].append(index)
    roots = np.full((len(episode_ids),), -1, dtype=np.int64)
    benefits = np.zeros((len(episode_ids),), dtype=np.float64)
    predicates = np.zeros((len(episode_ids),), dtype=np.int64)
    for output_index, episode_id in enumerate(episode_ids):
        candidates = by_episode.get(episode_id, [])
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda index: examples[index].root_index)
        selected = next((index for index in ordered if scores[index] >= threshold), None)
        if selected is None:
            continue
        example = examples[selected]
        roots[output_index] = example.root_index
        benefits[output_index] = example.benefit
        predicates[output_index] = example.predicate_gain
    return SelectionOutcome(np.asarray(episode_ids), roots, benefits, predicates)


def _oracle_outcome(
    examples: Sequence[RootExample],
    episode_ids: Sequence[int],
) -> SelectionOutcome:
    by_episode: dict[int, list[RootExample]] = collections.defaultdict(list)
    for example in examples:
        by_episode[example.episode_id].append(example)
    roots = np.full((len(episode_ids),), -1, dtype=np.int64)
    benefits = np.zeros((len(episode_ids),), dtype=np.float64)
    predicates = np.zeros((len(episode_ids),), dtype=np.int64)
    for index, episode_id in enumerate(episode_ids):
        safe = [
            item
            for item in by_episode.get(episode_id, ())
            if item.predicate_gain >= 0 and item.benefit > 0.0
        ]
        if not safe:
            continue
        selected = max(safe, key=lambda item: (item.benefit, -item.root_index))
        roots[index] = selected.root_index
        benefits[index] = selected.benefit
        predicates[index] = selected.predicate_gain
    return SelectionOutcome(np.asarray(episode_ids), roots, benefits, predicates)


def _basic_metrics(outcome: SelectionOutcome, *, tie_tolerance: float) -> dict[str, Any]:
    benefits = outcome.benefits
    wins = int(np.sum(benefits > tie_tolerance))
    losses = int(np.sum(benefits < -tie_tolerance))
    ties = int(benefits.size - wins - losses)
    selected = outcome.selected_root_indices >= 0
    distribution = collections.Counter(outcome.selected_root_indices[selected].tolist())
    return {
        "episodes": int(benefits.size),
        "refreshes": int(np.sum(selected)),
        "refresh_rate": float(np.mean(selected)),
        "mean_benefit": float(np.mean(benefits)),
        "median_benefit": float(np.median(benefits)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": wins - losses,
        "predicate_regressions": int(np.sum(outcome.predicate_gains < 0)),
        "predicate_improvements": int(np.sum(outcome.predicate_gains > 0)),
        "mean_predicate_gain": float(np.mean(outcome.predicate_gains)),
        "selected_root_distribution": {
            str(root): int(count) for root, count in sorted(distribution.items())
        },
    }


def _bootstrap_mean_ci(values: np.ndarray, *, samples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _reported_metrics(
    outcome: SelectionOutcome,
    fixed: SelectionOutcome,
    *,
    tie_tolerance: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result = _basic_metrics(outcome, tie_tolerance=tie_tolerance)
    result["mean_benefit_bootstrap_ci95"] = _bootstrap_mean_ci(
        outcome.benefits,
        samples=bootstrap_samples,
        seed=seed,
    )
    result["mean_benefit_delta_vs_fixed_root10"] = float(
        np.mean(outcome.benefits - fixed.benefits)
    )
    result["delta_vs_fixed_root10_bootstrap_ci95"] = _bootstrap_mean_ci(
        outcome.benefits - fixed.benefits,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return result


def _selection_rank(metrics: dict[str, Any]) -> tuple[Any, ...]:
    useful = metrics["wins"] > 0 and metrics["mean_benefit"] > 0.0
    balanced = metrics["wins"] >= 2 * metrics["losses"]
    return (
        metrics["predicate_regressions"] != 0,
        metrics["predicate_regressions"],
        not useful,
        not balanced,
        -metrics["mean_benefit"],
        -metrics["net_wins"],
        metrics["losses"],
        metrics["refreshes"],
    )


def _threshold_candidates(
    examples: Sequence[RootExample],
    scores: np.ndarray,
    episode_ids: Sequence[int],
) -> list[float]:
    by_episode: dict[int, list[float]] = collections.defaultdict(list)
    for example, score in zip(examples, scores, strict=True):
        by_episode[example.episode_id].append(float(score))
    maxima = np.asarray(
        [max(by_episode[episode]) for episode in episode_ids if by_episode.get(episode)],
        dtype=np.float64,
    )
    if maxima.size == 0:
        raise ValueError("No validation episode contains a root for threshold selection.")
    unique = np.unique(maxima)
    candidates = [float(np.nextafter(unique[-1], np.inf))]
    candidates.extend(float(value) for value in unique)
    if unique.size > 1:
        candidates.extend(float(value) for value in 0.5 * (unique[:-1] + unique[1:]))
    candidates.append(float(np.nextafter(unique[0], -np.inf)))
    return sorted(set(candidates))


def _select_threshold(
    examples: Sequence[RootExample],
    scores: np.ndarray,
    episode_ids: Sequence[int],
    *,
    tie_tolerance: float,
) -> tuple[float, SelectionOutcome, dict[str, Any]]:
    selected: tuple[tuple[Any, ...], float, SelectionOutcome, dict[str, Any]] | None = None
    for threshold in _threshold_candidates(examples, scores, episode_ids):
        outcome = _scored_outcome(
            examples,
            scores,
            episode_ids,
            threshold=threshold,
        )
        metrics = _basic_metrics(outcome, tie_tolerance=tie_tolerance)
        candidate = (_selection_rank(metrics), -threshold, outcome, metrics)
        if selected is None or candidate[:2] < selected[:2]:
            selected = candidate
    if selected is None:
        raise RuntimeError("No threshold candidate was evaluated.")
    return -selected[1], selected[2], selected[3]


def _choose_fixed_phase(
    examples: Sequence[RootExample],
    episode_ids: Sequence[int],
    *,
    tie_tolerance: float,
) -> tuple[int | None, SelectionOutcome]:
    candidates: list[tuple[tuple[Any, ...], int | None, SelectionOutcome]] = []
    none = _fixed_root_outcome(examples, episode_ids, root_index=None)
    candidates.append((_selection_rank(_basic_metrics(none, tie_tolerance=tie_tolerance)), None, none))
    for root_index in sorted({item.root_index for item in examples}):
        outcome = _fixed_root_outcome(examples, episode_ids, root_index=root_index)
        metrics = _basic_metrics(outcome, tie_tolerance=tie_tolerance)
        candidates.append((_selection_rank(metrics), root_index, outcome))
    _, root_index, outcome = min(
        candidates,
        key=lambda item: (item[0], item[1] is None, item[1] if item[1] is not None else -1),
    )
    return root_index, outcome


def _fit_variant(
    train_examples: Sequence[RootExample],
    validation_examples: Sequence[RootExample],
    train_features: np.ndarray,
    validation_features: np.ndarray,
    validation_episode_ids: Sequence[int],
    *,
    feature_variant: str,
    lambdas: Sequence[float],
    tie_tolerance: float,
) -> tuple[RidgeHead, float, SelectionOutcome, list[dict[str, Any]]]:
    target = np.asarray([item.benefit for item in train_examples], dtype=np.float64)
    candidates = []
    grid_summary = []
    for ridge_lambda in lambdas:
        head = _fit_ridge(
            train_features,
            target,
            ridge_lambda=ridge_lambda,
            feature_variant=feature_variant,
        )
        train_prediction = head.predict(train_features)
        validation_prediction = head.predict(validation_features)
        threshold, outcome, metrics = _select_threshold(
            validation_examples,
            validation_prediction,
            validation_episode_ids,
            tie_tolerance=tie_tolerance,
        )
        record = {
            "lambda": float(ridge_lambda),
            "threshold": float(threshold),
            "train_rmse": float(np.sqrt(np.mean(np.square(train_prediction - target)))),
            "validation": metrics,
        }
        grid_summary.append(record)
        candidates.append((_selection_rank(metrics), float(ridge_lambda), head, threshold, outcome))
    _, _, head, threshold, outcome = min(candidates, key=lambda item: (item[0], item[1]))
    return head, float(threshold), outcome, grid_summary


def _label_summary(examples: Sequence[RootExample]) -> dict[str, Any]:
    benefits = np.asarray([item.benefit for item in examples], dtype=np.float64)
    predicates = np.asarray([item.predicate_gain for item in examples], dtype=np.int64)
    h6 = np.asarray(
        [item.h6_benefit for item in examples if item.h6_benefit is not None],
        dtype=np.float64,
    )
    return {
        "roots": len(examples),
        "episodes": len({item.episode_id for item in examples}),
        "root_index_counts": {
            str(root): int(count)
            for root, count in sorted(collections.Counter(item.root_index for item in examples).items())
        },
        "handoff_benefit_mean": float(np.mean(benefits)),
        "handoff_benefit_std": float(np.std(benefits)),
        "handoff_benefit_positive": int(np.sum(benefits > 0.0)),
        "handoff_benefit_negative": int(np.sum(benefits < 0.0)),
        "handoff_benefit_zero": int(np.sum(benefits == 0.0)),
        "predicate_gain_positive": int(np.sum(predicates > 0)),
        "predicate_gain_negative": int(np.sum(predicates < 0)),
        "predicate_gain_zero": int(np.sum(predicates == 0)),
        "h6_benefit_count": int(h6.size),
        "h6_benefit_mean": float(np.mean(h6)) if h6.size else None,
    }


def _oracle_gap_metrics(
    method_mean: float,
    *,
    fixed_mean: float,
    oracle_mean: float,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "fraction_of_no_refresh_to_oracle_benefit": (
            float(method_mean / oracle_mean) if oracle_mean > 1e-12 else None
        )
    }
    gap = oracle_mean - fixed_mean
    result["fixed_root10_to_oracle_gap_closure"] = (
        float((method_mean - fixed_mean) / gap) if gap > 1e-12 else None
    )
    return result


def _write_head(path: pathlib.Path, head: RidgeHead, *, threshold: float) -> None:
    if head.feature_variant != FEATURE_TEMPORAL or head.mean.shape != (256,):
        raise ValueError(
            "Only a pure temporal_feature[256] head may be serialized for deployment."
        )
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        feature_variant=np.asarray("temporal"),
        mean=np.asarray(head.mean, dtype=np.float64),
        scale=np.asarray(head.scale, dtype=np.float64),
        coef=np.asarray(head.coef, dtype=np.float64),
        intercept=np.asarray(head.intercept, dtype=np.float64),
        threshold=np.asarray(threshold, dtype=np.float64),
        ridge_lambda=np.asarray(head.ridge_lambda, dtype=np.float64),
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    train_episode_ids = list(dict.fromkeys(args.train_episodes))
    validation_episode_ids = list(dict.fromkeys(args.validation_episodes))
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    head_path = output_dir / "head.npz"
    existing = [path for path in (summary_path, head_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist; pass --overwrite: {existing}")

    input_paths = _expand_inputs(args.collector_roots)
    examples, skipped = _load_examples(input_paths, task_id=args.task_id)
    train_set = set(train_episode_ids)
    validation_set = set(validation_episode_ids)
    train_examples = [item for item in examples if item.episode_id in train_set]
    validation_examples = [item for item in examples if item.episode_id in validation_set]
    observed_train = {item.episode_id for item in train_examples}
    observed_validation = {item.episode_id for item in validation_examples}
    missing_train = sorted(train_set - observed_train)
    missing_validation = sorted(validation_set - observed_validation)
    if missing_train or missing_validation:
        raise ValueError(
            f"Missing collector roots for train episodes {missing_train} or validation "
            f"episodes {missing_validation}."
        )

    train_temporal = _features(train_examples, FEATURE_TEMPORAL)
    validation_temporal = _features(validation_examples, FEATURE_TEMPORAL)
    train_combined = _features(train_examples, FEATURE_TEMPORAL_STATE_PLAN)
    validation_combined = _features(validation_examples, FEATURE_TEMPORAL_STATE_PLAN)

    fixed_validation = _fixed_root_outcome(
        validation_examples,
        validation_episode_ids,
        root_index=args.fixed_root_index,
    )
    phase_root, _ = _choose_fixed_phase(
        train_examples,
        train_episode_ids,
        tie_tolerance=args.tie_tolerance,
    )
    phase_validation = _fixed_root_outcome(
        validation_examples,
        validation_episode_ids,
        root_index=phase_root,
    )
    oracle_validation = _oracle_outcome(validation_examples, validation_episode_ids)

    temporal_head, temporal_threshold, temporal_outcome, temporal_grid = _fit_variant(
        train_examples,
        validation_examples,
        train_temporal,
        validation_temporal,
        validation_episode_ids,
        feature_variant=FEATURE_TEMPORAL,
        lambdas=args.ridge_lambdas,
        tie_tolerance=args.tie_tolerance,
    )
    combined_head, combined_threshold, combined_outcome, combined_grid = _fit_variant(
        train_examples,
        validation_examples,
        train_combined,
        validation_combined,
        validation_episode_ids,
        feature_variant=FEATURE_TEMPORAL_STATE_PLAN,
        lambdas=args.ridge_lambdas,
        tie_tolerance=args.tie_tolerance,
    )

    outcomes = {
        "fixed_root10": fixed_validation,
        "phase_only_fixed": phase_validation,
        "temporal": temporal_outcome,
        "temporal_state_plan_diagnostic": combined_outcome,
        "predicate_safe_oracle": oracle_validation,
    }
    validation_report = {}
    for index, (name, outcome) in enumerate(outcomes.items()):
        validation_report[name] = _reported_metrics(
            outcome,
            fixed_validation,
            tie_tolerance=args.tie_tolerance,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 100 * index,
        )
    validation_report["phase_only_fixed"]["train_selected_root_index"] = phase_root
    validation_report["temporal"]["lambda"] = temporal_head.ridge_lambda
    validation_report["temporal"]["threshold"] = temporal_threshold
    validation_report["temporal_state_plan_diagnostic"]["lambda"] = combined_head.ridge_lambda
    validation_report["temporal_state_plan_diagnostic"]["threshold"] = combined_threshold

    fixed_mean = validation_report["fixed_root10"]["mean_benefit"]
    oracle_mean = validation_report["predicate_safe_oracle"]["mean_benefit"]
    for name in ("fixed_root10", "phase_only_fixed", "temporal", "temporal_state_plan_diagnostic"):
        validation_report[name]["oracle_gap"] = _oracle_gap_metrics(
            validation_report[name]["mean_benefit"],
            fixed_mean=fixed_mean,
            oracle_mean=oracle_mean,
        )

    deployed = validation_report["temporal"]
    gates = {
        "not_worse_than_fixed_root10": deployed["mean_benefit"] + 1e-12 >= fixed_mean,
        "wins_at_least_twice_losses": deployed["wins"] >= 2 * deployed["losses"],
        "zero_predicate_regressions": deployed["predicate_regressions"] == 0,
        "positive_mean_benefit": deployed["mean_benefit"] > 0.0,
    }
    gates["go"] = all(gates.values())

    _write_head(head_path, temporal_head, threshold=temporal_threshold)
    summary = {
        "schema_version": 1,
        "method": {
            "name": "task8_plan_refresh_ridge_head",
            "target": (
                "fresh-minus-stale normalized progress after H6 plus same-seed fresh-teacher H9"
            ),
            "predicate_guard": (
                "fresh-minus-stale satisfied-predicate gain is validation-only and must have zero "
                "selected regressions"
            ),
            "decision_budget": "at most one selected refresh root per episode",
            "deployable_variant": "temporal_feature[256] ridge only",
            "diagnostic_variant": (
                "temporal plus anchor/current state and cached plan summaries; never serialized"
            ),
        },
        "args": {
            "collector_roots": args.collector_roots,
            "output_dir": str(output_dir),
            "task_id": args.task_id,
            "train_episodes": train_episode_ids,
            "validation_episodes": validation_episode_ids,
            "ridge_lambdas": [float(value) for value in args.ridge_lambdas],
            "fixed_root_index": args.fixed_root_index,
            "tie_tolerance": args.tie_tolerance,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "dataset": {
            "input_npz_files": len(input_paths),
            "valid_task_roots": len(examples),
            "skipped_invalid_or_other_task": skipped,
            "unused_valid_roots_outside_split": len(examples)
            - len(train_examples)
            - len(validation_examples),
            "train": _label_summary(train_examples),
            "validation": _label_summary(validation_examples),
        },
        "feature_schema": {
            "temporal": {
                "dimension": int(train_temporal.shape[1]),
                "fields": ["temporal_feature[256]"],
            },
            "temporal_state_plan_diagnostic": {
                "dimension": int(train_combined.shape[1]),
                "fields": [
                    "temporal_feature[256]",
                    "anchor_state,current_state,delta,abs_delta",
                    "mean,std,first,last for cached coarse/final/environment actions",
                ],
            },
            "standardization": "mean and scale fit on train roots only",
            "forbidden_deployment_inputs": [
                "root_index",
                "fresh outcomes",
                "stale outcomes",
                "privileged progress",
                "predicate gain",
            ],
        },
        "model_selection": {
            "threshold_source": "validation episodes only",
            "priority": (
                "online first-threshold-crossing; zero predicate regressions, positive mean, "
                "wins at least twice losses, then largest mean benefit"
            ),
            "temporal_lambda_grid": temporal_grid,
            "temporal_state_plan_diagnostic_lambda_grid": combined_grid,
        },
        "validation": validation_report,
        "deployment_artifact": {
            "path": str(head_path.resolve()),
            "feature_variant": "temporal",
            "fields": [
                "feature_variant",
                "mean",
                "scale",
                "coef",
                "intercept",
                "threshold",
            ],
            "diagnostic_combined_head_saved": False,
        },
        "go_no_go": {
            **gates,
            "bootstrap_ci_is_hard_gate": False,
            "mean_benefit_ci95": deployed["mean_benefit_bootstrap_ci95"],
            "delta_vs_fixed_root10_ci95": deployed[
                "delta_vs_fixed_root10_bootstrap_ci95"
            ],
        },
        "interpretation_guard": (
            "Lambda and threshold are selected and reported on the same validation episodes, so this "
            "is a fast development gate rather than an unbiased final estimate. A go result still "
            "requires held-out closed-loop confirmation."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
