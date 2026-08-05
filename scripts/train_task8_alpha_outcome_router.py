"""Train a tiny task-outcome router for Task8 alpha=0 versus alpha=.05.

Inputs are the per-root NPZ files written by
``collect_task8_alpha_snapshot_pairs.py``.  Labels are recomputed here instead
of trusting the collector's convenience preference field.  Their strict
lexicographic order is:

1. terminal success difference;
2. if both succeed, fewer remaining policy calls and then fewer steps;
3. if both fail, larger terminal privileged progress; and
4. H20 privileged progress as the final fallback.

The primary deployable feature is current-observation pooled prefix[2048] plus
normalized state[32].  A state[32]+decision-step compact control is evaluated
under the same episode-grouped protocol.  With at most tens of roots, both are
strongly regularized linear ridge classifiers; this script deliberately does
not fit an MLP.

Task8 formal episode IDs 10--29 are rejected by default.  Cross-validation is
leave-one-episode-out for at most ten episodes and five-fold episode-grouped
otherwise.  Ridge regularization is selected inside each outer fold, so the
reported OOF balanced accuracy, AUROC, coverage, and lexicographic utility do
not score a model on an episode used to fit it.

The exported scalar gate uses the existing runtime contract
``alpha=.05+.05*tanh(logit)``.  It is only a hard-routing approximation: ridge
score -1 maps to alpha=.01, score 0 to the 0/.05 decision boundary alpha=.025,
and score +1 to alpha=.05.  The NPZ outcome data cannot supervise arbitrary
continuous alpha values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import dataclasses
import json
import pathlib
from typing import Any

from flax import traverse_util
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as model_lib
from openpi.shared import download


TASK_ID = 8
FORMAL_EPISODES = frozenset(range(10, 30))
PRIMARY_FEATURE_DIM = 2048 + 32
COMPACT_FEATURE_DIM = 32 + 1
ALPHA0 = 0.0
ALTERNATIVE_ALPHA = 0.05
GATE_CENTER = 0.05
GATE_RADIUS = 0.05
GATE_DECISION_ALPHA = 0.025
GATE_LOW_APPROX_ALPHA = 0.01
PRIORITY_WEIGHTS = {
    1: 64.0,  # terminal rescue/regression
    2: 16.0,  # both-success remaining calls
    3: 8.0,   # both-success remaining steps
    4: 4.0,   # both-fail terminal privileged progress
    5: 1.0,   # H20 fallback
    6: 0.0,   # exact tie; excluded from fitting
}


@dataclasses.dataclass(frozen=True)
class Args:
    roots: tuple[str, ...]
    endpoint_student_params: str
    output_dir: str
    seed: int = 7
    ridge_lambdas: tuple[float, ...] = (10.0, 100.0, 1_000.0, 10_000.0)
    feature_std_floor: float = 1e-5
    expected_alternative_alpha: float = ALTERNATIVE_ALPHA
    require_terminal: bool = True
    allow_formal_episodes: bool = False
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class RootExample:
    path: pathlib.Path
    task_id: int
    episode_id: int
    decision_step: int
    physics_key: str
    alternative_alpha: float
    prefix_feature: np.ndarray
    normalized_state: np.ndarray
    terminal_evaluated: bool
    alpha0_success: bool
    alternative_success: bool
    alpha0_remaining_calls: int
    alternative_remaining_calls: int
    alpha0_remaining_steps: int
    alternative_remaining_steps: int
    alpha0_terminal_progress: float
    alternative_terminal_progress: float
    alpha0_h20_progress: float
    alternative_h20_progress: float
    collector_preference: int


@dataclasses.dataclass(frozen=True)
class Preference:
    label: int
    reason: str
    priority_tier: int


@dataclasses.dataclass(frozen=True)
class RidgeClassifier:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_scale: float
    target_mean: float
    weights: np.ndarray
    ridge_lambda: float
    training_class: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.feature_mean) / self.feature_std * self.feature_scale
        return (normalized @ self.weights + self.target_mean).astype(np.float64)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        action="append",
        required=True,
        help="Collector output directory, roots directory, or individual root NPZ; repeat as needed.",
    )
    parser.add_argument("--endpoint-student-params", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ridge-lambdas",
        nargs="+",
        type=float,
        default=[10.0, 100.0, 1_000.0, 10_000.0],
    )
    parser.add_argument("--feature-std-floor", type=float, default=1e-5)
    parser.add_argument("--expected-alternative-alpha", type=float, default=ALTERNATIVE_ALPHA)
    parser.add_argument(
        "--require-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-formal-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _args(namespace: argparse.Namespace) -> Args:
    return Args(
        roots=tuple(namespace.roots),
        endpoint_student_params=namespace.endpoint_student_params,
        output_dir=namespace.output_dir,
        seed=namespace.seed,
        ridge_lambdas=tuple(namespace.ridge_lambdas),
        feature_std_floor=namespace.feature_std_floor,
        expected_alternative_alpha=namespace.expected_alternative_alpha,
        require_terminal=namespace.require_terminal,
        allow_formal_episodes=namespace.allow_formal_episodes,
        overwrite=namespace.overwrite,
    )


def _validate_args(args: Args) -> None:
    if not args.roots:
        raise ValueError("At least one --roots input is required.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not args.ridge_lambdas or any(value < 1.0 for value in args.ridge_lambdas):
        raise ValueError(
            "This tiny-sample trainer requires strong positive ridge penalties; all values must be >= 1."
        )
    if args.feature_std_floor <= 0:
        raise ValueError("--feature-std-floor must be positive.")
    if not np.isclose(args.expected_alternative_alpha, ALTERNATIVE_ALPHA, atol=1e-8):
        raise ValueError("The deployable hard approximation currently supports only alpha=0 versus alpha=.05.")


def _discover_roots(inputs: Sequence[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for value in inputs:
        candidate = pathlib.Path(value).resolve()
        if candidate.is_file() and candidate.suffix == ".npz":
            paths.append(candidate)
            continue
        if not candidate.is_dir():
            raise FileNotFoundError(f"Root input is not an NPZ or directory: {candidate}")
        direct = sorted(candidate.glob("*.npz"))
        nested = sorted((candidate / "roots").glob("*.npz")) if (candidate / "roots").is_dir() else []
        paths.extend(direct or nested)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No root NPZ files found under {list(inputs)}.")
    return unique


def _scalar(data: Mapping[str, Any], name: str) -> Any:
    if name not in data:
        raise KeyError(f"Root NPZ is missing required field {name!r}.")
    return np.asarray(data[name]).reshape(()).item()


def _vector(data: Mapping[str, Any], name: str, size: int) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Root NPZ is missing required field {name!r}.")
    value = np.asarray(data[name], dtype=np.float64).reshape(-1)
    if value.shape != (size,):
        raise ValueError(f"{name} has shape {value.shape}; expected {(size,)}.")
    if np.any(~np.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return value


def _read_root(path: pathlib.Path) -> RootExample:
    with np.load(path, allow_pickle=False) as data:
        task_id = int(_scalar(data, "task_id"))
        episode_id = int(_scalar(data, "episode_id"))
        decision_step = int(_scalar(data, "decision_step"))
        alpha0_absolute_step = int(_scalar(data, "alpha0_terminal_absolute_step"))
        alternative_absolute_step = int(_scalar(data, "alternative_terminal_absolute_step"))
        return RootExample(
            path=path,
            task_id=task_id,
            episode_id=episode_id,
            decision_step=decision_step,
            physics_key=str(_scalar(data, "physics_key")),
            alternative_alpha=float(_scalar(data, "alternative_alpha")),
            prefix_feature=_vector(data, "acot_prefix_feature", 2048),
            normalized_state=_vector(data, "normalized_state", 32),
            terminal_evaluated=bool(_scalar(data, "terminal_evaluated")),
            alpha0_success=bool(_scalar(data, "alpha0_terminal_success")),
            alternative_success=bool(_scalar(data, "alternative_terminal_success")),
            # Include the initial action-generation call.  It is common to both
            # arms, but makes the reported quantity an actual remaining-call count.
            alpha0_remaining_calls=1 + int(_scalar(data, "alpha0_continuation_calls")),
            alternative_remaining_calls=1 + int(
                _scalar(data, "alternative_continuation_calls")
            ),
            alpha0_remaining_steps=max(0, alpha0_absolute_step - decision_step),
            alternative_remaining_steps=max(0, alternative_absolute_step - decision_step),
            alpha0_terminal_progress=float(_scalar(data, "alpha0_terminal_normalized_score")),
            alternative_terminal_progress=float(
                _scalar(data, "alternative_terminal_normalized_score")
            ),
            alpha0_h20_progress=float(_scalar(data, "alpha0_h20_normalized_score")),
            alternative_h20_progress=float(
                _scalar(data, "alternative_h20_normalized_score")
            ),
            collector_preference=int(_scalar(data, "preference_label")),
        )


def _validate_examples(examples: Sequence[RootExample], args: Args) -> None:
    if len(examples) < 2:
        raise ValueError("At least two roots are required.")
    wrong_task = sorted({item.task_id for item in examples if item.task_id != TASK_ID})
    if wrong_task:
        raise ValueError(f"Only zero-based Task8 roots are accepted; found task IDs {wrong_task}.")
    forbidden = sorted({item.episode_id for item in examples} & FORMAL_EPISODES)
    if forbidden and not args.allow_formal_episodes:
        raise ValueError(
            "Task8 formal episode IDs 10-29 are forbidden by default. "
            f"Refusing roots from episodes {forbidden}."
        )
    if args.require_terminal:
        incomplete = [str(item.path) for item in examples if not item.terminal_evaluated]
        if incomplete:
            raise ValueError(
                "Task-outcome routing requires terminal labels, but some roots are progress-only: "
                f"{incomplete[:5]}. Recollect/upgrade with --terminal or explicitly pass --no-require-terminal."
            )
    bad_alpha = [
        (str(item.path), item.alternative_alpha)
        for item in examples
        if not np.isclose(item.alternative_alpha, args.expected_alternative_alpha, atol=1e-7)
    ]
    if bad_alpha:
        raise ValueError(f"Roots contain an unexpected alternative alpha: {bad_alpha[:5]}.")
    keys = [(item.episode_id, item.decision_step, item.physics_key) for item in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate episode/decision/physics roots were supplied.")
    if len({item.episode_id for item in examples}) < 2:
        raise ValueError("Episode-grouped evaluation needs roots from at least two episodes.")
    outcome_values = np.asarray(
        [
            value
            for item in examples
            for value in (
                item.alpha0_terminal_progress,
                item.alternative_terminal_progress,
                item.alpha0_h20_progress,
                item.alternative_h20_progress,
            )
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(outcome_values)):
        raise FloatingPointError("Root outcome/progress labels contain non-finite values.")


def _compare(left: float, right: float, tolerance: float = 1e-8) -> int:
    difference = left - right
    if difference > tolerance:
        return 1
    if difference < -tolerance:
        return -1
    return 0


def _preference(example: RootExample) -> Preference:
    """Return +1 when alpha=.05 wins and -1 when alpha=0 wins."""

    if example.terminal_evaluated:
        success_difference = int(example.alternative_success) - int(example.alpha0_success)
        if success_difference:
            return Preference(
                success_difference,
                "terminal_rescue" if success_difference > 0 else "terminal_regression",
                1,
            )
        if example.alpha0_success and example.alternative_success:
            call_comparison = _compare(
                -example.alternative_remaining_calls,
                -example.alpha0_remaining_calls,
            )
            if call_comparison:
                return Preference(
                    call_comparison,
                    "both_success_fewer_calls" if call_comparison > 0 else "both_success_more_calls",
                    2,
                )
            step_comparison = _compare(
                -example.alternative_remaining_steps,
                -example.alpha0_remaining_steps,
            )
            if step_comparison:
                return Preference(
                    step_comparison,
                    "both_success_fewer_steps" if step_comparison > 0 else "both_success_more_steps",
                    3,
                )
        elif not example.alpha0_success and not example.alternative_success:
            terminal_progress = _compare(
                example.alternative_terminal_progress,
                example.alpha0_terminal_progress,
            )
            if terminal_progress:
                return Preference(
                    terminal_progress,
                    "both_fail_terminal_progress_gain"
                    if terminal_progress > 0
                    else "both_fail_terminal_progress_loss",
                    4,
                )

    h20_progress = _compare(
        example.alternative_h20_progress,
        example.alpha0_h20_progress,
    )
    if h20_progress:
        return Preference(
            h20_progress,
            "h20_progress_gain" if h20_progress > 0 else "h20_progress_loss",
            5,
        )
    return Preference(0, "lexicographic_tie", 6)


def _feature_arrays(examples: Sequence[RootExample]) -> dict[str, np.ndarray]:
    prefix = np.stack([item.prefix_feature for item in examples]).astype(np.float64)
    state = np.stack([item.normalized_state for item in examples]).astype(np.float64)
    decision_step = np.asarray([item.decision_step for item in examples], dtype=np.float64)[:, None]
    return {
        "prefix_state": np.concatenate([prefix, state], axis=-1),
        "state_decision_step": np.concatenate([state, decision_step], axis=-1),
    }


def _group_folds(groups: np.ndarray, seed: int) -> tuple[list[np.ndarray], str]:
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("Episode-grouped evaluation needs at least two groups.")
    if unique.size <= 10:
        return [np.flatnonzero(groups == group) for group in unique], "leave_one_episode_out"
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    chunks = np.array_split(shuffled, min(5, unique.size))
    return [np.flatnonzero(np.isin(groups, chunk)) for chunk in chunks], "five_fold_episode_grouped"


def _class_weights(labels: np.ndarray, importance: np.ndarray) -> np.ndarray:
    weights = np.asarray(importance, dtype=np.float64).copy()
    total = float(np.sum(weights))
    for value in (-1, 1):
        mask = labels == value
        if np.any(mask):
            class_total = float(np.sum(weights[mask]))
            if class_total <= 0:
                raise ValueError(f"Class {value} has non-positive priority weight.")
            weights[mask] *= total / (2.0 * class_total)
    return weights


def _fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    importance: np.ndarray,
    *,
    ridge_lambda: float,
    std_floor: float,
) -> RidgeClassifier:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    priority = np.asarray(importance, dtype=np.float64)
    if (
        values.ndim != 2
        or targets.shape != (values.shape[0],)
        or priority.shape != targets.shape
    ):
        raise ValueError(
            f"Invalid classifier shapes: features={values.shape}, labels={targets.shape}, "
            f"importance={priority.shape}."
        )
    feature_mean = np.mean(values, axis=0)
    feature_std = np.maximum(np.std(values, axis=0), std_floor)
    # Dimension-normalized features make one ridge grid meaningfully comparable
    # between the 2080D primary input and the 33D compact control.
    feature_scale = float(1.0 / np.sqrt(values.shape[1]))
    decisive = targets != 0
    decisive_targets = targets[decisive].astype(np.float64)
    if decisive_targets.size == 0:
        return RidgeClassifier(
            feature_mean,
            feature_std,
            feature_scale,
            -1.0,
            np.zeros(values.shape[1], dtype=np.float64),
            ridge_lambda,
            "no_decisive_labels_alpha0_fallback",
        )
    classes = np.unique(decisive_targets)
    if classes.size == 1:
        return RidgeClassifier(
            feature_mean,
            feature_std,
            feature_scale,
            float(classes[0]),
            np.zeros(values.shape[1], dtype=np.float64),
            ridge_lambda,
            "constant_positive" if classes[0] > 0 else "constant_negative",
        )

    x = (values[decisive] - feature_mean) / feature_std * feature_scale
    sample_weight = _class_weights(decisive_targets, priority[decisive])
    target_mean = float(np.average(decisive_targets, weights=sample_weight))
    sqrt_weight = np.sqrt(sample_weight)
    weighted_x = x * sqrt_weight[:, None]
    weighted_y = (decisive_targets - target_mean) * sqrt_weight
    system = weighted_x @ weighted_x.T + ridge_lambda * np.eye(weighted_x.shape[0])
    dual = np.linalg.solve(system, weighted_y)
    weights = weighted_x.T @ dual
    return RidgeClassifier(
        feature_mean,
        feature_std,
        feature_scale,
        target_mean,
        weights,
        ridge_lambda,
        "two_class_balanced_ridge",
    )


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    decisive = y != 0
    y = y[decisive]
    values = values[decisive]
    positive = y > 0
    negative = y < 0
    if not np.any(positive) or not np.any(negative):
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_count = int(np.sum(positive))
    negative_count = int(np.sum(negative))
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    prediction = np.where(np.asarray(scores) > 0.0, 1, -1)
    decisive = y != 0
    positive = y == 1
    negative = y == -1
    true_positive_rate = float(np.mean(prediction[positive] == 1)) if np.any(positive) else None
    true_negative_rate = float(np.mean(prediction[negative] == -1)) if np.any(negative) else None
    balanced_accuracy = (
        0.5 * (true_positive_rate + true_negative_rate)
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    predicted_positive = prediction == 1
    predicted_negative = prediction == -1
    return {
        "count": int(y.size),
        "decisive_count": int(np.sum(decisive)),
        "alternative_win_count": int(np.sum(positive)),
        "alpha0_win_count": int(np.sum(negative)),
        "tie_count": int(np.sum(y == 0)),
        "balanced_accuracy": balanced_accuracy,
        "auroc": _auc(y, scores),
        "decisive_accuracy": float(np.mean(prediction[decisive] == y[decisive]))
        if np.any(decisive)
        else None,
        "two_sided_win_coverage": {
            "alternative_win_recall": true_positive_rate,
            "alpha0_win_recall": true_negative_rate,
            "minimum_side_recall": min(true_positive_rate, true_negative_rate)
            if true_positive_rate is not None and true_negative_rate is not None
            else None,
            "alternative_selection_precision": float(np.mean(y[predicted_positive] == 1))
            if np.any(predicted_positive)
            else None,
            "alpha0_selection_precision": float(np.mean(y[predicted_negative] == -1))
            if np.any(predicted_negative)
            else None,
            "alternative_selection_coverage": float(np.mean(predicted_positive)),
            "alpha0_selection_coverage": float(np.mean(predicted_negative)),
        },
    }


def _router_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    priority_tiers: np.ndarray,
) -> dict[str, Any]:
    terminal = np.asarray(priority_tiers) == 1
    return {
        "overall_lexicographic_preference": _classification_metrics(labels, scores),
        "terminal_success_difference_priority": _classification_metrics(
            np.asarray(labels)[terminal],
            np.asarray(scores)[terminal],
        )
        if np.any(terminal)
        else None,
    }


def _metric_rank(
    metrics: Mapping[str, Any], ridge_lambda: float
) -> tuple[float, float, float, float]:
    overall = metrics["overall_lexicographic_preference"]
    terminal = metrics["terminal_success_difference_priority"]
    terminal_primary = -1.0
    if terminal is not None:
        terminal_primary = (
            terminal["balanced_accuracy"]
            if terminal["balanced_accuracy"] is not None
            else terminal["decisive_accuracy"]
            if terminal["decisive_accuracy"] is not None
            else -1.0
        )
    balanced = overall["balanced_accuracy"]
    auroc = overall["auroc"]
    return (
        float(terminal_primary),
        float(balanced) if balanced is not None else -1.0,
        float(auroc) if auroc is not None else -1.0,
        float(ridge_lambda),  # prefer stronger regularization on exact metric ties
    )


def _candidate_group_cv(
    features: np.ndarray,
    labels: np.ndarray,
    priority_tiers: np.ndarray,
    importance: np.ndarray,
    groups: np.ndarray,
    lambdas: Sequence[float],
    *,
    seed: int,
    std_floor: float,
) -> tuple[float, dict[str, Any]]:
    folds, protocol = _group_folds(groups, seed)
    candidate_report: dict[str, Any] = {}
    candidates: list[tuple[tuple[float, float, float, float], float]] = []
    all_indices = np.arange(labels.size)
    for ridge_lambda in lambdas:
        scores = np.zeros(labels.size, dtype=np.float64)
        for test in folds:
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            model = _fit_ridge_classifier(
                features[train],
                labels[train],
                importance[train],
                ridge_lambda=ridge_lambda,
                std_floor=std_floor,
            )
            scores[test] = model.predict(features[test])
        metrics = _router_metrics(labels, scores, priority_tiers)
        candidate_report[str(float(ridge_lambda))] = metrics
        candidates.append((_metric_rank(metrics, ridge_lambda), float(ridge_lambda)))
    selected = max(candidates, key=lambda item: item[0])[1]
    return selected, {
        "protocol": protocol,
        "candidates": candidate_report,
        "selected_lambda": selected,
        "selection_rule": (
            "max terminal-success-difference accuracy/balanced accuracy first, then overall "
            "balanced accuracy, AUROC, and stronger ridge"
        ),
    }


def _nested_oof(
    features: np.ndarray,
    labels: np.ndarray,
    priority_tiers: np.ndarray,
    importance: np.ndarray,
    groups: np.ndarray,
    lambdas: Sequence[float],
    *,
    seed: int,
    std_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    outer_folds, protocol = _group_folds(groups, seed)
    all_indices = np.arange(labels.size)
    scores = np.zeros(labels.size, dtype=np.float64)
    folds_report: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer_folds):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        train_groups = groups[train]
        if np.unique(train_groups).size >= 2:
            selected_lambda, inner = _candidate_group_cv(
                features[train],
                labels[train],
                priority_tiers[train],
                importance[train],
                train_groups,
                lambdas,
                seed=seed + 100 + fold_index,
                std_floor=std_floor,
            )
        else:
            selected_lambda = max(lambdas)
            inner = {
                "protocol": "single_training_episode_no_inner_cv",
                "selected_lambda": selected_lambda,
            }
        model = _fit_ridge_classifier(
            features[train],
            labels[train],
            importance[train],
            ridge_lambda=selected_lambda,
            std_floor=std_floor,
        )
        scores[test] = model.predict(features[test])
        folds_report.append(
            {
                "fold": fold_index,
                "test_episode_ids": sorted(int(value) for value in np.unique(groups[test])),
                "train_count": int(train.size),
                "test_count": int(test.size),
                "selected_lambda": selected_lambda,
                "training_class": model.training_class,
                "inner_selection": inner,
            }
        )
    return scores, {
        "protocol": protocol,
        "num_folds": len(outer_folds),
        "folds": folds_report,
        "metrics": _router_metrics(labels, scores, priority_tiers),
    }


def _arm_values(examples: Sequence[RootExample], use_alternative: np.ndarray) -> dict[str, np.ndarray]:
    choose = np.asarray(use_alternative, dtype=np.bool_)
    alpha0_success = np.asarray([item.alpha0_success for item in examples], dtype=np.bool_)
    alternative_success = np.asarray(
        [item.alternative_success for item in examples], dtype=np.bool_
    )
    return {
        "success": np.where(choose, alternative_success, alpha0_success),
        "calls": np.where(
            choose,
            [item.alternative_remaining_calls for item in examples],
            [item.alpha0_remaining_calls for item in examples],
        ).astype(np.float64),
        "steps": np.where(
            choose,
            [item.alternative_remaining_steps for item in examples],
            [item.alpha0_remaining_steps for item in examples],
        ).astype(np.float64),
        "terminal_progress": np.where(
            choose,
            [item.alternative_terminal_progress for item in examples],
            [item.alpha0_terminal_progress for item in examples],
        ).astype(np.float64),
        "h20_progress": np.where(
            choose,
            [item.alternative_h20_progress for item in examples],
            [item.alpha0_h20_progress for item in examples],
        ).astype(np.float64),
    }


def _utility(examples: Sequence[RootExample], use_alternative: np.ndarray) -> dict[str, Any]:
    values = _arm_values(examples, use_alternative)
    success = values["success"].astype(np.float64)
    failure = 1.0 - success
    vector = np.asarray(
        [
            np.mean(success),
            -np.mean(success * values["calls"]),
            -np.mean(success * values["steps"]),
            np.mean(failure * values["terminal_progress"]),
            np.mean(values["h20_progress"]),
        ],
        dtype=np.float64,
    )
    return {
        "lexicographic_vector": [float(value) for value in vector],
        "vector_semantics": [
            "terminal_success_rate (maximize first)",
            "negative success-weighted remaining calls",
            "negative success-weighted remaining steps",
            "failure-weighted terminal privileged progress",
            "H20 privileged progress fallback",
        ],
        "terminal_success_count": int(np.sum(values["success"])),
        "terminal_success_rate": float(np.mean(values["success"])),
        "mean_remaining_calls_on_success": float(np.mean(values["calls"][values["success"]]))
        if np.any(values["success"])
        else None,
        "mean_remaining_steps_on_success": float(np.mean(values["steps"][values["success"]]))
        if np.any(values["success"])
        else None,
        "mean_terminal_progress_on_failure": float(
            np.mean(values["terminal_progress"][~values["success"]])
        )
        if np.any(~values["success"])
        else None,
        "mean_h20_progress": float(np.mean(values["h20_progress"])),
        "alternative_selection_coverage": float(np.mean(use_alternative)),
    }


def _vector_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[float]:
    return [
        float(a - b)
        for a, b in zip(
            left["lexicographic_vector"],
            right["lexicographic_vector"],
            strict=True,
        )
    ]


def _utility_report(
    examples: Sequence[RootExample],
    labels: np.ndarray,
    main_scores: np.ndarray,
    compact_scores: np.ndarray,
) -> dict[str, Any]:
    count = len(examples)
    fixed_alpha0 = _utility(examples, np.zeros(count, dtype=np.bool_))
    fixed_alpha05 = _utility(examples, np.ones(count, dtype=np.bool_))
    oracle_selection = labels > 0
    oracle = _utility(examples, oracle_selection)
    main_router = _utility(examples, main_scores > 0.0)
    compact_router = _utility(examples, compact_scores > 0.0)
    for value in (fixed_alpha05, oracle, main_router, compact_router):
        value["vector_delta_vs_fixed_alpha0"] = _vector_delta(value, fixed_alpha0)
    main_router["vector_delta_vs_fixed_alpha05"] = _vector_delta(main_router, fixed_alpha05)
    compact_router["vector_delta_vs_fixed_alpha05"] = _vector_delta(
        compact_router, fixed_alpha05
    )
    return {
        "fixed_alpha0": fixed_alpha0,
        "fixed_alpha05": fixed_alpha05,
        "primary_prefix_state_oof_router": main_router,
        "compact_state_step_oof_router": compact_router,
        "per_root_hindsight_oracle": oracle,
        "oracle_gain": {
            "alternative_wins_over_alpha0": int(np.sum(labels > 0)),
            "alpha0_wins_over_alternative": int(np.sum(labels < 0)),
            "ties": int(np.sum(labels == 0)),
            "two_sided_oracle_improvement_coverage": float(np.mean(labels != 0)),
            "vector_delta_vs_fixed_alpha0": _vector_delta(oracle, fixed_alpha0),
            "vector_delta_vs_fixed_alpha05": _vector_delta(oracle, fixed_alpha05),
            "warning": "per-root hindsight upper bound, not deployable performance",
        },
    }


def _raw_score_affine(model: RidgeClassifier) -> tuple[np.ndarray, float]:
    kernel = model.feature_scale * model.weights / model.feature_std
    bias = model.target_mean - float(
        (model.feature_scale * model.feature_mean / model.feature_std) @ model.weights
    )
    return kernel.astype(np.float64), bias


def _gate_affine(model: RidgeClassifier) -> tuple[np.ndarray, np.ndarray]:
    raw_kernel, raw_bias = _raw_score_affine(model)
    decision_logit = float(
        np.arctanh((GATE_DECISION_ALPHA - GATE_CENTER) / GATE_RADIUS)
    )
    high_logit = 0.0  # alpha=.05
    score_scale = high_logit - decision_logit
    kernel = (score_scale * raw_kernel)[:, None].astype(np.float32)
    bias = np.asarray([decision_logit + score_scale * raw_bias], dtype=np.float32)
    return kernel, bias


def _save_classifier(path: pathlib.Path, model: RidgeClassifier, feature_name: str) -> None:
    raw_kernel, raw_bias = _raw_score_affine(model)
    np.savez_compressed(
        path,
        feature_mean=model.feature_mean.astype(np.float32),
        feature_std=model.feature_std.astype(np.float32),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float32),
        target_mean=np.asarray(model.target_mean, dtype=np.float32),
        weights=model.weights.astype(np.float32),
        raw_score_kernel=raw_kernel.astype(np.float32),
        raw_score_bias=np.asarray(raw_bias, dtype=np.float32),
        ridge_lambda=np.asarray(model.ridge_lambda, dtype=np.float32),
        training_class=np.asarray(model.training_class),
        feature_name=np.asarray(feature_name),
        score_semantics=np.asarray("score>0 selects alpha=.05; otherwise alpha=0"),
    )


def _save_orbax(target: pathlib.Path, params: dict[str, Any], overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(target, {"params": params}, force=overwrite)


def _save_gate_sidecars(
    output_dir: pathlib.Path,
    model: RidgeClassifier,
    endpoint_student_params: str,
    *,
    overwrite: bool,
) -> tuple[pathlib.Path, pathlib.Path]:
    kernel, bias = _gate_affine(model)
    if kernel.shape != (PRIMARY_FEATURE_DIM, 1) or bias.shape != (1,):
        raise ValueError(f"Unexpected gate shapes: kernel={kernel.shape}, bias={bias.shape}.")
    gate_flat = {
        ("adaptive_final_time_warp_gate", "kernel"): kernel,
        ("adaptive_final_time_warp_gate", "bias"): bias,
    }
    sidecar_path = pathlib.Path(download.maybe_download(endpoint_student_params))
    final_ir = model_lib.convert_str_keys_to_int(
        model_lib.restore_params(sidecar_path, restore_type=np.ndarray)
    )
    combined_flat = dict(traverse_util.flatten_dict(final_ir))
    overlap = set(combined_flat).intersection(gate_flat)
    if overlap:
        raise ValueError(f"Input sidecar already contains adaptive gate leaves: {sorted(overlap)}")
    combined_flat.update(gate_flat)

    gate_path = output_dir / "adaptive_gate_only" / "params"
    combined_path = output_dir / "combined_final_ir_adaptive_gate" / "params"
    _save_orbax(gate_path, traverse_util.unflatten_dict(gate_flat), overwrite)
    _save_orbax(combined_path, traverse_util.unflatten_dict(combined_flat), overwrite)
    return gate_path, combined_path


def _prepare_output(args: Args) -> pathlib.Path:
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = _prepare_output(args)
    paths = _discover_roots(args.roots)
    examples = [_read_root(path) for path in paths]
    examples.sort(key=lambda item: (item.episode_id, item.decision_step, item.physics_key))
    _validate_examples(examples, args)

    preferences = [_preference(item) for item in examples]
    labels = np.asarray([item.label for item in preferences], dtype=np.int8)
    priority_tiers = np.asarray([item.priority_tier for item in preferences], dtype=np.int8)
    importance = np.asarray(
        [PRIORITY_WEIGHTS[int(tier)] for tier in priority_tiers],
        dtype=np.float64,
    )
    episodes = np.asarray([item.episode_id for item in examples], dtype=np.int64)
    features = _feature_arrays(examples)
    if features["prefix_state"].shape[1] != PRIMARY_FEATURE_DIM:
        raise ValueError(f"Primary feature dimension is {features['prefix_state'].shape[1]}, expected 2080.")
    if features["state_decision_step"].shape[1] != COMPACT_FEATURE_DIM:
        raise ValueError(f"Compact feature dimension is {features['state_decision_step'].shape[1]}, expected 33.")

    oof_scores: dict[str, np.ndarray] = {}
    cv_reports: dict[str, Any] = {}
    final_models: dict[str, RidgeClassifier] = {}
    final_selection: dict[str, Any] = {}
    for feature_index, (name, values) in enumerate(features.items()):
        scores, cv_report = _nested_oof(
            values,
            labels,
            priority_tiers,
            importance,
            episodes,
            args.ridge_lambdas,
            seed=args.seed + feature_index * 1_000,
            std_floor=args.feature_std_floor,
        )
        selected_lambda, selection_report = _candidate_group_cv(
            values,
            labels,
            priority_tiers,
            importance,
            episodes,
            args.ridge_lambdas,
            seed=args.seed + 10_000 + feature_index,
            std_floor=args.feature_std_floor,
        )
        final_model = _fit_ridge_classifier(
            values,
            labels,
            importance,
            ridge_lambda=selected_lambda,
            std_floor=args.feature_std_floor,
        )
        oof_scores[name] = scores
        cv_reports[name] = cv_report
        final_models[name] = final_model
        final_selection[name] = selection_report

    utility = _utility_report(
        examples,
        labels,
        oof_scores["prefix_state"],
        oof_scores["state_decision_step"],
    )
    primary_model_path = output_dir / "prefix_state_ridge_router.npz"
    compact_model_path = output_dir / "state_step_ridge_router.npz"
    _save_classifier(primary_model_path, final_models["prefix_state"], "pooled_prefix_2048+state_32")
    _save_classifier(
        compact_model_path,
        final_models["state_decision_step"],
        "state_32+decision_step_1",
    )
    gate_path, combined_path = _save_gate_sidecars(
        output_dir,
        final_models["prefix_state"],
        args.endpoint_student_params,
        overwrite=args.overwrite,
    )

    examples_path = output_dir / "examples_and_oof_predictions.npz"
    np.savez_compressed(
        examples_path,
        source_file=np.asarray([str(item.path) for item in examples]),
        task_id=np.asarray([item.task_id for item in examples], dtype=np.int16),
        episode_id=episodes.astype(np.int32),
        decision_step=np.asarray([item.decision_step for item in examples], dtype=np.int32),
        physics_key=np.asarray([item.physics_key for item in examples]),
        prefix_feature=np.stack([item.prefix_feature for item in examples]).astype(np.float16),
        normalized_state=np.stack([item.normalized_state for item in examples]).astype(np.float16),
        preference_label=labels,
        preference_reason=np.asarray([item.reason for item in preferences]),
        preference_priority_tier=np.asarray(
            [item.priority_tier for item in preferences], dtype=np.int8
        ),
        preference_training_weight=importance.astype(np.float32),
        collector_preference_label=np.asarray(
            [item.collector_preference for item in examples], dtype=np.int8
        ),
        alpha0_terminal_success=np.asarray(
            [item.alpha0_success for item in examples], dtype=np.bool_
        ),
        alternative_terminal_success=np.asarray(
            [item.alternative_success for item in examples], dtype=np.bool_
        ),
        alpha0_remaining_calls=np.asarray(
            [item.alpha0_remaining_calls for item in examples], dtype=np.int16
        ),
        alternative_remaining_calls=np.asarray(
            [item.alternative_remaining_calls for item in examples], dtype=np.int16
        ),
        alpha0_remaining_steps=np.asarray(
            [item.alpha0_remaining_steps for item in examples], dtype=np.int32
        ),
        alternative_remaining_steps=np.asarray(
            [item.alternative_remaining_steps for item in examples], dtype=np.int32
        ),
        alpha0_terminal_progress=np.asarray(
            [item.alpha0_terminal_progress for item in examples], dtype=np.float32
        ),
        alternative_terminal_progress=np.asarray(
            [item.alternative_terminal_progress for item in examples], dtype=np.float32
        ),
        alpha0_h20_progress=np.asarray(
            [item.alpha0_h20_progress for item in examples], dtype=np.float32
        ),
        alternative_h20_progress=np.asarray(
            [item.alternative_h20_progress for item in examples], dtype=np.float32
        ),
        prefix_state_oof_score=oof_scores["prefix_state"].astype(np.float32),
        prefix_state_oof_selection=(oof_scores["prefix_state"] > 0).astype(np.int8),
        state_step_oof_score=oof_scores["state_decision_step"].astype(np.float32),
        state_step_oof_selection=(oof_scores["state_decision_step"] > 0).astype(np.int8),
    )

    collector_disagreements = int(
        np.sum(
            labels
            != np.asarray([item.collector_preference for item in examples], dtype=np.int8)
        )
    )
    reason_counts = {
        reason: int(sum(item.reason == reason for item in preferences))
        for reason in sorted({item.reason for item in preferences})
    }
    summary = {
        "name": "Task8 terminal-outcome alpha router",
        "status": "offline paired-root outcome model; formal Task8 closed-loop evaluation still required",
        "config": {
            **dataclasses.asdict(args),
            "roots": list(args.roots),
            "ridge_lambdas": [float(value) for value in args.ridge_lambdas],
        },
        "data": {
            "num_roots": len(examples),
            "episode_ids": sorted(int(value) for value in np.unique(episodes)),
            "num_episode_groups": int(np.unique(episodes).size),
            "formal_episode_ids_present": sorted(
                int(value) for value in set(episodes.tolist()) & FORMAL_EPISODES
            ),
            "formal_episode_override_used": bool(args.allow_formal_episodes),
            "terminal_evaluated_count": int(sum(item.terminal_evaluated for item in examples)),
            "alternative_alpha": args.expected_alternative_alpha,
            "label_priority": [
                "terminal success difference",
                "both-success remaining calls",
                "both-success remaining steps",
                "both-fail terminal privileged progress",
                "H20 privileged progress fallback",
            ],
            "training_priority_weights": {
                str(tier): weight for tier, weight in PRIORITY_WEIGHTS.items()
            },
            "preference_counts": {
                "alternative_wins": int(np.sum(labels > 0)),
                "alpha0_wins": int(np.sum(labels < 0)),
                "ties": int(np.sum(labels == 0)),
            },
            "preference_reasons": reason_counts,
            "collector_label_disagreements": collector_disagreements,
        },
        "model": {
            "family": (
                "class-balanced, lexicographic-tier-weighted, strongly-regularized linear ridge; no MLP"
            ),
            "primary_feature": "pooled current-observation prefix[2048] + normalized state[32]",
            "compact_control_feature": "normalized state[32] + decision_step[1]",
            "nested_episode_grouped_oof": cv_reports,
            "full_development_set_lambda_selection": final_selection,
            "full_development_set_training_class": {
                name: model.training_class for name, model in final_models.items()
            },
        },
        "oof_metrics": {
            "primary_prefix_state": cv_reports["prefix_state"]["metrics"],
            "compact_state_decision_step": cv_reports["state_decision_step"]["metrics"],
        },
        "lexicographic_utility": utility,
        "gate_export": {
            "runtime_contract": "alpha=.05+.05*tanh(logit)",
            "hard_decision": "nearest of {0,.05}; score>0 selects .05",
            "score_mapping": {
                "score_-1_alpha": GATE_LOW_APPROX_ALPHA,
                "score_0_alpha": GATE_DECISION_ALPHA,
                "score_+1_alpha": ALTERNATIVE_ALPHA,
            },
            "limitation": (
                "existing scalar gate is an unbounded affine logit and cannot exactly clamp alpha to {0,.05}; "
                "the sidecar is a hard-routing approximation trained only from those two arms"
            ),
        },
        "artifacts": {
            "examples_and_oof_predictions": str(examples_path),
            "primary_router": str(primary_model_path),
            "compact_control_router": str(compact_model_path),
            "adaptive_gate_only_sidecar": str(gate_path),
            "combined_final_ir_adaptive_gate_sidecar": str(combined_path),
        },
        "interpretation_guard": (
            "OOF root-level outcome prediction and hindsight oracle gain do not establish formal episode success. "
            "Use only development episodes here; evaluate the frozen combined sidecar once on IDs10-29."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(_args(_build_parser().parse_args()))
