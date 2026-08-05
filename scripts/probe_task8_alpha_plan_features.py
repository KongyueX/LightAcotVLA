"""Probe whether pre-final EAR/IAR features improve the Task8 alpha router.

This is an offline, episode-grouped probe over terminal paired roots.  It uses
the exact lexicographic labels, priority weights, ridge learner, metrics, and
group-fold construction from ``train_task8_alpha_outcome_router.py``.  It does
not read either arm's final actions as features and does not export an online
policy sidecar.

The compared feature families are:

* normalized state + decision step;
* state/step + a 35D EAR summary (mean/std/first/last/delta over 15x7);
* state/step + fold-train PCA of the pooled IAR token mean;
* state/step + EAR summary + fold-train PCA of the IAR token mean; and
* state/step + fold-train PCA of the pooled prefix, as a high-dimensional
  representation control.

Every PCA and every feature standardizer is fitted inside the relevant train
split.  The PCA dimension and ridge penalty are selected by inner grouped CV
inside every outer fold.  Thus the reported nested OOF scores never use a root
from the held-out episode to fit PCA, normalization, the classifier, or its
hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

import train_task8_alpha_outcome_router as base


EAR_SHAPE = (15, 7)
IAR_SHAPE = (18, 1024)
PREFIX_DIM = 2048
STATE_DIM = 32
EAR_SUMMARY_DIM = 5 * EAR_SHAPE[1]
EAR_SUMMARY_ORDER = (
    "mean[7]",
    "std[7]",
    "first[7]",
    "last[7]",
    "last_minus_first[7]",
)


@dataclasses.dataclass(frozen=True)
class Args:
    roots: tuple[str, ...]
    output_dir: str
    seed: int = 7
    ridge_lambdas: tuple[float, ...] = (10.0, 100.0, 1_000.0, 10_000.0)
    pca_dims: tuple[int, ...] = (2, 4, 8)
    feature_std_floor: float = 1e-5
    expected_roots: int = 29
    expected_terminal_discordant: int = 6
    expected_alternative_alpha: float = base.ALTERNATIVE_ALPHA
    coarse_arm_difference_tolerance: float = 1e-5
    allow_formal_episodes: bool = False
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    name: str
    fixed_blocks: tuple[str, ...]
    pca_source: str | None
    description: str


@dataclasses.dataclass(frozen=True)
class Projection:
    input_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        return (source - self.input_mean) @ self.components.T


@dataclasses.dataclass(frozen=True)
class Candidate:
    pca_dim: int
    ridge_lambda: float


@dataclasses.dataclass(frozen=True)
class FinalFit:
    candidate: Candidate
    projection: Projection | None
    classifier: base.RidgeClassifier
    feature_dim: int


FEATURE_SPECS = (
    FeatureSpec(
        "state_step",
        ("state_step",),
        None,
        "normalized_state[32] + decision_step[1]",
    ),
    FeatureSpec(
        "state_step_ear",
        ("state_step", "ear_summary"),
        None,
        "state/step + EAR mean/std/first/last/delta[35]",
    ),
    FeatureSpec(
        "state_step_iar",
        ("state_step",),
        "iar_token_mean",
        "state/step + fold-train PCA(IAR token mean[1024])",
    ),
    FeatureSpec(
        "state_step_ear_iar",
        ("state_step", "ear_summary"),
        "iar_token_mean",
        "state/step + EAR summary + fold-train PCA(IAR token mean[1024])",
    ),
    FeatureSpec(
        "state_step_prefix",
        ("state_step",),
        "prefix",
        "state/step + fold-train PCA(prefix[2048]) control",
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        action="append",
        required=True,
        help="Collector output directory, roots directory, or individual root NPZ; repeat as needed.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ridge-lambdas",
        nargs="+",
        type=float,
        default=[10.0, 100.0, 1_000.0, 10_000.0],
    )
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--feature-std-floor", type=float, default=1e-5)
    parser.add_argument("--expected-roots", type=int, default=29)
    parser.add_argument("--expected-terminal-discordant", type=int, default=6)
    parser.add_argument(
        "--expected-alternative-alpha", type=float, default=base.ALTERNATIVE_ALPHA
    )
    parser.add_argument("--coarse-arm-difference-tolerance", type=float, default=1e-5)
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
        output_dir=namespace.output_dir,
        seed=namespace.seed,
        ridge_lambdas=tuple(namespace.ridge_lambdas),
        pca_dims=tuple(namespace.pca_dims),
        feature_std_floor=namespace.feature_std_floor,
        expected_roots=namespace.expected_roots,
        expected_terminal_discordant=namespace.expected_terminal_discordant,
        expected_alternative_alpha=namespace.expected_alternative_alpha,
        coarse_arm_difference_tolerance=namespace.coarse_arm_difference_tolerance,
        allow_formal_episodes=namespace.allow_formal_episodes,
        overwrite=namespace.overwrite,
    )


def _validate_args(args: Args) -> None:
    if not args.roots:
        raise ValueError("At least one --roots input is required.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not args.ridge_lambdas or any(value < 1.0 for value in args.ridge_lambdas):
        raise ValueError("All ridge penalties must be >= 1 for this tiny-sample probe.")
    if not args.pca_dims or any(value <= 0 for value in args.pca_dims):
        raise ValueError("All PCA dimensions must be positive.")
    if len(set(args.pca_dims)) != len(args.pca_dims):
        raise ValueError("--pca-dims must not contain duplicates.")
    if args.feature_std_floor <= 0:
        raise ValueError("--feature-std-floor must be positive.")
    if args.expected_roots <= 0 or args.expected_terminal_discordant <= 0:
        raise ValueError("Expected root and terminal-discordant counts must be positive.")
    if args.coarse_arm_difference_tolerance < 0:
        raise ValueError("--coarse-arm-difference-tolerance must be non-negative.")
    if not np.isclose(
        args.expected_alternative_alpha, base.ALTERNATIVE_ALPHA, atol=1e-8
    ):
        raise ValueError("This probe supports only the collected alpha=0 versus alpha=.05 arms.")


def _prepare_output(args: Args) -> pathlib.Path:
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    return output_dir


def _required_array(
    data: Mapping[str, Any], name: str, shape: tuple[int, ...]
) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Root NPZ is missing required pre-final feature {name!r}.")
    value = np.asarray(data[name], dtype=np.float64)
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}; expected {shape}.")
    if np.any(~np.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return value


def _required_scalar(data: Mapping[str, Any], name: str) -> float:
    if name not in data:
        raise KeyError(f"Root NPZ is missing required field {name!r}.")
    value = float(np.asarray(data[name]).reshape(()).item())
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite.")
    return value


def _ear_summary(coarse_actions: np.ndarray) -> np.ndarray:
    coarse = np.asarray(coarse_actions, dtype=np.float64)
    return np.concatenate(
        [
            np.mean(coarse, axis=0),
            np.std(coarse, axis=0),
            coarse[0],
            coarse[-1],
            coarse[-1] - coarse[0],
        ]
    )


def _read_feature_arrays(
    examples: Sequence[base.RootExample], args: Args
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ear_rows: list[np.ndarray] = []
    iar_rows: list[np.ndarray] = []
    coarse_differences: list[float] = []
    for example in examples:
        with np.load(example.path, allow_pickle=False) as data:
            coarse = _required_array(data, "coarse_actions", EAR_SHAPE)
            iar = _required_array(data, "acot_iar_tokens", IAR_SHAPE)
            coarse_difference = _required_scalar(data, "coarse_max_abs_difference")
        if coarse_difference > args.coarse_arm_difference_tolerance:
            raise ValueError(
                "EAR is not arm-invariant at root "
                f"{example.path}: max difference {coarse_difference:.6g} exceeds "
                f"{args.coarse_arm_difference_tolerance:.6g}."
            )
        ear_rows.append(_ear_summary(coarse))
        iar_rows.append(np.mean(iar, axis=0))
        coarse_differences.append(coarse_difference)

    state = np.stack([item.normalized_state for item in examples]).astype(np.float64)
    step = np.asarray([item.decision_step for item in examples], dtype=np.float64)[:, None]
    prefix = np.stack([item.prefix_feature for item in examples]).astype(np.float64)
    arrays = {
        "state_step": np.concatenate([state, step], axis=1),
        "ear_summary": np.stack(ear_rows).astype(np.float64),
        "iar_token_mean": np.stack(iar_rows).astype(np.float64),
        "prefix": prefix,
    }
    expected_dims = {
        "state_step": STATE_DIM + 1,
        "ear_summary": EAR_SUMMARY_DIM,
        "iar_token_mean": IAR_SHAPE[1],
        "prefix": PREFIX_DIM,
    }
    for name, values in arrays.items():
        if values.shape != (len(examples), expected_dims[name]):
            raise ValueError(f"Unexpected {name} feature shape {values.shape}.")
        if np.any(~np.isfinite(values)):
            raise FloatingPointError(f"{name} feature contains non-finite values.")
    return arrays, np.asarray(coarse_differences, dtype=np.float64)


def _slice_arrays(
    arrays: Mapping[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[indices] for name, values in arrays.items()}


def _fit_projection(values: np.ndarray, output_dim: int) -> Projection:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"PCA input must be rank two; got {source.shape}.")
    maximum_dim = min(source.shape[0] - 1, source.shape[1])
    if output_dim > maximum_dim:
        raise ValueError(
            f"PCA dimension {output_dim} exceeds centered train rank bound {maximum_dim}."
        )
    input_mean = np.mean(source, axis=0)
    centered = source - input_mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors[:output_dim].copy()
    # SVD component signs are arbitrary.  Canonical signs make saved artifacts
    # stable without changing any transformed feature or classifier score.
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    energy = np.square(singular_values)
    total_energy = float(np.sum(energy))
    explained = (
        energy[:output_dim] / total_energy
        if total_energy > 0
        else np.zeros(output_dim, dtype=np.float64)
    )
    return Projection(input_mean, components, explained)


def _compose_features(
    spec: FeatureSpec,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    projection: Projection | None,
) -> np.ndarray:
    blocks = [np.asarray(arrays[name], dtype=np.float64)[indices] for name in spec.fixed_blocks]
    if spec.pca_source is not None:
        if projection is None:
            raise ValueError(f"Feature {spec.name} requires a fitted PCA projection.")
        blocks.append(projection.transform(np.asarray(arrays[spec.pca_source])[indices]))
    elif projection is not None:
        raise ValueError(f"Feature {spec.name} does not accept a PCA projection.")
    values = np.concatenate(blocks, axis=1)
    if np.any(~np.isfinite(values)):
        raise FloatingPointError(f"Composed feature {spec.name} contains non-finite values.")
    return values


def _fit_feature_projection(
    spec: FeatureSpec,
    arrays: Mapping[str, np.ndarray],
    train: np.ndarray,
    pca_dim: int,
) -> Projection | None:
    if spec.pca_source is None:
        if pca_dim != 0:
            raise ValueError(f"Non-PCA feature {spec.name} received pca_dim={pca_dim}.")
        return None
    return _fit_projection(np.asarray(arrays[spec.pca_source])[train], pca_dim)


def _headline(metrics: Mapping[str, Any]) -> dict[str, Any]:
    def summarize(values: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if values is None:
            return None
        coverage = values["two_sided_win_coverage"]
        return {
            "count": values["count"],
            "balanced_accuracy": values["balanced_accuracy"],
            "auroc": values["auroc"],
            "alternative_win_recall": coverage["alternative_win_recall"],
            "alpha0_win_recall": coverage["alpha0_win_recall"],
            "minimum_side_recall": coverage["minimum_side_recall"],
        }

    return {
        "overall_29_roots": summarize(metrics["overall_lexicographic_preference"]),
        "terminal_discordant_6_roots": summarize(
            metrics["terminal_success_difference_priority"]
        ),
    }


def _candidate_group_cv(
    spec: FeatureSpec,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    priority_tiers: np.ndarray,
    importance: np.ndarray,
    groups: np.ndarray,
    args: Args,
    *,
    seed: int,
) -> tuple[Candidate, dict[str, Any]]:
    folds, protocol = base._group_folds(groups, seed)
    all_indices = np.arange(labels.size)
    dimensions = args.pca_dims if spec.pca_source is not None else (0,)
    reports: dict[str, Any] = {}
    ranked: list[tuple[tuple[float, ...], Candidate]] = []
    for pca_dim in dimensions:
        for ridge_lambda in args.ridge_lambdas:
            scores = np.zeros(labels.size, dtype=np.float64)
            for test in folds:
                train = np.setdiff1d(all_indices, test, assume_unique=True)
                projection = _fit_feature_projection(
                    spec, arrays, train, int(pca_dim)
                )
                train_features = _compose_features(spec, arrays, train, projection)
                test_features = _compose_features(spec, arrays, test, projection)
                model = base._fit_ridge_classifier(
                    train_features,
                    labels[train],
                    importance[train],
                    ridge_lambda=float(ridge_lambda),
                    std_floor=args.feature_std_floor,
                )
                scores[test] = model.predict(test_features)
            metrics = base._router_metrics(labels, scores, priority_tiers)
            candidate = Candidate(int(pca_dim), float(ridge_lambda))
            key = f"pca_dim={candidate.pca_dim},ridge_lambda={candidate.ridge_lambda:g}"
            reports[key] = _headline(metrics)
            rank = base._metric_rank(metrics, candidate.ridge_lambda) + (
                -float(candidate.pca_dim),
            )
            ranked.append((rank, candidate))
    selected = max(ranked, key=lambda item: item[0])[1]
    return selected, {
        "protocol": protocol,
        "candidates": reports,
        "selected_pca_dim": selected.pca_dim,
        "selected_lambda": selected.ridge_lambda,
        "selection_rule": (
            "terminal-success-difference BA/accuracy, then overall BA, overall AUROC, "
            "stronger ridge, then smaller PCA dimension"
        ),
        "pca_leakage_control": "each candidate PCA was fitted separately on each inner-train fold",
    }


def _nested_oof(
    spec: FeatureSpec,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    priority_tiers: np.ndarray,
    importance: np.ndarray,
    groups: np.ndarray,
    args: Args,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    outer_folds, protocol = base._group_folds(groups, seed)
    all_indices = np.arange(labels.size)
    scores = np.zeros(labels.size, dtype=np.float64)
    folds_report: list[dict[str, Any]] = []
    for fold_index, test in enumerate(outer_folds):
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        train_groups = groups[train]
        if np.unique(train_groups).size >= 2:
            candidate, inner_report = _candidate_group_cv(
                spec,
                _slice_arrays(arrays, train),
                labels[train],
                priority_tiers[train],
                importance[train],
                train_groups,
                args,
                seed=seed + 100 + fold_index,
            )
        else:
            fallback_dim = min(args.pca_dims) if spec.pca_source is not None else 0
            candidate = Candidate(fallback_dim, max(args.ridge_lambdas))
            inner_report = {
                "protocol": "single_training_episode_no_inner_cv",
                "selected_pca_dim": candidate.pca_dim,
                "selected_lambda": candidate.ridge_lambda,
            }
        projection = _fit_feature_projection(spec, arrays, train, candidate.pca_dim)
        train_features = _compose_features(spec, arrays, train, projection)
        test_features = _compose_features(spec, arrays, test, projection)
        model = base._fit_ridge_classifier(
            train_features,
            labels[train],
            importance[train],
            ridge_lambda=candidate.ridge_lambda,
            std_floor=args.feature_std_floor,
        )
        scores[test] = model.predict(test_features)
        folds_report.append(
            {
                "fold": fold_index,
                "test_episode_ids": sorted(int(value) for value in np.unique(groups[test])),
                "train_count": int(train.size),
                "test_count": int(test.size),
                "selected_pca_dim": candidate.pca_dim,
                "selected_lambda": candidate.ridge_lambda,
                "training_class": model.training_class,
                "inner_selection": inner_report,
            }
        )
    metrics = base._router_metrics(labels, scores, priority_tiers)
    return scores, {
        "protocol": protocol,
        "num_folds": len(outer_folds),
        "folds": folds_report,
        "metrics": metrics,
        "headline": _headline(metrics),
        "leakage_control": (
            "outer-test episodes were excluded from inner hyperparameter selection, PCA, "
            "feature standardization, and ridge fitting"
        ),
    }


def _fit_full_development_model(
    spec: FeatureSpec,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    priority_tiers: np.ndarray,
    importance: np.ndarray,
    groups: np.ndarray,
    args: Args,
    *,
    seed: int,
) -> tuple[FinalFit, dict[str, Any]]:
    candidate, selection = _candidate_group_cv(
        spec,
        arrays,
        labels,
        priority_tiers,
        importance,
        groups,
        args,
        seed=seed,
    )
    all_indices = np.arange(labels.size)
    projection = _fit_feature_projection(spec, arrays, all_indices, candidate.pca_dim)
    features = _compose_features(spec, arrays, all_indices, projection)
    classifier = base._fit_ridge_classifier(
        features,
        labels,
        importance,
        ridge_lambda=candidate.ridge_lambda,
        std_floor=args.feature_std_floor,
    )
    return FinalFit(candidate, projection, classifier, int(features.shape[1])), selection


def _metric_value(metrics: Mapping[str, Any], section: str, field: str) -> float | None:
    values = metrics[section]
    if values is None:
        return None
    if field in ("alternative_win_recall", "alpha0_win_recall", "minimum_side_recall"):
        value = values["two_sided_win_coverage"][field]
    else:
        value = values[field]
    return float(value) if value is not None else None


def _metric_delta(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, dict[str, float | None]]:
    sections = (
        "overall_lexicographic_preference",
        "terminal_success_difference_priority",
    )
    fields = (
        "balanced_accuracy",
        "auroc",
        "alternative_win_recall",
        "alpha0_win_recall",
        "minimum_side_recall",
    )
    result: dict[str, dict[str, float | None]] = {}
    for section in sections:
        result[section] = {}
        for field in fields:
            left_value = _metric_value(left, section, field)
            right_value = _metric_value(right, section, field)
            result[section][field] = (
                left_value - right_value
                if left_value is not None and right_value is not None
                else None
            )
    return result


def _utility_report(
    examples: Sequence[base.RootExample],
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    count = len(examples)
    fixed_alpha0 = base._utility(examples, np.zeros(count, dtype=np.bool_))
    fixed_alpha05 = base._utility(examples, np.ones(count, dtype=np.bool_))
    oracle = base._utility(examples, labels > 0)
    routers = {
        name: base._utility(examples, values > 0.0) for name, values in scores.items()
    }
    for value in (fixed_alpha05, oracle, *routers.values()):
        value["vector_delta_vs_fixed_alpha0"] = base._vector_delta(value, fixed_alpha0)
    for value in routers.values():
        value["vector_delta_vs_fixed_alpha05"] = base._vector_delta(value, fixed_alpha05)
    baseline = routers["state_step"]
    for name, value in routers.items():
        value["vector_delta_vs_state_step_oof_router"] = base._vector_delta(
            value, baseline
        )
    return {
        "fixed_alpha0": fixed_alpha0,
        "fixed_alpha05": fixed_alpha05,
        "nested_oof_routers": routers,
        "per_root_hindsight_oracle": {
            **oracle,
            "vector_delta_vs_fixed_alpha05": base._vector_delta(oracle, fixed_alpha05),
            "warning": "per-root hindsight upper bound, not deployable performance",
        },
    }


def _save_model(
    path: pathlib.Path,
    spec: FeatureSpec,
    fit: FinalFit,
    episode_ids: np.ndarray,
) -> None:
    model = fit.classifier
    raw_kernel, raw_bias = base._raw_score_affine(model)
    projection = fit.projection
    pca_mean = (
        projection.input_mean
        if projection is not None
        else np.zeros(0, dtype=np.float64)
    )
    pca_components = (
        projection.components
        if projection is not None
        else np.zeros((0, 0), dtype=np.float64)
    )
    pca_explained = (
        projection.explained_variance_ratio
        if projection is not None
        else np.zeros(0, dtype=np.float64)
    )
    input_contract = {
        "fixed_feature_blocks": list(spec.fixed_blocks),
        "fixed_block_semantics": {
            "state_step": "normalized_state[32] followed by scalar decision_step",
            "ear_summary": list(EAR_SUMMARY_ORDER),
        },
        "pca_source": spec.pca_source or "none",
        "pca_transform": "(source - pca_input_mean) @ pca_components.T",
        "score": "concatenated_feature @ raw_score_kernel + raw_score_bias",
    }
    np.savez_compressed(
        path,
        feature_family=np.asarray(spec.name),
        feature_description=np.asarray(spec.description),
        fixed_feature_blocks=np.asarray(spec.fixed_blocks),
        ear_summary_order=np.asarray(EAR_SUMMARY_ORDER),
        pca_source=np.asarray(spec.pca_source or "none"),
        pca_input_mean=pca_mean.astype(np.float32),
        pca_components=pca_components.astype(np.float32),
        pca_explained_variance_ratio=pca_explained.astype(np.float32),
        pca_dim=np.asarray(fit.candidate.pca_dim, dtype=np.int16),
        composed_feature_dim=np.asarray(fit.feature_dim, dtype=np.int16),
        feature_mean=model.feature_mean.astype(np.float32),
        feature_std=model.feature_std.astype(np.float32),
        feature_scale=np.asarray(model.feature_scale, dtype=np.float32),
        target_mean=np.asarray(model.target_mean, dtype=np.float32),
        weights=model.weights.astype(np.float32),
        raw_score_kernel=raw_kernel.astype(np.float32),
        raw_score_bias=np.asarray(raw_bias, dtype=np.float32),
        ridge_lambda=np.asarray(model.ridge_lambda, dtype=np.float32),
        training_class=np.asarray(model.training_class),
        training_episode_ids=np.unique(episode_ids).astype(np.int16),
        score_semantics=np.asarray("score>0 selects alpha=.05; otherwise alpha=0"),
        feature_provenance=np.asarray(
            "pre-final one-NFE snapshot only; no alpha-arm final actions or outcomes are inputs"
        ),
        input_contract_json=np.asarray(json.dumps(input_contract, sort_keys=True)),
    )


def _family_rank(
    metrics: Mapping[str, Any], feature_dim: int, order: int
) -> tuple[float, ...]:
    # Hyperparameter-strength is deliberately excluded when comparing feature
    # families.  Lower-dimensional/earlier families win only exact metric ties.
    return base._metric_rank(metrics, 0.0)[:3] + (-float(feature_dim), -float(order))


def main(args: Args) -> None:
    _validate_args(args)
    output_dir = _prepare_output(args)
    paths = base._discover_roots(args.roots)
    examples = [base._read_root(path) for path in paths]
    examples.sort(key=lambda item: (item.episode_id, item.decision_step, item.physics_key))
    validation_args = base.Args(
        roots=args.roots,
        endpoint_student_params="unused_by_plan_probe",
        output_dir=args.output_dir,
        seed=args.seed,
        ridge_lambdas=args.ridge_lambdas,
        feature_std_floor=args.feature_std_floor,
        expected_alternative_alpha=args.expected_alternative_alpha,
        require_terminal=True,
        allow_formal_episodes=args.allow_formal_episodes,
        overwrite=args.overwrite,
    )
    base._validate_examples(examples, validation_args)
    if len(examples) != args.expected_roots:
        raise ValueError(f"Found {len(examples)} roots; expected exactly {args.expected_roots}.")

    preferences = [base._preference(item) for item in examples]
    labels = np.asarray([item.label for item in preferences], dtype=np.int8)
    priority_tiers = np.asarray(
        [item.priority_tier for item in preferences], dtype=np.int8
    )
    terminal_discordant = int(np.sum(priority_tiers == 1))
    if terminal_discordant != args.expected_terminal_discordant:
        raise ValueError(
            f"Found {terminal_discordant} terminal-discordant roots; "
            f"expected exactly {args.expected_terminal_discordant}."
        )
    importance = np.asarray(
        [base.PRIORITY_WEIGHTS[int(tier)] for tier in priority_tiers],
        dtype=np.float64,
    )
    episodes = np.asarray([item.episode_id for item in examples], dtype=np.int64)
    arrays, coarse_differences = _read_feature_arrays(examples, args)

    oof_scores: dict[str, np.ndarray] = {}
    oof_reports: dict[str, Any] = {}
    final_fits: dict[str, FinalFit] = {}
    full_selection: dict[str, Any] = {}
    artifact_paths: dict[str, pathlib.Path] = {}
    for feature_index, spec in enumerate(FEATURE_SPECS):
        scores, report = _nested_oof(
            spec,
            arrays,
            labels,
            priority_tiers,
            importance,
            episodes,
            args,
            seed=args.seed + feature_index * 1_000,
        )
        final_fit, selection = _fit_full_development_model(
            spec,
            arrays,
            labels,
            priority_tiers,
            importance,
            episodes,
            args,
            seed=args.seed + 10_000 + feature_index,
        )
        artifact_path = output_dir / "models" / f"{spec.name}.npz"
        _save_model(artifact_path, spec, final_fit, episodes)
        oof_scores[spec.name] = scores
        oof_reports[spec.name] = report
        final_fits[spec.name] = final_fit
        full_selection[spec.name] = selection
        artifact_paths[spec.name] = artifact_path

    ranked_families = [
        (
            _family_rank(
                oof_reports[spec.name]["metrics"],
                final_fits[spec.name].feature_dim,
                feature_index,
            ),
            spec,
        )
        for feature_index, spec in enumerate(FEATURE_SPECS)
    ]
    best_spec = max(ranked_families, key=lambda item: item[0])[1]
    best_path = output_dir / "best_deploy_plan_probe.npz"
    _save_model(best_path, best_spec, final_fits[best_spec.name], episodes)

    plan_specs = [spec for spec in FEATURE_SPECS if spec.name != "state_step"]
    best_plan_spec = max(
        plan_specs,
        key=lambda spec: _family_rank(
            oof_reports[spec.name]["metrics"],
            final_fits[spec.name].feature_dim,
            list(FEATURE_SPECS).index(spec),
        ),
    )
    baseline_metrics = oof_reports["state_step"]["metrics"]
    best_plan_metrics = oof_reports[best_plan_spec.name]["metrics"]
    plan_beats_baseline = base._metric_rank(best_plan_metrics, 0.0)[:3] > base._metric_rank(
        baseline_metrics, 0.0
    )[:3]

    utility = _utility_report(examples, labels, oof_scores)
    examples_path = output_dir / "examples_and_oof_predictions.npz"
    prediction_payload: dict[str, np.ndarray] = {
        "source_file": np.asarray([str(item.path) for item in examples]),
        "episode_id": episodes.astype(np.int16),
        "decision_step": np.asarray(
            [item.decision_step for item in examples], dtype=np.int32
        ),
        "preference_label": labels,
        "preference_reason": np.asarray([item.reason for item in preferences]),
        "preference_priority_tier": priority_tiers,
        "preference_training_weight": importance.astype(np.float32),
        "coarse_max_abs_difference": coarse_differences.astype(np.float32),
    }
    for name, scores in oof_scores.items():
        prediction_payload[f"{name}_oof_score"] = scores.astype(np.float32)
        prediction_payload[f"{name}_oof_selection"] = (scores > 0.0).astype(np.int8)
    np.savez_compressed(examples_path, **prediction_payload)

    reason_counts = {
        reason: int(sum(item.reason == reason for item in preferences))
        for reason in sorted({item.reason for item in preferences})
    }
    comparisons = {
        spec.name: _metric_delta(oof_reports[spec.name]["metrics"], baseline_metrics)
        for spec in FEATURE_SPECS
        if spec.name != "state_step"
    }
    summary = {
        "name": "Task8 pre-final plan-feature alpha-router probe",
        "status": "offline nested episode-grouped OOF probe; no policy sidecar exported",
        "config": {
            **dataclasses.asdict(args),
            "roots": list(args.roots),
            "ridge_lambdas": [float(value) for value in args.ridge_lambdas],
            "pca_dims": [int(value) for value in args.pca_dims],
        },
        "data": {
            "num_roots": len(examples),
            "episode_ids": sorted(int(value) for value in np.unique(episodes)),
            "num_episode_groups": int(np.unique(episodes).size),
            "terminal_evaluated_count": int(sum(item.terminal_evaluated for item in examples)),
            "terminal_discordant_count": terminal_discordant,
            "alternative_alpha": args.expected_alternative_alpha,
            "preference_counts": {
                "alternative_wins": int(np.sum(labels > 0)),
                "alpha0_wins": int(np.sum(labels < 0)),
                "ties": int(np.sum(labels == 0)),
            },
            "preference_reasons": reason_counts,
            "label_priority": [
                "terminal success difference",
                "both-success remaining calls",
                "both-success remaining steps",
                "both-fail terminal privileged progress",
                "H20 privileged progress fallback",
            ],
            "training_priority_weights": {
                str(tier): weight for tier, weight in base.PRIORITY_WEIGHTS.items()
            },
            "maximum_coarse_arm_difference": float(np.max(coarse_differences)),
        },
        "feature_contract": {
            "timing": "pre-final; one final-flow NFE is preserved",
            "state_step_dim": STATE_DIM + 1,
            "ear_input_shape": list(EAR_SHAPE),
            "ear_summary_dim": EAR_SUMMARY_DIM,
            "ear_summary_order": list(EAR_SUMMARY_ORDER),
            "iar_input_shape": list(IAR_SHAPE),
            "iar_pooling": "mean over 18 IAR tokens before fold-train PCA",
            "prefix_control_dim": PREFIX_DIM,
            "leakage_exclusion": (
                "feature loader reads only normalized state, decision step, alpha0 pre-final "
                "EAR/IAR/prefix snapshots, and the stored EAR arm-invariance check; neither "
                "arm's final actions nor outcome values are feature inputs"
            ),
            "pca_protocol": (
                "PCA is refit on every inner-train split for dimension/lambda selection, "
                "then on each outer-train split for OOF scoring"
            ),
        },
        "feature_families": {
            spec.name: {
                "description": spec.description,
                "nested_oof": oof_reports[spec.name],
                "full_development_set_selection": full_selection[spec.name],
                "full_development_set_fit": {
                    "pca_dim": final_fits[spec.name].candidate.pca_dim,
                    "ridge_lambda": final_fits[spec.name].candidate.ridge_lambda,
                    "composed_feature_dim": final_fits[spec.name].feature_dim,
                    "training_class": final_fits[spec.name].classifier.training_class,
                },
                "model_artifact": str(artifact_paths[spec.name]),
            }
            for spec in FEATURE_SPECS
        },
        "plan_feature_delta_vs_state_step": comparisons,
        "oof_lexicographic_utility": utility,
        "best_deployment": {
            "feature_family": best_spec.name,
            "selection_rule": (
                "nested-OOF terminal-discordant BA/accuracy, then overall BA, overall AUROC; "
                "lower feature dimension breaks exact ties"
            ),
            "artifact": str(best_path),
            "contains": (
                "full-development standardization, optional PCA mean/components, ridge weights, "
                "and an equivalent raw-score affine kernel"
            ),
        },
        "plan_signal_assessment": {
            "best_plan_family": best_plan_spec.name,
            "beats_state_step_under_prespecified_metric_rank": plan_beats_baseline,
            "headline_delta_vs_state_step": comparisons[best_plan_spec.name],
            "interpretation": (
                "A positive result is evidence of episode-generalizing pre-final plan signal on "
                "this 29-root development probe, not formal Task8 policy success. A non-positive "
                "result rejects these summaries/PCA choices, not all possible EAR/IAR uses."
            ),
        },
        "artifacts": {
            "examples_and_oof_predictions": str(examples_path),
            "best_deploy_plan_probe": str(best_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main(_args(_build_parser().parse_args()))
