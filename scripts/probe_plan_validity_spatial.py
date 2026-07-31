"""Probe whether spatial current-observation mismatch predicts refresh benefit.

This is an offline signal gate, not a deployable policy.  It deliberately keeps
all learned preprocessing inside episode-held-out folds and compares the real
current observation against two matched controls:

* ``anchor`` replaces the current observation with the anchor observation;
* ``shuffle`` replaces it with another episode's current observation at the
  same root index.

Only deployment-visible anchor/current observations, states, and cached plans
are features.  Fresh actions, progress, and predicate outcomes are labels only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torchvision.models import ResNet18_Weights, resnet18


VARIANTS = ("real", "anchor", "shuffle")


@dataclass(frozen=True)
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    target_mean: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale) @ self.coefficient + self.target_mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-roots", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=12)
    parser.add_argument(
        "--ridge-lambdas",
        nargs="+",
        type=float,
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--expected-state-lambda", type=float, default=10.0)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=32)
    return parser


def _scalar(data: Any, name: str, dtype: type) -> Any:
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"{name} must be scalar, got {value.shape}.")
    return dtype(value.reshape(()).item())


def _load_examples(path: pathlib.Path) -> dict[str, np.ndarray]:
    paths = sorted(path.rglob("*.npz")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"No NPZ roots found under {path}.")

    rows: list[dict[str, Any]] = []
    for root_path in paths:
        with np.load(root_path, allow_pickle=False) as data:
            if "advantage_handoff_normalized_score" not in data.files:
                continue
            rows.append(
                {
                    "episode": _scalar(data, "episode_id", int),
                    "root": _scalar(data, "root_index", int),
                    "anchor_images": np.asarray(data["anchor_images"], dtype=np.uint8),
                    "current_images": np.asarray(data["current_images"], dtype=np.uint8),
                    "anchor_state": np.asarray(data["anchor_state"], dtype=np.float64),
                    "current_state": np.asarray(data["current_state"], dtype=np.float64),
                    "coarse": np.asarray(data["cached_coarse_actions"], dtype=np.float64),
                    "final": np.asarray(data["cached_final_actions"], dtype=np.float64),
                    "environment": np.asarray(data["cached_env_actions"], dtype=np.float64),
                    "prefix": np.asarray(data["intended_prefix_env"], dtype=np.float64),
                    "benefit": _scalar(data, "advantage_handoff_normalized_score", float),
                    "predicate_gain": _scalar(
                        data,
                        "advantage_handoff_satisfied_count",
                        int,
                    ),
                }
            )
    if not rows:
        raise ValueError("No compatible plan-refresh roots were found.")

    def stack(name: str) -> np.ndarray:
        return np.stack([row[name] for row in rows])

    return {
        "episode": np.asarray([row["episode"] for row in rows], dtype=np.int64),
        "root": np.asarray([row["root"] for row in rows], dtype=np.int64),
        "anchor_images": stack("anchor_images"),
        "current_images": stack("current_images"),
        "anchor_state": stack("anchor_state"),
        "current_state": stack("current_state"),
        "coarse": stack("coarse"),
        "final": stack("final"),
        "environment": stack("environment"),
        "prefix": stack("prefix"),
        "benefit": np.asarray([row["benefit"] for row in rows], dtype=np.float64),
        "predicate_gain": np.asarray(
            [row["predicate_gain"] for row in rows],
            dtype=np.int64,
        ),
    }


def _sequence_summary(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (values.mean(axis=1), values.std(axis=1), values[:, 0], values[:, -1]),
        axis=1,
    )


def _plan_features(data: dict[str, np.ndarray]) -> np.ndarray:
    count = len(data["episode"])
    return np.concatenate(
        (
            data["anchor_state"],
            data["prefix"].reshape(count, -1),
            data["environment"][:, 4:10].reshape(count, -1),
            _sequence_summary(data["coarse"]),
            _sequence_summary(data["final"]),
        ),
        axis=1,
    )


def _extract_spatial_features(
    anchor_images: np.ndarray,
    current_images: np.ndarray,
    *,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.set_num_threads(threads)
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    scale = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = np.concatenate((anchor_images, current_images), axis=1)
    images = images.reshape(-1, *images.shape[2:])

    started = time.perf_counter()
    features: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(images), 64):
            batch = torch.from_numpy(images[offset : offset + 64])
            batch = batch.permute(0, 3, 1, 2).float() / 255.0
            batch = (batch - mean) / scale
            batch = model.conv1(batch)
            batch = model.bn1(batch)
            batch = model.relu(batch)
            batch = model.maxpool(batch)
            batch = model.layer1(batch)
            batch = model.layer2(batch)
            batch = model.layer3(batch)
            features.append(np.asarray(batch, dtype=np.float32))
    elapsed = time.perf_counter() - started
    maps = np.concatenate(features, axis=0).reshape(
        len(anchor_images),
        4,
        256,
        4,
        4,
    )
    return maps[:, :2], maps[:, 2:], elapsed


def _fit_pca(
    anchor_maps: np.ndarray,
    current_maps: np.ndarray,
    indices: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate((anchor_maps[indices], current_maps[indices]), axis=1)
    points = points.transpose(0, 1, 3, 4, 2).reshape(-1, points.shape[2])
    points = np.asarray(points, dtype=np.float64)
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[-dimension:][::-1]
    return mean, eigenvectors[:, order]


def _project_maps(
    maps: np.ndarray,
    pca: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, basis = pca
    points = maps.transpose(0, 1, 3, 4, 2).astype(np.float64)
    return (points - mean) @ basis


def _fit_ridge(values: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> Ridge:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (values - mean) / scale
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    dual = np.linalg.solve(
        normalized @ normalized.T + ridge_lambda * np.eye(len(normalized)),
        centered_targets,
    )
    return Ridge(
        mean=mean,
        scale=scale,
        coefficient=normalized.T @ dual,
        target_mean=target_mean,
    )


def _shuffle_mapping(
    episode_ids: np.ndarray,
    root_indices: np.ndarray,
    subsets: Sequence[np.ndarray],
) -> np.ndarray:
    mapping = np.arange(len(episode_ids))
    for subset in subsets:
        for root_index in sorted(set(root_indices[subset])):
            indices = subset[root_indices[subset] == root_index]
            indices = indices[np.argsort(episode_ids[indices])]
            mapping[indices] = np.roll(indices, 1)
    return mapping


def _build_features(
    *,
    variant: str,
    indices: np.ndarray,
    anchor_maps: np.ndarray,
    current_maps: np.ndarray,
    anchor_state: np.ndarray,
    current_state: np.ndarray,
    plan_features: np.ndarray,
    pca: tuple[np.ndarray, np.ndarray],
    expected_delta: Ridge,
    shuffle_mapping: np.ndarray,
) -> np.ndarray:
    projected_anchor = _project_maps(anchor_maps[indices], pca)
    if variant == "real":
        projected_current = _project_maps(current_maps[indices], pca)
        selected_state = current_state[indices]
    elif variant == "anchor":
        projected_current = projected_anchor.copy()
        selected_state = anchor_state[indices]
    elif variant == "shuffle":
        projected_current = _project_maps(current_maps[shuffle_mapping[indices]], pca)
        selected_state = current_state[shuffle_mapping[indices]]
    else:
        raise ValueError(f"Unknown variant {variant!r}.")

    visual_delta = projected_current - projected_anchor
    visual = np.concatenate(
        (
            projected_current,
            visual_delta,
            np.abs(visual_delta),
            projected_current * projected_anchor,
        ),
        axis=-1,
    ).reshape(len(indices), -1)
    state_delta = selected_state - anchor_state[indices]
    predicted_delta = expected_delta.predict(plan_features[indices])
    innovation = state_delta - predicted_delta
    state = np.concatenate(
        (
            selected_state,
            state_delta,
            np.abs(state_delta),
            innovation,
            np.abs(innovation),
        ),
        axis=1,
    )
    return np.concatenate((visual, state, plan_features[indices]), axis=1)


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    value = float(spearmanr(prediction, target).statistic)
    return value if np.isfinite(value) else 0.0


def _selection_outcome(
    *,
    scores: np.ndarray,
    indices: np.ndarray,
    threshold: float,
    episodes: np.ndarray,
    roots: np.ndarray,
    benefit: np.ndarray,
    predicate_gain: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    selected_benefits: list[float] = []
    selected_predicates: list[int] = []
    selected_roots: list[int] = []
    for episode_id in sorted(set(episodes[indices])):
        candidates = indices[episodes[indices] == episode_id]
        candidates = candidates[np.argsort(roots[candidates])]
        selected = candidates[scores[candidates] >= threshold]
        if len(selected):
            root_id = int(selected[0])
            selected_benefits.append(float(benefit[root_id]))
            selected_predicates.append(int(predicate_gain[root_id]))
            selected_roots.append(int(roots[root_id]))
        else:
            selected_benefits.append(0.0)
            selected_predicates.append(0)
            selected_roots.append(-1)
    values = np.asarray(selected_benefits, dtype=np.float64)
    predicates = np.asarray(selected_predicates, dtype=np.int64)
    return {
        "mean_benefit": float(values.mean()),
        "wins": int(np.sum(values > tolerance)),
        "losses": int(np.sum(values < -tolerance)),
        "ties_or_skips": int(np.sum(np.abs(values) <= tolerance)),
        "predicate_regressions": int(np.sum(predicates < 0)),
        "coverage": int(np.sum(np.asarray(selected_roots) >= 0)),
        "selected_roots": selected_roots,
    }


def _choose_threshold(
    *,
    scores: np.ndarray,
    indices: np.ndarray,
    episodes: np.ndarray,
    roots: np.ndarray,
    benefit: np.ndarray,
    predicate_gain: np.ndarray,
    tolerance: float,
) -> tuple[float, dict[str, Any]]:
    candidates = np.unique(np.quantile(scores[indices], np.linspace(0.0, 1.0, 41)))
    candidates = np.concatenate((candidates, np.asarray([np.inf])))
    best: tuple[tuple[Any, ...], float, dict[str, Any]] | None = None
    for threshold in candidates:
        outcome = _selection_outcome(
            scores=scores,
            indices=indices,
            threshold=float(threshold),
            episodes=episodes,
            roots=roots,
            benefit=benefit,
            predicate_gain=predicate_gain,
            tolerance=tolerance,
        )
        safe = int(outcome["predicate_regressions"] == 0)
        balanced = int(outcome["wins"] >= 2 * outcome["losses"] and outcome["wins"] > 0)
        key = (
            safe,
            balanced,
            outcome["mean_benefit"],
            outcome["wins"] - outcome["losses"],
            -outcome["coverage"],
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), outcome)
    assert best is not None
    return best[1], best[2]


def main() -> None:
    args = build_parser().parse_args()
    if args.folds < 2 or args.pca_dim <= 0:
        raise ValueError("folds must be >=2 and pca-dim must be positive.")
    if any(value <= 0 for value in args.ridge_lambdas):
        raise ValueError("ridge lambdas must be positive.")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_examples(pathlib.Path(args.collector_roots))
    episodes = data["episode"]
    roots = data["root"]
    benefit = data["benefit"]
    predicate_gain = data["predicate_gain"]
    train_episodes = np.asarray(args.train_episodes, dtype=np.int64)
    validation_episodes = np.asarray(args.validation_episodes, dtype=np.int64)
    if set(train_episodes) & set(validation_episodes):
        raise ValueError("Train and validation episodes overlap.")
    train_indices = np.flatnonzero(np.isin(episodes, train_episodes))
    validation_indices = np.flatnonzero(np.isin(episodes, validation_episodes))
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Train and validation selections must both be non-empty.")

    anchor_maps, current_maps, extraction_seconds = _extract_spatial_features(
        data["anchor_images"],
        data["current_images"],
        threads=args.torch_threads,
    )
    plan_features = _plan_features(data)
    fold_episodes = [train_episodes[index :: args.folds] for index in range(args.folds)]
    variant_results: dict[str, Any] = {}

    for variant in VARIANTS:
        predictions_by_lambda = {
            ridge_lambda: np.full(len(episodes), np.nan, dtype=np.float64)
            for ridge_lambda in args.ridge_lambdas
        }
        for held_out_episodes in fold_episodes:
            fold_validation = train_indices[np.isin(episodes[train_indices], held_out_episodes)]
            fold_train = train_indices[
                ~np.isin(episodes[train_indices], held_out_episodes)
            ]
            pca = _fit_pca(anchor_maps, current_maps, fold_train, args.pca_dim)
            expected_delta = _fit_ridge(
                plan_features[fold_train],
                data["current_state"][fold_train] - data["anchor_state"][fold_train],
                args.expected_state_lambda,
            )
            shuffle_mapping = _shuffle_mapping(
                episodes,
                roots,
                (fold_train, fold_validation),
            )
            train_features = _build_features(
                variant=variant,
                indices=fold_train,
                anchor_maps=anchor_maps,
                current_maps=current_maps,
                anchor_state=data["anchor_state"],
                current_state=data["current_state"],
                plan_features=plan_features,
                pca=pca,
                expected_delta=expected_delta,
                shuffle_mapping=shuffle_mapping,
            )
            validation_features = _build_features(
                variant=variant,
                indices=fold_validation,
                anchor_maps=anchor_maps,
                current_maps=current_maps,
                anchor_state=data["anchor_state"],
                current_state=data["current_state"],
                plan_features=plan_features,
                pca=pca,
                expected_delta=expected_delta,
                shuffle_mapping=shuffle_mapping,
            )
            for ridge_lambda in args.ridge_lambdas:
                model = _fit_ridge(
                    train_features,
                    benefit[fold_train, None],
                    ridge_lambda,
                )
                predictions_by_lambda[ridge_lambda][fold_validation] = model.predict(
                    validation_features
                ).reshape(-1)

        selected_lambda = max(
            args.ridge_lambdas,
            key=lambda value: _spearman(
                predictions_by_lambda[value][train_indices],
                benefit[train_indices],
            ),
        )
        oof_predictions = predictions_by_lambda[selected_lambda]
        threshold, oof_selection = _choose_threshold(
            scores=oof_predictions,
            indices=train_indices,
            episodes=episodes,
            roots=roots,
            benefit=benefit,
            predicate_gain=predicate_gain,
            tolerance=args.tie_tolerance,
        )
        fold_positive_count = 0
        for held_out_episodes in fold_episodes:
            fold_indices = train_indices[
                np.isin(episodes[train_indices], held_out_episodes)
            ]
            fold_outcome = _selection_outcome(
                scores=oof_predictions,
                indices=fold_indices,
                threshold=threshold,
                episodes=episodes,
                roots=roots,
                benefit=benefit,
                predicate_gain=predicate_gain,
                tolerance=args.tie_tolerance,
            )
            fold_positive_count += int(fold_outcome["mean_benefit"] > 0)

        pca = _fit_pca(anchor_maps, current_maps, train_indices, args.pca_dim)
        expected_delta = _fit_ridge(
            plan_features[train_indices],
            data["current_state"][train_indices] - data["anchor_state"][train_indices],
            args.expected_state_lambda,
        )
        shuffle_mapping = _shuffle_mapping(
            episodes,
            roots,
            (train_indices, validation_indices),
        )
        train_features = _build_features(
            variant=variant,
            indices=train_indices,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=data["anchor_state"],
            current_state=data["current_state"],
            plan_features=plan_features,
            pca=pca,
            expected_delta=expected_delta,
            shuffle_mapping=shuffle_mapping,
        )
        validation_features = _build_features(
            variant=variant,
            indices=validation_indices,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=data["anchor_state"],
            current_state=data["current_state"],
            plan_features=plan_features,
            pca=pca,
            expected_delta=expected_delta,
            shuffle_mapping=shuffle_mapping,
        )
        model = _fit_ridge(
            train_features,
            benefit[train_indices, None],
            selected_lambda,
        )
        validation_predictions = model.predict(validation_features).reshape(-1)
        validation_scores = np.full(len(episodes), np.nan, dtype=np.float64)
        validation_scores[validation_indices] = validation_predictions
        validation_selection = _selection_outcome(
            scores=validation_scores,
            indices=validation_indices,
            threshold=threshold,
            episodes=episodes,
            roots=roots,
            benefit=benefit,
            predicate_gain=predicate_gain,
            tolerance=args.tie_tolerance,
        )
        variant_results[variant] = {
            "ridge_lambda": selected_lambda,
            "threshold": threshold,
            "oof_spearman": _spearman(
                oof_predictions[train_indices],
                benefit[train_indices],
            ),
            "oof_positive_folds": fold_positive_count,
            "oof_selection": oof_selection,
            "validation_spearman": _spearman(
                validation_predictions,
                benefit[validation_indices],
            ),
            "validation_selection": validation_selection,
        }

    phase_candidates: list[dict[str, Any]] = []
    for root_index in sorted(set(roots[train_indices])):
        indices = train_indices[roots[train_indices] == root_index]
        phase_candidates.append(
            {
                "root": int(root_index),
                "mean_benefit": float(benefit[indices].mean()),
                "wins": int(np.sum(benefit[indices] > args.tie_tolerance)),
                "losses": int(np.sum(benefit[indices] < -args.tie_tolerance)),
                "predicate_regressions": int(np.sum(predicate_gain[indices] < 0)),
            }
        )
    safe_phases = [item for item in phase_candidates if item["predicate_regressions"] == 0]
    best_phase = max(safe_phases or phase_candidates, key=lambda item: item["mean_benefit"])
    phase_validation_indices = validation_indices[
        roots[validation_indices] == best_phase["root"]
    ]
    phase_validation = {
        "root": best_phase["root"],
        "mean_benefit": float(benefit[phase_validation_indices].mean()),
        "wins": int(np.sum(benefit[phase_validation_indices] > args.tie_tolerance)),
        "losses": int(np.sum(benefit[phase_validation_indices] < -args.tie_tolerance)),
        "predicate_regressions": int(
            np.sum(predicate_gain[phase_validation_indices] < 0)
        ),
    }
    oracle_values: list[float] = []
    for episode_id in sorted(set(episodes[validation_indices])):
        indices = validation_indices[episodes[validation_indices] == episode_id]
        oracle_values.append(max(0.0, float(np.max(benefit[indices]))))
    oracle_mean = float(np.mean(oracle_values))
    for result in variant_results.values():
        selected_mean = result["validation_selection"]["mean_benefit"]
        result["validation_oracle_gap_closure_pct"] = (
            100.0 * selected_mean / oracle_mean if oracle_mean > 0 else 0.0
        )

    real = variant_results["real"]
    controls = (variant_results["anchor"], variant_results["shuffle"])
    go_checks = {
        "oof_spearman_margin_at_least_0.10": real["oof_spearman"]
        >= max(item["oof_spearman"] for item in controls) + 0.10,
        "at_least_four_positive_oof_folds": real["oof_positive_folds"] >= 4,
        "validation_mean_positive": real["validation_selection"]["mean_benefit"] > 0,
        "validation_wins_at_least_twice_losses": real["validation_selection"]["wins"]
        >= 2 * real["validation_selection"]["losses"],
        "validation_predicate_regressions_zero": real["validation_selection"][
            "predicate_regressions"
        ]
        == 0,
        "validation_oracle_gap_closure_at_least_20pct": real[
            "validation_oracle_gap_closure_pct"
        ]
        >= 20.0,
        "validation_mean_exceeds_fixed_phase": real["validation_selection"][
            "mean_benefit"
        ]
        > phase_validation["mean_benefit"],
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "collector_roots": str(pathlib.Path(args.collector_roots).resolve()),
            "num_roots": len(episodes),
            "train_episodes": train_episodes.tolist(),
            "validation_episodes": validation_episodes.tolist(),
            "folds": args.folds,
            "pca_dim": args.pca_dim,
            "fresh_information_in_features": False,
            "root_or_absolute_step_in_features": False,
            "validation_is_development_only": True,
        },
        "spatial_extraction_seconds": extraction_seconds,
        "validation_oracle_mean_benefit": oracle_mean,
        "phase_train_candidates": phase_candidates,
        "fixed_phase_validation": phase_validation,
        "variants": variant_results,
        "go_checks": go_checks,
        "go": bool(all(go_checks.values())),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
