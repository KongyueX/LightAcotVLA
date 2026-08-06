"""Fit a tiny outcome-ranked router over four contextual fusion candidates.

The input is produced by ``collect_task8_contextual_fusion_candidates.py``.
Only deployment-available quantities are used as features: current normalized
state, absolute decision progress, EAR, compiler/expert chunks, their
disagreement, and the action-derived phase boundary.  Terminal outcome and
privileged task progress are used only to create pairwise training labels.

Episodes 30-39 are the default training partition and 40-49 are a strict
held-out validation partition.  The model is a four-head linear ranker fit with
pairwise logistic loss and L2 regularization selected by grouped CV over the
training episodes.  This deliberately small capacity is appropriate for the
limited same-root data and exports a portable ``router_params.npz`` sidecar.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


CANDIDATE_NAMES = ("compiler", "expert", "compiler_to_expert", "expert_to_compiler")
NUM_CANDIDATES = len(CANDIDATE_NAMES)


@dataclasses.dataclass(frozen=True)
class RootExample:
    path: pathlib.Path
    episode_id: int
    features: np.ndarray
    success: np.ndarray
    calls: np.ndarray
    absolute_step: np.ndarray
    progress: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-episode-start", type=int, default=30)
    parser.add_argument("--train-episode-end", type=int, default=40)
    parser.add_argument("--validation-episode-start", type=int, default=40)
    parser.add_argument("--validation-episode-end", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--l2-candidates", nargs="+", type=float, default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.train_episode_start < args.train_episode_end:
        raise ValueError("Training episode range must be non-empty.")
    if not args.validation_episode_start < args.validation_episode_end:
        raise ValueError("Validation episode range must be non-empty.")
    train_ids = set(range(args.train_episode_start, args.train_episode_end))
    validation_ids = set(range(args.validation_episode_start, args.validation_episode_end))
    if train_ids & validation_ids:
        raise ValueError("Training and validation episode ranges must be disjoint.")
    if args.cv_folds < 2 or args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("CV folds, steps, and learning rate must be positive.")
    if not args.l2_candidates or any(value < 0 for value in args.l2_candidates):
        raise ValueError("L2 candidates must be non-negative.")


def _group_rms(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values[:, start:end]), axis=-1) + 1e-8)


def _band_energy(actions: np.ndarray) -> np.ndarray:
    coefficients = np.fft.rfft(actions, axis=0)
    energy = np.square(np.abs(coefficients))
    # rFFT for T=10 has six bins.  Preserve coarse low/mid/high separation.
    return np.asarray(
        (
            np.mean(energy[0:2]),
            np.mean(energy[2:4]),
            np.mean(energy[4:6]),
        ),
        dtype=np.float32,
    )


def _features(data: Any) -> np.ndarray:
    compiler = np.asarray(data["compiler_actions"], dtype=np.float32)[:, :7]
    expert = np.asarray(data["expert_actions"], dtype=np.float32)[:, :7]
    coarse = np.asarray(data["coarse_actions"], dtype=np.float32)[:10, :7]
    state = np.asarray(data["normalized_state"], dtype=np.float32).reshape(-1)[:8]
    difference = expert - compiler
    boundary = int(np.asarray(data["phase_boundary"]).item())
    decision_step = int(np.asarray(data["decision_step"]).item())
    shared_speed = np.asarray(data["shared_speed"], dtype=np.float32)
    residual_jump = np.asarray(data["residual_jump"], dtype=np.float32)
    boundary_one_hot = np.zeros((7,), dtype=np.float32)
    boundary_one_hot[np.clip(boundary, 2, 8) - 2] = 1.0

    parts: list[np.ndarray] = [
        np.asarray((decision_step / 1000.0, boundary / 10.0), dtype=np.float32),
        boundary_one_hot,
        shared_speed,
        residual_jump,
        _group_rms(difference, 0, 3),
        _group_rms(difference, 3, 6),
        np.abs(difference[:, 6]),
        (np.signbit(expert[:, 6]) != np.signbit(compiler[:, 6])).astype(np.float32),
        compiler[:, 6],
        expert[:, 6],
        state,
    ]
    for actions in (compiler, expert):
        parts.extend(
            (
                np.linalg.norm(actions[:, :3], axis=-1).astype(np.float32),
                np.linalg.norm(actions[:, 3:6], axis=-1).astype(np.float32),
                _group_rms(actions - coarse, 0, 3),
                _group_rms(actions - coarse, 3, 6),
                _band_energy(actions[:, :3]),
                _band_energy(actions[:, 3:6]),
                np.asarray(
                    (
                        np.sqrt(np.mean(np.square(np.diff(actions[:, :3], axis=0)))),
                        np.sqrt(np.mean(np.square(np.diff(actions[:, 3:6], axis=0)))),
                        np.sum(np.signbit(actions[1:, 6]) != np.signbit(actions[:-1, 6])),
                    ),
                    dtype=np.float32,
                ),
            )
        )
    for start, end in ((0, boundary), (boundary, 10)):
        segment = difference[start:end]
        parts.append(
            np.asarray(
                (
                    np.sqrt(np.mean(np.square(segment[:, :3]))),
                    np.sqrt(np.mean(np.square(segment[:, 3:6]))),
                    np.sqrt(np.mean(np.square(segment[:, 6]))),
                ),
                dtype=np.float32,
            )
        )
    feature = np.concatenate([value.reshape(-1) for value in parts]).astype(np.float32)
    if not np.all(np.isfinite(feature)):
        raise ValueError("Router feature contains non-finite values.")
    return feature


def _load_examples(dataset: pathlib.Path) -> list[RootExample]:
    paths = sorted((dataset / "roots").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No root NPZ files found under {dataset / 'roots'}.")
    examples: list[RootExample] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            names = tuple(str(value) for value in data["candidate_names"].tolist())
            if names != CANDIDATE_NAMES:
                raise ValueError(f"Unexpected candidate order in {path}: {names}.")
            examples.append(
                RootExample(
                    path=path,
                    episode_id=int(np.asarray(data["episode_id"]).item()),
                    features=_features(data),
                    success=np.asarray(data["candidate_terminal_success"], dtype=np.bool_),
                    calls=np.asarray(data["candidate_continuation_calls"], dtype=np.int32),
                    absolute_step=np.asarray(
                        data["candidate_terminal_absolute_step"], dtype=np.int32
                    ),
                    progress=np.asarray(
                        data["candidate_terminal_normalized_score"], dtype=np.float32
                    ),
                )
            )
    widths = {example.features.shape for example in examples}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent router feature widths: {sorted(widths)}.")
    return examples


def _compare(example: RootExample, left: int, right: int) -> int:
    left_success = bool(example.success[left])
    right_success = bool(example.success[right])
    if left_success != right_success:
        return 1 if left_success else -1
    if left_success:
        if example.calls[left] != example.calls[right]:
            return 1 if example.calls[left] < example.calls[right] else -1
        if example.absolute_step[left] != example.absolute_step[right]:
            return 1 if example.absolute_step[left] < example.absolute_step[right] else -1
    delta = float(example.progress[left] - example.progress[right])
    if abs(delta) > 1e-6:
        return 1 if delta > 0 else -1
    return 0


def _statistics(examples: list[RootExample]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([example.features for example in examples])
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std = np.where(std > 1e-5, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _pair_matrix(
    examples: list[RootExample], mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[float] = []
    width = len(mean)
    for example in examples:
        feature = (example.features - mean) / std
        for left in range(NUM_CANDIDATES):
            for right in range(left + 1, NUM_CANDIDATES):
                comparison = _compare(example, left, right)
                if comparison == 0:
                    continue
                row = np.zeros((width * NUM_CANDIDATES + NUM_CANDIDATES,), dtype=np.float32)
                row[left * width : (left + 1) * width] = feature
                row[right * width : (right + 1) * width] = -feature
                row[width * NUM_CANDIDATES + left] = 1.0
                row[width * NUM_CANDIDATES + right] = -1.0
                rows.append(row)
                labels.append(float(comparison))
    if not rows:
        raise ValueError("No non-tied candidate pairs are available for training.")
    return np.stack(rows), np.asarray(labels, dtype=np.float32)


def _fit(
    examples: list[RootExample],
    *,
    l2: float,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mean, std = _statistics(examples)
    matrix, labels = _pair_matrix(examples, mean, std)
    rng = np.random.default_rng(seed)
    parameters = rng.normal(0.0, 1e-3, size=(matrix.shape[1],)).astype(np.float32)
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    for step in range(1, steps + 1):
        margins = labels * (matrix @ parameters)
        inverse = 1.0 / (1.0 + np.exp(np.clip(margins, -40.0, 40.0)))
        gradient = -np.mean((labels * inverse)[:, None] * matrix, axis=0) + l2 * parameters
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * np.square(gradient)
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        parameters -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
    margins = labels * (matrix @ parameters)
    loss = float(
        np.mean(np.logaddexp(0.0, -margins))
        + 0.5 * l2 * np.sum(np.square(parameters))
    )
    width = len(mean)
    weights = parameters[: width * NUM_CANDIDATES].reshape(NUM_CANDIDATES, width).T
    bias = parameters[width * NUM_CANDIDATES :]
    return mean, std, weights, bias, loss


def _scores(
    example: RootExample,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return ((example.features - mean) / std) @ weights + bias


def _metrics(
    examples: list[RootExample],
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    selected: list[int] = []
    pair_correct = 0
    pair_total = 0
    for example in examples:
        score = _scores(example, mean, std, weights, bias)
        selected.append(int(np.argmax(score)))
        for left in range(NUM_CANDIDATES):
            for right in range(left + 1, NUM_CANDIDATES):
                comparison = _compare(example, left, right)
                if comparison == 0:
                    continue
                pair_total += 1
                pair_correct += int((score[left] - score[right]) * comparison > 0)
    selected_success = np.asarray(
        [example.success[index] for example, index in zip(examples, selected, strict=True)],
        dtype=np.float32,
    )
    static_success = np.stack([example.success for example in examples]).astype(np.float32)
    discordant = np.ptp(static_success, axis=1) > 0
    return {
        "roots": len(examples),
        "episodes": sorted({example.episode_id for example in examples}),
        "pairwise_accuracy": pair_correct / pair_total if pair_total else None,
        "pairwise_pairs": pair_total,
        "selected_terminal_success_rate": float(np.mean(selected_success)),
        "selected_terminal_success_rate_discordant": (
            float(np.mean(selected_success[discordant])) if np.any(discordant) else None
        ),
        "outcome_discordant_roots": int(np.sum(discordant)),
        "candidate_oracle_terminal_success_rate": float(np.mean(np.max(static_success, axis=1))),
        "static_terminal_success_rate": {
            name: float(np.mean(static_success[:, index]))
            for index, name in enumerate(CANDIDATE_NAMES)
        },
        "selected_candidate_counts": {
            name: int(sum(index == candidate for index in selected))
            for candidate, name in enumerate(CANDIDATE_NAMES)
        },
    }


def _grouped_folds(examples: list[RootExample], folds: int) -> list[tuple[list[RootExample], list[RootExample]]]:
    episode_ids = sorted({example.episode_id for example in examples})
    if len(episode_ids) < folds:
        raise ValueError(f"Need at least {folds} training episodes, found {len(episode_ids)}.")
    result = []
    for fold in range(folds):
        validation_ids = set(episode_ids[fold::folds])
        result.append(
            (
                [example for example in examples if example.episode_id not in validation_ids],
                [example for example in examples if example.episode_id in validation_ids],
            )
        )
    return result


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    examples = _load_examples(pathlib.Path(args.dataset))
    train = [
        example
        for example in examples
        if args.train_episode_start <= example.episode_id < args.train_episode_end
    ]
    validation = [
        example
        for example in examples
        if args.validation_episode_start <= example.episode_id < args.validation_episode_end
    ]
    if not train or not validation:
        raise ValueError(
            f"Empty train/validation split: train={len(train)}, validation={len(validation)}."
        )

    cv: dict[str, Any] = {}
    best_l2 = None
    best_key = None
    for l2 in args.l2_candidates:
        fold_metrics = []
        for fold, (fold_train, fold_validation) in enumerate(
            _grouped_folds(train, args.cv_folds)
        ):
            mean, std, weights, bias, loss = _fit(
                fold_train,
                l2=l2,
                steps=args.steps,
                learning_rate=args.learning_rate,
                seed=args.seed + fold,
            )
            metrics = _metrics(fold_validation, mean, std, weights, bias)
            metrics["train_loss"] = loss
            fold_metrics.append(metrics)
        discordant_rates = [
            metric["selected_terminal_success_rate_discordant"]
            for metric in fold_metrics
            if metric["selected_terminal_success_rate_discordant"] is not None
        ]
        pairwise = [metric["pairwise_accuracy"] for metric in fold_metrics if metric["pairwise_accuracy"] is not None]
        selection_key = (
            float(np.mean(discordant_rates)) if discordant_rates else -1.0,
            float(np.mean(pairwise)) if pairwise else -1.0,
            -float(l2),
        )
        cv[str(l2)] = {
            "selection_key": selection_key,
            "folds": fold_metrics,
        }
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_l2 = float(l2)

    assert best_l2 is not None
    mean, std, weights, bias, loss = _fit(
        train,
        l2=best_l2,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    train_metrics = _metrics(train, mean, std, weights, bias)
    validation_metrics = _metrics(validation, mean, std, weights, bias)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "router_params.npz"
    np.savez_compressed(
        params_path,
        feature_mean=mean,
        feature_std=std,
        weights=weights,
        bias=bias,
        candidate_names=np.asarray(CANDIDATE_NAMES),
        selected_l2=np.asarray(best_l2, dtype=np.float32),
    )
    summary = {
        "status": "complete",
        "args": vars(args),
        "dataset": str(pathlib.Path(args.dataset).resolve()),
        "feature_dim": int(len(mean)),
        "candidate_names": list(CANDIDATE_NAMES),
        "selected_l2": best_l2,
        "train_loss": loss,
        "cv": cv,
        "train": train_metrics,
        "validation": validation_metrics,
        "artifacts": {"router_params": str(params_path.resolve())},
        "leakage_contract": {
            "deployment_features": [
                "current normalized state",
                "absolute decision progress",
                "current EAR",
                "current compiler and expert action chunks",
                "action-derived phase boundary and disagreement",
            ],
            "train_only_labels": [
                "terminal success",
                "remaining calls and steps",
                "privileged terminal task progress",
            ],
            "episode_id_is_model_input": False,
        },
    }
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    (output_dir / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main(build_parser().parse_args())
