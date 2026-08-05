"""Evaluate a frozen Task8 compact alpha router on held-out paired roots.

This script is deliberately offline: it never starts a policy server or edits
the existing LIBERO evaluation path.  It loads ``state_step_ridge_router.npz``
from ``train_task8_alpha_outcome_router.py``, routes each held-out root between
alpha=0 and alpha=.05, and replays the already-collected terminal outcomes.

The per-root hindsight preference exactly follows the trainer's lexicographic
order: terminal success; both-success remaining calls and steps; both-failure
terminal privileged progress; and H20 privileged progress as the final
fallback.  Task8 formal episode IDs 10--29 are refused by default.  Training
episode IDs are loaded from an explicit training summary, supplied directly,
or auto-discovered beside the router, and must not overlap held-out episodes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from openpi.policies.compact_alpha_router import CompactAlphaRouter
from openpi.policies.compact_alpha_router import load_compact_alpha_router


TASK_ID = 8
FORMAL_EPISODES = frozenset(range(10, 30))
ALPHA0 = 0.0
ALTERNATIVE_ALPHA = 0.05


@dataclasses.dataclass(frozen=True)
class Args:
    roots: tuple[str, ...]
    router: str
    output_dir: str
    train_summary: str | None
    train_episode_ids: tuple[int, ...] | None
    expected_alternative_alpha: float = ALTERNATIVE_ALPHA
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


@dataclasses.dataclass(frozen=True)
class Preference:
    label: int
    reason: str
    priority_tier: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        action="append",
        required=True,
        help="Held-out collector directory, roots directory, or root NPZ; repeat as needed.",
    )
    parser.add_argument(
        "--router",
        required=True,
        help="Frozen state_step_ridge_router.npz produced by the outcome-router trainer.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-summary",
        help=(
            "Training summary.json containing data.episode_ids. If neither this nor "
            "--train-episode-ids is set, summary.json is auto-discovered beside --router."
        ),
    )
    parser.add_argument(
        "--train-episode-ids",
        nargs="+",
        type=int,
        help="Exact episode IDs used to train the frozen router.",
    )
    parser.add_argument(
        "--expected-alternative-alpha",
        type=float,
        default=ALTERNATIVE_ALPHA,
    )
    parser.add_argument(
        "--allow-formal-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly allow Task8 formal IDs 10--29; disabled by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _args(namespace: argparse.Namespace) -> Args:
    return Args(
        roots=tuple(namespace.roots),
        router=namespace.router,
        output_dir=namespace.output_dir,
        train_summary=namespace.train_summary,
        train_episode_ids=(
            tuple(namespace.train_episode_ids)
            if namespace.train_episode_ids is not None
            else None
        ),
        expected_alternative_alpha=namespace.expected_alternative_alpha,
        allow_formal_episodes=namespace.allow_formal_episodes,
        overwrite=namespace.overwrite,
    )


def _discover_roots(inputs: Sequence[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for value in inputs:
        candidate = pathlib.Path(value).resolve()
        if candidate.is_file() and candidate.suffix == ".npz":
            paths.append(candidate)
            continue
        if not candidate.is_dir():
            raise FileNotFoundError(f"Held-out root input is not an NPZ or directory: {candidate}")
        direct = sorted(candidate.glob("*.npz"))
        nested = sorted((candidate / "roots").glob("*.npz")) if (candidate / "roots").is_dir() else []
        paths.extend(direct or nested)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No held-out root NPZ files found under {list(inputs)}.")
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
        decision_step = int(_scalar(data, "decision_step"))
        alpha0_absolute_step = int(_scalar(data, "alpha0_terminal_absolute_step"))
        alternative_absolute_step = int(
            _scalar(data, "alternative_terminal_absolute_step")
        )
        return RootExample(
            path=path,
            task_id=int(_scalar(data, "task_id")),
            episode_id=int(_scalar(data, "episode_id")),
            decision_step=decision_step,
            physics_key=str(_scalar(data, "physics_key")),
            alternative_alpha=float(_scalar(data, "alternative_alpha")),
            normalized_state=_vector(data, "normalized_state", 32),
            terminal_evaluated=bool(_scalar(data, "terminal_evaluated")),
            alpha0_success=bool(_scalar(data, "alpha0_terminal_success")),
            alternative_success=bool(_scalar(data, "alternative_terminal_success")),
            # Match the trainer: count the shared initial action-generation call.
            alpha0_remaining_calls=1 + int(_scalar(data, "alpha0_continuation_calls")),
            alternative_remaining_calls=1
            + int(_scalar(data, "alternative_continuation_calls")),
            alpha0_remaining_steps=max(0, alpha0_absolute_step - decision_step),
            alternative_remaining_steps=max(
                0, alternative_absolute_step - decision_step
            ),
            alpha0_terminal_progress=float(
                _scalar(data, "alpha0_terminal_normalized_score")
            ),
            alternative_terminal_progress=float(
                _scalar(data, "alternative_terminal_normalized_score")
            ),
            alpha0_h20_progress=float(_scalar(data, "alpha0_h20_normalized_score")),
            alternative_h20_progress=float(
                _scalar(data, "alternative_h20_normalized_score")
            ),
        )


def _summary_episode_ids(path: pathlib.Path) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Training summary does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = payload["data"]["episode_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Training summary {path} must contain data.episode_ids."
        ) from error
    if not isinstance(values, list) or not values:
        raise ValueError(f"Training summary {path} has no non-empty data.episode_ids list.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"Training summary {path} contains non-integer episode IDs.")
    return tuple(sorted(set(values)))


def _training_episode_ids(
    args: Args, router_path: pathlib.Path
) -> tuple[tuple[int, ...], str, pathlib.Path | None]:
    direct = None
    if args.train_episode_ids is not None:
        if not args.train_episode_ids or any(value < 0 for value in args.train_episode_ids):
            raise ValueError("--train-episode-ids must contain non-negative episode IDs.")
        direct = tuple(sorted(set(args.train_episode_ids)))

    explicit_summary = pathlib.Path(args.train_summary).resolve() if args.train_summary else None
    auto_summary = router_path.parent / "summary.json"
    summary_path = explicit_summary
    if summary_path is None and direct is None and auto_summary.is_file():
        summary_path = auto_summary
    summary_ids = _summary_episode_ids(summary_path) if summary_path is not None else None

    if direct is not None and summary_ids is not None and direct != summary_ids:
        raise ValueError(
            "--train-episode-ids disagree with data.episode_ids in --train-summary: "
            f"direct={list(direct)}, summary={list(summary_ids)}."
        )
    if summary_ids is not None:
        source = "explicit_train_summary" if explicit_summary is not None else "auto_discovered_train_summary"
        return summary_ids, source, summary_path
    if direct is not None:
        return direct, "explicit_train_episode_ids", None
    raise ValueError(
        "Training episode IDs are required to prove held-out separation. Pass "
        "--train-summary or --train-episode-ids, or keep summary.json beside the router."
    )


def _validate(
    args: Args,
    examples: Sequence[RootExample],
    train_episode_ids: Sequence[int],
) -> None:
    if not np.isclose(args.expected_alternative_alpha, ALTERNATIVE_ALPHA, atol=1e-8):
        raise ValueError("The frozen compact router supports only alpha=0 versus alpha=.05.")
    wrong_task = sorted({item.task_id for item in examples if item.task_id != TASK_ID})
    if wrong_task:
        raise ValueError(f"Only zero-based Task8 roots are accepted; found task IDs {wrong_task}.")
    incomplete = [str(item.path) for item in examples if not item.terminal_evaluated]
    if incomplete:
        raise ValueError(
            "Held-out outcome evaluation requires terminal labels; progress-only roots: "
            f"{incomplete[:5]}."
        )
    bad_alpha = [
        (str(item.path), item.alternative_alpha)
        for item in examples
        if not np.isclose(item.alternative_alpha, args.expected_alternative_alpha, atol=1e-7)
    ]
    if bad_alpha:
        raise ValueError(f"Roots contain an unexpected alternative alpha: {bad_alpha[:5]}.")
    if any(item.decision_step < 0 for item in examples):
        raise ValueError("Held-out roots contain a negative decision_step.")
    keys = [(item.episode_id, item.decision_step, item.physics_key) for item in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate episode/decision/physics roots were supplied.")

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
        raise FloatingPointError("Held-out outcome/progress labels contain non-finite values.")

    heldout = {item.episode_id for item in examples}
    train = set(train_episode_ids)
    overlap = sorted(train & heldout)
    if overlap:
        raise ValueError(
            "Training and held-out episode IDs overlap; refusing leaked evaluation: "
            f"{overlap}."
        )
    formal = sorted((train | heldout) & FORMAL_EPISODES)
    if formal and not args.allow_formal_episodes:
        raise ValueError(
            "Task8 formal episode IDs 10-29 are forbidden by default. "
            f"Refusing training/held-out episode IDs {formal}."
        )


def _compare(left: float, right: float, tolerance: float = 1e-8) -> int:
    difference = left - right
    if difference > tolerance:
        return 1
    if difference < -tolerance:
        return -1
    return 0


def _preference(example: RootExample) -> Preference:
    """Return +1 when alpha=.05 wins and -1 when alpha=0 wins."""

    success_difference = int(example.alternative_success) - int(example.alpha0_success)
    if success_difference:
        return Preference(
            success_difference,
            "terminal_rescue" if success_difference > 0 else "terminal_regression",
            1,
        )
    if example.alpha0_success and example.alternative_success:
        call_comparison = _compare(
            -example.alternative_remaining_calls, -example.alpha0_remaining_calls
        )
        if call_comparison:
            return Preference(
                call_comparison,
                "both_success_fewer_calls" if call_comparison > 0 else "both_success_more_calls",
                2,
            )
        step_comparison = _compare(
            -example.alternative_remaining_steps, -example.alpha0_remaining_steps
        )
        if step_comparison:
            return Preference(
                step_comparison,
                "both_success_fewer_steps" if step_comparison > 0 else "both_success_more_steps",
                3,
            )
    elif not example.alpha0_success and not example.alternative_success:
        progress = _compare(
            example.alternative_terminal_progress, example.alpha0_terminal_progress
        )
        if progress:
            return Preference(
                progress,
                "both_fail_terminal_progress_gain" if progress > 0 else "both_fail_terminal_progress_loss",
                4,
            )

    h20_progress = _compare(example.alternative_h20_progress, example.alpha0_h20_progress)
    if h20_progress:
        return Preference(
            h20_progress,
            "h20_progress_gain" if h20_progress > 0 else "h20_progress_loss",
            5,
        )
    return Preference(0, "lexicographic_tie", 6)


def _route(
    router: CompactAlphaRouter, examples: Sequence[RootExample]
) -> tuple[np.ndarray, np.ndarray, float]:
    scores: list[float] = []
    selections: list[bool] = []
    affine_errors: list[float] = []
    for item in examples:
        score, alpha = router.route(item.normalized_state, item.decision_step)
        feature = np.concatenate(
            [item.normalized_state, np.asarray([item.decision_step], dtype=np.float64)]
        )
        affine_score = float(feature @ router.raw_score_kernel + router.raw_score_bias)
        if not np.isclose(score, affine_score, rtol=1e-4, atol=1e-5):
            raise ValueError(
                f"Ridge/affine scores disagree at held-out root {item.path}: "
                f"{score} versus {affine_score}."
            )
        scores.append(score)
        selections.append(bool(np.isclose(alpha, ALTERNATIVE_ALPHA, atol=1e-8)))
        affine_errors.append(abs(score - affine_score))
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(selections, dtype=np.bool_),
        float(max(affine_errors, default=0.0)),
    )


def _arm_values(
    examples: Sequence[RootExample], use_alternative: np.ndarray
) -> dict[str, np.ndarray]:
    choose = np.asarray(use_alternative, dtype=np.bool_)
    return {
        "success": np.where(
            choose,
            [item.alternative_success for item in examples],
            [item.alpha0_success for item in examples],
        ).astype(np.bool_),
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


def _utility(
    examples: Sequence[RootExample], use_alternative: np.ndarray
) -> dict[str, Any]:
    values = _arm_values(examples, use_alternative)
    success_bool = values["success"]
    success = success_bool.astype(np.float64)
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
        "terminal_success_count": int(np.sum(success_bool)),
        "terminal_success_rate": float(np.mean(success_bool)),
        "mean_remaining_calls_on_success": (
            float(np.mean(values["calls"][success_bool])) if np.any(success_bool) else None
        ),
        "mean_remaining_steps_on_success": (
            float(np.mean(values["steps"][success_bool])) if np.any(success_bool) else None
        ),
        "mean_terminal_progress_on_failure": (
            float(np.mean(values["terminal_progress"][~success_bool]))
            if np.any(~success_bool)
            else None
        ),
        "mean_h20_progress": float(np.mean(values["h20_progress"])),
        "alternative_selection_coverage": float(np.mean(use_alternative)),
    }


def _vector_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[float]:
    return [
        float(a - b)
        for a, b in zip(
            left["lexicographic_vector"], right["lexicographic_vector"], strict=True
        )
    ]


def _balanced_accuracy(labels: np.ndarray, selected: np.ndarray) -> float | None:
    recalls: list[float] = []
    for label in (-1, 1):
        mask = labels == label
        if np.any(mask):
            prediction = np.where(selected[mask], 1, -1)
            recalls.append(float(np.mean(prediction == label)))
    return float(np.mean(recalls)) if recalls else None


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(args: Args) -> pathlib.Path:
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(args: Args) -> None:
    router_path = pathlib.Path(args.router).resolve()
    router = load_compact_alpha_router(router_path)
    train_ids, train_id_source, train_summary_path = _training_episode_ids(
        args, router_path
    )
    examples = [_read_root(path) for path in _discover_roots(args.roots)]
    examples.sort(key=lambda item: (item.episode_id, item.decision_step, item.physics_key))
    _validate(args, examples, train_ids)

    preferences = [_preference(item) for item in examples]
    labels = np.asarray([item.label for item in preferences], dtype=np.int8)
    scores, router_selection, max_affine_error = _route(router, examples)
    oracle_selection = labels > 0
    count = len(examples)

    utilities = {
        "fixed_alpha0": _utility(examples, np.zeros(count, dtype=np.bool_)),
        "fixed_alpha05": _utility(examples, np.ones(count, dtype=np.bool_)),
        "frozen_state_step_router": _utility(examples, router_selection),
        "per_root_hindsight_oracle": _utility(examples, oracle_selection),
    }
    fixed0 = utilities["fixed_alpha0"]
    fixed05 = utilities["fixed_alpha05"]
    for name in ("fixed_alpha05", "frozen_state_step_router", "per_root_hindsight_oracle"):
        utilities[name]["vector_delta_vs_fixed_alpha0"] = _vector_delta(
            utilities[name], fixed0
        )
    for name in ("frozen_state_step_router", "per_root_hindsight_oracle"):
        utilities[name]["vector_delta_vs_fixed_alpha05"] = _vector_delta(
            utilities[name], fixed05
        )

    output_dir = _prepare_output(args)
    predictions_path = output_dir / "per_root_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for index, (item, preference) in enumerate(zip(examples, preferences, strict=True)):
            use_alternative = bool(router_selection[index])
            chosen_success = item.alternative_success if use_alternative else item.alpha0_success
            chosen_calls = (
                item.alternative_remaining_calls if use_alternative else item.alpha0_remaining_calls
            )
            chosen_steps = (
                item.alternative_remaining_steps if use_alternative else item.alpha0_remaining_steps
            )
            chosen_progress = (
                item.alternative_terminal_progress
                if use_alternative
                else item.alpha0_terminal_progress
            )
            row = {
                "source_file": str(item.path),
                "task_id": item.task_id,
                "episode_id": item.episode_id,
                "decision_step": item.decision_step,
                "physics_key": item.physics_key,
                "router_score": float(scores[index]),
                "selected_alpha": ALTERNATIVE_ALPHA if use_alternative else ALPHA0,
                "preference_label": preference.label,
                "preference_reason": preference.reason,
                "preference_priority_tier": preference.priority_tier,
                "router_matches_hindsight_on_decisive_root": (
                    bool(use_alternative == bool(preference.label > 0))
                    if preference.label != 0
                    else None
                ),
                "alpha0_terminal_success": item.alpha0_success,
                "alternative_terminal_success": item.alternative_success,
                "selected_terminal_success": chosen_success,
                "selected_remaining_calls": chosen_calls,
                "selected_remaining_steps": chosen_steps,
                "selected_terminal_progress": chosen_progress,
            }
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    decisive = labels != 0
    terminal_discordant = np.asarray(
        [item.alpha0_success != item.alternative_success for item in examples],
        dtype=np.bool_,
    )
    selected_labels = np.where(router_selection, 1, -1)
    summary = {
        "name": "Task8 frozen compact alpha-router held-out replay",
        "status": "offline held-out paired-root replay; not closed-loop episode performance",
        "config": {
            **dataclasses.asdict(args),
            "roots": list(args.roots),
            "train_episode_ids": list(train_ids),
        },
        "separation_guard": {
            "train_episode_id_source": train_id_source,
            "train_summary": str(train_summary_path) if train_summary_path else None,
            "train_episode_ids": list(train_ids),
            "heldout_episode_ids": sorted({item.episode_id for item in examples}),
            "overlap": [],
            "formal_episode_ids_present": sorted(
                (set(train_ids) | {item.episode_id for item in examples}) & FORMAL_EPISODES
            ),
            "formal_episode_override_used": bool(args.allow_formal_episodes),
        },
        "data": {
            "num_roots": count,
            "num_heldout_episodes": len({item.episode_id for item in examples}),
            "preference_counts": {
                "alternative_wins": int(np.sum(labels > 0)),
                "alpha0_wins": int(np.sum(labels < 0)),
                "ties": int(np.sum(labels == 0)),
            },
            "terminal_discordant_roots": int(np.sum(terminal_discordant)),
            "label_priority": [
                "terminal success difference",
                "both-success remaining calls",
                "both-success remaining steps",
                "both-fail terminal privileged progress",
                "H20 privileged progress fallback",
            ],
        },
        "router": {
            "path": str(router_path),
            "sha256": _sha256(router_path),
            "feature": "normalized state[32] + absolute decision_step[1]",
            "ridge_lambda": router.ridge_lambda,
            "training_class": router.training_class,
            "score_semantics": "score>0 selects alpha=.05; otherwise alpha=0",
            "score_min": float(np.min(scores)),
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
            "alternative_selection_count": int(np.sum(router_selection)),
            "alternative_selection_coverage": float(np.mean(router_selection)),
            "decisive_root_accuracy": (
                float(np.mean(selected_labels[decisive] == labels[decisive]))
                if np.any(decisive)
                else None
            ),
            "decisive_root_balanced_accuracy": _balanced_accuracy(labels, router_selection),
            "terminal_discordant_accuracy": (
                float(
                    np.mean(
                        selected_labels[terminal_discordant]
                        == labels[terminal_discordant]
                    )
                )
                if np.any(terminal_discordant)
                else None
            ),
            "max_ridge_affine_score_error": max_affine_error,
        },
        "lexicographic_utility": utilities,
        "hindsight_oracle_guard": (
            "The per-root oracle uses collected outcomes from both arms and is an upper bound, "
            "not deployable performance."
        ),
        "artifacts": {"per_root_predictions": str(predictions_path)},
        "interpretation_guard": (
            "This evaluates a frozen router on unseen paired roots without refitting, but it replays "
            "counterfactual root outcomes rather than measuring closed-loop Task8 success."
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
