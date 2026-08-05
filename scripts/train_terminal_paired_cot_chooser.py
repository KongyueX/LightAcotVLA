"""Probe a post-query cached-vs-fresh Action-CoT chooser.

The exact refresh request has already happened when this chooser runs.  Its
shared candidate scorer therefore compares two available plans and treats the
refresh RPC as a common sunk cost.  Branch metadata, injected actions, absolute
steps, seeds, and privileged outcomes are never model inputs.

Episodes 0-9 and 30-39 form the episode-grouped five-fold OOF probe.  Episodes
40-49 are reported only as reused development diagnostics; they are not an
untouched test set and cannot by themselves make this probe a go.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax


TRAIN_EPISODES = (*range(0, 10), *range(30, 40))
REUSED_DEV_EPISODES = tuple(range(40, 50))
FORMAL_EPISODES = tuple(range(10, 30))
VARIANTS = ("real", "current_anchor", "fresh_shuffle", "cached_equals_fresh")


@dataclasses.dataclass(frozen=True)
class Example:
    path: str
    episode_id: int
    branch_name: str
    anchor_images: np.ndarray
    current_images: np.ndarray
    anchor_state: np.ndarray
    current_state: np.ndarray
    cached_ear: np.ndarray
    fresh_ear: np.ndarray
    cached_iar: np.ndarray
    fresh_iar: np.ndarray
    cached_final: np.ndarray
    fresh_final: np.ndarray
    cached_env: np.ndarray
    fresh_env: np.ndarray
    stale_success: bool
    fresh_success: bool
    success_delta: int
    progress_delta: float
    stale_remaining_calls: int
    fresh_remaining_calls: int
    stale_remaining_steps: int
    fresh_remaining_steps: int


@dataclasses.dataclass(frozen=True)
class Preprocessor:
    iar_mean: np.ndarray
    iar_basis: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray


@dataclasses.dataclass(frozen=True)
class FoldModel:
    preprocessor: Preprocessor
    params: Any
    score_mean: float
    score_scale: float
    train_episodes: tuple[int, ...]
    test_episodes: tuple[int, ...]


def _scalar(data: Any, key: str, cast: Any) -> Any:
    return cast(np.asarray(data[key]).reshape(()).item())


def _array(data: Any, key: str, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(data[key], dtype=dtype)


def _load_examples(root: pathlib.Path) -> list[Example]:
    paths = sorted(root.rglob("*.npz"))
    examples: list[Example] = []
    seen: set[tuple[int, str]] = set()
    allowed = set(TRAIN_EPISODES) | set(REUSED_DEV_EPISODES)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if not _scalar(data, "valid", bool):
                continue
            episode = _scalar(data, "episode_id", int)
            if episode not in allowed:
                continue
            branch = _scalar(data, "branch_canonical_name", str)
            key = (episode, branch)
            if key in seen:
                raise ValueError(f"Duplicate episode/branch pair: {key}")
            seen.add(key)
            success_delta = _scalar(data, "terminal_success_difference", int)
            stale_success = _scalar(data, "stale_terminal_success", bool)
            fresh_success = _scalar(data, "fresh_terminal_success", bool)
            if success_delta != int(fresh_success) - int(stale_success):
                raise ValueError(f"Inconsistent success label in {path}")
            examples.append(
                Example(
                    path=str(path),
                    episode_id=episode,
                    branch_name=branch,
                    anchor_images=_array(data, "anchor_images", np.uint8),
                    current_images=_array(data, "current_images", np.uint8),
                    anchor_state=_array(data, "anchor_state"),
                    current_state=_array(data, "current_state"),
                    cached_ear=_array(data, "cached_ear"),
                    fresh_ear=_array(data, "fresh_ear"),
                    cached_iar=_array(data, "cached_iar"),
                    fresh_iar=_array(data, "fresh_iar"),
                    cached_final=_array(data, "cached_final_actions"),
                    fresh_final=_array(data, "fresh_final_actions"),
                    cached_env=_array(data, "cached_env_actions"),
                    fresh_env=_array(data, "fresh_env_actions"),
                    stale_success=stale_success,
                    fresh_success=fresh_success,
                    success_delta=success_delta,
                    progress_delta=_scalar(
                        data, "terminal_privileged_progress_difference", float
                    ),
                    stale_remaining_calls=_scalar(
                        data, "stale_remaining_policy_calls", int
                    ),
                    fresh_remaining_calls=_scalar(
                        data, "fresh_remaining_policy_calls", int
                    ),
                    stale_remaining_steps=_scalar(
                        data, "stale_remaining_steps_after_h6", int
                    ),
                    fresh_remaining_steps=_scalar(
                        data, "fresh_remaining_steps_after_h6", int
                    ),
                )
            )
    episodes = {item.episode_id for item in examples}
    missing = sorted((set(TRAIN_EPISODES) | set(REUSED_DEV_EPISODES)) - episodes)
    if missing:
        raise ValueError(f"Missing requested episodes: {missing}")
    if len(examples) != 150:
        raise ValueError(f"Expected 150 valid examples, found {len(examples)}")
    return examples


def _preference(example: Example) -> tuple[float, float, str]:
    if example.success_delta > 0:
        return 1.0, 4.0, "rescue"
    if example.success_delta < 0:
        return -1.0, 8.0, "regression"
    if example.stale_success and example.fresh_success:
        # The exact refresh RPC has already happened for both candidate choices.
        # Compare only post-query continuation cost.
        if example.fresh_remaining_calls != example.stale_remaining_calls:
            return (
                float(np.sign(example.stale_remaining_calls - example.fresh_remaining_calls)),
                2.0,
                "both_success_remaining_calls",
            )
        if example.fresh_remaining_steps != example.stale_remaining_steps:
            return (
                float(np.sign(example.stale_remaining_steps - example.fresh_remaining_steps)),
                1.5,
                "both_success_remaining_steps",
            )
        return 0.0, 0.5, "both_success_tie"
    if example.progress_delta > 1e-6:
        return 1.0, 1.0, "both_fail_progress_gain"
    if example.progress_delta < -1e-6:
        return -1.0, 1.0, "both_fail_progress_loss"
    return 0.0, 0.5, "both_fail_tie"


def _context_features(example: Example, *, anchor_current: bool) -> np.ndarray:
    anchor = example.anchor_images.astype(np.float32) / 255.0
    current = anchor if anchor_current else example.current_images.astype(np.float32) / 255.0
    anchor_state = example.anchor_state.astype(np.float32)
    current_state = anchor_state if anchor_current else example.current_state.astype(np.float32)
    image_delta = current - anchor
    image_blocks = []
    for values in (anchor, current, image_delta, np.abs(image_delta)):
        image_blocks.extend(
            [values.mean(axis=(1, 2)).reshape(-1), values.std(axis=(1, 2)).reshape(-1)]
        )
    state_delta = current_state - anchor_state
    return np.concatenate(
        [*image_blocks, anchor_state, current_state, state_delta, np.abs(state_delta)]
    ).astype(np.float32)


def _aligned_plan(example: Example, arm: str) -> tuple[np.ndarray, ...]:
    if arm == "cached":
        # H4 corresponds to phase two for the stride-two explicit Action-CoT.
        ear = example.cached_ear[2:15]
        iar = example.cached_iar
        final = example.cached_final[4:10]
        env = example.cached_env[4:10]
    elif arm == "fresh":
        ear = example.fresh_ear[:13]
        iar = example.fresh_iar
        final = example.fresh_final[:6]
        env = example.fresh_env[:6]
    else:
        raise ValueError(arm)
    if ear.shape != (13, 32) or final.shape != (6, 32) or env.shape != (6, 7):
        raise ValueError(f"Unexpected aligned candidate shapes: {ear.shape}, {final.shape}, {env.shape}")
    return ear, iar, final, env


def _fit_iar_pca(examples: Sequence[Example], indices: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for index in indices:
        example = examples[int(index)]
        rows.extend([example.cached_iar.mean(axis=0), example.fresh_iar.mean(axis=0)])
    values = np.asarray(rows, dtype=np.float64)
    mean = values.mean(axis=0)
    _, _, right = np.linalg.svd(values - mean, full_matrices=False)
    return mean.astype(np.float32), right[: min(dim, right.shape[0])].T.astype(np.float32)


def _candidate_features(
    example: Example,
    arm: str,
    *,
    iar_mean: np.ndarray,
    iar_basis: np.ndarray,
    anchor_current: bool = False,
) -> np.ndarray:
    context = _context_features(example, anchor_current=anchor_current)
    ear, iar, final, env = _aligned_plan(example, arm)
    iar_summary = iar.mean(axis=0)
    iar_projected = (iar_summary - iar_mean) @ iar_basis
    positions = np.linspace(-1.0, 1.0, ear.shape[0], dtype=np.float32)[:, None]
    ear_blocks = [ear.mean(0), ear.std(0), ear[0], ear[-1], (ear * positions).mean(0)]
    boundary = env[0] - example.cached_env[3]
    smooth = np.diff(env, axis=0)
    plan = np.concatenate(
        [
            *ear_blocks,
            iar_projected,
            np.asarray([iar.std(), np.linalg.norm(iar_summary)], dtype=np.float32),
            final.reshape(-1),
            env.reshape(-1),
            boundary,
            np.asarray(
                [
                    np.mean(np.linalg.norm(smooth[:, :6], axis=1)),
                    np.max(np.linalg.norm(smooth[:, :6], axis=1)),
                    np.mean(np.sign(env[1:, 6]) != np.sign(env[:-1, 6])),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    return np.concatenate([context, plan]).astype(np.float32)


def _fit_preprocessor(
    examples: Sequence[Example], indices: np.ndarray, *, iar_pca_dim: int
) -> Preprocessor:
    iar_mean, iar_basis = _fit_iar_pca(examples, indices, iar_pca_dim)
    features = []
    for index in indices:
        example = examples[int(index)]
        for arm in ("cached", "fresh"):
            features.append(
                _candidate_features(
                    example, arm, iar_mean=iar_mean, iar_basis=iar_basis
                )
            )
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-5] = 1.0
    return Preprocessor(iar_mean, iar_basis, mean, scale)


def _features_for_indices(
    examples: Sequence[Example],
    indices: np.ndarray,
    preprocessor: Preprocessor,
    *,
    variant: str,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cached_rows = []
    fresh_rows = []
    rng = np.random.default_rng(shuffle_seed)
    donor_by_index: dict[int, int] = {}
    if variant == "fresh_shuffle":
        for index in indices:
            item = examples[int(index)]
            donors = [
                int(candidate)
                for candidate in indices
                if examples[int(candidate)].episode_id != item.episode_id
                and examples[int(candidate)].branch_name == item.branch_name
            ]
            donor_by_index[int(index)] = int(rng.choice(donors))
    for index in indices:
        item = examples[int(index)]
        anchor_current = variant == "current_anchor"
        cached = _candidate_features(
            item,
            "cached",
            iar_mean=preprocessor.iar_mean,
            iar_basis=preprocessor.iar_basis,
            anchor_current=anchor_current,
        )
        if variant == "cached_equals_fresh":
            fresh = cached.copy()
        elif variant == "fresh_shuffle":
            donor = examples[donor_by_index[int(index)]]
            # Keep the target current context but substitute only the fresh plan.
            target_context = _context_features(item, anchor_current=False)
            donor_feature = _candidate_features(
                donor,
                "fresh",
                iar_mean=preprocessor.iar_mean,
                iar_basis=preprocessor.iar_basis,
            )
            fresh = np.concatenate([target_context, donor_feature[target_context.size :]])
        else:
            fresh = _candidate_features(
                item,
                "fresh",
                iar_mean=preprocessor.iar_mean,
                iar_basis=preprocessor.iar_basis,
                anchor_current=anchor_current,
            )
        cached_rows.append((cached - preprocessor.feature_mean) / preprocessor.feature_scale)
        fresh_rows.append((fresh - preprocessor.feature_mean) / preprocessor.feature_scale)
    return np.asarray(cached_rows, np.float32), np.asarray(fresh_rows, np.float32)


def _init_params(key: jax.Array, feature_dim: int, hidden_dim: int) -> dict[str, jax.Array]:
    first, second = jax.random.split(key)
    return {
        "w1": jax.random.normal(first, (feature_dim, hidden_dim)) * np.sqrt(2.0 / feature_dim),
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "w2": jax.random.normal(second, (hidden_dim,)) * np.sqrt(1.0 / hidden_dim),
        "b2": jnp.zeros((), dtype=jnp.float32),
    }


def _score(params: dict[str, jax.Array], features: jax.Array) -> jax.Array:
    hidden = jax.nn.tanh(features @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def _train_model(
    cached: np.ndarray,
    fresh: np.ndarray,
    stale_success: np.ndarray,
    fresh_success: np.ndarray,
    preference: np.ndarray,
    weight: np.ndarray,
    *,
    hidden_dim: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> Any:
    x_cached = jnp.asarray(cached)
    x_fresh = jnp.asarray(fresh)
    s_success = jnp.asarray(stale_success, dtype=jnp.float32)
    f_success = jnp.asarray(fresh_success, dtype=jnp.float32)
    target = jnp.asarray(preference, dtype=jnp.float32)
    sample_weight = jnp.asarray(weight, dtype=jnp.float32)
    params = _init_params(jax.random.key(seed), cached.shape[1], hidden_dim)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, weight_decay=1e-4))
    state = optimizer.init(params)

    def loss_fn(model_params: Any) -> jax.Array:
        stale_logit = _score(model_params, x_cached)
        fresh_logit = _score(model_params, x_fresh)
        delta = fresh_logit - stale_logit
        non_tie = target != 0
        pair = jnp.where(
            non_tie,
            jax.nn.softplus(-target * delta),
            0.25 * jnp.square(delta),
        )
        pair_loss = jnp.sum(sample_weight * pair) / jnp.sum(sample_weight)
        arm_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(stale_logit, s_success)
            + optax.sigmoid_binary_cross_entropy(fresh_logit, f_success)
        )
        l2 = sum(jnp.sum(jnp.square(value)) for value in jax.tree_util.tree_leaves(model_params))
        return pair_loss + 0.5 * arm_loss + 1e-5 * l2

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(steps):
        _, gradients = value_and_grad(params)
        updates, state = optimizer.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
    return params


def _episode_folds(episodes: Sequence[int], folds: int, seed: int) -> list[np.ndarray]:
    values = np.asarray(sorted(set(episodes)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return [np.asarray(part, dtype=np.int32) for part in np.array_split(values, folds)]


def _indices(examples: Sequence[Example], episode_ids: Sequence[int]) -> np.ndarray:
    selected = set(int(value) for value in episode_ids)
    return np.asarray(
        [index for index, item in enumerate(examples) if item.episode_id in selected],
        dtype=np.int32,
    )


def _model_scores(
    model: FoldModel,
    examples: Sequence[Example],
    indices: np.ndarray,
    *,
    variant: str,
    seed: int,
) -> np.ndarray:
    cached, fresh = _features_for_indices(
        examples, indices, model.preprocessor, variant=variant, shuffle_seed=seed
    )
    raw = np.asarray(
        jax.device_get(_score(model.params, jnp.asarray(fresh)) - _score(model.params, jnp.asarray(cached))),
        dtype=np.float64,
    )
    return (raw - model.score_mean) / model.score_scale


def _selection_summary(
    examples: Sequence[Example], indices: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    chosen = [examples[int(index)] for index in indices]
    mask = np.asarray(selected, dtype=np.bool_)
    success_delta = np.asarray([item.success_delta for item in chosen], dtype=np.int32)
    stale_success = np.asarray([item.stale_success for item in chosen], dtype=np.bool_)
    fresh_success = np.asarray([item.fresh_success for item in chosen], dtype=np.bool_)
    stale_calls = np.asarray([item.stale_remaining_calls for item in chosen], dtype=np.int32)
    fresh_calls = np.asarray([item.fresh_remaining_calls for item in chosen], dtype=np.int32)
    policy_success = np.where(mask, fresh_success, stale_success)
    policy_calls = 1 + np.where(mask, fresh_calls, stale_calls)
    rescue = mask & (success_delta > 0)
    regression = mask & (success_delta < 0)
    branch_rows = {}
    for branch in sorted({item.branch_name for item in chosen}):
        members = np.asarray([item.branch_name == branch for item in chosen])
        branch_rows[branch] = {
            "selected": int(np.sum(mask & members)),
            "rescues": int(np.sum(rescue & members)),
            "regressions": int(np.sum(regression & members)),
        }
    return {
        "examples": len(chosen),
        "selected": int(np.sum(mask)),
        "rescues": int(np.sum(rescue)),
        "regressions": int(np.sum(regression)),
        "net_successes": int(np.sum(success_delta * mask)),
        "terminal_successes": int(np.sum(policy_success)),
        "always_stale_successes": int(np.sum(stale_success)),
        "always_fresh_successes": int(np.sum(fresh_success)),
        "oracle_successes": int(np.sum(stale_success | fresh_success)),
        "rescue_episodes": len({item.episode_id for item, flag in zip(chosen, rescue, strict=True) if flag}),
        "regression_episodes": len(
            {item.episode_id for item, flag in zip(chosen, regression, strict=True) if flag}
        ),
        "rescue_branches": len(
            {item.branch_name for item, flag in zip(chosen, rescue, strict=True) if flag}
        ),
        "post_query_remaining_calls_plus_common_query": int(np.sum(policy_calls)),
        "post_query_always_stale_calls": int(np.sum(1 + stale_calls)),
        "post_query_always_fresh_calls": int(np.sum(1 + fresh_calls)),
        "no_query_stale_remaining_calls": int(np.sum(stale_calls)),
        "by_branch": branch_rows,
    }


def _choose_threshold(
    scores: np.ndarray,
    examples: Sequence[Example],
    indices: np.ndarray,
    *,
    min_rate: float = 0.1,
    max_rate: float = 0.4,
) -> tuple[float, dict[str, Any]]:
    rows = []
    for threshold in np.unique(scores):
        selected = scores > threshold
        rate = float(np.mean(selected))
        if not min_rate <= rate <= max_rate:
            continue
        summary = _selection_summary(examples, indices, selected)
        rank = (
            summary["regression_episodes"] == 0,
            -summary["regression_episodes"],
            summary["rescue_episodes"],
            summary["rescues"],
            summary["net_successes"],
            -summary["selected"],
        )
        rows.append((rank, float(threshold), summary))
    if not rows:
        raise ValueError("No OOF threshold satisfies the 10%-40% selection interval")
    _, threshold, summary = max(rows, key=lambda item: item[0])
    return threshold, summary


def _branch_only_summary(
    examples: Sequence[Example], train_indices: np.ndarray, test_indices: np.ndarray
) -> dict[str, Any]:
    branches = sorted({examples[int(index)].branch_name for index in train_indices})
    means = {
        branch: float(
            np.mean(
                [
                    examples[int(index)].success_delta
                    for index in train_indices
                    if examples[int(index)].branch_name == branch
                ]
            )
        )
        for branch in branches
    }
    selected = np.asarray(
        [means[examples[int(index)].branch_name] > 0 for index in test_indices],
        dtype=np.bool_,
    )
    return {"train_branch_mean_success_delta": means, **_selection_summary(examples, test_indices, selected)}


def _write_predictions(
    path: pathlib.Path,
    examples: Sequence[Example],
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    threshold: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "episode_id",
            "branch_name",
            "success_delta",
            "preference",
            *[f"score_{name}" for name in VARIANTS],
            "selected_real",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for local, index in enumerate(indices):
            example = examples[int(index)]
            preference, _, _ = _preference(example)
            writer.writerow(
                {
                    "episode_id": example.episode_id,
                    "branch_name": example.branch_name,
                    "success_delta": example.success_delta,
                    "preference": preference,
                    **{f"score_{name}": predictions[name][local] for name in VARIANTS},
                    "selected_real": int(predictions["real"][local] > threshold),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--iar-pca-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.outer_folds < 2 or args.hidden_dim <= 0 or args.train_steps <= 0:
        raise ValueError("Invalid training configuration")
    examples = _load_examples(args.collector_root)
    train_indices = _indices(examples, TRAIN_EPISODES)
    dev_indices = _indices(examples, REUSED_DEV_EPISODES)
    preferences = np.asarray([_preference(item)[0] for item in examples], np.float32)
    weights = np.asarray([_preference(item)[1] for item in examples], np.float32)
    stale_success = np.asarray([item.stale_success for item in examples], np.float32)
    fresh_success = np.asarray([item.fresh_success for item in examples], np.float32)
    folds = _episode_folds(TRAIN_EPISODES, args.outer_folds, args.seed)
    oof = {name: np.full(len(examples), np.nan, dtype=np.float64) for name in VARIANTS}
    dev_fold_scores = {name: [] for name in VARIANTS}
    fold_models: list[FoldModel] = []
    fold_records = []
    for fold_id, test_episodes in enumerate(folds):
        test = _indices(examples, test_episodes)
        fit_episodes = sorted(set(TRAIN_EPISODES) - set(test_episodes.tolist()))
        fit = _indices(examples, fit_episodes)
        preprocessor = _fit_preprocessor(examples, fit, iar_pca_dim=args.iar_pca_dim)
        cached_fit, fresh_fit = _features_for_indices(
            examples, fit, preprocessor, variant="real", shuffle_seed=args.seed + fold_id
        )
        params = _train_model(
            cached_fit,
            fresh_fit,
            stale_success[fit],
            fresh_success[fit],
            preferences[fit],
            weights[fit],
            hidden_dim=args.hidden_dim,
            steps=args.train_steps,
            learning_rate=args.learning_rate,
            seed=args.seed + 100 * fold_id,
        )
        raw_fit = np.asarray(
            jax.device_get(_score(params, jnp.asarray(fresh_fit)) - _score(params, jnp.asarray(cached_fit)))
        )
        score_scale = max(float(np.std(raw_fit)), 1e-6)
        model = FoldModel(
            preprocessor,
            params,
            float(np.mean(raw_fit)),
            score_scale,
            tuple(fit_episodes),
            tuple(int(value) for value in test_episodes),
        )
        fold_models.append(model)
        for variant in VARIANTS:
            oof[variant][test] = _model_scores(
                model,
                examples,
                test,
                variant=variant,
                seed=args.seed + 1000 * fold_id,
            )
            dev_fold_scores[variant].append(
                _model_scores(
                    model,
                    examples,
                    dev_indices,
                    variant=variant,
                    seed=args.seed + 2000 * fold_id,
                )
            )
        fold_records.append(
            {
                "fold": fold_id,
                "train_episodes": fit_episodes,
                "test_episodes": test_episodes.tolist(),
                "feature_dim": int(preprocessor.feature_mean.size),
                "parameter_count": int(
                    sum(np.size(value) for value in jax.tree_util.tree_leaves(params))
                ),
            }
        )
    if any(np.any(~np.isfinite(oof[name][train_indices])) for name in VARIANTS):
        raise RuntimeError("OOF predictions are incomplete")
    threshold, threshold_train_summary = _choose_threshold(
        oof["real"][train_indices], examples, train_indices
    )
    train_results = {
        name: _selection_summary(
            examples, train_indices, oof[name][train_indices] > threshold
        )
        for name in VARIANTS
    }
    dev_predictions = {
        name: np.mean(np.stack(dev_fold_scores[name]), axis=0) for name in VARIANTS
    }
    dev_results = {
        name: _selection_summary(
            examples, dev_indices, dev_predictions[name] > threshold
        )
        for name in VARIANTS
    }
    train_branch_only = _branch_only_summary(examples, train_indices, train_indices)
    dev_branch_only = _branch_only_summary(examples, train_indices, dev_indices)
    real = train_results["real"]
    best_baseline_success = max(
        real["always_stale_successes"],
        real["always_fresh_successes"],
        train_branch_only["terminal_successes"],
    )
    oracle_gap = real["oracle_successes"] - best_baseline_success
    gap_closed = (
        (real["terminal_successes"] - best_baseline_success) / oracle_gap
        if oracle_gap > 0
        else 0.0
    )
    go_checks = {
        "oof_beats_strongest_static_or_branch_baseline_by_two": real[
            "terminal_successes"
        ]
        >= best_baseline_success + 2,
        "oof_closes_at_least_half_oracle_gap": gap_closed >= 0.5,
        "oof_zero_selected_regressions": real["regressions"] == 0,
        "oof_rescues_three_episodes": real["rescue_episodes"] >= 3,
        "oof_rescues_two_branches": real["rescue_branches"] >= 2,
        "oof_real_net_exceeds_fresh_shuffle": real["net_successes"]
        > train_results["fresh_shuffle"]["net_successes"],
        "oof_real_net_exceeds_current_anchor": real["net_successes"]
        > train_results["current_anchor"]["net_successes"],
    }
    summary = {
        "status": "complete",
        "go": bool(all(go_checks.values())),
        "go_checks": go_checks,
        "protocol": {
            "decision_timing": "post-query; exact fresh RPC is common sunk cost",
            "train_episodes": list(TRAIN_EPISODES),
            "reused_development_episodes": list(REUSED_DEV_EPISODES),
            "formal_episodes_forbidden": list(FORMAL_EPISODES),
            "unit_of_independence": "episode",
            "shared_candidate_scorer": True,
            "preference_logit": "V(current,fresh)-V(current,cached)",
            "cached_alignment": "EAR phase2, final/env action indices 4:10",
            "fresh_alignment": "EAR phase0, final/env action indices 0:6",
            "forbidden_inputs": [
                "branch metadata",
                "root/absolute step",
                "seed",
                "actual injected/executed prefix",
                "terminal outcome/progress/cost",
            ],
            "current_state_is_legal": (
                "yes, because the exact fresh RPC has already returned before this chooser"
            ),
            "reused_dev_is_untouched_test": False,
        },
        "data": {"examples": len(examples), "train": len(train_indices), "reused_dev": len(dev_indices)},
        "model": {
            "hidden_dim": args.hidden_dim,
            "iar_pca_dim": args.iar_pca_dim,
            "train_steps": args.train_steps,
            "folds": fold_records,
        },
        "threshold": threshold,
        "threshold_train_summary": threshold_train_summary,
        "train_oof": {"variants": train_results, "branch_only": train_branch_only},
        "reused_dev_diagnostic": {"variants": dev_results, "branch_only": dev_branch_only},
        "oof_strongest_baseline_successes": best_baseline_success,
        "oof_oracle_gap_closed": gap_closed,
        "next_step": (
            "freeze and collect prospective seed107 multi-root data" if all(go_checks.values())
            else "stop paired chooser; do not add capacity or collect more data"
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_predictions(
        args.output_dir / "train_oof_predictions.csv",
        examples,
        train_indices,
        {name: oof[name][train_indices] for name in VARIANTS},
        threshold,
    )
    _write_predictions(
        args.output_dir / "reused_dev_predictions.csv",
        examples,
        dev_indices,
        dev_predictions,
        threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
