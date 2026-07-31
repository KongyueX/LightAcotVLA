"""Offline signal probe for a verified Action-CoT refresh gate.

The collector provides multiple exact fresh-vs-stale counterfactual branches
for the same ``(anchor, episode, root)``.  This script uses those matched
branches to ask a narrow question before any closed-loop integration: does the
post-prefix observation contain phase-independent signal about the value of a
fresh ACoT query?

The probe is deliberately leakage-resistant:

* outer folds are held out by episode;
* ImageNet ResNet-18 layer-3 features are frozen and reduced by a PCA fitted on
  the training fold only;
* expected visual and state transitions are fitted on nominal training
  branches only;
* the primary gate sees only visual/state innovation and cached
  EAR/IAR/final-action tails; actual-minus-intended executed prefix is isolated
  in a separately reported telemetry-oracle upper bound;
* branch identifiers, names, strengths, root/control/decision steps, fresh
  plans, and privileged outcomes are never model inputs.

The linear objective combines absolute benefit regression and within-root
pairwise benefit-difference regression.  Four matched variants are evaluated:
real feedback, current equal to anchor, same-root shuffled current feedback,
and a same-phase shuffled cached tail.  Outputs are offline diagnostics only;
no deployable head or evaluation hook is written.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import pathlib
import time
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


PRIMARY_VARIANTS = (
    "real",
    "current_equals_anchor",
    "same_root_shuffled_current",
    "tail_shuffle",
)
TELEMETRY_ORACLE = "telemetry_oracle"
VARIANTS = (*PRIMARY_VARIANTS, TELEMETRY_ORACLE)

FORBIDDEN_MODEL_INPUTS = (
    "branch_id",
    "branch_name",
    "branch_canonical_name",
    "branch_strength",
    "root_index",
    "anchor_decision_step",
    "endpoint_decision_step",
    "anchor_control_step",
    "endpoint_control_step",
    "actual_prefix_env",
    "actual_prefix_executed_env",
    "actual_prefix_executed_valid",
    "fresh_*",
    "stale_*_score",
    "fresh_*_score",
    "advantage_*",
    "fresh_minus_stale_*",
)


@dataclasses.dataclass(frozen=True)
class BranchExample:
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
    executed_prefix: np.ndarray
    benefit_h6: float
    benefit_handoff: float
    predicate_gain_h6: int
    predicate_gain_handoff: int

    @property
    def root_key(self) -> tuple[int, int, int, int]:
        return (self.task_id, self.episode_id, self.anchor_id, self.root_index)

    @property
    def trajectory_key(self) -> tuple[int, int, str]:
        return (self.task_id, self.episode_id, self.branch_canonical_name)


@dataclasses.dataclass(frozen=True)
class LinearModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return ((array - self.mean) / self.scale) @ self.coefficient + self.intercept


@dataclasses.dataclass(frozen=True)
class FoldContext:
    pca_mean: np.ndarray
    pca_basis: np.ndarray
    expected_visual: LinearModel
    expected_state: LinearModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collector-roots",
        "--input",
        nargs="+",
        required=True,
        help="Branch NPZ files, directories, or shell-style glob patterns.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="Optional episode subset; the default uses every compatible episode.",
    )
    parser.add_argument(
        "--benefit-endpoint",
        choices=("h6", "handoff"),
        default="handoff",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=12)
    parser.add_argument("--pca-fit-tokens", type=int, default=65_536)
    parser.add_argument("--expected-transition-ridge", type=float, default=10.0)
    parser.add_argument("--benefit-ridge", type=float, default=10.0)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-root", type=int, default=32)
    parser.add_argument("--iar-pool-bins", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resnet-weights",
        default="default",
        help="'default', 'none', or a local ResNet18 state_dict path.",
    )
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    parser.add_argument("--selection-threshold", type=float, default=0.0)
    parser.add_argument("--go-min-spearman", type=float, default=0.15)
    parser.add_argument("--go-min-pairwise-accuracy", type=float, default=0.60)
    parser.add_argument("--go-min-control-margin", type=float, default=0.05)
    parser.add_argument("--go-min-selections", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.task_id < 0:
        raise ValueError("task-id must be non-negative.")
    if args.episodes is not None and (
        not args.episodes or any(value < 0 for value in args.episodes)
    ):
        raise ValueError("episodes must contain non-negative IDs.")
    if args.outer_folds < 2:
        raise ValueError("outer-folds must be at least two.")
    if not 0 < args.pca_dim <= 256:
        raise ValueError("pca-dim must be in [1, 256].")
    if args.pca_fit_tokens <= 0:
        raise ValueError("pca-fit-tokens must be positive.")
    if args.expected_transition_ridge <= 0 or args.benefit_ridge <= 0:
        raise ValueError("ridge penalties must be positive.")
    if args.pairwise_weight < 0:
        raise ValueError("pairwise-weight must be non-negative.")
    if args.max_pairs_per_root <= 0 or args.iar_pool_bins <= 0:
        raise ValueError("pair and IAR pooling limits must be positive.")
    if args.image_batch_size <= 0 or args.torch_threads <= 0:
        raise ValueError("image-batch-size and torch-threads must be positive.")
    if args.tie_tolerance < 0 or args.go_min_selections <= 0:
        raise ValueError("tie-tolerance and go-min-selections are invalid.")


def _expand_inputs(values: Sequence[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for value in values:
        matches = [pathlib.Path(item) for item in glob.glob(value)]
        if not matches:
            matches = [pathlib.Path(value)]
        for match in matches:
            if match.is_dir():
                paths.extend(sorted(match.rglob("*.npz")))
            elif match.is_file():
                paths.append(match)
            else:
                raise FileNotFoundError(f"Collector input does not exist: {match}")
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No collector NPZ files were found.")
    return unique


def _first_key(
    data: Any,
    names: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    for name in names:
        if name in data.files:
            return name
    if required:
        raise KeyError(
            f"Missing all compatible fields {tuple(names)}; available={sorted(data.files)}"
        )
    return None


def _scalar(
    data: Any,
    names: Sequence[str],
    dtype: type,
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    name = _first_key(data, names, required=required)
    if name is None:
        return default
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"Field {name!r} must be scalar, got {value.shape}.")
    return dtype(value.reshape(()).item())


def _array(
    data: Any,
    names: Sequence[str],
    *,
    dtype: np.dtype | type,
) -> np.ndarray:
    name = _first_key(data, names)
    value = np.asarray(data[name], dtype=dtype)
    if value.size == 0:
        raise ValueError(f"Field {name!r} must be non-empty.")
    if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
        raise ValueError(f"Field {name!r} contains non-finite values.")
    return value


def _benefit(
    data: Any,
    endpoint: str,
    *,
    dtype: type,
) -> Any:
    direct_names = (
        f"fresh_minus_stale_{endpoint}_normalized_score",
        f"advantage_{endpoint}_normalized_score",
    )
    direct = _scalar(data, direct_names, dtype, required=False, default=None)
    if direct is not None:
        return direct
    fresh = _scalar(data, (f"fresh_{endpoint}_normalized_score",), dtype)
    stale = _scalar(data, (f"stale_{endpoint}_normalized_score",), dtype)
    return dtype(fresh - stale)


def _predicate_gain(data: Any, endpoint: str) -> int:
    direct_names = (
        f"fresh_minus_stale_{endpoint}_satisfied_count",
        f"advantage_{endpoint}_satisfied_count",
    )
    direct = _scalar(data, direct_names, int, required=False, default=None)
    if direct is not None:
        return int(direct)
    fresh = _scalar(data, (f"fresh_{endpoint}_satisfied_count",), int)
    stale = _scalar(data, (f"stale_{endpoint}_satisfied_count",), int)
    return int(fresh - stale)


def _executed_prefix(data: Any, intended: np.ndarray) -> np.ndarray:
    executed = _array(
        data,
        ("actual_prefix_executed_env", "executed_prefix_env"),
        dtype=np.float64,
    )
    if executed.shape != intended.shape:
        raise ValueError(
            f"Executed/intended prefix shape mismatch: {executed.shape} vs {intended.shape}."
        )
    valid_name = _first_key(
        data,
        ("actual_prefix_executed_valid", "executed_prefix_valid"),
        required=False,
    )
    if valid_name is None:
        return executed
    valid = np.asarray(data[valid_name], dtype=np.bool_)
    if valid.shape != (intended.shape[0],):
        raise ValueError(f"Field {valid_name!r} has unexpected shape {valid.shape}.")
    # A missing command is represented as zero executed motion.  The validity
    # mask itself is not exposed as a separate model feature.
    return np.where(valid[:, None], executed, 0.0)


def _load_example(path: pathlib.Path, *, task_id: int) -> BranchExample | None:
    with np.load(path, allow_pickle=False) as data:
        valid = _scalar(data, ("valid",), bool, required=False, default=True)
        if not valid:
            return None
        current_task = _scalar(data, ("task_id",), int)
        if current_task != task_id:
            return None
        intended = _array(
            data,
            ("intended_prefix_env", "intended_prefix_actions"),
            dtype=np.float64,
        )
        canonical_name = _scalar(
            data,
            ("branch_canonical_name", "branch_name"),
            str,
        )
        branch_id = _scalar(data, ("branch_id",), int)
        anchor_id = _scalar(
            data,
            ("anchor_id", "initial_state_id"),
            int,
            required=False,
            default=0,
        )
        return BranchExample(
            path=str(path),
            task_id=current_task,
            episode_id=_scalar(data, ("episode_id", "episode_idx", "trial_id"), int),
            anchor_id=int(anchor_id),
            root_index=_scalar(data, ("root_index",), int),
            branch_id=branch_id,
            branch_name=_scalar(data, ("branch_name",), str),
            branch_canonical_name=canonical_name,
            is_nominal=canonical_name == "nominal" or branch_id == 0,
            anchor_images=_array(data, ("anchor_images",), dtype=np.uint8),
            current_images=_array(data, ("current_images",), dtype=np.uint8),
            anchor_state=_array(data, ("anchor_state",), dtype=np.float64),
            current_state=_array(data, ("current_state",), dtype=np.float64),
            cached_ear=_array(
                data,
                ("cached_ear", "cached_coarse_actions"),
                dtype=np.float64,
            ),
            cached_iar=_array(
                data,
                ("cached_iar", "cached_iar_tokens", "cached_implicit_features"),
                dtype=np.float64,
            ),
            cached_final=_array(
                data,
                ("cached_final_actions", "cached_actions"),
                dtype=np.float64,
            ),
            cached_env=_array(
                data,
                ("cached_env_actions", "cached_actions_env"),
                dtype=np.float64,
            ),
            intended_prefix=intended,
            executed_prefix=_executed_prefix(data, intended),
            benefit_h6=float(_benefit(data, "h6", dtype=float)),
            benefit_handoff=float(_benefit(data, "handoff", dtype=float)),
            predicate_gain_h6=_predicate_gain(data, "h6"),
            predicate_gain_handoff=_predicate_gain(data, "handoff"),
        )


def _validate_shapes(examples: Sequence[BranchExample]) -> None:
    fields = (
        "anchor_images",
        "current_images",
        "anchor_state",
        "current_state",
        "cached_ear",
        "cached_iar",
        "cached_final",
        "cached_env",
        "intended_prefix",
        "executed_prefix",
    )
    for field in fields:
        shapes = {getattr(example, field).shape for example in examples}
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent {field} shapes: {sorted(shapes)}")
    image_shape = examples[0].anchor_images.shape
    if len(image_shape) != 4 or image_shape[-1] != 3:
        raise ValueError(f"Expected [camera,height,width,3] images, got {image_shape}.")
    if examples[0].anchor_state.shape != examples[0].current_state.shape:
        raise ValueError("Anchor/current state shapes differ.")
    if examples[0].intended_prefix.shape != examples[0].executed_prefix.shape:
        raise ValueError("Intended/executed prefix shapes differ.")


def _load_examples(
    paths: Sequence[pathlib.Path],
    *,
    task_id: int,
    episodes: Sequence[int] | None,
) -> tuple[list[BranchExample], dict[str, int]]:
    selected_episodes = None if episodes is None else set(episodes)
    examples: list[BranchExample] = []
    skipped_invalid = 0
    skipped_task_or_episode = 0
    seen: dict[tuple[int, int, int, int], str] = {}
    for path in paths:
        example = _load_example(path, task_id=task_id)
        if example is None:
            with np.load(path, allow_pickle=False) as data:
                same_task = _scalar(
                    data,
                    ("task_id",),
                    int,
                    required=False,
                    default=task_id,
                ) == task_id
                valid = _scalar(
                    data,
                    ("valid",),
                    bool,
                    required=False,
                    default=True,
                )
            if same_task and not valid:
                skipped_invalid += 1
            else:
                skipped_task_or_episode += 1
            continue
        if selected_episodes is not None and example.episode_id not in selected_episodes:
            skipped_task_or_episode += 1
            continue
        key = (
            example.task_id,
            example.episode_id,
            example.root_index,
            example.branch_id,
        )
        if key in seen:
            raise ValueError(f"Duplicate branch key {key}: {seen[key]} and {path}.")
        seen[key] = str(path)
        examples.append(example)
    if not examples:
        raise ValueError(f"No valid Task{task_id} branch examples were loaded.")
    _validate_shapes(examples)
    root_counts = collections.Counter(example.root_key for example in examples)
    if max(root_counts.values()) < 2:
        raise ValueError("No root contains multiple valid counterfactual branches.")
    if not any(example.is_nominal for example in examples):
        raise ValueError("No nominal branch is available for transition fitting.")
    return examples, {
        "files_considered": len(paths),
        "valid_examples": len(examples),
        "skipped_invalid": skipped_invalid,
        "skipped_task_or_episode": skipped_task_or_episode,
    }


def _stack(examples: Sequence[BranchExample], field: str) -> np.ndarray:
    return np.stack([getattr(example, field) for example in examples])


def _extract_spatial_maps(
    anchor_images: np.ndarray,
    current_images: np.ndarray,
    *,
    weights_spec: str,
    device_name: str,
    batch_size: int,
    torch_threads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as torch_f
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("This probe requires torch and torchvision.") from exc

    torch.set_num_threads(torch_threads)
    if weights_spec == "default":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        weight_name = str(weights)
    elif weights_spec == "none":
        model = resnet18(weights=None)
        weight_name = "none_random_initialization"
    else:
        model = resnet18(weights=None)
        state = torch.load(weights_spec, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        weight_name = str(pathlib.Path(weights_spec).resolve())
    encoder = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        torch.nn.AdaptiveAvgPool2d((4, 4)),
    ).eval()
    encoder.requires_grad_(False)
    device = torch.device(device_name)
    encoder.to(device)

    images = np.stack((anchor_images, current_images), axis=1)
    count, _, cameras = images.shape[:3]
    flat = images.reshape(-1, *images.shape[-3:])
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.from_numpy(flat[start : start + batch_size]).to(device=device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
            batch = torch_f.interpolate(
                batch,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            batch = (batch - mean) / std
            outputs.append(encoder(batch).cpu().numpy().astype(np.float32, copy=False))
    maps = np.concatenate(outputs, axis=0).reshape(count, 2, cameras, 256, 4, 4)
    metadata = {
        "backbone": "torchvision_resnet18_layer3",
        "weights": weight_name,
        "frozen": True,
        "spatial_shape": [4, 4],
        "channels": 256,
        "cameras": int(cameras),
        "device": str(device),
        "seconds": float(time.perf_counter() - started),
    }
    return maps[:, 0], maps[:, 1], metadata


def _fit_pca(
    anchor_maps: np.ndarray,
    current_maps: np.ndarray,
    indices: np.ndarray,
    *,
    dimension: int,
    max_tokens: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate((anchor_maps[indices], current_maps[indices]), axis=1)
    points = points.transpose(0, 1, 3, 4, 2).reshape(-1, points.shape[2])
    if points.shape[0] > max_tokens:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(points.shape[0], size=max_tokens, replace=False)]
    values = np.asarray(points, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[-dimension:][::-1]
    return mean, eigenvectors[:, order]


def _project_maps(
    maps: np.ndarray,
    pca_mean: np.ndarray,
    pca_basis: np.ndarray,
) -> np.ndarray:
    points = maps.transpose(0, 1, 3, 4, 2).astype(np.float64)
    return (points - pca_mean) @ pca_basis


def _ridge_solve(
    features: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    if features.shape[1] <= features.shape[0]:
        system = features.T @ features + ridge_lambda * np.eye(features.shape[1])
        return np.linalg.solve(system, features.T @ targets)
    system = features @ features.T + ridge_lambda * np.eye(features.shape[0])
    return features.T @ np.linalg.solve(system, targets)


def _fit_linear(
    features: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
) -> LinearModel:
    values = np.asarray(features, dtype=np.float64)
    outputs = np.asarray(targets, dtype=np.float64)
    if outputs.ndim == 1:
        outputs = outputs[:, None]
    mean = values.mean(axis=0)
    raw_scale = values.std(axis=0)
    scale = np.where(raw_scale >= 1e-8, raw_scale, 1.0)
    normalized = (values - mean) / scale
    intercept = outputs.mean(axis=0)
    coefficient = _ridge_solve(normalized, outputs - intercept, ridge_lambda)
    return LinearModel(mean, scale, coefficient, intercept)


def _sequence_summary(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"Expected a non-empty sequence matrix, got {array.shape}.")
    return np.concatenate(
        (array.mean(axis=0), array.std(axis=0), array[0], array[-1])
    )


def _pooled_flat(values: np.ndarray, bins: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    chunks = np.array_split(flat, min(bins, flat.size))
    means = np.asarray([chunk.mean() for chunk in chunks], dtype=np.float64)
    rms = np.asarray(
        [np.sqrt(np.mean(np.square(chunk))) for chunk in chunks],
        dtype=np.float64,
    )
    return np.concatenate((means, rms))


def _cached_plan_features(
    example: BranchExample,
    *,
    iar_pool_bins: int,
) -> np.ndarray:
    prefix_length = example.intended_prefix.shape[0]
    final_tail = example.cached_final[prefix_length:]
    env_tail = example.cached_env[prefix_length:]
    if final_tail.shape[0] == 0 or env_tail.shape[0] == 0:
        raise ValueError(f"Cached action tail is empty in {example.path}.")
    return np.concatenate(
        (
            example.cached_ear.reshape(-1),
            _sequence_summary(example.cached_ear),
            _pooled_flat(example.cached_iar, iar_pool_bins),
            final_tail.reshape(-1),
            _sequence_summary(final_tail),
            env_tail.reshape(-1),
            _sequence_summary(env_tail),
        )
    )


def _transition_static_features(
    examples: Sequence[BranchExample],
    *,
    iar_pool_bins: int,
) -> np.ndarray:
    rows = []
    for example in examples:
        rows.append(
            np.concatenate(
                (
                    example.anchor_state,
                    example.intended_prefix.reshape(-1),
                    _cached_plan_features(example, iar_pool_bins=iar_pool_bins),
                )
            )
        )
    return np.stack(rows)


def _fit_fold_context(
    train_indices: np.ndarray,
    fold_id: int,
    *,
    examples: Sequence[BranchExample],
    anchor_maps: np.ndarray,
    current_maps: np.ndarray,
    anchor_state: np.ndarray,
    current_state: np.ndarray,
    transition_static: np.ndarray,
    pca_dim: int,
    pca_fit_tokens: int,
    expected_transition_ridge: float,
    seed: int,
) -> FoldContext:
    pca_mean, pca_basis = _fit_pca(
        anchor_maps,
        current_maps,
        train_indices,
        dimension=pca_dim,
        max_tokens=pca_fit_tokens,
        seed=seed + fold_id,
    )
    projected_anchor = _project_maps(anchor_maps, pca_mean, pca_basis)
    projected_current = _project_maps(current_maps, pca_mean, pca_basis)
    nominal_indices = np.asarray(
        [index for index in train_indices if examples[index].is_nominal],
        dtype=np.int64,
    )
    if nominal_indices.size < 2:
        raise ValueError(
            f"Fold {fold_id} has only {nominal_indices.size} nominal training branches."
        )
    transition_input = np.concatenate(
        (
            projected_anchor.reshape(len(examples), -1),
            transition_static,
        ),
        axis=1,
    )
    visual_delta = (projected_current - projected_anchor).reshape(len(examples), -1)
    state_delta = current_state - anchor_state
    expected_visual = _fit_linear(
        transition_input[nominal_indices],
        visual_delta[nominal_indices],
        expected_transition_ridge,
    )
    expected_state = _fit_linear(
        transition_input[nominal_indices],
        state_delta[nominal_indices],
        expected_transition_ridge,
    )
    return FoldContext(pca_mean, pca_basis, expected_visual, expected_state)


def _same_root_current_shuffle(
    examples: Sequence[BranchExample],
    indices: np.ndarray,
) -> np.ndarray:
    mapping = np.arange(len(examples), dtype=np.int64)
    groups: dict[tuple[int, int, int, int], list[int]] = collections.defaultdict(list)
    for index in indices:
        groups[examples[index].root_key].append(int(index))
    for members in groups.values():
        ordered = sorted(members, key=lambda index: examples[index].branch_id)
        if len(ordered) > 1:
            donors = ordered[1:] + ordered[:1]
            for target, donor in zip(ordered, donors, strict=True):
                mapping[target] = donor
    return mapping


def _same_phase_tail_shuffle(
    examples: Sequence[BranchExample],
    indices: np.ndarray,
) -> np.ndarray:
    mapping = np.arange(len(examples), dtype=np.int64)
    roots: dict[int, dict[tuple[int, int, int, int], list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index in indices:
        example = examples[index]
        roots[example.root_index][example.root_key].append(int(index))
    for groups in roots.values():
        ordered_keys = sorted(groups)
        if len(ordered_keys) <= 1:
            continue
        donor_keys = ordered_keys[1:] + ordered_keys[:1]
        for target_key, donor_key in zip(ordered_keys, donor_keys, strict=True):
            donor = sorted(groups[donor_key], key=lambda index: examples[index].branch_id)[0]
            for target in groups[target_key]:
                mapping[target] = donor
    return mapping


def _build_features(
    variant: str,
    indices: np.ndarray,
    *,
    examples: Sequence[BranchExample],
    context: FoldContext,
    anchor_maps: np.ndarray,
    current_maps: np.ndarray,
    anchor_state: np.ndarray,
    current_state: np.ndarray,
    transition_static: np.ndarray,
    plan_features: np.ndarray,
    current_shuffle: np.ndarray,
    tail_shuffle: np.ndarray,
    include_prefix_telemetry: bool,
) -> np.ndarray:
    projected_anchor_all = _project_maps(
        anchor_maps,
        context.pca_mean,
        context.pca_basis,
    )
    projected_current_all = _project_maps(
        current_maps,
        context.pca_mean,
        context.pca_basis,
    )
    transition_input = np.concatenate(
        (projected_anchor_all.reshape(len(examples), -1), transition_static),
        axis=1,
    )
    expected_visual = context.expected_visual.predict(transition_input).reshape(
        projected_anchor_all.shape
    )
    expected_state = context.expected_state.predict(transition_input)

    if variant in ("real", "tail_shuffle", TELEMETRY_ORACLE):
        feedback_indices = indices
    elif variant == "current_equals_anchor":
        feedback_indices = indices
    elif variant == "same_root_shuffled_current":
        feedback_indices = current_shuffle[indices]
    else:
        raise ValueError(f"Unknown variant {variant!r}.")

    anchor_visual = projected_anchor_all[indices]
    if variant == "current_equals_anchor":
        selected_visual = anchor_visual
        selected_state = anchor_state[indices]
    else:
        selected_visual = projected_current_all[feedback_indices]
        selected_state = current_state[feedback_indices]
    visual_innovation = selected_visual - (
        anchor_visual + expected_visual[indices]
    )
    state_innovation = selected_state - (
        anchor_state[indices] + expected_state[indices]
    )

    prefix_delta = np.stack(
        [
            examples[index].executed_prefix - examples[index].intended_prefix
            for index in indices
        ]
    )
    selected_plan = plan_features[
        tail_shuffle[indices] if variant == "tail_shuffle" else indices
    ]
    blocks = [
        visual_innovation.reshape(len(indices), -1),
        np.abs(visual_innovation).reshape(len(indices), -1),
        state_innovation,
        np.abs(state_innovation),
        selected_plan,
    ]
    if include_prefix_telemetry:
        blocks.extend(
            (
                prefix_delta.reshape(len(indices), -1),
                np.abs(prefix_delta).reshape(len(indices), -1),
            )
        )
    return np.concatenate(blocks, axis=1)


def _pair_rows(
    normalized: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    *,
    examples: Sequence[BranchExample],
    weight: float,
    max_pairs_per_root: int,
    tie_tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if weight <= 0:
        return (
            np.empty((0, normalized.shape[1]), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
            0,
        )
    local_by_global = {int(global_index): local for local, global_index in enumerate(indices)}
    groups: dict[tuple[int, int, int, int], list[int]] = collections.defaultdict(list)
    for global_index in indices:
        groups[examples[global_index].root_key].append(int(global_index))
    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    rng = np.random.default_rng(seed)
    for members in groups.values():
        pairs = [
            (left, right)
            for position, left in enumerate(members)
            for right in members[position + 1 :]
            if abs(targets[left] - targets[right]) > tie_tolerance
        ]
        if len(pairs) > max_pairs_per_root:
            chosen = rng.choice(len(pairs), size=max_pairs_per_root, replace=False)
            pairs = [pairs[int(index)] for index in chosen]
        for left, right in pairs:
            left_local = local_by_global[left]
            right_local = local_by_global[right]
            feature_rows.append(normalized[left_local] - normalized[right_local])
            target_rows.append(float(targets[left] - targets[right]))
    if not feature_rows:
        return (
            np.empty((0, normalized.shape[1]), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
            0,
        )
    root_weight = np.sqrt(weight)
    return (
        root_weight * np.stack(feature_rows),
        root_weight * np.asarray(target_rows, dtype=np.float64)[:, None],
        len(feature_rows),
    )


def _fit_benefit_model(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    *,
    examples: Sequence[BranchExample],
    ridge_lambda: float,
    pairwise_weight: float,
    max_pairs_per_root: int,
    tie_tolerance: float,
    seed: int,
) -> tuple[LinearModel, dict[str, float | int]]:
    values = np.asarray(features, dtype=np.float64)
    local_targets = targets[indices]
    mean = values.mean(axis=0)
    raw_scale = values.std(axis=0)
    scale = np.where(raw_scale >= 1e-8, raw_scale, 1.0)
    normalized = (values - mean) / scale
    intercept = np.asarray([local_targets.mean()], dtype=np.float64)
    absolute_target = (local_targets - intercept[0])[:, None]
    pair_features, pair_targets, pair_count = _pair_rows(
        normalized,
        targets,
        indices,
        examples=examples,
        weight=pairwise_weight,
        max_pairs_per_root=max_pairs_per_root,
        tie_tolerance=tie_tolerance,
        seed=seed,
    )
    design = np.concatenate((normalized, pair_features), axis=0)
    regression_target = np.concatenate((absolute_target, pair_targets), axis=0)
    coefficient = _ridge_solve(design, regression_target, ridge_lambda)
    model = LinearModel(mean, scale, coefficient, intercept)
    train_prediction = model.predict(values).reshape(-1)
    return model, {
        "absolute_rows": int(len(indices)),
        "pairwise_rows": int(pair_count),
        "absolute_rmse": float(
            np.sqrt(np.mean(np.square(train_prediction - local_targets)))
        ),
    }


def _episode_folds(
    examples: Sequence[BranchExample],
    folds: int,
    seed: int,
) -> list[list[int]]:
    episodes = np.asarray(sorted({example.episode_id for example in examples}), dtype=np.int64)
    if episodes.size < folds:
        raise ValueError(f"Cannot split {episodes.size} episodes into {folds} folds.")
    rng = np.random.default_rng(seed)
    shuffled = episodes[rng.permutation(episodes.size)]
    return [sorted(chunk.tolist()) for chunk in np.array_split(shuffled, folds)]


def _indices_for_episodes(
    examples: Sequence[BranchExample],
    episode_ids: Iterable[int],
) -> np.ndarray:
    selected = set(episode_ids)
    return np.asarray(
        [
            index
            for index, example in enumerate(examples)
            if example.episode_id in selected
        ],
        dtype=np.int64,
    )


def _safe_spearman(prediction: np.ndarray, target: np.ndarray) -> float | None:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.size < 3 or np.unique(target).size < 2:
        return None
    value = float(spearmanr(prediction, target).statistic)
    return value if np.isfinite(value) else None


def _within_root_pairwise_accuracy(
    prediction: np.ndarray,
    target: np.ndarray,
    examples: Sequence[BranchExample],
    *,
    tie_tolerance: float,
) -> dict[str, float | int | None]:
    groups: dict[tuple[int, int, int, int], list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.root_key].append(index)
    correct = 0.0
    count = 0
    for members in groups.values():
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                true_delta = target[left] - target[right]
                if abs(true_delta) <= tie_tolerance:
                    continue
                predicted_delta = prediction[left] - prediction[right]
                count += 1
                if abs(predicted_delta) <= 1e-12:
                    correct += 0.5
                elif np.sign(predicted_delta) == np.sign(true_delta):
                    correct += 1.0
    return {
        "pairs": int(count),
        "accuracy": float(correct / count) if count else None,
    }


def _correlation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    examples: Sequence[BranchExample],
    *,
    tie_tolerance: float,
) -> dict[str, Any]:
    nominal = np.asarray([example.is_nominal for example in examples], dtype=np.bool_)
    disturbed = ~nominal
    result = {
        "overall_spearman": _safe_spearman(prediction, target),
        "nominal_spearman": _safe_spearman(prediction[nominal], target[nominal]),
        "disturbed_spearman": _safe_spearman(
            prediction[disturbed], target[disturbed]
        ),
        "overall_count": int(len(examples)),
        "nominal_count": int(np.sum(nominal)),
        "disturbed_count": int(np.sum(disturbed)),
    }
    result["within_root_pairwise"] = _within_root_pairwise_accuracy(
        prediction,
        target,
        examples,
        tie_tolerance=tie_tolerance,
    )
    return result


def _selection_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    predicate_gain: np.ndarray,
    examples: Sequence[BranchExample],
    *,
    threshold: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    trajectories: dict[tuple[int, int, str], list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        trajectories[example.trajectory_key].append(index)
    selected_values: list[float] = []
    selected_predicates: list[int] = []
    selected_roots: list[int] = []
    oracle_values: list[float] = []
    for members in trajectories.values():
        ordered = sorted(members, key=lambda index: examples[index].root_index)
        selected = next(
            (index for index in ordered if prediction[index] > threshold),
            None,
        )
        if selected is None:
            selected_values.append(0.0)
            selected_predicates.append(0)
            selected_roots.append(-1)
        else:
            selected_values.append(float(target[selected]))
            selected_predicates.append(int(predicate_gain[selected]))
            selected_roots.append(int(examples[selected].root_index))
        oracle_values.append(max(0.0, max(float(target[index]) for index in ordered)))
    values = np.asarray(selected_values, dtype=np.float64)
    predicates = np.asarray(selected_predicates, dtype=np.int64)
    roots = np.asarray(selected_roots, dtype=np.int64)
    selected_mask = roots >= 0
    oracle_mean = float(np.mean(oracle_values))
    mean_benefit = float(np.mean(values))
    return {
        "trajectories": int(len(trajectories)),
        "selections": int(np.sum(selected_mask)),
        "selection_rate": float(np.mean(selected_mask)),
        "threshold": float(threshold),
        "mean_benefit": mean_benefit,
        "selected_mean_benefit": (
            float(np.mean(values[selected_mask])) if np.any(selected_mask) else None
        ),
        "wins": int(np.sum(values > tie_tolerance)),
        "losses": int(np.sum(values < -tie_tolerance)),
        "ties_or_skips": int(np.sum(np.abs(values) <= tie_tolerance)),
        "predicate_regressions": int(np.sum(predicates < 0)),
        "oracle_best_one_mean_benefit": oracle_mean,
        "oracle_gap_closure": (
            float(mean_benefit / oracle_mean) if oracle_mean > 1e-12 else None
        ),
        "selected_root_distribution": {
            str(root): int(count)
            for root, count in sorted(
                collections.Counter(roots[selected_mask].tolist()).items()
            )
        },
    }


def _metric_value(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def _go_no_go(
    variants: dict[str, dict[str, Any]],
    *,
    min_spearman: float,
    min_pairwise_accuracy: float,
    min_control_margin: float,
    min_selections: int,
) -> tuple[dict[str, bool], bool]:
    real = variants["real"]
    controls = [variants[name] for name in PRIMARY_VARIANTS if name != "real"]
    real_spearman = _metric_value(real["metrics"]["overall_spearman"])
    real_disturbed = _metric_value(real["metrics"]["disturbed_spearman"])
    real_pairwise = _metric_value(
        real["metrics"]["within_root_pairwise"]["accuracy"]
    )
    best_control_spearman = max(
        _metric_value(item["metrics"]["overall_spearman"]) for item in controls
    )
    best_control_pairwise = max(
        _metric_value(item["metrics"]["within_root_pairwise"]["accuracy"])
        for item in controls
    )
    selection = real["first_crossing"]
    checks = {
        "overall_spearman": real_spearman >= min_spearman,
        "disturbed_spearman": real_disturbed >= min_spearman,
        "within_root_pairwise_accuracy": real_pairwise >= min_pairwise_accuracy,
        "spearman_beats_best_control": (
            real_spearman >= best_control_spearman + min_control_margin
        ),
        "pairwise_accuracy_beats_best_control": (
            real_pairwise >= best_control_pairwise + min_control_margin
        ),
        "first_crossing_has_enough_selections": (
            selection["selections"] >= min_selections
        ),
        "first_crossing_mean_benefit_positive": selection["mean_benefit"] > 0.0,
        "first_crossing_wins_at_least_twice_losses": (
            selection["wins"] > 0
            and selection["wins"] >= 2 * selection["losses"]
        ),
        "first_crossing_predicate_regressions_zero": (
            selection["predicate_regressions"] == 0
        ),
    }
    return checks, bool(all(checks.values()))


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    output_dir = pathlib.Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "oof_predictions.npz"
    if not args.overwrite and (summary_path.exists() or predictions_path.exists()):
        raise FileExistsError(
            f"Output exists under {output_dir}; pass --overwrite to replace probe outputs."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _expand_inputs(args.collector_roots)
    examples, load_summary = _load_examples(
        input_paths,
        task_id=args.task_id,
        episodes=args.episodes,
    )
    benefit = np.asarray(
        [
            example.benefit_h6
            if args.benefit_endpoint == "h6"
            else example.benefit_handoff
            for example in examples
        ],
        dtype=np.float64,
    )
    predicate_gain = np.asarray(
        [
            example.predicate_gain_h6
            if args.benefit_endpoint == "h6"
            else example.predicate_gain_handoff
            for example in examples
        ],
        dtype=np.int64,
    )
    anchor_state = _stack(examples, "anchor_state")
    current_state = _stack(examples, "current_state")
    anchor_maps, current_maps, spatial_metadata = _extract_spatial_maps(
        _stack(examples, "anchor_images"),
        _stack(examples, "current_images"),
        weights_spec=args.resnet_weights,
        device_name=args.device,
        batch_size=args.image_batch_size,
        torch_threads=args.torch_threads,
    )
    plan_features = np.stack(
        [
            _cached_plan_features(example, iar_pool_bins=args.iar_pool_bins)
            for example in examples
        ]
    )
    transition_static = _transition_static_features(
        examples,
        iar_pool_bins=args.iar_pool_bins,
    )
    folds = _episode_folds(examples, args.outer_folds, args.seed)
    all_indices = np.arange(len(examples), dtype=np.int64)
    predictions = {
        variant: np.full(len(examples), np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    fold_records: list[dict[str, Any]] = []

    for fold_id, test_episodes in enumerate(folds):
        test_indices = _indices_for_episodes(examples, test_episodes)
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        context = _fit_fold_context(
            train_indices,
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
        train_current_shuffle = _same_root_current_shuffle(examples, train_indices)
        test_current_shuffle = _same_root_current_shuffle(examples, test_indices)
        train_tail_shuffle = _same_phase_tail_shuffle(examples, train_indices)
        test_tail_shuffle = _same_phase_tail_shuffle(examples, test_indices)
        fold_record: dict[str, Any] = {
            "fold": fold_id,
            "test_episodes": test_episodes,
            "train_examples": int(train_indices.size),
            "test_examples": int(test_indices.size),
            "nominal_train_examples": int(
                sum(examples[index].is_nominal for index in train_indices)
            ),
            "variants": {},
        }
        primary_train_features = _build_features(
            "real",
            train_indices,
            examples=examples,
            context=context,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=anchor_state,
            current_state=current_state,
            transition_static=transition_static,
            plan_features=plan_features,
            current_shuffle=train_current_shuffle,
            tail_shuffle=train_tail_shuffle,
            include_prefix_telemetry=False,
        )
        primary_model, primary_training_metrics = _fit_benefit_model(
            primary_train_features,
            benefit,
            train_indices,
            examples=examples,
            ridge_lambda=args.benefit_ridge,
            pairwise_weight=args.pairwise_weight,
            max_pairs_per_root=args.max_pairs_per_root,
            tie_tolerance=args.tie_tolerance,
            seed=args.seed + 10_000 * fold_id,
        )
        for variant in PRIMARY_VARIANTS:
            test_features = _build_features(
                variant,
                test_indices,
                examples=examples,
                context=context,
                anchor_maps=anchor_maps,
                current_maps=current_maps,
                anchor_state=anchor_state,
                current_state=current_state,
                transition_static=transition_static,
                plan_features=plan_features,
                current_shuffle=test_current_shuffle,
                tail_shuffle=test_tail_shuffle,
                include_prefix_telemetry=False,
            )
            predictions[variant][test_indices] = primary_model.predict(
                test_features
            ).reshape(-1)
            fold_record["variants"][variant] = {
                "shared_primary_model": True,
                "feature_dim": int(primary_train_features.shape[1]),
            }
        fold_record["primary_training"] = primary_training_metrics

        telemetry_train_features = _build_features(
            TELEMETRY_ORACLE,
            train_indices,
            examples=examples,
            context=context,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=anchor_state,
            current_state=current_state,
            transition_static=transition_static,
            plan_features=plan_features,
            current_shuffle=train_current_shuffle,
            tail_shuffle=train_tail_shuffle,
            include_prefix_telemetry=True,
        )
        telemetry_test_features = _build_features(
            TELEMETRY_ORACLE,
            test_indices,
            examples=examples,
            context=context,
            anchor_maps=anchor_maps,
            current_maps=current_maps,
            anchor_state=anchor_state,
            current_state=current_state,
            transition_static=transition_static,
            plan_features=plan_features,
            current_shuffle=test_current_shuffle,
            tail_shuffle=test_tail_shuffle,
            include_prefix_telemetry=True,
        )
        telemetry_model, telemetry_training_metrics = _fit_benefit_model(
            telemetry_train_features,
            benefit,
            train_indices,
            examples=examples,
            ridge_lambda=args.benefit_ridge,
            pairwise_weight=args.pairwise_weight,
            max_pairs_per_root=args.max_pairs_per_root,
            tie_tolerance=args.tie_tolerance,
            seed=args.seed + 10_000 * fold_id + 1,
        )
        predictions[TELEMETRY_ORACLE][test_indices] = telemetry_model.predict(
            telemetry_test_features
        ).reshape(-1)
        fold_record["variants"][TELEMETRY_ORACLE] = {
            "shared_primary_model": False,
            "feature_dim": int(telemetry_train_features.shape[1]),
            **telemetry_training_metrics,
        }
        fold_records.append(fold_record)

    for variant, values in predictions.items():
        if not np.all(np.isfinite(values)):
            missing = int(np.sum(~np.isfinite(values)))
            raise RuntimeError(f"Variant {variant} has {missing} missing OOF predictions.")

    variant_results: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        values = predictions[variant]
        variant_results[variant] = {
            "metrics": _correlation_metrics(
                values,
                benefit,
                examples,
                tie_tolerance=args.tie_tolerance,
            ),
            "first_crossing": _selection_summary(
                values,
                benefit,
                predicate_gain,
                examples,
                threshold=args.selection_threshold,
                tie_tolerance=args.tie_tolerance,
            ),
        }
    go_checks, go = _go_no_go(
        variant_results,
        min_spearman=args.go_min_spearman,
        min_pairwise_accuracy=args.go_min_pairwise_accuracy,
        min_control_margin=args.go_min_control_margin,
        min_selections=args.go_min_selections,
    )

    root_counts = collections.Counter(example.root_key for example in examples)
    branch_counts = collections.Counter(
        example.branch_canonical_name for example in examples
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "probe_only": True,
        "deployable_checkpoint_saved": False,
        "next_stage_if_go": (
            "fit and serialize the complete preprocessing and head on episodes "
            "0-9,30-39, freeze all choices on 40-49, then run formal episodes "
            "10-29 with zero tuning"
        ),
        "protocol": {
            "task_id": args.task_id,
            "benefit_endpoint": args.benefit_endpoint,
            "outer_split": "episode-held-out",
            "outer_folds": args.outer_folds,
            "fold_episodes": folds,
            "pca_dim": args.pca_dim,
            "pca_fit_scope": "outer-train-only",
            "expected_transition_fit_scope": "outer-train nominal branches only",
            "benefit_loss": (
                "absolute squared ridge + weighted same-root pairwise-difference squared"
            ),
            "benefit_ridge": args.benefit_ridge,
            "expected_transition_ridge": args.expected_transition_ridge,
            "pairwise_weight": args.pairwise_weight,
            "selection": (
                "off-policy root-selection diagnostic: score>threshold first crossing, "
                "at most one per episode/branch across independently restored roots"
            ),
            "first_crossing_is_closed_loop_evidence": False,
            "selection_threshold": args.selection_threshold,
            "primary_model_inputs": [
                "post-prefix visual innovation",
                "post-prefix state innovation",
                "cached EAR",
                "cached IAR",
                "cached final/environment tail",
            ],
            "telemetry_oracle_extra_inputs": [
                "actual-executed-minus-intended prefix",
            ],
            "telemetry_oracle_used_for_go": False,
            "forbidden_primary_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
            "fresh_information_in_model_features": False,
            "branch_or_phase_metadata_in_model_features": False,
        },
        "data": {
            **load_summary,
            "episodes": len({example.episode_id for example in examples}),
            "roots": len(root_counts),
            "branches_per_root_min": int(min(root_counts.values())),
            "branches_per_root_max": int(max(root_counts.values())),
            "branch_counts": {
                name: int(count) for name, count in sorted(branch_counts.items())
            },
            "benefit_mean": float(np.mean(benefit)),
            "benefit_std": float(np.std(benefit)),
            "benefit_positive": int(np.sum(benefit > args.tie_tolerance)),
            "benefit_negative": int(np.sum(benefit < -args.tie_tolerance)),
            "benefit_tie": int(np.sum(np.abs(benefit) <= args.tie_tolerance)),
        },
        "spatial_encoder": spatial_metadata,
        "folds": fold_records,
        "variants": variant_results,
        "go_thresholds": {
            "min_spearman": args.go_min_spearman,
            "min_pairwise_accuracy": args.go_min_pairwise_accuracy,
            "min_control_margin": args.go_min_control_margin,
            "min_selections": args.go_min_selections,
        },
        "go_checks": go_checks,
        "go": go,
    }

    np.savez_compressed(
        predictions_path,
        task_id=np.asarray([example.task_id for example in examples], dtype=np.int16),
        episode_id=np.asarray(
            [example.episode_id for example in examples], dtype=np.int32
        ),
        anchor_id=np.asarray([example.anchor_id for example in examples], dtype=np.int32),
        root_index=np.asarray(
            [example.root_index for example in examples], dtype=np.int32
        ),
        branch_id=np.asarray([example.branch_id for example in examples], dtype=np.uint8),
        branch_canonical_name=np.asarray(
            [example.branch_canonical_name for example in examples]
        ),
        is_nominal=np.asarray([example.is_nominal for example in examples], dtype=np.bool_),
        benefit=benefit.astype(np.float32),
        predicate_gain=predicate_gain.astype(np.int16),
        **{
            f"prediction_{variant}": predictions[variant].astype(np.float32)
            for variant in VARIANTS
        },
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
