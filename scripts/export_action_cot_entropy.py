"""Export offline Action-CoT entropy labels for ACoT-VLA policies.

Debug example:
mkdir -p checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion
ln -sfn /path/to/checkpoint/step checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/latest
python scripts/export_action_cot_entropy.py \
    --config_name acot_libero_action_cot_explicit_implicit_co_fusion \
    --num_samples 4 \
    --segment_mode fixed \
    --chunk_size 5 \
    --prune_ratio 0.3 \
    --max_items 200 \
    --output_dir ./action_cot_entropy_labels

Remote-policy example, with serve_policy.py already running:
python scripts/export_action_cot_entropy.py \
    --config_name acot_libero_action_cot_explicit_implicit_co_fusion \
    --policy_host 127.0.0.1 \
    --policy_port 8000 \
    --num_samples 4 \
    --segment_mode fixed \
    --chunk_size 5 \
    --prune_ratio 0.3 \
    --max_items 200 \
    --output_dir ./action_cot_entropy_labels
"""

import argparse
import csv
import json
import logging
import pathlib
import time
from typing import Any

import jax
import numpy as np

from openpi.action_cot import compression as acot_compression
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


LOGGER = logging.getLogger("export_action_cot_entropy")


def _status(message: str) -> None:
    print(f"[export_action_cot_entropy] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_name",
        "--config-name",
        default="acot_libero_action_cot_explicit_implicit_co_fusion",
        help="Training config name used to build the ACoT-VLA policy.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory. If omitted, config.checkpoint_dir is used.",
    )
    parser.add_argument("--num_samples", "--num-samples", type=int, default=4, help="MC samples K per input.")
    parser.add_argument("--segment_mode", "--segment-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=5)
    parser.add_argument("--min_len", "--min-len", type=int, default=3)
    parser.add_argument("--max_len", "--max-len", type=int, default=8)
    parser.add_argument("--max_segments", "--max-segments", type=int, default=8)
    parser.add_argument("--prune_ratio", "--prune-ratio", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_keep", "--min-keep", type=int, default=1)
    parser.add_argument("--max_skip_ratio", "--max-skip-ratio", type=float, default=0.7)
    parser.add_argument("--max_items", "--max-items", type=int, default=200)
    parser.add_argument("--output_dir", "--output-dir", default="./action_cot_entropy_labels")
    parser.add_argument("--default_prompt", "--default-prompt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy_host", "--policy-host", default=None)
    parser.add_argument("--policy_port", "--policy-port", type=int, default=None)
    parser.add_argument("--policy_api_key", "--policy-api-key", default=None)
    parser.add_argument(
        "--continue_on_error",
        "--continue-on-error",
        action="store_true",
        help="Skip bad dataset items instead of failing immediately with a traceback.",
    )
    parser.add_argument(
        "--dry_run_dataset",
        "--dry-run-dataset",
        action="store_true",
        help="Only construct the policy input dataset and print the first item summary.",
    )
    parser.add_argument("--gripper_indices", "--gripper-indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--offline_fallback_noise_std",
        "--offline-fallback-noise-std",
        type=float,
        default=0.0,
        help=(
            "Offline-analysis fallback only. If >0 and policy samples are identical, add Gaussian noise with this "
            "std to the normalized coarse samples used for entropy calculation. Raw saved coarse_samples are unchanged."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if args.max_items <= 0:
        raise ValueError("--max_items must be positive.")
    if args.offline_fallback_noise_std < 0:
        raise ValueError("--offline_fallback_noise_std must be non-negative.")
    if (args.policy_host is None) != (args.policy_port is None):
        raise ValueError("--policy_host and --policy_port must be provided together.")


def _resolve_checkpoint_dir(train_config: _config.TrainConfig, checkpoint_dir: str | None) -> pathlib.Path:
    if checkpoint_dir is not None and checkpoint_dir.strip():
        return download.maybe_download(checkpoint_dir)

    candidate_dirs = [
        pathlib.Path(train_config.checkpoint_base_dir) / train_config.name / "latest",
        pathlib.Path("checkpoints") / train_config.name / "latest",
        pathlib.Path("/root/autodl-tmp/acotvla/checkpoints") / train_config.name / "latest",
    ]
    seen = set()
    for candidate in candidate_dirs:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            _status(f"Using checkpoint symlink/default path: {candidate}")
            return candidate.resolve()

    try:
        return download.maybe_download(str(train_config.checkpoint_dir))
    except Exception as exc:
        candidates = "\n  ".join(str(path) for path in candidate_dirs)
        raise ValueError(
            "--checkpoint_dir is required when config.checkpoint_dir cannot be resolved. "
            "Alternatively create one of these symlinks:\n  "
            f"{candidates}"
        ) from exc


def _load_norm_stats(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | None,
    data_config: _config.DataConfig,
) -> dict[str, Any] | None:
    if data_config.norm_stats is not None:
        return data_config.norm_stats

    if data_config.asset_id is None or checkpoint_dir is None:
        return None

    try:
        return _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    except FileNotFoundError:
        LOGGER.warning("No normalization stats found in %s or config assets.", checkpoint_dir / "assets")
    except Exception as exc:  # Keep label export usable when stats are absent or stored differently.
        LOGGER.warning("Could not load normalization stats for %s: %s", train_config.name, exc)
    return None


def _fit_last_dim(values: np.ndarray, dim: int, pad_value: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] >= dim:
        return values[..., :dim]

    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (0, dim - values.shape[-1])
    return np.pad(values, pad_width, constant_values=pad_value)


def _normalize_actions(
    actions: np.ndarray,
    norm_stats: dict[str, Any] | None,
    *,
    use_quantiles: bool,
    preferred_key: str,
) -> tuple[np.ndarray, str | None]:
    if norm_stats is None:
        return actions.astype(np.float64), None

    stats_key = preferred_key if preferred_key in norm_stats else "actions" if "actions" in norm_stats else None
    if stats_key is None:
        return actions.astype(np.float64), None

    stats = norm_stats[stats_key]
    action_dim = actions.shape[-1]
    if use_quantiles and stats.q01 is not None and stats.q99 is not None:
        q01 = _fit_last_dim(stats.q01, action_dim, pad_value=0.0)
        q99 = _fit_last_dim(stats.q99, action_dim, pad_value=1.0)
        return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0, stats_key

    mean = _fit_last_dim(stats.mean, action_dim, pad_value=0.0)
    std = _fit_last_dim(stats.std, action_dim, pad_value=1.0)
    return (actions - mean) / (std + 1e-6), stats_key


def _create_policy_dataset(train_config: _config.TrainConfig, data_config: _config.DataConfig) -> _data_loader.Dataset:
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("This exporter currently supports random-access LeRobot-style datasets, not RLDS.")

    base_dataset = _data_loader.create_torch_dataset(data_config, train_config.model)
    return _data_loader.TransformedDataset(base_dataset, [*data_config.repack_transforms.inputs])


def _summarize_value(value: Any) -> str:
    try:
        array = np.asarray(value)
        return f"shape={array.shape}, dtype={array.dtype}"
    except Exception:
        return type(value).__name__


def _print_dataset_summary(dataset: _data_loader.Dataset, max_items: int) -> None:
    _status(f"Dataset length: {len(dataset)}; requested max_items: {max_items}")
    if len(dataset) == 0:
        return
    item = dataset[0]
    if item is None:
        _status("First dataset item is None")
        return
    _status("First dataset item:")
    for key in sorted(item):
        _status(f"  {key}: {_summarize_value(item[key])}")


def _create_policy(
    args: argparse.Namespace,
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | None,
    norm_stats: dict[str, Any] | None,
):
    if args.policy_host is not None:
        from openpi_client import websocket_client_policy as _websocket_client_policy

        _status(f"Connecting to remote policy at {args.policy_host}:{args.policy_port}")
        return _websocket_client_policy.WebsocketClientPolicy(
            args.policy_host,
            args.policy_port,
            api_key=args.policy_api_key,
            ping_interval=None,
            ping_timeout=None,
        )

    if checkpoint_dir is None:
        raise ValueError("--checkpoint_dir is required unless --policy_host/--policy_port are provided.")

    _status(f"Loading local policy from {checkpoint_dir}")
    return _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
    )


def _set_policy_seed(policy: Any, seed: int) -> None:
    if hasattr(policy, "_rng"):
        setattr(policy, "_rng", jax.random.key(seed))


def _sample_id(data: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "frame_index", "episode_index", "index"):
        if key not in data:
            continue
        value = np.asarray(data[key])
        if value.shape == ():
            return str(value.item())
    return str(index)


def _stack_output(samples: list[np.ndarray], key: str, sample_id: str) -> np.ndarray:
    try:
        return np.stack(samples, axis=0)
    except ValueError as exc:
        shapes = [sample.shape for sample in samples]
        raise ValueError(f"Cannot stack {key} for sample {sample_id}; observed shapes: {shapes}") from exc


def _collect_policy_samples(
    policy: Any,
    data: dict[str, Any],
    *,
    item_index: int,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coarse_samples = []
    action_samples = []
    timing_ms = []
    wall_start = time.perf_counter()

    for sample_index in range(num_samples):
        _set_policy_seed(policy, seed + item_index * num_samples + sample_index)
        _status(f"  policy.infer sample {sample_index + 1}/{num_samples}")
        result = policy.infer(data)
        if "coarse_actions" not in result:
            raise KeyError("Policy output does not contain 'coarse_actions'. Is this an explicit Action-CoT policy?")
        if "actions" not in result:
            raise KeyError("Policy output does not contain 'actions'.")

        coarse_samples.append(np.asarray(result["coarse_actions"], dtype=np.float32))
        action_samples.append(np.asarray(result["actions"], dtype=np.float32))
        policy_timing = result.get("policy_timing", {})
        timing_ms.append(float(policy_timing.get("infer_ms", np.nan)))

    wall_time_ms = (time.perf_counter() - wall_start) * 1000.0
    return (
        _stack_output(coarse_samples, "coarse_actions", str(item_index)),
        _stack_output(action_samples, "actions", str(item_index)),
        np.asarray(timing_ms, dtype=np.float32),
        wall_time_ms,
    )


def _segment(mean_coarse_normalized: np.ndarray, args: argparse.Namespace) -> list[acot_compression.Segment]:
    if args.segment_mode == "fixed":
        return acot_compression.segment_fixed(mean_coarse_normalized, chunk_size=args.chunk_size)

    return acot_compression.segment_adaptive(
        mean_coarse_normalized,
        min_len=args.min_len,
        max_len=args.max_len,
        max_segments=args.max_segments,
        gripper_indices=args.gripper_indices,
    )


def _entropy_samples(
    coarse_samples_normalized: np.ndarray,
    *,
    item_index: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool]:
    if not np.allclose(coarse_samples_normalized, coarse_samples_normalized[:1], rtol=1e-5, atol=1e-6):
        return coarse_samples_normalized, False

    if args.offline_fallback_noise_std <= 0:
        LOGGER.warning(
            "Sample %s produced identical coarse_actions across MC runs. Entropy will be constant unless stochastic "
            "sampling is enabled or --offline_fallback_noise_std is set.",
            item_index,
        )
        return coarse_samples_normalized, False

    rng = np.random.default_rng(args.seed + item_index)
    noise = rng.normal(
        loc=0.0,
        scale=args.offline_fallback_noise_std,
        size=coarse_samples_normalized.shape,
    )
    return coarse_samples_normalized + noise, True


def _write_metadata(args: argparse.Namespace, checkpoint_dir: pathlib.Path | None, output_dir: pathlib.Path) -> None:
    metadata = {
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "num_samples": args.num_samples,
        "segment_mode": args.segment_mode,
        "chunk_size": args.chunk_size,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "max_segments": args.max_segments,
        "prune_ratio": args.prune_ratio,
        "threshold": args.threshold,
        "min_keep": args.min_keep,
        "max_skip_ratio": args.max_skip_ratio,
        "max_items": args.max_items,
        "offline_fallback_noise_std": args.offline_fallback_noise_std,
        "fallback_note": "Fallback noise is for offline analysis only and does not modify saved raw coarse_samples.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _summary_row(sample_id: str, entropy: np.ndarray, skip_mask: np.ndarray) -> dict[str, str | int | float]:
    return {
        "sample_id": sample_id,
        "num_segments": int(entropy.shape[0]),
        "entropy_min": float(np.min(entropy)) if entropy.size else float("nan"),
        "entropy_max": float(np.max(entropy)) if entropy.size else float("nan"),
        "entropy_std": float(np.std(entropy)) if entropy.size else float("nan"),
        "skip_ratio": float(np.mean(skip_mask)) if skip_mask.size else 0.0,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = _parse_args()
    _validate_args(args)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _status(f"Loading config: {args.config_name}")
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        checkpoint_dir = _resolve_checkpoint_dir(train_config, args.checkpoint_dir)
    elif args.policy_host is None and not args.dry_run_dataset:
        checkpoint_dir = _resolve_checkpoint_dir(train_config, args.checkpoint_dir)
    norm_stats = _load_norm_stats(train_config, checkpoint_dir, data_config)

    _status("Creating policy input dataset")
    dataset = _create_policy_dataset(train_config, data_config)
    total_items = min(args.max_items, len(dataset))
    _status(f"Dataset ready: len={len(dataset)}, export_items={total_items}")
    if args.dry_run_dataset:
        _print_dataset_summary(dataset, args.max_items)
        return

    policy = _create_policy(args, train_config, checkpoint_dir, norm_stats)
    _write_metadata(args, checkpoint_dir, output_dir)

    summary_path = output_dir / "summary.csv"
    summary_fields = ["sample_id", "num_segments", "entropy_min", "entropy_max", "entropy_std", "skip_ratio"]
    processed = 0

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()

        for item_index in range(total_items):
            try:
                _status(f"Processing item {item_index + 1}/{total_items}")
                data = dataset[item_index]
                if data is None:
                    _status(f"Skipping item {item_index}: dataset returned None")
                    continue
                sample_id = _sample_id(data, item_index)
                _status(f"Collecting {args.num_samples} policy samples for sample_id={sample_id}")
                coarse_samples, actions_full, timing_ms, wall_time_ms = _collect_policy_samples(
                    policy,
                    data,
                    item_index=item_index,
                    num_samples=args.num_samples,
                    seed=args.seed,
                )

                coarse_mean = np.mean(coarse_samples, axis=0)
                coarse_samples_normalized, norm_key = _normalize_actions(
                    coarse_samples,
                    norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                    preferred_key="coarse_actions",
                )
                coarse_mean_normalized = np.mean(coarse_samples_normalized, axis=0)
                entropy_input, used_fallback = _entropy_samples(
                    coarse_samples_normalized,
                    item_index=item_index,
                    args=args,
                )

                segments = _segment(coarse_mean_normalized, args)
                entropy = acot_compression.compute_mc_predictive_entropy(entropy_input, segments)
                skip_mask = acot_compression.make_skip_mask(
                    entropy,
                    prune_ratio=args.prune_ratio,
                    threshold=args.threshold,
                    min_keep=args.min_keep,
                    max_skip_ratio=args.max_skip_ratio,
                )
                frame_skip_mask = acot_compression.expand_segment_mask(
                    skip_mask,
                    segments,
                    t_len=coarse_samples.shape[1],
                )

                output_path = output_dir / f"sample_{item_index:06d}.npz"
                np.savez_compressed(
                    output_path,
                    coarse_samples=coarse_samples,
                    coarse_mean=coarse_mean,
                    actions_full=actions_full,
                    segments=np.asarray(segments, dtype=np.int32),
                    entropy=entropy,
                    skip_mask=skip_mask.astype(np.int8),
                    frame_skip_mask=frame_skip_mask.astype(np.int8),
                    timing=timing_ms,
                    wall_time_ms=np.asarray(wall_time_ms, dtype=np.float32),
                    sample_id=np.asarray(sample_id),
                    normalization_key=np.asarray(norm_key or ""),
                    used_offline_fallback_noise=np.asarray(used_fallback),
                    offline_fallback_noise_std=np.asarray(args.offline_fallback_noise_std, dtype=np.float32),
                )
                writer.writerow(_summary_row(sample_id, entropy, skip_mask))
                processed += 1
                _status(
                    f"Wrote {output_path.name}: coarse_samples={coarse_samples.shape}, "
                    f"num_segments={len(segments)}, skip_ratio={float(np.mean(skip_mask)):.3f}"
                )

                if processed % 10 == 0:
                    LOGGER.info("Processed %s/%s items", processed, total_items)
            except Exception as exc:
                LOGGER.exception("Failed to export item %s", item_index)
                if not args.continue_on_error:
                    raise
                _status(f"Skipping item {item_index} due to export error: {exc}")

    if processed == 0:
        raise RuntimeError("No samples were exported. Check the dataset, checkpoint, and policy coarse_actions output.")

    LOGGER.info("Wrote %s samples to %s", processed, output_dir)
    LOGGER.info("CSV summary: %s", summary_path)


if __name__ == "__main__":
    main()
