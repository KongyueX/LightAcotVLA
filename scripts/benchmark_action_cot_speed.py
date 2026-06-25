"""Benchmark ACoT inference latency with and without explicit Action-CoT generation.

This script measures the speed budget that Stage B/Stage C can actually target:

1. full_acot:
   normal policy inference; runs explicit Action-CoT generation and final action head.
2. cached_coarse_override:
   reuses full coarse_actions as an override; skips the explicit Action-CoT
   generation loop and runs the final action head.
3. pruned_coarse_override:
   reuses a pruned/interpolated coarse trajectory as an override; this is the
   current Stage B injection path. It skips the whole explicit Action-CoT
   generation loop, but it does not yet reduce the final action-head token length.

If cached_coarse_override is faster than full_acot, explicit Action-CoT generation
has a real latency budget. If pruned_coarse_override is not faster than
cached_coarse_override, then segment-level pruning still needs a model path that
generates or consumes fewer Action-CoT tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import time
from typing import Any

import numpy as np

import eval_action_cot_pruning as stage_b
from openpi.action_cot import compression as acot_compression
from openpi.training import config as _config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy_dir", "--entropy-dir", required=True)
    parser.add_argument(
        "--policy.config",
        "--policy-config",
        "--config_name",
        "--config-name",
        dest="config_name",
        default=stage_b.DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--policy.dir",
        "--policy-dir",
        "--checkpoint_dir",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        required=True,
    )
    parser.add_argument("--default_prompt", "--default-prompt", default=None)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    parser.add_argument("--max_items", "--max-items", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--num_steps", "--num-steps", type=int, default=10)

    parser.add_argument("--strategy", choices=("low_entropy", "high_entropy", "random"), default="low_entropy")
    parser.add_argument("--segment_mode", "--segment-mode", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=5)
    parser.add_argument("--prune_ratio", "--prune-ratio", type=float, default=0.3)
    parser.add_argument("--replacement", choices=("interp", "hold", "zero"), default="interp")
    parser.add_argument("--min_keep_segments", "--min-keep-segments", type=int, default=1)
    parser.add_argument("--min_len", "--min-len", type=int, default=3)
    parser.add_argument("--max_len", "--max-len", type=int, default=6)
    parser.add_argument("--max_segments", "--max-segments", type=int, default=5)
    parser.add_argument("--gripper_indices", "--gripper-indices", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_items <= 0:
        raise ValueError("--max_items must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num_steps must be positive.")
    if not 0.0 <= args.prune_ratio <= 1.0:
        raise ValueError("--prune_ratio must be in [0, 1].")


def _status(message: str) -> None:
    print(f"[benchmark_action_cot_speed] {message}", flush=True)


def _infer_timed(policy: Any, policy_input: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], float, float]:
    stage_b._set_policy_seed(policy, seed)
    start = time.perf_counter()
    result = policy.infer(policy_input)
    wall_ms = (time.perf_counter() - start) * 1000.0
    policy_timing = result.get("policy_timing", {}) if isinstance(result, dict) else {}
    infer_ms = float(policy_timing.get("infer_ms", wall_ms))
    return result, infer_ms, wall_ms


def _make_pruned_coarse(
    sample: dict[str, Any],
    coarse_full: np.ndarray,
    *,
    args: argparse.Namespace,
    norm_stats: dict[str, Any] | None,
    data_config: _config.DataConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    skip_mask = stage_b._select_skip_mask(
        sample["entropy"],
        strategy=args.strategy,
        prune_ratio=args.prune_ratio,
        min_keep_segments=args.min_keep_segments,
        rng=rng,
    )
    frame_skip_mask = acot_compression.expand_segment_mask(
        skip_mask,
        sample["segments"],
        t_len=coarse_full.shape[0],
    )
    coarse_zero = stage_b._zero_action_value(
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        preferred_key="coarse_actions",
        dim=coarse_full.shape[-1],
    )
    coarse_pruned = stage_b._replace_masked_frames(
        coarse_full,
        frame_skip_mask,
        replacement=args.replacement,
        zero_value=coarse_zero,
    )
    return coarse_pruned, skip_mask, float(np.mean(frame_skip_mask)) if frame_skip_mask.size else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator


def _write_outputs(output_dir: pathlib.Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "latency_rows.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "item_index",
                "repeat",
                "full_ms",
                "cached_override_ms",
                "pruned_override_ms",
                "full_wall_ms",
                "cached_override_wall_ms",
                "pruned_override_wall_ms",
                "skip_ratio",
                "num_segments",
                "skipped_segments",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary["outputs"] = {"latency_rows_csv": str(rows_path)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_args(args)

    entropy_dir = pathlib.Path(args.entropy_dir)
    output_dir = pathlib.Path(args.output_dir)
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    checkpoint_dir = stage_b._resolve_checkpoint_dir(train_config, args.checkpoint_dir, allow_config_default=False)
    norm_stats = stage_b._load_norm_stats(train_config, data_config, checkpoint_dir)

    dataset = stage_b._create_policy_dataset(train_config, data_config)
    from openpi.policies import policy_config as _policy_config

    _status(f"Loading policy from {checkpoint_dir}")
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
        sample_kwargs={"num_steps": args.num_steps},
    )

    files = stage_b._entropy_files(entropy_dir, args.max_items)
    samples = [
        stage_b._prepare_sample(
            path,
            file_position=i,
            args=args,
            norm_stats=norm_stats,
            data_config=data_config,
        )
        for i, path in enumerate(files)
    ]
    _status(f"Prepared {len(samples)} samples, warmup={args.warmup}, repeat={args.repeat}")

    # Compile/warm up both policy signatures: normal inference and override inference.
    warmup_sample = samples[0]
    _, warmup_input = stage_b._expert_actions_for_item(dataset, warmup_sample["item_index"], train_config, data_config)
    if warmup_input is None:
        raise RuntimeError("Could not reconstruct a policy input from the dataset.")
    warmup_seed = args.seed + 10_000_000
    warmup_full = None
    for i in range(args.warmup):
        warmup_full, _, _ = _infer_timed(policy, dict(warmup_input), seed=warmup_seed + i)
    if warmup_full is None or "coarse_actions" not in warmup_full:
        raise RuntimeError("Policy output did not include coarse_actions.")
    warmup_override_input = dict(warmup_input)
    warmup_override_input["coarse_actions_override"] = np.asarray(warmup_full["coarse_actions"], dtype=np.float32)
    for i in range(args.warmup):
        _infer_timed(policy, dict(warmup_override_input), seed=warmup_seed + args.warmup + i)

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for sample_idx, sample in enumerate(samples):
        _status(f"Benchmark sample {sample_idx + 1}/{len(samples)} item={sample['item_index']}")
        _, policy_input = stage_b._expert_actions_for_item(dataset, sample["item_index"], train_config, data_config)
        if policy_input is None:
            raise RuntimeError(f"Could not reconstruct policy input for item {sample['item_index']}.")

        for repeat_idx in range(args.repeat):
            seed_base = args.seed + sample["item_index"] * max(args.repeat, 1) + repeat_idx

            full_result, full_ms, full_wall_ms = _infer_timed(policy, dict(policy_input), seed=seed_base)
            if "coarse_actions" not in full_result:
                raise RuntimeError("Policy output did not include coarse_actions.")
            coarse_full = np.asarray(full_result["coarse_actions"], dtype=np.float32)

            cached_input = dict(policy_input)
            cached_input["coarse_actions_override"] = coarse_full
            _, cached_ms, cached_wall_ms = _infer_timed(policy, cached_input, seed=seed_base)

            coarse_pruned, skip_mask, skip_ratio = _make_pruned_coarse(
                sample,
                coarse_full,
                args=args,
                norm_stats=norm_stats,
                data_config=data_config,
                rng=rng,
            )
            pruned_input = dict(policy_input)
            pruned_input["coarse_actions_override"] = coarse_pruned.astype(np.float32)
            _, pruned_ms, pruned_wall_ms = _infer_timed(policy, pruned_input, seed=seed_base)

            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "item_index": sample["item_index"],
                    "repeat": repeat_idx,
                    "full_ms": full_ms,
                    "cached_override_ms": cached_ms,
                    "pruned_override_ms": pruned_ms,
                    "full_wall_ms": full_wall_ms,
                    "cached_override_wall_ms": cached_wall_ms,
                    "pruned_override_wall_ms": pruned_wall_ms,
                    "skip_ratio": skip_ratio,
                    "num_segments": len(sample["segments"]),
                    "skipped_segments": ";".join(str(i) for i, value in enumerate(skip_mask.tolist()) if value),
                }
            )

    full_values = [float(row["full_ms"]) for row in rows]
    cached_values = [float(row["cached_override_ms"]) for row in rows]
    pruned_values = [float(row["pruned_override_ms"]) for row in rows]
    full_mean = _mean(full_values)
    cached_mean = _mean(cached_values)
    pruned_mean = _mean(pruned_values)
    summary = {
        "config": {
            "entropy_dir": str(entropy_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "policy_config": args.config_name,
            "num_steps": args.num_steps,
            "max_items": args.max_items,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "strategy": args.strategy,
            "segment_mode": args.segment_mode,
            "chunk_size": args.chunk_size,
            "prune_ratio": args.prune_ratio,
            "replacement": args.replacement,
        },
        "aggregate": {
            "num_measurements": len(rows),
            "full_acot_ms_mean": full_mean,
            "full_acot_ms_std": _std(full_values),
            "cached_coarse_override_ms_mean": cached_mean,
            "cached_coarse_override_ms_std": _std(cached_values),
            "pruned_coarse_override_ms_mean": pruned_mean,
            "pruned_coarse_override_ms_std": _std(pruned_values),
            "skip_ratio_mean": _mean([float(row["skip_ratio"]) for row in rows]),
            "full_to_cached_speedup_pct": (1.0 - _ratio(cached_mean, full_mean)) * 100.0,
            "full_to_pruned_speedup_pct": (1.0 - _ratio(pruned_mean, full_mean)) * 100.0,
            "cached_to_pruned_speedup_pct": (1.0 - _ratio(pruned_mean, cached_mean)) * 100.0,
        },
        "interpretation": {
            "full_to_cached": "Latency saved by bypassing the entire explicit Action-CoT generation loop.",
            "cached_to_pruned": (
                "Extra latency saved by changing coarse_actions values only. This should be near zero because the "
                "current Stage B injection path keeps the same final action-head token length."
            ),
            "next_requirement": (
                "If cached_to_pruned is near zero, real segment-level speedup requires a model path that generates "
                "or consumes fewer explicit Action-CoT tokens, not just masked values."
            ),
        },
    }
    _write_outputs(output_dir, rows, summary)
    _status(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
