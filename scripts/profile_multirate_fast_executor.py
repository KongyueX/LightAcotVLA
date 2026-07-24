"""Microbenchmark the real fixed-shape multi-rate fast executor on JAX."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import flax.nnx as nnx
import jax
import numpy as np
import tyro

from openpi.models import multirate_fast_executor


@dataclasses.dataclass(frozen=True)
class Args:
    output: str
    seed: int = 7
    batch_size: int = 1
    warmup_calls: int = 50
    timed_calls: int = 2_000
    full_refresh_ms: float = 95.844
    b6_ms: float = 68.929
    fast_mean_gate_ms: float = 20.0
    fast_p95_gate_ms: float = 25.0
    amortized_gate_ms: float = 40.0


def main(args: Args) -> None:
    if args.batch_size <= 0 or args.warmup_calls < 0 or args.timed_calls <= 0:
        raise ValueError("batch_size/timed_calls must be positive and warmup_calls non-negative.")
    config = multirate_fast_executor.MultiRateFastExecutorConfig()
    model = multirate_fast_executor.MultiRateFastExecutor(config, rngs=nnx.Rngs(args.seed))
    graphdef, params = nnx.split(model)

    @jax.jit
    def infer(
        current_params: nnx.State,
        current_images: jax.Array,
        state: jax.Array,
        cached_ear: jax.Array,
        cached_iar: jax.Array,
        cache_age: jax.Array,
    ) -> jax.Array:
        executor = nnx.merge(graphdef, current_params)
        return executor(current_images, state, cached_ear, cached_iar, cache_age)

    rng = np.random.default_rng(args.seed)
    host_inputs = {
        "current_images": rng.uniform(
            0.0,
            1.0,
            size=(args.batch_size, config.image_views, config.image_size, config.image_size, 3),
        ).astype(np.float32),
        "state": rng.normal(size=(args.batch_size, config.state_dim)).astype(np.float32),
        "cached_ear": rng.normal(size=(args.batch_size, config.ear_horizon, config.action_dim)).astype(np.float32),
        "cached_iar": rng.normal(size=(args.batch_size, 18, config.iar_dim)).astype(np.float32),
        "cache_age": rng.integers(
            0,
            config.max_cache_age + 1,
            size=(args.batch_size,),
            dtype=np.int32,
        ),
    }

    for _ in range(args.warmup_calls):
        infer(params, **host_inputs).block_until_ready()
    timings = np.empty((args.timed_calls,), dtype=np.float64)
    for index in range(args.timed_calls):
        start = time.perf_counter()
        infer(params, **host_inputs).block_until_ready()
        timings[index] = (time.perf_counter() - start) * 1_000

    mean_ms = float(np.mean(timings))
    p50_ms = float(np.quantile(timings, 0.50))
    p95_ms = float(np.quantile(timings, 0.95))
    amortized_ms = float((args.full_refresh_ms + 3.0 * mean_ms) / 4.0)
    result = {
        "status": "complete",
        "device": str(jax.devices()[0]),
        "config": dataclasses.asdict(config),
        "parameter_count": multirate_fast_executor.estimate_parameter_count(config),
        "batch_size": args.batch_size,
        "warmup_calls": args.warmup_calls,
        "timed_calls": args.timed_calls,
        "fast_mean_ms": mean_ms,
        "fast_p50_ms": p50_ms,
        "fast_p95_ms": p95_ms,
        "full_refresh_ms": args.full_refresh_ms,
        "b6_ms": args.b6_ms,
        "fixed_1_to_4_amortized_ms": amortized_ms,
        "speedup_vs_full": args.full_refresh_ms / amortized_ms,
        "speedup_vs_b6": args.b6_ms / amortized_ms,
        "gates": {
            "fast_mean_ms_max": args.fast_mean_gate_ms,
            "fast_p95_ms_max": args.fast_p95_gate_ms,
            "amortized_ms_max": args.amortized_gate_ms,
        },
        "speed_gate_pass": (
            mean_ms <= args.fast_mean_gate_ms
            and p95_ms <= args.fast_p95_gate_ms
            and amortized_ms <= args.amortized_gate_ms
        ),
        "note": (
            "Includes host-to-device inputs and synchronization for the executor call, "
            "but not websocket/RPC, image resizing, or the slow refresh implementation."
        ),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
