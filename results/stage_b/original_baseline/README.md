# LIBERO-10 Original Baseline Timing Summary

Source directory:

`/Users/kongyue/Downloads/libero_long_timing_logs/aligned_eval/acot_libero_long_timing_50/libero_10/episodes`

Parsed `500` trial files across `10` tasks. Each `trial_*.jsonl` row is one policy/server inference call within an episode.

Generated files:

- `libero10_original_per_trial.csv`: one row per trial, including success, episode steps, inference count, per-call latency, total episode inference time, and versions excluding `infer_idx=0`.
- `libero10_original_per_task_summary.csv`: one row per task plus an `all` row for the full LIBERO-10 baseline.
- `libero10_original_overall_summary.json`: machine-readable metadata and overall metrics.

Important timing fields:

- `policy_ms_mean_per_call_weighted`: total policy inference time divided by total inference rows.
- `policy_ms_total_mean_per_episode`: average total policy inference time per episode.
- `*_excluding_infer0`: same metrics after removing the first inference in each trial. This is useful because `infer_idx=0` may include initialization/JIT/cache effects.

Overall summary including all inference calls:

- success rate: 0.9660
- calls per episode: 58.5020
- weighted policy latency per call: 78.9558 ms
- weighted server latency per call: 84.3060 ms
- mean total policy inference time per episode: 4619.0703 ms
- mean total server inference time per episode: 4932.0689 ms

Overall summary excluding `infer_idx=0`:

- weighted policy latency per call: 77.9105 ms
- weighted server latency per call: 83.2552 ms
- mean total policy inference time per episode: 4480.0123 ms
- mean total server inference time per episode: 4787.3388 ms

Clean comparison baseline excluding the one warmup outlier episode (`task0 trial0`):

- success rate: 0.9659
- calls per episode: 58.5190
- weighted policy latency per call: 77.9287 ms
- weighted server latency per call: 83.2784 ms
- mean total policy inference time per episode: 4560.3128 ms
- mean total server inference time per episode: 4873.3741 ms

Recommended PK field against entropy-adaptive deployable results:

- baseline speed: `policy_ms_total_mean_per_episode` and `server_ms_total_mean_per_episode` from `libero10_original_per_task_summary_excluding_task0_trial0_warmup.csv`
- entropy-adaptive speed: `avg_total_deployable_policy_inference_ms_per_episode` and `avg_total_deployable_server_inference_ms_per_episode` from adaptive `summary.json`
- baseline calls: `inference_calls_mean_per_episode`
- entropy-adaptive calls: `avg_deployable_policy_calls_per_episode`
