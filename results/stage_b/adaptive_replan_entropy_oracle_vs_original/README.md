# Adaptive Entropy Oracle vs Original Baseline Per-Task Comparison

This comparison uses the clean original LIBERO-10 baseline and the user-provided entropy-adaptive deployable per-task metrics. Baseline wall timing is not available in the original timing logs, so only policy/server total speedups are computed.

- policy total speedup positive tasks: 7/10
- server total speedup positive tasks: 7/10
- calls/episode reduction positive tasks: 5/10
- success drop within 3 percentage points: 6/10
- policy speedup and success drop within 3pp: 5/10
- server speedup and success drop within 3pp: 5/10

Per-task details are in `adaptive_entropy_oracle_vs_original_per_task.csv`.
