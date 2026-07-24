# Transported Action-CoT pilot

This pilot tests whether the latest observation can transport a cached
Action-CoT EAR along a continuous time axis without allowing a direct
observation-to-action shortcut.

The positive result is structural: the balanced model reduces held-out
time-warp phase MAE from `0.3125` to `0.2203`, reduces full-EAR MSE by `22.43%`,
and becomes substantially worse when the current observation is replaced by
the anchor observation. Its fast path is also small (`474,530` parameters) and
fast (`0.3203 ms` p95).

The primary action gate fails. Nominal test action MSE is `0.12411`, versus
`0.11547` for Hold4 and the pre-registered continuation threshold `0.112335`.
Balancing nominal and warped samples did not recover the loss, so this is not
just a sampling-ratio issue. No closed-loop LIBERO evaluation or extra seeds
were run.

See [summary.json](summary.json) for the complete metrics and server artifact
paths.
