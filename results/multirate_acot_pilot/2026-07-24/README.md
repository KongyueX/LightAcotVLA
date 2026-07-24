# Fixed 1:4 multi-rate ACoT feasibility pilot

This directory records the final offline decision for the aggressive 2,000-window
pilot at code revision `da628a1`.

The slow branch runs full ACoT once every four policy calls and caches EAR/IAR.
The fast branch uses the latest two images and robot state to correct the cached
plan on every call. V2 keeps spatial visual tokens, attends over IAR tokens, and
forces the plan to modulate an observation-conditioned residual. SPRD adds a
training-only target for refreshing the current EAR phase.

## Result

The speed hypothesis is supported, but the accuracy hypothesis is rejected.
The final 1.78M-parameter executor runs in 0.644 ms mean / 0.702 ms p95 on the
pilot GPU. Fixed 1:4 amortization is 24.44 ms, corresponding to 3.92x versus the
95.844 ms full-ACoT reference and 2.82x versus the 68.929 ms B6 reference.

On 624 episode-disjoint stale test frames, lower 7D action MSE is better:

| Method | Stale MSE | Age-3 MSE | Gripper-transition accuracy |
| --- | ---: | ---: | ---: |
| B6 one-step reference | 0.08297 | 0.06962 | 73.08% |
| Hold the last action for four calls | 0.11547 | 0.14436 | 53.85% |
| V1 fast executor | 0.12333 | 0.12602 | 57.69% |
| V2 without SPRD | 0.12478 | 0.12810 | 57.69% |
| V2 with SPRD | 0.12560 | 0.12840 | 57.69% |

SPRD reduces phase-refresh MSE from 0.090308 to 0.089774, only 0.59%; the
pre-registered gate required at least 20%. The V1 fresh-plan oracle reaches
0.07935, indicating that stale-plan correction, not raw executor latency, is the
main unresolved bottleneck.

All V1/V2 variants failed the offline gate, so the pilot deliberately did not
run LIBERO closed-loop evaluation. The next experiment should change how the
slow plan is refreshed or provide genuinely corrective hard-state data; tuning
the SPRD weight or enlarging this executor on the same 2,000 windows is not
supported by these results.

Exact metrics and profiling scope are in
[`summary.json`](summary.json).
