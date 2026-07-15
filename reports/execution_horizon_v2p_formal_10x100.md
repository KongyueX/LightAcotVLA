# Budgeted Event V2-P formal LIBERO-10 evaluation

Evaluation date: 2026-07-14 to 2026-07-15 (Asia/Shanghai).

## Protocol

- Base policy: acot_libero_action_cot_explicit_implicit_co_fusion, checkpoint 50999; base weights were frozen.
- Predictor: round-0 validation-best sidecar from sft_round00_initial_200_seed7_8486631/params; the standalone SFT trainer did not load the base policy.
- LIBERO-10, 10 tasks x 100 trials per scheme, seed 7, Action-CoT denoising steps 10.
- All five schemes have exactly 1,000 unique (task_id, episode) rows. Their key sets and initial_state_id fields match exactly. LIBERO supplies 50 initial states per task, so 100 trials cycle each state twice with distinct episode seeds.
- Evaluations were run strictly serially on isolated deployment paths: pre-V2P pure-base for H5/H9, current base-only for Exact K20, and the sidecar server for distilled/value-refined.
- Exact K20 uses shared-prefix batched MC with K=20. Distilled/value run the predictor inside the same JAX/GPU policy call without another VLM pass or RPC.
- Policy, server, and RPC timings include all scheme-specific inference overhead. Predictor/teacher timing is a component and must not be added again. Full additionally includes environment reset/step/render/close work.

## Overall results

| Scheme | Predictor | Success | Timeout | Calls/ep | Avg H | Policy ms/call | Policy s/ep | Server s/ep | RPC s/ep | Full s/ep | Aux latency | Policy speedup vs H5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACoT-VLA original H5 | No | 92.7% | 7.3% | 63.518 | 5.000 | 89.540 | 5.687 | 6.212 | 6.290 | 16.091 | - | 1.000x |
| ACoT-VLA Fixed H9 | No | 95.0% | 5.0% | 33.568 | 9.000 | 88.855 | 2.983 | 3.249 | 3.288 | 12.567 | - | 1.907x |
| Exact batched-MC V2 K20 | No (K20) | 94.4% | 5.6% | 34.647 | 8.846 | 122.087 | 4.230 | 4.467 | 4.507 | 13.366 | teacher 105.818 ms/call | 1.345x |
| V2-P distilled | Yes | 95.2% | 4.8% | 30.536 | 9.874 | 110.904 | 3.387 | 3.666 | 3.702 | 12.430 | predictor 4.761 ms/call | 1.679x |
| V2-P value-refined | Yes | 94.3% | 5.7% | 32.892 | 9.328 | 109.780 | 3.611 | 3.903 | 3.945 | 12.796 | predictor 4.668 ms/call | 1.575x |

## Aggregate measured time

The totals below are sums of the 1,000 episode-level timers, not estimates from calls.

| Scheme | Policy total h | Server total h | RPC total h | Full episode total h |
|---|---:|---:|---:|---:|
| ACoT-VLA original H5 | 1.580 | 1.726 | 1.747 | 4.470 |
| ACoT-VLA Fixed H9 | 0.829 | 0.902 | 0.913 | 3.491 |
| Exact batched-MC V2 K20 | 1.175 | 1.241 | 1.252 | 3.713 |
| V2-P distilled | 0.941 | 1.018 | 1.028 | 3.453 |
| V2-P value-refined | 1.003 | 1.084 | 1.096 | 3.554 |

Observed evaluator run wall times from run_config.json to summary.json were: pure-base H5+H9 combined 7:57:48; Exact K20 3:43:42; distilled 3:28:03; value-refined 3:33:25.

## Outcome-stratified overall metrics

| Scheme | Success calls/ep | Success policy s/ep | Success RPC s/ep | Success full s/ep | Timeout calls/ep | Timeout policy s/ep | Timeout RPC s/ep | Timeout full s/ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACoT-VLA original H5 | 52.928 | 4.766 | 5.274 | 14.384 | 198.000 | 17.392 | 19.196 | 37.769 |
| ACoT-VLA Fixed H9 | 29.545 | 2.622 | 2.889 | 11.714 | 110.000 | 9.843 | 10.877 | 28.781 |
| Exact batched-MC V2 K20 | 30.121 | 3.685 | 3.927 | 12.301 | 110.946 | 13.420 | 14.289 | 31.316 |
| V2-P distilled | 27.054 | 3.001 | 3.281 | 11.620 | 99.604 | 11.027 | 12.049 | 28.507 |
| V2-P value-refined | 28.567 | 3.135 | 3.425 | 11.787 | 104.439 | 11.477 | 12.545 | 29.492 |

## Per-task metrics

| Scheme | Task | Success | Timeout | Calls/ep | Avg H | Policy s/ep | Server s/ep | RPC s/ep | Full s/ep | Predictor ms/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACoT-VLA original H5 | 0 | 97.0% | 3.0% | 65.230 | 5.000 | 6.093 | 6.689 | 6.777 | 19.228 | - |
| ACoT-VLA original H5 | 1 | 98.0% | 2.0% | 54.690 | 5.000 | 4.957 | 5.402 | 5.479 | 16.722 | - |
| ACoT-VLA original H5 | 2 | 98.0% | 2.0% | 53.240 | 5.000 | 4.782 | 5.211 | 5.283 | 11.821 | - |
| ACoT-VLA original H5 | 3 | 98.0% | 2.0% | 51.040 | 5.000 | 4.663 | 5.101 | 5.169 | 12.871 | - |
| ACoT-VLA original H5 | 4 | 98.0% | 2.0% | 51.950 | 5.000 | 4.688 | 5.111 | 5.182 | 17.162 | - |
| ACoT-VLA original H5 | 5 | 89.0% | 11.0% | 53.410 | 5.000 | 4.695 | 5.125 | 5.182 | 12.984 | - |
| ACoT-VLA original H5 | 6 | 89.0% | 11.0% | 64.640 | 5.000 | 5.901 | 6.504 | 6.574 | 17.048 | - |
| ACoT-VLA original H5 | 7 | 99.0% | 1.0% | 55.550 | 5.000 | 4.937 | 5.402 | 5.456 | 14.388 | - |
| ACoT-VLA original H5 | 8 | 66.0% | 34.0% | 125.140 | 5.000 | 10.863 | 11.813 | 11.963 | 24.052 | - |
| ACoT-VLA original H5 | 9 | 95.0% | 5.0% | 60.290 | 5.000 | 5.294 | 5.763 | 5.839 | 14.635 | - |
| ACoT-VLA Fixed H9 | 0 | 99.0% | 1.0% | 32.600 | 9.000 | 2.936 | 3.207 | 3.249 | 15.080 | - |
| ACoT-VLA Fixed H9 | 1 | 100.0% | 0.0% | 28.490 | 9.000 | 2.458 | 2.656 | 2.691 | 13.720 | - |
| ACoT-VLA Fixed H9 | 2 | 99.0% | 1.0% | 29.390 | 9.000 | 2.612 | 2.846 | 2.882 | 9.086 | - |
| ACoT-VLA Fixed H9 | 3 | 99.0% | 1.0% | 27.050 | 9.000 | 2.394 | 2.602 | 2.633 | 10.029 | - |
| ACoT-VLA Fixed H9 | 4 | 93.0% | 7.0% | 32.680 | 9.000 | 2.847 | 3.089 | 3.130 | 15.561 | - |
| ACoT-VLA Fixed H9 | 5 | 92.0% | 8.0% | 27.970 | 9.000 | 2.453 | 2.665 | 2.700 | 9.941 | - |
| ACoT-VLA Fixed H9 | 6 | 93.0% | 7.0% | 31.800 | 9.000 | 2.804 | 3.053 | 3.092 | 12.874 | - |
| ACoT-VLA Fixed H9 | 7 | 96.0% | 4.0% | 32.840 | 9.000 | 2.773 | 2.986 | 3.023 | 12.046 | - |
| ACoT-VLA Fixed H9 | 8 | 86.0% | 14.0% | 58.120 | 9.000 | 5.255 | 5.751 | 5.815 | 15.578 | - |
| ACoT-VLA Fixed H9 | 9 | 93.0% | 7.0% | 34.740 | 9.000 | 3.294 | 3.631 | 3.667 | 11.757 | - |
| Exact batched-MC V2 K20 | 0 | 98.0% | 2.0% | 34.530 | 8.837 | 4.277 | 4.524 | 4.565 | 16.056 | - |
| Exact batched-MC V2 K20 | 1 | 98.0% | 2.0% | 30.200 | 8.817 | 3.716 | 3.929 | 3.962 | 14.306 | - |
| Exact batched-MC V2 K20 | 2 | 99.0% | 1.0% | 29.280 | 8.835 | 3.549 | 3.746 | 3.779 | 9.431 | - |
| Exact batched-MC V2 K20 | 3 | 100.0% | 0.0% | 26.330 | 8.804 | 3.218 | 3.399 | 3.429 | 10.080 | - |
| Exact batched-MC V2 K20 | 4 | 97.0% | 3.0% | 30.260 | 8.833 | 3.743 | 3.961 | 3.997 | 15.429 | - |
| Exact batched-MC V2 K20 | 5 | 91.0% | 9.0% | 28.780 | 8.805 | 3.427 | 3.603 | 3.635 | 10.437 | - |
| Exact batched-MC V2 K20 | 6 | 92.0% | 8.0% | 34.410 | 8.863 | 4.199 | 4.432 | 4.473 | 13.980 | - |
| Exact batched-MC V2 K20 | 7 | 98.0% | 2.0% | 32.510 | 8.835 | 3.971 | 4.191 | 4.228 | 12.674 | - |
| Exact batched-MC V2 K20 | 8 | 82.0% | 18.0% | 60.710 | 8.902 | 7.366 | 7.781 | 7.851 | 17.420 | - |
| Exact batched-MC V2 K20 | 9 | 89.0% | 11.0% | 39.460 | 8.863 | 4.834 | 5.108 | 5.154 | 13.850 | - |
| V2-P distilled | 0 | 98.0% | 2.0% | 30.680 | 9.890 | 3.464 | 3.760 | 3.799 | 15.044 | 4.929 |
| V2-P distilled | 1 | 100.0% | 0.0% | 26.580 | 9.636 | 3.011 | 3.268 | 3.301 | 13.471 | 4.879 |
| V2-P distilled | 2 | 97.0% | 3.0% | 28.210 | 9.828 | 3.145 | 3.411 | 3.443 | 9.259 | 4.808 |
| V2-P distilled | 3 | 98.0% | 2.0% | 26.170 | 9.449 | 2.806 | 3.027 | 3.056 | 9.907 | 4.419 |
| V2-P distilled | 4 | 97.0% | 3.0% | 26.670 | 10.000 | 2.786 | 2.969 | 2.998 | 14.373 | 4.416 |
| V2-P distilled | 5 | 90.0% | 10.0% | 26.100 | 9.998 | 2.892 | 3.131 | 3.162 | 9.996 | 4.785 |
| V2-P distilled | 6 | 94.0% | 6.0% | 28.770 | 9.995 | 3.219 | 3.494 | 3.528 | 12.800 | 4.811 |
| V2-P distilled | 7 | 98.0% | 2.0% | 27.930 | 9.975 | 3.132 | 3.397 | 3.431 | 11.744 | 4.834 |
| V2-P distilled | 8 | 86.0% | 14.0% | 52.280 | 9.990 | 5.830 | 6.321 | 6.382 | 15.605 | 4.817 |
| V2-P distilled | 9 | 94.0% | 6.0% | 31.970 | 9.854 | 3.581 | 3.882 | 3.920 | 12.106 | 4.812 |
| V2-P value-refined | 0 | 97.0% | 3.0% | 31.670 | 9.708 | 3.510 | 3.800 | 3.839 | 15.257 | 4.719 |
| V2-P value-refined | 1 | 99.0% | 1.0% | 29.410 | 8.912 | 3.215 | 3.474 | 3.507 | 13.785 | 4.537 |
| V2-P value-refined | 2 | 95.0% | 5.0% | 32.180 | 9.067 | 3.420 | 3.691 | 3.726 | 9.764 | 4.504 |
| V2-P value-refined | 3 | 99.0% | 1.0% | 27.220 | 9.142 | 2.974 | 3.220 | 3.251 | 10.119 | 4.639 |
| V2-P value-refined | 4 | 96.0% | 4.0% | 30.310 | 9.030 | 3.317 | 3.589 | 3.627 | 15.098 | 4.704 |
| V2-P value-refined | 5 | 91.0% | 9.0% | 25.630 | 9.802 | 2.852 | 3.090 | 3.125 | 9.913 | 4.963 |
| V2-P value-refined | 6 | 92.0% | 8.0% | 32.330 | 9.369 | 3.539 | 3.825 | 3.868 | 13.406 | 4.690 |
| V2-P value-refined | 7 | 98.0% | 2.0% | 29.430 | 9.442 | 3.234 | 3.489 | 3.526 | 11.890 | 4.561 |
| V2-P value-refined | 8 | 84.0% | 16.0% | 56.760 | 9.245 | 6.180 | 6.671 | 6.745 | 16.188 | 4.586 |
| V2-P value-refined | 9 | 92.0% | 8.0% | 33.980 | 9.642 | 3.868 | 4.186 | 4.230 | 12.539 | 4.870 |

## Execution-horizon distributions

| Scheme | H distribution over decisions |
|---|---|
| ACoT-VLA original H5 | H5=63518 |
| ACoT-VLA Fixed H9 | H9=33568 |
| Exact batched-MC V2 K20 | H1=2, H2=4, H3=963, H4=259, H5=371, H6=744, H7=1564, H8=4520, H9=13067, H10=13153 |
| V2-P distilled | H1=3, H2=3, H3=98, H4=120, H5=134, H6=162, H7=141, H8=153, H9=340, H10=29382 |
| V2-P value-refined | H1=374, H2=311, H3=174, H4=406, H5=454, H6=435, H7=561, H8=541, H9=5808, H10=23828 |

## Same-state paired outcomes

A wins means A succeeds and B fails on the same (task_id, episode); p-values are two-sided exact McNemar tests and are not multiplicity-adjusted.

| A | B | A wins | B wins | Both success | Both fail | p |
|---|---|---:|---:|---:|---:|---:|
| ACoT-VLA original H5 | ACoT-VLA Fixed H9 | 32 | 55 | 895 | 18 | 0.01783 |
| ACoT-VLA original H5 | Exact batched-MC V2 K20 | 36 | 53 | 891 | 20 | 0.08932 |
| ACoT-VLA original H5 | V2-P distilled | 30 | 55 | 897 | 18 | 0.008836 |
| ACoT-VLA original H5 | V2-P value-refined | 31 | 47 | 896 | 26 | 0.08878 |
| ACoT-VLA Fixed H9 | Exact batched-MC V2 K20 | 38 | 32 | 912 | 18 | 0.5504 |
| ACoT-VLA Fixed H9 | V2-P distilled | 36 | 38 | 914 | 12 | 0.9076 |
| ACoT-VLA Fixed H9 | V2-P value-refined | 43 | 36 | 907 | 14 | 0.4999 |
| Exact batched-MC V2 K20 | V2-P distilled | 33 | 41 | 911 | 15 | 0.416 |
| Exact batched-MC V2 K20 | V2-P value-refined | 42 | 41 | 902 | 15 | 1 |
| V2-P distilled | V2-P value-refined | 40 | 31 | 912 | 17 | 0.3425 |

## Interpretation

- **Distillation succeeded as a single-call replacement for Exact K20.** Distilled reaches 95.2% success versus Exact's 94.4%, while reducing policy time from 4.230 to 3.387 s/episode (1.249x faster than Exact). The paired success difference is not significant (p=0.416), so the supported claim is preserved outcome quality with lower actual sampling cost.
- **Relative to pure original H5, distilled is the strongest adaptive result.** Success rises by 2.5 points and policy/RPC/full time improves by 1.679x/1.699x/1.294x. The same-state paired comparison is 55 wins versus 30 losses (p=0.00884).
- **Fixed H9 remains the policy-time leader.** It is 1.907x faster than H5 in policy time and has 95.0% success. Distilled is 13.5% slower in policy time and 12.6% slower in RPC time than H9, although its full episode timer is 1.1% lower. Their paired success is indistinguishable (38 distilled wins, 36 H9 wins; p=0.908), so V2-P has not demonstrated a clear Pareto improvement over H9.
- **Q refinement did not improve round-0.** Value-refined reduces average H from 9.874 to 9.328 but falls from 95.2% to 94.3% success, increases calls from 30.536 to 32.892, and is slower than distilled on policy, RPC, and full time. Distilled has 40 paired wins versus 31 for value-refined (p=0.342).
- **The predictor head itself is small but the sidecar path has broader overhead.** The measured head costs 4.761 ms/call (distilled) and 4.668 ms/call (value), about 4.3% of policy time. Sidecar policy latency is roughly 110 ms/call versus 89 ms/call for pure-base, because feature extraction/transfer is also included. Calls reduction is what produces the net episode-level speedup.
- **PPO remains deferred.** The completed evidence supports the requested SFT and iterative-relabel stage; no PPO code or result is included.

## Remote artifacts

- Pure-base H5/H9: /root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_pure_base_c5c08fc_original_h5_fixed_h9_10tasks_100trials
- Exact K20: /root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_current_base_exact_batched_mc_v2_k20_10tasks_100trials
- Distilled: /root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_round00_sidecar_v2_distilled_10tasks_100trials
- Value-refined: /root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_round00_sidecar_v2_value_refined_10tasks_100trials
- Final Feishu audit marker: /root/autodl-tmp/acotvla/execution_horizon_v2p/formal_five_scheme_final_audit_notification/.feishu_notification_sent
