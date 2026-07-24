# Execution Horizon V2-P results

该目录保存从服务器同步的结构化证据，人工汇总见 [`../../reports/execution_horizon_v2p_formal_10x100.md`](../../reports/execution_horizon_v2p_formal_10x100.md)。

## `formal_10x100`

- `pure_base_h5_h9/`：pre-V2P pure-base 的 original H5 与 Fixed H9，同一批 LIBERO-10 初始状态、每任务 100 局。
- `exact_k20/`：current base 上的 Exact K20 counterfactual teacher。
- `distilled/`：V2 distilled sidecar。
- `value_refined/`：V2 value-refined sidecar。

每个目录保留完整 `summary.json` 和 `per_task_summary.csv`。Exact K20 使用多次 teacher sampling，是诊断 oracle，不是可部署速度结果。

## `diagnostics`

- `headroom_aggregate596.json`：596 条离线 snapshot 的多 horizon headroom 汇总。
- `hard20_r5_final_audit.json`：20 个 hard roots、每 root 5 个 policy seeds 的 H1/H3/H6/H10 对照。
- `selector_rl_pilot_final_audit.json`：Task8 selector PPO pilot 的两轮配对评测。
- `action_chunk_candidate_hard16_r5.json`：16 个 hard roots 的 action candidate 分支诊断。
- `action_chunk_progress_pooled40.json`：40 个 roots 的 progress selector 汇总。

这些文件包含 counterfactual、hindsight 或小样本诊断结果，不能单独作为部署性能或稳定成功率提升结论。
