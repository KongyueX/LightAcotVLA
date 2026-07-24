# Task8/9 adaptation results

该目录整理 Task8/9 failure-focused adaptation 的小型服务器汇总，保留原始 JSON/CSV，不复制 checkpoint、视频或逐步日志。

## Targeted SFT

- `targeted_sft/round0_collection_summary.json`：targeted 数据清单统计。
- `targeted_sft/pilot1k_initial20/`：pilot 1k checkpoint 在 Task8/9 初始 20 个 episode 上的 Fixed H9 评测。

## Uniform control

- `uniform_control/initial20/`：与 targeted pilot 初始 20 个 episode 对齐的 uniform continued-SFT 控制。
- `uniform_control/ep20_99/`：Task8/9 episode 20–99 的扩展评测。

## Hard-state DAgger

- `hard_state_dagger/round1_collection/`：收集汇总和三条 accepted corrective trajectories 的训练 manifest。
- `hard_state_dagger/eval/base50999_heldout50/`：base 50999 的 Task8/9 held-out 50×2 对照。
- `hard_state_dagger/eval/dagger1k_heldout50/`：DAgger 1k checkpoint 在同一 held-out 集上的结果。
- `hard_state_dagger/eval/strict500_task8_ep25_49/` 与 `strict999_task8_ep25_49/`：strict-fine 路线的 Task8、25 局验证诊断。

目录名中的样本范围、checkpoint step 和协议差异不能混用。当前材料支持复盘和继续实验，不应被改写为多 seed 稳定提升或全 LIBERO-10 结论。
