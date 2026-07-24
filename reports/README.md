# 实验报告与证据索引

本目录保存适合长期复用的实验事实与人工汇总；结构化实验数据保存在 [`../results`](../results)。

## 本地报告

| 路径 | 用途 |
| --- | --- |
| [`context/experiment_log.md`](context/experiment_log.md) | 日报、周报和月计划的事实来源，持续追加实验配置、路径、指标、证据边界和下一步；本次快照约 480 KB。 |
| [`execution_horizon_v2p_formal_10x100.md`](execution_horizon_v2p_formal_10x100.md) | V2-P 五种方案的正式 LIBERO-10、每任务 100 局对照报告。 |

PPTX、XLSX、日报/周报草稿、`*.inspect.ndjson` 和 `.DS_Store` 均保留在本地，不进入 Git。

## 服务器实验汇总

2026-07-24（Asia/Shanghai）对服务器 `/root/ACoT-VLA/reports` 和 `/root/autodl-tmp/acotvla` 做了只读盘点，只同步适合代码审查和后续分析的小型汇总：

| 路径 | 内容 |
| --- | --- |
| [`../results/execution_horizon_v2p/`](../results/execution_horizon_v2p) | V2-P 正式 10×100 原始 JSON/CSV，以及 headroom、hard-state、selector RL、candidate/progress 诊断。 |
| [`../results/task89_adaptation/`](../results/task89_adaptation) | Task8/9 targeted SFT、uniform continued-SFT 对照、hard-state DAgger 收集与评测汇总。 |
| [`../results/ir_acot_pilot/`](../results/ir_acot_pilot) | IR-ACoT 快速否证 pilot 的开放环审计、clean-only profile、B6/IR 训练汇总和 metric streams。 |
| [`../results/stage_b/`](../results/stage_b) | 仓库原有的 Stage B pruning、closed-loop 和速度汇总。 |

服务器与本地的 `execution_horizon_v2p_formal_10x100.md` SHA-256 均为 `b036688416e65119ad9cb3e08c5cfca028f1f590b2b4536c5a74190effaf7305`，无需重复复制。

## 入库边界

同步 `summary.json`、`per_task_summary.csv`、`aggregate/audit`、小型 `training_manifest.jsonl` 和短训练的 `metrics.jsonl`。输出内保留原始绝对路径，便于回到服务器追溯来源。

不入库 checkpoint、Orbax 参数、HDF5 标签、原始逐步 rollout、视频、完整训练日志、环境、缓存或临时检查文件。这些材料体积大、噪声高，必要时应按索引中的服务器路径单独读取，而不应直接提交到 GitHub。

本次精确凭据模式扫描未发现 webhook、GitHub/OpenAI token、AWS key 或私钥。完整事实日志保留历史服务器入口和绝对路径用于实验追溯，但不包含认证密钥。
