# IR-ACoT pilot results

`2026-07-24/` 是快速否证型 MVP 的服务器快照，只保留小型 JSON/JSONL 汇总与训练指标流：

- `ear_audit200.json`：200 个状态、包含 intervention 的开放环 EAR 因果敏感性门槛。
- `coarse_teacher_2k.json`：teacher 10-step EAR 的 2,000 条 clean-only 导出/profile。
- `coarse_b6_train.json`：Fast-EAR B6-lite coarse 分支 300-step 训练汇总。
- `coarse_b6_metrics.jsonl`：coarse 分支训练/验证 metric stream。
- `student_coarse_2k.json`：已训练 one-step coarse student 的 2,000 条 clean-only 导出/profile。
- `naive_1x1_2k.json`：原始权重强制 coarse/final 1×1 的 2,000 条 clean-only profile。
- `final_teacher_2k.json`：以 student EAR 为条件的 2,000 条 final teacher relabel 汇总。
- `final_b6_train.json` 与 `final_b6_metrics.jsonl`：matched B6-lite final student 的 300-step 训练数据。
- `final_ir_train.json` 与 `final_ir_metrics.jsonl`：加入 intervention-response loss 的 IR-lite 300-step 训练数据。

`coarse_teacher_2k.json`、`student_coarse_2k.json` 和 `naive_1x1_2k.json` 均设置 `clean_only=true`，没有 intervention 计数；其中 `ear_causal_audit_pass=false` 只是审计字段不适用于 clean-only 模式，不能解释为方法失败。

最终验证点上，B6-lite 的 final MSE / IR-delta MSE / cosine 为
`0.00364254 / 0.00030865 / 0.50246`，IR-lite 为
`0.00363019 / 0.00031122 / 0.49612`；IR-lite 尚未显示相对 matched
B6-lite 的 response-alignment 优势。

这些结果不是闭环成功率，也不是论文结论。HDF5 shards、delta sidecar 参数、optimizer state 和完整日志仍留在服务器。
