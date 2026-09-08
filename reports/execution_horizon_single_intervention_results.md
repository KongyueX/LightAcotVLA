# 单次current介入：开发成功数净增1局，耗时略增

2026-09-09。同100个开发初态的200局完整闭环已完成：单次介入94/100，原A93/100，救回2局、退化1局；平均RPC增加1.72%，整局耗时增加1.27%。这是一个小幅正向成功信号与时间成本的权衡，尚不足以确认稳定收益。本批结束，A保持默认，单次介入策略冻结保留为备选，未启动预留测试或新训练。

## 完整同批结果

| 策略 | 成功 | 平均调用数 | 平均policy秒 | 平均RPC秒 | 平均整局秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原A | 93/100 | 16.10 | 1.732746 | 1.923678 | 10.751209 |
| 首次分歧仅一次current介入 | 94/100 | 16.44 | 1.763364 | 1.956796 | 10.887217 |

两者共同成功92局、共同失败5局；单次介入净救回1局。沿用既有配对汇总程序输出的成功率差95%区间为[−4,+6]个百分点；RPC平均差+33.118毫秒，95%区间[−192.114,+291.899]毫秒；整局平均差+136.008毫秒，95%区间[−563.807,+1012.824]毫秒。区间均跨0，不作显著提升、稳定加速或等价性声明，也没有为本次解释重新计算区间。

## 实际介入与差异episode

100局中70局介入一次、30局无H分歧而全程沿用A；介入中31次缩短H、39次延长H，中位发生步数105。真实decision输出确认每局最多一次介入、非介入的H来自A；介入后的765次决策均未再计算current小头，全部选择A的H。单次模式的平均selector后处理为0.137338毫秒/调用。

| Task / bank ID | 原A→单次介入成功 | 介入步数 | A H→current H | RPC变化秒 | 整局变化秒 |
| --- | --- | ---: | --- | ---: | ---: |
| 0 / 344 | 0→1，救回 | 360 | 20→25 | −0.897598 | −3.197638 |
| 3 / 339 | 1→0，退化 | 105 | 10→25 | +4.320368 | +14.437748 |
| 7 / 339 | 0→1，救回 | 75 | 15→5 | −4.268711 | −13.370075 |

救回既包含延长H，也包含缩短H；这些episode不足以支持统一延长或缩短规则。完整逐Task结果保留在配对汇总中。

## 研究判断与下一步

当前current权重并非只能产生无用改动：限制为一次介入后，在本批完整闭环观察到净+1局。但此前19个训练root中“current首H后交还A”的55/57，不能转化为开发收益大小的预期；这里的实际增益很小，也没有同时减少平均调用或时间。

该结果不能单独证明“反复介入导致旧候选失败”，更不能证明一次就是最优介入次数。当前不扩展门控网络、不扫描次数或阈值、不重新训练。若继续验证，只建议固定当前策略与权重，做一次独立初态的A配对对照，确认成功与成本的权衡能否保持；本轮未自动消耗预留IDs346–365。

## 协议、文件与状态

- Task0–9，bank IDs336–345，每策略100局，按初态交错；seed7、NFE10/10、resize224、等待10步，原8040 H25/A服务。
- 模式`ordered_feedback_current_once`，加载原`greedy_selection_9dca506/training_current/checkpoint.npz`。A/VLA/current权重与归一化均冻结，所有决策只需一次VLA响应。
- 实现提交`e32fd51`；2026-09-09 01:12启动、01:48:33完成，`development.exit=0`。
- [实验方案](execution_horizon_single_intervention_experiment.md)、[完整汇总与100局介入记录](../results/execution_horizon_single_intervention/development_20260909/summary.json)、[配对汇总](../results/execution_horizon_single_intervention/development_20260909/development/paired_analysis.json)、[逐局CSV](../results/execution_horizon_single_intervention/development_20260909/development/rollout_rows.csv)、[决策CSV](../results/execution_horizon_single_intervention/development_20260909/development/decisions.csv)。
- 远端输出保留于`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/single_intervention_20260909_v1`。本批评测结束，原A服务保留。

## Limitations

只有一轮100个开发初态，开发集此前已反复使用；净差仅1局，不构成独立泛化或稳定改进证据。此前全程current、架构及局部分支结果来自不同运行或cohort，不能与本批直接拼成匹配排名。真实整局时间包含环境和客户端开销，统计区间仍较宽；本轮没有预留测试结果。
