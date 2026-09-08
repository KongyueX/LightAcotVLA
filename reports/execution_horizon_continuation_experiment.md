# A接力与候选接力：有界测试

2026-09-09。按[迭代分析](execution_horizon_iteration_analysis.md)的第一优先级完成有界诊断：19个状态、228条分支，候选首H在A/current接力下均净增加3次成功，交互差0，无严格排序反转。条件组更新未触发，本批结束。完整数字与下一步建议见[结果报告](execution_horizon_continuation_results.md)。

## 判断实验

使用原H25模型及Round4 A，候选为已训练的current step200。该候选直接从原A一次VLA响应的已有特征计算H，两个策略共用同一8040服务和动作生成路径。A、VLA与候选权重在判断阶段均冻结。

只使用训练bank初态：Task0–9，每任务按IDs300–303顺序沿A贪心轨迹寻找首个A/候选H分歧，每任务最多保留2个root，总上限20。无分歧的episode保留记录；不按终局成功与否选root，不使用dev或预留测试状态。

每个root固定同一已生成动作chunk，比较四种情况：首H取A或候选，之后全程由A或候选继续选择H。每种情况3次配对重复，最多240条分支。每个分支从相同bank初态重放源轨迹记录的全部float32环境动作（含等待），恢复同一真实根状态；随后执行固定chunk的对应前缀。后续VLA随机流按repeat和call序号配对。NFE10/10、resize224、等待10步，原候选归一化保持。

记录每root两种接力下“候选首H减A首H”的成功差、实际RPC差，以及排序反转和整体交互变化。重构根状态的开销不计入续段性能，首个生成chunk的共同RPC成本单列。计数仅用于局部机制判断，不作为新模型成功率。

## 依据结果推进

若多个root的局部收益在换成候选接力后反转或消失，且整体结果支持这个方向，就推进一次带A参考的完整轨迹组更新；重点改变优势估计与当前策略覆盖，不再扩大同类Q^A单点标签。若未出现这个模式，则不将continuation差异当成已证实主因，不重复扩大诊断采集。

后续组更新如启动，复用已有current小头，A/VLA冻结，不增加网络、critic或训练轮次扫描；同一训练初态生成A参考和候选完整轨迹，使用实际成功与RPC回报。保持原开发100初态上与同批A比较，只有明确胜出的唯一候选才进入预留200初态。具体启动设置和结果追加到本文件。

该条件满足时，固定采用20组训练初态（每task IDs310/311），每组1条A greedy参考和3条冻结current step200行为策略的sampled完整轨迹，共80局，组内交错运行。首call因`history_valid=False`没有可训练残差，保持A greedy并排除actor更新；其余候选decision保存实际用于采样的概率、log-prob、H、原始feature、同状态A logits和真实duration/RPC。A参考轨迹仅用于组回报基准，不作为候选on-policy样本。

回报固定为`success × [1 + 0.02 × (RPC_A − RPC_i) / max(1秒, RPC_A, RPC_i)]`，优势只减去包括A在内的组均值，不除以组内标准差，避免放大微小耗时差。使用clipped ratio损失（clip0.1）、A→候选KL权重0.05，组间与候选轨迹间等权，轨迹内平均。仅更新原current四个小头参数，保留原归一化；lr1e-4、4 epochs、一次更新，不按训练回报扫描checkpoint。这是待条件触发的有界pilot配置，尚未执行，也不宣称复现BCP。

用户要求只保证逻辑和输入输出：本批仅验证同根动作/状态对应、候选H和实际运行，不扩充测试矩阵或一般审计。任何当前错误只作最小修复；原A服务和已有数据保留，阶段进度通过飞书汇报。

## 启动记录

2026-09-09 00:14（北京时间），`140d6ea`已推送main，并以独立代码快照在原服务器启动。运行脚本为`scripts/probe_execution_horizon_continuation.py`，tmux为`h25_continuation`，使用原8040服务。输出目录：

`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/continuation_diagnosis_20260909_v1`

00:58:33整批完成，`continuation_probe.exit=0`，`probe/summary.json`及全部逐分支JSON已生成并归档。Task5仅一个分歧root，因此共19个；不补采凑满上限。AA/CA/AC/CC成功52/55/50/53，各57次；严格反转0。未启动80局组采样、参数更新、开发或预留测试。运行阶段的飞书通知与15分钟任务心跳用于本批跟进，结束后清理心跳。

## Limitations

Q^A是合法的局部评价，排序变化是待检验假设。root按H分歧筛选，样本量有限，配对种子只减少部分随机性；结果不能作为总体成功率或统计优劣证明。此前开发集已多次使用，预留IDs346–365仍未使用。
