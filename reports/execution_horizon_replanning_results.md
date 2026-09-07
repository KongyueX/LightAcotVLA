# 重规划诊断与部署一致选模：完整结果

2026-09-08。本批诊断和200局开发闭环均已完成。按实际部署argmax重新选择的current step200成功92/100，同批A成功93/100，候选RPC增加6.17%、整局耗时增加4.06%。未达到既定晋级条件，保留A，不运行预留200初态测试，本批结束。

## 完整闭环结果

沿用LIBERO-10开发bank IDs336–345，每任务10个初态；A和current step200逐episode交错运行，共200局。同一冻结H25动作模型与A服务，seed7、CoT/final NFE10/10、resize224、等待10步。所有成功和失败局均计入耗时均值，表中A为本批实测结果。

| 模式 | 成功 | Calls/局 | Policy秒/局 | RPC秒/局 | 整局秒/局 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 93/100 | 16.09 | 1.745050 | 1.940129 | 10.864317 |
| current step200 | 92/100 | 17.15 | 1.853359 | 2.059842 | 11.305574 |

候选相对A救回2局、退化3局，成功率差-1个百分点；policy时间增加6.21%，RPC增加6.17%，整局增加4.06%。客户端小头平均0.233毫秒/call，位于RPC之外、整局计时之内；额外调用和轨迹变化也影响耗时，不能把总耗时增幅归因于小头计算。

| Task | A成功/10 | current step200成功/10 |
| --- | ---: | ---: |
| 0 | 9 | 10 |
| 1 | 10 | 10 |
| 2 | 9 | 9 |
| 3 | 10 | 8 |
| 4 | 9 | 9 |
| 5 | 7 | 7 |
| 6 | 10 | 10 |
| 7 | 9 | 10 |
| 8 | 10 | 10 |
| 9 | 10 | 9 |

救回为Task0/344、Task7/339；退化为Task3/339、Task3/342、Task9/345。晋级规则为成功数优先、成功数相同再比较平均RPC，严格优于同批A才进入预留测试；本次成功数及平均时间均未优于A。

## 诊断如何决定本次改进

先对上一批history小头与A的9组成败相反初态回放18局，8组复现，Task9/345双方本次均成功而排除出介入。各组首次H分歧前物理状态、生成和执行动作最大差均为0，首次差异均为history提前重规划。边界命令变化与成败没有统一对应关系：部分退化局变化更小，部分救回局更大。

随后对8个有效root各做3组配对、共48条分支：从同一bank初态重放记录的共同动作前缀，根状态和动作一致后，只改变首次执行H，后续全部由history继续。保持history首H成功22/24，首H换为A成功20/24，配对救回1次、退化3次。结果不支持统一恢复A的首H，也不足以把全部退化归因为动作不连续；因此本批没有修改动作专家。

独立审计原60个早停root发现，旧history step450的期望loss改善，但实际部署argmax所选缓存分支仍为245/300成功，与A相同且RPC稍高。于是将唯一改进变量限定为checkpoint选择：使用部署NumPy forward的argmax，按root等权成功差优先、RPC差次优选择。

复用原300训练root、60早停root以及原650次更新序列，loss、模型结构、优化器、学习率、batch、seed和归一化不变。current/history均选step200，早停缓存分别较A净多4/300和2/300次成功，仅按早停选current进入本次开发闭环。其缓存表现改善没有转化为超过A的全程部署结果，说明仅调整选模准则尚不足以解决问题。

## 后续判断

当前证据支持保留A，并停止本批继续增加训练轮数或更换checkpoint。若继续研究，优先检验“改变一个H后由A接力”的局部标签与“候选全程选择H”的整局回报差异：先在训练初态上比较候选自身的配对整局回报，再决定是否更新模型。该方向尚未启动，也尚未证明是本次差异的主要原因。

## 可复查材料

- [本批完整摘要](../results/execution_horizon_replanning/greedy_selection_20260908/summary.json)、[配对分析](../results/execution_horizon_replanning/greedy_selection_20260908/development/paired_analysis.json)、[200局原始CSV](../results/execution_horizon_replanning/greedy_selection_20260908/development/rollout_rows.csv)、[评测配置](../results/execution_horizon_replanning/greedy_selection_20260908/development/run_config.json)。
- [诊断详情](execution_horizon_replanning_diagnosis.md)、[选模协议](execution_horizon_greedy_selection.md)、[有效单H试验摘要](../results/execution_horizon_replanning/diagnosis_20260908/first_h_probe_summary.json)。
- 服务器目录：`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/replanning_diagnosis_20260908_v1/greedy_selection_9dca506`。候选权重位于`training_current/checkpoint.npz`，history权重及所有旧数据均保留。

被动回放代码57b6293，有效单H介入32e9c1e，部署一致选模与评测9dca506。开发评测和控制器均正常结束，原A服务未替换。

## Limitations

本次为单训练seed、100个开发初态，成功率差95%配对bootstrap区间为[-8,+5]个百分点，不能宣称统计非劣或稳定提升。开发初态已用于前批评测和9组差异诊断；预留IDs346–365未使用。

9组回放是按前批成败差异选择的案例，单H介入使用配对后续随机流；22/24与20/24不能当作模型成功率。早停300次分支计数来自60个root，局部标签后续由A继续，与候选全程部署存在差异。本批结果也不能单独证明历史信息或动作接续训练总体无效。
