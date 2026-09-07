# Predictor checkpoint按部署决策选择

2026-09-08。本批已结束：部署argmax选出的current step200在100开发初态上成功92/100，同批A为93/100，RPC增加6.17%。未达到晋级条件，保留A，未运行预留测试；详见[完整结果](execution_horizon_replanning_results.md)。被动回放未支持“命令接续变化普遍导致失败”的统一解释，本项检验选模口径能否改善闭环表现。

## 判断依据

仅复用原60个早停root、每root五次paired结果，用部署版NumPy selector实际argmax选择H，得到：

| checkpoint | 所选H分支成功次数 | RPC秒/root | 相对A改H的root数 | 救回/退化 |
| --- | ---: | ---: | ---: | ---: |
| A | 245/300 | 1.608343 | 0/60 | — |
| current step450 | 246/300 | 1.595767 | 11/60 | 4/3 |
| history step450 | 245/300 | 1.613509 | 15/60 | 5/5 |

两个小头的期望分布目标均改善，但history实际argmax没有成功净增，RPC略增。这些是缓存Q^A分支标签的审计，不是新闭环成绩；它们支持检验按实际部署决策选checkpoint，不能预先保证收益。

## 唯一研究变量

复用原300训练root、60早停root、A初始化和current/history两套小头；loss、网络、优化器、学习率、batch、seed和归一化不变。精确重放原来的650次更新，按原log25的候选checkpoint选择；不采新数据，不延长更新序列。

原准则是最小化验证期望loss。新准则为：用部署版NumPy forward取argmax，先最大化root等权的paired成功差，成功相同再选择RPC差更低者。成功差由原整数paired计数还原成精确分数，避免浮点并列误排；step0参与。训练仍使用原可微期望loss。

两个输入版本仅按这60个早停root选择一个候选；若最高分不优于A，或实际参数等同于已评测旧checkpoint，直接保留A并复用已有结果，不重复评测或改测次优版本。

## 测试

原单H介入诊断结束后串行测试，避免并发污染计时。候选与A在同一100开发初态（IDs336–345、10任务）逐episode交错，保持原H25服务、seed7/NFE10/10。晋级按完整成功数优先、成功数相同再比较平均RPC时间，严格优于同批A才进入尚未使用的200预留初态（IDs346–365）对A测试。Policy时间、整局时间和配对置信区间一并报告，不要求三种时间同时下降。原A服务不自动替换，不追加权重扫描或训练轮数。

入口为`train_execution_horizon_feedback.py --selection-metric greedy`及`run_execution_horizon_greedy_selection.py`。默认`expected_loss`保持原行为。单H介入用于机制诊断，本项推进依据是上述独立的greedy早停审计，两类结果分别解释。

## Limitations

Q^A只评价当前H改变后由A继续的局部回报，与候选全程部署存在差异；300次分支计数来自60个root，不是300个独立闭环episode。有限早停集与单训练seed可能选中偶然较好的checkpoint，最终判断必须依据实际闭环，不能把选模目标改善当作成功率提升。开发初态已用于前批评测及9组差异诊断，本轮复用它们作为开发门槛；只有预留IDs346–365尚未用于评测或选模。

## 已完成选模与测试

两套小头均重放原650次更新，greedy准则均选step200。current在早停缓存标签上相对A净多4/300次成功、RPC差-0.055050秒，只改2/60个root的H；history净多2/300、RPC差-0.013474秒、改7/60个root。按早停准则选定current，参数与旧step450不同，仅该候选进入闭环。

有效单H试验结束后，串行完成A/current step200的100开发初态配对测试，共200局；两者成功数分别93和92，RPC分别1.940129和2.059842秒/局，整局分别10.864317和11.305574秒。候选救回2局、退化3局，未晋级，预留200初态未运行。评测及控制器均exit0。运行代码9dca506，服务器目录为`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/replanning_diagnosis_20260908_v1/greedy_selection_9dca506`；未重新采集训练数据或修改VLA，原A与全部材料保留。
