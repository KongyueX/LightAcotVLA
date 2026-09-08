# 条件化视觉query与动作专家hidden：三组完整结果

2026-09-08。三组共300局开发闭环已全部完成：本批A成功90/100，条件化视觉query成功89/100，expert hidden成功88/100。两个候选的平均RPC和整局耗时均高于A，未达到晋级条件。本批保留原A，未运行预留200初态测试，也未启动组合模型。

## 同协议闭环结果

LIBERO-10，同一bank IDs336–345，每任务10个初态；三组逐episode交错，共100初态×3模式。冻结同一H25动作模型和A原参数，seed7、CoT/final NFE10/10、resize224、等待10步。三个实验服务驻留同一GPU，每次仅执行一个RPC。耗时均包含全部成功和失败局，正常评测不导出大特征缓存。

| 组别 | 所选checkpoint | 成功 | Calls/局 | Policy秒/局 | RPC秒/局 | 整局秒/局 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Round4原权重 | 90/100 | 16.98 | 1.855680 | 2.065527 | 11.089844 |
| 条件化视觉query | step75 | 89/100 | 17.79 | 1.949358 | 2.165828 | 11.502418 |
| Expert hidden | step0 | 88/100 | 18.82 | 2.030752 | 2.255192 | 11.586557 |

| 相对本批A | 救回/退化 | 成功率差 | Policy变化 | RPC变化 | 整局变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 条件化视觉query | 5/6 | -1个百分点 | +5.05% | +4.86% | +3.72% |
| Expert hidden | 5/7 | -2个百分点 | +9.43% | +9.18% | +4.48% |

表中A是本批重新运行的对照；此前其他批次的93/100不参与这些差值。新模块、hidden保留和相关服务计算均包含在本批实际计时内；调用次数与轨迹也改变，因此这些增幅不能当作孤立的模块延迟。

## 训练与选模

300个训练root（每任务IDs300–329）和60个早停root（330–335），每root五H×五次配对A-continuation，共9000分支。每个root的完整prefix、normalized输入、expert hidden和固定根动作在同一次真实调用捕获，随后执行该根动作的不同H前缀产生对应标签。训练与早停按episode划分，两候选共用全部数据。

| 模块 | 新增参数 | 更新次数 | Best step | 早停缓存成功净增 | 早停缓存RPC差/根状态 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 条件化视觉query | 65,792 | 650 | 75 | +5/300次配对分支 | -0.043457秒 |
| Expert hidden | 82,240 | 650 | 0 | 0/300 | 0秒 |

两组均实际完成650次更新，仅优化新增参数，A原参数保持不变；Adam lr1e-4、训练batch64、seed7。目标为配对成功差减0.02倍RPC秒差，加0.05倍对A的KL约束。验证按部署的batch1逐样本推理，step0和每25步参与选择，先比较root等权greedy成功差，再比较RPC差。

Query学到了非零增量，早停缓存上的改善未转化为超过A的闭环结果。Hidden也完成了训练，但所有候选checkpoint在早停准则下都未超过step0，所以最终部署的是零输出初始化。它的88/100是该初始化推理路径的成绩，不能称为训练学到hidden信息后的收益或退化。

## 逐任务结果

每任务每组10局。

| Task | A | Query | Hidden |
| --- | ---: | ---: | ---: |
| 0 | 10 | 10 | 10 |
| 1 | 10 | 10 | 10 |
| 2 | 9 | 9 | 9 |
| 3 | 10 | 9 | 8 |
| 4 | 8 | 8 | 8 |
| 5 | 7 | 7 | 7 |
| 6 | 9 | 8 | 9 |
| 7 | 10 | 10 | 8 |
| 8 | 8 | 9 | 10 |
| 9 | 9 | 9 | 9 |

Query救回：2/336、4/339、6/342、8/337、9/340；退化：2/340、3/338、4/338、6/338、6/345、9/342。

Hidden救回：2/336、4/339、8/337、8/342、9/340；退化：2/343、3/338、3/339、4/342、7/338、7/341、9/345。

## 结论与保留状态

本批没有证据替换A：query的已训练增量未提高完整成功数或降低实际耗时；hidden的训练与选模未产出优于初始化的权重。按预先规定的成功数优先、平局RPC次优准则，没有候选晋级，预留IDs346–365保持未使用。本批结束，不追加训练轮数、参数扫描或组合试验。

实验端口8050/8051/8052已关闭，原8040端口A服务仍运行；新旧权重和全部数据保留。模型、采集及评测代码8e0c0ad，训练及控制器代码5b70d1d；评测和控制器均正常退出。

## 可复查材料

- [实验协议](execution_horizon_architecture_experiment.md)。
- [总摘要](../results/execution_horizon_architecture/three_way_20260908/summary.json)、[配对分析](../results/execution_horizon_architecture/three_way_20260908/development/paired_analysis.json)、[300局CSV](../results/execution_horizon_architecture/three_way_20260908/development/rollout_rows.csv)、[评测配置](../results/execution_horizon_architecture/three_way_20260908/development/run_config.json)。
- [Query训练摘要](../results/execution_horizon_architecture/three_way_20260908/training_visual_query/summary.json)、[Hidden训练摘要](../results/execution_horizon_architecture/three_way_20260908/training_expert_hidden/summary.json)。

服务器目录：`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/architecture_query_hidden_20260908_v1`。两个所选权重分别位于`training_visual_query/params`和`training_expert_hidden/params`，末步权重在对应`last/params`；原始采集在`features_train`与`features_early_stop`。

## Limitations

单训练seed、每组100个开发初态，多轮研究复用该开发集。Query相对A的成功率差95%配对bootstrap区间为[-8,+6]个百分点，hidden约为[-11.03,+7]个百分点，均不能据此宣称统计优劣或非劣；独立预留测试未运行。

标签只估计当前H变化后由A继续的局部回报，与候选全程部署不同。第一组借鉴了ThinkProprio的状态指导读取及全局信息保留，但没有复现其VLM前筛选和状态词表编码；本结果不能外推为该论文思路无效。Hidden所选step0没有学到输出增量，其闭环差异也不能单独证明内部特征没有价值，或被唯一归因于某项数值操作。
