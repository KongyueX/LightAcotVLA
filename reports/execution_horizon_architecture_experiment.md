# A、条件化视觉query与动作专家hidden：三组对照

2026-09-08。360个root与9000条配对分支已全部采集完成，进入两个新增模块的训练与三组闭环阶段，尚无新闭环成绩。按用户要求，只保留逻辑、输入输出对应及运行成功所需的验证，不扩充检查。

## 唯一结构变量

| 组别 | 保留部分 | 新增部分 |
| --- | --- | --- |
| A | 冻结H25 VLA、Round4两层global ordered predictor | 无 |
| visual_query | 同一VLA与A的全部原参数 | 状态与已有prefix全局上下文调制四个视觉query；共享256→256零初始化增量 |
| expert_hidden | 同一VLA与A的全部原参数 | 最后一次已有去噪调用的25×1024 hidden，经1024→64→256投影逐位置残差融入动作token；末层零初始化 |

两新模块分别训练，首个决策也有效，不采用旧history小头的history-valid输出门。保留原全局context、两层Transformer和H5/10/15/20/25有序输出头。两模块参数量单独报告，不把完整方案差异单独归因于信息类型。

第一组借鉴ThinkProprio的状态指导视觉选择及保留全局信息的思路；任务上下文来自A已有prefix表示，未复现其词表状态编码、双分支硬筛选或VLM前剪枝。Seeker提供状态调制query的结构先例。预测动作chunk不输入第一组query，不运行组合模型。

## 数据与训练

沿用上一批的初态划分：300训练episode（每任务IDs300–329）及60早停episode（330–335），每episode以outcome-blind reservoir采一个root，各五H×五配对A-continuation，共9000分支。完整prefix、normalized输入、expert hidden和固定根动作在同一次真实source调用捕获；随后直接执行该动作的不同H前缀并记录分支结果，两个新结构共用这批特征与标签。

旧缓存不用于本批监督：旧服务与新特征服务在同一输入下存在动作数值差异，无法保证新特征与旧分支标签严格对应。保留旧数据，使用新采集结果。特征导出仅发生在source reservoir保留的调用，后续分支使用不导出缓存的A；共同root导出耗时在配对差中抵消。

两组同seed7、Adam lr1e-4、训练batch64、固定650次更新，每25步及step0选模。验证逐样本以batch1推理，与服务部署的批次一致；GPU浮点logit差记录为诊断，step0的实际H选择需与A一致。只优化新增参数，A和VLA冻结。目标沿用`-E[paired success delta - 0.02 * paired RPC delta seconds] + 0.05 KL(pi_A || pi)`；按早停root等权greedy成功差优先、RPC差次优选择checkpoint，实际训练满650步。已有归一化输入原样使用。

## 闭环与停止条件

三组都做完整开发比较：同bank IDs336–345、10任务，共100初态×3模式=300局；同seed7、CoT/final NFE10/10、等待10步、resize224。三个服务端点逐episode交错，仅一个RPC执行，不并发评测；正常评测不导出完整缓存，新模块及hidden保留开销包含在实际policy/RPC/整局计时中。

即使step0入选或离线没有改善，该组也完成既定100初态闭环。开发按成功数优先、成功数相同再比较平均RPC，严格优于同批A的最优一个候选才进入未用IDs346–365的200初态对A测试。否则本批结束。原8040端口A服务和全部权重/数据保留，不追加组合、扫参或训练轮数。

入口：`collect_execution_horizon_architecture_features.py --fresh-paired`（复用`collect_execution_horizon_feedback.py --architecture-cache`）、`train_execution_horizon_architecture.py`、`run_execution_horizon_architecture_experiment.py`。新评测模式为`ordered_visual_query`和`ordered_expert_hidden`，分别使用独立sidecar端点；原默认评测行为保持不变。

服务器输出目录：`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/architecture_query_hidden_20260908_v1`。实验端口8050/8051/8052，原A服务8040保持。

## Limitations

标签只衡量当前H改变后由A继续的局部回报，不包含新增模块未来开销；实际全程收益由闭环衡量。开发初态已用于前批评测，只有预留346–365仍未使用。冻结专家hidden可能与已有表示冗余，也不等于校准置信度；原始prefix含视觉与语言上下文，条件化pool不预设某个槽必然对应物体或接触区域。
