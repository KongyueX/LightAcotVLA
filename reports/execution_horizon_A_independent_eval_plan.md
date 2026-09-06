# A版独立完整性能评测

2026-09-06。冻结当前A，不重新训练或选择其他checkpoint；本轮只评测性能并分析改进方向。

## 样本与协议

- LIBERO-10，10任务×100个新增有效初态；A与Fixed H5各1000局，共2000局。用户若在正式运行前明确选择10×20，则缩小到各200局并记录变更。
- 使用LIBERO相同任务BDDL的reset采样器，seed=20260906；每任务保留50个官方预置作来源记录，另外生成100个有效、初始未成功、姿态不重复的reset样本，测试仅使用新bank中的ID50–149。
- 新测试姿态与此前0–299 bank及较早0–99 bank的姿态做一次去重核对；不同bank中的相同数字ID不代表同一状态。这是额外reset抽样的独立评测，不称为官方固定预置初态benchmark。
- 测试bank仅用于评测和结果分析，不参与训练、校准或checkpoint选择；A和H5在每个完全相同的初态上配对。逐对交错执行，交替先A或先H5。
- H25 policy、seed7、CoT/final NFE10、warmup各模式1次、图像尺寸224和10步初始等待保持。A每次call从H5/10/15/20/25选H；baseline使用同H25 policy并固定执行H5。

模型配置为`acot_libero_long_chunk_h25`。H25 policy：

`/root/autodl-tmp/acotvla/checkpoints/acot_libero_long_chunk_h25/h25_from_h20_lr1e5_seed42_split42_e2f9b27/5000`

A sidecar：

`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/ordered_dynamic_round4_a5e60b1_20260905_v1/predictor_seed7_lr1e4`

本轮输出根目录：

`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/independent_A_20260906_v1`

## 输出与改进分析

主要报告成功数/成功率、真实policy/RPC/整局耗时；同时报告calls、predictor开销、每任务结果、配对救回/退化及对应初态。复用现有paired McNemar与task/initial-state cluster bootstrap，最终只运行一次统计汇总；不能用“不显著”替代非劣证明。

成功和失败局全部纳入主要时间均值，双方都成功的子集仅作描述性补充。记录selected H与尾部截断的execution H、previous-H转移、episode progress分段，以及已计算好的ordered选择概率/entropy/margin。只记录小型诊断，不新增VLA调用或导出大型prefix特征。

改进分析关注退化是否集中于特定任务、阶段与H切换，以及耗时是否主要受重规划次数或单次计算影响。选择概率不是成功概率；不从终局失败断言某次H造成失败，不用它虚构success ECE/Brier或反事实false-long rate。

评测过程中不根据partial成绩调整参数。完整结果后区分“当前数据支持的结论”和“需要进一步实验验证的改进假设”；本轮授权不包含自动启动新的训练实验。

## 执行与汇报

新bank在CPU无渲染生成；代码仅增加可选的bank读取、交错模式和小型诊断，原eval默认及恢复行为保留。只做对应逻辑检查，正式评测的第一对运行承担实际数据路径确认，不另跑重复性能smoke。测试中不并发GPU工作。

每30分钟飞书发送真实完成局数、成功数、运行状态和按实际吞吐更新的时间估计；完整结束后另发完整结果与改进建议。保留所有原始结果、模型和日志。
