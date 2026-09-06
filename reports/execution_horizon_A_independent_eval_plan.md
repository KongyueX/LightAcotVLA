# 当前模型与原始ACoT-VLA历史结果对比

2026-09-06。仅运行当前完整系统（H25动作模型＋predictor A），复用已完成的原始ACoT-VLA 1000局结果，评估整体性能变化。原版不重测，不重新训练或选择checkpoint，不做模块消融。

## 样本与协议

- 沿用历史原版协议：LIBERO-10、10任务×100 trials，episode0–99，官方50个预置初态循环两次，总计1000次评测、500个不同初态。
- seed7、CoT NFE10、图像尺寸224、初始等待10、offset0；当前模型final NFE10。原始c5c08fc实现默认final NFE10，但历史summary未单独保存该字段，配置核对将如实标出缺失记录。
- 新模型每次call选择H5/10/15/20/25；原始历史数据来自50999、纯原版c5c08fc、无predictor、输出10步并按默认执行5步。5步属于历史原版配置，不是本轮新增的消融设置。
- 不更换初态分布，不使用额外生成的新bank；按照历史raw rows匹配task、episode和initial_state_id，不把不同样本或200局旧A成绩直接用来计算本轮提升。
- 只运行新模型一个评测进程，不并发其他GPU计算；不根据partial成绩调参。

新系统配置为`acot_libero_long_chunk_h25`，服务端口8040。H25 policy：

`/root/autodl-tmp/acotvla/checkpoints/acot_libero_long_chunk_h25/h25_from_h20_lr1e5_seed42_split42_e2f9b27/5000`

A sidecar：

`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/ordered_dynamic_round4_a5e60b1_20260905_v1/predictor_seed7_lr1e4`

原始历史系统配置为`acot_libero_action_cot_explicit_implicit_co_fusion`，代码c5c08fc，checkpoint：

`/root/autodl-tmp/acotvla/checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/acot_libero_long_run1/50999`

复用的历史结果目录：

`/root/autodl-tmp/acotvla/execution_horizon_v2p/eval_formal_pure_base_c5c08fc_original_h5_fixed_h9_10tasks_100trials`

只读取其中`mode=original`的1000行，忽略同目录的fixed_h9数据。原始结果为927/1000=92.7%，policy/RPC/整局均值5.687400/6.290472/16.091163秒。本次新模型1000局已完成，成功913/1000=91.3%，policy/RPC/整局均值1.771632/1.961989/11.230149秒，详见[完整结果](execution_horizon_current_vs_original_10x100.md)。

本轮输出根目录：

`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/current_vs_original_history_20260906_v1`

## 输出与改进分析

主要报告新完整系统相对原始ACoT-VLA的成功数/成功率变化、真实policy/RPC/整局耗时下降；同时报告calls、predictor开销、每任务结果、配对救回/退化及对应初态。差值代表整体系统改进，不能单独归因于predictor或长chunk某一模块。复用现有paired McNemar与task/initial-state cluster bootstrap，最终只运行一次统计汇总；不能用“不显著”替代非劣证明。

时间增减与区间属于跨历史运行参考，不能排除硬件、软件或负载变化；不宣称同一次运行、同一时间条件下的严格计时对照。历史与当前summary/config分别保存，不伪装成同一实验。该局限通过报告说明，不通过重跑原版扩大本轮工作。

成功和失败局全部纳入主要时间均值，双方都成功的子集仅作描述性补充。记录selected H与尾部截断的execution H、previous-H转移、episode progress分段，以及已计算好的ordered选择概率/entropy/margin。只记录小型诊断，不新增VLA调用或导出大型prefix特征。

改进分析关注退化是否集中于特定任务、阶段与H切换，以及耗时是否主要受重规划次数或单次计算影响。选择概率不是成功概率；不从终局失败断言某次H造成失败，不用它虚构success ECE/Brier或反事实false-long rate。

评测过程中不根据partial成绩调整参数。完整结果后区分“当前数据支持的结论”和“需要进一步实验验证的改进假设”；本轮授权不包含自动启动新的训练实验。

## 执行与汇报

当前仅通过8040调用新H25＋A；原始重测队列、原始server和额外初态生成已停止，已落盘文件保留。评测输出`eval_10x100`，后处理通过`--reference-eval-dir`读取历史原版，输出`historical_analysis`。必要逻辑检查后直接运行正式1000局，不另跑重复性能smoke。

每30分钟飞书发送真实完成局数、成功数、运行状态和按实际吞吐更新的时间估计；完整结束后另发完整结果与改进建议。保留所有原始结果、模型和日志。
