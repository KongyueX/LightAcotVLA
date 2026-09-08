# A与单次current介入：完整闭环对照

2026-09-09。用户授权“试一下单次介入”。200局完整开发对照已完成：单次介入94/100、原A93/100，救回2局、退化1局，RPC增加1.72%、整局耗时增加1.27%。70局各介入一次、30局无分歧；本批结束，保留A默认，未追加训练或预留测试。数字与结论见[完整结果](execution_horizon_single_intervention_results.md)。

## 固定策略与比较

原A为`ordered_transformer`。新策略为`ordered_feedback_current_once`，加载既有current step200，默认按A选择H；每局首次A/current建议H不同才采用一次current H，随后本局所有决策交还A。H建议相同不消耗机会，无分歧整局沿用A，每局重新获得一次机会。介入后不再计算current小头；所有决策使用原8040的一次VLA响应，不增加动作生成调用。

候选checkpoint：`replanning_diagnosis_20260908_v1/greedy_selection_9dca506/training_current/checkpoint.npz`。模型权重、归一化、H25动作模型与A均不变。previous actions、实际H和观测cache沿实际执行轨迹更新，不能在介入后沿用A反事实历史。

使用既有`initial_state_bank_0_365`的Task0–9、IDs336–345，共100个开发初态，每初态两种策略交错运行，共200局。seed7、NFE10/10、resize224、等待10步，H候选5/10/15/20/25；沿用原评测预算、终止和时延统计，不使用原诊断训练root作为开发结果。

比较实际成功数、配对救回/退化、policy总时间、RPC总时间与整局时间。记录单次介入时刻、A/current建议H及是否已介入；真实输出用于确认每局至多一次介入，避免把策略实现错误当实验结果。仅进行必要的语法和逻辑/输入输出验证，不扩展软件检查或扫描介入次数、阈值和checkpoint。

## 执行与结束

评测由`scripts/run_execution_horizon_single_intervention.py`运行，复用现有`run_stage`、配对结果汇总和飞书通知。远端输出目录：

`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/single_intervention_20260909_v1`

本轮固定开发200局，完成后归档完整结果并判断混合策略是否值得继续；不自动重训或扩大采样，不自动消耗预留IDs346–365。关键进展与最终结果通过飞书汇报，状态无实质变化时静默。

2026-09-09 01:12启动，01:48:33完成，`development.exit=0`。实现提交`e32fd51`已推送main并部署到独立代码快照；tmux为`h25_single_intervention`，评测初始PID161861。完整summary、paired_analysis、逐局/决策CSV和介入记录均已生成并归档。真实输出确认每局最多一次介入、非介入H来自A，介入后的765次决策没有再计算candidate。飞书启动与开发完成通知均获服务接受；本任务心跳在最终结果交付后删除。

## Limitations

一次介入是为了检验当前权重的局部改动能否转化为整局收益，不是最优介入次数的结论。若获益，只能说明该混合策略在本批对照中有效，不能单独证明持续介入就是原候选失败的原因。开发集已被多次使用，开发收益仍需后续独立测试；本轮不声称独立泛化成绩。
