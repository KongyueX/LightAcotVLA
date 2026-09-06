# 有序H选择器的SMDP闭环训练试验

2026-09-06。以现有H25＋A为起点，训练H选择而不是增加H10后处理规则。第一批冻结VLA和A的两层编码器，仅更新有序输出头及轻量critic；不改动作生成、夹爪、视觉主干或网络层数。

## 最终结果：本批未超过A

2026-09-07 00:34，两轮采集、更新和验证全部完成，controller.exit=0、summary complete。两轮均完成实际参数更新、各4 epochs接受，没有KL回滚；但验证成功率均为93/100，与A相同，RPC时间增加，因此保留A、未运行额外final200。不是3秒预算阻止候选，而是同成功数下没有成本收益。

以下仅比较官方IDs20–29、10tasks×10的同100个验证状态，时间单位秒/局，不混入A的200/1000局或训练随机探索成绩。

| 模型 | 成功数 | calls/局 | policy | RPC | 整局 | 对A救回/退化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A参考 | 93/100 | 15.930 | 1.658487 | 1.833013 | 10.904698 | — |
| SMDP Round1 | 93/100 | 16.040 | 1.704574 | 1.890671 | 10.889177 | 2/2 |
| SMDP Round2 | 93/100 | 16.740 | 1.781263 | 1.979155 | 10.986659 | 2/2 |

Round1/2相对A的RPC分别增加3.146%/7.973%，整局分别变化−0.142%/+0.752%。Round2在Task8由8/10提高到10/10，但Task4从9/10降到8/10、Task6从8/10降到7/10，净成功数不变。两轮 `eligible=false`，最终 `selected_system=A`、`final_run=false`。

### 训练与实际决策变化

| 项目 | Round1 | Round2 |
| --- | ---: | ---: |
| 新采随机轨迹／成功数 | 100／92 | 100／92 |
| 采集RPC秒/局 | 2.290280 | 2.336336 |
| 接受epochs／optimizer updates | 4／64 | 4／64 |
| 相对采样旧策略KL | 0.0015494 | 0.0018241 |
| 更新后成本乘子eta | 0.0129028 | 0.0062662 |
| CPU训练秒 | 2.111 | 2.304 |
| 验证中不同于当前状态A raw argmax的选择 | 52/1604（3.24%） | 177/1674（10.57%） |

参数确实更新，并非没有学习或全部回滚。上述不同率是在各候选自己访问的状态上与A raw argmax比较，不是两条轨迹逐call配对。Round2相对A的anchor KL约0.00535；H选择变化未转化为总体验证收益。

验证selected H5/10/15/20/25：A为144/44/163/540/702，Round1为141/54/190/530/689，Round2为239/47/213/434/741。H5和H25都增加，不能概括成单纯偏短或偏长；Round2在已访问状态上最常见的改选为raw H20→H25，共100次。

客户端actor/critic后处理为Round1/2的0.32448/0.32791毫秒/call、5.205/5.489毫秒/局，发生在RPC计时结束后，不解释主要RPC增量，也不重复计入policy/RPC。采样RPC低于3秒预算，所以对偶规则降低了成本乘子；该预算只限制成本上限，不自动保证比A更快。

### 本批结论与保留内容

已完成200条新训练轨迹及两次100局验证，建立了真实采样概率、实际duration和PPO更新链路；未取得超过A的验证结果。保留原A默认模式、两轮NPZ和全部原始记录，不追加epoch、阈值扫描、1000局或动作模型训练。

该结果只约束本次两轮、小数据、冻结encoder、线性输出增量及既定目标组合，不能推出RL整体无效。下一步应先结合现有优势/critic/决策记录区分学习信号与表示限制，再考虑反馈或动作接续方向，不把继续训练更多步视为必然改进。

## 目标与方法

目标是在保留明显速度收益的前提下提高完整任务成功率。H5/10/15/20/25全部可选，没有H10安全参照、概率阈值或手写候选mask；短H与长H的代价通过实际闭环后果学习。

冻结A每次call已计算出的256维`temporal_feature`和4个`ordered_continuation_logits`，客户端actor使用：

`z = z_A + feature @ delta_W + delta_b`

`delta_W/delta_b`初始化为0。这等价于在固定表示上微调A原有linear有序输出头，初始分布及贪心H与A一致，不是新建一个替代A的MLP actor。critic为64维tanh MLP，分别预测任务成功回报和剩余RPC秒数；其数值不用于过滤H。首轮不增加新的反馈输入，因此不能宣称解决了原编码器缺失的历史信息。

训练采用clipped PPO，参考[DEHP（Zhao等，2026预印本）](https://arxiv.org/html/2606.11408v2)的SMDP时序处理：chunk间discount为`gamma ** actual_duration`，GAE递推系数为`(gamma * lambda) ** actual_duration`。实际duration取相邻decision环境步差，末段取episode最终steps减decision起点，不把计划H或末段尚未执行动作计入。整局成功/超时均结束bootstrap；若启用折扣，末段成功奖励按该段内实际终止位置折扣。

本试验默认`gamma=1`，对应未折扣终局成功目标；`lambda=0.995`按环境步计算。与DEHP原实验的状态型策略、critic及初始化不同，这里是基于冻结VLA特征的项目适配，不复制论文性能结论。

## 成功与成本

- success reward：仅终局成功为1，失败为0。
- cost reward：每个call实测RPC wall秒数，包含本次VLA及服务器sidecar计算，不用calls替代真实时间。
- 两路GAE先合成`A_success - eta * A_cost`，再统一标准化；不分别标准化后改变两路量纲。
- `eta`初始0.02，根据训练整局RPC均值相对3秒预算进行小步对偶更新，学习率0.01、范围[0,0.05]。这是有限样本的成本调节，不是严格约束保证；验证阶段另检查实测预算。
- 保留相对A分布的KL约束，减少有限数据更新过大。H10不享有特殊地位。

单次更新最多4 epochs，batch128，actor/critic学习率1e-4/1e-3，PPO clip0.1，anchor KL权重0.1、target KL0.02，梯度裁剪1；超出KL限制时回退该epoch并结束该次更新。不扫描参数。

## 有界执行链

| 阶段 | 数据与规模 | 用途 |
| --- | --- | --- |
| Round1采集 | 10tasks×10，已有bank IDs100–109 | 从A初始分布随机采样H，收完整on-policy轨迹 |
| Round1更新/验证 | CPU更新；官方IDs20–29，共100局，贪心执行 | 判断完整成功率与RPC时间 |
| Round2采集 | 同bank IDs110–119，共100局 | 使用Round1输出重新采集，不混旧on-policy批次 |
| Round2更新/验证 | 同参数；相同预留验证100局 | 选择最终候选 |
| 最终候选测试 | 仅候选优于参考时，官方IDs0–19，共200局 | 报告完整系统性能，不再用于选模 |

所有运行均LIBERO-10、seed7、CoT/final NFE10、wait10、resize224、预测25步、warmup1；采集与计时评测串行使用现有8040服务。训练bank为`ROOT/round3_independent_5e18446_v1/initial_state_bank_0_299`，已确认10任务均包含IDs100–119；不生成新bank。只有训练使用该bank，验证及最终测试使用官方初态。

采集使用`--ordered-smdp-sample`，记录实际采样动作的old log probability、完整分布、两路value、原A logits及256维特征；无额外VLA调用、不保存大prefix张量。验证/测试不加sample flag，使用贪心H。初始随机采样轨迹与A贪心评测不是同一执行策略，不能把其成功率差直接当训练退化。

历史贪心A的1000局没有本轮随机采样行为概率，不能直接作为PPO样本。trainer要求当前完整rollout、sample模式、匹配的输入checkpoint与记录概率；每轮只用自己采集的批次。旧结果只用于相同keys的参考比较。

## 选择与停止

当前A官方IDs20–29的验证参考为93/100、RPC1.833013秒/局，各task成功10/10/10/10/9/9/8/10/8/9。参考来自`ROOT/current_vs_original_history_20260906_v1/eval_10x100/rollout_rows.csv`，不重跑A或原版。

候选验证RPC须不超过3秒/局，再按`(成功数, -RPC均值)`与参考及另一候选比较。只有优于参考的候选才运行最终200局；否则保留A，记录本次无收益，不重复测A作为“新结果”。最终测试无论结果好坏均报告，不自动扩大1000局或启动下一种算法。

最终对照使用同task/episode/initial_state_id的当前A200局（180成功）及原始50999历史200局（187成功）；另列旧A189/200的跨运行差异。H10规则188/200属于另一已完成工程试验，不作为PPO标签或强制模仿目标。

## 实现与运行边界

- 核心：`src/openpi/execution_horizon/ordered_smdp.py`。
- 训练：`scripts/train_ordered_horizon_smdp.py`。
- 有界runner：`scripts/run_ordered_horizon_smdp_pilot.py`。
- eval显式模式：`ordered_smdp --ordered-smdp-params <selector.npz>`。
- 现有eval默认模式、A、H10滞回及旧V2-P selector全部保留；新模式不调用旧selector的H10 safety mask。

服务继续使用H25 checkpoint `acot_libero_long_chunk_h25/h25_from_h20_lr1e5_seed42_split42_e2f9b27/5000`与A `ordered_dynamic_round4_a5e60b1_20260905_v1/predictor_seed7_lr1e4`。本试验NPZ仅包含输出头增量、critic与元数据，不修改这些服务权重。

`ROOT=/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19`。沿用现有Python3.11环境；CPU训练不与计时rollout并发，不重装环境。首批只做概率/GAE/更新/接口的基本检查，随后直接进行正式采集。阶段与最终结果按此前约定飞书汇报。

## 启动记录

代码提交`7da6e48`已通过代理推送main并同步服务器快照。39项相关CPU测试通过（5.60秒），包含概率初始化、采样记录、GAE实际duration、PPO更新/KL回退及两轮执行边界。

2026-09-06 23:18启动tmux `h25_ordered_smdp`，controller初始PID18401、首collector PID18403；继续使用原A服务PID1937/8040，不重启或修改其权重。输出根目录为`ROOT/ordered_smdp_7da6e48_20260906_v1`，外层`controller.log/exit`，阶段状态为`experiment/status.json`；各轮在`experiment/round01`、`round02`，保存collect/train/validate日志与exit、collection/training/validation结果。

初始化及两轮collect/train/validate均exit0；两轮输入checkpoint、参数输出、metrics和CSV均保留。controller于2026-09-07 00:34正常结束，未运行final200；原A服务PID1937/8040未重启或切换权重。

## Limitations

这是head-only PPO小试，不是完整Transformer或VLA的RL微调；不能补足被冻结表示遗漏的实际执行反馈。仅两轮、单seed，验证/最终状态此前已用于模型开发，不能宣称独立最终泛化。随机采样训练与贪心评测存在策略差异，效果以实际贪心闭环为准。原版时间为历史参考，旧/当前A同200keys有9局未解释差异，不将小幅变化当稳健结论。若当前动作生成的跨chunk不连续是主要限制，需另立动作接续训练试验，本批不捆绑实施。
