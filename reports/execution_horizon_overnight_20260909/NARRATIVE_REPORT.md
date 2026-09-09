# Predictor A：六小时窗口自动迭代结果

**本窗口没有找到可替换A的方案。** 最终固定事件策略的完整开发结果为91/100，原A为93/100；RPC增加7.08%，整局耗时增加2.84%。按预定停止条件保留A，结束当前候选路线。

授权窗口为2026-09-09北京时间03:12:30–09:12:30。所有实验作业在08:07:09前结束，随后完成结果复盘和归档；未为填满窗口追加训练或开发集调参。

## 完整闭环结果

| 策略 | 成功 | 平均A调用数 | 平均policy秒 | 平均RPC秒 | 平均整局秒 |
|---|---:|---:|---:|---:|---:|
| 原A | 93/100 | 16.10 | 1.741465 | 1.934455 | 10.825047 |
| 缓存上下文的事件策略 | 91/100 | 17.23 | 1.864488 | 2.071510 | 11.132152 |

候选救回2局、退化4局。Task6出现三个退化（10/10→7/10），贡献主要增时。候选执行298次事件观察、98次提前重规划；监视回调仅0.577毫秒/局，时间损失主要伴随额外A调用和失败轨迹出现。[完整对照与差异episode](refine-logs/round_3_results.md)

## 测试的问题与方法

问题是：动作chunk已经执行一部分后，新的本体感知是否有助于决定继续原计划或提前重规划，并改善成功与实际耗时的权衡？原H25动作模型和Round4 A始终冻结，A先给出名义H，监视器只可提前结束当前chunk，不延长H。

监视器为64隐藏单层MLP，预测提前重规划的效用差。共享输入包含A的256维缓存表示、起点proprio、原动作chunk、名义H和时钟信息；fresh额外使用当前8维proprio及相对起点差，masked遮蔽这16维。监督来自同一中途状态下continue/replan的配对真实续段，使用任务成功和幅度受限的相对RPC效用，后续均由A执行。部署不读取未来分支或仿真特权状态。

## 三个阶段得到的证据

| 阶段 | 完成范围 | 观察到的结果 | 决定 |
|---|---|---|---|
| 固定执行5步后观察 | 80 train+20 early root；600分支；fresh/masked各400更新 | early两动作均60/60成功，事后oracle仅约0.000185秒RPC余地；两模型均选step0 | 不重复评测等同A的零输出策略 |
| 夹爪事件后观察 | 48 train+19 early root；268分支；两模型各400更新 | fresh50与masked25在early恰好选相同6/19个介入，均36/38成功、RPC0.594491秒；continue为35/38、0.722538秒 | 未通过fresh超过masked的原条件；仅保留更简单的masked作后续验证 |
| 固定masked完整闭环 | A与候选在同100开发初态各跑100局 | 候选91/100对A93/100，RPC与整局均增时 | 拒绝候选，保留A |

合计完成868条配对续段、四个小头各400次更新，以及200局完整同批开发评测。续段不是独立完整episode，不能将868与200合并成模型成功率样本数。

第一阶段在无介入机会的early批次上选零输出有真实依据，训练拟合不能替代收益证据。第二阶段把观察移到夹爪命令转换后的两个动作之后，再向上取到A的5步档位，且要求k<H；跨chunk实际命令变化也纳入。源episode与规则先于结果确定，三个无合格事件的源episode照实保留，没有补入有利样本。[阶段1](refine-logs/round_1_results.md)；[阶段2](refine-logs/round_2_results.md)

事件early中的优势几乎全部来自Task6/332一个随机root，未能推广到最终候选的完整闭环。fresh和masked的early选择相同，只说明本轮未建立新proprio的额外价值证据，不代表两个函数在所有状态上等价。最终测试针对masked，不追认fresh条件通过。

## 最终决定与后续边界

原A继续作为默认模型。本窗口结束固定k5与本版事件MLP的迭代，不调整阈值、增加任务例外或强选训练末步。单次介入旧候选及本窗口所有权重、原始缓存和负结果均保留。

如未来继续研究，需要重新提出具有判别信息的输入或决策机制；本批没有验证新的视觉编码，也没有评估fresh的完整闭环。它们是尚未回答的问题，不是当前候选未达标后应自动追加的实验。预留IDs346–365始终未使用。

## 复现与文件

- [固定方案与阶段入口](refine-logs/EXPERIMENT_PLAN.md)、[完成清单](refine-logs/EXPERIMENT_TRACKER.md)。
- [R1采集汇总](../../results/execution_horizon_overnight_20260909/round1/collection/summary.json)、[R1训练汇总](../../results/execution_horizon_overnight_20260909/round1/training/summary.json)。
- [R2采集汇总](../../results/execution_horizon_overnight_20260909/round2/collection/summary.json)、[R2训练汇总](../../results/execution_horizon_overnight_20260909/round2/training/summary.json)。
- [R3完整汇总](../../results/execution_horizon_overnight_20260909/round3/summary.json)、[逐局结果](../../results/execution_horizon_overnight_20260909/round3/development/rollout_rows.csv)、[决策记录](../../results/execution_horizon_overnight_20260909/round3/development/decisions.csv)。
- 实现提交依次为`2550b4b`（固定k5）、`a6ca256`（事件对齐）、`97957a3`（冻结masked完整开发）。远端根目录为`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/overnight_20260909_031230`，各轮输出分开保留。RGB/NPZ原始缓存保留于本地及服务器；代码与轻量结果已归档。

## Limitations

每个root只有2或3次配对重复，early规模小，选模不确定性较大。R1与R2同时改变源episode和观察规则，不能把两批headroom差异唯一归因于事件时机。开发IDs336–345未参与本轮train/early，但在项目历史开发中已复用，不是独立最终测试。最终成功率差95%区间为[−11,+5]个百分点，RPC差区间[−190,+567]毫秒，整局差区间[−636,+1530]毫秒；因此不声称统计显著有害或等价，只依据缺乏改善的完整结果拒绝本候选。fresh未做完整开发，本结果不否定所有本体感知或事件反馈方法。
