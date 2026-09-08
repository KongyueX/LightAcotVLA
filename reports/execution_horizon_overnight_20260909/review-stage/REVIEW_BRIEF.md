# 结果驱动的研究复盘

本轮是用户授权的六小时模型迭代，不是论文投稿准备。研究复盘只在新的实质实验结果产生后运行，最多四轮；同一个reviewer延续上下文，不按心跳重置或重复无变化审查。

使用ARIS auto-review-loop的默认Codex reviewer路线：新建独立`gpt-5.6-sol`、`xhigh` reviewer，之后继续同一agent。将完整prompt与原始response保存在项目`.aris/traces/auto-review-loop/2026-09-09_run01/`（不提交），在这里的REVIEW_STATE.json记录agent_id、round、verdict、待完成实验和截止时间。若reviewer不可用，明确记录，不伪造评分或已接受结论。

每次要求reviewer直接读取研究合同、实现变更、原始训练/early/开发JSON及CSV，核对本轮到底学到了什么，哪些结论不受支持，以及截止09:12前最值得执行的一项最小改动。只关注影响方法含义、实际输入输出和结果判断的真实问题，不扩展防御性检查或论文级大规模实验。评分只描述本轮研究证据与下一步就绪程度，不解释为论文可投稿。

输出：Score、Verdict（ready/almost/not ready）、关键证据、仍未解决的问题、唯一优先下一步、Memory update。同家族结果记录`review_independence: same-family`、`acceptance_status: provisional`，不能标为跨家族accepted。

已验证边界：旧A默认保留，once冻结备选；新反馈的任何改进需相对同批A及遮蔽反馈控制解释，实际触发率/成本不同则不能唯一归因新信息。局部训练/early收益不能替代完整闭环，选中零输出模型不能证明训练后的信息表示失效。时间到点时必须报告未完成范围，不用评分掩盖缺失实验。
