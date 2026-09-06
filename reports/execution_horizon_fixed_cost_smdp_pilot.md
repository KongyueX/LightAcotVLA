# 固定正时间成本的单次末层PPO

2026-09-07。用户授权分析后自动迭代。上一批三轮MC critic预热＋末层PPO已完整结束，validation为90/92/92成功，均未超过A的93/100与RPC1.833013秒；本次是独立的一个新候选，不追加旧批次轮数。

## 动机与唯一变化

原预算3秒高于实际随机采样耗时，使eta由0.02降到0.016065、0.008010，最终为0，第三轮actor不再优化RPC成本。该设置符合原代码逻辑，但没有持续推动超过A的真实速度。

本次保持原初始eta=0.02，设置现有参数`--dual-learning-rate 0`。actor始终使用`A_success − 0.02 × A_cost`，再按原流程统一归一化。其余学习率、MC预热10epochs、PPO4epochs、KL、batch、输入及可训练层完全不变，不新增H规则或VLA计算。

不把greedy验证的1.833秒直接用作随机训练约束：训练/验证模式和初始状态集合不同。本次不读取验证标签参与梯度训练，也不扫描eta值。

## 数据与自动流程

- 复用原首轮100条完整采样轨迹：`ROOT/last_block_smdp_4574577_20260907_v1/experiment/round01/collection`。
- 从该采集`run_config.last_block_smdp_params`绑定的原step0 checkpoint开始，初始actor为原A，eta为0.02。不从末轮checkpoint配旧数据训练。
- 不新初始化、不重采，只运行一次现有`train_last_block_horizon_smdp.py`；所有新输出写独立目录，源数据和旧结果不改。
- 用新checkpoint做同official20–29的100局greedy validation；若actor未变化，复用其A参考结果。
- 仍按原`(success_count,-RPC)`优于A且RPC≤3秒的条件判定是否进入唯一official0–19 final200；未胜出就结束，不追加第二次更新或扫参。
- 若进入final，比较同keys的当前A200=180成功、原始历史200=187成功，并单列更早A189/200的跨运行差异；不重跑原始模型。

自动入口为`scripts/run_fixed_cost_horizon_smdp_pilot.py`。沿用现有49412 SSH、8040 VLA服务和Python3.11环境；本地改动代理push GitHub main后同步独立代码快照，不建分支。进度与结果继续飞书汇报。

`ROOT=/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19`。

## 启动与训练记录

2026-09-07 04:24（Asia/Shanghai）启动，代码及计划提交`6f94c5c`已通过本地代理推送GitHub main并同步服务器`LightAcotVLA_6f94c5c`。4项入口/命令/单次流程检查通过（0.05秒），未增加数据重采或VLA smoke。

- `STAGE=ROOT/fixed_cost_smdp_6f94c5c_20260907_v1`，实际输出为`STAGE/experiment`；外层`controller.log/exit`，内部`status.json`、`training/`、`validation/`及条件`final/`。
- tmux `h25_fixed_cost_smdp`，controller PID42695（shell42691）。VLA仍为原服务PID26134/8040，未重启或改权重。
- 04:26完成训练：4epochs/136updates全部接受、`actor_changed=true`、old-policy KL0.003704、无回退、耗时100.78秒；eta before/after均精确为0.02。
- checkpoint为`STAGE/experiment/training/checkpoint`，04:28已进入100局greedy验证并完成7局，尚无完整性能结果。
- 继续沿用30分钟飞书自动任务`h25-transformer`，已切换到本单次试验。只有该候选胜出才进入final200；结束后交付结果，不追加第二次训练或扫权重。

## Limitations

固定0.02与原首轮实际0.016065差距较小，本次是低成本目标设置试验，不保证提高成功率，也不能用单次结果排除所有成本建模方法。输入仍没有新历史观测或执行反馈。验证状态已反复用于开发，不构成独立泛化或统计非劣证明。所有模型都按成功率和真实时间比较，是否多选H25不是优劣标准。
