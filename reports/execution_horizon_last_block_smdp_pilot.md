# MC critic预热与现有最后一层Transformer微调

2026-09-07。用户授权在分析后自动迭代。本批在上一组head-only PPO未胜出的基础上，扩大到现有最后一层Transformer、summary projection和有序输出头，并先预热critic。不增加Transformer层数、不训练VLA动作模型、不恢复H10规则。

## 依据

上一组两轮validation均93/100，与A相同，RPC由A的1.833秒变为1.891/1.979秒。actor确实更新，但仅为固定256维表示上的线性增量；critic从成功值0.9、成本3秒的常数起步，未独立预热。其success MSE由0.02419→0.02381、0.02334→0.02031，是对使用旧value产生的GAE targets的拟合，不等于真实成败预测质量。

[DEHP（Zhao等，2026预印本）](https://arxiv.org/html/2606.11408v2)在更新长度策略前预热critic。本批采用真实Monte Carlo回报预热，并让已有最后一层重新组织时序信息；两项构成一个明确的新候选，结果不能分别归因某一项。

## 模型与缓存

- 从A原sidecar恢复并复制最后一个Transformer block、`summary_proj`和`raw_h_ordinal_head`，初始actor与A一致。
- 冻结视觉/VLA、输入投影、视觉query池和第一个Transformer block。
- critic为64维tanh双值头；其输入使用`stop_gradient(summary)`，value loss不更新actor表征。
- 服务器在显式请求时导出最后block的输入29×256（4视觉token＋25动作token）及context256；不另跑VLA，不传完整1024×2048 prefix。
- 新客户端重算可训练最后block。现有warmup响应同时预热其CPU JIT，避免把首次编译计入首局；不额外增加warmup VLA调用。
- 所有H5/10/15/20/25都可选，不设H10 safety mask或人为H变化限制。新模式为`ordered_smdp_last_block`。

缓存与旧256维summary-only日志不同，必须重新采集。训练/验证均记录实际动作概率和对应checkpoint，旧日志不会冒充可用于解冻的输入。

## 训练顺序

每一轮先从当前actor随机采集100条完整新轨迹：

1. **critic-only预热10 epochs。** success目标来自真实终局成败，cost目标来自该call起的实际剩余RPC累计，不用自举的GAE return作critic监督。按完整episode/task固定划分80/20预热训练/验证，step0参加选择；这20条仅用于本轮训练数据内部的critic选择，不是模型greedy validation或final。
2. **重算优势。** actor在预热中不变，old log probability仍有效；用预热后critic重新计算实际duration对应的SMDP GAE。
3. **PPO最多4 epochs。** 更新已有最后block、summary projection和ordered head，actor学习率2e-5；critic学习率1e-3，继续拟合MC目标。batch64、clip0.1、anchor KL权重0.1、target KL0.02、梯度裁剪1；超过KL限制时回退整个epoch。

沿用gamma1、lambda0.995，并按实际环境执行步数计算GAE，末段提前成功不以计划H代替已执行步数。成功优先、RPC成本乘子初始0.02，按3秒预算小步调节到[0,0.05]；这是训练目标的成本权衡，不是成功率保证。

## 自动实验范围

| 轮次 | 训练：已有bank中的新IDs | 贪心验证 |
| --- | --- | --- |
| 1 | 120–129，10tasks×10 | 官方20–29，100局 |
| 2 | 130–139，10tasks×10 | 同一预留100局 |
| 3 | 140–149，10tasks×10 | 同一预留100局 |

已有bank `ROOT/round3_independent_5e18446_v1/initial_state_bank_0_299`已确认10任务均覆盖120–149，不生成新bank。训练使用随机H，验证/测试使用贪心H；均seed7、CoT/final NFE10、wait10、resize224、H25、warmup1。

每轮checkpoint保存到独立目录，带Orbax params及metadata。若actor没有发生实际更新，则保留新critic供下一轮使用，同时复用该actor已有的validation，避免把无变化actor的重复测量当训练收益。

最多三轮。候选须在100局validation上按`(成功数,-RPC均值)`优于A（93/100、RPC1.833013秒），且RPC≤3秒；再选唯一validation-best完成官方0–19的200局最终测试。没有胜出候选则保留A，不重复测试原版，不无限加epoch或扫参数。

最终结果比较同keys的当前A200局180成功、原版历史200局187成功，同时保留旧A189/200的跨运行差异说明。训练、验证、最终结果分别报告；采集成功率不冒充部署成绩。

## 实现与服务

- 中间缓存：`ExecutionHorizonPredictor(..., return_training_cache=True)`，默认不返回；政策请求`execution_horizon_export_last_block_cache`仅新模式启用。
- 部分模型：`src/openpi/execution_horizon/last_block_smdp.py`。
- 训练：`scripts/train_last_block_horizon_smdp.py`。
- 自动链：`scripts/run_last_block_horizon_smdp_pilot.py`。
- 客户端与训练使用JAX CPU；EGL仍可使用GPU，VLA服务器继续使用GPU。计时评测与训练串行。

为支持缓存导出，需要升级服务代码，但仍加载原H25 `/5000`和A `predictor_seed7_lr1e4`权重。默认eval和其他模式保持原样。代码按本地修改、GitHub main代理推送、服务器快照同步的顺序部署；不创建新分支。

`ROOT=/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19`。阶段状态、日志、退出码与全部数据保留，30分钟飞书进度及阶段/异常/最终结果汇报。

## Limitations

本批依旧没有新的历史观测/执行反馈输入，也不直接约束action expert的跨chunk连续性。预热改善的critic误差不等于已校准成功概率；解冻最后层也不保证提高成功率。三轮、单seed、重复使用的开发验证状态不能构成独立最终泛化或统计非劣证明。服务代码升级与历史计时/复现差异均须在最终结果中说明，不把小幅差异当稳健增益。
