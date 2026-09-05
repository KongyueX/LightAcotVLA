# H25 候选执行长度感知 Transformer 方案

状态：2026-09-05 文献核对并实现。候选H readout、服务端配置、显式checkpoint迁移及2层比较入口已实现；新架构效果尚待训练和闭环测试。当前 Round4 原架构链作为对照继续运行。

## 目标与依据

在同一 H25 动作模型上，改善每次 policy call 对 H5/H10/H15/H20/H25 的选择，优先恢复任务成功率，同时保留实际 policy/RPC 与整局时间收益。

当前模型先对全部25个时序 token 求均值，再从共享向量输出所有 H。候选执行区间和停止边界缺少显式表示，是值得检验的架构假设。现有全注意力已覆盖25步，加深网络增强的是特征组合深度，不会扩大时序感受野。

## 文献依据与迁移范围

本节核对原文方法及附录。除 MoH 的 arXiv 元数据明确标注 ICML 2026 录用外，其余下列工作按预印本引用；不把它们在其他模型、任务上的结果当成本项目预期成绩。

| 论文与来源 | 原文相关证据 | 对本方案的作用与限制 |
| --- | --- | --- |
| Shin et al., 2026, [ACH：Adaptive Action Chunking via Multi-Chunk Q Value Estimation](https://arxiv.org/html/2605.10044v1) | §4.1–4.2 对同一动作序列的嵌套前缀分别估值；causal Transformer 一次输出多个前缀Q。附录B.4为2层、128维、4 heads。 | 支持显式区分候选前缀。本文的attention pooling、边界特征拼接及H embedding是项目假设，不是ACH原文给出的readout；ACH同时训练动作策略与critic，不是冻结VLA的SFT sidecar。 |
| Xu et al., 2026, [BCP：Continue or Replan?](https://arxiv.org/html/2608.03483v1) | §3.2采用共享CLS、上下文与逐步动作/最终速度场特征，预测有序continue概率；附录Table A.4比较深度。 | 支持ordered continuation与动作条件输入，但并未采用我们提出的候选专属readout。其RoboTwin Hanging Mug Clean对照中1/2/4/6层成功率为82/87/84/83%，不支持预设加深会提高效果。 |
| Zhao et al., 2026, [DEHP：Dynamic Execution Horizon Prediction](https://arxiv.org/html/2606.11408v2) | §3将当前状态与完整动作chunk联合用于H选择，冻结动作策略，并按实际执行步数构造SMDP-PPO更新。 | 支持状态与动作共同决定H、考虑后续策略的训练视角。其状态型Diffusion Policy实验未给出horizon head深度对照，不能作为4层sidecar依据；本项目目前未实现该PPO训练。 |
| Nie et al., 2026, [PACE：Phase-Aware Chunk Execution](https://arxiv.org/html/2606.00537v2) | §3从单次预测动作块的平滑速度曲线提取低速转折点作为重规划边界，不增加策略前向。 | 为后续边界特征提供依据。迁移到LIBERO时需按实际delta-action和归一化定义物理量；不能直接把任意动作差分叫速度，且速度谷不是安全概率。 |
| Jing et al., ICML 2026, [MoH：Mixture of Horizons in Action Chunking](https://arxiv.org/html/2511.19433v2) | §3对多个horizon分别mask并共享action transformer，形成不同预测后融合，并用跨horizon分歧决定执行范围。 | 可作为动作模型层面的后续路线；真正采用需要改造/训练action expert。同一个H25 chunk截取多个前缀不会产生独立预测，不能直接复制其consensus。§4.4的2.5× throughput指标不能等同于墙钟时间加速。 |

这些证据支持“前缀/边界表示＋有序选择”的方向，但尚未验证本项目的具体组合。BCP的层数结果只覆盖其指定单任务，不证明所有任务2层都最好；它要求我们把4层保留为待测变量，而非默认更优方案。

## 推荐候选

保留输入、物理时间对齐和4个 prefix attention queries，hidden=256、attention heads=4、FFN multiplier=4。优先实现2层候选H感知 Transformer；4层是后续容量对照，不作为预先选定的胜出版本。

时序编码器输出 Z_1,...,Z_25。每个候选 H 构建自己的表示：

`r_H = MLP([attention_pool(q_H, Z_1:H), Z_H, context, horizon_embedding(H)])`

- 前缀 attention pooling：汇聚本次将执行的动作区间。
- 边界 token Z_H：保留执行终点附近的局部事件信息。
- H embedding：显式表示不同执行长度。
- 所有 H 共享 readout 网络，避免五套独立大模型。

Z 是已有全时序编码器的输出，包含整个已生成动作计划的上下文；前缀 readout 不代表严格因果编码，也不使用未来真实观测。

相邻候选的继续决策使用共享小头：

`continue_logit_k = f([r_Hk, r_H(k+1), r_H(k+1) - r_Hk])`

四个 continue logits 仍按已有有序概率公式形成五档 H 分布，部署端继续读取 `ordered_selected_h`。该因子化借鉴BCP；将相邻候选表示作为共享头输入是本项目的待验证改动。第一批仅改变 ordered policy 的 readout，保持其他辅助任务与选择规则，便于判别架构改动的作用。

输入特征的后续候选为PACE启发的减速谷位置/显著性，以及已有VLA计算中的最终去噪速度场。二者要区分：动作序列的时间差分与flow velocity不是同一个量。当前数据可重算前者的合适代理，后者未保存时只能在后续采集增加导出。先不将新增信号与readout/层数变化捆绑测试。

## 参数恢复与训练

- 冻结 H25 VLA；复用当前 sidecar 的共享投影、视觉池和前两层时序参数。
- 新 readout 使用独立参数，不把旧 cumulative ordinal 参数直接解释为新边界头。
- 4层版本新增两层采用残差输出零初始化，使新增层初始接近恒等映射。
- 本批使用与对照相同的weight decay和validation-best/early stopping，保持dropout设置一致；readout dropout作为后续独立变量。
- 每增加一层约增加0.79M参数，两层约1.58M；另有readout参数。实际推理耗时需要测量，不预先承诺毫秒增量。

复用正在采集的 Q^dynamic 数据与既定 episode split，无需重新采集输入。旧 Q^H5 数据只通过初始化提供知识，训练监督仍使用新动态数据。

## 最小比较

| 版本 | 层数 | ordered readout |
| --- | --- | --- |
| A：当前对照 | 2 | 全25步mean summary |
| B：优先升级 | 2 | 候选H前缀attention与边界表示 |
| C：容量对照 | 4 | 与B相同 |

使用同一数据、split和训练目标，记录A/B的独立early-stop表现。B是本次唯一新增主版本，训练成功后完成相同200初始状态闭环，验证损失差只作诊断，不单独阻止该闭环测试。依据B的泛化和实际闭环收益，再决定是否运行C。

实现入口：`train_execution_horizon_predictor.py --ordered-readout candidate --resume-candidate-readout`；比较runner为`run_execution_horizon_candidate_readout_experiment.py`。该runner复用Round4的100/30/30/20划分、lr1e-4、batch128、相同loss，从相同θ0初始化；默认2层，4层需显式指定。运行排在原Round4的采集、微调和复测全部结束后，避免并发污染计时。部署仍使用`ordered_transformer`，sidecar配置决定global或candidate readout。

最终以闭环成功率、policy/RPC时间、整局耗时和predictor开销判断，不以参数量、训练loss或H分布多样性作为成功标准。若4层只有训练loss更低而验证及闭环没有收益，保留2层readout改进版。

## 限制与后续

该比较用于判别架构改动，不同时更换时间软标签规则，以免混淆两者作用。已发现的计时噪声放大与较晚恢复状态覆盖不足仍需单独处理；增加层数不保证弥补这些问题。本项目当前训练是多seed反事实SFT与动态relabel，不是BCP的GRPO、DEHP的SMDP-PPO或ACH的TD训练复现，不能直接继承相应论文的理论或实验收益。
