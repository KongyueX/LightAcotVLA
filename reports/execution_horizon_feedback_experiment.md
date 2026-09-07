# Predictor A 配对监督与历史反馈实验

2026-09-07。目标是在保留H25动作模型和A的基础上，检验更直接的配对监督及实际观测历史能否改善执行长度选择。用户已授权实施、服务器测试和飞书进度汇报。

## 方法

冻结H25 VLA及A全部参数，在客户端对A的四个continue logits增加零初始化残差。比较同数据、同目标的两个候选：

- `ordered_feedback_current`：只使用A当前256维summary。
- `ordered_feedback_history`：增加上一真实policy call到当前的prefix/state变化、上次执行H和实际环境步间隔。

历史prefix复用A已有的2048→256投影，投影后swish，取当前与上次之差。特征为summary256＋prefix变化256＋state变化32＋previous H/25、elapsed steps/25、history valid，共547维。当前输入对照把后291维置零，两个候选都在无历史的首call保持A。归一化只用训练root拟合；547→64 tanh→4的最后投影从零开始，step0与A的logits一致。

不增加VLA调用，不传输完整prefix tokens；已有A响应提供summary、pooled prefix及normalized state。新增客户端计算计入完整episode，单列selector毫秒，不能重复加到RPC时间。

## 配对数据

沿用现有generation seed20260830，将旧bank的0–299扩展到0–365，直接核对旧前缀一致。每任务角色在采集前固定：

| 角色 | 每任务初态ID | 总初态数 |
| --- | --- | ---: |
| 训练 | 300–329 | 300 |
| 早停 | 330–335 | 60 |
| 开发闭环 | 336–345 | 100 |
| 预留测试 | 346–365 | 200 |

训练和早停每episode采一个root。A先完成真实source轨迹，outcome-blind reservoir均匀选一个实际policy call，保留物理快照、当时已生成的动作、A选择和真实上一call特征。source终局成败仅用于分组分析，不参与root选择或模型输入。

每root五H5/10/15/20/25，各五次paired continuation。只改变当前chunk执行H，随后都由原A继续；每repeat内H顺序打乱，共享continuation随机种子。每分支从同一快照恢复，当前动作不重新生成。source与continuation均显式CoT/final NFE10及同步profile，与闭环eval一致。

每root独立NPZ保存输入、每H/repeat的success、RPC秒、elapsed秒、calls、steps及valid。保存的root RPC作为五H共同成本，在配对差值中抵消。单独记录完整分支elapsed和RPC，不混用。

首先完成训练集中的40个诊断root：Task3/4各10，Task8六个，其他任务各两个。报告相对root处A选择的成败变化、全H失败与leave-one-repeat-out选择诊断；这些不是可部署成功率，也不把5/5当作稳定保证。随后按既定范围完成剩余训练与早停数据。

## 训练

对每root当前A所选H作为配对参照，计算：

`adv_H = mean(success_H − success_A) − 0.02 × mean(RPC_H − RPC_A)`

只使用共同有效repeat，RPC单位为秒。两个候选相同目标：

`loss = − E_{H~pi}[adv_H] + 0.05 × KL(pi_A || pi)`

直接使用多次分支的Monte Carlo后果，不训练critic。固定seed7、学习率1e-4、batch64、最多2000 updates，每25步在全部早停root检查，8次未改善停止；step0参与best选择。两个系数固定，不扫参。只训练新增残差头。

当前输入对照检验新数据/监督能否带来收益；历史候选与它比较才能区分历史输入的贡献。旧反事实数据不混入本批，也不把旧开发状态用于梯度训练。

## 闭环测试与完成条件

新100个开发初态上，A/current/history三个模式逐episode交错，保持同一H25与A服务、seed7、NFE10/10、resize224及等待10步。每模式完整100局，报告成功数、配对救回/退化、policy/RPC/整局时间及客户端额外开销。

开发阶段按成功数优先、同成功数RPC更低选一个严格优于A的候选。若没有候选胜出，结束并保留A；若有，则在预留200初态上仅比较该候选与A。最终报告同时呈现成功率和时间，成功率提高但更慢时明确写作权衡；不以开发胜出冒充预留测试提升，也不自动替换A服务。

每个阶段保存command JSON、log和exit，闭合root和已完成stage可续跑。完整评测使用既有episode journal。飞书发送阶段结果，30分钟监控发送有实际进度的更新；失败时保留日志与所有数据，结束后停止监控。

## 入口

- 采集：`scripts/collect_execution_horizon_feedback.py`
- 残差模型：`src/openpi/execution_horizon/feedback.py`
- 训练：`scripts/train_execution_horizon_feedback.py`
- 闭环：`scripts/eval_libero_execution_horizon.py`的新显式feedback模式；原默认模式与原resume配置保留。
- 全流程：`scripts/run_execution_horizon_feedback_experiment.py`

服务器沿用SSH49412、8040的A服务及现有Python3.11环境；不重建环境或重训练VLA。

## 实施验证

采集、残差模型、训练、eval接入及流程恢复共26项针对性检查通过；另4项原有eval默认/单次请求/ordered选择/resume兼容检查通过。真实A checkpoint成功恢复prefix投影，新增输出层为零。检查包括训练/推理一致性、一次实际Optax更新、配对valid与RPC量纲、train-only归一化、step0选择、历史快照和中断恢复；没有运行本地训练或额外GPU性能smoke。

## Limitations

本批为单训练seed、每root五次重复；局部H标签之后仍由A继续，不能假设其估值等于新候选的整局表现。历史仅一段，能否区分停滞与正常推进需要实测；不能据此保证失败恢复。候选仅控制重规划时机，动作模型能力仍限制最终效果。新增200个预留初态提供本批选模外检验，但有限样本不自动构成严格统计非劣证明。
