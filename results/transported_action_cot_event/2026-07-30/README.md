# Event-factorized Action-CoT 快速验证

本轮只验证一个问题：在普通 EAR 相位传输之外，再给夹爪通道一个独立的全局时间偏移，是否能以极小开销改善动作质量。该分支只增加一个 `Linear(128, 1)`，即 129 个参数；连续六维仍从原始 transported EAR 解码，事件头只改变夹爪在缓存 EAR 上的采样位置。

先做的无训练上限检查支持这一表示：固定 split7 的 624 个 test nominal pairs 中，`δ=0` 可完整复现 67.63% 的目标夹爪序列，而 `δ∈[-2,2]` 的 oracle 可覆盖 97.12%；test target 中只有 2.72% 是多事件序列。GT gripper 替换后的 phase nominal 7D MSE 下界约为 0.0563–0.0570，说明夹爪仍有足够理论改进空间。

按照快速 go/no-go 方案，只跑 seed7、600 steps，并比较完全相同的 event 架构：`event_zero` 固定 `δ=0`，`event_learned` 学习偏移。learned 分支将 test nominal MSE 从 0.12322 降至 0.12187，将 all-warp MSE 从 0.12046 降至 0.11860；all-warp event F1 从 0.4405 升至 0.4882。代价基本不可见，p95 仍约 0.40 ms，理论加速约 3.95×。

结论是“有方向性信号，但不足以成为当前主方法”。它仍未达到 nominal MSE 0.112335 的门槛，all-warp phase MAE 为 0.25058，略高于 0.25；而且收益不对称：对 progress offset `-1` 的 MSE 改善 0.004274，对 `+1` 反而恶化 0.001019。因此不继续扩成三种子训练矩阵。下一步更值得验证的是带置信度和 fallback 的事件修正，使模型可以新增、抑制或翻转事件，而不只是平移已有事件。

完整机器可读结果见 [summary.json](summary.json)。这些结果来自离线四帧 teacher windows，不等同于 LIBERO 闭环成功率。
