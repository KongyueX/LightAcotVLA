# H25 动态执行长度迭代结果

2026-09-06。本批保留 **A：Round4 的两层 global ordered Transformer**，作为当前最佳动态版本。A 成功率为94.5%，相对 Fixed H5 的 RPC 时间减少68.593%、整局时间减少29.394%；本批后续改动均未超过A。

## 共同评测协议

所有版本使用同一H25 policy checkpoint：
`/root/autodl-tmp/acotvla/checkpoints/acot_libero_long_chunk_h25/h25_from_h20_lr1e5_seed42_split42_e2f9b27/5000`。

LIBERO-10，10 tasks × initial states 0–19，各200局，seed7，CoT/final denoising均10步。动态版本在每次policy call重新选择H5/10/15/20/25；Fixed H5固定执行5步。下表时间为全部成功及失败局的均值，predictor开销按policy call计。

| 版本 | 成功数/率 | policy秒/局 | RPC秒/局 | 整局秒/局 | predictor毫秒/call |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed H5 | 190/200，95.0% | 5.487679 | 6.053083 | 15.370983 | — |
| θ0：初始ordered warm-start | 177/200，88.5% | 1.768532 | 1.957603 | 11.506057 | 5.78950 |
| **A：Round4动态数据微调** | **189/200，94.5%** | **1.715501** | **1.901117** | **10.852770** | **6.08885** |
| B：两层候选H读取替换头 | 184/200，92.0% | 3.550558 | 3.923397 | 13.208927 | 5.92184 |
| R：冻结A、候选H读取残差 | 185/200，92.5% | 1.721342 | 1.898837 | 10.992981 | 5.70550 |
| D5：A整轨迹采样动态relabel | 177/200，88.5% | 1.728112 | 1.921472 | 11.441148 | 6.16828 |
| N：D5数据、paired-noise时间软标签 | 183/200，91.5% | 2.560540 | 2.836112 | 12.180373 | 5.90980 |

A相对H5配对救回9局、退化10局，净少1局。N相对A救回9局、退化15局，RPC增加49.181%、整局增加12.233%；相对D5救回19局、退化13局，成功率增加3个百分点，但RPC增加47.601%、整局增加6.461%。N相对H5救回6局、退化13局，RPC减少53.146%、整局减少20.757%。

## 本批结论

1. **交付A。** A具有本批动态版本中最高成功率和最低整局耗时；R的RPC仅略低，成功率和整局耗时均未胜出。Fixed H5仍保留最高观察成功率。
2. 替换候选读取头、增加候选残差、扩大轨迹采样范围和单独调整计时噪声目标，均未带来优于A的闭环结果。本批未继续增加到4层；这些结果只否定本批具体配置，不说明候选读取或更深架构永远无效。
3. A/B/R/D5/N的本批微调各仅100个训练root、30个早停root，训练与验证很快分离；D5验证NLL首步最佳，训练NLL随后下降而验证NLL上升。D5训练中89/100个root存在最佳成功率并列，46个root的并列候选耗时范围不足0.5秒，计时强排序的可靠性有限。N缓和时间目标后仍未胜出，说明仅这一处修改不足。本批实验结束，下一批应先明确增加独立root和稳定标签的方案，再决定模型改动及运行预算。

## 可用模型与启动参数

代码版本：`6af844a`。配置：`acot_libero_long_chunk_h25`。A sidecar：
`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/ordered_dynamic_round4_a5e60b1_20260905_v1/predictor_seed7_lr1e4`。

2026-09-06服务已切回A：tmux `h25_best_server`，端口8040；运行日志为`ROOT/best_available_A_20260906_v1/server.log`。A checkpoint及所有对照数据均保留。

在服务器对应代码目录，沿用已配置的Python环境、`PYTHONPATH`及FFmpeg动态库路径，启动参数为：

```bash
/root/autodl-tmp/acotvla/envs/acotvla-py311/bin/python scripts/serve_policy.py \
  --port 8040 \
  policy:checkpoint \
  --policy.config acot_libero_long_chunk_h25 \
  --policy.dir /root/autodl-tmp/acotvla/checkpoints/acot_libero_long_chunk_h25/h25_from_h20_lr1e5_seed42_split42_e2f9b27/5000 \
  --policy.execution-horizon-predictor-params /root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19/ordered_dynamic_round4_a5e60b1_20260905_v1/predictor_seed7_lr1e4
```

评测入口为`scripts/eval_libero_execution_horizon.py`，显式使用`--modes ordered_transformer`；A不使用`hierarchical_transformer`模式，原有eval默认行为保持不变。

## 原始结果

以下路径均相对于服务器根目录
`/root/autodl-tmp/acotvla/execution_horizon_h25/snapshot_relabel_4770d19`：

| 版本 | 评测目录 |
| --- | --- |
| H5、θ0 | `ordered_closedloop_6e533f8_20260905_v1/pilot_10x20` |
| A | `ordered_dynamic_round4_a5e60b1_20260905_v1/eval_ordered_10x20` |
| B | `candidate_readout_63e2165_20260905_v1/eval_ordered_10x20` |
| R | `residual_candidate_6f24666_20260905_v1/eval_ordered_10x20` |
| D5 | `ordered_dynamic_round5_e096630_20260905_v1/eval_ordered_10x20` |
| N | `time_noise_6af844a_20260906_v1/eval_ordered_10x20` |

## Limitations

结果来自单seed、每版本200局pilot，多轮复用相同初始状态，属于development comparison，不构成正式成功率非劣证明或独立最终验证。不同时间软标签目标的NLL不能直接用于跨方案比较；当前选择依据是实际闭环成功率与耗时。
