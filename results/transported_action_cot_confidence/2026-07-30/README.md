# Action-CoT update confidence 快速验证

本轮在固定的 `event_learned` checkpoint 上训练一个独立的 129 参数 plan-level router。Router 读取 frozen editor 的 stopped-gradient context，预测完整事件修正是否应被采用；拒绝时原子回退到同一 checkpoint 的未修改 phase-transport plan。Router 不与 editor residual 相乘，也不向 editor 或视觉编码器反传。

严格 oracle 表明选择空间存在：held-out all-warp test 上，always-base、always-update 和逐样本 oracle route 的 7D action MSE 分别为 `0.12306442`、`0.11860245` 和 `0.10837799`。但是 learned confidence 没有学成：validation 选择阈值 `0.47560` 后，test AUROC 仅 `0.54262`；接受覆盖率 `63.10%` 时 routed action MSE 为 `0.12050465`，差于 always-update，accepted regression rate 仍为 `47.33%`，因此预注册 offline gate 失败。

Router 的额外 mean latency 为 `0.02328 ms`；editor+router mean/p95 为 `0.42651/0.43587 ms`，固定 1:4 的理论加速仍为 `3.9473×`。当前结果只说明现有 nominal/synthetic local-time-warp 2k 数据无法可靠区分有益和有害修正，不能否定 confidence 在 branched disturbed 数据上的潜力，也不能证明 full-ACoT refresh。完整机器可读结果见 [summary.json](summary.json)。
