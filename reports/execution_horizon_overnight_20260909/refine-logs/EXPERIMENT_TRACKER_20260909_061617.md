# 实验进度

| 阶段 | 状态 | 证据/下一步 |
|---|---|---|
| 已有方向与结果接续 | 完成 | 保留A与once，转向执行中新proprio |
| Round1实现接口 | 完成 | 457维MLP、k5执行回调、配对collector、trainer与deadline pipeline；必要I/O通过 |
| Round1训练/early配对采集 | 完成 | 80 train+20 early，600分支，无缺失eligible |
| Round1小头训练/锁模 | 完成 | 各400updates；early均60/60且无实质RPC headroom，均选step0 |
| Round1开发对照 | 跳过 | 所选fresh恒定continue，未产生新的闭环策略 |
| 结果驱动复盘/下一轮 | 第1次复盘完成 | 5/10、not ready；改为事件对齐观察，保留原始负结果 |
| Round2事件接口 | 完成 | 含跨chunk指令、458维shared k、280分支协议，必要I/O通过 |
| Round2配对采集/训练/条件开发 | 待部署 | 新train308–312、early332–333、seed107007；无结果 |
| 最终归档/飞书 | 未启动 | 截止09:12:30，保留负结果与未完成边界 |
