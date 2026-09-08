# 第二轮结果：非零事件策略出现，fresh未超过masked

预定70个源episode完成；48个train与19个early有合格事件，共67root/268条分支。没有合格事件的Task5/train309、Task6/train308、Task5/early332照实记录，未替换或补固定k5样本。原始数据位于`results/execution_horizon_overnight_20260909/round2/`。

| 范围 | continue成功 | always-replan成功 | continue均值RPC秒 | always-replan均值RPC秒 |
|---|---:|---:|---:|---:|
| train48root | 92/96 | 89/96 | 0.912068340 | 1.046605650 |
| early19root | 35/38 | 36/38 | 0.722537550 | 0.635857939 |

fresh选step50、masked选step25，均为非零参数更新。经过预定的一次触发率校准，masked阈值固定为0.027104230597615242；fresh和masked在19个early root上恰好选择相同6个root重规划，均36/38成功、RPC0.594490897秒。相对continue的净救回为1，平均RPC减少0.128046653秒。

该结果未达到预先设定的fresh必须超过matched masked的条件，因此没有自动启动fresh三组开发。19个early选择相同不代表两函数在其他状态上等价，更不支持新proprio有额外价值。大部分early优势来自Task6/332一个随机分支差异：continue一失败一成功，replan两成功，其平均RPC改善在全19root均值中约占0.1268秒。

第二次研究复盘为6/10、not ready：允许一个更窄的后续问题，即冻结masked事件策略及阈值，在既有开发100初态上与A各跑100局，判断只用缓存上下文的事件调度是否有完整闭环价值。不再训练、调阈值、换特征或换checkpoint；这一后续验证不追认fresh-proprio条件通过。

这些开发初态未参与本轮train/early，但在项目历史开发中已使用；不能称为全新独立测试。预留346–365仍不使用。完整模型成功率与部署耗时目前尚无新结果。
