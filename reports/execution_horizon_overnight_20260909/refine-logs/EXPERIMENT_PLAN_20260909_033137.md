# 六小时执行计划

窗口：2026-09-09北京时间03:12:30–09:12:30；UTC2026-09-08T19:12:30Z至2026-09-09T01:12:30Z。使用原单GPU服务器，不新租资源。ARIS配置：AUTO_PROCEED=true、HUMAN_CHECKPOINT=false、CODE_REVIEW=false、AUTO_WRITE=false、RENDER_HTML=false、MAX_ROUNDS=4。范围为研究实现与实验，不生成论文。

## 第一轮固定方案

1. 原A提出H，在eligible H>5 chunk执行5步后读取新proprio。一个轻量continue/replan小头，最多一次中途决策，原H是执行上限。
2. 80个train root与20个early root，Task0–9均衡、来自固定训练/early episode。源A轨迹每episode仅随机选择一个eligible chunk，选样不参考最终成功。每root continue/replan各3次配对续段，最多600条分支；不足如实记录，不伪造或补满标签。
3. 采用成功门控、幅度受限的相对RPC效用差，失败奖励0，成功时间项限制在±2%；不按每root微小标准差放大奖励。输入为调用时缓存A上下文/原始动作/旧proprio，以及新增mid-chunk proprio变化。新反馈版与遮蔽版使用同容量小MLP。
4. 固定训练预算，在early范围选模并一次性对齐两监视器的触发率后锁定；不根据dev结果调阈值。完整dev为A、fresh、masked三组，各100局交错，共300局。报告成功、配对救回/退化、实际policy/RPC/整局时间及触发次数；开发触发/成本不同则不将全部差异唯一归因新信息。
5. 新结果产生后进入一次有记录的研究复盘，决定继续、简化或结构调整。时钟唤醒只负责恢复实际等待阶段，不按时钟重做相同审查。

若early所选fresh为恒定continue的零输出模型，其闭环策略等同A，先进入结果复盘而不浪费三组评测；不能把它解释为已经有效训练并否定了新反馈。完整训练/early结果仍保留，后续结构调整需形成新的非零候选再评测。

部署设置已锁定：train每Task固定IDs300–307，early每Task330–331，各episode只取一个reservoir root；source/branch seed97007。每root保存457维raw feature、2×3的success/RPC/policy/segment时延和完整动作前缀，起点与中途原RGB仅作后续表示研究缓存，不进入首版模型。continue之后首次A请求保留完整旧chunk并累计previous_h=5+实际剩余执行，立即replan首次请求则previous_h=5。

两模型各400updates、lr1e-3、batch64、seed7、每25step用真实部署版NumPy输出做early选模，step0参与。归一化仅拟合一次train，masked标准化后遮蔽末16维；校准仅按early score顺序匹配fresh触发数，ties记录最近可达到数量，不用dev反馈调阈值。新增模型与主循环必要逻辑/输入输出检查已通过，首批真实root直接纳入正式采集。

## 时间与资源边界

目标前3小时内完成实现、采集和小头训练；后3小时完成锁定后的三组闭环及归档。最晚08:40不再启动长任务，所有本轮worker有剩余窗口timeout，09:12:30停止。若一轮实际运行显著超时，保留数据并缩减尚未启动的可选项，不将半批结果当完整对照。

优先完成一个可解释的闭环结果，再考虑下一轮。不为了填满六小时重复无变化审查或旧实验；也不提前把失败研究称为成功。成功信号不足时可在窗口内选择实质不同的表示/监督方案，须先更新此计划与tracker再执行。

原8040 A服务保留；新脚本/权重/结果单独存放。git沿main提交，禁止新建分支；不提交experiment_log、research_summary或私有review trace。飞书只推送关键阶段、异常和最后结果，凭证不进入输出。
