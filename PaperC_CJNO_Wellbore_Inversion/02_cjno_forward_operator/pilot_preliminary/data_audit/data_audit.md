# Pilot 数据准入审计
范围：阶段二 pilot 预研。不宣布阶段一正式验收、H1、正式 OOD 或全场精度。
- 源 manifest 案例：2000（ok=1896, rejected=104）
- 文件缺失 0，SHA256 不符 0，深度检查失败 0
- **可用 1896**，切分 seed=20260906：{'train': 1514, 'val': 191, 'test': 191}，泄漏组 []
- 104 例拒绝：稳态井口压力越出 [5e6, 1.2e8] Pa。manifest 未保留其参数，**当前有效分布 ≠ 原始 LHS 先验**。
- 实际保存：原生井口 `t,p_head,Q_head`；降采样节点 `node_H,Q−,Q+,sum q, mean p_c, node_z`；网格在 `meta_json`。
- **没有**观测链波形、没有全场快照。第一轮监督 = 井口 + 已存节点迹/记忆，不冒充全场算子。
- 井口 Δt 中位 2.8429e-03 s（Nyquist ≈ 175.9 Hz）。
- 节点 Δt 中位 4.5490e-02 s（Nyquist ≈ 11.0 Hz）。**节点标签不能验收 70 Hz。**
- Cr_target=1，但训练域中全部段 Cr=1 的比例只有 0.000；短段 Cr 中位最小 0.879。
- 当前 `goldens/metrics.json` 的 Newton 门状态：PENDING（历史 pilot 覆盖率仍为 1896/1896；smoke 验收会改写该文件）。
- formal smoke 未混入本研究集。
