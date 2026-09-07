# 阶段二 Pilot 预研结论

**本文件不宣布**：阶段一正式验收通过、H1 成立、正式 OOD 达标、全场精度达标、M3 记忆通道逐点 <2%（网络侧）通过。

环境：训练正式档使用 `D:\\Anaconda\\envs\\torch24`（torch 2.4.0 + CUDA / RTX 4060 Ti）。CPU Torch 2.2 的 `hard_n256` 仅作对照基线，不作为正式过拟合结论。

## 0. 证据边界（先读）

- 标签来自 Stage-1 **pilot 2000**（1896 ok），不是 formal 30k。
- 模型选择只看 **val 前 16 例** 的 pert L2；全量 val/test 只在调参封闭后由 `eval.py` 跑一次。
- 训练窗：停泵后 4 个 \(4L/a\)，井口 decim=4（Nyquist ≈ 40–50 Hz），节点 ≤128 点。
- **不能验收**：70 Hz 到时、0.05–1/1–10/10–100 Hz 分频带、全场 \(E_{L2}\)、网络记忆通道 vs MOC \(z_l(x,t)\) 逐点 <2%、正式 OOD、H1 的 5-seed/32-trial 对标。
- 原型 **不是** 完整 CJ-NO：无特征坐标段传播、无全场 trunk；硬节点 Newton 使用物理 \(B=a/(gA)\)，但 \(K/c_1/a_0\) 为简化占位，不是逐孔 MOC 系数。
- A1b 在本轮是「同一外壳换节点头」，井口头仍是共享 1D FNO；硬层不改写井口路径。
- 未完成项：extra seed 43 hard, extra seed 43 soft, extra seed 44 hard, extra seed 44 soft。

## 1. 实际可用案例与可信监督

- 可用 **1896** 例（manifest ok=1896，SHA256 全匹配，深度检查 0 失败）。
- 切分 seed=20260906：{'train': 1514, 'val': 191, 'test': 191}，组泄漏为空。
- 104 例因稳态井口压力越界拒绝；参数未保留。**有效分布 ≠ 原始 LHS 先验。**
- 可信监督：原生井口 \(p,Q\)（Nyquist 中位约 176 Hz）；节点迹/记忆为 decim≈16（Nyquist 中位约 11 Hz）。
- **没有**观测链波形、**没有**全场快照。训练窗口再 decim=4 后井口 Nyquist 降至约 40–50 Hz。
- formal smoke 未混入训练集。
- 源 generator frozen digest 前 8 位：`22bd3563`。

## 2. 节点层前向与梯度；记忆通道

- 单元门：`node_validation_metrics.json` 状态 **PASS**，失败数 0。
- 前向与 Stage-1 `solve_cluster_node` 一致；无量纲残差 \(\sim 10^{-13}\sim10^{-11}<10^{-8}\)。
- 光滑点 IFT vs 中心差分相对误差 \(\sim 10^{-10}<10^{-4}\)；条件数 \(O(10^2)\)，未触发 Tikhonov。
- \(q\approx 0\) 单独报告（孔眼 \(q|q|\) 非光滑）。
- 求解器侧记忆一致性（`memory_moc_consistency.json`）：**PASS**。 递推 vs 同核指数卷积 rel L2 = 4.283e-14；相对完整 Zielke 的拟合残差 = 0.007398（只报告，不是本测试失败项）。 Joukowsky 小算例 `snap_every=1` 重建 \(z\) rel L2 = 2.241e-16。**这不是网络记忆通道门。**
- 网络记忆 \(z_l(x,t)\) 逐点 <2%：**待补证据**（pilot NPZ `snap_every=0`，本轮未批量补生成）。

## 3. 256 例过拟合

- run_id（hard）=`s2_hard_n256_torch24_20260906_165737_999034df`，device=`cuda`，params=`256839`。
- 硬节点正式档：train pert L2 mean = 0.2596（p50=0.1482，max=1.745），val pert L2（16 例选模）= 0.2043，best_epoch=16，epochs=29，wall=1122.8 s。
- 节点水头扰动 L2（train 诊断）= 1.199；非有限条数 = 0。
- 过拟合门（井口扰动相对 L2 ≤2%）：**未达到**。
- 失败归因（按现有证据，不是事后改门）：(i) 原型缺少特征坐标段传播，井口只靠条件化 1D FNO；(ii) 损失尺度上 \(L_{node}/L_{hard}\) 远大于井口头，优化先压节点/占位 Newton 而非波形细节；(iii) 井口 proj 零初始化后仍学到 ~0.2 量级相对误差，说明容量/架构未贴到逐点波形；**不是**节点 IFT 算错（B 门已过），**不是**标签 SHA 损坏（审计 0 失败）。
- 长时自由滚动：评价窗为停泵后 4 个 \(4L/a\)；不外推到 20 个周期全窗。
- **相位诊断口径**：正式 `eval.py` 使用物理时间 \(t\)。`hard/soft_n256_torch24` 训练环里的 `phase_4La` 曾误用采样下标当时间，**那些 history 里的相位数作废**；n=64 及以后的训练环已改用 `s.t`。
- 全量 val（191，选模后一次）：hard pert mean=0.2355（p50=0.2011），soft 0.2547。16 例选模偏乐观。
- 调参封闭后 ID-test（**不是正式 OOD**）：hard pert mean=0.2339（p50=0.1969，phase p50=0.06155 of \(4L/a\)），soft 0.2654，n=191。
- 正式相位（物理时间）：hard val p50=0.06543，test p50=0.06155；均值被错峰案例拉高，**未达 1% 的 \(4L/a\) 门**。

## 4. 硬节点 vs 软节点（A1b 预研，非正式 H1）

| tag | n | val16 pert | full val | ID-test | wall s | infer s |
|---|---:|---:|---:|---:|---:|---:|
| hard 256 | 256 | 0.2043 | 0.2355 | 0.2339 | 1122.8 | 0.002997 |
| soft 256 | 256 | 0.1931 | 0.2547 | 0.2654 | 31.9 | 0.003011 |
| fno 256 | 256 | 0.1123 | 0.1361 | 0.1508 | 32.6 | 0.005 |
| hard 64 | 64 | 0.3755 | — | — | 301.3 | 0.003008 |
| soft 64 | 64 | 0.3754 | — | — | 13.0 | 0.001995 |

- 参数量：hard/soft/fno 均为 256839（同一外壳，未拆头）。
- 16 例选模上 soft 略优于 hard（0.193 vs 0.204）；**全量 val/test 上 hard 略优于 soft**（val 0.2355/0.2547，test 0.2339/0.2654）。单 seed，差值远小于误差本身，**不能主张硬层带来显著井口精度收益**。
- 嵌套 n：64→256 使 val16 pert 从 0.375 降到 ~0.20，有样本增益，仍远高于 2%。
- FNO 对照（无 \(L_{node}/L_{hard}\)）val16 pert = 0.1123，train pert = 0.122。井口更好是因为损失不被节点项淹没，**不是**「纯 FNO 已达到 CJ-NO 目标」，更不是 H1 成立。仍未过 2% 门。
- 成本：hard 256 wall ≈ 1123 s，soft/FNO ≈ 32 s；batch=1 井口推理均约 3 ms（`apply_hard_times=0`）。硬层训练贵在 CPU Newton 循环，不在 GPU 前向。
- 节点 H 扰动 L2（全量 eval）：hard val=1.046 / test=1.011，soft val=1.011 / test=0.9851。FNO 无节点头（NaN）。硬/软节点误差同量级 ≈1，说明当前硬层占位 Newton **没有**把节点监督做成可用读出。
- 不是 5 seeds / 32-trial HPO。seeds 43/44 未跑，不算统计。
- 图：`figures/Fig_pilot_hard_vs_soft_n256.png`、`Fig_pilot_nested_n.png`、`Fig_pilot_waveforms_hard_n256_torch24_val.png`（边注 run_id/ckpt）。
- 表：`tables/table_pilot_ab.tex`。

## 5. 限制主要来自哪里

- **实现**：节点层门已过；训练瓶颈是 Python 循环 + CPU Newton + NPZ 读盘。硬模式 GPU 并不比 CPU 快一个数量级，因为 Newton 不在 GPU 上。
- **数据质量**：1896 例完整、哈希一致；缺全场与观测链；节点带宽不够 70 Hz。
- **样本量**：256 例未过拟合到 2%；全量 1514 例不是本轮必达。
- **架构**：当前收益更像共享 FNO 井口头；结构优势未表现出来。

## 6. 下一轮最值得补的数据/实验

1. 对一小撮 case 打开 `snap_every>0`（不必 30k），专门验**网络**记忆通道 vs \(z_l(x,t)\)。
2. 实现真正的特征坐标段传播后再做 A1b，并把硬层输出接到井口路径，而不是只换节点头。
3. 硬层改用逐孔 MOC 的 \(K,c_1,a_0\)，去掉占位系数。
4. 训练直接用原生井口 Δt（或 decim=2）的短窗，避免把 70 Hz 信息先抽掉再谈高频门。
5. 满量 30k 仍单独排队，不与本预研结论捆绑。

## 图表与表格

- `tables/pilot_summary.json`：官方 tag 摘要。
- `tables/*_seed42.json`：各 run 完整 history。
- `figures/*_seed42.png`：训练曲线，边注 run_id。
- `figures/Fig_pilot_hard_vs_soft_n256.png`、`Fig_pilot_nested_n.png`。
- 波形叠图：`python plot_waveforms.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt`。

## 可复现命令

```text
set PYTHON=D:\Anaconda\envs\torch24\python.exe
cd PaperC_CJNO_Wellbore_Inversion/02_cjno_forward_operator/pilot_preliminary/code
%PYTHON% test_interface.py
%PYTHON% test_dataset.py
%PYTHON% test_checkpoint.py
%PYTHON% test_memory_moc.py
%PYTHON% audit_pilot.py
%PYTHON% train.py --mode hard --n-train 256 --seed 42 --tag hard_n256_torch24
%PYTHON% train.py --mode soft --n-train 256 --seed 42 --tag soft_n256_torch24
%PYTHON% train.py --mode hard --n-train 64 --seed 42 --tag hard_n64_torch24
%PYTHON% train.py --mode soft --n-train 64 --seed 42 --tag soft_n64_torch24
%PYTHON% train.py --mode fno --n-train 256 --seed 42 --tag fno_n256_torch24
%PYTHON% eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val
%PYTHON% eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split test
%PYTHON% plot_waveforms.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val
%PYTHON% write_pilot_report.py
```

禁止：`--confirm-30k`；改写 Stage-1 goldens / pilot manifests。
