# Handoff Report — explorer_p3_data

**Agent**: `explorer_p3_data` (Teamwork Explorer: Dataset, MOC Simulation & Robustness Test Investigator)  
**Parent Agent**: `orchestrator_5` (`ef4964b0-6990-42ce-9c4d-bbe3ee64b8cd`)  
**Timestamp**: 2026-09-13T14:45:00Z  
**Type**: Hard Handoff (Investigation Complete)  
**Deliverables**:
- Full Technical Report: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_p3_data\report.md`
- Handoff Report: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_p3_data\handoff.md`

---

## 1. Observation (客观观察事实)

1. **数据集文件与存储物理特征**：
   - 文件 `data/datasets/moc_v2_1k_dataset.h5`：大小 103,332,056 字节（98.55 MiB），`gzip` 压缩比 2.32。包含两个组：`waveforms` 与 `labels`。
   - `waveforms` 包含 `timestamps` (60001,), `wellhead_head` (1000, 60001), `wellhead_velocity` (1000, 60001)。
   - `labels` 包含 `n_frac` (1000,), `fracture_positions` (1000, 8), `fracture_weights` (1000, 8), `fracture_Cf` (1000, 8), `fracture_Kp` (1000, 8), `fracture_kleak` (1000, 8), `fracture_types` (1000, 8), `has_fault` (1000,), `fault_cluster_idx` (1000,), `pump_closure_tc` (1000,), `wavespeed` (1000,), `H_ext` (1000,)。
   - 快速缓存文件 `PaperC_CJNO_Wellbore_Inversion/data/cache_1k_t4096_c1024_g500.npz`（32,899,729 字节）包含 11 个键：`waveforms` (1000, 2, 4096), `cepstrums` (1000, 1, 1024), `conds` (1000, 3), `n_frac` (1000,), `positions` (1000, 6), `norm_positions` (1000, 6), `masks` (1000, 6), `alphas` (1000, 6), `cf` (1000, 6), `log_cf` (1000, 6), `m_alpha_grid` (1000, 500)。
2. **簇数分布排查**：
   - 经对全量 1,000 例执行 `Counter(n_frac)` 统计，簇数分布精确为：`Nc=1: 166, Nc=2: 153, Nc=3: 154, Nc=4: 175, Nc=5: 177, Nc=6: 175`；
   - 槽位 6 和 7（第 7、8 簇）在 H5 中全部为 0.0（`pos[:, 6:].max() == 0.0`），无活跃的 7 簇或 8 簇算例；
   - 簇间距分布：$\min=5.00\,\mathrm{m}, \max=40.00\,\mathrm{m}, \mathrm{mean}=22.59\,\mathrm{m}$；间距 $<10\,\mathrm{m}$ 占比 13.7%（353 对），$10\sim 20\,\mathrm{m}$ 占比 28.7%（737 对）。
3. **计算软硬件环境基准**：
   - 系统为 Windows 10 (10.0.19045)，Python 3.12.4，PyTorch 2.2.0+cpu（`torch.cuda.is_available() == False`，10 线程并发）；
   - 在 100 例测试集上执行实测基准（CPU 推理耗时与参数量）：
     * 1D-ResNet: 331,394 参数，47.60 ms (2,100.8 samples/s)；
     * 1D-FNO: 1,089,090 参数，119.30 ms (838.2 samples/s)；
     * Vanilla DeepONet: 204,738 参数，25.50 ms (3,921.6 samples/s)；
     * CJ-Cep-DeepONet: 149,218 参数，33.20 ms (3,012.1 samples/s)；
     * TG-DeepONet: 230,598 参数，42.70 ms (2,342.0 samples/s)；
     * 训练步耗时（batch 32, fwd+bwd+opt）：55.0 ms / step，单 Epoch 耗时 1.38 秒，100 Epochs 训练仅需 2.3 分钟。
4. **现有评测脚本与鲁棒性审计现状**：
   - 现有 `experiments/evaluate_ablation.py` 与 `evaluate_pilot.py` 仅支持 Clean 纯净评估，无加噪或扰动逻辑；
   - 对现有 Phase 2 最优模型 `tg_relative_bias` 执行独立噪声压力测试实测：
     * Clean 基准：$\alpha$ MAE = 0.1336，$R^2$ = 0.5666，$W_1$ = 8.39 m，$C_f$ MRE = 46.1%；
     * 白噪声 20dB：$\alpha$ MAE = 0.1437（相对恶化 +7.6%），$R^2$ = 0.5104（衰减 9.9%），$W_1$ = 8.93 m；
     * 粉红噪声 20dB：$\alpha$ MAE = 0.1417（相对恶化 +6.1%），$R^2$ = 0.5314（衰减 6.2%），$W_1$ = 8.73 m；
     * 声速扰动 $-1\%$：$\alpha$ MAE = 0.1502（相对恶化 +12.5%），$R^2$ = 0.4660（衰减 17.7%），$W_1$ = 9.46 m；
     * 声速扰动 $+1\%$：$\alpha$ MAE = 0.2111（相对恶化 +58.0%），$R^2$ = -0.0950（崩溃 -116.8%），$W_1$ = 15.58 m。
5. **密集多簇与稀疏工况分层评估实测**：
   - 稀疏样本（$N_c \in [1, 3]$，53 例）：TG-DeepONet $\alpha$ MAE = 0.1596，$R^2$ = 0.5609；
   - 密集样本（$N_c \in [4, 6]$，47 例）：TG-DeepONet $\alpha$ MAE = 0.1220，$R^2$ = 0.1319（基准模型 ResNet/FNO 为 0.0340，CJ-Cep 为 -0.1211）；
   - 单簇工况（$N_c=1$，17 例）：全模型 $\alpha$ MAE = 0.0000，$W_1 = 0.00\,\mathrm{m}$；
   - 物理单纯形守恒偏差：全模型最大单纯形偏差为 $1.19 \times 10^{-7}$；
   - 裂缝位置检出与起裂分类 F1-score：当前代码库中直接输入已知完井几何深度，未包含位置偏差微调头与起裂二分类头，指标库 `src/metrics.py` 中亦未实现 F1 评测。

---

## 2. Logic Chain (推导逻辑链)

1. **由 Observation 2 与 Observation 5**：
   - 密集多簇工况（$N_c \in [4, 6]$）虽然在样本量上占据全集的主导地位（52.7%），但其当前决定系数 $R^2$ 仅为 0.1319（基准模型近乎为 0 或负值），而全集 $R^2=0.5666$ 主要是由单簇（$R^2=1.00$）和稀疏工况拉高；
   - 根本原因在于水锤波沿水平井传播通过密集多簇（间距 5~40m）时，上游跟部裂缝的透射扼流与强烈反射衰减了传向下游的能量，且下游反射波被上游裂缝多次混响掩盖（透射衰减与多径串扰）；
   - 在缺少一维波动方程逆散射层剥离算子（DIS Layer）显式解混的情况下，神经网络受期望风险最小化驱动，被迫对密集多簇输出趋近于先验均值（$\approx 1/N_c$）的预测，形成均方差均值退化陷阱；
   - **推论**：要实现密集多簇 $R^2 > 0.75$，必须构建严格沿井筒（$x_1 \to x_{N_c}$）补偿透射衰减与剥离多径回波的可微逆散射层。
2. **由 Observation 4**：
   - 在 20dB 高斯白噪声与粉红噪声下，TG-DeepONet 的核心指标衰减幅度均控制在 6%~10% 之内，充分满足“20dB 强噪声扰动下衰减幅度 < 15%”的标准，证明卷积滤波与时延偏置注意力的自适应去噪能力；
   - 然而，当声速发生 $+1\%$ 扰动时，在 4500m 完井深度处往返到时产生高达 $\mp 61.5\,\mathrm{ms}$ 的漂移，直接超出了相对到时窗机制 $[-50\,\mathrm{ms}, +250\,\mathrm{ms}]$ 的容限，导致关键波前起跳丢失或相位混乱，$R^2$ 崩溃至 $-0.0950$；
   - **推论**：纯信号局部窗截取机制具有结构性硬脆性，无法泛化至声速存在测量误差的现场工况，必须将声学参数与阻抗反射率解耦。
3. **由 Observation 3**：
   - 本机纯 CPU 推理 100 例样本仅需 42.7 ms，单步训练耗时 55 ms，单 Epoch 仅 1.38 秒，训练 100 Epochs 仅需 2.3 分钟；
   - **推论**：算力完全充裕，无需 GPU 加速即可支持 Phase 3 可微逆散射层算子与神经算子的联合训练与高频消融。
4. **由 Observation 5**：
   - Phase 3 验收标准要求“多裂缝位置检出与起裂分类准确率 F1-score > 0.88（容差 $\pm 10\mathrm{m}$）”；
   - 当前网络架构仅预测 $\alpha_j$ 与 $C_{f,j}$，默认设计位置 $x_j$ 完全精确已知，且当前数据集标签中存在未起裂/砂堵死簇（Type IV）及断层沟通簇（Type V）；
   - **推论**：必须在网络中显式增加起裂存在性分类头 $\hat{e}_j \in [0, 1]$ 与位置偏差微调头 $\Delta x_j \in [-20, 20]\,\mathrm{m}$，并在 `src/metrics.py` 中补充容差 $\pm 10\mathrm{m}$ 的 F1 计算模块。

---

## 3. Caveats (局限与未涉领域)

1. **数据集簇数上限局限**：现存权威数据集 `moc_v2_1k_dataset.h5` 中 $N_c$ 仅覆盖 1~6 簇，未包含 7~8 簇活跃案例（虽预留了 8 槽位）；若 Phase 3 需强制验证 7~8 簇超密工况，需运行 `moc_simulate.v2` 正演流水线补充对应算例。
2. **测试集声速摄动假设**：本调查通过在工况特征向量 `cond[1]` 中注入 $\pm 1\%$ 声速摄动模拟操作现场的声速估计误差，未对 100 例原始时程波形重新运行变声速 MOC 正演物理仿真。
3. **断层沟通与强滤失标签利用**：原始 H5 中存储的拟达西滤失系数 $k_{\mathrm{leak}}$ 与限流流阻 $K_p$ 尚未引入反演损失函数，当前模型仅约束流量分配 $\alpha$、顺应性 $C_f$ 与空间连续场 $m_\alpha(x)$。

---

## 4. Conclusion (结论与行动方案)

1. **数据与性能就绪**：1,000 例物理数据集质量高、规范完备，CPU 训练与评估吞吐量极高（100 例推理仅 42.7 ms，训练单轮 1.38 s），具备立即开启 Phase 3 研发的坚实工程底座；
2. **加性噪声达标，声速漂移需解混**：20dB 强噪声衰减率 6%~10% 完全达标，但声速摄动下脆弱性极强，验证了引入波动方程逆散射层剥离算子的必然性；
3. **核心攻关路径明确**：
   - **路径 A (R1 算子)**：实现端到端可微逆散射层剥离算子（Differentiable Layer-Stripping Layer），沿井筒由跟部到趾端逐级消除多径串扰与透射衰减，输出反射率序列 $\Gamma_j$ 与支路导纳 $Y_{b,j}$，将密集多簇 $R^2$ 由 0.1319 突破至 0.75 以上；
   - **路径 B (R2 网络)**：在离散头中补齐起裂存在性分类概率 $\hat{e}_j$ 与位置偏差微调 $\Delta x_j$，在 `metrics.py` 中实现 F1-score（$\pm 10\mathrm{m}$ 容差）评测，达成 F1 > 0.88 与全集 $\alpha$ MAE < 0.08；
   - **路径 C (R3 审计)**：编写独立的 `evaluate_robustness.py` 自动化评测流水线，系统覆盖 30dB, 20dB, 10dB 白/粉红噪声与 $\pm 1\%$ 声速摄动。

---

## 5. Verification Method (独立验证方法)

任何接替智能体或审计员可通过以下标准命令独立复现并核验本报告的全部观测事实与指标数据：

1. **测试套件回归验证**：
   ```powershell
   pytest -q PaperC_CJNO_Wellbore_Inversion/tests
   ```
   *预期结果*：12 项测试 100% 全部通过（通过耗时约 3.5 秒）。
2. **基准指标与全模型推理速度核验**：
   ```powershell
   python -c "from PaperC_CJNO_Wellbore_Inversion.experiments.evaluate_ablation import evaluate_test_set; evaluate_test_set()"
   ```
   *预期结果*：成功加载 100 例测试集，输出 TG-DeepONet $\alpha$ MAE = 0.1336，$R^2$ = 0.5666，$W_1$ = 8.39 m，单纯形最大偏差 $1.19 \times 10^{-7}$。
3. **20dB 噪声与声速扰动压力测试复现**：
   在工作目录中调用报告脚本核验 20dB 加性白噪声恶化率（7.6%）、粉红噪声恶化率（6.1%）及 $+1\%$ 声速扰动下的指标崩溃现象。
4. **失效条件 (Invalidation Conditions)**：
   若重新加载 `cache_1k_t4096_c1024_g500.npz` 时样本数不为 1000、或密集多簇（$N_c \ge 4$）在无层剥离下的 $R^2$ 超过 0.50、或 20dB 噪声恶化超过 15%，则本调研报告的对应结论失效。
