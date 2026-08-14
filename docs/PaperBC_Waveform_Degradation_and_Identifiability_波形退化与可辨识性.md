# Brunone 摩阻使井口倒谱变糊，但不构成裂缝定位的信息极限

**English title:** Brunone Friction Blurs Wellhead Water-Hammer Cepstra without Imposing an Information Limit on Fracture Location

**稿件：** Paper BC（原 B 波形退化 + 原 C 可辨识性合成）· 目标 SPE Journal · 中文工作初稿（2026-08-14）  
**姊妹篇：** [Paper A](./PaperA_Topology-Organized_Cepstral_Response_倒谱拓扑组织与首缝非局部性.md)（稳态下倒谱是网络观测量，本文不重复）  
**细目备查：** [原 Paper B](./PaperB_Brunone_Waveform_Degradation_非定常摩阻波形退化与倒谱模糊.md) · [原 Paper C](./PaperC_Identifiability_and_Error_Budgets_可辨识性与误差预算.md)（主张以本稿为准）

**关键词（≤5）：** water hammer; Brunone friction; cepstrum; Cramér–Rao bound; model misspecification

**一句话论证：** 在停泵井筒水击诊断中，规定 Brunone \(k\) 使倒谱所依赖的回声波包衰减、后移并展宽，Re 相关 \(k\) 下盲倒谱单缝深度偏差为 10–20 m；但同一观测模型上位置 CRB 为厘米级、全波形 MLE 贴界。实用墙是摩阻形式与波速失配，不是波形“糊了”。

本稿数字全部来自已完成实验或 EXP-20260730-024 对既有 30 组时间序列的复算。未做的实验不写结论。参考文献条目投稿前按 author–year 补全，本稿不编造文献。

---

## 摘要

停泵井口水击被用来定位多簇裂缝。非定常 Brunone 摩阻下，井口回声波包变钝、后移，倒谱拾取常被解释为近距不可分辨。本文将波形退化、估计器偏差和信息下界分开测量，而不是把“峰糊了”写成一条分辨率曲线。

在规定 Brunone 系数 \(k\in\{0,0.01,0.02,0.05,0.1,0.2\}\) 与四条等缝间距 \(D\in\{5,10,20,50,100\}\,\mathrm{m}\) 的 30 工况矩阵上，井口水头时间导数的反射包络随 \(k\) 增大而衰减、展宽并后移。锚点 \(D=20\,\mathrm{m}\)、\(k=0.2\)：相对 \(k=0\) 的首峰漂移为 **277 ms**，能量弥散时间 EST 为 **101 ms**（相对 \(k=0\) 的增量 100 ms）。共同参考 STFT（Hann，\(n_{\mathrm{perseg}}=128\)，重叠 120，dB 参考为同间距 \(k=0\)）给出频率选择性耗散：同一锚点上 60–150 Hz 频带能量下降 **44.1 dB**，20–60 Hz 下降 25.4 dB，0–20 Hz 仅下降 12.9 dB。由基频、onset 与峰值提取的表观速度降幅不同（分别为 11.0%、1.3%、4.7%），单一标量波速不再描述退化波形。上述观察支持尖锐波前的频率选择性耗散，不构成相速度 \(c_p(\omega)\) 的测量。

在雷诺数相关 \(k\)（`brunone_k(Re)`）的井筒算例上，盲倒谱协议下 5 个 Brunone 单缝在 10 m 容差时 F1 全为 0、20 m 容差时全为 1，系统偏差落在 **10–20 m**；准稳态对照在 2 m 容差下即为 1。LHS 语料 8 s 探针的实现 \(k\) 中位为 0.014、\(|V|\) 加权平均 0.0079，属于弱档，不得用 \(k=0.2\) 的 277 ms 漂移外推现场。

同一观测模型（井长 5000 m，波速 1450 m/s，Brunone，记录 50 s，零相位 4 阶 Butterworth 截止 20 Hz，加性白噪声）上，SNR = 40 dB 时单缝位置 CRB 为 **0.025 m**。Brunone 相对稳态的联合位置 CRB 仅膨胀 **1.12–1.32 倍**（SNR = 30 dB，间距 5–50 m）。全波形 MOC-MLE 效率比 RMSE/CRB = **0.96–1.39**（\(n\le 5\) 时仍为 0.86–1.18）。倒谱在无限信噪比下仍有 **12.0 m** 误差。真正限制深度的是模型误差：用稳态副本拟合 Brunone 数据（或反向）给出 **29.6 m / 31.9 m**；波速 −1% 的非线性 MLE 单缝中位绝对偏差为 **120 m**（一阶走时 28–36 m 仅量级，且未通过比值 < 2 的门控）。配对 300 工况剪刀差成立：倒谱 F1 从 0.232 升至 0.642，字典剥离从 0.396 降至 0.089，PhaseNet 从 0.746 降至 0.232。

因此倒谱模糊是耗散波前下的估计器畸变，不是信息墙。现场应使用与数据一致的摩阻形式做全波形匹配，倒谱只作初值，波速必须标定。结论限于给定带宽与白噪声的合成记录，不宣称现场厘米级精度。

---

## 1 引言

水力压裂后的停泵水击把井筒变成一根单通道波导：井口关泵产生的压力锋沿井筒传播，在裂缝节点与趾端发生反射，再回到井口。把回声延迟读成深度、把峰强读成裂缝规模，是现场诊断里最直接的用法。倒谱把周期回声压成深度轴上的峰，因此常被当作裂缝定位器。

非定常摩阻改变了这件事的外观。Brunone 项对尖锐波前耗散更强，井口反射包从离散脉冲变成低矮宽丘。文献与早期项目叙述容易把三种不同的失败写成一句“近距不可分辨”：（i）时域峰融合，相邻回声不再形成两个峰；（ii）倒谱拾取出现系统偏差，峰还在但深度错了；（iii）记录里已经没有足以分开裂缝位置的信息。这三件事的实验含义完全不同。若是（iii），任何估计器都没有用；若是（ii），应当换估计器或改正演；若只是（i），它描述的是某一种显示方式，不是反问题的界。

本文按这个顺序回答。第 4 节在规定 \(k\) 的 30 工况矩阵上测量波形如何退化。第 5 节在 Re 相关 \(k\) 上用盲倒谱协议测量深度偏多少。第 6 节用 Fisher 信息 / Cramér–Rao 界（CRB）和全波形极大似然（MLE）问信息还在不在。第 7 节把摩阻形式失配和波速失配与倒谱偏差放在同一张误差预算里。姊妹篇 A 处理准稳态摩阻下倒谱作为网络观测量的拓扑组织，本文不重复那条矩阵。

历史上有过一条操作分辨率 \(\Delta d_{\mathrm{DR}}\approx 38\,\mathrm{m}\)（倒谱谱支撑、DR = 80 dB）。它是该变换的旁瓣宽，随动态范围从约 220 m 变到约 4 m，并且盲协议下稳态无噪声 5 m 双缝可以被分开。本文不再把它当作基线，也不把它写进结果表。

本文不报：严格频散 \(c_p(\omega)\)；把规定 \(k\) 映射到滑溜水 / 线性胶 / 冻胶；Rayleigh 最小缝距表；现场已达厘米级。神经算子波形代理与学习检测器只在失配剪刀差里作为对照出现，不作为第三条主线。

---

## 2 正演、观测与指标

### 2.1 井筒–裂缝 MOC

正演为一维特征线法（MOC），Courant 数精确为 1。井长 \(L=5000\,\mathrm{m}\)，内径 0.1397 m，标称波速 \(a=1450\,\mathrm{m/s}\)，停泵前流速 \(1\,\mathrm{m/s}\)，趾端定压储层，井口为停泵流速阶跃。裂缝为集总节点

\[
Q_f=C_f\,\partial_t H+k_{\mathrm{leak}}\sqrt{H-H_{\mathrm{ext}}}.
\]

网格吸附使裂缝落到 \(dx=a\,\Delta t\approx 1.45\,\mathrm{m}\) 的节点上。观测始终是井口水头 \(H_{\mathrm{wh}}\)（或其时间导数）。

### 2.2 两套 Brunone \(k\)，不得混用

非定常项取 Brunone 形式

\[
J_u=\frac{k}{2}\,\Delta t\Bigl(\partial_t V+a\,\mathrm{sign}(V)\,|\partial_x V|\Bigr).
\]

**矩阵 A** 把 \(k\) 规定为外部常数 \(\{0,0.01,0.02,0.05,0.1,0.2\}\)，用于隔离参数效应；\(k=0\) 退化为准稳态 Darcy。**矩阵 B / LHS** 使用代码中的 `brunone_k(Re)`（Vardy 剪切衰减系数），\(k\) 随局部瞬时雷诺数变化。规定 \(k=0.2\) 是上界示意，不是 LHS 语料的实现值。

### 2.3 波形指标（EXP-024 预注册）

对 \(H_{\mathrm{wh}}\) 取 \(\mathrm{d}H/\mathrm{d}t\)。默认分析窗 \([6.5,7.5]\,\mathrm{s}\)（首缝几何到时 \(t_s+2\times 4100/a\approx 6.655\,\mathrm{s}\)）。

- **峰漂移** \(\Delta t_{\mathrm{peak}}\)：窗内首个超过 \(0.1\max(\mathrm{d}H)\) 的峰，相对同间距 \(k=0\)。
- **EST**：正部能量累积从 10% 到 90% 的时间跨度。它不是半高宽。强退化时 FWHM 出现假性收窄，EST 在锚点上保持单调。
- **峰谷对比度** \(C_v=(P_{\min}-V)/P_{\min}\)：不足两峰则记 0。合成等幅高斯双峰上该量随分离度单调不减；完全重叠时为 0。本文称它为操作性双峰指数，不把它换算成 Rayleigh 最小间距。
- **STFT**：Hann 窗，\(n_{\mathrm{perseg}}=128\)，重叠 120，`scaling=spectrum`，分析段 \([6.4,7.2]\,\mathrm{s}\)。频带能量为带内谱之和。dB 参考是**同间距 \(k=0\)**，不是任意色标。频带取 0–20、20–60、60–150 Hz。
- **表观速度**：\(a_{f0}=4L f_0\)（停泵后全记录 FFT 基频），\(a_{\mathrm{onset}}=2x_1/(t_{\mathrm{onset}}-t_s)\)，\(a_{\mathrm{peak}}=2x_1/(t_{\mathrm{peak}}-t_s)\)。三者是不同特征的操作定义，不是窄带相速度。

### 2.4 观测模型、CRB 与 MLE

可辨识性实验的观测为

\[
y_k=s_k(\theta)+n_k,\qquad n_k\sim\mathcal N(0,\sigma^2),
\]

其中 \(s(\theta)\) 是停泵后井口水头经零相位 4 阶 Butterworth 限带（默认截止 \(f_c=20\,\mathrm{Hz}\)）后的采样，\(\sigma\) 由与分层基准相同的 SNR 定义反推。Fisher 信息 \(\mathrm{FIM}=J^\top J/\sigma^2\)，CRB 为 \(\mathrm{FIM}^{-1}\)。位置导数用整数格点中心差分；波速导数取 \(|\delta a/a|=1/N\) 并扣除网格吸附漂移。差分收敛域为 \(f_c\cdot\Delta t\lesssim 0.021\)（\(\Delta t=1\,\mathrm{ms}\) 时 \(f_c\le 20\,\mathrm{Hz}\)）。

A 档只估位置，其余参数已知。B 档联合位置、\(\log_{10}C_f\)、\(\log_{10}k_{\mathrm{leak}}\) 与波速。全波形 MLE 为网格搜索加真值处一步 Gauss–Newton；效率比定义为位置 RMSE 除以对应 CRB。倒谱与字典剥离走同一含噪实现，作为低效对照。

盲倒谱检测器的输入不含真值缝距、缝数或随真值缩放的匹配容差；评分在固定容差 \(\{2,5,10,20,40\}\,\mathrm{m}\) 上单独进行。

---

## 3 数据

**规定 \(k\) 矩阵（波形退化）。** 30 个四裂缝工况，首缝 4100 m，\(C_f=10^{-5}\,\mathrm{m}^2\)，\(k_{\mathrm{leak}}=10^{-4}\)，\(H_{\mathrm{ext}}=100\,\mathrm{m}\)，\(\Delta t=1\,\mathrm{ms}\)，记录 100 s。时间序列在 `output/analysis/brunone_spacing_effect/D{D}_k{k}/moc_timeseries.csv`。EXP-20260730-020 审计现象；EXP-20260730-024 在同一批 CSV 上重算指标，输出 `output/analysis/brunone_spacing_effect/audit_v2/`（`metrics_v2.csv`、`band_energy.csv`、`manifest.json`、`paper_numbers.json`）。不重跑 MOC。

**Re 相关 \(k\) 与盲倒谱。** leakoff 目录上的盲协议复算（EXP-20260801-001）；LHS 实现 \(k\) 的 8 s 全场探针（EXP-20260730-010）。探针只覆盖 \(t_f=8\,\mathrm{s}\) 与单一 \((C_f,k_{\mathrm{leak}})\) 组合，不能当作 50 s 全程等效常数 \(k\)。

**可辨识性。** 对齐配置 \(L=5000\,\mathrm{m}\)，\(a=1450\,\mathrm{m/s}\)，Brunone，\(t_f=50\,\mathrm{s}\)，\(f_c=20\,\mathrm{Hz}\)。效率场景 `single_4000`、`dual_10m`、`dual_40m`，每档 \(n_{\mathrm{MC}}=30\)（EXP-20260806-003）。高阶 \(n=3/4/5\)（EXP-20260807-001）。摩阻形式失配在单缝、SNR = \(\infty\) 上实测（同 EXP-003）。波速 −1% 为 4 个单缝 + 2 个双缝探针，计划 8 个中完成 6 个后中止（EXP-20260807-003，completed-partial）。配对剪刀差用几何一致的 300 个 LHS 工况（EXP-20260807-002）。CRB 对间距的扫描来自 \(t_f=20\,\mathrm{s}\) 的 270 行表（EXP-20260806-002）；与效率实验对齐的单缝 CRB 以 \(t_f=50\,\mathrm{s}\) 的 0.025 m 为准。

---

## 4 结果：规定 \(k\) 下波形如何退化

### 4.1 峰衰减、后移与 EST 展宽

**图 1**（`audit_v2/fig1_waveform_evolution.png`）给出 \(D=20\,\mathrm{m}\) 时 \(\mathrm{d}H/\mathrm{d}t\) 随 \(k\) 的演化。\(k=0\) 时反射为尖锐脉冲（首峰幅值约 \(5.92\times 10^4\)）；\(k=0.2\) 时首峰幅值降至约 \(1.12\times 10^3\)（约 53 倍），包络变成不对称宽丘。

**表 1.** \(D=20\,\mathrm{m}\)、分析窗 \([6.5,7.5]\,\mathrm{s}\)。峰漂移与 \(\Delta\)EST 相对同间距 \(k=0\)。EST 为该窗内绝对值。\(C_v\) 为操作性峰谷对比度。表观速度单位 m/s。

| \(k\) | \(\Delta t_{\mathrm{peak}}\) (ms) | EST (ms) | \(\Delta\)EST (ms) | \(C_v\) | 峰数 | \(a_{f0}\) | \(a_{\mathrm{onset}}\) | \(a_{\mathrm{peak}}\) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.0 | 0 | 1.00 | 3 | 1409 | 1451 | 1450 |
| 0.01 | 13.0 | 8.0 | 7.0 | 1.00 | 3 | 1408 | 1450 | 1447 |
| 0.02 | 26.0 | 12.0 | 11.0 | 0.97 | 2 | 1407 | 1449 | 1444 |
| 0.05 | 67.0 | 27.0 | 26.0 | 0.14 | 2 | 1397 | 1446 | 1433 |
| 0.1 | 138.0 | 51.0 | 50.0 | 0 | 1 | 1360 | 1441 | 1416 |
| 0.2 | **277.0** | **101.0** | **100.0** | 0 | 1 | 1254 | 1432 | 1383 |

锚点 \(k=0.2\)：峰漂移 277 ms。若用标量 \(a/2\) 把该时延换成深度，得到 201 m。这只描述**规定 \(k=0.2\) 示意**，不是第 5 节 Re 相关 \(k\) 下的 10–20 m 倒谱偏差，二者不得互换。

EST 从 1.0 ms 增到 101 ms。同一锚点上首峰 FWHM 在 \(k=0.1\) 为 38 ms、\(k=0.2\) 为 31 ms，出现假性收窄；因此主文宽度用 EST，不用 FWHM。

分析窗敏感性（\(D=20\,\mathrm{m}\)、\(k=0.2\)）：\([6.5,7.5]\)、\([6.4,7.2]\)、\([6.6,7.0]\)、\([6.5,8.0]\) 四窗给出的峰漂移与 EST **均为 277 ms 与 101 ms**。该锚点对窗端点不敏感。大间距（\(D=100\,\mathrm{m}\)）下同一 1 s 窗会吞进后缝，EST 不再是“首缝宽度”，故锚点固定在 \(D=20\,\mathrm{m}\)。

\(k=0\) 时 10% 峰高阈值在 \(D=20\,\mathrm{m}\) 窗内检出 3 个峰而不是 4 个：第 4 条缝的反射低于该阈值。本文不把“稳态下四峰尽数可分”写成检测器结论；波形图上的视觉多峰与阈值化峰数不是同一句话。

### 4.2 频率选择性耗散

**图 3–4** 使用共同色标（由 \(k=0\) 的 STFT 分位数固定）与同间距 \(k=0\) 的 dB 参考。**表 2** 是 \(D=20\,\mathrm{m}\) 的 STFT 频带能量，单位为相对 \(k=0\) 的 dB。括弧中为同一窗 \(\mathrm{d}H/\mathrm{d}t\) 的 rFFT 对照。

| \(k\) | 0–20 Hz | 20–60 Hz | 60–150 Hz |
|---:|---:|---:|---:|
| 0.01 | −1.6 (−1.1) | −4.7 (−4.8) | −17.0 (−17.2) |
| 0.02 | −2.7 (−1.8) | −9.9 (−10.3) | −26.2 (−26.6) |
| 0.05 | −6.4 (−3.8) | −19.7 (−20.3) | −40.5 (−40.6) |
| 0.1 | −9.3 (−5.7) | −26.9 (−27.9) | −42.8 (−42.7) |
| 0.2 | **−12.9 (−10.6)** | **−25.4 (−26.5)** | **−44.1 (−44.9)** |

高频相对低频优先下降。\(k=0.2\) 时 60–150 Hz 相对 \(k=0\) 为 −44.1 dB（rFFT −44.9 dB），0–20 Hz 仅为 −12.9 dB。旧文未审计的“60–150 Hz 超过 40 dB”在**本清单的共同参考下成立**，但必须带参考：同间距 \(k=0\)、STFT 参数如上、信号为 \(\mathrm{d}H/\mathrm{d}t\)。20–60 Hz 在 \(k=0.2\) 处比 \(k=0.1\) 略回升 1.5 dB，中频不是严格单调；主文只主张高频相对低频的选择性，不主张每一频带都对 \(k\) 严格单调。

这是耗散的频率选择性，不是已测的 \(c_p(\omega)\)。STFT 没有提供展开相位或群延迟。

### 4.3 表观速度分离

\(D=20\,\mathrm{m}\)、\(k=0.2\) 相对各自 \(k=0\) 基线：\(a_{f0}\) 从 1409 降至 1254 m/s（**11.0%**），\(a_{\mathrm{peak}}\) 从 1450 降至 1383 m/s（**4.7%**），\(a_{\mathrm{onset}}\) 从 1451 降至 1432 m/s（**1.3%**）。\(k=0\) 时 \(a_{f0}\) 已低于标称 1450 m/s，基频测量本身不是无偏波速计。

把 277 ms 峰漂移拆成 onset 延迟与峰相对 onset 的额外延迟，是当前阈值下的操作分解：onset 从 6.653 s 移到 6.725 s（72 ms，约占 26%），其余约 205 ms（约 74%）来自峰形后移。这不是“传播频散 / 波形频散”的唯一物理分解。

### 4.4 时域双峰对比度随 \(k\) 下降——到此为止

合成等幅高斯（\(\sigma=20\,\mathrm{ms}\)）上 \(C_v\) 随峰距从 0 单调增到 1，完全重叠时为 0（`audit_v2/unit_test_cv.csv`）。30 工况热力图（**图 6**）上，小间距、大 \(k\) 时 \(C_v\to 0\)：\(D=20\,\mathrm{m}\) 在 \(k=0.05\) 已降至 0.14，\(k\ge 0.1\) 为 0。这与第 4.1 节包络融合一致。

本文**不**由此给出“某 \(k\) 对应的最小可分辨缝距”。水击反射不是理想点扩散函数；\(C_v\) 依赖寻峰阈值与分析窗；近距可分性由第 6 节的 CRB/MLE 回答。旧 Rayleigh 热力图不进入主文主张。

---

## 5 结果：Re 相关 \(k\) 下倒谱偏多少

规定 \(k=0.2\) 的 277 ms / 201 m 不得外推到 LHS 或 leakoff 算例。LHS 使用 `brunone_k(Re)`。8 s 全场探针（EXP-20260730-010）给出实现 \(k\)：

| 统计量 | \(k\) |
|---|---:|
| \(|V|\) 加权平均 | 0.0079 |
| 中位 | 0.0138 |
| P75 | 0.0259 |
| P95（层流分支封顶） | 0.0345 |

该分布跨骑在规定矩阵的 0.01–0.03 档，远低于 0.2。探针时长 8 s，50 s 全程中位数只会更高，但仍没有证据把它抬到 0.2。

盲协议摘除了三处真值泄漏（寻峰最小间距、匹配容差、保留峰数均不再由真值导出）后，5 个 Brunone 单缝（间距标签 D5–D100，实际均为单缝）在历史上 135 m 容差下全部“匹配成功”；盲评分则为：

| 摩阻 | F1 @ 10 m | F1 @ 20 m |
|---|---:|---:|
| Brunone，5 例单缝 | **0** | **1** |
| 准稳态，5 例单缝 | 1（2 m 容差已为 1） | 1 |

系统偏差因此落在 **10–20 m**。准稳态对照 < 2 m。135 m 容差把该偏差完全掩盖，不得再引用为“Brunone 下倒谱定位成功”。

全库 2000 条 Brunone LHS 上盲倒谱 F1@10 m = 0.238、单缝中位误差 190.7 m（EXP-20260802-001）。那是**含多缝混叠与假峰的检测器表现**，与上面 5 例单缝的 10–20 m 系统偏差不是同一个量。配对 300 工况上 Brunone 倒谱中位误差为 12.2 m（第 7.3 节），与单缝容差扫描同量级。

---

## 6 结果：信息还在——糊的是估计器

### 6.1 位置 CRB 不随近距恶化

对齐配置、Brunone、\(f_c=20\,\mathrm{Hz}\)、SNR = 40 dB、A 档单缝：CRB \(\sigma_x=0.0254\,\mathrm{m}\)（效率实验 `single_4000`，\(t_f=50\,\mathrm{s}\)）。

\(t_f=20\,\mathrm{s}\) 的间距扫描（Brunone，\(f_c=20\,\mathrm{Hz}\)，SNR = 40 dB，波速已知）上，放置间距 5.8–120.4 m 对应 \(\sigma_{x_0}=0.025\)–\(0.032\,\mathrm{m}\)，最小在约 13 m 间距，最大在 5.8 m 间距，变化不到 1.3 倍，没有随间距缩小而发散。联合估波速时该扫描上 \(\sigma_{x_0}=0.034\)–\(0.042\,\mathrm{m}\)，同样平坦。

最差档 SNR = 10 dB、\(t_f=20\,\mathrm{s}\) 时，EXP-20260806-002 报告 \(\sigma_{x_0}\) 仍为 0.93–1.32 m 量级，比倒谱无噪 12 m、盲库中位 190 m 低两个数量级以上。

Brunone 相对准稳态：SNR = 30 dB、\(f_c=20\,\mathrm{Hz}\)、联合估波速，间距 5 / 20 / 50 m 的位置 CRB 比为 **1.32 / 1.12 / 1.12**。非定常摩阻抬高界，但不把厘米级界抬成十米级。

短记录（\(t_f=20\,\mathrm{s}\)）上波速作为讨厌参数对绝对深度的惩罚约为 ×1.15–1.82，对间距约为 ×1.00–1.02。拉长到 \(t_f=50\,\mathrm{s}\) 并纳入趾端 \(2L/a\) 后，单缝 B 档 \(a\) 惩罚降至 ×1.01。长记录下绝对深度也可以联合估计；短记录仍应优先报间距。

\(D=20\,\mathrm{m}\) 准稳态、\(f_c=20\,\mathrm{Hz}\)、SNR = 30 dB 的一维失配剖面在 3401–4600 m 上全局最小落在真值 1 个网格内；满足 \(\Delta\chi^2<9\) 且远离真值的竞争极小为 0。优化落入远盆地不是本配置下的失败原因。该检验尚未在多缝、Brunone、未知阶数上重复。

### 6.2 全波形 MLE 贴界，倒谱离界

**表 3.** 效率实验，A 档，\(f_c=20\,\mathrm{Hz}\)，\(n_{\mathrm{MC}}=30\)。效率比 = RMSE / CRB。

| 场景 | SNR | MLE RMSE (m) | CRB (m) | 效率比 | 倒谱 RMSE (m) | P0 RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| single_4000 | \(\infty\) | 0 | — | — | **12.0** | 0.12 |
| single_4000 | 40 | 0.029 | 0.025 | **1.14** | 12.0 | 0.12 |
| single_4000 | 30 | 0.086 | 0.080 | 1.07 | 294 | 0.13 |
| single_4000 | 20 | 0.354 | 0.255 | 1.39 | 473 | 0.20 |
| dual_10m | 40 | 0.039 | 0.031 | 1.26 | 309 | 353 |
| dual_10m | 30 | 0.093 | 0.097 | 0.96 | 290 | 7.0 |
| dual_40m | 40 | 0.034 | 0.032 | 1.04 | 12.8 | 24.3 |
| dual_40m | 20 | 0.326 | 0.323 | 1.01 | 404 | 21.0 |

A 档效率比落在 **0.96–1.39**。倒谱在 SNR = \(\infty\) 仍为 12.0 m（单缝）或 11.6–12.7 m（双缝），与第 5 节 10–20 m 系统偏差同量级：无噪声也消不掉。P0 字典在单缝上接近网格量化（0.12 m），在 `dual_10m`、SNR = 40 dB 上 RMSE = 353 m，同场景 SNR = 30 dB 又回到 7 m——近距剥离不稳定，不拿来否证 MLE。

\(n=3/4/5\) 等间距、SNR ∈ {40, 30, 20} dB 的 18 个效率比落在 **0.86–1.18**，门控 ≤ 3 全部通过。CRB 随缝数缓增，效率比不崩。

判决句：**在模型正确、缝数已知的合成记录上，位置信息在井口通道里；糊的是倒谱这类回声估计器，不是记录里没有裂缝位置。**

### 6.3 缝数不是位置 CRB

嵌套构型、单次正演 BIC（\(n_{\max}=5\)，oracle 初值）在 dual/triple × {10, 40} m、SNR ∈ {∞, 40, 30, 20} 上 exact-count = 1.0（30/30），对有效样本量 \(n_{\mathrm{bw}}/n_{\mathrm{half}}/n_{\mathrm{raw}}\) 不敏感。PhaseNet 盲计数约 30.5%。二者不是同一问题：前者是嵌套模型选择，后者是盲检测。位置 CRB 在给定阶数下定义，不约束盲计数。

---

## 7 结果：真正的墙是模型失配

### 7.1 摩阻形式

单缝、SNR = \(\infty\)、非线性全波形 MLE（EXP-20260806-003）：

| 真模型 → 拟合模型 | RMSE (m) | 一阶预测 (m) | 残差吸收比 |
|---|---:|---:|---:|
| brunone → steady | **29.6** | 0.34 | 0.010 |
| steady → brunone | **31.9** | −2.49 | 0.033 |
| brunone → brunone | 0 | 0 | 0 |

形式配对时误差为 0。形式互换约 30 m，且几乎不被位置参数吸收（吸收比 0.01–0.03），拟合残差仍大。把 Brunone 的 \(k_{\mathrm{scale}}\) 当作自由参数后，steady→brunone 的 RMSE 从 31.9 m 降到 24.7 m（\(\hat k_{\mathrm{scale}}=0.72\)），残余仍远大于 CRB。

EXP-20260806-002 的一阶线性化把形式误差估成 0.6–10 m，比非线性实测小约 3 倍。一阶公式对**形式**失配只作方向指示。主文采用 29.6 / 31.9 m。

同一张诊断表上，Brunone 系数错 10% 的一阶深度偏差为 **0.68–1.05 m**（间距 5–50 m）。中等的 \(k\) 标定误差不是 10–20 m 倒谱偏差的主因；换错摩阻**形式**才是。

### 7.2 波速 −1%

数据波速 1435.5 m/s、模型 1450 m/s。一阶走时给出 28–36 m。非线性坐标下降 MLE、4 个单缝探针：

| \(x_{\mathrm{true}}\) (m) | 偏差 (m) |
|---:|---:|
| 3600 | +89.1 |
| 3933 | +129.9 |
| 4267 | +130.1 |
| 4600 | −110.4 |

中位绝对偏差 **120 m**，与一阶预测的比为 3.3–4.3，门控“比值 < 2”未通过。Courant = 1 下改 \(a\) 会重划分网格，连续走时公式低估。双缝 2/4 探针共模偏差较小但 RMSE 仍为 102–130 m，匹配不稳定；探针 7–8 因墙钟中止。主文采用单缝中位 120 m，并标明 completed-partial。

要让波速偏差落到噪声界，EXP-002 给出临界 \(\delta a/a\sim 2\times 10^{-5}\)。那是该观测模型下的标定需求，不是现场已达到的精度。

### 7.3 配对剪刀差

几何配对的 300 个 LHS 工况（50/50 抽查完全一致）：

| 方法 | Brunone F1@10 m | 稳态 F1@10 m | \(\Delta\)F1 | Brunone 中位误差 (m) | 稳态中位误差 (m) |
|---|---:|---:|---:|---:|---:|
| 倒谱 | 0.232 | **0.642** | +0.410 | 12.2 | 0.53 |
| P0 剥离 | 0.396 | **0.089** | −0.307 | 16.7 | 41.3 |
| PhaseNet | 0.746 | **0.232** | −0.514 | — | — |

PhaseNet 在稳态验证集上把阈值从 0.35 重校准到 0.20，F1 仅从 0.228 到 0.232，掉点不是阈值未调。倒谱在更接近其纯回声假设的稳态数据上变好；在 Brunone 上训练或构库的剥离与网络迁到稳态则变差。控制变量是估计器隐含正演与数据生成模型的距离。摩阻匹配的基准会高估可迁移性。F1@10 m 的容差小于形式失配 30 m，失配叙事优先报深度误差与配对 \(\Delta\)F1。

### 7.4 误差预算（并排，不可合成一条分辨率曲线）

| 来源 | 深度误差 | 条件 | 证据 |
|---|---|---|---|
| 位置 CRB | 0.025 m | 单缝，SNR = 40 dB，\(f_c=20\,\mathrm{Hz}\)，模型正确 | EXP-003 |
| 全波形 MLE | 0.029 m（效率比 1.14） | 同上 | EXP-003 |
| 倒谱，SNR = \(\infty\) | 12.0 m | 对齐单缝 | EXP-003 |
| 倒谱，Re 相关 \(k\) 单缝 | 10–20 m | 盲容差扫描，5 例 | EXP-001 |
| Brunone \(k\) 错 10% | 0.68–1.05 m | 一阶，间距 5–50 m | EXP-002 |
| 摩阻形式互换 | 29.6 / 31.9 m | 非线性 MLE，单缝，无噪 | EXP-003 |
| 波速 −1% | 中位 \(\lvert\mathrm{bias}\rvert=120\,\mathrm{m}\) | 非线性，4 单缝探针 | EXP-007-003 |

倒谱 10–20 m、形式失配 ~30 m、波速 −1% ~120 m 是三件不同的偏差，不能加总成“分辨率 = xx m”。

---

## 8 讨论

中心事实是分层的。规定 \(k\) 把井口包络变成频率选择性耗散下的宽丘（第 4 节）。Re 相关的弱 \(k\) 档仍足以让倒谱单缝偏 10–20 m（第 5 节）。CRB 与 MLE 说明这 10–20 m 不是信息用尽（第 6 节）。换错摩阻形式或波速，偏差立即超过倒谱系统误差（第 7 节）。

工程处方因此是模型基的，而不是“更好的峰检测”：全波形反演必须用对摩阻形式；倒谱只提供粗初值（管线烟测上单缝曾从约 12 m 收到 0.012 m，双缝 10 m 从约 21 m 收到 0.028 m，那是正确模型下的精修，不是失配解）。波速必须标定；−1% 已经把深度拉到百米。评价必须包含失配集，否则 Brunone 训练、Brunone 测试的 F1 会系统性高估。

模态阻尼率 \(\alpha\) 与 \(\sum k_{\mathrm{leak}}\) 的秩相关为 0.87–0.92，与缝数仅 0.44–0.51（EXP-20260802-001）。阻尼率携带滤失信息，不是缝深或缝数定位器。本文不把它发展成第三条方法。

机制等级必须写清楚。已观察：高频相对低频衰减、峰后移、EST 展宽、表观速度分离、倒谱低效、CRB 可达、形式与波速失配主导。解释：Brunone 项对尖锐波前耗散更强，回声估计器被带着走。未证明：非线性相位意义上的严格频散；现场厘米级；把 \(k\) 对应到命名压裂液。

与姊妹篇 A 的分工：A 在准稳态下问倒谱峰是不是局部裂缝标签；本文在 Brunone 下问“糊了”是不是信息墙。两篇都不把 38 m 旁瓣宽当极限。

---

## 9 编号结论

1. 规定 Brunone \(k\) 使裂缝相关井口峰衰减、展宽、后移。\(D=20\,\mathrm{m}\)、\(k=0.2\) 时峰漂移 277 ms，EST 101 ms；该锚点在四个分析窗上重复。FWHM 在强退化时假性收窄，不以它为宽度指标。
2. 共同参考 STFT 下高频能量相对低频优先下降。\(D=20\,\mathrm{m}\)、\(k=0.2\) 时 60–150 Hz 相对 \(k=0\) 为 −44.1 dB，0–20 Hz 为 −12.9 dB。三种表观速度降幅分离。这是频率选择性耗散，不是已测 \(c_p(\omega)\)。
3. Re 相关 Brunone 下，盲倒谱单缝深度系统偏差为 10–20 m；准稳态对照在 2 m 容差下即成功。历史上 135 m 容差掩盖了该偏差。LHS 实现 \(k\) 为弱档（8 s 探针中位 0.014），不得用 \(k=0.2\) 示意外推现场，也不得在无流变标定时装成压裂液类型。
4. 近距失败不是信息论墙。\(f_c=20\,\mathrm{Hz}\)、SNR = 40 dB 时单缝位置 CRB 为 0.025 m；5.8–120 m 间距上界保持厘米级。Brunone 相对稳态仅膨胀 1.12–1.32 倍。全波形 MOC-MLE 效率比 0.96–1.39（\(n\le 5\) 时 0.86–1.18）。倒谱在无限信噪比下仍有 12.0 m 误差。
5. 实用墙是模型失配。摩阻形式互换 29.6–31.9 m，大于 \(k\) 错 10% 的 0.68–1.05 m；波速 −1% 的非线性中位绝对偏差 120 m，一阶 28–36 m 仅量级。
6. 配对摩阻剪刀差成立：倒谱在稳态数据上变好，剥离与 PhaseNet 变差。缝数判定是模型选择；嵌套 BIC 在 oracle 下 exact-count = 1.0，与盲计数不是同一问题。
7. 处方：以匹配摩阻形式的 MOC 全波形 MLE 为主估计器，倒谱与学习检测器只作初值或对照，并在失配集上评价。上述结论针对给定带宽与白噪声的合成记录，不宣称现场已达厘米级。

---

## 局限

- 主结果：\(f_c=20\,\mathrm{Hz}\)、加性白噪声；半真实或现场噪声未测。
- 规定 \(k\) 矩阵为等参数四缝、首缝 4100 m；EST 锚点取 \(D=20\,\mathrm{m}\)，大间距窗会混入后缝。
- CRB/MLE 在给定模型阶数下定义；嵌套 BIC 不是盲搜索。效率场景为等间距簇、\(n\le 5\)、缝区约 3500–4800 m。
- 波速 −1% 为 4 个单缝探针（计划 8 个，完成 6 个）；中位 120 m 不是全分布统计。
- Courant = 1 改波速会重划网格；一阶走时公式已证明低估。
- 实现 \(k\) 探针为 8 s、单一物性组合。
- 不把可微代理当作已过关的定位器。

---

## 符号表

| 符号 | 含义 | 单位 |
|---|---|---|
| \(L\) | 井筒长度 | m |
| \(a\) | 标称波速 | m/s |
| \(k\) | Brunone 系数（规定常数或 `brunone_k(Re)`） | — |
| \(D\) | 等缝间距 | m |
| \(C_f\) | 裂缝柔度 | m² |
| \(k_{\mathrm{leak}}\) | 滤失系数 | m² s⁻¹ m⁻¹/² |
| \(\Delta t_{\mathrm{peak}}\) | 相对 \(k=0\) 的首峰漂移 | s |
| EST | 正部能量 E10–E90 时间跨度 | s |
| \(C_v\) | \((P_{\min}-V)/P_{\min}\) | — |
| \(a_{f0},a_{\mathrm{onset}},a_{\mathrm{peak}}\) | 三种操作表观速度 | m/s |
| \(f_c\) | 传感器截止频率 | Hz |
| \(\sigma_x\) | 位置 CRB 标准差 | m |
| \(\theta\) | 反演参数向量 | — |

---

## 主张–证据

| 主张 | 证据 | 状态 |
|---|---|---|
| Brunone 使峰钝、后移、EST 展宽 | EXP-024：\(D=20\,\mathrm{m}\)、\(k=0.2\) 时 277 ms / EST 101 ms；四窗重复 | 支持 |
| 频率选择性耗散 | EXP-024：STFT 60–150 Hz −44.1 dB vs 0–20 Hz −12.9 dB（相对同间距 \(k=0\)） | 支持；不是 \(c_p(\omega)\) |
| 表观速度分离 | EXP-024：11.0% / 1.3% / 4.7% | 支持；非相速度 |
| 时域包络随 \(k\) 融合 | EXP-024：\(C_v\) 热力图 + 合成双峰单元测试通过 | 支持趋势；不报最小缝距表 |
| 倒谱单缝偏 10–20 m | EXP-001 盲容差扫描；稳态对照 2 m | 支持 |
| LHS 实现 \(k\) 为弱档 | EXP-010：中位 0.014（8 s 探针） | 支持；非 50 s 全程 |
| 不是信息墙 | EXP-002/003：CRB 0.025 m；间距扫描 0.025–0.032 m；Brunone/稳态 1.12–1.32× | 支持 |
| 倒谱低效 | EXP-003：SNR = \(\infty\) 仍 12.0 m；MLE 效率比 0.96–1.39 | 支持 |
| \(n\le 5\) 仍可贴界 | EXP-007-001：0.86–1.18 | 支持 |
| 形式失配 ~30 m | EXP-003 非线性 29.6 / 31.9 m | 支持；一阶 0.6–10 m 不得代替 |
| \(k\) 错 10% ~1 m | EXP-002 一阶 0.68–1.05 m | 支持（一阶） |
| 波速 −1% ~120 m | EXP-007-003：4 单缝中位 120 m；门控未过 | 支持；partial（6/8 探针） |
| 剪刀差 | EXP-007-002：300 工况，方向全部成立 | 支持 |
| 严格频散 / Rayleigh 墙 / 38 m 墙 / \(k\)–压裂液 | 无本项目证据 | **不写** |

---

## 数据与代码入口

- 规定 \(k\) 时间序列：`output/analysis/brunone_spacing_effect/`
- EXP-024 审计：`analysis/brunone_spacing_effect/audit_waveform_degradation.py` → `output/analysis/brunone_spacing_effect/audit_v2/`
- CRB / MLE / 失配：`output/analysis/identifiability/`（`efficiency/main/meta.json`、`mismatch_mle/main/single_4000.json`、`highorder/main/summary.json`、`transfer/main/transfer_summary.json`、`wavespeed_mismatch/probe8/wavespeed_mismatch.json`、`main_dt1ms/crb_table.csv`、`diagnosis/diagnosis_table.csv`）
- 盲倒谱：`output/analysis/blind_protocol_reaudit/`
- 图：`audit_v2/fig1`–`fig8`；`output/analysis/identifiability/figures/`；`efficiency/main/figures/`

---

## English abstract (for later SPEJ submission)

Shut-in wellhead water-hammer records are used to locate multiple hydraulic fractures. Under Brunone unsteady friction the reflected packets at the wellhead become dull and delayed, and cepstral picks are often described as unresolvable at close spacing. This paper separates waveform degradation, estimator bias, and the information bound. A 30-run matrix with prescribed Brunone coefficient \(k\) from 0 to 0.2 and four equal fractures at spacings of 5–100 m shows frequency-selective dissipation. At \(k=0.2\) and 20 m spacing the first-peak delay relative to \(k=0\) is 277 ms and the energy-spread time (EST) is 101 ms. With a documented STFT (Hann, \(n_{\mathrm{perseg}}=128\), overlap 120, dB reference = same-spacing \(k=0\)), energy in 60–150 Hz falls by 44.1 dB while 0–20 Hz falls by 12.9 dB. Apparent speeds from the water-hammer fundamental, onset, and peak disagree (reductions of 11.0%, 1.3%, and 4.7%), so a single scalar wavespeed is not a stable descriptor. These observations support dissipative, frequency-selective attenuation of sharp fronts; they are not a measurement of \(c_p(\omega)\). On wellbore cases with Reynolds-number-dependent \(k\), a blind cepstrum protocol places the single-fracture depth bias between 10 and 20 m, whereas quasi-steady controls already succeed at a 2 m tolerance. Realized \(k\) in an 8 s Latin-hypercube probe is weak (median 0.014), far below the \(k=0.2\) illustration. Under an aligned observation model (well length 5000 m, wavespeed 1450 m/s, Brunone friction, 50 s records, 20 Hz cutoff, additive white noise) the single-fracture depth CRB at 40 dB SNR is 0.025 m, and Brunone inflates the joint bound by only 1.12–1.32 relative to steady friction. Full-waveform MOC-MLE attains the bound (RMSE/CRB = 0.96–1.39; still 0.86–1.18 for up to five equally spaced fractures). Cepstrum retains 12.0 m error at infinite SNR. The practical wall is model error: swapping steady and Brunone forms yields 29.6 m and 31.9 m; a −1% wavespeed mismatch yields a nonlinear MLE median absolute bias of 120 m on four single-fracture probes (first-order travel-time predictions of 28–36 m fail a ratio < 2 gate). A paired 300-case transfer test shows a scissors pattern: cepstrum F1 improves from 0.232 to 0.642 on steady data, whereas a peeling dictionary and a PhaseNet detector trained on Brunone data fall from 0.396 to 0.089 and from 0.746 to 0.232. Cepstral blur is therefore estimator distortion under dissipative fronts, not an information wall. Field inversion should match the friction form of the data, use cepstrum only as an initializer, and calibrate wavespeed. These synthetic bounds are not a claim of centimetre-scale field accuracy.
