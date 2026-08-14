# 级联裂缝中停泵水击倒谱响应的拓扑组织与首缝非局部性

**English title:** Topology-Organized Cepstral Response of Shut-In Water-Hammer Transients in Cascaded Hydraulic Fractures

**目标期刊：** SPE Journal · **Paper A** · 姊妹篇：[Paper B](./PaperB_Brunone_Waveform_Degradation_非定常摩阻波形退化与倒谱模糊.md) · [Paper C](./PaperC_Identifiability_and_Error_Budgets_可辨识性与误差预算.md)

**关键词：** water hammer; hydraulic fracturing; cepstrum; fracture network; method of characteristics

**一句话问题：** 在准稳态 Darcy 摩阻下，井口倒谱峰是随传播距离衰减的局部裂缝标签，还是随节点拓扑与下游网络变化的非局部观测量？

本稿由 `paper_steady_state_mechanisms.md` 的方法与首缝叙述扩写，主张以已审计观察与 2026-08-14 重算为准。数字来源：`output/analysis/decay_regression/03_extracted_peaks_csv/decay_table.csv` 与 `output/analysis/decay_regression/paperA_stats/`。本文不报定位误差、不报 F1、不以 Brunone 为主结果。

---

## 摘要

停泵水击常被用于多簇裂缝诊断，井口倒谱峰往往被当作随距离衰减的局部裂缝标签：回声时延对应深度，峰强对应簇规模。本文用一维特征线井筒模型和准稳态 Darcy 摩阻检验该假设。主矩阵为 420 个等参数、等间距工况（首缝深度 2000–4500 m，间距 10–100 m，缝数 2–8；2100 条裂缝级记录）。后缝相对倒谱幅度 \(\alpha_i=P_i/P_1\) 对物理距离呈发散曲线族，对裂缝序号则更紧凑。在可比较的 360 个 \(n\ge 3\) 工况上，拓扑幂律 Pow(idx) 的中位 \(R^2=0.987\)，空间幂律 Pow(dx) 为 0.908；\(\Delta R^2\) 中位 0.080（工况级 bootstrap 95% 区间 [0.075, 0.084]），358/360 个工况拓扑模型更优。首缝表观二维倒谱峰 \(P_{2D}\) 随深度、缝数和间距普遍非单调：70/70 条深度序列、60/60 条缝数序列、41/42 条间距序列出现先降后升或振荡，因而不是局部柔度或滤失加上单调沿程衰减的单值函数。同一矩阵上，一维全局倒谱的峰–峰值相对变幅明显小于二维时间积分剖面，说明非局部性是该观测算子的性质，而不是简单的一维旁瓣噪声。观察与节点反射、透射和分流比单纯路径长度更有描述力相一致；它本身并不唯一识别相长/相消路径，展宽指数拟合也不构成反常扩散证明。工程含义是：即使在准稳态摩阻下，把第一回声幅度当作孤立首簇标定已经设定错误。相对后缝响应与首缝绝对峰必须分报。非定常摩阻与全波形可辨识性见姊妹篇。

---

## 1 引言

段内多簇压裂是非常规储层改造的常规做法。停泵瞬间流速突变激发水击波，波在井筒中传播，并在阻抗突变处（起裂缝口、趾端、井口）反射。井口压力因此携带裂缝位置与水力性质的信息，被用作压裂后诊断的候选观测。

现场解释常常默认两个可分离性：回声时延映射为深度，峰强映射为该簇的局部规模。多簇级联下，这一默认并未在受控矩阵上被系统检验。裂缝节点之间存在反复透射与反射，井口记录是整个网络的卷积，而不是互不干扰的单散射峰列。若后缝相对幅度主要由物理传播距离组织，则沿程粘性耗散是主控因素，深度–幅度标定有明确意义。若它主要由裂缝序号组织，则节点处的反射、透射和分流比路径长度更有描述力，用距离衰减曲线去标定后簇就会设定错误。首缝作为最近的反射界面，其表观峰是否只取决于自身柔度、滤失和到井口的距离，同样未被拆开。

本文严格限定在准稳态 Darcy 摩阻，把观测物理从定位算法和非定常摩阻中分开。工作假说为：

- **H1.** 后缝相对倒谱响应由物理距离 \(\Delta x\) 组织，还是由裂缝序号组织。
- **H2.** 首缝倒谱峰是局部 \(C_f\)/\(k_{\mathrm{leak}}\) 加上单调沿程衰减的单值函数，还是受下游网络耦合。

第 2 节给出正演与倒谱操作定义；第 3 节给出 420 工况矩阵；第 4 节检验 H1；第 5 节检验 H2；第 6–7 节讨论工程含义与边界。本文讨论的是观测算子，不是定位分辨率。

---

## 2 理论与方法

### 2.1 一维特征线井筒模型

控制方程为不可压缩–弹性耦合的一维水击方程：

\[
\frac{\partial H}{\partial t}+\frac{a^{2}}{g}\frac{\partial V}{\partial x}=0,\qquad
\frac{\partial V}{\partial t}+g\frac{\partial H}{\partial x}+\frac{f}{2D}V|V|+g\sin\theta=0.
\]

离散采用特征线法（MOC），Courant 数精确为 1，无额外数值耗散。达西摩阻因子 \(f\) 由层流解析式与紊流 Zigrang–Swamee 显式近似给出。本文主结果全部使用准稳态 Darcy 摩阻，不加入 Brunone 非定常项。

井口边界：\(t<t_s\) 时 \(V=V_0\)，停泵后 \(V=0\)（柱塞泵瞬时截流）。趾端为定压水库边界。裂缝簇等效为集总柔度加滤失节点：

\[
Q_f=C_f\frac{\mathrm{d}H}{\mathrm{d}t}+k_{\mathrm{leak}}\sqrt{H-H_{\mathrm{ext}}},
\]

并满足连续性 \(A(V_{\mathrm{left}}-V_{\mathrm{right}})=Q_f\)。代入 C⁺/C⁻ 后得到关于节点水头的非线性方程，用牛顿迭代求解。观测为井口水头 \(H_{\mathrm{wh}}(t)\)。

### 2.2 一维与二维倒谱，以及两个必须分开的量

压力记录可写成源子波与稀疏反射序列的卷积。实倒谱把卷积变为加法，双程走时 \(\tau=2x/a\) 对应倒频率峰，深度映射为 \(x=q\cdot a/2\)。

**一维全局实倒谱**对停泵后全序列作 FFT–对数谱–IFFT，把时间轴压到倒频率轴。多径尾波与高次反射因此全部叠进同一条深度曲线。

**二维倒谱图**对去趋势井口水头加滑动窗（本文：Hamming 窗，窗长 30 s，hop 5 s）。主矩阵中二维峰并不采用“沿 \(t=\tau\) 对角脊积分”的理想化方案，而是对倒谱图时间轴求和后，在真值深度邻域内取局部最大：

\[
P_{2D}(x_i)=\max_{|x-x_i|\le r}\, \Bigl(-\sum_{t} C(q(x),t)\Bigr),\qquad
r=\min\bigl(15\,\mathrm{m},\,0.49S\bigr).
\]

一维峰 \(P_{1D}\) 用同一邻域规则从全局实倒谱提取。相对响应定义为 \(\alpha_i=P_i/P_1\)。后文 H1 使用 \(\alpha_{2D}\)；H2 使用首缝绝对峰 \(P_{2D}\)。二者不得都称为“能量衰减”：归一化把首缝固定为 1，会掩盖第 5 节的非局部性。

两个互斥工作假说对应两个经验模型。空间幂律

\[
\alpha=(1+\Delta x)^{-k_{dx}}
\]

把后缝相对幅度归因于物理偏移。拓扑幂律

\[
\alpha=\mathrm{idx}^{-k}
\]

把同一幅度归因于节点序号。比较在工况级进行，不用裂缝级行充当独立样本。展宽指数 \(\alpha=\exp\bigl(-(b\Delta x)^{\beta}\bigr)\) 仅作为空间轴上的经验拟合进入补充讨论，不解释为反常扩散或非德拜弛豫。

---

## 3 数据与实验设计

主矩阵与仓库衰减回归扫描一致，固定水力参数见表 1，扫描轴见表 2。每个工况等参数、等间距。井筒长度取 \(\max(5000, x_{\mathrm{last}}+500)\,\mathrm{m}\)，以保证末缝后方仍有管段；仿真时长 \(t_f=\max(100, 2L/a+1)\,\mathrm{s}\)。

**表 1.** 正演与倒谱固定参数

| 参数 | 取值 | 单位 |
| :--- | :--- | :--- |
| 井筒内径 \(D\) | 0.1397 | m |
| 流体密度 \(\rho\) | 1000 | kg/m³ |
| 运动黏度 \(\nu\) | \(1.0\times 10^{-6}\) | m²/s |
| 波速 \(a\) | 1450 | m/s |
| 粗糙度 | \(4.5\times 10^{-5}\) | m |
| 初始流速 \(V_0\) | 1.0 | m/s |
| 初始水头 \(H_0\) | 300 | m |
| 井斜 \(\theta\) | 0 | rad |
| 停泵时刻 \(t_s\) | 1.0 | s |
| 时间步 \(\Delta t\) | \(1.0\times 10^{-3}\) | s |
| 趾端 | 定压水库，水头 300 | m |
| 裂缝柔度 \(C_f\) | \(1.0\times 10^{-5}\) | m² |
| 滤失 \(k_{\mathrm{leak}}\) | \(1.0\times 10^{-4}\) | m²/s/√m |
| 地层水头 \(H_{\mathrm{ext}}\) | 100 | m |
| 2D 窗长 / hop / 窗型 | 30 / 5 / Hamming | s |

**表 2.** 主扫描矩阵（稳态摩阻）

| 轴 | 取值 | 水平数 |
| :--- | :--- | ---: |
| 首缝深度 \(X_1\) | 2000, 2500, 3000, 3500, 4000, 4500 m | 6 |
| 裂缝间距 \(S\) | 10, 20, …, 100 m | 10 |
| 缝数 \(n\) | 2, 3, …, 8 | 7 |
| 工况 | \(6\times 10\times 7\) | **420** |
| 裂缝级记录 | 每工况 \(n\) 条 | **2100** |

口径已经 EXP-20260730-019 审计。历史草稿中的 350 工况（缺 \(X_1=4500\,\mathrm{m}\)）作废。数据入口：`decay_table.csv`。拓扑拟合与首缝摘录于 2026-08-14 从该表重算（EXP-20260814-001）；统计单元是工况 \((X_1,S,n)\)，裂缝级行不独立进入 bootstrap。\(n=2\) 只有一个后缝点，两模型都能过两点，不进入 \(R^2\) 比较，故比较样本为 360 个 \(n\ge 3\) 工况。

Cf/Kleak 历史表混有 \(n=3\) 与 \(n=5\)，全表不得当作统一稳健性证据。下文只引用其中已经是 \(n=5\) 的 42 个稳态单元格作为**受限**参考，并标明缺 30 格。

---

## 4 结果：后缝相对响应的拓扑组织

### 4.1 距离轴发散，序号轴收紧

图 2 以 \(X_1=4000\,\mathrm{m}\)、\(n=8\) 为签名图（`steady_collapse_vs_divergence_x1_4000.png`）。横轴为物理偏移 \(\Delta x=x_i-x_1\) 时，不同间距的 \(\alpha_{2D}\) 曲线呈扇形发散：同一序号的裂缝因 \(S\) 不同而落在不同距离上，包络被拉开。横轴改为裂缝序号后，同一组曲线收进更窄的带。\(n=8\) 全间距上，后缝 \(\alpha_{2D}\) 按序号分组的中位四分位距为 0.077，按 20 m 距离分箱则为 0.120。这是 H1 的几何观察：在等参数级联里，节点计数比路径长度更能把曲线叠到一起。

### 4.2 工况级 Pow(idx) 对 Pow(dx)

对每个 \(n\ge 3\) 工况，用 \(\alpha_{2D}>0\) 的后缝点最小二乘拟合 \(k\) 与 \(k_{dx}\)，再在该工况全部点上计算 \(R^2\)。表 3 汇总 360 个稳态工况。拓扑模型中位 \(R^2=0.987\)（分组 bootstrap 95% 区间 [0.985, 0.989]），空间模型为 0.908（[0.901, 0.914]）。\(\Delta R^2=R^2_{\mathrm{idx}}-R^2_{\mathrm{dx}}\) 中位 0.080（[0.075, 0.084]）。358/360（99.4%）个工况 \(\Delta R^2>0\)。\(\Delta R^2\) 随缝数增加：\(n=3\) 中位 0.041，\(n=8\) 中位 0.129，与“多一个节点就多一次阻抗失配”的方向一致，但还不是单节点透射系数的推导。

**表 3.** 稳态主矩阵工况级拟合（\(\alpha_{2D}\)，\(n\ge 3\)，\(N=360\)）

| 量 | 中位 | 95% CI | P25–P75 |
| :--- | ---: | :--- | :--- |
| \(R^2\) Pow(idx) | 0.987 | [0.985, 0.989] | 0.977–0.997 |
| \(R^2\) Pow(dx) | 0.908 | [0.901, 0.914] | 0.873–0.935 |
| \(\Delta R^2\) | 0.080 | [0.075, 0.084] | 0.055–0.108 |
| \(k\)（拓扑） | 1.22 | [1.18, 1.25] | 0.99–1.41 |
| \(k_{dx}\)（空间） | 0.327 | [0.318, 0.339] | 0.263–0.382 |
| Pow(idx) 更优 | 358/360 = 99.4% | — | — |

仅有的两个反例都在最浅、最密的格子上：\((X_1,S,n)=(2000\,\mathrm{m},10\,\mathrm{m},4)\)，\(\Delta R^2=-0.013\)；\((2000\,\mathrm{m},10\,\mathrm{m},5)\)，\(\Delta R^2=-0.020\)。两例中两个模型的 \(R^2\) 都已经高于 0.978，差别是“谁更好”，不是“拓扑模型崩溃”。细间距下搜索邻域 \(r=0.49S=4.9\,\mathrm{m}\) 与峰融合可能掺入误差，不能单独解释为空间耗散主导。

图 3（`paperA_stats/steady_r2_idx_vs_dx.png`）给出工况级散点与 \(\Delta R^2\) 直方图：绝大多数点位于对角线上方，质量集中在 \(\Delta R^2>0\)。

### 4.3 Cf/Kleak 受限参考，以及不得写入主贡献的拟合

历史 Cf/Kleak 表在固定几何 \(X_1=3000\,\mathrm{m}\)、\(S=20\,\mathrm{m}\) 上扫描，但 30 个单元格跑的是 \(n=3\)、42 个是 \(n=5\)，全表不能当作统一 \(n=5\) 稳健性。只保留 \(n=5\) 的 42 格：42/42 拓扑模型更优，中位 \(\Delta R^2=0.093\)，与主矩阵同数量级。拓扑指数 \(k\) 却从 0.32 变到 2.33，说明**组织方式**（按序号优于按距离）在该子集上仍然成立，**指数大小**随节点水力参数变化，不能写成与 \(C_f\)、\(k_{\mathrm{leak}}\) 无关的普适常数。缺的 30 格未重跑正演，本结论不得外推到全网格。

同一 360 工况上，展宽指数拟合中位 \(\beta=0.70\)（P25–P75：0.64–0.79），90.6% 的工况 \(\beta<1\)，中位 \(R^2=0.997\)。这只说明空间轴上的经验曲线比单指数更弯，不证明反常扩散。该拟合进补充材料，不进编号结论。

Brunone 对照矩阵同样是 420 工况，但 \(\Delta R^2\) 中位仅 0.008，拓扑更优比例降到 58.6%。本文**不**把跨摩阻普适标度写成结果；该对照只用于划定适用范围。

---

## 5 结果：首缝表观峰的非局部性

主矩阵中所有裂缝的 \(C_f\) 与 \(k_{\mathrm{leak}}\) 相同。若首缝峰只是局部属性加沿程单调衰减，则固定 \(n,S\) 时 \(P_{2D}\) 应随 \(X_1\) 下降，固定 \(X_1,S\) 时随下游缝数下降或至少不回升，固定 \(X_1,n\) 时随间距单调。表 4 与既有三面板图（`06_first_frac_energy/`）给出反例。非单调判定为：一条完整扫描序列上同时出现正增量与负增量。

**表 4.** 稳态首缝 \(P_{2D}\) 摘录（从 `decay_table.csv` 重算）

| 扫描 | 固定量 | 序列（\(P_{2D}\)） |
| :--- | :--- | :--- |
| 深度 | \(n=2\)，\(S=50\,\mathrm{m}\) | 2000 m: **2.24**；2500 m: **1.04**；3000 m: **2.73**；3500 m: 2.71；4000 m: 3.03；4500 m: 2.69 |
| 缝数 | \(X_1=2000\,\mathrm{m}\)，\(S=50\,\mathrm{m}\) | \(n=2\): 2.24；\(n=3\): **1.48**；\(n=4\): 1.76；\(n=5\): 1.58；\(n=6\): 1.96；\(n=7\): 1.74；\(n=8\): 1.76 |
| 缝数 | \(X_1=2000\,\mathrm{m}\)，\(S=100\,\mathrm{m}\) | \(n=2\): 2.18；…；\(n=8\): **2.31**（下游加密后首缝峰高于双缝） |
| 间距 | \(X_1=2000\,\mathrm{m}\)，\(n=8\) | \(S=10\): 1.86；30 m: 1.67；**40 m: 1.65**；90 m: 2.30；**100 m: 2.31** |

全矩阵统计而不是只看上表：

- 对 \(X_1\)：70/70 条 \((n,S)\) 序列非单调；峰–峰值相对均值的中位为 0.59。
- 对 \(n\)：60/60 条 \((X_1,S)\) 序列非单调；中位相对变幅 0.25。
- 对 \(S\)：41/42 条 \((X_1,n)\) 序列非单调；中位相对变幅 0.17。

因此 H2 的单调/局部版本在本矩阵上不成立。首缝 \(P_{2D}\) 随下游缝数和间距变化，尽管首缝自身的 \(C_f\)、\(k_{\mathrm{leak}}\) 从未改过。孤立单缝标定或“下游只屏蔽、不反馈”的模型与数据不符。

同一批首缝记录上，一维峰 \(P_{1D}\) 的相对变幅小得多：对 \(X_1\)、\(n\)、\(S\) 的中位峰–峰值/均值分别为 0.11、0.059、0.027；缝数轴上只有 45/60 条序列非单调。二维时间积分剖面把网络耦合放大了，而不是把它滤掉。旧稿把二维方案写成“对角累加以隔离一次反射”；就本仓库的实际提取规则而言，更准确的说法是：二维峰是一种对下游网络敏感的观测算子。倒谱窗、邻域寻峰和多次反射都可能进入 \(P_{2D}\)，现有输出没有复谱相位或逐缝开启，因此正文只写与全局网络耦合一致，不写已经识别某一条相长/相消路径。主矩阵无 \(n=1\) 基线，不能把多缝与单缝的差分解为“纯网络增量”。

---

## 6 讨论

相对后缝响应与首缝绝对峰必须分报。用 \(P_1\) 归一化之后，第 5 节的非单调被固定成 1，第 4 节看到的只是后缝如何跟着第一个峰走。现场若把井口第一回声幅度当作首簇规模的标定曲线，等于假设 H2 的局部版本，而该版本已被等参数扫描否证。

对反演的含义是设定，而不是提出新的拾峰阈值。需要能表达节点级联的正演（通向姊妹篇的全波形匹配），而不是把各簇当作独立单散射再线性叠加。这里否定的是“可分离的局部标签”假设，不是线性波动方程本身的叠加原理。

拓扑指数 \(k\) 在主矩阵中位约 1.22，但在 \(n=5\) 的 Cf/Kleak 子集上可以从 0.3 变到 2.3。因此不能把 \(k\) 直接当成现场簇均匀性的通用判据；能守住的是组织方式：在准稳态、等参数、等间距条件下，序号比距离更合适。

局限如下。正演是一维的。裂缝等参数、等间距；非等间距与非等水力参数未覆盖。摩阻是准稳态 Darcy；Brunone 下坍缩明显变弱，不在本文主张范围内。\(P_{2D}\) 依赖窗长、hop 与邻域半径，非单调里可能含窗函数贡献。无 \(n=1\)、无节点消融、无复谱相位，机制停在网络耦合这一层。

---

## 7 结论

1. 在准稳态 Darcy 摩阻、420 个等参数级联工况中，后缝相对倒谱幅度按裂缝序号组织比按物理距离更紧凑。360 个可比较工况上，Pow(idx) 中位 \(R^2=0.987\)，Pow(dx) 为 0.908，\(\Delta R^2\) 中位 0.080；358/360 个工况拓扑模型更优。
2. 首缝表观二维倒谱峰随首缝深度、缝数和间距非单调（深度 70/70、缝数 60/60、间距 41/42 条序列），不是首缝局部属性加单调沿程衰减的单值函数。下游网络改变首缝观测量。
3. 相对后缝响应与绝对首缝峰必须分报；用第一峰归一化会掩盖非局部性。二维时间积分峰对网络的敏感度高于一维全局倒谱。
4. 节点水力参数变化时，拓扑**排序**在已有 \(n=5\) 子集上仍然成立，但拓扑指数不是普适常数。展宽指数 \(\beta<1\) 只是经验拟合，不构成反常扩散证明。
5. 上述结论限于一维、等间距、等参数裂缝与准稳态摩阻，不宣称 Brunone 下的普适规律。本文讨论观测算子，不是定位分辨率。

**可写：** 拓扑组织的观察；首缝非局部观察；对孤立首缝标定的否证。  
**不可写：** 已导出单节点透射损失；已证明相长/相消路径；反常扩散；现场导流能力定量反演；定位 F1。

---

## 符号表

| 符号 | 含义 |
| :--- | :--- |
| \(H,V,a,D,f\) | 水头、流速、波速、内径、达西摩阻因子 |
| \(C_f,k_{\mathrm{leak}},H_{\mathrm{ext}}\) | 裂缝柔度、滤失系数、地层水头 |
| \(X_1,S,n,\mathrm{idx}\) | 首缝深度、间距、缝数、裂缝序号 |
| \(\Delta x\) | 相对首缝的物理偏移 |
| \(P_{1D},P_{2D}\) | 一维 / 二维倒谱局部峰 |
| \(\alpha_i\) | 相对响应 \(P_i/P_1\) |
| \(k,k_{dx}\) | 拓扑 / 空间幂律指数 |
| \(\beta\) | 展宽指数（补充材料） |

---

## English abstract

Shut-in water-hammer records are widely used to diagnose multi-cluster hydraulic fractures, yet wellhead cepstral peaks are often treated as local, distance-attenuated tags of individual clusters. This paper tests that assumption with a one-dimensional method-of-characteristics wellbore model and quasi-steady Darcy friction. A controlled matrix of 420 equal-parameter, equally spaced cases spans first-fracture depths of 2000–4500 m, cluster spacings of 10–100 m, and 2–8 fractures (2100 fracture-level records). Relative later-fracture cepstral amplitudes form a diverging family of curves when plotted against physical offset, but collapse more tightly when plotted against fracture index. On the 360 cases with \(n\ge 3\), a topological power law in fracture index attains a median \(R^2\) of 0.987, versus 0.908 for a spatial power law in offset; the median \(\Delta R^2\) is 0.080 (case-level bootstrap 95% interval [0.075, 0.084]), and the topological model is better in 358 of 360 cases. The apparent primary-fracture two-dimensional cepstral peak is non-monotonic in first-fracture depth (70/70 series), cluster count (60/60), and spacing (41/42), and is therefore not a single-valued function of local compliance or leakoff plus monotonic path attenuation. One-dimensional global cepstral peaks vary far less along the same axes, so the nonlocality is a property of the two-dimensional observation operator rather than one-dimensional sidelobe noise alone. These observations are consistent with node-wise reflection, transmission, and flow diversion dominating over simple path length in this cascade. They do not identify a unique constructive or destructive travel path, nor does a stretched-exponential fit establish anomalous diffusion. Inversion schemes that assign the first wellhead echo amplitude to an isolated first cluster are misspecified even under quasi-steady friction. Relative later-fracture response and the absolute primary-fracture peak must be reported separately.

---

## 数据、图表与仍缺实验

**本初稿使用的已完成证据**

| 内容 | 路径 |
| :--- | :--- |
| 主表 | `output/analysis/decay_regression/03_extracted_peaks_csv/decay_table.csv` |
| 口径审计 | EXP-20260730-019 |
| 工况级重算 | `output/analysis/decay_regression/paperA_stats/paperA_recomputed_summary.json` |
| 拟合表 | `paperA_stats/steady_case_fits_alpha2d.csv` |
| 图 2 候选 | `04_collapse_and_scaling_pidx/steady_collapse_vs_divergence_x1_4000.png` |
| 图 3 | `paperA_stats/steady_r2_idx_vs_dx.png` |
| 图 5 三面板 | `06_first_frac_energy/steady_*_2d.png` |

**参数更新后的参考结论（相对旧框架稿）**

- 旧稿 Table 2 写 \(X_1\) 五档、隐含 350 工况；主表实为六档 **420** 工况。
- 旧稿首缝数字与重算一致到约 0.01：\(n=2,S=50\,\mathrm{m}\) 时 \(P_{2D}=2.24/1.04/2.73\)（旧写 2.23/1.04/2.73）；\(S=100\,\mathrm{m}\) 时 \(n=8\) 对 \(n=2\) 为 2.31>2.18（旧写 2.30>2.18）；\(n=8\) 时间距低谷在 40 m、1.65，100 m 回升到 2.31。
- 旧框架把 Pow(idx) 优于 Pow(dx) 标为待做 EXP-022。现已在 360 工况上算完：99.4% 更优，中位 \(\Delta R^2=0.080\)。
- Cf/Kleak **不能**写成统一 \(n=5\) 全网格稳健；仅 42 格可用，\(k\) 随参数大幅变化。
- 旧稿 1D/2D“对角累加隔离一次反射”与代码不符；实际是时间轴求和。二维峰对网络更敏感，不是更“干净”的局部标签。

**投稿前仍缺（未在本稿当作已完成贡献）**

- 统一 \(n=5\) 补齐缺的 30 个 Cf/Kleak 正演格（EXP-022 余下部分）。
- \(n=1\) 基线与固定几何逐缝开启（EXP-023）。无此则机制语言保持“网络耦合 / 非局部”。

**本稿明确不做：** 非等间距全矩阵；Brunone 普适标度；把倒谱当定位器报 F1。
