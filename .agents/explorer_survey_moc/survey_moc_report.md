# MOC 仿真引擎与摩阻模型深度调研报告
# (MOC Simulation Engine & Friction Formulations Survey Report)

**报告作者**：Teamwork Explorer (MOC Simulation Engine & Friction Formulations)  
**工作目录**：`e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc\`  
**工程根目录**：`e:\water_hammer_research\wellbore_moc_method`  
**完成时间**：2026-09-09  
**参考规范**：`ORIGINAL_REQUEST.md`、`docs/MOC仿真原理与数据生成规范_v2.1.md`、`docs/Brunone非定常摩阻仿真建模.md`

---

## 1. 调研执行摘要 (Executive Summary)

针对水力压裂井筒水锤瞬变流与声学倒谱分析的敏感性消融实验需求，本调研深入剖析了 `moc_simulate/` 目录下的核心 MOC 求解器 `wellbore_moc.py`、参数配置模块 `config.py` / `lhs_config.py`、批处理生成器 `run_lhs_batch_simulate.py`、倒谱库 `cepstrum_mocdata.py` 以及完整的测试套件（`tests/`）。

**核心调研结论**：
1. **MOC 求解器高度成熟且严谨**：代码实现了严格满足 $CFL = \frac{a \Delta t}{\Delta x} = 1.0$ 的无耗散特征线网格，通过微调实际波速 $a_{adj} = \Delta x / \Delta t$ 彻底消除网格插值数值耗散与数值色散；
2. **多裂缝集总动力学建模完备**：裂缝节点采用“水头柔度 $C_H$ 储能 + 压差滤失 $k_{leak}$ + 射孔阻抗 $R_p$”耦合方程。当 $R_p=0$ 时采用闭式解析二次方程精确求根，彻底避免了 $H \to H_{ext}$ 处的导数奇异；当 $R_p>0$ 时采用牛顿迭代求解，具有硬残差阈值门控（质量残差 $<10^{-9}\,\mathrm{m^3/s}$，射孔残差 $<10^{-8}\,\mathrm{m}$）；
3. **双摩阻模型实现精准**：系统具备纯稳态 Darcy-Weisbach 摩阻（常数 $f$ 或雷诺数实时更新 quasi-steady）以及 Brunone 非定常瞬态剪切摩阻。Brunone 模型基于 Vardy 剪切衰减理论，通过时间差分与沿特征线空间差分计算瞬时加速度，并创新性地引入了 **$\tanh(V/V_{smooth})$ 符号平滑** 与 **裂缝邻域 $J_u$ 置零隔离** 机制，确保了强冲击下的绝对数值稳定；
4. **质量守恒与稳态基线达到机器精度**：对于分段压裂典型的封闭趾端（`dead_end`），求解器内置了全井离散特征线稳态流场递推算法，保证停泵前注入量 $Q_{in} = A V_0$ 严格等于各簇泄流之和 $\sum Q_{f,i}^{ss} = Q_{in}$，最后一缝至趾端流速严格为零，停泵前（$t < t_s$）井口水头波动 $< 10^{-8}\,\mathrm{m}$，完全消除了 $t=0$ 伪水击干扰；
5. **测试套件 100% 通过**：运行 `tests/test_steady_state_and_toe.py`（3 项）、`tests/test_fracture_storage_physics.py`（4 项）、`tests/test_final_acceptance.py`（14 项）、`tests/test_visualization.py`（5 项）共 26 个单元测试，全部通过（PASS），确认满足新版生产规范 `moc_lhs_v2.1`。

---

## 2. MOC 求解器核心控制方程与计算网格架构

### 2.1 物理控制方程 (Governing Equations)

在考虑可压缩水力瞬变流动的一维井筒中，连续性方程与动量方程表述为：

$$
\frac{\partial H}{\partial t} + \frac{a^2}{g} \frac{\partial V}{\partial x} = 0
$$

$$
\frac{\partial V}{\partial t} + g \frac{\partial H}{\partial x} + \frac{f}{2D} V|V| + S_u + g \sin\theta = 0
$$

其中：
- $H(x, t)$ 为井筒测压水头 $[\mathrm{m}]$；
- $V(x, t)$ 为流体截面平均轴向流速 $[\mathrm{m/s}]$；
- $a$ 为水锤波速 $[\mathrm{m/s}]$；
- $g = 9.81\,\mathrm{m/s^2}$ 为重力加速度；
- $D$ 为井筒内径 $[\mathrm{m}]$；
- $f$ 为 Darcy-Weisbach 摩阻系数；
- $S_u$ 为非定常摩阻加速度源项 $[\mathrm{m/s^2}]$（Brunone 模型启用）；
- $\theta$ 为井筒倾角正弦项（水平井 $\theta=0$）。

### 2.2 特征线相容关系与内节点格式

引入声阻抗比例常数 $\gamma_a = \frac{g}{a}$，控制方程沿特征线方向对角化为 $C^+$ 和 $C^-$ 两条特征线相容关系：

- **沿正特征线 $C^+: \frac{dx}{dt} = +a$（自上游节点 $i-1$ 传向节点 $i$）**：

$$
C_P = V_{i-1}^{n-1} + \gamma_a H_{i-1}^{n-1} - J_{i-1}^+ + \gamma_a \Delta t V_{i-1}^{n-1} \sin\theta
$$

- **沿负特征线 $C^-: \frac{dx}{dt} = -a$（自下游节点 $i+1$ 传向节点 $i$）**：

$$
C_M = -V_{i+1}^{n-1} + \gamma_a H_{i+1}^{n-1} + J_{i+1}^- + \gamma_a \Delta t V_{i+1}^{n-1} \sin\theta
$$

其中 $J_{i-1}^+$ 和 $J_{i+1}^-$ 为摩阻冲量项（量纲 $\mathrm{m/s}$，包含稳态 $J_s$ 与非定常 $J_u$）。

对于常规无裂缝的内部节点（$i = 1, \dots, N-1$），两特征方程联立即可得到显式解析解：

$$
H_i^n = \frac{C_P + C_M}{2 \gamma_a}
$$

$$
V_i^n = C_P - \gamma_a H_i^n = -C_M + \gamma_a H_i^n
$$

### 2.3 双流速数组架构 (`V_prev_left` 与 `V_prev_right`)

常规管流 MOC 在裂缝或支管节点处常简单采用单流速 $V_i$，这会导致裂缝处因侧向分流 $Q_f$ 引起的流速跃变被虚假平均。

`moc_simulate/wellbore_moc.py` 严格维护了双流速数组：
- `V_prev_left[i]`：存储节点 $i$ 截面上游侧流速 $V_{left}$，提供给上游邻居节点 $i-1$ 计算 $C^-$；
- `V_prev_right[i]`：存储节点 $i$ 截面下游侧流速 $V_{right}$，提供给下游邻居节点 $i+1$ 计算 $C^+$；
- 对于无分流的常规节点，$V_{left} = V_{right} = V$；
- 对于裂缝节点，$A \cdot (V_{left} - V_{right}) = Q_f$。

该设计完全保留了跨缝流速跃变的物理真实性，杜绝了数值耗散。

---

## 3. 裂缝物理参数建模与配置细节

裂缝动力学响应是水力瞬变波形反射、衰减及倒谱定位的物理源泉。在 `wellbore_moc.py` 中，裂缝节点作为集总水动力边界嵌入求解器：

```
                    井筒流动 (A, a)
   V_left (节点 i-)  ───────┬───────> V_right (节点 i+)
                            │
                       Q_f  │  (侧向分流)
                            ▼
                    ┌───────────────┐
                    │ 射孔孔眼压降   │  H_w - H_f = R_p * Q_f * |Q_f|
                    └───────┬───────┘
                            │ H_f (裂缝内部水头)
                            ▼
                    ┌───────────────┐
                    │ 裂缝储液柔度   │  Q_storage = C_H * (dH_f/dt)
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ 地层压差滤失   │  Q_leak = k_leak * sqrt(max(H_f - H_ext, 0))
                    └───────────────┘
                            │
                            ▼ H_ext (孔隙压力)
```

### 3.1 裂缝位置 $x_f$ 与网格对齐机制

- **参数定义**：裂缝距离井口的位置坐标 $x_f \in (0, L)$ $[\mathrm{m}]$。
- **网格对齐逻辑**：
  在 MOC 网格空间步长 $\Delta x$ 下，裂缝实际求解点必须对齐至网格离散节点：
  $$i_f = \operatorname{round}\left(\frac{x_{f,raw}}{\Delta x}\right), \quad x_{f,aligned} = i_f \cdot \Delta x$$
- **边界约束与冲突检测**：
  1. 严格要求 $0 < x_{f,raw} < L$，严禁置于井口（$i_f=0$）或趾端（$i_f=N$），否则抛出 `ValueError`；
  2. 裂缝重叠检测：若多条裂缝间距过小导致 $\operatorname{round}(x_f / \Delta x)$ 相同，抛出 `ValueError("裂缝对齐后网格索引重合")`；
  3. 数据集落盘标准：新版 schema `moc_lhs_v2.1` 同时记录 `x_f_requested`（请求真实值）、`x_f_aligned`（网格对齐值）与 `grid_index`（网格索引），保证反演评估基准的严密性。

### 3.2 裂缝水头柔度 $C_H$（储能机制与特征弛豫时间）

- **物理意义与换算**：
  水力压裂缝壁具有弹性顺应性，压力升高时裂缝体积膨胀储液，压力释放时液体回吐。
  - 压力柔度 $C_p = \frac{dV_f}{dp} \quad [\mathrm{m^3/Pa}]$，工程常用单位为 $\mathrm{bbl/psi}$（$1\,\mathrm{bbl/psi} \approx 2.3059 \times 10^{-5}\,\mathrm{m^3/Pa}$）；
  - 水头柔度 $C_H = \frac{dV_f}{dH} = \rho g C_p \quad [\mathrm{m^2}]$。
  - 在 `wellbore_moc.py` 中参数名为 `fracture_compliance_m2`（兼容别名 `Cf` 或 `fracture_Cf`）。
- **数学方程与离散格式**：
  $$Q_{storage} = C_H \frac{\partial H_f}{\partial t} \approx C_H \frac{H_f^n - H_f^{n-1}}{\Delta t}$$
  **关键物理分离**：储液时间导数中的历史水头必须使用裂缝内部水头上一时刻状态 $H_f^{n-1}$（`Hf_prev`），而严禁使用井筒水头 $H_w^{n-1}$！
- **特征弛豫时间常数 $\tau$**：
  理论推导表明，单缝瞬态水击响应具有一阶阻抗弛豫特性，其特征衰减时间为：
  $$\tau = \frac{a \cdot C_H}{2 g A}$$
  - 当冲击阶跃波到达裂缝时，裂缝反射波解析形式为 $h_r(t) = -\Delta H \exp(-t/\tau)$；
  - 透射波解析形式为 $h_t(t) = \Delta H (1 - \exp(-t/\tau))$；
  - $\tau$ 决定了水锤波高频分量的滤除截止频率与波前倒圆角程度。
- **取值区间与敏感性覆盖**：
  基准参考值 $C_H = 1.0 \times 10^{-5}\,\mathrm{m^2}$（占 $5000\,\mathrm{m}$ 井筒总流体体积储液约 $2.8\%$）。LHS 采样覆盖对数区间 $[10^{-6}, 10^{-4}]\,\mathrm{m^2}$（储液占比 $0.28\% \sim 28\%$）。

### 3.3 滤失系数 $k_{leak}$ 与等效反求

- **物理模型**：
  流体穿透缝壁滤失进入多孔介质地层，按压力平方根非线性漏失：
  $$Q_{leak} = k_{leak} \sqrt{\max(H_f - H_{ext}, 0)}$$
  其中 $H_{ext}$ 为地层孔隙压力水头 $[\mathrm{m}]$（基准默认 $100.0\,\mathrm{m}$）。
- **稳态等效滤失反求机制 (`equivalent_kleak`)**：
  在封闭趾端模式下，由于总注入流量 $Q_{in}$ 必须在稳态时完全由各压裂簇漏入地层，若直接人工指定 $k_{leak}$，极易造成初始流场与注入流量冲突。
  因此，求解器先根据分流权重 $w_i$ 确定稳态泄流 $Q_{f,i}^{ss} = w_i Q_{in}$，并根据全井离散摩阻递推得到稳态裂缝水头 $H_{f,i}^{ss}$，进而反向精确解析求解等效滤失系数：
  $$k_{leak,i}^{equiv} = \frac{Q_{f,i}^{ss}}{\sqrt{H_{f,i}^{ss} - H_{ext}}}$$
- **稳态水头裕度硬校验 (`head_margin`)**：
  求解器强制要求 $H_{f,i}^{ss} - H_{ext} > 1.0 \times 10^{-3}\,\mathrm{m}$。若由于井底水头过低、射孔阻力过大或地层水头过高导致压差不足，直接抛出 `ValueError` 阻断，杜绝产生虚假的负压截断数据。

### 3.4 射孔阻抗 $R_p$（孔眼压降机制与求解分支）

- **物理方程**：
  流体流经射孔孔眼时发生孔流节流，井筒网格水头 $H_w$ 与裂缝内部水头 $H_f$ 产生二次方压降损失：
  $$H_w - H_f = R_p \cdot Q_f |Q_f|$$
  其中 $R_p$ 为孔眼流动阻力系数 $[\mathrm{s^2/m^5}]$。
- **解法分支 1：$R_p = 0$（无孔眼压降基准，闭式二次方程解）**：
  当 $R_p = 0$ 时，$H_w \equiv H_f = H_P$。井筒特征线方程与裂缝连续性方程结合得：
  $$\alpha H_P + \beta \sqrt{\max(H_P - H_{ext}, 0)} + \gamma = 0$$
  其中：
  $$\alpha = 2 A \gamma_a + \frac{C_H}{\Delta t}, \quad \beta = k_{leak}, \quad \gamma = -A(C_P + C_M) - \frac{C_H}{\Delta t} H_f^{n-1}$$
  通过变量代换 $y = \sqrt{\max(H_P - H_{ext}, 0)} \ge 0$，方程化为关于 $y$ 的严格一元二次方程：
  $$\alpha y^2 + \beta y + (\alpha H_{ext} + \gamma) = 0$$
  直接使用求根公式 $y = \frac{-\beta + \sqrt{\beta^2 - 4\alpha C}}{2\alpha}$ 解析求解。**该解析解彻底消除了常规牛顿法在 $H_P \to H_{ext}$ 处 $\frac{1}{2\sqrt{H_P - H_{ext}}} \to \infty$ 的导数奇异与数值发散隐患！**
- **解法分支 2：$R_p > 0$（包含射孔阻抗，高精度 Newton-Raphson 迭代）**：
  未知量设定为裂缝内部水头 $H_f$。定义目标残差函数：
  $$\Phi(H_f) = H_w(H_f) - H_f - R_p Q_f(H_f) |Q_f(H_f)| = 0$$
  迭代求解 $H_f$，步长容差 `newton_tol = 1.0e-10`，最大迭代步数 20。求解完成后执行严格残差双重检验：
  - 裂缝质量守恒残差 $|r_{mass}| \le 1.0 \times 10^{-9}\,\mathrm{m^3/s}$；
  - 射孔压降残差 $|r_{Rp}| \le 1.0 \times 10^{-8}\,\mathrm{m}$。

### 3.5 分流权重 $w_i$（进液分布与 Dirichlet 采样）

- **物理约束**：
  各簇稳态进液权重 $w_i$ 代表该压裂簇分担总注入流量的比例：
  $$Q_{f,i}^{ss} = w_i \cdot Q_{in}, \quad w_i \ge 0, \quad \sum_{i=1}^{N_{cl}} w_i = 1.0$$
- **Dirichlet-LHS 映射采样**：
  在多参数采样中，通过对称狄利克雷分布生成单纯形均匀变量：
  $$u_i \in (0, 1) \implies g_i = \mathrm{gammaincinv}(\alpha_{dirichlet}, u_i), \quad w_i = \frac{g_i}{\sum g_j}$$
  - $\alpha_{dirichlet} = 0.3$：强不均衡进液（模拟优势进液簇吸入 $80\%+$ 流量）；
  - $\alpha_{dirichlet} = 1.0$：单纯形均匀分布；
  - $\alpha_{dirichlet} = 3.0$：弱不均衡进液；
  - $\alpha_{dirichlet} = 10.0$：各簇近乎等额进液。
- **校验拦截**：
  若输入存在负权重（如 `[1.2, -0.2]`）或总和偏离 1.0 超出 $10^{-10}$，求解器禁止静默归一化，强制抛出异常报错。

### 3.6 裂缝间距 $\Delta x$ 与多尺度排布

- **预设间距**：`SPACING_PRESETS_M = (5, 10, 20, 50, 100)` 米。
- **排布函数**：`build_cases(spacing_m)`，首缝默认 $x_0 = 4100.0\,\mathrm{m}$，生成 1 到 8 簇（`single`, `dual`, `triple`, `quad`, `quint`, `hex`, `hept`, `oct`）。
- **物理极限与谱支撑**：
  根据井筒参数（$a \approx 1450\,\mathrm{m/s}$），双缝间距 $D$ 产生的往返时间延迟为 $\Delta t_{delay} = \frac{2 D}{a}$。
  - $D = 100\,\mathrm{m} \implies \Delta t_{delay} \approx 0.138\,\mathrm{s}$（频域调制周期约 $7.25\,\mathrm{Hz}$）；
  - $D = 10\,\mathrm{m} \implies \Delta t_{delay} \approx 0.0138\,\mathrm{s}$（频域调制周期约 $72.5\,\mathrm{Hz}$）；
  - 当 $D \le 10\,\mathrm{m}$ 时，裂缝干涉波峰在倒谱空间高度重叠，是对倒谱分辨率与高频瞬变流摩阻耗散的最严峻考验。

---

## 4. 摩阻模型实现：Steady Darcy-Weisbach vs. Brunone

井筒瞬变流的阻尼耗散直接影响水锤波幅度的衰减速度、波形平滑度以及倒谱峰值的信噪比。

### 4.1 稳态 Darcy-Weisbach 摩阻

稳态摩阻项沿特征线的冲量表示为：

$$
J_s = \frac{f \cdot \Delta t}{2 D} V |V|
$$

摩阻系数 $f$ 依据雷诺数 $Re = \frac{|V| D}{\nu}$ 计算：
- $Re < 10^{-3}$：$f = 0$；
- $10^{-3} \le Re < 2000$（层流）：$f = \frac{64}{Re}$；
- $Re \ge 2000$（紊流）：采用 Zigrand-Swami 显式逼近式（与 TSNet 标准一致）：
  $$f = \left[ -1.8 \log_{10} \left( \frac{6.9}{Re} + \frac{\varepsilon}{D} \right) \right]^{-2}$$

求解器支持两种稳态模式：
1. `model="steady"`：仅用初始流速 $V_0$ 计算一次常数 $f_{steady}$，计算极快，广泛用于标准基准对比；
2. `model="quasi-steady"`：每个时间步、每个节点根据局部瞬时雷诺数 $Re(x, t)$ 动态重算 $f$。

### 4.2 Brunone 非定常摩阻理论与数值实现

经典达西摩阻假定瞬时管壁剪切应力与恒定流相同（$\tau_w \propto V^2$）。但在水力截流关泵瞬间，流速出现极陡峭的时间梯度 $\partial V/\partial t$ 与空间激波面 $\partial V/\partial x$，管壁边界层剧烈变形，产生远高于稳态的瞬态高频剪切耗散。

#### 4.2.1 控制方程与冲量公式
Brunone (1991) 提出了基于局部与对流加速度加权的非定常摩阻模型：

$$
\tau_w = \tau_{ws} + \tau_{wu} = \frac{1}{8} \rho f V |V| + \frac{k_{brunone} \rho D}{4} \left( \frac{\partial V}{\partial t} + a \cdot \operatorname{sign}(V) \left| \frac{\partial V}{\partial x} \right| \right)
$$

对应于 MOC 特征线中的摩阻冲量为：

$$
J_u = \frac{k}{2} \cdot \Delta t \left( \frac{\partial V}{\partial t} + a \cdot \operatorname{sign}(V) \left| \frac{\partial V}{\partial x} \right| \right)
$$

总摩阻冲量为 $J = J_s + J_u$。

#### 4.2.2 Brunone 衰减系数 $k$ 与流体流变对应关系
求解器采用 **Vardy & Brown (1996, 2003) 剪切波衰减理论** 计算无量纲系数 $k$：

$$
k = \frac{\sqrt{C}}{2} \times \text{brunone\_k\_scale}
$$

其中剪切衰减常数 $C$ 按雷诺数分段：
- 层流（$Re < 2000$）：$C = 4.76 \times 10^{-3} \implies k \approx 0.0345$；
- 紊流（$Re \ge 2000$）：
  $$C = \frac{7.41}{Re^{\log_{10}(14.3 / Re^{0.05})}}$$

在实际工程中，不同压裂液黏度直接映射到截然不同的 Brunone 阻尼水平：
- **滑溜水 (Slickwater)**：黏度极低（$\nu \approx 10^{-6}\,\mathrm{m^2/s}$），$Re > 10^5$，$k$ 极小（$\sim 0.005 - 0.01$），高频衰减慢，裂缝反射尖锐；
- **线性胶 (Linear Gel)**：$\nu \sim 10^{-5}\,\mathrm{m^2/s}$，$Re \sim 10^4$，$k \sim 0.02 - 0.05$，水锤波显著展宽；
- **交联冻胶 (Crosslinked Gel)**：表观黏度高，$Re < 2000$，$k > 0.1$，非定常高频阻尼极强，倒谱分辨率面临挑战。

#### 4.2.3 关键数值工程稳定化技术

代码在 `wellbore_moc.py` 中实现了四项关键数值稳定化措施：

1. **速度符号连续平滑（消震荡）**：
   在流速过零点附近，$\operatorname{sign}(V)$ 的阶跃跳变极易引发数值锯齿。求解器采用双曲正切函数进行平滑替换：
   $$\operatorname{sign}(V) \to \tanh\left(\frac{V}{V_{smooth}}\right), \quad V_{smooth} = 0.05\,\mathrm{m/s}$$
2. **裂缝分流邻域置零（防速度梯度爆炸）**：
   裂缝节点由于侧向出流，左右速度天然不连续（$V_{left} \neq V_{right}$）。若直接用差分算 $\partial V/\partial x$，会出现极大的虚假空间导数导致非定常项爆炸。
   实现中对所有裂缝节点前序位置强制置零：
   ```python
   if has_fractures:
       for i_f in frac_indices:
           Ju1[i_f - 1] = 0.0
           Ju2[i_f - 1] = 0.0
   ```
3. **两步历史速度存储（时间二阶前向）**：
   计算 $\partial V/\partial t \approx \frac{V_j^n - V_j^{n-1}}{\Delta t}$ 需要上上时步速度。求解器维护 `V_prev2_left` 与 `V_prev2_right`，在时间步 $n \ge 2$ 时才启动非定常项，初始步平滑过渡。
4. **特征线单边空间差分**：
   - $C^+$ 来源点（上游点 $j=0 \dots N-2$）：采用前向差分 $\frac{\partial V}{\partial x} \approx \frac{V_{j+1} - V_j}{\Delta x}$；
   - $C^-$ 来源点（下游点 $j=2 \dots N$）：采用后向差分 $\frac{\partial V}{\partial x} \approx \frac{V_j - V_{j-1}}{\Delta x}$；
   - 井口与趾端边界节点分别采用精确单边差分匹配。

---

## 5. 数值稳定性约束与离散参数解析

MOC 方法能够成为水锤计算工业标准的根本原因在于其无耗散特性。

### 5.1 Courant-Friedrichs-Lewy (CFL) 严格等式条件

在管流双曲型偏微分方程中，Courant 数定义为：

$$
CFL = \frac{a \Delta t}{\Delta x}
$$

- 若 $CFL < 1$：特征线落在网格之间，必须进行空间插值，这将引入严重的**数值耗散（虚假人工衰减）与数值弥散（伪波头振荡）**；
- 若 $CFL > 1$：数值格式失稳发散；
- **当前 MOC 求解器做法**：
  在 `MocConfig.__post_init__` 中执行：
  $$N = \operatorname{round}\left( \frac{L}{a_{req} \cdot \Delta t_{req}} \right), \quad \Delta x = \frac{L}{N}, \quad a_{adj} = \frac{\Delta x}{\Delta t_{req}}$$
  从而强制保证：
  $$CFL = \frac{a_{adj} \Delta t}{\Delta x} \equiv 1.0$$
  这种微调实际波速（调整幅度通常 $< 0.1\%$）而保持时间步长精确的做法，**保证了特征线严格精准连接相邻网格节点，数值耗散精确为零**，所有观察到的衰减完全来自于物理 Darcy 摩阻、Brunone 剪切耗散及裂缝滤失。

### 5.2 离散参数与井筒流体几何规格

| 物理量 | 符号 | 默认基准值 | 敏感性测试常见范围 | 影响与约束 |
| :--- | :---: | :---: | :---: | :--- |
| 井筒长度 | $L$ | $5000.0\,\mathrm{m}$ | $1000 \sim 6000\,\mathrm{m}$ | 决定水锤全周期 $T_0 = 4L/a \approx 13.8\,\mathrm{s}$ |
| 井筒内径 | $D$ | $0.1397\,\mathrm{m}$ (5.5"套管) | 固定 | 截面积 $A \approx 0.015328\,\mathrm{m^2}$ |
| 标称波速 | $a_{req}$ | $1450.0\,\mathrm{m/s}$ | $1350 \sim 1550\,\mathrm{m/s}$ | 决定往返走时与 Joukowsky 压力阶跃幅值 |
| 时间步长 | $\Delta t$ | $1.0 \times 10^{-3}\,\mathrm{s}$ (1 ms) | $0.5 \sim 2.0\,\mathrm{ms}$ | 采样率 $f_s = 1000\,\mathrm{Hz}$，奈奎斯特频率 $500\,\mathrm{Hz}$ |
| 空间步长 | $\Delta x$ | $\approx 1.45\,\mathrm{m}$ | $0.7 \sim 3.0\,\mathrm{m}$ | 空间网格节点数 $N \approx 3448$ |
| 流体密度 | $\rho$ | $1000.0\,\mathrm{kg/m^3}$ | 固定（滑溜水） | 决定流体动量与压力-水头转换常数 |
| 运动黏度 | $\nu$ | $1.0 \times 10^{-6}\,\mathrm{m^2/s}$ | $10^{-6} \sim 10^{-4}\,\mathrm{m^2/s}$ | 控制雷诺数与 Brunone 衰减强度 |
| 绝对粗糙度 | $\varepsilon$ | $4.5 \times 10^{-5}\,\mathrm{m}$ | 商用钢管壁 | 相对粗糙度 $K_D = \varepsilon/D \approx 3.22 \times 10^{-4}$ |
| 初始流速 | $V_0$ | $1.0\,\mathrm{m/s}$ | $0.5 \sim 3.0\,\mathrm{m/s}$ | 注入排量 $Q_0 \approx 0.0153\,\mathrm{m^3/s}$ ($0.92\,\mathrm{m^3/min}$) |
| 初始井口水头 | $H_0$ | $300.0\,\mathrm{m}$ | $200 \sim 600\,\mathrm{m}$ | 对应表压 $\sim 2.94\,\mathrm{MPa}$ |
| 停泵时刻 | $t_s$ | $1.0\,\mathrm{s}$ | 固定 | 前 $1.0\,\mathrm{s}$ 校验稳态平稳度 |
| 仿真时长 | $t_f$ | $50.0\,\mathrm{s}$ (LHS) / $100.0\,\mathrm{s}$ | $30 \sim 100\,\mathrm{s}$ | 覆盖 $3 \sim 7$ 次完整井筒混响往返 |

---

## 6. 仿真调用链路、输入输出与收敛守恒保障机制

### 6.1 模块调用链路架构

```
[用户/批处理入口]
  │
  ├── run_lhs_batch_simulate.py (多进程并行调度: 14 worker processes)
  │     ├── generate_lhs_params() (LHS 空间采样, Dirichlet 权重分配)
  │     └── _worker_simulate() (单样本执行器)
  │           │
  │           ▼
  │     simulate_wellbore(cfg, ...)  <-- [核心求解器: wellbore_moc.py]
  │           │
  │           ├── compute_steady_state_profile() [严格稳态流场初始化]
  │           │
  │           ├── [时间推进主循环: n = 1 .. n_steps]
  │           │     ├── 内部节点特征线 C+/C- (向量化)
  │           │     ├── 稳态 J_s + Brunone J_u 叠加计算
  │           │     ├── solve_fracture_node() [裂缝节点耦合方程求解]
  │           │     ├── 井口边界条件更新 (velocity_step / ramp)
  │           │     └── 趾端边界条件更新 (dead_end / reservoir)
  │           │
  │           └── 返回结果字典 (1D 时程 + 可选全场)
  │
  └── 保存为标准格式: output/lhs_dataset_v2/data/case_XXXXX.npz
```

### 6.2 严格稳态流场初始化算法（消除初始伪激扰）

在以往的水锤数值模拟中，若直接以均匀水头 $H_0$ 和均匀流速 $V_0$ 启动仿真，由于井筒沿程摩阻损失以及裂缝侧向漏失的存在，初始流场并不满足控制方程，会在 $t=0$ 瞬间激发出强烈的非物理压力脉冲，污染随后的水锤信号。

`wellbore_moc.py` 中的 `compute_steady_state_profile()` 函数实现了离散特征线稳态反演：
1. **流速分段**：自井口向井底推进，每经过一条裂缝，轴向流速扣除该簇稳态泄流 $Q_{leak,i}^{ss} = w_i \cdot Q_{in}$；在封闭趾端模式下，最后一缝之后流速严格为零（死水区）；
2. **水头沿程积分**：从井口水头 $H[0] = H_0$ 开始，完全按 MOC 离散特征线形式逐段递推水头损失：
   $$H[j] = H[j-1] - \frac{1}{\gamma_a} J_s(V_{right}[j-1]) + \Delta t V_{right}[j-1] \sin\theta$$
3. **等效滤失反求**：在得到的各缝稳态水头 $H_{w,i}^{ss}$ 基础上反算 $k_{leak,i}^{equiv}$；
4. **结果**：在 $t < t_s$ 期间，$\partial H/\partial t \equiv 0$，数值波动 $< 10^{-8}\,\mathrm{m}$（达机器精度），彻底消除伪激扰。

### 6.3 局部与全局质量守恒保障

1. **裂缝局部望远镜守恒**：
   储液项采用局部时间后向差分 $\frac{H_f^n - H_f^{n-1}}{\Delta t}$。由 `tests/test_fracture_storage_physics.py` 的严格测试证明，该格式具有望远镜伸缩求和特性：
   $$\sum_{n=1}^{N} (Q_f^n - Q_{leak}^n) \Delta t = C_H (H_f^N - H_f^0)$$
   相对守恒误差 $< 1.0 \times 10^{-12}$（机器精度）。对比之下，若采用空间平均水头插值，质量误差将显著放大数个数量级。
2. **井筒节点流速连续性**：
   裂缝节点两侧流速由特征线直接决定：
   $$V_{left} = C_P - \gamma_a H_w, \quad V_{right} = -C_M + \gamma_a H_w$$
   相减即得：
   $$A(V_{left} - V_{right}) = A(C_P + C_M) - 2 A \gamma_a H_w \equiv Q_f$$
   流量完全守恒，无任何质量渗漏。

### 6.4 数据存储与 `moc_lhs_v2.1` Schema 规范

批处理脚本生成的单工况 `.npz` 文件压缩体积仅约十几 KB，内嵌 23+ 项物理元数据与时序特征：

| 键名 (Key) | 数据类型 / 形状 | 物理含义与用途 |
| :--- | :---: | :--- |
| `schema_version` | scalar `str` | 数据模式版本，固定为 `"moc_lhs_v2.1"` |
| `seed` | scalar `int` | 随机数种子，用于 100% 物理复现 |
| `t` | `(n_steps+1,)` float64 | 时间序列 $[0.0, 0.001, \dots, t_f]\,\mathrm{s}$ |
| `H_wh` | `(n_steps+1,)` float64 | **核心观测特征**：井口水头时程 $[\mathrm{m}]$ |
| `Q_wh` | `(n_steps+1,)` float64 | 辅助观测特征：井口体积流量时程 $[\mathrm{m^3/s}]$ |
| `x_f` / `x_f_aligned`| `(n_frac,)` float64 | **反演真实标签**：网格对齐后的各缝深坐标 $[\mathrm{m}]$ |
| `x_f_raw` | `(n_frac,)` float64 | 原始连续采样裂缝位置 $[\mathrm{m}]$ |
| `grid_index` | `(n_frac,)` int32 | 各裂缝对应的网格点索引 $i_f$ |
| `compliance_head_m2`| `(n_frac,)` float64 | **反演真实标签**：各缝水头柔度 $C_H$ $[\mathrm{m^2}]$ |
| `inflow_weight` | `(n_frac,)` float64 | **反演真实标签**：各簇稳态进液权重 $w_i$ ($\sum w_i = 1$) |
| `kleak_equiv` | `(n_frac,)` float64 | 各簇物理闭合反算的等效滤失系数 $[\mathrm{m^{5/2}/s}]$ |
| `Rp` | `(n_frac,)` float64 | 各缝射孔流动阻抗系数 $[\mathrm{s^2/m^5}]$ |
| `friction` | scalar `str` | 摩阻模型类型 (`"steady"` 或 `"brunone"`) |
| `brunone_k_scale` | scalar `float` | Brunone 衰减系数标定缩放因子 |
| `wavespeed` | scalar `float` | MOC 求解实际使用波速 $a_{adj}$ $[\mathrm{m/s}]$ |
| `dx`, `dt` | scalar `float` | 离散网格空间与时间步长 |
| `Hw_ss`, `Hf_ss` | `(n_frac,)` float64 | 稳态井筒侧水头与裂缝内部水头 $[\mathrm{m}]$ |
| `Qf_ss`, `Qin_ss` | float64 | 稳态侧向流量与总注入流量 $[\mathrm{m^3/s}]$ |

---

## 7. 针对后续消融实验的建议与指导方案 (Recommendations)

为顺利执行 `ORIGINAL_REQUEST.md` 中的单变量消融、正交矩阵仿真及时频-倒谱指标提取，基于本调研提出以下实施建议：

### 7.1 基准单变量消融物理网格设计

建议在进行单变量敏感性扫描（Ablation Study）时，以四缝（`quad`）或双缝（`dual`）为标准参照系，设定如下基准值与扫描区间：

| 物理参数 | 符号 | 基准默认值 | 敏感性扫描网格建议 | 物理观测重点 |
| :--- | :---: | :---: | :---: | :--- |
| **首缝位置** | $x_{f,1}$ | $4100.0\,\mathrm{m}$ | $1000, 2000, 3000, 4100, 4800\,\mathrm{m}$ | 走时延迟、衰减包络、倒谱主峰深度 |
| **裂缝间距** | $\Delta x$ | $50.0\,\mathrm{m}$ | $5, 10, 15, 20, 30, 50, 100\,\mathrm{m}$ | 频域谱陷波间距、倒谱次峰分离度与混叠极限 |
| **水头柔度** | $C_H$ | $1.0 \times 10^{-5}\,\mathrm{m^2}$ | $10^{-6}, 3\times 10^{-6}, 10^{-5}, 3\times 10^{-5}, 10^{-4}\,\mathrm{m^2}$ | 波前梯级跌落 $(\partial H/\partial t)_{max}$、倒谱峰幅值 |
| **滤失系数** | $k_{leak}$ | 等效稳态反求 | $10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}\,\mathrm{m^{5/2}/s}$ | 长期直流漂移、残余稳态水头恢复、整体能量衰减率 |
| **射孔阻抗** | $R_p$ | $0.0\,\mathrm{s^2/m^5}$ | $0, 10, 50, 100, 300, 1000\,\mathrm{s^2/m^5}$ | 高频毛刺滤波效应、缝内水头阶跃滞后、倒谱幅值屏蔽 |
| **分流权重** | $w_i$ | 均分 $[0.25 \dots]$ | 集中度 $\alpha \in \{0.3, 1.0, 3.0, 10.0\}$，单簇偏流 | 稳态压力降、各簇反射相对强度对比 |

### 7.2 稳态 Darcy vs. Brunone 对照执行准则

1. **同参对照**：对所有参数网格点，必须严格在保持几何与裂缝参数 100% 相同的前提下，分别在 `friction="steady"` 与 `friction="brunone"` 下运行，输出成对波形；
2. **物理对照指标分离**：
   - Darcy 稳态模型用于观察纯裂缝动力学引起的瞬变阶跃与无耗散混响；
   - Brunone 模型用于量化管壁边界层高频剪切衰减对裂缝倒谱峰的“模糊/平滑效应”；
   - 对比两者的 RMS 衰减率、高频（$f > 1.5\,\mathrm{Hz}$）能量衰减谱斜率以及倒谱识别峰值信噪比（PVR）。

### 7.3 保证 100% 收敛与 PASS 的实践要点

1. **防止低压截断**：由于瞬态波反射会产生负压波（停泵降压），应确保初始井口水头 $H_0 \ge 300\,\mathrm{m}$，地层水头 $H_{ext} \le 100\,\mathrm{m}$，保证整个瞬变时程中 $H_f(t) - H_{ext} > 0$，避免单向滤失截断；
2. **避免网格碰撞**：在扫描间距 $\Delta x$ 时，若 $\Delta x < 5\,\mathrm{m}$，需特别核对 $\operatorname{round}(x_f / \Delta x_{grid})$ 是否发生重合；若需更高分辨率，可将时间步长由 $1\,\mathrm{ms}$ 减小至 $0.5\,\mathrm{ms}$（对应空间网格细化至 $\sim 0.725\,\mathrm{m}$）；
3. **射孔阻力模式**：若消融实验开启 $R_p > 0$，务必确保 $H_0$ 足够大以补偿稳态射孔压降，防止触发稳态内部水头不足报错。

---

## 8. 调研结论与后续衔接

通过对 MOC 仿真引擎与摩阻模型的全方位源码剖析与测试验证，证实该套仿真代码完全满足高精度数值实验的要求，控制方程完备，数值格式无人工耗散，裂缝物理机制详尽，且已经完全跑通全部测试套件。该调研为团队后续开展参数消融扫描、倒谱特征提取及敏感性量化报告编写提供了坚实可靠的底层支撑。
