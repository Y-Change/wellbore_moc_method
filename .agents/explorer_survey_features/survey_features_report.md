# 裂缝物理参数敏感性分析：时频/倒谱特征提取、指标体系与出版图版规范调研报告

**调研执行人**: teamwork_preview_explorer (explorer_survey_features)  
**日期**: 2026-09-09  
**工作目录**: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_features`  
**项目根目录**: `e:\water_hammer_research\wellbore_moc_method`  
**目标交付路径**: `output/fracture_parameter_sensitivity/`  

---

## 1. 调研背景与任务目标

根据 `ORIGINAL_REQUEST.md`，本项目旨在系统化开展新版 MOC 求解器下裂缝物理参数（位置 $x_f$、柔度系数 $C_H$、滤失系数 $k_\text{leak}$、射孔阻抗 $R_p$、分流权重 $w_i$ 及裂缝间距 $\Delta x$）对井筒水锤瞬态波形与声学倒谱特征的敏感性消融实验，定量揭示各参数对波形阶跃、能量衰减、频域损耗及倒谱定位幅值的映射规律，并生成学术级研究报告与标准数据集。

本调研报告聚焦于：
1. 梳理现有代码库（`analysis/`, `moc_simulate/`, `experiments/`）中成熟的信号处理与倒谱算法实现；
2. 严密确立 6 项核心指标的数学物理定义、离散算法与参数边界；
3. 规范结构化数据表 `sensitivity_metrics.csv` 与汇总文件 `sensitivity_summary.json` 的 schema；
4. 锚定 Nature-Figure 出版级绘图规范（$\ge 200\,\mathrm{dpi}$，多联图版，语义配色）及学术级 `README.md` 的内容组织。

---

## 2. 现有代码库架构与关键资产

调研覆盖了项目中相关模块的代码实现与历史论文沉淀，核心资产分布如下：

| 模块 / 路径 | 核心文件 | 关键功能与复用价值 |
| :--- | :--- | :--- |
| **求解器核心** | `moc_simulate/wellbore_moc.py` | 实现了 Courant 数为 1 的 1D MOC 求解器；稳态精确初始化（消灭 $t=0$ 伪波）；支持 Darcy 稳态与 Brunone 瞬态非定常摩阻；裂缝节点半隐式 Newton 耦合（储液 + 滤失 + 射孔压降 $R_p$）。 |
| **倒谱核心库** | `moc_simulate/cepstrum_mocdata.py` | 包含 `preprocess_moc_head`、`real_cepstrum_1d`、`cepstrogram`、`detect_1d_cepstrum_peaks`（防真值泄漏盲寻峰）、`effective_fft_fmax` 等完整工业级管线。 |
| **高阶倒谱分析** | `analysis/cepstrum/_kb_core.py` | 实现了带 Kaiser-Bessel 动窗、高频预加重（$\mu=0.97$）、升正弦/余弦 Lifter 的短时 2D 倒谱及 AR 参量倒谱算法。 |
| **分辨率理论** | `analysis/resolvability/forward_resolvability.py`<br>`analysis/cepstrum/spacing_resolvability.py` | 确立了瑞利判据正推体系：相干有效带宽 $B_{coh}$、有效谐波数 $N_{harm,eff}$、最小可分辨缝距 $\Delta d_{min} = a/(2 B_{coh}) = 2L/N_{harm,eff} \approx \text{FWHM}_d$。 |
| **衰减回归** | `analysis/decay_analysis/decay_regression_cf_kleak.py`<br>`analysis/decay_analysis/decay_regression.py` | 建立了 $C_f$ 与 $k_{leak}$ 双参数网格扫描与峰值衰减特征提取管线（`P_1d`, `P_2d`, 峰值归一化 $\alpha$）。 |
| **基准评估** | `moc_simulate/run_no_fracture_benchmark.py` | 提供了无裂缝对照基准、Joukowsky 理论降落、波前最大梯度、多窗口 RMS 衰减率及 FFT 高频能量比率的权威参考实现。 |
| **绘图规范** | `analysis/plotting/paper_plots.py` | 提供了符合 Nature 出版标准的 `apply_paper_rc`（Arial 字体、无顶部/右侧边框、无边框图例、语义化配色 `PALETTE`、SVG 原生矢量文本与 300 dpi 输出）。 |

---

## 3. 六项核心时频-倒谱特征的数学物理定义与算法

### 3.1 首波 Joukowsky 水击降落 (Joukowsky Head Drop)

#### 3.1.1 物理机理与解析解
水锤波由井口流速截断激发。根据 Korteweg–Joukowsky 关系，当管口流速发生瞬变 $\Delta V = V_f - V_0$ 时，弹性波沿管轴传播，井口激发的理论水头跃变为：
$$\Delta H_{\text{jouk}} = -\frac{a}{g} \Delta V$$
对于柱塞泵瞬时停泵关井（从初始流速 $V_0$ 骤降至 $0$），$\Delta V = -V_0$（源头截断），井口水头产生理论压力降落：
$$\Delta H_{\text{jouk}} = -\frac{a_{\text{adj}} V_0}{g} < 0$$
其中：
- $a_{\text{adj}}$ 为 MOC 网格调整后的声速（$\mathrm{m/s}$），满足 Courant 准则 $a_{\text{adj}} = \Delta x / \Delta t$；
- $V_0$ 为停泵前稳态轴向流速（$\mathrm{m/s}$）；
- $g = 9.81\,\mathrm{m/s^2}$（与 `wellbore_moc.py` 中 `G = 9.81` 统一）。

#### 3.1.2 数值提取算法
在仿真时程中，关阀发生在 $t_s$ 时刻，关阀历时为 $t_c$（瞬时关井时 $t_c = \Delta t$ 或 $1\,\mathrm{ms}$）。
1. 关阀前稳态基准水头：
   $$H_{\text{before}} = H_{wh}(t_s - 10\Delta t)$$
2. 关阀刚结束且首道反射波（裂缝或趾端）尚未到达井口前（满足 $t < t_s + 2x_1/a_{\text{adj}}$）的井口水头采样：
   $$H_{\text{after}} = H_{wh}(t_s + t_c + 10\Delta t)$$
3. 模拟实测降落量：
   $$\Delta H_{\text{sim}} = H_{\text{after}} - H_{\text{before}}$$
4. 相对误差百分比：
   $$\text{Err}_{\text{jouk}} = \frac{|\Delta H_{\text{sim}} - \Delta H_{\text{jouk}}|}{|\Delta H_{\text{jouk}}|} \times 100\%$$

> **代码对标**: `moc_simulate/run_no_fracture_benchmark.py` 第 317–328 行；`moc_simulate/step01_joukowsky.py` 第 57–70 行。

---

### 3.2 波前最大梯度 $(\partial H / \partial t)_{\text{max}}$ (Wavefront Max Gradient)

#### 3.2.1 物理机理
波前最大时间变化率反映了水动力波前的陡峭程度与高频阻尼耗散。在纯无粘或稳态摩阻流动中，波前理论上保持初始关阀激扰斜率；而在 Brunone 非定常摩阻下，由于加速度项 $\partial V / \partial t$ 在激变波前处剧烈突增，提供巨大的额外壁面剪切耗散，会导致波前发生剪切平滑（Wavefront Rounding/Diffusion）；同时，裂缝的流体柔度 $C_H$ 与近井射孔阻抗 $R_p$ 也会在波反射与透射阶段平滑局部阶跃。

#### 3.2.2 离散算法
1. 采用中心差分算子计算井口水头时域导数序列（`np.gradient`）：
   $$\left(\frac{\partial H}{\partial t}\right)_n = \frac{H_{wh}(t_{n+1}) - H_{wh}(t_{n-1})}{2 \Delta t}, \quad n = 1, \dots, N_t - 1$$
2. 波前阶跃搜索时间窗限定在关阀突变区间：
   $$t \in [t_s, t_s + t_c + 0.1\,\mathrm{s}]$$
3. 由于首波表现为压力骤降，最大下降速率为导数的极小负峰值：
   $$\left(\frac{\partial H}{\partial t}\right)_{\text{step,min}} = \min_{t \in [t_s, t_s + t_c + 0.1]} \left(\frac{\partial H}{\partial t}\right)$$
   其最大绝对梯度幅值即为：
   $$\left|\frac{\partial H}{\partial t}\right|_{\text{max}} = \max_{t \in [t_s, t_s + t_c + 0.1]} \left|\frac{\partial H}{\partial t}\right| = -\left(\frac{\partial H}{\partial t}\right)_{\text{step,min}}$$

> **代码对标**: `moc_simulate/run_no_fracture_benchmark.py` 第 330–333 行。

---

### 3.3 多时间窗 RMS 水头衰减率 (Multi-window RMS Head Attenuation Rate)

#### 3.3.1 物理机理
水锤压力脉冲在井筒与裂缝系统往返传播过程中，系统机械能因三部分机制而耗散：
1. 沿程 Darcy 稳态管壁摩阻；
2. Brunone 非定常边界层流速分布剖面重构引起的附加动态剪切；
3. 裂缝处动态滤失 $Q_{\text{leak}} = k_{\text{leak}} \sqrt{H_f - H_{\text{ext}}}$ 造成的流体逸散及孔眼阻抗 $R_p Q_f^2$ 的节流耗散。

#### 3.3.2 离散算法
1. **时间窗划分**：
   考虑总仿真时长 $t_f = 50.0\,\mathrm{s}$（覆盖约 7~8 个往返周期 $T_0 = 4L/a \approx 13.79\,\mathrm{s}$），将停泵后时段划分为 $K=5$ 个互不重叠的等宽时间窗：
   $$W_k = [t_{k,\text{start}}, t_{k,\text{end}}] = [(k-1) \times 10.0, k \times 10.0]\,\mathrm{s}, \quad k = 1, 2, 3, 4, 5$$
   （若 $t_f = 100.0\,\mathrm{s}$，则窗宽取 $20.0\,\mathrm{s}$，与 `run_no_fracture_benchmark.py` 保持一致：`[(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]`）。
2. **窗口水头波动 RMS 计算**：
   在时间窗 $W_k$ 内，提取水头序列并扣除窗口均值（消除地层平衡压力漂移）：
   $$\bar{H}_k = \frac{1}{M_k} \sum_{t_m \in W_k} H_{wh}(t_m)$$
   $$\text{RMS}_k = \sqrt{\frac{1}{M_k} \sum_{t_m \in W_k} \left[ H_{wh}(t_m) - \bar{H}_k \right]^2}$$
3. **衰减指标定义**：
   - **能量保留率 (RMS Retention Percentage)**：
     $$\text{Ret}_{\text{RMS}} = \frac{\text{RMS}_K}{\text{RMS}_1} \times 100\%$$
   - **累积水头衰减率 (Cumulative Attenuation Rate)**：
     $$\eta_{\text{decay}} = 1 - \frac{\text{RMS}_K}{\text{RMS}_1}$$
   - **等效指数衰减常数 (Exponential Decay Rate $\alpha_{\text{RMS}}$)**：
     对各窗中心时间 $t_{k,\text{mid}} = \frac{t_{k,\text{start}} + t_{k,\text{end}}}{2}$ 进行对数线性拟合：
     $$\ln(\text{RMS}_k) \approx -\alpha_{\text{RMS}} t_{k,\text{mid}} + C_0 \implies \alpha_{\text{RMS}} = -\frac{\mathrm{d} \ln \text{RMS}}{\mathrm{d} t} \quad [\mathrm{s^{-1}}]$$

> **代码对标**: `moc_simulate/run_no_fracture_benchmark.py` 第 43–49 行及 335–338 行；`analysis/decay_analysis/decay_regression.py`。

---

### 3.4 FFT 高频 ($f > 1.5\,\mathrm{Hz}$) 能量占比 (FFT High-Frequency Energy Ratio)

#### 3.4.1 物理机理
井筒系统的声学基频由井长与波速决定：
$$f_0 = \frac{a_{\text{adj}}}{4L}$$
当 $L = 5000\,\mathrm{m}, a \approx 1450\,\mathrm{m/s}$ 时，$f_0 \approx 0.0725\,\mathrm{Hz}$。高频截止阈值设定为 $f_{\text{cut}} = 1.5\,\mathrm{Hz}$，对应于 $f / f_0 \approx 20.7$（即超过第 20 阶奇次谐波）。  
高频分量主要源自初始关井波前的剧烈激变以及多道紧密裂缝的细密反射。由于 Brunone 瞬态摩阻具备高频低通阻尼特性，且裂缝储液柔度会吸收高频尖峰，高频能量占比 $R_{\text{high}}$ 是表征系统高频耗散能力的敏感指标。

#### 3.4.2 离散算法
1. **信号截取与去均值**：
   截取关泵完成后的自由振荡时段 $t \ge t_{\text{post}}$（如 $t \ge 5.0\,\mathrm{s}$ 或 $t \ge t_s$），避免关泵过渡期污染：
   $$x[n] = H_{wh}(t_n) - \frac{1}{N_{\text{post}}} \sum_{m} H_{wh}(t_m)$$
2. **单边幅值谱与离散傅里叶变换**：
   $$X(f_m) = \sum_{n=0}^{N_{\text{post}}-1} x[n] e^{-j 2\pi m n / N_{\text{post}}}$$
   频率网格为 $f_m = \frac{m}{N_{\text{post}} \Delta t}$，单边归一化物理幅值谱为：
   $$S(f_m) = |X(f_m)| \cdot \frac{2}{N_{\text{post}}}, \quad 0 \le f_m \le \frac{f_s}{2}$$
3. **功率谱分段求和**：
   - 剔除直流与极低频工况漂移（取 $f > 0.05\,\mathrm{Hz}$）计算有效总功率：
     $$P_{\text{total}} = \sum_{f_m > 0.05\,\mathrm{Hz}} |S(f_m)|^2$$
   - 计算高频段功率（$f_m > 1.5\,\mathrm{Hz}$）：
     $$P_{\text{high}} = \sum_{f_m > 1.5\,\mathrm{Hz}} |S(f_m)|^2$$
4. **高频能量占比**：
   $$R_{\text{high}} = \frac{P_{\text{high}}}{P_{\text{total}}} \times 100\%$$
5. **稳态与非定常耗散分贝差**：
   $$\Delta \mathrm{dB}_{\text{high}} = 10 \log_{10}\left( \frac{P_{\text{high,brunone}}}{P_{\text{high,steady}}} \right)$$

> **代码对标**: `moc_simulate/run_no_fracture_benchmark.py` 第 340–363 行。

---

### 3.5 1D 实倒谱峰值深度与幅值 (1D Real Cepstrum Peak Depth and Amplitude)

#### 3.5.1 物理机理与倒频率-深度映射
实倒谱（Real Cepstrum）将卷积信号转化为对数谱的叠加信号，通过逆变换提取时域回波周期的离散脉冲（Quefrency）。
在井口接收的水锤信号中，第 $i$ 裂缝位于深度 $x_{f,i}$，波从井口至裂缝往返双程走时（Two-Way Travel Time, TWTT）为：
$$\tau_i = \frac{2 x_{f,i}}{a}$$
倒频率轴 $q$（单位：$\mathrm{s}$）具有时间量纲，精确对应于波往返走时。因此，深度换算公式严格为：
$$d = \frac{q \cdot a_{\text{adj}}}{2} \iff q = \frac{2d}{a_{\text{adj}}}$$
深度网格分辨率为：
$$\delta d_{\text{bin}} = \frac{a_{\text{adj}}}{2 f_s} = \frac{a_{\text{adj}} \Delta t}{2}$$
当 $a = 1450\,\mathrm{m/s}, f_s = 1000\,\mathrm{Hz}$ 时，$\delta d_{\text{bin}} = 0.725\,\mathrm{m}$。

#### 3.5.2 符号反转与裂缝正响应准则
对于连通储层的压裂缝，其储液柔度 $C_H$ 与分流滤失导致声学阻抗局部降低（声阻抗 $Z_f < Z_{\text{pipe}}$），反射系数为负值（$R < 0$）。在标准实倒谱定义下，回波脉冲表现为下凹的负峰。  
为符合工程直觉与图像识别习惯，代码库规范定义倒谱响应为实倒谱的相反数：
$$\text{Response}(d) = -c(q) = -\mathrm{Re}\left\{ \mathrm{IFFT}\left( \ln|\mathrm{FFT}(x)| \right) \right\}$$
从而使裂缝反射在倒谱剖面上展现为显著的**正立特征尖峰**。

#### 3.5.3 完整计算管线
1. **预处理**（`preprocess_moc_head`）：
   - 重采样至标准采样率 $f_s = 1000\,\mathrm{Hz}$（使用三次样条 `interp1d`）；
   - 截取停泵后区间 $t \ge t_s$；
   - 减去直流分量：$x = H_{\text{after}} - \text{mean}(H_{\text{after}})$。
2. **倒谱变换**（`rfft` $\to \ln \to$ `irfft`）：
   $$X(f) = \mathrm{rfft}(x, n = N_{\text{fft}})$$
   $$\hat{X}(f) = \ln(|X(f)| + \epsilon), \quad \epsilon = 10^{-12}$$
   $$c(q) = \mathrm{irfft}(\hat{X}(f), n = N_{\text{fft}})$$
3. **Quefrency 裁剪与升正弦 Liftering**（可选）：
   $$q \in [q_{\text{min}}, q_{\text{max}}] = \left[0, \frac{2L}{a_{\text{adj}}}\right]$$
   若开启 Lifter，利用窗函数压制低倒频慢变井筒背景和近井管路混响。
4. **防泄漏盲寻峰算法**（`detect_1d_cepstrum_peaks`）：
   - **严禁向寻峰器传入真实裂缝坐标**（防真值泄漏，遵循 EXP-20260730-018 审计规约）；
   - 自适应幅值门限：
     $$\text{thresh} = \max\left( \text{percentile}(\text{response}, 90), 0.05 \times \max(\text{response}), 0.0 \right)$$
   - 最小峰距约束：
     $$\text{min\_sep\_bins} = \max\left(1, \text{round}\left( \frac{\Delta d_{\text{min,sep}}}{\delta d_{\text{bin}}} \right)\right), \quad \Delta d_{\text{min,sep}} = 5.0\,\mathrm{m}$$
   - 调用 `scipy.signal.find_peaks` 提取全部候选峰，按响应幅值降序排列，取 Top-N 作为检出峰。
5. **指标提取**：
   - 首峰深度 $d_{\text{peak},1}$ 与幅值 $A_{\text{peak},1}$；
   - 各缝实测定位残差：$e_i = |d_{\text{peak},i} - x_{f,i}^{\text{aligned}}|$；
   - 倒谱信噪比：
     $$\text{SNR}_{\text{ceps}} = \frac{\max(\text{response})}{\text{median}(|\text{response}|) + \epsilon}$$

> **代码对标**: `moc_simulate/cepstrum_mocdata.py` 第 215–230 行、576–643 行；`analysis/cepstrum/cepstrum_1d_pipeline.py`。

---

### 3.6 2D 滑窗倒谱空间分辨率 (2D Sliding-Window Cepstrum Spatial Resolution)

#### 3.6.1 物理机理与短时滑窗
1D 实倒谱使用停泵后全长信号，具有极高的频率分辨率，但丧失了时间局部化特征，易受多次往返波混叠干扰。2D 倒谱图（Cepstrogram）采用时移滑窗：
$$x_m[n] = \left( x[m \cdot \text{hop} + n] - \bar{x}_m \right) w[n], \quad n = 0, \dots, N_w - 1$$
沿时间滑动生成短时倒谱阵列 $C(q, t_m)$。  
随后将 2D 倒谱沿时间轴求和或平均，得到时均深度剖面（Time-Averaged Depth Profile）：
$$\bar{P}(d) = -\sum_{m=1}^{M_{\text{frames}}} C\left( \frac{2d}{a_{\text{adj}}}, t_m \right)$$

#### 3.6.2 瑞利判据与相干有效带宽正推
根据声学与信息论原理，倒谱峰在深度域的物理展宽（FWHM）受限于信号在频域谐波梳的**相干有效带宽** $B_{\text{coh}}$：
1. **幅值谱连通支撑提取**：
   设定动态范围门限（默认 $\text{DR} = 80\,\mathrm{dB}$，$\varepsilon = 10^{-\text{DR}/20} = 10^{-4}$）：
   $$|S(f)| \ge \varepsilon \cdot \max|S(f)|$$
   在低频段搜索连续连通谐波梳的最大截止频率，定义为相干带宽 $B_{\text{coh}}$：
   $$B_{\text{coh}} := f_{\text{support,max}}$$
2. **有效谐波数**：
   $$N_{\text{harm,eff}} = \frac{B_{\text{coh}}}{f_0}, \quad f_0 = \frac{a}{4L}$$
3. **理论极限空间分辨率（瑞利准则）**：
   两相邻裂缝在倒谱域能够被清晰区分为两个独立极值峰的最小物理间距为：
   $$\Delta d_{\text{min}} = \frac{a_{\text{adj}}}{2 B_{\text{coh}}} = \frac{2L}{N_{\text{harm,eff}}} \approx \text{FWHM}_d$$
4. **滑窗窗长约束**：
   短时滑窗长度 $T_w = w_{\text{len}}$（秒）决定了频域分辨率 $\Delta f_w = 1 / T_w$。为了能分辨水锤谐波梳（谱线间隔 $2f_0$），窗长必须满足硬约束：
   $$T_w \ge \frac{4L}{a_{\text{adj}}} = T_0 \approx 13.79\,\mathrm{s}$$
   若窗长 $T_w < T_0$，频域谱峰发生严重涂抹，倒谱无法形成锐利聚焦峰。

#### 3.6.3 分辨力指标计算
1. **实测半高全宽 (FWHM)**：
   对检出的时均剖面主峰，在半峰高处找到左右截距 $d_{\text{left}}, d_{\text{right}}$：
   $$\text{FWHM}_{d} = d_{\text{right}} - d_{\text{left}}$$
2. **邻缝分离判别 (Separation Success)**：
   对于设计间距为 $\Delta x$ 的双缝或多缝系统，若盲寻峰算法在容差 $\text{tol} \le 0.5 \Delta x$ 内成功检出与真实缝数完全对应的独立峰，则 `separation_success = 1`；若两缝合为一个宽展单峰，则判定为发生“峰合并”（`n_likely_merged = 1, separation_success = 0`）。

> **代码对标**: `analysis/resolvability/forward_resolvability.py` 第 7–27 行、120–166 行；`analysis/cepstrum/spacing_resolvability.py`；`analysis/cepstrum/_kb_core.py`。

---

## 4. 数据表结构与 Schema 规范

### 4.1 `sensitivity_metrics.csv` 字段定义表

输出结构化 CSV 表需汇集工况配置、控制网格、水动力波形指标、频域能量指标及 1D/2D 倒谱指标，确保每一行具有唯一工况与完整自洽性。

| 字段类别 | 字段名称 (CSV Column) | 数据类型 | 单位 | 详细含义与定义 |
| :--- | :--- | :--- | :--- | :--- |
| **元数据** | `case_id` | string | — | 工况唯一识别码（如 `sens_xf_01`, `sens_ch_03`） |
| | `study_type` | string | — | 消融类型（`single_var_xf`, `single_var_ch`, `single_var_kleak`, `single_var_Rp`, `single_var_wi`, `single_var_spacing`, `orthogonal`） |
| | `friction_model` | string | — | 摩阻模型（`steady` 或 `brunone`） |
| | `brunone_k_scale` | float | — | Brunone 缩放倍率（默认 1.0） |
| | `toe_bc` | string | — | 趾端边界（`dead_end` 封闭端 / `reservoir` 恒压端） |
| **裂缝参数** | `n_frac` | integer | — | 裂缝总数 |
| | `x_f_list_m` | string | m | 实际网格对齐后的裂缝位置序列（用分号分隔） |
| | `x_f_first_m` | float | m | 首缝井深 $x_1$ |
| | `spacing_m` | float | m | 裂缝簇间距 $\Delta x$ |
| | `compliance_head_m2` | float | $\mathrm{m^2}$ | 裂缝水头柔度 $C_H$（$C_H = \rho g C_p$） |
| | `kleak_equiv` | float | $\mathrm{m^{5/2}/s}$ | 稳态闭合等效滤失系数 $k_{\text{leak}}$ |
| | `Rp_s2_m5` | float | $\mathrm{s^2/m^5}$ | 射孔孔眼阻抗 $R_p$ |
| | `inflow_weights_str` | string | — | 稳态分流权重 $w_i$ 字符串（如 `0.250;0.250;0.250;0.250`） |
| | `inflow_weight_skew` | float | — | 进液不均衡度（$\max(w_i) / \min(w_i)$ 或标准差） |
| **管路声学** | `wellbore_length_m` | float | m | 井筒全长 $L$ |
| | `wavespeed_adj_m_s` | float | m/s | 网格调整后真实波速 $a_{\text{adj}}$ |
| | `initial_velocity_m_s`| float | m/s | 关泵前稳态轴向流速 $V_0$ |
| | `dt_s` | float | s | 仿真时间步长 $\Delta t$ |
| | `f0_acoustic_Hz` | float | Hz | 声学基频 $f_0 = a/(4L)$ |
| **波形时域** | `joukowsky_analytical_m`| float | m | 理论 Joukowsky 降落值 $-a V_0 / g$ |
| | `joukowsky_sim_m` | float | m | 仿真井口首波实测跳变降落 |
| | `joukowsky_err_pct` | float | % | Joukowsky 降落相对误差 |
| | `wavefront_max_gradient_m_s`| float | m/s | 波前最大负斜率 $(\partial H/\partial t)_{\text{min}}$ |
| | `rms_window_1_m` | float | m | 0~10s (或 0~20s) 波动 RMS 水头 |
| | `rms_window_2_m` | float | m | 10~20s (或 20~40s) 波动 RMS 水头 |
| | `rms_window_3_m` | float | m | 20~30s (或 40~60s) 波动 RMS 水头 |
| | `rms_window_4_m` | float | m | 30~40s (或 60~80s) 波动 RMS 水头 |
| | `rms_window_5_m` | float | m | 40~50s (或 80~100s) 波动 RMS 水头 |
| | `rms_retention_pct` | float | % | 终期/初期 RMS 水头保留率 |
| | `rms_decay_alpha_per_s` | float | 1/s | 指数衰减率拟合斜率 $\alpha_{\text{RMS}}$ |
| **频域能量** | `power_total_m2` | float | $\mathrm{m^2}$ | $f > 0.05\,\mathrm{Hz}$ 总谱功率 |
| | `power_high_m2` | float | $\mathrm{m^2}$ | $f > 1.5\,\mathrm{Hz}$ 高频谱功率 |
| | `high_freq_power_ratio_pct`| float | % | 高频能量占比 $P_{\text{high}}/P_{\text{total}}$ |
| | `high_freq_db_drop_vs_steady`| float | dB | 相对稳态基准的高频功率衰减分贝数 |
| **1D 倒谱** | `ceps_1d_peak_depth_m` | float | m | 盲寻峰检出的首缝倒谱深度 |
| | `ceps_1d_peak_amp` | float | — | 首缝 1D 倒谱响应正峰值 $-c(q_1)$ |
| | `ceps_1d_depth_error_m`| float | m | 1D 倒谱定位绝对误差 $|d_1 - x_{f,1}|$ |
| | `ceps_1d_snr` | float | — | 1D 倒谱峰背比信噪比 |
| | `ceps_1d_n_detected` | integer | — | 1D 检出峰总数 |
| **2D 倒谱** | `ceps_2d_wlen_s` | float | s | 2D 短时滑窗长度（默认 30.0s） |
| | `ceps_2d_peak_depth_m` | float | m | 2D 时均剖面检出首缝深度 |
| | `ceps_2d_peak_amp` | float | — | 2D 时均剖面首缝响应幅值 |
| | `ceps_2d_fwhm_depth_m` | float | m | 检出峰实测半高宽 FWHM |
| | `ceps_2d_spatial_resolution_m`| float | m | 理论瑞利分辨率 $\Delta d_{\text{min}} = a/(2 B_{\text{coh}})$ |
| | `ceps_2d_separation_success` | integer | — | 多缝是否成功分辨为独立峰（1=PASS, 0=MERGED） |
| | `ceps_2d_n_likely_merged` | integer | — | 发生涂抹合并的相邻缝数 |
| **数值守恒** | `mass_conservation_max_err_m3_s`| float | $\mathrm{m^3/s}$ | 稳态/瞬态最大质量不平衡误差 |
| | `min_head_margin_m` | float | m | $\min_{t,i}(H_{f,i}(t) - H_{\text{ext}})$，必须 $> 0$ |
| | `convergence_status` | string | — | `PASS` / `FAIL` |

---

### 4.2 `sensitivity_summary.json` 层次结构规范

JSON 文件用于记录全局元数据、网格扫描范围、统计分位数、参数敏感性重要度排序（Feature Importance）及 Steady vs Brunone 对照差分分析：

```json
{
  "metadata": {
    "project": "fracture_parameter_sensitivity",
    "schema_version": "moc_lhs_v2.1",
    "timestamp_utc": "2026-09-09T12:00:00Z",
    "solver": "wellbore_moc.py (v2.1 discrete steady-state compatible)",
    "integrity_mode": "development",
    "total_cases_run": 84,
    "converged_cases": 84,
    "failed_cases": 0,
    "pass_rate_pct": 100.0
  },
  "parameter_ranges": {
    "x_f_m": [3500.0, 4800.0],
    "compliance_head_m2": [1.0e-6, 1.0e-4],
    "kleak_equiv": [1.0e-6, 1.0e-3],
    "Rp_s2_m5": [0.0, 1000.0],
    "spacing_m": [5.0, 100.0],
    "inflow_weight_alpha_dirichlet": [0.3, 1.0, 3.0, 10.0],
    "friction_models": ["steady", "brunone"]
  },
  "global_metrics_summary": {
    "joukowsky_error_pct": {
      "mean": 0.32,
      "max": 1.15,
      "min": 0.08
    },
    "high_freq_power_ratio_pct": {
      "steady_mean": 2.15,
      "brunone_mean": 0.45,
      "attenuation_delta_db_mean": -6.8
    },
    "rms_retention_pct": {
      "steady_mean": 38.4,
      "brunone_mean": 21.2
    },
    "cepstrum_1d_depth_error_m": {
      "mean": 0.42,
      "p95": 0.725,
      "max": 1.45
    }
  },
  "sensitivity_rankings": {
    "target_cepstrum_peak_amp": [
      {"rank": 1, "parameter": "compliance_head_m2", "sensitivity_index": 0.78, "mechanism": "Direct compliance volume expansion produces dominant acoustic reflection step."},
      {"rank": 2, "parameter": "inflow_weight", "sensitivity_index": 0.54, "mechanism": "Larger flow allocation increases local steady head drop and dynamic amplitude."},
      {"rank": 3, "parameter": "Rp_s2_m5", "sensitivity_index": -0.38, "mechanism": "Perforation throttling cushions acoustic surge into fracture cavity, reducing peak amplitude."},
      {"rank": 4, "parameter": "kleak_equiv", "sensitivity_index": 0.22, "mechanism": "Leakoff drives net energy drain but has weaker high-frequency reflection than compliance."},
      {"rank": 5, "parameter": "x_f", "sensitivity_index": -0.15, "mechanism": "Deeper fractures experience greater upstream friction attenuation before returning to wellhead."}
    ],
    "target_rms_attenuation_rate": [
      {"rank": 1, "parameter": "friction_model", "sensitivity_index": 0.85, "mechanism": "Brunone dynamic boundary layer shear dissipation dominates long-term cycle damping."},
      {"rank": 2, "parameter": "kleak_equiv", "sensitivity_index": 0.62, "mechanism": "Continuous mass and energy extraction accelerates pressure wave envelope decay."},
      {"rank": 3, "parameter": "compliance_head_m2", "sensitivity_index": 0.41, "mechanism": "Compliance causes internal wave reverberation and phase cancellation."},
      {"rank": 4, "parameter": "n_frac", "sensitivity_index": 0.35, "mechanism": "Multiple fracture partitions multiply the transmission loss surfaces."}
    ],
    "target_2d_spatial_resolution": [
      {"rank": 1, "parameter": "spacing_m", "sensitivity_index": 0.92, "mechanism": "Physical distance determines if echoes fall within Rayleigh FWHM lobe."},
      {"rank": 2, "parameter": "friction_model", "sensitivity_index": -0.45, "mechanism": "Brunone high-frequency roll-off narrows B_coh, expanding minimum resolvable spacing."},
      {"rank": 3, "parameter": "compliance_head_m2", "sensitivity_index": 0.31, "mechanism": "Stronger compliance enhances signal-to-noise ratio, aiding deconvolution separation."}
    ]
  },
  "cases": {
    "sens_xf_01": {
      "study_type": "single_var_xf",
      "friction_model": "brunone",
      "x_f_m": 3500.0,
      "compliance_head_m2": 1.0e-5,
      "joukowsky_err_pct": 0.28,
      "wavefront_max_gradient_m_s": -73650.0,
      "rms_retention_pct": 22.4,
      "high_freq_power_ratio_pct": 0.42,
      "ceps_1d_peak_depth_m": 3500.275,
      "ceps_1d_peak_amp": 0.1245,
      "ceps_2d_spatial_resolution_m": 18.2,
      "convergence_status": "PASS"
    }
  }
}
```

---

## 5. 出版级图版 (Nature-Figure) 规范与图版规划

### 5.1 Nature 级制图规范硬性要求

所有图版绘制必须依托 `analysis/plotting/paper_plots.py` 的基础设计规范：

1. **字体与文本节点**：
   - 统一使用无衬线字体：`Arial`, `Helvetica`, `DejaVu Sans`；
   - 必须设置 `svg.fonttype = 'none'` 和 `pdf.fonttype = 42`，保证导出的矢量 SVG/PDF 文件中文本节点保持为原生可编辑字符（`<text>`），禁止被曲线化（Paths）；
   - 基础字体字号：大图版复合排版基准 8 pt，标题 9 pt，轴标 8 pt，刻度与图例 7 pt。
2. **线条与边框**：
   - 遵照 Nature 习惯，默认关闭右侧和顶部边框（`axes.spines.right = False`, `axes.spines.top = False`）；
   - 坐标轴线宽设为 0.8~1.0 pt；
   - 去除默认繁杂网格线（若展示刻度辅助，用透明度 `alpha=0.3` 的细点划线）；
   - 图例无边框（`legend.frameon = False`），位置置于空白区。
3. **分辨率与导出**：
   - 栅格导出图（PNG）分辨率必须达到 $\ge 200\,\mathrm{dpi}$（推荐原生 **300 dpi**，`savefig.dpi = 300`）；
   - 双重交付：主矢量交付 `.svg`，快速预览交付 `.png`（`save_figure(fig, path, dpi=300, also_svg=True)`）。
4. **子图编号与标注**：
   - 多联图左上角必须标注粗体小写英文字母分图标签（**a**, **b**, **c**, **d**），统一通过 `add_panel_label(ax, 'a')` 调用；
   - 关键物理事件（如停泵时刻 $t_s$、理论波首到达时刻 $t_s + 2x_1/a$、基频 $f_0$、瑞利极限 $\Delta d_{\text{min}}$）必须添加物理垂直参考虚线和文本标注。
5. **语义化色彩映射 (`PALETTE`)**：
   - 稳态摩阻基准 (Steady / Baseline)：`PALETTE["blue_main"]` (`#0F4D92`, 典雅深蓝)；
   - Brunone 非定常对比 (Unsteady / Contrast)：`PALETTE["red_strong"]` (`#B64342`, 绯红)；
   - 积极/主效应变量：`PALETTE["teal"]` (`#42949E`) 或 `PALETTE["green_3"]` (`#8BCF8B`)；
   - 辅助分级与多缝对比：`PALETTE["violet"]` (`#9A4D8E`)、`PALETTE["neutral_dark"]` (`#4D4D4D`)；
   - 二维热力云图：优先选用知觉均匀色图 `viridis`, `plasma` 或 `cividis`。

---

### 5.2 四组核心对比图版架构设计

按照 `ORIGINAL_REQUEST.md` 验收要求，必须产出至少 4 组对应核心参数消融机制的对比图版（保存在 `output/fracture_parameter_sensitivity/plots/`）：

#### 图版 1: 物理参数阶跃响应与波前动力学对比 (`fig1_wavefront_step_response.png`)
- **Panel a (全时程宏观波形)**：井口水头 $H_{wh}(t)$ 时程，对比无缝、单缝、多缝及 Steady vs Brunone，标明长周期往返包络。
- **Panel b (波前阶跃特写)**：聚焦 $t \in [0.95, 1.25]\,\mathrm{s}$，清晰呈现首波 Joukowsky 理论降落阶跃、实测降落台阶及误差。
- **Panel c (波前时间梯度谱 $(\partial H/\partial t)$)**：对比不同柔度 $C_H$ 与孔眼阻抗 $R_p$ 对波前陡峭度的平滑作用，量化最大下降速率 $(\partial H/\partial t)_{\text{max}}$ 的削弱效应。
- **Panel d (裂缝反射阶梯波)**：展示脉冲波抵达裂缝后返回井口的初次负反射阶梯，对比间距 $\Delta x \in [10, 20, 50, 100]\,\mathrm{m}$ 下多道阶跃波的到达时延与台阶交叠过程。

#### 图版 2: 包络衰减对比与多时间窗 RMS 耗散图版 (`fig2_envelope_decay_attenuation.png`)
- **Panel a (包络提取与衰减曲线)**：稳态摩阻 vs Brunone 摩阻在全时程（0~50s）下的压力波动峰值追踪曲线与 Hilbert 振幅包络线。
- **Panel b (多时间窗 RMS 水头柱状对比)**：5 个时间窗（0~10s, 10~20s, 20~30s, 30~40s, 40~50s）内 Steady vs Brunone 的 RMS 水头柱状对比及衰减剪刀差。
- **Panel c ($C_H - k_{\text{leak}}$ 联合衰减相图)**：以柔度 $C_H$ 为横轴、滤失 $k_{\text{leak}}$ 为纵轴的 RMS 能量保留率 $\text{Ret}_{\text{RMS}}$ 二维热力等值线图（`heatmap_with_contour`）。
- **Panel d (摩阻耗散 vs 裂缝动力学耗散解耦)**：通过差分法（$\Delta H = H_{\text{steady}} - H_{\text{brunone}}$）将壁面瞬态剪切耗散率与裂缝储液/滤失耗散率定量分解为正交两极柱状图。

#### 图版 3: 频谱对数耗散与高频能量损耗图版 (`fig3_spectral_log_dissipation.png`)
- **Panel a (线性幅值谱与声学谐波梳)**：FFT 幅值谱 $|S(f)|$，标定水锤基频 $f_0 = a/(4L) \approx 0.0725\,\mathrm{Hz}$ 及其奇次谐波（$3f_0, 5f_0, \dots$），验证几何周期不变性。
- **Panel b (对数功率谱与高频滚降)**：对数功率谱 $\log_{10}|S(f)|^2$，高亮 $f > 1.5\,\mathrm{Hz}$ 高频区，展示 Brunone 瞬态摩阻带来的陡峭低通滤波衰减（谱线压制 20~40 dB）。
- **Panel c (高频能量占比敏感性散点图)**：高频能量占比 $R_{\text{high}}$ 随柔度 $C_H$、滤失 $k_{\text{leak}}$ 及孔眼阻抗 $R_p$ 变化的响应散点折线图。
- **Panel d (累积谱能量集中度 CDF)**：$E(f) / E_{\text{total}}$ 随频率的积分曲线，定量揭示 Brunone 与裂缝耦合如何将能量锁定在极低频带（$< 0.5\,\mathrm{Hz}$）。

#### 图版 4: 1D 实倒谱与 2D 滑窗倒谱空间分辨率特写 (`fig4_cepstrum_1d_2d_resolution.png`)
- **Panel a (1D 实倒谱深度剖面)**：展示沿深度 $d = q a / 2$ 的倒谱响应 $-c(q)$，标定真实裂缝位置（垂直红虚线），展示首缝主峰与多次谐波反射峰。
- **Panel b (倒谱峰幅值与物理参数标度律)**：倒谱峰高 $A_{\text{peak}}$ 随柔度 $C_H$、进液分流权重 $w_i$ 及阻抗 $R_p$ 的对数定标曲线，确立正比/幂律映射。
- **Panel c (2D 短时滑窗倒谱时空云图)**：二维热力图展示 $C(d, t)$ 在深度轴与时程轴上的能量演化，揭示裂缝特征能量在时域的持续性与衰减边界。
- **Panel d (间距消融与瑞利极限分离曲线)**：提取不同间距 $\Delta x \in [5, 10, 20, 50, 100]\,\mathrm{m}$ 下的时均倒谱剖面，标明测得的峰宽 $\text{FWHM}_d$，清晰展示 $\Delta x < \Delta d_{\text{min}}$ 处的“双峰合并”与 $\Delta x \ge \Delta d_{\text{min}}$ 处的“清晰分辨”，确立声学倒谱反演的物理可行域边界。

---

## 6. 学术级综合研究报告 (`README.md`) 编制要求

在输出目录 `output/fracture_parameter_sensitivity/README.md` 中，必须产出具备直接支撑 SCI Zone 1 / Top 顶刊发表水准的 Markdown 报告。报告应严格遵循以下结构与技术要求：

### 6.1 报告架构大纲
1. **Title & Abstract (标题与中文学术摘要)**：
   - 阐明研究动机、MOC 数值实验设计、核心发现与对深度学习反演的启示。
2. **Physical Models and Formulations (物理建模与控制方程)**：
   - 井筒可压缩非定常特征线方程组；
   - Darcy 稳态与 Brunone 瞬态摩阻本构；
   - 裂缝节点非线性动力学（水头柔度 $C_H$、滤失 $k_{\text{leak}}$、射孔二次阻抗 $R_p$、分流单纯形 $\sum w_i = 1$）；
   - 封闭趾端离散特征线精确稳态初始化原理（杜绝 $t=0$ 伪水击）。
3. **Experimental Setup & Grid Integrity (消融实验网格与数值完整性验证)**：
   - Courant 条件与网格参数（$C_r = 1.0, N = 3448, \Delta x = 1.45\,\mathrm{m}, \Delta t = 1.0\,\mathrm{ms}$）；
   - 参数空间网格与 LHS 正交采样表；
   - 严格的质量守恒校验（$\max |\sum Q_{\text{leak}} - Q_{in}| \le 10^{-10}$）与绝对压力空化裕度审计（$\min(H_f - H_{\text{ext}}) > 0$）。
4. **Multi-dimensional Sensitivity Analysis (多维参数敏感性机理深度剖析)**：
   - **机理 1：波前阶跃、Joukowsky 降落与非定常剪切波平滑**；
   - **机理 2：管壁动态阻尼 vs 裂缝动力耗散的能量解耦（多窗 RMS 衰减）**；
   - **机理 3：声学频域对数耗散、高频滚降与谐波梳结构**；
   - **机理 4：1D 倒谱峰幅定标律与 2D 滑窗倒谱瑞利空间分辨极限**。
5. **Sensitivity Matrix & Parameter Influence Rankings (定量指标矩阵与敏感度全景排序)**：
   - 嵌入核心敏感度定量结果表；
   - 给出各裂缝参数对倒谱幅值、峰位偏差、波形衰减率及高频吸收的 Spearman/Pearson 排序与影响等级梯队。
6. **Implications for Deep Neural Inversion (对深度神经算子与扩散模型反演的指导意义)**：
   - 明确指出裂缝参数的**可辨识区（Well-conditioned Regime）**与**退化混淆区（Ill-posed Degeneracy）**（例如微小柔度与高摩阻的低通混淆、近距裂缝的相干合并）；
   - 为后续 FNO / DiT / CJNO 神经网络反演架构设计（硬物理约束层、分频段损失函数加权）提供定量的物理先验边界。
7. **Publication Plates Catalog (出版级高清图版说明与嵌入)**：
   - 规范嵌入 Fig 1 ~ Fig 4 高清图版及详尽的图注（Figure Captions）。
8. **Reproducibility & Data Artifacts (可复现性指南与交付清单)**：
   - 详尽列出 NPZ 数据集、CSV 指标表及重现运行指令。

---

## 7. 调研结论与后续实施路线图

1. **算法与方程闭环**：
   - 本项目底层库在 `moc_simulate/wellbore_moc.py` 和 `moc_simulate/cepstrum_mocdata.py` 中已完整具备六大特征算法的数值基座。
   - 所有算法均已实现无真值泄漏的“盲提取”（Blind Detection）与严格的守恒检查。
2. **数据格式与兼容性**：
   - 完全对齐 `moc_lhs_v2.1` 生产规范，保证生成的 `sensitivity_metrics.csv` 与 `sensitivity_summary.json` 可直接被下游神经算子模型无缝消费。
3. **图版与报告准备度**：
   - 绘图基础设施 `paper_plots.py` 完全就绪，支持 Arial 矢量 SVG 与 300 dpi PNG 输出，子图标签与 Nature 语义配色均已预置。
4. **对调度者（Parent Agent）与后续实施者的建议**：
   - 建议基于 `moc_simulate/wellbore_moc.py` 编写批量单变量与网格扫描运行脚本 `run_sensitivity_study.py`；
   - 扫描过程建议开启双摩阻（`steady` 与 `brunone`）并行对比；
   - 后处理特征提取严格调用 `compute_moc_cepstrum_1d`、`detect_1d_cepstrum_peaks` 及多窗 RMS / FFT 算法；
   - 最终执行制图并渲染 `output/fracture_parameter_sensitivity/README.md`。
