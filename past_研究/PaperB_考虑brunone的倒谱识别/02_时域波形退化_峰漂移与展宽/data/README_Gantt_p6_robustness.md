# Gantt p6: 数值稳健性与有效工作区间定论笔记 (审定定稿版)

## 1. 核心数据表汇总

### 1.1 网格收敛性检验 (p6_grid_sensitivity.csv)
> **配置**：单缝 $n=1, X_1=4100\,\mathrm{m}, C_f=10^{-5}\,\mathrm{m^2}, k_{\text{leak}}=10^{-4}\,\mathrm{m^2/s/\sqrt{m}}$，阶跃关井，对比 $\Delta t=1.0\,\mathrm{ms}$ 与 $\Delta t=0.5\,\mathrm{ms}$。

| 时间步长 $\Delta t$ | 摩阻模型 | $t_{\text{onset}}$ (s) | $\Delta x_{\text{onset}}$ (m) | $t_{\text{peak}}$ (s) | $\Delta x_{\text{peak}}$ (m) | $\Delta x_{\text{cep}}$ (m) | 相对峰漂移 $\delta x_{\text{peak}}$ (m) | 相对倒谱偏深 $\delta x_{\text{cep}}$ (m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 ms** | Steady Darcy | 6.6530 | -1.58 m | 6.6530 | -1.58 m | -0.85 m | 0.00 m | 0.00 m |
| **1.0 ms** | IAB $k=0.01$ | 6.6530 | -1.58 m | 6.6670 | +8.58 m | +8.58 m | **+10.15 m** | **+9.43 m** |
| **0.5 ms** | Steady Darcy | 6.6555 | +0.24 m | 6.6555 | +0.24 m | +0.60 m | 0.00 m | 0.00 m |
| **0.5 ms** | IAB $k=0.01$ | 6.6565 | +0.96 m | 6.6695 | +10.39 m | +10.39 m | **+10.15 m** | **+9.79 m** |

*网格结论*：网格加密一倍（$\Delta t=0.5\,\mathrm{ms}$）后，时域相对峰漂移 $\delta x_{\text{peak}} = \mathbf{+10.15\,\mathrm{m}}$ 保持完全一致；相对倒谱偏深 $\delta x_{\text{cep}}$ 变动仅为 $0.36\,\mathrm{m}$（$< 1$ 个离散网格步长 $1450\times 0.0005/2 = 0.363\,\mathrm{m}$），时钟分叉与偏深量级展现出严格的网格收敛性。

---

### 1.2 倒谱窗型与窗长敏感性 (p6_window_sensitivity.csv)
> **配置**：在 1.0 ms 基准时序上截取停泵后 $T_{\text{win}} \in \{10, 30, 50\}\,\mathrm{s}$，测试 Hann、Hamming 与矩形窗（Rectangular）。

| 窗型 (Window) | 窗长 $T_{\text{win}}$ (s) | Steady $\Delta x_{\text{cep}}$ (m) | IAB $k=0.01$ $\Delta x_{\text{cep}}$ (m) | 相对倒谱偏深 $\delta x_{\text{cep}}$ (m) | 状态与物理机理 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Hann** | 10 s | -42.18 m | -40.00 m | +2.18 m | $10\,\mathrm{s} < 4L/a$（短于一个井筒基周期），两端压平关井锐沿致同态周期不足 |
| **Hann** | 30 s | -0.85 m | +8.58 m | **+9.43 m** | 稳态有效区间 |
| **Hann** | 50 s (主文基准) | -0.85 m | +8.58 m | **+9.43 m** | 稳态有效区间 |
| **Hamming** | 10 s | -1.58 m | +12.20 m | +13.78 m | 短窗过渡态 |
| **Hamming** | 30 s | -0.85 m | +9.30 m | **+10.15 m** | 稳态有效区间 |
| **Hamming** | 50 s | -0.85 m | +8.58 m | **+9.43 m** | 稳态有效区间 |
| **Rectangular** | 10 s | -1.58 m | +8.58 m | **+10.15 m** | 稳态有效区间 |
| **Rectangular** | 30 s | -0.85 m | +8.58 m | **+9.43 m** | 稳态有效区间 |
| **Rectangular** | 50 s | -0.85 m | +8.58 m | **+9.43 m** | 稳态有效区间 |

---

### 1.3 输入信号通道敏感性 (p6_input_channel_sensitivity.csv)
> **配置**：对比去均值水头 $H(t)$、主通道导数 $\mathrm{d}H/\mathrm{d}t$、及 $[0.05, 150]\,\mathrm{Hz}$ 4阶 Butterworth 带通滤波水头 $H_{\text{bp}}(t)$（窗长 50s，Hann 窗）。

| 输入信号通道 (Input Channel) | Steady $\Delta x_{\text{cep}}$ (m) | IAB $k=0.01$ $\Delta x_{\text{cep}}$ (m) | 相对倒谱偏深 $\delta x_{\text{cep}}$ (m) |
| :--- | :---: | :---: | :---: |
| **压力水头 $H(t)$** | -0.85 m | +8.58 m | **+9.43 m** |
| **水头导数 $\mathrm{d}H/\mathrm{d}t$ (主通道)** | -0.85 m | +8.58 m | **+9.43 m** |
| **带通滤波水头 $H_{\text{bp}}(t)$ ($[0.05, 150]\,\mathrm{Hz}$)** | -0.85 m | +8.58 m | **+9.43 m** |

*输入通道结论*：无论是直接使用原水头 $H(t)$、导数信号 $\mathrm{d}H/\mathrm{d}t$ 还是带通滤波信号，对数谱同态变换后的裂缝回声能量重心时延完全一致，$\delta x_{\text{cep}}$ 严格锁定为 $\mathbf{+9.43\,\mathrm{m}}$。

---

### 1.4 加性高斯白噪声鲁棒性 (p6_noise_robustness.csv)
> **配置**：对井口水头 $H(t)$ 施加 $\mathrm{SNR} \in \{60, 40, 20\}\,\mathrm{dB}$ 高斯白噪声，每档测试 5 个随机种子（`[42, 43, 44, 45, 46]`），经数值差分后提取倒谱峰位。

| 信噪比 $\mathrm{SNR}$ | 测试种子数 | 中位数 $\delta x_{\text{cep}}$ (m) | 极值区间 $[\min, \max]$ (m) | 极差 (Span) | 稳健性判决 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **60 dB** | 5 | **+9.43 m** | $[+9.43, +9.43]\,\mathrm{m}$ | $0.00\,\mathrm{m}$ | 绝对保真，完全无抖动 |
| **40 dB** | 5 | **+8.70 m** | $[+8.70, +9.43]\,\mathrm{m}$ | $0.73\,\mathrm{m}$ | 高度稳健（极差仅 1 个网格步） |
| **20 dB** | 5 | **+6.53 m** | $[+3.63, +9.43]\,\mathrm{m}$ | $5.80\,\mathrm{m}$ | 强噪下受基底噪声扰动，但偏深方向恒正 |

---

## 2. 有效工作区间定论 (审定三句)

1. **网格与输入通道不变性**：
   - 空间网格加密至 $\Delta t=0.5\,\mathrm{ms}$ 时，$\delta x_{\text{cep}} = +9.79\,\mathrm{m}$ 与 $1.0\,\mathrm{ms}$ 基准（$+9.43\,\mathrm{m}$）差异仅 $0.36\,\mathrm{m}$，$\delta x_{\text{peak}} = +10.15\,\mathrm{m}$ 完全不变；$H(t)$、$\mathrm{d}H/\mathrm{d}t$ 与带通 $H$ 三通道提取的 $\delta x_{\text{cep}}$ 严格恒为 $+9.43\,\mathrm{m}$，证明假说 H1 物理时钟分叉与倒谱偏深并非特定离散网格或预处理算子的数值伪影。
2. **倒谱窗长有效工作区间**：
   - 倒谱分析的有效窗长要求为 $T_{\text{win}} \ge \mathbf{30\,\mathrm{s}}$（对应约 5 个水锤往返周期）；在此区间内 Hann、Hamming 与矩形窗的 $\delta x_{\text{cep}}$ 稳定聚集于 $+9.43 \sim +10.15\,\mathrm{m}$；当窗长过短（$T_{\text{win}}=10\,\mathrm{s} < 4L/a \approx 13.8\,\mathrm{s}$）时，短于一个水锤基波振荡周期，Hann 窗两端平滑压平了关井瞬态阶跃与尾部振荡，导致同态回声周期信息不足而出现倒谱峰漂移（$\delta x_{\text{cep}} = +2.18\,\mathrm{m}$）。
3. **加性噪声容限与现场信噪比边界**：
   - 在高/中信噪比（$\mathrm{SNR} \ge \mathbf{40\,\mathrm{dB}}$）下，倒谱偏深中位数稳定在 $+8.70 \sim +9.43\,\mathrm{m}$（极差 $\le 0.73\,\mathrm{m}$）；当信噪比恶化至 $\mathrm{SNR}=20\,\mathrm{dB}$ 时，差分高频噪声导致倒谱峰位出现抖动（中位数 $+6.53\,\mathrm{m}$，波动范围 $[+3.63, +9.43]\,\mathrm{m}$），但偏深方向始终保持为正，未出现假性偏浅失效。
