# 03 频域高频耗散_STFT与表观波速模块

## 模块定位
从傅里叶变换、短时傅里叶变换（STFT）以及表观速度（Apparent Wave Speed）的分裂现象，深入揭示 Brunone 摩阻引起的“频率选择性耗散（Frequency-Selective Dissipation）”与波速表观畸变。

## 核心物理现象与机理
1. **频率选择性耗散（Frequency-Selective Dissipation）**：
   - Brunone 阻尼对高频波动成分具有极强的抑制能力。
   - 在 STFT（Hann 窗，$n_{\text{perseg}} = 128$，重叠 120，分析段 $[6.4, 7.2]\text{ s}$）下对比同间距 $k=0$ 基准：
     * **低频段（0–20 Hz）**：相对衰减约为 $-12.9\text{ dB}$（保留较多主干水锤能量）；
     * **中频段（20–60 Hz）**：相对衰减达到 $-25.4\text{ dB}$；
     * **高频段（60–150 Hz）**：相对衰减高达 **$-44.1\text{ dB}$**（高频锐脉冲能量近乎灭绝）。
   - 该现象直接导致倒谱变换赖以解析细微时延的高频微结构信息被强行“低通滤波”。

2. **表观速度（Apparent Wave Speed）分离**：
   - 随 $k$ 增大，由于波前、峰值与全域驻波模式遭受不同程度的耗散畸变，传统单一标量波速 $a = 1450\text{ m/s}$ 发生明显表观解耦：
     * **基频速度 $a_{f0} = 4L f_0$**：由水锤基频谐波测定，从 $1409\text{ m/s}$ 降至 $1254\text{ m/s}$（降幅 **11.0%**）；
     * **起跳速度 $a_{\text{onset}} = \frac{2 X_1}{t_{\text{onset}} - t_s}$**：由波前起跳微小扰动测定，仅从 $1451\text{ m/s}$ 降至 $1432\text{ m/s}$（降幅 **1.3%**，物理首波前受耗散影响较小）；
     * **峰值速度 $a_{\text{peak}} = \frac{2 X_1}{t_{\text{peak}} - t_s}$**：由波峰最大值测定，从 $1450\text{ m/s}$ 降至 $1383\text{ m/s}$（降幅 **4.7%**）。
   - **科学定论**：这一速度分离直接证明了传统使用单一标量波速将倒谱时延线性映射为物理深度的做法存在根本性物理失配。

## 关键文件与图件
- `频带能量表/band_energy.csv`：各工况在 0–20 Hz、20–60 Hz、60–150 Hz 频带的能量积分与相对稳态的 dB 衰减值。
- `图/Figure_3_1_STFT_Spectrogram_Comparison.png`：稳态 vs Brunone 的高分辨率 STFT 时频衰减对比图。
- `图/Figure_3_2_Frequency_Band_Energy_Dissipation.png`：三大频带能量衰减随 $k$ 的阶梯柱状对比图。
- `图/Figure_3_3_Apparent_Speed_Discrepancy.png`：$a_{f0}, a_{\text{onset}}, a_{\text{peak}}$ 三种表观速度随 $k$ 的分化曲线。
