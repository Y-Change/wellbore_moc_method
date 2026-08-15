# 02 时域波形退化_峰漂移与展宽模块

## 模块定位
量化非定常摩阻在时域中对井口反射波形造成的脉冲衰减、峰后移（Peak Drift）、波包展宽（EST 能量弥散）以及多裂缝反射双峰重叠融合现象。

## 核心物理现象与机制
1. **幅值剧烈衰减**：
   - 随 $k$ 增大，波前回声幅值发生断崖式跌落。在 $D = 20\text{ m}, k = 0.2$ 锚点下，首反射峰幅值相比稳态（$k=0$）下降超过 50 倍。
   - 物理机理：Brunone 项正比于瞬态流速时空梯度 $\partial_t V + a \text{ sign}(V) |\partial_x V|$，高梯度的阶跃波前遭受极强的局部黏滞耗散。

2. **峰值到时后移（Peak Drift / Time Delay）**：
   - 强烈的非定常阻尼削平了原始波前的尖锐突变，使波形最大值（Peak）向后严重滞后。在 $D = 20\text{ m}, k = 0.2$ 下，峰漂移 $\Delta t_{\text{peak}} \approx 277\text{ ms}$。
   - 若直接将此峰值时延按常规波速折算深度 $\Delta x = \frac{a \Delta t_{\text{peak}}}{2}$，将导致超过 $200\text{ m}$ 的表观视深度偏移（这是倒谱产生系统性深度偏差的直接时域根源）。

3. **能量弥散与波包展宽（Energy Spread Time, EST）**：
   - 传统半高宽（FWHM）在强退化下会因主峰矮化和平台化产生虚假变窄，本模块采用能量累积积分 $E_{10} \sim E_{90}$ 的时间跨度 EST 作为客观展宽指标。
   - 在锚点工况下，EST 从 $k=0$ 时的 $1.0\text{ ms}$ 剧烈膨胀至 $101.0\text{ ms}$（增量达 $100.0\text{ ms}$）。

4. **双峰融合与干涉（Double-Peak Merging）**：
   - 采用峰谷对比度指标 $C_v = \frac{P_{\min} - V_{\text{trough}}}{P_{\min}}$ 评估相邻裂缝回声的可分离性。
   - 随 $k$ 增大与间距 $D$ 缩小，相邻反射波包互相叠合，$C_v \to 0$，离散峰融合为单峰。

## 关键文件与图件
- `指标表/metrics_v2.csv`：包含 30 组工况的 $\Delta t_{\text{peak}}$、$\text{EST}$、$\Delta\text{EST}$、$C_v$ 及有效波峰数统计。
- `图/Figure_2_1_Waveform_Evolution.png`：不同 $k$ 下 $\mathrm{d}H/\mathrm{d}t$ 的时程退化演化图。
- `图/Figure_2_2_PeakShift_and_EST_Trend.png`：峰漂移与 EST 随 $k$ 及间距 $D$ 的双轴演变趋势。
- `图/Figure_2_3_Cv_Heatmap.png`：双峰对比度 $C_v$ 在 $(k, D)$ 空间中的退化热力图。
