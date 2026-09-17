# steady 模式：B_coh / N_harm,eff / Δd_min **正推**

> 数据：`E:\water_hammer_research\TSNet-water-master\wellbore_moc_method\output\leakoff\steady_D50\single\moc_timeseries.csv`  
> **不含**用 dual 匹配间距反推 N_harm。

## 1. 瑞利判据（定义）

倒谱峰半高全宽由对数谱中相干谐波梳的有效带宽决定：

$$
\mathrm{FWHM}_\tau \approx \frac{1}{B_{\mathrm{coh}}}
$$

$$
B_{\mathrm{coh}} = N_{\mathrm{harm,eff}} \cdot f_0,
\qquad f_0 = \frac{a}{4L}
$$

深度域 $d=q\cdot a/2$：

$$
\Delta d_{\min}
= \frac{a}{2\,B_{\mathrm{coh}}}
= \frac{2L}{N_{\mathrm{harm,eff}}}
= \mathrm{FWHM}_d
$$

物理含义：两缝 quefrency 间距 $\Delta\tau=2\Delta d/a$ 须达到约一个 FWHM，才能在倒谱上分开。

## 2. B_coh 的正推操作定义

steady + `velocity_step` 下，停泵后去均值井口水头的幅值谱为 $|S(f)|$。  
取动态范围门限（相对峰值）：

$$
\varepsilon = 10^{-\mathrm{DR}/20}
\quad(\text{默认 DR}=80\ \mathrm{dB}\Rightarrow\varepsilon=10^{-4})
$$

从低频起找 $|S(f)|\ge\varepsilon\,\max|S|$ 的连通支撑（允许短于 $0.5 f_0$ 的空洞），右端频率定义为：

$$
B_{\mathrm{coh}} := f_{\mathrm{support,max}},
\qquad
N_{\mathrm{harm,eff}} := B_{\mathrm{coh}}/f_0
$$

说明：

- 关断历时 ≈ $dt$（毫秒级），**不能**再用 $1/t_s=1\,\mathrm{Hz}$ 当作 $B_{\mathrm{coh}}$ 上限。
- DR=80 dB 是谱可用动态范围约定（峰值以下 80 dB 视为跌出相干谐波梳），不是由匹配矩阵标定。

## 3. 本算例几何

| 量 | 数值 |
|----|------|
| L | 5000 m |
| a_adj | 1450.1160 m/s |
| f0 = a/(4L) | 0.07251 Hz |
| T_1d | 99.0 s |
| 关断 | velocity_step，≈1.0 ms |

## 4. 正推结果（DR = 80 dB）

| 量 | 数值 |
|----|------|
| ε | 1.0e-04 |
| **B_coh** | **18.899 Hz** |
| **N_harm,eff** | **260.7** |
| FWHM_τ = 1/B_coh | 0.05291 s |
| FWHM_d = a/(2 B_coh) | 38.36 m |
| **Δd_min** | **38.36 m** |
| Δd_min = 2L/N | 38.36 m |

代入验算：

$$
N_{\mathrm{harm,eff}} = \frac{B_{\mathrm{coh}}}{f_0}
= \frac{18.899}{0.07251}
\approx 261
$$

$$
\Delta d_{\min} = \frac{2L}{N}
= \frac{10000}{260.7}
\approx 38.4\ \mathrm{m}
$$

## 5. DR 门限敏感性（正推族）

| DR [dB] | ε | B_coh [Hz] | N_harm,eff | Δd_min [m] |
|---------|---|------------|------------|------------|
| 60 | 1.0e-03 | 3.30 | 45.6 | 219.5 |
| 70 | 3.2e-04 | 6.70 | 92.4 | 108.3 |
| 80 | 1.0e-04 | 18.90 | 260.7 | 38.4 |
| 90 | 3.2e-05 | 56.75 | 782.7 | 12.8 |
| 100 | 1.0e-05 | 173.47 | 2392.6 | 4.2 |

## 6. 与匹配实验的关系

匹配实验（`steady_Dall`）用于**验证**正推下界，不参与本计算。  
若正推 Δd_min≈38 m，则预期 Δd≳该值的 dual 可分、明显更小则不可分。
