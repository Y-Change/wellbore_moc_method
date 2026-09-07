# 02 时域波形退化_峰漂移与展宽模块

## 模块定位
本模块负责时域三时钟（物理起跳时钟 $t_{\text{onset}}$、峰值时钟 $t_{\text{peak}}$、能量中位时钟 $t_{E50}$）的精确提取，量化非稳态摩阻在时域中对井口反射波形造成的脉冲衰减、峰后移（Peak Drift）、能量弥散（EST 展宽）、有限关井时间 $T_c$ 影响，以及网格收敛性与噪声鲁棒性分析。

## 目录结构
- `./code/`:
  * `audit_waveform_degradation.py`: 时域退化指标审计主脚本；
  * `compute_four_clocks.py`: 四种时钟与能量重心抽取脚本 (p1)；
  * `evaluate_p2_onset_verdict.py`: Onset 时钟校正判决计算脚本 (p2)；
  * `run_tc_sweep.py`: 有限关井时间 $T_c$ 扫描仿真与时钟提取脚本 (p3)；
  * `run_p6_robustness_analysis.py`: 网格、窗、输入通道与加性噪声稳健性综合分析脚本 (p6)。
- `./data/`:
  * `four_clocks_n1_vs_n4.csv`: 单缝 vs 四缝四时钟对比表；
  * `metrics_v2.csv`: 时域审计指标总表 (EST, 峰漂移, Cv)；
  * `onset_correction_verdict.csv`: p2 Onset 绝对误差判决全表；
  * `Tc_sweep_n1_k0.01.csv`: p3 关井时间扫描时钟响应表；
  * `p5_model_comparison_clocks.csv`: p5 模型等级三角对照四时钟全表；
  * `p6_grid_sensitivity.csv`: p6 网格收敛性检验表 ($\Delta t=0.5\,\mathrm{ms}$ vs $1.0\,\mathrm{ms}$)；
  * `p6_noise_robustness.csv`: p6 加性高斯白噪声鲁棒性全表 ($\mathrm{SNR}=20,40,60\,\mathrm{dB}$，5 随机种子)；
  * `README_Gantt_p1_clocks_audit.md`: p1 审计定论笔记；
  * `README_Gantt_p2_onset_verdict.md`: p2 Onset 判决定论笔记；
  * `README_Gantt_p3_Tc_sweep.md`: p3 关井时间定论笔记；
  * `README_Gantt_p6_robustness.md`: p6 数值稳健性与有效工作区间定论笔记。
- `./figures/`:
  * `fig1_waveform_evolution.png`: 时域波形演化图；
  * `fig2_peakshift_est.png`: 峰漂移与 EST 展宽图；
  * `fig6_cv_heatmap.png`: 双峰对比度 Cv 热力图；
  * `fig_p6_robustness_diagnostic.png`: 符合 SPEJ 规范的网格/窗/通道/噪声综合稳健性诊断图件；
  * `fig_p6_robustness_diagnostic.svg`: 矢量图件。

## 核心审定数据与稳健性定论 (p6)
1. **网格收敛性**：
   - $\Delta t=0.5\,\mathrm{ms}$ 网格下 $\delta x_{\text{peak}} = \mathbf{+10.15\,\mathrm{m}}$ 与 $1.0\,\mathrm{ms}$ 完全相同，$\delta x_{\text{cep}} = \mathbf{+9.79\,\mathrm{m}}$ 与 $1.0\,\mathrm{ms}$ 基准（$+9.43\,\mathrm{m}$）差异仅 $0.36\,\mathrm{m}$（$<1$ 个离散网格步），网格收敛性完全成立。
2. **输入通道与窗长有效区间**：
   - 压力水头 $H(t)$、导数 $\mathrm{d}H/\mathrm{d}t$ 及带通 $H$ 三通道提取的 $\delta x_{\text{cep}}$ **严格恒为 $+9.43\,\mathrm{m}$**；
   - 倒谱窗长有效工作区间为 $T_{\text{win}} \ge \mathbf{30\,\mathrm{s}}$（$\delta x_{\text{cep}}$ 稳定聚集于 $+9.43 \sim +10.15\,\mathrm{m}$）。
3. **加性白噪声容限**：
   - $\mathrm{SNR} \ge \mathbf{40\,\mathrm{dB}}$ 下中位数稳定在 $+8.70 \sim +9.43\,\mathrm{m}$（极差 $\le 0.73\,\mathrm{m}$）；$\mathrm{SNR}=20\,\mathrm{dB}$ 强噪下中位数为 $+6.53\,\mathrm{m}$，偏深方向恒正。
