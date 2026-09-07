# 01 非定常摩阻正演与基准模块

## 模块定位
负责井筒水击与裂缝耦合系统在稳态 Darcy、Brunone 瞬时加速度 (IAB) 与 Vardy–Brown 卷积加权 (WFB) 非稳态摩阻条件下的 MOC 正演数值模拟计算，为后续各分析模块提供基准时间序列。

## 目录结构
- `./code/`:
  * `run_simulations.py`: 四缝多间距矩阵 A 仿真脚本（`velocity_step` 阶跃关井）；
  * `run_n1_simulations.py`: 单缝机制核 $n=1$ 仿真脚本（`velocity_step` 阶跃关井，常数 $k \in \{0, 0.01, 0.02, 0.05\}$）；
  * `run_tc_sweep.py`: 有限关井时间 $T_c$ 扫描仿真脚本（`ramp` 线性斜坡关井，$T_c \in \{1, 50, 200, 1000\}\,\mathrm{ms}$）；
  * `run_p5_iab_wfb_comparison.py`: Darcy vs IAB vs WFB 模型三角对照仿真与时钟提取脚本。
- `./data/`:
  * `cases_manifest.csv`: 正演工况定义与参数配置清单；
  * `p5_model_comparison_clocks.csv`: 模型等级三角对照四时钟与阻尼比全表；
  * `README_Gantt_p5_IAB_WFB.md`: Gantt p5 模型等级三角对照审定定论笔记（审定定稿版）。
- `./figures/`:
  * `fig1_waveform_evolution.png`: 基础正演水击波形时程与衰减基准对比图；
  * `fig_p5_iab_wfb_comparison.png`: 符合 SPEJ 规范的模型等级三角对照波形与时钟误差对比图件；
  * `fig_p5_iab_wfb_comparison.svg`: 矢量格式图件。

## 模型等级三角对照定论 (Gantt p5 审定定稿)
1. **时钟分叉与偏深在 IAB 上显著，在 1D WFB 上不出现**：
   - 阶跃关井下 Brunone IAB ($k=0.01$) 的相对偏深 $\delta x_{\text{cep}} = \mathbf{+9.43\,\mathrm{m}}$ 来源于全截面瞬时加速度对流耗散假说，是理论上限；纯 1D WFB 粘性壁面扩散下 $\delta x_{\text{cep}} = \mathbf{0.00\,\mathrm{m}}$。
2. **$+9.43\,\mathrm{m}$ 严格限定为“IAB、$k=0.01$、阶跃关井”理论上界**：
   - 主文倒谱相对偏移量统一使用 $+9.43\,\mathrm{m}$，不作为各类流态的恒定绝对值。
3. **$[0, 9.4]\,\mathrm{m}$ 为水动力学模型等级间的理论差异范围，不得称为现场测量误差带**：
   - 纯 1D WFB 平滑管壁给出理论下界（$\approx 0\,\mathrm{m}$），IAB 给出理论上界（$\approx 9.4\,\mathrm{m}$）；现场复杂井筒由于存在接箍突变与 2D/3D 紊流畸变，其实际偏深位于该理论区间内。
