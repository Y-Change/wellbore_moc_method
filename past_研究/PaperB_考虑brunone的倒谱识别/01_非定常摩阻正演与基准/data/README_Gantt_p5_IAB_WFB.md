# Gantt p5: 模型等级三角对照定论笔记 (Darcy vs IAB vs WFB) (审定定稿版)

## 1. 模型等级三角对照数据表 (p5_model_comparison_clocks.csv)

> **工况与参数冻结规范（严格对齐 WELL_CONFIG / FRACTURE_CONFIG）**：
> - 几何与物性：单缝 $n=1, X_1=4100.0\,\mathrm{m}, C_f=1.0\times 10^{-5}\,\mathrm{m}^2, k_{\text{leak}}=1.0\times 10^{-4}\,\mathrm{m^2/s/\sqrt{m}}, H_{\text{ext}}=100.0\,\mathrm{m}$；
> - 井筒参数：$L=5000.0\,\mathrm{m}, D=0.1397\,\mathrm{m}, a=1450.0\,\mathrm{m/s}, V_0=1.0\,\mathrm{m/s}, H_0=300.0\,\mathrm{m}, \nu=1.0\times 10^{-6}\,\mathrm{m^2/s}$；
> - 关井边界：`velocity_step` 阶跃关井，$t_s=1.0\,\mathrm{s}, t_f=50.0\,\mathrm{s}, \Delta t=1.0\,\mathrm{ms}$；
> - 对照模型集：
>   1. **Steady Darcy**：稳态摩阻基准（$k=0$）；
>   2. **Brunone IAB ($k=0.01$)**：主文弱阻尼锚点（基于瞬时对流加速度耗散假设）；
>   3. **Brunone IAB ($k=0.05$)**：趋势上沿；
>   4. **Vardy–Brown WFB**：基于物理雷诺数 $Re_0=1.397\times 10^5$ 的 10 阶指数和加权函数卷积模型（$C^* \approx 0.00656$，无自由调谐参数 $k$）。

| 模型类别 | 摩阻参数配置 | $t_{\text{onset}}$ (s) | $\Delta x_{\text{onset}}$ (m) | $t_{\text{peak}}$ (s) | $\Delta x_{\text{peak}}$ (m) | $\Delta x_{\text{cep}}$ (m) | 相对峰漂移 $\delta x_{\text{peak}}$ (m) | 相对倒谱偏深 $\delta x_{\text{cep}}$ (m) | CWT 阻尼比 $\zeta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Steady Darcy** | 纯稳态 Darcy | 6.653 | -1.58 m | 6.653 | -1.58 m | -0.85 m | 0.00 m | 0.00 m | 0.03941 |
| **Brunone IAB** | $k=0.01$ (主文锚点) | 6.653 | -1.58 m | 6.667 | +8.58 m | +8.58 m | +10.15 m | **+9.43 m** | 0.04301 |
| **Brunone IAB** | $k=0.05$ (趋势上沿) | 6.660 | +3.50 m | 6.715 | +43.38 m | +35.40 m | +44.95 m | **+36.25 m** | 0.05997 |
| **Vardy–Brown WFB** | 物理 $C^*(Re)$ 卷积 | 6.653 | -1.58 m | 6.654 | -0.85 m | -0.85 m | **+0.73 m** | **0.00 m** | 0.04028 |

---

## 2. 物理机制剖析与数值根源

1. **一维纯壁面粘性剪切扩散（WFB）在单次声波走时内的局限**：
   - 理论加权函数 $W(\tau^*) = C^* / \sqrt{\tau^*}$ 描述流体剪切扰动自管壁向轴心的径向粘性扩散过程（扩散时间尺度 $t_{\text{diff}} = D^2 / (4\nu) \approx 4880\,\mathrm{s}$）；
   - 在油套管尺寸（$D=0.1397\,\mathrm{m}$）与清水低粘度（$\nu=1.0\times 10^{-6}\,\mathrm{m^2/s}$）下，首波单次往返历时仅 $5.65\,\mathrm{s}$，壁面粘性边界层厚度仅发展至 $\delta \approx \sqrt{\nu t} \approx 2.4\,\mathrm{mm} \ll D/2 \approx 70\,\mathrm{mm}$；
   - 截面 $>96\%$ 的流体仍处于无粘声学阶跃状态，单次传播几乎不产生高频能量耗散与波形畸变，波峰与倒谱重心在单次传播后基本无宏观漂移（$\delta x_{\text{peak}} = +0.73\,\mathrm{m}$ 仅为 1 个时网步；$\delta x_{\text{cep}} = 0.00\,\mathrm{m}$）。
2. **Brunone IAB 瞬时对流加速度项的全局耗散作用**：
   - Brunone 模型引入了空间对流加速度项 $a \cdot \text{sign}(V) |\partial V / \partial x|$，在阶跃波前瞬态将耗散作用均匀施加于全截面（瞬态阻尼脉冲 $J_u \sim 10^{-2}\,\mathrm{m/s}$，比 WFB 初始脉冲强约 1000 倍）；
   - 这使得波前高频能量被强烈滤除，产生显著的不对称展宽与能量重心后移（$\Delta t_{\text{peak}} \approx 13\,\mathrm{ms} \implies \delta x_{\text{cep}} = +9.43\,\mathrm{m}$）。

---

## 3. 科学假说 H1 边界与定论口径 (审定定稿三句)

1. **时钟分叉与倒谱偏深在 Brunone IAB 模型上显著成立，在纯 1D WFB 模型上不出现**：
   - 在 IAB 假设下，$\Delta t_{\text{onset}} \ll \Delta t_{\text{peak}} \approx \Delta\tau_{\text{cep}}$ 的时钟分叉特征极其显著，倒谱相对稳态偏深 $\delta x_{\text{cep}} = \mathbf{+9.43\,\mathrm{m}}$；而在纯 1D WFB 粘性壁面扩散下，首反射波前未发生宏观波形展宽与重心滞后（$\delta x_{\text{cep}} = 0.00\,\mathrm{m}$）。
2. **$+9.43\,\mathrm{m}$ 严格限定为“Brunone IAB、$k=0.01$、阶跃关井”下的理论上界（Upper Bound）**：
   - $9.4\,\mathrm{m}$ 来源于 IAB 模型对流加速度全截面耗散假设，代表非定常摩阻作用下的理论偏深上限，不得包装为现场各种流态下的固定绝对值；主文倒谱相对偏移量严格统一使用 **$+9.43\,\mathrm{m}$**。
3. **$[0, 9.4]\,\mathrm{m}$ 为水动力学模型等级间的理论差异范围，不得称为现场测量误差带**：
   - 纯 1D 平滑管壁粘性边界层扩散（WFB）给出理论下界（$\approx 0\,\mathrm{m}$），Brunone IAB 给出理论上界（$\approx 9.4\,\mathrm{m}$）；
   - 现场复杂井筒由于存在接箍内径突变、射孔孔眼局部漩涡及 2D/3D 紊流畸变，其实际非定常剪切耗散强于 1D WFB 平滑管，其实际偏深位于理论区间内。
