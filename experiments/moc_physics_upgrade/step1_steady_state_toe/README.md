# Step 1 物理升级报告：初始稳态流场自洽与封闭趾端无激波

## 1. 升级背景与解决的核心问题

在原始 MOC 仿真（`moc_simulate/wellbore_moc.py`）中，存在严重的初始稳态与边界不自洽问题：
1. **流速全井常数与趾端突变断裂**：原版将整条 5000 米井筒统一设为 $V(x) = V_0 = 1.0\,\mathrm{m/s}$，直到最末网格点强行令 $V[-1] = 0$。在流体力学上，这在 $t=0$ 仿真伊始就在趾端激发起一个非物理的 $+147.8\,\mathrm{m}$ 水击假激波，并在全井反复震荡，导致停泵前井口水头漂移至 $347.8\,\mathrm{m}$。
2. **缺失稳态沿程达西摩阻坡降**：原版将全井水头初始设为平直常数 $H_0 = 300\,\mathrm{m}$，违反了稳态动量平衡 $\frac{\partial H}{\partial x} = -\frac{f V |V|}{2 g D}$。
3. **忽略多簇流量分流守恒**：实际水平井中注入流量全部进入各射孔簇（$\sum q_{j,0} = Q_{pump}$），最末簇下游至封闭趾端桥塞之间为死水区（$V = 0$）。

## 2. 数学模型与修复方案

在独立子文件夹 `experiments/moc_physics_upgrade/step1_steady_state_toe/` 下构建了全新的升级求解器 `solver.py`（**原有 `moc_simulate` 代码保持 100% 完好未改**）：

1. **流量分流连续性方程**：
   $$V(x) = V_0 - \frac{1}{A} \sum_{x_j \le x} q_{j,0}$$
   最末压裂簇之后（$x \ge x_{N_c}$），$V(x) = 0.0$。封闭趾端桥塞处自然满足 $V[-1] = 0$，无速度间断。
2. **沿程达西摩阻稳态水头场积分**：
   $$H_0(x) = H_0(0) - \int_0^x \frac{f(s) V(s) |V(s)|}{2 g D} ds$$
   在最末簇与趾端间的死水段，$\frac{dH}{dx} = 0$，$H(x) = \text{const}$。
3. **稳态滤失动平衡自洽**：
   校准稳态缝内滤失 $k_{leak,eff} \sqrt{H_{f,ss} - H_{ext}} = q_{j,0}$，使时间推进在 $t < t_s$ 期间 $\frac{\partial H}{\partial t} = 0$，$\frac{\partial V}{\partial t} = 0$ 严格恒等于零。

## 3. 验证工况与结果 (PaperA brunone_D20/quad)

- 裂缝形态：`quad`（四裂缝，深度 $4100, 4120, 4140, 4160\,\mathrm{m}$，间距 $D = 20\,\mathrm{m}$）
- 摩阻模型：Brunone 非定常摩阻
- 仿真时长：$100.0\,\mathrm{s}$（$10$ 万步）

### 关键量化指标对比

| 指标 | PaperA 原版 | Step 1 升级版 | 提升效果 |
| :--- | :---: | :---: | :--- |
| **关泵前稳态水头平均值** | $347.8557\,\mathrm{m}$ | **$300.0000\,\mathrm{m}$** | 完美贴合设定基准 $300\,\mathrm{m}$ |
| **关泵前水头波动标准差** | $\sim 25\,\mathrm{m}$ (剧烈震荡) | **$4.32 \times 10^{-12}\,\mathrm{m}$** | **达到双精度浮点零波动** |
| **Joukowsky 理论跳变误差** | $6.34 \times 10^{-13}\,\%$ | **$4.99 \times 10^{-13}\,\%$** | 严格动量守恒 |
| **1D 倒谱检出峰深误差** | $\sim 9.7\,\mathrm{m}$ | **$1.10\,\mathrm{m}$** | 定位精度大幅提高 |

## 4. 交付文件清单 (`output/moc_physics_upgrade/step1_steady_state_toe/quad/`)

1. `moc_timeseries.csv`: 完整 $100\,\mathrm{s}$ 时程数据（含井口与各缝水头、流量）；
2. `moc_leakoff.png`: 标准 2×2 四联物理验证图；
3. `cepstrum_standard.png`: 5 联标准倒谱图；
4. `cepstrum_fracture_zoom.png`: 裂缝区 3 联高分辨率特写；
5. `comparison_step1_vs_original.png`: Step 1 vs 原版叠合对比大图；
6. `moc_leakoff.json`: 结构化评测指标。
