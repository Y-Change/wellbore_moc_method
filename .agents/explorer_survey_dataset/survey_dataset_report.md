# 多裂缝井筒 MOC 数据集规范、参数空间与已有实验脚本深度调研报告

> **调研执行节点**：`teamwork_preview_explorer` (Dataset & Parameter Space Specialist)  
> **归档路径**：`e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_dataset\survey_dataset_report.md`  
> **调研目标**：支撑任务目标 —— 系统化开展新版 MOC 求解器下裂缝物理参数对井筒水锤瞬态波形与声学倒谱特征的敏感性消融实验 (`output/fracture_parameter_sensitivity`)，确立标准数据契约、参数取值网格与实验设计范式。

---

## 1. 调研执行摘要 (Executive Summary)

依据 `ORIGINAL_REQUEST.md` 的四项核心需求（R1 单变量消融与正交仿真、R2 双摩阻稳态与 Brunone 对照、R3 时频-倒谱多维特征矩阵构建、R4 学术级综合研究报告与图版交付），本调研全面遍历了代码库中 `moc_simulate/`、`experiments/`、`analysis/`、`PaperA...`、`PaperB...`、`PaperC...` 及 `output/` 现存的大量脚本与数据集。

**核心调研结论**：
1. **统一数据标准已确立**：项目已冻结 `moc_lhs_v2.1` 标准 schema，其单样本 `.npz` 格式包含 **41 项完备元数据键值**，支持网格/波速自适应修正、旧接口别名兼容（如 `Cf` 与 `compliance_head_m2`、`kleak` 与 `kleak_equiv`）及稳态双射闭合验证。标准时程 CSV 文件名为 `moc_timeseries.csv`（列结构为 `t, H_wh, Q_wh, H_f1, Q_f1, ...`）。
2. **物理基准与取值范围已系统标定**：基准井长 $L=5000\,\mathrm{m}$，内径 $D=0.1397\,\mathrm{m}$，水头 $H_0=300\,\mathrm{m}$，流速 $V_0=1.0\,\mathrm{m/s}$，波速 $a=1450\,\mathrm{m/s}$。裂缝敏感性区间：位置 $x_f \in [3500, 4800]\,\mathrm{m}$（基准 $4100\,\mathrm{m}$），间距 $\Delta x \in [5, 20]\,\mathrm{m}$（离散预设 $[5, 10, 20, 50, 100]\,\mathrm{m}$），水头柔度 $C_H \in [10^{-6}, 10^{-4}]\,\mathrm{m}^2$（对数均匀，基准 $10^{-5}\,\mathrm{m}^2$），滤失 $k_\text{leak} \in [10^{-6}, 10^{-3}]\,\mathrm{m}^{5/2}/\mathrm{s}$（基准 $10^{-4}$），分流权重 $w_i \sim \mathrm{Dirichlet}(\alpha)$（$\alpha \in \{0.3, 1.0, 3.0, 10.0\}$），射孔阻抗 $R_p \in [0, 10^5]\,\mathrm{s^2/m^5}$（基准 $0.0$）。
3. **三种成熟实验设计范式**：
   - **单变量消融 (OAT)**：固定背景锚点，逐维扫描（如 $t_c$ 扫描、间距扫描、柔度单变量扫描）；
   - **小规模全正交网格 (Orthogonal Grid)**：笛卡尔积正交采样（如 PaperB 采用的 $k \times k_\text{leak}$ 16 点正交阻尼混淆面、$D \times k$ 30 工况矩阵、decay 框架下的 $C_f \times k_\text{leak}$ 72 点全网格）；
   - **多维拉丁超立方 (LHS)**：基于 `scipy.stats.qmc.LatinHypercube`，内嵌间距非重叠链长约束与 Dirichlet 逆 Gamma 累计分布映射。
4. **统一落盘架构规范**：标准实验输出应组织为 `data/`（`case_XXXXX.npz`）、`timeseries_csv/`、`figures/`、`tables/`（`sensitivity_metrics.csv` 及 JSON）和顶层 `README.md`。

---

## 2. 标准数据集 Schema 契约分析 (`moc_lhs_v2.1` & CSV)

### 2.1 NPZ 数据集结构 (`case_XXXXX.npz`)

在 `moc_simulate/run_lhs_batch_simulate.py`、`moc_simulate/build_reservoir_dataset.py` 及 `tests/test_final_acceptance.py` 中，标准数据结构被统一定义为 `schema_version = "moc_lhs_v2.1"`。经对现场生成的 `output/acceptance_10_test/data/case_00000.npz` 执行实际加载分析，单文件内包含完整的 **41 个字段**，具体定义如下：

| 分类 | 字段名 (Key) | 形状 (Shape) | 数据类型 (dtype) | 物理含义与单位 / 语义说明 |
|:---|:---|:---|:---|:---|
| **版本与索引** | `schema_version` | `()` | `<U12` | 模式版本号，固定为 `"moc_lhs_v2.1"` |
| | `case_id` | `()` | `int32` | 工况连续整数 ID（如 0, 1, 2...） |
| | `seed` | `()` | `int32` | 仿真生成时的随机数种子，保障 100% 可复现 |
| **网格与时间** | `N` | `()` | `int32` | 井筒空间剖分离散段数（$N = \operatorname{round}(L / (a \Delta t))$） |
| | `dx` | `()` | `float64` | 空间网格步长 [m]（$\Delta x = L / N$） |
| | `dt` | `()` | `float64` | 实际仿真时间步长 [s] |
| | `dt_requested` | `()` | `float64` | 用户请求的时间步长 [s]（默认 $1.0\times 10^{-3}\,\mathrm{s}$） |
| | `dt_adj` | `()` | `float64` | 微调后的实际时间步长 [s]（保留请求步长） |
| | `tf` | `()` | `float64` | 仿真总历时截断时间 [s]（如 30.0s 或 50.0s） |
| | `t` | `(M,)` | `float64` | 时间步戳序列 [s]（$M = \operatorname{round}(t_f/\Delta t) + 1$） |
| **几何与声学** | `wellbore_length` | `()` | `float64` | 井筒总测深 $L$ [m]（默认 5000.0 m） |
| | `wavespeed` | `()` | `float64` | 实际数值波速 [m/s]（对齐 $\Delta x / \Delta t$） |
| | `wavespeed_requested` | `()` | `float64` | 请求标称波速 [m/s]（默认 1450.0 m/s） |
| | `wavespeed_adj` | `()` | `float64` | 调整后的实际波速 [m/s]（严格保障 Courant 数 $Cr=1$） |
| | `wavespeed_nominal` | `()` | `float64` | 标称波速备份 [m/s] |
| **工况与背景** | `friction` | `()` | `<U7` | 摩阻模式：`"steady"` 或 `"brunone"` |
| | `brunone_k_scale` | `()` | `float64` | Brunone 系数修正倍率（默认 1.0） |
| | `toe_bc` | `()` | `<U8` | 趾端边界：`"dead_end"` (封闭) 或 `"reservoir"` (水库) |
| | `initial_head` | `()` | `float64` | 井筒初始静水头 $H_0$ [m]（默认 300.0 m） |
| | `initial_velocity` | `()` | `float64` | 停泵前初始流速 $V_0$ [m/s]（默认 1.0 m/s） |
| | `H_ext` | `()` | `float64` | 地层孔隙压力参考水头 [m]（默认 100.0 m） |
| **观测特征 (波形)** | `H_wh` | `(M,)` | `float64` | 井口测压管水头时程 $H_{wh}(t)$ [m]（含可选噪声） |
| | `Q_wh` | `(M,)` | `float64` | 井口体积流量时程 $Q_{wh}(t)$ [m³/s] |
| **裂缝物理参数** | `n_frac` | `()` | `int32` | 该工况实际裂缝条数 $N_{cl}$ |
| | `x_f_requested` | `(N_cl,)` | `float64` | 采样生成的裂缝理论位置 [m] |
| | `x_f_aligned` | `(N_cl,)` | `float64` | 对齐到 MOC 网格节点的实际裂缝位置 [m] |
| | `fracture_indices` | `(N_cl,)` | `int32` | 裂缝在井筒网格上的离散节点编号 $i_f$ |
| | `compliance_head_m2` | `(N_cl,)` | `float64` | 裂缝水头柔度 $C_H = \rho g C_p$ [m²] |
| | `inflow_weight` | `(N_cl,)` | `float64` | 稳态进液分流权重 $w_i$（$\sum w_i = 1.0$） |
| | `kleak_equiv` | `(N_cl,)` | `float64` | 稳态反求的等效滤失系数 $k_i^\text{equiv}$ [m$^{5/2}$/s] |
| | `Rp` | `(N_cl,)` | `float64` | 射孔阻抗系数 $R_p$ [s²/m⁵]（基准为 0.0） |
| | `alpha_dirichlet` | `()` | `float64` | 采样所采用的 Dirichlet 浓度参数 $\alpha$ |
| **稳态双射闭合** | `Hw_ss` | `(N_cl,)` | `float64` | 裂缝外侧（井筒节点）稳态水头 [m] |
| | `Hf_ss` | `(N_cl,)` | `float64` | 裂缝内侧稳态水头 [m] |
| | `Qf_ss` | `(N_cl,)` | `float64` | 裂缝稳态进液流量 [m³/s] ($Q_{f,i}^{ss} = w_i Q_{in}$) |
| | `Qin_ss` | `()` | `float64` | 井口初始稳态总注入流量 [m³/s] ($A V_0$) |
| **兼容类别名 (Aliases)**| `x_f` | `(N_cl,)` | `float64` | 对齐别名，等于 `x_f_aligned` |
| | `x_f_raw` | `(N_cl,)` | `float64` | 原始别名，等于 `x_f_requested` |
| | `grid_index` | `(N_cl,)` | `int32` | 节点别名，等于 `fracture_indices` |
| | `Cf` | `(N_cl,)` | `float64` | 柔度别名，等于 `compliance_head_m2` |
| | `kleak` | `(N_cl,)` | `float64` | 滤失别名，等于 `kleak_equiv` |

### 2.2 时程 CSV 文件结构 (`moc_timeseries.csv`)

在全仓库的所有正演与深度分析中（`moc_simulate/leakoff_multi.py`、`PaperA`、`PaperB` 等），时程 CSV 格式完全统一：

1. **文件头 (Header)**：
   ```csv
   t,H_wh,Q_wh,H_f1,Q_f1,H_f2,Q_f2,...,H_fn,Q_fn
   ```
2. **各列定义**：
   - `t`：时间步戳 [s]，单调递增，步长 $\Delta t$（通常 1 ms）；
   - `H_wh`：井口测压管水头 [m]；
   - `Q_wh`：井口体积流量 [m³/s]；
   - `H_fi`：第 $i$ 条裂缝处内部水头 [m]；
   - `Q_fi`：第 $i$ 条裂缝的侧向汇流量 [m³/s]。
3. **加载契约**：在 `moc_simulate/leakoff_multi.py` 的 `load_timeseries_csv()` 中，代码自动检测 `H_f{k}` 与 `Q_f{k}` 列数来确定裂缝数 $n_\text{frac}$，并提取为矩阵数组。

### 2.3 元数据与索引文件结构 (`lhs_metadata.json` & `lhs_summary.csv`)

1. **`lhs_summary.csv` 字段**：
   ```csv
   case_id,status,n_frac,positions_str,Cf_str,wi_str,kleak_str,elapsed_s,npz_file,error,wavespeed,friction_model,snr_db
   ```
   其中各物理量以分号 `;` 拼接为字符串，方便快速在 Pandas 中过滤工况。
2. **`lhs_metadata.json` 键值**：
   - `schema_version`: `"moc_lhs_v2.1"`
   - `seed`: 采样基准种子
   - `param_ranges`: 记录该批次采样的上下界字典
   - `moc_config`: 固定的井筒背景参数字典
   - `sample_count`: `{"total": N, "pass": N_pass, "fail": N_fail}`
   - `generation_time`: 生成时间戳
   - `samples_index`: 各工况结果字典列表

---

## 3. 物理参数空间、基准值与分布特性

结合 `moc_simulate/config.py`、`moc_simulate/lhs_config.py` 以及各个专项研究，多裂缝水锤仿真的典型参数基准与采样分布汇总如下：

### 3.1 井筒与流体基础参数 (Wellbore & Fluid Background)

| 参数名称 | 符号 | 代码变量 | 基准值 (Baseline) | 扫描/域随机化区间 | 物理意义与工程背景 |
|:---|:---|:---|:---|:---|:---|
| 井筒总长 | $L$ | `L`, `wellbore_length` | $5000.0\,\mathrm{m}$ | 固定 (5000 m) 或 [3000, 5000] m | 页岩油气典型长水平井总测深 |
| 井筒内径 | $D$ | `wellbore_diameter` | $0.1397\,\mathrm{m}$ | $5.5''$ 套管 (标称内径约 0.1214~0.124m，代码固定 0.1397m) | 标准压裂生产管柱 |
| 流体密度 | $\rho$ | `fluid_density` | $1000.0\,\mathrm{kg/m^3}$ | [990, 1060] | 清水/滑溜水压裂液 |
| 运动黏度 | $\nu$ | `fluid_viscosity` | $1.0\times 10^{-6}\,\mathrm{m^2/s}$ | $[0.8\times 10^{-6}, 3.0\times 10^{-6}]$ | 常温水相流体黏度 |
| 水击波速 | $a$ | `wavespeed` | $1450.0\,\mathrm{m/s}$ | $[1350.0, 1550.0]\,\mathrm{m/s}$ | 钢套管与水综合弹性等效声速 |
| 管壁粗糙度 | $\varepsilon$ | `roughness_height` | $4.5\times 10^{-5}\,\mathrm{m}$ | $[1.5\times 10^{-5}, 1.0\times 10^{-4}]$ | 商业钢管工业级绝对粗糙度 |
| 初始注入流速 | $V_0$ | `initial_velocity`, `V0` | $1.0\,\mathrm{m/s}$ | $[0.5, 2.0]\,\mathrm{m/s}$ | 对应管截面排量 $Q_0 \approx 0.0153\,\mathrm{m^3/s} \approx 5.8\,\mathrm{bpm}$ |
| 初始水头 | $H_0$ | `initial_head`, `H0` | $300.0\,\mathrm{m}$ | $[200.0, 500.0]\,\mathrm{m}$ | 停泵前井口基准测压水头 ($\approx 2.94\,\mathrm{MPa}$) |
| 地层孔隙水头 | $H_{ext}$ | `H_ext` | $100.0\,\mathrm{m}$ | 固定 $100\,\mathrm{m}$ ($\approx 0.98\,\mathrm{MPa}$) | 裂缝外侧远场基底孔隙压力 |
| 井斜角 | $\theta$ | `theta` | $0.0\,\mathrm{rad}$ | 现行约定固定为 0（水平井） | 重力轴向分量为 0 |
| 停泵开始时刻 | $t_s$ | `ts`, `pump_shut_time` | $1.0\,\mathrm{s}$ | 固定 $1.0\,\mathrm{s}$ | 给定 1s 前置稳态流动阶段 |
| 关阀历时 | $t_c$ | `tc`, `pump_closure_duration` | $0.0\,\mathrm{s}$ (瞬时) | $[0.001, 1.0]\,\mathrm{s}$ (线性 ramp) | 模拟瞬时截断与阀门线性关闭过程 |

### 3.2 裂缝物理参数空间 (Fracture Parameters)

依据 R1 任务要求，以下六大核心裂缝参数是消融与正交矩阵的直接考察对象：

| 参数名称 | 符号 | 代码变量 | 基准值 (Baseline) | 敏感性扫描网格 / LHS 区间 | 采样分布类型 | 物理机制与水锤效应 |
|:---|:---|:---|:---|:---|:---|:---|
| **裂缝位置** | $x_f$ | `x_f`, `positions` | $4100.0\,\mathrm{m}$ (首缝) | $[3500.0, 4800.0]\,\mathrm{m}$ | 均匀连续抽样 / 网格对齐 | 决定水锤反射到达时刻 $t_r = t_s + 2x_f/a$ 及倒谱主峰深度 |
| **裂缝间距** | $\Delta x$ | `spacing_m`, `delta_x` | $20.0\,\mathrm{m}$ | $[5.0, 20.0]\,\mathrm{m}$ (LHS 连续); <br>`[5, 10, 20, 50, 100]` m (离散) | LHS 映射 / 预设离散网格 | 决定多缝高频谐波干涉周期 $\Delta f = a/(2\Delta x)$ 与倒谱分辨极限 |
| **水头柔度** | $C_H$ | `compliance_head_m2`, `Cf` | $1.0\times 10^{-5}\,\mathrm{m^2}$ | $[1.0\times 10^{-6}, 1.0\times 10^{-4}]\,\mathrm{m^2}$ <br>(宽幅扩展: $10^{-7}\sim 5\times 10^{-4}$) | **对数均匀** (Log-Uniform) | 控制储能电容效应，主导波头阶跃反射幅值与高频吸收 |
| **滤失系数** | $k_\text{leak}$ | `kleak_equiv`, `kleak` | $1.0\times 10^{-4}\,\mathrm{m^{5/2}/s}$ | $[1.0\times 10^{-6}, 1.0\times 10^{-3}]\,\mathrm{m^{5/2}/s}$ <br>(宽幅扩展: $0\sim 5\times 10^{-3}$) | **对数均匀** / 稳态闭合反求 | 诱发拟稳态阻尼泄流，主导水锤包络整体衰减率与基线恢复 |
| **分流权重** | $w_i$ | `inflow_weight`, `wi` | $1/n_\text{frac}$ (均匀) | 单纯形约束：$\sum w_i = 1, w_i > 0$ | **Dirichlet-LHS** 映射 ($\alpha \in \{0.3, 1, 3, 10\}$) | 控制各簇稳态压降与局部等效滤失开度，表征多簇不均匀进液 |
| **射孔阻抗** | $R_p$ | `Rp` | $0.0\,\mathrm{s^2/m^5}$ | $[0, 100, 500, 2000, 10000]\,\mathrm{s^2/m^5}$ | 阶梯离散 / 对数扫描 | 产生近井局部二次阻力压降 $R_p Q |Q|$，削弱瞬态压力传递 |
| **裂缝条数** | $N_{cl}$ | `n_frac` | $3$ 或 $4$ | $1 \sim 6$ (LHS 生产); $1 \sim 8$ (PaperA) | 离散均匀整数 | 簇群空间规模 |

### 3.3 物理单位转换关系

代码中内置了完备的单位换算辅助函数（位于 `moc_simulate/lhs_config.py`）：
- **压力柔度 $C_p$ [$m^3/\mathrm{Pa}$] 转水头柔度 $C_H$ [$m^2$]**：
  $$C_H = \rho g C_p$$
- **油田现场单位 $[\mathrm{bbl/psi}]$ 转水头柔度 $C_H$ [$m^2$]**：
  $$1\,\mathrm{bbl} \approx 0.1589873\,\mathrm{m^3}, \quad 1\,\mathrm{psi} \approx 6894.757\,\mathrm{Pa}$$
  $$1\,\mathrm{bbl/psi} \approx 2.3059\times 10^{-5}\,\mathrm{m^3/Pa} \implies C_H \approx 0.2261\,\mathrm{m^2}$$
  基准值 $C_H = 10^{-5}\,\mathrm{m^2}$ 对应约 $4.42\times 10^{-5}\,\mathrm{bbl/psi}$。
- **等效全井储液量占比**：
  全井截面总体积 $V_\text{well} = A \cdot L \approx 0.01533 \times 5000 \approx 76.6\,\mathrm{m^3}$。全井水头压缩柔度 $C_\text{well} = A L / a^2 \approx 76.6 / (1450^2) \approx 3.64\times 10^{-5}\,\mathrm{m^2}$。因此 $C_H = 10^{-5}\,\mathrm{m^2}$ 约占全井流体弹性的 **27.5%**，具备极显著且符合物理实际的穿透反射特征。

---

## 4. 已有实验脚本与数据集生成器全景调研

### 4.1 `experiments/` 目录

| 脚本路径 | 实验名称 | 扫描参数与设计结构 | 输出产物与评估指标 |
|:---|:---|:---|:---|
| `experiments/exp_shutdown_friction_2d_cepstrum/run_simulation.py` | 停泵时间与摩阻对照实验 | 8 组工况：$T_c \in [0.01, 0.1, 0.5, 1.0]\,\mathrm{s} \times \{\text{steady}, \text{brunone}\}$；3 条裂缝 $[3000, 3050, 3100]\,\mathrm{m}$ | `output/exp_shutdown_friction_2d_cepstrum/data/{case}/moc_timeseries.csv` |
| `experiments/exp_shutdown_friction_2d_cepstrum/analyze_and_plot.py` | 2D 倒谱提取与 PVR 评估 | 计算 2D 倒谱 (Kaiser 窗)、FFT 频谱、峰谷比 PVR、定位误差 $\Delta x$、峰高演化 | `figures/` 图版、`metrics_summary.csv`、`resolvability_verdict` 判定 |
| `experiments/exp_window_sweep_comparison/run_and_analyze.py` | 滑移窗长扫描对比实验 | 4 窗长 $[10, 20, 40, 60]\,\mathrm{s} \times \{\text{steady}, \text{brunone}\}$ 对照；提取 PVR、半高宽 FWHM、SNR | `output/exp_window_sweep_comparison/`：出版级 2x4 云图矩阵与指标趋势图 |

### 4.2 `moc_simulate/` 目录

| 脚本路径 | 核心定位 | 技术实现机制 | 输出规格 |
|:---|:---|:---|:---|
| `moc_simulate/run_lhs_batch_simulate.py` | 生产级进液反演黄金数据集生成器 | 14 进程并发，LHS 3D 物理采样（位置链、对数柔度、Dirichlet 进液比），$Cr=1$ 网格对齐，稳态双射闭合校验 | 导出 `case_XXXXX.npz` (schema v2.1)、`lhs_summary.csv`、`lhs_metadata.json` |
| `moc_simulate/build_reservoir_dataset.py` & `build_reservoir_steady_dataset.py` | 历史恒压水库边界对照集生成器 | 定水头边界条件 `reservoir` 下的 LHS 仿真集 | 同属 `moc_lhs_v2.1` schema，带 `toe_head` |
| `moc_simulate/run_no_fracture_benchmark.py` | 光管零裂缝经典 Joukowsky 基准 | 全场快照提取、理论 Joukowsky 水头降 $\Delta H = -a \Delta V/g$ 解析对比 | `no_fracture_steady.npz`、`steady_timeseries.csv`、水动力指标 JSON |
| `moc_simulate/stratified_bench.py` & `run_stratified_bench.py` | 分层间距格子与加噪基准集 | 9 组固定间距格子 $(5\sim 120\,\mathrm{m})$，每格 30 组清洁样本，后处理注入 5 档 AWGN 噪声 $(10\sim 40\,\mathrm{dB})$ | `clean/` 与 `noisy_snr{X}db/` npz 序列、`manifest.json` |
| `moc_simulate/leakoff_multi.py` | 基础 1~8 缝正演验证与标准图版生成 | 支持 `single` 至 `oct` 8 种形态，等间距扫描 $D \in [5, 10, 20, 50, 100]\,\mathrm{m}$，生成标准倒谱五联图 | `output/leakoff/{series}/{case}/`：`moc_timeseries.csv`, `moc_leakoff.png`, `cepstrum_standard.png`, `moc_leakoff.json` |

### 4.3 `analysis/` 目录

| 模块 / 脚本路径 | 研究重点与扫描维度 | 关键特征提取与分析方法 |
|:---|:---|:---|
| `analysis/decay_analysis/decay_regression_cf_kleak.py` | 裂缝柔度 $C_f$ 与滤失 $k_\text{leak}$ 二维全正交敏感性矩阵 | $8 \times 9 = 72$ 组参数组合 $\times \{\text{steady}, \text{brunone}\} = 144$ 次仿真。提取 1D/2D 倒谱峰值深度、峰值响应度及衰减斜率 |
| `analysis/decay_analysis/first_frac_energy_analysis.py` | 首缝能量与衰减深度回归 | 深度 $x_1 \in [1000, 4000]\,\mathrm{m}$、间距 $D$、缝数 $n$ 联合分析水锤初波能量透射率 |
| `analysis/brunone_spacing_effect/run_simulations.py` | 间距与 Brunone $k$ 系数正交矩阵 (Matrix A) | $D \in [5, 10, 20, 50, 100]\,\mathrm{m} \times k \in [0, 0.01, 0.02, 0.05, 0.1, 0.2]$ 共 30 组工况 |
| `analysis/brunone_spacing_effect/run_tc_sweep.py` | 关阀历时 $t_c$ 扫描与四时钟时延发散 | $t_c \in [1, 50, 200, 1000]\,\mathrm{ms} \times \{0, 0.01\}$，分析波首到时、峰值到时、能量 $E_{50}$ 到时与倒谱峰漂移 |
| `analysis/cepstrum/wlen_hop_sweep.py` | 倒谱时频窗长与步长敏感性扫描 | 二维热力图网格：窗长 $10\sim 60\,\mathrm{s}$，hop $0.2\sim 5.0\,\mathrm{s}$，评估时空分辨率权衡 |
| `analysis/unified_evaluation/detection_protocol.py` | 倒谱盲检测协议与综合评分基准 | 解耦真值泄漏的盲峰值检测器，多容差扫描 $[2, 5, 10, 20, 40]\,\mathrm{m}$ 精度评估 |

### 4.4 Paper 系列归档

1. **PaperA (井口多裂缝水击响应)**：
   - 包含五大模块：`01_几何网格`、`02_裂缝属性_CfKleak`、`03_leakoff验证`、`04_能量回归`、`05_Brunone常数k`。
   - 数据全部归档为 `moc_timeseries.csv` 与 `cf_kleak_table.csv`。
2. **PaperB (考虑 Brunone 的倒谱识别，对标 SPE Journal)**：
   - 五大模块均按 `code/`, `data/`, `figures/` 严格划分；
   - `05_雷诺数动态k与物性敏感性/code/run_k_kleak_surface.py` 实现了精密的 $(k \times k_\text{leak})$ 16 点正交仿真，提取 CWT 包络阻尼比 $\zeta$ 与倒谱漂移量。
3. **PaperC (CJNO Wellbore Inversion)**：
   - 基于 Sobol 拟随机采样与 RCI 簇模型，定义了详尽的 OOD（分布外）测试集及传感器/抗混叠滤波测量链。

---

## 5. 单变量消融 (OAT) 与正交矩阵/LHS 实现机制

在水锤多裂缝动力学中，不同参数的作用机制存在极强的非线性耦合。代码库中提炼出了以下三种设计与实现机制：

### 5.1 单变量消融机制 (One-At-a-Time, OAT)

**设计原理**：
选定一组物理合理的“基准算例”（Anchor/Nominal Case），每次仅改变且细密扫描 1 个目标参数，其余全部参数严格钉死在基准值上。每组扫描均同步在 `steady` 和 `brunone` 两种摩阻下运行，以分离管壁阻尼效应。

**各参数扫描网格推荐方案**：

```
基准锚点（Anchor Case）:
- 裂缝形态: 3 缝 (triple), x_f = [4000, 4020, 4040] m (首缝 4000m, 间距 20m)
- 物理属性: C_H = 1.0e-5 m², k_leak = 1.0e-4 m^{5/2}/s, w_i = [1/3, 1/3, 1/3], R_p = 0.0 s²/m⁵
- 井筒参数: L = 5000 m, D = 0.1397 m, a = 1450 m/s, V0 = 1.0 m/s, H0 = 300 m, H_ext = 100 m
- 关井条件: 瞬时 step 或快速 ramp (tc = 0.05s)
```

1. **消融轴 1：水头柔度 $C_H$** (7 点对数扫描)
   - 取值：$[10^{-7}, 3\times 10^{-7}, 10^{-6}, 3\times 10^{-6}, 10^{-5}, 3\times 10^{-5}, 10^{-4}]\,\mathrm{m^2}$
   - 目标：揭示初波反射台阶幅值 $\Delta H_\text{step}$ 与倒谱主峰幅值随储液电容的标度律。
2. **消融轴 2：滤失系数 $k_\text{leak}$** (7 点对数扫描)
   - 取值：$[0.0, 10^{-5}, 3\times 10^{-5}, 10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}]\,\mathrm{m^{5/2}/s}$
   - 目标：揭示波包整体衰减率（指数衰减常数 $\lambda$）与基线恢复速度随泄流阻尼的变化。
3. **消融轴 3：裂缝间距 $\Delta x$** (6 点物理扫描)
   - 取值：$[5, 10, 15, 20, 35, 50]\,\mathrm{m}$
   - 目标：确定 2D Kaiser 倒谱与 1D 实倒谱的混叠临界间距（Rayleigh 极限与 PVR 跌落临界点）。
4. **消融轴 4：分流权重偏流度 $w_i$** (5 组比例形态)
   - 均匀型：$[0.333, 0.333, 0.333]$
   - 首缝占优：$[0.70, 0.15, 0.15]$
   - 中缝占优：$[0.15, 0.70, 0.15]$
   - 尾缝占优：$[0.15, 0.15, 0.70]$
   - 双峰不均：$[0.48, 0.04, 0.48]$
   - 目标：分析多簇进液分配对各倒谱峰相对峰高比例的敏感映射关系。
5. **消融轴 5：射孔阻抗 $R_p$** (5 点阶梯扫描)
   - 取值：$[0, 100, 500, 2000, 10000]\,\mathrm{s^2/m^5}$
   - 目标：考察近井限流压降对瞬态波包高频成分的滤除与波前平缓化效应。
6. **消融轴 6：裂缝位置 $x_f$** (5 点深度扫描)
   - 取值：$[3500, 3800, 4100, 4400, 4700]\,\mathrm{m}$
   - 目标：检验走时双程差 $2x_f/a$ 的线性标定度与沿程管壁摩阻累积对深部倒谱识别的影响。

### 5.2 正交矩阵 (Orthogonal Array / Full Factorial Grid)

在探究多参数交互耦合（尤其是管壁非定常剪切与裂缝动态参数的混淆）时，采用正交网格。
代码库典型实现（对标 PaperB `run_k_kleak_surface.py` 与 `decay_regression_cf_kleak.py`）：
- **核心正交面**：$4 \times 4$ 正交阻尼混淆面
  - 因子 A（裂缝属性）：$C_H \in [10^{-6}, 5\times 10^{-6}, 2\times 10^{-5}, 10^{-4}]\,\mathrm{m^2}$
  - 因子 B（间距与形态）：$\Delta x \in [10, 20, 50, 100]\,\mathrm{m}$
  - 因子 C（摩阻与管壁损耗）：$\{\text{steady}, \text{brunone (k=0.01)}, \text{brunone (k=0.02)}, \text{brunone (k=0.05)}\}$
  通过该正交网格，可计算方差分析 (ANOVA) 与各因子的主效应与交互效应灵敏度指数。

### 5.3 拉丁超立方采样 (LHS with Physical Constraints)

在 `moc_simulate/run_lhs_batch_simulate.py` 中实现了具备高度物理自洽的 LHS 生成器：
1. **几何链长无干涉约束**：
   对于采样点 $u_0 \in (0, 1)$ 及簇间距因子 $u_k \in (0, 1)$：
   $$\Delta x_k = \Delta x_\text{min} + u_k (\Delta x_\text{max} - \Delta x_\text{min})$$
   $$\text{chain\_len} = \sum_{k=1}^{N_{cl}-1} \Delta x_k$$
   起始有效跨度 $\text{start\_span} = (z_\text{end} - z_\text{start}) - \text{chain\_len}$。首缝位置 $x_0 = z_\text{start} + u_0 \cdot \text{start\_span}$，严格保障所有裂缝落入 $[z_\text{start}, z_\text{end}]$ 且无缝位重叠。
2. **单纯形 Dirichlet-LHS 权重映射**：
   对于均匀随机变量 $u_i \in (0, 1)$，利用逆不完全 Gamma 函数构造标准 Gamma 变量：
   $$g_i = F^{-1}_{\Gamma(\alpha, 1)}(u_i) = \text{scipy.special.gammaincinv}(\alpha, u_i)$$
   归一化进液比：
   $$w_i = \frac{g_i}{\sum_{j=1}^{N_{cl}} g_j}, \quad \text{满足} \sum_{i=1}^{N_{cl}} w_i = 1.0, \; w_i > 0$$
3. **稳态双射滤失反求与数值保护**：
   在封闭趾端模式下，注入量全部进入裂缝，稳态流量 $Q_{f,i}^{ss} = w_i Q_{in}$。从井口向趾端逐段扣除流量并积分摩阻降，求出各裂缝处内侧稳态水头 $H_{f,i}^{ss}$。反算等效滤失：
   $$k_i^\text{equiv} = \frac{Q_{f,i}^{ss}}{\sqrt{H_{f,i}^{ss} - H_{ext}}}$$
   强制要求稳态裕度 $H_{f,i}^{ss} - H_{ext} > 10^{-3}\,\mathrm{m}$，若不满足则显式拒绝样本，从而保证反演目标间具备严格的双射映射。

---

## 6. 输出目录结构、命名规范与序列化标准

依据 `paths.py` 与 `ORIGINAL_REQUEST.md` 的规范，在执行裂缝物理参数敏感性消融实验时，`output/fracture_parameter_sensitivity/` 建议的规范层级组织如下：

```
output/fracture_parameter_sensitivity/
├── data/                                 # 生产级 NPZ 数据集 (schema v2.1)
│   ├── case_00000.npz                   # 100% 收敛无 NaN 的各仿真真解
│   ├── case_00001.npz
│   └── ...
│
├── timeseries_csv/                       # 标准高精度时程数据
│   ├── oat/                             # 单变量消融时程
│   │   ├── cf_sweep/
│   │   │   ├── steady_cf_1.0e-06_timeseries.csv
│   │   │   ├── brunone_cf_1.0e-06_timeseries.csv
│   │   │   └── ...
│   │   ├── kleak_sweep/
│   │   ├── spacing_sweep/
│   │   └── ...
│   └── orthogonal/                      # 正交矩阵工况时程
│
├── figures/                              # 出版级高分辨率图集 (>= 200 dpi)
│   ├── fig1_cf_step_response.png        # 柔度阶跃响应与反射幅值对照图
│   ├── fig2_kleak_decay_envelope.png    # 滤失阻尼与包络衰减对比图
│   ├── fig3_spectrum_dissipation.png    # 频谱对数耗散与高频能量分析图
│   ├── fig4_spacing_cepstrum_zoom.png   # 间距与 1D/2D 倒谱分辨特写图
│   ├── fig5_brunone_vs_steady_shear.png # 双摩阻形态对比与瞬态剪切耗散图
│   └── fig6_sensitivity_ranking_radar.png # 各参数敏感度雷达/排序图
│
├── tables/                               # 结构化定量评估指标
│   ├── sensitivity_metrics.csv          # 全部工况量化特征总表
│   ├── sensitivity_metrics.json         # 结构化 JSON 格式指标表
│   ├── parameter_ranking.json           # 灵敏度排序与主效应等级评定
│   ├── lhs_summary.csv                  # 数据集索引与工况耗时表
│   └── lhs_metadata.json                # 数据集元数据与物理配置描述
│
└── README.md                             # 学术级综合机理研究报告 (Markdown)
```

### 6.1 文件序列化标准

1. **NPZ 序列化**：
   - 必须调用 `np.savez_compressed(filepath, **payload)`；
   - 数组采用 `float64`（时程信号与物理坐标）与 `int32`（网格索引与计数器）；
   - 标量存储为原生 float/int/str；
   - 严格禁止存储任何 `NaN`、`Inf`，在存储前必须通过 `np.all(np.isfinite(...))` 断言检查。
2. **CSV 序列化**：
   - 采用标准 UTF-8 编码，`newline=""`，逗号 `,` 分隔；
   - 时程数据首列必须为 `t`，随后为 `H_wh`, `Q_wh`，后续严格按裂缝次序输出 `H_f1, Q_f1, ..., H_fn, Q_fn`；
   - 浮点数建议格式化至保留 4~6 位有效数字，或使用 `np.savetxt(..., fmt='%.6e')` 保障跨平台复现。
3. **JSON 序列化**：
   - 采用 `json.dump(..., indent=2, ensure_ascii=False)`；
   - 包含 `schema_version`、`seed`、`sample_count` 及完整的物理背景边界；
   - 所有 NumPy 浮点/整型标量必须显式转换为 Python 原生 `float()` 或 `int()` 避免序列化报错。

---

## 7. 时频-倒谱关键指标提取算法索引 (R3 需求支撑)

为在后续任务中产出 `sensitivity_metrics.csv`，调研确认了以下关键指标的计算标准：

1. **首波 Joukowsky 降落 ($\Delta H_J$) 与误差**：
   - 理论值：$\Delta H_\text{jouk} = -\frac{a V_0}{g}$
   - 提取时刻：$t_s + t_c$ 关阀完成瞬间的第一个下落低谷，$\Delta H = H(t_\text{first\_trough}) - H_0$。
2. **波前最大压力梯度**：
   $$(\partial H / \partial t)_\text{max} = \max_{t \in [t_s, t_s + 2L/a]} \left| \frac{H(t+\Delta t) - H(t)}{\Delta t} \right|$$
   反映波前阶跃陡峭程度，受 $R_p$、关阀历时 $t_c$ 及 Brunone 频散的直接平滑影响。
3. **多时间窗 RMS 水头衰减率 ($\lambda_\text{RMS}$)**：
   在时间段 $[t_s, t_f]$ 划分为多个往返周期窗 $W_k = [t_s + (k-1)T, t_s + k T]$（$T = 2L/a \approx 6.9\,\mathrm{s}$），计算各窗内中心化能量 $E_k = \sqrt{\frac{1}{|W_k|}\int_{W_k} (H(t) - \bar{H})^2 dt}$，按对数回归 $E_k = E_0 e^{-\lambda_\text{RMS} t}$ 提取阻尼率 $\lambda_\text{RMS}$。
4. **FFT 高频能量占比 ($\eta_\text{HF}$)**：
   对稳态段之后信号进行去均值 FFT，计算 $f > 1.5\,\mathrm{Hz}$ 频段能量占总交流能量的比例：
   $$\eta_\text{HF} = \frac{\int_{1.5\,\mathrm{Hz}}^{f_\text{Nyquist}} |X(f)|^2 df}{\int_{0^+}^{f_\text{Nyquist}} |X(f)|^2 df}$$
5. **1D 实倒谱峰值深度与幅值**：
   对 $H_{wh}(t)$ 去均值加窗后计算实倒谱 $\hat{x}(q) = \operatorname{Re}(\mathcal{F}^{-1}\{\ln |\mathcal{F}\{x(t)\}|\})$，深度轴 $z = q \cdot a / 2$。在 $z \in [x_f - 20, x_f + 20]\,\mathrm{m}$ 邻域提取峰值深度 $z_\text{peak}$ 与幅值 $P_\text{1d}$，定位偏差 $\delta x = z_\text{peak} - x_f$。
6. **2D 滑移窗倒谱峰谷比 (PVR) 与解离度**：
   采用 Kaiser 或 Hamming 滑移窗提取各时间切片的时间平均剖面，在相邻两裂缝峰值 $P_1, P_2$ 之间搜寻局部极小谷值 $V_{12}$，计算：
   $$\mathrm{PVR}_{12} = \frac{\min(P_1, P_2)}{\max(V_{12}, 10^{-6})}$$
   若 $\mathrm{PVR} \ge 2.0$ 判定为完全解离，$\mathrm{PVR} < 1.03$ 判定为融合粘连。

---

## 8. 下一步实验与落地指导建议

1. **求解器调用路径**：直接引用 `moc_simulate/wellbore_moc.py` 中的 `simulate_wellbore()` 与 `MocConfig`，保证底层求解器与已验收代码完全一致。
2. **并行与收敛保证**：借鉴 `run_lhs_batch_simulate.py` 的多进程框架，由于 $Cr=1$ 且裂缝求解内嵌了解析求根（$R_p=0$ 时）或带回溯的 Newton 法（$R_p>0$ 时），所有工况均可实现 100% 收敛通过（PASS）。
3. **摩阻分离控制**：单变量与正交实验必须严格成对运行（`friction_model='steady'` 与 `'brunone'`），通过相减直接得到 $\Delta_\text{Brunone} = \text{Metric}_\text{brunone} - \text{Metric}_\text{steady}$，以此定量分离管壁剪切耗散。
4. **图表绘制风格**：建议调用 `analysis/plotting/paper_plots.py` 中的 `apply_paper_rc()`，直接产出满足高冲击力期刊标准（Nature/SPEJ 规范，分辨率 $\ge 200\,\mathrm{dpi}$，矢量级字体与清楚物理标注）的高清图集。
