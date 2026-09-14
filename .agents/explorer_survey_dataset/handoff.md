# Handoff Report: Dataset Schemas, Parameter Spaces, and Existing Experiment Scripts

> **Agent**: `teamwork_preview_explorer` (explorer_survey_dataset)  
> **Parent Conversation ID**: `0e5df0b9-cf56-4e73-90ab-a6315bcdc031`  
> **Date**: 2026-09-09T06:51:00Z  
> **Handoff Type**: Hard (Task Complete)  
> **Target Delivery Report**: `survey_dataset_report.md`

---

## 1. Observation (直接观测事实)

1. **`ORIGINAL_REQUEST.md` (lines 12–37)**:
   - 明确实验目标：开展新版 MOC 求解器下裂缝物理参数 ($x_f, C_H, k_\text{leak}, R_p, w_i, \Delta x$) 敏感性消融实验；
   - 明确输出目录：`output/fracture_parameter_sensitivity`；
   - 明确双摩阻对比：Darcy-Weisbach 稳态 vs Brunone 非定常摩阻；
   - 明确数据规范：收敛通过（PASS，无 NaN/Inf），导出 `moc_lhs_v2.1` 兼容 NPZ 与高精度 CSV 时程文件，产出结构化指标表 (`sensitivity_metrics.csv` 及 JSON)、出版级图版（$\ge 200\,\mathrm{dpi}$）与学术级机理报告 `README.md`。

2. **Schema 规范与 NPZ 实体字段**:
   - `docs/MOC仿真原理与数据生成规范_v2.1.md` (lines 445–470): 定义了 schema v2.1 字段与别名机制（`Cf` 为 `compliance_head_m2` 别名，`kleak` 为 `kleak_equiv` 别名，`x_f` 为 `x_f_aligned` 别名）。
   - 实测已生成的 `output/acceptance_10_test/data/case_00000.npz`:
     - 包含 **41 个独立键值**；
     - 核心观测与物理数组：`H_wh` (shape=(50001,), dtype=float64), `Q_wh` (shape=(50001,), dtype=float64), `t` (shape=(50001,), dtype=float64), `x_f_aligned` (shape=(1,), dtype=float64), `compliance_head_m2` (shape=(1,), dtype=float64), `inflow_weight` (shape=(1,), dtype=float64), `kleak_equiv` (shape=(1,), dtype=float64), `Rp` (shape=(1,), dtype=float64)；
     - 闭合量：`Hw_ss`, `Hf_ss`, `Qf_ss`, `Qin_ss`；
     - 网格量：`N=3448`, `dx=1.450116`, `dt=0.001`, `wavespeed_adj=1450.116009`；
     - 元数据标量：`schema_version="moc_lhs_v2.1"`, `seed=42`, `case_id=0`, `friction="brunone"`, `toe_bc="dead_end"`。

3. **CSV 时程文件规范**:
   - `moc_simulate/leakoff_multi.py` (lines 578–592) 及 `README.md` (line 139):
     - 列结构：`csv_header = ['t', 'H_wh', 'Q_wh'] + [f'H_f{k+1}', f'Q_f{k+1}' for k in range(n_frac)]`；
     - 命名规范：`moc_timeseries.csv`；
     - 典型行示例：`t, H_wh, Q_wh, H_f1, Q_f1, ...`，全场浮点数连续记录。

4. **基准参数与范围取值**:
   - `moc_simulate/config.py` (lines 13–40):
     - 井筒基准：$L=5000.0\,\mathrm{m}, D=0.1397\,\mathrm{m}, \rho=1000.0\,\mathrm{kg/m^3}, \nu=1.0\times 10^{-6}\,\mathrm{m^2/s}, a=1450.0\,\mathrm{m/s}, \varepsilon=4.5\times 10^{-5}\,\mathrm{m}, V_0=1.0\,\mathrm{m/s}, H_0=300.0\,\mathrm{m}, \theta=0.0\,\mathrm{rad}, \text{toe\_bc}='dead\_end'$；
     - 裂缝基准：$C_H = 1.0\times 10^{-5}\,\mathrm{m^2}, k_\text{leak} = 1.0\times 10^{-4}\,\mathrm{m^{5/2}/s}, H_\text{ext}=100.0\,\mathrm{m}, R_p=0.0\,\mathrm{s^2/m^5}$；
     - 离散间距预设：`SPACING_PRESETS_M = (5, 10, 20, 50, 100)`；首缝基准 `FRAC_FIRST_M = 4100.0`。
   - `moc_simulate/lhs_config.py` (lines 38–70):
     - 缝网分布区间：$[3500.0, 4800.0]\,\mathrm{m}$；
     - 间距约束：`min_spacing = 5.0` m, `max_spacing = 20.0` m；
     - 水头柔度：$\log_{10} C_H \in [-6.0, -4.0]$（对数均匀）；
     - Dirichlet 权重参数：$\alpha \in [0.3, 1.0, 3.0, 10.0]$；
     - 滤失范围：$\log_{10} k \in [-6.0, -3.0]$；
     - 域随机化：波速 $[1350, 1550]\,\mathrm{m/s}$，信噪比 $[20, 60]\,\mathrm{dB}$，摩阻 `["steady", "brunone"]`。

5. **消融与正交设计成熟代码**:
   - `experiments/exp_shutdown_friction_2d_cepstrum/`：停泵关阀历时 $T_c \in [0.01, 0.1, 0.5, 1.0]\,\mathrm{s} \times \{\text{steady}, \text{brunone}\}$；
   - `analysis/decay_analysis/decay_regression_cf_kleak.py`：$C_f$ (8 点) $\times k_\text{leak}$ (9 点) 全网格 $\times \{\text{steady}, \text{brunone}\}$；
   - `PaperB_考虑brunone的倒谱识别/.../run_k_kleak_surface.py`：$k \in \{0, 0.01, 0.02, 0.05\} \times k_\text{leak} \in \{10^{-5}, 5\times 10^{-5}, 10^{-4}, 5\times 10^{-4}\}$ 16 点正交阻尼混淆面；
   - `moc_simulate/run_lhs_batch_simulate.py`：3D 物理约束 LHS，结合 `gammaincinv` 的 Dirichlet 单纯形映射。

---

## 2. Logic Chain (推理链条)

1. **从任务需求到数据规范统一性**：
   - 需求 R1 与 Acceptance Criteria 要求输出 `moc_lhs_v2.1` 兼容的 NPZ 与高精度 CSV 时程文件；
   - 观测到代码库已在 `run_lhs_batch_simulate.py`、`test_final_acceptance.py` 及生产集严格实施了该标准；
   - 推理：新设计的敏感性实验无需创造新数据格式，直接采用该 41 字段 NPZ 规范与 `t, H_wh, Q_wh, H_f*, Q_f*` 的 CSV 格式，可完全实现与现有 AI 算子（`dataset_surrogate.py`）及倒谱提取脚本的零摩擦对接。

2. **从物理机理到单变量消融 (OAT) 设计**：
   - 观测到 $C_H$、$\Delta x$、$k_\text{leak}$、$R_p$、$w_i$ 对水锤波具有截然不同的物理主导机制（$C_H$ 决定初波阶跃台阶幅值，$\Delta x$ 决定高频干涉干涉波峰重叠，$k_\text{leak}$ 决定中长时包络指数衰减，$R_p$ 决定高频波前滤波，$w_i$ 决定多簇相对反射比重）；
   - 推理：必须以物理典型工况为基准锚点（3 缝，首缝 4000m，间距 20m，$C_H=10^{-5}\,\mathrm{m}^2$，$k_\text{leak}=10^{-4}$），每次仅扰动 1 个物理参数进行 5~7 档阶梯扫描，并强制成对运行 `steady` 与 `brunone` 摩阻，方能干净解耦各参数的独立边际敏感性。

3. **从参数耦合到正交矩阵设计**：
   - 观测到 PaperB 与 Decay 分析中发现管壁非定常剪切耗散与裂缝局部动态阻尼存在“阻尼混淆”；
   - 推理：仅做 OAT 无法评估交互效应。必须构建小规模全正交网格（如 $C_H \times \Delta x \times \text{Friction}$ 或 $k \times k_\text{leak}$ 正交面），通过方差分析 (ANOVA) 与灵敏度排序雷达图定量评定各裂缝参数的主效应与交叉影响等级。

4. **从代码组织到输出目录标准**：
   - 观测到 `paths.py` 与各模块输出规范一致遵循 `data/` (npz), `timeseries_csv/`, `figures/`, `tables/`, `README.md` 的层级结构；
   - 推理：`output/fracture_parameter_sensitivity` 应严格依此构建，确保所有中间及最终交付物结构清晰，自包含且易于审查。

---

## 3. Caveats (局限与未调研区域)

1. **倾角项假设限制**：
   当前 MOC 控制方程约定固定 `theta=0`，因此所有参数空间调研结论与基准均局限于水平井段；若后续扩展至大斜度井或垂直井，需重构重力源项。
2. **射孔阻抗采样实践经验相对较少**：
   现有 LHS 生产集默认固定 $R_p=0.0$ 以加速计算；虽然 `wellbore_moc.py` 内嵌了带残差保障的 Newton 迭代求解器且通过物理测试，但在敏感性消融实验中引入 $R_p > 0$ 时，需微调步长或预留稍微多一点的求解耗时。
3. **未直接修改业务代码**：
   作为调研探索智能体 (explorer)，本任务严格遵守只读分析原则，未直接生成仿真波形或修改业务源码，全部成果均汇总于结构化报告。

---

## 4. Conclusion (确定性结论)

1. **Schema 冻结契约**：新敏感性实验数据集应全面采纳 `moc_lhs_v2.1`（包含 41 个标准 NPZ 字段）以及 `moc_timeseries.csv`（`t, H_wh, Q_wh, H_f1, Q_f1, ...`）。
2. **基准参数空间推荐**：
   - 井筒基准：$L=5000\,\mathrm{m}, D=0.1397\,\mathrm{m}, a=1450\,\mathrm{m/s}, V_0=1.0\,\mathrm{m/s}, H_0=300\,\mathrm{m}, H_{ext}=100\,\mathrm{m}$；
   - 裂缝扫描区间：$x_f \in [3500, 4800]\,\mathrm{m}$，$\Delta x \in [5, 50]\,\mathrm{m}$，$C_H \in [10^{-7}, 10^{-4}]\,\mathrm{m}^2$，$k_\text{leak} \in [0, 3\times 10^{-3}]\,\mathrm{m}^{5/2}/\mathrm{s}$，$w_i \in \text{Simplex}$，$R_p \in [0, 10000]\,\mathrm{s^2/m^5}$。
3. **仿真方案架构推荐**：
   - 模块 1：OAT 单变量消融（6 组物理轴，每轴 5~7 档，共约 36 组工况 $\times 2$ 摩阻 = 72 组仿真）；
   - 模块 2：双摩阻正交矩阵（$4 \times 4 \times 2 = 32$ 组工况）；
   - 提取指标表：`sensitivity_metrics.csv` 与 `sensitivity_metrics.json`；
   - 图集：4+ 组出版级图版（阶跃响应、包络衰减、频谱对数耗散、1D/2D 倒谱特写）；
   - 报告：`README.md`。

---

## 5. Verification Method (独立验证方法)

后续执行智能体可运行以下具体命令核验本调研报告中的事实与结论：

1. **核验 schema v2.1 结构与测试通过性**：
   ```powershell
   python -m pytest tests/test_final_acceptance.py -q
   ```
2. **核验现存 NPZ 字段及 41 个键值**：
   ```powershell
   python -c "import numpy as np; d=np.load(r'output/acceptance_10_test/data/case_00000.npz'); print('Keys:', len(d.files)); assert d['schema_version']=='moc_lhs_v2.1'; print('Passed!')"
   ```
3. **查阅详细调研报告**：
   ```powershell
   type .agents\explorer_survey_dataset\survey_dataset_report.md
   ```
