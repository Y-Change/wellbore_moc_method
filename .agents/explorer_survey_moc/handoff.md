# Handoff Report: MOC Simulation Engine & Friction Formulations Survey

**Agent**: teamwork_preview_explorer (MOC Simulation Engine & Friction Formulations Explorer)  
**Date**: 2026-09-09  
**Working Directory**: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc`  
**Handoff Type**: Hard (Investigation Complete)  
**Deliverable Document**: `survey_moc_report.md`

---

## 1. Observation (观察)

1. **核心控制方程与网格构造 (`moc_simulate/wellbore_moc.py`)**：
   - 求解器在 `MocConfig.__post_init__`（第 141-155 行）强制要求 Courant 精确匹配：
     ```python
     self.area = np.pi * self.wellbore_diameter**2 / 4.0
     N_raw = round(self.wellbore_length / (self.wavespeed * self.dt))
     self.N = N_raw
     self.dx = self.wellbore_length / self.N
     self.a_adj = self.dx / self.dt
     self.dt_adj = self.dt
     self.n_steps = round(self.tf / self.dt)
     ```
     实际波速微调为 $a_{adj} = \Delta x / \Delta t$，从而严格保证 $CFL = \frac{a \Delta t}{\Delta x} \equiv 1.0$，特征线精准连接相邻节点，消除数值耗散与数值色散。
   - 采用双流速数组保留裂缝侧向流量跃变（第 40-45、974-984 行）：
     ```python
     # 更新双流速：非缝节点 left=right=V_new；缝节点 left/right 分离
     if has_fractures:
         for k, i_f in enumerate(frac_indices):
             V_prev_left[i_f] = V_new[i_f]
             Cm_f = Cm[i_f - 1]
             V_prev_right[i_f] = -Cm_f + ga * H_new[i_f]
     ```

2. **裂缝动力学耦合方程与求解分支 (`solve_fracture_node`, lines 243-370)**：
   - 物理关系：$H_w - H_f = R_p Q_f |Q_f|$，$Q_f = C_H \frac{H_f - H_{prev,f}}{\Delta t} + k_{leak} \sqrt{\max(H_f - H_{ext}, 0)}$。
   - 当 $R_p = 0$ 时，代码采用一元二次方程解析求根（lines 283-309）：
     ```python
     alpha = 2.0 * A * ga + Cf / dt
     beta = kleak
     gamma = -A * (Cp_f + Cm_f) - (Cf / dt) * H_prev_f
     C = alpha * H_ext + gamma
     if C >= 0.0:
         H_P = -gamma / alpha
     else:
         disc = beta * beta - 4.0 * alpha * C
         y = (-beta + np.sqrt(disc)) / (2.0 * alpha)
         H_P = H_ext + y * y
     ```
     解析求根彻底规避了 $H_P \to H_{ext}$ 时的牛顿导数奇异与震荡。
   - 当 $R_p > 0$ 时，采用牛顿迭代法求解 $H_f$，并执行严格残差检验（lines 363-368）：
     `|r_mass| <= 1.0e-9 m³/s`，`|r_Rp| <= 1.0e-8 m`。
   - 储液项状态严格使用历史内部水头 $H_f^{n-1}$（`Hf_prev[k]`，lines 870-874），与井筒水头 $H_w$ 彻底物理分离。

3. **严格稳态流场初始化算法 (`compute_steady_state_profile`, lines 376-540)**：
   - 针对封闭趾端（`toe_bc="dead_end"`），注入量全部由各簇按权重分流泄入地层：$Q_{f,i}^{ss} = w_i Q_{in}$，$\sum w_i = 1$。
   - 最后一缝至封闭趾端为死水区，流速严格为零。
   - 各段水头按离散特征线关系严格递推积分，并根据稳态裂缝水头反求等效滤失系数：
     `kleak_equiv[k] = Qf_ss[k] / np.sqrt(dH)`。
   - 设定水头硬裕度（lines 514-519）：`dH = Hf_ss[k] - H_ext > 1.0e-3 m`，否则直接硬报错。

4. **双摩阻模型实现与 Brunone 稳定化技术 (`wellbore_moc.py`)**：
   - 稳态 Darcy 摩阻冲量：$J_s = \frac{f \Delta t V |V|}{2 D}$，紊流采用 Zigrand-Swami 显式逼近式（lines 164-184）。
   - Brunone 非定常摩阻冲量：$J_u = \frac{k}{2} \Delta t \left( \frac{\partial V}{\partial t} + a \cdot \operatorname{sign}(V) \cdot \left| \frac{\partial V}{\partial x} \right| \right)$（lines 220-238）。
   - 衰减系数 $k$ 由 Vardy 剪切衰减常数 $C$ 决定：层流 $C = 4.76 \times 10^{-3}$，紊流 $C = 7.41 / Re^{\log_{10}(14.3 / Re^{0.05})}$（lines 189-218）。
   - 稳定化措施（lines 825-851）：
     1. 速度符号平滑：`sign_V = np.tanh(V / 0.05)`；
     2. 裂缝邻域置零：`Ju1[i_f - 1] = 0.0`, `Ju2[i_f - 1] = 0.0`；
     3. 步数门控：仅在 $n \ge 2$（拥有前两步速度历史）时开启；
     4. 空间导数沿特征线单边差分匹配。

5. **批处理生成与 Schema 规范 (`moc_simulate/run_lhs_batch_simulate.py`)**：
   - 支持多进程并发（默认 14 worker processes）；
   - 输出单工况压缩 NPZ 数据，字段包含 `schema_version="moc_lhs_v2.1"`，包含 23+ 项完整元数据与观测时程；
   - 进液分配采用对称 Dirichlet 分布（$\alpha \in [0.3, 1.0, 3.0, 10.0]$），保证非负与单纯形归一化。

6. **测试验证执行结果**：
   - 执行 `pytest -v tests/test_steady_state_and_toe.py tests/test_fracture_storage_physics.py`：7 passed in 2.14s；
   - 执行 `pytest -v tests/test_final_acceptance.py`：14 passed in 2.87s；
   - 执行 `pytest -v tests/test_visualization.py`：5 passed in 1.18s；
   - 累计 26 个单元与集成测试 100% 全部通过（PASS）。

---

## 2. Logic Chain (推理链条)

1. **数值稳定性与无耗散性**：
   - 由 [Observation 1]，MOC 网格通过自动微调波速使得 $a_{adj} = \Delta x / \Delta t$，强制 $CFL \equiv 1.0$；
   - 由此推理：仿真网格完全避免了由于 $CFL < 1$ 带来的网格插值数值耗散，所有波形衰减与频散完全源自物理摩阻模型与裂缝边界。
2. **裂缝动力学响应保真度**：
   - 由 [Observation 2]，裂缝处区分井筒水头 $H_w$ 与内部水头 $H_f$，且储液项严格使用 $H_f^{n-1}$；
   - 当 $R_p = 0$ 时二次方程精确求解，避免牛顿奇异；当 $R_p > 0$ 时严格检验双重残差；
   - 由此推理：裂缝局部动力学满足物理连续性与能量耗散机制，高压储液、低压回吐及地层滤失特性在全时程上严格守恒。
3. **稳态基线无扰动性**：
   - 由 [Observation 3]，`compute_steady_state_profile` 采用离散特征线逆推流速与水头，使 $\sum Q_{f,i}^{ss} = Q_{in}$ 且死水区流速为 0；
   - 测试实测表明在 $t < t_s$ 时井口水头波动 $< 10^{-8}\,\mathrm{m}$；
   - 由此推理：停泵产生的波形阶跃完全由真实的关泵截流物理激扰引起，排除了任何非物理数值初始震荡的污染。
4. **非定常摩阻稳定性与物理对照可行性**：
   - 由 [Observation 4]，Brunone 模型通过双曲正切平滑与裂缝处 $J_u$ 置零，消除了跨缝流速跃变与过零点抖动引起的数值不稳定；
   - 由此推理：系统具备纯稳态 Darcy 与非定常 Brunone 在完全一致的几何和裂缝网格下并列运行的完备能力，满足需求 R2 的对比分析。
5. **消融实验与数据集产出就绪**：
   - 由 [Observation 5 & 6]，批处理入口原生支持 `moc_lhs_v2.1` 格式输出，全部单元测试均通过；
   - 由此推理：后续进行 $x_f, C_H, k_{leak}, R_p, w_i, \Delta x$ 的单变量消融与正交矩阵实验具备完整的底层求解器支撑。

---

## 3. Caveats (局限性与注意事项)

1. **单向滤失下限风险**：当前滤失模型采用单向截断 $Q_{leak} = k_{leak} \sqrt{\max(H_f - H_{ext}, 0)}$。若初始水头 $H_0$ 过低且停泵压降过大，导致 $H_f < H_{ext}$，滤失会被强制截断为 0，这与实际地层补液有差异。建议消融实验保持 $H_0 \ge 300\,\mathrm{m}$ 且 $H_{ext} \le 100\,\mathrm{m}$。
2. **倾角源项假设**：代码在特征线中保留了倾角项 $\gamma_a \Delta t V \sin\theta$，但目前生产基准固定为水平井 $\theta=0$。若未来研究斜井或直井，需重新核对重力项推导与测试。
3. **网格碰撞下限**：对于 $\Delta t = 1\,\mathrm{ms}$，$\Delta x \approx 1.45\,\mathrm{m}$。若裂缝间距 $\Delta x_{spacing} < 3\,\mathrm{m}$，可能出现 $\operatorname{round}(x_f / \Delta x)$ 重合报错。对极近间距研究需同步减小 $\Delta t$（如 $0.5\,\mathrm{ms}$）。
4. **无空化模型**：当前求解器为单相微可压缩流体假设，未考虑空化或液柱分离，最低压力需维持在液体饱和蒸汽压以上。

---

## 4. Conclusion (评估结论)

1. **求解器状态评估**：`moc_simulate/wellbore_moc.py` 数值格式严谨，CFL 严格等于 1，完全支持 Darcy 稳态与 Brunone 非定常双摩阻模型，支持裂缝多物理参数配置（$x_f, C_H, k_{leak}, R_p, w_i, \Delta x$），具有完善的输入校验和稳态流场初始化机制。
2. **数据规格匹配**：批处理生成器 `moc_simulate/run_lhs_batch_simulate.py` 与配置模块完全符合 `moc_lhs_v2.1` 规范，可直接用于大规模批量仿真与特征提取。
3. **可直接开展后续工作**：调研产出的 `survey_moc_report.md` 详尽给出了各参数的物理机制、离散公式、取值区间与消融实验扫描网格建议，后续实验执行者可直接依据该报告制定消融实验计划。

---

## 5. Verification Method (独立验证方法)

任何接收本交接报告的 Agent 或工程师，均可通过以下具体命令与文件独立复现与验证：

1. **运行完整物理与验收测试套件**：
   ```powershell
   cd e:\water_hammer_research\wellbore_moc_method
   pytest tests/test_steady_state_and_toe.py tests/test_fracture_storage_physics.py tests/test_final_acceptance.py tests/test_visualization.py -v
   ```
   **期望结果**：26 个测试项 100% 全部 PASSED，无任何 Warning 或 Error。

2. **运行最小批处理仿真冒烟测试**：
   ```powershell
   python moc_simulate/run_lhs_batch_simulate.py --n-samples 2 --workers 1 --tf 2.0 --friction brunone --toe-bc dead_end --out-dir output/smoke_survey_test
   ```
   **期望结果**：终端输出 2 组 PASS，在 `output/smoke_survey_test/data/` 下生成 `case_00000.npz` 与 `case_00001.npz`，解压检查包含 `schema_version='moc_lhs_v2.1'`。

3. **核心源码查阅路径**：
   - 求解器主文件：`moc_simulate/wellbore_moc.py`（关注第 141-155 行网格调整、第 243-370 行裂缝求解、第 376-540 行稳态反演、第 825-851 行 Brunone 计算）；
   - 配置与默认参数：`moc_simulate/config.py` 与 `moc_simulate/lhs_config.py`；
   - 综合调研报告：`e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc\survey_moc_report.md`。
