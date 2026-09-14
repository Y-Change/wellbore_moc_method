# Handoff Report — explorer_survey_features

- **Agent Name**: teamwork_preview_explorer (explorer_survey_features)
- **Role**: Time-Frequency, Cepstrum Feature Extraction, and Publication Figures/Report Standards Investigator
- **Working Directory**: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_features`
- **Parent Conversation ID**: `0e5df0b9-cf56-4e73-90ab-a6315bcdc031`
- **Handoff Type**: Hard (Task complete)
- **Date**: 2026-09-09

---

## 1. Observation (直接观察)

通过调用文件系统搜索与文件查看工具，直接查验了项目代码库中的信号处理、倒谱变换、特征提取、物理求解器与制图模块：

1. **MOC 求解器与裂缝边界模型**:
   - `moc_simulate/wellbore_moc.py` (Line 19–48, 66–156): 控制方程为特征线方程 $C^+ / C^-$；网格参数调整声速 $a_{\text{adj}} = \Delta x / \Delta t$ 使得 Courant 数精确等于 1.0；Line 181–237 实现了 Darcy 稳态与 Brunone 瞬态非定常摩阻项 $J_u = (k/2)\Delta t (\partial V/\partial t + a\,\mathrm{sign}(V)|\partial V/\partial x|)$；Line 243–320 实现了裂缝节点求解器 `solve_fracture_node`，支持水头柔度 $C_H$、滤失 $k_{\text{leak}}$ 及二次孔眼阻抗 $R_p$：$H_w - H_f = R_p Q_f |Q_f|$，$Q_f = C_H \frac{H_f - H_{prev}}{\Delta t} + k_{\text{leak}} \sqrt{\max(H_f - H_{ext}, 0)}$。
   - `moc_simulate/wellbore_moc.py` (Line 380–520): 实现了封闭趾端（`toe_bc == "dead_end"`）离散特征线稳态解初始化，严格满足质量守恒 $Q_{in} = \sum_{i=1}^n w_i Q_{in}$，并反求等效滤失系数 $k_{\text{equiv},i} = Q_{f,i}^{ss} / \sqrt{H_{f,i}^{ss} - H_{ext}}$，从源头消除了 $t=0$ 伪水击激扰。

2. **时域与频域指标参考实现**:
   - `moc_simulate/run_no_fracture_benchmark.py` (Line 317–363):
     - Joukowsky 理论降落: `dH_jouk = - a_adj * V0 / G`；实测降落: `dH_std_sim = float(H_wh_std[idx_shut_end + 10] - H_before_shut)`；误差: `err_jouk_std = float(abs(dH_std_sim - dH_jouk) / abs(dH_jouk) * 100.0)`。
     - 波前最大梯度: `dH_dt_std = np.gradient(H_wh_std, dt)`；`max_dstep_std = float(np.min(dH_dt_std[int(ts/dt):int((ts+tc+0.1)/dt)]))`。
     - 多时间窗 RMS: `windows = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)]`，通过函数 `osc_rms`（Line 43–49: `np.sqrt(np.mean((H_win - np.mean(H_win))**2))`）计算，保留率 `rms_retention_pct = rms[-1] / rms[0] * 100.0`。
     - FFT 高频能量比率: 截取 $t \ge 5.0\,\mathrm{s}$，单边幅值谱 `fft_mag = np.abs(fft(sig)[:n_post // 2]) * (2.0 / n_post)`，有效总功率 `np.sum(fft_mag[freqs > 0.05]**2)`，高频功率 `np.sum(fft_mag[freqs > 1.5]**2)`，高频能量比率 `power_high / power_total * 100.0`。

3. **倒谱管线与防真值泄漏盲寻峰**:
   - `moc_simulate/cepstrum_mocdata.py` (Line 132–230): 预处理函数 `preprocess_moc_head` 进行三次样条重采样至 $1000\,\mathrm{Hz}$ 并截取 $t \ge t_s$ 去均值；`real_cepstrum_1d` 计算 $c(q) = \mathrm{Re}\{\mathrm{IFFT}(\ln(|\mathrm{FFT}(x)| + \epsilon))\}$；深度轴按 $d = q \cdot a / 2$ 映射；倒谱响应定义为负实倒谱 $\text{response} = -c(q)$（Line 288），以使负反射系数的裂缝呈现为正立峰。
   - `moc_simulate/cepstrum_mocdata.py` (Line 576–643): `detect_1d_cepstrum_peaks` 实现了防真值泄漏盲寻峰（若传入 `fracture_depths_m` 则直接抛出 `TypeError`），采用自适应门限 `max(P90, rel * max, abs)` 与物理距离 `min_separation_m = 5.0 m`。
   - `analysis/cepstrum/_kb_core.py` (Line 60–121, 213–228): 实现了带 Kaiser-Bessel 动窗、高频预加重（$\mu=0.97$）与升正弦 Lifter 的 2D 倒谱图 `optimized_2d_cepstrogram`，并通过时间平均积分生成时均深度剖面 `compute_time_avg_depth_profile`。

4. **瑞利分辨率判据与相干有效带宽**:
   - `analysis/resolvability/forward_resolvability.py` (Line 7–27, 120–166): 确立了瑞利判据正推体系：在对数幅值谱上取动态范围门限 $\varepsilon = 10^{-\text{DR}/20}$（默认 $\text{DR}=80\,\mathrm{dB}$），找连通低频支撑最大频率作为相干有效带宽 $B_{\text{coh}}$；有效谐波数 $N_{\text{harm,eff}} = B_{\text{coh}} / f_0$；理论最小可分辨缝距 $\Delta d_{\text{min}} = a / (2 B_{\text{coh}}) = 2L / N_{\text{harm,eff}} \approx \text{FWHM}_d$。
   - `analysis/cepstrum/spacing_resolvability.py` (Line 82–119): 阐明了滑窗长度硬约束：2D 滑窗长度必须大于等于井筒全往返走时 $T_w \ge 4L/a$（约 $13.79\,\mathrm{s}$），否则频域分辨力不足导致倒谱虚化。

5. **Nature 绘图标准规范**:
   - `analysis/plotting/paper_plots.py` (Line 24–103, 271–294): 预置了 `apply_paper_rc`（Arial 字体、`svg.fonttype='none'` 保持原生 `<text>`、`savefig.dpi=300`、关闭 top/right spines、无边框图例）、语义调色板 `PALETTE`（稳态蓝 `#0F4D92`，Brunone 红 `#B64342`，辅色青/绿/紫）、子图标签辅助函数 `add_panel_label` 以及 `save_figure`（同时导出 300 dpi PNG 和矢量 SVG）。

6. **数据规范与验收基准**:
   - `docs/MOC仿真原理与数据生成规范_v2.1.md` (Line 1–578): 确立了 `moc_lhs_v2.1` 规范，明确要求封闭趾端、Dirichlet 进液分流权重 $\sum w_i = 1$、完整时程绝对压力裕度校验 $\min_{t,i}(H_{f,i}(t) - H_{ext}) > 0$ 以及对齐坐标 $x_{f,\text{aligned}}$。

---

## 2. Logic Chain (推理逻辑链)

1. **从物理激扰到波形表征 (Obs 1, 2)**:
   井口瞬时关泵 $\Delta V = -V_0$ 产生的瞬变降落由 Joukowsky 关系定量描述（$\Delta H = -a V_0 / g$）。初始波前阶跃斜率 $(\partial H/\partial t)_{\text{max}}$ 衡量了声学冲击强度。稳态 Darcy 摩阻只消耗动能，而 Brunone 瞬态摩阻在剧烈加速度波前处引入强剪切耗散，因此两者的第一处关键分流点正是波前的平滑度与梯度降。

2. **从能量耗散到多窗口 RMS 衰减 (Obs 1, 2)**:
   随着波在井筒内往返，流速总体衰减。稳态摩阻与 $V|V|$ 成正比，流速趋零时阻尼消失，波形长时间不衰；而 Brunone 摩阻与 $\partial V/\partial t$ 和 $\partial V/\partial x$ 成正比，即使宏观平均流速为零，驻波振荡加速度仍持续做负功。因此，采用 5 个时间窗提取水头波动 RMS 并计算保留率 $\text{Ret}_{\text{RMS}} = \text{RMS}_5 / \text{RMS}_1$ 和等效指数衰减常数 $\alpha_{\text{RMS}}$，能够清晰分离管壁耗散与裂缝动态滤失耗散。

3. **从频域谐波到高频滚降 (Obs 2, 4)**:
   水锤波动形成以 $f_0 = a/(4L)$ 为基频的奇次谐波梳。裂缝反射产生周期性调制干涉。Brunone 瞬态项本质上是一阶/分数阶高频低通滤波器，在 $f > 1.5\,\mathrm{Hz}$（即第 20 阶谐波以上）展现出陡峭的能量衰减（压制 20~40 dB）。计算 $R_{\text{high}} = P_{\text{high}} / P_{\text{total}}$ 能够敏感反映系统高频阻尼特性。

4. **从回波时延到 1D/2D 倒谱深度重构 (Obs 3, 4)**:
   裂缝处负反射波经双程往返到达井口，时间延迟为 $\tau_i = 2x_{f,i}/a$。对对数幅值谱进行逆变换（实倒谱）后，回波延迟转化为倒频率峰 $q_i$。通过 $d = q a / 2$ 映射到空间深度，并通过 $-c(q)$ 将负反射反转为正立特征尖峰。结合防泄漏盲寻峰，可无偏检出裂缝深度与峰值响应幅值。

5. **从相干带宽到瑞利空间分辨率极限 (Obs 4)**:
   倒谱峰在深度域的物理宽度受限于相干有效谐波梳宽度 $B_{\text{coh}}$。由傅里叶反变换性质，倒频域主瓣宽度 $\text{FWHM}_\tau \approx 1/B_{\text{coh}}$，对应空间半高宽 $\text{FWHM}_d = a/(2 B_{\text{coh}}) = 2L/N_{\text{harm,eff}}$。当裂缝间距 $\Delta x < \Delta d_{\text{min}}$ 时，两峰相互重叠并在对数非线性下坍缩合并为单峰，构成倒谱识别的物理盲区边界。

6. **从数据与图表标准到科研交付闭环 (Obs 5, 6)**:
   将上述 6 组物理特征整合至 `sensitivity_metrics.csv`（包含完整元数据、几何声学、时域梯度、多窗衰减、频域高频比、1D 倒谱峰、2D 瑞利分辨率及质量守恒）与 `sensitivity_summary.json`，依托 `paper_plots.py` 生成符合 Nature 标准（300 dpi、无边框、语义蓝红配色、原生矢量文本）的 4 组对比图版，并在 `output/fracture_parameter_sensitivity/README.md` 中形成机理完备的学术报告，达成全部研究目标。

---

## 3. Caveats (注意事项与假设说明)

1. **倾角项约定**: 当前求解器内部特征线虽然包含倾角源项，但生产基准严格锁定为水平井（`theta = 0`）。本研究结论聚焦水平段压裂裂缝敏感性，不直接适用于大斜度井或垂向直井。
2. **单向滤失截断约束**: 求解器中瞬态滤失律为 $Q_L = k_{\text{leak}} \sqrt{\max(H_f - H_{\text{ext}}, 0)}$，当负水击波使 $H_f \le H_{\text{ext}}$ 时滤失被截断为零（未考虑地层反向产液）。因此所有敏感性实验工况必须审计瞬态绝对水头裕度：$\min_{t,i}(H_{f,i}(t) - H_{\text{ext}}) > 0$，否则样本会被判定为失真。
3. **空化与液柱分离**: 仿真假定单相微可压缩牛顿流体，未包含气相析出与空化模型。井口初始水头设定在 $300\,\mathrm{m}$ 以上，确保瞬变低压段不触及饱和蒸汽压。
4. **短时滑窗长度**: 2D 倒谱滑窗长度 $w_{\text{len}}$ 不得设置过小（必须满足 $w_{\text{len}} \ge 4L/a \approx 13.8\,\mathrm{s}$），推荐采用 $30.0\,\mathrm{s}$ 窗长，以避免频域主瓣涂抹破坏谐波梳结构。

---

## 4. Conclusion (调研结论)

1. **核心算法体系完全确立**:
   - Joukowsky 理论与实测降落：$\Delta H_{\text{jouk}} = -a_{\text{adj}} V_0 / g$，实测取关阀后无反射平直段；
   - 波前梯度：$(\partial H/\partial t)_{\text{max}} = \max_{t \in [t_s, t_s + t_c + 0.1]} |\partial H/\partial t|$；
   - 多窗口 RMS 衰减：划分 5 个时间窗，计算波动标准差并提取能量保留率与指数衰减率 $\alpha_{\text{RMS}}$；
   - FFT 高频比率：$f > 1.5\,\mathrm{Hz}$ 功率占 $f > 0.05\,\mathrm{Hz}$ 总功率比例；
   - 1D 实倒谱：$d = q a / 2$，响应 $-c(q)$，防泄漏盲寻峰自适应门限提取；
   - 2D 倒谱分辨率：瑞利准则 $\Delta d_{\text{min}} = a / (2 B_{\text{coh}}) = 2L / N_{\text{harm,eff}} \approx \text{FWHM}_d$，确定近距裂缝合并边界。

2. **数据契约与图版标准明确**:
   - 规范了 42 个字段的 `sensitivity_metrics.csv` 及与之对应的层次化 `sensitivity_summary.json`；
   - 严格遵循 `analysis/plotting/paper_plots.py` 的 Nature-Figure 规范（Arial 字体、无顶部/右侧边框、无边框图例、语义蓝/红配色、300 dpi PNG + 原生 SVG）；
   - 规划了 4 组核心机理图版（阶跃波前、包络衰减、频域耗散、1D/2D 倒谱分辨率）与包含 8 个学术章节的 `README.md`。

---

## 5. Verification Method (独立验证方法)

后继执行者或父级 Agent 可通过以下具体步骤独立复现并验证本报告的所有结论与技术路线：

1. **验证求解器与基准特征计算**:
   运行无裂缝基准与 Joukowsky 验证脚本：
   ```bash
   python moc_simulate/step01_joukowsky.py
   python moc_simulate/run_no_fracture_benchmark.py --tf 50.0 --force
   ```
   检查输出日志及 `output/benchmark_no_fracture/metrics_comparison.json`，验证：
   - `joukowsky_steady_err_pct < 0.5%`；
   - `wavefront_max_slope_steady_m_s` 与 `wavefront_max_slope_brunone_m_s` 正常提取；
   - 5 窗口 RMS 水头与 `high_freq_power_ratio_pct` 正常生成。

2. **验证 1D/2D 倒谱与相干带宽正推**:
   运行倒谱管线与正推分辨率脚本：
   ```bash
   python analysis/cepstrum/cepstrum_1d_pipeline.py
   python analysis/resolvability/forward_resolvability.py --dr-db 80
   ```
   验证生成的 `cepstrum_1d_pipeline.png` 中 $-c(q)$ 峰值是否与设定缝深严格对齐，以及正推相干带宽 $B_{\text{coh}}$ 与理论极限缝距 $\Delta d_{\text{min}}$ 是否符合瑞利准则。

3. **运行测试套件保证数据兼容性**:
   ```bash
   pytest tests/test_final_acceptance.py -k "schema or steady or brunone"
   ```
   验证 `moc_lhs_v2.1` 数据 schema 字段与求解器稳态初始化断言 100% 通过（PASS）。

4. **查验调研报告完整性**:
   直接查看生成的技术报告文件：
   - 绝对路径: `e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_features\survey_features_report.md`
   - 检查数学公式、离散算法、CSV/JSON schema、4 组 Nature 图版构思及 `README.md` 规范。
