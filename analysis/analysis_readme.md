# Analysis 目录结构说明

`analysis` 目录主要用于深度物理机理分析、倒谱特征提取、参数回归、以及最终的高质量论文制图。该目录下的脚本通常依赖于 `validation` 生成的仿真数据，或者独立调用 `wellbore_moc` 跑批量算例。

经过重组，当前的代码被划分为以下五个核心功能模块：

## 1. decay_analysis (衰减机理与能量回归)
用于提取水击波形的衰减特征（如首波峰值、整体能量衰减率），并进行物理标度坍缩和指数/扩展指数回归分析。
- **`decay_regression.py`**：衰减机理主线脚本，负责生成基础波形、进行 1D/2D 倒谱处理，并提取衰减峰值数据（输出到 01~03 阶段）。
- **`decay_stretched_exp_extraction.py`**：基于扩展指数模型，对衰减形态进行参数 $b$ 和 $\beta$ 的空间提取。
- **`decay_regression_cf_kleak.py`**：专门研究裂缝柔度 $C_f$ 和滤失系数 $K_{leak}$ 交叉敏感性的参数扫描仿真脚本。
- **`decay_regression_collapse.py` & `decay_regression_compare.py`**：针对特征波形折叠和对比的专门分析。
- **`first_frac_energy_analysis.py`**：聚焦首缝能量，进行深度、间距和缝数的三维联合分析。
- **`energy_regression.py` & `energy_regression_fit.py`**：整体裂缝系统能量的回归拟合与指标计算。
- **`decay_k_parameter_analysis.py` & `test_new_fits.py`**：测试与验证新的衰减常数 $k$ 公式拟合效果。

## 2. plotting (论文级制图)
集中了所有用于生成出版级别（Publication-ready）图表的脚本。
- **`paper_plots.py`**：提供底层的 `apply_paper_rc` 和 `save_figure` 函数，统一样式（如字体、字号、边距、热力图规范等）。
- **`plot_divergence_vs_collapse.py`**：绘制时间域色散与基于无量纲位置指数 $P_{idx}$ 的尺度坍缩图。
- **`plot_friction_comparisons_pidx.py`**：对比稳态摩阻与 Brunone 摩阻在 $P_{idx}$ 标度下的响应差异。
- **`plot_cf_kleak.py`**：绘制 $C_f$ 和 $K_{leak}$ 敏感性矩阵结果图。
- **`plot_new_fits_x1_4000.py`**：为新拟合公式绘制在深井 (x1=4000m) 处的对比图。
- **`plot_all_cases_zoom.py`**：批量绘制各仿真 case 裂缝放大区域视图的脚本。

## 3. cepstrum (倒谱专项分析)
用于对比和优化倒谱（Cepstrum）识别裂缝特征的能力。
- **`cepstrum_1d_pipeline.py`**：1D 实倒谱处理过程的详细步骤与可视化展示。
- **`kaiser_bessel_multi.py`**：核心脚本，对比多缝情况下 Kaiser-Bessel 等不同窗函数 2D 倒谱的识别效果与缝深残差。
- **`_kb_core.py`**：实现 Kaiser-Bessel 及其它倒谱方法核心算法的底层支持库。
- **`window_comparison.py`**：基础的窗函数效果比对。
- **`wlen_sweep.py` & `wlen_hop_sweep.py`**：进行窗长 (Window Length) 及重叠率 (Hop Ratio) 扫描，以寻找最佳的时频分辨平衡点。
- **`wlen_hop_heatmap.py` & `wlen_hop_study2_plots.py`**：绘制扫参结果的热力图与切片对比图。
- **`spacing_resolvability.py`**：汇总不同缝间距 (D10/D20/D50/D100) 下的倒谱分辨能力指标。

## 4. resolvability (缝距分辨力预测)
- **`forward_resolvability.py`**：水击多裂缝倒谱分辨力的“正推”预测脚本。根据设定的动态范围 (`--dr-db`) 计算相干带宽 $B_{coh}$、有效谐波数 $N_{harm,eff}$ 及理论最小可分辨缝距 $\Delta d_{min}$。

## 5. reporting (自动报告生成)
- **`report_render.py`**：批量读取 `output` 下的图表与 JSON 数据，自动生成综合分析报告（Markdown/HTML）。
- **`research_report.py`**：定制化的研究报告生成脚本。

---

**导入规范：**
该目录下的所有脚本均通过 `while True:` 动态循环寻找 `paths.py` 来挂载项目根目录到 `sys.path`。因此在任何层级下直接通过 `python script_name.py` 执行均不会产生 `ModuleNotFoundError`。
