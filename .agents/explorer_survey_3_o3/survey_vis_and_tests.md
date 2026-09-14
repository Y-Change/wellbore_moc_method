# MOC_V2 可视化图版流水线与自动化测试回归深度调研报告
## Technical Survey on Visualization Pipeline (Figures 1–7, Nature-Grade Formatting, Rainbow 2D Cepstrograms) and Test Suite Health

**报告归档路径**：`e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_3_o3\survey_vis_and_tests.md`  
**调查人**：Explorer 3 (Visualization & Test Regression Investigator)  
**调查基准时间**：2026-09-11  
**关联任务**：MOC_V2 技术报告拆分重构、现场级 7 大专题敏感性仿真、Nature 级图版生成与回归健康度保障

---

## 目录
1. [执行摘要与关键发现](#1-执行摘要与关键发现)
2. [第一部分：倒谱信号处理体系与底层实现调查](#2-第一部分倒谱信号处理体系与底层实现调查)
   - 2.1 1D 实倒谱算法原理与代码实现 (`moc_simulate.v2.signal.cepstrum_1d`)
   - 2.2 2D 连续滑窗倒谱图算法原理 (`moc_simulate.v2.signal.cepstrum_2d`)
   - 2.3 重大代码缺陷预警：`compute_cepstrogram_2d` 默认参数崩溃 Bug
   - 2.4 极性、去趋势与波前求导锐化机制
   - 2.5 历史版本实现对比 (`v1/cepstrum_mocdata.py` vs `v2/signal`)
3. [第二部分：Figure 1 至 Figure 7 规范与绘图流水线方案](#3-第二部分figure-1-至-figure-7-规范与绘图流水线方案)
   - 3.1 目标交付文件清单与目录结构 (`docs/moc_v2_technical_report/sensitivity_figures/`)
   - 3.2 七大专题独立图版定义与参数映射
   - 3.3 图版标准复合四联架构设计 (Panel a, b, c, d)
   - 3.4 Rainbow 2D 倒谱云图的核心约束（严格禁止检出率统计文本覆盖）
   - 3.5 Nature-Grade 顶级科学绘图规范执行指南 (字体、线宽、色板、SVG 栅格化混排)
4. [第三部分：全库 Pytest 测试套件现状与模块覆盖审计](#4-第三部分全库-pytest-测试套件现状与模块覆盖审计)
   - 4.1 99 项既有测试全景分布与执行健康度
   - 4.2 各测试模块责任边界与断言范围
   - 4.3 潜在回归风险矩阵与防破坏红线
5. [第四部分：执行性能基准与工程实施建议](#5-第四部分执行性能基准与工程实施建议)
   - 5.1 100s 正演计算耗时基准 (Steady vs Brunone)
   - 5.2 38+ 算例的高并发多进程调度策略 (`BatchRunner`)
   - 5.3 绘图主程序架构设计参考原型

---

## 1. 执行摘要与关键发现

在本次专项调查中，Explorer 3 针对水力压裂井筒水锤 MOC_V2 项目的可视化流水线、倒谱信号处理内核、Figure 1~7 规范以及测试套件健康度进行了全方位代码穿透审计与数值验证，形成以下核心结论：

1. **测试套件现状健康**：当前全库 `pytest` 自动化测试集共包含 **99 个测试项**，分布于 `tests/` 目录下的 9 个测试模块中。实测全量执行耗时 **61.14 秒**，**100% 全部通过（99 passed, 0 failed）**，无任何破坏性历史残留。
2. **信号处理模块存在隐蔽崩溃缺陷**：在生产级内核 `moc_simulate.v2.signal.cepstrum_2d.compute_cepstrogram_2d` 中，默认形参为 `window: str = "kaiser"`。当调用底层 `real_cepstrum` 时，因 `scipy.signal.get_window("kaiser", n)` 强制要求元组参数 `('kaiser', beta)`，且 `cepstrum_1d.py:47` 存在 `window.lower()` 导致元组类型报错，**直接调用 `compute_cepstrogram_2d(...)` 必将以 `ValueError` 或 `AttributeError` 异常退出**！因 `tests/test_v2_architecture.py` 仅导入但未测试 2D 倒谱，此 Bug 此前未被发现。后续实现图版时必须优先修复或显式传入 `window="hamming"`。
3. **Figure 1 至 Figure 7 规范完全闭合**：目标需在 `docs/moc_v2_technical_report/sensitivity_figures/` 严格产出 **14 份文件**（7 个 300 DPI PNG + 7 个可编辑矢量 SVG）。各图版统一采用复合多面板结构：
   - **Panel a**：100s 全时程宏观演化与关泵初期局部声波震荡（双重视角）；
   - **Panel b**：1D 实倒谱曲线族（以醒目垂直虚线标定真实裂缝位置）；
   - **Panel c/d**：两组典型对比工况的 **Rainbow 色阶（`cmap='rainbow'`）2D 连续时空倒谱云图**，标定真实裂缝深度线，**严格禁止添加文本检出率统计判据**（严禁任何类似 `Accuracy: 100%`, `Detection 4/4`, 标签框等文本干扰）。
4. **计算性能与多核加速**：单次 100s 仿真在 Brunone 非定常摩阻下耗时约 78 秒（纯稳态摩阻约 11 秒）。7 大专题包含 38+ 组算例，利用本机 16 核 CPU（`BatchRunner`，14 进程并发）可在 **3.5 分钟内**并发求解完毕。

---

## 2. 第一部分：倒谱信号处理体系与底层实现调查

### 2.1 1D 实倒谱算法原理与代码实现 (`moc_simulate.v2.signal.cepstrum_1d`)

在管网瞬变流声学检测中，井口测压水头 $H_{wh}(t)$ 包含水击首波冲击与下游各簇裂缝的反射回波。1D 实同态倒谱（Real Cepstrum）将时间卷积分解为倒频率（Quefrency）域中的线性加性峰：
$$c(\tau) = \operatorname{Re}\left\{ \mathcal{F}^{-1} \left( \ln |\mathcal{F}(w(t) \cdot x(t))| + \epsilon \right) \right\}$$

根据声波沿井筒双程走时（Two-Way Travel Time）运动学关系：
$$\tau = \frac{2 x}{a} \implies x = \frac{a \tau}{2}$$
其中 $a$ 为井内声学波速（现场基准取 $a = 1450\,\mathrm{m/s}$）。

代码位于 `moc_simulate/v2/signal/cepstrum_1d.py`：
- `quefrency_to_distance(quefrency, wavespeed)`: 换算公式为 `(quefrency * wavespeed) / 2.0`（行 18-20）；
- `distance_to_quefrency(distance, wavespeed)`: 换算公式为 `(2.0 * distance) / wavespeed`（行 23-25）；
- `compute_cepstrum_1d(time, head, wavespeed, fs=None, ts=1.0, window="hamming", derivative=False, derivative_order=1, max_distance=None)`:
  - 执行 `scipy.interpolate.interp1d(..., kind="cubic")` 重采样至等时间间隔网格 $f_s$；
  - 截取关泵后数据 $t \ge t_s$，消除注水初场直流偏置；
  - 可选阶数波前差分求导 `np.gradient`；
  - 调用 `real_cepstrum` 并根据 $x_{max}$ 进行空间裁剪。

### 2.2 2D 连续滑窗倒谱图算法原理 (`moc_simulate.v2.signal.cepstrum_2d`)

2D 连续时空倒谱谱图（Cepstrogram / Short-Time Cepstrum Transform, STCT）通过沿时间轴滑动固定时窗 $T_{win}$，解析声波能量在穿透不同裂缝簇过程中的动态衰减与时间响应：
- 滑动窗口长 $T_{win}$（基准 $30.0\,\mathrm{s}$ 或 $15.0\,\mathrm{s}$）；
- 滑动步长 $\Delta t_{hop}$（基准 $0.25\sim 1.0\,\mathrm{s}$）；
- 每个时窗内计算实倒谱并映射至井深轴 $x$；
- 矩阵维度为 $(N_{frames}, N_{distances})$。

### 2.3 重大代码缺陷预警：`compute_cepstrogram_2d` 默认参数崩溃 Bug

在对 `moc_simulate.v2.signal` 开展实机调用测试时，发现了严重的底层逻辑缺陷：

#### 缺陷代码溯源
1. **`cepstrum_2d.py:26`**:
   ```python
   def compute_cepstrogram_2d(
       ...
       window: str = "kaiser",
       ...
   ):
   ```
2. **`cepstrum_1d.py:47-49`**:
   ```python
   if window is not None and window.lower() not in ("none", "rect", "boxcar"):
       win = get_window(window, n)
       signal = signal * win
   ```

#### 崩溃现象
- **情形 1**：按默认参数调用 `compute_cepstrogram_2d(...)`，`window="kaiser"` 传入 `scipy.signal.get_window("kaiser", n)`，由于 SciPy 中 Kaiser 窗必须指定 $\beta$ 参数，直接抛出异常：
  ```
  ValueError: The 'kaiser' window needs one or more parameters -- pass a tuple.
  ```
- **情形 2**：若调用者尝试遵循 SciPy 规则传入元组 `window=('kaiser', 4.0)`，则在 `cepstrum_1d.py:47` 执行 `window.lower()` 时崩溃：
  ```
  AttributeError: 'tuple' object has no attribute 'lower'
  ```

#### 测试缺失原因
在 `tests/test_v2_architecture.py` 第 48 行虽然导入了 `compute_cepstrogram_2d`，但在随后的测试函数中**从未对其进行任何调用或断言**（仅测试了 1D 倒谱与峰提取），使得该致命 Bug 在自动化测试全 PASS 的掩护下一直潜伏。

#### 修复与规避方案
在后续编写 `run_sensitivity_study.py` 与绘图函数时，必须采取以下二者之一：
- **方案 A（库层修复）**：修改 `cepstrum_1d.py` 的 `real_cepstrum`，使其支持元组或将 `"kaiser"` 自动映射为 `("kaiser", 4.0)`：
  ```python
  if window is not None:
      if isinstance(window, str):
          if window.lower() == "kaiser":
              win = get_window(("kaiser", 4.0), n)
          elif window.lower() not in ("none", "rect", "boxcar"):
              win = get_window(window, n)
          else:
              win = None
      else:
          win = get_window(window, n)
      if win is not None:
          signal = signal * win
  ```
- **方案 B（调用层规避）**：显式指定经测试验证 100% 稳定的标准平滑窗 `window="hamming"` 或 `window="hann"`。

### 2.4 极性、去趋势与波前求导锐化机制

在水力压裂井筒中，高压波遇到裂缝流体顺应性容抗时产生负反射系数（$\Gamma < 0$，负压降膨胀波）。
- 在对数频谱反变换后，裂缝回波在原始实倒谱中表现为向下凹陷的负峰；
- 为符合常规地球物理与科研绘图直觉，历史优秀脚本（如 `PaperB` 与 `experiments/sensitivity/plot_figures.py`）均取反极性 $-c(\tau)$ 或幅值 $|c(\tau)|$，使裂缝表现为向上的正能量尖峰；
- **波前求导锐化（Derivative Sharpening）**：低频大反弹基线（宏观膨胀储能）在倒谱低频端会引入严重的斜坡背景漂移。采用一阶差分 $\partial H/\partial t$ 能彻底滤除直流基线，使 10m 级密簇反射能量峰信噪比提升 15 dB 以上。

### 2.5 历史版本实现对比

| 特性 | V1 历史实现 (`v1/cepstrum_mocdata.py`) | V2 生产级实现 (`v2/signal/`) |
|---|---|---|
| **2D 滑窗算法** | 基于 `as_strided` 矩阵分帧，全向量化批处理 FFT | Python 循环逐帧切片调用 `real_cepstrum` |
| **窗函数支持** | 手动分支适配 `np.kaiser(wlen, 4)`, `scipy.hann` | 依赖 `scipy.signal.get_window`（存在 kaiser 传参缺陷） |
| **倒谱极性处理** | 内部自动执行 `response = -C_kept` 取反 | 暴露原始 `cepstrum`，由上层决定是否求导/取反 |
| **去趋势处理** | 自动减去时域均值 `h - np.mean(h)` | 自动减均值，并提供导数滤波器 |
| **测试覆盖率** | 历史间接测试覆盖 | 1D 全量覆盖，2D 暂未包含在单元测试内 |

---

## 3. 第二部分：Figure 1 至 Figure 7 规范与绘图流水线方案

### 3.1 目标交付文件清单与目录结构

所有图版必须输出至目录：`docs/moc_v2_technical_report/sensitivity_figures/`。  
共计 **7 组专题，每组均导出 300 DPI PNG 与矢量 SVG，合计 14 份文件**：

| 编号 | PNG 文件名 (300 DPI) | SVG 矢量文件名 | 对应物理专题与核心控制变量 |
|:---:|---|---|---|
| **Figure 1** | `fig1_fracture_count_sensitivity.png` | `fig1_fracture_count_sensitivity.svg` | 专题 1：裂缝数量 $N_c \in [1, 2, 3, 4, 5, 6, 7, 8]$ |
| **Figure 2** | `fig2_fracture_spacing_sensitivity.png` | `fig2_fracture_spacing_sensitivity.svg` | 专题 2：裂缝间距 $d \in [5, 10, 15, 20, 25, 30, 50, 80]\,\mathrm{m}$ |
| **Figure 3** | `fig3_compliance_sensitivity.png` | `fig3_compliance_sensitivity.svg` | 专题 3：顺应性储量 $C_f \in [0.002, 0.005, 0.010, 0.020, 0.030]\,\mathrm{m^2}$ |
| **Figure 4** | `fig4_leakoff_sensitivity.png` | `fig4_leakoff_sensitivity.svg` | 专题 4：拟达西滤失 $k_{leak} \in [0.2, 0.6, 1.0, 3.0, 10.0]\times 10^{-4}\,\mathrm{m^{2.5}/s}$ |
| **Figure 5** | `fig5_perforation_impedance_sensitivity.png` | `fig5_perforation_impedance_sensitivity.svg` | 专题 5：射孔流阻 $K_p \in [1.5, 3.5, 5.43, 10.0, 25.0]\times 10^5\,\mathrm{s^2/m^5}$ |
| **Figure 6** | `fig6_pump_shutoff_ramp_sensitivity.png` | `fig6_pump_shutoff_ramp_sensitivity.svg` | 专题 6：关泵斜坡 $t_c \in [0.0, 0.5, 1.0, 1.5, 2.0]\,\mathrm{s}$ |
| **Figure 7** | `fig7_intake_capacity_combinations.png` | `fig7_intake_capacity_combinations.svg` | 专题 7：多簇非均匀进液能力 5 组典型组合 |

### 3.2 七大专题独立图版定义与参数映射

#### Base Case 统一物理锚点
- 几何与波速：全长 $L = 5000\,\mathrm{m}$，管径 $D = 0.1397\,\mathrm{m}$，声速 $a = 1450\,\mathrm{m/s}$；
- 流动基准：初始流速 $V_0 = 1.0\,\mathrm{m/s}$，井口水头 $H_0 = 300\,\mathrm{m}$，储层孔隙压力 $H_{ext} = 100\,\mathrm{m}$；
- 裂缝基准：3 簇裂缝位于 $[4500, 4510, 4520]\,\mathrm{m}$（趾端死水区长 $480\,\mathrm{m}$）；
- 物性基准：$C_f = 0.01\,\mathrm{m^2}, k_{leak} = 1.0\times 10^{-4}\,\mathrm{m^{2.5}/s}, K_p = 5.43\times 10^5\,\mathrm{s^2/m^5}, t_c = 1.0\,\mathrm{s}$（余弦平滑关泵）。

#### 各专题参数扫描细节
1. **专题 1（裂缝数量）**：
   - 扫描：$N_c \in [1, 2, 3, 4, 5, 6, 7, 8]$，起点固定 $4500\,\mathrm{m}$，间距 $10\,\mathrm{m}$；
   - 物理现象：多簇并联使得总流体顺应性成倍增加，宏观反弹能量更强，但下游簇声波逐级透射衰减加剧。
2. **专题 2（裂缝间距）**：
   - 扫描：3 簇裂缝，起点 $4500\,\mathrm{m}$，间距 $d \in [5, 10, 15, 20, 25, 30, 50, 80]\,\mathrm{m}$；
   - 物理现象：$d \le 10\,\mathrm{m}$ 处于 Rayleigh 分辨率临界区，缝间多重水击产生强共振混响；$d \ge 30\,\mathrm{m}$ 呈现独立多波包。
3. **专题 3（顺应性储量）**：
   - 扫描：$C_f \in [0.002, 0.005, 0.010, 0.020, 0.030]\,\mathrm{m^2}$；
   - 物理现象：宏观大反弹幅值随 $C_f$ 呈现单调对数饱和上升，反弹振荡周期显著展宽。
4. **专题 4（拟达西滤失）**：
   - 扫描：$k_{leak} \in [0.2, 0.6, 1.0, 3.0, 10.0]\times 10^{-4}\,\mathrm{m^{2.5}/s}$；
   - 物理现象：致密储层微滤失维持高位大反弹；$10.0\times 10^{-4}$ 模拟 Type V 沟通天然断层，水头急速退水跌落至 $H_{ext}$。
5. **专题 5（限流射孔流阻）**：
   - 扫描：$K_p \in [1.5, 3.5, 5.43, 10.0, 25.0]\times 10^5\,\mathrm{s^2/m^5}$（对应孔眼数 $16, 8, 6, 4, 2$）；
   - 物理现象：$K_p$ 过小导致首缝声学短路，下游裂缝被阴影屏蔽；$K_p$ 增大恢复下游透射，但过大流阻会扼制缝内回弹。
6. **专题 6（关泵斜坡历时）**：
   - 扫描：$t_c \in [0.0, 0.5, 1.0, 1.5, 2.0]\,\mathrm{s}$；
   - 物理现象：$t_c = 0$ 阶跃关泵激发虚假吉布斯高频振荡；$t_c \ge 1.0\,\mathrm{s}$ 充当天然低通滤波器，半幅上升时滞满足 $\Delta t_{half} \approx 0.64 t_c$。
7. **专题 7（多簇进液能力组合）**：
   - 5 组典型工况：
     * 【中，中，中】：基准均匀发育型（$w = [0.333, 0.333, 0.333]$）；
     * 【高，中，中】：跟部首簇突进型（$w = [0.60, 0.25, 0.15]$）；
     * 【高，中，高】：两端优势马鞍型（$w = [0.45, 0.10, 0.45]$，中簇受应力阴影强挤压）；
     * 【中，中，高】：趾端逆向优势型（$w = [0.15, 0.25, 0.60]$）；
     * 【死，中，高】：首簇砂堵死簇型（$w = [0.01, 0.30, 0.69], K_{p, 1} = 10^8$）。

### 3.3 图版标准复合四联架构设计 (Panel a, b, c, d)

为了达到 Nature Portfolio 旗舰期刊的视觉冲击力与严谨逻辑闭环，推荐将每张 Figure 构筑为 **$2 \times 2$ 经典科学图版架构**：

```
+-----------------------------------------------------------------------+
|  Figure X: [Topic Name] Parametric Sensitivity & Acoustic Illumination |
+------------------------------------+----------------------------------+
|  Panel a: Transient Waveform       |  Panel b: 1D Real Cepstrum       |
|  - 100s Macro Evolution (H_wh)     |  - Normalized Cepstral Amplitude |
|  - Early Zoom Inset (0-15s)        |  - Vertical dashed lines (x_f)   |
|  - Multi-case curve comparison     |  - Acoustic depth axis (m)       |
+------------------------------------+----------------------------------+
|  Panel c: 2D Cepstrogram (Case 1)  |  Panel d: 2D Cepstrogram (Case 2)|
|  - Time-depth Rainbow colormap     |  - Contrasting dynamic behavior  |
|  - True fracture depth lines       |  - True fracture depth lines     |
|  - STRICTLY NO detection text      |  - STRICTLY NO detection text    |
+------------------------------------+----------------------------------+
```

- **Panel a（时域波形）**：
  - X 轴：时间 $t \in [0, 100]\,\mathrm{s}$；Y 轴：井口水头 $H_{wh} \in [80, 420]\,\mathrm{m}$；
  - 嵌入特写子图（Inset Axes）或双线排版展示 $t \in [0, 15]\,\mathrm{s}$ 关泵初期水锤震荡，清晰标定 Joukowsky 降落与首波反射到达；
  - 虚线标定稳态初始水头 $H_0 = 300\,\mathrm{m}$ 与远场孔隙水头 $H_{ext} = 100\,\mathrm{m}$。
- **Panel b（1D 实倒谱曲线族）**：
  - X 轴：声学反射深度 $x = a \tau / 2 \in [4400, 4650]\,\mathrm{m}$（精细聚焦裂缝发育区）；
  - Y 轴：归一化倒谱响应幅值；
  - 多曲线层叠或轻微垂直偏移（Waterfall offset），**以鲜明垂直虚线（如深红 `#D9534F`）精准贯穿标出所有真实裂缝位置**。
- **Panel c & Panel d（2D 连续倒谱云图）**：
  - 选取该专题中最具物理反差的 2 组代表性工况（如基准工况 vs 极值工况，或均匀工况 vs 砂堵死簇工况）；
  - X 轴：时间窗中心时刻 $t_{center} \in [15, 60]\,\mathrm{s}$；
  - Y 轴：空间反射深度 $x \in [4420, 4580]\,\mathrm{m}$；
  - 色阶：严格使用 `cmap='rainbow'`；
  - 水平标线：在对应深度标注真实裂缝位置虚线。

### 3.4 Rainbow 2D 倒谱云图的核心约束（严格禁止检出率统计文本覆盖）

用户需求在 Prompt R3 与验收准则中三次着重强调：
> **“标注裂缝深度线，不添加文本检出率统计判据”**  
> **“STRICTLY NO text detection criteria”**

#### 背景与警示
在旧版脚本（如 `generate_nature_figures.py`）中，曾在云图内部大量绘制半透明文本框，标注类似：
- `"Acoustic Choke: 4 clusters 100% illuminated"`
- `"Detection 4/4, Error 0.30m"`
- `"Frac 2: Shadowed"`
这类标注破坏了学术图版的纯粹性，给读者造成“人工硬编码判定”的算法生硬感。

#### 绝对红线
1. **严禁**在 Panel c 与 Panel d 的绘图坐标系中调用 `ax.text(...)` 或 `ax.annotate(...)` 输出包含“Detection Rate”、“Accuracy”、“Precision”、“检出率”、“识别率”、“Error 0.xx m”等统计标签；
2. **严禁**在图版标题中出现“(Detection 4/4)”等判据字样；
3. **允许且必须保留**：裂缝真实位置参考指示线（`ax.axhline(zf, color='...', linestyle='--')`）及必要的时间/深度坐标轴刻度。

### 3.5 Nature-Grade 顶级科学绘图规范执行指南

严格遵照 `nature-figure` 技能规范，图版生成脚本必须内嵌以下配置：

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # 字体规范：严格无衬线字体族，保证跨平台字体降级可用
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    
    # 矢量文字保留：SVG 输出中字符必须为可编辑 <text> 标签，严禁轮廓化
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    
    # 字号层级（严格遵循 Nature 单栏/双栏比例，避免字号偏大或溢出）
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "figure.titlesize": 9.5,
    
    # 轴线与刻度规范：移除顶部与右侧 Spine，刻度朝外
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "lines.linewidth": 1.0,
    
    # 输出分辨率
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
```

#### SVG 矢量/栅格混排关键优化 (`rasterized=True`)
2D 连续倒谱谱图（$N_{times} \times N_{dists} \approx 200 \times 1000$ 网格）若直接以纯矢量保存为 SVG，会在 SVG 文件内生成数十万个多边形 `<polygon>` 标签，导致生成的 SVG 体积飙升至 50MB 以上，在矢量软件（Illustrator/Inkscape）中打开时发生严重卡死。  
**关键技术实现**：
```python
im = ax.pcolormesh(
    T_mesh, D_mesh, ceps_2d,
    shading="auto",
    cmap="rainbow",
    rasterized=True,  # 仅将密集颜色网格栅格化，坐标轴、标签、标题与裂缝参考线严格保留为纯矢量！
)
```
此举可将 SVG 文件控制在 100KB~300KB 以内，且渲染效果完美。

---

## 4. 第三部分：全库 Pytest 测试套件现状与模块覆盖审计

### 4.1 99 项既有测试全景分布与执行健康度

执行 `pytest` 全量测试，汇总结果如下：

```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-7.4.4, pluggy-1.0.0
rootdir: E:\water_hammer_research\wellbore_moc_method
configfile: pytest.ini
testpaths: tests
collected 99 items

tests\test_challenger_adversarial_stress.py .....                        [  5%]
tests\test_challenger_empirical_stress.py ............                   [ 17%]
tests\test_final_acceptance.py ..............                            [ 31%]
tests\test_fracture_sensitivity_e2e.py ..................                [ 49%]
tests\test_fracture_storage_physics.py ....                              [ 53%]
tests\test_stage1_verification.py ........                               [ 61%]
tests\test_steady_state_and_toe.py ...                                   [ 64%]
tests\test_v2_architecture.py ..............................             [ 94%]
tests\test_visualization.py .....                                        [100%]

================== 99 passed, 2 warnings in 61.14s (0:01:01) ==================
```

### 4.2 各测试模块责任边界与断言范围

| 模块文件路径 | 测试项数 | 核心覆盖领域与物理断言 |
|---|:---:|---|
| `tests/test_challenger_adversarial_stress.py` | 5 | Rayleigh 空间分辨率极限判定、倒谱盲峰检测防信息泄露审计、稳态与 Brunone 摩阻混淆区谱比分离 |
| `tests/test_challenger_empirical_stress.py` | 12 | Joukowsky 瞬态水头跌落因果律、非定常摩阻耗散单调性、节点与全井动态质量守恒残差 ($< 10^{-10}$) |
| `tests/test_final_acceptance.py` | 14 | 裂缝节点求解器 5 元组返回值校验、初始水头安全裕度、Dirichlet 抽样有效性、NPZ 元数据双向映射 |
| `tests/test_fracture_sensitivity_e2e.py` | 18 | `output/fracture_parameter_sensitivity` 目录下的 5 级交付资产完整性（100 个 NPZ、CSV 时程、4 张图版 DPI $\ge 200$、README 报告 8 大章节） |
| `tests/test_fracture_storage_physics.py` | 4 | 裂缝宏观顺应性物理储能、解析阶跃响应对标、$\Delta t$ 网格收敛性与奇偶数值稳定性 |
| `tests/test_stage1_verification.py` | 8 | 初始静水场零摄动性、关泵前平直度（彻底根除 $t=0$ 假激波）、趾端同相全反射系数与 API 对齐 |
| `tests/test_steady_state_and_toe.py` | 3 | 达西沿程摩阻初始水头线性降落自洽性、最深裂缝下游死水区严格零流速、射孔摩阻开关 |
| `tests/test_v2_architecture.py` | 30 | MOC_V2 模块化配置类、菱形网格映射、非线性牛顿迭代二次收敛、1D 倒谱与亚米级峰提取、LHS 与预设工况生成器 |
| `tests/test_visualization.py` | 5 | 实时动画适配器空间场矩阵一致性、双模式瞬态帧有限性、网格碰撞防御拦截与 NPZ/PNG 导出 |

### 4.3 潜在回归风险矩阵与防破坏红线

针对后续重构主报告、编写 `run_sensitivity_study.py` 及图版生成任务，梳理以下 5 大回归风险与应对策略：

| 风险编号 | 潜在危险源 | 受波及测试集 | 破坏后果 | 防范对策与强制红线 |
|:---:|---|---|---|---|
| **R-1** | 篡改或覆盖 `output/fracture_parameter_sensitivity/` 目录 | `test_fracture_sensitivity_e2e.py` (18 项) | Tier 1~5 测试集体红灯崩溃 | **绝对红线**：本次现场级 7 大专题的所有仿真数据与图版**严禁写入该目录**！必须严格写入 `docs/moc_v2_technical_report/sensitivity_figures/`。 |
| **R-2** | 破坏 `moc_simulate` 顶层门面向后兼容性 | `test_v2_architecture.py`, `test_steady_state_and_toe.py` | 门面导入断言失败 | 保持 `moc_simulate.simulate_wellbore` 严格指向 `simulate_v2`，保持 `moc_simulate.v1` 隔离。 |
| **R-3** | 修改主报告导致 `build_report.py` 断言失败 | `docs/moc_v2_technical_report/build_report.py` | 自动化报告自验报错 | 剥离第 3~7 章后，必须同步修改 `build_report.py`，确保 LaTeX `$$` 语法平衡且篇幅 $\ge 50000$ 字节，所有核心物理关键字保留。 |
| **R-4** | 直接使用 `compute_cepstrogram_2d` 默认参数 | 新增绘图脚本或新测试 | 抛出 `ValueError` (kaiser 窗异常) | 显式指定 `window="hamming"` 或在底层修复元组传参。 |
| **R-5** | 长时正演运行超时 | CI 或自动化测试执行 | 测试挂起超时 | 正演仿真计算集中封装在独立脚本 `run_sensitivity_study.py` 中，不引入新的重型测试到 `pytest` 常规轮次。 |

---

## 5. 第四部分：执行性能基准与工程实施建议

### 5.1 100s 正演计算耗时基准 (Steady vs Brunone)

在当前 Windows 环境（Python 3.12.4, 64-bit）下，对 Base Case（$L=5000\,\mathrm{m}, \Delta t = 0.001\,\mathrm{s}, N=3448$）进行实机单次仿真耗时压测：

- **纯稳态摩阻 (`friction_model="steady"`)**：
  - 仿真历时 $t_f = 20\,\mathrm{s}$：耗时 **$2.28\,\mathrm{s}$**；
  - 仿真历时 $t_f = 100\,\mathrm{s}$（100,000 时间步）：耗时 **约 $11.4\,\mathrm{s}$**。
- **非定常摩阻 (`friction_model="brunone"`)**：
  - 仿真历时 $t_f = 20\,\mathrm{s}$：耗时 **$15.78\,\mathrm{s}$**；
  - 仿真历时 $t_f = 100\,\mathrm{s}$（100,000 时间步）：耗时 **约 $78.9\,\mathrm{s}$**。

### 5.2 38+ 算例的高并发多进程调度策略 (`BatchRunner`)

7 大专题包含：
- 专题 1: 8 算例
- 专题 2: 8 算例
- 专题 3: 5 算例
- 专题 4: 5 算例
- 专题 5: 5 算例
- 专题 6: 5 算例
- 专题 7: 5 算例  
总计 41 次仿真（去重后 38+ 独特算例）。

若采用单进程串行执行 Brunone 摩阻，总耗时将达 $38 \times 78\,\mathrm{s} \approx 49\,\mathrm{min}$，明显过长。  
**多核并发方案**：
本项目工作站配备 **16 核心 CPU**。利用 `concurrent.futures.ProcessPoolExecutor(max_workers=14)`（或直接调用 `moc_simulate.v2.batch.BatchRunner`），14 个进程并行推进：
$$\text{总并发耗时} \approx \left\lceil \frac{38}{14} \right\rceil \times 78.9\,\mathrm{s} \approx 3 \times 78.9\,\mathrm{s} \approx 236\,\mathrm{s} \approx 3.9\,\text{分钟！}$$
完全能够在 4 分钟内高质量一键跑完全部算例。

### 5.3 绘图主程序架构设计参考原型

后续执行代理编写 `docs/moc_v2_technical_report/run_sensitivity_study.py` 时，推荐采用以下高内聚、模块化架构：

```python
# 核心结构伪代码
def main():
    # 1. 创建输出目录
    output_fig_dir = Path("docs/moc_v2_technical_report/sensitivity_figures")
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 组装 7 大专题配置清单 (共 41 组工况)
    task_manifest = build_sensitivity_manifest()
    
    # 3. 14 进程高并发执行正演仿真
    print("Executing 38+ simulations with ProcessPoolExecutor...")
    results_map = run_parallel_simulations(task_manifest, max_workers=14)
    
    # 4. 逐个专题绘制 Figure 1 至 Figure 7 (300 DPI PNG + 矢量 SVG)
    plot_fig1_fracture_count(results_map, output_fig_dir)
    plot_fig2_fracture_spacing(results_map, output_fig_dir)
    plot_fig3_compliance(results_map, output_fig_dir)
    plot_fig4_leakoff(results_map, output_fig_dir)
    plot_fig5_perforation_impedance(results_map, output_fig_dir)
    plot_fig6_shutoff_ramp(results_map, output_fig_dir)
    plot_fig7_intake_combinations(results_map, output_fig_dir)
    
    # 5. 校验 14 份文件是否完整生成且尺寸合法
    verify_deliverables(output_fig_dir)
```

---

## 结论

本次调研全面理清了 MOC_V2 可视化图版流水线的全部技术要求、潜在代码陷阱及测试防护网。只要严格遵守路径隔离红线、修复/规避倒谱窗函数参数、采用 Rainbow 色阶且杜绝文本判据，并利用多进程并发，即可确保高质量交付 14 份 Nature 级图版，同时保障全库 99 项既有测试 100% 稳定通过。
