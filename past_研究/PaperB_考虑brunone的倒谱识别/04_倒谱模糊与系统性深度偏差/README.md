# 04 倒谱分析特性基础响应模块

## 模块定位
本模块深入解剖倒谱变换（Cepstral Analysis）作为一个已有估计器，在非稳态耗散子波输入下产生特征峰形态畸变与系统性测距偏差的物理与信号机理，并评估窗型、窗长与输入通道对倒谱测距的影响。

## 目录结构
- `./code/`:
  * `compute_cepstrum_clocks.py`: 统一倒谱变换与 $|C(\tau)|$ 极值拾取脚本；
  * `evaluate_p2_onset_verdict.py`: 倒谱与 Onset 对照判决分析脚本。
- `./data/`:
  * `four_clocks_n1_vs_n4.csv`: 四时钟与倒谱时延对照表（自 02 模块同步）；
  * `onset_correction_verdict.csv`: Onset 对照判决全表；
  * `Tc_sweep_n1_k0.01.csv`: 有限关井时间下的倒谱时延响应表；
  * `p5_model_comparison_clocks.csv`: 模型等级三角对照四时钟全表；
  * `p6_window_sensitivity.csv`: 窗型 (Hann, Hamming, Rect) $\times$ 窗长 (10, 30, 50s) 倒谱敏感性表；
  * `p6_input_channel_sensitivity.csv`: 输入通道 ($H$, $\mathrm{d}H/\mathrm{d}t$, 带通 $H$) 倒谱响应表；
  * `blind_protocol_summary.csv`: 盲协议复算统计总表。
- `./figures/`:
  * `cepstrum_profiles_proof.png`: 倒谱剖面验证图（证伪正旁瓣误抓，确认负谷翻转）；
  * `cepstrum_profiles_proof.svg`: 矢量图。

## 核心审定数据与三笔账分列核算
严禁将不同工况下的倒谱偏差混为一谈，必须严格分为三笔账核算：

1. **第 1 笔账（阶跃关井 + 弱常数 $k=0.01$ 单缝）**：
   - 倒谱时钟绝对误差 $\Delta x_{\text{cep}} = \mathbf{+8.58\,\mathrm{m}}$，相对稳态基准后移量 $\delta x_{\text{cep}} = \mathbf{+9.43\,\mathrm{m}}$（与峰时钟 $+9.43\,\mathrm{m}$ 完全一致）；
2. **第 2 笔账（现场斜坡关井 $T_c = 0.2 \sim 1.0\,\mathrm{s}$ + 弱常数 $k=0.01$）**：
   - 倒谱时钟相对稳态漂移量随关井时间延长单调收敛：$T_c=200\,\mathrm{ms}$ 时为 $\delta x_{\text{cep}} = \mathbf{+5.80\,\mathrm{m}}$，$T_c=1000\,\mathrm{ms}$ 时为 $\delta x_{\text{cep}} = \mathbf{+5.08\,\mathrm{m}}$；偏深幅度缩小但**未完全消失**，稳定在 $\mathbf{5 \sim 7\,\mathrm{m}}$；
3. **第 3 笔账（阶跃关井 + 动态 $k(Re)$ 单缝）**：
   - 历史 1D 实倒谱估计器偏差为 $\Delta x_{\text{cep}} = \mathbf{+11.80\,\mathrm{m}}$，冻结全谱 $|C(\tau)|$ 倒谱估计器偏差为 $\Delta x_{\text{cep}} = \mathbf{+18.73\,\mathrm{m}}$；**10–20 m 系统性偏深仅特指动态 $k(Re)$ 下的盲倒谱估计器**。

## 窗长与输入通道有效工作区间 (p6)
1. **窗长有效区间**：有效窗长要求 $T_{\text{win}} \ge \mathbf{30\,\mathrm{s}}$（Hann/Hamming/Rect 窗下 $\delta x_{\text{cep}}$ 稳定聚集于 $+9.43 \sim +10.15\,\mathrm{m}$）；$T_{\text{win}}=10\,\mathrm{s} < 4L/a \approx 13.8\,\mathrm{s}$ 短于一个井筒基周期时，Hann 窗两端平滑压平了关井阶跃与尾部振荡，导致同态周期信息不足而出现漂移（$\delta x_{\text{cep}} = +2.18\,\mathrm{m}$）。
2. **输入通道完全不变性**：$H(t)$、$\mathrm{d}H/\mathrm{d}t$ 及带通滤波水头 $H_{\text{bp}}(t)$ 三种通道提取的 $\delta x_{\text{cep}}$ 严格恒为 $\mathbf{+9.43\,\mathrm{m}}$。
