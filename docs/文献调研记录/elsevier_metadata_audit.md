# Elsevier 水击文献元数据交叉核验（2026-09-05）

**最终证据状态：已通过用户认证的 ScienceDirect 浏览器读取本文末节所列摘要。** 前两节保留早期公开 API 核验记录：Crossref REST `https://api.crossref.org/works/{DOI}`、OpenAlex REST `https://api.openalex.org/works/https://doi.org/{DOI}` 与 DOI 重定向（Elsevier PII）。当时摘要字段为空、非认证请求返回 HTTP 403，因此早期表中仅作题录判断；这些限制不代表后来认证浏览器仍无法读摘要。当前技术范围以末节逐篇摘要证据为准，仍不把摘要核验冒充全文审查。日期为数据库返回的卷期/出版日期；OpenAlex 引用计数不作为 WoS 引用数。

| 条目 | 核验作者 | DOI、PII 与日期 | 摘要证据与可用性判断 |
|---|---|---|---|
| *Evaluation of multi-fractures geometry based on water hammer signals: A new comprehensive model and field application* | Xiaodong Hu; Yinghao Luo; Fujian Zhou; Yang Qiu; Zhuolong Li; Yujiao Li | DOI `10.1016/j.jhydrol.2022.128240`; PII `S0022169422008125`; Journal of Hydrology 612, article 128240; Crossref 2022-09; DOI 解析到 Elsevier | Crossref/OpenAlex 摘要字段为空；仅可确认题名、作者、期刊、卷号和文章号。可作为“多裂缝几何水击模型/现场应用”文献，但不能声称其验证了特定 PINN/FNO 或给出未核验的精度数字。 |
| *Identification of fluid-entry clusters and diagnosis of downhole events based on high-frequency water hammer pressure* | Shuangshuang Sun; Yongming He; Lijun Liu; Yanchao Li; Longqing Zou; Liang Yang | DOI `10.1016/j.ijrmms.2026.106437`; PII `S1365160926000419`; International Journal of Rock Mechanics and Mining Sciences; Crossref 2026-04; DOI 解析到 Elsevier | 摘要未由 Crossref/OpenAlex 暴露，ScienceDirect 受 403。题名支持“高频水击压力识别入液簇/井下事件”的范围判断；不能把题名等同于已证明的多簇反演可辨识性或神经算子结果。 |
| *Development and effectiveness verification of ultra-high-frequency water-hammer wave monitoring equipment for large-scale fracturing of unconventional oil and gas* | Yijing Cheng; Jiaxuan You; Fengqi Guo; Jinancheng Wang; Lei Li; Jingping Zhu | DOI `10.1016/j.flowmeasinst.2026.103424`; PII `S0955598626002384`; Flow Measurement and Instrumentation 111, article 103424; Crossref 2026-09; DOI 解析到 Elsevier | 摘要字段为空，未取得开放摘要文本。可确认这是监测设备开发/有效性验证论文；不能据此断言采样率、SNR、现场井数或对裂缝参数反演能力。 |
| *The influence of filtering methods and parameters on reflection period from pump shut-in water hammer signals: A comprehensive study* | Xuelian Dong; Xingming Wang; Haiyan Zhu; Yuanyuan Yang; Le He; Ziping Liu; Wei Gong | DOI `10.1016/j.geoen.2025.214278`; PII `S2949891025006360`; Geoenergy Science and Engineering 257, article 214278; Crossref 2026-02（OpenAlex publication year 2025，属在线/卷期年份差异）; DOI 解析到 Elsevier | 摘要字段为空。题名足以支持“滤波方法/参数影响停泵水击反射周期”的主题归类；不得把其结论扩展为所有滤波器均导致某一固定偏差，除非直接阅读正文。 |
| *Research and application of hydraulic fracturing fluid entry depth detection method based on water-hammer signal* | Xuelian Dong; Haiyan Zhu; Xingming Wang; Le He; Ziping Liu; Wei Gong; Zhan Li; Jingran Tang; Xiangyi Yi | DOI `10.1016/j.geoen.2024.213556`; PII `S2949891024009266`; Geoenergy Science and Engineering 246, article 213556; Crossref 2025-03; DOI 解析到 Elsevier | 摘要字段为空。题名支持“基于水击信号的压裂液进入深度检测方法及应用”这一保守描述；不能无摘要证据地声称其实现逐簇导流能力反演、特定误差或实时性能。 |

## 结论及对 fable5.1 的修订建议

1. 上述五篇 Elsevier 文献均为真实 DOI/PII，可保留在水击诊断与仪器背景中；应按核验作者和 DOI 引用。
2. “Journal of Hydrology Sep 2022”对应第一篇（卷 612，article 128240）；“IJRMMS Apr 2026”对应第二篇（article 106437）；“Flow Measurement and Instrumentation Sep 2026”对应第三篇（volume 111，article 103424）；“Geoenergy Feb 2026”对应第四篇（online/卷期年份分别为 2025/2026）；第五篇 Crossref 日期为 2025-03。
3. 这些条目是工程水击信号、滤波、仪器和入液深度检测的 prior art，元数据没有证据表明它们采用 PINO、FNO、DeepONet 或给出“单点井口高频波形反演 50–100 维裂缝参数”的理论结果。fable5.1 中涉及 AI 算子空白的判断不能以这些论文作为直接依据。
4. 摘要未公开时，最终综述应明确标注“仅核验元数据/题名”，并把方法细节、样本规模、采样率、误差和现场结论留待获得全文后再写。

## 补充核验：Ye、Fourier-DeepONet 与 NIO

| 条目 | 核验结果 | 摘要/范围证据 |
|---|---|---|
| Ye, Do, Zeng & Lambert (2022), *Physics-informed neural networks for hydraulic transient analysis in pipeline systems* | **真实、应恢复**。Crossref DOI `10.1016/j.watres.2022.118828`; Water Research 221, article 118828; published 2022-08; PII `S0043135422007771`; authors Jiawei Ye, Nhu Cuong Do, Wei Zeng, Martin Lambert。 | Crossref 的参考文献与 Elsevier text-mining 链接可核验该文主题确为管道水力瞬变 PINN；Crossref `abstract` 字段为空，故不能逐句转述摘要。可保留为“简单管道水力瞬变 PINN prior art”，不能扩写为多簇裂缝结论。 |
| Zhu, Feng, Lin & Lu, *Fourier-DeepONet: Fourier-enhanced deep operator networks for full waveform inversion with improved accuracy, generalizability, and robustness* | **真实、DOI需纠正**。Crossref: Min Zhu, Shihang Feng, Youzuo Lin, Lu Lu; *Computer Methods in Applied Mechanics and Engineering*; DOI `10.1016/j.cma.2023.116300`; published 2023-11; PII `S0045782523004243`（用户给出的 PII 与 DOI 对应）。另有同题 SSRN DOI `10.2139/ssrn.4461079`。 | Crossref 未提供摘要字段；题名本身支持其用于 full-waveform inversion、Fourier-enhanced DeepONet 及 accuracy/generalizability/robustness 评估。不得将其结果等同于水击裂缝反演验证。 |
| Molinaro, Yang, Engquist & Mishra, *Neural Inverse Operators for Solving PDE Inverse Problems* | **公开真实的 arXiv 预印本；未核验到 Nature Machine Intelligence 正式版**。arXiv `2301.11167v2` (2023-06-03)，作者 Roberto Molinaro, Yunan Yang, Björn Engquist, Siddhartha Mishra；标题与摘要可由 `http://export.arxiv.org/api/query?search_query=all:%22neural%20inverse%20operator%22` 复核。 | arXiv 摘要明确：NIO 组合 DeepONet 与 FNO，学习 operator-to-function 的 PDE 逆映射，并报告较直接/PDE-constrained optimization 更快。Crossref 题名检索未找到同名 Nature Machine Intelligence DOI；最终稿应引用 arXiv，除非作者提供正式出版信息。 |

### 对原稿的直接改动建议

- 恢复 Ye 2022 Water Research 条目，但把“首次/明确指出致命瓶颈”等强结论改为该文实际覆盖范围内的描述，除非取得全文逐句核对。
- 将 Fourier-DeepONet 的错误/缺失 DOI 修为 `10.1016/j.cma.2023.116300`，并把它归入全波形反演算子方法，不作为水击领域实证。
- NIO 条目按 arXiv 预印本标注；“Nature Machine Intelligence 正式版”目前没有公开 DOI 证据，不应写入正式期刊文献表。

## 认证 ScienceDirect 摘要语义摘记（主线程页面读取）

以下内容来自主线程在认证 ScienceDirect 页面通过 PII 对应文章读取的摘要摘记；本子任务未逐篇阅读全文，因此仅作为摘要层证据，不替代全文方法和统计审计。

| 条目 | 摘要层可核验内容 | 使用边界 |
|---|---|---|
| Ye 2022 Water Research | PINN 同时使用观测数据与瞬变物理；针对不完整管网模型学习隐含边界和阻尼，并预测未监测点压力；包含 2 个数值案例和 1 个实验案例；分析传感器布置与超参数敏感性。 | 支持“管道水力瞬变 PINN prior art”；不能外推为多簇裂缝、频变卷积摩阻或井口反演结果。 |
| Fourier-DeepONet 2023 CMAME | 以 FNO 作为 DeepONet decoder；输入包括震源频率和位置；在 FWI-F、FWI-L、FWI-FL 三类数据集上测试 Gaussian 噪声、缺失道和震源噪声。 | 这是地震全波形反演算子学习，不是 water-hammer 研究；可用于算子架构先例，不能作为水击实证。 |
| Hu 2022 Journal of Hydrology | 多裂缝 RCI 扩展，考虑应力干扰和总裂缝体积守恒等效；开展多裂缝数、簇间距、体积敏感性分析，并与 microseismic 现场结果对比。 | 支持多裂缝水击工程模型 prior art；摘要未证明逐簇动态阻抗后验或神经算子反演。 |
| Sun 2026 IJRMMS | 复合滤波、时频分析、倒谱和时间-深度转换用于识别段内多进液簇与动态流体分布；现场事件原文为 `diverter effectiveness, plug leakage, and plug slippage`，即转向剂有效性、封隔塞泄漏及滑移，不能擅自改称“暂堵塞球”；含合成验证和现场案例。 | 摘要没有给出逐簇动态阻抗后验；不得把“识别事件/簇”写成连续参数可辨识性定理。[ScienceDirect 原文](https://www.sciencedirect.com/science/article/pii/S1365160926000419) 已于末轮重读，页面另列数据需向作者申请。 |
| Dong 2026 Filtering Geoenergy | 比较 7 种滤波：Median、Kalman、FIR、low-pass、Gaussian、Gaussian 一阶导数、Butterworth；案例含 50 Hz 电干扰、5–10 Hz 停泵噪声和 0–5 Hz 有效信号，并比较 200 Hz/1 kHz/2 kHz 采样。摘要称 Gaussian 属于幅度衰减/周期失真较小的方法之一。 | 不能将该案例的频带或 Gaussian 结论推广为所有井的固定频带/最优滤波器。 |
| Dong 2025 FDM Geoenergy | 井筒裂缝模型与现场验证；以频差法取得反射周期，再由回波时间和波速换算进液深度；2 口井、10 个段；摘要报告 correlation 99.6%。 | “correlation 99.6%”不能改写成“相对误差 0.4%”；摘要未建立信息论可辨识性极限。 |
| Cheng 2026 Flow Measurement and Instrumentation | 10 kHz 采样；摘要报告 accuracy 0.02%，并给出约 ±0.15 m 位置精度（1 kHz 约 ±1.5 m）；结合滤波、倒谱和 cloud map；与 microseismic、fiber-optic 结果对比，用于进液位置/体积、暂堵密封和裂缝沟通判断。 | 这些是设备及应用指标，不是 FIM、空间分辨率上界或信息论可辨识性证明；需在正文中核对 accuracy 定义和测试协议。 |

上述摘要摘记可支持工程背景和现有诊断能力的客观描述，但不能支撑 fable5.1 中尚未有 DOI/全文证据的 AI 算子理论归因。所有“首次”“完全空白”“理论上界”等措辞仍需独立文献检索或改为审慎的“尚未检出公开报道”。
