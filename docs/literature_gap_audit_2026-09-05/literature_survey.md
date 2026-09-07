# 井筒–多簇裂缝水击动力学与智能算子反演：交叉验证后的最终文献调研报告

**核查日期：2026-09-05；本版已补入认证 WoS Core Collection 与 Elsevier ScienceDirect 检索。** 分区采用 JCR 的 JIF 学科分区，并记录页面年度；除历史刊名 JPSE 为 JCR 2023 外，本次已读取的刊物均为 JCR 2025。分区是期刊属性，引用次数是论文的数据库快照，两者均不替代技术证据。

本综述以 DOI、出版方摘要、WoS 记录及项目内已发表论文全文相互核对。下文明确区分“前人已经实现”“本报告由方程推导”“尚待验证的研究设计”；摘要没有提供的网格、训练样本和耗时不予补造。本次是有检索日志的批判性综述，尚非全库逐篇全文审查的系统综述。“本次未发现同时覆盖”不构成对 WoS、Scopus、CNKI 或万方的穷尽否定。

逐项来源见 [physics_evidence.md](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/physics_evidence.md)、[fracture_evidence.md](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/fracture_evidence.md)、[operator_evidence.md](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/operator_evidence.md)、[WoS/JCR 证据](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/wos_jcr_evidence.md) 和 [Elsevier 交叉核验](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/elsevier_metadata_audit.md)。早期审计中的“未确认”属于阶段性状态，以本报告补查后的处理为准。

## 交叉验证与压力测试结论

指定的 Stage 1 文档（`fable5.1（llmarea）-stage1_literature_survey_and_gaps.md`）提供了有价值的方程骨架和候选文献，但其中若干结论不能原样进入最终论文。按“原文断言 → 交叉证据 → 最终处理”审计如下：

| Stage 1 断言（原文行） | 交叉验证结果 | 最终处理 |
|---|---|---|
| `Cr=1` 是 MOC 稳定且无弥散的充要条件（约第131行） | `Cr=1` 是经典无插值特征网格的相容取值；一般显式/隐式格式的稳定范围取决于离散方案。 | 改为“无插值模板”，并给出 `Cr<1` 插值、分管时间步和隐式方案。 |
| Brunone 可用固定 `C*` 常数并称为权重模型（约第51–63行） | Ghidaoui/Bergant 体系区分 Zielke 历史卷积与 Brunone 瞬时加速度型；系数依流态、符号约定和标定而变。 | 删除未经核验的固定常数，保留两类模型及适用边界。 |
| “Ye et al., Water Research 2022 首次 PINN 水击”（约第179行） | 补查已由 Crossref、WoS 与 ScienceDirect 三方确认 Ye, Do, Zeng & Lambert (2022), DOI `10.1016/j.watres.2022.118828`；出版方摘要给出两次数值与一次实验验证。 | 恢复该文，纠正早期审计的漏检；保留“管路瞬变 PINN 先例”，不将“首次”扩大至全部水击问题。 |
| “FNO 在 delta 输入处必然 Gibbs 失效”“OOD 误差随 Wasserstein 距离超线性”（约第191–196行） | Lanthaler 等证明的是特定不连续解族的逼近下界；非线性重构、shift-DeepONet、clawNO 等已显示可缓解间断误差。未核实 Stage 1 所列 Bartolucci/Zhu/Benitez/Lippe 组合能支持这些普遍定律。 | 改成“标准连续算子定理不能直接覆盖未知位置跳跃；需做界面编码和 OOD 实验”。 |
| “没有任何纯 PINN 成功求解高频水击”（约第179行） | Ye 2022 已做管路瞬变重建；Fourier-feature PINN 已有波传播/反问题基准；2026 GEAW-PINN 已处理 Brunone 水击和系数反演。 | 管路 PINN 成功不等于任意高频、多簇问题已解；缺口限定为未知界面、记忆摩阻、稀疏后验和跨工况验证的交集。 |
| 多裂缝反演属于空白（约第274–279行） | Hu 2022 已有多裂缝体积等效与应力干涉模型；Liu 2023 MCMC、Deng/Sun 2025、Sun 2026 IJRMMS 等已有几何、数量、位置或事件诊断。 | Gap 改为逐簇动态阻抗的可辨识组合、稀疏观测后验和可核验的端到端速度。 |
| “真实现场必然存在时变波速、孔眼不可逆冲蚀和裂缝闭合反馈”（约第293–298行） | 这些是工程上合理的待验证机制，但在当前检索中没有一组统一水击数据证明它们在同一观测窗内都占主导。 | 作为候选非自治扩展与实验假设，不写成已证实普遍事实。 |

因此，最终稿删除了“完全空白、唯一处女地、必然崩塌、数小时至数天”等不可由已核验证据推出的绝对措辞；保留“本次检索未发现同时覆盖”的可审计表述。

## 一、物理系统与数学建模

### 1. 控制方程

对单相液体、低 Mach 数、圆管截面和可忽略管壁惯性的常用一维近似，取流量 `Q(x,t)`、平均速度 `V=Q/A`、测压管水头 `H=p/(rho g)+z`；井筒截面、波速和参考密度在每个管段内取常数，可写为

\[
 \frac{\partial Q}{\partial t}+gA\frac{\partial H}{\partial x}
 +\frac{fQ|Q|}{2DA}+J_{u,Q}=0,
 \qquad
 \frac{\partial H}{\partial t}+\frac{a^2}{gA}\frac{\partial Q}{\partial x}=0. \tag{1}
\]

这里 `A=pi D^2/4`，`f` 是 Darcy–Weisbach 摩阻因子，`J_{u,Q}` 是以流量变量写出的非定常摩阻源项，单位为 `m3 s-2`。用速度变量时，等价源项是 `J_{u,V}=J_{u,Q}/A`，单位 `m s-2`；若用水头坡降表示，必须再除以 `g`。由于 `gA H_x=(A/rho)p_x+gA z_x`，倾斜井段的重力已经包含在 `H_x` 中，不能再加一次 `gA sin(theta)`；只有改用压力变量时才显式写 `gA z_x`。被省略对流项相对声学主项的量级为 `V/a`，因此式(1)不是高速、强气液两相或显著管壁惯性情形的普适方程。非定常壁面剪应力与源项的换算为 `J_{u,Q}=4A tau_{w,u}/(rho D)`。

一维模型还需要检查有效信号频带内是否以所选轴对称管波模态为主。对这里的截面均匀长波近似，`f_sig D/a << 1` 是一个基本量级检查；不满足时应检查径向模态、管壁动态和流固耦合，而不是仅减小 MOC 步长。`f_sig` 是实际用于推断的信号频率，不是传感器采样率。

流体状态方程给出 `K_f=rho (partial p/partial rho)`。含可压缩管壁的 Korteweg 型波速可写成

\[
 a^2=\frac{K_e/\rho_e}{1+cK_eD/(Ee)}, \tag{2}
\]

其中 `E` 为管材杨氏模量、`e` 为壁厚，`c` 由泊松比和轴向约束决定，`K_e,rho_e` 在单相时分别退化为 `K_f,rho`。因此 `K` 与 `E` 都是 Pa，但前者是流体/混合物有效体积模量，后者是固体弹性模量，不能互换。经典依据为 Ghidaoui et al. (2005, DOI `10.1115/1.1828050`) 和 Chaudhry (2014, DOI `10.1007/978-1-4614-8538-4`)。

**历史型摩阻。** Zielke (1968, DOI `10.1115/1.3605049`) 从层流径向动量扩散得到壁面剪应力历史卷积。例如以速度写，

\[
 \tau_w(t)=\frac{4\nu\rho}{R}V(t)+\frac{2\nu\rho}{R}
 \int_0^t \frac{\partial V(s)}{\partial s}
 W_Z\!\left(\frac{\nu(t-s)}{R^2}\right)\,ds, \tag{3}
\]

`R=D/2`。一种严格级数为 `W_Z(theta)=sum_{n>=1} exp(-j_{2,n}^2 theta)`，`j_{2,n}` 是 `J_2` 的正根，首根平方约 `26.3744`；工程上常用五指数长时拟合和短时幂函数。积分下限为零隐含扰动前充分发展稳态，非稳态预历史必须保留或用递归记忆状态初始化。Vardy–Brown (1995, DOI `10.1080/00221689509498654`) 把权重函数扩展到湍流，但其参数依赖流态和等效黏性假设。

**瞬时加速度型摩阻。** Brunone–Golia–Greco 模型不是 Zielke 权重卷积，常见速度形式是

\[
 \frac{\partial H}{\partial x}+\frac{1}{g}\frac{\partial V}{\partial t}
 +\frac{fV|V|}{2gD}
 +\frac{k}{g}\left(\frac{\partial V}{\partial t}-a\frac{\partial V}{\partial x}\right)=0, \tag{4}
\]

`k` 为经验系数，逆流时需按 Pezzinga/Bergant 类修正。文献指出该附加项在波反射边界附近尤其影响衰减与相位；不能把 `k` 当普适常数，也不能把 Brunone 称为频率权重模型。Ghidaoui et al. (2005) 和 Bergant, Simpson & Vitkovsky (2001, DOI `10.1080/00221680109499828`) 是主要综述依据。

### 2. 多簇射孔/裂缝内边界

第 `j` 个簇位于 `x_j`，井筒左右流量为 `Q_j^- , Q_j^+`，从井筒流向第 `m` 个裂缝的流量为 `q_{j,m}`。节点质量守恒为

\[
 Q_j^- -Q_j^+=\sum_{m=1}^{M_j}q_{j,m}+q_{leak,j}. \tag{5}
\]

忽略节点局部惯性和主线局部损失时，井筒左右共享节点压力 `p_{w,j}`；射孔下游入口压力 `p_{in,j,m}` 与该压力存在节流压差：

\[
 p_{w,j}-p_{in,j,m}=\Delta p_{perf,jm}+\Delta p_{nw,jm},\qquad
 \Delta p_{perf,jm}=\frac{\rho}{2C_d^2A_{perf,jm}^2}q_{j,m}|q_{j,m}|. \tag{6}
\]

多个同质孔眼只有在同压降时才能先合并为总面积；否则必须逐孔求解后再求和。`Delta p_perf` 是二次非线性耗散，不应直接替换成线性电阻。

以下给出拓扑一致的最低阶集总闭合，用于统一端口和状态记号，不声称逐式复刻任一文献的全部 RCI 模型。令 `q_f` 为入口流入裂缝的流量、`p_c` 为裂缝储容压力、`p_res` 为恒定储层参考压力、`q_l` 为裂缝向储层的漏失，取

\[
 C_f\dot p_c=q_f-q_l,\qquad
 p_{in}-p_c=I_f\dot q_f+R_fq_f,\qquad
 q_l=G_l(p_c-p_{res}). \tag{7}
\]

其中 `C_f` (`m3 Pa-1`) 为有效储容，`I_f` (`Pa s2 m-3`) 为入口后流体惯性，`R_f` (`Pa s m-3`) 为入口后裂缝流动阻力，`G_l>=0` (`m3 Pa-1 s-1`) 为线性漏失导纳。已放入 `Delta p_nw` 的近井损失不得再次计入 `R_f`；已由 `q_f` 带入裂缝并计入式(7)的漏失，也不得作为同一流量再次加入式(5)，其中 `q_leak,j` 仅表示直接从井筒节点流失的独立支路。保留裂缝内流体压缩性时，最低阶有效储容为 `C_f=dV_geom/dp_c+V_f/K_f`；忽略流体压缩时才近似为几何体积顺应性。文献中单位为 `m Pa-1` 的法向顺应性还需按面积和几何换算，不能直接与集总 `C_f` 比较。

Holzhausen & Gooch (1985, SPE-13892-MS)、Paige et al. (1995, SPE-26525-PA) 建立并发展了 hydraulic-impedance/RCI 思路。Luo et al. (2023, SPE-214658-PA) 将多裂缝和射孔干涉纳入 MOC 衰减模型，所核验模型使用等效底部边界，其假设包括定高定长椭圆裂缝、短时无漏失。这里应区分“全部簇聚合至同一端口”与“每簇各有集总支路且由井筒传播边连接”：后者仍能保留簇间传播时延。

式(6)–(7)在稳态流量 `bar q_f` 附近、零扰动初值下的小信号端口阻抗为

\[
Z_b(s)=R_{perf}+R_{nw}+R_f+sI_f+\frac{1}{sC_f+G_l},\qquad
R_{perf}=\frac{\rho|\bar q_f|}{C_d^2A_{perf}^2},\quad
R_{nw}=\left.\frac{d\Delta p_{nw}}{dq_f}\right|_{\bar q_f}.
\]

其中 `R_nw` 仅在所选近井压降律于工作点可微时定义；对非可微压降律应保留非线性求解。该式明确区分串联入口损失/惯性与并联储容/漏失。若只采用 `q_f=C_f dot(p_c)+(p_c-p_res)/R_l` 且 `p_in=p_c`，支路导纳才是 `sC_f+1/R_l`；不能把同一个 `R` 同时用于串联损失与并联漏失。传播矩阵若状态向量为 `[H,Q]`，特征阻抗应取 `Z_H=a/(gA)`；`rho a/A` 只适用于 `[p,Q]` 压力–流量变量。

有限长裂缝若近似为均匀一维有损传输线，令单位长度串联阻抗为 `z_f(s)`、并联导纳为 `y_f(s)`，可写为

\[
 Z_f(s)=Z_{cf}(s)\frac{Z_L(s)+Z_{cf}(s)\tanh[\gamma_f(s)l]}
 {Z_{cf}(s)+Z_L(s)\tanh[\gamma_f(s)l]},\qquad
 \gamma_f(s)=\sqrt{z_f(s)y_f(s)},\quad
 Z_{cf}(s)=\sqrt{z_f(s)/y_f(s)}. \tag{8}
\]

`Z_L` 是尖端端接阻抗，根支须满足因果性和衰减方向。传播常数与特征阻抗须来自同一有损闭合；无损常参数情形 `z_f=sL_f'`、`y_f=sC_f'` 才有 `Z_cf=sqrt(L_f'/C_f')=rho_f a_f/A_f` 与 `gamma_f=s/a_f`。封闭端且 `|gamma_f l|<<1` 时 `Z_f≈1/[y_f(s)l]`，无漏失且 `y_f=sC_f'` 时才进一步退化为 `1/(sC_f'l)`。该式说明传输线的结构，不把二维/径向裂缝流固耦合等同于普通圆管。Bakku et al. (2013, DOI `10.1190/geo2012-0521.1`) 已由管波幅比和衰减估计频率相关裂缝顺应性、孔径和延伸长度；Liang et al. (2017, DOI `10.1190/geo2016-0480.1`) 用可压缩 Navier–Stokes 与固体弹性波耦合得到频变阻抗。因此缺口不是“无人研究频变裂缝阻抗”，而是本次已核验证据尚未同时展示其与长井筒、多簇射孔节流、非定常摩阻和井口稀疏反演的统一验证。

### 3. 反射/透射与 MOC

井筒特征阻抗 `Z_w=rho a/A_w`，局部支路等效阻抗 `Z_b`。射孔裂缝是 **shunt** 节点：压力连续、流量分流；若两侧主井筒阻抗相同，线性小扰动反射系数为 `Gamma_shunt=-Z_w/(2Z_b+Z_w)`，透射压力为 `1+Gamma_shunt`。只有主线末端唯一端接裂缝时，才使用 `Gamma_end=(Z_b-Z_w)/(Z_b+Z_w)`。截面/材料突变是 **series** 跃变，应使用两侧特征阻抗和压力/流量连续条件。多簇往返传播相位叠加，产生频率选择性的峰、谷和包络调制。频变线性时不变支路对应的是 `Gamma(s)`，不应简化为常数；非线性孔眼或真正时变的裂缝状态则需工作点线性化或局部冻结近似，不能由单一时不变传递函数精确覆盖。

沿 `C+/-` 特征线，经典无插值网格取 `Delta x=a Delta t`，即 `Cr=a Delta t/Delta x=1`。它是特征线与网格恰好重合的相容取值，不是所有 MOC 离散的普适稳定必要条件。多管路不同 `a` 时通常需要 `Cr<1` 插值、分管时间步、波速调整或隐式求解。若最短水力长度控制全局步长，`N_x≈L/Delta x`、`N_t≈T/Delta t`；朴素直接历史卷积的全程工作量为 `O(N_x N_t^2)`、记忆为 `O(N_x N_t)`，`N_exp` 项指数递归可分别降至 `O(N_x N_exp N_t)` 与 `O(N_x N_exp)`，但宽频拟合可能增加所需指数数目。这些是给定实现的复杂度，不能替代实测耗时。

令 `B=a/(gA)`、`S_Q=fQ|Q|/(2DA)+J_{u,Q}`。当 `S_Q` 作为离散源项且采用经典声学主部时，沿 `dx/dt=+a` 有 `d(H+BQ)/dt=-B S_Q`，沿 `dx/dt=-a` 有 `d(H-BQ)/dt=+B S_Q`。在无插值网格上，旧时层显式源项对应的完整离散形式为
\[
\begin{aligned}
C_P&=H_{i-1}^{n}+B Q_{i-1}^{n}-rQ_{i-1}^{n}|Q_{i-1}^{n}|-B\Delta t J_{u,Q,i-1}^{n},\\
C_M&=H_{i+1}^{n}-B Q_{i+1}^{n}+rQ_{i+1}^{n}|Q_{i+1}^{n}|+B\Delta t J_{u,Q,i+1}^{n},\\
H_i^{n+1}+B Q_i^{n+1}&=C_P,\qquad
H_i^{n+1}-B Q_i^{n+1}=C_M,\\
H_i^{n+1}&=(C_P+C_M)/2,\qquad
Q_i^{n+1}=(C_P-C_M)/(2B).
\end{aligned}
\]
其中 `r=f Delta x/(2gDA^2)`；忽略非定常摩阻时令 `J_{u,Q}=0`。第 `j` 个内节点使用左侧 `H_j+B_-Q_j^-=C_{P,-}`、右侧 `H_j-B_+Q_j^+=C_{M,+}`，再与式(5)–(7)联立。不同管段应使用各自 `B`；非线性源项的显式、半隐式或迭代处理会影响精度和稳定范围。

Zielke 卷积可用历史积分或递归状态求取。Brunone 项本身含 `V_t,V_x`，须声明滞后、迭代或并入主部的处理方式，不能无条件当作普通已知源项。例如，将式(4)按所写符号精确并入主部得到 `(1+k)V_t+gH_x-kaV_x+fV|V|/(2D)=0`；与连续性方程配对后，常数 `k` 下的特征速度为 `-a` 和 `a/(1+k)`。这一结果仅针对式(4)的具体变体，说明固定 `+/-a` 模板需要相应源项近似，不能概括所有 Brunone 修正式。

完整初边值问题还须给出 `H(x,0),Q(x,0)`、裂缝状态和摩阻预历史，初值应与节点方程相容。停泵问题可在井口给定 `Q(0,t)=Q_pump(t)`，封闭井趾/桥塞处给定 `Q(L,t)=0`；实际端接不同则使用对应压力或阻抗边界。每个亚声速外边界通常只给一个独立标量条件，不能同时任意指定压力和流量。

单点井口压力反演 `N_f,x_j,R_f,C_f,I_f,A_perf` 和漏失率是有限带宽、强阻尼和参数相关的非线性问题。对 Liu et al. (2023, ARMA-2023-0865) 的储量闭合 `d(2N_fh_fl_fw)/dt=Q-Q_leak`，若 `h_f` 给定、`w_f(t)` 采用同一外定函数且漏失闭合保持不变，变换 `(N_f,l_f)->(kN_f,l_f/k)` 保持这一储容贡献不变。但当弹性宽度、漏失面积、孔眼数或阻力也依赖几何时，该变换未必保持完整输出不变。因此这只是储容子模型的等效性和参数相关来源；完整模型的结构非唯一性必须检查全部闭合及允许激励，不能由一条质量方程直接推出，更不能归因于 MOC 本身。

### 4. WoS / Elsevier 补查后的直接技术脉络

**多裂缝传播与几何诊断已形成明确先例。** Minato & Ghose (2017), *Journal of Applied Physics* 121:104902，DOI `10.1063/1.4978250`，研究多裂缝与流体井孔相交时的管波生成和散射；Liang et al. (2017) 更明确把裂缝频变阻抗接入频域井筒管波反射/透射模型，包含井口骤停水击激励。因此“多裂缝散射”与“频变端口”均不是新概念。其与本文 1D 节点闭合的联系是端口压力–流量关系，不能直接认定管波弹性模型和工程 MOC 具有相同空间维数、频带及闭合假设。

Hu, Luo, Zhou, Qiu, Z. Li & Y. Li (2022), *Journal of Hydrology* 612:128240，DOI `10.1016/j.jhydrol.2022.128240`，已经把应力干涉引入多裂缝 RCI 诊断，通过总裂缝体积守恒构造等效关系，研究缝数、间距、体积敏感性，并与现场微地震几何结果比较。这是对“2023 年以前没有多裂缝水击几何反演”及“全体 RCI 都是单缝”的直接反证。其体积等效并不等于每簇频变导纳被独立测定。

**由主进液深度走向多进液簇与井下事件。** Dong et al. (2025), DOI `10.1016/j.geoen.2024.213556`，由频域共振频差提取反射时间，再结合波速估计进液深度；摘要为两口井、10 段数据，报告相关系数 99.6%，不能改写成相对误差 0.4%。Sun et al. (2026), *IJRMMS* 200:106437，DOI `10.1016/j.ijrmms.2026.106437`，通过复合滤波、倒谱和时间–深度转换识别段内多个进液簇，并诊断转向剂作用、桥塞泄漏/滑移及流体动态分配。合成和现场验证已说明“多簇识别完全空白”不成立；尚需区别“回波定位/事件判别”与“每簇阻抗、储容、孔眼面积的联合后验”。

**测量带宽与处理链必须进入逆问题。** Dong et al. (2026), DOI `10.1016/j.geoen.2025.214278`，比较七类滤波器及 200 Hz、1 kHz、2 kHz 采样对反射周期和峰衰减的影响。其案例把 50 Hz 识别为电源干扰、5–10 Hz 为停泵噪声、0–5 Hz 为有效水击信号；这些是该数据集的结论，不是所有长井筒的普适频带。Cheng et al. (2026), *Flow Measurement and Instrumentation* 111:103424，DOI `10.1016/j.flowmeasinst.2026.103424`，报告 10 kHz、0.02% 精度监测设备及约 ±0.15 m 的定位指标，并结合滤波、倒谱、云图和微地震/光纤对比。上述数值是作者摘要中的设备与应用结果，本报告未独立复现实验，也不能把它当作任意多簇波形的空间可辨识极限。

仪器采样率 `f_s`、有效信号带宽 `B_eff`、数值步长及最小可辨间距是四个不同量。提高 `f_s` 只改变数字采样上限；传感器响应、井筒耗散、激励带宽、信噪比及多重散射共同决定可用于反演的信息。简单孤立回波的双程延时为 `tau≈2x/a`，因此 `delta x≈(a/2)delta tau+(tau/2)delta a` 只是线性误差传播关系，不能在未知波速、重叠回波和强滤波条件下直接给出唯一簇位置。

## 二、PINN 与神经算子现状

### 1. PINN 训练病态及已经存在的水击反例

Raissi, Perdikaris & Karniadakis (2019, *JCP*, DOI `10.1016/j.jcp.2018.10.045`) 建立将观测误差与 PDE 残差共同训练的通用 PINN 框架。对单个工况，其基本目标是 `L=lambda_d L_data+lambda_r L_PDE+lambda_b L_BC+lambda_0 L_IC`；输出是该工况的解近似，参数化工况族的算子泛化并非普通 PINN 自动具备的性质。

Wang, Teng & Perdikaris (2021, DOI `10.1137/20M1318043`) 证明复合损失项反向梯度不平衡和数值刚性会造成 PINN 训练病态；Wang, Yu & Perdikaris (2022, DOI `10.1016/j.jcp.2021.110768`) 用 NTK 特征值解释不同损失项收敛速度差异；Wang, Sankaran & Perdikaris (2024, DOI `10.1016/j.cma.2024.116813`) 以因果损失权重改善多尺度/长时间动力学。Wang, Wang & Perdikaris (2021, DOI `10.1016/j.cma.2021.113938`) 还明确展示 Fourier features 可缓解高频波传播与反问题的谱偏差。因此不能写“纯 PINN 原理上不能求高频水击”；可辩护的命题是：在多重反射、频变记忆摩阻、未知内部节点和稀疏逆问题共同存在时，普通点采样残差会出现尺度失衡、时间因果不足和接口欠采样，需做专门验证。PINN 是连续函数近似，严格意义上的“数值弥散”不是固定网格格式的截断误差；这里应测量的是训练/外推造成的频率相关相位滞后、波包展宽和耗散偏差，并与 MOC 的离散色散分开报告。

Ye, Do, Zeng & Lambert (2022), *Water Research* 221:118828，研究在不掌握完整管网初边值模型时，由实测数据和瞬变方程重建未监测位置的压力；摘要明确涉及隐含边界和波衰减信息、两次数值与一次实验案例，以及传感器配置敏感性。这是比泛用波方程更接近本课题的直接先例。但摘要没有展示逐簇射孔入口、裂缝动态储容、变簇数算子族及其后验可辨识性，不能据此宣布多簇反演已完成。

### 2. FNO、DeepONet 与波场逆算子

FNO (Li et al., ICLR 2021；预印本 `2010.08895`) 用 Fourier 积分核学习参数化 PDE 解算子；DeepONet (Lu et al., 2021, DOI `10.1038/s42256-021-00302-5`) 用 branch/trunk 网络逼近函数到函数映射；Kovachki et al. (2023, *JMLR* 24(89):1–97) 给出连续算子的逼近与离散化不变性框架。其理论前提是所选函数拓扑中的连续算子。变量位置、高对比阻抗跳变若未作为输入函数/几何编码，不能直接套用连续算子定理。

令输入 `mu` 包含物性场、泵/阀操作函数、初值、摩阻预历史和有序节点集合 `{x_j,theta_j}`；输出 `u=(H,Q,p_c,q_f,摩阻状态)`。目标是 `G:mu -> u`，实际网络只看到输入的有限编码与有限训练样本。典型 FNO 层和 DeepONet 读出为

\[
v_{l+1}(\xi)=\sigma\!\left(W_lv_l(\xi)+\mathcal F^{-1}[R_l(k)\mathcal Fv_l(k)](\xi)\right),\qquad
\widehat{\mathcal G}(\mu)(\xi)=\sum_{r=1}^{p}b_r(\mathcal E\mu)t_r(\xi)+b_0. \tag{9}
\]

`xi` 为时空查询点，`R_l(k)` 为有限保留模态上的可学习矩阵，`E mu` 为传感点/参数/几何编码。FNO 使用该谱层堆叠再解码；DeepONet 的 `b_r,t_r` 分别来自 branch 与 trunk。无限维理论不意味着有限训练后能够解析任意高频或任意节点数量；截断模态数、输入采样和物理测量带宽均须单独报告。

**跳跃与算子连续性不能混为一谈。** 由简单示例即可直接计算：令 `chi_x(s)=1_{s>x}`，在有限区间内且 `x!=y` 时，

\[
\|\chi_x-\chi_y\|_{L^2(0,L)}=|x-y|^{1/2},\qquad
\|\chi_x-\chi_y\|_{L^\infty(0,L)}=1. \tag{9a}
\]

因此移动跳跃在 `L2` 下可以连续，在一致范数下却不连续。这是本报告的示例推导，不是某篇文献已经证明井筒解算子连续。实际工作应声明输入拓扑与输出范数，并将分段体场误差和节点左右迹值误差分别评价；仅有较小的全局 `L2` 误差不能保证零测度节点上的分流条件正确。变簇数、簇合并或阻抗趋于奇异时还需另外规定允许参数域，不能沿用固定拓扑的结论。

Zhu, Feng, Lin & Lu (2023), *CMAME* 416:116300，DOI `10.1016/j.cma.2023.116300`，提出 **Fourier-DeepONet**：以 FNO 为 DeepONet 解码器，将震源频率、位置纳入输入，并构建 FWI-F、FWI-L、FWI-FL 全波形反演数据集，测试 Gaussian 噪声与缺失道。该文已经针对操作输入变化和观测不完整开展泛化研究。因此可讨论的是“这些泛化机制能否迁移至非线性孔眼、摩阻记忆及井口单点观测”，而非泛称纯数据算子没有泛化研究。其所测试震源变化也不构成对任意 OOD 分布的保证。

对本课题，主井筒节点处压力可以连续而流量跳变，射孔两侧则有有限压差；这与在全部输出场中出现移动激波不是同一数学问题。标准平滑网络或谱层可能在未解析的界面附近平滑流量迹值，普通监督损失也不强制分流守恒。然而“可能违反”须通过残差、相位和 OOD 试验量化，不能写成所有 FNO/DeepONet 必然发生守恒崩塌。标注数据成本同样取决于学习目标、样本复杂度及物理约束，不能无依据指定必须需要多少万条 MOC 波形。

### 3. 物理信息算子与界面约束的融合

PI-DeepONet (Wang, Wang & Perdikaris, 2021, DOI `10.1126/sciadv.abi8605`) 用 PDE 残差降低配对数据需求；PINO 正式论文 (Li et al., 2024, DOI `10.1145/3648506`) 结合低分辨率数据与高分辨率残差；Rosofsky et al. (2023, DOI `10.1088/2632-2153/acd168`) 在 1D 波、Burgers 和浅水基准展示物理信息算子。它们尚未验证射孔节点质量守恒、二次压降、非局部 Zielke 记忆状态或 `Z_f(omega)` 跳跃条件。

不能忽略反例：Lanthaler et al. (arXiv `2210.01074`) 对不连续解给出线性重构的下界并展示 nonlinear reconstruction；clawNO (Liu et al., ICML 2024, arXiv `2312.11176`) 用结构设计自动满足连续方程；CPINN (Jagtap et al., 2020, DOI `10.1016/j.cma.2020.113028`) 已把守恒残差用于正/逆问题；XPINN 理论 (Hu et al., 2022, DOI `10.1137/21M1447039`) 明确子域分解有样本不足和过拟合权衡。故创新点应是特定井筒系统的联合拓扑、耗散、守恒和逆不确定度，而非泛称“首次物理约束”。

更直接的水击反例是 Li, Ren & Wang (2026, *Water* 18:1900, DOI `10.3390/w18151900`)：GEAW-PINN 在 Brunone 摩阻、管路/网络和 25 dB 噪声下自适应平衡方程损失，并训练/反演水击系数。摘要报告单管压力、速度相对误差约 0.00742 和 0.0183。它关闭了“Brunone 水击 PINN 无先例”的说法，但没有给出多簇射孔动态阻抗、未知簇位置或神经算子族泛化。

将上述残差思想用于本报告式(1)、(5)–(7)，可得到以下**待检验的研究设计**，不是声称某篇 PINO 论文已经实现井筒多裂缝耦合。各分段内部计算 `r_PDE`；每个节点分别计算左右迹值及

\[
\begin{aligned}
r_{m,j}&=Q_j^- -Q_j^+-\sum_m q_{j,m}-q_{leak,j}, &r_{p,j}&=p_j^- -p_j^+,\\
r_{o,jm}&=p_{w,j}-p_{in,jm}-\frac{\rho q_{jm}|q_{jm}|}{2C_{d,jm}^2A_{perf,jm}^2}-\Delta p_{nw,jm},\\
r_{C,jm}&=C_{f,jm}\dot p_{c,jm}-q_{jm}+G_{l,jm}(p_{c,jm}-p_{res}),\\
r_{I,jm}&=p_{in,jm}-p_{c,jm}-I_{f,jm}\dot q_{jm}-R_{f,jm}q_{jm}.
\end{aligned} \tag{10}
\]

先以各自的流量、压力尺度无量纲化，再训练

\[
\mathcal L=\lambda_d\mathcal L_{data}+\lambda_v\mathcal L_{PDE}
+\lambda_0\mathcal L_{IC}+\lambda_b\mathcal L_{BC}
+\sum_j\lambda_j\left(\|\widetilde r_{m,j}\|^2+\|\widetilde r_{p,j}\|^2+\sum_m[\|\widetilde r_{o,jm}\|^2+\|\widetilde r_{C,jm}\|^2+\|\widetilde r_{I,jm}\|^2]\right). \tag{11}
\]

局部节点在体积分域中为零测度集合，不能期待仅靠随机体残差采样自动识别其分流条件。井筒的带支路质量条件也不是直接套用无源激波 Rankine–Hugoniot 条件。软惩罚降低残差并不保证精确守恒；代数消元、保守数值通量或图节点硬约束仍应作为对比方案。

例如将式(3)中核函数在目标时窗近似为 `W_Z(nu t/R^2)≈sum_l alpha_l exp(-beta_l t)`，可引入 `dot eta_l=-beta_l eta_l+dot V`，从而 `tau_w,u≈(2rho nu/R)sum_l alpha_l eta_l`。`beta_l` 单位为 `s^-1`，预历史进入 `eta_l(0)`；这只是原有指数递归摩阻的可微状态实现，核拟合误差和网络误差应分开。不能以一个无记忆网络输出自动获得相同频率响应。

## 三、Benchmark Matrix

| 代表文献/学者体系 | 核心算法与建模架构 | 裂缝与井筒边界处理方式 | 适用场景与时空分辨率 | 正演耗时/反演能力 | 核心局限性 |
|---|---|---|---|---|---|
| Wylie–Streeter–Suo；Chaudhry；Ghidaoui et al. | 1D 可压缩水击方程 + MOC | 端部/支路边界；可扩展节点质量守恒 | 长管路；`Cr=1` 特征网格或插值 | 单次正演可靠；逐工况成本随 `N_xN_t` 增长 | 参数不确定度、频变摩阻记忆和多簇稀疏反演需额外处理 |
| Zielke；Vardy–Brown；Brunone | 历史卷积、指数记忆或瞬时加速度源项 | 摩阻进入特征源项，不自动给裂缝节点 | 层流/湍流瞬变；频率依赖耗散 | 递归实现较快；不提供逆问题唯一性 | 模型系数/预历史/流态适用范围不同，Brunone 与 Zielke 不能混称 |
| Qiu et al. 2022, DOI `10.1016/j.petrol.2022.110425` | 井筒–支管 MOC，时域/频域/倒谱分析 | 三管 junction 压力连续、流量守恒；一次一个支管裂缝 | 验证算例：10 m 井筒、`Delta x=0.01 m`、`a=1490 m/s`；由此算得 MOC `Delta t≈6.71 us`，不是仪器采样率 | 验证传播与干涉；未做逐簇参数后验 | 单支管理想化，未形成场尺度多簇空间算子 |
| Luo et al. 2023, SPE-214658-PA | MOC + RCI 等效底部边界 + 多裂缝/射孔衰减 | 多簇以等效集总 RCI；孔眼/迂曲损失 | 190 m 实验管、10 kHz 监测；场例长井段 | 以衰减和波形特征估计裂缝数 | `N_f` 不等于每个 `x_j` 的独立阻抗、顺应性和时延 |
| Liu et al. 2023, ARMA-2023-0865 | MOC + 射孔/漏失边界 + MCMC 后验 | 质量守恒、孔眼二次压降、RCI/几何和 Carter 漏失 | 约 5 簇合成和现场案例 | 有后验；真 `N=5` 时点估计 4，半长 SD 约 44.5 m | 等效几何参数存在相关，尚未证明逐簇位置/阻抗可辨识；会议论文，非端到端快速代理 |
| Zeng et al. 2024, DOI `10.3389/fenrg.2023.1336148` | RCI 瞬变模型 + 几何尺寸反演 | 裂缝 R/C/I 作用于等效边界 | 页岩储层场景 | 几何误差摘要约 6.28%，近井趾误差更高 | RCI 与真实几何映射不充分，未用神经算子 |
| Sheludko et al. 2024, SPE-217781-MS | 曲线拟合特征 + Random Forest 分类 | 无 PDE/节点损失；直接用井口特征 | 8 井 78 段、1 s 现场数据（方法部分另有 17/210 计数矛盾） | 阶段产能分类测试 accuracy 0.71/F1 0.72 | 1 Hz 丢失高频传播相位；不是逐簇阻抗反演 |
| Deng et al. 2025, DOI `10.1016/j.engfracmech.2025.111347` | CEEMD/倒谱定位 + MOC + 漏失/应力影 RCI | 可定位簇数/位置；多缝仍以等效 E-frac 并联 | 现场高频信号，保留约 0.1–20 Hz | 几何诊断与微地震比较 | 仍有定高平面裂缝、单簇等效和常物性假设 |
| Sun et al. 2025, SPE-228403-PA | 可压缩方程 + 管摩阻/孔眼/过滤 + 几何拟合 | 单一等效底部边界，含 `d(2N_fh_fl_fw)/dt` | 现场拟合 | 压力拟合约 0.5%，半长/高度与微地震误差 15.4/4.2 m | 不能从一个拟合曲线推出逐簇可辨识性 |
| Waqar et al. 2025, DOI `10.1016/j.ymssp.2024.111967` | domain-guided NN 输出可变数量 leak function | 管道泄漏函数，无压裂裂缝 RCI | 250 Hz、1200 点、6000 合成样本；实验/网络 | 训练 2 h 36 min（CPU），在线推理与正演加速未等同 | 可变异常数先例，但不是多簇裂缝、非线性孔眼或频变阻抗 |
| Li et al. FNO；Lu et al. 2021 DeepONet | 纯监督谱算子 / branch–trunk 算子 | 标准版本无显式 PDE 或射孔节点损失 | 原论文的参数化 PDE 基准；分辨率泛化有适用域 | 多查询推理优势；原文加速不能移植为本课题耗时 | 有限编码/训练分布不保证跨簇拓扑、守恒或后验校准 |
| Wang et al. 2021 PI-DeepONet；Li et al. 2024 PINO | 算子网络 + PDE 残差；PINO 可结合粗数据与细残差 | 标准 PDE 初边值约束；未验证式(10)全套端口 | 参数化 PDE / 波类基准；不是现场多簇井筒 | 降低配对标签依赖，支持多查询；仍需计入训练成本 | 软残差不是精确节点守恒，未知界面与记忆实现仍须设计 |
| Ye et al. 2022, Water Research | 观测 + 瞬变 PDE 的 PINN | 从局部观测学习隐含边界和阻尼 | 2 数值 + 1 实验；摘要未给统一分辨率/耗时 | 重建未监测位置压力，传感器配置敏感性 | 单工况场重建先例；未证明多簇参数后验或算子族泛化 |
| Zhu et al. 2023, Fourier-DeepONet | FNO 解码器 + DeepONet；震源条件输入 | 地震 FWI 边界与源函数，不是射孔分流端口 | FWI-F/L/FL，变频率/位置、噪声/缺失道 | 直接从波形反演地下结构；本次摘要未核端到端秒数 | 已有源条件泛化；仍未验证井口单点、摩阻记忆和孔眼非线性 |
| Hu et al. 2022, J Hydrology | 多缝 RCI + 应力干涉 + 体积守恒等效 | 多缝几何映射为等效响应 | 缝数/间距/体积参数分析；现场微地震对照 | 多裂缝几何评价；摘要未报统一耗时 | 体积等效不保证逐簇独立动态阻抗可辨识 |
| Sun et al. 2026, IJRMMS | 复合滤波 + 倒谱 + 时间–深度转换 | 识别多进液簇与井下事件的信号模型 | 合成及现场；摘要未给统一采样率 | 多簇定位、转向/桥塞事件及分配诊断 | 未建立每簇 R/C/I、摩阻及操作不确定度的联合后验 |
| Dong et al. 2025/2026, Geoenergy | 频差法定位；七类滤波性能评估 | 回波时间–深度关系，依赖波速 | FDM 两井10段；滤波200 Hz/1 kHz/2 kHz | 深度诊断；周期失真与噪声抑制评价 | 预处理和波速误差可能与几何误差混淆，不能只看拟合相关性 |
| Cheng et al. 2026, Flow Meas Instrum | 10 kHz 监测设备 + 信号处理 | 现场压力信号与多种诊断对照 | 摘要0.02%精度、约±0.15 m定位指标 | 已有高分辨率测量与应用；非算子训练计时 | 设备精度不等于强干涉条件下的结构可辨识性 |
| Li, Ren & Wang 2026, DOI `10.3390/w18151900` | GEAW-PINN + Brunone + 可训练系数 | 水库–管路–阀门/网络边界，未见多簇裂缝 | 25 dB 噪声，数值参考解 | 系数反演与状态预测；无算子族 OOD 报告 | 已关闭“Brunone PINN 无先例”，仍未覆盖多簇动态阻抗 |
| Zhu & Wang 2026, DOI `10.3390/modelling7030087` | Kalman 去噪 + 自动片段 + PSO 几何反演 | 高度依赖等效模型与片段特征 | 现场高频停泵信号 | 片段抽取 0.84–1.22 s；不是端到端反演 runtime | 无 PDE/jump loss、逐簇后验或 OOD 几何测试 |

矩阵中的“摘要未报”表示本次证据层级未取得该指标，不表示原论文必然没有报告。会议论文及必要基础文献保留用于 prior-art 判定，不因缺少 JCR 分区而抹去；期刊分区与引用快照另见第五节。不同论文的压力误差、几何误差、分类 F1 与设备精度不可直接横向排名。

## 四、可辩护的 SCI 1 区科研断层

以下四项是前述技术之间尚待解决的具体问题，不是把若干方法名称拼接后宣布首创，也不保证某个期刊的录用。顶刊价值取决于能否给出可复验的理论、方法或实验结论。

### Gap 1：非连续多簇边界的算子表征

已有 Liu/Sun/Deng 的质量守恒、孔眼压降和漏失闭合，也有 Bakku/Liang 的频变裂缝阻抗；本次已核验证据尚未同时展示这些要素在可学习解算子中按每个 `x_j` 独立耦合。将全部簇聚合至同一底部端口会损失簇间传播时延，而逐簇集总 RCI 与有限长度井筒传播边的组合仍可保留时延，必须按原文拓扑区分。研究切入点是把簇数量/布局、二次孔眼节流和动态端口作为算子输入，在未见布局上检验质量守恒、耗散和反射相位，并与逐簇 MOC、已有界面学习方法及删减模型比较。单纯组合已有要素不能自动构成顶刊创新；需要证明表示、误差控制或数据效率方面的可重复增益。

### Gap 2：频变摩阻与多重反射下的物理一致学习

Zielke/Vardy–Brown 是带历史状态的非局部耗散，Brunone 是瞬时加速度型；Fourier features、因果 PINN、NTK/GEAW 已分别缓解相关训练病态。高频多重反射把 PDE 残差、节点跳跃和记忆状态置于不同时间/频率尺度，均匀采样可能导致首波与小幅回波训练失衡，但这一具体机制仍需在井筒系统中通过梯度及分频段误差验证。可研究带递归摩阻状态的 PINO/PI-DeepONet、因果时间窗和界面采样，并以端口能量收支与整体被动性检验物理一致性。对有外部输入、裂缝储容/惯性和历史摩阻状态的系统，不能无条件强制观测场能量逐时刻不增；应核验输入功率、总储能变化及耗散，并计入记忆状态。其方法价值应由频谱、相位、能量平衡和适用参数域内的长时间稳定验证共同支撑。

### Gap 3：井口稀疏波形下的逐簇结构可辨识性

补查后的最近竞争工作包括 Hu 2022 的多裂缝等效几何反演、Dong 2025 的频差定位和 Sun 2026 的多进液簇/事件诊断。它们使本 Gap 必须聚焦于“定位之外，何种动态参数可由实际测量链识别”，不能将多簇位置识别本身列为无人涉足。

Liu 2023 已有 MCMC，Zeng 2024、Sun 2025、Yang 2026 和 Zhu 2026 已有几何/特征反演，因此“没有贝叶斯反演”或“没有自动几何诊断”均不成立。需进一步回答的是：单个井口传感器在有限带宽、未知摩阻和未知操作条件下，哪些逐簇 `x_j,N_{p,j},C_{f,j},R_{f,j},I_{f,j}` 或其组合能够被识别，哪些只能约束为等效量。结构等效变换应保持完整模型及允许激励下的输出不变；Fisher 信息矩阵可诊断局部相关，但满秩不证明全局唯一性，离散簇数和参数置换需另行分析。在此基础上结合多工况激励、可微 MOC 或物理神经算子开展概率反演，并报告先验敏感性、簇位置召回、可信区间覆盖及跨噪声/跨流体验证。可辨识边界和校准后的不确定度比单条压力拟合更能形成可检验的方法学贡献。

### Gap 4：从昂贵正演到可信极速代理的验证域

PINO/DeepONet 的快速多查询优势和 Waqar 2025 的 domain-guided leak function 表明代理加速可行；本次核验的压裂水击工作主要采用 MOC、信号处理、RCI 拟合或 MCMC。本次检索尚未确认同时满足以下条件的公开基准：长水平井、多个未知簇、频变摩阻、孔眼非线性、实验/现场观测、未见几何和未见泵停机工况，并同时核查质量守恒、节点跳跃、相位和后验不确定度。Sun 2026 的 ScienceDirect 页面标注数据可向作者申请，这与已经公开可下载的标准基准也有区别。在明确的基准上相对 MOC 报告端到端 wall-clock、误差、稳定性和 OOD 失效边界，才能检验智能算子的实用价值。

### 可检验的数学目标与验收设计

以式(1)、(5)–(7)给出的正演模型为基础，实际观测应写成

\[
y_{r,n}=\left[\mathcal M_{\psi}\mathcal G(\theta,\eta;u_r)\right](t_n)+\epsilon_{r,n},\qquad
\epsilon\sim\mathcal N(0,\Sigma). \tag{12}
\]

`theta` 为簇数/位置及端口参数；`eta` 为波速、摩阻等干扰参数；`u_r` 为第 `r` 个泵/阀激励；`M_psi` 显式包含井口取样、传感器传递函数和采用的滤波。误差模型在此作为研究假设，需要由数据残差检查。固定离散簇数与标签顺序时，令无量纲参数向量为 `vartheta`，局部灵敏度及 Fisher 矩阵为

\[
S_{(r,n),k}=\frac{\partial (\mathcal M_{\psi}\mathcal G)_{r,n}}{\partial\vartheta_k},\qquad
\mathcal I=S^T\Sigma^{-1}S. \tag{13}
\]

小奇异值提示一阶噪声放大，秩亏提示局部一阶退化；满秩仍不排除远处的等效解，秩亏也不能单独排除由更高阶项携带的辨识信息。未知摩阻、波速或传感器参数必须加入灵敏度分析或联合边缘化，不能固定这些量后声称现场全部参数均可唯一辨识。跨激励信息矩阵只在相应噪声独立假设下相加，增益应由最小奇异值、后验覆盖或实验验证评估。贝叶斯形式为

\[
\pi(\theta,\eta,\psi\mid y)\propto
\exp\!\left[-\tfrac12(y-\mathcal M_{\psi}\mathcal G)^T\Sigma^{-1}(y-\mathcal M_{\psi}\mathcal G)\right]
\pi(\theta,\eta,\psi). \tag{14}
\]

对孔眼，单一二次压降律只依赖乘积 `C_d A_perf`；没有额外信息时，两者分离需要新的观测或约束。该结论可由式(6)直接验证，比从波形拟合误差推断所有几何量恢复成功更严格。若用算子代理替代 `G`，代理误差应通过参考求解器复核或误差模型计入后验，不能在加速时静默丢弃。

能量验证也须闭合到端口。对静止参考附近、常数非负 `C_f,I_f,G_l,R_f` 及具有被动实现的摩阻模型，可采用

\[
E_{tot}=\frac12\int\!\left[\frac{\rho Q^2}{A}+\frac{A p'^2}{\rho a^2}\right]dx
+\sum_{jm}\left[\frac12C_{f,jm}p_{c,jm}'^2+\frac12I_{f,jm}q_{jm}^2\right]+E_{mem},
\quad \frac{dE_{tot}}{dt}=P_{ports}-\mathcal D. \tag{15}
\]

`p',p_c'` 是相对同一静止参考的压力扰动；`E_mem` 由所选历史摩阻的被动状态实现决定，不能任意设零。无其他能量输入时，孔眼耗散为 `Delta p_perf q=rho |q|^3/(2C_d^2 A_perf^2)>=0`，线性裂缝阻力和漏失耗散分别为 `R_f q^2`、`G_l p_c'^2`。只有把输入功率、全部储能和耗散闭合后才可要求能量非增；这不是对任意 Brunone 变体、流动参考态或时变裂缝的无条件定理。

| Gap | 已有工作为什么仍不足以回答本问题 | 可以形成高水平论文贡献的结果 | 必需的反证基线/验收 |
|---|---|---|---|
| 1：逐簇界面算子 | Hu 的体积等效、Minato/Liang 的散射与标准算子分别覆盖不同部分 | 在变簇数/布局和有限压降界面上给出稳定表征及可验证误差控制 | 逐簇 MOC、聚合 RCI、纯 FNO/DeepONet、界面分域模型；节点流量残差和复反射系数误差 |
| 2：记忆耗散一致性 | Fourier features、因果训练和 PINO 并未自动闭合多簇历史摩阻端口 | 分离核截断、求解器和学习误差，证明或实测长时程能量及相位一致性 | Zielke/Vardy–Brown/Brunone 分开；无记忆/有记忆、软/硬守恒消融；首波与弱回波分别评价 |
| 3：稀疏可辨识反演 | 已有多簇定位和 MCMC 不能替代未知物性、有限带宽下的逐簇辨识分析 | 给出可辨识参数组合和失效域，并得到先验敏感性受检验的后验 | MOC+优化/MCMC、频差/倒谱、Fourier-DeepONet 类逆模型；几何召回、置信区间覆盖、传感器/滤波敏感性 |
| 4：可信代理基准 | FWI-F/L/FL 已有源变化基准，水击监测设备与滤波研究也已存在 | 发布井筒多簇专用的正演/稀疏逆问题基准，含模型差异和真实测量链 | 按井/布局/工况划分训练测试，禁止相邻窗口泄漏；实验/现场外部验证与端到端计时 |

计算速度须区分 `T_label`、`T_train`、`T_infer`、反演迭代及预处理。对 `M` 次同类查询，只有在 `T_MOC>T_infer` 且

\[
M>\frac{T_{label}+T_{train}}{T_{MOC}-T_{infer}} \tag{16}
\]

时，计入训练成本的简化盈亏平衡才成立；多次训练、调参、GPU/CPU差异及后验采样另计。公式(16)是计时定义下的代数推导，不是本次测得的性能。传统 MOC 对单次 1D 正演可以很高效，算子的潜在价值主要在大量多工况查询或反复反演，不应以未测量的“数小时至数天”制造差距。

## 五、WoS / Elsevier 扩展覆盖与影响力核验

### 1. 检索范围及纳入原则

2026-09-05 在用户已认证的 WoS **Core Collection / All Editions / Fielded Search** 使用 Topic 字段检索，按 `Citations: highest first` 筛查强相关记录，再用两组 DOI OR 查询核验核心文献。主题查询的标准字段表达如下：

```text
TS=(("water hammer" OR "water-hammer" OR "hydraulic impedance" OR "tube wave*")
AND ("hydraulic fractur*" OR "fracture diagnostic*" OR "wellbore"))
```

页面返回 **74** 条。

```text
TS=(("physics-informed neural network*" OR "neural operator*" OR DeepONet)
AND (wave* OR "inverse problem*" OR gradient OR causality OR "tangent kernel"))
```

页面返回 **2,855** 条。ScienceDirect 使用 `"water hammer" AND "fracture"` 返回 **1,412** 个结果，其中筛选器显示 research articles 812、Engineering 735、Energy 311；后两者不是互斥集合。该宽检索也包含材料断裂和设备失效等旁支，不能把 1,412 当作井筒多簇研究论文数。补充按精确题名/DOI阅读 Ye 2022 和 Fourier-DeepONet 2023。

本次是相关性筛查加关键题录/摘要核验，未下载并全文审查上述全部记录。两组 DOI OR 检索分别是 13 个 DOI 返回 12 条、7 个 DOI 返回 5 条；Luo 2023、Sun 2025 SPE 及 Ghidaoui 2005 在对应查询中未显示，不等于论文不存在，仍依据出版方/项目论文保留。现有 SPE 论文的 JCR 分区和 WoS 次数在本次未取得，不予补造。中文作者的 SCI 论文已纳入，但 CNKI/万方中文期刊的系统检索尚未完成。

数据库还显示卷期为 **2026 年 11 月** 的井筒–地层停泵实验条目（PII `S2949891026002551`）。本次没有核实其正式在线发表日，故没有把该未来卷期条目用于截至 2026-09-05 的创新性否定。期刊名或“Full text access”标记本身也不能证明本次已阅读全文。

### 2. JCR 学科与年度表

以下数值均来自 WoS `View Journal Impact` 中的 **Journal Impact Factor Summary / Source: Journal Citation Reports**，采用 JIF 分区，不混用旁边的 JCI 表。均为页面标注的 SCIE 学科。**这是检索日页面所列年度，不声称是每篇论文发表当年的分区。** 学科原名与排名保留以方便复核。

| 期刊 | JCR 年度 / JIF | 学科、排名与分区 | 纳入作用 |
|---|---|---|---|
| Nature Machine Intelligence | 2025 / 29.8 | COMPUTER SCIENCE, ARTIFICIAL INTELLIGENCE：2/210，Q1；COMPUTER SCIENCE, INTERDISCIPLINARY APPLICATIONS：1/185，Q1 | DeepONet 理论与实证 |
| CMAME | 2025 / 7.6 | ENGINEERING, MULTIDISCIPLINARY：7/178，Q1；MATHEMATICS, INTERDISCIPLINARY APPLICATIONS：3/137，Q1；MECHANICS：9/172，Q1 | Fourier features、因果 PINN、cPINN、Fourier-DeepONet |
| Journal of Computational Physics | 2025 / 3.9 | PHYSICS, MATHEMATICAL：2/61，Q1；COMPUTER SCIENCE, INTERDISCIPLINARY APPLICATIONS：81/185，Q2 | 通用 PINN、NTK 训练动力学 |
| SIAM Journal on Scientific Computing | 2025 / 2.9 | MATHEMATICS, APPLIED：27/345，Q1 | 梯度病态与计算数学 |
| Science Advances | 2025 / 13.9 | MULTIDISCIPLINARY SCIENCES：12/140，Q1 | PI-DeepONet |
| Water Research | 2025 / 12.8 | ENGINEERING, ENVIRONMENTAL：3/88，Q1；ENVIRONMENTAL SCIENCES：15/395，Q1；WATER RESOURCES：2/132，Q1 | 管路瞬变 PINN 实验先例 |
| Journal of Hydrology | 2025 / 7.3 | ENGINEERING, CIVIL：17/193，Q1；GEOSCIENCES, MULTIDISCIPLINARY：17/259，Q1；WATER RESOURCES：10/132，Q1 | 多裂缝几何与应力干涉 |
| Geophysics | 2025 / 3.6 | GEOCHEMISTRY & GEOPHYSICS：22/103，Q1 | 裂缝共振、频变阻抗及管波 |
| Engineering Fracture Mechanics | 2025 / 6.2 | MECHANICS：14/172，Q1 | 自然裂缝与漏失水击诊断 |
| International Journal of Rock Mechanics and Mining Sciences | 2025 / 8.5 | ENGINEERING, GEOLOGICAL：4/68，Q1；MINING & MINERAL PROCESSING：4/35，Q1 | 多进液簇及井下事件 |
| Mechanical Systems and Signal Processing | 2025 / 10.2 | ENGINEERING, MECHANICAL：6/184，Q1 | 水力瞬变的可变数量泄漏反演 |
| Flow Measurement and Instrumentation | 2025 / 3.3 | ENGINEERING, MECHANICAL：55/184，Q2；INSTRUMENTS & INSTRUMENTATION：33/81，Q2 | 10 kHz 水击监测设备与测量链 |
| Geoenergy Science and Engineering | 2025 / 4.4 | ENGINEERING, PETROLEUM：6/22，Q2；ENERGY & FUELS：109/191，Q3 | 频差定位、滤波与水击现场研究；保留跨学科差异 |
| Journal of Petroleum Science and Engineering | **2023** / 4.7 | ENGINEERING, PETROLEUM：4/24，Q1；ENERGY & FUELS：77/171，Q2 | Qiu 2022 等历史刊名文献；不能套用 Geoenergy 2025 数据 |
| Journal of Applied Physics | 2025 / 2.7 | PHYSICS, APPLIED：107/191，Q3 | 必要的多裂缝散射先例，未列为 Q1/Q2 成果 |
| Journal of Hydraulic Research | 2025 / 1.7 | ENGINEERING, CIVIL：130/193，Q3；WATER RESOURCES：105/132，Q4 | Zielke/Brunone 比较的必要经典依据，未列为 Q1/Q2 成果 |

这次核验覆盖上表 16 个刊名，其中 14 个至少在一个所列学科/年度为 Q1/Q2。优先覆盖高水平期刊不应删除反驳新颖性的基础文献；分区较低也不使经典物理规律失效。SPE Journal、JMLR、PINO 正式刊物及 *Water* 等本次没有读取其分区，因此不在表中冒充已核验的 Q1/Q2。

### 3. 论文级 WoS 引用快照

以下均为 **WoS Core Collection 页面 Citations 数值，核验日 2026-09-05**，未排除作者自引，也不等同于 ESI Highly Cited Paper 认定。对最新直接相关论文不设统一高引用门槛；“未显示”不填零。更多完整题名、DOI 和 accession number 见独立 [WoS 证据文件](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/wos_jcr_evidence.md)。

| 论文 | WoS 次数 | 技术层级 / 使用边界 |
|---|---:|---|
| Raissi et al. 2019，PINNs，JCP | 15,165 | 高被引通用方法；不是多裂缝水击验证 |
| Lu et al. 2021，DeepONet，Nature Machine Intelligence | 2,731 | 高被引算子基础 |
| Wang et al. 2021，gradient-flow pathologies，SIAM JSC | 1,535 | 高被引训练病态依据 |
| Wang et al. 2022，NTK perspective，JCP | 1,132 | 高被引训练动力学依据 |
| Jagtap et al. 2020，cPINN，CMAME | 999 | 界面通量守恒先例 |
| Kovachki et al. 2023，Neural Operator，JMLR | 827 | 连续算子理论；本次 JCR 分区未核 |
| Wang et al. 2021，Fourier-feature eigenvector bias，CMAME | 548 | 高频/多尺度表示与缓解方法 |
| Bergant et al. 2001，unsteady friction modelling，J Hydraulic Research | 284 | 核心摩阻比较；分区按上表如实保留 |
| Wang et al. 2024，Respecting causality，CMAME | 215 | 因果训练方法 |
| Zhu et al. 2023，Fourier-DeepONet，CMAME | 93 | 震源条件变化、噪声及缺失道的逆算子 |
| Ye et al. 2022，hydraulic transient PINN，Water Research | 63 | 直接管路瞬变重建与实验 |
| Wang et al. 2021，PI-DeepONet，Science Advances | 42 | 按所读 WoS 数值记录，不替换为其他库统计 |
| Liang et al. 2017，Krauklis-wave / tube-wave diagnostics，Geophysics | 31 | 裂缝频变阻抗直接先例 |
| Qiu et al. 2022，time/frequency/quefrency，JPSE | 25 | 井筒–裂缝传播与倒谱 |
| Hu et al. 2022，multi-fractures geometry，J Hydrology | 19 | 多裂缝几何等效模型；不是同年 Hu 的 XPINN 论文 |
| Waqar et al. 2025，domain-guided ML，MSSP | 15 | 多泄漏函数反演的邻近先例 |
| Minato & Ghose 2017，多裂缝管波散射，J Applied Physics | 11 | 直接物理先例；不因引用较低而删除 |
| Dong et al. 2025，fluid entry depth / FDM，Geoenergy | 6 | 新近直接相关现场方法 |
| Deng et al. 2025，naturally fractured reservoir diagnosis，EFM | 3 | 新近直接相关多物理闭合 |
| Dong et al. 2026，filtering / reflection period，Geoenergy | 2 | 测量预处理与周期失真 |
| Sun et al. 2026，多进液簇与井下事件，IJRMMS | 未显示 | 最新直接相关，未称高被引 |
| Cheng et al. 2026，10 kHz 监测设备，Flow Meas Instrum | 未显示 | 最新测量技术，未称高被引 |

## 结论性判断

本次证据支持把 MOC 所保留的双曲传播、节点分流和耗散记忆转化为可学习、可校准、可量化不确定度的算子。具备进一步研究依据的方向是“显式逐簇界面 + 频变摩阻状态 + 稀疏观测结构可辨识性 + OOD 基准”的联合验证，其论文贡献仍须落实到误差控制、可辨识边界、数据效率或实验外部验证。公开 prior art 已反驳“首次水击 PINN”和“首次多裂缝反演”的宽泛声明；“FNO 自动解决间断”同样没有得到上述文献的普遍保证。

## 参考文献与核验入口

完整题名、作者、期刊/会议、卷页或文章号及可点击 DOI 见 [已核验参考文献表](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/bibliography_verified.md)。该表按 DOI 去重，并将会议论文、正式期刊论文和预印本分别标识；下列为主题索引，不代替完整题录。

经典水击：Ghidaoui et al. (2005) `10.1115/1.1828050`；Zielke (1968) `10.1115/1.3605049`；Bergant et al. (2001) `10.1080/00221680109499828`；Vardy & Brown (1995) `10.1080/00221689509498654`；Chaudhry (2014) `10.1007/978-1-4614-8538-4`。

裂缝水击：Qiu et al. (2022) `10.1016/j.petrol.2022.110425`；Luo et al. (2023) `10.2118/214658-PA`；Liu et al. (2023) `10.56952/ARMA-2023-0865`；Deng et al. (2025) `10.1016/j.engfracmech.2025.111347`；Sun et al. (2025) `10.2118/228403-PA`；Zhu & Wang (2026) `10.3390/modelling7030087`。

WoS / Elsevier 补充：Ye et al. (2022) `10.1016/j.watres.2022.118828`；Hu et al. (2022) `10.1016/j.jhydrol.2022.128240`；Sun et al. (2026) `10.1016/j.ijrmms.2026.106437`；Cheng et al. (2026) `10.1016/j.flowmeasinst.2026.103424`；Dong et al. (2025) `10.1016/j.geoen.2024.213556`；Dong et al. (2026) `10.1016/j.geoen.2025.214278`；Zhu et al. (2023, Fourier-DeepONet) `10.1016/j.cma.2023.116300`；Minato & Ghose (2017, JAP) `10.1063/1.4978250`；Liang et al. (2017) `10.1190/geo2016-0480.1`。

机器学习与算子：Wang et al. (2021) `10.1137/20M1318043`；Wang et al. (2022) `10.1016/j.jcp.2021.110768`；Wang et al. (2024) `10.1016/j.cma.2024.116813`；Wang et al. (2021 Fourier features) `10.1016/j.cma.2021.113938`；Lu et al. (2021 DeepONet) `10.1038/s42256-021-00302-5`；PI-DeepONet `10.1126/sciadv.abi8605`；PINO `10.1145/3648506`；CPINN `10.1016/j.cma.2020.113028`；XPINN `10.1137/21M1447039`；GEAW-PINN 水击 `10.3390/w18151900`。

## 检索与证据日志

2026-09-05 使用 Crossref REST 精确 DOI 检索、OpenAlex 公开元数据/摘要、arXiv、JMLR、PMLR 和项目内已发表论文 PDF 抽取。Crossref 对宽查询触发 HTTP 429 时，只保留已有 DOI 记录或公开摘要支持的条目。会议论文明确标记 proceedings；本地笔记仅用于发现线索，不作为独立发表证据。补充原始元数据脚本和响应见 [collect_public_metadata.ps1](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/collect_public_metadata.ps1) 与 `public_metadata.json`。

用户指定的 [grill-me 技能文件](C:/Users/Change/.agents/skills/grill-me/SKILL.md) 仅包含转调 `grilling` 的指令，当前会话未提供对应 Skill 工具；本报告实际执行的是逐条文献反证、量纲/边界核验和独立复核，审计结果保存在 [Stage 1 压力测试记录](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/grill_fable_audit.md)，不声称运行了缺失的技能。
