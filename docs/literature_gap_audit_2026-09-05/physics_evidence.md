# 井筒–多簇裂缝水击：物理与数值证据审计

## 1. 可核验的经典方程

### 1.1 低 Mach 数一维模型

在管径、材料和截面沿轴向分段常数，液体单相且管壁惯性可忽略时，常用守恒变量为流量 `Q`（m3 s-1）和压力水头 `H=p/(rho g)+z`（m）。Chaudhry（2014，DOI:10.1007/978-1-4614-8538-4）及 Ghidaoui *et al.*（2005，DOI:10.1115/1.1828050）给出的低 Mach 近似可写成

\[
 \partial_t Q+gA\partial_x H+\frac{fQ|Q|}{2DA}+J_u=0,
 \qquad
 \partial_t H+\frac{a^2}{gA}\partial_xQ=0.                 \tag{1}
\]

 `A=pi D^2/4`（m2），`f` 是 Darcy–Weisbach 摩阻因子（无量纲）。在式(1)的 `Q` 动量形式中，`J_u` 是非定常摩阻源项，单位 m3 s-2；若改写成平均速度 `V=Q/A` 方程，则应使用 `J_u/A`（m s-2），或在水头梯度形式中除以 `gA`。重力项可以写成 `gA sin(theta)`，水平段为零。被省略的对流项是 `Q/A * partial_x(Q/A)`，其相对量级为 `U/a=Mach <<1`，不是对所有气液两相或高速条件都成立的恒等式。

有效体积模量和波速应保持符号一致。流体状态方程为 `d p/d rho=K_f/rho`；含可压缩管壁的 Korteweg 型结果是

\[
 a^2=\frac{K_e/\rho_e}{1+cK_eD/(Ee)},
 \tag{2}
\]

其中 `E`（Pa）为管材杨氏模量、`e`（m）为壁厚、`D`（m）为内径，`c` 由轴向约束和 Poisson 比决定（Ghidaoui *et al.*, 2005：自由膨胀接头 `c=1`；全锚固等情形取相应约束系数）。单相液体时 `K_e=K_f`、`rho_e=rho`；混合物才可引入有效 `K_e,rho_e`。因此把 `E` 直接称作“流体有效体积模量”是量纲错误；`K` 与 `E` 都是 Pa，但物理对象不同。

### 1.2 非定常摩阻：Zielke 与 Brunone 不能混称

Zielke（1968，ASME *J. Basic Engineering* 90(1):109–115，DOI:10.1115/1.3605049）由层流径向动量扩散得到历史卷积。Ghidaoui *et al.* 的等价壁面剪应力形式为

\[
 \tau_w(t)=\frac{4\nu\rho}{R}V(t)+\frac{2\nu\rho}{R}
 \int_0^t \frac{\partial V(s)}{\partial s}
 W_Z\!\left(\frac{\nu(t-s)}{R^2}\right)ds,                 \tag{3}
\]

`R=D/2`、`nu`（m2 s-1）、`V=Q/A`（m s-1）。若将壁面项换成水头梯度，第二项系数可写成 `16 nu/(g D^2)`，但必须同时保持与核的无量纲自变量约定一致。严格无限级数常写为 `W_Z(theta)=sum_n exp(-j_{2,n}^2 theta)`，其中 `j_{2,n}` 为二阶 Bessel 函数 `J_2` 的正根，首根平方约 `26.3744`；工程实现通常采用 Zielke 的短时幂函数/长时五指数拟合（Ghidaoui *et al.*, 2005，Eq.39–40）。积分下限 0 隐含“扰动前为充分发展稳态”；任意非稳态预历史必须保留 `t<0` 记忆或初始化递归状态，不能无条件截断。

卷积直接实现需保存全部历史（内存和 CPU 随时间步增长）；Trikha/Suzuki 的指数递归和 Vardy–Brown 权重模型将其压缩为少数记忆变量。Vardy *et al.*（1993，*J. Hydraul. Res.* 31:533–548）和 Vardy & Brown（1995，DOI:10.1080/00221689509498654）针对低至高 Reynolds 数建立湍流权重近似。Vardy–Brown 依赖冻结/稳态湍流黏性等假设，不能声称对任意含颗粒压裂液普适。

Brunone–Golia–Greco（1991）则是瞬时加速度型模型，不是 Zielke 权重卷积。其常见压力水头写法为

\[
 \partial_xH+\frac1g\partial_tV+\frac{fV|V|}{2gD}
 +\frac{k}{g}\left(\partial_tV- a\,\partial_xV\right)=0,       \tag{4}
\]

并有 Pezzinga/Bergant 等对逆流方向的符号修正。`k` 为经验/模型系数，非普适常数；Ghidaoui *et al.* 指出其附加耗散主要在波反射边界体现。故报告中应将“Zielke/Vardy–Brown 历史型”和“Brunone 瞬时加速度型”列作并列备选，而非统称“Zielke 或 Brunone 权重模型”。

## 2. 多簇节点、射孔与裂缝阻抗

### 2.1 一般内节点条件

设井筒内第 `j` 个簇位于 `x_j`，井筒左右流量为 `Q_j^- , Q_j^+`，流入各裂缝支路流量为 `q_{j,m}`（从井筒指向裂缝为正）。节点质量守恒是

\[
 Q_j^- -Q_j^+ = \sum_{m=1}^{M_j}q_{j,m}+q_{\rm leak,j}.          \tag{5}
\]

若忽略节点局部惯性，节点压力只有一个值 `p_j`，但射孔和裂缝两侧压力不相等：

\[
 p_{w,j}-p_{f,j,m}=\Delta p_{\rm perf,jm}+\Delta p_{\rm nw,jm},
\quad
 \Delta p_{\rm perf,jm}=\frac{\rho}{2C_d^2A_{\rm perf,jm}^2}
 q_{j,m}|q_{j,m}| .                                             \tag{6}
\]

多个并联孔眼应使用各自 `A_perf` 后求和；把总孔面积和总流量直接代入只有在孔眼同质、同压降时才成立。局部孔眼压力降是二次非线性耗散，不能等同于线性电阻 `R q`。

### 2.2 裂缝顺应性、储集和惯性

最低阶集总模型可写成 `C_f= dV_f/dp_f`（m3 Pa-1）和

\[
 q_f=C_f\,\dot p_f+q_{\rm leak},
\qquad
 I_f\dot q_f+p_f-p_{\rm res}+R_fq_f=0,                    \tag{7}
\]

其中 `I_f` 的单位为 Pa s2 m-3，`R_f` 为 Pa s m-3；这是 Holzhausen & Gooch（1985，SPE-13892，DOI:10.2118/13892-MS）提出的阻抗思想，Paige *et al.*（1995，SPE-26525，DOI:10.2118/26525-PA）加入流体惯性，Mondal/Carey 等用于 RCI 井底边界。Luo *et al.*（2023，SPE Journal 28:1973–1985，SPE-214658）明确指出其等效 R 主要代表井筒近端/射孔/弯曲阻力，并假设短时无漏失、定长定高裂缝；该模型不是空间上每个簇的独立传输线。

对于有限长裂缝，不能把体积顺应性 `C_f` 与频变裂缝阻抗混为一谈。将裂缝近似为传输线时，可用启发性的

\[
 Z_f(s)=Z_{cf}\frac{Z_L+Z_{cf}\tanh[\gamma(s)l]}
 {Z_{cf}+Z_L\tanh[\gamma(s)l]},\quad
 Z_{cf}=\rho_f a_f/A_f,                                      \tag{8}
\]

其中 `gamma(s)` 为含黏性/弹性耗散的传播常数，`l` 为裂缝长度，`Z_L` 为尖端端接阻抗。低频封闭端极限才可约化为 `Z≈1/(sC_f)`（并联泄漏时再加导纳）。Bakku *et al.*（2013，*Geophysics* 78:D249–D260，DOI:10.1190/geo2012-0521.1）通过频率相关井筒管波估计裂缝顺应性、孔径和延伸长度，报告现场顺应性约 `10^-10–10^-9 m Pa^-1`，并强调过渡频率由黏性 skin depth 与孔径相当决定。Liang *et al.*（2017，*Geophysics* 82:D171–D186，DOI:10.1190/geo2016-0480.1）以可压缩 Navier–Stokes 与周围固体弹性波耦合求频变阻抗，再计算井筒管波反射/透射。故“文献尚未建立频变裂缝阻抗”是不准确的；更谨慎的缺口是尚无将其与长井筒、多簇射孔二次节流、非定常摩阻和稀疏井口反演统一起来的验证框架。

### 2.3 反射/透射的方向性

对线性化局部支路，井筒特征阻抗 `Z_w=rho a/A_w`，支路等效阻抗 `Z_b`。单一侧支是 **shunt（并联）**：节点压力连续、流量分流。若井筒两侧均有传播管段，节点方程给出
`Gamma_shunt=p_r/p_i=-Z_w/(2Z_b+Z_w)`，`p_t/p_i=1+Gamma_shunt`；这是并联导纳的结果。只有在主井筒在该处终止、裂缝作为唯一端接时，才可用单端终端式 `Gamma_end=(Z_b-Z_w)/(Z_b+Z_w)`（正方向约定会改变整体符号）。若裂缝段沿井筒串联改变截面或材料，则是 **series** 跃变，应使用两侧特征阻抗和压力/流量连续条件；不能把并联裂缝统一套串联公式。非线性孔眼压降和时变 `Z_b(s)` 会使上述常系数 Gamma 仅为小扰动近似。多簇位置之间往返传播相位叠加，产生频率选择性振荡；Luo *et al.*（2023）已用 MOC 加底部等效边界分析多裂缝/射孔衰减，Hu *et al.*（2022，*Journal of Hydrology* 612:128240，DOI:10.1016/j.jhydrol.2022.128240）用于多裂缝几何场例，但两者不等于一般空间分布、频变阻抗网络的解析解。

## 3. MOC 的稳定性、成本与可辨识性边界

沿 `C+/-` 特征线，固定网格无插值的经典网格取 `Delta x=a Delta t`，即 `Cr=a Delta t/Delta x=1`。这只是最简显式特征网格；Ghidaoui *et al.* 明确说明多管路统一 `Delta t` 时不同 `a,Delta x` 往往无法同时满足 `Cr=1`，实际用 `Cr<1` 加空间插值、分管时间步、波速调整或混合方案。因而“CFL=1 是 MOC 普适必要稳定条件”应改为“经典无插值格式的相容取值；一般显式格式要求按离散分析满足 `Cr` 稳定范围”。隐式中心差分可在 `Cr>1` 保持稳定，但代价是每步稀疏/非线性矩阵求解。

若最短水力长度控制全局步长，`N_x≈L/Delta x`、`N_t≈T/Delta t`，每步成本还包括节点非线性方程和摩阻记忆状态。直接 Zielke/Vardy 历史卷积额外存储 `O(N_xN_t)`；指数递归降至 `O(N_x N_exp)`，但高频宽带拟合需更多指数。长水平井、多个局部簇和高采样率会将正演成本放大，然而这只是计算瓶颈，不构成逆问题唯一性。

从井口单点 `p(t)` 反演簇数、位置、`R_f,C_f,I_f` 或 `Z_f(omega)` 是非线性、有限带宽、强阻尼下的病态问题：不同的射孔面积、近井扭曲阻力、裂缝顺应性、漏失和摩阻记忆可产生近似同一衰减/相位；未测的下井压力和簇间流量使参数相关性高。Luo *et al.*（2023）直接承认其 RCI 等效电阻在复杂多裂缝时物理意义不足；Deng *et al.*（2025，*Engineering Fracture Mechanics* 325:111347，DOI:10.1016/j.engfracmech.2025.111347）仍需 cepstrum/EMD 先估位置和数量，再用 MOC、漏失和应力阴影条件估计几何；Liu *et al.*（2023，ARMA-2023-0865，DOI:10.56952/arma-2023-0865）采用 MCMC 而非单次确定性拟合。这些证据支持“可辨识性与不确定度量化缺口”，但不支持“此前完全没有水击裂缝反演”。

## 4. 可用于综述的审慎研究断层

1. **拓扑–物理联合表征**：已有工作分别处理 RCI 集总端、有限频率裂缝阻抗或多簇衰减；缺少经实验校准的、可切换 shunt/series 拓扑、孔眼二次压降、簇间传播和 `Z_f(omega)` 同时存在的算子。
2. **宽频耗散的一致离散**：Zielke/Vardy–Brown 记忆和 Brunone 模型各有适用范围。如何在高频多重反射下保持能量耗散、因果性和离散守恒，并给出可证实的误差/稳定界，仍是数值与学习算法共同的难题。
3. **稀疏观测的可辨识性**：单点井口波形对位置、开启数、孔眼面积、`R/C/I`、漏失与摩阻存在等效变换。需要 Fisher 信息/后验不确定度、可辨识参数组合和多工况激励设计，而不是仅报告一条拟合曲线的误差。
4. **代理模型的验证域**：FNO/DeepONet/PINO 可降低重复正演成本，但现有公开水击裂缝研究仍以 MOC、信号处理或 MCMC 为主。真正有说服力的突破必须在未见簇位置、未见射孔方案、频变摩阻和实验/现场数据上同时验证守恒残差、反射相位及后验覆盖率。

## 5. 主要文献（DOI 已通过 CrossRef 核验）

- Ghidaoui, M. S., Zhao, M., McInnis, D. A., & Axworthy, D. H. (2005). *A Review of Water Hammer Theory and Practice*. **Applied Mechanics Reviews**, 58, 49–76. DOI:10.1115/1.1828050.
- Zielke, W. (1968). *Frequency-Dependent Friction in Transient Pipe Flow*. **ASME Journal of Basic Engineering**, 90, 109–115. DOI:10.1115/1.3605049.
- Bergant, A., Simpson, A. R., & Vitkovsky, J. (2001). *Developments in Unsteady Pipe Flow Friction Modelling*. **Journal of Hydraulic Research**, 39, 249–257. DOI:10.1080/00221680109499828.
- Vardy, A. E., & Brown, J. M. B. (1995). *Transient, Turbulent, Smooth Pipe Friction*. **Journal of Hydraulic Research**, 33, 435–456. DOI:10.1080/00221689509498654.
- Holzhausen, G. R., & Gooch, R. P. (1985). SPE-13892-MS. DOI:10.2118/13892-MS.
- Liang, C., O'Reilly, O., Dunham, E. M., & Moos, D. (2017). *Hydraulic fracture diagnostics from Krauklis-wave resonance and tube-wave reflections*. **Geophysics**, 82, D171–D186. DOI:10.1190/geo2016-0480.1.
- Bakku, S. K., Fehler, M., & Burns, D. (2013). *Fracture compliance estimation using borehole tube waves*. **Geophysics**, 78, D249–D260. DOI:10.1190/geo2012-0521.1.
- Luo, Y. *et al.* (2023). *A New Water Hammer Decay Model: Analyzing the Interference of Multiple Fractures and Perforations on Decay Rate*. **SPE Journal**, 28, 1973–1985. SPE-214658.
- Deng, S. *et al.* (2025). *A diagnostic model for hydraulic fracture in naturally fractured reservoir utilising water-hammer signal*. **Engineering Fracture Mechanics**, 325, 111347. DOI:10.1016/j.engfracmech.2025.111347.
- Wylie, E. B., Streeter, V. L., & Suo, L. (1993). *Fluid Transients in Systems*. Prentice Hall.
- Chaudhry, M. H. (2014). *Applied Hydraulic Transients*. Springer. DOI:10.1007/978-1-4614-8538-4.

本文件的方程、假设和局限均以项目内 Ghidaoui 综述及上述公开条目的可核验内容为依据；尚未将未检索到全文的模型细节扩写为“文献事实”。
