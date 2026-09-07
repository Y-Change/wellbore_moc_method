# 多簇井筒非局部水击倒谱响应：机理证伪与物理信息非循环重建框架

## 摘要

井口水击记录可以在无需井下阵列的条件下用于识别实际进液的射孔簇，但多簇井筒中的响应不是局部量。下游界面会改变上游表观峰；中等簇间距可能使后序峰的衰减强于更小或更大的间距；多次反射还会在未设计射孔的位置产生伪峰。本文采用一维特征线法（method of characteristics，MOC）和二维边际倒谱，对经过认证的 426 个独立拓扑网格进行定量分析。固定 \(X_1=3000\,\mathrm{m}\)、\(S=20\,\mathrm{m}\) 时，增加第二簇使首簇表观峰由 5.157 降至 2.705 a.u.，下降 47.5%。在 \(n=4\) 时，\(S=30\,\mathrm{m}\) 的第三、第四峰分别降至 0.536 和 0.324 a.u.，在 \(S=100\,\mathrm{m}\) 时恢复至 1.21 和 0.92 a.u.。总跨度同为 60 m 时，\(n=2,3,4,7\) 的末首峰比 \(R_{\mathrm{end}}\) 分别为 0.472、0.188、0.158 和 0.072。保存的 \(S=30\,\mathrm{m},n=4\) 连续剖面还在约 3120 m 和 3150 m 处出现与多路径时延一致的次级峰。

基于这些结果，本文提出四算子重建框架。名义射孔深度是已知工程先验，实际破裂并进液的簇集合是未知的非负状态。多路径剥离算子 \(G_{\mathrm{peel}}\) 仅作用于不与名义射孔重合的预测伪峰位置；传输补偿 \(K_{\mathrm{comp}}\) 使用由前向波动校准得到的有限增益 \([T_{\mathrm{eff}}^{2(k-1)}\eta(S)]^{-1}\)，不通过局部实测峰除法强行均衡；对拟合出的活跃分量施加核心宽度 0.80 m 的超高斯聚焦 \(F_{\mathrm{focus}}\)；连续掩膜 \(M(x)\) 报告绿色、黄色和红色证据带。非负活动量与有限增益共同保证零保持性：若某簇未进液，补偿不会凭空生成裂缝。四个极端几何验证了前向失真和现有 oracle 实现的适用边界，但 426 个峰值表本身不能证明盲恢复率或普适的伪峰抑制能力；这些指标需要具有独立激活状态和复相位信息的留出基准。

**关键词：** 水击；多簇压裂；特征线法；倒谱；多路径混响；物理信息反问题；不确定性掩膜

## 1 引言

多簇压裂阶段的效果取决于哪些设计簇真正破裂并接收流体。井下成像和分布式传感可以提供直接信息；已有管波研究表明，裂缝柔度、共振和反射也能从井筒波场中得到约束[1,2]。在成本和作业约束下，关井或停泵水击瞬态能够采样井筒中的阻抗变化，并可变换到时延或深度表示，因此具有潜在的非侵入式诊断价值[3,4]。近期研究已把这一思路推进到进液深度检测、簇级事件识别和超高频井口监测装备验证[5–7]。

常见解释把一个时延归给一个簇，并将相应倒谱峰幅值视为该簇响应的代理量，其倒频率解释源于同态倒谱框架[8,9]。该解释只有在网络耦合很弱时才近似成立。多簇井筒是连通界面网络：一个簇处的透射和反射会改变所有下游簇的入射波，也会改变井口接收到的返回波。因此，总跨度相同而界面数不同的网络可以有不同的末端响应；改变间距也可能改变波形幅值，而不只是平移到时[10–13]。这一“局部标签—网络观测”的差异见图 1。

本文回答两个相互独立的问题。第一，受控 MOC 网格能够否定哪些关于局部性、间距和总跨度的默认假设？第二，在给定射孔枪设计时，如何去除可预测的多路径贡献，而不预先假定哪些设计簇已经破裂？第二个问题被明确写成受约束反问题：名义位置已知，活化状态未知。把峰值直接插到所有名义位置会回答另一个问题，并形成循环论证。

本文使用项目中经认证的峰值表 `01_几何网格/峰值表/decay_table.csv` 以及 5.1--5.7 分析。独立单位是拓扑 MOC 网格，而不是单个时间采样或倒谱箱。该设计空间扫描是确定性的，没有现场噪声或重复实验，因此不报告 \(p\) 值。本文的贡献限定为一组可证伪的网络观测和一套可在未来盲基准上检验的非循环算子规范；近期贝叶斯诊断、裂缝尺寸反演、自然裂缝模型、连续小波衰减分析和自动参数反演研究提供了问题背景，但不替代本研究的前向证据[14–18]。

## 2 前向模型与观测算子

### 2.1 控制方程

井筒被表示为恒定截面、弱可压缩的一维管道。压力变量用水头 \(H(x,t)\) 表示，\(Q(x,t)\) 为体积流量；图 2(a) 给出簇节点的集总边界示意：

\[
\frac{\partial H}{\partial t}+\frac{a^2}{gA}\frac{\partial Q}{\partial x}=0,
\qquad
\frac{\partial Q}{\partial t}+gA\frac{\partial H}{\partial x}+
\frac{f}{2DA}Q|Q|=0 .
\tag{1}
\]

其中 \(a\) 为声速，\(A=\pi D^2/4\) 为截面积，\(D\) 为内径，\(g\) 为重力加速度，\(f\) 为 Darcy--Weisbach 摩阻系数。频变与工程非定常摩阻模型可见文献[19–21]。主基准采用定常摩阻，\(f\) 由初始雷诺数确定，不随瞬时速度更新；Brunone 非定常摩阻不在本文结论边界内。

在簇节点 \(x_k\) 处，质量守恒和集总裂缝/滤失关系写为（该类井筒—裂缝水击前向建模已有应用[10–13]）

\[
Q_{L,k}-Q_{R,k}=Q_{f,k},\qquad
Q_{f,k}=C_{f,k}\frac{\mathrm d H_{f,k}}{\mathrm dt}+
k_{\mathrm{leak},k}\sqrt{H_{f,k}-H_{\mathrm{ext}}} .
\tag{2}
\]

\(C_{f,k}\) 为集总柔度，\(k_{\mathrm{leak},k}\) 为滤失系数，\(H_{\mathrm{ext}}\) 为外部水头。基准中各簇共享 \(C_f\) 和 \(k_{\mathrm{leak}}\)，因此主要考察拓扑和间距，而非裂缝属性异质性。MOC 沿 \(\mathrm dx/\mathrm dt=\pm a\) 积分，Courant 数为 1。公共参数为 \(D=0.1397\,\mathrm{m}\)、\(a=1450\,\mathrm{m/s}\)、密度 \(1000\,\mathrm{kg/m^3}\)、运动黏度 \(10^{-6}\,\mathrm{m^2/s}\)、粗糙度 \(4.5\times10^{-5}\,\mathrm{m}\)、初始速度 \(1.0\,\mathrm{m/s}\)、初始水头 300 m、\(\Delta t=10^{-3}\,\mathrm{s}\)、\(C_f=10^{-5}\,\mathrm{m^2}\)、\(k_{\mathrm{leak}}=10^{-4}\,\mathrm{m^{5/2}/s}\)，停泵时刻为 1 s。

### 2.2 设计空间与独立网格

名义射孔深度为

\[
x_{\mathrm{perf},k}=X_1+(k-1)S,\qquad k=1,\ldots,n .
\tag{3}
\]

其中 \(X_1\in\{2000,2500,3000,3500,4000,4500\}\,\mathrm{m}\)，\(S\in\{10,20,\ldots,100\}\,\mathrm{m}\)，\(n\in\{1,\ldots,8\}\)。CSV 含 4320 条簇级记录，因为同时保存了定常和 Brunone 摩阻表，且单簇工况在不同间距标签下重复。定常摩阻拓扑分析中，\(n=2\)--8 提供 \(6\times7\times10=420\) 个网格，单簇基线贡献另外 6 个网格，总计 426 个独立拓扑。

### 2.3 倒谱观测

井口水头记录采用 30 s Hamming 窗和 5 s 步长变换；已有研究表明，滤波方法和参数会影响停泵水击反射周期的提取[22]。记所得二维边际倒谱场为 \(C(x,t)\)。图 2(b,c) 展示单簇与多簇剖面及峰值窗口。在基准分析中，第 \(i\) 个名义位置的表观峰按

\[
P_i=\max_{|x-x_i|\le r}\left[-\sum_t C(x,t)\right],
\qquad r=\min(15\,\mathrm m,0.49S) ,
\tag{4}
\]

提取，其中 \(x_i\) 是模拟几何深度。\(P_i\) 是局部表观响应，不是裂缝能量的直接测量。本文还使用 \(\alpha_i=P_i/P_1\)、\(R_{\mathrm{end}}=P_n/P_1\)、\(L_{\mathrm{span}}=(n-1)S\)，以及固定几何下的经验包络 \(P_i=A\exp[-\gamma(i-1)]\)。参数 \(\gamma\) 只是工况内包络参数，不等同于单界面透射系数。

## 3 机理证伪试验

图 2(c) 的窗口峰值提取使用模拟几何邻域，因此仅用于基准峰值提取，不构成盲定位精度验证。

### 3.1 下游反馈否定局部首峰标签

这一首峰反馈与图 6(a) 的簇数主效应相对应。

在 \(X_1=3000\,\mathrm{m}\)、\(S=20\,\mathrm{m}\) 时，单簇峰为 5.157 a.u.，增加第二簇后降至 2.705 a.u.，下降 47.5%。首峰因此依赖下游拓扑，即使其名义深度和局部簇参数不变。多簇网格中的首峰约在 2.45--3.55 a.u. 之间变化。这足以否定未经网络条件修正的单簇标定外推，但不意味着存在第二个普适常数。

### 3.2 间距同时改变幅值和时延

图 5 汇总了连续剖面、绝对峰值和相对响应的间距变化。

表 1 给出 \(X_1=3000\,\mathrm{m}\)、\(n=4\) 的定常摩阻峰值。

| \(S\) (m) | \(P_1\) | \(P_2\) | \(P_3\) | \(P_4\) |
|---:|---:|---:|---:|---:|
| 10 | 2.738 | 1.421 | 0.964 | 0.634 |
| 20 | 2.844 | 1.317 | 0.614 | 0.450 |
| 30 | 3.122 | 1.442 | 0.536 | 0.324 |
| 40 | 3.050 | 1.514 | 0.596 | 0.327 |
| 50 | 3.063 | 1.541 | 0.609 | 0.436 |
| 60 | 3.088 | 1.464 | 0.592 | 0.437 |
| 70 | 3.040 | 1.527 | 1.064 | 0.693 |
| 80 | 3.090 | 1.581 | 1.158 | 0.878 |
| 100 | 3.159 | 1.705 | 1.209 | 0.920 |

第三、第四簇在 20--60 m 的低响应以及更大间距的恢复不是简单的到时平移。双簇对照相对平坦：\(P_1\approx2.70\)--2.74 a.u.，\(P_2\approx1.24\)--1.30 a.u.。因此，\(n\ge3\) 才明显出现的谷值支持多界面网络交互。仅凭峰值表不能在相消、波包重叠和有限分析窗之间作唯一归因。

### 3.3 总跨度不是充分变量

图 8(a–c) 给出间距与簇数的交互相图，显示相同跨度下的路径并不等价。

倒谱时延和倒频解释的基础见文献[8,9]；本节的峰值仍是网络条件化的表观量。

当 \(L_{\mathrm{span}}=60\,\mathrm{m}\) 时，\(n=2,3,4,7\) 的 \(R_{\mathrm{end}}\) 分别为 0.4722、0.1877、0.1582 和 0.0716。相同总跨度不保证相同响应，界面数和间距必须显式保留。

### 3.4 深度不是可分离的标量增益

图 7、图 8(d–f) 和图 9 用主效应、联合相图及固定几何包络展示深度依赖性。

联合 \(X_1,S\) 分析显示，相对响应尤其是后序簇响应会随首簇深度变化。在 \(X_1=3000\,\mathrm{m}\) 时，\(n=3\)--8 的固定几何指数拟合平均 \(R^2=0.9864\)，但 \(\gamma\) 随间距改变；例如 \(n=8\) 时，\(S=30\,\mathrm{m}\) 的 \(\gamma=0.684\)，而 \(S=80\,\mathrm{m}\) 为 \(\gamma=0.294\)。较高的 \(R^2\) 只说明固定几何内峰列可被紧凑描述，不能证明存在普适空间衰减律。

## 4 物理信息重建

图 3 将四个算子表示为前向推导的模板和敏感度代理；其中的核函数和 \(M_{\mathrm{sens}}\) 不代表已估计的活化状态或真实性概率。

### 4.1 非循环反问题

重建输入为观测剖面 \(y(x)=P_{2D}(x)\) 及式（3）的工程先验。未知状态是非负活动向量 \(z=(z_1,\ldots,z_n)\)，其中 \(z_k=0\) 表示该簇不贡献进液响应。单位响应模板 \(h_k(x)\) 与多路径模板 \(g_m(x)\) 由同一 MOC 前向模型或独立校准的脉冲响应库生成。活动量和干扰伪峰系数通过

\[
\min_{z\ge0,\,c\ge0}
\left\|W^{1/2}\left[y-Hz-Gc\right]\right\|_2^2
+\lambda_z\|z\|_1+\lambda_c\|c\|_2^2
\tag{5}
\]

估计，其中 \(H=[h_1,\ldots,h_n]\)，\(G=[g_2,\ldots,g_M]\)，\(W\) 对可靠深度样本加权，\(c\) 为伪峰幅值。伪峰系数受前向路径上界约束，例如 \(0\le c_m\le\bar\rho_m\sum_k z_k\)，从而完全不活化的阶段不能生成仅由伪峰构成的裂缝剖面。正则参数必须在留出前向模拟上选择，不能根据目标剖面调节。

### 4.2 多路径剥离算子 \(G_{\mathrm{peel}}\)

多路径候选深度为

\[
x_{\mathrm{ghost},m}=X_1+mS,\qquad m\ge2,
\tag{6}
\]

并删除与名义射孔集合重合的位置。令 \(\kappa(u)=\exp(-u^2)\)，\(w_g\) 为校准伪峰宽度，\(\rho_m\in[0,0.95]\) 为由前向脉冲响应获得的路径系数，则

\[
G_{\mathrm{peel}}(x)=
\operatorname{clip}\left[1-
\sum_{m\in\mathcal G}\rho_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right),
G_{\min},1\right].
\tag{7}
\]

其中 \(G_{\min}\in[0,1)\) 是预注册的最低保留系数，并在留出正演上校准。算子抑制的是预测路径族，不宣称该深度的全部样本都是伪峰。与名义射孔重合的候选位置保留在活动拟合中，不进行凹口抑制。

### 4.3 传输与相位补偿 \(K_{\mathrm{comp}}\)

第 \(k\) 个名义簇的物理增益为

\[
g_k(S)=\frac{1}{T_{\mathrm{eff}}^{2(k-1)}\eta(S)},
\qquad
K_{\mathrm{comp}}(x)=1+\sum_{k=1}^{n}[g_k(S)-1]\omega_k(x),
\tag{8}
\]

其中 \(T_{\mathrm{eff}}\in(0,1]\) 来自前向传输校准，\(\eta(S)\in[\eta_{\min},1]\) 表示经校准的相消幅值。可采用物理相位模型

\[
\eta(S)=\operatorname{clip}\left(\left|1+R_{\mathrm{frac}}e^{-\mathrm i4\pi S/\lambda_0}\right|,\eta_{\min},1\right),
\tag{9}
\]

其中 \(R_{\mathrm{frac}}\) 为复反射系数，\(\lambda_0\) 为参考波长。\(\omega_k\) 以名义深度为中心并具有有限支撑。它不通过局部观测峰除法计算，也不把输出设为预先规定的目标幅值。重建簇系数为 \(\tilde z_k=g_kz_k\)，故只要增益有限，\(z_k=0\Rightarrow\tilde z_k=0\)；补偿不能凭空制造裂缝。图 3 采用固定的示意值 \(T_{\mathrm{eff}}=0.784\) 展示有限级联增益模板；该值未由 426 个峰值表识别，也不构成本文的参数估计结果。

### 4.4 超高斯聚焦 \(F_{\mathrm{focus}}\)

对 \(z_k\) 超过预注册活化阈值的分量，在名义窗口内进行局部模板拟合，得到中心 \(\hat x_k\)。核心聚焦核为

\[
F_{\mathrm{focus}}(x)=
\sum_{k:z_k>\tau_z}q_k
\exp\left[-\left(\frac{x-\hat x_k}{w_{\mathrm{core}}}\right)^4\right],
\qquad w_{\mathrm{core}}=0.80\,\mathrm m .
\tag{10}
\]

其中 \(q_k\) 是拟合活动置信量。聚焦只作用于拟合出的活跃分量，不使用真实状态标签。非活跃分量的 \(q_k=0\)，因此不会把噪声极大值变为裂缝。

### 4.5 连续真实性掩膜 \(M(x)\)

其中 \(w_t\)、\(u_k\) 和 \(v_m\) 的尺度与权重由留出正演和噪声扰动校准，并在测试前冻结。

令 \(u_k\) 综合归一化系数信号不确定度、与 \(h_k\) 的残差一致性和间距重叠惩罚，令 \(v_m\) 表示预测伪峰证据，则

\[
M(x)=\operatorname{clip}\left[
\sum_k u_k\tilde z_k\,\kappa\left(\frac{x-\hat x_k}{w_t}\right)
-\sum_{m\in\mathcal G}v_m\kappa\left(\frac{x-x_{\mathrm{ghost},m}}{w_g}\right),0,1\right].
\tag{11}
\]

颜色带是操作性标签：绿色 \(M\ge0.70\)，黄色 \(0.20\le M<0.70\)，红色 \(M<0.20\)。它们是该反问题的置信带，不是概率，也不是全油田检测限。最终剖面写为

\[
\tilde P(x)=M(x)G_{\mathrm{peel}}(x)K_{\mathrm{comp}}(x)
\left[\sum_k z_kh_k(x)\right]F_{\mathrm{focus}}(x).
\tag{12}
\]

部署时必须在查看测试状态之前固定 \(H,G,T_{\mathrm{eff}},\eta\)、阈值和不确定度校准。项目中的旧探索脚本把目标幅值除以局部原始峰，并用输入的真实位置构造 \(M\)，因此不作为本文验证实现。

## 5 四个极端工况验证

四个工况的原始剖面、名义先验和路径候选见图 4；该图不显示校正后的盲重建结果。

426 网格用于验证前向失真。下表将算子要求写成可复现实验准则，而不把使用 oracle 几何的现有输出宣称为盲重建精度。

| 工况 | 设计 | 前向证据 | 非循环检查 |
|---|---|---|---|
| 1 相消/伪峰陷阱 | \(X_1=3000\,\mathrm m,S=30\,\mathrm m,n=4\) | \(P=[3.122,1.442,0.536,0.324]\) a.u.；约 3119.2 和 3149.7 m 的次峰接近 3120、3150 m 预测深度 | 用名义模板拟合 \(z\)，只剥离非射孔伪峰，报告 \(z_3,z_4\) 支持度和残差 |
| 2 密集簇极限 | \(S=10\,\mathrm m,n=4\) | \(P=[2.738,1.421,0.964,0.634]\) a.u.；相邻窗口接近 30 s 分析响应尺度 | 报告系数协方差；不能仅凭设计位置声称四个独立峰，未分辨区域用黄色掩膜 |
| 3 深级联耗散 | \(S=20\,\mathrm m,n=8\) | \(P=[3.420,1.616,0.695,0.554,0.406,0.274,0.231,0.184]\) a.u.，\(R_{\mathrm{end}}=0.0537\) | 用独立传输校准得到有限 \(g_k\)，以未活化合成簇检验零保持性，并报告增益导致的不确定度膨胀 |
| 4 高间距原始基线 | \(S=80\,\mathrm m,n=4\) | \(P=[3.090,1.581,1.158,0.878]\) a.u.，\(R_{\mathrm{end}}=0.2841\) | 仅在独立校准预测较高传输时检验 \(K_{\mathrm{comp}}\) 是否接近 1，并检查掩膜是否产生额外峰 |

工况 1 的连续 MOC 剖面在约 3000.29、3029.29、3058.29、3088.75、3119.20 和 3149.65 m 处有峰。后两个峰接近多路径深度；其幅值约 0.223 和 0.140 a.u.，足以支持路径几何存在，但不足以声称所有伪峰都占主导，或固定 notch 在无损条件下总能去除它们。

旧探索实现的结果应作为负例保留：在提供真实位置时，\(S=30\,\mathrm m,n=4\) 的第三、第四真实位置值在阈值化后归零；\(S=20\,\mathrm m,n=8\) 的两个后序簇也会丢失。这是目标除法和 oracle 掩膜设计的结果，不纳入证据链。可发表的算子基准应在留出激活状态上报告精确率、召回率、簇数误差、定位误差、伪峰抑制比和系数不确定度，并在评估前冻结全部参数。

## 6 讨论与下游反演接口

图 10 的操作包络用于界定门限和可辨识度的适用范围，而不是现场检测概率。

前向网格改变了井口峰的解释对象：\(P_i\) 是网络条件化的表观响应，而不是第 \(i\) 簇的局部标签。47.5% 首峰变化、\(n\ge3\) 的间距谷值和等跨度曲线分离，共同与连通多界面系统相容，但不能唯一识别某一个反射系数或某一种相消机制。要作更强归因，需要复谱、路径消融和受控相位扰动。

式（5）把已知设计几何与未知进液活动分开，式（8）只恢复来自前向传输和相位因子的有限物理增益。目标均衡可能产生视觉上整齐的剖面，却掩盖未活化簇或放大伪峰；非负活动状态承担诊断决策，零状态在补偿后仍为零。

掩膜 \(M\) 是下游反演的决策辅助。只有在系数不确定度和残差检查满足预注册条件时，绿色区间才可传递给水力开度或进液阻力反演；黄色区间应作为备选状态传播，例如采样后验或报告 \(z_k\) 范围；红色区间应排除在定量反演外，但保留在审计记录中，因为红色可能来自预测伪峰，也可能来自信息不足。掩膜是证据记账工具，不是“发生破裂”的概率。

下游反问题可抽象写为 \(z_k=\mathcal H(w_{f,k},C_{f,k},k_{\mathrm{leak},k})\)，其中 \(w_{f,k}\) 为水力开度，\(\mathcal H\) 为前向校准映射。本文只提供带物理条件和不确定度的 \(z_k\)，不在裂缝属性固定的基准上识别 \(w_{f,k}\)。任何开度反演都必须独立评估属性扰动、波速误差、摩阻模型不确定度、现场噪声和偏离设计位置的簇。

本文存在几个实质限制：主表是确定性模拟而非现场重复；峰值提取搜索了真实邻域，不能证明盲定位精度；426 个网格没有覆盖所有部分破裂激活状态；保存的连续伪峰示例也未覆盖每个拓扑。经验指数包络虽在固定几何内具有较高 \(R^2\)，却不是普适传输定律。这些限制定义了下一步验证，而不削弱当前前向证伪结果。

## 7 结论

1. 下游簇会改变上游簇的表观响应。在 \(X_1=3000\,\mathrm m,S=20\,\mathrm m\) 时，\(n\) 从 1 变为 2，\(P_1\) 从 5.157 降至 2.705 a.u.。
2. 当 \(n\ge3\) 时，间距对后序簇幅值呈非单调调制。\(n=4,S=30\,\mathrm m\) 的 \(P_3/P_4=0.536/0.324\) a.u.，在 \(S=100\,\mathrm m\) 时为 1.209/0.920 a.u.。双簇对照近似平坦，支持多界面网络解释。
3. 总跨度不是充分描述量。总跨度 60 m 时，\(n=2,3,4,7\) 的 \(R_{\mathrm{end}}\) 为 0.472、0.188、0.158 和 0.072，标定必须保留间距和界面数。
4. 只有在名义射孔深度作为先验、进液活动作为非负未知量时，\(G_{\mathrm{peel}}\)-\(K_{\mathrm{comp}}\)-\(F_{\mathrm{focus}}\)-\(M\) 框架才是非循环的。有限物理增益保持零活动，不会制造假裂缝。
5. 四个极端几何构成可复现的盲基准协议。当前 426 网格验证了前向陷阱和部分伪峰深度，但没有证明 100% 恢复、普适伪峰抑制或定量开度反演；这些结论需要留出激活状态、独立的 \(T_{\mathrm{eff}}\) 与 \(\eta(S)\) 校准，以及不确定度感知的评估。

### 可复现性说明

前向证据位于 `01_几何网格/峰值表/decay_table.csv`；分析目录为 `5.1_间距S主效应分析`、`5.2_裂缝总数n主效应分析`、`5.3_首缝深度X1主效应分析`、`5.4_间距与缝数交互相图分析`、`5.5_深度与间距交互相图分析`、`5.6_空间衰减包络与模型拟合` 和 `5.7_可辨识度边界与可行域评估`。MOC 和倒谱设置见第 2 节。旧 oracle 探索脚本保留用于审计，但不是本文验证实现。正式发布版本还应包含式（5）的受约束求解器、前向生成的 \(H\) 与 \(G\) 模板库、留出激活基准以及全部掩膜和不确定度指标的机器可读表。

## 参考文献

[1] Bakku S.K., Fehler M.C., Burns D.R. Fracture compliance estimation using borehole tube waves. *Geophysics*, 2013, 78: D249–D260. DOI: 10.1190/geo2012-0521.1.

[2] Liang C., O'Reilly O., Dunham E.M., Moos D. Hydraulic fracture diagnostics from Krauklis-wave resonance and tube-wave reflections. *Geophysics*, 2017, 82: D171–D186. DOI: 10.1190/geo2016-0480.1.

[3] Ghidaoui M.S., Zhao M., McInnis D.A., Axworthy D.H. A review of water hammer theory and practice. *Applied Mechanics Reviews*, 2005, 58: 49–76. DOI: 10.1115/1.1828050.

[4] Chaudhry M.H. *Applied Hydraulic Transients*. 3rd ed. Springer, 2014. DOI: 10.1007/978-1-4614-8538-4.

[5] Dong X.L. et al. Research and application of hydraulic fracturing fluid entry depth detection method based on water-hammer signal. *Geoenergy Science and Engineering*, 2025, 246: 213556. DOI: 10.1016/j.geoen.2024.213556.

[6] Sun S.S., He Y.M., Liu L.J., Li Y.C., Zou L.Q., Yang L. Identification of fluid-entry clusters and diagnosis of downhole events based on high-frequency water hammer pressure. *International Journal of Rock Mechanics and Mining Sciences*, 2026, 200: 106437. DOI: 10.1016/j.ijrmms.2026.106437.

[7] Cheng Y.J., You J.X., Guo F.Q., Wang J.C., Li L., Zhu J.P. Development and effectiveness verification of ultra-high-frequency water-hammer wave monitoring equipment for large-scale fracturing of unconventional oil and gas. *Flow Measurement and Instrumentation*, 2026, 111: 103424. DOI: 10.1016/j.flowmeasinst.2026.103424.

[8] Childers D.G., Skinner D.P., Kemerait R.C. The cepstrum: a guide to processing. *Proceedings of the IEEE*, 1977, 65: 1428–1443. DOI: 10.1109/PROC.1977.10747.

[9] Oppenheim A.V., Schafer R.W. From frequency to quefrency: a history of the cepstrum. *IEEE Signal Processing Magazine*, 2004, 21: 95–106. DOI: 10.1109/MSP.2004.1328092.

[10] Qiu Y. et al. Water hammer response characteristics of wellbore-fracture system: multi-dimensional analysis in time, frequency and quefrency domain. *Journal of Petroleum Science and Engineering*, 2022, 213: 110425. DOI: 10.1016/j.petrol.2022.110425.

[11] Luo Y. et al. A new water hammer decay model: analyzing the interference of multiple fractures and perforations on decay rate. *SPE Journal*, 2023, 28: 1973–1985. DOI: 10.2118/214658-PA.

[12] Hu X. et al. Evaluation of multi-fractures geometry based on water hammer signals: a new comprehensive model and field application. *Journal of Hydrology*, 2022, 612: 128240. DOI: 10.1016/j.jhydrol.2022.128240.

[13] Sun S.S. et al. A novel comprehensive water hammer pressure model for fracture geometry evaluation. *SPE Journal*, 2025, 30(9): 5350–5366. DOI: 10.2118/228403-PA.

[14] Liu Lijun, Liu Yongzan, Wang Xiaoguang. A novel MCMC-based hydraulic fracture diagnostics approach using water hammer data. In: *57th US Rock Mechanics/Geomechanics Symposium*, 2023. DOI: 10.56952/ARMA-2023-0865.

[15] Zeng B. et al. Fracture size inversion method based on water hammer signal for shale reservoir. *Frontiers in Energy Research*, 2024, 11: 1336148. DOI: 10.3389/fenrg.2023.1336148.

[16] Deng S. et al. A diagnostic model for hydraulic fracture in naturally fractured reservoir utilising water-hammer signal. *Engineering Fracture Mechanics*, 2025, 325: 111347. DOI: 10.1016/j.engfracmech.2025.111347.

[17] Gabry M.A., Ramadan A., Soliman M.Y. Estimating water hammer damping ratios using continuous wavelet transform for induced hydraulic fracture complexity characterization. *SPE Journal*, 2025, 30(6): 3587–3611. DOI: 10.2118/225459-PA.

[18] Zhu M., Wang H. Automated water hammer analysis for fracture parameter inversion using high-frequency shut-in pressure signals during hydraulic fracturing. *Modelling*, 2026, 7(3): 87. DOI: 10.3390/modelling7030087.

[19] Zielke W. Frequency-dependent friction in transient pipe flow. *Journal of Basic Engineering*, 1968, 90: 109–115. DOI: 10.1115/1.3605049.

[20] Vardy A.E., Brown J.M.B. Transient, turbulent, smooth pipe friction. *Journal of Hydraulic Research*, 1995, 33: 435–456. DOI: 10.1080/00221689509498654.

[21] Bergant A., Simpson A.R., Vitkovsky J. Developments in unsteady pipe flow friction modelling. *Journal of Hydraulic Research*, 2001, 39: 249–257. DOI: 10.1080/00221680109499828.

[22] Dong X.L., Wang X.M., Zhu H.Y., Yang Y.Y., He L., Liu Z.P., Gong W. The influence of filtering methods and parameters on reflection period from pump shut-in water hammer signals: a comprehensive study. *Geoenergy Science and Engineering*, 2026, 257: 214278. DOI: 10.1016/j.geoen.2025.214278.

## 图件清单与图注

**图 1｜局部标签与网络观测。** 左图为被本文前向证据否定的朴素局部标签解释；右图表示井口观测是经过多个离散界面、下游反馈和二维倒谱算子后的网络条件化响应。图件：`图表数据/Figure_1_Local_vs_Network.{png,svg,pdf}`。

**图 2｜前向模型和观测算子。** (a) 集总裂缝节点的流量守恒与柔度/滤失关系；(b) 单簇基线与四簇网络的累积倒谱对比；(c) 名义射孔窗口、局部峰提取半径和网络条件化峰值。面板 (c) 使用模拟几何邻域，是基准峰值提取而非盲重建。图件：`图表数据/Figure_2_Forward_and_Operator.{png,svg,pdf}`。

**图 3｜物理信息去混叠算子示意。** (a) 原始连续倒谱及名义射孔和非射孔路径候选；(b) 仅由前向波场定义的候选路径凹口与有限传输增益；(c) 有限增益和 0.80 m 超高斯聚焦候选核；(d) 先验/传输敏感性代理量 \(M_{\mathrm{sens}}\)。图中采用固定的示意值 \(T_{\mathrm{eff}}=0.784\) 并对增益归一化显示，且未包含 \(\eta(S)\)；该值不是由 426 个峰值表反演得到。该图是算子模板和代理量示意，不是盲恢复率验证或真实性概率。图件：`../Section3_可信度指数模型与方法论/figures/Figure_3_3_Physics_Informed_Dealiasing_Workflow.{png,svg,pdf}`。

**图 4｜四个极端工况的前向证据矩阵。** (a) \(S=30\,\mathrm{m}\)、\(n=4\) 的低响应和 3120/3150 m 路径候选；(b) \(S=10\,\mathrm{m}\) 的窗口重叠；(c) \(S=20\,\mathrm{m}\)、\(n=8\) 的深级联耗散；(d) \(S=80\,\mathrm{m}\) 的高间距原始基线。绿色、黄色和灰色分别表示随簇序号变化的较高、歧义和低信息敏感度代理，未包含 \(\eta(S)\)；酒红色表示非射孔路径候选。色带不表示真实状态概率。图件：`../Section3_可信度指数模型与方法论/figures/Figure_3_4_MultiCase_Physics_Correction_Matrix.{png,svg,pdf}`。

**图 5｜间距主效应。** (a) \(n=4\) 的连续倒谱剖面随 \(S\) 的变化；(b) 各簇表观峰值 \(P_i(S)\)；(c) 相对响应 \(\alpha_i=P_i/P_1\)。图件：`../5.1_间距S主效应分析/figures/Figure_5_1_Spacing_Main_Effect.{png,svg,pdf}`。

**图 6｜簇数主效应和末端衰减。** 汇总不同 \(n\) 下的峰值、末首峰比和累计能量代理量，用于展示级联界面数的作用。图件：`../5.2_裂缝总数n主效应分析/figures/Figure_5_2_Multiplicity_Main_Effect.{png,svg,pdf}`；累计衰减补充图为同目录的 `Figure_5_2_Supp_Cumulative_Energy_Decay`。

**图 7｜首簇深度效应。** 比较 \(X_1\) 改变时的绝对峰值和相对响应，检验深度能否被一个独立标量增益表示。图件：`../5.3_首缝深度X1主效应分析/figures/Figure_5_3_Depth_Main_Effect.{png,svg,pdf}`。

**图 8｜间距、簇数与深度的交互。** (a) \(P_1(S,n)\) 相图；(b) \(R_{\mathrm{end}}(S,n)\) 相图；(c) 按簇数分组的 \(R_{\mathrm{end}}\)-\(L_{\mathrm{span}}\) 关系；(d) \(P_1(S,X_1)\) 相图；(e) \(\alpha_2(S,X_1)\) 相图；(f) 不同 \(X_1\) 下的 \(\alpha_2(S)\) 曲线。灰点表示经认证的离散网格，连续色面仅作插值可视化。图件：`图表数据/Figure_8_Interaction_Phase_Maps.{png,svg,pdf}`。

**图 9｜固定几何衰减包络。** 展示 \(P_i=A\exp[-\gamma(i-1)]\) 在固定几何内的描述性拟合；\(\gamma\) 是工况内经验参数，不解释为普适透射系数。图件：`../5.6_空间衰减包络与模型拟合/figures/Figure_5_6_Spatial_vs_Topological_Decay.{png,svg,pdf}`。

**图 10｜可辨识度与操作包络。** 给出不同门限下的可识别簇数、表观信噪比和间距可行域。图件：`../5.7_可辨识度边界与可行域评估/figures/Figure_5_7_Operational_Envelope.{png,svg,pdf}`。

上述主图均以同一 Matplotlib 字体、线宽和色板导出为 400 dpi PNG、可编辑 SVG 与 PDF。旧版 `Figure_3_4_a` 至 `Figure_3_4_d` 以及 `Figure_Psi_k_Calibration_and_Transfer` 属于 target-equalization/oracle 探索输出，不作为本文定量验证图件。
