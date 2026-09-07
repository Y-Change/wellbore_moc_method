# 井筒–多簇裂缝高频水击物理信息算子学习攻关计划

**执行周期：8个月｜可信物理基准：高保真MOC｜目标：CMAME/JCP主投，Computers & Geosciences与SPE Journal分轨投稿**

本计划将研究对象限定为“可压缩单相井筒 + 多簇射孔/裂缝支路 + 非定常摩阻 + 井口稀疏压力观测”。研究成果必须同时通过三类验收：MOC网格收敛与端口守恒、算子在未见布局/工况上的相位和耗散泛化、单点反演的后验覆盖率与结构可辨识性。以下样本数和误差门槛是项目执行时的预注册建议值，不是前人论文已经报告的结果。

## 一、选题立意与创新定标

### 1. 论文题目备选

**方法主线（CMAME/JCP）**

1. *Physics-informed neural operators for multi-cluster wellbore–fracture water-hammer transients with frequency-dependent friction and sparse inverse identification*
2. *Interface-aware PI-FNO/DeepONet for compressible wellbore water-hammer waves with discrete impedance jumps and memory friction*

对应中文：

1. **含频变摩阻与离散阻抗跳跃的井筒–多簇裂缝水击物理信息神经算子及稀疏反演**
2. **面向多簇离散内边界和记忆摩阻的井筒高频水击界面感知物理算子学习**

**地学/工程主线（C&G/SPE Journal）**

1. *A physics-constrained operator surrogate and uncertainty-aware inversion for multi-fracture frac-water-hammer diagnostics*
2. *Fast and uncertainty-calibrated multi-cluster fracture diagnosis from high-frequency shut-in water-hammer pressure*

对应中文：

1. **面向多簇压裂水击诊断的物理约束算子代理与不确定性反演**
2. **基于高频停泵水击压力的多簇裂缝快速诊断与校准不确定性反演**

题目不使用“首次”“完全解决”等不可由 prior art 推出的措辞。已有Ye等的管路瞬变PINN、Hu/Qiu/Luo/Deng/Sun等的多裂缝水击诊断和Liu等的MCMC反演，因此本文的创新对象是下列交集，而非“AI用于水击”本身：

> **未知簇布局的非连续节点 + 非线性射孔压降 + 频变/记忆摩阻 + 单点井口后验反演 + 未见布局和未见操作工况验证。**

### 2. 三个理论创新声明

**创新A：拓扑感知的物理归纳偏置。** 令 `Gamma_N={x_j}_{j=1}^N` 为有序簇位置，采用带mask的簇集合编码、signed-distance/indicator通道和逐段Fourier block。每个井筒段独立进行低模态传播更新，在节点由可微散射层将左右特征变量与裂缝端口阻抗连接。散射层输入 `Z_j(s)` 或其RCI状态参数，而不是把所有簇聚合成一个底部端口；因此网络必须显式表示往返时延、压力迹连续、流量分流和局部阻抗差异。固定网格的PI-FNO作为对照，变量数量簇和非规则位置优先采用Branch–Trunk查询解码。

**创新B：残差、界面、记忆和能量的联合约束。** 在无量纲化后的连续性/动量残差之外，逐簇加入质量跳跃、压力迹、孔眼二次压降、裂缝储容/惯性/漏失残差；用指数递归状态逼近Zielke卷积，Brunone作为独立瞬时加速度模型，禁止二者混称。训练中报告输入功率–储能变化–耗散收支；只有无外部输入、闭合被动时才要求能量非增。

**创新C：从正演代理到物理引导逆算子。** 训练一个由井口波形或波形特征到参数后验/初值的逆编码器，再用可微MOC或冻结的物理算子做少量梯度校正。输出不只是一组点估计，还要给出簇数/位置的离散概率、`C_d A_perf`等只能联合识别的参数组合、95%可信区间和后验波形重构。以Fisher信息奇异值、profile likelihood和多工况后验覆盖率界定“可识别到什么”，不声称单点观测能够唯一恢复全部几何参数。

### 3. 目标期刊匹配

| 期刊 | JCR/定位 | 必须呈现的卖点 | 不能缺失的证据 |
|---|---|---|---|
| *Computer Methods in Applied Mechanics and Engineering* | JCR2025三个相关学科均Q1 | 界面感知算子、守恒/能量约束、频变摩阻、误差与泛化 | 与cPINN、Fourier-feature PINN、Fourier-DeepONet、标准FNO/DeepONet的严格消融；独立节点残差 |
| *Journal of Computational Physics* | JCR2025 Physics Mathematical Q1、CS Interdisciplinary Applications Q2 | 双曲PDE、长时稳定、相位/频谱误差和可辨识性理论 | MOC离散收敛、CFL/插值、记忆状态、能量收支、Fisher退化分析 |
| *Computers & Geosciences* | 本项目未核验JCR分区 | 可复用地学基准、数据字典、传感器/滤波测量算子、外部井数据 | 完整容器和随机种子、按井/布局切分、现场外部验证；不把网络层数当贡献 |
| *SPE Journal* | 本项目未核验JCR分区 | 井口停泵诊断、簇间干涉、操作决策和现场可用的后验 | 与Luo 2023、Sun 2025等工程模型对比；仪器带宽、滤波、现场案例和决策误差 |

## 二、MOC高保真基准与数据集

### 1. 求解器和节点闭合

每个井筒段使用

$$
r_Q=Q_t+gA H_x+\frac{fQ|Q|}{2DA}+J_{u,Q}=0,\qquad
r_H=H_t+\frac{a^2}{gA}Q_x=0.
$$

簇节点采用

$$
Q_j^- -Q_j^+=\sum_m q_{jm}+q_{leak,j},
$$

$$
p_{w,j}-p_{in,jm}=\frac{\rho q_{jm}|q_{jm}|}{2C_{d,jm}^2A_{perf,jm}^2}+\Delta p_{nw,jm},
$$

$$
C_{f,jm}\dot p_{c,jm}=q_{jm}-G_{l,jm}(p_{c,jm}-p_{res}),quad
p_{in,jm}-p_{c,jm}=I_{f,jm}\dot q_{jm}+R_{f,jm}q_{jm}.
$$

历史摩阻采用Zielke/指数递归作为一条基线，Brunone瞬时加速度作为另一条基线。两者分别标定、分别训练和分别报告，不能用一个网络的耗散误差代替摩阻模型验证。

为使记忆项可微且可复现，先对选定时间窗内的Zielke核做指数拟合，而不是把拟合误差隐藏在网络中：

$$
\dot z_l=-\beta_l z_l+\alpha_l\dot V,\qquad
J_{u,V}=\sum_{l=1}^{M}w_l z_l,\qquad J_{u,Q}=A J_{u,V}.
$$

`z_l(0)`保存扰动前预历史；`alpha_l,beta_l,w_l`按Reynolds数、时间窗和目标核拟合。Brunone分支单独加入
`H_x+V_t/g+fV|V|/(2gD)+k(V_t-aV_x)/g=0` 中的瞬时加速度项。训练/测试必须分别报告“真实MOC卷积、指数核截断和网络误差”，否则无法判断波形衰减来自物理还是代理。

空间网格取 `Nx=512,1024,2048` 三档，簇位置与单元对齐，节点左右各保留迹值；局部加密只用于独立验证，不允许测试集信息反向改变训练网格。时间步

$$\Delta t=Cr\,\Delta x/a_{min},\qquad Cr\in\{0.8,1.0\}.$$

`Cr=1`只作为均匀无插值特征网格模板；多波速、多段或局部插值时必须使用`Cr<1`、分段时间步或隐式处理。每个算例做三网格Richardson检查：井口峰值差小于0.5%、首波到时差小于0.2%、全场收敛阶不低于1.8；节点质量残差小于`10^-6`，端口能量不平衡小于`10^-4`。保存完整CPU/GPU wall-clock（含I/O），作为代理加速分母。

### 2. 参数空间和采样

初始工程先验范围如下，正式实验前用现场井径、流体实验和压力记录校准；范围不是对所有井的普适结论。

| 参数 | 初始范围/分布 | 参数化处理 |
|---|---|---|
| `L,D,t_w,E,nu` | `L=1–5 km`；`D=0.076–0.127 m`；`t_w=6–15 mm`；`E=180–220 GPa`；`nu=0.25–0.32` | 截断均匀或现场先验 |
| `rho,K,mu,epsilon` | `rho=950–1200 kg m^-3`；`K=1.6–2.6 GPa`；`mu=1–20 mPa s`；粗糙度`10^-6–5x10^-4 m` | `mu,K,C_i,I_i,G_i`使用log-uniform或实测先验 |
| 簇拓扑 | `N=3–12`；间距20–120 m，允许非均匀 | 有序位置，禁止标签随机置换泄漏 |
| 射孔 | `C_d=0.55–0.90`；每簇`A_perf=0.5x10^-4–8x10^-4 m^2` | 逐簇参数；识别时优先报告乘积可辨识性 |
| 裂缝端口 | `C_i=10^-9–10^-5 m^3 Pa^-1`；`I_i=10^3–10^7 Pa s^2 m^-3`；`G_i=10^-13–10^-8 m^3 Pa^-1 s^-1`；漏失`0–5x10^-8 m^3 Pa^-1 s^-1` | 正参数log-uniform；零漏失单独混合点 |
| 激励 | `tau_stop=0.02–2 s`；`Q_0=0.01–0.10 m^3 s^-1` | 停泵/阀闭合波形和滤波作为输入函数 |

由Korteweg关系派生波速并保留`800–1600 m s^-1`物理筛选。先用2,000组maximin-LHS做网格、守恒和动态范围pilot；正式数据用Owen-scrambled Sobol（8 seeds）30,000组：20,000 train、3,000 validation、3,000 ID-test、4,000 OOD-test。OOD包括`N=13–16`、密/疏簇距、黏度/顺应性尾部、训练未共现的参数组合和随机删簇/阻抗突变。另设5个challenge场景，每场景100个随机seed。

每个波形覆盖至少20个井筒往返周期，输出`p(x,t),v(x,t),p_head(t),Q_j(t),p_j(t)`及摩阻记忆状态；波头附近输出步长不超过MOC步长的0.1倍。按井、簇布局和泵停工况切分，禁止同一长波形的相邻时间窗跨训练/测试泄漏。保存参数、网格、MOC版本、随机种子、残差和运行时间。

## 三、PINO / PI-DeepONet架构

### 1. 输入输出流形

令`Gamma_N={x_j}_{j=1}^N`，允许的输入空间为

$$
\mathcal A=\bigsqcup_{N=N_{min}}^{N_{max}}
\left\{(u_{in},u_{out},\mu_{smooth},\{x_j,\vartheta_j\}_{j=1}^N):
u_{in/out}\in H^1(0,T),\ \vartheta_j\in\Theta_j\right\},
$$

其中`vartheta_j=(N_{p,j},C_{d,j},A_{perf,j},C_{f,j},I_{f,j},R_{f,j},G_{l,j})`，离散簇数使用带mask的集合拓扑，不强行把不同`N`嵌入同一光滑欧氏参数场。输出解空间为断片域

$$
\mathcal U_N=\{(H,Q,z_1,\ldots,z_M):H,Q\in L^2(0,T;H^1(\Omega\setminus\Gamma_N)),\ \text{左右迹值存在}\},
$$

并附加节点状态`p_c,q_f`和观测算子

$$y_n=\left[\mathcal M_\psi\mathcal G(\mu)\right](t_n)+\epsilon_n,
$$

其中`M_psi`包含井口采样、传感器传递函数和滤波。预测空间场不应掩盖零测度节点的迹值错误。

### 2. 推荐骨架：Fourier-Jump PI-DeepONet

推荐以PI-DeepONet为主、分段FNO为局部传播模块，并保留纯PI-FNO作为对照。

1. **Global branch encoder**：对`u_in(t)`、初始流量、波速、摩阻类型和指数记忆初值做1D因果卷积/小型Transformer编码。
2. **Cluster set encoder**：每个簇输入`(x_j,mask,N_p,C_d,A_perf,C_f,I_f,R_f,G_l)`，加入Fourier positional embedding和signed distance；用DeepSets/attention得到可变簇数的集合表示。
3. **Segment Fourier blocks**：在每个`(x_j,x_{j+1})`上做有限模态传播更新，输出bulk latent states`(H,Q,z_l)`；不把跳跃点用零填充后直接当作光滑场。
4. **Interface scattering layer**：将两侧特征变量与支路端口状态送入可微Newton/固定点层，求解节点质量、孔眼二次压降和RCI闭合；其雅可比和条件数必须记录。训练时可用代数投影作hard constraint，残差损失作soft constraint的对比。
5. **Trunk decoder**：查询`(x,t,segment_id,d(x,Gamma_N))`，输出`H,Q`；节点查询另输出左右迹、`p_c,q_f`和井口`p_head`。物理输出按`p=rho g(H-z)`、`v=Q/A`转换，避免把水头和压力在损失函数中混用。
6. **Physics head**：自动微分计算PDE、历史状态、初边值和节点残差；独立energy head只作验收，不允许网络通过任意负耗散“作弊”。

标准FNO适合固定规则网格和多查询，PI-DeepONet适合不规则查询、稀疏井口输入和变量簇数；二者都必须在同一MOC划分和训练预算下比较。Fourier-Jump并不预设一定优于FNO，优越性必须由节点相位、`R_jump`和OOD结果证明。

## 四、物理损失与逆问题

### 1. 复合损失

先以`Q_ref,H_ref,p_ref,q_ref`逐项无量纲化，在断片域`Omega_b=Omega\setminus Gamma_N`上定义

$$
\mathcal L_{pde}=\frac{1}{|\Omega_b|T}\int_0^T\int_{\Omega_b}
\left(\frac{r_Q^2}{Q_{ref}^2}+\frac{r_H^2}{H_{ref}^2}\right)dxdt.
$$

观测/全场监督项为

$$
\mathcal L_{data}=\frac{1}{|\mathcal O|}\sum_{(x,t)\in\mathcal O}
\left[\frac{(\hat H-H^{MOC})^2}{H_{ref}^2}+\frac{(\hat Q-Q^{MOC})^2}{Q_{ref}^2}\right],
$$

其中训练可用全场低分辨率标签和稀疏井口标签两种设置，测试统一用高保真MOC。

初边值项为

$$
\mathcal L_{bc}=\frac{1}{T}\int_0^T\left(\frac{r_{pump}^2}{Q_{ref}^2}+\frac{r_{end}^2}{Q_{ref}^2}\right)dt
 +\frac{1}{|\Omega|}\int_\Omega\left(\frac{r_{IC,H}^2}{H_{ref}^2}+\frac{r_{IC,Q}^2}{Q_{ref}^2}\right)dx.
$$

第`j`个节点的界面项为

$$
\begin{aligned}
r_{m,j}&=Q_j^- -Q_j^+-\sum_mq_{jm}-q_{leak,j},\\
r_{p,j}&=p_j^- -p_j^+,\\
r_{o,jm}&=p_{w,j}-p_{in,jm}-\frac{\rho q_{jm}|q_{jm}|}{2C_{d,jm}^2A_{perf,jm}^2}-\Delta p_{nw,jm},\\
r_{C,jm}&=C_{f,jm}\dot p_{c,jm}-q_{jm}+G_{l,jm}(p_{c,jm}-p_{res}),\\
r_{I,jm}&=p_{in,jm}-p_{c,jm}-I_{f,jm}\dot q_{jm}-R_{f,jm}q_{jm}.
\end{aligned}
$$

$$
\mathcal L_{jump}=\sum_{j=1}^N\frac{1}{T}\int_0^T\left[
\frac{r_{m,j}^2}{Q_{ref}^2}+\frac{r_{p,j}^2}{p_{ref}^2}+
\sum_m\left(\frac{r_{o,jm}^2}{p_{ref}^2}+\frac{r_{C,jm}^2}{q_{ref}^2}+\frac{r_{I,jm}^2}{p_{ref}^2}\right)\right]dt.
$$

总损失为

$$\mathcal L_{total}=\lambda_d\mathcal L_{data}+\lambda_p\mathcal L_{pde}+\lambda_b\mathcal L_{bc}+\lambda_j\mathcal L_{jump}+\lambda_e\mathcal L_{energy}.$$

对静止参考压力扰动，定义

$$
E_{tot}=E_{wb}+\sum_{jm}\left[\frac12C_{f,jm}(p_{c,jm}-p_{res})^2+\frac12I_{f,jm}q_{jm}^2\right]+E_{mem},
$$

$$
r_E=P_{ports}-\frac{dE_{tot}}{dt}-D_{wall}-D_{perf}-D_{nw}-D_{frac}-D_{leak},\qquad
\mathcal L_{energy}=\frac1T\int_0^T\frac{r_E(t)^2}{P_{ref}^2}\,dt.
$$

其中`D_frac=sum_j R_{f,j}q_j^2`，`E_mem`只在所选记忆状态实现具有被动性时启用。`L_energy`只作端口功率验收；无外部输入且闭合被动时才检查能量非增，不能在有泵输入时强迫压力场能量单调。损失训练使用三段课程：先数据/初边值稳定波头，再加入PDE与jump，最后加入长时窗、摩阻记忆和OOD扰动。

### 2. 自适应权重和数值实现

所有损失先无量纲化；每个训练周期测量`g_k=||grad_theta(lambda_k L_k)||_2`。GradNorm用初始相对下降率构造目标

$$
g_k^*=\bar g\left(\frac{L_k/L_{k,0}}{\frac{1}{K}\sum_l L_l/L_{l,0}}\right)^\alpha,
\qquad \mathcal L_{GN}=\sum_k|g_k-g_k^*|,
$$

再更新`lambda_k`使各块梯度接近`g_k^*`；`lambda_k`截断在`[10^-3,10^3]`并记录轨迹。NTK块特征值只作独立诊断，可用`lambda_k^{NTK}\propto1/(tr(K_k)/dim(K_k)+epsilon)`作为可复现实验分支，而不是无证据宣称NTK存在通用幂律。节点和波头采用重要性采样，界面点比例单独报告。主结果必须与固定权重、仅GradNorm和仅NTK三组对比。

### 3. 单点井口逆演闭环

观测只有`p_head(t)`时，先用测量算子`M_psi`去噪、校准采样/滤波响应，再用逆编码器输出`q(phi|p_head)`和初值`theta_0`。第二步通过冻结正演算子或可微MOC最小化

$$
J(\theta,\eta)=\frac12\|\Sigma^{-1/2}[M_\psi G(\theta,\eta)-p_{head}]\|_2^2
 +\beta R_{prior}(\theta,\eta),
$$

其中`eta`包含波速、摩阻、传感器增益等干扰参数，`R_prior`是物理边界、正性、相邻簇最小间距和裂缝数先验。对小型challenge集用MCMC/SMC校准后验；大规模用ensemble/拉普拉斯近似或多起点L-BFGS。若`C_d`与`A_perf`只以乘积进入压降律，则先验或独立孔眼观测不足时只能报告其乘积，不能分别报出伪精确值。

抗噪矩阵使用SNR`∞,30,20,10,5 dB`，加入1–3%脉冲野值；似然同时用Gaussian和Huber/Student-t。报告参数MAPE、log-RMSE、簇数F1、位置误差、95%覆盖率、CRPS/NLL、波形重构误差和后验相关矩阵。对多簇标签置换使用排序约束或Hungarian matching，不能把标签交换误判成物理失败。

## 五、基准、指标与消融

### 1. 正演和逆演基线

| 组别 | 基线 | 目的 |
|---|---|---|
| 可信正演 | 高保真MOC；粗网格MOC | 真值与传统速度上限 |
| 无物理网络 | vanilla FNO；vanilla DeepONet；纯PINN MLP | 量化谱偏差、数据依赖和长时外推 |
| 部分物理 | PI-FNO/PI-DeepONet（无`L_jump`）；CPINN式界面残差 | 分离体PDE与离散节点约束 |
| 本文正演 | Fourier-Jump/Branch–Trunk PINO | 联合拓扑、记忆和物理损失 |
| 逆演 | MOC+L-BFGS/adjoint；小规模MCMC/SMC；1D CNN/Transformer；DeepONet encoder | 比较点估计、后验和速度 |

所有学习基线共享数据划分、归一化、训练预算和至少5个随机seed；报告均值、标准差、95% bootstrap区间和效应量，不只报告最优seed。

### 2. 定量指标与建议门槛

$$E_{L2}=\frac{||u_{pred}-u_{MOC}||_2}{||u_{MOC}||_2+\epsilon},\quad
E_{PSD}=\frac{\int_{f_l}^{f_h}|S_{pred}(f)-S_{MOC}(f)|df}{\int_{f_l}^{f_h}S_{MOC}(f)df}.$$

另报告井口NRMSE、峰谷幅值误差、首波及第`k`波相位`Delta phi_k=2pi|t_hat_k-t_k|/T_ref`、log峰包络衰减率误差、`R_mass`、`R_jump`和端口能量不平衡。推理加速

$$S_{infer}=T_{MOC,wall}/T_{model,wall},
$$

分开batch=1和batch=256，并另报`T_label,T_train,T_infer,T_inverse`和盈亏平衡查询数；不把训练时间隐藏在“实时”表述中。

预注册的主结果门槛为：ID全场`E_L2<=2%`、OOD`<=5%`；首波相位误差不超过周期1%、峰值误差不超过3%、PSD误差不超过5%、衰减率误差不超过10%；独立节点`R_jump<=10^-3`。单case推理加速目标`>=100x`、batch=256目标`>=1000x`，若未达到则报告实际Pareto前沿而不删结果。逆演目标为ID MAPE`<=10%`、OOD`<=20%`、SNR=10 dB时MAPE增幅`<=50%`、95%区间覆盖率0.90–0.98、簇数F1`>=0.90`。

### 3. 消融矩阵

| 编号 | 去除/替换项 | 预期可检验现象 |
|---|---|---|
| A0 | 完整模型 | 主结果 |
| A1 | 去除`L_jump` | 节点邻域`R_jump`、反射相位和簇数OOD显著恶化 |
| A2 | 去除Zielke/Brunone非定常项，仅稳态Darcy | 长时高频衰减和PSD误差恶化 |
| A3 | 固定权重替代GradNorm/NTK | 梯度块失衡、波头欠拟合或训练方差上升 |
| A4 | 去除簇token/Fourier-Jump | 多簇和未见布局泛化下降 |
| A5 | 仅监督无PDE | 数据外推和守恒下降 |
| A6 | 仅PDE无数据 | 相位/幅值校准和噪声鲁棒性下降 |
| A7 | 去除参数先验/UQ | 后验过窄、覆盖率下降或非唯一方向发散 |
| A8 | 仅井口单点训练 vs 多传感器训练 | 识别性与测量配置敏感性差异 |

每项至少5 seeds，以配对bootstrap/Wilcoxon给出置信区间和效应量。另做`N=1,3,6,12`、Zielke/Brunone、白噪/脉冲/带限噪声三因素分析；若某“预期现象”没有出现，按数据报告，不能为了支持叙事删去。

## 六、8个月里程碑

| 阶段 | 月份 | 工作包 | 质检验收 |
|---|---:|---|---|
| I | 1–2 | PDE/RCI节点闭合、Zielke/Brunone分支、MOC单元测试、2,000组pilot | 单管阀闭合和单节点阻抗解析/半解析算例峰值误差<1%；三网格阶>=1.8；质量残差<`10^-6`；冻结数据生成器 |
| II | 3–4 | 30,000组Sobol数据、PI-FNO/PI-DeepONet、jump/memory损失、正演训练 | ID`E_L2<=3%`、首波相位<=2%；5 seeds方差<10%；单GPU batch=1目标>=100x，完整记录失败域 |
| III | 5–6 | 逆编码器、可微MOC校正、MCMC/SMC challenge、抗噪、OOD、A0–A8 | ID/OOD MAPE<=10%/20%；SNR10 dB覆盖率>=0.90；簇数F1>=0.90；完成后验相关/Fisher报告 |
| IV | 7–8 | 冻结盲测、现场/实验外部验证、开源容器、论文和盲审 | 测试集只运行一次；公开代码、数据字典、seed、环境和wall-clock；完成CMAME/JCP主稿与C&G/SPE应用稿；两位同行盲审逐条关闭 |

每月形成可审计tag：MOC版本、数据manifest、训练配置、指标JSON、图表脚本和失败案例。任何阶段门槛未达，优先缩小可辨识参数集合或申报“可靠等效参数反演”，不能扩大网络复杂度掩盖物理不可辨识性。

## 七、论文结构与图表蓝图

### 1. 正文结构

1. **Introduction**：联合科研断层、prior art边界、贡献列表和可验证假设。
2. **Governing physics and identifiable parameterization**：PDE、Zielke/Brunone、孔眼/RCI节点、测量算子和参数可辨识性。
3. **MOC benchmark and dataset**：CFL、历史状态、网格收敛、Sobol/LHS、ID/OOD切分和数据字典。
4. **Interface-aware physics-informed operator**：函数空间、Fourier-Jump/Branch–Trunk、损失、GradNorm/NTK和可微节点层。
5. **Sparse inverse identification**：逆编码器、物理校正、先验/UQ、Fisher和后验覆盖率。
6. **Results, limitations and reproducibility**：正演、消融、OOD、速度盈亏平衡、外部验证、失效域、代码数据和结论边界。

CMAME/JCP可将“Results”拆成误差/稳定性、逆演和消融三节；C&G增加数据/代码与现场测量章节；SPE版本把节点参数映射、泵停工况和现场决策放在主结果而非补充材料。

### 2. Main Figures 1–8

| 图 | 内容 | 必须标注的证据 |
|---|---|---|
| Fig.1 | 物理拓扑：长井筒、簇位置、孔眼、裂缝端口、正反射路径和测量算子 | `Z_w`、`Z_f(omega)`、shunt质量守恒与压力迹，区分端接反射 |
| Fig.2 | MOC→PINO→逆演流程和网络维度图 | 输入函数、簇token、分段Fourier、interface scattering、trunk查询 |
| Fig.3 | 三网格/两CFL收敛、首波和多反射时空云图 | `E_L2`、峰值、到时、质量/能量残差 |
| Fig.4 | MOC、PINN、FNO、DeepONet、本文模型全场波形、PSD、相位和峰谷衰减 | ID/OOD分面，Zielke/Brunone分开 |
| Fig.5 | 去`L_jump`、聚合RCI、逐簇jump、硬投影/软惩罚对比 | 节点质量残差、复反射系数、局部相位和端口功率 |
| Fig.6 | NTK/梯度范数、因果窗口、模态截断和OOD误差曲线 | 训练动力学与失败域，不只展示loss下降 |
| Fig.7 | 井口波形→簇数/位置/`R,C,I`后验的散点、coverage、Fisher奇异值和SNR曲线 | MAP/MAPE、F1、95%覆盖、参数相关和标签置换处理 |
| Fig.8 | `T_label,T_train,T_infer,T_inverse` Pareto图及实验/现场外部验证 | CPU/GPU、batch=1/256、查询盈亏平衡、实测带宽和滤波链 |

补充材料放完整参数表、每个seed的结果、失败案例、MOC输入输出schema、PINO权重轨迹、测量传递函数和容器运行日志。

## 最终决策门

只有同时满足以下条件才以“逐簇参数物理算子”作为主张：节点残差和相位误差在独立OOD上达标；逆演后验覆盖率接近名义水平；`C_d`与`A_perf`等结构等效方向被正确报告；代理速度在计入训练成本后仍对目标查询量有正盈亏平衡；至少一个实验/现场测量链外部验证通过。否则将论文主张收缩为“物理约束的等效端口/可辨识参数组合代理”，这比把不可辨识问题包装成逐簇几何恢复更符合CMAME、JCP和SPE Journal审稿标准。

本计划继承 [最终文献调研报告](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/literature_survey.md) 的证据边界；JCR分区、WoS引用快照和题录见 [wos_jcr_evidence.md](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/wos_jcr_evidence.md) 与 [bibliography_verified.md](E:/water_hammer_research/wellbore_moc_method/docs/literature_gap_audit_2026-09-05/bibliography_verified.md)。
