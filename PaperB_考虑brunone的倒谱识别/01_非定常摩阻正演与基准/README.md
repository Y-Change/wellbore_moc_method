# 01 非定常摩阻正演与基准数据模块

## 模块定位
本模块提供 Paper B 所需的所有正演基准时程数据，包含一维特征线法（MOC）在稳态 Darcy 摩阻与 Brunone 非定常摩阻下的对比仿真结果。

## 物理与数值设定
1. **井筒参数**：
   - 井长 $L = 5000\text{ m}$，内径 $D_p = 0.1397\text{ m}$，粗糙度 $K_D = 10^{-4}$
   - 水击波速 $a = 1450\text{ m/s}$，CFL 条件数精确为 1（$\Delta x = a \cdot \Delta t = 1.45\text{ m}$，$\Delta t = 1\text{ ms}$）
   - 初始稳态排量 $Q_0 = 0.0153\text{ m}^3/\text{s}$（对应流速 $V_0 = 1.0\text{ m/s}$）
   - 井口停泵水击激励：流速在 $\Delta t$ 内阶跃至 0

2. **裂缝节点参数**：
   - 首缝深度 $X_1 = 4100\text{ m}$（往返几何到时 $2 X_1 / a \approx 5.655\text{ s}$，停泵起始时间 $t_s = 1.0\text{ s}$ 时几何首波到达井口时间为 $6.655\text{ s}$）
   - 裂缝顺应系数 $C_f = 10^{-5}\text{ m}^2$，滤失系数 $k_{\text{leak}} = 10^{-4}\text{ m}^{2.5}/\text{s}$
   - 裂缝簇间距 $D \in \{5, 10, 20, 50, 100\}\text{ m}$，裂缝数 $n = 4$（主矩阵）与 $n = 1, 2$（对照组）

3. **摩阻模型**：
   - **稳态摩阻**：$J_s = \frac{f \Delta t V |V|}{2 D_p}$，采用 Zigrand-Swami 显式摩阻系数。
   - **Brunone 非定常摩阻**：$J_u = \frac{k}{2} \Delta t \left( \frac{\partial V}{\partial t} + a \text{ sign}(V) \left| \frac{\partial V}{\partial x} \right| \right)$。
   - **Matrix A（机制隔离）**：固定 $k \in \{0.0, 0.01, 0.02, 0.05, 0.1, 0.2\}$。
   - **Matrix B（实际流动）**：动态 $k(Re) = \frac{\sqrt{C(Re)}}{2}$，由局部瞬时雷诺数计算 Vardy 剪切衰减系数。

## 数据文件说明
- `波形/`：保存各工况的 `moc_timeseries.csv` 文件。
- `工况/`：按 `D{间距}_k{常数}` 归档的完整原始结果副本。
