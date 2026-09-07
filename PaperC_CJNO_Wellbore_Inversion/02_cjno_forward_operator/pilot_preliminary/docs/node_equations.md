# 可微节点层：未知量、约束与约减残差

依据 `node_newton.py` 与方案 §2.3 / §4.1。实现见 `interface_newton.py`。

## 全系统未知量（时刻 n+1）

\(H,\; Q^-,\; Q^+,\; \{q_m, p_{\mathrm{in},m}, p_{c,m}\}_{m=1}^{M}\)

## 已解析消元的仿射/显式约束

1. 压力迹：\(H^-=H^+=H\)。
2. C+/C−（含梯形摩阻）显式给出 \(Q^-(H), Q^+(H)\)：
   \(B Q + \tfrac12 R Q|Q| = D\)。
3. 分支 ODE 仿射化（与 MOC 同一套 `branch_coefficients`）：
   \(p_c = a_0 + a_1 q\)，\(p_{\mathrm{in}}-p_c = c_1 q + c_0\)。
   储容默认 Crank–Nicolson；\(G_l\Delta t/C_f>1\) 时切到指数积分器（刚性格）。

## 约减未知量与残差

\(\tilde u = (H, q_1,\ldots,q_M)\)

\[
\tilde r_0 = Q^-(H)-Q^+(H)-\sum_m q_m
\]
\[
\tilde r_m = p_w(H) - K_m q|q| - \kappa_m\phi(q) - (c_1 q+c_0) - (a_0+a_1 q)
\]

无量纲：\(\tilde r_0/Q_{\mathrm{scale}}\)，\(\tilde r_m/p_{\mathrm{scale}}\)。门：\(\|\tilde r\|_\infty<10^{-8}\)。

## IFT

\(\partial\tilde u^*/\partial\theta = -(\partial\tilde r/\partial\tilde u)^{-1}\partial\tilde r/\partial\theta\)，反向用线性求解，不显式求逆。条件数 \(>10^8\) 时 Tikhonov \(\lambda=10^{-8}\)，该步梯度有偏。

前向 Newton 调用 Stage-1 `solve_cluster_node`，与 MOC 同一离散根。dtype：前向与反向 float64。

近 \(q=0\) 处孔眼 \(q|q|\) 非光滑，有限差分不当作光滑导数。
