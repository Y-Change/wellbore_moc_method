# Brunone 非定常摩阻方案说明

**文件**：`wellbore_moc_method/wellbore_moc.py`
**适用模块**：`friction_model="brunone"`（MocConfig 字段）
**最后更新**：2026-07-08

---

## 1. 控制方程

水击一维特征线方程中，摩阻项分为稳态部分 \(J_s\) 和非定常部分 \(J_u\)：

\[
J = J_s + J_u
\]

特征线相容方程（C⁺/C⁻）：

\[
C^+: \quad V_1 + g_a H_1 - J_1 + g_a \Delta t\, V_1 \sin\theta = C_P
\]
\[
C^-: \quad -V_2 + g_a H_2 + J_2 + g_a \Delta t\, V_2 \sin\theta = C_M
\]

其中 \(g_a = a/(gA)\)，\(J = J_s + J_u\) 是总摩阻冲量（量纲 m/s），\(a\) 为波速，\(A\) 为截面积，\(\theta\) 为井斜角。

内节点求解：

\[
H = \frac{C_P + C_M}{2 g_a}, \quad V = C_P - g_a H
\]

---

## 2. 稳态达西摩阻 \(J_s\)

\[
J_s = \frac{f\, \Delta t\, V|V|}{2D}
\]

达西系数 \(f\) 由 Zigrand-Swami 显式近似计算：

\[
f = \begin{cases}
\dfrac{64}{Re} & Re < 2000 \quad \text{(层流)} \\[8pt]
\dfrac{1}{\left[-1.8\, \log_{10}\!\left(\dfrac{6.9}{Re} + K_D\right)\right]^2} & Re \ge 2000 \quad \text{(紊流)}
\end{cases}
\]

其中 \(Re = |V|D/\nu\)，\(K_D\) 为相对粗糙度，\(D\) 为内径，\(\nu\) 为运动黏度。

**对应代码**：`friction_term_J`（`wellbore_moc.py:144-146`）、`darcy_friction_factor`（`wellbore_moc.py:127-141`）

---

## 3. Brunone 非定常摩阻 \(J_u\)

### 3.1 核心公式

\[
\boxed{\, J_u = \frac{k}{2}\, \Delta t \left( \frac{\partial V}{\partial t} + a\, \text{sign}(V)\, \left|\frac{\partial V}{\partial x}\right| \right) \,}
\]

**物理含义**：

| 项 | 表达式 | 含义 |
|----|--------|------|
| 局部加速度 | \(\partial V/\partial t\) | 流速随时间变化率 |
| 对流加速度 | \(a\, \text{sign}(V)\, \|\partial V/\partial x\|\) | 流速沿空间变化率，乘波速得加速度量纲 |
| 系数 | \(k/2\) | Brunone 系数，由 Vardy 公式确定 |

两者之和即"总瞬时加速度"，是 Brunone 模型区别于稳态摩阻的关键——稳态摩阻只用 \(V|V|\)，无法反映流速变化时的额外阻尼。

**对应代码**：`brunone_friction_Ju`（`wellbore_moc.py:186-203`）

```python
def brunone_friction_Ju(k, dt, dVdt, dVdx, V, a):
    return (k / 2.0) * dt * (dVdt + a * np.sign(V) * np.abs(dVdx))
```

### 3.2 Brunone 系数 \(k\)

采用 **Vardy 剪切衰减系数** \(C\)：

\[
k = \frac{\sqrt{C}}{2}
\]

\[
C = \begin{cases}
4.76 \times 10^{-3} & \text{层流 } (Re < 2000) \\[8pt]
\dfrac{7.41}{Re^{\log_{10}(14.3/Re^{0.05})}} & \text{紊流 } (Re \ge 2000)
\end{cases}
\]

**对应代码**：`brunone_k`（`wellbore_moc.py:152-165`）、向量化版 `brunone_k_vec`（`wellbore_moc.py:168-183`）

该公式来自 Vardy & Brown (1996, 2003) 的剪切波衰减理论：\(C\) 与雷诺数相关，反映频率相关的能量耗散。层流 \(C\) 为常数，紊流 \(C\) 随 \(Re\) 增大而减小（高频振荡阻尼更强）。

---

## 4. 数值离散方案

### 4.1 时间导数（需上上步 \(V^{n-1}\)）

\[
\frac{\partial V}{\partial t}\bigg|_{j} \approx \frac{V_j^{n} - V_j^{n-1}}{\Delta t}
\]

需存储 **前两步** 速度场（`V_prev2_left/right`），Brunone 模式在 `n >= 2` 时才启用。

### 4.2 空间导数（沿特征线方向差分）

**C⁺ 来源点**（前向差分）：

\[
\frac{\partial V}{\partial x}\bigg|_{j} \approx \frac{V_{j+1}^{n} - V_j^{n}}{\Delta x}
\]

**C⁻ 来源点**（后向差分）：

\[
\frac{\partial V}{\partial x}\bigg|_{j} \approx \frac{V_{j}^{n} - V_{j-1}^{n}}{\Delta x}
\]

### 4.3 符号函数平滑

为消除 \(V \approx 0\) 处 \(\text{sign}(V)\) 跳变导致的数值锯齿，用 tanh 平滑：

\[
\text{sign}(V) \to \tanh\!\left(\frac{V}{V_{\text{smooth}}}\right), \quad V_{\text{smooth}} = 0.05\ \text{m/s}
\]

### 4.4 裂缝邻域置零

裂缝处流速 \(V\) 有跃变（侧向流量分流），跨缝 \(\partial V/\partial x\) 不可靠。实现中将裂缝邻域的 \(J_u\) 强制置零：

```python
if has_fractures:
    for i_f in frac_indices:
        Ju1[i_f - 1] = 0.0
        Ju2[i_f - 1] = 0.0
```

### 4.5 内节点实现

**对应代码**：`wellbore_moc.py:455-485`

```python
# C⁺ 来源点 j=0..N-2（V_prev_right）：前向差分 dV/dx
dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
k1 = brunone_k_vec(Re1b)
sign_V1 = np.tanh(V1 / V_smooth)
Ju1 = (k1 / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

# C⁻ 来源点 j=2..N（V_prev_left）：后向差分 dV/dx
dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
k2 = brunone_k_vec(Re2b)
sign_V2 = np.tanh(V2 / V_smooth)
Ju2 = (k2 / 2.0) * dt * (dVdt2 + a * sign_V2 * np.abs(dVdx2))

# 裂缝邻域置零
if has_fractures:
    for i_f in frac_indices:
        Ju1[i_f - 1] = 0.0
        Ju2[i_f - 1] = 0.0

# 叠加到稳态摩阻
J1 = J1 + Ju1
J2 = J2 + Ju2
```

### 4.6 边界节点

井口（node 0，C⁻ 来源 node 1）和趾端（node N，C⁺ 来源 node N-1）单独处理，使用单边差分：

```python
# 井口 C⁻（后向差分）
dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
J_0 += brunone_friction_Ju(k_0, dt, dVdt_0, dVdx_0, V2_0, a)

# 趾端 C⁺（前向差分）
dVdt_N = (V_prev_right[N-1] - V_prev2_right[N-1]) / dt
dVdx_N = (V_prev_right[N] - V_prev_right[N-1]) / dx
J_N += brunone_friction_Ju(k_N, dt, dVdt_N, dVdx_N, V1_N, a)
```

**对应代码**：`wellbore_moc.py:529-535`（井口）、`wellbore_moc.py:562-568`（趾端）

---

## 5. 总摩阻组合

最终叠加到特征线系数：

\[
J = J_s + J_u
\]

\[
C_P = V_1 + g_a H_1 - J + g_a \Delta t\, V_1 \sin\theta
\]
\[
C_M = -V_2 + g_a H_2 + J + g_a \Delta t\, V_2 \sin\theta
\]

```python
Cp = V1 + ga * H1 - J1 + ga * dt * V1 * theta
Cm = -V2 + ga * H2 + J2 + ga * dt * V2 * theta
H_new[1:-1] = (Cp + Cm) / (2.0 * ga)
V_new[1:-1] = Cp - ga * H_new[1:-1]
```

---

## 6. 物理意义总结

| 项 | 公式 | 作用 |
|----|------|------|
| 稳态 \(J_s\) | \(\dfrac{f \Delta t V\|V\|}{2D}\) | 与流速平方成正比，反映稳态能耗 |
| 非定常 \(J_u\) | \(\dfrac{k}{2}\Delta t (\partial_t V + a\,\text{sign}(V)\|\partial_x V\|)\) | 与瞬时加速度成正比，反映频率相关阻尼 |
| 系数 \(k\) | \(\dfrac{\sqrt{C}}{2}\)，\(C\) 由 Vardy 公式确定 | 随 \(Re\) 变化，层流/紊流分段 |

### Brunone vs 稳态摩阻的关键差异

**稳态摩阻**：阻尼力 \(\propto V|V|\)，在停泵瞬间 \(V\) 跳变但 \(V|V|\) 变化有限，无法提供足够瞬态阻尼 → 振荡不衰减。

**Brunone 摩阻**：阻尼力 \(\propto \partial V/\partial t + a\,\text{sign}(V)|\partial V/\partial x|\)，停泵瞬间 \(|\partial V/\partial t|\) 极大（流速阶跃），提供强瞬态阻尼 → 振荡快速衰减，与现场观测一致。

### 数值稳定性措施

| 措施 | 目的 |
|------|------|
| tanh 平滑 sign(V) | 消除 \(V \approx 0\) 处符号跳变导致的锯齿 |
| 裂缝邻域 \(J_u\) 置零 | 隔离跨缝 \(V\) 跃变，防 \(\partial V/\partial x\) 爆炸 |
| \(n \ge 2\) 才启用 | 需上上步 \(V^{n-1}\) 计算时间导数 |
| CFL=1 严格匹配 | 特征线精确落网格，无数值耗散污染摩阻判定 |

---

## 7. 验证脚本

| 脚本 | 用途 |
|------|------|
| `validation/leakoff_multi.py --friction brunone` | Brunone + 滤失多缝 MOC 验证 |
| `validation/cepstrum/kaiser_bessel_multi.py --friction brunone` | Brunone 下倒谱方法对比 |
| `validation/cepstrum/wlen_sweep.py --friction brunone` | Brunone 下窗长扫描 |
| `validation/step03b_brunone.py` | 单缝 Brunone 衰减率验证（对照现场 ~7s 周期） |

---

## 8. 参考文献

- Brunone B., Golia U.M., Greco M. (1991). *Some remarks on the momentum equations for fast transients*. Proc. 6th Int. Conf. on Pressure Surges.
- Vardy A.E., Brown J.M.B. (1996). *On turbulent, unsteady, smooth-pipe friction*. Proc. 7th Int. Conf. on Pressure Surges.
- Vardy A.E., Brown J.M.B. (2003). *Transient turbulent friction in smooth pipe*. J. Hydraul. Eng., 129(6), 429-439.
- Wylie E.B., Streeter V.L. (1993). *Fluid Transients in Systems*. Prentice-Hall.
