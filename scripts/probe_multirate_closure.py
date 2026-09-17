# -*- coding: utf-8 -*-
"""
多工况阻抗桥闭合性验证。

背景：单工况下「声学给 (x_j, K_p,j)，稳态给 H0 一个标量，未知 k_leak,j 有 N_c 个」
→ 欠定 N_c-1 维。若改用 M 个不同排量工况，每个工况给 1 个质量守恒方程，
M >= N_c 时系统是否闭合？本脚本直接数值验证（只用稳态求解器，无需跑 MOC）。

方法：
  1. 设定真值 k_leak,j 与 K_p,j、位置 x_j；
  2. 用 solve_physical_steady_state 在 M 个排量 Q0^(m) 下正向解出 H0^(m)；
  3. 假装只知 (x_j, K_p,j, H0^(m), Q0^(m))，最小二乘反解 k_leak,j；
  4. 检查 M 增大时反解是否收敛到真值（欠定性是否被闭合）。
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moc_simulate.v2.core.initial_field import solve_physical_steady_state

G = 9.80665
L, D, NU, KD = 5000.0, 0.1397, 1.0e-6, 0.02 * 0.1397 / (2 * np.sqrt(2.0))
AREA = np.pi * D ** 2 / 4.0
DT = 1.0e-3
N = int(round(L / (1450.0 * DT)))
DX = L / N
A_ADJ = DX / DT

X = [4500.0, 4520.0, 4540.0]       # 20m 间距，可分
KP = [5.43e5, 8.0e5, 4.0e5]
KL_TRUE = [1.0e-4, 2.5e-4, 0.6e-4]
HEXT = 100.0


def forward_h0(q0, kl, n_active):
    """给定总排量与逐簇 k_leak，正向解出井口稳态水头 H0。"""
    idx = [int(round(x / DX)) for x in X[:n_active]]
    res = solve_physical_steady_state(
        L, N, DX, D, AREA, NU, KD, q0 / AREA, G, "dead_end",
        idx, np.asarray(kl), np.asarray(KP[:n_active]),
        X[:n_active], H_ext=HEXT, H0_max=1e9,
    )
    return res[0]


def solve_inverse(multi_h0, q0_list, n_active):
    """已知 M 个 (Q0, H0) 与 (x_j, K_p,j)，反解 k_leak,j。"""
    def resid(p):
        kl = np.clip(np.exp(p), 1e-7, 1e-2)
        out = [forward_h0(q0, kl, n_active) for q0 in q0_list]
        return np.asarray(out) - np.asarray(multi_h0)

    # 软正则：欠定时把解拉向先验中心，避免 LM 对 M<N_c 直接拒绝
    def resid_full(p):
        r = resid(p)
        return np.concatenate([r, 1e3 * (p - np.log(1.0e-4))])

    p0 = np.log(np.full(n_active, 1.0e-4))
    sol = least_squares(resid_full, p0, method="trf",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=20000)
    return np.clip(np.exp(sol.x), 1e-7, 1e-2), sol


print("=" * 78)
print("多工况阻抗桥闭合性验证  N_c=3, 20m 间距")
print("=" * 78)
print(f"真值 k_leak = {KL_TRUE}")
print(f"H_ext = {HEXT} m, K_p = {KP}")
print()

# 单簇基准（Nc=1 时应精确可解，作为求解器正确性对照）
print("--- 对照：N_c=1 单簇，单工况应精确闭合 ---")
kl1 = [KL_TRUE[0]]
h0_1 = forward_h0(0.0113, kl1, 1)
inv1, _ = solve_inverse([h0_1], [0.0113], 1)
print(f"  真值 {kl1[0]:.4e}  反解 {inv1[0]:.4e}  相对误差 {abs(inv1[0]/kl1[0]-1)*100:.3f}%")
print()

Q0_ALL = [0.008, 0.011, 0.014, 0.017, 0.020]  # 5 个排量工况
print("--- N_c=3：逐步增加工况数 M，看反解是否收敛 ---")
print(f"{'M':>3}  {'反解 k_leak (3簇)':>44}  {'max 相对误差':>12}")
print("-" * 78)
for m in range(1, 6):
    qs = Q0_ALL[:m]
    h0s = [forward_h0(q, KL_TRUE, 3) for q in qs]
    inv, sol = solve_inverse(h0s, qs, 3)
    err = np.max(np.abs(inv / np.asarray(KL_TRUE) - 1))
    print(f"{m:>3}  " + "  ".join(f"{v:.3e}" for v in inv) +
          f"  {err*100:>11.2f}%")

print("\n判读：")
print("  M=1,2 → 欠定，反解应显著偏离真值；")
print("  M>=3  → 系统闭合，反解应收敛到真值。")
print("  若 M>=3 仍不收敛，说明 H0 对 k_leak 的映射存在多解/弱可观测，")
print("  需重新评估多工况方案的信息量。")
