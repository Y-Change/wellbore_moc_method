# -*- coding: utf-8 -*-
"""
倒谱峰幅度 vs 真实导纳标签：在现有 1000 例上做零成本直接回归。

问题：FNO 从「井口压力 + 倒谱」能否拿到井下导纳？
先不训网络——直接测「倒谱峰高度」这一最强手工特征与真实 Y_eq 的关系。
若手工上限就低，任何网络都不会更高；若高，说明信息确实在波形里。
"""
import numpy as np
import h5py
from scipy.stats import spearmanr, pearsonr

P = "data/datasets/moc_v2_physical_steady_1k_newa/moc_v2_physical_steady_1k_x4500_4950.h5"
G, RHO = 9.80665, 1000.0

with h5py.File(P, "r") as f:
    H = np.asarray(f["waveforms/wellhead_head"], dtype=np.float64)
    ts = np.asarray(f["waveforms/timestamps"], dtype=np.float64)
    n_frac = np.asarray(f["labels/n_frac"], dtype=np.int64)
    xpos = np.asarray(f["labels/fracture_positions"], dtype=np.float64)
    Cf = np.asarray(f["labels/fracture_Cf"], dtype=np.float64)
    Kl = np.asarray(f["labels/fracture_kleak"], dtype=np.float64)
    Kp = np.asarray(f["labels/fracture_Kp"], dtype=np.float64)
    alpha = np.asarray(f["labels/fracture_alpha_ss"], dtype=np.float64)
    Qss = np.asarray(f["labels/fracture_Q_ss"], dtype=np.float64)
    wav = np.asarray(f["labels/wavespeed"], dtype=np.float64)

dt = float(ts[1] - ts[0])
N = H.shape[1]


def rceps(x):
    """实倒谱 = IFFT(log|FFT|)，用相位无关的实部。"""
    n = 1 << int(np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x - x.mean(), n=n)
    return np.fft.irfft(np.log(np.abs(X) + 1e-12), n=n)[: len(x)]


# 每簇标签：R_perf = 2 K_p |q̄_j|；q̄ 由稳态 Q_ss 给出
rows = []
for i in range(H.shape[0]):
    nf = int(n_frac[i])
    if nf < 2:
        continue
    w = H[i]
    i0 = int(np.argmin(np.abs(ts - 0.99)))
    H0 = float(np.mean(w[max(0, i0 - 10): i0 + 1]))
    tr = w[ts >= 1.0] - H0
    c = rceps(tr)
    q = np.linspace(0, (len(c) - 1) * dt, len(c))
    a = float(wav[i])
    for j in range(nf):
        xj = float(xpos[i, j])
        tau = 2.0 * xj / a
        k = int(round(tau / dt))
        if k < 2 or k >= len(c):
            continue
        # 峰高 = 局部极大减去左右谷底（相对背景）
        win = 25
        seg = c[max(0, k - win): k + win + 1]
        pk = c[k] - np.median(seg)
        qbar = float(Qss[i, j])  # m^3/s 单簇稳态流量
        R_perf = 2.0 * float(Kp[i, j]) * abs(qbar)
        G_leak = float(Kl[i, j]) ** 2 / (2.0 * max(abs(qbar), 1e-12))
        Y0 = G * (np.pi * 0.1397 ** 2 / 4.0) / a
        # 声学导纳探针：Re[1/(R + 1/(G + j w C))]
        w1 = 2.0 * np.pi / 6.0  # ~首波周期 6s
        den = R_perf + 1.0 / (G_leak + 1j * w1 * float(Cf[i, j]))
        Y_eq = (1.0 / den).real
        rows.append((i, j, pk, Y_eq, 1.0 / max(R_perf, 1e-30),
                     float(Cf[i, j]), float(Kl[i, j]), float(Kp[i, j]),
                     float(alpha[i, j]), xj, a))

R = np.array(rows)
print("=" * 84)
print("实验 4：倒谱峰幅度（真实声学特征）vs 导纳标签  N =", len(R))
print("=" * 84)

pk, Yeq, invR, cf, kl, kp, al, xj, a = [R[:, k] for k in range(2, 11)]

print(f"\n{'标签':<22}{'spearman':>11}{'pearson':>11}{'中位比':>10}")
print("-" * 84)
for name, tgt in (("log10 Y_eq", np.log10(Yeq + 1e-12)),
                  ("log10 1/R_perf", np.log10(invR)),
                  ("log10 C_f", np.log10(cf)),
                  ("log10 k_leak", np.log10(kl)),
                  ("log10 K_p", np.log10(kp)),
                  ("alpha_ss", al)):
    sp = spearmanr(pk, tgt).statistic
    pe = pearsonr(pk, np.log10(np.abs(pk) + 1e-12) * 0 + tgt).statistic
    print(f"{name:<22}{sp:>11.4f}{pe:>11.4f}")

print("\n关键判读：")
print("  spearman(倒谱峰高, log10 Y_eq) 与 spearman(倒谱峰高, log10 1/R_perf)")
print("  若几乎相等 → 峰高携带的是射孔摩阻信息，不是裂缝顺应性信息。")
print(f"\n  实测 spearman(峰高, log10 1/R_perf) = {spearmanr(pk, np.log10(invR)).statistic:.4f}")
print(f"  实测 spearman(峰高, log10 C_f)     = {spearmanr(pk, np.log10(cf)).statistic:.4f}")
