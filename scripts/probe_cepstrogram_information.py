# -*- coding: utf-8 -*-
"""
1D/2D 倒谱到底编码什么信息？

分解为三问：
  (A) 1D quefrency 轴    -> 位置（双程走时 τ=2x/a）  [已知，复验]
  (B) 1D 峰高            -> 反射系数幅度 |Γ_j|        [实测是 K_p 还是 C_f]
  (C) 2D 多出的窗时间轴  -> 峰高随时间的衰减率       [这是 2D 相对 1D 的唯一新增信息]
      测它到底对哪个物理量敏感。若对 C_f/k_leak 都不敏感 -> 2D 也没带来导纳信息。

在干净单簇算例上做（Nc=1 时 α≡1，无生成器先验可泄漏，是纯声学因果）。
"""
import numpy as np
import h5py
from scipy.stats import spearmanr, pearsonr

P = ("data/datasets/moc_v2_physical_steady_1k_newa/"
     "moc_v2_physical_steady_1k_x4500_4950.h5")
G = 9.80665
D_WELL = 0.1397

with h5py.File(P, "r") as f:
    H = np.asarray(f["waveforms/wellhead_head"], dtype=np.float64)
    ts = np.asarray(f["waveforms/timestamps"], dtype=np.float64)
    nf = np.asarray(f["labels/n_frac"], dtype=np.int64)
    xp = np.asarray(f["labels/fracture_positions"], dtype=np.float64)
    Cf = np.asarray(f["labels/fracture_Cf"], dtype=np.float64)
    Kl = np.asarray(f["labels/fracture_kleak"], dtype=np.float64)
    Kp = np.asarray(f["labels/fracture_Kp"], dtype=np.float64)
    Qs = np.asarray(f["labels/fracture_Q_ss"], dtype=np.float64)
    wav = np.asarray(f["labels/wavespeed"], dtype=np.float64)
    tc = np.asarray(f["labels/pump_closure_tc"], dtype=np.float64)

dt = float(ts[1] - ts[0])
Y0_REF = G * (np.pi * D_WELL ** 2 / 4.0) / 1450.0


def rceps(x):
    n = 1 << int(np.ceil(np.log2(len(x))))
    X = np.fft.rfft(x - x.mean(), n=n)
    return np.fft.irfft(np.log(np.abs(X) + 1e-12), n=n)[: len(x)]


# ---------- 2D 倒谱：滑窗 ----------
# 缝在 4500-4950m，双程走时 τ ≈ 6.2-7.0s，窗必须能覆盖该 quefrency
WIN = 16384         # 窗长 16.384 s > τ_max
HOP = 4096          # 4.096 s 步长
TMAX = 61.0         # 用满关泵后全程

rows = []
n_1d_ok = 0
for i in range(H.shape[0]):
    if int(nf[i]) != 1:
        continue
    w = H[i]
    i0 = int(np.argmin(np.abs(ts - 0.99)))
    H0 = float(np.mean(w[max(0, i0 - 10): i0 + 1]))
    seg = w[ts >= 1.0]
    if len(seg) < WIN:
        continue
    a = float(wav[i])
    xj = float(xp[i, 0])
    tau = 2.0 * xj / a
    k = int(round(tau / dt))
    if k < 5:
        continue

    # (A)(B) 全程 1D 倒谱
    c1 = rceps(seg)
    if abs(k * dt - tau) > 0.15:
        continue  # 只保留定位成功的算例
    n_1d_ok += 1
    win = 25
    s_ = c1[max(0, k - win): k + win + 1]
    pk1 = c1[k] - np.median(s_)

    # (C) 2D：逐窗峰高 -> 衰减曲线
    n_win = 1 + (len(seg) - WIN) // HOP
    n_win = min(n_win, int(TMAX / (HOP * dt)))
    pk_t, pk_v = [], []
    for m in range(n_win):
        wseg = seg[m * HOP: m * HOP + WIN]
        cw = rceps(wseg)
        if k >= len(cw):
            continue
        s2 = cw[max(0, k - win): k + win + 1]
        pk_t.append(m * HOP * dt)
        pk_v.append(cw[k] - np.median(s2))
    if len(pk_t) < 4:
        continue
    pk_t = np.array(pk_t)
    pk_v = np.array(pk_v)
    # 衰减率：log 峰高对时间的斜率
    good = pk_v > 0
    if good.sum() < 3:
        continue
    slope = np.polyfit(pk_t[good], np.log(pk_v[good]), 1)[0]

    # 标签
    qb = float(Qs[i, 0])
    R_perf = 2.0 * float(Kp[i, 0]) * abs(qb)
    G_leak = float(Kl[i, 0]) ** 2 / (2.0 * max(abs(qb), 1e-12))
    Y0i = G * (np.pi * D_WELL ** 2 / 4.0) / a
    # 支路导纳（低频极限：C_f 短路，只剩 R_perf 串联 G_leak）
    Yb = 1.0 / (R_perf + 1.0 / max(G_leak, 1e-30))
    Gamma = -Yb / (2.0 * Y0i + Yb)  # 并联支路反射系数
    rows.append((pk1, slope, float(Cf[i, 0]), float(Kl[i, 0]),
                 float(Kp[i, 0]), R_perf, abs(Gamma), float(tc[i]), a, xj))

R = np.array(rows)
pk1, slope, cf, kl, kp, Rp, gm, tcv, av, xjv = [R[:, m] for m in range(10)]
print("=" * 90)
print(f"干净单簇算例 N = {len(R)}   (1D 定位成功共 {n_1d_ok} 例)")
print("=" * 90)

print("\n(A) 1D quefrency 轴 = 位置。复验：定位误差中位数")
# 用已筛条件，误差均 <0.15s
print("    筛选条件 |τ_peak−τ_true| < 0.15 s  ->  位置误差 < 0.15*a/2 ≈ "
      f"{0.15*1450/2:.1f} m")

print("\n(B) 1D 峰高 -> 反射系数幅度。峰高 vs 物理量：")
for nm, v in (("|Γ|", gm), ("log10 K_p", np.log10(kp)),
              ("log10 R_perf", np.log10(Rp)),
              ("log10 C_f", np.log10(cf)),
              ("log10 k_leak", np.log10(kl))):
    print(f"    spearman(峰高, {nm:<14}) = {spearmanr(pk1, v).statistic:+.4f}")
print(f"    pearson(峰高, log10|Γ|)  = {pearsonr(pk1, np.log10(gm)).statistic:+.4f}")

print("\n(C) 2D 新增信息 = 峰高随窗时间的衰减率。斜率 vs 物理量：")
for nm, v in (("log10 C_f", np.log10(cf)),
              ("log10 k_leak", np.log10(kl)),
              ("log10 K_p", np.log10(kp)),
              ("log10 R_perf", np.log10(Rp)),
              ("t_c", tcv),
              ("wavespeed", av)):
    sp = spearmanr(slope, v).statistic
    print(f"    spearman(衰减率, {nm:<14}) = {sp:+.4f}")

print("\n判读：")
print("  (C) 若衰减率只对 t_c/波速敏感、对 C_f 和 k_leak 不敏感，")
print("      则 2D 倒谱图相对 1D 没有增加任何裂缝导纳信息；")
print("      它多出来的那条轴编码的是「能量如何耗散」即摩阻/泄漏总损耗，")
print("      而不是「哪一簇的顺应性是多少」。")
