# -*- coding: utf-8 -*-
"""
导纳可辨识性探针 v2：把「稳态通道(DC)」与「声学通道(瞬态波形)」彻底分离。

判据：把井口波形分解为
  H(t) = H0(参数) + h(t; 参数)      H0 = 关泵前稳态水头
FNO 若从波形学参数，可以走 H0 这条非声学捷径。真正携带导纳信息的只有
归一化瞬态包络 h/H0。逐参数测两条通道的相对敏感度，与噪声底 (20dB=0.10,
10dB=0.32) 对比，判定哪些参数在声学通道内可见。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moc_simulate.v2.core.solver import simulate_v2
from moc_simulate.v2.configs import MocV2Config

# ---- Base Case: 3 簇，现场尺度 ----
L = 5000.0
X = [4500.0, 4510.0, 4520.0]
A_WAVE = 1450.0
CF, KLEAK, KP = 0.01, 1.0e-4, 5.43e5
TF = 20.0


def run(cf, kleak, kp, tc=1.0, a=A_WAVE):
    cfg = MocV2Config(
        wellbore_length=L,
        wavespeed=a,
        dt=1.0e-3,
        tf=TF,
        pump_shut_time=1.0,
        pump_closure_duration=tc,
        initial_velocity=1.0,
        initial_head=300.0,
        friction_model="brunone",
        store_full_field=False,
    )
    n = len(X)
    return simulate_v2(
        cfg,
        fracture_positions=X,
        fracture_Cf=[cf] * n,
        fracture_kleak=[kleak] * n,
        fracture_Kp=[kp] * n,
        H_ext=100.0,
    )


def split(res):
    h = np.asarray(res["wellhead_head"], dtype=np.float64).ravel()
    t = np.asarray(res["timestamps"], dtype=np.float64).ravel()
    i0 = int(np.argmin(np.abs(t - 0.99)))  # 关泵前最后一刻
    H0 = float(np.mean(h[max(0, i0 - 10): i0 + 1]))
    m = t >= 1.0
    tr = (h[m] - H0) / max(abs(H0), 1e-9)  # 归一化瞬态（去掉 DC 通道）
    return H0, tr


ref = run(CF, KLEAK, KP)
H0r, tr_r = split(ref)
amp = float(np.max(np.abs(tr_r)))
print("=" * 88)
print("实验 2：DC 通道(稳态水头) vs 声学通道(归一化瞬态) 分离敏感度")
print("=" * 88)
print(f"参考: H0 = {H0r:.3f} m, 归一化瞬态峰值 |h/H0|max = {amp:.5f}\n")

TESTS = [
    ("K_p    x2.0", dict(kp=KP * 2.0)),
    ("K_p    x0.5", dict(kp=KP * 0.5)),
    ("C_f    x3.0", dict(cf=CF * 3.0)),
    ("C_f    x0.33", dict(cf=CF / 3.0)),
    ("k_leak x5.0", dict(kleak=KLEAK * 5.0)),
    ("k_leak x0.5", dict(kleak=KLEAK * 0.5)),
    ("t_c    x2.0", dict(tc=2.0)),
    ("a      +1%", dict(a=A_WAVE * 1.01)),
    ("a      -1%", dict(a=A_WAVE * 0.99)),
]

print(f"{'扰动':<16}{'dH0/H0':>10}{'DC-dB':>9}{'||dh||':>11}{'AC-dB':>9}{'时移敏感度':>12}")
print("-" * 88)
out = []
for name, kw in TESTS:
    try:
        r = run(**{k: v for k, v in dict(
            cf=CF, kleak=KLEAK, kp=KP, tc=1.0, a=A_WAVE).items()})
        r = run(**{**dict(cf=CF, kleak=KLEAK, kp=KP, tc=1.0, a=A_WAVE), **kw})
    except Exception as e:
        print(f"{name:<16}{'FEASIBLE-FAIL':>10}")
        continue
    H0, tr = split(r)
    dc = abs(H0 - H0r) / abs(H0r)
    # 波形差异（先做时移对齐，分离「幅度信息」与「时延信息」）
    d_raw = np.max(np.abs(tr - tr_r))
    # 时延敏感度：允许 ±50ms 平移后的最小差异
    kmax = 50
    best = d_raw
    for k in range(-kmax, kmax + 1):
        a_ = np.roll(tr, k)
        if k > 0:
            cmp_ = tr_r[k:]
            sl = a_[k:]
        elif k < 0:
            cmp_ = tr_r[:k]
            sl = a_[:k]
        else:
            cmp_, sl = tr_r, a_
        if sl.size:
            best = min(best, float(np.max(np.abs(sl - cmp_))))
    ac = best / max(amp, 1e-12)
    d_amp = d_raw / max(amp, 1e-12)
    out.append((name, dc, ac, d_amp))
    print(f"{name:<16}{dc:>10.5f}{20*np.log10(max(dc,1e-12)):>9.2f}"
          f"{ac:>11.5f}{20*np.log10(max(ac,1e-12)):>9.2f}{d_amp:>12.5f}")

print("\n判读：AC-dB < -20 dB 表示该参数在声学通道低于 20dB 噪声底 → 水击波里看不见。")
print("      时移敏感度列若远大于 AC 列，说明该参数主要改变时延而非幅度（位置信息）。")
