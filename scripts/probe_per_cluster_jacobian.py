# -*- coding: utf-8 -*-
"""
逐簇导纳雅可比秩探针：FNO 能否从井口波形反演「逐簇」导纳？

方法：对 N_c 簇的 K_p（或 C_f）逐一施加相同比例扰动，收集井口归一化瞬态的
变化向量 d_j = Δh_j / ||·||，组成雅可比 J = [d_1 ... d_Nc]。J 的奇异值谱直接
给出可辨识性：
  - σ_max/σ_min （条件数）>> 1 → 逐簇不可分，网络只能学到总导纳/平均导纳；
  - σ_k 与噪声底 σ_noise 之比 → 该方向的 SNR，即 Cramér-Rao 意义的可测性。
这是与网络架构无关的信息论上限：任何 FNO/DeepONet 都不可能超过它。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moc_simulate.v2.core.solver import simulate_v2
from moc_simulate.v2.configs import MocV2Config

TF = 20.0
G = 9.80665
RHO = 1000.0


def make_case(n_frac, x_start, spacing, a, cf, kleak, kp_list):
    X = [x_start + i * spacing for i in range(n_frac)]
    cfg = MocV2Config(
        wellbore_length=5000.0,
        wavespeed=a,
        dt=1.0e-3,
        tf=TF,
        pump_shut_time=1.0,
        pump_closure_duration=1.0,
        initial_velocity=1.0,
        initial_head=300.0,
        friction_model="brunone",
        store_full_field=False,
    )
    return cfg, X, [cf] * n_frac, [kleak] * n_frac, list(kp_list)


def transient(res):
    h = np.asarray(res["wellhead_head"], dtype=np.float64).ravel()
    t = np.asarray(res["timestamps"], dtype=np.float64).ravel()
    i0 = int(np.argmin(np.abs(t - 0.99)))
    H0 = float(np.mean(h[max(0, i0 - 10): i0 + 1]))
    m = t >= 1.0
    return (h[m] - H0) / max(abs(H0), 1e-9), t[m]


def jacobian(n_frac, spacing, param, base_val, pert=2.0, a=1450.0, x_start=4500.0):
    """逐簇扰动收集雅可比列。param: 'kp' | 'cf' | 'kleak'"""
    kp0 = 5.43e5
    base_kw = dict(cf=0.01, kleak=1.0e-4, kp=kp0)
    if param == "kp":
        base_kw["kp"] = base_val
    elif param == "cf":
        base_kw["cf"] = base_val
    else:
        base_kw["kleak"] = base_val

    def call(overrides):
        cfg, X, cfs, lks, kps = make_case(n_frac, x_start, spacing, a,
                                          base_kw["cf"], base_kw["kleak"],
                                          [kp0] * n_frac)
        if param == "kp":
            kps = [overrides.get(i, base_kw["kp"]) for i in range(n_frac)]
        elif param == "cf":
            cfs = [overrides.get(i, base_kw["cf"]) for i in range(n_frac)]
        else:
            lks = [overrides.get(i, base_kw["kleak"]) for i in range(n_frac)]
        return simulate_v2(cfg, fracture_positions=X, fracture_Cf=cfs,
                           fracture_kleak=lks, fracture_Kp=kps, H_ext=100.0)

    tr0, t = transient(call({}))
    amp = float(np.max(np.abs(tr0)))
    cols, labels = [], []
    for j in range(n_frac):
        tr1, _ = transient(call({j: base_val * pert}))
        cols.append((tr1 - tr0) / amp)
        labels.append(f"cluster{j+1}")
    J = np.vstack(cols).T  # (T, Nc)
    return J, labels, tr0, t, amp


def report(J, labels, tr0, amp, title, snr_db):
    u, s, vt = np.linalg.svd(J, full_matrices=False)
    # 观测噪声底：SNR 相对瞬态峰值
    sigma_n = 10.0 ** (-snr_db / 20.0)
    print(f"\n--- {title}  (SNR={snr_db}dB) ---")
    print(f"  奇异值: " + "  ".join(f"σ{k+1}={v:.4f}" for k, v in enumerate(s)))
    if s[-1] > 0:
        cond = s[0] / s[-1]
        print(f"  条件数 κ(J) = {cond:8.2f}")
    # 每个奇异方向的噪声比 = σ_k / σ_noise
    print("  方向 SNR (σ_k/σ_n): " + "  ".join(
        f"{v/sigma_n:8.2f}" for v in s))
    # 有效可辨识维数（σ_k > σ_n 的个数）
    rank = int(np.sum(s > sigma_n))
    print(f"  有效可辨识维数（σ>σ_n）= {rank} / {len(s)}")
    # 逐簇可分离性：归一化后列间最大相关
    Jn = J / (np.linalg.norm(J, axis=0, keepdims=True) + 1e-30)
    Cm = np.abs(Jn.T @ Jn)
    off = Cm[~np.eye(len(labels), dtype=bool)]
    print(f"  簇间雅可比余弦相似度: min={off.min():.4f} max={off.max():.4f} "
          f"(1.0=完全不可分)")
    return s, rank


print("=" * 92)
print("实验 3：逐簇导纳雅可比秩分析 —— FNO 反演逐簇导纳的信息上限")
print("=" * 92)
print("判据：κ(J) 越大、簇间相似度越接近 1.0，逐簇越不可分；")
print("      有效维数 < N_c 时，网络只能反演总导纳而非逐簇导纳。")

for snr_db in (40, 20):
    for n_frac, spacing in ((3, 10.0), (4, 10.0), (6, 50.0), (8, 10.0)):
        try:
            J, lab, tr0, t, amp = jacobian(n_frac, spacing, "kp", 5.43e5, 2.0)
            report(J, lab, tr0, amp,
                   f"N_c={n_frac} 间距={spacing:g}m  参数=K_p (×2)", snr_db)
        except Exception as e:
            print(f"\n--- N_c={n_frac} d={spacing}m K_p: FAILED {type(e).__name__} ---")

for snr_db in (40, 20):
    for n_frac, spacing in ((3, 10.0), (6, 50.0)):
        try:
            J, lab, tr0, t, amp = jacobian(n_frac, spacing, "cf", 0.01, 2.0)
            report(J, lab, tr0, amp,
                   f"N_c={n_frac} 间距={spacing:g}m  参数=C_f (×2)", snr_db)
        except Exception as e:
            print(f"\n--- N_c={n_frac} d={spacing}m C_f: FAILED {type(e).__name__} ---")
