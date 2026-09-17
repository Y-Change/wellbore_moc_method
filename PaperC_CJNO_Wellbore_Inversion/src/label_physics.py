# -*- coding: utf-8 -*-
"""
CJ-AlphaNet 离线标签构造：

- 从停泵后井口记录估首回波周期 T1，结合设计首簇深度得到 â = 2 x1 / T1
- 由 (q, k_leak, Kp, Cf, T1) 构造工作点等效并联导纳 Y_eq
- 活动簇 y_act、进液密度 m_alpha

不使用 HDF5 中的设计波速 wavespeed。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from moc_simulate.common.constants import CASING_OD_5_5, G

ALPHA_ACTIVE_THR: float = 0.04
A_HAT_MIN: float = 1200.0
A_HAT_MAX: float = 1600.0
Y_EQ_FLOOR: float = 1.0e-10
Q_EPS: float = 1.0e-12
SIGMA_M_DEFAULT: float = 20.0
L_DEFAULT: float = 5000.0
N_GRID_DEFAULT: int = 500
POST_SHUT_DURATION_S: float = 60.0


def pipe_area(diameter: float = CASING_OD_5_5) -> float:
    return float(np.pi * (diameter ** 2) / 4.0)


def characteristic_admittance(a_hat: float, g: float = G, area: Optional[float] = None) -> float:
    """Y0 = g A / a  [m^2/s]"""
    A = pipe_area() if area is None else float(area)
    a = max(float(a_hat), 1.0)
    return float(g * A / a)


def extract_post_shut_head(
    timestamps: np.ndarray,
    head: np.ndarray,
    ts: float = 1.0,
    duration: float = POST_SHUT_DURATION_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """截取停泵后 duration 秒的 (t, H)。"""
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    h = np.asarray(head, dtype=np.float64).reshape(-1)
    t0 = float(ts)
    t1 = t0 + float(duration)
    mask = (t >= t0 - 1.0e-12) & (t <= t1 + 1.0e-9)
    if int(np.count_nonzero(mask)) < 8:
        mask = t >= t0 - 1.0e-12
    return t[mask], h[mask]


def estimate_round_trip_period(
    head_post: np.ndarray,
    dt: float,
    x1: float,
    a_min: float = A_HAT_MIN,
    a_max: float = A_HAT_MAX,
) -> Tuple[float, bool]:
    """
    估首回波双程走时 T1。

    停泵后 dH/dt 在 T ∈ [2 x1 / a_max, 2 x1 / a_min] 内：
    1. 若自相关有显著内点主峰，用该峰（单簇/脉冲串更稳）；
    2. 否则用同一时窗内 |dH/dt| 最强峰，即首回波到时。

    返回 (T1 [s], 是否找到可用峰)。
    """
    h = np.asarray(head_post, dtype=np.float64).reshape(-1)
    n = int(h.size)
    dt = float(dt)
    x1 = float(x1)
    fallback = 2.0 * max(x1, 1.0) / (0.5 * (a_min + a_max))
    if n < 16 or not np.isfinite(x1) or x1 <= 0.0 or dt <= 0.0:
        return float(fallback), False

    dh = np.gradient(h, dt)
    dh = dh - float(np.mean(dh))

    t_lo = 2.0 * x1 / float(a_max)
    t_hi = 2.0 * x1 / float(a_min)
    lag_min = int(np.floor(t_lo / dt))
    lag_max = int(np.ceil(t_hi / dt))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, n - 2)
    if lag_max <= lag_min:
        return float(0.5 * (t_lo + t_hi)), False

    def _parabolic_lag(series: np.ndarray, peak: int) -> float:
        if peak <= 0 or peak >= int(series.size) - 1:
            return float(peak)
        y0, y1, y2 = float(series[peak - 1]), float(series[peak]), float(series[peak + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1.0e-12:
            delta = 0.5 * (y0 - y2) / denom
            return float(peak) + float(np.clip(delta, -0.5, 0.5))
        return float(peak)

    n_ac = min(n, max(lag_max + 2, int(np.round(20.0 / dt))))
    spec = np.fft.rfft(dh[:n_ac], n=2 * n_ac)
    ac = np.fft.irfft(np.abs(spec) ** 2, n=2 * n_ac)[:n_ac]
    ac = ac / (float(ac[0]) + 1.0e-12)
    lag_max_ac = min(lag_max, n_ac - 2)
    if lag_max_ac > lag_min:
        window_ac = ac[lag_min : lag_max_ac + 1]
        peak_ac = lag_min + int(np.argmax(window_ac))
        interior = peak_ac > lag_min and peak_ac < lag_max_ac
        ac_med = float(np.median(window_ac))
        ac_val = float(ac[peak_ac])
        prominent = interior and (ac_val > 0.01) and (ac_val > ac_med + 0.01)
        if prominent:
            return float(_parabolic_lag(ac, peak_ac) * dt), True

    abs_dh = np.abs(dh)
    window_e = abs_dh[lag_min : lag_max + 1]
    peak_e = lag_min + int(np.argmax(window_e))
    interior_e = peak_e > lag_min and peak_e < lag_max
    t1 = float(_parabolic_lag(abs_dh, peak_e) * dt)
    return t1, bool(interior_e)


def wavespeed_from_first_cluster(x1: float, t1: float) -> float:
    t1 = max(float(t1), 1.0e-6)
    a_hat = 2.0 * float(x1) / t1
    return float(np.clip(a_hat, A_HAT_MIN, A_HAT_MAX))


def equivalent_admittance(
    q: float,
    kleak: float,
    kp: float,
    cf: float,
    omega: float,
    q_eps: float = Q_EPS,
) -> float:
    """
    Y_eq = Re 1 / (R_perf + 1 / (G_leak + j ω Cf))
    R_perf = 2 Kp |q|,  G_leak = k^2 / (2 |q|)
    """
    q_abs = abs(float(q))
    k = float(kleak)
    kp = float(kp)
    cf = max(float(cf), 0.0)
    omega = float(omega)
    if q_abs < q_eps:
        return 0.0
    r_perf = 2.0 * kp * q_abs
    g_leak = (k ** 2) / (2.0 * q_abs)
    inner = g_leak + 1.0j * omega * cf
    if abs(inner) < 1.0e-30:
        return 0.0
    z_b = r_perf + 1.0 / inner
    if abs(z_b) < 1.0e-30:
        return 0.0
    y = 1.0 / z_b
    val = float(np.real(y))
    if not np.isfinite(val) or val < 0.0:
        return 0.0
    return val


def log10_y_over_y0(y_eq: float, y0: float, y_floor: float = Y_EQ_FLOOR) -> float:
    y0 = max(float(y0), 1.0e-16)
    return float(np.log10((max(float(y_eq), 0.0) + y_floor) / y0))


def renormalize_alpha(alpha: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    s = float(np.sum(a))
    if s > 1.0e-12:
        return (a / s).astype(np.float64)
    n = max(int(a.size), 1)
    return np.full(n, 1.0 / n, dtype=np.float64)


def active_flags(alpha: np.ndarray, threshold: float = ALPHA_ACTIVE_THR) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    s = float(np.sum(a))
    if s <= 1.0e-12:
        return np.zeros(int(a.size), dtype=np.int32)
    return ((a / s) >= float(threshold)).astype(np.int32)


def inflow_density_field(
    positions: np.ndarray,
    alpha: np.ndarray,
    L: float = L_DEFAULT,
    n_grid: int = N_GRID_DEFAULT,
    sigma_m: float = SIGMA_M_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """高斯铺开 m_alpha(x)，Riemann 积分归一化为 1。返回 (grid_x, field)."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    w = renormalize_alpha(alpha)
    n = min(int(pos.size), int(w.size))
    grid_x = np.linspace(0.0, float(L), int(n_grid), dtype=np.float64)
    dx = float(L) / max(int(n_grid) - 1, 1)
    field = np.zeros(int(n_grid), dtype=np.float64)
    if n == 0:
        return grid_x.astype(np.float32), field.astype(np.float32)
    inv = 1.0 / (np.sqrt(2.0 * np.pi) * float(sigma_m))
    two_s2 = 2.0 * (float(sigma_m) ** 2)
    for j in range(n):
        field += w[j] * inv * np.exp(-((grid_x - pos[j]) ** 2) / two_s2)
    integ = float(np.sum(field) * dx)
    if integ > 1.0e-12:
        field = field / integ
    return grid_x.astype(np.float32), field.astype(np.float32)


def build_sample_labels(
    timestamps: np.ndarray,
    wellhead_head: np.ndarray,
    positions: np.ndarray,
    alpha: np.ndarray,
    q_ss: np.ndarray,
    kleak: np.ndarray,
    kp: np.ndarray,
    cf: np.ndarray,
    ts: float = 1.0,
    duration: float = POST_SHUT_DURATION_S,
    g: float = G,
    diameter: float = CASING_OD_5_5,
) -> Dict[str, np.ndarray]:
    """单井：估 T1/â，构造 y_act、Y_eq、m_alpha。"""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    n = int(pos.size)
    alpha_n = renormalize_alpha(alpha)
    q = np.asarray(q_ss, dtype=np.float64).reshape(-1)[:n]
    k = np.asarray(kleak, dtype=np.float64).reshape(-1)[:n]
    kp_arr = np.asarray(kp, dtype=np.float64).reshape(-1)[:n]
    cf_arr = np.asarray(cf, dtype=np.float64).reshape(-1)[:n]

    x1 = float(np.min(pos)) if n > 0 else 0.0
    t_post, h_post = extract_post_shut_head(timestamps, wellhead_head, ts=ts, duration=duration)
    if t_post.size >= 2:
        dt = float(np.median(np.diff(t_post)))
    else:
        dt = 1.0e-3
    t1, ok = estimate_round_trip_period(h_post, dt=dt, x1=x1)
    a_hat = wavespeed_from_first_cluster(x1, t1) if n > 0 else 0.5 * (A_HAT_MIN + A_HAT_MAX)
    omega = 2.0 * np.pi / max(t1, 1.0e-6)
    y0 = characteristic_admittance(a_hat, g=g, area=pipe_area(diameter))

    y_act = active_flags(alpha_n)
    y_eq = np.zeros(n, dtype=np.float64)
    log_y = np.zeros(n, dtype=np.float64)
    for j in range(n):
        if int(y_act[j]) == 0:
            y_eq[j] = 0.0
        else:
            y_eq[j] = equivalent_admittance(q[j], k[j], kp_arr[j], cf_arr[j], omega)
        log_y[j] = log10_y_over_y0(y_eq[j], y0)

    grid_x, m_alpha = inflow_density_field(pos, alpha_n)

    return {
        "n_frac": np.array(n, dtype=np.int32),
        "positions": pos.astype(np.float32),
        "alpha": alpha_n.astype(np.float32),
        "y_active": y_act.astype(np.int32),
        "Y_eq": y_eq.astype(np.float32),
        "log10_Y_over_Y0": log_y.astype(np.float32),
        "x1": np.array(x1, dtype=np.float32),
        "T1": np.array(t1, dtype=np.float32),
        "a_hat": np.array(a_hat, dtype=np.float32),
        "a_hat_ok": np.array(1 if ok else 0, dtype=np.int32),
        "Y0": np.array(y0, dtype=np.float32),
        "omega_star": np.array(omega, dtype=np.float32),
        "grid_x": grid_x,
        "m_alpha_grid": m_alpha,
        "dt_post": np.array(dt, dtype=np.float32),
    }
