# -*- coding: utf-8 -*-
"""
测试 B — 2D 短时倒谱优化方案对比验证（Kaiser-Bessel / Epsilon / Lifter / AR）

对比四种方案相对基线 Hamming / Kaiser(beta=4) 在裂缝深度定位上的效果：
  方案1  动态约束 Kaiser-Bessel 加窗
  方案2  epsilon 偏置对数变换 + 预加重
  方案3  升余弦倒谱 Lifter
  方案4  参数化 AR 现代谱倒谱估计

运行（在 wellbore_moc_method/ 目录下）:
    python validate_moc_test_b_Kaiser-Bessel.py
"""
from __future__ import annotations

import os
import sys
import time as time_module
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal

_METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
if _METHOD_DIR not in sys.path:
    sys.path.insert(0, _METHOD_DIR)

from cepstrum_mocdata import cepstrogram, preprocess_moc_head
from paths import moc_output_dir
from wellbore_moc import MocConfig, simulate_wellbore

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ── 方案 1–3：复合优化 2D 短时倒谱 ────────────────────────────────

def _preemphasis(frame: np.ndarray, mu: float = 0.97) -> np.ndarray:
    """一阶预加重: y(n) = x(n) - mu * x(n-1)"""
    y = np.empty_like(frame)
    y[0] = frame[0]
    y[1:] = frame[1:] - mu * frame[:-1]
    return y


def _dynamic_kaiser_beta(
    target_depth_m: Optional[float],
    fs: float,
    wavespeed: float,
) -> float:
    """方案1：由目标裂缝深度估算 Kaiser 形状参数 beta。"""
    if target_depth_m is None:
        return 8.6
    tau_expected = 2.0 * target_depth_m / wavespeed
    return float(np.clip(tau_expected * fs / 100.0, 4.0, 14.0))


def _raised_sine_lifter(
    n_bins: int,
    fs: float,
    tau_max: float,
    K: float = 22.0,
    zero_cutoff_sec: float = 0.01,
) -> np.ndarray:
    """方案3：升余弦倒谱 Lifter  L(tau) = 1 + (K/2) sin(pi tau / tau_max)。"""
    q = np.arange(n_bins) / fs
    lifter = np.ones(n_bins, dtype=float)
    mask = (q > 0) & (q < tau_max)
    lifter[mask] = 1.0 + (K / 2.0) * np.sin(np.pi * q[mask] / tau_max)
    lifter[q < zero_cutoff_sec] = 0.0
    return lifter


def optimized_2d_cepstrogram(
    x: np.ndarray,
    fs: float,
    window_length_sec: float,
    wavespeed: float = 1450.0,
    wellbore_length: float = 5000.0,
    hop_ratio: float = 0.2,
    target_depth_m: Optional[float] = None,
    *,
    use_dynamic_kaiser: bool = True,
    use_epsilon_log: bool = True,
    use_preemphasis: bool = True,
    use_lifter: bool = True,
    epsilon: float = 1e-12,
    preemph_mu: float = 0.97,
    lifter_K: float = 22.0,
    lifter_tau_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    针对深层压裂裂缝特征优化的 2D 短时倒谱。

    返回
    ----
    time_axis, quefrency_axis, cepstrogram  (shape: [n_q, n_frames])
    """
    x = np.asarray(x, dtype=float).flatten()
    n_win = max(4, int(window_length_sec * fs))
    n_hop = max(1, int(n_win * hop_ratio))
    n_fft = int(2 ** np.ceil(np.log2(n_win) + 2))
    n_cep = _max_quefrency_bins(fs, n_win, wavespeed, wellbore_length)

    beta = _dynamic_kaiser_beta(target_depth_m, fs, wavespeed) if use_dynamic_kaiser else 8.6
    window = signal.windows.kaiser(n_win, beta)

    if lifter_tau_max is None:
        lifter_tau_max = 2.0 * wellbore_length / wavespeed
    lifter = _raised_sine_lifter(n_cep, fs, lifter_tau_max, K=lifter_K)

    num_frames = 1 + (len(x) - n_win) // n_hop
    cgram = np.zeros((n_cep, num_frames))
    time_axis = np.zeros(num_frames)
    quefrency_axis = np.arange(n_cep) / fs

    for i in range(num_frames):
        start = i * n_hop
        frame = x[start:start + n_win]
        if len(frame) < n_win:
            break

        if use_preemphasis:
            frame = _preemphasis(frame, mu=preemph_mu)

        frame = frame - np.mean(frame)
        windowed = frame * window
        spectrum = np.fft.rfft(windowed, n=n_fft)

        if use_epsilon_log:
            power_spec = np.abs(spectrum) ** 2 + epsilon
            log_spec = np.log(power_spec)
        else:
            log_spec = np.log(np.abs(spectrum) + np.finfo(float).eps)

        ceps_frame = np.fft.irfft(log_spec, n=n_fft)[:n_cep]
        if use_lifter:
            ceps_frame = ceps_frame * lifter

        cgram[:, i] = ceps_frame
        time_axis[i] = (start + n_win / 2.0) / fs

    return time_axis, quefrency_axis, cgram


# ── 方案 4：AR 递推倒谱 ───────────────────────────────────────────

def ar_cepstrum_frame(
    frame: np.ndarray,
    p_order: int = 30,
    num_coeffs: Optional[int] = None,
) -> np.ndarray:
    """
    单帧 AR 现代谱倒谱（Yule-Walker + 递推公式）。
    """
    frame = np.asarray(frame, dtype=float).flatten()
    n = len(frame)
    if num_coeffs is None:
        num_coeffs = p_order * 2
    frame = frame - np.mean(frame)

    if n <= p_order + 2:
        return np.zeros(num_coeffs)

    r = np.correlate(frame, frame, mode='full')[n - 1:n + p_order + 1]
    R = linalg.toeplitz(r[:p_order])
    try:
        a_pos = linalg.solve(R, r[1:p_order + 1])
    except linalg.LinAlgError:
        return np.zeros(num_coeffs)

    # x(n) = -sum a_k x(n-k) + e(n)  中的 a_k
    a = a_pos

    c = np.zeros(num_coeffs + 1)
    c[1] = -a[0]
    for n_idx in range(2, p_order + 1):
        c[n_idx] = -a[n_idx - 1]
        for k in range(1, n_idx):
            c[n_idx] -= (1.0 - k / n_idx) * a[k - 1] * c[n_idx - k]
    for n_idx in range(p_order + 1, num_coeffs + 1):
        for k in range(1, p_order + 1):
            c[n_idx] -= (1.0 - k / n_idx) * a[k - 1] * c[n_idx - k]

    return c[1:num_coeffs + 1]


def ar_2d_cepstrogram(
    x: np.ndarray,
    fs: float,
    window_length_sec: float,
    hop_ratio: float = 0.2,
    p_order: int = 60,
    wavespeed: float = 1450.0,
    wellbore_length: float = 5000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """滑窗 AR 倒谱图；num_coeffs 覆盖全井深倒频率范围。"""
    x = np.asarray(x, dtype=float).flatten()
    n_win = max(p_order + 4, int(window_length_sec * fs))
    n_hop = max(1, int(n_win * hop_ratio))
    # 覆盖 depth <= L 所需倒频率 q_max = 2L/v
    num_coeffs = int(np.ceil(2.0 * wellbore_length / wavespeed * fs)) + 64
    p_order = min(p_order, n_win // 3)

    num_frames = 1 + (len(x) - n_win) // n_hop
    cgram = np.zeros((num_coeffs, num_frames))
    time_axis = np.zeros(num_frames)
    quefrency_axis = np.arange(1, num_coeffs + 1) / fs

    for i in range(num_frames):
        start = i * n_hop
        frame = x[start:start + n_win]
        if len(frame) < n_win:
            break
        cgram[:, i] = ar_cepstrum_frame(frame, p_order=p_order, num_coeffs=num_coeffs)
        time_axis[i] = (start + n_win / 2.0) / fs

    return time_axis, quefrency_axis, cgram


# ── 基线方法 ─────────────────────────────────────────────────────

def _depth_axis(q: np.ndarray, v: float) -> np.ndarray:
    return q * v / 2.0


def baseline_2d_cepstrogram(
    x: np.ndarray,
    fs: float,
    window_length_sec: float,
    hop_ratio: float = 0.2,
    win_type: str = 'hamming',
    kaiser_beta: float = 4.0,
    wavespeed: float = 1450.0,
    wellbore_length: float = 5000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基线：Hamming 或固定 beta Kaiser + log|X|，深度裁剪至井长。"""
    n_win = max(4, int(window_length_sec * fs))
    n_hop = max(1, int(n_win * hop_ratio))
    C, q, t = cepstrogram(x, wlen=n_win, hop=n_hop, fs=fs, win_type=win_type)
    if win_type == 'kaiser':
        _ = kaiser_beta
    n_cep = _max_quefrency_bins(fs, n_win, wavespeed, wellbore_length)
    return t, q[:n_cep], C[:n_cep, :]


# ── 评估指标 ─────────────────────────────────────────────────────

def _max_quefrency_bins(
    fs: float,
    n_win: int,
    wavespeed: float,
    wellbore_length: float,
) -> int:
    """倒谱有效 bin 数：不超过窗长分辨率，也不超过全井深 2L/v。"""
    n_by_window = int(np.ceil((1 + n_win) / 2))
    q_max_phys = 2.0 * wellbore_length / wavespeed
    n_by_depth = int(np.floor(q_max_phys * fs)) + 1
    return min(n_by_window, n_by_depth)


def _crop_cepstrum_by_depth(
    C: np.ndarray,
    q: np.ndarray,
    v: float,
    depth_min: float,
    depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """按物理深度裁剪倒谱矩阵。"""
    depth = _depth_axis(q, v)
    mask = (depth >= depth_min) & (depth <= depth_max)
    if not np.any(mask):
        return C, q
    return C[mask, :], q[mask]


def evaluate_fracture_peak(
    C: np.ndarray,
    q: np.ndarray,
    v: float,
    true_depth_m: float,
    depth_min: float = 100.0,
    depth_max: Optional[float] = None,
) -> Dict:
    """
    沿时间平均的倒谱深度剖面，找主峰并报告与真实缝深误差。
    """
    depth = _depth_axis(q, v)
    if depth_max is None:
        depth_max = float(depth[-1]) if len(depth) else 5000.0

    mask = (depth >= depth_min) & (depth <= depth_max)
    if not np.any(mask):
        return {
            'peak_depth_m': np.nan,
            'error_m': np.nan,
            'peak_val': np.nan,
            'snr': np.nan,
            'profile_depth': depth[mask] if np.any(mask) else depth,
            'profile': np.array([]),
        }

    profile = -np.mean(C[mask, :], axis=1)
    depth_kept = depth[mask]
    i_peak = int(np.argmax(profile))
    peak_depth = float(depth_kept[i_peak])
    bg = np.percentile(profile, 50)
    peak_val = float(profile[i_peak])
    snr = peak_val / max(abs(bg), 1e-12)

    return {
        'peak_depth_m': peak_depth,
        'error_m': abs(peak_depth - true_depth_m),
        'peak_val': peak_val,
        'snr': float(snr),
        'profile_depth': depth_kept,
        'profile': profile,
    }


def _plot_2d_panel(
    ax,
    t: np.ndarray,
    depth: np.ndarray,
    C: np.ndarray,
    title: str,
    fracture_depth: float,
    depth_min: float = 0.0,
    depth_max: float = 5000.0,
    vmin_pct: float = 2.0,
    vmax_pct: float = 98.0,
) -> None:
    mask = (depth >= depth_min) & (depth <= depth_max)
    depth_plot = depth[mask]
    C_plot = C[mask, :]

    T, D = np.meshgrid(t, depth_plot)
    data = -C_plot
    vmin = float(np.percentile(data, vmin_pct))
    vmax = float(np.percentile(data, vmax_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.pcolormesh(T, D, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    ax.axhline(fracture_depth, color='yellow', ls='--', lw=0.8, alpha=0.7)
    ax.invert_yaxis()
    ax.set_ylim([depth_max, depth_min])
    ax.set_xlabel('时间 [s]')
    ax.set_ylabel('深度 [m]')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, fraction=0.046, aspect=30)


def run_validation():
    print("=" * 72)
    print("测试 B — 2D 倒谱优化方案对比（Kaiser-Bessel / AR）")
    print("=" * 72)

    # ── 与 validate_moc_test_b.py 相同物理配置 ──
    L = 5000.0
    a = 1450.0
    V0 = 1.0
    H0 = 300.0
    ts = 1.0
    dt = 1.0e-3
    tf = 100.0
    x_f = 4300.0
    Cf = 1.0e-5
    kleak = 0.0001
    H_ext = 100.0
    fs = 1.0 / dt
    wlen_sec = 30.0
    hop_ratio = 0.2

    cfg = MocConfig(
        wellbore_length=L,
        wellbore_diameter=0.1397,
        fluid_density=1000.0,
        fluid_viscosity=1.0e-6,
        wavespeed=a,
        roughness_height=4.5e-5,
        friction_model='steady',
        dt=dt,
        tf=tf,
        wellhead_bc='velocity_step',
        pump_shut_time=ts,
        initial_velocity=V0,
        initial_head=H0,
        theta=0.0,
        toe_bc='reservoir',
        toe_head=H0,
    )

    print(f"\n运行 MOC 仿真 (x_f={x_f}m, steady, kleak={kleak})...")
    t0 = time_module.time()
    res = simulate_wellbore(
        cfg,
        fracture_positions=[x_f],
        fracture_Cf=[Cf],
        fracture_kleak=[kleak],
        H_ext=H_ext,
        store_full_field=False,
    )
    print(f"  耗时: {time_module.time() - t0:.1f}s")

    t_sim = res['timestamps']
    H_wh = res['wellhead_head']
    x_f_aligned = float(res['x_grid'][res['fracture_indices'][0]])
    v = cfg.a_adj

    pre = preprocess_moc_head(t_sim, H_wh, fs=fs, ts=ts)
    x_work = pre['h_detrended']

    print(f"  缝对齐深度: {x_f_aligned:.2f} m")
    print(f"  工作信号长度: {len(x_work)/fs:.1f}s, wlen={wlen_sec}s, hop_ratio={hop_ratio}")

    # ── 计算各方法 2D 倒谱 ──
    methods: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}

    t0, q0, C0 = baseline_2d_cepstrogram(
        x_work, fs, wlen_sec, hop_ratio, win_type='hamming',
        wavespeed=v, wellbore_length=L,
    )
    methods['基线 Hamming'] = (t0, q0, C0, f'Hamming, log|X|, beta=N/A')

    t1, q1, C1 = baseline_2d_cepstrogram(
        x_work, fs, wlen_sec, hop_ratio, win_type='kaiser', kaiser_beta=4.0,
        wavespeed=v, wellbore_length=L,
    )
    methods['基线 Kaiser beta=4'] = (t1, q1, C1, 'Kaiser beta=4, log|X|')

    t2, q2, C2 = optimized_2d_cepstrogram(
        x_work, fs, wlen_sec, wavespeed=v, wellbore_length=L, hop_ratio=hop_ratio,
        target_depth_m=x_f_aligned,
        use_dynamic_kaiser=True,
        use_epsilon_log=False,
        use_preemphasis=False,
        use_lifter=False,
    )
    methods['方案1 动态Kaiser'] = (t2, q2, C2, f'动态 beta={_dynamic_kaiser_beta(x_f_aligned, fs, v):.1f}')

    t3, q3, C3 = optimized_2d_cepstrogram(
        x_work, fs, wlen_sec, wavespeed=v, wellbore_length=L, hop_ratio=hop_ratio,
        target_depth_m=x_f_aligned,
        use_dynamic_kaiser=True,
        use_epsilon_log=True,
        use_preemphasis=True,
        use_lifter=False,
    )
    methods['方案1+2 eps+预加重'] = (t3, q3, C3, '动态Kaiser + ln(|X|^2+eps) + 预加重')

    t4, q4, C4 = optimized_2d_cepstrogram(
        x_work, fs, wlen_sec, wavespeed=v, wellbore_length=L, hop_ratio=hop_ratio,
        target_depth_m=x_f_aligned,
        use_dynamic_kaiser=True,
        use_epsilon_log=True,
        use_preemphasis=True,
        use_lifter=True,
    )
    methods['方案1+2+3 全复合'] = (t4, q4, C4, '动态Kaiser + eps + 预加重 + Lifter')

    t5, q5, C5 = ar_2d_cepstrogram(
        x_work, fs, wlen_sec, hop_ratio=hop_ratio, p_order=60,
        wavespeed=v, wellbore_length=L,
    )
    methods['方案4 AR倒谱'] = (t5, q5, C5, f'Yule-Walker AR p=60, n={len(q5)}')

    # ── 定量评估 ──
    print("\n" + "-" * 72)
    print(f"{'方法':<22} {'beta/备注':<28} {'峰深[m]':>8} {'误差[m]':>8} {'SNR':>8}")
    print("-" * 72)

    metrics = {}
    for name, (t_ax, q_ax, C, note) in methods.items():
        m = evaluate_fracture_peak(C, q_ax, v, x_f_aligned, depth_min=100.0, depth_max=L)
        metrics[name] = m
        print(
            f"{name:<22} {note:<28} "
            f"{m['peak_depth_m']:8.1f} {m['error_m']:8.1f} {m['snr']:8.2f}"
        )
    print("-" * 72)
    print(f"真实缝深: {x_f_aligned:.1f} m")

    # ── 可视化：2D 对比 2×3 ──
    out_dir = moc_output_dir()
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        f'测试 B — 2D 倒谱优化方案对比\n'
        f'x_f={x_f_aligned:.0f}m, v={v:.1f}m/s, wlen={wlen_sec}s, hop_ratio={hop_ratio}',
        fontsize=13, fontweight='bold',
    )

    for ax, (name, (t_ax, q_ax, C, note)) in zip(axes.flat, methods.items()):
        depth = _depth_axis(q_ax, v)
        _plot_2d_panel(
            ax, t_ax, depth, C, f'{name}\n{note}', x_f_aligned,
            depth_min=0.0, depth_max=L,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path_2d = os.path.join(out_dir, 'test_b_kaiser_bessel_2d_comparison.png')
    fig.savefig(path_2d, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n2D 对比图: {path_2d}")

    # ── 深度剖面对比 ──
    fig2, axp = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for (name, m), color in zip(metrics.items(), colors):
        axp.plot(m['profile_depth'], m['profile'], lw=1.2, label=name, color=color)

    axp.axvline(x_f_aligned, color='k', ls='--', lw=1.5, label=f'真实缝 {x_f_aligned:.0f}m')
    axp.set_xlabel('深度 [m]')
    axp.set_ylabel('时间平均倒谱响应 (-C)')
    axp.set_title('各方法深度剖面对比（时间维平均）')
    axp.set_xlim([0, L])
    axp.legend(fontsize=8, loc='upper right')
    axp.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    path_prof = os.path.join(out_dir, 'test_b_kaiser_bessel_depth_profile.png')
    fig2.savefig(path_prof, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"深度剖面图: {path_prof}")

    best = min(metrics.items(), key=lambda kv: kv[1]['error_m'])
    print(f"\n最优方法（深度误差最小）: {best[0]}  "
          f"峰={best[1]['peak_depth_m']:.1f}m, 误差={best[1]['error_m']:.1f}m, SNR={best[1]['snr']:.2f}")

    return {'metrics': metrics, 'x_f_aligned': x_f_aligned, 'figures': [path_2d, path_prof]}


if __name__ == '__main__':
    run_validation()
