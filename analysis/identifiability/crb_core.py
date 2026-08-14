# -*- coding: utf-8 -*-
"""
crb_core.py — 井筒-多裂缝反问题的 Fisher 信息矩阵与 Cramér-Rao 下界。

观测模型
--------
    y_k = s_k(theta) + n_k ,   n_k ~ N(0, sigma^2) i.i.d.

其中 s(theta) 为停泵后井口水头 H_wh 经传感器带宽限制后的采样序列，
theta 为待估参数向量。对高斯白噪声，

    FIM(theta) = J^T J / sigma^2 ,      J_{kj} = d s_k / d theta_j
    Cov(theta_hat) >= FIM^{-1}          （任意无偏估计器）

sigma 由 SNR 约定给出（与 moc_simulate/stratified_bench.apply_awgn 一致）：
    sigma^2 = Var(s_post_shutin) / 10^(SNR/10)

参数向量
--------
    x_i          裂缝位置 [m]           （关注量）
    log10 Cf_i   缝柔度                 （讨厌参数）
    log10 kl_i   滤失系数               （讨厌参数）
    a            波速 [m/s]             （讨厌参数，本项目的核心怀疑对象）

数值要点
--------
1. MOC 以 Courant=1 运行，裂缝位置被吸附到网格（dx = L/N ≈ 1.45 m）。
   因此对 x_i 的有限差分步长必须取网格步长的整数倍，否则差分恒为 0。
2. 改变波速会改变 N，从而改变 dx 与裂缝的吸附位置。本模块用整数 N 偏移来
   施加波速扰动，并用已算出的 ds/dx_i 扣除吸附抖动带来的伪贡献。
3. 传感器带宽 fc 是必需的正则化：仿真中的停泵是理想阶跃，带宽直达 Nyquist，
   不加带宽限制得到的 CRB 会乐观到没有物理意义。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore

__all__ = [
    "Scenario",
    "ObsModel",
    "ForwardResult",
    "forward",
    "observe",
    "noise_std_from_snr",
    "build_jobs",
    "run_jobs",
    "assemble_jacobian",
    "compute_jacobian",
    "fim_from_jacobian",
    "crb_report",
    "linear_functional_std",
]


# =====================================================================
# 场景与观测模型
# =====================================================================
@dataclass(frozen=True)
class Scenario:
    """一次 CRB 评估所对应的物理场景（真值/线性化点）。"""

    x_f: Tuple[float, ...]                 # 裂缝目标位置 [m]
    Cf: Tuple[float, ...]                  # 缝柔度 [m^2]
    kleak: Tuple[float, ...]               # 滤失系数 [m^2/s/sqrt(m)]
    L: float = 5000.0                      # 井筒长度 [m]
    diameter: float = 0.1397               # 内径 [m]
    a_nominal: float = 1450.0              # 名义波速 [m/s]
    dt: float = 1.0e-3                     # 时间步 [s]
    tf: float = 20.0                       # 总时长 [s]
    ts: float = 1.0                        # 停泵时刻 [s]
    V0: float = 1.0                        # 停泵前流速 [m/s]
    H0: float = 300.0                      # 初始水头 [m]
    H_ext: float = 100.0                   # 地层孔隙压力水头 [m]
    toe_bc: str = "reservoir"
    friction: str = "steady"               # steady / brunone
    k_scale: float = 1.0                   # Brunone 系数标定倍率（讨厌参数）
    viscosity: float = 1.0e-6
    roughness: float = 4.5e-5

    @property
    def n_frac(self) -> int:
        return len(self.x_f)

    def n_cells_nominal(self) -> int:
        return int(round(self.L / (self.a_nominal * self.dt)))

    def dx_for(self, n_cells: int) -> float:
        return self.L / n_cells

    def a_for(self, n_cells: int) -> float:
        return self.L / (n_cells * self.dt)


@dataclass(frozen=True)
class ObsModel:
    """观测（传感器）模型：带宽限制 + 白噪声。"""

    fc_hz: float = 200.0        # 传感器 -3dB 带宽 [Hz]；None/<=0 表示不限带
    order: int = 4              # Butterworth 幅频阶数（零相位）
    t_start: Optional[float] = None   # 观测起点 [s]，默认 = 停泵时刻
    t_end: Optional[float] = None     # 观测终点 [s]，默认 = tf


@dataclass
class ForwardResult:
    H_wh: np.ndarray
    t: np.ndarray
    placed_x: np.ndarray        # 吸附到网格后的实际裂缝位置 [m]
    a_adj: float                # MOC 实际使用的波速 [m/s]
    dx: float
    n_cells: int


# =====================================================================
# 正演与观测算子
# =====================================================================
def forward(
    scn: Scenario,
    *,
    n_cells: Optional[int] = None,
    x_f: Optional[Sequence[float]] = None,
    Cf: Optional[Sequence[float]] = None,
    kleak: Optional[Sequence[float]] = None,
    k_scale: Optional[float] = None,
) -> ForwardResult:
    """在指定网格数 n_cells 上运行一次 MOC 正演。

    以 n_cells 而非 wavespeed 作为波速自由度，可保证 MocConfig 内部的
    ``round(L/(a*dt))`` 精确落到期望网格数，避免舍入造成的不可控跳变。
    """
    n_cells = int(n_cells if n_cells is not None else scn.n_cells_nominal())
    x_f = np.asarray(scn.x_f if x_f is None else x_f, dtype=np.float64)
    Cf = np.asarray(scn.Cf if Cf is None else Cf, dtype=np.float64)
    kleak = np.asarray(scn.kleak if kleak is None else kleak, dtype=np.float64)

    wavespeed_arg = scn.L / (n_cells * scn.dt)
    cfg = MocConfig(
        wellbore_length=scn.L,
        wellbore_diameter=scn.diameter,
        fluid_viscosity=scn.viscosity,
        wavespeed=wavespeed_arg,
        roughness_height=scn.roughness,
        friction_model=scn.friction,
        brunone_k_scale=float(scn.k_scale if k_scale is None else k_scale),
        dt=scn.dt,
        tf=scn.tf,
        pump_shut_time=scn.ts,
        initial_velocity=scn.V0,
        initial_head=scn.H0,
        toe_bc=scn.toe_bc,
        toe_head=scn.H0,
    )
    if cfg.N != n_cells:  # 防御：舍入未落到期望格数
        raise RuntimeError(f"网格数不匹配：期望 {n_cells}，实得 {cfg.N}")

    res = simulate_wellbore(
        cfg,
        fracture_positions=list(x_f),
        fracture_Cf=list(Cf),
        fracture_kleak=list(kleak),
        H_ext=scn.H_ext,
        store_full_field=False,
    )
    placed = np.asarray(res["fracture_indices"], dtype=np.float64) * cfg.dx
    return ForwardResult(
        H_wh=np.asarray(res["wellhead_head"], dtype=np.float64),
        t=np.asarray(res["timestamps"], dtype=np.float64),
        placed_x=placed,
        a_adj=cfg.a_adj,
        dx=cfg.dx,
        n_cells=cfg.N,
    )


def _butterworth_lowpass(y: np.ndarray, dt: float, fc: float, order: int) -> np.ndarray:
    """零相位 Butterworth 幅频低通（频域实现，严格线性算子）。

    用奇对称延拓抑制 FFT 环绕不连续；延拓与裁剪同样是线性算子，
    因此对信号与 Jacobian 各列施加同一算子后 FIM 仍然精确。
    """
    n = y.size
    pad = min(n, max(64, int(round(4.0 / (fc * dt)))))
    left = 2.0 * y[0] - y[pad:0:-1]
    right = 2.0 * y[-1] - y[-2:-pad - 2:-1]
    ext = np.concatenate([left, y, right])
    m = ext.size
    freqs = np.fft.rfftfreq(m, d=dt)
    gain = 1.0 / np.sqrt(1.0 + (freqs / fc) ** (2 * order))
    filt = np.fft.irfft(np.fft.rfft(ext) * gain, n=m)
    return filt[left.size:left.size + n]


def observe(H_wh: np.ndarray, t: np.ndarray, scn: Scenario, obs: ObsModel) -> np.ndarray:
    """把井口水头时程映射为观测向量 s(theta)。"""
    t0 = scn.ts if obs.t_start is None else obs.t_start
    t1 = t[-1] if obs.t_end is None else obs.t_end
    mask = (t >= t0 - 1e-12) & (t <= t1 + 1e-12)
    seg = H_wh[mask]
    if obs.fc_hz is not None and obs.fc_hz > 0:
        nyq = 0.5 / scn.dt
        if obs.fc_hz < nyq:
            seg = _butterworth_lowpass(seg, scn.dt, obs.fc_hz, obs.order)
    return seg


def noise_std_from_snr(s: np.ndarray, snr_db: Optional[float]) -> float:
    """按项目约定由 SNR 反推白噪声标准差（信号功率 = 观测段去均值方差）。"""
    if snr_db is None or not np.isfinite(snr_db):
        raise ValueError("snr_db 必须为有限值才能定义噪声水平")
    p_sig = float(np.var(s - np.mean(s)))
    if p_sig <= 0.0:
        raise ValueError("观测段信号功率为零，无法定义 SNR")
    return math.sqrt(p_sig / (10.0 ** (float(snr_db) / 10.0)))


# =====================================================================
# Jacobian
# =====================================================================
def _param_names(n_frac: int, include: Sequence[str]) -> List[str]:
    names: List[str] = []
    if "x" in include:
        names += [f"x{i}" for i in range(n_frac)]
    if "cf" in include:
        names += [f"log_cf{i}" for i in range(n_frac)]
    if "kleak" in include:
        names += [f"log_kl{i}" for i in range(n_frac)]
    if "a" in include:
        names.append("a")
    if "kbr" in include:
        names.append("log_kbr")
    return names


def build_jobs(
    scn: Scenario,
    *,
    include: Sequence[str] = ("x", "cf", "kleak", "a"),
    grid_step: int = 1,
    log_step: float = 0.02,
    a_cell_step: int = 1,
) -> Tuple[Scenario, List[str], List[Dict[str, object]]]:
    """构造中心差分所需的全部正演任务。

    与 :func:`assemble_jacobian` 分离，使昂贵的正演只跑一次，而不同的观测
    模型（带宽 fc）与 SNR 可以在缓存波形上反复复用。

    参数
    ----
    grid_step  : 位置扰动 = grid_step * dx（正整数；dx = a*dt）
    log_step   : Cf / kleak 的 log10 扰动量
    a_cell_step: 波速扰动通过网格数 N -> N -/+ a_cell_step 实现，
                 |da/a| = a_cell_step / N。注意改变 a 近似等价于时间轴伸缩，
                 ds/da ~ -(t/a) ds/dt 随 t 线性增长，故必须保证
                 (a_cell_step/N) * tf * fc << 1，否则差商越出线性区。

    返回
    ----
    (对齐到网格后的 scenario, 参数名列表, 任务列表)
    """
    if grid_step < 1:
        raise ValueError("grid_step 必须 >= 1")
    names = _param_names(scn.n_frac, include)
    n_cells0 = scn.n_cells_nominal()
    dx0 = scn.dx_for(n_cells0)

    # 把目标位置对齐到格点，使 ±grid_step*dx 的扰动精确落到相邻格点
    idx0 = np.round(np.asarray(scn.x_f, dtype=np.float64) / dx0).astype(int)
    if len(set(idx0.tolist())) != len(idx0):
        raise ValueError(f"裂缝在网格上重合：idx={idx0.tolist()}，间距过小")
    x0 = idx0 * dx0
    scn = replace(scn, x_f=tuple(x0.tolist()))

    jobs: List[Dict[str, object]] = [{"kind": "base", "par": -1, "sgn": 0,
                                      "n_cells": n_cells0}]
    for j, name in enumerate(names):
        if name.startswith("x"):
            i = int(name[1:])
            for sgn in (+1, -1):
                xx = x0.copy()
                xx[i] = (idx0[i] + sgn * grid_step) * dx0
                jobs.append({"kind": "pert", "par": j, "sgn": sgn,
                             "n_cells": n_cells0, "x_f": xx})
        elif name.startswith("log_cf"):
            i = int(name[len("log_cf"):])
            for sgn in (+1, -1):
                cc = np.asarray(scn.Cf, dtype=np.float64).copy()
                cc[i] *= 10.0 ** (sgn * log_step)
                jobs.append({"kind": "pert", "par": j, "sgn": sgn,
                             "n_cells": n_cells0, "Cf": cc})
        elif name.startswith("log_kl"):
            i = int(name[len("log_kl"):])
            for sgn in (+1, -1):
                kk = np.asarray(scn.kleak, dtype=np.float64).copy()
                kk[i] *= 10.0 ** (sgn * log_step)
                jobs.append({"kind": "pert", "par": j, "sgn": sgn,
                             "n_cells": n_cells0, "kleak": kk})
        elif name == "a":
            for sgn in (+1, -1):   # N 减小 -> dx 增大 -> a 增大
                jobs.append({"kind": "pert", "par": j, "sgn": sgn,
                             "n_cells": n_cells0 - sgn * a_cell_step})
        elif name == "log_kbr":
            for sgn in (+1, -1):
                jobs.append({"kind": "pert", "par": j, "sgn": sgn,
                             "n_cells": n_cells0,
                             "k_scale": scn.k_scale * 10.0 ** (sgn * log_step)})
        else:
            raise ValueError(f"未知参数名 {name}")
    return scn, names, jobs


def _run_job(args) -> Dict[str, object]:
    """单个正演任务（模块级函数，可被 ProcessPoolExecutor 序列化）。"""
    scn, job = args
    res = forward(
        scn,
        n_cells=int(job["n_cells"]),
        x_f=job.get("x_f"),
        Cf=job.get("Cf"),
        kleak=job.get("kleak"),
        k_scale=job.get("k_scale"),
    )
    return {
        "H_wh": res.H_wh,
        "t": res.t,
        "a_adj": res.a_adj,
        "placed_x": res.placed_x,
    }


def run_jobs(scn: Scenario, jobs: Sequence[Dict[str, object]], map_fn=None) -> List[Dict]:
    """执行任务列表，返回原始波形（未施加观测算子）。"""
    runner = map_fn if map_fn is not None else map
    return list(runner(_run_job, [(scn, job) for job in jobs]))


def assemble_jacobian(
    scn: Scenario,
    names: Sequence[str],
    jobs: Sequence[Dict[str, object]],
    raws: Sequence[Dict[str, object]],
    obs: ObsModel,
    *,
    grid_step: int = 1,
    log_step: float = 0.02,
) -> Dict[str, object]:
    """在给定观测模型下由缓存波形组装 Jacobian。"""
    names = list(names)
    dx0 = scn.dx_for(scn.n_cells_nominal())

    def _obs(raw) -> np.ndarray:
        return observe(np.asarray(raw["H_wh"]), np.asarray(raw["t"]), scn, obs)

    base = raws[0]
    s0 = _obs(base)
    J = np.zeros((s0.size, len(names)), dtype=np.float64)

    plus: Dict[int, Dict] = {}
    minus: Dict[int, Dict] = {}
    for job, raw in zip(jobs[1:], raws[1:]):
        (plus if job["sgn"] > 0 else minus)[int(job["par"])] = raw

    a0 = float(base["a_adj"])
    diag: Dict[str, object] = {
        "a0": a0,
        "placed_x0": np.asarray(base["placed_x"]).tolist(),
        "dx0": dx0,
        "fc_hz": obs.fc_hz,
        "fc_times_dt": (obs.fc_hz * scn.dt) if obs.fc_hz else 0.0,
    }

    for j, name in enumerate(names):
        sp, sm = _obs(plus[j]), _obs(minus[j])
        if name.startswith("x"):
            J[:, j] = (sp - sm) / (2.0 * grid_step * dx0)
        elif name in ("log_kbr",) or name.startswith("log_cf") or name.startswith("log_kl"):
            J[:, j] = (sp - sm) / (2.0 * log_step)
        elif name == "a":
            da = float(plus[j]["a_adj"]) - float(minus[j]["a_adj"])
            J[:, j] = (sp - sm) / da
            diag["a_rel_step"] = abs(da / (2.0 * a0))
            diag["a_phase_number"] = (
                abs(da / (2.0 * a0)) * scn.tf * (obs.fc_hz or 0.0)
            )

    # 波速列的吸附抖动修正：改变 N 会改变 dx，裂缝重新吸附产生位置漂移，
    # 该漂移并非波速的物理效应，用已算出的 ds/dx_i 扣除。
    if "a" in names and any(n.startswith("x") for n in names):
        ja = names.index("a")
        da = float(plus[ja]["a_adj"]) - float(minus[ja]["a_adj"])
        dplaced = np.asarray(plus[ja]["placed_x"]) - np.asarray(minus[ja]["placed_x"])
        corr = np.zeros(s0.size)
        for i in range(scn.n_frac):
            if f"x{i}" in names:
                corr += J[:, names.index(f"x{i}")] * (dplaced[i] / da)
        raw_norm = float(np.linalg.norm(J[:, ja]))
        J[:, ja] -= corr
        diag["a_snap_correction_rel"] = (
            float(np.linalg.norm(corr) / raw_norm) if raw_norm > 0 else 0.0
        )

    return {"J": J, "names": names, "s0": s0, "scenario": scn, "obs": obs,
            "grid_step": grid_step, "log_step": log_step, "diagnostics": diag}


def compute_jacobian(
    scn: Scenario,
    obs: ObsModel,
    *,
    include: Sequence[str] = ("x", "cf", "kleak", "a"),
    grid_step: int = 1,
    log_step: float = 0.02,
    a_cell_step: int = 1,
    map_fn=None,
) -> Dict[str, object]:
    """便捷入口：构造任务、执行正演、组装 Jacobian 一步完成。"""
    scn2, names, jobs = build_jobs(
        scn, include=include, grid_step=grid_step,
        log_step=log_step, a_cell_step=a_cell_step,
    )
    raws = run_jobs(scn2, jobs, map_fn=map_fn)
    jac = assemble_jacobian(scn2, names, jobs, raws, obs,
                            grid_step=grid_step, log_step=log_step)
    jac["a_cell_step"] = a_cell_step
    return jac


# =====================================================================
# FIM / CRB
# =====================================================================
def fim_from_jacobian(J: np.ndarray, sigma: float) -> np.ndarray:
    return (J.T @ J) / (sigma ** 2)


def _safe_inverse(F: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """对角归一化后求逆，返回 (逆矩阵, 归一化 FIM 条件数, 归一化 FIM 特征值)。"""
    d = np.sqrt(np.diag(F))
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise np.linalg.LinAlgError("FIM 对角元非正，参数无信息")
    S = np.outer(d, d)
    Fn = F / S
    Fn = 0.5 * (Fn + Fn.T)
    evals = np.linalg.eigvalsh(Fn)
    cond = float(evals[-1] / evals[0]) if evals[0] > 0 else float("inf")
    Finv = np.linalg.inv(Fn) / S
    return Finv, cond, evals


def crb_report(
    jac: Dict[str, object],
    snr_db: float,
    *,
    nuisance: Sequence[str] = ("a",),
) -> Dict[str, object]:
    """由 Jacobian 与 SNR 计算 CRB 及其分解。

    返回三种口径的位置标准差下界：
      - ``std_full``   : 所有参数联合估计（波速未知）
      - ``std_no_nuis``: 讨厌参数已知（把对应行列从 FIM 中删除后求逆）
      - ``std_alone``  : 只有该参数未知，1/sqrt(F_ii)
    """
    J = np.asarray(jac["J"])
    names: List[str] = list(jac["names"])
    s0 = np.asarray(jac["s0"])
    sigma = noise_std_from_snr(s0, snr_db)

    F = fim_from_jacobian(J, sigma)
    Finv, cond, evals = _safe_inverse(F)
    std_full = np.sqrt(np.diag(Finv))
    std_alone = 1.0 / np.sqrt(np.diag(F))

    keep = [i for i, n in enumerate(names) if n not in set(nuisance)]
    Fsub = F[np.ix_(keep, keep)]
    Fsub_inv, cond_sub, _ = _safe_inverse(Fsub)
    std_sub = np.full(len(names), np.nan)
    std_sub[keep] = np.sqrt(np.diag(Fsub_inv))
    # 嵌回全尺寸矩阵，便于用同一套 c 向量计算线性泛函
    crb_sub_full = np.full_like(Finv, np.nan)
    crb_sub_full[np.ix_(keep, keep)] = Fsub_inv

    # 归一化 FIM 的最小特征方向 = 最难辨识的参数组合
    d = np.sqrt(np.diag(F))
    Fn = 0.5 * ((F / np.outer(d, d)) + (F / np.outer(d, d)).T)
    w, V = np.linalg.eigh(Fn)
    weakest = V[:, 0]

    corr = Finv / np.outer(std_full, std_full)

    return {
        "names": names,
        "sigma": sigma,
        "snr_db": float(snr_db),
        "n_obs": int(s0.size),
        "fim": F,
        "crb": Finv,
        "crb_no_nuisance": crb_sub_full,
        "nuisance": list(nuisance),
        "std_full": std_full,
        "std_no_nuisance": std_sub,
        "std_alone": std_alone,
        "corr": corr,
        "cond_normalized": cond,
        "cond_no_nuisance": cond_sub,
        "eigvals_normalized": evals,
        "weakest_direction": weakest,
        "weakest_eigval": float(w[0]),
    }


def linear_functional_std(
    report: Dict[str, object],
    c: np.ndarray,
    *,
    which: str = "full",
) -> float:
    """线性组合 c^T theta 的 CRB 标准差，例如缝间距 (x1-x0) 或缝群质心。

    which='full' 为讨厌参数未知；which='no_nuisance' 为讨厌参数已知。
    """
    key = {"full": "crb", "no_nuisance": "crb_no_nuisance"}[which]
    C = np.asarray(report[key], dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if np.any(~np.isfinite(C[np.ix_(c != 0, c != 0)])):
        return float("nan")
    C = np.nan_to_num(C, nan=0.0)
    return float(np.sqrt(max(c @ C @ c, 0.0)))
