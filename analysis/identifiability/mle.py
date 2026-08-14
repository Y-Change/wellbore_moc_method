# -*- coding: utf-8 -*-
"""
mle.py — 全波形极大似然估计器（网格搜索 + Gauss-Newton 亚网格精修）。

观测模型与 CRB 一致：
    y = s(theta) + n,  n ~ N(0, sigma^2 I)
    theta_hat = argmin ||y - s(theta)||^2

两档估计器
----------
A 档：只估裂缝位置 x（Cf / kleak / a 已知）→ 对比 CRB std_no_nuisance
B 档：估 x + log10(Cf) + log10(kleak) + a   → 对比 CRB std_full

数值策略
--------
1. 位置在真值邻域做网格枚举（MOC 裂缝吸附到 dx 格点），用真实非线性正演，
   不是线性代理（否则贴 CRB 是循环论证）。
2. 在网格最优处用 assemble_jacobian 做一步 Gauss-Newton，得到亚网格连续估计。
3. 同一场景的全部正演缓存后，多噪声实现只做残差比较 + GN，几乎免费。

缓存键：位置索引元组（相对名义网格）+ 可选的 (log_cf, log_kl, n_cells) 扰动。
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.identifiability.crb_core import (
    ObsModel,
    Scenario,
    assemble_jacobian,
    build_jobs,
    compute_jacobian,
    forward,
    noise_std_from_snr,
    observe,
    run_jobs,
)

__all__ = [
    "ForwardCache",
    "MLEResult",
    "build_position_cache",
    "mle_from_cache",
    "mle_grid_coord_descent",
    "mle_refine_gn",
    "one_step_from_jacobian",
    "add_noise",
    "match_estimate_to_truth",
]


@dataclass
class MLEResult:
    x_hat: np.ndarray
    x_grid: np.ndarray          # 网格搜索最优（吸附位置）
    Cf_hat: np.ndarray
    kleak_hat: np.ndarray
    a_hat: float
    rss: float                  # 最终残差平方和
    rss_grid: float             # 网格搜索残差平方和
    mode: str                   # "A" | "B"
    n_forwards_cached: int
    gn_steps: int
    extras: Dict[str, object]


@dataclass
class ForwardCache:
    """位置网格上的观测波形缓存（Cf/kleak/a 固定在场景真值）。"""

    scn: Scenario               # 已对齐到网格的场景
    obs: ObsModel
    idx0: np.ndarray            # 真值位置的格点索引
    dx: float
    n_cells: int
    radius: int
    # keys: tuple of absolute cell indices (len = n_frac)
    signals: Dict[Tuple[int, ...], np.ndarray]
    t: np.ndarray
    s_truth: np.ndarray         # 真值处观测
    placed_truth: np.ndarray


def _snap_scenario(scn: Scenario) -> Tuple[Scenario, np.ndarray, float, int]:
    """把裂缝位置吸附到名义网格，返回 (对齐后场景, 索引, dx, n_cells)。"""
    n_cells = scn.n_cells_nominal()
    dx = scn.dx_for(n_cells)
    idx = np.round(np.asarray(scn.x_f, dtype=np.float64) / dx).astype(int)
    if len(set(idx.tolist())) != len(idx):
        raise ValueError(f"裂缝在网格上重合：idx={idx.tolist()}")
    x0 = idx * dx
    return replace(scn, x_f=tuple(x0.tolist())), idx, dx, n_cells


def _enumerate_index_combos(
    idx0: np.ndarray,
    radius: int,
    n_cells: int,
    *,
    min_sep_cells: int = 1,
) -> List[Tuple[int, ...]]:
    """真值邻域 ±radius 的全部位置组合（保持缝序、最小间距）。"""
    n = len(idx0)
    ranges = []
    for i in range(n):
        lo = max(1, int(idx0[i]) - radius)
        hi = min(n_cells - 1, int(idx0[i]) + radius)
        ranges.append(range(lo, hi + 1))
    combos: List[Tuple[int, ...]] = []
    for combo in itertools.product(*ranges):
        ok = True
        for a, b in zip(combo, combo[1:]):
            if b - a < min_sep_cells:
                ok = False
                break
        if ok:
            combos.append(tuple(int(c) for c in combo))
    return combos


def _forward_job(args) -> Tuple[Tuple[int, ...], np.ndarray, np.ndarray, float]:
    """单次正演：返回 (idx_tuple, H_wh, t, a_adj)。模块级以便进程池序列化。"""
    scn, idx_tuple, n_cells, dx = args
    x_f = np.asarray(idx_tuple, dtype=np.float64) * dx
    res = forward(scn, n_cells=n_cells, x_f=x_f)
    return idx_tuple, res.H_wh, res.t, res.a_adj


def build_position_cache(
    scn: Scenario,
    obs: ObsModel,
    *,
    radius: int = 30,
    workers: int = 1,
    min_sep_cells: int = 1,
    progress_every: int = 200,
) -> ForwardCache:
    """在真值邻域构建位置网格正演缓存。"""
    scn2, idx0, dx, n_cells = _snap_scenario(scn)
    combos = _enumerate_index_combos(
        idx0, radius, n_cells, min_sep_cells=min_sep_cells,
    )
    if not combos:
        raise RuntimeError("位置网格为空：检查 radius / 真值间距")

    jobs = [(scn2, c, n_cells, dx) for c in combos]
    signals: Dict[Tuple[int, ...], np.ndarray] = {}
    t_ref: Optional[np.ndarray] = None

    def _consume(results):
        nonlocal t_ref
        for k, (idx_tuple, H_wh, t, _a) in enumerate(results):
            if t_ref is None:
                t_ref = np.asarray(t, dtype=np.float64)
            s = observe(np.asarray(H_wh, dtype=np.float64), t_ref, scn2, obs)
            signals[idx_tuple] = s
            if progress_every and (k + 1) % progress_every == 0:
                print(f"      缓存进度 {k + 1}/{len(jobs)}")

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            _consume(ex.map(_forward_job, jobs, chunksize=max(1, len(jobs) // (workers * 4))))
    else:
        _consume(map(_forward_job, jobs))

    assert t_ref is not None
    truth_key = tuple(int(i) for i in idx0)
    if truth_key not in signals:
        # 真值恰在边界外时补一次
        _, H, t, _ = _forward_job((scn2, truth_key, n_cells, dx))
        t_ref = np.asarray(t, dtype=np.float64)
        signals[truth_key] = observe(H, t_ref, scn2, obs)

    placed = idx0.astype(np.float64) * dx
    return ForwardCache(
        scn=scn2,
        obs=obs,
        idx0=idx0,
        dx=dx,
        n_cells=n_cells,
        radius=radius,
        signals=signals,
        t=t_ref,
        s_truth=signals[truth_key],
        placed_truth=placed,
    )


def add_noise(
    s: np.ndarray,
    snr_db: Optional[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """按项目 SNR 约定加性高斯白噪声。snr_db=None 表示无噪声（σ=0）。"""
    if snr_db is None or (isinstance(snr_db, float) and not np.isfinite(snr_db)):
        return s.copy(), 0.0
    sigma = noise_std_from_snr(s, float(snr_db))
    return s + rng.normal(0.0, sigma, size=s.shape), sigma


def _rss(y: np.ndarray, s: np.ndarray) -> float:
    d = y - s
    return float(d @ d)


def _grid_search(cache: ForwardCache, y: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    best_key: Optional[Tuple[int, ...]] = None
    best_rss = float("inf")
    for key, s in cache.signals.items():
        r = _rss(y, s)
        if r < best_rss:
            best_rss = r
            best_key = key
    assert best_key is not None
    return best_key, best_rss


def mle_grid_coord_descent(
    scn: Scenario,
    obs: ObsModel,
    y: np.ndarray,
    *,
    radius: int = 15,
    n_rounds: int = 3,
    min_sep_cells: int = 1,
    x_init: Optional[Sequence[float]] = None,
    signal_cache: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
    workers: int = 1,
) -> Dict[str, object]:
    """循环坐标下降：逐缝扫描位置网格，其余固定。

    复杂度约 ``n_rounds * n_frac * (2*radius+1)`` 次正演（可跨调用复用
    ``signal_cache``），避免 n≥3 时穷举乘积网格。

    EXP-002 似然面真值邻域单峰是可用性前提；与穷举对照的验收见
    ``run_coord_validate``。
    """
    scn2, idx0, dx, n_cells = _snap_scenario(scn)
    if x_init is None:
        idx = idx0.copy()
    else:
        idx = np.round(np.asarray(x_init, dtype=np.float64) / dx).astype(int)
        idx = np.clip(idx, 1, n_cells - 1)

    cache = signal_cache if signal_cache is not None else {}
    t_ref: Optional[np.ndarray] = None
    n_forward = 0
    history: List[Dict[str, object]] = []

    def _get_signal(key: Tuple[int, ...]) -> np.ndarray:
        nonlocal t_ref, n_forward
        if key in cache:
            return cache[key]
        _, H_wh, t, _ = _forward_job((scn2, key, n_cells, dx))
        if t_ref is None:
            t_ref = np.asarray(t, dtype=np.float64)
        s = observe(np.asarray(H_wh, dtype=np.float64), t_ref, scn2, obs)
        # 对齐观测长度到 y
        m = min(len(y), len(s))
        s = s[:m]
        cache[key] = s
        n_forward += 1
        return s

    def _rss_key(key: Tuple[int, ...]) -> float:
        s = _get_signal(key)
        m = min(len(y), len(s))
        return _rss(y[:m], s[:m])

    # 预取当前点
    cur = tuple(int(v) for v in idx.tolist())
    best_rss = _rss_key(cur)

    for rnd in range(n_rounds):
        improved = False
        for i in range(len(idx)):
            lo = max(1, int(idx[i]) - radius)
            hi = min(n_cells - 1, int(idx[i]) + radius)
            # 邻居约束：保持顺序与最小间距
            if i > 0:
                lo = max(lo, int(idx[i - 1]) + min_sep_cells)
            if i < len(idx) - 1:
                hi = min(hi, int(idx[i + 1]) - min_sep_cells)
            if lo > hi:
                continue

            cand_keys: List[Tuple[int, ...]] = []
            for j in range(lo, hi + 1):
                trial = list(idx)
                trial[i] = j
                cand_keys.append(tuple(int(v) for v in trial))

            # 可并行预计算缺失正演
            missing = [k for k in cand_keys if k not in cache]
            if missing and workers > 1 and len(missing) > 1:
                jobs = [(scn2, k, n_cells, dx) for k in missing]
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    for key, H_wh, t, _a in ex.map(
                        _forward_job, jobs, chunksize=max(1, len(jobs) // (workers * 4))
                    ):
                        if t_ref is None:
                            t_ref = np.asarray(t, dtype=np.float64)
                        s = observe(np.asarray(H_wh, dtype=np.float64), t_ref, scn2, obs)
                        m = min(len(y), len(s))
                        cache[key] = s[:m]
                        n_forward += 1

            local_best = cur
            local_rss = best_rss
            for key in cand_keys:
                r = _rss_key(key)
                if r < local_rss - 1e-18:
                    local_rss = r
                    local_best = key

            if local_best != cur:
                idx = np.asarray(local_best, dtype=int)
                cur = local_best
                best_rss = local_rss
                improved = True

        history.append({
            "round": rnd + 1,
            "idx": list(cur),
            "x": (np.asarray(cur, dtype=np.float64) * dx).tolist(),
            "rss": best_rss,
            "improved": improved,
        })
        if not improved:
            break

    x_hat = np.asarray(cur, dtype=np.float64) * dx
    return {
        "x_hat": x_hat,
        "idx_hat": cur,
        "rss": best_rss,
        "n_forwards": n_forward,
        "n_cache": len(cache),
        "history": history,
        "signal_cache": cache,
        "dx": dx,
        "scn": scn2,
    }


def mle_refine_gn(
    scn: Scenario,
    obs: ObsModel,
    y: np.ndarray,
    *,
    x_init: Sequence[float],
    mode: str = "A",
    n_steps: int = 1,
    include_override: Optional[Sequence[str]] = None,
    map_fn=None,
) -> Dict[str, object]:
    """在 x_init 附近做 Gauss-Newton 精修。

    mode='A'：只更新位置（include=x）
    mode='B'：更新 x + cf + kleak + a
    """
    if include_override is not None:
        include = list(include_override)
    elif mode.upper() == "A":
        include = ["x"]
    else:
        include = ["x", "cf", "kleak", "a"]

    scn_cur = replace(scn, x_f=tuple(float(v) for v in x_init))
    history: List[Dict[str, object]] = []

    for step in range(n_steps):
        jac = compute_jacobian(scn_cur, obs, include=include, map_fn=map_fn)
        names = list(jac["names"])
        J = np.asarray(jac["J"], dtype=np.float64)
        s0 = np.asarray(jac["s0"], dtype=np.float64)
        # 场景已被 build_jobs 吸附
        scn_cur = jac["scenario"]
        m = min(len(y), len(s0))
        residual = y[:m] - s0[:m]
        J = J[:m, :]
        # 列归一化最小二乘，避免 m 与 log10 混用病态
        scale = np.linalg.norm(J, axis=0)
        scale[scale <= 0] = 1.0
        Js = J / scale
        delta_s, *_ = np.linalg.lstsq(Js, residual, rcond=None)
        delta = delta_s / scale

        # 应用更新
        x_new = np.asarray(scn_cur.x_f, dtype=np.float64).copy()
        Cf_new = np.asarray(scn_cur.Cf, dtype=np.float64).copy()
        kl_new = np.asarray(scn_cur.kleak, dtype=np.float64).copy()
        n_cells = scn_cur.n_cells_nominal()
        a_new = scn_cur.a_for(n_cells)
        k_scale_new = float(scn_cur.k_scale)

        for j, name in enumerate(names):
            d = float(delta[j])
            if name.startswith("x"):
                i = int(name[1:])
                x_new[i] += d
            elif name.startswith("log_cf"):
                i = int(name[len("log_cf"):])
                Cf_new[i] *= 10.0 ** d
            elif name.startswith("log_kl"):
                i = int(name[len("log_kl"):])
                kl_new[i] *= 10.0 ** d
            elif name == "a":
                a_new = a_new + d
                # 映射回最近的 n_cells（保持 Courant=1）
                n_cells = max(10, int(round(scn_cur.L / (a_new * scn_cur.dt))))
                a_new = scn_cur.a_for(n_cells)
            elif name == "log_kbr":
                k_scale_new *= 10.0 ** d

        # 位置保持顺序与井筒内
        order = np.argsort(x_new)
        x_new = x_new[order]
        Cf_new = Cf_new[order]
        kl_new = kl_new[order]
        x_new = np.clip(x_new, scn_cur.dx_for(n_cells), scn_cur.L - scn_cur.dx_for(n_cells))

        scn_cur = replace(
            scn_cur,
            x_f=tuple(x_new.tolist()),
            Cf=tuple(Cf_new.tolist()),
            kleak=tuple(kl_new.tolist()),
            a_nominal=float(a_new),
            k_scale=float(k_scale_new),
        )
        # 用更新后的参数重跑正演，得到新 RSS
        res = forward(scn_cur, n_cells=n_cells)
        s_new = observe(res.H_wh, res.t, scn_cur, obs)
        # 对齐长度（波速变化时采样数可能变）
        m = min(len(y), len(s_new))
        rss = _rss(y[:m], s_new[:m])
        history.append({
            "step": step + 1,
            "x": x_new.tolist(),
            "Cf": Cf_new.tolist(),
            "kleak": kl_new.tolist(),
            "a": float(a_new),
            "k_scale": float(k_scale_new),
            "rss": rss,
            "delta": {nm: float(delta[j]) for j, nm in enumerate(names)},
        })

    last = history[-1] if history else {
        "x": list(x_init), "Cf": list(scn.Cf), "kleak": list(scn.kleak),
        "a": scn.a_nominal, "k_scale": scn.k_scale, "rss": float("nan"),
    }
    return {
        "x_hat": np.asarray(last["x"], dtype=np.float64),
        "Cf_hat": np.asarray(last["Cf"], dtype=np.float64),
        "kleak_hat": np.asarray(last["kleak"], dtype=np.float64),
        "a_hat": float(last["a"]),
        "k_scale_hat": float(last.get("k_scale", scn.k_scale)),
        "rss": float(last["rss"]),
        "history": history,
        "n_steps": len(history),
    }


def one_step_from_jacobian(
    jac: Dict[str, object],
    y: np.ndarray,
    *,
    s_ref: Optional[np.ndarray] = None,
    x_ref: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """用预计算 Jacobian 做一步 Gauss-Newton（无额外正演）。

    典型用法：jac 在真值处算一次；每个噪声实现用
    ``s_ref = s_grid``（来自缓存）+ ``x_ref = x_grid``，
    得到亚网格连续估计。高 SNR 下网格命中真值时等价于
    经典一步有效估计量，渐近达到 CRB。
    """
    names = list(jac["names"])
    J = np.asarray(jac["J"], dtype=np.float64)
    s0 = np.asarray(jac["s0"] if s_ref is None else s_ref, dtype=np.float64)
    scn = jac["scenario"]
    m = min(len(y), len(s0), J.shape[0])
    residual = y[:m] - s0[:m]
    J = J[:m]

    scale = np.linalg.norm(J, axis=0)
    scale[scale <= 0] = 1.0
    delta_s, *_ = np.linalg.lstsq(J / scale, residual, rcond=None)
    delta = delta_s / scale

    x_new = np.asarray(
        scn.x_f if x_ref is None else x_ref, dtype=np.float64
    ).copy()
    Cf_new = np.asarray(scn.Cf, dtype=np.float64).copy()
    kl_new = np.asarray(scn.kleak, dtype=np.float64).copy()
    n_cells = scn.n_cells_nominal()
    a_new = scn.a_for(n_cells)
    k_scale_new = float(scn.k_scale)

    for j, name in enumerate(names):
        d = float(delta[j])
        if name.startswith("x"):
            i = int(name[1:])
            if i < len(x_new):
                x_new[i] += d
        elif name.startswith("log_cf"):
            i = int(name[len("log_cf"):])
            if i < len(Cf_new):
                Cf_new[i] *= 10.0 ** d
        elif name.startswith("log_kl"):
            i = int(name[len("log_kl"):])
            if i < len(kl_new):
                kl_new[i] *= 10.0 ** d
        elif name == "a":
            a_new = a_new + d
        elif name == "log_kbr":
            k_scale_new *= 10.0 ** d

    order = np.argsort(x_new)
    x_new = x_new[order]
    Cf_new = Cf_new[order]
    kl_new = kl_new[order]
    return {
        "x_hat": x_new,
        "Cf_hat": Cf_new,
        "kleak_hat": kl_new,
        "a_hat": float(a_new),
        "k_scale_hat": float(k_scale_new),
        "delta": {nm: float(delta[j]) for j, nm in enumerate(names)},
        "rss_linear": float(residual @ residual),
    }


def mle_from_cache(
    cache: ForwardCache,
    y: np.ndarray,
    *,
    mode: str = "A",
    gn_steps: int = 1,
    skip_gn: bool = False,
    map_fn=None,
    jac_precomputed: Optional[Dict[str, object]] = None,
) -> MLEResult:
    """在已缓存的位置网格上做 MLE，可选一步/多步 Gauss-Newton 精修。

    若提供 ``jac_precomputed``（通常在真值处算一次），则用
    :func:`one_step_from_jacobian` 做无额外正演的一步精修——
    蒙特卡洛阶段几乎免费。否则回退到每次重算 Jacobian 的
    :func:`mle_refine_gn`（昂贵，仅用于失配等小样本诊断）。
    """
    best_key, rss_grid = _grid_search(cache, y)
    x_grid = np.asarray(best_key, dtype=np.float64) * cache.dx
    s_grid = cache.signals[best_key]

    if skip_gn or gn_steps <= 0:
        return MLEResult(
            x_hat=x_grid.copy(),
            x_grid=x_grid,
            Cf_hat=np.asarray(cache.scn.Cf, dtype=np.float64),
            kleak_hat=np.asarray(cache.scn.kleak, dtype=np.float64),
            a_hat=cache.scn.a_for(cache.n_cells),
            rss=rss_grid,
            rss_grid=rss_grid,
            mode=mode.upper(),
            n_forwards_cached=len(cache.signals),
            gn_steps=0,
            extras={"best_key": best_key},
        )

    if jac_precomputed is not None:
        step = one_step_from_jacobian(
            jac_precomputed, y, s_ref=s_grid, x_ref=x_grid,
        )
        # 线性化残差作为 RSS 代理；高 SNR 下与真 RSS 接近
        return MLEResult(
            x_hat=step["x_hat"],
            x_grid=x_grid,
            Cf_hat=step["Cf_hat"],
            kleak_hat=step["kleak_hat"],
            a_hat=step["a_hat"],
            rss=float(step["rss_linear"]),
            rss_grid=rss_grid,
            mode=mode.upper(),
            n_forwards_cached=len(cache.signals),
            gn_steps=1,
            extras={"best_key": best_key, "one_step_delta": step["delta"],
                    "k_scale_hat": step["k_scale_hat"]},
        )

    refined = mle_refine_gn(
        cache.scn, cache.obs, y,
        x_init=x_grid, mode=mode, n_steps=gn_steps, map_fn=map_fn,
    )
    return MLEResult(
        x_hat=refined["x_hat"],
        x_grid=x_grid,
        Cf_hat=refined["Cf_hat"],
        kleak_hat=refined["kleak_hat"],
        a_hat=refined["a_hat"],
        rss=refined["rss"],
        rss_grid=rss_grid,
        mode=mode.upper(),
        n_forwards_cached=len(cache.signals),
        gn_steps=refined["n_steps"],
        extras={"best_key": best_key, "gn_history": refined["history"]},
    )


def match_estimate_to_truth(
    x_hat: Sequence[float],
    x_true: Sequence[float],
) -> Dict[str, object]:
    """把估计位置与真值做最优一一匹配（匈牙利 / 对小 n 穷举），返回误差。"""
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    n = len(x_true)
    if len(x_hat) != n:
        # 缝数固定时不应发生；兜底按长度截断/填充
        m = min(len(x_hat), n)
        err = np.full(n, np.nan)
        err[:m] = np.sort(x_hat)[:m] - np.sort(x_true)[:m]
        return {
            "errors_m": err,
            "rmse_m": float(np.sqrt(np.nanmean(err ** 2))),
            "bias_m": err,
            "spacing_err_m": float("nan"),
            "perm": list(range(n)),
        }

    # n 很小，穷举排列
    best_perm = tuple(range(n))
    best_cost = float("inf")
    for perm in itertools.permutations(range(n)):
        cost = float(np.sum((x_hat[list(perm)] - x_true) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    matched = x_hat[list(best_perm)]
    err = matched - x_true
    spacing_err = float("nan")
    if n >= 2:
        spacing_err = float((matched[-1] - matched[0]) - (x_true[-1] - x_true[0]))
    return {
        "errors_m": err,
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "bias_m": err,
        "spacing_err_m": spacing_err,
        "x_matched": matched,
        "perm": list(best_perm),
    }


def jacobian_at_truth(
    scn: Scenario,
    obs: ObsModel,
    *,
    mode: str = "A",
    map_fn=None,
) -> Dict[str, object]:
    """在真值处组装 Jacobian，供 CRB 对照。"""
    include = ["x"] if mode.upper() == "A" else ["x", "cf", "kleak", "a"]
    return compute_jacobian(scn, obs, include=include, map_fn=map_fn)
