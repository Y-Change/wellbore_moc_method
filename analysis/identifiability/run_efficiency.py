# -*- coding: utf-8 -*-
"""
run_efficiency.py — 全波形 MLE 效率检验 + 三方对照 + 波速联合估计代价。

对每个场景 × SNR：
  1. 在对齐物理配置上算 CRB（A 档 / B 档）
  2. 建位置网格正演缓存（一次）
  3. 30 次噪声实现：MLE-A、MLE-B、P0 剥离、倒谱
  4. 报告 RMSE / CRB 效率比；B 档另报绝对深度 vs 缝间距经验方差

用法
----
    # 冒烟（小半径、少试验）
    python -m analysis.identifiability.run_efficiency --tag smoke \\
        --cases single_4000 --radius 5 --n-mc 3 --snr 30,40 --workers 8

    # 正式（计划配置）
    python -m analysis.identifiability.run_efficiency --tag main \\
        --cases single_4000,dual_10m,dual_40m --radius 30 --n-mc 30 \\
        --snr inf,40,30,20 --fc 20 --workers 14
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import (  # noqa: E402
    crb_report,
    linear_functional_std,
    noise_std_from_snr,
)
from analysis.identifiability.mle import (  # noqa: E402
    ForwardCache,
    add_noise,
    build_position_cache,
    jacobian_at_truth,
    match_estimate_to_truth,
    mle_from_cache,
)
from analysis.identifiability.scenarios import (  # noqa: E402
    FC_DEFAULT,
    A_WAVE,
    case_by_name,
    default_cases,
    make_obs,
    make_scenario,
    scenario_summary,
)

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "efficiency")


def _parse_snr(s: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for part in s.split(","):
        part = part.strip().lower()
        if part in ("inf", "infty", "infinity", "none"):
            out.append(None)
        else:
            out.append(float(part))
    return out


def _snr_label(snr: Optional[float]) -> str:
    return "inf" if snr is None else f"{snr:g}"


def compute_crb_baselines(
    scn,
    obs,
    *,
    workers: int = 1,
) -> Dict[str, object]:
    """A/B 两档在真值处的 CRB，供效率比分母。"""
    map_fn = None
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)
        map_fn = pool.map
    try:
        jac_a = jacobian_at_truth(scn, obs, mode="A", map_fn=map_fn)
        jac_b = jacobian_at_truth(scn, obs, mode="B", map_fn=map_fn)
    finally:
        if pool is not None:
            pool.shutdown()

    names_a = list(jac_a["names"])
    names_b = list(jac_b["names"])
    n = scn.n_frac
    idx_x_a = [names_a.index(f"x{i}") for i in range(n)]
    idx_x_b = [names_b.index(f"x{i}") for i in range(n)]

    # 用一个参考 SNR 取相对结构；绝对 CRB 按 SNR 缩放
    # CRB ∝ 1/SNR_linear，即 σ_x(snr) = σ_x(ref) * 10^((ref-snr)/20)
    ref_snr = 40.0
    rep_a = crb_report(jac_a, ref_snr, nuisance=())
    rep_b = crb_report(jac_b, ref_snr, nuisance=("a",))

    c_sep_b = np.zeros(len(names_b))
    if n >= 2:
        c_sep_b[idx_x_b[0]] = -1.0
        c_sep_b[idx_x_b[-1]] = 1.0

    def scale_std(std_ref: float, snr: Optional[float]) -> float:
        if snr is None:
            return 0.0
        return float(std_ref) * (10.0 ** ((ref_snr - float(snr)) / 20.0))

    return {
        "jac_a": jac_a,
        "jac_b": jac_b,
        "ref_snr_db": ref_snr,
        "std_x_a_ref": [float(rep_a["std_full"][j]) for j in idx_x_a],
        "std_x_b_ref": [float(rep_b["std_full"][j]) for j in idx_x_b],
        "std_x_b_a_known_ref": [float(rep_b["std_no_nuisance"][j]) for j in idx_x_b],
        "std_spacing_b_ref": (
            linear_functional_std(rep_b, c_sep_b) if n >= 2 else float("nan")
        ),
        "std_spacing_b_a_known_ref": (
            linear_functional_std(rep_b, c_sep_b, which="no_nuisance")
            if n >= 2 else float("nan")
        ),
        "a_penalty_x0": float(
            rep_b["std_full"][idx_x_b[0]] / rep_b["std_no_nuisance"][idx_x_b[0]]
        ),
        "a_penalty_spacing": (
            float(
                linear_functional_std(rep_b, c_sep_b)
                / linear_functional_std(rep_b, c_sep_b, which="no_nuisance")
            )
            if n >= 2 else float("nan")
        ),
        "placed_x": list(jac_a["diagnostics"]["placed_x0"]),
        "scale_std": scale_std,  # 函数，不进 JSON
    }


def _load_p0_dictionary(variant: str = "nominal") -> Optional[Dict[str, np.ndarray]]:
    path = Path("output/analysis/method_migration/dictionary") / variant / "dictionary.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {
            "depth_grid": z["depth_grid"],
            "cf_grid": z["cf_grid"],
            "intact": z["intact"],
            "templates": z["templates"],
        }


def _reconstruct_full_H(
    cache: ForwardCache,
    y_obs: np.ndarray,
    snr_db: Optional[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """把观测段噪声映射回完整 H_wh，供 P0 / 倒谱使用。

    观测算子是停泵后截取 + 低通。这里用「真值全时程 + 同分布噪声」近似：
    对完整时程加噪声后再限带截取，与 y_obs 统计一致；P0/倒谱吃全时程。
    """
    # 重新正演真值全时程（已在 cache 构建时跑过，这里再跑一次代价可接受；
    # 也可从 cache 外另存。为简单起见用 forward。）
    from analysis.identifiability.crb_core import forward, observe

    res = forward(cache.scn, n_cells=cache.n_cells)
    H = res.H_wh.copy()
    if snr_db is None:
        return H
    # 用观测段功率定义 σ，再加到全时程
    s = observe(H, res.t, cache.scn, cache.obs)
    sigma = noise_std_from_snr(s, float(snr_db))
    H = H + rng.normal(0.0, sigma, size=H.shape)
    return H


def run_cepstrum_baseline(
    t: np.ndarray,
    H_wh: np.ndarray,
    x_true: Sequence[float],
    *,
    v: float = A_WAVE,
) -> Dict[str, object]:
    from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum_1d
    from analysis.unified_evaluation.detection_protocol import (
        DetectorConfig,
        detect_peaks,
        score_detections,
    )

    cep = compute_moc_cepstrum_1d(
        t, H_wh, v=v, ts=1.0, wellbore_length=5000.0,
        depth_min=3400.0,
    )
    depth = np.asarray(cep["depth"], dtype=np.float64)
    resp = np.asarray(cep["response"], dtype=np.float64)
    det_cfg = DetectorConfig(
        min_separation_m=5.0,
        height_pct=85.0,
        height_rel=0.03,
        max_peaks=max(10, len(x_true) + 2),
        search_min_m=3400.0,
        search_max_m=4900.0,
    )
    peaks = detect_peaks(depth, resp, det_cfg)
    # 取前 n_true 个峰做定位误差（缝数已知，与 MLE 对齐）
    n = len(x_true)
    pred = [p["depth_m"] for p in peaks[:n]]
    if len(pred) < n:
        pred = pred + [float("nan")] * (n - len(pred))
    matched = match_estimate_to_truth(pred, x_true)
    score = score_detections(
        [p["depth_m"] for p in peaks], list(x_true), tolerance_m=10.0,
    )
    return {
        "x_hat": matched.get("x_matched", pred),
        "errors_m": matched["errors_m"].tolist()
        if hasattr(matched["errors_m"], "tolist") else list(matched["errors_m"]),
        "rmse_m": matched["rmse_m"],
        "spacing_err_m": matched["spacing_err_m"],
        "f1": score["f1"],
        "n_pred": len(peaks),
    }


def run_p0_baseline(
    H_wh: np.ndarray,
    dic: Dict[str, np.ndarray],
    x_true: Sequence[float],
) -> Dict[str, object]:
    from analysis.method_migration.run_peeling import peel_signal
    from analysis.unified_evaluation.detection_protocol import score_detections

    det = peel_signal(
        H_wh, dic["intact"], dic["templates"],
        dic["depth_grid"], dic["cf_grid"],
    )
    n = len(x_true)
    pred_all = [d["depth_m"] for d in det]
    pred = pred_all[:n]
    if len(pred) < n:
        pred = pred + [float("nan")] * (n - len(pred))
    matched = match_estimate_to_truth(pred, x_true)
    score = score_detections(pred_all, list(x_true), tolerance_m=10.0)
    return {
        "x_hat": matched.get("x_matched", pred),
        "errors_m": matched["errors_m"].tolist()
        if hasattr(matched["errors_m"], "tolist") else list(matched["errors_m"]),
        "rmse_m": matched["rmse_m"],
        "spacing_err_m": matched["spacing_err_m"],
        "f1": score["f1"],
        "n_pred": len(pred_all),
    }


def _summarize_errors(errs: List[np.ndarray]) -> Dict[str, object]:
    """errs: list of (n_frac,) error vectors。"""
    if not errs:
        return {
            "rmse_per_frac_m": [],
            "bias_per_frac_m": [],
            "rmse_m": float("nan"),
            "mae_m": float("nan"),
            "n_mc": 0,
        }
    E = np.asarray(errs, dtype=np.float64)  # (n_mc, n_frac)
    with np.errstate(all="ignore"):
        rmse_per = np.sqrt(np.nanmean(E ** 2, axis=0))
        bias_per = np.nanmean(E, axis=0)
        rmse_all = float(np.sqrt(np.nanmean(E ** 2)))
        mae_all = float(np.nanmean(np.abs(E)))
    return {
        "rmse_per_frac_m": np.where(np.isfinite(rmse_per), rmse_per, np.nan).tolist(),
        "bias_per_frac_m": np.where(np.isfinite(bias_per), bias_per, np.nan).tolist(),
        "rmse_m": rmse_all if np.isfinite(rmse_all) else float("nan"),
        "mae_m": mae_all if np.isfinite(mae_all) else float("nan"),
        "n_mc": int(E.shape[0]),
    }


def run_case(
    case_name: str,
    *,
    snrs: Sequence[Optional[float]],
    fc: float,
    radius: int,
    n_mc: int,
    gn_steps: int,
    workers: int,
    seed: int,
    out_dir: Path,
    skip_baselines: bool = False,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs(fc)
    print(f"\n{'=' * 72}")
    print(f"场景 {case_name}: x={case.x_f}, spacing={case.spacing_m} m, "
          f"friction={scn.friction}, tf={scn.tf}, fc={fc} Hz")
    print(f"  网格半径 ±{radius} → 约 "
          f"{(2 * radius + 1) ** len(case.x_f)} 次正演（含间距约束后更少）")

    # ---- CRB ----
    t0 = time.time()
    print("  [1/4] 计算 CRB 基线…")
    crb = compute_crb_baselines(scn, obs, workers=workers)
    placed = np.asarray(crb["placed_x"], dtype=np.float64)
    print(f"       placed_x={placed.tolist()}  "
          f"a_penalty_x0=×{crb['a_penalty_x0']:.2f}  "
          f"a_penalty_spacing=×{crb['a_penalty_spacing']:.3f}  "
          f"({time.time() - t0:.1f}s)")

    # 用吸附后的场景做缓存（与 CRB 一致）
    scn_aligned = crb["jac_a"]["scenario"]

    # ---- 位置缓存 ----
    t0 = time.time()
    print("  [2/4] 构建位置正演缓存…")
    cache = build_position_cache(
        scn_aligned, obs, radius=radius, workers=workers,
    )
    print(f"       缓存 {len(cache.signals)} 个网格点  ({time.time() - t0:.1f}s)")

    # 保存真值全时程供基线用
    from analysis.identifiability.crb_core import forward
    truth_fwd = forward(scn_aligned, n_cells=cache.n_cells)
    t_full = truth_fwd.t
    H_truth = truth_fwd.H_wh

    dic = None if skip_baselines else _load_p0_dictionary("nominal")
    if not skip_baselines and dic is None:
        print("  [警告] P0 字典不存在，跳过剥离基线")

    # ---- 蒙特卡洛 ----
    print(f"  [3/4] 蒙特卡洛 n_mc={n_mc} × SNR={[_snr_label(s) for s in snrs]}…")
    rng_master = np.random.default_rng(seed)
    results_by_snr: Dict[str, object] = {}

    for snr in snrs:
        label = _snr_label(snr)
        print(f"    SNR={label} dB")
        # 为可复现，每个 SNR 用独立子流
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        crb_a = [crb["scale_std"](s, snr) for s in crb["std_x_a_ref"]]
        crb_b = [crb["scale_std"](s, snr) for s in crb["std_x_b_ref"]]
        crb_sp = crb["scale_std"](crb["std_spacing_b_ref"], snr) if case.spacing_m > 0 else float("nan")

        errs_a: List[np.ndarray] = []
        errs_b: List[np.ndarray] = []
        errs_cep: List[np.ndarray] = []
        errs_p0: List[np.ndarray] = []
        spacing_err_b: List[float] = []
        spacing_err_a: List[float] = []
        rows: List[Dict] = []

        for trial in range(n_mc):
            y, sigma = add_noise(cache.s_truth, snr, rng)

            # MLE-A：只估位置；用预计算 Jacobian 一步精修（无额外正演）
            mle_a = mle_from_cache(
                cache, y, mode="A", gn_steps=gn_steps, skip_gn=(gn_steps <= 0),
                jac_precomputed=crb["jac_a"],
            )
            ma = match_estimate_to_truth(mle_a.x_hat, placed)
            errs_a.append(np.asarray(ma["errors_m"], dtype=np.float64))
            spacing_err_a.append(ma["spacing_err_m"])

            # MLE-B：联合估波速等
            mle_b = mle_from_cache(
                cache, y, mode="B", gn_steps=gn_steps, skip_gn=(gn_steps <= 0),
                jac_precomputed=crb["jac_b"],
            )
            mb = match_estimate_to_truth(mle_b.x_hat, placed)
            errs_b.append(np.asarray(mb["errors_m"], dtype=np.float64))
            spacing_err_b.append(mb["spacing_err_m"])

            row = {
                "trial": trial,
                "sigma": sigma,
                "mle_a_x": mle_a.x_hat.tolist(),
                "mle_a_err": errs_a[-1].tolist(),
                "mle_a_rss": mle_a.rss,
                "mle_b_x": mle_b.x_hat.tolist(),
                "mle_b_a": mle_b.a_hat,
                "mle_b_err": errs_b[-1].tolist(),
                "mle_b_spacing_err": mb["spacing_err_m"],
                "mle_b_rss": mle_b.rss,
            }

            if not skip_baselines:
                # 全时程含噪信号
                H_noisy = H_truth.copy()
                if snr is not None:
                    H_noisy = H_noisy + rng.normal(0.0, sigma, size=H_noisy.shape)

                cep = run_cepstrum_baseline(t_full, H_noisy, placed, v=A_WAVE)
                errs_cep.append(np.asarray(cep["errors_m"], dtype=np.float64))
                row["cep_err"] = cep["errors_m"]
                row["cep_f1"] = cep["f1"]
                row["cep_rmse"] = cep["rmse_m"]

                if dic is not None:
                    # P0 字典长度需与 H 对齐
                    n_t = min(len(H_noisy), len(dic["intact"]))
                    p0 = run_p0_baseline(H_noisy[:n_t], {
                        "depth_grid": dic["depth_grid"],
                        "cf_grid": dic["cf_grid"],
                        "intact": dic["intact"][:n_t],
                        "templates": dic["templates"][:, :, :n_t],
                    }, placed)
                    errs_p0.append(np.asarray(p0["errors_m"], dtype=np.float64))
                    row["p0_err"] = p0["errors_m"]
                    row["p0_f1"] = p0["f1"]
                    row["p0_rmse"] = p0["rmse_m"]

            rows.append(row)
            if (trial + 1) % max(1, n_mc // 5) == 0 or trial == n_mc - 1:
                print(f"      trial {trial + 1}/{n_mc}")

        sum_a = _summarize_errors(errs_a)
        sum_b = _summarize_errors(errs_b)
        # 效率比：RMSE / CRB（用首缝 CRB；多缝用平均）
        crb_a_mean = float(np.mean(crb_a)) if crb_a else float("nan")
        crb_b_mean = float(np.mean(crb_b)) if crb_b else float("nan")
        eff_a = (
            sum_a["rmse_m"] / crb_a_mean
            if snr is not None and crb_a_mean > 0 else float("nan")
        )
        eff_b = (
            sum_b["rmse_m"] / crb_b_mean
            if snr is not None and crb_b_mean > 0 else float("nan")
        )

        sp_std_b = float(np.nanstd(spacing_err_b, ddof=1)) if case.spacing_m > 0 else float("nan")
        sp_rmse_b = float(np.sqrt(np.nanmean(np.square(spacing_err_b)))) if case.spacing_m > 0 else float("nan")
        sp_std_a = float(np.nanstd(spacing_err_a, ddof=1)) if case.spacing_m > 0 else float("nan")

        block: Dict[str, object] = {
            "snr_db": snr,
            "snr_label": label,
            "crb_x_a_m": crb_a,
            "crb_x_b_m": crb_b,
            "crb_spacing_b_m": crb_sp,
            "mle_a": {**sum_a, "efficiency_ratio": eff_a,
                      "spacing_std_m": sp_std_a},
            "mle_b": {
                **sum_b,
                "efficiency_ratio": eff_b,
                "spacing_std_m": sp_std_b,
                "spacing_rmse_m": sp_rmse_b,
                "spacing_efficiency": (
                    sp_rmse_b / crb_sp if snr is not None and crb_sp > 0 else float("nan")
                ),
                "depth_vs_spacing": {
                    "depth_rmse_m": sum_b["rmse_m"],
                    "spacing_rmse_m": sp_rmse_b,
                    "crb_depth_m": crb_b_mean,
                    "crb_spacing_m": crb_sp,
                    "empirical_depth_over_spacing": (
                        sum_b["rmse_m"] / sp_rmse_b if sp_rmse_b > 0 else float("nan")
                    ),
                },
            },
            "trials": rows,
        }
        if errs_cep:
            sum_cep = _summarize_errors(errs_cep)
            block["cepstrum"] = sum_cep
        if errs_p0:
            sum_p0 = _summarize_errors(errs_p0)
            block["p0"] = sum_p0

        print(f"      MLE-A RMSE={sum_a['rmse_m']:.4g} m  CRB={crb_a_mean:.4g}  "
              f"eff={eff_a:.3g}")
        print(f"      MLE-B RMSE={sum_b['rmse_m']:.4g} m  CRB={crb_b_mean:.4g}  "
              f"eff={eff_b:.3g}  spacing_RMSE={sp_rmse_b:.4g} m")
        if errs_cep:
            print(f"      倒谱  RMSE={block['cepstrum']['rmse_m']:.4g} m")
        if errs_p0:
            print(f"      P0    RMSE={block['p0']['rmse_m']:.4g} m")

        results_by_snr[label] = block

    # ---- 汇总写出 ----
    print("  [4/4] 写出结果…")
    # 去掉不可 JSON 的函数
    crb_public = {k: v for k, v in crb.items()
                  if k not in ("jac_a", "jac_b", "scale_std")}
    # Jacobian 诊断保留轻量信息
    crb_public["diagnostics_a"] = crb["jac_a"]["diagnostics"]
    crb_public["diagnostics_b"] = crb["jac_b"]["diagnostics"]
    crb_public["param_names_a"] = crb["jac_a"]["names"]
    crb_public["param_names_b"] = crb["jac_b"]["names"]

    payload = {
        "case": case_name,
        "scenario": scenario_summary(scn_aligned),
        "placed_x": placed.tolist(),
        "fc_hz": fc,
        "radius": radius,
        "n_mc": n_mc,
        "gn_steps": gn_steps,
        "n_cache": len(cache.signals),
        "crb": crb_public,
        "by_snr": results_by_snr,
        "seed": seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_name}.json"

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None if np.isnan(o) else ("Infinity" if o > 0 else "-Infinity")
        raise TypeError(type(o))

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=_default)
    print(f"  → {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--cases", default="single_4000,dual_10m,dual_40m")
    ap.add_argument("--snr", default="inf,40,30,20")
    ap.add_argument("--fc", type=float, default=FC_DEFAULT)
    ap.add_argument("--radius", type=int, default=30)
    ap.add_argument("--n-mc", type=int, default=30)
    ap.add_argument("--gn-steps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    snrs = _parse_snr(args.snr)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"效率检验  tag={args.tag}")
    print(f"  cases  = {cases}")
    print(f"  SNR    = {[_snr_label(s) for s in snrs]}")
    print(f"  fc     = {args.fc} Hz")
    print(f"  radius = ±{args.radius}")
    print(f"  n_mc   = {args.n_mc}")
    print(f"  out    = {out_dir}")
    print("=" * 72)

    meta = {
        "tag": args.tag,
        "cases": cases,
        "snr": [_snr_label(s) for s in snrs],
        "fc_hz": args.fc,
        "radius": args.radius,
        "n_mc": args.n_mc,
        "gn_steps": args.gn_steps,
        "workers": args.workers,
        "seed": args.seed,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "available_cases": [c.name for c in default_cases()],
    }
    t_all = time.time()
    summaries = []
    for name in cases:
        payload = run_case(
            name,
            snrs=snrs,
            fc=args.fc,
            radius=args.radius,
            n_mc=args.n_mc,
            gn_steps=args.gn_steps,
            workers=args.workers,
            seed=args.seed,
            out_dir=out_dir,
            skip_baselines=args.skip_baselines,
        )
        # 轻量摘要
        for lab, block in payload["by_snr"].items():
            summaries.append({
                "case": name,
                "snr": lab,
                "mle_a_rmse": block["mle_a"]["rmse_m"],
                "mle_a_eff": block["mle_a"]["efficiency_ratio"],
                "mle_b_rmse": block["mle_b"]["rmse_m"],
                "mle_b_eff": block["mle_b"]["efficiency_ratio"],
                "mle_b_spacing_rmse": block["mle_b"].get("spacing_rmse_m"),
                "cep_rmse": block.get("cepstrum", {}).get("rmse_m"),
                "p0_rmse": block.get("p0", {}).get("rmse_m"),
                "crb_a": float(np.mean(block["crb_x_a_m"])),
                "crb_b": float(np.mean(block["crb_x_b_m"])),
            })

    meta["elapsed_s"] = round(time.time() - t_all, 1)
    meta["summary"] = summaries
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"{'case':<14} {'SNR':>6} {'MLE-A':>10} {'effA':>7} "
          f"{'MLE-B':>10} {'effB':>7} {'P0':>10} {'cep':>10} {'CRB-A':>10}")
    for s in summaries:
        def fmt(v):
            return f"{v:.4g}" if v is not None and np.isfinite(v) else "—"
        print(f"{s['case']:<14} {s['snr']:>6} {fmt(s['mle_a_rmse']):>10} "
              f"{fmt(s['mle_a_eff']):>7} {fmt(s['mle_b_rmse']):>10} "
              f"{fmt(s['mle_b_eff']):>7} {fmt(s['p0_rmse']):>10} "
              f"{fmt(s['cep_rmse']):>10} {fmt(s['crb_a']):>10}")
    print(f"总耗时 {meta['elapsed_s']}s → {out_dir}")


if __name__ == "__main__":
    main()
