# -*- coding: utf-8 -*-
"""
run_order_selection.py — 缝数未知：嵌套模型 BIC 选择（P2c）。

对每个候选阶数 n̂，在真值邻域构造嵌套构型并单次正演得 RSS，
用两种有效样本量（带宽自由度 / 其半值）算 BIC；与 PhaseNet exact-count
30.5% 对照。

嵌套构型（避免 n_max×坐标下降的组合爆炸）：
  n̂ < n_true : 取真值中能量最强的 n̂ 条（以 |log Cf| 代理；此处 Cf 相同则取居中子集）
  n̂ = n_true : 真值
  n̂ > n_true : 在真值缝间插入点，直至 n̂

有效样本量：时间点高度相关，禁止直接用 N_t；主报告用 n_bw=2·fc·T。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from analysis.identifiability.crb_core import forward, observe  # noqa: E402
from analysis.identifiability.mle import add_noise  # noqa: E402
from analysis.identifiability.scenarios import (  # noqa: E402
    CF_DEFAULT,
    KLEAK_DEFAULT,
    case_by_name,
    make_obs,
    make_scenario,
)

DEFAULT_OUT = os.path.join("output", "analysis", "identifiability", "order_selection")
PHASENET_EXACT_COUNT = 0.305


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


def effective_sample_sizes(n_t: int, dt: float, fc_hz: float) -> Dict[str, float]:
    T = float(n_t) * dt
    n_bw = max(10.0, 2.0 * fc_hz * T)
    return {
        "n_raw": float(n_t),
        "n_bw": n_bw,
        "n_half": max(10.0, 0.5 * n_bw),
    }


def bic_score(rss: float, n_eff: float, n_params: int) -> float:
    rss = max(rss, 1e-30)
    n_eff = max(n_eff, 2.0)
    return float(n_eff * np.log(rss / n_eff) + n_params * np.log(n_eff))


def nested_positions(x_true: Sequence[float], n_hat: int) -> Tuple[float, ...]:
    x = list(np.sort(np.asarray(x_true, dtype=float)))
    n = len(x)
    if n_hat == n:
        return tuple(x)
    if n_hat < n:
        # 居中子集
        start = (n - n_hat) // 2
        return tuple(x[start:start + n_hat])
    # 插入中点直至达到 n_hat
    cur = x[:]
    while len(cur) < n_hat:
        # 在最大间距处插入
        gaps = np.diff(cur)
        i = int(np.argmax(gaps))
        mid = 0.5 * (cur[i] + cur[i + 1])
        cur.insert(i + 1, float(mid))
    # 若仍不足（单缝），向外扩展
    while len(cur) < n_hat:
        cur.append(cur[-1] + 10.0)
    return tuple(cur[:n_hat])


def rss_for_positions(scn, obs, y: np.ndarray, x_f: Sequence[float]) -> Dict[str, object]:
    scn_c = replace(
        scn,
        x_f=tuple(float(v) for v in x_f),
        Cf=tuple(CF_DEFAULT for _ in x_f),
        kleak=tuple(KLEAK_DEFAULT for _ in x_f),
    )
    res = forward(scn_c)
    s = observe(res.H_wh, res.t, scn_c, obs)
    m = min(len(y), len(s))
    d = y[:m] - s[:m]
    return {
        "x_f": list(x_f),
        "rss": float(d @ d),
        "n_params": len(x_f),
        "s": s,
    }


def select_order(scn, obs, y: np.ndarray, *, n_max: int) -> Dict[str, object]:
    neff = effective_sample_sizes(len(y), scn.dt, obs.fc_hz)
    fits = []
    for n_hat in range(1, n_max + 1):
        xs = nested_positions(scn.x_f, n_hat)
        fit = rss_for_positions(scn, obs, y, xs)
        fit["n_hat"] = n_hat
        for key, n_eff in neff.items():
            fit[f"bic_{key}"] = bic_score(fit["rss"], n_eff, fit["n_params"])
        fits.append(fit)

    choices = {}
    for key in ("n_bw", "n_half", "n_raw"):
        best = min(fits, key=lambda f: f[f"bic_{key}"])
        choices[key] = {
            "n_hat": best["n_hat"],
            "bic": best[f"bic_{key}"],
            "rss": best["rss"],
        }
    return {
        "n_true": scn.n_frac,
        "fits": fits,
        "choices": choices,
        "n_eff": neff,
    }


def run_mc(
    case_name: str,
    *,
    snrs: List[Optional[float]],
    n_mc: int,
    n_max: int,
    seed: int,
    workers: int,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs()
    res = forward(scn)
    s0 = observe(res.H_wh, res.t, scn, obs)

    print(f"\n===== order selection: {case_name} n_true={scn.n_frac} =====")
    by_snr = {}
    rng = np.random.default_rng(seed)

    # 预计算各 n_hat 的干净信号，噪声实现只改 y — 但嵌套位置固定，
    # 干净信号可缓存；噪声下 RSS = ||(s0+n) - s_hat||^2
    cache_s: Dict[int, np.ndarray] = {}
    for n_hat in range(1, n_max + 1):
        xs = nested_positions(scn.x_f, n_hat)
        fit = rss_for_positions(scn, obs, s0, xs)
        cache_s[n_hat] = fit.pop("s")
        print(f"  cached n_hat={n_hat} rss_clean={fit['rss']:.4e}")

    for snr in snrs:
        label = _snr_label(snr)
        records = []
        for i in range(n_mc):
            y, _ = add_noise(s0, snr, rng)
            neff = effective_sample_sizes(len(y), scn.dt, obs.fc_hz)
            fits = []
            for n_hat in range(1, n_max + 1):
                s = cache_s[n_hat]
                m = min(len(y), len(s))
                d = y[:m] - s[:m]
                rss = float(d @ d)
                fit = {"n_hat": n_hat, "rss": rss, "n_params": n_hat,
                       "x_f": list(nested_positions(scn.x_f, n_hat))}
                for key, n_eff in neff.items():
                    fit[f"bic_{key}"] = bic_score(rss, n_eff, n_hat)
                fits.append(fit)
            choices = {}
            for key in ("n_bw", "n_half", "n_raw"):
                best = min(fits, key=lambda f: f[f"bic_{key}"])
                choices[key] = {
                    "n_hat": best["n_hat"],
                    "bic": best[f"bic_{key}"],
                    "rss": best["rss"],
                }
            records.append({"trial": i, "choices": choices, "fits": fits, "n_eff": neff})

        def exact_rate(key: str) -> float:
            return float(np.mean([
                r["choices"][key]["n_hat"] == scn.n_frac for r in records
            ]))

        by_snr[label] = {
            "exact_count_n_bw": exact_rate("n_bw"),
            "exact_count_n_half": exact_rate("n_half"),
            "exact_count_n_raw": exact_rate("n_raw"),
            "n_mc": n_mc,
            "records": records,
        }
        print(
            f"  SNR={label}: exact n_bw={by_snr[label]['exact_count_n_bw']:.2f} "
            f"n_half={by_snr[label]['exact_count_n_half']:.2f} "
            f"n_raw={by_snr[label]['exact_count_n_raw']:.2f} "
            f"(PhaseNet ref={PHASENET_EXACT_COUNT})"
        )

    return {
        "case": case_name,
        "n_true": scn.n_frac,
        "spacing_m": case.spacing_m,
        "n_max": n_max,
        "method": "nested_single_forward_bic",
        "phasenet_exact_count_ref": PHASENET_EXACT_COUNT,
        "by_snr": by_snr,
    }


def main():
    p = argparse.ArgumentParser(description="缝数 BIC 模型选择")
    p.add_argument("--tag", default="main")
    p.add_argument("--cases", default="dual_10m,triple_10m,dual_40m,triple_40m")
    p.add_argument("--snr", default="inf,40,30,20")
    p.add_argument("--n-mc", type=int, default=30)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    out_dir = Path(DEFAULT_OUT) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    snrs = _parse_snr(args.snr)

    all_rows = []
    for name in [c.strip() for c in args.cases.split(",") if c.strip()]:
        t0 = time.time()
        row = run_mc(
            name,
            snrs=snrs,
            n_mc=args.n_mc,
            n_max=args.n_max,
            seed=args.seed,
            workers=args.workers,
        )
        print(f"  case wall {time.time() - t0:.1f}s")
        (out_dir / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        slim = {
            "case": row["case"],
            "n_true": row["n_true"],
            "by_snr": {
                k: {
                    "exact_count_n_bw": v["exact_count_n_bw"],
                    "exact_count_n_half": v["exact_count_n_half"],
                    "exact_count_n_raw": v["exact_count_n_raw"],
                    "n_mc": v["n_mc"],
                }
                for k, v in row["by_snr"].items()
            },
        }
        all_rows.append(slim)

    summary = {
        "tag": args.tag,
        "method": "nested_single_forward_bic",
        "phasenet_exact_count_ref": PHASENET_EXACT_COUNT,
        "cases": all_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for slim in all_rows:
            snr_keys = list(slim["by_snr"].keys())

            def sk(s):
                return -1.0 if s == "inf" else float(s)

            snr_keys = sorted(snr_keys, key=sk, reverse=True)
            xs = [0.0 if k == "inf" else float(k) for k in snr_keys]
            ax.plot(xs, [slim["by_snr"][k]["exact_count_n_bw"] for k in snr_keys],
                    "o-", label=f"{slim['case']} n_bw")
            ax.plot(xs, [slim["by_snr"][k]["exact_count_n_half"] for k in snr_keys],
                    "s--", label=f"{slim['case']} n_half")
        ax.axhline(PHASENET_EXACT_COUNT, color="k", ls=":", label="PhaseNet 30.5%")
        ax.set_xlabel("SNR (dB); 0 = inf")
        ax.set_ylabel("exact-count rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / "exact_count_vs_snr.png", dpi=150)
        fig.savefig(fig_dir / "exact_count_vs_snr.svg")
        plt.close(fig)
    except Exception as e:
        print(f"  (plot skipped: {e})")

    print(f"\n写入 {out_dir}")


if __name__ == "__main__":
    main()
