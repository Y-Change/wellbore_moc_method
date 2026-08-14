# -*- coding: utf-8 -*-
"""
run_fno_mle_gate.py — P1 反演门槛：FNO-MLE vs MOC-MLE 位置差 vs CRB。

硬门槛（咨询性）：median |x_FNO - x_MOC| < CRB_A(σ_x)（有限 SNR）。
SNR=∞ 时 CRB=0，改报绝对米制差并以 CRB@40dB 作参照。

用法
----
    D:\\Anaconda\\envs\\torch24\\python.exe -m analysis.identifiability.run_fno_mle_gate \\
        --tag smoke --cases single_4000 --snr 40 --n-mc 3 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _p = os.path.dirname(_d)
    if _p == _d:
        break
    _d = _p

from analysis.identifiability.crb_core import forward, observe
from analysis.identifiability.mle import add_noise, one_step_from_jacobian
from analysis.identifiability.run_efficiency import compute_crb_baselines
from analysis.identifiability.scenarios import case_by_name, make_obs, make_scenario
from neural_operator.encode_impulse_torch import (
    encode_fno_input,
    interp1d_torch,
    observe_torch,
)
from neural_operator.fno_1d import FNO1dSurrogate


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_snr(s: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for tok in _parse_list(s):
        if tok.lower() in ("inf", "infty", "infinity", "none"):
            out.append(None)
        else:
            out.append(float(tok))
    return out


def load_fno(
    ckpt: str,
    device: torch.device,
    *,
    width: int = 64,
    modes: int = 32,
    layers: int = 4,
) -> nn.Module:
    model = FNO1dSurrogate(
        in_channels=4, out_channels=1, width=width, modes=modes, num_layers=layers
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def fno_signal(
    model: nn.Module,
    x_f: torch.Tensor,
    Cf: Sequence[float],
    kleak: Sequence[float],
    *,
    t_moc: torch.Tensor,
    ts: float,
    dt: float,
    tf: float,
    a: float,
    fc_hz: float,
    n_time: int,
    device: torch.device,
) -> torch.Tensor:
    """FNO 正演 → 插到 MOC 时间轴 → 观测算子，返回与 MOC ``observe`` 同长的 s。"""
    cf_t = torch.as_tensor(Cf, dtype=x_f.dtype, device=device)
    kl_t = torch.as_tensor(kleak, dtype=x_f.dtype, device=device)
    x_in, t_fno = encode_fno_input(
        x_f,
        cf_t,
        kl_t,
        n_time=n_time,
        tf=tf,
        ts=ts,
        wavespeed=a,
        device=device,
        dtype=x_f.dtype,
    )
    H = model(x_in)[0, 0]
    H_moc = interp1d_torch(t_fno, H, t_moc)
    return observe_torch(H_moc, t_moc, ts=ts, dt=dt, fc_hz=fc_hz)


def fno_mle(
    model: nn.Module,
    y: np.ndarray,
    x_init: np.ndarray,
    Cf: Sequence[float],
    kleak: Sequence[float],
    *,
    t_moc: np.ndarray,
    ts: float,
    dt: float,
    tf: float,
    a: float,
    fc_hz: float,
    n_time: int,
    device: torch.device,
    n_steps: int = 80,
    lr: float = 1.0,
) -> Dict[str, object]:
    """A 档：只估位置，Cf/kleak/a 固定。Adam 真值邻域精修。"""
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t_moc, dtype=torch.float32, device=device)
    x_param = nn.Parameter(
        torch.as_tensor(np.asarray(x_init, dtype=np.float32), device=device)
    )
    opt = torch.optim.Adam([x_param], lr=lr)
    best_rss = float("inf")
    best_x = x_param.detach().cpu().numpy().copy()
    hist = []

    for step in range(int(n_steps)):
        opt.zero_grad()
        # 保持深度顺序
        x_sorted, _ = torch.sort(x_param)
        s = fno_signal(
            model,
            x_sorted,
            Cf,
            kleak,
            t_moc=t_t,
            ts=ts,
            dt=dt,
            tf=tf,
            a=a,
            fc_hz=fc_hz,
            n_time=n_time,
            device=device,
        )
        m = min(s.numel(), y_t.numel())
        rss = torch.mean((s[:m] - y_t[:m]) ** 2)
        rss.backward()
        opt.step()
        # 软约束：井筒内
        with torch.no_grad():
            # 井筒长度默认 5000 m；用双程时延上界作软夹紧
            x_max = 0.5 * float(a) * (float(tf) - float(ts)) - 50.0
            x_param.clamp_(50.0, max(100.0, x_max))

        r = float(rss.detach().cpu())
        hist.append(r)
        if r < best_rss:
            best_rss = r
            best_x = torch.sort(x_param.detach())[0].cpu().numpy().copy()

    return {
        "x_hat": best_x.astype(np.float64),
        "rss": best_rss,
        "n_steps": int(n_steps),
        "rss_final": float(hist[-1]) if hist else float("nan"),
    }


def run_case(
    case_name: str,
    *,
    model: nn.Module,
    device: torch.device,
    snr_list: Sequence[Optional[float]],
    n_mc: int,
    seed: int,
    workers: int,
    n_time: int,
    fno_steps: int,
    fno_lr: float,
    init_jitter_m: float,
) -> Dict[str, object]:
    case = case_by_name(case_name)
    scn = make_scenario(case)
    obs = make_obs()
    print(f"\n=== case {case_name}  forward + CRB ===")
    t0 = time.time()
    fr = forward(scn)
    s0 = observe(fr.H_wh, fr.t, scn, obs)
    crb = compute_crb_baselines(scn, obs, workers=workers)
    jac_a = crb["jac_a"]
    placed = np.asarray(crb["placed_x"], dtype=np.float64)
    print(f"  placed_x={placed.tolist()}  CRB@40dB A={crb['std_x_a_ref']}  ({time.time()-t0:.1f}s)")

    t_moc = np.asarray(fr.t, dtype=np.float64)
    scale_std = crb["scale_std"]
    rng = np.random.default_rng(seed)

    snr_blocks = []
    for snr in snr_list:
        snr_label = "inf" if snr is None else f"{snr:g}"
        crb_std = [
            scale_std(ref, snr) for ref in crb["std_x_a_ref"]
        ]
        crb_ref40 = list(crb["std_x_a_ref"])
        rows = []
        print(f"  SNR={snr_label}  n_mc={n_mc} ...")
        for i in range(n_mc):
            y, sigma = add_noise(s0, snr, rng)
            moc = one_step_from_jacobian(jac_a, y)
            x_moc = np.asarray(moc["x_hat"], dtype=np.float64)

            jitter = rng.uniform(-init_jitter_m, init_jitter_m, size=placed.shape)
            x_init = placed + jitter
            # 保持序
            x_init = np.sort(x_init)

            fno = fno_mle(
                model,
                y,
                x_init,
                case.Cf,
                case.kleak,
                t_moc=t_moc,
                ts=scn.ts,
                dt=scn.dt,
                tf=scn.tf,
                a=scn.a_nominal,
                fc_hz=float(obs.fc_hz),
                n_time=n_time,
                device=device,
                n_steps=fno_steps,
                lr=fno_lr,
            )
            x_fno = np.asarray(fno["x_hat"], dtype=np.float64)
            # 长度对齐（排序后）
            n = min(len(x_fno), len(x_moc))
            diff = np.abs(x_fno[:n] - x_moc[:n])
            rows.append(
                {
                    "trial": i,
                    "x_moc": x_moc.tolist(),
                    "x_fno": x_fno.tolist(),
                    "x_init": x_init.tolist(),
                    "abs_diff_m": diff.tolist(),
                    "max_abs_diff_m": float(np.max(diff)),
                    "mean_abs_diff_m": float(np.mean(diff)),
                    "fno_rss": float(fno["rss"]),
                    "sigma": float(sigma),
                }
            )
            if (i + 1) % max(1, n_mc // 5) == 0 or i == 0:
                print(
                    f"    trial {i+1}/{n_mc}  |Δx|_max={diff.max():.4f} m  "
                    f"x_fno={x_fno.tolist()}  x_moc={x_moc.tolist()}"
                )

        max_diffs = np.array([r["max_abs_diff_m"] for r in rows], dtype=float)
        mean_diffs = np.array([r["mean_abs_diff_m"] for r in rows], dtype=float)
        # 门槛：用各缝 CRB 的均值作标尺
        crb_scale = float(np.mean(crb_std)) if snr is not None else float(np.mean(crb_ref40))
        med_max = float(np.median(max_diffs))
        strict = bool(med_max < crb_scale) if snr is not None else bool(med_max < crb_scale)
        advisory = bool(med_max < 2.0 * crb_scale)
        block = {
            "snr_db": snr_label,
            "crb_std_x_a_m": crb_std,
            "crb_std_x_a_ref40_m": crb_ref40,
            "crb_scale_m": crb_scale,
            "median_max_abs_diff_m": med_max,
            "mean_max_abs_diff_m": float(np.mean(max_diffs)),
            "median_mean_abs_diff_m": float(np.median(mean_diffs)),
            "ratio_to_crb": float(med_max / crb_scale) if crb_scale > 0 else float("inf"),
            "pass_strict_lt_crb": strict,
            "pass_advisory_lt_2crb": advisory,
            "trials": rows,
        }
        snr_blocks.append(block)
        print(
            f"  → SNR={snr_label}: median max|Δx|={med_max:.4f} m  "
            f"CRB_scale={crb_scale:.4f} m  ratio={block['ratio_to_crb']:.2f}  "
            f"strict={strict}  advisory={advisory}"
        )

    return {
        "case": case_name,
        "placed_x": placed.tolist(),
        "x_true": list(case.x_f),
        "n_mc": n_mc,
        "fno_steps": fno_steps,
        "fno_lr": fno_lr,
        "init_jitter_m": init_jitter_m,
        "snr_blocks": snr_blocks,
        "pass_all_strict": all(b["pass_strict_lt_crb"] for b in snr_blocks),
        "pass_all_advisory": all(b["pass_advisory_lt_2crb"] for b in snr_blocks),
    }


def main():
    if sys.platform.startswith("win") and hasattr(sys.stdout, "buffer") and not sys.stdout.closed:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="FNO-MLE vs MOC-MLE inversion gate")
    p.add_argument("--tag", default="main")
    p.add_argument("--cases", default="single_4000,dual_10m")
    p.add_argument("--snr", default="40")
    p.add_argument("--n-mc", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--ckpt",
        default="output/models/fno_close2000/fno_surrogate_best.pt",
    )
    p.add_argument("--n-time", type=int, default=4096)
    p.add_argument("--fno-steps", type=int, default=80)
    p.add_argument("--fno-lr", type=float, default=1.0)
    p.add_argument("--init-jitter-m", type=float, default=2.0)
    p.add_argument(
        "--out-dir",
        default="",
        help="默认 output/analysis/identifiability/fno_mle_gate/<tag>",
    )
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else Path(
        f"output/analysis/identifiability/fno_mle_gate/{args.tag}"
    )
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("P1 FNO-MLE vs MOC-MLE inversion gate")
    print(f"  device={device}  ckpt={args.ckpt}")
    print(f"  cases={args.cases}  snr={args.snr}  n_mc={args.n_mc}")
    print("=" * 72)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    model = load_fno(args.ckpt, device)

    cases = _parse_list(args.cases)
    snr_list = _parse_snr(args.snr)
    results = []
    t_all = time.time()
    for name in cases:
        r = run_case(
            name,
            model=model,
            device=device,
            snr_list=snr_list,
            n_mc=args.n_mc,
            seed=args.seed,
            workers=args.workers,
            n_time=args.n_time,
            fno_steps=args.fno_steps,
            fno_lr=args.fno_lr,
            init_jitter_m=args.init_jitter_m,
        )
        results.append(r)
        (out / f"{name}.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8"
        )

    summary = {
        "tag": args.tag,
        "ckpt": os.path.abspath(args.ckpt),
        "device": str(device),
        "cases": cases,
        "snr": [ ("inf" if s is None else s) for s in snr_list ],
        "n_mc": args.n_mc,
        "gate_note": (
            "Hard gate: median max|x_FNO-x_MOC| < CRB_A (finite SNR). "
            "Advisory: < 2*CRB. Waveform PASS does not imply inversion PASS."
        ),
        "pass_all_strict": all(r["pass_all_strict"] for r in results),
        "pass_all_advisory": all(r["pass_all_advisory"] for r in results),
        "results": [
            {
                "case": r["case"],
                "pass_all_strict": r["pass_all_strict"],
                "pass_all_advisory": r["pass_all_advisory"],
                "snr_summary": [
                    {
                        "snr_db": b["snr_db"],
                        "median_max_abs_diff_m": b["median_max_abs_diff_m"],
                        "crb_scale_m": b["crb_scale_m"],
                        "ratio_to_crb": b["ratio_to_crb"],
                        "pass_strict_lt_crb": b["pass_strict_lt_crb"],
                        "pass_advisory_lt_2crb": b["pass_advisory_lt_2crb"],
                    }
                    for b in r["snr_blocks"]
                ],
            }
            for r in results
        ],
        "wall_s": float(time.time() - t_all),
    }
    (out / "gate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2))
    for r in summary["results"]:
        print(json.dumps(r, indent=2))
    print(f"wrote {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
