# -*- coding: utf-8 -*-
"""
plot_efficiency.py — 效率检验主图与波速代价图。

输入：output/analysis/identifiability/efficiency/<tag>/*.json
输出：output/analysis/identifiability/efficiency/<tag>/figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "figure.constrained_layout.use": True,
    "font.size": 10,
})


def _load_case(path: Path) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _snr_sort_key(lab: str) -> float:
    return float("inf") if lab == "inf" else float(lab)


def fig1_rmse_vs_snr(cases: Dict[str, Dict], out: Path) -> None:
    """主图：SNR–定位误差，MLE / P0 / 倒谱 + CRB 线。"""
    names = list(cases.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        data = cases[name]
        by = data["by_snr"]
        labels = sorted(by.keys(), key=_snr_sort_key)
        # 跳过 inf 画在最左，用高 SNR 代替轴位置
        snr_x, crb_a, mle_a, mle_b, p0, cep = [], [], [], [], [], []
        for lab in labels:
            if lab == "inf":
                continue
            b = by[lab]
            snr_x.append(float(lab))
            crb_a.append(float(np.mean(b["crb_x_a_m"])))
            mle_a.append(b["mle_a"]["rmse_m"])
            mle_b.append(b["mle_b"]["rmse_m"])
            p0.append(b.get("p0", {}).get("rmse_m"))
            cep.append(b.get("cepstrum", {}).get("rmse_m"))

        ax.semilogy(snr_x, crb_a, "k--", lw=1.8, label="CRB (A档, a已知)")
        ax.semilogy(snr_x, mle_a, "o-", color="#2166ac", lw=1.6, label="全波形 MLE-A")
        ax.semilogy(snr_x, mle_b, "s-", color="#92c5de", lw=1.4, label="全波形 MLE-B (联合a)")
        if any(v is not None for v in p0):
            ax.semilogy(snr_x, [v if v is not None else np.nan for v in p0],
                        "D-", color="#f4a582", lw=1.4, label="P0 字典剥离")
        if any(v is not None for v in cep):
            ax.semilogy(snr_x, [v if v is not None else np.nan for v in cep],
                        "^-", color="#b2182b", lw=1.4, label="倒谱")

        # SNR=∞ 注释
        if "inf" in by:
            inf_a = by["inf"]["mle_a"]["rmse_m"]
            ax.annotate(
                f"SNR=∞\nMLE RMSE={inf_a:.2g} m",
                xy=(0.03, 0.97), xycoords="axes fraction",
                va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
            )

        ax.set_xlabel("SNR / dB")
        ax.set_title(name.replace("_", " "))
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.invert_xaxis()  # 高 SNR 在左，阅读更自然？不——保持常规递增
        ax.invert_xaxis()  # 撤销，保持 SNR 递增

    axes[0].set_ylabel("定位 RMSE / m")
    axes[-1].legend(loc="best", fontsize=8, framealpha=0.92)
    fig.suptitle("全波形 MLE 效率检验：能否贴到 Cramér-Rao 界", fontsize=12)
    fig.savefig(out / "fig1_rmse_vs_snr.png", dpi=160)
    fig.savefig(out / "fig1_rmse_vs_snr.pdf")
    plt.close(fig)


def fig2_efficiency_ratio(cases: Dict[str, Dict], out: Path) -> None:
    """效率比 RMSE/CRB 柱状图。"""
    rows = []
    for name, data in cases.items():
        for lab, b in data["by_snr"].items():
            if lab == "inf":
                continue
            rows.append({
                "case": name,
                "snr": float(lab),
                "eff_a": b["mle_a"]["efficiency_ratio"],
                "eff_b": b["mle_b"]["efficiency_ratio"],
            })
    if not rows:
        return

    cases_u = list(dict.fromkeys(r["case"] for r in rows))
    snrs_u = sorted(set(r["snr"] for r in rows), reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.35
    x = np.arange(len(cases_u) * len(snrs_u))
    # 简化：按 case 分组，组内不同 SNR
    xpos, labels = [], []
    k = 0
    for c in cases_u:
        for s in snrs_u:
            xpos.append(k)
            labels.append(f"{c}\n{s:g}dB")
            k += 1
        k += 0.5  # 组间距

    eff_a, eff_b, xt = [], [], []
    k = 0
    for c in cases_u:
        for s in snrs_u:
            hit = next(r for r in rows if r["case"] == c and r["snr"] == s)
            eff_a.append(hit["eff_a"] if hit["eff_a"] is not None else np.nan)
            eff_b.append(hit["eff_b"] if hit["eff_b"] is not None else np.nan)
            xt.append(k)
            k += 1
        k += 0.6

    xt = np.asarray(xt, dtype=float)
    ax.bar(xt - width / 2, eff_a, width, label="MLE-A / CRB-A", color="#2166ac")
    ax.bar(xt + width / 2, eff_b, width, label="MLE-B / CRB-B", color="#92c5de")
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="效率=1（贴界）")
    ax.axhline(3.0, color="0.5", ls=":", lw=1.0, label="效率=3（可接受）")
    ax.set_xticks(xt)
    ax.set_xticklabels(
        [f"{c.split('_')[0]}\n{s:g}" for c in cases_u for s in snrs_u],
        fontsize=8,
    )
    ax.set_ylabel("效率比 RMSE / CRB")
    ax.set_xlabel("场景 × SNR (dB)")
    ax.set_ylim(0, max(5, np.nanmax(eff_a + eff_b) * 1.1))
    ax.legend(fontsize=8)
    ax.set_title("估计器效率：接近 1 表示信息已被用尽")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.savefig(out / "fig2_efficiency_ratio.png", dpi=160)
    fig.savefig(out / "fig2_efficiency_ratio.pdf")
    plt.close(fig)


def fig3_depth_vs_spacing(cases: Dict[str, Dict], out: Path) -> None:
    """B 档：绝对深度 RMSE vs 缝间距 RMSE。"""
    dual = {k: v for k, v in cases.items() if "dual" in k}
    if not dual:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    markers = {"dual_10m": "o", "dual_40m": "s"}
    for name, data in dual.items():
        by = data["by_snr"]
        depth_rmse, spacing_rmse, snr_labs = [], [], []
        for lab in sorted(by.keys(), key=_snr_sort_key):
            if lab == "inf":
                continue
            b = by[lab]
            d = b["mle_b"]["rmse_m"]
            s = b["mle_b"].get("spacing_rmse_m")
            if s is None or not np.isfinite(s):
                continue
            depth_rmse.append(d)
            spacing_rmse.append(s)
            snr_labs.append(lab)
            ax.annotate(lab, (s, d), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        ax.plot(spacing_rmse, depth_rmse, markers.get(name, "o") + "-",
                label=name, lw=1.5)

    # 参考线 y=x
    lims = ax.get_xlim(), ax.get_ylim()
    lo = min(lims[0][0], lims[1][0], 1e-4)
    hi = max(lims[0][1], lims[1][1], 1e-2)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="depth = spacing")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("缝间距 RMSE / m（B 档联合估波速）")
    ax.set_ylabel("绝对深度 RMSE / m（B 档）")
    ax.set_title("波速联合估计：间距稳健、绝对深度更脆弱")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.savefig(out / "fig3_depth_vs_spacing.png", dpi=160)
    fig.savefig(out / "fig3_depth_vs_spacing.pdf")
    plt.close(fig)


def fig4_gap_orders(cases: Dict[str, Dict], out: Path) -> None:
    """数量级鸿沟：同一 SNR 下各方法 RMSE 对比（选 SNR=30）。"""
    target = "30"
    methods = ["CRB-A", "MLE-A", "MLE-B", "P0", "倒谱"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(cases))
    width = 0.15
    colors = ["#000000", "#2166ac", "#92c5de", "#f4a582", "#b2182b"]
    for i, (meth, color) in enumerate(zip(methods, colors)):
        vals = []
        for name, data in cases.items():
            b = data["by_snr"].get(target)
            if b is None:
                vals.append(np.nan)
                continue
            if meth == "CRB-A":
                vals.append(float(np.mean(b["crb_x_a_m"])))
            elif meth == "MLE-A":
                vals.append(b["mle_a"]["rmse_m"])
            elif meth == "MLE-B":
                vals.append(b["mle_b"]["rmse_m"])
            elif meth == "P0":
                vals.append(b.get("p0", {}).get("rmse_m", np.nan))
            else:
                vals.append(b.get("cepstrum", {}).get("rmse_m", np.nan))
        ax.bar(x + (i - 2) * width, vals, width, label=meth, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(list(cases.keys()), fontsize=8)
    ax.set_ylabel(f"定位 RMSE / m  (SNR={target} dB)")
    ax.set_title("信息下界 vs 现有估计器：数量级鸿沟")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)
    fig.savefig(out / "fig4_gap_orders.png", dpi=160)
    fig.savefig(out / "fig4_gap_orders.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root", default="output/analysis/identifiability/efficiency")
    args = ap.parse_args()

    root = Path(args.root) / args.tag
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    cases = {}
    for p in sorted(root.glob("*.json")):
        if p.name == "meta.json":
            continue
        cases[p.stem] = _load_case(p)
    if not cases:
        raise SystemExit(f"未找到结果 JSON：{root}")

    fig1_rmse_vs_snr(cases, out)
    fig2_efficiency_ratio(cases, out)
    fig3_depth_vs_spacing(cases, out)
    fig4_gap_orders(cases, out)
    print(f"写出图 → {out}")


if __name__ == "__main__":
    main()
