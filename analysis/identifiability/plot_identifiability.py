# -*- coding: utf-8 -*-
"""
plot_identifiability.py — 可辨识性分析成图。

图 1  CRB vs 缝间距：噪声下界是否随间距恶化（是否存在信息意义上的分辨率墙）
图 2  噪声下界 vs 模型误差偏差：判定反问题是「信息受限」还是「偏差受限」
图 3  波速讨厌参数的代价：绝对深度 vs 缝间距（共模 / 差模）
图 4  (共模平移, 波速) 失配面：深度-波速简并谷

用法
----
    python -m analysis.identifiability.plot_identifiability
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["figure.constrained_layout.use"] = True

ROOT = os.path.join("output", "analysis", "identifiability")
FIGDIR = os.path.join(ROOT, "figures")

# 项目已有的经验参照（供对照，非本分析产出）
REF_LINES = {
    "倒谱谱支撑宽度 Δd_DR=38.4 m (DR=80dB)": 38.36,
    "倒谱盲检测单缝中位误差 190.7 m": 190.7,
    "P0 字典剥离单缝中位误差 4.2 m": 4.2,
    "Brunone 单缝深度系统偏差 10~20 m": 15.0,
}


def fig1_crb_vs_spacing(df: pd.DataFrame) -> None:
    sub = df[(df.fc_hz == 20) & (df.fd_converged)]
    snrs = sorted(sub.snr_db.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, fr in zip(axes, ["steady", "brunone"]):
        d = sub[sub.friction == fr]
        for snr in snrs:
            q = d[d.snr_db == snr].sort_values("spacing_placed_m")
            ax.plot(q.spacing_placed_m, q.std_x0_full_m, "o-", ms=4,
                    label=f"SNR={snr:g} dB")
        for name, val in REF_LINES.items():
            ax.axhline(val, ls="--", lw=0.9, color="0.55")
            ax.text(150, val * 1.12, name, fontsize=6.8, color="0.35", ha="right")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(4.2, 165)
        ax.set_xlabel("缝间距 [m]")
        ax.set_title(f"{fr}  (fc=20 Hz, 联合估计波速)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("位置 CRB 标准差下界  std(x0) [m]")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("图1  任何无偏估计器的位置精度下界与缝间距几乎无关\n"
                 "—— 5–20 m 区间不存在信息论意义上的分辨率墙", fontsize=11)
    fig.savefig(os.path.join(FIGDIR, "fig1_crb_vs_spacing.png"))
    plt.close(fig)


def fig2_bias_vs_crb(dg: pd.DataFrame) -> None:
    """CRB 随 SNR 严格按 10^(-SNR/20) 标度，故用一个锚点解析延拓到连续 SNR 轴。

    并标注每种模型误差的「等效 SNR」：噪声单独作用时产生同等误差所需的 SNR。
    """
    snr_axis = np.linspace(-30, 60, 400)
    biases = [
        ("bias_x0_a_rel_1pct_m", "波速错 1%", "#d62728", "-"),
        ("form_bias_x0_m", "摩阻模型形式错\n(steady ↔ brunone)", "#9467bd", ":"),
        ("bias_x0_kbr_rel_10pct_m", "Brunone k 错 10%", "#ff7f0e", "-."),
    ]
    spacings = sorted(dg.spacing_m.unique())
    fig, axes = plt.subplots(1, len(spacings), figsize=(14.5, 4.8), sharey=True)
    for ax, sp in zip(np.atleast_1d(axes), spacings):
        d = dg[(dg.spacing_m == sp) & (dg.friction == "brunone")
               & (dg.fc_hz == 20)].sort_values("snr_db")
        anchor_snr = float(d.snr_db.iloc[0])
        for key, sty, lbl in [("crb_std_x0_m", "-", "CRB 下界（波速已知）"),
                              ("crb_std_x0_joint_m", "--", "CRB 下界（联合估计波速）")]:
            c0 = float(d[key].iloc[0])
            ax.plot(snr_axis, c0 * 10 ** (-(snr_axis - anchor_snr) / 20.0),
                    sty, color="#1f77b4" if sty == "-" else "#4fa3d1",
                    lw=2 if sty == "-" else 1.4, label=lbl)
        ax.plot(d.snr_db, d.crb_std_x0_m, "o", color="#1f77b4", ms=5, zorder=5)

        c30 = float(d[d.snr_db == 30].crb_std_x0_m.iloc[0])
        notes = []
        for key, lbl, col, ls in biases:
            if key not in d or not np.isfinite(d[key].iloc[0]):
                continue
            b = abs(float(d[key].iloc[0]))
            ax.axhline(b, color=col, lw=1.8, ls=ls)
            snr_eq = 30.0 - 20.0 * np.log10(b / c30)
            ax.plot([snr_eq], [b], "v", color=col, ms=8, zorder=6)
            ax.text(58, b * 1.15, lbl.replace("\n", " "), fontsize=7.2,
                    color=col, ha="right", va="bottom")
            notes.append((f"{lbl.splitlines()[0]}：偏差 {b:.1f} m，"
                          f"等效 SNR {snr_eq:.0f} dB", col))
        ax.text(0.03, 0.035,
                "\n".join(n for n, _ in notes),
                transform=ax.transAxes, fontsize=7.0, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
        ax.set_yscale("log")
        ax.set_xlim(-30, 60)
        ax.set_ylim(1e-3, 3e2)
        ax.set_xlabel("SNR [dB]")
        ax.set_title(f"缝间距 {sp:g} m")
        ax.grid(alpha=0.3, which="both")
        ax.axvspan(10, 40, color="0.85", alpha=0.5, zorder=0)
        ax.text(25, 1.6e2, "项目基准\nSNR 区间", fontsize=7, ha="center",
                va="top", color="0.35")
    np.atleast_1d(axes)[0].set_ylabel("首缝位置误差 [m]")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("图2  噪声下界 vs 模型误差偏差（brunone, fc=20 Hz）\n"
                 "模型误差的「等效 SNR」远低于任何真实工况：反问题是偏差受限，不是信息受限",
                 fontsize=11)
    fig.savefig(os.path.join(FIGDIR, "fig2_bias_vs_crb.png"))
    plt.close(fig)


def fig3_wavespeed_penalty(df: pd.DataFrame) -> None:
    sub = df[(df.fc_hz == 20) & (df.snr_db == 30) & df.fd_converged]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for fr, mk in [("steady", "o"), ("brunone", "s")]:
        d = sub[sub.friction == fr].sort_values("spacing_placed_m")
        ax.plot(d.spacing_placed_m, d.a_penalty_x0, mk + "-", ms=5,
                label=f"{fr}：绝对深度 x0（共模）")
        ax.plot(d.spacing_placed_m, d.a_penalty_spacing, mk + "--", ms=5,
                alpha=0.75, label=f"{fr}：缝间距 x1-x0（差模）")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("缝间距 [m]")
    ax.set_ylabel("方差惩罚倍数  std(波速未知) / std(波速已知)")
    ax.set_title("图3  把波速当未知量联合估计的代价\n"
                 "绝对深度 ×1.15~1.6；缝间距 ×1.00~1.02（缝间距是波速不变量）",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.savefig(os.path.join(FIGDIR, "fig3_wavespeed_penalty.png"))
    plt.close(fig)


def fig4_valley(js: dict) -> None:
    rec = js["grid2d"]
    a_rel = np.array([r["a_rel"] for r in rec])
    dx = np.array([r["x0_offset_m"] for r in rec])
    dchi = np.array([r["delta_chi2"] for r in rec])
    ua, ux = np.unique(a_rel), np.unique(dx)
    Z = np.full((ua.size, ux.size), np.nan)
    for r in rec:
        Z[np.searchsorted(ua, r["a_rel"]), np.searchsorted(ux, r["x0_offset_m"])] = \
            r["delta_chi2"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.35, 1]})
    pc = ax.pcolormesh(ux, ua * 100, np.log10(np.maximum(Z, 1e-1)),
                       cmap="viridis", shading="nearest")
    x0 = js["x_true_m"][0]
    ax.plot(x0 * ua, ua * 100, "w--", lw=1.6, label=r"简并预测 $\Delta x = x_0\,\delta a/a$")
    vb = [min([r for r in rec if abs(r["a_rel"] - v) < 1e-12],
              key=lambda r: r["delta_chi2"]) for v in ua]
    ax.plot([b["x0_offset_m"] for b in vb], ua * 100, "r.-", ms=7, lw=1.2,
            label="各波速下的最优深度")
    ax.set_xlabel("缝群共模平移量 Δx [m]")
    ax.set_ylabel("波速相对误差 δa/a [%]")
    cb = fig.colorbar(pc, ax=ax); cb.set_label(r"$\log_{10}\,\Delta\chi^2$")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("(a) 失配面：最优深度精确跟随简并方向", fontsize=10)

    # 沿谷底的剖面：波速错误在「已重新拟合深度」之后仍留下多少失配
    vy = np.array([b["delta_chi2"] for b in vb])
    ax2.semilogy(ua * 100, np.maximum(vy, 1e-1), "ro-", ms=5)
    ax2.axhline(9.0, color="0.35", ls=":", lw=1.2,
                label=r"$\Delta\chi^2=9$（3σ 可区分阈值）")
    ax2.set_xlabel("波速相对误差 δa/a [%]")
    ax2.set_ylabel(r"谷底残余 $\Delta\chi^2$（深度已重新最优化）")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.set_title("(b) 谷底沿波速方向并不平坦\n→ 波速可由同一组数据联合估计出来",
                  fontsize=10)

    fig.suptitle(f"图4  深度–波速简并（{js['friction']}, D={js['spacing_m']:g} m, "
                 f"fc={js['fc_hz']:g} Hz, SNR={js['snr_db']:g} dB）",
                 fontsize=11.5)
    fig.savefig(os.path.join(FIGDIR, "fig4_depth_wavespeed_valley.png"))
    plt.close(fig)


def fig5_profile(js: dict) -> None:
    p = js.get("profile1d")
    if not p:
        return
    x = np.array(p["x0_m"]); d = np.array(p["delta_chi2"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].semilogy(x, np.maximum(d, 1e-2), lw=1.0)
    axes[0].axvline(js["x_true_m"][0], color="r", ls="--", lw=1, label="真值")
    axes[0].axhline(9.0, color="0.4", ls=":", lw=1, label=r"$\Delta\chi^2=9$（3σ 可区分）")
    axes[0].set_xlabel("首缝深度假设 x0 [m]"); axes[0].set_ylabel(r"$\Delta\chi^2$")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title("全域剖面：无竞争性局部极小")

    m = np.abs(x - js["x_true_m"][0]) <= 25
    axes[1].semilogy(x[m], np.maximum(d[m], 1e-2), "o-", ms=3.5, lw=1.0)
    axes[1].axvline(js["x_true_m"][0], color="r", ls="--", lw=1)
    axes[1].axhline(9.0, color="0.4", ls=":", lw=1)
    axes[1].axvline(js["x_true_m"][1], color="g", ls="-.", lw=1, label="次缝真值")
    axes[1].set_xlabel("首缝深度假设 x0 [m]")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title("真值邻域放大：单峰、陡峭")
    fig.suptitle(f"图5  似然失配剖面（{js['friction']}, D={js['spacing_m']:g} m, "
                 f"SNR={js['snr_db']:g} dB）—— 排除「全局多峰」这一失败解释", fontsize=11)
    fig.savefig(os.path.join(FIGDIR, "fig5_misfit_profile.png"))
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "main_dt1ms", "crb_table.csv"))
    dg = pd.read_csv(os.path.join(ROOT, "diagnosis", "diagnosis_table.csv"))
    fig1_crb_vs_spacing(df)
    fig2_bias_vs_crb(dg)
    fig3_wavespeed_penalty(df)

    lp = os.path.join(ROOT, "landscape", "steady_D20_fc20_snr30.json")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as fh:
            js = json.load(fh)
        if "grid2d" in js:
            fig4_valley(js)
        fig5_profile(js)
    print(f"图已写入 {os.path.abspath(FIGDIR)}")
    for f in sorted(os.listdir(FIGDIR)):
        print("   ", f)


if __name__ == "__main__":
    main()
