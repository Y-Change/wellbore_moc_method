"""
Figure 2: lumped fracture node + 2-D cepstral operator + truth-neighborhood peak pick.
Example profiles are computed from archived MOC npz (same operator as decay_table).
No quality grades.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from moc_simulate.cepstrum_mocdata import compute_moc_cepstrum

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.75,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "legend.frameon": False,
    "figure.dpi": 300,
})

COLOR_BLUE = "#1B4F72"
COLOR_TEAL = "#117864"
COLOR_ORANGE = "#D35400"
COLOR_RED = "#900C3F"
COLOR_SLATE = "#2C3E50"
COLOR_GREY = "#BDC3C7"
FRAC_COLORS = [COLOR_BLUE, COLOR_TEAL, COLOR_ORANGE, COLOR_RED]

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
DATA_BASE = os.path.join(BASE_DIR, "01_几何网格")
NPZ_DIR = os.path.join(
    DATA_BASE, [d for d in os.listdir(DATA_BASE) if "npz" in d.lower()][0]
)
OUT_DIR = os.path.join(BASE_DIR, "文章撰写", "稿件", "图表数据")


def load_profile(fn):
    data = np.load(os.path.join(NPZ_DIR, fn))
    t = data["t_sim"]
    H_wh = data["H_wh"]
    v = float(np.atleast_1d(data["v"])[0])
    fs = float(np.atleast_1d(data["fs"])[0])
    ts = float(np.atleast_1d(data["ts"])[0])
    L = float(np.atleast_1d(data["L"])[0])
    x_f = np.asarray(data["x_f_aligned"], dtype=float)
    out = compute_moc_cepstrum(
        t, H_wh, v, fs=fs, ts=ts, wellbore_length=L,
        wlen_sec=30.0, hop_sec=5.0, win_type="hamming",
    )
    depth = np.asarray(out["depth"], dtype=float)
    prof = -np.sum(out["C"], axis=1)
    return depth, prof, x_f


def draw_node_panel(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(4 / 5)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    y = 0.62
    ax.plot([0.06, 0.38], [y, y], color=COLOR_SLATE, lw=2.2, solid_capstyle="round")
    ax.plot([0.62, 0.94], [y, y], color=COLOR_SLATE, lw=2.2, solid_capstyle="round")
    node = Circle((0.50, y), 0.085, facecolor="#EBF5FB", edgecolor=COLOR_BLUE, lw=1.1, zorder=3)
    ax.add_patch(node)
    ax.text(0.50, y, r"$x_k$", ha="center", va="center", fontsize=8.0, color=COLOR_BLUE)

    ax.annotate("", xy=(0.38, y), xytext=(0.14, y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_BLUE, lw=1.0, mutation_scale=8))
    ax.annotate("", xy=(0.86, y), xytext=(0.62, y),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_TEAL, lw=1.0, mutation_scale=8))
    ax.text(0.22, y + 0.10, r"$Q_{L,k}$", ha="center", fontsize=7.0, color=COLOR_BLUE)
    ax.text(0.78, y + 0.10, r"$Q_{R,k}$", ha="center", fontsize=7.0, color=COLOR_TEAL)

    ax.annotate("", xy=(0.50, 0.28), xytext=(0.50, y - 0.085),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_ORANGE, lw=1.0, mutation_scale=8))
    ax.text(0.63, 0.42, r"$Q_{f,k}$", fontsize=7.0, color=COLOR_ORANGE)

    box = FancyBboxPatch(
        (0.12, 0.06), 0.76, 0.18,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#FDF2E9", edgecolor=COLOR_ORANGE, lw=0.8,
    )
    ax.add_patch(box)
    ax.text(
        0.50, 0.15,
        r"$Q_{f,k}=C_f\dot H_f+k_{\mathrm{leak}}\sqrt{\Delta H}$",
        ha="center", va="center", fontsize=6.8, color=COLOR_SLATE,
    )
    ax.text(0.50, 0.88, "Lumped fracture node", ha="center", fontsize=7.5, color=COLOR_SLATE)


def draw_profile_panel(ax, depth1, prof1, xf1, depth4, prof4, xf4):
    ax.set_box_aspect(4 / 5)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)

    def rel(depth, xf0):
        return depth - xf0

    m1 = (depth1 > xf1[0] - 40) & (depth1 < xf1[0] + 120)
    m4 = (depth4 > xf4[0] - 40) & (depth4 < xf4[-1] + 40)
    ax.plot(rel(depth1[m1], xf1[0]), prof1[m1], color=COLOR_BLUE, lw=1.15, label=r"$n=1$")
    ax.plot(rel(depth4[m4], xf4[0]), prof4[m4], color=COLOR_ORANGE, lw=1.15, label=r"$n=4$, $S=20$ m")
    for x in xf4:
        ax.axvline(x - xf4[0], color=COLOR_GREY, lw=0.55, ls="--", zorder=0, alpha=0.7)
    ax.set_xlabel(r"Relative depth $x-X_1$ (m)")
    ax.set_ylabel(r"Accumulated cepstrum $P_{2D}$ (a.u.)")
    ax.set_xlim(-30, 100)
    ax.set_ylim(-0.25, 6.2)
    ax.legend(loc="upper right", handlelength=1.1)
    ax.set_xticks([0, 20, 40, 60, 80])


def draw_neighborhood_panel(ax, depth, prof, xf, S=20.0):
    ax.set_box_aspect(4 / 5)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    r = min(15.0, 0.49 * S)
    x0 = xf[0]
    m = (depth > x0 - 25) & (depth < xf[-1] + 35)
    xr = depth[m] - x0
    pr = prof[m]
    ax.plot(xr, pr, color=COLOR_SLATE, lw=1.15, zorder=2)
    for i, x in enumerate(xf):
        ax.axvspan(x - x0 - r, x - x0 + r, color=FRAC_COLORS[i], alpha=0.12, zorder=0, lw=0)
        ax.axvline(x - x0, color=FRAC_COLORS[i], lw=0.6, ls="--", zorder=1, alpha=0.75)
        mask = np.abs(depth - x) <= r
        if np.any(mask):
            j = np.argmax(prof[mask])
            xp = depth[mask][j] - x0
            yp = prof[mask][j]
            ax.plot(xp, yp, "o", color=FRAC_COLORS[i], ms=4.2, zorder=4)
            ax.text(xp, yp + 0.18, r"$P_{%d}$" % (i + 1), ha="center", va="bottom",
                    fontsize=6.5, color=FRAC_COLORS[i])
    ax.set_xlabel(r"Relative depth $x-X_1$ (m)")
    ax.set_ylabel(r"Accumulated cepstrum $P_{2D}$ (a.u.)")
    ax.set_xlim(-20, 95)
    ymax = float(np.max(pr))
    ax.set_ylim(-0.2, ymax * 1.38)
    ax.text(
        0.04, 0.92,
        r"$r=\min(15\,\mathrm{m},\,0.49S)$",
        transform=ax.transAxes, fontsize=6.5, color=COLOR_SLATE, va="top",
    )


def save_figure(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, stem)
    fig.savefig(path + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(path + ".svg", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".{png,svg,pdf}")


def main():
    print("loading example profiles...")
    depth1, prof1, xf1 = load_profile("steady_x1_3000_sp_20_n_1.npz")
    depth4, prof4, xf4 = load_profile("steady_x1_3000_sp_20_n_4.npz")

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.45), constrained_layout=True)
    draw_node_panel(axes[0])
    draw_profile_panel(axes[1], depth1, prof1, xf1, depth4, prof4, xf4)
    draw_neighborhood_panel(axes[2], depth4, prof4, xf4, S=20.0)
    save_figure(fig, "Figure_2_Forward_and_Operator")


if __name__ == "__main__":
    main()
