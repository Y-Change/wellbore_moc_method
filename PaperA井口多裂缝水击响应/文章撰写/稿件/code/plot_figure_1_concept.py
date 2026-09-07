"""
Figure 1: local-label reading vs network observation (schematic).
No Psi flowchart. English labels for SPE; 4:5 boxes, Times + STIX.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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

PALETTE = ["#1B4F72", "#117864", "#D35400", "#900C3F"]
SLATE = "#2C3E50"
GREY = "#BDC3C7"

REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
OUT_DIR = os.path.join(
    REPO_ROOT, "PaperA井口多裂缝水击响应", "文章撰写", "稿件", "图表数据"
)


def _frame(ax, title):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(4 / 5)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=5)


def _wellbore(ax, y=0.38, n=4):
    ax.plot([0.10, 0.96], [y, y], color=SLATE, lw=2.2, solid_capstyle="round", zorder=2)
    valve = FancyBboxPatch(
        (0.035, y - 0.032), 0.055, 0.064,
        boxstyle="square,pad=0", facecolor="#EBF5FB", edgecolor=SLATE, lw=0.9, zorder=3,
    )
    ax.add_patch(valve)
    ax.plot([0.062, 0.062], [y - 0.018, y + 0.018], color=SLATE, lw=0.8, zorder=4)
    ax.plot([0.048, 0.076], [y, y], color=SLATE, lw=0.8, zorder=4)
    ax.text(0.062, y - 0.085, "WH", ha="center", va="top", fontsize=6.5, color=SLATE)
    xs = np.linspace(0.28, 0.82, n)
    for i, x in enumerate(xs):
        ax.plot([x, x], [y - 0.045, y + 0.045], color=PALETTE[i], lw=2.0, zorder=3)
        ax.plot(x, y, "o", color=PALETTE[i], ms=5.2, zorder=4)
        ax.text(x, y - 0.085, f"$x_{i + 1}$", ha="center", va="top", fontsize=6.5, color=PALETTE[i])
    ax.text(0.96, y - 0.085, "toe", ha="right", va="top", fontsize=6.5, color=SLATE)
    return xs, y


def _stems(ax, xs, heights, y0=0.58, color_list=None, label_prefix="P"):
    ax.plot([xs[0] - 0.06, xs[-1] + 0.06], [y0, y0], color=GREY, lw=0.7, zorder=1)
    for i, (x, h) in enumerate(zip(xs, heights)):
        c = color_list[i] if color_list is not None else PALETTE[i]
        ax.plot([x, x], [y0, y0 + h], color=c, lw=1.4, zorder=2)
        ax.plot(x, y0 + h, "o", color=c, ms=3.6, zorder=3)
        ax.text(x, y0 + h + 0.025, f"${label_prefix}_{i + 1}$", ha="center", va="bottom",
                fontsize=6.5, color=c)
    return y0


def draw_panel_a(ax):
    _frame(ax, "(a)")
    xs, yb = _wellbore(ax)
    heights = np.array([0.28, 0.21, 0.15, 0.10])
    _stems(ax, xs, heights)
    ax.text(0.50, 0.93, "Local-label reading", ha="center", va="center", fontsize=8.0, color=SLATE)
    ax.text(
        0.50, 0.86,
        r"each peak $\equiv$ that cluster; decay with distance only",
        ha="center", va="center", fontsize=6.5, color=SLATE,
    )
    for x, h in zip(xs, heights):
        ax.annotate(
            "",
            xy=(x, 0.38 + 0.05),
            xytext=(x, 0.58),
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6, ls=(0, (2, 1.5))),
        )
    ax.text(0.50, 0.12, "independent echoes; no downstream coupling", ha="center",
            fontsize=6.5, color=SLATE)


def draw_panel_b(ax):
    _frame(ax, "(b)")
    xs, yb = _wellbore(ax)
    # downstream-to-upstream coupling arrows along the pipe
    for i in range(len(xs) - 1, 0, -1):
        ax.annotate(
            "",
            xy=(xs[i - 1] + 0.02, yb + 0.07),
            xytext=(xs[i] - 0.02, yb + 0.07),
            arrowprops=dict(
                arrowstyle="-|>",
                color=PALETTE[i],
                lw=0.9,
                mutation_scale=7,
            ),
        )
    ax.text(0.55, yb + 0.115, "downstream feedback", ha="center", fontsize=6.3, color=PALETTE[2])

    box = FancyBboxPatch(
        (0.04, 0.70), 0.30, 0.18,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#EBF5FB", edgecolor="#1B4F72", lw=0.9, zorder=3,
    )
    ax.add_patch(box)
    ax.text(0.19, 0.84, "2-D cepstrum", ha="center", va="center", fontsize=7.0, color="#1B4F72")
    ax.text(0.19, 0.76, "30 s Hamming", ha="center", va="center", fontsize=6.3, color="#1B4F72")
    ax.annotate(
        "",
        xy=(0.19, 0.70),
        xytext=(0.062, yb + 0.03),
        arrowprops=dict(arrowstyle="-|>", color="#1B4F72", lw=0.9, mutation_scale=7),
    )

    # apparent peaks: leading peak reduced, later peaks non-monotonic
    heights = np.array([0.16, 0.11, 0.055, 0.09])
    _stems(ax, xs, heights, y0=0.58)
    ax.text(0.62, 0.93, "Network observation", ha="center", va="center", fontsize=8.0, color=SLATE)
    ax.text(
        0.66, 0.86,
        r"wellhead $P_i$: apparent, nonlocal",
        ha="center", va="center", fontsize=6.5, color=SLATE,
    )
    ax.text(0.50, 0.12, "fixed operator on a discrete multi-interface network",
            ha="center", fontsize=6.5, color=SLATE)


def save_figure(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, stem)
    fig.savefig(path + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(path + ".svg", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", path + ".{png,svg,pdf}")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.55), constrained_layout=True)
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    save_figure(fig, "Figure_1_Local_vs_Network")


if __name__ == "__main__":
    main()
