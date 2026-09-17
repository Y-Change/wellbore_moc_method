"""Build manuscript Figure 8 from the validated Section 5.4 and 5.5 metrics.

The six panels preserve the calculations used by the two section figures while
placing the spacing--multiplicity and depth--spacing evidence on one consistent
2 x 3 canvas.  Deterministic simulation-grid values are shown without invented
uncertainty; interpolated phase maps retain all original sample locations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

from papera_plot_style import (
    FRACTURE_COLORS,
    NATURE_BLUE_WARM,
    NATURE_TEAL_WARM,
    PALETTE,
    add_reference_line,
    configure_papera_style,
    format_colorbar,
    legend_overlaps_data,
    make_papera_colormap,
    style_axis,
    verify_figure_exports,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PAPER_DIR = REPO_ROOT / "PaperA井口多裂缝水击响应"

DATA_SN = (
    PAPER_DIR
    / "文章撰写"
    / "5.4_间距与缝数交互相图分析"
    / "data"
    / "interaction_S_n_metrics.csv"
)
DATA_XS = (
    PAPER_DIR
    / "文章撰写"
    / "5.5_深度与间距交互相图分析"
    / "data"
    / "interaction_X1_S_metrics.csv"
)
OUTPUT_STEM = (
    PAPER_DIR
    / "文章撰写"
    / "稿件"
    / "图表数据"
    / "Figure_8_Interaction_Phase_Maps"
)

# Nature-style render-time alignment gate.  The plotting backend remains
# Matplotlib; this helper only audits the final axes geometry.
NATURE_FIGURE_SCRIPTS = (
    Path.home() / ".agents" / "skills" / "nature-figure" / "scripts"
)
if str(NATURE_FIGURE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(NATURE_FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


FIGSIZE = (7.35, 5.00)
PANEL_LABELS = tuple("abcdef")


def _load_and_validate_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both complete metric grids and fail loudly on missing observations."""
    sn = pd.read_csv(DATA_SN)
    xs = pd.read_csv(DATA_XS)

    required_sn = {
        "x1",
        "n_total",
        "spacing_m",
        "L_span_m",
        "P_1",
        "R_end",
    }
    required_xs = {"n_total", "x1", "spacing_m", "P_1", "alpha_2"}
    for name, frame, required in (
        ("spacing-multiplicity", sn, required_sn),
        ("depth-spacing", xs, required_xs),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} metrics are missing columns: {missing}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} metrics contain missing values")

    if len(sn) != 70 or sn[["spacing_m", "n_total"]].duplicated().any():
        raise ValueError("Expected a unique 10 x 7 spacing-multiplicity grid")
    if set(sn["spacing_m"].unique()) != set(np.arange(10.0, 101.0, 10.0)):
        raise ValueError("Unexpected spacing coordinates in Section 5.4 metrics")
    if set(sn["n_total"].unique()) != set(range(2, 9)):
        raise ValueError("Unexpected multiplicity coordinates in Section 5.4 metrics")
    if not np.allclose(sn["x1"], 3000.0):
        raise ValueError("Section 5.4 metrics must use X1 = 3000 m")

    if len(xs) != 60 or xs[["spacing_m", "x1"]].duplicated().any():
        raise ValueError("Expected a unique 10 x 6 depth-spacing grid")
    if set(xs["spacing_m"].unique()) != set(np.arange(10.0, 101.0, 10.0)):
        raise ValueError("Unexpected spacing coordinates in Section 5.5 metrics")
    if set(xs["x1"].unique()) != set(np.arange(2000.0, 4501.0, 500.0)):
        raise ValueError("Unexpected depth coordinates in Section 5.5 metrics")
    if not np.allclose(xs["n_total"], 4):
        raise ValueError("Section 5.5 metrics must use n = 4")

    return sn, xs


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a Nature-compatible label at a common physical panel anchor."""
    ax.text(
        -0.17,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        clip_on=False,
    )


def _surface_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    x_dense: np.ndarray,
    y_dense: np.ndarray,
    cmap,
    colorbar_label: str,
) -> None:
    """Draw the original cubic interpolation and retain every grid observation."""
    xx, yy = np.meshgrid(x_dense, y_dense)
    points = frame[[x_col, y_col]].to_numpy(float)
    values = frame[value_col].to_numpy(float)
    surface = griddata(points, values, (xx, yy), method="cubic")
    if np.isnan(surface).any():
        raise ValueError(f"Cubic interpolation for {value_col} left undefined cells")

    levels = np.linspace(values.min(), values.max(), 15)
    contour = ax.contourf(
        xx,
        yy,
        surface,
        levels=levels,
        cmap=cmap,
        extend="both",
        alpha=0.96,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=6,
        color=PALETTE["slate"],
        alpha=0.48,
        edgecolors="none",
        zorder=3,
    )

    cax = inset_axes(
        ax,
        width="4.2%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.035, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cax.set_in_layout(False)
    cbar = fig.colorbar(contour, cax=cax, orientation="vertical")
    format_colorbar(cbar, colorbar_label, fontsize=7.5)


def _style_phase_map_axis(
    ax: plt.Axes,
    *,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
) -> None:
    ax.set_xlabel(r"Fracture spacing, $S$ (m)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(10, 100)
    ax.set_ylim(*ylim)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_yticks(yticks)


def _freeze_equal_gutters(fig: plt.Figure, axes: np.ndarray) -> None:
    """Freeze constrained layout after centring each middle-column panel.

    Different right-side decoration widths can shift a constrained-layout
    middle panel by a few points even when all plot areas have equal widths.
    The symmetric shift below preserves every panel size while making the two
    repeated horizontal gutters identical at the final physical dimensions.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for row in axes:
        left, middle, right = (axis.get_position().frozen() for axis in row)
        gutter_left = middle.x0 - left.x1
        gutter_right = right.x0 - middle.x1
        shift = 0.5 * (gutter_left - gutter_right)
        row[1].set_position(
            [middle.x0 - shift, middle.y0, middle.width, middle.height]
        )
    fig.canvas.draw()


def build_figure(sn: pd.DataFrame, xs: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Assemble the six complementary interaction panels."""
    configure_papera_style()
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(
        w_pad=0.075,
        h_pad=0.075,
        wspace=0.24,
        hspace=0.17,
    )

    for ax, label in zip(axes.flat, PANEL_LABELS):
        style_axis(ax, box_aspect=4 / 5)
        _add_panel_label(ax, label)

    cmap_blue_warm = make_papera_colormap(
        "figure8_blue_warm", NATURE_BLUE_WARM
    )
    cmap_teal_warm = make_papera_colormap(
        "figure8_teal_warm", NATURE_TEAL_WARM
    )

    _surface_panel(
        fig,
        axes[0, 0],
        sn,
        x_col="spacing_m",
        y_col="n_total",
        value_col="P_1",
        x_dense=np.linspace(10, 100, 120),
        y_dense=np.linspace(2, 8, 120),
        cmap=cmap_blue_warm,
        colorbar_label=r"$P_1$ (a.u.)",
    )
    _style_phase_map_axis(
        axes[0, 0],
        ylabel=r"Fracture multiplicity, $n$",
        ylim=(2, 8),
        yticks=list(range(2, 9)),
    )

    _surface_panel(
        fig,
        axes[0, 1],
        sn,
        x_col="spacing_m",
        y_col="n_total",
        value_col="R_end",
        x_dense=np.linspace(10, 100, 120),
        y_dense=np.linspace(2, 8, 120),
        cmap=cmap_teal_warm,
        colorbar_label=r"$R_{\mathrm{end}}$",
    )
    _style_phase_map_axis(
        axes[0, 1],
        ylabel=r"Fracture multiplicity, $n$",
        ylim=(2, 8),
        yticks=list(range(2, 9)),
    )

    ax = axes[0, 2]
    for n_value, color in zip(range(2, 9), FRACTURE_COLORS[:7]):
        subset = sn.loc[sn["n_total"] == n_value].sort_values("L_span_m")
        ax.plot(
            subset["L_span_m"],
            subset["R_end"],
            marker="o",
            color=color,
            lw=1.0,
            markersize=3.2,
            label=f"n = {n_value}",
        )
    add_reference_line(ax, 0.0)
    ax.set_xlabel(r"$L_{\mathrm{span}}=(n-1)S$ (m)")
    ax.set_ylabel(r"End ratio, $R_{\mathrm{end}}$")
    ax.set_xlim(-15, 720)
    ax.set_ylim(0.0, 0.78)
    legend_c = ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=6.5,
        handlelength=1.1,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.25,
    )

    _surface_panel(
        fig,
        axes[1, 0],
        xs,
        x_col="spacing_m",
        y_col="x1",
        value_col="P_1",
        x_dense=np.linspace(10, 100, 120),
        y_dense=np.linspace(2000, 4500, 120),
        cmap=cmap_blue_warm,
        colorbar_label=r"$P_1$ (a.u.)",
    )
    _style_phase_map_axis(
        axes[1, 0],
        ylabel=r"First-cluster depth, $X_1$ (m)",
        ylim=(2000, 4500),
        yticks=[2000, 2500, 3000, 3500, 4000, 4500],
    )

    _surface_panel(
        fig,
        axes[1, 1],
        xs,
        x_col="spacing_m",
        y_col="x1",
        value_col="alpha_2",
        x_dense=np.linspace(10, 100, 120),
        y_dense=np.linspace(2000, 4500, 120),
        cmap=cmap_teal_warm,
        colorbar_label=r"$\alpha_2$",
    )
    _style_phase_map_axis(
        axes[1, 1],
        ylabel=r"First-cluster depth, $X_1$ (m)",
        ylim=(2000, 4500),
        yticks=[2000, 2500, 3000, 3500, 4000, 4500],
    )

    ax = axes[1, 2]
    depths = [2000, 2500, 3000, 3500, 4000, 4500]
    for depth, color in zip(depths, FRACTURE_COLORS[:6]):
        subset = xs.loc[xs["x1"] == depth].sort_values("spacing_m")
        ax.plot(
            subset["spacing_m"],
            subset["alpha_2"],
            marker="o",
            color=color,
            lw=1.0,
            markersize=3.2,
            label=f"X₁ = {depth} m",
        )
    add_reference_line(ax, 0.5)
    ax.set_xlabel(r"Fracture spacing, $S$ (m)")
    ax.set_ylabel(r"Relative ratio, $\alpha_2=P_2/P_1$")
    ax.set_xlim(5, 105)
    ax.set_ylim(0.25, 0.90)
    ax.set_xticks([20, 40, 60, 80, 100])
    legend_f = ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=6.5,
        handlelength=1.1,
        columnspacing=0.65,
        handletextpad=0.3,
        borderaxespad=0.25,
    )

    _freeze_equal_gutters(fig, axes)
    for axis, legend, panel in (
        (axes[0, 2], legend_c, "c"),
        (axes[1, 2], legend_f, "f"),
    ):
        if legend_overlaps_data(axis, legend):
            raise RuntimeError(f"Legend overlaps plotted data in panel {panel}")

    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes.flat),
        panel_ids=PANEL_LABELS,
        row_groups=[
            {"id": "top-row", "panels": ["a", "b", "c"]},
            {"id": "bottom-row", "panels": ["d", "e", "f"]},
        ],
        column_groups=[
            {"id": "left-column", "panels": ["a", "d"]},
            {"id": "middle-column", "panels": ["b", "e"]},
            {"id": "right-column", "panels": ["c", "f"]},
        ],
        json_out=OUTPUT_STEM.with_suffix(".panel-alignment.json"),
        overlay_svg=OUTPUT_STEM.with_suffix(".panel-alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    return fig, axes


def save_figure(fig: plt.Figure) -> None:
    """Export at the fixed manuscript size without tight-bbox rescaling."""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=400)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    plt.close(fig)
    verify_figure_exports(OUTPUT_STEM)


def main() -> None:
    sn, xs = _load_and_validate_metrics()
    figure, _ = build_figure(sn, xs)
    save_figure(figure)
    print(f"Figure 8 source rows: {len(sn)} + {len(xs)}")
    print(f"Figure 8 saved to: {OUTPUT_STEM}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
