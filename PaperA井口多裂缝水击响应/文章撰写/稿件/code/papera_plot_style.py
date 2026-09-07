"""Shared publication style utilities extracted from the Paper A plotting scripts.

Copy this file next to a Paper A figure script and import the helpers locally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PALETTE = {
    "blue_main": "#1B4F72",
    "teal": "#117864",
    "orange": "#D35400",
    "red_strong": "#900C3F",
    "purple": "#6C3483",
    "blue_light": "#2874A6",
    "gold": "#B7950B",
    "slate": "#2C3E50",
    "grey_ref": "#BDC3C7",
}
FRACTURE_COLORS = [
    PALETTE["blue_main"], PALETTE["teal"], PALETTE["orange"],
    PALETTE["red_strong"], PALETTE["purple"], PALETTE["blue_light"],
    PALETTE["gold"], "#5D6D7E",
]
NATURE_BLUE_WARM = ["#EBF5FB", "#AED6F1", "#5DADE2", "#F5B041", "#E74C3C"]
NATURE_TEAL_WARM = ["#E8F8F5", "#A3E4D7", "#48C9B0", "#E67E22", "#C0392B"]


def configure_papera_style() -> None:
    """Apply the Paper A SPE/MSSP publication defaults before creating figures."""
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
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.75,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.8,
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


def style_axis(ax: plt.Axes, panel_label: str | None = None, *, box_aspect: float | None = 4 / 5) -> plt.Axes:
    """Apply borders, inward ticks, optional 4:5 panel aspect and panel label."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.75)
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    if panel_label:
        ax.set_title(panel_label, loc="left", fontsize=9.5, fontweight="bold", pad=5)
    return ax


def make_panels(nrows: int = 1, ncols: int = 3, *, figsize: tuple[float, float] | None = None,
                labels: Sequence[str] | None = None, sharex: bool | str = False,
                sharey: bool | str = False, box_aspect: float | None = 4 / 5):
    """Create and style a conventional Paper A panel grid."""
    if figsize is None:
        figsize = (7.20 if ncols <= 3 else 7.35, 2.35 if nrows == 1 else 5.50)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True,
                             sharex=sharex, sharey=sharey, squeeze=False)
    flat_axes = list(axes.flat)
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(flat_axes))]
    for ax, label in zip(flat_axes, labels):
        style_axis(ax, label, box_aspect=box_aspect)
    return fig, np.asarray(flat_axes, dtype=object)


def make_papera_colormap(name: str = "nature_blue_warm", colors: Iterable[str] | None = None):
    """Build one of the project colour maps for continuous response surfaces."""
    if colors is None:
        colors = NATURE_BLUE_WARM if name == "nature_blue_warm" else NATURE_TEAL_WARM
    return LinearSegmentedColormap.from_list(name, list(colors))


def format_colorbar(cbar, label: str, *, fontsize: float = 7.0) -> None:
    """Format a vertical Paper A colourbar placed next to its panel."""
    cbar.set_label(label, fontsize=fontsize, labelpad=4)
    cbar.ax.tick_params(labelsize=6.0, direction="in", length=2.0, width=0.6)


def add_reference_line(ax: plt.Axes, value: float, *, orientation: str = "h", label: str | None = None, **kwargs):
    """Add a subdued baseline/reference line beneath the scientific data."""
    style = {"color": PALETTE["grey_ref"], "ls": ":", "lw": 0.6, "alpha": 0.7, "zorder": 0}
    style.update(kwargs)
    return ax.axhline(value, label=label, **style) if orientation == "h" else ax.axvline(value, label=label, **style)


def legend_overlaps_data(ax: plt.Axes, legend=None) -> bool:
    """Return True when the legend overlaps an axis line or collection.

    This is a warning mechanism. It cannot judge all artwork (e.g. annotations), so
    visually inspect dense panels after resolving any reported collision.
    """
    legend = legend or ax.get_legend()
    if legend is None:
        return False
    fig = ax.figure
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    artists = [*ax.lines, *ax.collections]
    for artist in artists:
        if not artist.get_visible() or artist.get_zorder() <= 0:
            continue
        try:
            if legend_box.overlaps(artist.get_window_extent(fig.canvas.get_renderer())):
                return True
        except (AttributeError, ValueError):
            continue
    return False


def save_papera_figure(fig: plt.Figure, output_stem: str | Path, *, dpi: int = 400, close: bool = True) -> dict[str, Path]:
    """Write the required PNG, editable-text SVG and publication-ready PDF outputs."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {suffix: stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf")}
    fig.savefig(outputs[".png"], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[".svg"], bbox_inches="tight")
    fig.savefig(outputs[".pdf"], bbox_inches="tight")
    if close:
        plt.close(fig)
    return outputs


def verify_figure_exports(output_stem: str | Path) -> dict[str, Path]:
    """Raise FileNotFoundError unless all three mandatory Paper A deliverables exist."""
    stem = Path(output_stem)
    outputs = {suffix: stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf")}
    missing = [str(path) for path in outputs.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty Paper A figure export(s): {', '.join(missing)}")
    return outputs
