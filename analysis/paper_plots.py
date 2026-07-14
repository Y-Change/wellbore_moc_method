# -*- coding: utf-8 -*-
"""Nature-style publication figure toolkit.

rcParams, color palette, and helpers for heatmap/contour/curve figures.
Adheres to Nature figure conventions:
- Arial sans-serif, editable SVG text (svg.fonttype='none')
- Top/right spines off, frameless legends
- 7-9pt base font for dense multi-panel composites
- Semantic palette: blue=hero, green=positive, red=baseline/contrast, neutral=support
- SVG as primary export, PNG 300dpi as raster preview
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ── Nature semantic palette ───────────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",   # hero method
    "blue_secondary": "#3775BA",   # second method
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",          # positive variant
    "red_1":   "#F6CFCB",
    "red_2":   "#E9A6A1",
    "red_strong": "#B64342",       # baseline / contrast
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
    "magenta": "#EA84DD",
}

DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral_light"],
]

# Friction-model semantic mapping (used across study2 figures)
FRICTION_COLORS = {
    "steady":   PALETTE["blue_main"],     # hero / reference
    "brunone":  PALETTE["red_strong"],    # contrast
}

CASE_COLORS = {
    "dual":   PALETTE["blue_main"],
    "triple": PALETTE["teal"],
    "quad":   PALETTE["violet"],
}

CASE_MARKERS = {"dual": "o", "triple": "s", "quad": "^"}


# ── Mandatory rcParams (editable SVG text) ────────────────
def apply_paper_rc(font_size: int = 8, axes_linewidth: float = 1.0) -> None:
    """Apply Nature-style rcParams. Call once before creating any figures.

    font_size=8 is the Nature dense multi-panel default; use 15-16 for
    compact analytic plots, 24 for large slide-sized bar panels.
    """
    plt.rcParams.update({
        # ── MANDATORY: editable SVG text ──
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans',
                            'Liberation Sans', 'SimHei', 'Microsoft YaHei'],
        'svg.fonttype': 'none',          # text stays as <text> nodes
        'pdf.fonttype': 42,              # editable TrueType in PDF
        # ── Layout & style ──
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': axes_linewidth,
        'legend.frameon': False,
        'axes.unicode_minus': False,
        'axes.grid': False,              # Nature: no grid by default
        'grid.alpha': 0.0,
        'image.cmap': 'viridis',
        'contour.negative_linestyle': 'dashed',
        'mathtext.fontset': 'dejavusans',
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ── Helpers ───────────────────────────────────────────────
def is_dark(hex_color: str, threshold: int = 128) -> bool:
    """Return True if hex color is dark (use white text on it)."""
    c = hex_color.lstrip('#')
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def add_panel_label(
    ax, label: str, x: float = -0.08, y: float = 1.02,
    fontsize: int = 10, color: str = 'black', fontweight: str = 'bold',
) -> None:
    """Place a Nature-style panel letter (a, b, c…) near top-left."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=fontsize, fontweight=fontweight, color=color,
        ha='left', va='bottom',
    )


def style_heatmap_ax(ax) -> None:
    """Heatmap axes: keep all 4 spines but thin; no ticks marks, labels stay."""
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#767676')


# ── heatmap + contour ─────────────────────────────────────
def heatmap_with_contour(
    ax,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    cmap: str = 'viridis',
    levels: Optional[Sequence[float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    cbar_label: str = '',
    cbar_fraction: float = 0.046,
    cbar_pad: float = 0.04,
    contour_color: str = 'white',
    contour_lw: float = 0.6,
    contour_alpha: float = 0.5,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    mark_best: Optional[Tuple[float, float]] = None,
    mark_color: str = 'red',
) -> 'QuadMesh':
    """pcolormesh + contour overlay on ax. Returns mesh handle."""
    Z = np.asarray(Z, dtype=float)
    if vmin is None:
        vmin = np.nanmin(Z)
    if vmax is None:
        vmax = np.nanmax(Z)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    if levels is None:
        levels = np.linspace(vmin, vmax, 7)
    mesh = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading='auto')
    if Z.shape[0] >= 2 and Z.shape[1] >= 2:
        cs = ax.contour(X, Y, Z, levels=levels, colors=contour_color,
                        linewidths=contour_lw, alpha=contour_alpha)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')
    if show_colorbar:
        cbar = ax.figure.colorbar(mesh, ax=ax, shrink=0.9,
                                  fraction=cbar_fraction, pad=cbar_pad)
        cbar.set_label(cbar_label, fontsize=plt.rcParams['axes.labelsize'])
        cbar.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'], length=0)
        cbar.outline.set_linewidth(0.5)
    if mark_best is not None:
        bx, by = mark_best
        ax.scatter([bx], [by], marker='*', s=80, c=mark_color,
                   edgecolors='white', linewidths=0.8, zorder=10)
    if title:
        ax.set_title(title, fontweight='bold', loc='left')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_heatmap_ax(ax)
    return mesh


# ── Faceted heatmap (shared colorbar) ─────────────────────
def faceted_heatmap(
    facets: Iterable[Tuple[plt.Axes, np.ndarray, np.ndarray, np.ndarray]],
    *,
    common_title: str = '',
    cbar_label: str = '',
    cmap: str = 'viridis',
    xlabel: str = '',
    ylabel: str = '',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Multi-panel shared colorbar. facets: [(ax, X, Y, Z), ...]"""
    all_z = np.concatenate([np.asarray(Z).ravel() for _, _, _, Z in facets])
    zmin = np.nanmin(all_z) if vmin is None else vmin
    zmax = np.nanmax(all_z) if vmax is None else vmax
    if zmax - zmin < 1e-12:
        zmax = zmin + 1.0
    levels = np.linspace(zmin, zmax, 7)
    last_mesh = None
    axes_list = []
    for ax, X, Y, Z in facets:
        last_mesh = heatmap_with_contour(
            ax, X, Y, Z, cmap=cmap, levels=levels,
            vmin=zmin, vmax=zmax,
            show_colorbar=False, xlabel=xlabel, ylabel=ylabel,
        )
        axes_list.append(ax)
    if last_mesh is not None:
        cbar = last_mesh.figure.colorbar(
            last_mesh, ax=axes_list, shrink=0.85, pad=0.02,
            fraction=0.046,
        )
        cbar.set_label(cbar_label, fontsize=plt.rcParams['axes.labelsize'])
        cbar.ax.tick_params(labelsize=plt.rcParams['ytick.labelsize'], length=0)
        cbar.outline.set_linewidth(0.5)
    if common_title:
        last_mesh.figure.suptitle(common_title, fontweight='bold')


# ── Dual-axis curve ───────────────────────────────────────
def dual_axis_curve(
    ax,
    x: np.ndarray,
    y1: np.ndarray,
    y2: Optional[np.ndarray] = None,
    *,
    label1: str = '',
    label2: str = '',
    color1: str = PALETTE['blue_main'],
    color2: str = PALETTE['red_strong'],
    marker1: str = 'o',
    marker2: str = 's',
    xlabel: str = '',
    ylabel1: str = '',
    ylabel2: str = '',
    title: str = '',
    lw: float = 1.6,
    ms: float = 4,
) -> None:
    """Left axis y1; optional right axis y2. Nature-style frameless legend."""
    ax.plot(x, y1, color=color1, marker=marker1, lw=lw, ms=ms, label=label1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1, color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    if y2 is not None:
        ax2 = ax.twinx()
        ax2.spines['right'].set_visible(True)   # re-enable right spine for twin
        ax2.plot(x, y2, color=color2, marker=marker2, lw=lw, ms=ms, label=label2)
        ax2.set_ylabel(ylabel2, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
    if title:
        ax.set_title(title, fontweight='bold', loc='left')


# ── Export: SVG primary, PNG preview ──────────────────────
def save_figure(
    fig,
    path: str,
    *,
    dpi: int = 300,
    also_svg: bool = True,
    also_pdf: bool = False,
    pad: float = 1.5,
) -> None:
    """Save figure as PNG (path) + optional SVG/PDF with same base name.

    SVG is primary (editable text); PNG is raster preview.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.tight_layout(pad=pad)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if also_svg:
        svg_path = os.path.splitext(path)[0] + '.svg'
        fig.savefig(svg_path, bbox_inches='tight')
    if also_pdf:
        pdf_path = os.path.splitext(path)[0] + '.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
