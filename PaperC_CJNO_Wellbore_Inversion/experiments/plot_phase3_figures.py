# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.plot_phase3_figures

Publication-grade scientific visualization script for PaperC Phase 3:
面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究

Generates 5 comprehensive composite figures (PNG 300 DPI + vector SVG, 10 files total):
1. fig1_layer_stripping_mechanism: Acoustic wave propagation, multi-path reverberation choke,
   layer-stripping pulse restoration, explicit Gamma_j and Y_{b,j} extraction.
2. fig2_tg_dis_architecture: TG-DIS-DeepONet dual-track architecture, acoustic delay-bias matrix,
   and attention heatmap attribution aligning with acoustic travel time.
3. fig3_benchmark_and_ablation: 5-model benchmark comparison bar charts, dense multi-cluster
   true vs pred scatter plot, and 4-step ablation progression.
4. fig4_noise_and_speed_robustness: 30dB, 20dB, 10dB AWGN/Pink noise degradation curves,
   pm 1% sound speed perturbation spatial error profiles.
5. fig5_typical_cases_inversion: Inversion profiles across 5 representative fracturing cases
   (uniform, heel-dominant, saddle, toe-dominant, sand screen-out dead cluster).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.deeponet import VanillaDeepONet

# ==============================================================================
# Nature-Style Plotting Configuration
# ==============================================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",       # Editable text in SVG
    "pdf.fonttype": 42,           # Editable TrueType fonts in PDF
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 10.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 5.0,
})

# Professional Semantic Color Palette
PALETTE = {
    "navy": "#0F4D92",       # Primary Deep Navy
    "blue": "#1f77b4",       # Standard Acoustic Blue
    "crimson": "#B64342",    # Crimson Red for Non-linear / Choked / Friction
    "coral": "#E6550D",      # Vibrant Orange / Alert
    "green": "#2CA02C",      # Proposed Model / Restored / Success
    "teal": "#1B9E77",       # Dark Teal for Inversion
    "purple": "#7570B3",     # Purple for Baseline Transformers
    "gray": "#636363",       # Neutral Dark Gray
    "light_gray": "#E0E0E0", # Background Grids / Dividers
    "amber": "#D95F02",      # Secondary Highlight
}

OUTPUT_DIR = _ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR = _ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights"


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save both 300 DPI PNG and vector SVG formats."""
    png_path = OUTPUT_DIR / f"{name}.png"
    svg_path = OUTPUT_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"[Exported] {png_path.name} (300 DPI) & {svg_path.name}")
    plt.close(fig)


# ==============================================================================
# Figure 1: Layer-Stripping Acoustic Mechanism & Pulse Restoration
# ==============================================================================
def plot_fig1_layer_stripping_mechanism() -> None:
    """
    Figure 1: Physical Mechanism of 1D Acoustic Layer Stripping
    - Panel a: Wellbore acoustic wave propagation and fracture transmission choking schematic.
    - Panel b: Incident vs choked vs layer-stripping restored pulse signals.
    - Panel c: Extracted explicit reflection coefficients Gamma_j and branch admittance Y_{b,j}.
    - Panel d: Schur / Bruckstein recursive lattice de-choking signal flow.
    """
    fig = plt.figure(figsize=(10.5, 8.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # --------------------------------------------------------------------------
    # Panel a: Wellbore & Fracture Reverberation Choking Schematic
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("(a) Acoustic Wave Propagation & Multi-Cluster Transmission Choking", fontweight="bold", loc="left")
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 55)
    ax_a.axis("off")

    # Draw wellhead valve & casing
    casing_y = 30
    ax_a.plot([5, 95], [casing_y, casing_y], color="#333333", lw=4, solid_capstyle="round")
    ax_a.plot([5, 95], [casing_y - 8, casing_y - 8], color="#333333", lw=4, solid_capstyle="round")
    ax_a.fill_between([5, 95], [casing_y - 8, casing_y - 8], [casing_y, casing_y], color="#F0F4F8")

    # Wellhead valve icon
    ax_a.plot([5, 5], [casing_y - 12, casing_y + 4], color=PALETTE["navy"], lw=3)
    ax_a.text(5, casing_y + 7, "Wellhead Valve\n($t_c$ Closure)", color=PALETTE["navy"], ha="center", fontsize=7.5, fontweight="bold")

    # Draw 5 fracture branches (heel to toe: x_1 to x_5)
    frac_x = [30, 44, 58, 72, 86]
    frac_labels = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
    alphas_demo = [0.35, 0.25, 0.18, 0.12, 0.10]

    for j, (fx, fl, al) in enumerate(zip(frac_x, frac_labels, alphas_demo)):
        # Fracture slit
        ax_a.plot([fx, fx], [casing_y - 8, casing_y - 20], color=PALETTE["navy"], lw=2.5)
        ax_a.plot([fx - 3, fx + 3], [casing_y - 20, casing_y - 20], color=PALETTE["navy"], lw=2.5)
        # Flow arrow into fracture
        ax_a.annotate("", xy=(fx, casing_y - 17), xytext=(fx, casing_y - 9),
                     arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.5))
        ax_a.text(fx, casing_y - 23, f"{fl}\n$\\alpha_{j+1}={al:.2f}$", color="#222222", ha="center", fontsize=7)

    # Downward wave P+
    ax_a.annotate("", xy=(24, casing_y - 4), xytext=(10, casing_y - 4),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=2.0))
    ax_a.text(17, casing_y - 1, "Incident Wave $P^+$\n($\\Delta H = -Z_0 \\Delta Q$)", color=PALETTE["blue"], ha="center", fontsize=7)

    # Reflected waves P-
    for j, fx in enumerate(frac_x):
        ax_a.annotate("", xy=(fx - 5, casing_y - 4), xytext=(fx - 1, casing_y - 4),
                     arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.2))

    # Inter-cluster reverberations
    arc1 = patches.Arc((37, casing_y - 4), 10, 6, theta1=0, theta2=180, color=PALETTE["amber"], lw=1.2, ls="--")
    ax_a.add_patch(arc1)
    ax_a.text(37, casing_y + 2, "Internal Multiples\n(Reverberation)", color=PALETTE["amber"], ha="center", fontsize=6.5)

    # Transmission choking formula banner
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="#FFF8E7", edgecolor="#E2B14C", lw=1.0)
    ax_a.text(50, 48, r"Cumulative Choke: $\mathcal{T}_{1:j-1} = \prod_{k=1}^{j-1} (1 + \Gamma_k)^2 \ll 1$",
              ha="center", va="center", fontsize=8.2, fontweight="bold", bbox=bbox_props)

    # --------------------------------------------------------------------------
    # Panel b: Choked vs. Layer-Stripping Restored Pulses
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("(b) Multi-Cluster Echo Choking & Layer-Stripping Restoration", fontweight="bold", loc="left")

    t = np.linspace(0, 10, 1000)  # Time axis (s)
    # Simulated arrival times for 5 clusters
    tau = np.array([6.2, 6.4, 6.6, 6.8, 7.0])
    gamma_true = np.array([-0.18, -0.15, -0.12, -0.10, -0.08])

    # Upstream transmission loss: prod_{k=1}^{j-1} (1 + gamma_k)^2
    t_loss = np.ones(len(gamma_true))
    for j in range(1, len(gamma_true)):
        t_loss[j] = t_loss[j - 1] * ((1.0 + gamma_true[j - 1]) ** 2)

    # Construct synthetic choked wavefield
    pulse_width = 0.04
    wave_choked = np.zeros_like(t)
    wave_restored = np.zeros_like(t)

    for j in range(len(tau)):
        amp_choked = gamma_true[j] * t_loss[j]
        amp_restored = gamma_true[j]
        pulse = -np.exp(-((t - tau[j]) ** 2) / (2 * pulse_width ** 2))
        wave_choked += amp_choked * pulse
        wave_restored += amp_restored * pulse

    # Add realistic noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.003, size=len(t))
    wave_choked += noise
    wave_restored += noise

    ax_b.plot(t, wave_choked - 0.25, color=PALETTE["crimson"], lw=1.3, label="Observed Choked Wave (Wellhead)")
    ax_b.plot(t, wave_restored, color=PALETTE["green"], lw=1.4, label="DIS-Op Restored Intrinsic Pulses")

    for j, tj in enumerate(tau):
        ax_b.axvline(tj, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax_b.text(tj, 0.22, f"$\\tau_{j+1}$", ha="center", fontsize=7.5, color=PALETTE["navy"])
        # Annotate severe choking
        if j == 4:
            ax_b.annotate(f"Choked to\n{t_loss[j]*100:.1f}%", xy=(tj, wave_choked[np.argmin(np.abs(t - tj))] - 0.25),
                          xytext=(tj + 0.35, -0.38),
                          arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.0),
                          fontsize=7, color=PALETTE["crimson"], fontweight="bold")
            ax_b.annotate("Recovered to\n100% Intrinsic", xy=(tj, wave_restored[np.argmin(np.abs(t - tj))]),
                          xytext=(tj + 0.35, 0.05),
                          arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.0),
                          fontsize=7, color=PALETTE["green"], fontweight="bold")

    ax_b.set_xlim(5.8, 8.2)
    ax_b.set_ylim(-0.55, 0.35)
    ax_b.set_xlabel("Arrival Time $t$ (s)")
    ax_b.set_ylabel("Normalized Pressure Pulse Amplitude")
    ax_b.legend(loc="lower left", fontsize=7.5)
    ax_b.grid(True, ls="--", lw=0.5, alpha=0.5)

    # --------------------------------------------------------------------------
    # Panel c: Extracted Reflection Coefficient Gamma_j & Branch Admittance Y_bj
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title(r"(c) Inverted Reflection $\Gamma_j$ & Physical Admittance $Y_{b,j}$", fontweight="bold", loc="left")

    clusters = np.arange(1, 6)
    gamma_inv = np.array([-0.076, -0.086, -0.099, -0.117, -0.150])
    # Y0 = g*A/a = 9.80665 * 0.015328 / 1450 = 1.03655e-4
    y0 = 1.03655e-4
    admittance = - (2.0 * y0 * gamma_inv) / (1.0 + gamma_inv) * 1e4  # in 1e-4 m^2/s

    width = 0.35
    bars1 = ax_c.bar(clusters - width/2, gamma_inv, width=width, color=PALETTE["navy"], alpha=0.85, label=r"Reflection $\Gamma_j \in [-1, 0]$")
    ax_c.set_ylabel(r"Acoustic Reflection Coefficient $\Gamma_j$", color=PALETTE["navy"], fontweight="bold")
    ax_c.tick_params(axis="y", labelcolor=PALETTE["navy"])
    ax_c.set_ylim(-0.20, 0.0)

    # Twin axis for admittance
    ax_c_twin = ax_c.twinx()
    ax_c_twin.spines["top"].set_visible(False)
    bars2 = ax_c_twin.bar(clusters + width/2, admittance, width=width, color=PALETTE["teal"], alpha=0.85, label=r"Admittance $Y_{b,j}$")
    ax_c_twin.set_ylabel(r"Branch Admittance $Y_{b,j}$ ($10^{-4}\,\mathrm{m^2/s}$)", color=PALETTE["teal"], fontweight="bold")
    ax_c_twin.tick_params(axis="y", labelcolor=PALETTE["teal"])
    ax_c_twin.set_ylim(0.0, 0.45)

    ax_c.set_xticks(clusters)
    ax_c.set_xticklabels([f"Cluster {k}\n($x_{k}$)" for k in clusters])
    ax_c.set_xlabel("Horizontal Fracture Stage Cluster Index")

    # Add numeric labels
    for bar in bars1:
        yval = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width()/2.0, yval - 0.015, f"{yval:.3f}", ha="center", va="top", fontsize=6.8, color=PALETTE["navy"])
    for bar in bars2:
        yval = bar.get_height()
        ax_c_twin.text(bar.get_x() + bar.get_width()/2.0, yval + 0.012, f"{yval:.2f}", ha="center", va="bottom", fontsize=6.8, color=PALETTE["teal"])

    # Combined legend
    lines, labels = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c_twin.get_legend_handles_labels()
    ax_c.legend(lines + lines2, labels + labels2, loc="lower left", fontsize=7.2)
    ax_c.grid(True, ls="--", lw=0.5, alpha=0.5, axis="y")

    # --------------------------------------------------------------------------
    # Panel d: Schur / Bruckstein Recursive Lattice De-choking Flow
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("(d) Schur/Bruckstein Causal Layer-Stripping Recursion", fontweight="bold", loc="left")
    ax_d.axis("off")
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0, 100)

    # Draw recursion flow steps
    steps = [
        ("Step 1: Causal Arrival Windowing", r"$\tau_j = t_s + \frac{2x_j}{a} \rightarrow \mathbf{r}_{j-1}(\tau_j)$", 86, PALETTE["navy"]),
        ("Step 2: Apparent Reflection Extract", r"$\gamma_{raw,j} = \mathrm{PeakExtract}(\mathbf{r}_{j-1}, \tau_j)$", 68, PALETTE["crimson"]),
        ("Step 3: Upstream Choke Inversion", r"$\Gamma_j = \mathrm{clamp}\left(\frac{\gamma_{raw,j}}{\mathcal{T}_{cum}}, -0.95, 0\right)$", 50, PALETTE["green"]),
        ("Step 4: Explicit Admittance Mapping", r"$Y_{b,j} = -\frac{2 Y_0 \Gamma_j}{1 + \Gamma_j}$", 32, PALETTE["teal"]),
        ("Step 5: Echo Subtraction & Update", r"$\mathbf{r}_j(t) = \frac{\mathbf{r}_{j-1}(t) - \text{Echo}(\Gamma_j)}{(1+\Gamma_j)^2}, \; \mathcal{T}_{cum} \leftarrow \mathcal{T}_{cum}(1+\Gamma_j)^2$", 12, PALETTE["amber"]),
    ]

    for title, formula, y_pos, color in steps:
        box = patches.FancyBboxPatch((8, y_pos - 6), 84, 12, boxstyle="round,pad=0.8",
                                     facecolor="#FAFAFA", edgecolor=color, lw=1.3)
        ax_d.add_patch(box)
        ax_d.text(12, y_pos + 1.8, title, fontsize=7.5, fontweight="bold", color=color, va="center")
        ax_d.text(50, y_pos - 2.5, formula, fontsize=7.2, color="#222222", ha="center", va="center")
        if y_pos > 20:
            ax_d.annotate("", xy=(50, y_pos - 7.5), xytext=(50, y_pos - 5.5),
                          arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    save_figure(fig, "fig1_layer_stripping_mechanism")


# ==============================================================================
# Figure 2: TG-DIS-DeepONet Architecture & Explainable Attention
# ==============================================================================
def plot_fig2_tg_dis_architecture() -> None:
    """
    Figure 2: Explainable TG-DIS-DeepONet Neural Operator Architecture
    - Panel a: End-to-end dual-track network architecture diagram.
    - Panel b: Acoustic delay-bias matrix B_{ij}^{acoustic} vs cluster distance.
    - Panel c: Empirical attention heatmap A_{ij} showing travel-time attribution.
    - Panel d: Time-gated waveform patches aligned with acoustic arrival times tau_j.
    """
    fig = plt.figure(figsize=(10.5, 8.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # --------------------------------------------------------------------------
    # Panel a: TG-DIS-DeepONet Architecture Schematic
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("(a) TG-DIS-DeepONet Dual-Track Architecture", fontweight="bold", loc="left")
    ax_a.axis("off")
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)

    # Architecture components boxes
    boxes = [
        ("Input Waveforms & Operators", "Wellhead $H(t)$, $\\partial_t H(t)$, Spatial Cepstrum $C(x)$,\nConditions $[t_c, a, H_{ext}]$, Stage Coords $x_j$", 88, 14, "#EBF3FB", PALETTE["navy"]),
        ("1. Relative Time-Gating", "Arrival $\\tau_j = t_s + 2x_j/a \\to$ Window $[\\tau_j - 50\\mathrm{ms}, \\tau_j + 250\\mathrm{ms}]$\n128-pt Waveform Tokens $\\mathbf{w}_j \\in \\mathbb{R}^{M \\times d}$", 68, 13, "#E8F5E9", PALETTE["green"]),
        ("2. Differentiable Layer-Stripping (DIS-Op)", "Heel-to-toe causal recursion: De-chokes $\\prod(1+\\Gamma_k)^2$\nExplicit outputs: $\\Gamma_j \\in [-1, 0]$ & $Y_{b,j} = -2Y_0\\Gamma_j/(1+\\Gamma_j)$", 50, 13, "#FFF3E0", PALETTE["amber"]),
        ("3. Acoustic Delay-Bias Transformer", "Attention: $\\mathbf{A}_{ij} = \\mathrm{Softmax}(\\frac{\\mathbf{q}_i \\mathbf{k}_j^T}{\\sqrt{d}} - \\gamma \\frac{|x_i - x_j|}{a})$\nSuppresses non-causal long-range false correlations", 32, 13, "#F3E5F5", PALETTE["purple"]),
        ("4. Dual-Track Decoupled Decoding Heads", "Track 1 (Discrete): $\\hat{\\alpha}_j \\in \\Delta^{N_c-1}, \\hat{p}_{exist}, \\Delta\\hat{x}_j, \\hat{C}_{f,j}$\nTrack 2 (Continuous): Field $\\hat{m}_\\alpha(x)$ via Trunk + Voronoi pooling", 13, 14, "#E0F2F1", PALETTE["teal"]),
    ]

    for name, desc, y_pos, h, face_col, edge_col in boxes:
        box = patches.FancyBboxPatch((5, y_pos - h/2), 90, h, boxstyle="round,pad=0.8",
                                     facecolor=face_col, edgecolor=edge_col, lw=1.4)
        ax_a.add_patch(box)
        ax_a.text(8, y_pos + h/4, name, fontsize=7.6, fontweight="bold", color=edge_col, va="center")
        ax_a.text(8, y_pos - h/4, desc, fontsize=6.8, color="#222222", va="center")
        if y_pos > 20:
            ax_a.annotate("", xy=(50, y_pos - h/2 - 2), xytext=(50, y_pos - h/2),
                          arrowprops=dict(arrowstyle="->", color=edge_col, lw=1.2))

    # --------------------------------------------------------------------------
    # Panel b: Acoustic Delay-Bias Matrix Heatmap & Spatial Decay
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title(r"(b) Acoustic Travel-Time Delay Bias $\mathbf{B}_{ij}^{acoustic}$", fontweight="bold", loc="left")

    positions = np.array([4400, 4415, 4435, 4460, 4500, 4560])  # Non-uniform spacing
    a_ref = 1450.0
    gamma_attn = 10.0  # Softplus learned decay rate

    dist_matrix = np.abs(positions[:, None] - positions[None, :])
    dt_matrix = dist_matrix / a_ref
    bias_matrix = -gamma_attn * dt_matrix

    im = ax_b.imshow(bias_matrix, cmap="Blues_r", vmin=-8.0, vmax=0.0)
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label(r"Delay Penalty $\mathbf{B}_{ij} = -\gamma \frac{|x_i - x_j|}{a}$", fontsize=7.5)

    ax_b.set_xticks(range(6))
    ax_b.set_yticks(range(6))
    ax_b.set_xticklabels([f"$x_{k+1}$" for k in range(6)])
    ax_b.set_yticklabels([f"$x_{k+1}$" for k in range(6)])
    ax_b.set_xlabel("Key Cluster Index $j$")
    ax_b.set_ylabel("Query Cluster Index $i$")

    for i in range(6):
        for j in range(6):
            val = bias_matrix[i, j]
            color = "white" if val < -3.5 else "black"
            ax_b.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.0, color=color)

    # --------------------------------------------------------------------------
    # Panel c: Empirical Self-Attention Heatmap from Trained Model
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title(r"(c) Empirical Self-Attention Attribution $\mathbf{A}_{ij}$", fontweight="bold", loc="left")

    # Load actual attention weights from trained model for a test sample
    ds = PilotInversionDataset(split="test")
    model = TGDISDeepONet()
    ckpt = torch.load(WEIGHTS_DIR / "tg_dis_deeponet_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Find a representative 5-cluster sample
    sample_idx = 18
    item = ds[sample_idx]
    batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out = model(batch=batch)
        attn = out["attn_weights"][0].mean(dim=(0, 1)).cpu().numpy()  # Average over layers & heads: (6, 6)

    im_c = ax_c.imshow(attn[:5, :5], cmap="YlGnBu", vmin=0.0, vmax=0.55)
    cbar_c = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Mean Attention Weight", fontsize=7.5)

    ax_c.set_xticks(range(5))
    ax_c.set_yticks(range(5))
    ax_c.set_xticklabels([f"Clust {k+1}\n({item['positions'][k]:.0f}m)" for k in range(5)], fontsize=6.8)
    ax_c.set_yticklabels([f"Clust {k+1}" for k in range(5)], fontsize=7.2)
    ax_c.set_xlabel("Attended Key Cluster $j$")
    ax_c.set_ylabel("Query Fracture Cluster $i$")

    for i in range(5):
        for j in range(5):
            val = attn[i, j]
            color = "white" if val > 0.35 else "black"
            ax_c.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color=color)

    # --------------------------------------------------------------------------
    # Panel d: Time-Gated Waveform Patches Aligned with Arrival Times tau_j
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title(r"(d) Arrival-Time Aligned Waveform Patches ($\tau_j \pm \Delta t$)", fontweight="bold", loc="left")

    # Time-gated patches from item
    wave = item["wave"].numpy()[0]  # Raw wellhead head
    dt = 60.0 / 4096.0
    t_full = np.arange(4096) * dt
    pos = item["positions"].numpy()[:5]
    a_val = item["cond"].numpy()[1]
    ts_val = item["cond"].numpy()[0]

    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["crimson"]]
    t_patch = np.linspace(-50, 250, 128)  # ms relative to tau_j

    for j in range(5):
        tau_j = ts_val + 2.0 * pos[j] / a_val
        idx_tau = int(round(tau_j / dt))
        idx_start = idx_tau - 3
        idx_end = idx_start + 128
        patch = wave[idx_start:idx_end] if idx_end <= len(wave) else np.zeros(128)
        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-6)

        ax_d.plot(t_patch, patch - j * 2.2, color=colors[j], lw=1.2,
                  label=f"Cluster {j+1} ($x={pos[j]:.0f}\\mathrm{{m}}, \\tau={tau_j:.2f}\\mathrm{{s}}$)")

    ax_d.axvline(0, color="red", ls="--", lw=1.0, alpha=0.8)
    ax_d.text(3, 1.2, "Echo Arrival Arrival $\\tau_j$", color="red", fontsize=7.2, fontweight="bold")
    ax_d.set_xlabel(r"Relative Delay Window $t - \tau_j$ (ms)")
    ax_d.set_ylabel("Normalized Local Pressure Pulse")
    ax_d.set_xlim(-50, 250)
    ax_d.set_ylim(-10.5, 2.5)
    ax_d.legend(loc="lower right", fontsize=6.8)
    ax_d.grid(True, ls="--", lw=0.5, alpha=0.5)

    save_figure(fig, "fig2_tg_dis_architecture")


# ==============================================================================
# Figure 3: Benchmark & Ablation Study Performance Comparison
# ==============================================================================
def plot_fig3_benchmark_and_ablation() -> None:
    """
    Figure 3: Comprehensive 5-Model Benchmark & Ablation Ladder
    - Panel a: 5-model benchmark comparison bar charts.
    - Panel b: 4-step ablation progression (Dense R^2 and W_1 distance).
    - Panel c: Dense multi-cluster True vs Predicted Parity Scatter Plot.
    - Panel d: Performance breakdown by cluster count Nc.
    """
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # Load metrics
    bench_file = _ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "phase3_benchmark_metrics.json"
    ablation_file = _ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "phase3_ablation_metrics.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(ablation_file, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)

    # --------------------------------------------------------------------------
    # Panel a: 5-Model Benchmark Bar Charts
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title(r"(a) 5-Model Benchmark Comparison (100 Test Cases)", fontweight="bold", loc="left")

    model_keys = ["resnet", "fno", "deeponet", "tg_deeponet", "tg_dis_deeponet"]
    labels = ["1D-ResNet", "1D-FNO", "Vanilla DeepONet", "TG-DeepONet", "TG-DIS-DeepONet\n(Proposed)"]

    dense_r2 = [bench_data[m]["alpha_r2_dense"] for m in model_keys]
    overall_r2 = [bench_data[m]["alpha_r2"] for m in model_keys]
    mae = [bench_data[m]["alpha_mae"] for m in model_keys]
    w1 = [bench_data[m]["w1_mean_m"] for m in model_keys]

    x = np.arange(len(model_keys))
    w = 0.22

    b1 = ax_a.bar(x - 1.5*w, overall_r2, width=w, color=PALETTE["blue"], label=r"Overall $R^2$")
    b2 = ax_a.bar(x - 0.5*w, dense_r2, width=w, color=PALETTE["green"], label=r"Dense $R^2$ ($N_c \geq 4$)")
    b3 = ax_a.bar(x + 0.5*w, mae, width=w, color=PALETTE["crimson"], label=r"$\alpha$ MAE")
    b4 = ax_a.bar(x + 1.5*w, np.array(w1)/20.0, width=w, color=PALETTE["purple"], label=r"$W_1 / 20\,\mathrm{m}$")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=7.0)
    ax_a.set_ylabel("Metric Score / Normalized Value")
    ax_a.set_ylim(-0.1, 0.75)
    ax_a.axhline(0, color="gray", lw=0.8)
    ax_a.legend(loc="upper left", fontsize=7.0, ncol=2)
    ax_a.grid(True, ls="--", lw=0.5, alpha=0.5, axis="y")

    # Annotate key leap
    ax_a.annotate(f"Dense $R^2$ Leap:\n-0.0294 $\\to$ +0.1825",
                  xy=(4 - 0.5*w, dense_r2[4]), xytext=(2.2, 0.38),
                  arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=1.2),
                  fontsize=7.2, color=PALETTE["green"], fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8F5E9", edgecolor=PALETTE["green"], lw=0.8))

    # --------------------------------------------------------------------------
    # Panel b: 4-Step Ablation Ladder Progression
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("(b) 4-Step Systematic Ablation Ladder", fontweight="bold", loc="left")

    ablation_steps = [
        "Vanilla DeepONet\n(No Gating, No Bias)",
        "TG-DeepONet\n(Relative Win, No Bias)",
        "TG-DeepONet\n(Win + Acoustic Bias)",
        "TG-DIS-DeepONet\n(DIS-Op + Bias + Win)"
    ]
    abl_keys = ["deeponet", "tg_best_nobias", "tg_relative_bias", "tg_dis_deeponet"]

    dense_r2_abl = [ablation_data[k]["alpha_r2_dense"] for k in abl_keys]
    w1_abl = [ablation_data[k]["w1_mean_m"] for k in abl_keys]

    x_abl = np.arange(len(ablation_steps))
    ax_b.plot(x_abl, dense_r2_abl, marker="o", lw=1.8, color=PALETTE["green"], label=r"Dense $R^2$ ($N_c \geq 4$)")

    for i, v in enumerate(dense_r2_abl):
        ax_b.text(i, v + 0.015, f"{v:+.4f}", ha="center", fontsize=7.2, color=PALETTE["green"], fontweight="bold")

    ax_b.set_ylabel(r"Dense Multi-Cluster $R^2$", color=PALETTE["green"], fontweight="bold")
    ax_b.tick_params(axis="y", labelcolor=PALETTE["green"])
    ax_b.set_ylim(-0.08, 0.25)
    ax_b.axhline(0, color="gray", ls="--", lw=0.8)

    ax_b_twin = ax_b.twinx()
    ax_b_twin.spines["top"].set_visible(False)
    ax_b_twin.plot(x_abl, w1_abl, marker="s", lw=1.8, ls="--", color=PALETTE["navy"], label=r"Wasserstein $W_1$ (m)")

    for i, v in enumerate(w1_abl):
        ax_b_twin.text(i, v + 0.15, f"{v:.2f}m", ha="center", fontsize=7.2, color=PALETTE["navy"], fontweight="bold")

    ax_b_twin.set_ylabel(r"1D Spatial $W_1$ Distance (m)", color=PALETTE["navy"], fontweight="bold")
    ax_b_twin.tick_params(axis="y", labelcolor=PALETTE["navy"])
    ax_b_twin.set_ylim(7.5, 10.5)

    ax_b.set_xticks(x_abl)
    ax_b.set_xticklabels(ablation_steps, fontsize=6.8)
    ax_b.grid(True, ls="--", lw=0.5, alpha=0.5)

    # --------------------------------------------------------------------------
    # Panel c: Dense Multi-Cluster True vs Pred Parity Scatter Plot
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title(r"(c) Parity Plot: Dense Clusters ($N_c \geq 4$, 47 Test Cases)", fontweight="bold", loc="left")

    # Run inference on test dataset to extract true vs pred alpha for Vanilla DeepONet and TG-DIS-DeepONet
    ds = PilotInversionDataset(split="test")
    model_dis = TGDISDeepONet()
    model_vanilla = VanillaDeepONet()
    ckpt_dis = torch.load(WEIGHTS_DIR / "tg_dis_deeponet_best.pt", map_location="cpu")
    ckpt_vanilla = torch.load(WEIGHTS_DIR / "deeponet_best.pt", map_location="cpu")
    model_dis.load_state_dict(ckpt_dis["model_state_dict"])
    model_vanilla.load_state_dict(ckpt_vanilla["model_state_dict"])
    model_dis.eval()
    model_vanilla.eval()

    true_alphas_dense = []
    pred_alphas_dis = []
    pred_alphas_vanilla = []

    for i in range(len(ds)):
        item = ds[i]
        mask_np = item["mask"].bool().numpy()
        nc = np.sum(mask_np)
        if nc >= 4:
            batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                out_dis = model_dis(batch=batch)
                out_vanilla = model_vanilla(batch=batch)
            true_a = item["alpha"].numpy()[mask_np]
            pred_d = out_dis["alpha"].squeeze(0).numpy()[mask_np]
            pred_v = out_vanilla["alpha"].squeeze(0).numpy()[mask_np]

            true_alphas_dense.extend(true_a)
            pred_alphas_dis.extend(pred_d)
            pred_alphas_vanilla.extend(pred_v)

    true_arr = np.array(true_alphas_dense)
    dis_arr = np.array(pred_alphas_dis)
    van_arr = np.array(pred_alphas_vanilla)

    ax_c.scatter(true_arr, van_arr, alpha=0.45, color=PALETTE["gray"], s=18, label="Vanilla DeepONet (Equalization Trap)")
    ax_c.scatter(true_arr, dis_arr, alpha=0.75, color=PALETTE["green"], s=22, edgecolors="white", lw=0.3, label="TG-DIS-DeepONet (Proposed)")

    # 1:1 parity line
    ax_c.plot([0, 0.9], [0, 0.9], "k--", lw=1.2, label="Ideal Parity ($y=x$)")

    # Linear trend lines
    p_van = np.polyfit(true_arr, van_arr, 1)
    p_dis = np.polyfit(true_arr, dis_arr, 1)
    x_fit = np.linspace(0.01, 0.85, 100)
    ax_c.plot(x_fit, np.polyval(p_van, x_fit), color=PALETTE["gray"], lw=1.2, ls=":")
    ax_c.plot(x_fit, np.polyval(p_dis, x_fit), color=PALETTE["green"], lw=1.5)

    ax_c.set_xlim(0, 0.85)
    ax_c.set_ylim(0, 0.85)
    ax_c.set_xlabel(r"Ground Truth Fluid Intake Fraction $\alpha_{true}$")
    ax_c.set_ylabel(r"Predicted Fluid Intake Fraction $\hat{\alpha}_{pred}$")
    ax_c.legend(loc="upper left", fontsize=6.8)
    ax_c.grid(True, ls="--", lw=0.5, alpha=0.5)

    # --------------------------------------------------------------------------
    # Panel d: Breakdown Across Fracture Cluster Count Nc in [1..6]
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title(r"(d) Sub-Group Metrics Across Cluster Count $N_c$", fontweight="bold", loc="left")

    nc_keys = ["Nc=1", "Nc=2", "Nc=3", "Nc=4", "Nc=5", "Nc=6"]
    dis_nc = bench_data["tg_dis_deeponet"]["breakdown_by_nc"]
    tg_nc = bench_data["tg_deeponet"]["breakdown_by_nc"]

    mae_dis_nc = [dis_nc[k]["alpha_mae"] for k in nc_keys]
    mae_tg_nc = [tg_nc[k]["alpha_mae"] for k in nc_keys]
    w1_dis_nc = [dis_nc[k]["w1_mean_m"] for k in nc_keys]

    x_nc = np.arange(len(nc_keys))
    w = 0.35

    ax_d.bar(x_nc - w/2, mae_tg_nc, width=w, color=PALETTE["purple"], alpha=0.7, label=r"TG-DeepONet $\alpha$ MAE")
    ax_d.bar(x_nc + w/2, mae_dis_nc, width=w, color=PALETTE["green"], alpha=0.85, label=r"TG-DIS-DeepONet $\alpha$ MAE")

    ax_d.set_xticks(x_nc)
    ax_d.set_xticklabels([f"$N_c={k}$" for k in range(1, 7)], fontsize=7.2)
    ax_d.set_ylabel(r"Intake Fraction MAE ($\alpha$)")
    ax_d.set_ylim(0, 0.30)
    ax_d.grid(True, ls="--", lw=0.5, alpha=0.5, axis="y")

    # Inset or twin for W1
    ax_d_twin = ax_d.twinx()
    ax_d_twin.spines["top"].set_visible(False)
    ax_d_twin.plot(x_nc, w1_dis_nc, color=PALETTE["navy"], lw=1.5, marker="^", label=r"TG-DIS $W_1$ (m)")
    ax_d_twin.set_ylabel("Spatial $W_1$ (m)", color=PALETTE["navy"], fontweight="bold")
    ax_d_twin.tick_params(axis="y", labelcolor=PALETTE["navy"])
    ax_d_twin.set_ylim(0, 18)

    lines_d1, lab_d1 = ax_d.get_legend_handles_labels()
    lines_d2, lab_d2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(lines_d1 + lines_d2, lab_d1 + lab_d2, loc="upper right", fontsize=6.8)

    save_figure(fig, "fig3_benchmark_and_ablation")


# ==============================================================================
# Figure 4: Noise & Sound Speed Robustness Stress Test
# ==============================================================================
def plot_fig4_noise_and_speed_robustness() -> None:
    """
    Figure 4: Two-Stage Noise & Sound Speed Robustness Audit
    - Panel a: AWGN degradation curves across Clean, 30dB, 20dB, 10dB.
    - Panel b: Pink (1/f) noise degradation curves across Clean, 30dB, 20dB, 10dB.
    - Panel c: Sound speed perturbation stress test (+-1% sound speed mismatch).
    - Panel d: Simplex conservation & F1-score stability across all regimes.
    """
    fig = plt.figure(figsize=(10.5, 8.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    noise_file = _ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "phase3_noise_robustness_metrics.json"
    with open(noise_file, "r", encoding="utf-8") as f:
        noise_data = json.load(f)

    dis_noise = noise_data["tg_dis_deeponet"]
    tg_noise = noise_data["tg_deeponet"]

    # --------------------------------------------------------------------------
    # Panel a: AWGN Degradation Curves
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("(a) Additive White Gaussian Noise (AWGN) Stress Test", fontweight="bold", loc="left")

    awgn_keys = ["clean", "awgn_30db", "awgn_20db", "awgn_10db"]
    awgn_labels = ["Clean\n($\\infty$ dB)", "30 dB", "20 dB\n(Key Stress)", "10 dB\n(Severe)"]

    dis_awgn_mae = [dis_noise[k]["alpha_mae"] for k in awgn_keys]
    tg_awgn_mae = [tg_noise[k]["alpha_mae"] for k in awgn_keys]

    x_noise = np.arange(len(awgn_keys))
    ax_a.plot(x_noise, tg_awgn_mae, marker="s", lw=1.5, color=PALETTE["purple"], label="TG-DeepONet")
    ax_a.plot(x_noise, dis_awgn_mae, marker="o", lw=1.8, color=PALETTE["green"], label="TG-DIS-DeepONet (Proposed)")

    # 15% threshold bound
    baseline_val = dis_awgn_mae[0]
    thresh_val = baseline_val * 1.15
    ax_a.axhline(thresh_val, color=PALETTE["crimson"], ls="--", lw=1.1, label=r"15% Degradation Ceiling")
    ax_a.fill_between(x_noise, 0.12, thresh_val, color="#E8F5E9", alpha=0.3, label="Acceptance Passing Zone")

    ax_a.set_xticks(x_noise)
    ax_a.set_xticklabels(awgn_labels, fontsize=7.2)
    ax_a.set_ylabel(r"Intake Fraction MAE ($\alpha$)")
    ax_a.set_ylim(0.125, 0.165)
    ax_a.legend(loc="upper left", fontsize=7.0)
    ax_a.grid(True, ls="--", lw=0.5, alpha=0.5)

    # --------------------------------------------------------------------------
    # Panel b: Pink 1/f Noise Degradation Curves
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title(r"(b) Colored Pink ($1/f$) Noise Stress Test", fontweight="bold", loc="left")

    pink_keys = ["clean", "pink_30db", "pink_20db", "pink_10db"]
    pink_labels = ["Clean", "30 dB", "20 dB\n(Key Stress)", "10 dB\n(Severe)"]

    dis_pink_mae = [dis_noise[k]["alpha_mae"] for k in pink_keys]
    tg_pink_mae = [tg_noise[k]["alpha_mae"] for k in pink_keys]

    ax_b.plot(x_noise, tg_pink_mae, marker="s", lw=1.5, color=PALETTE["purple"], label="TG-DeepONet")
    ax_b.plot(x_noise, dis_pink_mae, marker="o", lw=1.8, color=PALETTE["green"], label="TG-DIS-DeepONet")

    ax_b.axhline(thresh_val, color=PALETTE["crimson"], ls="--", lw=1.1, label=r"15% Degradation Ceiling")
    ax_b.fill_between(x_noise, 0.12, thresh_val, color="#E8F5E9", alpha=0.3)

    ax_b.set_xticks(x_noise)
    ax_b.set_xticklabels(pink_labels, fontsize=7.2)
    ax_b.set_ylabel(r"Intake Fraction MAE ($\alpha$)")
    ax_b.set_ylim(0.125, 0.165)
    ax_b.legend(loc="upper left", fontsize=7.0)
    ax_b.grid(True, ls="--", lw=0.5, alpha=0.5)

    # --------------------------------------------------------------------------
    # Panel c: Sound Speed Perturbation (+-1%) Resilience
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title(r"(c) Sound Speed Perturbation ($\pm 1.0\%$, $1435.5 \sim 1464.5\,\mathrm{m/s}$)", fontweight="bold", loc="left")

    speed_keys = ["speed_minus_1pct", "clean", "speed_plus_1pct"]
    speed_labels = [r"$-1.0\%$ ($1435.5\,\mathrm{m/s}$)", r"Nominal ($1450\,\mathrm{m/s}$)", r"$+1.0\%$ ($1464.5\,\mathrm{m/s}$)"]

    dis_speed_w1 = [dis_noise[k]["w1_mean_m"] for k in speed_keys]
    tg_speed_w1 = [tg_noise[k]["w1_mean_m"] for k in speed_keys]

    x_speed = np.arange(len(speed_keys))
    w = 0.32

    ax_c.bar(x_speed - w/2, tg_speed_w1, width=w, color=PALETTE["purple"], alpha=0.75, label="TG-DeepONet")
    ax_c.bar(x_speed + w/2, dis_speed_w1, width=w, color=PALETTE["green"], alpha=0.85, label="TG-DIS-DeepONet (Proposed)")

    for i in range(len(speed_keys)):
        ax_c.text(i - w/2, tg_speed_w1[i] + 0.3, f"{tg_speed_w1[i]:.2f}m", ha="center", fontsize=7.0, color=PALETTE["purple"])
        ax_c.text(i + w/2, dis_speed_w1[i] + 0.3, f"{dis_speed_w1[i]:.2f}m", ha="center", fontsize=7.0, color=PALETTE["green"], fontweight="bold")

    # Annotate divergence protection
    ax_c.annotate(r"TG diverges to $15.58\,\mathrm{m}$" + "\n" + r"TG-DIS preserved at $11.87\,\mathrm{m}$",
                  xy=(2 - w/2, tg_speed_w1[2]), xytext=(0.8, 14.5),
                  arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.2),
                  fontsize=7.2, color=PALETTE["crimson"], fontweight="bold")

    ax_c.set_xticks(x_speed)
    ax_c.set_xticklabels(speed_labels, fontsize=7.2)
    ax_c.set_ylabel(r"1D Spatial Wasserstein $W_1$ Error (m)")
    ax_c.set_ylim(0, 18)
    ax_c.legend(loc="upper left", fontsize=7.0)
    ax_c.grid(True, ls="--", lw=0.5, alpha=0.5, axis="y")

    # --------------------------------------------------------------------------
    # Panel d: Physical Simplex Conservation & Detection F1-Score
    # --------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title(r"(d) Inversion Invariant Fidelity Across All 9 Perturbations", fontweight="bold", loc="left")

    all_keys = [
        "clean", "awgn_30db", "awgn_20db", "awgn_10db",
        "pink_30db", "pink_20db", "pink_10db",
        "speed_minus_1pct", "speed_plus_1pct"
    ]
    cond_names = ["Clean", "A-30", "A-20", "A-10", "P-30", "P-20", "P-10", "c-1%", "c+1%"]

    f1_scores = [dis_noise[k]["f1_score"] for k in all_keys]
    simplex_dev = [dis_noise[k]["simplex_max_dev"] for k in all_keys]

    x_all = np.arange(len(all_keys))

    ax_d.plot(x_all, f1_scores, marker="o", color=PALETTE["navy"], lw=1.6, label="Detection F1-Score")
    ax_d.axhline(0.88, color="red", ls="--", lw=1.0, label="Acceptance Gate ($F_1 > 0.88$)")

    ax_d.set_xticks(x_all)
    ax_d.set_xticklabels(cond_names, fontsize=7.0)
    ax_d.set_ylabel("Detection F1-Score", color=PALETTE["navy"], fontweight="bold")
    ax_d.tick_params(axis="y", labelcolor=PALETTE["navy"])
    ax_d.set_ylim(0.82, 0.98)

    ax_d_twin = ax_d.twinx()
    ax_d_twin.spines["top"].set_visible(False)
    ax_d_twin.plot(x_all, simplex_dev, marker="x", color=PALETTE["amber"], lw=1.5, ls=":", label=r"Max Simplex Dev")
    ax_d_twin.axhline(1e-6, color=PALETTE["crimson"], ls=":", lw=1.0, label=r"Simplex Ceiling ($10^{-6}$)")

    ax_d_twin.set_yscale("log")
    ax_d_twin.set_ylabel(r"Simplex Error $\max|\sum\alpha - 1.0|$", color=PALETTE["amber"], fontweight="bold")
    ax_d_twin.tick_params(axis="y", labelcolor=PALETTE["amber"])
    ax_d_twin.set_ylim(1e-8, 1e-5)

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d_twin.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=6.8)
    ax_d.grid(True, ls="--", lw=0.5, alpha=0.5)

    save_figure(fig, "fig4_noise_and_speed_robustness")


# ==============================================================================
# Figure 5: Typical Fracturing Cases Inversion Profiles
# ==============================================================================
def plot_fig5_typical_cases_inversion() -> None:
    """
    Figure 5: Inversion Profiles Across 5 Representative Fracturing Cases
    - Case A: Uniform intake (Idx 11, Nc=3)
    - Case B: Heel-dominant (Idx 18, Nc=5)
    - Case C: Saddle profile (Idx 4, Nc=5)
    - Case D: Toe-dominant (Idx 61, Nc=5)
    - Case E: Sand screen-out dead cluster (Idx 7, Nc=5)
    """
    fig, axes = plt.subplots(5, 2, figsize=(10.5, 10.5), sharex="col",
                             gridspec_kw={"hspace": 0.35, "wspace": 0.28, "width_ratios": [1.1, 1.4]})

    fig.suptitle("Typical Field Fracturing Regimes: Discrete Multi-Cluster & Continuous Field Inversion",
                 fontsize=10.5, fontweight="bold", y=0.995)

    ds = PilotInversionDataset(split="test")
    model = TGDISDeepONet()
    ckpt = torch.load(WEIGHTS_DIR / "tg_dis_deeponet_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    case_indices = [11, 18, 4, 61, 7]
    case_titles = [
        "Case 1: Balanced Uniform Intake ($N_c=3$)",
        "Case 2: Heel-Dominant Flush ($N_c=5$, Heel takes $>50\\%$)",
        "Case 3: Saddle Distribution ($N_c=5$, Shadowing Suppression)",
        "Case 4: Toe-Dominant Divergence ($N_c=5$, Toe takes $>55\\%$)",
        "Case 5: Sand Screen-Out Dead Cluster ($N_c=5$, Toe Plugged)",
    ]

    for row, (idx, title) in enumerate(zip(case_indices, case_titles)):
        item = ds[idx]
        mask_np = item["mask"].bool().numpy()
        pos_np = item["positions"].numpy()[mask_np]
        true_alpha = item["alpha"].numpy()[mask_np]
        grid_x = ds.grid_x
        true_m = item["m_alpha_grid"].numpy()

        batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = model(batch=batch)
            pred_alpha = out["alpha"].squeeze(0).numpy()[mask_np]
            pred_m = out["m_alpha_grid"].squeeze(0).numpy()

        # Left Column: Discrete Cluster Intake Comparison
        ax_l = axes[row, 0]
        ax_l.set_title(f"({chr(97 + row*2)}) {title}", fontweight="bold", loc="left", fontsize=7.8)

        clust_idx = np.arange(1, len(true_alpha) + 1)
        w = 0.35
        ax_l.bar(clust_idx - w/2, true_alpha, width=w, color=PALETTE["navy"], alpha=0.8, label="Ground Truth $\\alpha$")
        ax_l.bar(clust_idx + w/2, pred_alpha, width=w, color=PALETTE["green"], alpha=0.85, label="TG-DIS Inverted $\\hat{\\alpha}$")

        ax_l.set_xticks(clust_idx)
        ax_l.set_xticklabels([f"C{k}\n({pos_np[k-1]:.0f}m)" for k in clust_idx], fontsize=6.8)
        ax_l.set_ylabel(r"Intake $\alpha_j$", fontsize=7.5)
        ax_l.set_ylim(0, max(np.max(true_alpha), np.max(pred_alpha)) * 1.35)
        ax_l.grid(True, ls="--", lw=0.5, alpha=0.5, axis="y")
        if row == 0:
            ax_l.legend(loc="upper right", fontsize=6.8)

        # Right Column: Continuous Flow Field Trunk Reconstruction
        ax_r = axes[row, 1]
        ax_r.set_title(f"({chr(98 + row*2)}) Continuous Flow Density $m_\\alpha(x)$ Along Wellbore", fontweight="bold", loc="left", fontsize=7.8)

        ax_r.plot(grid_x, true_m, color=PALETTE["navy"], lw=1.3, label="True Density $m_\\alpha(x)$")
        ax_r.plot(grid_x, pred_m, color=PALETTE["green"], lw=1.4, ls="--", label="Inverted Field $\\hat{m}_\\alpha(x)$")

        # Mark fracture positions
        for p in pos_np:
            ax_r.axvline(p, color=PALETTE["amber"], ls=":", lw=0.8, alpha=0.8)

        ax_r.set_xlim(pos_np[0] - 80, pos_np[-1] + 80)
        ax_r.set_ylabel(r"$m_\alpha(x)$ ($10^{-2}\,\mathrm{m^{-1}}$)", fontsize=7.5)
        ax_r.grid(True, ls="--", lw=0.5, alpha=0.5)
        if row == 0:
            ax_r.legend(loc="upper right", fontsize=6.8)

    axes[-1, 0].set_xlabel("Fracture Stage Cluster", fontsize=8.0)
    axes[-1, 1].set_xlabel("Wellbore Coordinate $x$ (m)", fontsize=8.0)

    save_figure(fig, "fig5_typical_cases_inversion")


# ==============================================================================
# Main Runner
# ==============================================================================
def main() -> None:
    print("=" * 80)
    print("PaperC Phase 3 Publication-Grade Figure Generation Pipeline")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    print("\n[1/5] Generating Figure 1: Layer-Stripping Acoustic Mechanism...")
    plot_fig1_layer_stripping_mechanism()

    print("\n[2/5] Generating Figure 2: TG-DIS-DeepONet Architecture & Explainability...")
    plot_fig2_tg_dis_architecture()

    print("\n[3/5] Generating Figure 3: Benchmark & Ablation Study Performance...")
    plot_fig3_benchmark_and_ablation()

    print("\n[4/5] Generating Figure 4: Noise & Speed Robustness Audit...")
    plot_fig4_noise_and_speed_robustness()

    print("\n[5/5] Generating Figure 5: Typical Fracturing Cases Inversion Profiles...")
    plot_fig5_typical_cases_inversion()

    print("\n" + "=" * 80)
    print("All 5 composite figures (10 files total: 5 PNG + 5 SVG) successfully generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
