"""
Generate LaTeX Formula Images for PPT Presentation
Author: Antigravity (Pair Programming with User)
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Strict Academic STIX LaTeX Math Rendering
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
})

OUT_DIR = r"E:\water_hammer_research\wellbore_moc_method\PaperA井口多裂缝水击响应\文章撰写\Section3_可信度指数模型与方法论\figures\formulas"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

COLOR_PRIMARY = "#0E2F44"  # Deep Navy
COLOR_TEAL    = "#117864"  # Teal Accent
COLOR_CRIMSON = "#900C3F"  # Crimson Accent
COLOR_BLUE    = "#1B4F72"  # Blue Accent

formulas = [
    {
        "name": "formula_master_reconstruction.png",
        "text": r"$\tilde{P}_{2D}(x) = P_{2D}(x) \cdot \mathcal{G}_{peel}(x) \cdot \mathcal{K}_{comp}(x) \cdot \mathcal{F}_{focus}(x)$",
        "fontsize": 20,
        "color": COLOR_PRIMARY
    },
    {
        "name": "formula_ghost_peeling.png",
        "text": r"$\mathcal{G}_{peel}(x) = 1 - \sum_{m=2}^{M} \alpha_m \cdot \exp\left( -\left[\frac{x - (x_1 + mS)}{w_{ghost}}\right]^2 \right)$",
        "fontsize": 18,
        "color": COLOR_CRIMSON
    },
    {
        "name": "formula_gain_equalization.png",
        "text": r"$\mathcal{K}_{comp}(x) = 1 + \sum_{k=1}^n \left[ \frac{1}{T_{eff}^{2(k-1)} \cdot \eta_{phase}(S, f_0)} - 1 \right] \cdot \exp\left( -\left[\frac{x - x_{f,k}}{w_{comp}}\right]^2 \right)$",
        "fontsize": 17,
        "color": COLOR_BLUE
    },
    {
        "name": "formula_phase_factor.png",
        "text": r"$\text{where } \eta_{phase}(S, f_0) = \left| 1 + R_{frac} \, e^{-j \frac{4\pi S}{\lambda_0}} \right|, \quad T_{eff} = 1 - R_{frac}$",
        "fontsize": 15,
        "color": "#566573"
    },
    {
        "name": "formula_peak_focusing.png",
        "text": r"$\mathcal{F}_{focus}(x) = \sum_{k=1}^n \exp\left( -\left[\frac{x - x_{peak, k}}{w_{core}}\right]^4 \right), \quad x_{peak, k} = \arg\max_{x \in [x_{f,k}-\delta, x_{f,k}+\delta]} P_{2D}(x)$",
        "fontsize": 17,
        "color": COLOR_TEAL
    },
    {
        "name": "formula_confidence_mask.png",
        "text": r"$\mathcal{M}(x) = \operatorname{clip}\left( \sum_{k=1}^n \Psi_k \, e^{-\left[\frac{x - x_{f,k}}{w_m}\right]^2} - \sum_{m=2}^M \beta_m \, e^{-\left[\frac{x - (x_1+mS)}{w_m}\right]^2}, \, 0, \, 1 \right)$",
        "fontsize": 17,
        "color": COLOR_PRIMARY
    }
]

def render_all():
    for f in formulas:
        p = os.path.join(OUT_DIR, f["name"])
        fig = plt.figure(figsize=(0.1, 0.1), dpi=400)
        text = fig.text(0, 0, f["text"], fontsize=f["fontsize"], color=f["color"], va='bottom', ha='left')
        fig.savefig(p, dpi=400, bbox_inches='tight', transparent=True, pad_inches=0.04)
        plt.close(fig)
        print(f"--> Rendered formula: {p}")

if __name__ == "__main__":
    render_all()
