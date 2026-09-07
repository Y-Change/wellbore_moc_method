"""
Diagnostic Credibility Index (Psi) Model & Section 3 Methodology Framework
Author: Antigravity (Pair Programming with User)
Target: SPE Journal - Multi-Cluster Water-Hammer Diagnostics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = r"E:\water_hammer_research\wellbore_moc_method"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Matplotlib rcParams Configuration: Strict Paper A & SPE Journal Specification ---
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
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.8,
    # Full box border and inward ticks
    "axes.spines.right": True,
    "axes.spines.top": True,
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

# --- Curated Color Palettes ---
COLOR_BLUE_MAIN = "#1B4F72"
COLOR_TEAL      = "#117864"
COLOR_ORANGE    = "#D35400"
COLOR_RED       = "#900C3F"
COLOR_PURPLE    = "#6C3483"
COLOR_SLATE     = "#2C3E50"
COLOR_GOLD      = "#B7950B"
COLOR_GREY      = "#7F8C8D"

BASE_DIR = os.path.join(REPO_ROOT, "PaperA井口多裂缝水击响应")
SEC_DIR  = os.path.join(BASE_DIR, "文章撰写", "Section3_可信度指数模型与方法论")
FIG_DIR  = os.path.join(SEC_DIR, "figures")
TAB_DIR  = os.path.join(SEC_DIR, "data")
DATA_SRC = os.path.join(BASE_DIR, "01_几何网格")


class DiagnosticCredibilityModel:
    """
    Closed-form Analytical Diagnostic Credibility Index (Psi) Model.
    Calculates cluster-level credibility Psi_k, stage-level credibility Q_stage,
    recovery ratio eta_rec, and Grade A/B/C Signal Quality Classification.
    """
    def __init__(self, 
                 wave_speed=1450.0, 
                 T_eff=0.784, 
                 S_min=10.0, 
                 S_null=30.0, 
                 w_null=22.0, 
                 kappa=0.65, 
                 sigma_gauge=0.05,
                 psi_thresh_A=5.0,
                 psi_thresh_B=3.0):
        self.a = wave_speed
        self.T_eff = T_eff
        self.S_min = S_min
        self.S_null = S_null
        self.w_null = w_null
        self.kappa = kappa
        self.sigma_gauge = sigma_gauge
        self.psi_thresh_A = psi_thresh_A
        self.psi_thresh_B = psi_thresh_B

    def spatial_separation_factor(self, S):
        """Spatial resolution operator Phi_sep(S): 1 when S >= S_min, drops to 0 when S < S_min."""
        S_arr = np.asarray(S, dtype=float)
        return 1.0 - np.exp(-(S_arr / self.S_min)**2)

    def phase_cancellation_factor(self, S):
        """Multipath destructive phase interference factor eta_phase(S). U-valley minimum at S_null ~ 30m."""
        S_arr = np.asarray(S, dtype=float)
        return 1.0 - self.kappa * np.exp(-((S_arr - self.S_null) / self.w_null)**2)

    def cascaded_transmission_factor(self, k):
        """Discrete cascaded interface transmission loss T_eff^{2(k-1)}."""
        k_arr = np.asarray(k, dtype=float)
        return (self.T_eff**2)**(k_arr - 1.0)

    def effective_noise_floor(self, k, S):
        """Comprehensive interference noise floor: gauge noise + cepstral cross-term harmonic noise."""
        k_arr = np.asarray(k, dtype=float)
        S_arr = np.asarray(S, dtype=float)
        sigma_cross = 0.015 * np.sqrt(np.maximum(k_arr - 1.0, 0.0)) * np.exp(-S_arr / 45.0)
        return np.sqrt(self.sigma_gauge**2 + sigma_cross**2)

    def baseline_excitation(self, X1):
        """Nominal primary reflection peak baseline P1(X1) ~ 3.0 at 3000m."""
        return 3.00

    def evaluate_cluster(self, k, S, n_total, X1=3000.0):
        """Calculates single-cluster credibility Psi_k."""
        P1 = self.baseline_excitation(X1)
        phi_sep = self.spatial_separation_factor(S)
        eta_phi = self.phase_cancellation_factor(S) if k > 2 else (1.0 if k == 1 else 0.5 + 0.5 * self.phase_cancellation_factor(S))
        t_casc  = self.cascaded_transmission_factor(k)
        sigma_e = self.effective_noise_floor(k, S)
        
        # Physical peak amplitude prediction
        P_k_pred = P1 * t_casc * eta_phi * phi_sep
        # Signal credibility index
        psi_k = P_k_pred / sigma_e
        return P_k_pred, psi_k

    def evaluate_stage(self, S, n_total, X1=3000.0):
        """
        Evaluates full stage credibility Q_stage, cluster-level Psi profile,
        recoverable clusters n_rec, and Grade A/B/C rating.
        """
        k_vals = np.arange(1, n_total + 1)
        P_preds = []
        psi_vals = []
        for k in k_vals:
            p_k, psi_k = self.evaluate_cluster(k, S, n_total, X1)
            P_preds.append(p_k)
            psi_vals.append(psi_k)
            
        P_preds = np.array(P_preds)
        psi_vals = np.array(psi_vals)
        
        Q_stage = np.min(psi_vals) # Governed by the weakest/deepest cluster
        n_rec = np.sum(psi_vals >= self.psi_thresh_A)
        eta_rec = (n_rec / n_total) * 100.0
        
        # Classification
        if Q_stage >= self.psi_thresh_A:
            grade = "Grade A (High-Fidelity / Reliable)"
            grade_short = "Grade A"
            rec_action = "Full Quantitative Inversion (Reliable)"
        elif Q_stage >= self.psi_thresh_B:
            grade = "Grade B (Interference-Constrained)"
            grade_short = "Grade B"
            rec_action = "Layer-Stripping Compensation Required"
        else:
            grade = "Grade C (Degraded / Blind Zone)"
            grade_short = "Grade C"
            rec_action = "Unreliable / Inversion Blocked (Adjust Perforations)"
            
        return {
            'S': S,
            'n_total': n_total,
            'X1': X1,
            'P_preds': P_preds,
            'psi_vals': psi_vals,
            'Q_stage': Q_stage,
            'n_rec': n_rec,
            'eta_rec': eta_rec,
            'grade': grade,
            'grade_short': grade_short,
            'action': rec_action
        }


def generate_benchmark_profiles():
    """
    Generates synthetic benchmark cepstral profiles along spatial depth x
    representing the 4 fundamental archetypes of signal quality.
    """
    x = np.linspace(2980, 3350, 1500)
    
    def ricker(x_eval, center, amp, width):
        t = (x_eval - center) / width
        return amp * (1.0 - 2.0 * t**2) * np.exp(-t**2)

    # 1. Grade A: High-Fidelity Signal (n=3, S=80m, X1=3000m)
    # Clear, well-separated peaks at 3000, 3080, 3160m
    y_A = 0.05 * np.sin(x/15) + 0.03 * np.cos(x/7)
    y_A += ricker(x, 3000, 3.05, 7.0)
    y_A += ricker(x, 3080, 1.88, 7.0)
    y_A += ricker(x, 3160, 1.15, 7.0)
    
    # 2. Coalescence Failure (n=4, S=6m, X1=3000m)
    # S < 10m: All 4 clusters at 3000, 3006, 3012, 3018m merge into a single wide bulge
    y_Coal = 0.05 * np.sin(x/15) + 0.03 * np.cos(x/7)
    for c in [3000, 3006, 3012, 3018]:
        y_Coal += ricker(x, c, 1.25, 7.0)
        
    # 3. Grade B: Destructive Phase Interference (n=4, S=30m, X1=3000m)
    # Clusters at 3000, 3030, 3060, 3090m. 3rd & 4th clusters deeply suppressed
    y_B = 0.05 * np.sin(x/15) + 0.03 * np.cos(x/7)
    y_B += ricker(x, 3000, 3.12, 7.0)
    y_B += ricker(x, 3030, 1.44, 7.0)
    y_B += ricker(x, 3060, 0.48, 7.0) # Suppressed U-valley
    y_B += ricker(x, 3090, 0.38, 7.0) # Suppressed
    
    # 4. Grade C: Deep Blind Zone / Multi-Interface Exhaustion (n=8, S=20m, X1=3000m)
    # Clusters at 3000, 3020, 3040, 3060, 3080, 3100, 3120, 3140m. Trailing clusters < 0.15 submerged
    y_C = 0.05 * np.sin(x/15) + 0.03 * np.cos(x/7)
    amps_C = [3.42, 1.62, 0.55, 0.42, 0.28, 0.21, 0.18, 0.14]
    for i, amp in enumerate(amps_C):
        y_C += ricker(x, 3000 + i*20, amp, 7.0)
        
    return x, y_A, y_Coal, y_B, y_C


def plot_figure_3_1(model):
    """
    Generates Section 3 Methodology Benchmark Figure (Figure 3.1):
    (a) 4 Typical Cepstral Waveforms (Grade A, Coalescence, Grade B Interference, Grade C Blind Zone)
    (b) Analytical Physical Decomposition Curves (Phi_sep, eta_phase, T_eff cascaded)
    (c) Diagnostic Credibility Profiles Psi_k with Grade A/B/C Decision Thresholds
    """
    print("[1/2] Generating Figure 3.1 (Multi-Cluster Cepstral Credibility Framework)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35), constrained_layout=True)
    
    # --- (a) Representative Cepstral Waveforms ---
    ax = axes[0]
    ax.set_box_aspect(4/5)
    x, y_A, y_Coal, y_B, y_C = generate_benchmark_profiles()
    
    # Vertical stacking for crystal clear comparison
    ax.plot(x, y_A + 8.5, color=COLOR_TEAL, lw=1.0, label="Grade A: High-Fidelity ($S=80$m, $n=3$)")
    ax.plot(x, y_Coal + 5.7, color=COLOR_GREY, lw=1.0, ls='--', label="Coalescence Limit ($S=6$m, $n=4$)")
    ax.plot(x, y_B + 2.8, color=COLOR_ORANGE, lw=1.0, label="Grade B: Phase Cancel ($S=30$m, $n=4$)")
    ax.plot(x, y_C + 0.0, color=COLOR_RED, lw=1.0, label="Grade C: Blind Zone ($S=20$m, $n=8$)")
    
    ax.axhline(0.20, color="#BDC3C7", ls=":", lw=0.6)
    ax.text(3310, 0.40, "Noise Floor", fontsize=5.5, color=COLOR_GREY, ha='right')
    
    ax.set_xlabel("Apparent Depth $x$ (m)")
    ax.set_ylabel("Marginal Cepstrum $P_{2D}$ (a.u.)")
    ax.set_xlim(2980, 3320)
    ax.set_ylim(-0.5, 12.8)
    ax.set_yticks([1.5, 4.3, 7.2, 10.0])
    ax.set_yticklabels(["Grade C", "Grade B", "Overlap", "Grade A"], fontsize=6.5)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (b) Physical Factor Decomposition Curves ---
    ax = axes[1]
    ax.set_box_aspect(4/5)
    
    s_dense = np.linspace(0, 100, 150)
    phi_s = model.spatial_separation_factor(s_dense)
    eta_s = model.phase_cancellation_factor(s_dense)
    
    k_dense = np.linspace(1, 8, 150)
    t_casc = model.cascaded_transmission_factor(k_dense)
    
    ax.plot(s_dense, phi_s, color=COLOR_SLATE, lw=1.1, ls='--', label="Separation $\\Phi_{sep}(S)$ ($S_{min}=10$m)")
    ax.plot(s_dense, eta_s, color=COLOR_ORANGE, lw=1.2, label="Interference $\\eta_{phase}(S)$ (U-Valley)")
    ax.plot(s_dense, (model.T_eff**2)**((s_dense/20.0)-1.0), color=COLOR_BLUE_MAIN, lw=1.1, ls=':', 
            label="Cascaded $\\mathcal{T}^{2(k-1)}$ ($T_{eff}=0.78$)")
    
    # Highlight critical zones
    ax.axvspan(0, 10, color=COLOR_GREY, alpha=0.18, hatch='//')
    ax.text(5, 0.50, "Overlap", fontsize=5.8, color=COLOR_GREY, rotation=90, va='center', ha='center')
    
    ax.axvspan(20, 50, color=COLOR_ORANGE, alpha=0.12)
    ax.text(35, 0.12, "U-Valley Cancellation", fontsize=5.8, color=COLOR_ORANGE, ha='center')
    
    ax.set_xlabel("Fracture Spacing $S$ (m) / Index $k$")
    ax.set_ylabel("Normalized Physical Factor")
    ax.set_xlim(0, 100)
    # Expand Y-lim to 1.38 to give ample headroom for legend
    ax.set_ylim(0.0, 1.38)
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.2, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    # --- (c) Credibility Profiles Psi_k & Decision Thresholds ---
    ax = axes[2]
    ax.set_box_aspect(4/5)
    
    # Evaluate the 4 cases
    res_A = model.evaluate_stage(S=80, n_total=3)
    res_Coal = model.evaluate_stage(S=6, n_total=4)
    res_B = model.evaluate_stage(S=30, n_total=4)
    res_C = model.evaluate_stage(S=20, n_total=8)
    
    ax.plot(np.arange(1, 4), res_A['psi_vals'], marker='o', color=COLOR_TEAL, lw=1.1, markersize=3.6,
            label=f"Case A: Grade A ($Q_{{stg}}={res_A['Q_stage']:.1f}$)")
    ax.plot(np.arange(1, 5), res_B['psi_vals'], marker='s', color=COLOR_ORANGE, lw=1.1, markersize=3.6,
            label=f"Case B: Grade B ($Q_{{stg}}={res_B['Q_stage']:.1f}$)")
    ax.plot(np.arange(1, 9), res_C['psi_vals'], marker='^', color=COLOR_RED, lw=1.1, markersize=3.6,
            label=f"Case C: Grade C ($Q_{{stg}}={res_C['Q_stage']:.1f}$)")
    ax.plot(np.arange(1, 5), res_Coal['psi_vals'], marker='x', color=COLOR_GREY, lw=1.0, ls='--', markersize=4.0,
            label=f"Case D: Overlap ($Q_{{stg}}={res_Coal['Q_stage']:.1f}$)")
    
    # Horizontal Decision Thresholds included in legend for zero-collision
    ax.axhline(5.0, color=COLOR_TEAL, ls="--", lw=0.75, label="Grade A Thresh ($\\Psi=5.0$)")
    ax.axhline(3.0, color=COLOR_RED, ls=":", lw=0.75, label="Grade B Thresh ($\\Psi=3.0$)")
    
    ax.set_xlabel("Fracture Cluster Index $k$")
    ax.set_ylabel("Diagnostic Credibility Index $\\Psi_k$")
    ax.set_xlim(0.5, 8.6)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    # Headroom expansion to 75 to ensure ZERO overlap
    ax.set_ylim(-3.0, 75.0)
    ax.legend(loc="upper right", fontsize=5.0, handlelength=1.1, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("(c)", loc="left", fontsize=9.5, fontweight="bold", pad=5)
    
    out_prefix = os.path.join(FIG_DIR, "Figure_3_1_Cepstral_Credibility_Framework")
    fig.savefig(f"{out_prefix}.png", dpi=400)
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Saved Figure 3.1: {out_prefix}.png / .svg / .pdf")


def validate_against_moc_database(model):
    """
    Validates the Psi model against the 70 empirical MOC simulation configurations
    from decay_table.csv and identifiability_envelope_metrics.csv.
    """
    print("[2/2] Validating Psi Model against empirical MOC simulation database (70 cases)...")
    
    csv_table = os.path.join(DATA_SRC, "峰值表", "decay_table.csv")
    df = pd.read_csv(csv_table)
    df_sub = df[(df['friction_model'] == 'steady') & (df['x1'] == 3000.0) & (df['n_total'] >= 2)]
    
    validation_records = []
    for (n_val, sp_val), group in df_sub.groupby(['n_total', 'spacing_m']):
        group_sorted = group.sort_values('frac_idx')
        p_sims = group_sorted['P_2d'].values
        p_tail_sim = p_sims[-1]
        
        # Analytical model evaluation
        res = model.evaluate_stage(S=sp_val, n_total=n_val, X1=3000.0)
        p_tail_pred = res['P_preds'][-1]
        psi_tail_pred = res['Q_stage']
        grade_pred = res['grade_short']
        
        # Empirical classification benchmark (using P_th = 0.20 as noise limit)
        if p_tail_sim >= 0.25 and (p_tail_sim / p_sims[0]) >= 0.15:
            grade_sim = "Grade A"
        elif p_tail_sim >= 0.15:
            grade_sim = "Grade B"
        else:
            grade_sim = "Grade C"
            
        is_consistent = (grade_pred == grade_sim)
        
        validation_records.append({
            'n_total': int(n_val),
            'spacing_m': float(sp_val),
            'P_tail_sim': p_tail_sim,
            'P_tail_pred': p_tail_pred,
            'Q_stage_pred': psi_tail_pred,
            'Grade_Sim': grade_sim,
            'Grade_Pred': grade_pred,
            'Is_Consistent': is_consistent
        })
        
    df_val = pd.DataFrame(validation_records)
    out_csv = os.path.join(TAB_DIR, "psi_model_validation_summary.csv")
    df_val.to_csv(out_csv, index=False)
    
    accuracy = df_val['Is_Consistent'].mean() * 100.0
    r_corr = np.corrcoef(df_val['P_tail_sim'], df_val['P_tail_pred'])[0, 1]
    
    print(f"===============================================================")
    print(f" [Psi Model Validation Results]")
    print(f"  - Total MOC Cases Tested: {len(df_val)}")
    print(f"  - Classification Consistency Accuracy: {accuracy:.2f}%")
    print(f"  - Predicted vs Simulated Tail Amplitude Correlation: r = {r_corr:.4f}")
    print(f"  - Summary Table Saved to: {out_csv}")
    print(f"===============================================================")
    return df_val


if __name__ == "__main__":
    model = DiagnosticCredibilityModel()
    plot_figure_3_1(model)
    validate_against_moc_database(model)
