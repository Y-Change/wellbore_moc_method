# -*- coding: utf-8 -*-
"""
plot_fig5a_cepstrogram.py — 2D Sliding-Window Cepstrogram Bridge Figure for SPE Journal.

STRICT DATA INTEGRITY & AUDITED REPRODUCIBILITY:
- 100% bound to Paper B canonical wellhead traces in
  01_非定常摩阻正演与基准/data/timeseries/*.csv.gz
  (a) n1_k0.csv.gz
  (b) n1_k0.01.csv.gz
  (c) n4_D20_k0.01.csv.gz
  (d) n4_D5_k0.01.csv.gz

FROZEN SLIDING-WINDOW SPECIFICATIONS:
- Channel: dH/dt (wellhead pressure derivative)
- Window: Full Hann taper, Twin = 30.0 s (p6 audit: window shorter than 4L/a approx 13.8 s is ineffective)
- Hop: hop = 0.25 s (for dense publication-quality visualization)
- Depth axis: x = a * tau / 2, with acoustic wave speed a = 1450 m/s
- Horizontal range: 4060 to 4190 m
- Vertical range: Window center time t_center in [15, 35] s
- Colorbar: Shared unified scale across all 4 subplots
- True fracture coordinates: Single fracture X1 = 4100 m; Multi-cluster Xi = 4100 + i * D

IMPORTANT SCIENTIFIC NOTE:
The 2D sliding-window cepstrogram visually demonstrates the progressive loss of high-frequency homomorphic
sharpness and the spatial resolution limit across observation windows. Numerical depth benchmarks throughout
the main text strictly adhere to the audited 1D full-window |C(tau)| metrics (+9.43 m, +5.80 m, +11.80 m, +18.73 m).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
matplotlib_backend = 'Agg'
mpl.use(matplotlib_backend)
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# SPEJ Publication Styling
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 3.0
mpl.rcParams['ytick.major.size'] = 3.0
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['ytick.major.width'] = 0.75
mpl.rcParams['font.size'] = 7.5
mpl.rcParams['axes.labelsize'] = 8.0
mpl.rcParams['axes.titlesize'] = 8.0
mpl.rcParams['xtick.labelsize'] = 7.0
mpl.rcParams['ytick.labelsize'] = 7.0
mpl.rcParams['legend.fontsize'] = 6.5

_root = r"e:\water_hammer_research\wellbore_moc_method"
TS_DIR = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "01_非定常摩阻正演与基准", "data", "timeseries")
fig_dir = os.path.join(_root, "PaperB_考虑brunone的倒谱识别", "文章撰写", "figures")
os.makedirs(fig_dir, exist_ok=True)

def compute_sliding_cepstrum(t, h, twin=30.0, hop=0.25, a=1450.0, t_max=50.0):
    mask = (t >= 0.0) & (t <= t_max)
    t_sub = t[mask]
    h_sub = h[mask]
    
    dt = t_sub[1] - t_sub[0]
    dhdt = np.gradient(h_sub, dt)
    
    nwin = int(round(twin / dt))
    nhop = int(round(hop / dt))
    
    centers = []
    cep_list = []
    
    for i_start in range(0, len(t_sub) - nwin + 1, nhop):
        i_end = i_start + nwin
        t_w = t_sub[i_start:i_end]
        sig_w = dhdt[i_start:i_end]
        
        sig_w = sig_w - np.mean(sig_w)
        w = np.hanning(len(sig_w))
        sig_w = sig_w * w
        
        # Real cepstrum
        spec = fft(sig_w)
        log_mag = np.log(np.abs(spec) + 1e-12)
        cep = np.real(ifft(log_mag))
        cep_mag = np.abs(cep)
        
        center_t = (t_w[0] + t_w[-1]) / 2.0
        centers.append(center_t)
        cep_list.append(cep_mag)
        
    tau = np.arange(nwin) * dt
    depth = a * tau / 2.0
    return np.array(centers), depth, np.array(cep_list)

def main():
    cases = [
        ("(a) Single Fracture ($n=1, k=0$)",
         os.path.join(TS_DIR, "n1_k0.csv.gz"),
         [4100]),
        ("(b) Single Fracture ($n=1, k=0.01$)",
         os.path.join(TS_DIR, "n1_k0.01.csv.gz"),
         [4100]),
        ("(c) 4 Clusters ($n=4, D=20\\,\\mathrm{m}, k=0.01$)",
         os.path.join(TS_DIR, "n4_D20_k0.01.csv.gz"),
         [4100, 4120, 4140, 4160]),
        ("(d) 4 Clusters ($n=4, D=5\\,\\mathrm{m}, k=0.01$)",
         os.path.join(TS_DIR, "n4_D5_k0.01.csv.gz"),
         [4100, 4105, 4110, 4115]),
    ]

    all_data = []
    all_vals = []
    for title, path, fracs in cases:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required timeseries file: {path}")
        df = pd.read_csv(path)
        centers, depth, C2d = compute_sliding_cepstrum(df['t'].values, df['H_wh'].values, twin=30.0, hop=0.25, t_max=50.0)
        idx_d = (depth >= 4060) & (depth <= 4190)
        d_sub = depth[idx_d]
        C_sub = C2d[:, idx_d]
        all_data.append((title, centers, d_sub, C_sub, fracs))
        all_vals.append(C_sub)

    # Shared color scaling
    vmax = np.percentile(np.concatenate([v.ravel() for v in all_vals]), 99.2)
    vmin = 0.0

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True, constrained_layout=True)

    for i, (ax, (title, centers, d_sub, C_sub, fracs)) in enumerate(zip(axes, all_data)):
        D_mesh, T_mesh = np.meshgrid(d_sub, centers)
        # Use rasterized=True for clean vector SVG export
        im = ax.pcolormesh(D_mesh, T_mesh, C_sub, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
        
        # Plot true fracture reference depth lines
        for f_idx, x_f in enumerate(fracs):
            col = '#E74C3C' if f_idx == 0 else '#F39C12'
            ax.axvline(x_f, color=col, linestyle='--', linewidth=0.85, alpha=0.9, zorder=3)
            
        ax.set_title(title, pad=4, fontsize=7.2)
        ax.set_xlabel('Depth $x$ (m)', fontsize=7.5)
        if i == 0:
            ax.set_ylabel('Window Center Time $t_{\\mathrm{center}}$ (s)', fontsize=7.5)
        ax.set_xlim(4060, 4190)
        ax.set_ylim(centers[0], centers[-1])
        ax.tick_params(direction='in', top=True, right=True, labelsize=6.8)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.018, pad=0.015)
    cbar.set_label('Cepstral Mag. $|C(\\tau)|$', fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.5, direction='in')

    png_path = os.path.join(fig_dir, "Fig5A_cepstrogram_bridge.png")
    svg_path = os.path.join(fig_dir, "Fig5A_cepstrogram_bridge.svg")
    
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    
    print(f"Successfully generated:\n  {png_path}\n  {svg_path}")

if __name__ == "__main__":
    main()
