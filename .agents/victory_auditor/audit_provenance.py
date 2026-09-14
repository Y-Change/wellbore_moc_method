import os
import sys
import time
import json
import glob
import numpy as np
import pandas as pd

print('=== VICTORY AUDITOR FORENSIC PROVENANCE INSPECTOR ===')

files = [
    'ORIGINAL_REQUEST.md',
    'TEST_READY.md',
    'tests/test_fracture_sensitivity_e2e.py',
    'experiments/sensitivity/generate_matrix.py',
    'output/fracture_parameter_sensitivity/manifest.json',
    'experiments/sensitivity/run_simulation.py',
    'output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv',
    'output/fracture_parameter_sensitivity/tables/sensitivity_summary.json',
    'experiments/sensitivity/extract_features.py',
    'experiments/sensitivity/plot_figures.py',
    'output/fracture_parameter_sensitivity/figures/fig1_wavefront_step_gradient.png',
    'output/fracture_parameter_sensitivity/figures/fig1_wavefront_step_gradient.svg',
    'output/fracture_parameter_sensitivity/figures/fig2_envelope_rms_decay.png',
    'output/fracture_parameter_sensitivity/figures/fig2_envelope_rms_decay.svg',
    'output/fracture_parameter_sensitivity/figures/fig3_frequency_spectral_dissipation.png',
    'output/fracture_parameter_sensitivity/figures/fig3_frequency_spectral_dissipation.svg',
    'output/fracture_parameter_sensitivity/figures/fig4_cepstrum_rayleigh_resolution.png',
    'output/fracture_parameter_sensitivity/figures/fig4_cepstrum_rayleigh_resolution.svg',
    'output/fracture_parameter_sensitivity/README.md',
]

for f in files:
    if os.path.exists(f):
        mtime = os.path.getmtime(f)
        size = os.path.getsize(f)
        t_str = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(mtime))
        print(f'{f:75s} | {size:8d} B | {t_str}')
    else:
        print(f'{f:75s} | NOT FOUND')

# NPZ and CSV time ranges
npz_files = glob.glob('output/fracture_parameter_sensitivity/data/*.npz')
csv_files = glob.glob('output/fracture_parameter_sensitivity/timeseries_csv/*.csv')

print(f'\nTotal NPZ files: {len(npz_files)}')
print(f'Total CSV files: {len(csv_files)}')

if npz_files:
    mtimes_npz = [os.path.getmtime(f) for f in npz_files]
    min_npz = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(min(mtimes_npz)))
    max_npz = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(max(mtimes_npz)))
    print(f'NPZ generation interval: {min_npz} to {max_npz} (duration: {max(mtimes_npz)-min(mtimes_npz):.1f}s)')

if csv_files:
    mtimes_csv = [os.path.getmtime(f) for f in csv_files]
    min_csv = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(min(mtimes_csv)))
    max_csv = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(max(mtimes_csv)))
    print(f'CSV generation interval: {min_csv} to {max_csv} (duration: {max(mtimes_csv)-min(mtimes_csv):.1f}s)')

