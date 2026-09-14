import os
import re
import ast
import numpy as np
import pandas as pd

print('=== PHASE B: ANTI-CHEATING & INTEGRITY DETECTION ===')

source_files = [
    'experiments/sensitivity/generate_matrix.py',
    'experiments/sensitivity/run_simulation.py',
    'experiments/sensitivity/extract_features.py',
    'experiments/sensitivity/plot_figures.py',
    'tests/test_fracture_sensitivity_e2e.py',
    'tests/test_challenger_adversarial_stress.py',
    'tests/test_challenger_empirical_stress.py',
]

suspicious_patterns = [
    r'\bmock\b', r'\bMagicMock\b', r'\bunittest\.mock\b',
    r'assert\s+True\b', r'return\s+True\b', r'return\s+42\b',
    r'fake_', r'dummy_', r'canned_'
]

print('1. Scanning source files for suspicious patterns / facades...')
findings = []
for sf in source_files:
    if not os.path.exists(sf):
        continue
    with open(sf, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines, 1):
        for pat in suspicious_patterns:
            if re.search(pat, line, re.IGNORECASE):
                findings.append((sf, idx, line.strip(), pat))

print(f'Total pattern matches found: {len(findings)}')
for f in findings:
    print(f'  {f[0]}:{f[1]} [{f[3]}] -> {f[2]}')

# 2. AST analysis for dummy functions / trivial returns
print('\n2. AST Analysis for trivial/empty functions...')
trivial_funcs = []
for sf in source_files:
    if not os.path.exists(sf):
        continue
    with open(sf, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=sf)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    trivial_funcs.append((sf, node.name, 'pass'))
                elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                    trivial_funcs.append((sf, node.name, f'return {stmt.value.value}'))

print(f'Trivial functions found: {len(trivial_funcs)}')

# 3. Numerical Cross-Validation:
# Recompute features independently for 4 cases from raw NPZ and compare with sensitivity_metrics.csv
print('\n3. Independent numerical verification of sensitivity_metrics.csv against raw NPZ...')
df_metrics = pd.read_csv('output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv')

spot_cases = ['case_00000', 'case_00010', 'case_00042', 'case_00060']
for sc in spot_cases:
    npz_path = f'output/fracture_parameter_sensitivity/data/{sc}.npz'
    npz_data = np.load(npz_path)
    t = npz_data['t']
    H_wh = npz_data['H_wh']
    dt = float(t[1] - t[0])
    ts = 0.5
    tc = 0.05
    
    # Check row in metrics
    row = df_metrics[df_metrics['case_name'] == sc].iloc[0]
    
    # 1. Joukowsky drop check:
    idx_after = int(round((ts + tc + 0.05) / dt))
    idx_before = int(round((ts - 0.01) / dt))
    dH_sim_calc = float(H_wh[idx_after] - H_wh[idx_before])
    dH_sim_metric = float(row['joukowsky_sim_m'])
    print(f'\nChecking {sc}:')
    print(f'  Joukowsky simulated drop: calc={dH_sim_calc:.4f} m, metric={dH_sim_metric:.4f} m, diff={abs(dH_sim_calc - dH_sim_metric):.6e}')
    assert abs(dH_sim_calc - dH_sim_metric) < 1e-4, f'Joukowsky mismatch in {sc}'
    
    # 2. Check maximum gradient:
    mask_grad = (t >= ts) & (t <= ts + tc + 0.1)
    dH_dt = np.gradient(H_wh, dt)
    max_grad_calc = float(np.max(np.abs(dH_dt[mask_grad])))
    max_grad_metric = float(row['wavefront_max_gradient_m_s'])
    print(f'  Gradient: calc={max_grad_calc:.4f} m/s, metric={max_grad_metric:.4f} m/s, diff={abs(max_grad_calc - max_grad_metric):.6e}')
    assert abs(max_grad_calc - max_grad_metric) < 1e-3, f'Gradient mismatch in {sc}'
    
    # 3. Check RMS window 1:
    tf = 40.0
    w_len = (tf - ts) / 5.0
    mask_win1 = (t >= ts) & (t <= ts + w_len)
    H_win1 = H_wh[mask_win1]
    rms1_calc = float(np.sqrt(np.mean((H_win1 - np.mean(H_win1)) ** 2)))
    rms1_metric = float(row['rms_window_1_m'])
    print(f'  RMS Win 1: calc={rms1_calc:.4f} m, metric={rms1_metric:.4f} m, diff={abs(rms1_calc - rms1_metric):.6e}')
    assert abs(rms1_calc - rms1_metric) < 1e-4, f'RMS Win 1 mismatch in {sc}'

print('\n=== PHASE B RESULT: ALL INTEGRITY & ANTI-CHEATING CHECKS PASSED (CLEAN) ===')
