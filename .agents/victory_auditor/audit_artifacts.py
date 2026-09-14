import os
import json
import numpy as np
import pandas as pd
from PIL import Image

print('=== PHASE A: ARTIFACT & DELIVERABLES VERIFICATION ===')

# 1. Manifest verification
manifest_path = 'output/fracture_parameter_sensitivity/manifest.json'
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)

cases = manifest.get('cases', [])
print(f'Manifest cases count: {len(cases)}')
assert len(cases) == 84, f'Expected 84 cases, found {len(cases)}'

steady_cases = [c for c in cases if c.get('friction') == 'steady']
brunone_cases = [c for c in cases if c.get('friction') == 'brunone']
print(f'Steady cases: {len(steady_cases)}, Brunone cases: {len(brunone_cases)}')
assert len(steady_cases) == 42, f'Expected 42 steady cases, got {len(steady_cases)}'
assert len(brunone_cases) == 42, f'Expected 42 Brunone cases, got {len(brunone_cases)}'

# Check groups
groups = set(c.get('group') for c in cases)
print(f'Manifest groups: {sorted(list(groups))}')

# 2. NPZ schema and numerical sanity
data_dir = 'output/fracture_parameter_sensitivity/data'
csv_dir = 'output/fracture_parameter_sensitivity/timeseries_csv'

required_keys = [
    't', 'H_wh', 'Q_wh', 'x_f_aligned', 'compliance_head_m2', 
    'kleak_equiv', 'Rp', 'inflow_weight', 'friction', 'schema_version'
]

nan_inf_found = 0
keys_missing_count = 0
total_npz_checked = 0
total_csv_checked = 0

for c in cases:
    cname = c.get('case_name') or ('case_%05d' % c['case_id'])
    npz_path = os.path.join(data_dir, cname + '.npz')
    csv_path = os.path.join(csv_dir, cname + '.csv')
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f'Missing NPZ: {npz_path}')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Missing CSV: {csv_path}')
        
    data = np.load(npz_path, allow_pickle=True)
    total_npz_checked += 1
    
    # Check schema
    for rk in required_keys:
        if rk not in data.files:
            keys_missing_count += 1
            print(f'Missing key {rk} in {cname}')
            
    # Check schema version
    ver = str(data['schema_version'])
    if 'moc_lhs_v2.1' not in ver:
        print(f'Unexpected schema_version: {ver} in {cname}')
        
    # Check NaN / Inf
    H_wh = data['H_wh']
    Q_wh = data['Q_wh']
    if not np.all(np.isfinite(H_wh)) or not np.all(np.isfinite(Q_wh)):
        nan_inf_found += 1
        print(f'NaN/Inf detected in {cname}')
        
    # Check CSV
    df = pd.read_csv(csv_path)
    total_csv_checked += 1
    if df.shape[0] < 100 or df.shape[1] < 3:
        print(f'Abnormal CSV shape for {cname}: {df.shape}')
    if df.isna().sum().sum() > 0:
        nan_inf_found += 1
        print(f'NaN detected in CSV {cname}')

print(f'NPZ checked: {total_npz_checked}/84, CSV checked: {total_csv_checked}/84')
print(f'Missing keys count: {keys_missing_count}')
print(f'NaN/Inf occurrences: {nan_inf_found}')
assert keys_missing_count == 0, 'Schema keys missing in NPZ'
assert nan_inf_found == 0, 'NaN/Inf detected in simulation outputs'

# 3. Metrics files verification
metrics_csv = 'output/fracture_parameter_sensitivity/tables/sensitivity_metrics.csv'
summary_json = 'output/fracture_parameter_sensitivity/tables/sensitivity_summary.json'

assert os.path.exists(metrics_csv), 'sensitivity_metrics.csv missing'
assert os.path.exists(summary_json), 'sensitivity_summary.json missing'

df_metrics = pd.read_csv(metrics_csv)
print(f'Metrics CSV shape: {df_metrics.shape} (84 rows x {df_metrics.shape[1]} columns)')
assert len(df_metrics) == 84, f'Expected 84 rows in metrics CSV, got {len(df_metrics)}'

with open(summary_json, 'r', encoding='utf-8') as f:
    summ = json.load(f)
print(f'Summary JSON top keys: {list(summ.keys())}')
if 'sensitivity_rankings' in summ:
    print('Sensitivity ranking in summary JSON: PRESENT')
    print('Rankings keys:', list(summ['sensitivity_rankings'].keys()))
else:
    print('Sensitivity ranking in summary JSON: MISSING!')
    raise AssertionError('sensitivity_rankings missing in summary JSON')

# 4. Publication figures verification
figures = [
    'fig1_wavefront_step_gradient',
    'fig2_envelope_rms_decay',
    'fig3_frequency_spectral_dissipation',
    'fig4_cepstrum_rayleigh_resolution'
]
fig_dir = 'output/fracture_parameter_sensitivity/figures'

for fig in figures:
    png_path = os.path.join(fig_dir, fig + '.png')
    svg_path = os.path.join(fig_dir, fig + '.svg')
    assert os.path.exists(png_path), f'PNG figure missing: {png_path}'
    assert os.path.exists(svg_path), f'SVG figure missing: {svg_path}'
    
    with Image.open(png_path) as im:
        dpi = im.info.get('dpi', (None, None))
        width, height = im.size
        print(f'{fig}: Size=({width}x{height}), DPI={dpi}, PNG bytes={os.path.getsize(png_path)}, SVG bytes={os.path.getsize(svg_path)}')
        if dpi[0] is not None:
            assert dpi[0] >= 200, f'{fig} DPI {dpi[0]} is below 200'

# 5. README.md verification
readme_path = 'output/fracture_parameter_sensitivity/README.md'
assert os.path.exists(readme_path), 'README.md missing'
with open(readme_path, 'r', encoding='utf-8') as f:
    readme_content = f.read()

print(f'README.md length: {len(readme_content)} characters, {len(readme_content.splitlines())} lines')
assert len(readme_content) > 5000, 'README.md is too short'

required_terms = [
    'Joukowsky', 'Brunone', 'Darcy', 'Rayleigh', 'cepstrum', 
    'sensitivity', 'neural', 'damping'
]
missing_terms = [t for t in required_terms if t.lower() not in readme_content.lower()]
print(f'Missing required terms in README.md: {missing_terms}')
assert len(missing_terms) == 0, f'README.md missing required concepts: {missing_terms}'

print('\n=== PHASE A RESULT: ALL 5 DELIVERABLES 100% VERIFIED ===')
