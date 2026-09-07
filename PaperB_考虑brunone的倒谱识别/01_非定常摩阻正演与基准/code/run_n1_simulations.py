# -*- coding: utf-8 -*-
"""
Script to run n=1 constant k simulations (k in {0, 0.01, 0.02, 0.05})
and extract four clocks (t_onset, t_peak, t_E50, tau_cep) for both n=1 and n=4 (D=20m).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import fft, ifft

_d = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_d))
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.config import WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG
import moc_simulate.wellbore_moc as wm

output_base = os.path.join(_root, 'output', 'analysis', 'brunone_spacing_effect')
os.makedirs(output_base, exist_ok=True)

k_values = [0, 0.01, 0.02, 0.05]
FRAC_FIRST_M = 4100.0
WAVESPEED = 1450.0

def patch_brunone(k_val):
    def mock_k(Re: float) -> float:
        return k_val
    def mock_k_vec(Re_arr: np.ndarray) -> np.ndarray:
        return np.full_like(Re_arr, k_val, dtype=np.float64)
    wm.brunone_k = mock_k
    wm.brunone_k_vec = mock_k_vec

def run_n1_simulations():
    original_brunone_k = wm.brunone_k
    original_brunone_k_vec = wm.brunone_k_vec

    for k in k_values:
        case_dir = os.path.join(output_base, f'n1_k{k}')
        os.makedirs(case_dir, exist_ok=True)
        csv_path = os.path.join(case_dir, 'moc_timeseries.csv')
        
        if os.path.exists(csv_path):
            print(f"n=1, k={k} already exists at {csv_path}, skipping simulation.")
            continue
            
        print(f"Running n=1, k={k}...")
        fric_model = 'steady' if k == 0 else 'brunone'
        if k > 0:
            patch_brunone(k)
            
        cfg = MocConfig(
            wellbore_length=WELL_CONFIG['L'],
            wellbore_diameter=WELL_CONFIG['wellbore_diameter'],
            fluid_density=WELL_CONFIG['fluid_density'],
            fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
            wavespeed=WAVESPEED,
            roughness_height=WELL_CONFIG['roughness_height'],
            friction_model=fric_model,
            dt=0.001,
            tf=50.0,
            wellhead_bc='velocity_step',
            pump_shut_time=1.0,
            initial_velocity=WELL_CONFIG['V0'],
            initial_head=WELL_CONFIG['H0'],
            theta=WELL_CONFIG['theta'],
            toe_bc='reservoir',
            toe_head=WELL_CONFIG['H0'],
        )
        
        x_f_list = [FRAC_FIRST_M]
        Cf = FRACTURE_CONFIG['Cf']
        kleak = FRACTURE_CONFIG['kleak']
        H_ext = FRACTURE_CONFIG['H_ext']
        
        t0 = time.time()
        res = simulate_wellbore(
            cfg,
            fracture_positions=x_f_list,
            fracture_Cf=[Cf],
            fracture_kleak=[kleak],
            H_ext=H_ext,
            store_full_field=False,
        )
        print(f"  Done in {time.time()-t0:.2f}s")
        
        t_arr = res["timestamps"]
        H_wh = res["wellhead_head"]
        V_wh = res["wellhead_velocity"]
        Q_wh = V_wh * cfg.area
        
        df = pd.DataFrame({'t': t_arr, 'H_wh': H_wh, 'Q_wh': Q_wh})
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")
        
        wm.brunone_k = original_brunone_k
        wm.brunone_k_vec = original_brunone_k_vec

if __name__ == '__main__':
    run_n1_simulations()
