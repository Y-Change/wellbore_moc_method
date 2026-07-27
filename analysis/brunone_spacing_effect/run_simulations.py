import os
import sys
import time
import numpy as np
import pandas as pd

_d = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_d))
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.config import WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG
import moc_simulate.wellbore_moc as wm

output_dir = os.path.join(_root, 'output', 'analysis', 'brunone_spacing_effect')
os.makedirs(output_dir, exist_ok=True)

spacings = [5, 10, 20, 50, 100]
k_values = [0, 0.01, 0.02, 0.05, 0.1, 0.2]

FRAC_FIRST_M = 4100.0

def patch_brunone(k_val):
    def mock_k(Re: float) -> float:
        return k_val
    def mock_k_vec(Re_arr: np.ndarray) -> np.ndarray:
        return np.full_like(Re_arr, k_val, dtype=np.float64)
    
    wm.brunone_k = mock_k
    wm.brunone_k_vec = mock_k_vec

def run_all():
    original_brunone_k = wm.brunone_k
    original_brunone_k_vec = wm.brunone_k_vec

    for spacing in spacings:
        for k in k_values:
            case_dir = os.path.join(output_dir, f'D{spacing}_k{k}')
            os.makedirs(case_dir, exist_ok=True)
            csv_path = os.path.join(case_dir, 'moc_timeseries.csv')
            
            if os.path.exists(csv_path):
                print(f"Skipping D={spacing}, k={k}, already exists.")
                continue
                
            print(f"Running D={spacing}, k={k}...")
            
            fric_model = 'steady' if k == 0 else 'brunone'
            if k > 0:
                patch_brunone(k)
                
            cfg = MocConfig(
                wellbore_length=WELL_CONFIG['L'],
                wellbore_diameter=WELL_CONFIG['wellbore_diameter'],
                fluid_density=WELL_CONFIG['fluid_density'],
                fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
                wavespeed=WELL_CONFIG['wavespeed'],
                roughness_height=WELL_CONFIG['roughness_height'],
                friction_model=fric_model,
                dt=SIM_CONFIG['dt'],
                tf=SIM_CONFIG['tf'],
                wellhead_bc='velocity_step',
                pump_shut_time=SIM_CONFIG['ts'],
                initial_velocity=WELL_CONFIG['V0'],
                initial_head=WELL_CONFIG['H0'],
                theta=WELL_CONFIG['theta'],
                toe_bc='reservoir',
                toe_head=WELL_CONFIG['H0'],
            )
            
            x_f_list = [FRAC_FIRST_M + i * spacing for i in range(4)]
            Cf = FRACTURE_CONFIG['Cf']
            kleak = FRACTURE_CONFIG['kleak']
            H_ext = FRACTURE_CONFIG['H_ext']
            
            t0 = time.time()
            res = simulate_wellbore(
                cfg,
                fracture_positions=x_f_list,
                fracture_Cf=[Cf] * 4,
                fracture_kleak=[kleak] * 4,
                H_ext=H_ext,
                store_full_field=False,
            )
            print(f"  Simulation done in {time.time()-t0:.2f}s")
            
            t_arr = res["timestamps"]
            H_wh = res["wellhead_head"]
            V_wh = res["wellhead_velocity"]
            Q_wh = V_wh * cfg.area
            
            df = pd.DataFrame({'t': t_arr, 'H_wh': H_wh, 'Q_wh': Q_wh})
            df.to_csv(csv_path, index=False)
            print(f"  Saved to {csv_path}")
            
            # Restore original
            wm.brunone_k = original_brunone_k
            wm.brunone_k_vec = original_brunone_k_vec

if __name__ == '__main__':
    run_all()
