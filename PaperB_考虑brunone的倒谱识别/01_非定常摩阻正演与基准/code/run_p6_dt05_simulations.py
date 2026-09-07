# -*- coding: utf-8 -*-
"""
Gantt p6: Forward MOC simulation on refined grid (dt = 0.5 ms).
Simulates n=1, X1=4100m, Cf=1.0e-5 m^2, kleak=1.0e-4 m^2/s/sqrt(m), H_ext=100.0m
for k=0.0 (Steady Darcy) and k=0.01 (Brunone IAB) with velocity_step shut-in.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

_root = r"e:\water_hammer_research\wellbore_moc_method"
if _root not in sys.path:
    sys.path.insert(0, _root)

from moc_simulate.wellbore_moc import (
    MocConfig, reynolds, darcy_friction_factor, friction_term_J,
    brunone_k, solve_fracture_node
)
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG

sim_base = os.path.join(_root, "output", "analysis", "brunone_spacing_effect", "p6_robustness")
os.makedirs(sim_base, exist_ok=True)

WAVESPEED = 1450.0
FRAC_X1 = 4100.0
CF = 1.0e-5
KLEAK = 1.0e-4
TS = 1.0
TF = 50.0
DT = 0.0005  # 0.5 ms refined time step
G = 9.80665

def simulate_dt05(k_val: float, case_label: str):
    csv_path = os.path.join(sim_base, f"{case_label}.csv")
    if os.path.exists(csv_path):
        print(f"Skipping {case_label}, already exists: {csv_path}")
        return

    print(f"Running simulation {case_label} (dt=0.5ms, k={k_val}, tf=50s)...")
    t0 = time.time()

    cfg = MocConfig(
        wellbore_length=WELL_CONFIG['L'],
        wellbore_diameter=WELL_CONFIG['wellbore_diameter'],
        fluid_density=WELL_CONFIG['fluid_density'],
        fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
        wavespeed=WAVESPEED,
        roughness_height=WELL_CONFIG['roughness_height'],
        friction_model="steady" if k_val == 0.0 else "brunone",
        dt=DT,
        tf=TF,
        wellhead_bc="velocity_step",
        pump_shut_time=TS,
        initial_velocity=WELL_CONFIG['V0'],
        initial_head=WELL_CONFIG['H0'],
        theta=WELL_CONFIG['theta'],
        toe_bc="reservoir",
        toe_head=WELL_CONFIG['H0'],
    )

    N = cfg.N
    dt = cfg.dt
    dx = cfg.dx
    a = cfg.a_adj
    A = cfg.area
    D = cfg.wellbore_diameter
    nu = cfg.fluid_viscosity
    K_D = cfg.roughness_height / D
    n_steps = cfg.n_steps
    ga = G / a
    t_s = cfg.pump_shut_time
    V0 = cfg.initial_velocity
    H0 = cfg.initial_head

    Re0 = reynolds(V0, D, nu)
    f_steady = darcy_friction_factor(Re0, K_D, "steady")

    H = np.full(N + 1, H0, dtype=np.float64)
    V = np.full(N + 1, V0, dtype=np.float64)
    H[:-1] = cfg.toe_head + (f_steady * V0 * abs(V0) / (2.0 * G * D)) * (cfg.wellbore_length - np.linspace(0.0, cfg.wellbore_length, N + 1)[:-1])

    i_f = int(round(FRAC_X1 / dx))

    V_prev_left = V.copy()
    V_prev_right = V.copy()
    V_prev2_left = V.copy()
    V_prev2_right = V.copy()
    H_prev = H.copy()

    timestamps = np.zeros(n_steps + 1)
    wh_head_hist = np.zeros(n_steps + 1)
    wh_head_hist[0] = H[0]

    V_smooth = 0.05
    is_brunone = (k_val > 0.0)

    for n in range(1, n_steps + 1):
        t = n * dt
        timestamps[n] = t

        V1 = V_prev_right[:-2]
        H1 = H_prev[:-2]
        V2 = V_prev_left[2:]
        H2 = H_prev[2:]

        J1 = friction_term_J(f_steady, D, V1, dt)
        J2 = friction_term_J(f_steady, D, V2, dt)

        if is_brunone and n >= 2:
            dVdt1 = (V_prev_right[:-2] - V_prev2_right[:-2]) / dt
            dVdx1 = (V_prev_right[1:-1] - V_prev_right[:-2]) / dx
            sign_V1 = np.tanh(V1 / V_smooth)
            Ju1 = (k_val / 2.0) * dt * (dVdt1 + a * sign_V1 * np.abs(dVdx1))

            dVdt2 = (V_prev_left[2:] - V_prev2_left[2:]) / dt
            dVdx2 = (V_prev_left[2:] - V_prev_left[1:-1]) / dx
            sign_V2 = np.tanh(V2 / V_smooth)
            Ju2 = (k_val / 2.0) * dt * (dVdt2 + a * sign_V2 * np.abs(dVdx2))

            Ju1[i_f - 1] = 0.0
            Ju2[i_f - 1] = 0.0
            J1 += Ju1
            J2 += Ju2

        Cp = V1 + ga * H1 - J1
        Cm = -V2 + ga * H2 + J2

        H_new = np.empty(N + 1, dtype=np.float64)
        V_new = np.empty(N + 1, dtype=np.float64)

        H_new[1:-1] = (Cp + Cm) / (2.0 * ga)
        V_new[1:-1] = Cp - ga * H_new[1:-1]

        H_w, H_f, V_left_f, V_right_f, _ = solve_fracture_node(
            Cp[i_f - 1], Cm[i_f - 1], H_prev[i_f], A, ga,
            CF, KLEAK, FRACTURE_CONFIG['H_ext'], dt
        )
        H_new[i_f] = H_w
        V_new[i_f] = V_left_f

        # Wellhead
        V2_0 = V_prev_left[1]
        H2_0 = H_prev[1]
        J_0 = friction_term_J(f_steady, D, V2_0, dt)
        if is_brunone and n >= 2:
            dVdt_0 = (V_prev_left[1] - V_prev2_left[1]) / dt
            dVdx_0 = (V_prev_left[1] - V_prev_left[0]) / dx
            J_0 += (k_val / 2.0) * dt * (dVdt_0 + a * np.sign(V2_0) * np.abs(dVdx_0))

        Cm_0 = -V2_0 + ga * H2_0 + J_0
        V_wh = 0.0 if t >= t_s else V0
        V_new[0] = V_wh
        H_new[0] = (V_wh + Cm_0) / ga

        # Toe
        V1_N = V_prev_right[N - 1]
        H1_N = H_prev[N - 1]
        J_N = friction_term_J(f_steady, D, V1_N, dt)
        if is_brunone and n >= 2:
            dVdt_N = (V_prev_right[N - 1] - V_prev2_right[N - 1]) / dt
            dVdx_N = (V_prev_right[N] - V_prev_right[N - 1]) / dx
            J_N += (k_val / 2.0) * dt * (dVdt_N + a * np.sign(V1_N) * np.abs(dVdx_N))

        Cp_N = V1_N + ga * H1_N - J_N
        H_new[N] = cfg.toe_head
        V_new[N] = Cp_N - ga * H_new[N]

        V_prev2_left[:] = V_prev_left
        V_prev2_right[:] = V_prev_right
        V_prev_left[:] = V_new
        V_prev_right[:] = V_new
        V_prev_left[i_f] = V_left_f
        V_prev_right[i_f] = V_right_f
        H_prev[:] = H_new
        wh_head_hist[n] = H_new[0]

    df_out = pd.DataFrame({'t': timestamps, 'H_wh': wh_head_hist})
    df_out.to_csv(csv_path, index=False)
    print(f"  Completed {case_label} in {time.time()-t0:.2f}s -> {csv_path}")

if __name__ == '__main__':
    simulate_dt05(0.0, "dt05_k0.0")
    simulate_dt05(0.01, "dt05_k0.01")
