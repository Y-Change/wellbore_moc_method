# -*- coding: utf-8 -*-
"""
experiments/exp_shutdown_friction_2d_cepstrum/run_simulation.py
--------------------------------------------------------------
运行 5000m MOC 井筒仿真：
- 3 条裂缝：起点 3000m，间距 50m (3000, 3050, 3100 m)
- 4 种停泵历时：Tc in [0.01, 0.1, 0.5, 1.0] s
- 2 种摩阻模型：steady (稳态达西), brunone (非定常摩阻)
共 8 组工况。
"""
from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# 加入项目根目录
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(_curr_dir))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore
from moc_simulate.config import WELL_CONFIG, SIM_CONFIG, FRACTURE_CONFIG

OUTPUT_BASE = os.path.join(_root_dir, 'output', 'exp_shutdown_friction_2d_cepstrum')
DATA_DIR = os.path.join(OUTPUT_BASE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 实验参数定义
WELLBORE_LENGTH = 5000.0       # 井长 [m]
WAVESPEED = 1450.0             # 波速 [m/s]
DIAMETER = 0.1397              # 内径 [m]
V0 = 1.0                       # 初始流速 [m/s]
H0 = 300.0                     # 初始水头 [m]
TS = 1.0                       # 停泵开始时间 [s]
TF = 50.0                      # 仿真总时间 [s]
DT = 1.0e-3                    # 时间步长 [s]

# 裂缝参数
FRACTURE_POSITIONS = [3000.0, 3050.0, 3100.0]  # 3 条裂缝
N_FRAC = len(FRACTURE_POSITIONS)
CF_LIST = [1.0e-5] * N_FRAC                     # 缝柔度 [m²]
KLEAK_LIST = [1.0e-4] * N_FRAC                  # 滤失系数 [m^{5/2}/s]
H_EXT = 100.0                                   # 外部地层水头 [m]

# 变量列表
TC_LIST = [0.01, 0.1, 0.5, 1.0]                # 停泵时间 [s]
FRICTION_LIST = ['steady', 'brunone']           # 摩阻模型


def run_all_cases():
    cases_meta = []
    print("=" * 70)
    print("开始执行 5000m MOC 三裂缝停泵时间与摩阻形态仿真实验 (8 工况)")
    print(f"裂缝位置: {FRACTURE_POSITIONS} m (间距 50 m)")
    print(f"停泵时间: {TC_LIST} s")
    print(f"摩阻类型: {FRICTION_LIST}")
    print(f"总仿真时间: {TF} s, 时间步长: {DT} s")
    print("=" * 70)

    for tc in TC_LIST:
        for fric in FRICTION_LIST:
            tc_str = str(tc).replace('.', 'p')
            case_name = f"tc_{tc_str}s_{fric}"
            case_dir = os.path.join(DATA_DIR, case_name)
            os.makedirs(case_dir, exist_ok=True)
            csv_path = os.path.join(case_dir, 'moc_timeseries.csv')
            meta_path = os.path.join(case_dir, 'meta.json')

            print(f"\n[运行中] 工况: {case_name} (Tc={tc}s, 摩阻={fric})...")

            cfg = MocConfig(
                wellbore_length=WELLBORE_LENGTH,
                wellbore_diameter=DIAMETER,
                fluid_density=WELL_CONFIG['fluid_density'],
                fluid_viscosity=WELL_CONFIG['fluid_viscosity'],
                wavespeed=WAVESPEED,
                roughness_height=WELL_CONFIG['roughness_height'],
                friction_model=fric,
                dt=DT,
                tf=TF,
                wellhead_bc='ramp',
                pump_shut_time=TS,
                pump_closure_duration=tc,
                initial_velocity=V0,
                initial_head=H0,
                theta=0.0,
                toe_bc='reservoir',
                toe_head=H0,
            )

            t0 = time.time()
            res = simulate_wellbore(
                cfg,
                fracture_positions=FRACTURE_POSITIONS,
                fracture_Cf=CF_LIST,
                fracture_kleak=KLEAK_LIST,
                H_ext=H_EXT,
                store_full_field=False,
            )
            elapsed = time.time() - t0
            print(f"  -> 仿真完成，耗时: {elapsed:.2f} s")

            t_arr = res["timestamps"]
            H_wh = res["wellhead_head"]
            V_wh = res["wellhead_velocity"]
            Q_wh = V_wh * cfg.area

            # 保存 CSV
            df = pd.DataFrame({
                't': t_arr,
                'H_wh': H_wh,
                'V_wh': V_wh,
                'Q_wh': Q_wh,
            })
            df.to_csv(csv_path, index=False)
            print(f"  -> 时程数据已保存: {csv_path} (行数: {len(df)})")

            # 保存元数据
            meta = {
                'case_name': case_name,
                'Tc': tc,
                'friction': fric,
                'fracture_positions': FRACTURE_POSITIONS,
                'fracture_indices': res['fracture_indices'],
                'N_grid': cfg.N,
                'dx': cfg.dx,
                'a_adj': cfg.a_adj,
                'elapsed_s': elapsed,
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            cases_meta.append(meta)

    # 导出总清单
    manifest_path = os.path.join(DATA_DIR, 'cases_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(cases_meta, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print(f"全部 8 组工况仿真完成！数据已统一归档至: {DATA_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    run_all_cases()
