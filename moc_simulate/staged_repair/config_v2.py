# -*- coding: utf-8 -*-
"""
moc_simulate.staged_repair.config_v2 - 沙箱配置与参数定义

专供 staged_repair 沙箱使用，严格与原生产配置隔离。
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

# 工作区根目录
REPO_ROOT = Path(__file__).resolve().parents[2]

# Stage 1 专用物理与仿真基准配置
STAGE1_CONFIG = {
    # 几何与网格
    "wellbore_length": 5000.0,           # [m]
    "wellbore_diameter": 0.1397,         # [m] (5.5" 套管)
    "roughness_height": 4.5e-5,          # [m] 商用钢管粗糙度
    # 流体特性
    "fluid_density": 1000.0,             # [kg/m^3]
    "fluid_viscosity": 1.0e-6,           # [m^2/s]
    "wavespeed": 1450.0,                 # [m/s]
    # 摩阻
    "friction_model": "brunone",         # Brunone 非定常摩阻
    "brunone_k_scale": 1.0,
    # 时间控制
    "dt": 1.0e-3,                        # [s]
    "tf": 100.0,                         # [s]
    "pump_shut_time": 1.0,               # [s] 停泵时刻
    "pump_closure_duration": 1.0e-3,     # [s]
    # 边界与初始场
    "wellhead_bc": "velocity_step",      # 停泵瞬时截流
    "initial_velocity": 1.0,             # [m/s]
    "initial_head": 300.0,               # [m]
    "theta": 0.0,                        # 水平井
    "toe_bc": "dead_end",                # 阶段一严格锁定死端 (Gamma = +1.0)
    "toe_head": 300.0,
    # 裂缝系统 (quad, 间距 10m)
    "fracture_positions": [4100.0, 4110.0, 4120.0, 4130.0],
    "Cf": 1.0e-5,                        # [m^2]
    "kleak": 1.0e-4,                     # [m^{5/2}/s]
    "H_ext": 100.0,                      # [m] 地层孔隙水头
}

# 基准文件路径 (PaperA 03_leakoff验证/brunone_D10/quad)
PAPERA_BENCHMARK_DIR = REPO_ROOT / "PaperA井口多裂缝水击响应" / "03_leakoff验证" / "brunone_D10" / "quad"
PAPERA_BENCHMARK_CSV = PAPERA_BENCHMARK_DIR / "moc_timeseries.csv"
PAPERA_BENCHMARK_JSON = PAPERA_BENCHMARK_DIR / "moc_leakoff.json"

# Stage 1 成果产出目录
STAGE1_OUTPUT_DIR = REPO_ROOT / "output" / "staged_repair" / "stage_1"
