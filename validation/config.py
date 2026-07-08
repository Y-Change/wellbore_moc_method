# -*- coding: utf-8 -*-
"""
validation/config.py — 集中管理验证脚本的物理参数、仿真参数、缝形态与摩阻配置。

供 leakoff_multi.py / kaiser_bessel_multi.py / wlen_sweep.py 等验证脚本共享。
修改本文件即可全局调整所有验证的物理与仿真设定。
"""
from __future__ import annotations

# ── 井参数 ────────────────────────────────────────────────
WELL_CONFIG = {
    'L': 5000.0,                # 井筒长度 [m]
    'wellbore_diameter': 0.1397,  # 内径 [m]
    'fluid_density': 1000.0,    # 密度 [kg/m³]
    'fluid_viscosity': 1.0e-6,  # 运动黏度 [m²/s]
    'wavespeed': 1450.0,        # 波速 [m/s]
    'roughness_height': 4.5e-5, # 粗糙度 [m]
    'V0': 1.0,                  # 初始流速 [m/s]
    'H0': 300.0,                # 初始水头 [m]
    'theta': 0.0,               # 井斜角 [rad]
}

# ── 仿真参数 ──────────────────────────────────────────────
SIM_CONFIG = {
    'ts': 1.0,       # 停泵时刻 [s]
    'dt': 1.0e-3,    # 时间步长 [s]
    'tf': 100.0,     # 总仿真时长 [s]
}

# ── 裂缝参数 ──────────────────────────────────────────────
FRACTURE_CONFIG = {
    'Cf': 1.0e-5,       # 缝柔度 [m²]
    'kleak': 0.0001,    # 滤失系数 [m²/s/√m]
    'H_ext': 100.0,     # 地层孔隙压力水头 [m]
}

# ── 缝形态配置（单/双/三/四/五缝）─────────────────────────
CASES = {
    'single':  {'label': '单缝', 'x_f_list': [4300.0]},
    'dual':    {'label': '双缝', 'x_f_list': [4300.0, 4600.0]},
    'triple':  {'label': '三缝', 'x_f_list': [4100.0, 4300.0, 4500.0]},
    'quad':    {'label': '四缝', 'x_f_list': [4100.0, 4300.0, 4500.0, 4700.0]},
    'quint':   {'label': '五缝', 'x_f_list': [3700.0, 3900.0, 4100.0, 4300.0, 4500.0]},
}

# ── 摩阻模型配置 ──────────────────────────────────────────
# judgment4: steady → smoothness（抖动比 < 2.0）
#            brunone → oscillation_decay（RMS 衰减比 < 0.5）
# stab_factor: 判定 5 长期稳定性阈值因子 (H_range < stab_factor × ΔH)
FRICTION_PARAMS = {
    'steady': {
        'friction_model': 'steady',
        'label': '稳态达西+滤失',
        'judgment4': 'smoothness',
        'stab_factor': 1.0,
    },
    'brunone': {
        'friction_model': 'brunone',
        'label': 'Brunone非定常+滤失',
        'judgment4': 'oscillation_decay',
        'stab_factor': 0.5,
    },
}
