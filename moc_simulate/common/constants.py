# -*- coding: utf-8 -*-
"""
moc_simulate.common.constants

标准物理常数与基准参数定义
"""
from __future__ import annotations

# 重力加速度 [m/s^2]
# ISO 80000-3 / CODATA 标准重力加速度
G_STANDARD: float = 9.80665

# 工程水锤及自研 MOC 基准所采用的实用重力加速度
G: float = 9.81
GRAVITY: float = G

# 清水与压裂液参考物性
RHO_WATER: float = 1000.0         # 清水密度 [kg/m^3]
NU_WATER: float = 1.0e-6          # 运动粘度 [m^2/s] (20°C 清水)
MU_WATER: float = 1.0e-3          # 动力粘度 [Pa*s]

# 经典套管几何与材料属性
CASING_ROUGHNESS: float = 4.5e-5  # 典型钢制套管绝对粗糙度 [m] (0.045 mm)
CASING_OD_5_5: float = 0.1397     # 5.5 英寸套管外径/内径参考 [m]
WAVESPEED_DEFAULT: float = 1450.0 # 压裂井筒典型声学波速 [m/s]

# 标准大气参数
P_ATM: float = 101325.0           # 标准大气压 [Pa]
H_ATM: float = 10.33              # 大气压当量清水水头 [m]

# 封闭趾端压裂井口最大工作水头 [m]
# 约 294 MPa (P = ρ g H)。足以覆盖单簇 + 默认 k_leak=1e-4 的研究算例 (~24 km)，
# 同时切断现有 pilot 中 46~60 km 的非物理长尾。采样器可再收紧 h0_max。
H0_MAX_WORKING: float = 30000.0
