# -*- coding: utf-8 -*-
"""
moc_simulate.v2

【生产级】物理自洽 MOC_V2 正演求解内核与智能反演基座
全面集成 Step 1~4 物理升级特性：
1. 稳态自洽流场空间积分与死端零虚假激波
2. 地质尺度宏观顺应性流体储能
3. 限流射孔非线性二次节流压降牛顿迭代器
4. 现场斜坡关泵动力学与波前平滑
5. 倒谱 1D/2D 空间连续特征与 HDF5 大规模数据集生成管道
"""
from __future__ import annotations

from moc_simulate.v2.configs import (
    WellboreConfig,
    FractureConfig,
    PerforationConfig,
    BoundaryConfig,
    SimulationConfig,
    MocV2Config,
)
from moc_simulate.v2.core import (
    MocGrid,
    compute_riemann_invariants,
    update_internal_nodes,
    reynolds,
    darcy_friction_factor,
    friction_term_J,
    brunone_k_vec,
    brunone_friction_Ju,
    compute_steady_state_field,
    solve_physical_steady_state,
    InfeasibleSteadyStateError,
    solve_fracture_node_v2,
    compute_ramp_velocity,
    compute_ramp_acceleration,
    apply_wellhead_bc,
    apply_toe_bc,
    WellboreMocV2Solver,
    simulate_v2,
    simulate_wellbore_v2,
)
from moc_simulate.v2.signal import (
    real_cepstrum,
    quefrency_to_distance,
    distance_to_quefrency,
    compute_cepstrum_1d,
    compute_cepstrogram_2d,
    apply_wavefront_derivative_filter,
    suppress_cluster_harmonics,
    detect_fracture_peaks,
    evaluate_peak_matching,
    compute_psnr,
)
from moc_simulate.v2.batch import (
    LatinHypercubeSampler,
    LhsSamplingBounds,
    BatchRunner,
    save_hdf5_dataset,
    load_hdf5_dataset,
)

__all__ = [
    # 配置
    "WellboreConfig",
    "FractureConfig",
    "PerforationConfig",
    "BoundaryConfig",
    "SimulationConfig",
    "MocV2Config",
    # 核心力学与求解器
    "MocGrid",
    "compute_riemann_invariants",
    "update_internal_nodes",
    "reynolds",
    "darcy_friction_factor",
    "friction_term_J",
    "brunone_k_vec",
    "brunone_friction_Ju",
    "compute_steady_state_field",
    "solve_physical_steady_state",
    "InfeasibleSteadyStateError",
    "solve_fracture_node_v2",
    "compute_ramp_velocity",
    "compute_ramp_acceleration",
    "apply_wellhead_bc",
    "apply_toe_bc",
    "WellboreMocV2Solver",
    "simulate_v2",
    "simulate_wellbore_v2",
    # 信号分析
    "real_cepstrum",
    "quefrency_to_distance",
    "distance_to_quefrency",
    "compute_cepstrum_1d",
    "compute_cepstrogram_2d",
    "apply_wavefront_derivative_filter",
    "suppress_cluster_harmonics",
    "detect_fracture_peaks",
    "evaluate_peak_matching",
    "compute_psnr",
    # 批处理与数据集
    "LatinHypercubeSampler",
    "LhsSamplingBounds",
    "BatchRunner",
    "save_hdf5_dataset",
    "load_hdf5_dataset",
]
