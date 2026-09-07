# -*- coding: utf-8 -*-
"""
moc_simulate/lhs_config.py — 拉丁超立方采样 (LHS) 与 AI 进液反演核心参数配置
继承并扩展自 moc_simulate/config.py 的基础井筒与流体核心参数。
"""
from __future__ import annotations
from typing import Dict, Any

# 从基础配置中导入井筒物理常数与默认裂缝背景配置
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG, CEPSTRUM_CONFIG

# ── 单位换算与物理辅助函数 ──────────────────────────────────
def pressure_to_head_compliance(Cp_Pa: float, rho: float = 1000.0, g: float = 9.80665) -> float:
    """
    压力柔度 [m³/Pa] 转换为水头柔度 [m²]：
        C_H = ρ · g · C_p
    """
    return float(Cp_Pa * rho * g)


def bbl_psi_to_head_compliance(Cp_bbl_psi: float, rho: float = 1000.0, g: float = 9.80665) -> float:
    """
    现场常用柔度单位 [bbl/psi] 转换为水头柔度 [m²]：
        1 bbl ≈ 0.1589873 m³, 1 psi ≈ 6894.757 Pa
        1 bbl/psi ≈ 2.3059e-5 m³/Pa
    """
    Cp_m3_pa = Cp_bbl_psi * 0.1589873 / 6894.757
    return pressure_to_head_compliance(Cp_m3_pa, rho, g)


# ── 1. 仿真时间与步长配置 (独立设置 tf 为 50.0s) ────────────────
SIM_CONFIG: Dict[str, float] = {
    'ts': 1.0,       # 停泵时刻 [s]
    'dt': 1.0e-3,    # 时间步长 [s] (默认 1ms)
    'tf': 50.0,      # 总仿真时长 [s] (设定为 50s，覆盖约 7~8 次完整的井筒往返水击混响)
}

# ── 2. 拉丁超立方 (LHS) 物理参数采样区间配置 ──────────────────
LHS_PARAM_RANGES: Dict[str, Any] = {
    "n_frac_min": 1,               # 最少压裂簇数
    "n_frac_max": 6,               # 最多压裂簇数 (多簇穿孔压裂对标)
    "frac_zone_start": 3500.0,     # 缝网分布起始井深 [m] (中深层水平段)
    "frac_zone_end": 4800.0,       # 缝网分布结束井深 [m] (最大井深 L=5000m)
    "min_spacing": 5.0,            # 最小簇间距 [m] (硬约束，防止物理空间重叠)
    "max_spacing": 20.0,           # 最大簇间距 [m] (硬约束，相邻簇间距落在 [min_spacing, max_spacing])
    
    # 水头柔度 compliance_head_m2 [m²]：对数均匀采样
    # 锚定基准值 10^-5 m² (约占井筒储液 2.8%)，覆盖 [10^-6, 10^-4] m² (0.28% ~ 28%)
    # 具有明确且可穿透的水击反射响应特征
    "cf_head_m2_log_min": -6.0,    # 10^(-6.0) = 1.0e-6 m²
    "cf_head_m2_log_max": -4.0,    # 10^(-4.0) = 1.0e-4 m²
    "cf_log_min": -6.0,            # 向后兼容键
    "cf_log_max": -4.0,            # 向后兼容键
    
    # 封闭趾端分流模式：各簇稳态进液比例 w_i 采用 Dirichlet 分布采样 (∑w_i = 1)
    "alpha_dirichlet": 1.0,        # 对称狄利克雷分布浓度参数 (1.0 = 单纯形均匀分布)
    "alpha_dirichlet_choices": [0.3, 1.0, 3.0, 10.0], # 典型工况 (0.3:强偏流, 1.0:均匀, 3.0:弱集中, 10.0:均等)
    
    # 等效滤失参考背景区间 (主要用于非封闭或显示参考)
    "kleak_log_min": -6.0,         # 10^-6
    "kleak_log_max": -3.0,         # 10^-3
    
    # ── 域随机化新增参数 (Phase 1) ───────────────────────
    "a_min": 1350.0,               # 最小波速 [m/s]
    "a_max": 1550.0,               # 最大波速 [m/s]
    "snr_db_min": 20.0,            # 最小信噪比 [dB]
    "snr_db_max": 60.0,            # 最大信噪比 [dB]
    "friction_models": ["steady", "brunone"], # 混合摩阻模型采样池
    "toe_bc": "dead_end",          # 默认封闭趾端
}

# ── 3. 批量多进程运行并发与生成默认设定 ───────────────────────
LHS_BATCH_CONFIG: Dict[str, Any] = {
    "default_workers": 14,         # 针对 i5-12600KF(16线程) 优化，默认保留 2 线程给系统与I/O
    "default_n_samples": 1500,     # 推荐黄金训练集总样本数
    "default_friction": "brunone", # 默认采用 Brunone 非定常摩阻（真实体现频散与衰减物理特征）
    "output_dir": "output/lhs_dataset_v2", # 数据集升级至 v2 版本，避免与旧微柔度数据混淆
    "seed": 42,                    # 随机数种子，确保生成真解的数据集可复现
}

# ── 4. 神经算子与扩散去噪器 AI 反演配置 ───────────────────────
INVERSION_CONFIG: Dict[str, Any] = {
    "input_feature": "H_wh",       # 默认观测特征：井口水头时序信号 [m]
    "target_labels": ["x_f", "compliance_head_m2", "inflow_weight", "kleak"], # 物理严密的重构目标列表
    "max_n_frac": 6,               # 深度网络 (DiT / FNO) 统一对齐的定长维度上限 (不足部分以 0 填充)
    "normalize_method": "min_max", # 特征归一化建议配置
}

# ── 5. 分层含噪基准集（EXP-011 阶段 S1 / EXP-021 输入）────────
# 与 LHS_PARAM_RANGES 的连续间距采样不同：这里按固定间距格子分层，
# 补齐 lhs_dataset_2000(5–20 m) 与 lhs_dataset_6000(≥50 m) 之间的 20–50 m 空白。
# 噪声不进 MOC，后处理加到停泵后 H_wh，使同一清洁波形可复用于全部 SNR。
BENCH_STRATIFIED_CONFIG: Dict[str, Any] = {
    "spacing_grid_m": (5.0, 8.0, 12.0, 20.0, 30.0, 38.0, 50.0, 80.0, 120.0),
    "snr_db_levels": (None, 40.0, 30.0, 20.0, 10.0),  # None = 无限 SNR（清洁）
    "n_per_spacing": 30,           # 每间距格子清洁样本数（≥ EXP-021 预注册下限）
    "n_frac": 2,                   # 分辨率基准默认双缝（排除缝数混杂）
    "frac_zone_start": LHS_PARAM_RANGES["frac_zone_start"],
    "frac_zone_end": LHS_PARAM_RANGES["frac_zone_end"],
    "cf_log_min": LHS_PARAM_RANGES["cf_log_min"],
    "cf_log_max": LHS_PARAM_RANGES["cf_log_max"],
    "kleak_log_min": LHS_PARAM_RANGES["kleak_log_min"],
    "kleak_log_max": LHS_PARAM_RANGES["kleak_log_max"],
    "default_friction": "brunone",
    "default_tf": 30.0,            # 约 4 次往返；可用 CLI 覆盖
    "default_dt": SIM_CONFIG["dt"],
    "default_workers": LHS_BATCH_CONFIG["default_workers"],
    "output_dir": "output/bench_stratified",
    "seed": 20260801,
    # AWGN：相对停泵后信号功率；seed = base_seed + case_id * 1000 + snr_tag
    "noise_segment": "post_shut_in",
}
