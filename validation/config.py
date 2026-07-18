# -*- coding: utf-8 -*-
"""
validation/config.py — 集中管理验证脚本的物理参数、仿真参数、缝形态与摩阻配置。

供 leakoff_multi.py / kaiser_bessel_multi.py / wlen_sweep.py 等验证脚本共享。
修改本文件即可全局调整所有验证的物理与仿真设定。
"""
from __future__ import annotations

from typing import Dict, List

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

# ── 倒谱参数（leakoff 标准倒谱图 / 2D cepstrogram）────────
# win_type: rect | hamming | hanning | kaiser | gauss
# 寻峰（1D 实倒谱 & 2D 时间平均剖面共用 detect_1d_cepstrum_peaks）:
#   height = max(P{peak_height_pct}, peak_height_rel*max, peak_height_abs)
#   Brunone 次峰偏弱：把 peak_height_abs 降到 0、peak_height_rel 降到 0.03~0.05
#   可检出更多峰；过大则噪声峰也会进来。
CEPSTRUM_CONFIG = {
    'wlen_sec': 30.0,        # 2D 倒谱窗长 [s]
    'hop_sec': 5.0,          # 2D 倒谱 hop [s]
    'win_type': 'hamming',   # 2D 倒谱窗型
    # ---- 寻峰 ----
    'peak_height_pct': 85.0,     # 高度下界：响应分位数 [%]（原隐含 95）
    'peak_height_rel': 0.03,     # 高度下界：相对全局 max 的比例（替代硬门限 0.01）
    'peak_height_abs': 0.0,      # 绝对高度下限；0=关闭（原为 0.01，Brunone 易漏次峰）
    'peak_distance_frac': 0.25,  # 最小峰间距 = frac × 最小缝距 / Δd_bin
    'peak_top_n': 10,            # 最多保留峰数
    # ---- 裂缝区放大图（cepstrum_fracture_zoom.png）----
    'fracture_zoom_margin_m': 100.0,  # 缝群两侧各扩展 [m]
}

# ── 缝形态：首缝 + 等间距生成 ─────────────────────────────
FRAC_FIRST_M = 4100.0
SPACING_PRESETS_M = (5, 10, 20, 50, 100)


def build_cases(spacing_m: float) -> Dict[str, Dict]:
    """首缝 FRAC_FIRST_M，其后按 spacing_m 等间距排布 single~oct。"""
    d = float(spacing_m)
    x0 = FRAC_FIRST_M
    return {
        'single': {'label': '单缝', 'x_f_list': [x0]},
        'dual':   {'label': '双缝', 'x_f_list': [x0, x0 + d]},
        'triple': {'label': '三缝', 'x_f_list': [x0 + i * d for i in range(3)]},
        'quad':   {'label': '四缝', 'x_f_list': [x0 + i * d for i in range(4)]},
        'quint':  {'label': '五缝', 'x_f_list': [x0 + i * d for i in range(5)]},
        'hex':    {'label': '六缝', 'x_f_list': [x0 + i * d for i in range(6)]},
        'hept':   {'label': '七缝', 'x_f_list': [x0 + i * d for i in range(7)]},
        'oct':    {'label': '八缝', 'x_f_list': [x0 + i * d for i in range(8)]},
    }



# 默认 CASES：D=50 m（兼容 --friction steady / brunone）
CASES = build_cases(50)

# ── 摩阻模型配置 ──────────────────────────────────────────
# judgment4: steady → smoothness（抖动比 < 2.0）
#            brunone → oscillation_decay（RMS 衰减比 < 0.5）
# stab_factor: 判定 5 长期稳定性阈值因子 (H_range < stab_factor × ΔH)
# spacing_m: 若存在，leakoff 用 build_cases(spacing_m)，输出目录为键名
#            （如 steady_D10 → output/leakoff/steady_D10/）
_STEADY_BASE = {
    'friction_model': 'steady',
    'judgment4': 'smoothness',
    'stab_factor': 1.0,
}

_BRUNONE_BASE = {
    'friction_model': 'brunone',
    'judgment4': 'oscillation_decay',
    'stab_factor': 0.5,
}

FRICTION_PARAMS = {
    'steady': {
        **_STEADY_BASE,
        'label': '稳态达西+滤失',
    },
    'brunone': {
        **_BRUNONE_BASE,
        'label': 'Brunone非定常+滤失',
    },
}

# 等间距变体：steady_D* / brunone_D* → output/leakoff/{key}/{case}/
for _d in SPACING_PRESETS_M:
    FRICTION_PARAMS[f'steady_D{_d}'] = {
        **_STEADY_BASE,
        'label': f'稳态达西+滤失 D={_d}m',
        'spacing_m': _d,
    }
    FRICTION_PARAMS[f'brunone_D{_d}'] = {
        **_BRUNONE_BASE,
        'label': f'Brunone非定常+滤失 D={_d}m',
        'spacing_m': _d,
    }

# 批量别名：一次跑完 SPACING_PRESETS_M（不写入 FRICTION_PARAMS，仅 CLI 展开）
FRICTION_BATCH_ALIASES: Dict[str, List[str]] = {
    'steady_Dall': [f'steady_D{d}' for d in SPACING_PRESETS_M],
    'brunone_Dall': [f'brunone_D{d}' for d in SPACING_PRESETS_M],
}


def expand_friction_keys(friction: str) -> List[str]:
    """将 steady_Dall / brunone_Dall 展开为各间距键；其余原样返回。"""
    if friction in FRICTION_BATCH_ALIASES:
        return list(FRICTION_BATCH_ALIASES[friction])
    return [friction]


def friction_cli_choices() -> List[str]:
    """CLI --friction 可选值：FRICTION_PARAMS 键 + 批量别名。"""
    return list(FRICTION_PARAMS.keys()) + list(FRICTION_BATCH_ALIASES.keys())

