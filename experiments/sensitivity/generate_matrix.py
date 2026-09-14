# -*- coding: utf-8 -*-
"""
experiments/sensitivity/generate_matrix.py
-----------------------------------------
构建新版 MOC 求解器下裂缝物理参数敏感性消融实验仿真矩阵（OAT 单变量消融 + 正交交互矩阵）。

覆盖 6 大物理维度：
1. x_f: 裂缝位置扫描 [3500, 3800, 4100, 4400, 4700] m (基准间距 20m, 3 缝)
2. C_H: 水头柔度扫描 [1e-7, 1e-6, 1e-5, 3e-5, 1e-4] m²
3. k_leak: 地层滤失扫描 [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3] m^{5/2}/s
4. R_p: 射孔流动阻抗扫描 [0.0, 100.0, 500.0, 2000.0, 10000.0] s²/m⁵
5. w_i: 稳态分流权重扫描 (uniform, toe-dominant, heel-dominant, middle-dominant)
6. delta_x: 裂缝簇间距扫描 [5.0, 10.0, 20.0, 35.0, 50.0] m (首缝 4000m, 3 缝)

正交跨参数网格：
C_H (3 档: 1e-6, 1e-5, 1e-4) × delta_x (2 档: 10.0, 35.0) × k_leak (2 档: 1e-5, 1e-4) = 12 组

每个工况均生成成对仿真（paired runs）：
- friction = 'steady' (Darcy-Weisbach 纯稳态摩阻)
- friction = 'brunone' (Brunone 非定常摩阻)

总计 42 个物理参数组 × 2 种摩阻 = 84 个独立仿真工况。
清单导出至：
- output/fracture_parameter_sensitivity/manifest.json
- experiments/sensitivity/experiment_manifest.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 基准井筒参数 (Baseline Wellbore)
BASELINE_WELLBORE: Dict[str, Any] = {
    "L": 5000.0,                # 井长 [m]
    "D": 0.1397,                # 内径 [m] (5.5" 套管)
    "rho": 1000.0,              # 流体密度 [kg/m³]
    "nu": 1.0e-6,               # 运动黏度 [m²/s]
    "a": 1450.0,                # 标称水锤波速 [m/s]
    "roughness": 4.5e-5,         # 绝对粗糙度 [m]
    "V0": 1.0,                  # 初始稳态流速 [m/s]
    "H0": 300.0,                # 初始井口水头 [m]
    "H_ext": 100.0,             # 外部孔隙压力水头 [m]
    "theta": 0.0,               # 井斜角正弦（水平井=0.0）
    "toe_bc": "dead_end",       # 趾端边界：封闭端
    "tf": 40.0,                 # 仿真历时 [s]
    "tc": 0.05,                 # 关井历时 [s] (快速 ramp)
    "ts": 0.5,                  # 停泵起始时刻 [s]
    "dt": 0.001,                # 时间步长 [s] (1 ms, CFL=1.0)
    "wellhead_bc": "ramp",      # 井口边界类型
    "brunone_k_scale": 1.0,     # Brunone 衰减系数缩放倍率
}

# 基准 3 裂缝系统 (Baseline 3-fracture system)
BASELINE_FRACTURES: Dict[str, Any] = {
    "n_frac": 3,
    "positions": [4000.0, 4020.0, 4040.0],
    "compliance_m2": [1.0e-5, 1.0e-5, 1.0e-5],
    "kleak": [1.0e-4, 1.0e-4, 1.0e-4],
    "Rp": [0.0, 0.0, 0.0],
    "inflow_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
}

# 扫描网格定义 (Ablation Grid)
SWEEP_DEFINITIONS = {
    "xf": {
        "group": "oat_xf",
        "param_name": "x_f",
        "values": [3500.0, 3800.0, 4100.0, 4400.0, 4700.0],
        "delta_x_default": 20.0,
    },
    "ch": {
        "group": "oat_ch",
        "param_name": "compliance_head_m2",
        "values": [1.0e-7, 1.0e-6, 1.0e-5, 3.0e-5, 1.0e-4],
    },
    "kleak": {
        "group": "oat_kleak",
        "param_name": "kleak",
        "values": [0.0, 1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3],
    },
    "rp": {
        "group": "oat_rp",
        "param_name": "Rp",
        "values": [0.0, 100.0, 500.0, 2000.0, 10000.0],
    },
    "wi": {
        "group": "oat_wi",
        "param_name": "inflow_weight",
        "patterns": {
            "uniform": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "toe_dominant": [0.1, 0.2, 0.7],
            "heel_dominant": [0.7, 0.2, 0.1],
            "middle_dominant": [0.1, 0.8, 0.1],
        },
    },
    "spacing": {
        "group": "oat_spacing",
        "param_name": "delta_x",
        "values": [5.0, 10.0, 20.0, 35.0, 50.0],
        "x1_default": 4000.0,
    },
}

# 正交矩阵定义 (Orthogonal Grid)
ORTHOGONAL_CH = [1.0e-6, 1.0e-5, 1.0e-4]
ORTHOGONAL_SPACING = [10.0, 35.0]
ORTHOGONAL_KLEAK = [1.0e-5, 1.0e-4]


def is_baseline_condition(
    positions: List[float],
    compliance_m2: List[float],
    kleak: List[float],
    Rp: List[float],
    inflow_weights: List[float],
) -> bool:
    """判断是否严格匹配基准工况参数"""
    base_pos = BASELINE_FRACTURES["positions"]
    base_ch = BASELINE_FRACTURES["compliance_m2"]
    base_kl = BASELINE_FRACTURES["kleak"]
    base_rp = BASELINE_FRACTURES["Rp"]
    base_wi = BASELINE_FRACTURES["inflow_weights"]

    pos_match = all(abs(p - bp) < 1e-3 for p, bp in zip(positions, base_pos))
    ch_match = all(abs(c - bc) < 1e-12 for c, bc in zip(compliance_m2, base_ch))
    kl_match = all(abs(k - bk) < 1e-12 for k, bk in zip(kleak, base_kl))
    rp_match = all(abs(r - br) < 1e-6 for r, br in zip(Rp, base_rp))
    wi_match = all(abs(w - bw) < 1e-6 for w, bw in zip(inflow_weights, base_wi))

    return pos_match and ch_match and kl_match and rp_match and wi_match


def build_parameter_matrix() -> List[Dict[str, Any]]:
    """生成 42 个物理参数组合（30 个 OAT + 12 个正交）"""
    parameter_sets: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. OAT: x_f location scan (5 组)
    # -------------------------------------------------------------
    for xf in SWEEP_DEFINITIONS["xf"]["values"]:
        pos = [float(xf), float(xf + 20.0), float(xf + 40.0)]
        is_base = is_baseline_condition(
            pos,
            BASELINE_FRACTURES["compliance_m2"],
            BASELINE_FRACTURES["kleak"],
            BASELINE_FRACTURES["Rp"],
            BASELINE_FRACTURES["inflow_weights"],
        )
        parameter_sets.append({
            "group": "oat_xf",
            "param_name": "x_f",
            "param_value": xf,
            "param_description": f"First fracture at {xf:.0f}m (spacing 20m)",
            "is_baseline": is_base,
            "positions": pos,
            "compliance_m2": list(BASELINE_FRACTURES["compliance_m2"]),
            "kleak": list(BASELINE_FRACTURES["kleak"]),
            "Rp": list(BASELINE_FRACTURES["Rp"]),
            "inflow_weights": list(BASELINE_FRACTURES["inflow_weights"]),
            "spacing_m": 20.0,
        })

    # -------------------------------------------------------------
    # 2. OAT: C_H compliance scan (5 组)
    # -------------------------------------------------------------
    for ch in SWEEP_DEFINITIONS["ch"]["values"]:
        ch_list = [float(ch), float(ch), float(ch)]
        is_base = is_baseline_condition(
            BASELINE_FRACTURES["positions"],
            ch_list,
            BASELINE_FRACTURES["kleak"],
            BASELINE_FRACTURES["Rp"],
            BASELINE_FRACTURES["inflow_weights"],
        )
        parameter_sets.append({
            "group": "oat_ch",
            "param_name": "compliance_head_m2",
            "param_value": ch,
            "param_description": f"Head compliance C_H = {ch:.1e} m2",
            "is_baseline": is_base,
            "positions": list(BASELINE_FRACTURES["positions"]),
            "compliance_m2": ch_list,
            "kleak": list(BASELINE_FRACTURES["kleak"]),
            "Rp": list(BASELINE_FRACTURES["Rp"]),
            "inflow_weights": list(BASELINE_FRACTURES["inflow_weights"]),
            "spacing_m": 20.0,
        })

    # -------------------------------------------------------------
    # 3. OAT: k_leak leakoff scan (6 组)
    # -------------------------------------------------------------
    for kl in SWEEP_DEFINITIONS["kleak"]["values"]:
        kl_list = [float(kl), float(kl), float(kl)]
        is_base = is_baseline_condition(
            BASELINE_FRACTURES["positions"],
            BASELINE_FRACTURES["compliance_m2"],
            kl_list,
            BASELINE_FRACTURES["Rp"],
            BASELINE_FRACTURES["inflow_weights"],
        )
        parameter_sets.append({
            "group": "oat_kleak",
            "param_name": "kleak",
            "param_value": kl,
            "param_description": f"Leakoff coefficient k_leak = {kl:.1e} m^(5/2)/s",
            "is_baseline": is_base,
            "positions": list(BASELINE_FRACTURES["positions"]),
            "compliance_m2": list(BASELINE_FRACTURES["compliance_m2"]),
            "kleak": kl_list,
            "Rp": list(BASELINE_FRACTURES["Rp"]),
            "inflow_weights": list(BASELINE_FRACTURES["inflow_weights"]),
            "spacing_m": 20.0,
        })

    # -------------------------------------------------------------
    # 4. OAT: R_p perforation resistance scan (5 组)
    # -------------------------------------------------------------
    for rp in SWEEP_DEFINITIONS["rp"]["values"]:
        rp_list = [float(rp), float(rp), float(rp)]
        is_base = is_baseline_condition(
            BASELINE_FRACTURES["positions"],
            BASELINE_FRACTURES["compliance_m2"],
            BASELINE_FRACTURES["kleak"],
            rp_list,
            BASELINE_FRACTURES["inflow_weights"],
        )
        parameter_sets.append({
            "group": "oat_rp",
            "param_name": "Rp",
            "param_value": rp,
            "param_description": f"Perforation resistance Rp = {rp:.0f} s2/m5",
            "is_baseline": is_base,
            "positions": list(BASELINE_FRACTURES["positions"]),
            "compliance_m2": list(BASELINE_FRACTURES["compliance_m2"]),
            "kleak": list(BASELINE_FRACTURES["kleak"]),
            "Rp": rp_list,
            "inflow_weights": list(BASELINE_FRACTURES["inflow_weights"]),
            "spacing_m": 20.0,
        })

    # -------------------------------------------------------------
    # 5. OAT: w_i flow split scan (4 组)
    # -------------------------------------------------------------
    for name, wi_list in SWEEP_DEFINITIONS["wi"]["patterns"].items():
        is_base = is_baseline_condition(
            BASELINE_FRACTURES["positions"],
            BASELINE_FRACTURES["compliance_m2"],
            BASELINE_FRACTURES["kleak"],
            BASELINE_FRACTURES["Rp"],
            wi_list,
        )
        parameter_sets.append({
            "group": "oat_wi",
            "param_name": "inflow_weight",
            "param_value": name,
            "param_description": f"Flow split pattern: {name} {wi_list}",
            "is_baseline": is_base,
            "positions": list(BASELINE_FRACTURES["positions"]),
            "compliance_m2": list(BASELINE_FRACTURES["compliance_m2"]),
            "kleak": list(BASELINE_FRACTURES["kleak"]),
            "Rp": list(BASELINE_FRACTURES["Rp"]),
            "inflow_weights": list(wi_list),
            "spacing_m": 20.0,
        })

    # -------------------------------------------------------------
    # 6. OAT: delta_x cluster spacing scan (5 组)
    # -------------------------------------------------------------
    for dx in SWEEP_DEFINITIONS["spacing"]["values"]:
        pos = [4000.0, float(4000.0 + dx), float(4000.0 + 2 * dx)]
        is_base = is_baseline_condition(
            pos,
            BASELINE_FRACTURES["compliance_m2"],
            BASELINE_FRACTURES["kleak"],
            BASELINE_FRACTURES["Rp"],
            BASELINE_FRACTURES["inflow_weights"],
        )
        parameter_sets.append({
            "group": "oat_spacing",
            "param_name": "delta_x",
            "param_value": dx,
            "param_description": f"Cluster spacing delta_x = {dx:.1f} m",
            "is_baseline": is_base,
            "positions": pos,
            "compliance_m2": list(BASELINE_FRACTURES["compliance_m2"]),
            "kleak": list(BASELINE_FRACTURES["kleak"]),
            "Rp": list(BASELINE_FRACTURES["Rp"]),
            "inflow_weights": list(BASELINE_FRACTURES["inflow_weights"]),
            "spacing_m": float(dx),
        })

    # -------------------------------------------------------------
    # 7. Orthogonal Cross-parameter matrix (12 组)
    # C_H (3) × delta_x (2) × k_leak (2)
    # -------------------------------------------------------------
    for ch in ORTHOGONAL_CH:
        for dx in ORTHOGONAL_SPACING:
            for kl in ORTHOGONAL_KLEAK:
                pos = [4000.0, float(4000.0 + dx), float(4000.0 + 2 * dx)]
                ch_list = [float(ch), float(ch), float(ch)]
                kl_list = [float(kl), float(kl), float(kl)]
                is_base = is_baseline_condition(
                    pos,
                    ch_list,
                    kl_list,
                    BASELINE_FRACTURES["Rp"],
                    BASELINE_FRACTURES["inflow_weights"],
                )
                parameter_sets.append({
                    "group": "orthogonal",
                    "param_name": "ch_spacing_kleak",
                    "param_value": f"CH_{ch:.0e}_dx_{dx:.0f}_kl_{kl:.0e}",
                    "param_description": (
                        f"Orthogonal grid: C_H={ch:.1e} m2, delta_x={dx:.0f}m, k_leak={kl:.1e} m^(5/2)/s"
                    ),
                    "is_baseline": is_base,
                    "positions": pos,
                    "compliance_m2": ch_list,
                    "kleak": kl_list,
                    "Rp": list(BASELINE_FRACTURES["Rp"]),
                    "inflow_weights": list(BASELINE_FRACTURES["inflow_weights"]),
                    "spacing_m": float(dx),
                })

    return parameter_sets


def generate_full_manifest() -> Dict[str, Any]:
    """
    生成完整的仿真执行清单，包含每个参数组在 steady 和 brunone 两种摩阻下的配对算例。
    总计 42 × 2 = 84 个工况。
    """
    param_sets = build_parameter_matrix()
    assert len(param_sets) == 42, f"Expected 42 parameter sets, got {len(param_sets)}"

    cases: List[Dict[str, Any]] = []
    case_idx = 0

    for pset in param_sets:
        for friction in ["steady", "brunone"]:
            case_id_str = f"case_{case_idx:05d}"
            case_entry = {
                "case_id": case_idx,
                "case_name": case_id_str,
                "group": pset["group"],
                "param_name": pset["param_name"],
                "param_value": pset["param_value"],
                "param_description": pset["param_description"],
                "is_baseline": pset["is_baseline"] and (friction == "steady"),
                "friction": friction,
                "brunone_k_scale": 1.0 if friction == "brunone" else 0.0,
                "wellbore": dict(BASELINE_WELLBORE),
                "fractures": {
                    "n_frac": 3,
                    "positions": list(pset["positions"]),
                    "compliance_m2": list(pset["compliance_m2"]),
                    "kleak": list(pset["kleak"]),
                    "Rp": list(pset["Rp"]),
                    "inflow_weights": list(pset["inflow_weights"]),
                    "spacing_m": pset["spacing_m"],
                },
                "seed": 20260909 + case_idx,
            }
            cases.append(case_entry)
            case_idx += 1

    assert len(cases) == 84, f"Expected 84 total cases, got {len(cases)}"

    manifest = {
        "metadata": {
            "project": "fracture_parameter_sensitivity",
            "schema_version": "moc_lhs_v2.1",
            "description": (
                "Simulation matrix for fracture parameter sensitivity ablation study "
                "across 6 physical dimensions + orthogonal interaction grid, paired with "
                "steady and Brunone unsteady friction models."
            ),
            "total_parameter_sets": len(param_sets),
            "total_cases": len(cases),
            "friction_models": ["steady", "brunone"],
            "groups": [
                "oat_xf",
                "oat_ch",
                "oat_kleak",
                "oat_rp",
                "oat_wi",
                "oat_spacing",
                "orthogonal",
            ],
            "baseline_case_id": 0,
        },
        "baseline_wellbore": BASELINE_WELLBORE,
        "baseline_fractures": BASELINE_FRACTURES,
        "cases": cases,
    }

    return manifest


def save_manifest(manifest: Dict[str, Any], output_paths: List[Path]) -> None:
    """序列化 manifest 到指定文件路径"""
    for out_path in output_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[OK] Manifest written to: {out_path} ({len(manifest['cases'])} cases)")


def main():
    print("=" * 70)
    print("生成裂缝物理参数敏感性消融仿真矩阵 (Simulation Matrix Generator)")
    print("=" * 70)

    manifest = generate_full_manifest()

    out_dirs = [
        PROJECT_ROOT / "output" / "fracture_parameter_sensitivity" / "manifest.json",
        PROJECT_ROOT / "experiments" / "sensitivity" / "experiment_manifest.json",
    ]

    save_manifest(manifest, out_dirs)

    # 统计汇总输出
    group_counts: Dict[str, int] = {}
    for c in manifest["cases"]:
        grp = c["group"]
        group_counts[grp] = group_counts.get(grp, 0) + 1

    print("\n[工况分组统计 Summary]")
    for grp, count in group_counts.items():
        print(f"  - {grp:15s}: {count} 组仿真 (含 steady + brunone 对照)")
    print(f"  总计工况总数: {len(manifest['cases'])}")


if __name__ == "__main__":
    main()
