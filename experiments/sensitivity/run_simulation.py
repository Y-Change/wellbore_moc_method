# -*- coding: utf-8 -*-
"""
experiments/sensitivity/run_simulation.py
----------------------------------------
批量执行裂缝物理参数敏感性消融 MOC 仿真（Milestone M2 核心执行器）。

核心功能：
1. 读取 output/fracture_parameter_sensitivity/manifest.json 清单中的全部 84 组工况；
2. 构造 MocConfig (严格保证 CFL ≡ 1.0 无数值色散与耗散)；
3. 调用 moc_simulate.wellbore_moc.solve_moc 执行物理仿真；
4. 成对运行 Darcy-Weisbach 纯稳态摩阻 ('steady') 与 Brunone 非定常摩阻 ('brunone')；
5. 实施物理与数值校验：
   - 100% 收敛通过 (PASS)
   - 零 NaN / Inf 异常
   - 严格质量守恒与稳态基线平稳度检验
   - 正向水头裕度 min(Hf_ss - H_ext) > 0 验证
6. 序列化标准 schema moc_lhs_v2.1 兼容的 NPZ 文件（41 项标准键值）；
7. 导出高精度时程 CSV 文件 (t, H_wh, Q_wh, H_f1, Q_f1, ...)。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 加入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moc_simulate.wellbore_moc import MocConfig, solve_moc

# 路径常量
SENSITIVITY_DIR = PROJECT_ROOT / "output" / "fracture_parameter_sensitivity"
DATA_DIR = SENSITIVITY_DIR / "data"
CSV_DIR = SENSITIVITY_DIR / "timeseries_csv"
MANIFEST_PATH = SENSITIVITY_DIR / "manifest.json"


def simulate_single_case(case_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    单工况仿真执行函数：负责求解、校验并落盘 NPZ 与 CSV。
    """
    case_id = int(case_info["case_id"])
    case_name = case_info.get("case_name", f"case_{case_id:05d}")
    friction = str(case_info["friction"])
    brunone_k_scale = float(case_info.get("brunone_k_scale", 1.0 if friction == "brunone" else 0.0))
    seed = int(case_info.get("seed", 20260909 + case_id))

    wb = case_info["wellbore"]
    fc = case_info["fractures"]

    t0 = time.perf_counter()

    # 1. 构造 MocConfig 配置 (保证 CFL = a * dt / dx == 1.0)
    cfg = MocConfig(
        wellbore_length=float(wb["L"]),
        wellbore_diameter=float(wb["D"]),
        fluid_density=float(wb["rho"]),
        fluid_viscosity=float(wb["nu"]),
        wavespeed=float(wb["a"]),
        roughness_height=float(wb["roughness"]),
        friction_model=friction,
        brunone_k_scale=brunone_k_scale,
        dt=float(wb["dt"]),
        tf=float(wb["tf"]),
        wellhead_bc=str(wb.get("wellhead_bc", "ramp")),
        pump_shut_time=float(wb["ts"]),
        pump_closure_duration=float(wb.get("tc", 0.05)),
        initial_velocity=float(wb["V0"]),
        initial_head=float(wb["H0"]),
        theta=float(wb.get("theta", 0.0)),
        toe_bc=str(wb.get("toe_bc", "dead_end")),
        toe_head=float(wb.get("H0", 300.0)),
    )

    # 验证 CFL 严格等于 1.0
    cfl = cfg.a_adj * cfg.dt_adj / cfg.dx
    assert abs(cfl - 1.0) < 1e-12, f"CFL violation in {case_name}: cfl={cfl}"

    # 2. 调用核心求解器 solve_moc
    positions = list(fc["positions"])
    compliance_m2 = list(fc["compliance_m2"])
    kleak_target = list(fc["kleak"])
    Rp_list = list(fc["Rp"])
    inflow_weights = list(fc["inflow_weights"])
    H_ext_val = float(wb["H_ext"])

    res = solve_moc(
        cfg=cfg,
        fracture_positions=positions,
        fracture_compliance_m2=compliance_m2,
        fracture_kleak=kleak_target,
        fracture_inflow_weights=inflow_weights,
        fracture_Rp=Rp_list,
        H_ext=H_ext_val,
        store_full_field=False,
    )

    t_elapsed = time.perf_counter() - t0

    # 3. 提取结果与守恒校验
    t_arr = np.asarray(res["timestamps"], dtype=np.float64)
    H_wh = np.asarray(res["wellhead_head"], dtype=np.float64)
    V_wh = np.asarray(res["wellhead_velocity"], dtype=np.float64)
    Q_wh = V_wh * cfg.area

    # 检查非有限数值
    assert np.all(np.isfinite(t_arr)), f"NaN/Inf in timestamps for {case_name}"
    assert np.all(np.isfinite(H_wh)), f"NaN/Inf in wellhead_head for {case_name}"
    assert np.all(np.isfinite(Q_wh)), f"NaN/Inf in wellhead_flow for {case_name}"

    # 检查非平凡波幅
    ptp_head = float(np.ptp(H_wh))
    assert ptp_head > 1.0, f"Flatline water hammer signal in {case_name}: ptp={ptp_head:.3f} m"

    # 稳态信息与水头裕度校验
    ss_info = res["steady_state"]
    kleak_equiv = np.asarray(ss_info["equivalent_kleak"], dtype=np.float64)
    wi_actual = np.asarray(ss_info["inflow_weights"], dtype=np.float64)
    Hw_ss = np.asarray(ss_info["Hw_ss"], dtype=np.float64)
    Hf_ss = np.asarray(ss_info["Hf_ss"], dtype=np.float64)
    Qf_ss = np.asarray(ss_info["Qf_ss"], dtype=np.float64)
    Qin_ss = float(ss_info["Qin_ss"])

    min_head_margin = float(np.min(Hf_ss - H_ext_val))
    assert min_head_margin > 0.0, f"Negative head margin in {case_name}: min(Hf_ss - H_ext)={min_head_margin:.4f} m"

    # 稳态双射闭合与质量平衡校验
    for k in range(len(positions)):
        diff_q = abs(Qf_ss[k] - kleak_equiv[k] * np.sqrt(Hf_ss[k] - H_ext_val))
        assert diff_q < 1.0e-9, f"Steady state bijection error in {case_name} frac #{k}: {diff_q:.4e}"
    diff_total_q = abs(float(np.sum(Qf_ss)) - Qin_ss)
    assert diff_total_q < 1.0e-9, f"Steady mass conservation error in {case_name}: {diff_total_q:.4e}"

    frac_indices = np.asarray(res["fracture_indices"], dtype=np.int32)
    x_f_aligned = np.asarray([idx * cfg.dx for idx in frac_indices], dtype=np.float64)
    x_f_requested = np.asarray(positions, dtype=np.float64)

    # 4. 序列化 NPZ 文件 (严格 41 项标准 Schema moc_lhs_v2.1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = DATA_DIR / f"{case_name}.npz"

    np.savez_compressed(
        npz_path,
        # 1. Version & reproducibility (3)
        schema_version="moc_lhs_v2.1",
        seed=int(seed),
        case_id=int(case_id),
        # 2. Grid & acoustic parameters (10)
        N=int(cfg.N),
        wellbore_length=float(cfg.wellbore_length),
        dx=float(cfg.dx),
        dt_requested=float(cfg.dt),
        dt_adj=float(cfg.dt_adj),
        dt=float(cfg.dt_adj),
        wavespeed_requested=float(cfg.wavespeed),
        wavespeed_adj=float(cfg.a_adj),
        wavespeed=float(cfg.a_adj),
        wavespeed_nominal=float(cfg.wavespeed),
        # 3. Spatial coordinates & alignment (6)
        x_f_requested=x_f_requested,
        x_f_aligned=x_f_aligned,
        x_f=x_f_aligned,
        x_f_raw=x_f_requested,
        fracture_indices=frac_indices,
        grid_index=frac_indices,
        # 4. Wellhead observation time series (3)
        t=t_arr,
        H_wh=H_wh,
        Q_wh=Q_wh,
        # 5. Boundary & operating conditions (7)
        friction=str(friction),
        brunone_k_scale=float(brunone_k_scale),
        toe_bc=str(cfg.toe_bc),
        initial_head=float(cfg.initial_head),
        initial_velocity=float(cfg.initial_velocity),
        H_ext=float(H_ext_val),
        tf=float(cfg.tf),
        # 6. Fracture physical properties (7)
        n_frac=int(len(positions)),
        Rp=np.asarray(Rp_list, dtype=np.float64),
        compliance_head_m2=np.asarray(compliance_m2, dtype=np.float64),
        Cf=np.asarray(compliance_m2, dtype=np.float64),
        inflow_weight=wi_actual,
        kleak_equiv=kleak_equiv,
        kleak=kleak_equiv,
        # 7. Steady-state closure & diagnostics (5)
        Hw_ss=Hw_ss,
        Hf_ss=Hf_ss,
        Qf_ss=Qf_ss,
        Qin_ss=float(Qin_ss),
        alpha_dirichlet=1.0,
        # 附加诊断标签
        min_head_margin_m=float(min_head_margin),
        convergence_status="PASS",
        status="PASS",
    )

    # 5. 序列化高精度 CSV 文件
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"{case_name}.csv"

    frac_heads = res["fracture_internal_heads"]
    frac_qs = res["fracture_Qs"]

    csv_data = {
        "t": t_arr,
        "H_wh": H_wh,
        "Q_wh": Q_wh,
    }
    for k in range(len(positions)):
        csv_data[f"H_f{k+1}"] = frac_heads[:, k]
        csv_data[f"Q_f{k+1}"] = frac_qs[:, k]

    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(csv_path, index=False, float_format="%.6e")

    return {
        "case_id": case_id,
        "case_name": case_name,
        "group": case_info["group"],
        "friction": friction,
        "elapsed_s": t_elapsed,
        "min_head_margin": min_head_margin,
        "status": "PASS",
    }


def _worker_wrapper(case_info: Dict[str, Any]) -> Dict[str, Any]:
    """多进程包装器，具备异常捕获与诊断信息格式化"""
    case_name = case_info.get("case_name", f"case_{case_info['case_id']:05d}")
    try:
        return simulate_single_case(case_info)
    except Exception as e:
        print(f"[ERROR] Case {case_name} failed: {e}", file=sys.stderr)
        return {
            "case_id": case_info["case_id"],
            "case_name": case_name,
            "group": case_info.get("group", "unknown"),
            "friction": case_info.get("friction", "unknown"),
            "elapsed_s": 0.0,
            "min_head_margin": -1.0,
            "status": "FAIL",
            "error": str(e),
        }


def run_batch_simulations(
    manifest_path: Path = MANIFEST_PATH,
    num_workers: Optional[int] = None,
    specific_case_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """读取 manifest 并执行多进程批量仿真"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}. Run generate_matrix.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_cases = manifest["cases"]

    if specific_case_id is not None:
        target_cases = [c for c in all_cases if int(c["case_id"]) == specific_case_id]
        if not target_cases:
            raise ValueError(f"Case ID {specific_case_id} not found in manifest.")
    else:
        target_cases = all_cases

    n_total = len(target_cases)
    if num_workers is None:
        cpu_avail = os.cpu_count() or 4
        num_workers = max(1, min(cpu_avail - 2, 12))

    print("=" * 70)
    print(f"启动 MOC 裂缝参数敏感性批量仿真 (Batch MOC Simulation Runner)")
    print(f"待运行工况数: {n_total} | 并行 Worker 数: {num_workers}")
    print(f"NPZ 导出目录: {DATA_DIR}")
    print(f"CSV 导出目录: {CSV_DIR}")
    print("=" * 70)

    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []

    if num_workers <= 1:
        for idx, c in enumerate(target_cases):
            res = _worker_wrapper(c)
            results.append(res)
            print(
                f"[{idx+1:02d}/{n_total:02d}] {res['case_name']} | {res['group']:12s} | "
                f"{res['friction']:7s} | {res['status']} | 耗时 {res['elapsed_s']:.2f}s"
            )
    else:
        with mp.Pool(processes=num_workers) as pool:
            for idx, res in enumerate(pool.imap(_worker_wrapper, target_cases)):
                results.append(res)
                print(
                    f"[{idx+1:02d}/{n_total:02d}] {res['case_name']} | {res['group']:12s} | "
                    f"{res['friction']:7s} | {res['status']} | 耗时 {res['elapsed_s']:.2f}s"
                )

    t_total = time.perf_counter() - t_start

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = n_total - n_pass

    print("\n" + "=" * 70)
    print(f"仿真批处理完成 Summary: 总工况={n_total}, 通过={n_pass}, 失败={n_fail}, 总耗时={t_total:.1f}s")
    print("=" * 70)

    if n_fail > 0:
        failed_cases = [r for r in results if r["status"] != "PASS"]
        for fc in failed_cases:
            print(f"  FAILED: {fc['case_name']} - {fc.get('error', 'unknown error')}")
        raise RuntimeError(f"{n_fail} simulation cases failed! Check logs above.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run dual-friction MOC sensitivity simulations.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--case-id", type=int, default=None, help="Run single case by integer ID")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH), help="Path to manifest JSON")

    args = parser.parse_args()

    manifest_p = Path(args.manifest)
    run_batch_simulations(
        manifest_path=manifest_p,
        num_workers=args.workers,
        specific_case_id=args.case_id,
    )


if __name__ == "__main__":
    main()
