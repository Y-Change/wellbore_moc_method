# -*- coding: utf-8 -*-
"""
moc_simulate.staged_repair.run_stage_verification — 阶段一仿真与对标全流程执行脚本

执行内容：
1. 使用 wellbore_moc_v2.py 运行 brunone_D10/quad 全时程仿真 (tf=100.0s, dt=0.001s);
2. 导出 stage1_timeseries.csv 至 output/staged_repair/stage_1/;
3. 读取 PaperA 基准数据进行对标分析，计算准出判定指标并导出 stage1_metrics.json;
4. 生成 stage1_pre_shut_in_zoom.png 与 stage1_vs_papera_delta.png 高清学术图表;
5. 输出全套验收结论。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# 确保路径可解析
REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moc_simulate.staged_repair.config_v2 import (
    STAGE1_CONFIG,
    PAPERA_BENCHMARK_CSV,
    STAGE1_OUTPUT_DIR,
)
from moc_simulate.staged_repair.wellbore_moc_v2 import MocConfig, simulate_wellbore
from moc_simulate.staged_repair.compare_utils import (
    load_timeseries_csv,
    compute_stage1_metrics,
    plot_pre_shut_in_zoom,
    plot_full_timeseries_delta,
)


def run_stage1_verification(reuse_csv: bool = False) -> Dict:
    print("=" * 72)
    print("【阶段一：趾端边界与死水段静止化（Stage 1）】仿真与对标验证启动")
    print("=" * 72)

    # 1. 确保输出目录存在
    out_dir = STAGE1_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] 成果输出目录就绪: {out_dir}")

    # 2. 构建仿真配置
    cfg_data = STAGE1_CONFIG
    cfg = MocConfig(
        wellbore_length=cfg_data["wellbore_length"],
        wellbore_diameter=cfg_data["wellbore_diameter"],
        fluid_density=cfg_data["fluid_density"],
        fluid_viscosity=cfg_data["fluid_viscosity"],
        wavespeed=cfg_data["wavespeed"],
        roughness_height=cfg_data["roughness_height"],
        friction_model=cfg_data["friction_model"],
        brunone_k_scale=cfg_data.get("brunone_k_scale", 1.0),
        dt=cfg_data["dt"],
        tf=cfg_data["tf"],
        wellhead_bc=cfg_data["wellhead_bc"],
        pump_shut_time=cfg_data["pump_shut_time"],
        pump_closure_duration=cfg_data.get("pump_closure_duration", 1.0e-3),
        initial_velocity=cfg_data["initial_velocity"],
        initial_head=cfg_data["initial_head"],
        theta=cfg_data["theta"],
        toe_bc="dead_end",
        toe_head=cfg_data.get("toe_head", 300.0),
    )

    print(f"[2/5] 求解器配置已锁定:")
    print(f"      - 井长: {cfg.wellbore_length} m, 波速: {cfg.a_adj:.2f} m/s, 网格数 N={cfg.N}, dx={cfg.dx:.4f} m")
    print(f"      - 时步: dt={cfg.dt_adj} s, 总长: tf={cfg.tf} s (总步数: {cfg.n_steps})")
    print(f"      - 边界: wellhead={cfg.wellhead_bc}, toe_bc={cfg.toe_bc} (Gamma = +1.0 严格死端)")
    print(f"      - 裂缝: {cfg_data['fracture_positions']} m (quad, 间距 10m)")
    print(f"      - 柔度 Cf={cfg_data['Cf']}, 滤失 kleak={cfg_data['kleak']}, H_ext={cfg_data['H_ext']} m")

    # 3. 运行仿真或复用已有数据
    csv_path = out_dir / "stage1_timeseries.csv"
    if reuse_csv and csv_path.is_file() and csv_path.stat().st_size > 10_000_000:
        print(f"[3/5] 检测到已存在仿真数据且指定了复用模式，正在读取: {csv_path}")
        stage1_df = pd.read_csv(csv_path)
    else:
        print("[3/5] 开始执行 MOC 全时程时间推进计算...")
        t0 = time.time()
        res = simulate_wellbore(
            cfg=cfg,
            fracture_positions=cfg_data["fracture_positions"],
            fracture_Cf=[cfg_data["Cf"]] * len(cfg_data["fracture_positions"]),
            fracture_kleak=[cfg_data["kleak"]] * len(cfg_data["fracture_positions"]),
            H_ext=cfg_data["H_ext"],
            store_full_field=False,
        )
        t_elapsed = time.time() - t0
        print(f"      MOC 计算完成，耗时: {t_elapsed:.2f} s")

        # 4. 构建输出 DataFrame 并保存 timeseries CSV
        t_arr = res["timestamps"]
        wh_head = res["wellhead_head"]
        wh_vel = res["wellhead_velocity"]
        wh_flow = wh_vel * cfg.area
        frac_heads = res["fracture_heads"]
        frac_Qs = res["fracture_Qs"]

        data_cols = {
            "t": t_arr,
            "H_wh": wh_head,
            "Q_wh": wh_flow,
        }
        for k in range(frac_heads.shape[1]):
            data_cols[f"H_f{k+1}"] = frac_heads[:, k]
            data_cols[f"Q_f{k+1}"] = frac_Qs[:, k]

        stage1_df = pd.DataFrame(data_cols)
        stage1_df.to_csv(csv_path, index=False)
        print(f"[4/5] Stage 1 完整时程数据已保存: {csv_path} (行数: {len(stage1_df)})")

    # 5. 加载 PaperA 基准进行物理比对并生成报告与图表
    print(f"[5/5] 读取 PaperA 基准文件: {PAPERA_BENCHMARK_CSV}")
    if not PAPERA_BENCHMARK_CSV.is_file():
        raise FileNotFoundError(f"未找到 PaperA 基准文件: {PAPERA_BENCHMARK_CSV}")
    papera_df = load_timeseries_csv(PAPERA_BENCHMARK_CSV)

    # 计算判定指标
    metrics = compute_stage1_metrics(stage1_df, papera_df, cfg_data)
    json_path = out_dir / "stage1_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"      判定指标 JSON 已生成: {json_path}")

    # 绘制图表
    zoom_png = out_dir / "stage1_pre_shut_in_zoom.png"
    plot_pre_shut_in_zoom(stage1_df, papera_df, zoom_png)
    print(f"      图表 1 已保存: {zoom_png}")

    delta_png = out_dir / "stage1_vs_papera_delta.png"
    plot_full_timeseries_delta(stage1_df, papera_df, delta_png)
    print(f"      图表 2 已保存: {delta_png}")

    # 打印总结汇报
    print("-" * 72)
    print("【阶段一 验收判定汇总】")
    for k, v in metrics["verdicts"].items():
        print(f"  - {k:<25}: {v}")
    print(f"  - 停泵前最大残差波动       : {metrics['pre_shut_in_analysis']['stage1_max_drift_m']:.2e} m (机器浮点极限)")
    print(f"  - Joukowsky 跳变仿真/理论 : {metrics['joukowsky_verification']['dH_sim_m']:.3f} m / {metrics['joukowsky_verification']['dH_theory_m']:.3f} m (误差 {metrics['joukowsky_verification']['error_pct']:.4f}%)")
    print(f"  - 趾端反射波特性           : Stage 1 = {metrics['wave_arrival_verification']['stage1_toe_polarity']}, PaperA = {metrics['wave_arrival_verification']['papera_toe_polarity']}")
    print("=" * 72)
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Verification Runner")
    parser.add_argument("--reuse-csv", action="store_true", help="Reuse existing stage1_timeseries.csv if available")
    args = parser.parse_args()
    run_stage1_verification(reuse_csv=args.reuse_csv)
