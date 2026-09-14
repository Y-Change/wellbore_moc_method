# -*- coding: utf-8 -*-
"""
tests/test_stage1_verification.py — 阶段一（Stage 1）核心物理与交付物自动化回归测试
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from moc_simulate.staged_repair.config_v2 import (
    STAGE1_CONFIG,
    STAGE1_OUTPUT_DIR,
)
from moc_simulate.staged_repair.wellbore_moc_v2 import MocConfig, simulate_wellbore


@pytest.fixture(scope="module")
def stage1_short_sim():
    """运行前 8.5 秒短时仿真以快速校验物理机制"""
    cfg_data = dict(STAGE1_CONFIG)
    cfg_data["tf"] = 8.5
    cfg = MocConfig(
        wellbore_length=cfg_data["wellbore_length"],
        wellbore_diameter=cfg_data["wellbore_diameter"],
        fluid_density=cfg_data["fluid_density"],
        fluid_viscosity=cfg_data["fluid_viscosity"],
        wavespeed=cfg_data["wavespeed"],
        roughness_height=cfg_data["roughness_height"],
        friction_model=cfg_data["friction_model"],
        dt=cfg_data["dt"],
        tf=cfg_data["tf"],
        wellhead_bc=cfg_data["wellhead_bc"],
        pump_shut_time=cfg_data["pump_shut_time"],
        initial_velocity=cfg_data["initial_velocity"],
        initial_head=cfg_data["initial_head"],
        theta=cfg_data["theta"],
        toe_bc="dead_end",
        toe_head=cfg_data.get("toe_head", 300.0),
    )
    res = simulate_wellbore(
        cfg=cfg,
        fracture_positions=cfg_data["fracture_positions"],
        fracture_Cf=[cfg_data["Cf"]] * len(cfg_data["fracture_positions"]),
        fracture_kleak=[cfg_data["kleak"]] * len(cfg_data["fracture_positions"]),
        H_ext=cfg_data["H_ext"],
        store_full_field=True,
    )
    return res


def test_stage1_initial_field_quiescent(stage1_short_sim):
    """验证初始场：死水段 (x > x_{f,last}) 初始流速严格置零"""
    res = stage1_short_sim
    cfg = res["cfg"]
    x_grid = res["x_grid"]
    frac_indices = res["fracture_indices"]
    last_idx = frac_indices[-1]

    V_field_0 = res["velocity"][0]
    H_field_0 = res["head"][0]

    # 1. 验证末簇裂缝至趾端死水段流速全为 0 (初始时刻 t=0 与推进首步 t=dt)
    dead_zone_V = V_field_0[last_idx + 1:]
    assert np.all(dead_zone_V == 0.0), f"死水段流速非零: {dead_zone_V}"
    assert V_field_0[-1] == 0.0, "趾端流速非零"

    # 进一步检验首个时步推进后，死水段流速依旧保持静止
    V_field_1 = res["velocity"][1]
    assert np.all(V_field_1[last_idx + 1:] == 0.0), "首步死水段未保持静止"

    # 2. 验证死水段水头为静止恒定值 H0
    dead_zone_H = H_field_0[last_idx:]
    assert np.allclose(dead_zone_H, cfg.initial_head, atol=1e-10), "死水段水头非恒定"


def test_stage1_pre_shut_in_flatness(stage1_short_sim):
    """验证关泵前 (t < 1.0s) 水头波动为严格水平线 (残差 < 1e-6m)"""
    res = stage1_short_sim
    t = res["timestamps"]
    wh_head = res["wellhead_head"]

    pre_mask = t < 1.0
    drift = np.max(np.abs(wh_head[pre_mask] - wh_head[0]))
    assert drift < 1e-6, f"停泵前水头漂移过大: {drift:.2e} m"


def test_stage1_joukowsky_drop(stage1_short_sim):
    """验证停泵时刻 Joukowsky 瞬态跳变符合解析理论 (误差 < 0.1%)"""
    res = stage1_short_sim
    cfg = res["cfg"]
    wh_head = res["wellhead_head"]
    ts_idx = int(round(cfg.pump_shut_time / cfg.dt))

    dH_sim = wh_head[ts_idx] - wh_head[ts_idx - 1]
    dH_theory = -(cfg.a_adj * cfg.initial_velocity) / 9.81
    rel_err = abs((dH_sim - dH_theory) / dH_theory)
    assert rel_err < 0.001, f"Joukowsky 跳变误差超标: {rel_err*100:.4f}%"


def test_stage1_toe_in_phase_reflection(stage1_short_sim):
    """验证趾端同相反射 (Gamma = +1.0)：到达后水头继续处于负向差量区间"""
    res = stage1_short_sim
    cfg = res["cfg"]
    dt = cfg.dt
    t = res["timestamps"]
    wh_head = res["wellhead_head"]
    dH = wh_head - wh_head[0]

    # 趾端波到达理论时刻 ts + 2*L/a = 1.0 + 2*5000/1450.12 = 7.896s
    t_toe = cfg.pump_shut_time + 2.0 * cfg.wellbore_length / cfg.a_adj
    idx_before = int(round((t_toe - 0.05) / dt))
    idx_after = int(round((t_toe + 0.15) / dt))

    # 此时应保持负压降 (死端同相反射使减压波反射为减压波)
    assert dH[idx_after] < -10.0, f"趾端反射后水头未保持负向: dH={dH[idx_after]:.2f} m"
    assert dH[idx_after] < dH[idx_before], "死端反射减压波应使水头进一步降低"


def test_stage1_deliverables_complete():
    """验证阶段一 4 大交付产物文件完备性"""
    out_dir = STAGE1_OUTPUT_DIR
    csv_file = out_dir / "stage1_timeseries.csv"
    json_file = out_dir / "stage1_metrics.json"
    zoom_png = out_dir / "stage1_pre_shut_in_zoom.png"
    delta_png = out_dir / "stage1_vs_papera_delta.png"

    assert csv_file.is_file() and csv_file.stat().st_size > 10_000_000, "stage1_timeseries.csv 缺失或不完整"
    assert json_file.is_file() and json_file.stat().st_size > 500, "stage1_metrics.json 缺失或为空"
    assert zoom_png.is_file() and zoom_png.stat().st_size > 50_000, "stage1_pre_shut_in_zoom.png 缺失或异常"
    assert delta_png.is_file() and delta_png.stat().st_size > 50_000, "stage1_vs_papera_delta.png 缺失或异常"

    # 读取 JSON 校验各判定均为 PASS
    with open(json_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    for verdict_name, verdict_val in metrics["verdicts"].items():
        assert verdict_val == "PASS", f"指标 {verdict_name} 未通过: {verdict_val}"


def test_stage1_toe_bc_strict_locking():
    """验证 Stage 1 沙箱中 toe_bc 严格锁定为 dead_end，防意外篡改"""
    cfg = MocConfig(toe_bc="reservoir")
    assert cfg.toe_bc == "dead_end", "MocConfig 必须自动纠正/锁定 toe_bc 为 dead_end"

    res = simulate_wellbore(cfg)
    assert res["cfg"].toe_bc == "dead_end", "simulate_wellbore 执行时必须强制锁定 toe_bc=dead_end"
    assert res["toe_velocity"][0] == 0.0, "趾端初始流速必须为 0"


def test_stage1_simulate_case_api():
    """验证统一对外接口 simulate_case 的兼容性与返回数据契约"""
    from moc_simulate.staged_repair.wellbore_moc_v2 import simulate_case
    cfg_data = dict(STAGE1_CONFIG)
    cfg_data["tf"] = 0.05
    res = simulate_case(cfg_data)

    assert "timestamps" in res
    assert "wellhead_head" in res
    assert "toe_head" in res
    assert "junction_heads" in res
    assert len(res["junction_heads"]) == 4
    assert res["wellhead_head"][0] > 330.0, "井口初值水头需含达西水力坡降"


def test_stage1_anti_rubber_stamp_toe_verdict():
    """反向验证：构造倒相数据时，判定逻辑必须真实给出 FAIL 而非被假 PASS 蒙蔽"""
    from moc_simulate.staged_repair.compare_utils import compute_stage1_metrics
    dummy_s1 = pd.DataFrame({
        "t": [0.0, 1.0, 7.846, 8.046],
        "H_wh": [339.5, 191.7, 300.0, 350.0],  # 故意构造反向跃变 +50m
        "Q_wh": [0.015, 0.0, 0.0, 0.0],
    })
    dummy_pa = pd.DataFrame({
        "t": [0.0, 1.0, 7.846, 8.046],
        "H_wh": [347.8, 200.0, 173.6, 228.7],
        "Q_wh": [0.015, 0.0, 0.0, 0.0],
    })
    metrics = compute_stage1_metrics(dummy_s1, dummy_pa, STAGE1_CONFIG)
    assert metrics["verdicts"]["toe_reflection_physics"] == "FAIL", "倒相跳变必须触发 FAIL 判定"
    assert metrics["verdicts"]["overall_stage1"] == "FAIL", "任何子项 FAIL 必须导致 overall_stage1=FAIL"

