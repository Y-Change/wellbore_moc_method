# -*- coding: utf-8 -*-
"""
common.py — 方法迁移试验（EXP-20260802-001）的公共工具。

统一约定
--------
* 数据：``output/lhs_dataset_2000``（2000 case，brunone，tf=50s，dt=1ms，
  L=5000m，a=1450m/s，缝区 3500–4800 m，间距 5–20 m，n_frac 1–6）。
* 检测：``detection_protocol.detect_peaks`` 盲寻峰（搜索窗 [3400, 4900] m
  为先验井段，与逐样本真值无关）。
* 评分：``detection_protocol.score_detections``，主容差 10 m + 容差扫描。
* 聚合：F1@10m、分离成功率、计数 MAE，按 n_frac 与最小间距分档。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from analysis.unified_evaluation.detection_protocol import (
    DetectorConfig,
    ScoringConfig,
    evaluate_blind,
)

DATA_ROOT = Path("output/lhs_dataset_2000")
OUT_ROOT = Path("output/analysis/method_migration")

A_WAVE = 1450.0          # m/s
TS_SHUT = 1.0            # s 停泵时刻
L_WELL = 5000.0          # m
FRAC_ZONE = (3500.0, 4800.0)  # m 先验缝区（LHS 采样区间）
SEARCH_WIN = (3400.0, 4900.0)  # m 固定搜索窗（先验井段，含 100 m 余量）

# 盲检测超参数（与真值无关的固定量，沿用 CEPSTRUM_CONFIG 语义）
DET_CFG = DetectorConfig(
    min_separation_m=5.0,
    height_pct=85.0,
    height_rel=0.03,
    height_abs=0.0,
    prominence_rel=0.0,
    max_peaks=10,
    search_min_m=SEARCH_WIN[0],
    search_max_m=SEARCH_WIN[1],
)
SCORE_CFG = ScoringConfig(tolerance_m=10.0)


@dataclass
class CaseTruth:
    case_id: int
    n_frac: int
    x_f: np.ndarray
    Cf: np.ndarray
    kleak: np.ndarray


def load_index() -> List[Dict]:
    """读取 lhs_metadata.json 的 samples_index。"""
    with (DATA_ROOT / "lhs_metadata.json").open(encoding="utf-8") as fh:
        md = json.load(fh)
    return md["samples_index"]


def load_case(case_id: int) -> Dict:
    """加载单个 case 的 npz（t, H_wh, 真值）。"""
    npz = np.load(DATA_ROOT / "data" / f"case_{case_id:05d}.npz")
    return {
        "t": npz["t"],
        "H_wh": npz["H_wh"],
        "x_f": np.asarray(npz["x_f"], dtype=float),
        "Cf": np.asarray(npz["Cf"], dtype=float),
        "kleak": np.asarray(npz["kleak"], dtype=float),
        "n_frac": int(npz["n_frac"]),
    }


def min_spacing(x_f: Sequence[float]) -> float:
    x = np.sort(np.asarray(x_f, dtype=float))
    if x.size < 2:
        return np.inf
    return float(np.min(np.diff(x)))


def spacing_band(sp: float) -> str:
    if sp <= 10.0:
        return "5-10"
    if sp <= 15.0:
        return "10-15"
    return "15-20"


def eval_depth_response(
    depth: np.ndarray,
    response: np.ndarray,
    true_depths: Sequence[float],
) -> Dict:
    """盲检测 + 评分（检测侧无真值）。"""
    return evaluate_blind(
        np.asarray(depth, dtype=float),
        np.asarray(response, dtype=float),
        np.asarray(true_depths, dtype=float),
        detector_config=DET_CFG,
        scoring_config=SCORE_CFG,
    )


def summarize_results(rows: List[Dict]) -> Dict:
    """聚合逐 case 结果。

    rows 每项至少含：case_id, n_frac, spacing, band, f1, sep, count_err, ...
    """
    f1 = np.array([r["f1"] for r in rows], dtype=float)
    sep = np.array([1.0 if r["sep"] else 0.0 for r in rows], dtype=float)
    cnt = np.array([r["count_err"] for r in rows], dtype=float)
    n = len(rows)

    def _mean(vals: np.ndarray) -> float:
        return float(vals.mean()) if vals.size else float("nan")

    return {
        "n_cases": n,
        "f1_mean": _mean(f1),
        "f1_by_n_frac": {
            str(k): _mean(f1[[r["n_frac"] for r in rows] == k])
            for k in sorted({r["n_frac"] for r in rows})
        },
        "f1_by_band": {
            b: _mean(f1[[r["band"] for r in rows] == b])
            for b in ["5-10", "10-15", "15-20"]
        },
        "sep_rate": _mean(sep),
        "sep_rate_by_band": {
            b: _mean(sep[[r["band"] for r in rows] == b])
            for b in ["5-10", "10-15", "15-20"]
        },
        "count_mae": _mean(np.abs(cnt)),
    }


def save_json(obj, rel: str) -> Path:
    out = OUT_ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return out


def row_from_eval(case_id: int, n_frac: int, sp: float, ev: Dict) -> Dict:
    """把 evaluate_blind 输出压成一行汇总记录。"""
    primary = ev["primary"]
    return {
        "case_id": case_id,
        "n_frac": n_frac,
        "spacing": float(sp),
        "band": spacing_band(sp),
        "n_pred": primary["n_pred"],
        "tp": primary["tp"],
        "fp": primary["fp"],
        "fn": primary["fn"],
        "f1": primary["f1"],
        "sep": bool(primary["separation_success"]),
        "count_err": primary["count_error"],
        "median_err_m": primary["median_error_m"],
        "detected_depths": [d["depth_m"] for d in ev["detected_peaks"]],
    }


def parse_positions(s: str) -> List[float]:
    return [float(x) for x in s.replace(";", ",").split(",")]
