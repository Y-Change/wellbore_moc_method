# -*- coding: utf-8 -*-
"""
make_report.py — 汇总方法迁移试验全部结果，生成 REPORT.md。

读取各阶段 JSON（baselines / peeling / G0 / music / damping / ita），
输出 output/analysis/method_migration/REPORT.md。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from analysis.method_migration import common

OUT = Path("output/analysis/method_migration")
DATE = "2026-08-03"


def _load(rel: str) -> Dict:
    with (OUT / rel).open(encoding="utf-8") as fh:
        return json.load(fh)


def _f1_at(rows: List[Dict], truths: Dict[int, List[float]], tol: float) -> float:
    from analysis.unified_evaluation.detection_protocol import score_detections
    vals = []
    for r in rows:
        vals.append(score_detections(r["detected_depths"], truths[r["case_id"]], tol)["f1"])
    return float(np.mean(vals))


def main() -> None:
    idx = {s["case_id"]: s for s in common.load_index()}
    truths = {cid: common.parse_positions(s["positions_str"]) for cid, s in idx.items()}

    cep = _load("baselines/cepstrum_summary.json")
    cwt = _load("baselines/cwt_summary.json")
    peel = _load("peeling_final/nominal_summary.json")
    g0 = _load("peeling_g0/G0_mismatch.json")
    music = _load("music/music_summary.json")
    damp = _load("damping/damping_summary.json")
    ita = _load("ita/ita_refine_summary.json")

    rows = {
        "倒谱 cepstrum": cep["rows"],
        "小波 CWT": cwt["rows"],
        "P0 层剥离": peel["rows"],
    }

    lines: List[str] = []
    lines += [
        "# 方法迁移试验结果报告（lhs_dataset_2000）",
        "",
        f"> 生成：{DATE} ｜ 代码：`analysis/method_migration/` ｜ 数据：`output/lhs_dataset_2000`",
        "> 背景：EXP-20260802-001（方法迁移尝试计划 G0→P0→P1→P2→P3→P4）",
        "",
        "## 0. 数据与协议",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 数据集 | lhs_dataset_2000（brunone，tf=50 s，dt=1 ms，全部 PASS） |",
        "| 井参数 | L=5000 m，a=1450 m/s，V0=1 m/s，reservoir 趾端 |",
        "| 裂缝 | n=1–6，深度 3500–4800 m，间距 5–20 m，Cf/Kleak 对数均匀 |",
        "| 盲协议 | `detection_protocol.py`：固定搜索窗 [3400,4900] m，min-sep 5 m，max 10 峰 |",
        "| 评分 | 固定容差 + 容差扫描（F1 均注明容差） |",
        "",
        "## 1. 总览：F1@10m 与失败模式",
        "",
        "| 方法 | n | F1@10m | F1@20m | F1@40m | 分离成功率 | 计数 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rr in rows.items():
        f1_10 = _f1_at(rr, truths, 10)
        f1_20 = _f1_at(rr, truths, 20)
        f1_40 = _f1_at(rr, truths, 40)
        sep = np.mean([1.0 if r["sep"] else 0.0 for r in rr])
        mae = np.mean([abs(r["count_err"]) for r in rr])
        lines.append(
            f"| {name} | {len(rr)} | {f1_10:.3f} | {f1_20:.3f} | {f1_40:.3f} | "
            f"{sep:.3f} | {mae:.2f} |"
        )
    lines += [
        "",
        "**解读**：",
        "- 倒谱的 F1@10m 仅 0.238，且分离成功率≈0——Brunone 下的深度偏差（已知 10–20 m）",
        "  超过 10 m 容差，且多缝峰重叠无法分离。",
        "- P0 层剥离把单缝深度中位误差从倒谱的 ~190 m 压到 ~4.7 m（模板携带同物理偏差，",
        "  匹配峰天然对准真值），计数 MAE 从 6.4 降到 2.1，且首次出现 9.8% 的完全分离成功率。",
        "- 但 5–20 m 间距仍远低于分辨率极限，多数多缝 case 只能检出 1–2 峰（混叠未根本解决）。",
        "",
        "### 1.1 单缝定位精度（n_frac=1，360 case）",
        "",
        "| 方法 | 中位误差 [m] | ≤10 m 占比 | ≤20 m 占比 |",
        "|---|---:|---:|---:|",
    ]
    for name, rr in rows.items():
        errs = []
        for r in rr:
            if r["n_frac"] != 1 or not r["detected_depths"]:
                continue
            errs.append(abs(r["detected_depths"][0] - truths[r["case_id"]][0]))
        if not errs:
            continue
        e = np.array(errs)
        lines.append(
            f"| {name} | {np.median(e):.1f} | {(e<=10).mean():.3f} | {(e<=20).mean():.3f} |"
        )
    lines += [
        "",
        "## 2. G0 核失配门（固定 198 case 子集）",
        "",
        "| 字典核 | F1@10m | 变化 |",
        "|---|---:|---:|",
    ]
    for v in ["nominal", "a_plus2pct", "V0_plus10pct", "roughness_x2"]:
        s = g0["by_variant"][v]
        delta = s["f1_mean"] - g0["by_variant"]["nominal"]["f1_mean"]
        lines.append(f"| {v} | {s['f1_mean']:.4f} | {delta:+.4f} |")
    lines += [
        "",
        "**G0 门结论：不通过（波速敏感）**。波速 +2% 失配使 F1 从 0.385 崩至 0.021：",
        "模板与基线的全部到时结构整体错位，互相关全部失锁。V0 +10% 与粗糙度 ×2 仅轻微退化",
        "（0.369 / 0.357），说明字典法对波速核极端敏感、对其他核稳健。",
        "→ 任何字典类方法（P0/P2）落地前提：**波速必须独立标定（误差 <1%）**，否则先做波速估计或波速并列字典。",
        "",
        "## 3. P3 阻尼率特征（TWD 视角，全 2000 case）",
        "",
        "| 谐波模态 | 与 Σkleak 秩相关 | 与 n_frac 秩相关 | 与 ΣCf 秩相关 | 与间距 秩相关 |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in range(4):
        sp = damp["summary"]["spearman"][f"mode{m+1}"]
        lines.append(
            f"| mode{m+1} | {sp['vs_logsumkleak']:.3f} | {sp['vs_n_frac']:.3f} | "
            f"{sp['vs_logsumCf']:.3f} | {sp['vs_spacing']:.3f} |"
        )
    lines += [
        "",
        "**结论**：模态阻尼率携带裂缝信息但主要由滤失（kleak）主导（0.87–0.92），",
        "与缝数中等相关（0.44–0.51）、与 Cf 弱相关。作为独立缝数估计器潜力有限，",
        "但作为物理可解释特征（滤失诊断通道）成立——回答 T04「阻尼既是噪声也是信息」。",
        "",
        "## 4. P4 频域子空间（MUSIC，200 case 子集）",
        "",
        "| 方法 | n | F1@10m | 分离成功率 | 计数 MAE |",
        "|---|---:|---:|---:|---:|",
        f"| MUSIC（差分谱/比值谱） | {music['summary']['n_cases']} | "
        f"{music['summary']['f1_mean']:.3f} | {music['summary']['sep_rate']:.3f} | "
        f"{music['summary']['count_mae']:.2f} |",
        "",
        "**负结果**：MUSIC 与比值谱反卷积均无法定位。诊断：",
        "1. 比值谱 R(f)=H(f)/H0(f)−1 并非干净的复指数和——多缝透射-反射耦合（T03）",
        "   与趾端多次回波污染相位，单散射假设不成立；",
        "2. Brunone 频率选择衰减使反射系数随 f 变化，破坏 e^{-j2πfτ} 线性相位模型。",
        "",
        "## 5. P2 两阶段 ITA 精修（60 单缝 case）",
        "",
        "| 阶段 | 中位误差 [m] | ≤5 m | ≤10 m | 改进占比 |",
        "|---|---:|---:|---:|---:|",
        f"| 粗（P0 层剥离） | {ita['summary']['coarse']['median_m']:.2f} | "
        f"{ita['summary']['coarse']['within5m']:.3f} | {ita['summary']['coarse']['within10m']:.3f} | — |",
        f"| 精（ITA 全波形） | {ita['summary']['refined']['median_m']:.2f} | "
        f"{ita['summary']['refined']['within5m']:.3f} | "
        f"{ita['summary']['refined']['within10m']:.3f} | "
        f"{ita['summary']['improved_frac']:.3f}（32% 变差） |",
        "",
        "**结论**：正确核条件下 ITA 精修有真实收益（≤10 m 命中 80%→95%，中位 4.1→3.0 m），",
        "但代价高（60 case × 21 次 MOC ≈ 19 min，12 进程），且继承 G0 的波速敏感约束；",
        "粗定位离群时（少数 case）精修无法挽回。",
        "",
        "## 6. 总体结论与建议",
        "",
        "1. **P0 层剥离是唯一在混叠/偏差场景下有实质改进的经典方法**：单缝定位精度提升一个",
        "   量级、计数大幅改善、首次出现完全分离，且计算开销可接受（2000 case 约 3 min 推理）。",
        "2. **混叠的根本瓶颈未被经典方法打破**：5–20 m 间距远低于有效带宽（约 38 m 级）",
        "   对应的分辨率极限，所有时域方法的分离成功率 <10%。",
        "3. **G0 波速敏感是字典类方法最大的工程风险**（±2% 即崩），优先级高于方法本身调优。",
        "4. **频域子空间路线在本数据上不成立**（单散射假设被多缝干涉与 Brunone 破坏），",
        "   不建议继续投入；若需超分辨，应转向学习型方法（T07）或波速联合估计。",
        "5. **阻尼率可作为滤失诊断通道**（T04 视角互换落地），但不足以单独定缝。",
        "",
        "## 7. 局限",
        "",
        "- G0 失配字典用 20 m 粗网格（nominal 为 10 m），网格差异会轻微放大失配影响；",
        "  但 F1 崩塌幅度（0.385→0.021）远超网格效应。",
        "- P0 字典固定 kleak=1e-4、Cf∈{1e-7,1e-6}，模板核为简化（cf/kleak 第三维未展开）。",
        "- P2/P4 仅子集验证；P2 仅单缝 case。",
        "- 全部为无噪声清洁数据；含噪行为未评估。",
        "",
        "## 8. 关联",
        "",
        "- 计划：[[方法迁移尝试计划]]（research_ob）",
        "- 综述：[[瞬变波异常检测五类方法综述]]（research_ob）",
        "- 代码：`analysis/method_migration/`（common / run_baselines / build_dictionary /",
        "  run_peeling / run_damping / run_music / run_ita_refine / make_report）",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
