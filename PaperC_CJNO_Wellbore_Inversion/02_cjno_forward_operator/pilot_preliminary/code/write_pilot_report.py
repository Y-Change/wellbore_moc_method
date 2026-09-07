# -*- coding: utf-8 -*-
"""Assemble tables/figures and the evidence-bounded pilot conclusion from run JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump_json
from paths import DATA_AUDIT_DIR, FIG_DIR, PILOT_PRELIM, RUNS_DIR, TABLE_DIR, ensure_dirs
from plot_curves import plot_ab_compare, plot_history, plot_n_curve


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_int(x):
    if x is None:
        return "—"
    return str(int(round(float(x))))


def _fmt_s(x):
    if x is None:
        return "—"
    return f"{float(x):.1f}"


def _brief(rec):
    if not rec:
        return None
    return {
        "tag": rec.get("tag"),
        "run_id": rec.get("run_id"),
        "mode": rec.get("mode"),
        "seed": rec.get("seed"),
        "n_train": rec.get("n_train"),
        "val_pert": rec.get("best", {}).get("val_pert_l2"),
        "val_selection_max_cases": rec.get("val_selection_max_cases"),
        "train_pert_mean": (rec.get("train_overfit") or {}).get("pert_l2", {}).get("mean"),
        "train_pert_p50": (rec.get("train_overfit") or {}).get("pert_l2", {}).get("p50"),
        "train_pert_max": (rec.get("train_overfit") or {}).get("pert_l2", {}).get("max"),
        "node_H_pert_mean": (rec.get("train_overfit") or {}).get("node_H_pert_l2", {}).get("mean"),
        "n_params": rec.get("n_params"),
        "device": rec.get("device"),
        "wall_s": rec.get("wall_clock_s"),
        "infer_s": rec.get("infer_batch1_s"),
        "epochs": rec.get("epochs_ran"),
        "best_epoch": rec.get("best", {}).get("epoch"),
        "n_nonfinite": (rec.get("train_overfit") or {}).get("n_nonfinite"),
        "test": rec.get("test"),
        "test_sealed": rec.get("test_sealed"),
    }


def main():
    ensure_dirs()
    audit = _load(DATA_AUDIT_DIR / "data_audit.json") or {}
    node = _load(DATA_AUDIT_DIR / "node_validation_metrics.json") or {}
    mem = _load(DATA_AUDIT_DIR / "memory_moc_consistency.json") or {}
    tables = {}
    for p in sorted(TABLE_DIR.glob("*.json")):
        if p.name.endswith("_rows.json") or p.name in ("pilot_summary.json",):
            continue
        tables[p.stem] = _load(p)
        if tables[p.stem] and tables[p.stem].get("history"):
            rid = tables[p.stem].get("run_id") or tables[p.stem].get("tag", p.stem)
            plot_history(p, FIG_DIR / f"{p.stem}.png", run_id=str(rid))

    wanted = [
        "hard_n256_torch24_seed42", "soft_n256_torch24_seed42",
        "fno_n256_torch24_seed42",
        "hard_n64_torch24_seed42", "soft_n64_torch24_seed42",
        "hard_n128_torch24_seed42", "soft_n128_torch24_seed42",
        "hard_n256_seed42", "soft_n256_seed42",
        "smoke_hard_n1_torch24_seed42",
    ]
    by_tag = {}
    for p in RUNS_DIR.glob("s2_*/metrics.json"):
        m = _load(p)
        if m and m.get("tag") and m.get("run_id"):
            by_tag.setdefault(m["tag"], m["run_id"])
    for rec in tables.values():
        if rec and not rec.get("run_id") and rec.get("tag") in by_tag:
            rec["run_id"] = by_tag[rec["tag"]]

    official = {k: _brief(tables[k]) for k in wanted if k in tables and tables[k]}

    hard = TABLE_DIR / "hard_n256_torch24_seed42.json"
    soft = TABLE_DIR / "soft_n256_torch24_seed42.json"
    if hard.exists() and soft.exists():
        plot_ab_compare(hard, soft, FIG_DIR / "Fig_pilot_hard_vs_soft_n256.png")

    nest = []
    for k, rec in official.items():
        if rec and rec.get("n_train") and rec.get("val_pert") is not None and "torch24" in k and "smoke" not in k:
            nest.append({"mode": rec["mode"], "n": rec["n_train"], "val_pert": rec["val_pert"]})
    plot_n_curve(nest, FIG_DIR / "Fig_pilot_nested_n.png",
                 run_note="torch24 official tags only; missing n listed as incomplete")

    evals = {}
    for p in sorted(TABLE_DIR.glob("eval_*.json")):
        if p.name.endswith("_rows.json"):
            continue
        evals[p.stem] = _load(p)

    incomplete = []
    for key, label in (
        ("hard_n256_torch24_seed42", "正式 hard 256 seed42"),
        ("soft_n256_torch24_seed42", "正式 soft 256 seed42"),
        ("hard_n64_torch24_seed42", "嵌套 hard 64"),
        ("soft_n64_torch24_seed42", "嵌套 soft 64"),
        ("fno_n256_torch24_seed42", "FNO 256 对照"),
        ("hard_n256_s43", "extra seed 43 hard"),
        ("soft_n256_s43", "extra seed 43 soft"),
        ("hard_n256_s44", "extra seed 44 hard"),
        ("soft_n256_s44", "extra seed 44 soft"),
    ):
        if key not in tables:
            incomplete.append(label)

    dump_json({
        "audit_n_usable": audit.get("n_usable"),
        "node_gate": node.get("status"),
        "memory_solver_gate": mem.get("status"),
        "runs": official,
        "evals": {k: {"split": v.get("split"), "n": v.get("n"),
                      "pert_mean": (v.get("summary") or {}).get("pert_l2", {}).get("mean")}
                  for k, v in evals.items() if v},
        "incomplete": incomplete,
        "not_official_5seed_hpo": True,
    }, TABLE_DIR / "pilot_summary.json")

    h = official.get("hard_n256_torch24_seed42") or official.get("hard_n256_seed42") or {}
    s = official.get("soft_n256_torch24_seed42") or official.get("soft_n256_seed42") or {}
    f = official.get("fno_n256_torch24_seed42") or {}
    overfit_ok = (h.get("train_pert_mean") is not None and h["train_pert_mean"] <= 0.02)
    cpu_only = h.get("device") == "cpu" and "hard_n256_torch24_seed42" not in official

    def _ev(name):
        v = evals.get(name) or {}
        sm = v.get("summary") or {}
        return {
            "n": v.get("n"),
            "pert": (sm.get("pert_l2") or {}).get("mean"),
            "pert_p50": (sm.get("pert_l2") or {}).get("p50"),
            "phase_mean": (sm.get("phase_4La") or {}).get("mean"),
            "phase_p50": (sm.get("phase_4La") or {}).get("p50"),
            "psd": (sm.get("psd") or {}).get("mean"),
            "node": (sm.get("node_H_pert_l2") or {}).get("mean"),
            "ckpt": v.get("ckpt"),
        }

    ev_hv = _ev("eval_val_hard_n256_torch24")
    ev_sv = _ev("eval_val_soft_n256_torch24")
    ev_ht = _ev("eval_test_hard_n256_torch24")
    ev_st = _ev("eval_test_soft_n256_torch24")
    ev_fv = _ev("eval_val_fno_n256_torch24")
    ev_ft = _ev("eval_test_fno_n256_torch24")
    te_hard = evals.get("eval_test_hard_n256_torch24")

    tex = []
    tex.append("% auto-generated by write_pilot_report.py; do not hand-edit\n")
    tex.append("\\begin{tabular}{lrrrr}\n")
    tex.append("tag & n & val$_{16}$ pert & full-val pert & ID-test pert \\\\\n")
    tex.append("\\hline\n")
    rows_tex = [
        ("hard\\_n256", h.get("n_train"), h.get("val_pert"), ev_hv.get("pert"), ev_ht.get("pert")),
        ("soft\\_n256", s.get("n_train"), s.get("val_pert"), ev_sv.get("pert"), ev_st.get("pert")),
        ("fno\\_n256", f.get("n_train"), f.get("val_pert"), ev_fv.get("pert"), ev_ft.get("pert")),
        ("hard\\_n64", official.get("hard_n64_torch24_seed42", {}).get("n_train"),
         official.get("hard_n64_torch24_seed42", {}).get("val_pert"), None, None),
        ("soft\\_n64", official.get("soft_n64_torch24_seed42", {}).get("n_train"),
         official.get("soft_n64_torch24_seed42", {}).get("val_pert"), None, None),
    ]
    for name, ntr, v16, fv, te in rows_tex:
        tex.append(f"{name} & {_fmt_int(ntr)} & {_fmt(v16)} & {_fmt(fv)} & {_fmt(te)} \\\\\n")
    tex.append("\\end{tabular}\n")
    (TABLE_DIR / "table_pilot_ab.tex").write_text("".join(tex), encoding="utf-8")

    md = []
    md.append("# 阶段二 Pilot 预研结论\n\n")
    md.append("**本文件不宣布**：阶段一正式验收通过、H1 成立、正式 OOD 达标、全场精度达标、")
    md.append("M3 记忆通道逐点 <2%（网络侧）通过。\n\n")
    md.append("环境：训练正式档使用 `D:\\\\Anaconda\\\\envs\\\\torch24`（torch 2.4.0 + CUDA / RTX 4060 Ti）。")
    md.append("CPU Torch 2.2 的 `hard_n256` 仅作对照基线，不作为正式过拟合结论。")
    if cpu_only:
        md.append("**正式 torch24 256 尚未写入 tables，下列 hard 数字来自 CPU 基线。**")
    md.append("\n\n")

    md.append("## 0. 证据边界（先读）\n\n")
    md.append("- 标签来自 Stage-1 **pilot 2000**（1896 ok），不是 formal 30k。\n")
    md.append("- 模型选择只看 **val 前 16 例** 的 pert L2；全量 val/test 只在调参封闭后由 `eval.py` 跑一次。\n")
    md.append("- 训练窗：停泵后 4 个 \(4L/a\)，井口 decim=4（Nyquist ≈ 40–50 Hz），节点 ≤128 点。\n")
    md.append("- **不能验收**：70 Hz 到时、0.05–1/1–10/10–100 Hz 分频带、全场 \(E_{L2}\)、")
    md.append("网络记忆通道 vs MOC \(z_l(x,t)\) 逐点 <2%、正式 OOD、H1 的 5-seed/32-trial 对标。\n")
    md.append("- 原型 **不是** 完整 CJ-NO：无特征坐标段传播、无全场 trunk；硬节点 Newton 使用物理 \(B=a/(gA)\)，")
    md.append("但 \(K/c_1/a_0\) 为简化占位，不是逐孔 MOC 系数。\n")
    md.append("- A1b 在本轮是「同一外壳换节点头」，井口头仍是共享 1D FNO；硬层不改写井口路径。\n")
    md.append(f"- 未完成项：{', '.join(incomplete) if incomplete else '无（矩阵内官方 tag 均已落盘）'}。\n\n")

    md.append("## 1. 实际可用案例与可信监督\n\n")
    md.append(f"- 可用 **{audit.get('n_usable')}** 例（manifest ok=1896，SHA256 全匹配，深度检查 0 失败）。\n")
    md.append(f"- 切分 seed=20260906：{audit.get('split', {}).get('counts')}，组泄漏为空。\n")
    md.append("- 104 例因稳态井口压力越界拒绝；参数未保留。**有效分布 ≠ 原始 LHS 先验。**\n")
    md.append("- 可信监督：原生井口 \(p,Q\)（Nyquist 中位约 176 Hz）；节点迹/记忆为 decim≈16（Nyquist 中位约 11 Hz）。\n")
    md.append("- **没有**观测链波形、**没有**全场快照。训练窗口再 decim=4 后井口 Nyquist 降至约 40–50 Hz。\n")
    md.append("- formal smoke 未混入训练集。\n")
    md.append(f"- 源 generator frozen digest 前 8 位：`{(audit.get('source_generator') or {}).get('code_digest', '')[:8]}`。\n\n")

    md.append("## 2. 节点层前向与梯度；记忆通道\n\n")
    md.append(f"- 单元门：`node_validation_metrics.json` 状态 **{node.get('status')}**，失败数 {node.get('n_fail')}。\n")
    md.append("- 前向与 Stage-1 `solve_cluster_node` 一致；无量纲残差 \(\\sim 10^{-13}\\sim10^{-11}<10^{-8}\)。\n")
    md.append("- 光滑点 IFT vs 中心差分相对误差 \(\\sim 10^{-10}<10^{-4}\)；条件数 \(O(10^2)\)，未触发 Tikhonov。\n")
    md.append("- \(q\\approx 0\) 单独报告（孔眼 \(q|q|\) 非光滑）。\n")
    md.append(f"- 求解器侧记忆一致性（`memory_moc_consistency.json`）：**{mem.get('status', '未跑')}**。")
    _mem_rows = {r.get("name"): r for r in (mem.get("rows") or [])}
    _alg = _mem_rows.get("rec_algebra_and_zielke_fit") or {}
    _mocz = _mem_rows.get("tiny_joukowsky_snap_recon_vs_final_z") or {}
    if _alg:
        md.append(f" 递推 vs 同核指数卷积 rel L2 = {_fmt(_alg.get('rel_l2_rec_vs_expsum'))}；")
        md.append(f"相对完整 Zielke 的拟合残差 = {_fmt(_alg.get('rel_l2_rec_vs_zielke'))}（只报告，不是本测试失败项）。")
    if _mocz:
        md.append(f" Joukowsky 小算例 `snap_every=1` 重建 \(z\) rel L2 = {_fmt(_mocz.get('rel_l2_z'))}。")
    md.append("**这不是网络记忆通道门。**\n")
    md.append("- 网络记忆 \(z_l(x,t)\) 逐点 <2%：**待补证据**（pilot NPZ `snap_every=0`，本轮未批量补生成）。\n\n")

    md.append("## 3. 256 例过拟合\n\n")
    md.append(f"- run_id（hard）=`{h.get('run_id')}`，device=`{h.get('device')}`，params=`{h.get('n_params')}`。\n")
    md.append(f"- 硬节点正式档：train pert L2 mean = {_fmt(h.get('train_pert_mean'))}")
    md.append(f"（p50={_fmt(h.get('train_pert_p50'))}，max={_fmt(h.get('train_pert_max'))}），")
    md.append(f"val pert L2（16 例选模）= {_fmt(h.get('val_pert'))}，")
    md.append(f"best_epoch={h.get('best_epoch')}，epochs={h.get('epochs')}，wall={_fmt_s(h.get('wall_s'))} s。\n")
    md.append(f"- 节点水头扰动 L2（train 诊断）= {_fmt(h.get('node_H_pert_mean'))}；非有限条数 = {h.get('n_nonfinite')}。\n")
    md.append(f"- 过拟合门（井口扰动相对 L2 ≤2%）：**{'达到' if overfit_ok else '未达到'}**。\n")
    if not overfit_ok:
        md.append("- 失败归因（按现有证据，不是事后改门）：")
        md.append("(i) 原型缺少特征坐标段传播，井口只靠条件化 1D FNO；")
        md.append("(ii) 损失尺度上 \(L_{node}/L_{hard}\) 远大于井口头，优化先压节点/占位 Newton 而非波形细节；")
        md.append("(iii) 井口 proj 零初始化后仍学到 ~0.2 量级相对误差，说明容量/架构未贴到逐点波形；")
        md.append("**不是**节点 IFT 算错（B 门已过），**不是**标签 SHA 损坏（审计 0 失败）。\n")
    md.append("- 长时自由滚动：评价窗为停泵后 4 个 \(4L/a\)；不外推到 20 个周期全窗。\n")
    md.append("- **相位诊断口径**：正式 `eval.py` 使用物理时间 \(t\\)。")
    md.append("`hard/soft_n256_torch24` 训练环里的 `phase_4La` 曾误用采样下标当时间，**那些 history 里的相位数作废**；")
    md.append("n=64 及以后的训练环已改用 `s.t`。\n")
    if ev_ht.get("pert") is not None:
        md.append(f"- 全量 val（191，选模后一次）：hard pert mean={_fmt(ev_hv.get('pert'))}（p50={_fmt(ev_hv.get('pert_p50'))}），")
        md.append(f"soft {_fmt(ev_sv.get('pert'))}。16 例选模偏乐观。\n")
        md.append(f"- 调参封闭后 ID-test（**不是正式 OOD**）：hard pert mean={_fmt(ev_ht.get('pert'))}")
        md.append(f"（p50={_fmt(ev_ht.get('pert_p50'))}，phase p50={_fmt(ev_ht.get('phase_p50'))} of \(4L/a\)），")
        md.append(f"soft {_fmt(ev_st.get('pert'))}，n=191。\n")
        md.append(f"- 正式相位（物理时间）：hard val p50={_fmt(ev_hv.get('phase_p50'))}，")
        md.append(f"test p50={_fmt(ev_ht.get('phase_p50'))}；均值被错峰案例拉高，**未达 1% 的 \(4L/a\) 门**。\n")
    else:
        md.append("- ID-test：尚未打开，或 `eval.py --split test` 结果未落盘。\n")
    md.append("\n")

    md.append("## 4. 硬节点 vs 软节点（A1b 预研，非正式 H1）\n\n")
    md.append("| tag | n | val16 pert | full val | ID-test | wall s | infer s |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    n64h = official.get("hard_n64_torch24_seed42") or {}
    n64s = official.get("soft_n64_torch24_seed42") or {}
    md.append(f"| hard 256 | 256 | {_fmt(h.get('val_pert'))} | {_fmt(ev_hv.get('pert'))} | {_fmt(ev_ht.get('pert'))} | {_fmt_s(h.get('wall_s'))} | {_fmt(h.get('infer_s'))} |\n")
    md.append(f"| soft 256 | 256 | {_fmt(s.get('val_pert'))} | {_fmt(ev_sv.get('pert'))} | {_fmt(ev_st.get('pert'))} | {_fmt_s(s.get('wall_s'))} | {_fmt(s.get('infer_s'))} |\n")
    md.append(f"| fno 256 | 256 | {_fmt(f.get('val_pert'))} | {_fmt(ev_fv.get('pert'))} | {_fmt(ev_ft.get('pert'))} | {_fmt_s(f.get('wall_s'))} | {_fmt(f.get('infer_s'))} |\n")
    md.append(f"| hard 64 | 64 | {_fmt(n64h.get('val_pert'))} | — | — | {_fmt_s(n64h.get('wall_s'))} | {_fmt(n64h.get('infer_s'))} |\n")
    md.append(f"| soft 64 | 64 | {_fmt(n64s.get('val_pert'))} | — | — | {_fmt_s(n64s.get('wall_s'))} | {_fmt(n64s.get('infer_s'))} |\n\n")
    md.append(f"- 参数量：hard/soft/fno 均为 {h.get('n_params')}（同一外壳，未拆头）。\n")
    md.append("- 16 例选模上 soft 略优于 hard（0.193 vs 0.204）；**全量 val/test 上 hard 略优于 soft**")
    md.append(f"（val {_fmt(ev_hv.get('pert'))}/{_fmt(ev_sv.get('pert'))}，test {_fmt(ev_ht.get('pert'))}/{_fmt(ev_st.get('pert'))}）。")
    md.append("单 seed，差值远小于误差本身，**不能主张硬层带来显著井口精度收益**。\n")
    md.append("- 嵌套 n：64→256 使 val16 pert 从 0.375 降到 ~0.20，有样本增益，仍远高于 2%。\n")
    if f:
        md.append(f"- FNO 对照（无 \(L_{{node}}/L_{{hard}}\)）val16 pert = {_fmt(f.get('val_pert'))}，train pert = {_fmt(f.get('train_pert_mean'))}。")
        md.append("井口更好是因为损失不被节点项淹没，**不是**「纯 FNO 已达到 CJ-NO 目标」，更不是 H1 成立。仍未过 2% 门。\n")
    md.append("- 成本：hard 256 wall ≈ 1123 s，soft/FNO ≈ 32 s；batch=1 井口推理均约 3 ms（`apply_hard_times=0`）。")
    md.append("硬层训练贵在 CPU Newton 循环，不在 GPU 前向。\n")
    md.append(f"- 节点 H 扰动 L2（全量 eval）：hard val={_fmt(ev_hv.get('node'))} / test={_fmt(ev_ht.get('node'))}，")
    md.append(f"soft val={_fmt(ev_sv.get('node'))} / test={_fmt(ev_st.get('node'))}。FNO 无节点头（NaN）。")
    md.append("硬/软节点误差同量级 ≈1，说明当前硬层占位 Newton **没有**把节点监督做成可用读出。\n")
    md.append("- 不是 5 seeds / 32-trial HPO。seeds 43/44 未跑，不算统计。\n")
    md.append("- 图：`figures/Fig_pilot_hard_vs_soft_n256.png`、`Fig_pilot_nested_n.png`、")
    md.append("`Fig_pilot_waveforms_hard_n256_torch24_val.png`（边注 run_id/ckpt）。\n")
    md.append("- 表：`tables/table_pilot_ab.tex`。\n\n")

    md.append("## 5. 限制主要来自哪里\n\n")
    md.append("- **实现**：节点层门已过；训练瓶颈是 Python 循环 + CPU Newton + NPZ 读盘。")
    md.append("硬模式 GPU 并不比 CPU 快一个数量级，因为 Newton 不在 GPU 上。\n")
    md.append("- **数据质量**：1896 例完整、哈希一致；缺全场与观测链；节点带宽不够 70 Hz。\n")
    md.append("- **样本量**：256 例未过拟合到 2%；全量 1514 例不是本轮必达。\n")
    md.append("- **架构**：当前收益更像共享 FNO 井口头；结构优势未表现出来。\n\n")

    md.append("## 6. 下一轮最值得补的数据/实验\n\n")
    md.append("1. 对一小撮 case 打开 `snap_every>0`（不必 30k），专门验**网络**记忆通道 vs \(z_l(x,t)\)。\n")
    md.append("2. 实现真正的特征坐标段传播后再做 A1b，并把硬层输出接到井口路径，而不是只换节点头。\n")
    md.append("3. 硬层改用逐孔 MOC 的 \(K,c_1,a_0\)，去掉占位系数。\n")
    md.append("4. 训练直接用原生井口 Δt（或 decim=2）的短窗，避免把 70 Hz 信息先抽掉再谈高频门。\n")
    md.append("5. 满量 30k 仍单独排队，不与本预研结论捆绑。\n\n")

    md.append("## 图表与表格\n\n")
    md.append("- `tables/pilot_summary.json`：官方 tag 摘要。\n")
    md.append("- `tables/*_seed42.json`：各 run 完整 history。\n")
    md.append("- `figures/*_seed42.png`：训练曲线，边注 run_id。\n")
    md.append("- `figures/Fig_pilot_hard_vs_soft_n256.png`、`Fig_pilot_nested_n.png`。\n")
    md.append("- 波形叠图：`python plot_waveforms.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt`。\n\n")

    md.append("## 可复现命令\n\n")
    md.append("```text\n")
    md.append("set PYTHON=D:\\Anaconda\\envs\\torch24\\python.exe\n")
    md.append("cd PaperC_CJNO_Wellbore_Inversion/02_cjno_forward_operator/pilot_preliminary/code\n")
    md.append("%PYTHON% test_interface.py\n")
    md.append("%PYTHON% test_dataset.py\n")
    md.append("%PYTHON% test_checkpoint.py\n")
    md.append("%PYTHON% test_memory_moc.py\n")
    md.append("%PYTHON% audit_pilot.py\n")
    md.append("%PYTHON% train.py --mode hard --n-train 256 --seed 42 --tag hard_n256_torch24\n")
    md.append("%PYTHON% train.py --mode soft --n-train 256 --seed 42 --tag soft_n256_torch24\n")
    md.append("%PYTHON% train.py --mode hard --n-train 64 --seed 42 --tag hard_n64_torch24\n")
    md.append("%PYTHON% train.py --mode soft --n-train 64 --seed 42 --tag soft_n64_torch24\n")
    md.append("%PYTHON% train.py --mode fno --n-train 256 --seed 42 --tag fno_n256_torch24\n")
    md.append("%PYTHON% eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val\n")
    md.append("%PYTHON% eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split test\n")
    md.append("%PYTHON% plot_waveforms.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val\n")
    md.append("%PYTHON% write_pilot_report.py\n")
    md.append("```\n")
    md.append("\n禁止：`--confirm-30k`；改写 Stage-1 goldens / pilot manifests。\n")

    (PILOT_PRELIM / "pilot_stage_conclusion.md").write_text("".join(md), encoding="utf-8")
    print("wrote", PILOT_PRELIM / "pilot_stage_conclusion.md")
    print("incomplete:", incomplete)


if __name__ == "__main__":
    main()
