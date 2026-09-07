# -*- coding: utf-8 -*-
"""Assemble Round-3 markdown deliverables from measured JSON."""
from __future__ import annotations

import json
from pathlib import Path

from paths import MANIFEST_DIR, ROUND3, TABLE_DIR


def load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def status_of(x, default="NOT_RUN"):
    if x is None:
        return default
    if isinstance(x, dict):
        return x.get("status", default)
    return str(x)


def main():
    a = load(MANIFEST_DIR / "metrics_regression_tests.json")
    b = load(MANIFEST_DIR / "optimization_diagnosis.json") or load(TABLE_DIR / "optimization_diagnosis.json")
    c = load(MANIFEST_DIR / "production_rollout_gates.json")
    fields = load(MANIFEST_DIR / "input_field_consumption.json")
    docs = ROUND3 / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    # --- optimization_diagnosis.md ---
    lines = ["# Round-3 B：8 案有限预算对照\n",
             "冻结嵌套前缀 8 案、dt=0.02 s、T=400、QueryFourierMLP、同一特征与 batch 计划。\n",
             "门槛：train mean 扰动 L2 ≤ 2%。不把预算耗尽写成收敛。\n"]
    if b is None:
        lines.append("**状态：NOT_RUN / 证据未落地。**\n")
    else:
        for arm in ("B1", "B2"):
            m = b.get(arm)
            if not m:
                lines.append(f"## {arm}\n\nNOT_RUN\n")
                continue
            lines.append(f"## {arm} ({m.get('status')})\n")
            lines.append(f"- run_id: `{m.get('run_id')}`\n")
            lines.append(f"- stop_reason: **{m.get('stop_reason')}**（不是收敛声明）\n")
            lines.append(f"- steps: {m.get('optimizer_steps_completed')}  wall: {m.get('wall_clock_s'):.1f}s\n")
            be = m.get("best", {})
            lines.append(
                f"- best train pert L2 mean/p90/max: "
                f"{be.get('pert_l2_mean'):.4f} / {be.get('pert_l2_p90'):.4f} / {be.get('pert_l2_max'):.4f}\n"
            )
            lines.append(f"- clip_frac: {m.get('clip_frac')}\n")
            lines.append("- 逐案:\n")
            for r in be.get("rows", []):
                lines.append(
                    f"  - {r['case_id']}: rel={r['rel_l2']:.4f} silent={r['silent_p']} "
                    f"||y||={r['target_l2']:.3e}\n"
                )
        lines.append(f"\nboth_fail_within_budget: {b.get('both_fail_within_budget')}\n")
        if b.get("singles"):
            lines.append("\n## 单例拟合\n")
            for s in b["singles"]:
                be = s.get("best", {})
                lines.append(
                    f"- {s.get('case_ids')} status={s.get('status')} "
                    f"mean={be.get('pert_l2_mean')} stop={s.get('stop_reason')} "
                    f"steps={s.get('optimizer_steps_completed')}\n"
                )
        if b.get("extra_seed"):
            lines.append(f"\n预先登记 extra seed 复跑：{b['extra_seed'].get('status')} "
                         f"mean={b['extra_seed'].get('best', {}).get('pert_l2_mean')}\n")
        else:
            lines.append("\n预先登记 extra seed：**未触发**（无过门臂）。\n")
    (docs / "optimization_diagnosis.md").write_text("".join(lines), encoding="utf-8")

    # --- reference_parity.md ---
    pl = ["# Round-3 C：同物理参考对齐\n",
          f"F 定义：`{json.dumps((c or {}).get('F_definition'), ensure_ascii=False)}`\n",
          "禁止用等分总流量 / 复制平均 pc / F=0 冒充真状态。\n"]
    if c is None:
        pl.append("**状态：NOT_RUN。**\n")
    else:
        pl.append(f"cases: {c.get('case_ids')}\n")
        pl.append(f"alignment all_pass (0.5% Darcy twin vs ref): {c.get('alignment_gate', {}).get('all_pass')}\n")
        pl.append("这是新设数值对齐门，不替换学习 2% 门。Darcy 过门 ≠ 生产 ZVB 过门。\n")
        for r in c.get("rows", []):
            pl.append(f"\n## {r['case_id']}\n")
            pl.append(f"- T_long={r.get('T_long_s'):.3f}s  period={r.get('period_s'):.3f}s\n")
            rf = r.get("refine_darcy", {})
            pl.append(f"- grid 64 vs 128 pert L2: {rf.get('rel_pert_l2_64_vs_128')} "
                      f"small_enough={rf.get('small_enough_for_0p5pct_gate')}\n")
            for fr in ("none", "darcy"):
                tw = r.get(f"twin_vs_ref_{fr}", {})
                pl.append(f"- twin vs ref [{fr}] pert L2={tw.get('wh_pert_l2')} "
                          f"pass_0.5%={tw.get('pass_0p5pct')} n_fail={tw.get('n_fail_twin')}/{tw.get('n_fail_ref')}\n")
                tt = r.get(f"torch_vs_twin_{fr}_short", {})
                pl.append(f"- torch vs twin [{fr}] short: {tt}\n")
            pl.append(f"- zvb vs darcy ref: {r.get('zvb_vs_darcy_ref')}\n")
            pl.append(f"- steady drift: {r.get('steady_drift_none')}\n")
            pl.append(f"- node→WH one-way: {r.get('node_to_wh_one_way')}\n")
            pl.append(f"- valve→node: {r.get('valve_to_node')}\n")
    (docs / "reference_parity.md").write_text("".join(pl), encoding="utf-8")

    # --- production_rollout_gates.md ---
    gl = ["# Round-3 C：生产 Tensor rollout 门\n",
          "同一 `ProductionMOC`，不用 mini_loop 过门。\n"]
    if c is None:
        gl.append("**状态：NOT_RUN。**\n")
    else:
        gl.append(f"- grad: {c.get('production_grad_status')} `{c.get('production_grad')}`\n")
        gl.append(f"- real Newton found: {c.get('real_newton_failure', {}).get('found')}\n")
        gl.append(f"- Newton blocks train step: {c.get('newton_blocks_train_step')}\n")
        gl.append(f"- tensor ZVB: {c.get('zvb_tensor')}\n")
        gl.append(f"- mini_loop_used: {c.get('mini_loop_used')}\n")
    (docs / "production_rollout_gates.md").write_text("".join(gl), encoding="utf-8")

    # --- claim_evidence_table.md ---
    claims = []
    claims.append(("A metrics counterexamples", "PASS" if a and a.get("all_pass") else "FAIL",
                   "tests/test_metrics.py", "负峰/多峰/极性/窗外/截断/静默/抵消/歧义", "合成信号，不是现场首达"))
    claims.append(("A isolation by case_id", "PASS" if a and a.get("all_pass") else "FAIL",
                   "tests/test_dataset.py + isolation.py", "test split / all_usable / test case_id 均 raise",
                   "仅登记 train；未宣称 val 诊断"))
    claims.append(("A checkpoint resume", "PASS" if a and a.get("all_pass") else "FAIL",
                   "tests/test_checkpoint.py", "连续 vs 恢复后下一步权重一致", "Tiny SGD，非 CJ-NO"))
    if b:
        for arm in ("B1", "B2"):
            m = b.get(arm) or {}
            claims.append((f"{arm} 8-case within budget", m.get("status", "NOT_RUN"),
                           m.get("ckpt_best", "train_b.py"),
                           f"train mean pert L2={m.get('best', {}).get('pert_l2_mean')} "
                           f"stop={m.get('stop_reason')}",
                           "8s 查询、共享 MLP、无节点损失"))
    else:
        claims.append(("B1/B2", "NOT_RUN", "train_b.py", "—", "A 通过后才跑"))
    if c:
        claims.append(("C Darcy twin alignment 0.5%",
                       "PASS" if c.get("alignment_gate", {}).get("all_pass") else "FAIL",
                       "moc_twin.py vs moc_solver.simulate",
                       str(c.get("alignment_gate")),
                       "none/Darcy 离散孪生；不是 ZVB 生产物理"))
        claims.append(("C production autograd vs FD",
                       c.get("production_grad_status", "NOT_RUN"),
                       "production_rollout.py ProductionMOC",
                       str(c.get("production_grad", {}).get("best_rel")),
                       "冻结初态；光滑点；分支切换另报"))
        claims.append(("C real Newton blocks step",
                       "PASS" if c.get("newton_blocks_train_step") else
                       ("INSUFFICIENT" if not c.get("real_newton_failure", {}).get("found") else "FAIL"),
                       "NewtonFailedError in ProductionMOC.step",
                       str(c.get("newton_block_error")),
                       "病理 K 使 solve_cluster_node ok=False，非 inject raise"))
        claims.append(("C ZVB tensor production", c.get("zvb_tensor", "NOT_RUN"),
                       "—", "未实现 Tensor ZVB 记忆", "Darcy 过门不得记为生产物理过门"))
    else:
        claims.append(("C gates", "NOT_RUN", "run_c_gates.py", "—", "—"))

    cl = ["# claim_evidence_table\n\n",
          "| 主张 | 状态 | 代码路径 | 数值断言 | 适用范围 |\n|---|---|---|---|---|\n"]
    for name, st, path, num, scope in claims:
        cl.append(f"| {name} | {st} | `{path}` | {num} | {scope} |\n")
    (docs / "claim_evidence_table.md").write_text("".join(cl), encoding="utf-8")

    # --- round3_audit.md ---
    b1s = status_of((b or {}).get("B1"))
    b2s = status_of((b or {}).get("B2"))
    calign = "NOT_RUN" if c is None else ("PASS" if c.get("alignment_gate", {}).get("all_pass") else "FAIL")
    cgrad = "NOT_RUN" if c is None else c.get("production_grad_status", "NOT_RUN")
    czvb = "NOT_RUN" if c is None else c.get("zvb_tensor", "NOT_RUN")
    seg_ok = (calign == "PASS" and cgrad == "PASS" and c.get("newton_blocks_train_step") and czvb == "PASS")
    audit = f"""# Round-3 审计

日期：2026-09-06。工作根：`02_cjno_forward_operator/pilot_round3`。  
未覆盖 Round1/Round2 原始结果。未打开旧 test。未训 256、未开 D 组。

本文件区分：**保留事实** / **必须修正的解释** / **未验证主张**。  
未运行、超时、证据不足单独标记，不与 FAIL/PASS 混写。

## 1. 哪些 Round2 数字保留，哪些解释必须修正？

### 保留事实（Round2 落盘，本轮不改写）

- n=1 PASS：`r2_overfit_n1_20260906_174918_934f991e`，train pert L2 mean=0.0100。
- n=1 失败保留：`r2_overfit_n1_20260906_174839_708825a3`，best=0.0120，last=0.063。
- n=8 FAIL：`r2_overfit_n8_20260906_174935_f6f7a808`，0.135 / 0.274 / 0.430。
- n=64 FAIL：`r2_overfit_n64_20260906_174959_f8c81858`，0.196 / 0.414 / 0.633。
- n=256 **未跑**。
- 节点分支覆盖 PASS（stiff_guard / 反向 q / 近零根等）。
- Round2 闭环门当时报 PASS，但证据范围是 **1-cell + mini_loop 梯度 + 注入异常**，以及 F=0 / 等分 q、pc。
- 8 案 ID：`pilot_01342, 00163, 00543, 01320, 00049, 00888, 01900, 00880`。

### 必须修正的解释

- Round2「首波」因 8 s 查询短于 ~9 s 周期全部 `ref_no_peak`：这是**窗口定义**，不是相位已测。Round3 指标已区分「关阀后第一峰」与「节点反射首达」，歧义/失败不进成功均值，验收用绝对相位误差。
- Round2 静默曾让预测进入分母：已废止。静默只由目标 + 冻结 Joukowsky 尺度决定。
- Round2 1-cell vs MOC 26–56% **不能**读成「同物理闭环已准」。当时不是真逐孔状态，摩阻/网格也未对齐。
- Round2 梯度过门不能代表生产 rollout：用了另一个 mini_loop。
- Round2 Newton-fail 测试是注入异常，不是真实求解失败。
- 新归一化（float64 raw、常数维=0）**不得**套用旧检查点冒充同模型数字。旧结果留在旧协议。

### 未验证 / 本轮仍禁止声称

- 完整 CJ-NO、正式 OOD、70 Hz、全场精度、H1、正式 30k。

## 2. 8 案失败：哪些替代原因被排除，哪些仍未排除？

B1 状态：**{b1s}**  
B2 状态：**{b2s}**  
both_fail_within_budget：{(b or {}).get('both_fail_within_budget', 'NOT_RUN')}

解释规则（门槛未改）：

- 仅当 B1 在本预算过门：支持「原短预算/调度是重要因素」。
- 仅当只有 B2 过门：支持「幅值权重是重要因素」。
- 均未过门：只报本预算内未过门，**不**宣布架构不可能；单例拟合用来区分单波形 vs 联合拟合。

QueryFourierMLP **实际未消费**：I_f, R_f, G_l, kappa, p_res, roughness, TVD, 逐孔异质性, 初始 q/pc/F。  
输入补全未混进 B1/B2。

仍未排除（若两臂未过门则全部保留）：容量/谱偏置、8 s 窗截断周期、缺少节点损失、缺少未消费物理场、联合拟合干扰等。

## 3. 实际闭环现在是否同时做到准确、因果、可微？

必须三项同一实现：

| 项 | 状态 | 说明 |
|---|---|---|
| 同物理准确（none→Darcy 对齐门 ≤0.5%） | {calign} | 新数值门，不是学习 2% 门 |
| 生产 ZVB 物理 | {czvb} | Darcy 过 ≠ 生产过 |
| 因果（x/a 与往返） | 见 reference_parity | 与「关阀后第一峰」不是同一指标 |
| 同一 ProductionMOC 可微 | {cgrad} | autograd vs 多步长中心差分；冻结初态 |
| 真实 Newton 失败阻断训练步 | {c.get('newton_blocks_train_step') if c else 'NOT_RUN'} | |

## 4. 段传播学习可以启动，还是仍有具体阻塞项？

启动条件（本轮先写接口、不训大模型）：同物理前向 + 真状态递推 + **生产** rollout 梯度过门。

当前判定：{'可以提交最小接口方案' if seg_ok else '仍有阻塞，不启动段传播学习训练'}。

阻塞项：
- ZVB Tensor 生产路径：{czvb}
- 对齐门：{calign}
- 梯度门：{cgrad}
- Newton 阻断：{c.get('newton_blocks_train_step') if c else 'NOT_RUN'}

若提交接口，学习模块只替代/修正段传播出射特征；hard 节点固定；记忆由确定性递推维护；跨时间梯度走 Tensor 状态；predict 仍禁止未来标签。下一轮若启动：1→8，再讨论 64，不直接 256。
"""
    (docs / "round3_audit.md").write_text(audit, encoding="utf-8")

    readme = f"""# PaperC Stage-2 Round 3

指标修复、短训失败归因、同物理可微闭环。  
**不**证明 CJ-NO 已成功。历史证据在 `pilot_preliminary/` 与 `pilot_round2/`，本目录不覆盖它们。

环境：`D:\\Anaconda\\envs\\torch24\\python.exe`（Python 3.9, torch 2.4.0 + CUDA）。

## 已实际跑过的复现命令

在 `02_cjno_forward_operator/pilot_round3` 下：

```text
D:\\Anaconda\\envs\\torch24\\python.exe tests\\run_a_suite.py
D:\\Anaconda\\envs\\torch24\\python.exe code\\run_a_and_fields.py
```

A 未通过时不要跑 B。A 已通过后：

```text
D:\\Anaconda\\envs\\torch24\\python.exe code\\run_b_suite.py
D:\\Anaconda\\envs\\torch24\\python.exe code\\run_c_gates.py
D:\\Anaconda\\envs\\torch24\\python.exe code\\write_round3_reports.py
```

单臂：

```text
D:\\Anaconda\\envs\\torch24\\python.exe code\\train_b.py --arm B1 --n 8
D:\\Anaconda\\envs\\torch24\\python.exe code\\train_b.py --arm B2 --n 8
```

## 交付

- `docs/round3_audit.md`
- `docs/metrics_regression_tests.md` 与 `manifests/metrics_regression_tests.json`
- `docs/optimization_diagnosis.md`
- `docs/reference_parity.md`
- `docs/production_rollout_gates.md`
- `docs/claim_evidence_table.md`

## 状态摘要

- A: {"PASS" if a and a.get("all_pass") else "NOT_PASS"}
- B1: {b1s}
- B2: {b2s}
- C Darcy alignment: {calign}
- C grad: {cgrad}
- C ZVB tensor: {czvb}
"""
    (ROUND3 / "README.md").write_text(readme, encoding="utf-8")
    print("wrote round3 reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
