# BRIEFING — 2026-09-11T19:46:02+08:00

## Mission
拆分原技术报告并新建《MOC_V2 现场工况正演与参数敏感性分析报告》（MOC_V2_Simulation_Sensitivity_Report.md），基于生产级 moc_simulate.v2 求解器执行 7 大专题敏感性仿真实验（38+用例全部收敛），采用 Nature 级图版规范绘制 Figure 1-7（含 Rainbow 色阶 2D 连续倒谱图），输出完整专业技术分析报告，确保全库 pytest 100% 通过。

## 🔒 My Identity
- Archetype: teamwork_preview_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_3
- Original parent: parent
- Original parent conversation ID: 73334a36-8ab4-493c-8072-ec15a7d15bc0

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_3\PROJECT.md
1. **Decompose**: Survey codebase with 3 Explorers, create PROJECT.md with Feature Inventory, Milestones, and Interface Contracts.
2. **Dispatch & Execute** (pick ONE):
   - **Direct (iteration loop)**: For each milestone: Explorer (3) -> Worker (1) -> Reviewer (2) + Challenger (2) + Auditor (1) -> Gate.
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (last resort)
4. **Succession**: At 16 spawns, write soft handoff, spawn successor.
- **Work items**:
  1. Survey & Codebase Mapping [in-progress]
  2. M1 (R1): Physics Upgrade Report Streamlining & build_report.py validation [pending]
  3. M2 (R2): 7 Sensitivity Studies Simulation Pipeline [pending]
  4. M3 (R3): Nature-grade Figure Generation (7 figures, 14 files, Rainbow 2D cepstrum) [pending]
  5. M4 (R4): MOC_V2_Simulation_Sensitivity_Report.md Technical Report [pending]
  6. M5 (R5): Full Regression Testing (pytest 99+ tests PASS) [pending]
- **Current phase**: 0 (Survey)
- **Current focus**: Survey phase with 3 Explorers

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers for technical investigation.
- File-editing tools ONLY for metadata/state files (.md) in .agents/ folder.
- Always communicate results back to parent (73334a36-8ab4-493c-8072-ec15a7d15bc0) via send_message.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.
- Binary veto on audit failure (Forensic Auditor).
- 2D cepstrogram must strictly use Rainbow colormap and mark true fracture positions without text detection criteria.

## Current Parent
- Conversation ID: 73334a36-8ab4-493c-8072-ec15a7d15bc0
- Updated: 2026-09-11T19:46:02+08:00

## Key Decisions Made
- Initialized Project Orchestrator 3 for MOC_V2 Report Streamlining and Simulation Sensitivity Study.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_survey_1 | teamwork_preview_explorer | Survey MOC Solver & Simulation Pipeline | completed | 15a91bd9-3c97-4fde-9d89-50276e85a90c |
| explorer_survey_2 | teamwork_preview_explorer | Survey Reports & Build Pipeline | completed | 4e85c354-d883-43a6-9e73-6666611031b3 |
| explorer_survey_3 | teamwork_preview_explorer | Survey Visualization & Test Regression | completed | 402e66d9-f928-4cdc-a2df-36ace667dee9 |
| worker_m1 | teamwork_preview_worker | M1: Theory Report Streamlining & Build Validation | in-progress | afa8512b-cde3-43e4-947c-399a8dd6535f |
| worker_m2 | teamwork_preview_worker | M2 & M3: Sensitivity Simulation Pipeline & Figures | in-progress | d981ac91-2bbf-4e7e-89f8-b55f9b40469a |

## Succession Status
- Succession required: no
- Spawn count: 5 / 16
- Pending subagents: afa8512b-cde3-43e4-947c-399a8dd6535f, d981ac91-2bbf-4e7e-89f8-b55f9b40469a
- Predecessor: orchestrator_2
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: 378ebea8-5954-4263-ba6d-94834c1aab6c/task-20
- Safety timer: 378ebea8-5954-4263-ba6d-94834c1aab6c/task-112
- On succession: kill all timers before spawning successor
- On context truncation: run manage_task(Action="list") — re-create if missing

## Artifact Index
- e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_3\DISPATCH.md — Dispatch instructions and user request record
- e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md — Original request
