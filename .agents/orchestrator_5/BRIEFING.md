# BRIEFING — 2026-09-13T22:30:00+08:00

## Mission
面向水平井压裂多裂缝识别与进液能力反演的可解释物理-信号智能反演系统研究：构建可微波动方程逆散射层剥离算子与可解释双轨神经算子网络（TG-DIS-DeepONet），完成1,000例基准对标与鲁棒性审计，生成出版级图版并交付完整的学术研究报告 phase3_inverse_scattering_report.md。

## 🔒 My Identity
- Archetype: orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\
- Original parent: Sentinel
- Original parent conversation ID: 353f55fb-3f88-4189-9dda-a2729f28be40

## 🔒 My Workflow
- **Pattern**: Project Pattern (Survey → Decompose & Delegate → Iteration Loop: Explorer → Worker → Reviewer / Challenger / Auditor → Final Deliverable & Report)
- **Scope document**: e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\PROJECT.md
1. **Decompose**:
   - Survey Phase: Dispatch Explorers to map existing PaperC codebase (`src/`, `data/`, `experiments/`, `phase2_ablation_report.md`), MOC models, datasets, baseline models, layer-stripping formulation, and current benchmark metrics.
   - Milestone Decomposition:
     - M1: 波动方程可微逆散射与声学层剥离算子 (Differentiable Layer-Stripping Layer & Scattering Matrix Formulation)
     - M2: 可解释神经算子与双轨物理映射网络 (TG-DIS-DeepONet Architecture, Forward/Backward Gradient, Physical Simplex α, Admittance Spectrum, Delay-bias Attention)
     - M3: 1,000例物理数据集基准对标 (1D-ResNet, 1D-FNO, DeepONet, TG-DeepONet vs TG-DIS-DeepONet) 与 20dB/10dB 噪声/声速鲁棒性审计实验
     - M4: 论文级图版生成与学术技术研究报告交付 (Publication Figures & phase3_inverse_scattering_report.md)
     - Verification & Dual-track: Independent Reviewers, Challengers, and Forensic Auditor verification against all 7 Acceptance Criteria.
2. **Dispatch & Execute**:
   - Iteration loop per milestone with specialist subagents: Explorers → Workers → Reviewers & Challengers & Forensic Auditor.
3. **On failure**:
   - Retry -> Replace -> Skip -> Redistribute -> Redesign.
4. **Succession**:
   - Self-succeed at 16 spawns if necessary.
- **Work items**:
  1. Survey & Codebase Exploration [in-progress]
  2. M1: Differentiable Layer-Stripping Operator [pending]
  3. M2: Explainable TG-DIS-DeepONet Inversion Model [pending]
  4. M3: Benchmark & Robustness Audit (1,000 dataset, SNR 30/20/10dB) [pending]
  5. M4: Publication-grade Figures & Technical Report Deliverable [pending]
  6. Final Independent Review & Audit Verification [pending]
- **Current phase**: 0 (Survey)
- **Current focus**: Survey of codebase, data, and previous phase results via parallel Explorers

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers for technical investigation.
- You MAY use file-editing tools ONLY for metadata/state files (.md) in your .agents/ folder.
- DO NOT CHEAT. All implementations must be genuine.
- Hard audit veto: Forensic Auditor verdict must be CLEAN.

## Current Parent
- Conversation ID: 353f55fb-3f88-4189-9dda-a2729f28be40
- Updated: 2026-09-13T22:30:00+08:00

## Key Decisions Made
- Initialized orchestrator_5 for Phase 3 Inverse Scattering research project.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|---|---|---|---|---|
| explorer_p3_theory | teamwork_preview_spec_miner | Survey theoretical foundations & mathematical specifications for R1/R2 | completed | eaeca433-e7e4-4668-bb39-d87adf7e7e89 |
| explorer_p3_codebase | teamwork_preview_explorer | Survey PaperC codebase architecture, metrics, models, and integration plan | completed | 80015ac9-721d-4a0e-871d-b3758d7b4623 |
| explorer_p3_data | teamwork_preview_explorer | Survey 1,000 cases dataset, noise robustness pipelines, and env benchmarks | completed | 56eef2f2-f099-435c-9dca-5d0876e72526 |
| worker_m1_m2_dis | teamwork_preview_worker | Implement DIS Layer, TGDISDeepONet, F1 metric, and unit tests | completed | 80de2223-8a2c-40c2-a18d-c353191a45a5 |
| worker_m3_benchmark | teamwork_preview_worker | Multi-model benchmark training, ablation, and noise robustness audit | completed | 052234b8-6f1f-41ad-ac1f-5c3c29733b0b |
| worker_m4_report | teamwork_preview_worker | Generate Figures 1-5 (PNG/SVG) and author phase3_inverse_scattering_report.md | completed | 533d13c8-3945-4824-899d-2432d4daa775 |
| reviewer_1_p3 | teamwork_preview_reviewer | Code & Architecture Review and test suite execution | completed | 37ee43aa-8530-4254-823d-b11b176a4247 |
| reviewer_2_p3 | teamwork_preview_reviewer | Benchmark, Figures & Scientific Report Review | completed | bea196a0-bdae-4deb-9bc6-4406c6233ffd |
| challenger_1_p3 | teamwork_preview_challenger | Physical Consistency & Numerical Stress Testing | completed | ad1d85fe-e593-43c4-9a25-0d10898f39fa |
| challenger_2_p3 | teamwork_preview_challenger | Noise & Generalization Robustness Stress Testing | completed | ea25d16f-6758-4f69-9cb6-90c96e41f3b7 |
| auditor_p3 | teamwork_preview_auditor | Forensic Integrity Audit across code, checkpoints & report | completed | 80f1465f-122c-4a22-ba2d-c8cea05c36bb |

## Succession Status
- Succession required: no
- Spawn count: 11 / 16
- Pending subagents: none
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: ef4964b0-6990-42ce-9c4d-bbe3ee64b8cd/task-16
- Safety timer: none

## Artifact Index
- e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\DISPATCH.md — Task assignment
- e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md — Authoritative User Request
- e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\BRIEFING.md — Working memory
- e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_5\progress.md — Progress & liveness
