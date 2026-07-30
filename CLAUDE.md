<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 82 entries.
Manifest: `.aris/installed-skills.txt`
ARIS repo root: `E:\water_hammer_research\wellbore_moc_method\ARIS\Auto-claude-code-research-in-sleep`
Project skill path: `.claude/skills/<skill-name>`
For ARIS workflows, prefer the project-local skills under `.claude/skills/`.
Do not edit or delete junctioned skills in place; update upstream or rerun:
`powershell -NoProfile -ExecutionPolicy Bypass -File "E:\water_hammer_research\wellbore_moc_method\ARIS\Auto-claude-code-research-in-sleep\tools\install_aris.ps1" "E:\water_hammer_research\wellbore_moc_method" -Platform claude -Reconcile`
<!-- ARIS:END -->


## Research Obsidian Synchronization

The project research vault is `research_ob/`. Whenever an experiment-related task produces, changes, analyzes, or verifies results, update the vault before considering the task complete.

### Vault layout (simplified)

- `首页.md` — single entry (topics + todos)
- `主题/` — one page per research line (science + evidence + paper claims)
- `实验/` — reproducible experiment notes
- `日记/` — process notes
- `文献/` — literature notes
- `概念/` — concept / formula notes
- `模板/`、`附件/`、`归档/`

### Required synchronization

1. Record reproducible experiment details under `research_ob/实验/` using `research_ob/模板/实验记录模板.md`. Include experiment ID, purpose, hypothesis, code entry point, command, key parameters, input/output paths, quantitative results, anomalies, conclusion, and next steps. Prefer the matching category subdirectory (`01-MOC正演与验证/` … `06-神经算子与DCCDM/`).
2. Update the corresponding **topic page** under `research_ob/主题/` (science conclusions, evidence links, and paper claim–evidence table in the same note):
   - Cepstrum and resolvability → `T01-倒谱分辨率极限.md`
   - Decay and topological scaling → `T02-多裂缝拓扑衰减.md`
   - Primary-fracture interference → `T03-首缝干涉机制.md`
   - Brunone friction and dispersion → `T04-Brunone频散效应.md`
   - Sparse deconvolution → `T05-稀疏反卷积.md`
   - Dispersion compensation → `T06-频散补偿后向传播.md`
   - Neural operators and DCCDM → `T07-DCCDM代理模型.md`
3. Do **not** maintain separate paper-project homepages; paper status lives in the topic page’s「论文」section. Long drafts stay in `research_ob/主题/drafts/`.
4. Link vault notes to repository-relative code, configuration, data, and `output/` paths instead of copying reproducible outputs into the vault. Store only irreplaceable screenshots, PDFs, or manually produced material under `research_ob/附件/`.
5. Record failed runs, negative results, invalid metrics, and rejected hypotheses under `research_ob/实验/99-失败与负结果/`; do not omit them.

### Integrity rules

- Synchronize only observed results; never invent metrics, conclusions, citations, or completed runs.
- Clearly distinguish observations, interpretations, and hypotheses.
- If an experiment is incomplete or verification fails, record that status and the failure details rather than presenting a final conclusion.
- Preserve existing notes and manually authored content. Extend or revise the relevant sections without replacing unrelated material.
- Keep `docs/` unchanged unless the task explicitly requests a documentation update; `research_ob/` is the primary location for evolving research knowledge.
- In the final task response, list the experiment note and topic pages updated. If no vault update was appropriate, state the reason explicitly.
