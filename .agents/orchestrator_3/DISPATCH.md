# Dispatch Instructions for Orchestrator 3

## Role
Project Orchestrator (`teamwork_preview_orchestrator`)

## Working Directory
`e:\water_hammer_research\wellbore_moc_method\.agents\orchestrator_3`

## Authoritative User Request
`e:\water_hammer_research\wellbore_moc_method\.agents\ORIGINAL_REQUEST.md`

## Mission
拆分原技术报告并新建《MOC_V2 现场工况正演与参数敏感性分析报告》（`MOC_V2_Simulation_Sensitivity_Report.md`）。以实际油田水平井现场尺度为基准（Base Case: 3 簇裂缝，起点 4500m，间距 10m），利用生产级 `moc_simulate.v2` 求解器执行 7 大专题敏感性仿真实验（裂缝数量 1~8、间距 5~80m、顺应性 Cf、滤失 kleak、射孔流阻 Kp、关泵斜坡 tc、非均匀进液能力组合），采用统一的 Nature 级图版规范绘制时域波形、1D 倒谱及 Rainbow 色阶 2D 连续倒谱图（标注真实裂缝深度，不作检出率文本判据），输出完整的专业技术分析报告。

## Key Deliverables & Acceptance Criteria
1. R1: 修改 `docs/moc_v2_technical_report/MOC_V2_Physics_Upgrade_Report.md`（删除第6、7章，剥离第3、4、5章移交新报告，保留并完善第1、2章；调整 `docs/moc_v2_technical_report/build_report.py` 确保断言校验全部PASS）。
2. R2: 编写 `docs/moc_v2_technical_report/run_sensitivity_study.py`，基于 `moc_simulate.v2` 生产级内核执行全流程正演仿真（Base Case及专题1~7共38+用例全部收敛，无NaN/Inf）。
3. R3: 在 `docs/moc_v2_technical_report/sensitivity_figures/` 下输出 7 大专题独立图版（`fig1_fracture_count_sensitivity` 至 `fig7_intake_capacity_combinations`，每个提供 300 DPI PNG 与矢量 SVG，共14份文件；2D倒谱云图严格使用 Rainbow 色阶；仅标定真实裂缝位置，不添加文本检出率判据）。
4. R4: 撰写完整的学术技术报告 `docs/moc_v2_technical_report/MOC_V2_Simulation_Sensitivity_Report.md`（深度融入7组仿真数据与图版、物理机理深度剖析、公式规范配齐SI单位）。
5. 全库回归：运行 `pytest` 确保全库99项以上测试 100% 全部通过。

## 2026-09-11T11:46:02Z
拆分原技术报告并新建《MOC_V2 现场工况正演与参数敏感性分析报告》（`MOC_V2_Simulation_Sensitivity_Report.md`）。以实际油田水平井现场尺度为基准（Base Case: 3 簇裂缝，起点 4500m，间距 10m），利用生产级 `moc_simulate.v2` 求解器执行 7 大专题敏感性仿真实验（裂缝数量 1~8、间距 5~80m、顺应性 Cf、滤失 kleak、射孔流阻 Kp、关泵斜坡 tc、非均匀进液能力组合），采用统一的 Nature 级图版规范绘制时域波形、1D 倒谱及 Rainbow 色阶 2D 连续倒谱图（标注真实裂缝深度，不作检出率文本判据），输出完整的专业技术分析报告。
