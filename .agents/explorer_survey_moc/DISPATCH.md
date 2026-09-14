# Explorer Survey MOC Dispatch

You are teamwork_preview_explorer for surveying the MOC simulation engine and friction models.
Working directory: e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc
Project root: e:\water_hammer_research\wellbore_moc_method
Original request: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md

## 2026-09-09T06:46:29Z
You are teamwork_preview_explorer focusing on MOC Simulation Engine & Friction Formulations.
Your working directory is: e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc
Project root is: e:\water_hammer_research\wellbore_moc_method
Path to ORIGINAL_REQUEST.md: e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md (MANDATORY: Read it first!)

Tasks:
1. Read e:\water_hammer_research\wellbore_moc_method\ORIGINAL_REQUEST.md.
2. Investigate the codebase in moc_simulate/ (specifically wellbore_moc.py and any related MOC solver files).
3. Detail how fracture parameters are modeled and configured: location x_f, fracture compliance C_H, leakoff coefficient k_leak, perforation resistance R_p, branch flow distribution w_i, fracture spacing delta x.
4. Detail the friction formulation implementation: Steady Darcy-Weisbach vs Brunone unsteady friction formulation. How is Brunone decay coefficient k_u or friction term calculated, switched, and parameterized in wellbore_moc.py or other modules?
5. Check numerical stability constraints (Courant-Friedrichs-Lewy condition CFL <= 1, spatial discretization delta x, time step delta t, wave speed a, fluid density, viscosity, pipe geometry).
6. Identify how simulations are invoked, inputs configured, outputs returned, and how numerical convergence and mass conservation are guaranteed.
7. Write a comprehensive report survey_moc_report.md and handoff.md in your working directory e:\water_hammer_research\wellbore_moc_method\.agents\explorer_survey_moc\. Send a message back to parent when done.
