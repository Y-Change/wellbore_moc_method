## 2026-09-11T11:39:45Z

Mission: Survey MOC_V2 simulation production kernel (`moc_simulate.v2`), multi-cluster fracture modeling, and forward simulation pipeline requirements for R2.
Key Tasks:
1. Inspect the codebase for `moc_simulate` and specifically `moc_simulate.v2` (modules, classes, methods, configuration dataclasses).
   - How are fractures configured? (Positions, compliance Cf, leakoff k_leak, perforation impedance Kp, intake flow rates or ratios)?
   - How is the wellbore geometry and boundary conditions set up (L=5000m, diameter, wave speed a, friction model, shut-in ramp tc)?
   - How are initial steady-state / self-consistent flow fields established?
   - How does Newton iteration solve the boundary and internal fracture nodes?
2. Analyze the requirements for R2:
   - Base Case: L=5000m, 3 clusters at [4500, 4510, 4520]m (start 4500m, spacing 10m, toe dead zone 480m), Cf=0.01 m^2, k_leak=1.0e-4 m^2.5/s, Kp=5.43e5 s^2/m^5, tc=1.0s.
   - Topic 1: Nc in [1, 2, 3, 4, 5, 6, 7, 8], start 4500m, spacing 10m.
   - Topic 2: 3 clusters, start 4500m, spacing d in [5, 10, 15, 20, 25, 30, 50, 80]m.
   - Topic 3: Cf in [0.002, 0.005, 0.010, 0.020, 0.030] m^2.
   - Topic 4: k_leak in [0.2, 0.6, 1.0, 3.0, 10.0]e-4 m^2.5/s (including Type V fault strong leakoff).
   - Topic 5: Kp in [1.5, 3.5, 5.43, 10.0, 25.0]e5 s^2/m^5 (perforation holes 16, 8, 6, 4, 2).
   - Topic 6: tc in [0.0, 0.5, 1.0, 1.5, 2.0]s.
   - Topic 7: 5 intake combinations: [med,med,med] uniform; [high,med,med] heel breakthrough; [high,med,high] saddle; [med,med,high] toe reverse dominance; [dead,med,high] heel sandout.
3. Check simulation run time, grid resolution (dx, dt), simulation duration (100s?), data storage/caching architecture for `run_sensitivity_study.py`.
4. Deliver a comprehensive survey report in `e:\water_hammer_research\wellbore_moc_method\.agents\teamwork_preview_explorer_survey_2\handoff.md` and send a message when done.
