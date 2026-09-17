# Figure audit for the physics-informed manuscript

## Main-figure inventory (2026-09-06)

Figures 1--10 have been traced to their plotting outputs and visually inspected at original PNG resolution. Every cited main figure is available as a 400-dpi PNG, an editable SVG and a PDF; no cited main figure has a clipped axis label, title or legend. Figure 8 was redrawn as the single six-panel file `稿件/图表数据/Figure_8_Interaction_Phase_Maps.{png,svg,pdf}` because the former manuscript mapping incorrectly treated two separate three-panel figures as panels 8a and 8b. The replacement reads the full 70-row spacing--multiplicity grid and 60-row depth--spacing grid, labels panels a--f, and removes the clipped physical-span axis label present in the former 5.4 export. Its panel-alignment, PDF text-size and collision audits pass.

The current Figure 3 and Figure 4 PNGs were visually checked after regeneration; both are 400 dpi and their render-time panel-alignment gates pass with zero failures or warnings. Older standalone collision reports pre-date the latest exports and are not treated as pass evidence.

The manuscript figures use the following audited outputs:

- `Figure_3_3_Physics_Informed_Dealiasing_Workflow.{png,svg,pdf}`: operator proposal with raw profile, nominal design prior, non-perforation path candidates, finite transmission compensation, focusing kernel, and a sensitivity proxy. It does not display target-equalized peaks or claim blind recovery.
- `Figure_3_4_MultiCase_Physics_Correction_Matrix.{png,svg,pdf}`: four-case forward evidence matrix. It reports raw MOC/cepstrum evidence and nominal/path locations; it does not display an oracle-corrected profile.

The standalone files `Figure_3_4_a_*` through `Figure_3_4_d_*` are legacy exploratory exports. Their labels and source scripts were produced by the former target-equalization/oracle-mask implementation (including “100% Authentic Fracture Band”) and are excluded from the manuscript evidence chain. They should not be cited as validation figures.

The numerical benchmark remains the authenticated `decay_table.csv` and the continuous Case 1 profile described in the Chinese and English manuscripts. Any future blind-validation figure must be generated from held-out activation states with the physical gains and mask thresholds fixed before evaluation.
