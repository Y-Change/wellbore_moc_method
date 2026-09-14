import os, time

paths = [
    r'PaperC_CJNO_Wellbore_Inversion/src/modules/layer_stripping.py',
    r'PaperC_CJNO_Wellbore_Inversion/src/models/tg_dis_deeponet.py',
    r'PaperC_CJNO_Wellbore_Inversion/src/metrics.py',
    r'PaperC_CJNO_Wellbore_Inversion/src/losses.py',
    r'PaperC_CJNO_Wellbore_Inversion/tests/test_dis_layer.py',
    r'PaperC_CJNO_Wellbore_Inversion/tests/test_tg_dis_model.py',
    r'PaperC_CJNO_Wellbore_Inversion/experiments/train_dis.py',
    r'PaperC_CJNO_Wellbore_Inversion/checkpoints/tg_dis_deeponet_best.pt',
    r'PaperC_CJNO_Wellbore_Inversion/output/phase3_benchmark_metrics.json',
    r'PaperC_CJNO_Wellbore_Inversion/output/phase3_ablation_metrics.json',
    r'PaperC_CJNO_Wellbore_Inversion/output/phase3_noise_robustness_metrics.json',
    r'PaperC_CJNO_Wellbore_Inversion/experiments/plot_phase3_figures.py',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig1_layer_stripping_mechanism.png',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig1_layer_stripping_mechanism.svg',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig2_tg_dis_architecture.png',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig2_tg_dis_architecture.svg',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig3_benchmark_and_ablation.png',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig3_benchmark_and_ablation.svg',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig4_noise_and_speed_robustness.png',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig4_noise_and_speed_robustness.svg',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig5_typical_cases_inversion.png',
    r'PaperC_CJNO_Wellbore_Inversion/output/figures/fig5_typical_cases_inversion.svg',
    r'PaperC_CJNO_Wellbore_Inversion/phase3_inverse_scattering_report.md',
    r'.agents/orchestrator_5/handoff.md'
]

for p in paths:
    if os.path.exists(p):
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p)))
        size = os.path.getsize(p)
        print(f'{mtime} | {size:>8} bytes | {p}')
    else:
        print(f'MISSING | {p}')
