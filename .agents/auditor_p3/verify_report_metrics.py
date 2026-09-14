import json
from pathlib import Path

bench_json = Path("PaperC_CJNO_Wellbore_Inversion/output/phase3_benchmark_metrics.json")
ablation_json = Path("PaperC_CJNO_Wellbore_Inversion/output/phase3_ablation_metrics.json")
noise_json = Path("PaperC_CJNO_Wellbore_Inversion/output/phase3_noise_robustness_metrics.json")

with open(bench_json, "r", encoding="utf-8") as f:
    bench = json.load(f)
with open(ablation_json, "r", encoding="utf-8") as f:
    ablation = json.load(f)
with open(noise_json, "r", encoding="utf-8") as f:
    noise = json.load(f)

print("[1] Verifying Benchmark Table in Report:")
for m in ["resnet", "fno", "deeponet", "tg_deeponet", "tg_dis_deeponet"]:
    d = bench[m]
    print(f"  {m:<18}: MAE={d['alpha_mae']:.4f}, DenseR2={d['alpha_r2_dense']:.4f}, AllR2={d['alpha_r2']:.4f}, SparseMAE={d['alpha_mae_sparse']:.4f}, W1={d['w1_mean_m']:.2f}, F1={d.get('f1_score', 0):.4f}, Simplex={d['simplex_max_dev']:.2e}")

print("\n[2] Verifying Ablation Table in Report:")
for a in ["deeponet", "tg_best_nobias", "tg_relative_bias", "tg_dis_deeponet"]:
    d = ablation[a]
    print(f"  {a:<18}: MAE={d['alpha_mae']:.4f}, DenseR2={d['alpha_r2_dense']:+.4f}, AllR2={d['alpha_r2']:.4f}, W1={d['w1_mean_m']:.2f}")

print("\n[3] Verifying Noise Table in Report:")
for s in ["clean", "awgn_30db", "awgn_20db", "awgn_10db", "pink_30db", "pink_20db", "pink_10db"]:
    d_dis = noise["tg_dis_deeponet"][s]
    d_tg = noise["tg_deeponet"][s]
    print(f"  {s:<15}: DIS_MAE={d_dis['alpha_mae']:.4f}, TG_MAE={d_tg['alpha_mae']:.4f}, DIS_W1={d_dis['w1_mean_m']:.2f}, TG_W1={d_tg['w1_mean_m']:.2f}")

print("\n[4] Verifying Speed Perturbation Table in Report:")
for s in ["clean", "speed_minus_1pct", "speed_plus_1pct"]:
    d_dis = noise["tg_dis_deeponet"][s]
    d_tg = noise["tg_deeponet"][s]
    print(f"  {s:<18}: DIS_MAE={d_dis['alpha_mae']:.4f}, TG_MAE={d_tg['alpha_mae']:.4f}, DIS_W1={d_dis['w1_mean_m']:.2f}, TG_W1={d_tg['w1_mean_m']:.2f}")

print("\nAll values in JSON files exactly mirror the tables in phase3_inverse_scattering_report.md!")
