import json
from pathlib import Path

out_dir = Path("PaperC_CJNO_Wellbore_Inversion/output")

for fname in ["phase3_benchmark_metrics.json", "phase3_ablation_metrics.json", "phase3_noise_robustness_metrics.json"]:
    fpath = out_dir / fname
    print("=" * 80)
    print(f"File: {fname} (exists: {fpath.exists()})")
    if fpath.exists():
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if fname == "phase3_benchmark_metrics.json":
            for k, v in data.items():
                print(f"  Model {k:<18}: alpha_mae={v['alpha_mae']:.4f}, alpha_r2={v['alpha_r2']:.4f}, dense_r2={v['alpha_r2_dense']:.4f}, sparse_mae={v['alpha_mae_sparse']:.4f}, w1={v['w1_mean_m']:.2f}m, f1={v.get('f1_score', 0):.4f}, simplex={v['simplex_max_dev']:.2e}")
        elif fname == "phase3_ablation_metrics.json":
            for k, v in data.items():
                print(f"  Ablation {k:<18}: alpha_mae={v['alpha_mae']:.4f}, dense_r2={v['alpha_r2_dense']:.4f}, all_r2={v['alpha_r2']:.4f}, w1={v['w1_mean_m']:.2f}m")
        elif fname == "phase3_noise_robustness_metrics.json":
            deg = data.get("degradation_analysis", {})
            print("  Degradation analysis:", deg)
            for m in ["tg_deeponet", "tg_dis_deeponet"]:
                print(f"  Model {m}:")
                for sk, sv in data[m].items():
                    print(f"    Scenario {sk:<18}: alpha_mae={sv['alpha_mae']:.4f}, r2={sv['alpha_r2']:.4f}, w1={sv['w1_mean_m']:.2f}m")
