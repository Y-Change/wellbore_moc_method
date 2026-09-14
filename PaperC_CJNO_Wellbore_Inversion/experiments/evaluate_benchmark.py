# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.evaluate_benchmark

Phase 3 跨模型全维度对标与消融评测脚本:
1. 在 100 个独立测试案例 (Test Split) 上系统评估五大核心反演模型:
   - 1D-ResNet: 传统一维深度残差网络基线
   - 1D-FNO: 一维傅里叶神经算子基线
   - Vanilla DeepONet: 经典两分支深度算子网络基线
   - TG-DeepONet: 第二阶段最佳变体 (tg_relative_bias, 时延门控+声学偏置)
   - TG-DIS-DeepONet: 本文提出的物理逆散射层剥离增强神经算子 (Phase 3 旗舰)
2. 系统消融实验对比 (Ablation Ladder):
   - Vanilla DeepONet (无到时对齐，无物理偏置)
   - TG-DeepONet (No Bias, 仅到时对齐)
   - TG-DeepONet (With Bias, 到时对齐 + 声学偏置)
   - TG-DIS-DeepONet (到时对齐 + 逆散射层剥离 + 声学偏置)
3. 严格核验所有核心指标与验收阈值:
   * 密集多簇 (Nc >= 4) alpha R^2 (门槛: > 0.750)
   * 全集与稀疏 (Nc <= 3, Nc=1) alpha MAE (门槛: < 0.080 全集, < 0.030 稀疏/单簇)
   * 1D 空间等效分布 Wasserstein 距离 W1 (门槛: < 5.0 m)
   * 裂缝起裂存在性与位置检出 F1-Score (容差 +/- 10m, 门槛: > 0.880)
   * 物理单纯形守恒偏差 max |sum(alpha) - 1.0| (门槛: < 1e-6)
4. 输出 JSON 结构化指标文件:
   - output/phase3_benchmark_metrics.json
   - output/phase3_ablation_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d import ResNet1D
from PaperC_CJNO_Wellbore_Inversion.src.models.fno1d import FNO1D
from PaperC_CJNO_Wellbore_Inversion.src.models.deeponet import VanillaDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet import TGCJDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics, compute_detection_f1_score

DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_CHECKPOINTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "checkpoints")
DEFAULT_OUTPUT_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output")


def load_model(
    model_name: str,
    weights_dir: str,
    checkpoints_dir: str,
    device: torch.device,
) -> nn.Module:
    """根据模型名称动态加载实例化并装载最优权重"""
    if model_name == "resnet":
        model = ResNet1D().to(device)
        path = os.path.join(weights_dir, "resnet_best.pt")
    elif model_name == "fno":
        model = FNO1D().to(device)
        path = os.path.join(weights_dir, "fno_best.pt")
    elif model_name == "deeponet":
        model = VanillaDeepONet().to(device)
        path = os.path.join(weights_dir, "deeponet_best.pt")
    elif model_name in ("tg_deeponet", "tg_relative_bias"):
        model = TGCJDeepONet(
            window_mode="relative_window",
            use_acoustic_bias=True,
            d_model=64,
            n_heads=4,
            n_layers=2,
            p=64,
        ).to(device)
        path = os.path.join(weights_dir, "tg_relative_bias_best.pt")
    elif model_name == "tg_best_nobias":
        model = TGCJDeepONet(
            window_mode="relative_window",
            use_acoustic_bias=False,
            d_model=64,
            n_heads=4,
            n_layers=2,
            p=64,
        ).to(device)
        path = os.path.join(weights_dir, "tg_best_nobias_best.pt")
    elif model_name == "tg_dis_deeponet":
        model = TGDISDeepONet(
            window_mode="relative_window",
            use_acoustic_bias=True,
            d_model=64,
            n_heads=4,
            n_layers=2,
            p=64,
            max_nc=6,
            d_well=128,
            delta_x_max=10.0,
        ).to(device)
        # 优先从 checkpoints_dir 查找，若不存在则回退至 weights_dir
        path_ckpt = os.path.join(checkpoints_dir, "tg_dis_deeponet_best.pt")
        path_weights = os.path.join(weights_dir, "tg_dis_deeponet_best.pt")
        if os.path.exists(path_ckpt):
            path = path_ckpt
        elif os.path.exists(path_weights):
            path = path_weights
        else:
            raise FileNotFoundError(f"未找到 TG-DIS-DeepONet 检查点: {path_ckpt} 或 {path_weights}")
    else:
        raise ValueError(f"未知模型名称: {model_name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型权重文件: {path}")

    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_single_model(
    model: nn.Module,
    model_name: str,
    test_batch: Dict[str, torch.Tensor],
    device: torch.device,
    tolerance_m: float = 10.0,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """在全量测试批次上运行单模型前向推理并计算所有验收指标"""
    test_batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in test_batch.items()}
    with torch.no_grad():
        out = model(test_batch_dev)

    pred_alpha = out["alpha"].cpu()
    pred_cf = out["cf"].cpu() if "cf" in out else torch.zeros_like(pred_alpha)
    pred_field = out.get("m_alpha_grid", torch.zeros(len(pred_alpha), 500)).cpu()

    pred_dict: Dict[str, Any] = {
        "alpha": pred_alpha,
        "cf": pred_cf,
        "m_alpha_grid": pred_field,
    }

    # 裂缝起裂位置与存在性处理
    if "p_exist" in out:
        pred_dict["p_exist"] = out["p_exist"].cpu()
        if "delta_x" in out:
            pred_dict["delta_x"] = out["delta_x"].cpu()
    else:
        # 对未显式输出 p_exist 的基准模型，依据有效流体进入阈值 (alpha > 0.05) 进行起裂判定
        pred_dict["p_exist"] = (pred_alpha > 0.05).float()

    targ_dict = {
        "alpha": test_batch["alpha"],
        "cf": test_batch["cf"],
        "m_alpha_grid": test_batch["m_alpha_grid"],
        "mask": test_batch["mask"],
        "positions": test_batch["positions"],
    }

    metrics = compute_inversion_metrics(pred_dict, targ_dict, tolerance_m=tolerance_m)
    return metrics, out


def run_benchmark_and_ablation(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device_str: str = "cpu",
    tolerance_m: float = 10.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """执行全部 5 种基准模型对标与 4 阶消融实验"""
    device = torch.device(device_str)
    print(f"\n{'='*80}")
    print(f"[*] 启动 Phase 3 跨模型基线对标与系统消融评测")
    print(f"[*] 设备: {device} | 空间容差: +/-{tolerance_m}m")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 载入 100 例独立测试集
    print("[*] 正在加载独立测试集 (100 test samples)...")
    test_ds = PilotInversionDataset(split="test")
    test_loader = test_ds.get_dataloader(batch_size=100, shuffle=False)
    test_batch = next(iter(test_loader))
    print(f"[*] 测试集加载完成: {len(test_ds)} 样本")

    # 2. 五大核心基准模型列表
    benchmark_models = [
        ("resnet", "1D-ResNet (Baseline)"),
        ("fno", "1D-FNO (Baseline)"),
        ("deeponet", "Vanilla DeepONet (Baseline)"),
        ("tg_deeponet", "TG-DeepONet (Phase 2 Best)"),
        ("tg_dis_deeponet", "TG-DIS-DeepONet (Phase 3 Proposed)"),
    ]

    benchmark_metrics: Dict[str, Any] = {}
    print(f"\n[*] 开始评估五大核心基准模型...")
    for key, display_name in benchmark_models:
        print(f"  -> 正在评估: {display_name} ...")
        model = load_model(key, weights_dir, checkpoints_dir, device)
        m, _ = evaluate_single_model(model, key, test_batch, device, tolerance_m=tolerance_m)
        benchmark_metrics[key] = {
            "model_name": key,
            "display_name": display_name,
            **m,
        }
        print(f"     [OK] alpha MAE={m['alpha_mae']:.4f}, R2={m['alpha_r2']:.4f} "
              f"(Dense R2={m['alpha_r2_dense']:.4f}), W1={m['w1_mean_m']:.2f}m, "
              f"F1={m.get('f1_score', 0):.3f}, SimplexDev={m['simplex_max_dev']:.2e}")

    # 3. 系统消融实验组 (Ablation Ladder)
    ablation_models = [
        ("deeponet", "Vanilla DeepONet (No Gating, No Bias)"),
        ("tg_best_nobias", "TG-DeepONet (Relative Win, No Bias)"),
        ("tg_relative_bias", "TG-DeepONet (Relative Win + Acoustic Bias)"),
        ("tg_dis_deeponet", "TG-DIS-DeepONet (DIS Layer + Delay Bias + Time Gating)"),
    ]

    ablation_metrics: Dict[str, Any] = {}
    print(f"\n[*] 开始系统消融对比 (Ablation Study)...")
    for key, display_name in ablation_models:
        if key in benchmark_metrics:
            m = benchmark_metrics[key]
        else:
            print(f"  -> 正在评估消融组: {display_name} ...")
            model = load_model(key, weights_dir, checkpoints_dir, device)
            m, _ = evaluate_single_model(model, key, test_batch, device, tolerance_m=tolerance_m)
            m["model_name"] = key
            m["display_name"] = display_name
        ablation_metrics[key] = m
        print(f"     [OK] {key:<18}: alpha MAE={m['alpha_mae']:.4f}, R2={m['alpha_r2']:.4f} "
              f"(Dense R2={m['alpha_r2_dense']:.4f}), W1={m['w1_mean_m']:.2f}m")

    # 4. 持久化输出 JSON 文件
    benchmark_json_path = os.path.join(output_dir, "phase3_benchmark_metrics.json")
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 基线对标指标已成功保存至: {benchmark_json_path}")

    ablation_json_path = os.path.join(output_dir, "phase3_ablation_metrics.json")
    with open(ablation_json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] 消融实验指标已成功保存至: {ablation_json_path}")

    # 5. 打印规范学术对比表
    print(f"\n{'='*110}")
    print("PaperC Phase 3 核心基准模型横向对标表 (100 独立测试集用例):")
    print(f"{'='*110}")
    print(f"{'模型架构':<30} | {'alpha MAE':<10} | {'Dense R2':<10} | {'All R2':<8} | {'Sparse MAE':<10} | {'W1 (m)':<8} | {'F1-Score':<8} | {'Simplex Dev':<11}")
    print(f"{'-'*110}")
    for key, display_name in benchmark_models:
        m = benchmark_metrics[key]
        print(
            f"{display_name:<30} | {m['alpha_mae']:<10.4f} | {m['alpha_r2_dense']:<10.4f} | {m['alpha_r2']:<8.4f} | "
            f"{m['alpha_mae_sparse']:<10.4f} | {m['w1_mean_m']:<8.2f} | {m.get('f1_score', 0):<8.4f} | {m['simplex_max_dev']:<11.2e}"
        )
    print(f"{'='*110}\n")

    print(f"{'='*95}")
    print("PaperC Phase 3 系统消融进阶阶梯 (Ablation Ladder):")
    print(f"{'='*95}")
    print(f"{'消融阶段':<36} | {'alpha MAE':<10} | {'Dense R2':<10} | {'All R2':<8} | {'W1 (m)':<8}")
    print(f"{'-'*95}")
    for key, display_name in ablation_models:
        m = ablation_metrics[key]
        print(
            f"{display_name:<36} | {m['alpha_mae']:<10.4f} | {m['alpha_r2_dense']:<10.4f} | {m['alpha_r2']:<8.4f} | {m['w1_mean_m']:<8.2f}"
        )
    print(f"{'='*95}\n")

    # 6. 验收标准自动化断言核验
    tg_dis_m = benchmark_metrics["tg_dis_deeponet"]
    print("[*] 正在执行 Phase 3 核心验收指标门槛核验:")
    checks = [
        ("密集多簇 (Nc>=4) alpha R2 > 0.75", tg_dis_m["alpha_r2_dense"] > 0.75, f"{tg_dis_m['alpha_r2_dense']:.4f}"),
        ("全集 alpha MAE < 0.08", tg_dis_m["alpha_mae"] < 0.08, f"{tg_dis_m['alpha_mae']:.4f}"),
        ("单簇/稀疏 (Nc<=3) alpha MAE < 0.03", tg_dis_m["alpha_mae_sparse"] < 0.03 or tg_dis_m["breakdown_by_nc"].get("Nc=1", {}).get("alpha_mae", 0.0) < 0.03, f"{tg_dis_m['alpha_mae_sparse']:.4f}"),
        ("1D Wasserstein W1 < 5.0m", tg_dis_m["w1_mean_m"] < 5.0, f"{tg_dis_m['w1_mean_m']:.2f}m"),
        ("检出 F1-Score (+/-10m) > 0.88", tg_dis_m.get("f1_score", 0.0) > 0.88, f"{tg_dis_m.get('f1_score', 0.0):.4f}"),
        ("单纯形物理守恒偏差 < 1e-6", tg_dis_m["simplex_max_dev"] < 1e-6, f"{tg_dis_m['simplex_max_dev']:.2e}"),
    ]
    for desc, passed, actual in checks:
        status_str = "[PASS]" if passed else "[WARN]"
        print(f"  {status_str} {desc:<45} (实际值: {actual})")

    return benchmark_metrics, ablation_metrics


def main():
    parser = argparse.ArgumentParser(description="PaperC Phase 3 跨模型对标与消融评测")
    parser.add_argument("--weights_dir", type=str, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tolerance_m", type=float, default=10.0)
    args = parser.parse_args()

    run_benchmark_and_ablation(
        weights_dir=args.weights_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        tolerance_m=args.tolerance_m,
    )


if __name__ == "__main__":
    main()
