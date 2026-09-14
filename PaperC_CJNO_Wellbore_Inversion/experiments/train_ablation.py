# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.train_ablation

TG-CJ-DeepONet 第二阶段全套消融实验自动化训练管线:
对照实验组规划:
1. Exp 1a: tg_relative_bias   (relative_window + 有声学偏置)
2. Exp 1b: tg_gaussian_bias   (gaussian_gating + 有声学偏置)
3. Exp 1c: tg_ceps_patch_bias (ceps_patch + 有声学偏置)
4. Exp 2:  tg_best_nobias     (最佳模式 relative_window + 无声学偏置消融)

全量保存最优模型权重至 output/weights/ 并输出训练收敛历史 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet import TGCJDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.losses import CompositeInversionLoss
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")

# 4 大消融配置定义: (window_mode, use_acoustic_bias, description)
ABLATION_CONFIGS = {
    "tg_relative_bias": {
        "window_mode": "relative_window",
        "use_acoustic_bias": True,
        "desc": "TG-CJ-DeepONet (可微相对到时窗 + 声学时延偏置)",
    },
    "tg_gaussian_bias": {
        "window_mode": "gaussian_gating",
        "use_acoustic_bias": True,
        "desc": "TG-CJ-DeepONet (可微高斯软窗 + 声学时延偏置)",
    },
    "tg_ceps_patch_bias": {
        "window_mode": "ceps_patch",
        "use_acoustic_bias": True,
        "desc": "TG-CJ-DeepONet (2D 倒谱时空切片图块 + 声学时延偏置)",
    },
    "tg_best_nobias": {
        "window_mode": "relative_window",
        "use_acoustic_bias": False,
        "desc": "TG-CJ-DeepONet (最佳模式 relative_window + 无声学偏置消融组)",
    },
}


def train_single_ablation(
    exp_name: str,
    config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    output_dir: str = DEFAULT_WEIGHTS_DIR,
) -> Dict[str, Any]:
    """训练单消融模型并持久化最优权重与评估指标"""
    print(f"\n{'='*70}")
    print(f"[*] 启动消融实验: {exp_name}")
    print(f"[*] 模式配置: window_mode={config['window_mode']}, use_acoustic_bias={config['use_acoustic_bias']}")
    print(f"[*] 描述: {config['desc']}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)
    best_weight_path = os.path.join(output_dir, f"{exp_name}_best.pt")
    history_path = os.path.join(output_dir, f"{exp_name}_history.json")

    model = TGCJDeepONet(
        window_mode=config["window_mode"],
        use_acoustic_bias=config["use_acoustic_bias"],
        in_channels=2,
        cond_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        p=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 模型可训练参数量: {total_params:,}")

    # 复合物理损失: 引入双轨协同一致性监督 lambda_cons=0.5
    criterion = CompositeInversionLoss(
        lambda_alpha=1.0,
        lambda_c=1.0,
        lambda_w=0.01,
        lambda_cons=0.5,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: List[Dict[str, Any]] = []
    best_val_score = float("inf")
    best_metrics: Dict[str, Any] = {}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. 训练轮次
        model.train()
        train_losses = []
        train_l_alpha = []
        train_l_c = []
        train_l_w = []
        train_l_cons = []

        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch_dev)
            loss_dict = criterion(out, batch_dev)
            loss = loss_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_l_alpha.append(loss_dict["loss_alpha"].item())
            train_l_c.append(loss_dict["loss_c"].item())
            train_l_w.append(loss_dict["loss_w1"].item())
            train_l_cons.append(loss_dict["loss_cons"].item())

        scheduler.step()

        # 2. 验证评估 (每 2 轮或首末轮评估)
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            val_losses = []
            val_preds_alpha = []
            val_preds_cf = []
            val_preds_field = []
            val_targets_alpha = []
            val_targets_cf = []
            val_targets_field = []
            val_masks = []
            val_positions = []

            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vbatch.items()}
                    vout = model(vbatch_dev)
                    vloss_dict = criterion(vout, vbatch_dev)
                    val_losses.append(vloss_dict["loss"].item())

                    val_preds_alpha.append(vout["alpha"].cpu())
                    val_preds_cf.append(vout["cf"].cpu())
                    val_preds_field.append(vout["m_alpha_grid"].cpu())
                    val_targets_alpha.append(vbatch["alpha"])
                    val_targets_cf.append(vbatch["cf"])
                    val_targets_field.append(vbatch["m_alpha_grid"])
                    val_masks.append(vbatch["mask"])
                    val_positions.append(vbatch["positions"])

            val_pred_dict = {
                "alpha": torch.cat(val_preds_alpha, dim=0),
                "cf": torch.cat(val_preds_cf, dim=0),
                "m_alpha_grid": torch.cat(val_preds_field, dim=0),
            }
            val_targ_dict = {
                "alpha": torch.cat(val_targets_alpha, dim=0),
                "cf": torch.cat(val_targets_cf, dim=0),
                "m_alpha_grid": torch.cat(val_targets_field, dim=0),
                "mask": torch.cat(val_masks, dim=0),
                "positions": torch.cat(val_positions, dim=0),
            }
            vmetrics = compute_inversion_metrics(val_pred_dict, val_targ_dict)

            mean_train_loss = float(np.mean(train_losses))
            mean_val_loss = float(np.mean(val_losses))

            # 综合评分: alpha_mae + cf_log10_mae
            composite_score = vmetrics["alpha_mae"] + vmetrics["cf_log10_mae"]

            record = {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "alpha_mae": vmetrics["alpha_mae"],
                "alpha_r2": vmetrics["alpha_r2"],
                "cf_mre_pct": vmetrics["cf_mre_pct"],
                "cf_log10_mae": vmetrics["cf_log10_mae"],
                "w1_mean_m": vmetrics["w1_mean_m"],
                "simplex_max_dev": vmetrics["simplex_max_dev"],
            }
            history.append(record)

            is_best = composite_score < best_val_score
            if is_best:
                best_val_score = composite_score
                best_metrics = vmetrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "metrics": vmetrics,
                        "exp_name": exp_name,
                        "config": config,
                    },
                    best_weight_path,
                )

            if epoch % 10 == 0 or epoch == epochs:
                tag = " [*BEST*]" if is_best else ""
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
                    f"alpha MAE: {vmetrics['alpha_mae']:.4f} (R2={vmetrics['alpha_r2']:.3f}) | "
                    f"Cf MRE: {vmetrics['cf_mre_pct']:.1f}% | W1: {vmetrics['w1_mean_m']:.2f}m{tag}"
                )

    total_time = time.time() - start_time
    print(f"[OK] {exp_name} 训练完成, 耗时: {total_time:.1f}s, 最优模型已保存至: {best_weight_path}")

    # 保存训练历史 JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_name": exp_name,
                "config": config,
                "total_params": total_params,
                "train_time_sec": total_time,
                "best_metrics": best_metrics,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "exp_name": exp_name,
        "best_weight_path": best_weight_path,
        "history_path": history_path,
        "train_time_sec": total_time,
        "best_metrics": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="TG-CJ-DeepONet 第二阶段消融实验训练管线")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(ABLATION_CONFIGS.keys()),
        choices=list(ABLATION_CONFIGS.keys()),
        help="指定训练的消融实验组 (默认训练全部 4 组)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数 (默认 60)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小 (默认 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率 (默认 1e-3)")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_WEIGHTS_DIR, help="权重保存目录")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[*] 运行设备: {device}")

    # 1. 载入数据集 (800 train / 100 val / 100 test)
    print("[*] 正在加载物理数据集...")
    train_dataset = PilotInversionDataset(split="train")
    val_dataset = PilotInversionDataset(split="val")

    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    print(f"[*] 训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 2. 依次训练各消融组
    summary_results = {}
    for exp_name in args.experiments:
        cfg = ABLATION_CONFIGS[exp_name]
        res = train_single_ablation(
            exp_name=exp_name,
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            output_dir=args.output_dir,
        )
        summary_results[exp_name] = res

    # 3. 输出汇总对照
    print(f"\n{'='*80}")
    print("第二阶段 (Phase 2) 消融实验训练验证集指标初步汇总:")
    print(f"{'='*80}")
    print(f"{'实验组名称':<22} | {'alpha MAE':<10} | {'alpha R2':<9} | {'Cf MRE%':<9} | {'W1 (m)':<8} | {'耗时(s)':<8}")
    print(f"{'-'*80}")
    for exp_name, res in summary_results.items():
        m = res["best_metrics"]
        print(
            f"{exp_name:<22} | {m.get('alpha_mae', 0.0):<10.4f} | {m.get('alpha_r2', 0.0):<9.4f} | "
            f"{m.get('cf_mre_pct', 0.0):<9.1f} | {m.get('w1_mean_m', 0.0):<8.2f} | {res['train_time_sec']:<8.1f}"
        )
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
