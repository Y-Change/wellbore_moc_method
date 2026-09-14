# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.train_embedding_ablation

网络端到端连续时空 Embedding 架构升级与消融对标训练管线:
1. 保持 TG-DIS-DeepONet 旗舰网络后端架构与损失函数 100% 严格一致;
2. 在相同的 50 Epochs 轮次、相同随机种子 (seed=42) 与相同超参数下，
   横向并行/顺序训练 4 个模型变体:
   - 'baseline_resample': 离线 4096 点机械降采样基准 (TG-DIS-Baseline);
   - 'physics_query':     方案一 (物理到时锚定 Query + 原生 1000 Hz 波包声学偏置 Cross-Attention);
   - 'fourier_feature':   方案二 (多尺度高频连续傅里叶字典嵌入 0.0725 Hz ~ 500 Hz);
   - 'sinc_filterbank':   方案三 (井筒声学奇数次驻波谐频 Sinc 带通可微滤波器组);
3. 实时记录各轮次训练损失、验证损失、密集多簇 (Nc>=4) R^2, alpha MAE, Cf Log10 MAE, W1 距离与耗时;
4. 最佳检查点与训练历史持久化至 checkpoints/ 与 output/weights/。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from PaperC_CJNO_Wellbore_Inversion.src.dataset import PilotInversionDataset
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_embedding_variants import TGDISEmbeddingModel
from PaperC_CJNO_Wellbore_Inversion.src.losses import CompositeInversionLoss
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_CHECKPOINTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "checkpoints")

VARIANT_NAMES = {
    "baseline_resample": "TG-DIS-Baseline (4096-pts Downsampling)",
    "physics_query": "TG-DIS-QueryEmbed (Physics Query + Delay Bias Attention)",
    "fourier_feature": "TG-DIS-FourierEmbed (Continuous Fourier Features 0.07-500Hz)",
    "sinc_filterbank": "TG-DIS-SincEmbed (Acoustic Harmonic Sinc Filterbank)",
}


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """在验证集上全面评测模型物理指标"""
    model.eval()
    val_losses = []
    val_preds_alpha = []
    val_preds_cf = []
    val_preds_exist = []
    val_preds_dx = []
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
            if "p_exist" in vout:
                val_preds_exist.append(vout["p_exist"].cpu())
            if "delta_x" in vout:
                val_preds_dx.append(vout["delta_x"].cpu())
            val_preds_field.append(vout["m_alpha_grid"].cpu())

            val_targets_alpha.append(vbatch["alpha"])
            val_targets_cf.append(vbatch["cf"])
            val_targets_field.append(vbatch["m_alpha_grid"])
            val_masks.append(vbatch["mask"])
            val_positions.append(vbatch["positions"])

    pred_dict: Dict[str, Any] = {
        "alpha": torch.cat(val_preds_alpha, dim=0),
        "cf": torch.cat(val_preds_cf, dim=0),
        "m_alpha_grid": torch.cat(val_preds_field, dim=0),
    }
    if val_preds_exist:
        pred_dict["p_exist"] = torch.cat(val_preds_exist, dim=0)
    if val_preds_dx:
        pred_dict["delta_x"] = torch.cat(val_preds_dx, dim=0)

    targ_dict = {
        "alpha": torch.cat(val_targets_alpha, dim=0),
        "cf": torch.cat(val_targets_cf, dim=0),
        "m_alpha_grid": torch.cat(val_targets_field, dim=0),
        "mask": torch.cat(val_masks, dim=0),
        "positions": torch.cat(val_positions, dim=0),
    }

    metrics = compute_inversion_metrics(pred_dict, targ_dict)
    mean_val_loss = float(np.mean(val_losses))
    return mean_val_loss, metrics


def train_single_variant(
    embedding_type: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 6e-4,
    weight_decay: float = 1e-3,
    device_str: str = "cpu",
    output_dir: str = DEFAULT_WEIGHTS_DIR,
    checkpoint_dir: str = DEFAULT_CHECKPOINTS_DIR,
    lambda_alpha: float = 3.0,
    lambda_c: float = 1.0,
    lambda_w: float = 0.0002,
    lambda_cons: float = 0.05,
    lambda_exist: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """训练单个 Embedding 变体模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str)
    variant_desc = VARIANT_NAMES.get(embedding_type, embedding_type)
    print(f"\n{'='*75}")
    print(f"[*] 启动变体训练: {embedding_type} ({variant_desc})")
    print(f"[*] 设备: {device} | Epochs: {epochs} | LR: {lr} | Seed: {seed}")
    print(f"{'='*75}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 构造模型
    model = TGDISEmbeddingModel(
        embedding_type=embedding_type,
        use_acoustic_bias=True,
        in_channels=2,
        cond_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        p=64,
        max_nc=6,
        d_well=128,
        delta_x_max=10.0,
        n_win_pts=301,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 可训练参数量: {total_params:,}")

    # 统一损失函数与优化器
    criterion = CompositeInversionLoss(
        lambda_alpha=lambda_alpha,
        lambda_c=lambda_c,
        lambda_w=lambda_w,
        lambda_cons=lambda_cons,
        lambda_exist=lambda_exist,
        lambda_pos=0.0,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_metrics: Dict[str, Any] = {}
    best_epoch = 0

    best_weight_out = os.path.join(output_dir, f"embedding_ablation_{embedding_type}_best.pt")
    best_weight_ckpt = os.path.join(checkpoint_dir, f"embedding_ablation_{embedding_type}_best.pt")
    history_path = os.path.join(output_dir, f"embedding_ablation_{embedding_type}_history.json")

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []

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

        scheduler.step()
        ep_duration = time.time() - t0

        # 验证评估
        val_loss, vmetrics = evaluate(model, criterion, val_loader, device)

        mean_train_loss = float(np.mean(train_losses))
        r2_dense = float(vmetrics.get("alpha_r2_dense", vmetrics["alpha_r2"]))
        r2_overall = float(vmetrics["alpha_r2"])
        alpha_mae = float(vmetrics["alpha_mae"])
        cf_log10_mae = float(vmetrics["cf_log10_mae"])
        w1_m = float(vmetrics["w1_mean_m"])
        f1 = float(vmetrics.get("f1_score", 1.0))

        # 综合评分: 越低越好，用于遴选最佳模型
        composite_score = alpha_mae - 0.6 * r2_dense + 0.05 * cf_log10_mae + 0.005 * w1_m

        is_best = composite_score < best_score
        if is_best:
            best_score = composite_score
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "alpha_r2_dense": r2_dense,
                "alpha_r2_overall": r2_overall,
                "alpha_mae": alpha_mae,
                "cf_log10_mae": cf_log10_mae,
                "w1_mean_m": w1_m,
                "f1_score": f1,
                "composite_score": composite_score,
            }
            # 保存最佳权重
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "embedding_type": embedding_type,
                    "metrics": best_metrics,
                },
                best_weight_out,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "embedding_type": embedding_type,
                    "metrics": best_metrics,
                },
                best_weight_ckpt,
            )

        ep_record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            "alpha_r2_dense": r2_dense,
            "alpha_r2_overall": r2_overall,
            "alpha_mae": alpha_mae,
            "cf_log10_mae": cf_log10_mae,
            "w1_mean_m": w1_m,
            "f1_score": f1,
            "composite_score": composite_score,
            "is_best": is_best,
            "epoch_duration_s": ep_duration,
        }
        history.append(ep_record)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            star = " [*BEST*]" if is_best else ""
            print(
                f"[{embedding_type:17s}] Ep {epoch:02d}/{epochs:02d} | "
                f"TrLoss: {mean_train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"Dense R^2: {r2_dense:+.4f} | All R^2: {r2_overall:+.4f} | "
                f"MAE: {alpha_mae:.4f} | W1: {w1_m:.2f}m | F1: {f1:.3f} | {ep_duration:.2f}s{star}"
            )

    total_time = time.time() - start_train_time
    print(f"\n[*] 变体 {embedding_type} 训练完成! 总耗时: {total_time:.1f}s")
    print(f"[*] 最佳 Epoch {best_epoch}: Dense R^2={best_metrics.get('alpha_r2_dense', 0):+.4f}, "
          f"All R^2={best_metrics.get('alpha_r2_overall', 0):+.4f}, MAE={best_metrics.get('alpha_mae', 0):.4f}")

    # 保存历史 JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_type": embedding_type,
                "variant_name": variant_desc,
                "total_train_time_s": total_time,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "embedding_type": embedding_type,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history_path": history_path,
        "weight_path": best_weight_out,
        "total_time_s": total_time,
    }


def run_embedding_ablation_training(
    variants: Optional[List[str]] = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 6e-4,
    device_str: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """顺序执行指定 Embedding 变体的全套消融训练"""
    if variants is None:
        variants = ["baseline_resample", "physics_query", "fourier_feature", "sinc_filterbank"]

    print(f"\n{'#'*80}")
    print(f"[*] 启动端到端连续时空 Embedding 四大变体消融对标训练实验")
    print(f"[*] 对标变体列表: {variants}")
    print(f"[*] 统一超参数: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Device={device_str}, Seed={seed}")
    print(f"{'#'*80}\n")

    # 1. 载入数据集 (启用原生 1000 Hz 全速率波形)
    print("[*] 正在载入物理仿真数据集 (含 60,001 点原生波形)...")
    train_dataset = PilotInversionDataset(split="train", load_raw_wave=True)
    val_dataset = PilotInversionDataset(split="val", load_raw_wave=True)
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    print(f"[*] 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    results: Dict[str, Any] = {}
    for var in variants:
        res = train_single_variant(
            embedding_type=var,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device_str=device_str,
            seed=seed,
        )
        results[var] = res

    # 打印全局汇总表
    print(f"\n{'='*85}")
    print(f"{'Embedding 变体名称':<25} {'最佳轮次':<8} {'Dense R^2':<12} {'Overall R^2':<12} {'Alpha MAE':<10} {'W1 距离':<10} {'耗时 (s)':<8}")
    print(f"{'-'*85}")
    for var, res in results.items():
        bm = res["best_metrics"]
        print(
            f"{var:<25} "
            f"{res['best_epoch']:<8} "
            f"{bm.get('alpha_r2_dense', 0.0):+10.4f}  "
            f"{bm.get('alpha_r2_overall', 0.0):+10.4f}  "
            f"{bm.get('alpha_mae', 0.0):8.4f}  "
            f"{bm.get('w1_mean_m', 0.0):6.2f}m   "
            f"{res['total_time_s']:6.1f}"
        )
    print(f"{'='*85}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="端到端连续时空 Embedding 架构升级消融训练")
    parser.add_argument("--variants", nargs="+", default=["baseline_resample", "physics_query", "fourier_feature", "sinc_filterbank"],
                        choices=["baseline_resample", "physics_query", "fourier_feature", "sinc_filterbank"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_embedding_ablation_training(
        variants=args.variants,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
