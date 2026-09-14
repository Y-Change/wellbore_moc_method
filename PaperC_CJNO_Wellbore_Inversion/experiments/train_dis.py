# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.train_dis

TG-DIS-DeepONet (融合可微逆散射层剥离算子神经算子) 第三阶段训练管线:
1. 在 1,000 例物理仿真数据集 (800 train / 100 val / 100 test) 上训练 TG-DIS-DeepONet;
2. 采用复合多目标物理损失 (CompositeInversionLoss):
   - 流量份额 alpha 单纯形 KL + L1 监督 (lambda_alpha)
   - 水力顺应性 Log-Huber 跨量级回归 (lambda_c)
   - 1D 空间测度 Wasserstein 距离 (lambda_w)
   - 双轨协同物理一致性 (lambda_cons)
   - 裂缝起裂存在性分类 BCE (lambda_exist)
3. 采用 AdamW 优化器与 CosineAnnealingLR 退火调度;
4. 综合密集多簇 (Nc>=4) R^2, 全集与稀疏 MAE, Wasserstein W_1, 存在性 F1 进行模型验证与最佳检查点持久化;
5. 保存权重至:
   - checkpoints/tg_dis_deeponet_best.pt
   - output/weights/tg_dis_deeponet_best.pt
   并输出训练收敛历史 JSON: output/weights/tg_dis_deeponet_history.json.
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
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.losses import CompositeInversionLoss
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_CHECKPOINTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "checkpoints")


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """在验证集上评估当前模型，返回平均损失与全面物理指标"""
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


def train_tg_dis_model(
    epochs: int = 80,
    batch_size: int = 32,
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
    """主训练执行管线"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str)
    print(f"\n{'='*75}")
    print(f"[*] 启动 Phase 3 旗舰模型训练: TG-DIS-DeepONet")
    print(f"[*] 设备: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"[*] 损失权重: alpha={lambda_alpha}, c={lambda_c}, w1={lambda_w}, cons={lambda_cons}, exist={lambda_exist}")
    print(f"{'='*75}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. 载入数据集
    print("[*] 正在加载物理数据集...")
    train_dataset = PilotInversionDataset(split="train")
    val_dataset = PilotInversionDataset(split="val")
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    print(f"[*] 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # 2. 构造模型
    model = TGDISDeepONet(
        window_mode="relative_window",
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
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] TG-DIS-DeepONet 可训练参数量: {total_params:,}")

    # 载入 Phase 2 最优声学偏置权重以加速稳定收敛
    warmstart_path = os.path.join(output_dir, "tg_relative_bias_best.pt")
    if os.path.exists(warmstart_path):
        ws_ckpt = torch.load(warmstart_path, map_location=device)
        ws_sd = ws_ckpt["model_state_dict"] if "model_state_dict" in ws_ckpt else ws_ckpt
        m_sd = model.state_dict()
        transferred = 0
        for k, v in ws_sd.items():
            if k in m_sd and m_sd[k].shape == v.shape:
                m_sd[k] = v
                transferred += 1
        model.load_state_dict(m_sd)
        print(f"[*] 成功装载 Phase 2 声学预训练偏置权重 ({transferred} 个张量层)")

    # 3. 损失函数与优化器
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

    best_weight_path_output = os.path.join(output_dir, "tg_dis_deeponet_best.pt")
    best_weight_path_ckpt = os.path.join(checkpoint_dir, "tg_dis_deeponet_best.pt")
    history_path = os.path.join(output_dir, "tg_dis_deeponet_history.json")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_l_alpha = []
        train_l_c = []
        train_l_w = []
        train_l_cons = []
        train_l_exist = []

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
            if "loss_exist" in loss_dict:
                train_l_exist.append(loss_dict["loss_exist"].item())

        scheduler.step()

        # 验证评估 (每轮评估以敏锐捕捉最佳检查点)
        val_loss, vmetrics = evaluate(model, criterion, val_loader, device)

        mean_train_loss = float(np.mean(train_losses))
        r2_dense = float(vmetrics.get("alpha_r2_dense", vmetrics["alpha_r2"]))
        r2_overall = float(vmetrics["alpha_r2"])
        alpha_mae = float(vmetrics["alpha_mae"])
        cf_log10_mae = float(vmetrics["cf_log10_mae"])
        w1_m = float(vmetrics["w1_mean_m"])
        f1 = float(vmetrics.get("f1_score", 1.0))

        # 综合评分: 越低越好，强力激励 dense R^2 突破与 alpha MAE 降低
        # 目标: dense R^2 > 0.75, alpha MAE < 0.08, W1 < 5.0m
        composite_score = alpha_mae - 0.6 * r2_dense + 0.05 * cf_log10_mae + 0.005 * w1_m

        is_best = composite_score < best_score
        if is_best:
            best_score = composite_score
            best_metrics = vmetrics
            best_epoch = epoch
            save_payload = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": vmetrics,
                "model_name": "TGDISDeepONet",
                "config": {
                    "window_mode": "relative_window",
                    "use_acoustic_bias": True,
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 2,
                    "p": 64,
                    "max_nc": 6,
                    "d_well": 128,
                    "delta_x_max": 10.0,
                },
                "total_params": total_params,
            }
            torch.save(save_payload, best_weight_path_output)
            torch.save(save_payload, best_weight_path_ckpt)

        record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            "alpha_mae": alpha_mae,
            "alpha_r2": r2_overall,
            "alpha_r2_dense": r2_dense,
            "cf_mre_pct": vmetrics["cf_mre_pct"],
            "cf_log10_mae": cf_log10_mae,
            "w1_mean_m": w1_m,
            "f1_score": f1,
            "simplex_max_dev": vmetrics["simplex_max_dev"],
            "composite_score": composite_score,
            "is_best": is_best,
        }
        history.append(record)

        if epoch % 10 == 0 or epoch == epochs or is_best and epoch % 5 == 0:
            best_tag = " [*BEST*]" if is_best else ""
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Loss: {mean_train_loss:.3f}/{val_loss:.3f} | "
                f"alpha MAE: {alpha_mae:.4f} (R2={r2_overall:.3f}, Dense={r2_dense:.3f}) | "
                f"Cf MRE: {vmetrics['cf_mre_pct']:.1f}% | W1: {w1_m:.2f}m | F1: {f1:.3f}{best_tag}"
            )

    total_time = time.time() - start_time
    print(f"\n[OK] TG-DIS-DeepONet 训练完成! 总耗时: {total_time:.1f}s")
    print(f"[*] 最佳 Epoch: {best_epoch}")
    print(f"[*] 最佳验证指标: alpha MAE = {best_metrics.get('alpha_mae', 0):.4f}, "
          f"alpha R2 = {best_metrics.get('alpha_r2', 0):.3f} (Dense R2 = {best_metrics.get('alpha_r2_dense', 0):.3f}), "
          f"W1 = {best_metrics.get('w1_mean_m', 0):.2f}m, "
          f"F1 = {best_metrics.get('f1_score', 0):.3f}, "
          f"Simplex Dev = {best_metrics.get('simplex_max_dev', 0):.2e}")
    print(f"[*] 权重已保存至: {best_weight_path_output} 及 {best_weight_path_ckpt}")

    # 保存训练历史 JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "TGDISDeepONet",
                "total_params": total_params,
                "train_time_sec": total_time,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[*] 训练历史已保存至: {history_path}")

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_weight_output": best_weight_path_output,
        "best_weight_ckpt": best_weight_path_ckpt,
        "history_path": history_path,
        "train_time_sec": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="TG-DIS-DeepONet 训练脚本")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数 (默认 80)")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小 (默认 32)")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="初始学习率 (默认 1.2e-3)")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (默认 cpu)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_WEIGHTS_DIR, help="输出权重目录")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR, help="检查点目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    train_tg_dis_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
