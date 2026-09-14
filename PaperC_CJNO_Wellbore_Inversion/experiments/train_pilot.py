# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.train_pilot

水击波物理反演四大模型自动化训练管线：
1. ResNet1D: 经典 1D 残差卷积时序回归模型；
2. FNO1D: 1D 傅里叶神经算子频域谱卷积模型；
3. VanillaDeepONet: 经典 DeepONet (Wave Branch + Trunk 连续坐标)；
4. CJCepDeepONet: 倒谱增强特征线算子原型网络 (含 Voronoi 守恒积分池化层)。

统一基于 800 train / 100 val / 100 test 确定性切分，按验证集指标保存最优权重。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Type

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
from PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d import ResNet1D
from PaperC_CJNO_Wellbore_Inversion.src.models.fno1d import FNO1D
from PaperC_CJNO_Wellbore_Inversion.src.models.deeponet import VanillaDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_cep_deeponet import CJCepDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.losses import CompositeInversionLoss
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "resnet": ResNet1D,
    "fno": FNO1D,
    "deeponet": VanillaDeepONet,
    "cj_cep_deeponet": CJCepDeepONet,
}


DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")


def train_single_model(
    model_name: str,
    model_cls: Type[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    output_dir: str = DEFAULT_WEIGHTS_DIR,
) -> Dict[str, Any]:
    """训练单模型并返回历史记录与最佳指标"""
    print(f"\n========================================================")
    print(f"[*] 开始训练模型: {model_name} (Epochs: {epochs}, lr: {lr})")
    print(f"========================================================")

    os.makedirs(output_dir, exist_ok=True)
    best_weight_path = os.path.join(output_dir, f"{model_name}_best.pt")
    history_path = os.path.join(output_dir, f"{model_name}_history.json")

    model = model_cls().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 模型可训练参数量: {total_params:,}")

    # 损失物理尺度归一化平衡 (lambda_w=0.01 使 W1 误差 ~10m 平衡在 0.1 尺度)
    criterion = CompositeInversionLoss(lambda_alpha=1.0, lambda_c=1.0, lambda_w=0.01).to(device)
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

        scheduler.step()

        # 2. 验证评估 (每 2 轮或首末轮评估)
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            val_losses = []
            val_preds_alpha = []
            val_preds_cf = []
            val_targets_alpha = []
            val_targets_cf = []
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
                    val_targets_alpha.append(vbatch["alpha"])
                    val_targets_cf.append(vbatch["cf"])
                    val_masks.append(vbatch["mask"])
                    val_positions.append(vbatch["positions"])

            val_pred_dict = {
                "alpha": torch.cat(val_preds_alpha, dim=0),
                "cf": torch.cat(val_preds_cf, dim=0),
            }
            val_targ_dict = {
                "alpha": torch.cat(val_targets_alpha, dim=0),
                "cf": torch.cat(val_targets_cf, dim=0),
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
                        "model_name": model_name,
                    },
                    best_weight_path,
                )

            if epoch % 10 == 0 or epoch == epochs:
                tag = " [*BEST*]" if is_best else ""
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
                    f"alpha MAE: {vmetrics['alpha_mae']:.4f} | Cf MRE: {vmetrics['cf_mre_pct']:.1f}% | "
                    f"W1: {vmetrics['w1_mean_m']:.2f}m{tag}"
                )

    total_time = time.time() - start_time
    print(f"[OK] {model_name} finished training, time: {total_time:.1f}s, best model saved to: {best_weight_path}")

    # 保存训练历史
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
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
        "model_name": model_name,
        "best_weight_path": best_weight_path,
        "history_path": history_path,
        "train_time_sec": total_time,
        "best_metrics": best_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="PaperC 物理反演多模型自动化训练")
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数 (默认 60)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小 (默认 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率 (默认 1e-3)")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["resnet", "fno", "deeponet", "cj_cep_deeponet", "all"],
        help="指定训练的模型名称或 all",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 确定硬件设备与线程数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"[*] 执行环境: Device={device}, Torch CPU Threads={num_threads}")

    # 数据加载器
    print("[*] 加载数据集切分 (800 train / 100 val / 100 test)...")
    train_dataset = PilotInversionDataset(split="train", seed=args.seed)
    val_dataset = PilotInversionDataset(split="val", seed=args.seed)

    train_loader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    val_loader = val_dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)

    target_models = list(MODEL_REGISTRY.keys()) if args.models == "all" else [args.models]

    results = {}
    for m_name in target_models:
        m_cls = MODEL_REGISTRY[m_name]
        res = train_single_model(
            model_name=m_name,
            model_cls=m_cls,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        results[m_name] = res

    print("\n========================================================")
    print("                全套模型训练结果汇总                      ")
    print("========================================================")
    header = f"{'Model':<18} | {'alpha MAE':<10} | {'alpha R2':<10} | {'Cf MRE (%)':<11} | {'W1 (m)':<8} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))
    for m_name, res in results.items():
        bm = res["best_metrics"]
        print(
            f"{m_name:<18} | "
            f"{bm.get('alpha_mae', 0.0):<10.4f} | "
            f"{bm.get('alpha_r2', 0.0):<10.4f} | "
            f"{bm.get('cf_mre_pct', 0.0):<11.1f} | "
            f"{bm.get('w1_mean_m', 0.0):<8.2f} | "
            f"{res['train_time_sec']:<8.1f}"
        )


if __name__ == "__main__":
    main()
