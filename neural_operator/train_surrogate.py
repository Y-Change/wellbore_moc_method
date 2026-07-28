# -*- coding: utf-8 -*-
"""
neural_operator/train_surrogate.py — 神经算子正演代理模型训练与评估脚本

核心特性：
1. 混合精度加速 (AMP)：充分发挥 RTX 4060 Ti 16G 显存与 Tensor Core 算力，显存低、收敛快；
2. 时-频联合物理损失 (Composite Physics-Informed Loss)：
     L_total = MSE(y_pred, y_true) + 0.1 * MSE(d/dt y_pred, d/dt y_true) + 0.05 * MSE(FFT(y_pred), FFT(y_true))
   确保波头拐点锐度与高频反射混响不衰减；
3. 自动化 checkpoint 与图表测评：自动保存最佳模型至 output/models/fno_surrogate_best.pt，并在验证集上绘制对比图。

运行方式：
    python neural_operator/train_surrogate.py --epochs 100 --batch-size 32 --lr 1e-3
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time as time_module
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# 确保导入根目录模块
_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'README.md')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError('Cannot find wellbore_moc_method root')
    _d = _parent

from neural_operator.fno_1d import FNO1dSurrogate
from neural_operator.dataset_surrogate import get_surrogate_dataloaders


class CompositePhysicsLoss(nn.Module):
    """时域-导数-频域三位一体综合物理损失函数"""
    def __init__(self, alpha_grad: float = 0.1, beta_fft: float = 0.05):
        super(CompositePhysicsLoss, self).__init__()
        self.alpha = alpha_grad
        self.beta = beta_fft

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. 时域基础均方误差 (MSE)
        loss_time = F.mse_loss(pred, target)

        # 2. 时域一阶差分/导数误差 —— 强制拉升波头阶跃与反射拐点的锐度
        diff_pred = pred[:, :, 1:] - pred[:, :, :-1]
        diff_target = target[:, :, 1:] - target[:, :, :-1]
        loss_grad = F.mse_loss(diff_pred, diff_target)

        # 3. 频域/谱能量一致性误差 —— 保证反射波倒谱主频与次频不丢干净
        fft_pred = torch.fft.rfft(pred, dim=-1)
        fft_target = torch.fft.rfft(target, dim=-1)
        loss_fft = F.mse_loss(torch.abs(fft_pred), torch.abs(fft_target))

        return loss_time + self.alpha * loss_grad + self.beta * loss_fft


def evaluate_and_plot(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    epoch: int,
):
    """挑选 4 条验证样本绘制 真解 vs FNO 预测 对比图"""
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            break  # 仅取第一批次

    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    preds_np = preds.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    t_axis = np.linspace(0.0, 50.0, targets_np.shape[-1])

    for i in range(min(4, len(targets_np))):
        ax = axes[i]
        ax.plot(t_axis, targets_np[i, 0], 'k-', label="MOC 物理真解 (Ground Truth)", linewidth=1.5, alpha=0.85)
        ax.plot(t_axis, preds_np[i, 0], 'r--', label="FNO-1D 神经算子代理预测", linewidth=1.2)
        
        # 标出前 15 秒核心进液混响区
        ax.set_xlim(0.0, 25.0)
        ax.set_title(f"验证集样本 #{i+1} — 井口水头时程对比 (前 25s 放大)", fontsize=11)
        ax.set_xlabel("时间 t [s]", fontsize=10)
        ax.set_ylabel("水头 H_wh [m]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"fno_eval_epoch_{epoch:03d}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  [图表生成] 验证集对比效果图已保存至: {plot_path}")


def main():
    if sys.platform.startswith('win') and hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="FNO-1D 神经算子代理模型训练器")
    parser.add_argument("--data-dir", type=str, default="output/lhs_dataset/data", help="LHS 数据集目录")
    parser.add_argument("--out-dir", type=str, default="output/models", help="模型与日志保存目录")
    parser.add_argument("--epochs", type=int, default=80, help="训练总迭代轮次 (推荐 80~150)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch Size (显存充足可设 64 或 128)")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="初始学习率")
    parser.add_argument("--width", type=int, default=64, help="FNO 隐层通道数 width")
    parser.add_argument("--modes", type=int, default=32, help="保留傅里叶模式数 modes")
    parser.add_argument("--layers", type=int, default=4, help="FNO 残差块级联层数")
    parser.add_argument("--n-time", type=int, default=4096, help="对齐插值的时序长度 (2的幂次)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 72)
    print("FNO-1D 神经算子正演代理模型 —— 物理守恒训练启动")
    print("=" * 72)
    print(f"  运行设备 : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  网络结构 : FNO1dSurrogate(width={args.width}, modes={args.modes}, layers={args.layers})")
    print(f"  数据集   : {args.data_dir} (目标对齐点数 N={args.n_time})")
    print(f"  训练轮次 : {args.epochs} Epochs | Batch Size = {args.batch_size} | LR = {args.lr}")
    print("=" * 72)

    # 1. 挂载 DataLoader
    print("\n[Step 1] 正在挂载与解析物理真解数据集...")
    train_loader, val_loader = get_surrogate_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=0, n_time_target=args.n_time
    )

    # 2. 构建模型与优化器
    model = FNO1dSurrogate(
        in_channels=4, out_channels=1, width=args.width, modes=args.modes, num_layers=args.layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.0e-6)
    criterion = CompositePhysicsLoss(alpha_grad=0.1, beta_fft=0.05).to(device)
    scaler = GradScaler()

    best_val_loss = float("inf")
    log_path = os.path.join(args.out_dir, "train_surrogate_log.csv")
    
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

    # 3. 训练循环
    print("\n[Step 2] 开始迭代优化模型权重...\n")
    t_start_train = time_module.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time_module.time()
        model.train()
        train_loss_total = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # 自动混合精度前向计算
            with autocast():
                preds = model(inputs)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item() * inputs.size(0)

        train_loss_avg = train_loss_total / len(train_loader.dataset)
        scheduler.step()

        # 验证评估
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with autocast():
                    preds = model(inputs)
                    loss = criterion(preds, targets)
                val_loss_total += loss.item() * inputs.size(0)

        val_loss_avg = val_loss_total / len(val_loader.dataset)
        elapsed = time_module.time() - t0

        # 记录与存档
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss_avg:.6f}", f"{val_loss_avg:.6f}", f"{optimizer.param_groups[0]['lr']:.2e}", f"{elapsed:.2f}"])

        # 打印日志
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1 or epoch == args.epochs or val_loss_avg < best_val_loss:
            mark = ""
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                best_model_path = os.path.join(args.out_dir, "fno_surrogate_best.pt")
                torch.save(model.state_dict(), best_model_path)
                mark = "★ [保存新最佳权重]"
                
            print(f"  Epoch [{epoch:3d}/{args.epochs}] | Train Loss: {train_loss_avg:.6f} | "
                  f"Val Loss: {val_loss_avg:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | 耗时: {elapsed:.1f}s {mark}")

        # 每 20 轮或最后一轮生成图表
        if epoch % max(1, args.epochs // 4) == 0 or epoch == args.epochs:
            evaluate_and_plot(model, val_loader, device, args.out_dir, epoch)

    total_time = time_module.time() - t_start_train
    print("\n" + "=" * 72)
    print(f"FNO 神经算子代理模型训练圆满完毕！总耗时: {total_time/60:.2f} min")
    print(f"最佳验证集误差 (Best Val Loss): {best_val_loss:.6f}")
    print(f"模型权重归档: {os.path.join(args.out_dir, 'fno_surrogate_best.pt')}")
    print(f"训练记录日志: {log_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
