# -*- coding: utf-8 -*-
"""
evaluate.py — 倒谱 Cep-DeepONet 模型性能评估与指标审计

严格遵循 CMAME 2023 论文《Fourier-DeepONet》Section 4 评估准则：
1. MAE (Mean Absolute Error)
2. RMSE (Root Mean Squared Error)
3. Rel-L2 (Relative L2 Norm Error)
4. Pearson 相关系数 (Pearson Correlation) 与 1D-SSIM (结构相似度)
5. 裂缝检出绝对位置偏差 (Depth Localization Error, 米)
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from dataset import get_dataloaders
from model import CepDeepONet


def compute_1d_ssim(x: np.ndarray, y: np.ndarray, win_size: int = 21) -> float:
    """计算一维信号的局部结构相似度 (1D Structural Similarity)"""
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    # 滑窗均值与方差
    kernel = np.ones(win_size) / win_size
    pad = win_size // 2
    x_pad = np.pad(x, pad, mode='edge')
    y_pad = np.pad(y, pad, mode='edge')

    mu_x = np.convolve(x_pad, kernel, mode='valid')
    mu_y = np.convolve(y_pad, kernel, mode='valid')

    sigma_x_sq = np.convolve(x_pad ** 2, kernel, mode='valid') - mu_x ** 2
    sigma_y_sq = np.convolve(y_pad ** 2, kernel, mode='valid') - mu_y ** 2
    sigma_xy = np.convolve(x_pad * y_pad, kernel, mode='valid') - mu_x * mu_y

    sigma_x_sq = np.maximum(0, sigma_x_sq)
    sigma_y_sq = np.maximum(0, sigma_y_sq)

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return float(np.mean(ssim_map))


def match_peaks_to_groundtruth(
    y_pred: np.ndarray,
    x_grid: np.ndarray,
    x_true: List[float],
    height_thresh: float = 0.2,
    match_tol_m: float = 80.0
) -> Dict:
    """
    基于极大值检出预测裂缝位置，并与真实裂缝进行就近匹配与偏差统计
    """
    peaks_idx, props = scipy.signal.find_peaks(y_pred, height=height_thresh, distance=5)
    pred_depths = x_grid[peaks_idx]
    pred_heights = y_pred[peaks_idx]

    matched_errors = []
    matched_pairs = []
    unmatched_true = list(x_true)

    # 贪心就近匹配
    for true_depth in x_true:
        if len(pred_depths) == 0:
            break
        dists = np.abs(pred_depths - true_depth)
        min_idx = np.argmin(dists)
        if dists[min_idx] <= match_tol_m:
            err = float(dists[min_idx])
            matched_errors.append(err)
            matched_pairs.append({
                'true_m': float(true_depth),
                'pred_m': float(pred_depths[min_idx]),
                'error_m': err,
                'peak_height': float(pred_heights[min_idx])
            })
            if true_depth in unmatched_true:
                unmatched_true.remove(true_depth)

    return {
        'n_true': len(x_true),
        'n_detected': len(pred_depths),
        'n_matched': len(matched_pairs),
        'match_rate': len(matched_pairs) / max(len(x_true), 1),
        'mean_error_m': float(np.mean(matched_errors)) if matched_errors else float('nan'),
        'max_error_m': float(np.max(matched_errors)) if matched_errors else float('nan'),
        'matched_details': matched_pairs
    }


def evaluate_checkpoint(
    ckpt_path: str,
    base_dir: str,
    output_dir: str,
    device_str: str = "cpu"
) -> Dict:
    device = torch.device(device_str)
    print(f"[Evaluate] 正在加载 Checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 加载模型
    model = CepDeepONet(
        T_steps=16384,
        Nx=1000,
        L=5000.0,
        param_dim=8,
        latent_channels=64,
        fourier_modes=32,
        num_fourier_layers=4
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载验证集
    _, val_loader, _, _ = get_dataloaders(
        base_dir=base_dir,
        batch_size=1,
        train_ratio=0.8,
        seed=checkpoint.get('seed', 42),
        augment_train=False
    )

    x_grid = np.linspace(0.0, 5000.0, 1000, endpoint=True)
    results = []

    print("=" * 85)
    print(f"{'Case':<22} | {'Rel-L2':^9} | {'MAE':^8} | {'RMSE':^8} | {'SSIM':^7} | {'Pearson':^8} | {'Mean Err (m)':^12}")
    print("=" * 85)

    with torch.no_grad():
        for batch in val_loader:
            case_name = batch['case_name'][0]
            H = batch['H'].to(device)
            xi = batch['xi'].to(device)
            y_true_t = batch['y'].to(device)
            x_f_true = [float(val) for val in batch['x_f'][0]]

            y_pred_t = model(H, xi)

            y_pred = y_pred_t[0].cpu().numpy()
            y_true = y_true_t[0].cpu().numpy()

            # 论文评估指标
            diff_l2 = np.linalg.norm(y_pred - y_true)
            true_l2 = np.linalg.norm(y_true) + 1e-8
            rel_l2 = float(diff_l2 / true_l2)
            mae = float(np.mean(np.abs(y_pred - y_true)))
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

            # 结构相似度与相关性
            ssim_val = compute_1d_ssim(y_pred, y_true)
            pearson = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 1e-6 else 0.0

            # 峰值定位审计
            peak_stats = match_peaks_to_groundtruth(y_pred, x_grid, x_f_true)

            print(
                f"{case_name:<22} | {rel_l2:^9.4f} | {mae:^8.4f} | {rmse:^8.4f} | "
                f"{ssim_val:^7.4f} | {pearson:^8.4f} | {peak_stats['mean_error_m']:^12.2f}"
            )

            results.append({
                'case': case_name,
                'rel_l2': rel_l2,
                'mae': mae,
                'rmse': rmse,
                'ssim': ssim_val,
                'pearson': pearson,
                'peak_stats': peak_stats,
                'y_pred': y_pred.tolist(),
                'y_true': y_true.tolist()
            })

    print("=" * 85)
    mean_rel_l2 = np.mean([r['rel_l2'] for r in results])
    mean_mae = np.mean([r['mae'] for r in results])
    mean_rmse = np.mean([r['rmse'] for r in results])
    mean_ssim = np.mean([r['ssim'] for r in results])
    mean_pearson = np.mean([r['pearson'] for r in results])
    mean_loc_err = np.nanmean([r['peak_stats']['mean_error_m'] for r in results])

    print(f"验证集平均指标汇总:")
    print(f"  - Mean Relative L2:   {mean_rel_l2:.4f}")
    print(f"  - Mean Absolute Error: {mean_mae:.4f}")
    print(f"  - Mean RMSE:          {mean_rmse:.4f}")
    print(f"  - Mean SSIM (1D):     {mean_ssim:.4f}")
    print(f"  - Mean Pearson r:     {mean_pearson:.4f}")
    print(f"  - Mean Location Error:{mean_loc_err:.2f} m")

    summary = {
        'checkpoint': ckpt_path,
        'summary_metrics': {
            'mean_rel_l2': float(mean_rel_l2),
            'mean_mae': float(mean_mae),
            'mean_rmse': float(mean_rmse),
            'mean_ssim': float(mean_ssim),
            'mean_pearson': float(mean_pearson),
            'mean_loc_err_m': float(mean_loc_err)
        },
        'cases': results
    }

    eval_json = os.path.join(output_dir, 'evaluation_summary.json')
    with open(eval_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[Evaluate] 详细评估报告已保存至: {eval_json}")

    return summary


if __name__ == '__main__':
    base_dir = r"e:\water_hammer_research\wellbore_moc_method\PaperA井口多裂缝水击响应\03_leakoff验证"
    ckpt_path = os.path.join(base_dir, "cep_deeponet", "results", "best_cep_deeponet.pt")
    output_dir = os.path.join(base_dir, "cep_deeponet", "results")

    evaluate_checkpoint(ckpt_path, base_dir, output_dir)
