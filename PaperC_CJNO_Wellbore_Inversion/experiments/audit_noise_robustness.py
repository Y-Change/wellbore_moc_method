# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.experiments.audit_noise_robustness

Phase 3 噪声鲁棒性与物理摄动压力审计系统:
1. 对标评估 TG-DIS-DeepONet (Phase 3 旗舰) vs TG-DeepONet (Phase 2 基线);
2. 注入多源多模态现场工况应力干扰:
   - Clean Baseline (无干扰基准)
   - 高斯白噪声 (AWGN): SNR 30dB, 20dB, 10dB
   - 有色粉红噪声 (Colored 1/f Pink Noise): SNR 30dB, 20dB, 10dB
   - 声波传播速度物理扰动 (Sound Speed Perturbations): +1%, -1%
3. 量化核心指标退化率 (Relative Performance Degradation):
   Degradation = (|Metric_perturbed - Metric_clean| / Metric_clean) * 100%
   核心验收门槛: 在 20dB 强噪声干扰下，核心指标衰减幅度 < 15.0%
4. 输出结构化指标文件: output/phase3_noise_robustness_metrics.json.
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
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet import TGCJDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.metrics import compute_inversion_metrics

DEFAULT_WEIGHTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output" / "weights")
DEFAULT_CHECKPOINTS_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "checkpoints")
DEFAULT_OUTPUT_DIR = str(_ROOT / "PaperC_CJNO_Wellbore_Inversion" / "output")


def add_awgn_noise(wave: torch.Tensor, snr_db: float, seed: int = 42) -> torch.Tensor:
    """向时域波形注入指定信噪比的高斯白噪声 (AWGN)"""
    torch.manual_seed(seed)
    noisy_wave = wave.clone()
    B, C, T = wave.shape

    for b in range(B):
        for c in range(C):
            sig = wave[b, c]
            sig_power = torch.mean(sig ** 2)
            if sig_power <= 1e-12:
                continue
            noise_power = sig_power * (10.0 ** (-snr_db / 10.0))
            noise_std = torch.sqrt(noise_power)
            noise = torch.randn_like(sig) * noise_std
            noisy_wave[b, c] = sig + noise

    return noisy_wave


def generate_pink_noise_1d(n_pts: int, seed: int = 42) -> torch.Tensor:
    """生成标准 1/f 粉红噪声序列"""
    np.random.seed(seed)
    # 频域合成法
    white = np.random.randn(n_pts)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_pts)
    freqs[0] = freqs[1] * 0.5  # 避免 DC 零除
    # 幅度按 1/sqrt(f) 衰减
    S = 1.0 / np.sqrt(freqs)
    X_pink = X * S
    pink = np.fft.irfft(X_pink, n=n_pts)
    pink = pink / (np.std(pink) + 1e-8)
    return torch.from_numpy(pink).float()


def add_pink_noise(wave: torch.Tensor, snr_db: float, seed: int = 42) -> torch.Tensor:
    """向时域波形注入指定信噪比的 1/f 有色粉红噪声"""
    noisy_wave = wave.clone()
    B, C, T = wave.shape

    for b in range(B):
        for c in range(C):
            sig = wave[b, c]
            sig_power = torch.mean(sig ** 2)
            if sig_power <= 1e-12:
                continue
            noise_power = sig_power * (10.0 ** (-snr_db / 10.0))
            noise_std = torch.sqrt(noise_power)
            pink_unit = generate_pink_noise_1d(T, seed=seed + b * 10 + c).to(sig.device)
            noisy_wave[b, c] = sig + pink_unit * noise_std

    return noisy_wave


def perturb_sound_speed(
    batch: Dict[str, torch.Tensor],
    pct_perturbation: float,
) -> Dict[str, torch.Tensor]:
    """对工况中的声波传播速度进行物理摄动 (如 +1.0% 或 -1.0%)"""
    perturbed_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    cond = perturbed_batch["cond"].clone()
    # cond[:, 1] = (a - 1450.0) / 20.0  => a = cond[:, 1] * 20.0 + 1450.0
    a_nom = cond[:, 1] * 20.0 + 1450.0
    a_pert = a_nom * (1.0 + pct_perturbation / 100.0)
    cond[:, 1] = (a_pert - 1450.0) / 20.0
    perturbed_batch["cond"] = cond
    return perturbed_batch


def evaluate_batch_with_model(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """在指定批次上运行前向推理并计算核心指标"""
    model.eval()
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        out = model(batch_dev)

    pred_alpha = out["alpha"].cpu()
    pred_cf = out["cf"].cpu() if "cf" in out else torch.zeros_like(pred_alpha)
    pred_field = out.get("m_alpha_grid", torch.zeros(len(pred_alpha), 500)).cpu()

    pred_dict: Dict[str, Any] = {
        "alpha": pred_alpha,
        "cf": pred_cf,
        "m_alpha_grid": pred_field,
    }
    if "p_exist" in out:
        pred_dict["p_exist"] = out["p_exist"].cpu()
        if "delta_x" in out:
            pred_dict["delta_x"] = out["delta_x"].cpu()
    else:
        pred_dict["p_exist"] = (pred_alpha > 0.05).float()

    targ_dict = {
        "alpha": batch["alpha"],
        "cf": batch["cf"],
        "m_alpha_grid": batch["m_alpha_grid"],
        "mask": batch["mask"],
        "positions": batch["positions"],
    }
    return compute_inversion_metrics(pred_dict, targ_dict)


def run_robustness_audit(
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    device_str: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """运行全套噪声与物性应力审计测试"""
    device = torch.device(device_str)
    print(f"\n{'='*85}")
    print(f"[*] 启动 Phase 3 两阶噪声与物理扰动鲁棒性压力审计")
    print(f"[*] 测试对标: TG-DIS-DeepONet (Phase 3) vs TG-DeepONet (Phase 2)")
    print(f"{'='*85}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载测试集
    test_ds = PilotInversionDataset(split="test")
    test_loader = test_ds.get_dataloader(batch_size=100, shuffle=False)
    base_batch = next(iter(test_loader))

    # 2. 加载两个对比模型
    # A. TG-DeepONet (Phase 2 基线)
    tg_model = TGCJDeepONet(
        window_mode="relative_window",
        use_acoustic_bias=True,
        d_model=64,
        n_heads=4,
        n_layers=2,
        p=64,
    ).to(device)
    tg_weight = os.path.join(weights_dir, "tg_relative_bias_best.pt")
    ckpt_tg = torch.load(tg_weight, map_location=device)
    tg_model.load_state_dict(ckpt_tg["model_state_dict"] if "model_state_dict" in ckpt_tg else ckpt_tg)
    tg_model.eval()

    # B. TG-DIS-DeepONet (Phase 3 旗舰)
    dis_model = TGDISDeepONet(
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
    path_ckpt = os.path.join(checkpoints_dir, "tg_dis_deeponet_best.pt")
    path_weights = os.path.join(weights_dir, "tg_dis_deeponet_best.pt")
    dis_path = path_ckpt if os.path.exists(path_ckpt) else path_weights
    ckpt_dis = torch.load(dis_path, map_location=device)
    dis_model.load_state_dict(ckpt_dis["model_state_dict"] if "model_state_dict" in ckpt_dis else ckpt_dis)
    dis_model.eval()

    # 3. 扰动场景定义
    scenarios = [
        ("clean", "Clean Baseline", None, None),
        ("awgn_30db", "AWGN (SNR 30dB)", "awgn", 30.0),
        ("awgn_20db", "AWGN (SNR 20dB) [Key Stress]", "awgn", 20.0),
        ("awgn_10db", "AWGN (SNR 10dB) [Severe]", "awgn", 10.0),
        ("pink_30db", "Pink Noise (SNR 30dB)", "pink", 30.0),
        ("pink_20db", "Pink Noise (SNR 20dB) [Key Stress]", "pink", 20.0),
        ("pink_10db", "Pink Noise (SNR 10dB) [Severe]", "pink", 10.0),
        ("speed_plus_1pct", "Sound Speed +1.0%", "speed", +1.0),
        ("speed_minus_1pct", "Sound Speed -1.0%", "speed", -1.0),
    ]

    audit_results: Dict[str, Any] = {
        "tg_deeponet": {},
        "tg_dis_deeponet": {},
        "degradation_analysis": {},
    }

    print(f"\n[*] 正在逐项执行环境应力压力测试...")
    for s_key, s_name, s_type, s_param in scenarios:
        # 构建当前工况测试批次
        if s_type is None:
            cur_batch = base_batch
        elif s_type == "awgn":
            cur_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_batch.items()}
            cur_batch["wave"] = add_awgn_noise(base_batch["wave"], snr_db=s_param, seed=seed)
        elif s_type == "pink":
            cur_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_batch.items()}
            cur_batch["wave"] = add_pink_noise(base_batch["wave"], snr_db=s_param, seed=seed)
        elif s_type == "speed":
            cur_batch = perturb_sound_speed(base_batch, pct_perturbation=s_param)
        else:
            cur_batch = base_batch

        # 评估两模型
        m_tg = evaluate_batch_with_model(tg_model, cur_batch, device)
        m_dis = evaluate_batch_with_model(dis_model, cur_batch, device)

        audit_results["tg_deeponet"][s_key] = {"scenario": s_name, **m_tg}
        audit_results["tg_dis_deeponet"][s_key] = {"scenario": s_name, **m_dis}

        print(f"  [{s_name:<32}]")
        print(f"    - TG-DeepONet     : alpha MAE={m_tg['alpha_mae']:.4f}, R2={m_tg['alpha_r2']:.4f}, W1={m_tg['w1_mean_m']:.2f}m")
        print(f"    - TG-DIS-DeepONet : alpha MAE={m_dis['alpha_mae']:.4f}, R2={m_dis['alpha_r2']:.4f}, W1={m_dis['w1_mean_m']:.2f}m")

    # 4. 计算 20dB 强噪声核心退化率
    clean_dis = audit_results["tg_dis_deeponet"]["clean"]
    awgn_20_dis = audit_results["tg_dis_deeponet"]["awgn_20db"]
    pink_20_dis = audit_results["tg_dis_deeponet"]["pink_20db"]

    def calc_deg(clean_val: float, noisy_val: float, higher_is_better: bool = False) -> float:
        if higher_is_better:
            return float(max(0.0, (clean_val - noisy_val) / (abs(clean_val) + 1e-8) * 100.0))
        return float(max(0.0, (noisy_val - clean_val) / (abs(clean_val) + 1e-8) * 100.0))

    deg_awgn20_mae = calc_deg(clean_dis["alpha_mae"], awgn_20_dis["alpha_mae"])
    deg_awgn20_r2 = calc_deg(clean_dis["alpha_r2"], awgn_20_dis["alpha_r2"], higher_is_better=True)
    deg_awgn20_w1 = calc_deg(clean_dis["w1_mean_m"], awgn_20_dis["w1_mean_m"])

    deg_pink20_mae = calc_deg(clean_dis["alpha_mae"], pink_20_dis["alpha_mae"])
    deg_pink20_r2 = calc_deg(clean_dis["alpha_r2"], pink_20_dis["alpha_r2"], higher_is_better=True)
    deg_pink20_w1 = calc_deg(clean_dis["w1_mean_m"], pink_20_dis["w1_mean_m"])

    # 核心指标在 20dB 强噪声下的最大退化率
    max_deg_20db = max(deg_awgn20_mae, deg_pink20_mae, deg_awgn20_w1, deg_pink20_w1)

    audit_results["degradation_analysis"] = {
        "awgn_20db_mae_degradation_pct": deg_awgn20_mae,
        "awgn_20db_r2_degradation_pct": deg_awgn20_r2,
        "awgn_20db_w1_degradation_pct": deg_awgn20_w1,
        "pink_20db_mae_degradation_pct": deg_pink20_mae,
        "pink_20db_r2_degradation_pct": deg_pink20_r2,
        "pink_20db_w1_degradation_pct": deg_pink20_w1,
        "max_degradation_20db_pct": max_deg_20db,
        "target_threshold_pct": 15.0,
        "passed_20db_audit": max_deg_20db < 15.0,
    }

    # 5. 持久化输出 JSON
    json_path = os.path.join(output_dir, "phase3_noise_robustness_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 噪声鲁棒性审计报告已持久化至: {json_path}")

    # 6. 控制台汇总表格
    print(f"\n{'='*95}")
    print("PaperC Phase 3 噪声与摄动应力对标对比表 (TG-DIS-DeepONet vs TG-DeepONet):")
    print(f"{'='*95}")
    print(f"{'工况扰动场景':<32} | {'DIS alpha MAE':<13} | {'TG alpha MAE':<12} | {'DIS W1 (m)':<10} | {'TG W1 (m)':<9}")
    print(f"{'-'*95}")
    for s_key, s_name, _, _ in scenarios:
        dis_m = audit_results["tg_dis_deeponet"][s_key]
        tg_m = audit_results["tg_deeponet"][s_key]
        print(
            f"{s_name:<32} | {dis_m['alpha_mae']:<13.4f} | {tg_m['alpha_mae']:<12.4f} | "
            f"{dis_m['w1_mean_m']:<10.2f} | {tg_m['w1_mean_m']:<9.2f}"
        )
    print(f"{'='*95}")
    print(f"[*] 20dB 强噪声退化审计结果:")
    print(f"    - AWGN 20dB alpha MAE 退化率: {deg_awgn20_mae:.2f}% (门槛: < 15.0%)")
    print(f"    - Pink 20dB alpha MAE 退化率: {deg_pink20_mae:.2f}% (门槛: < 15.0%)")
    print(f"    - AWGN 20dB W1 空间退化率  : {deg_awgn20_w1:.2f}% (门槛: < 15.0%)")
    print(f"    - Pink 20dB W1 空间退化率  : {deg_pink20_w1:.2f}% (门槛: < 15.0%)")
    status_tag = "[PASS] 验收通过" if max_deg_20db < 15.0 else "[WARN] 未达标"
    print(f"[*] 20dB 综合退化核验结论: {status_tag} (最大退化率: {max_deg_20db:.2f}%)")
    print(f"{'='*95}\n")

    return audit_results


def main():
    parser = argparse.ArgumentParser(description="PaperC Phase 3 噪声与物性扰动鲁棒性审计")
    parser.add_argument("--weights_dir", type=str, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_robustness_audit(
        weights_dir=args.weights_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
