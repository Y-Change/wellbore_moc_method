# -*- coding: utf-8 -*-
"""
moc_simulate/run_lhs_batch_simulate.py — 水力压裂井后进液诊断：多进程拉丁超立方批量仿真脚本

核心特性：
1. 拉丁超立方采样 (LHS)：科学均匀覆盖各压裂簇的集总柔度(Cf)、分布滤失(kleak)、裂缝位置与条数。
2. 物理截断加速：默认将 tf 从 100s 缩短至 30s，聚焦进液诊断核心混响波头，仿真提速 70%。
3. 多进程并发：针对 i5-12600KF(16线程) 优化，默认开启 14 个并发 worker，秒级生成样本。
4. AI4S 友好数据结构：输出标准 .npz (零依赖快速加载) 与总结 CSV/JSON，直接对接 PyTorch 神经算子与扩散去噪器。

运行方式：
    # 快速测试跑 10 组
    python moc_simulate/run_lhs_batch_simulate.py --n-samples 10 --workers 4
    
    # 生产模式跑 1,500 组 (推荐 14 并发，耗时约 40~50 分钟)
    python moc_simulate/run_lhs_batch_simulate.py --n-samples 1500 --workers 14 --tf 30.0 --friction brunone
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time as time_module
from typing import Dict, List, Tuple, Any

if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 确保能导入项目根目录模块
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

import numpy as np

try:
    from scipy.stats import qmc
    HAS_SCIPY_QMC = True
except ImportError:
    HAS_SCIPY_QMC = False

from moc_simulate.wellbore_moc import MocConfig, simulate_wellbore, G
from moc_simulate.config import WELL_CONFIG, FRACTURE_CONFIG, FRICTION_PARAMS
from moc_simulate.lhs_config import SIM_CONFIG, LHS_PARAM_RANGES, LHS_BATCH_CONFIG


def generate_lhs_params(n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """生成 n_samples 组独立的多缝物理参数组合"""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # 确定每条样本的裂缝条数 N_cl
    n_min = LHS_PARAM_RANGES["n_frac_min"]
    n_max = LHS_PARAM_RANGES["n_frac_max"]
    frac_counts = rng.integers(n_min, n_max + 1, size=n_samples)
    
    # 为保证所有样本最大维数一致以便于 LHS，我们以最大条数 n_max 分配采样空间
    # 每个裂缝需要 3 个参数：位置比例、log(Cf)、log(kleak) -> 共 3 * n_max 维
    dim = 3 * n_max
    
    if HAS_SCIPY_QMC:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        lhs_raw = sampler.random(n=n_samples)
    else:
        print("[警告] 未检测到 scipy.stats.qmc，退化为标准均匀伪随机采样。")
        lhs_raw = rng.uniform(0, 1, size=(n_samples, dim))
        
    samples = []
    z_start = LHS_PARAM_RANGES["frac_zone_start"]
    z_end = LHS_PARAM_RANGES["frac_zone_end"]
    min_sp = float(LHS_PARAM_RANGES["min_spacing"])
    max_sp = float(LHS_PARAM_RANGES["max_spacing"])
    if max_sp < min_sp:
        raise ValueError(f"max_spacing ({max_sp}) must be >= min_spacing ({min_sp})")
    
    cf_lmin, cf_lmax = LHS_PARAM_RANGES["cf_log_min"], LHS_PARAM_RANGES["cf_log_max"]
    kl_lmin, kl_lmax = LHS_PARAM_RANGES["kleak_log_min"], LHS_PARAM_RANGES["kleak_log_max"]
    
    for i in range(n_samples):
        n_cl = int(frac_counts[i])
        row = lhs_raw[i]
        
        # 1. 相邻簇间距约束在 [min_spacing, max_spacing]，整条缝链落入缝网区间
        if n_cl == 1:
            positions = [float(z_start + float(row[0]) * (z_end - z_start))]
        else:
            spacings = [
                min_sp + float(row[k + 1]) * (max_sp - min_sp) for k in range(n_cl - 1)
            ]
            chain_len = float(sum(spacings))
            start_span = (z_end - z_start) - chain_len
            if start_span < 0:
                raise ValueError(
                    f"n_frac={n_cl} with spacing [{min_sp}, {max_sp}] m "
                    f"cannot fit in zone [{z_start}, {z_end}] m"
                )
            x0 = float(z_start + float(row[0]) * start_span)
            positions = [x0]
            for gap in spacings:
                positions.append(float(positions[-1] + gap))
        
        # 2. 生成对数均匀分布的 Cf 和 kleak
        cf_vals = [float(10.0 ** (cf_lmin + row[n_max + k] * (cf_lmax - cf_lmin))) for k in range(n_cl)]
        kleak_vals = [float(10.0 ** (kl_lmin + row[2 * n_max + k] * (kl_lmax - kl_lmin))) for k in range(n_cl)]
        
        samples.append({
            "case_id": i,
            "n_frac": n_cl,
            "positions": positions,
            "Cf_list": cf_vals,
            "kleak_list": kleak_vals,
        })
    return samples


def _worker_simulate(args: Tuple[Dict[str, Any], str, float, float, str]) -> Dict[str, Any]:
    """多进程单个 Worker 执行函数：调用 simulate_wellbore 跑单条真解并落盘"""
    sample, friction_model, tf_cut, dt_sim, out_data_dir = args
    case_id = sample["case_id"]
    n_cl = sample["n_frac"]
    positions = sample["positions"]
    Cf_list = sample["Cf_list"]
    kleak_list = sample["kleak_list"]
    
    t0 = time_module.time()
    try:
        # 构造配置
        w = WELL_CONFIG
        s = SIM_CONFIG
        fc = FRACTURE_CONFIG
        
        cfg = MocConfig(
            wellbore_length=w['L'],
            wellbore_diameter=w['wellbore_diameter'],
            fluid_density=w['fluid_density'],
            fluid_viscosity=w['fluid_viscosity'],
            wavespeed=w['wavespeed'],
            roughness_height=w['roughness_height'],
            friction_model=friction_model,
            dt=dt_sim,
            tf=tf_cut,             # 物理截断提速
            wellhead_bc='velocity_step',
            pump_shut_time=s['ts'],
            initial_velocity=w['V0'],
            initial_head=w['H0'],
            theta=w['theta'],
            toe_bc='reservoir',
            toe_head=w['H0'],
        )
        
        res = simulate_wellbore(
            cfg,
            fracture_positions=positions,
            fracture_Cf=Cf_list,
            fracture_kleak=kleak_list,
            H_ext=fc['H_ext'],
            store_full_field=False,
        )
        
        elapsed = time_module.time() - t0
        
        # 提取关键时序 (AI 反演特征)
        t_arr = res["timestamps"]
        H_wh = res["wellhead_head"]
        V_wh = res["wellhead_velocity"]
        Q_wh = V_wh * cfg.area
        
        # 保存为 .npz
        npz_filename = f"case_{case_id:05d}.npz"
        npz_path = os.path.join(out_data_dir, npz_filename)
        np.savez_compressed(
            npz_path,
            t=t_arr,
            H_wh=H_wh,
            Q_wh=Q_wh,
            x_f=np.array(positions, dtype=np.float32),
            Cf=np.array(Cf_list, dtype=np.float32),
            kleak=np.array(kleak_list, dtype=np.float32),
            n_frac=n_cl,
            friction=str(friction_model),
            tf=float(tf_cut),
        )
        
        return {
            "case_id": case_id,
            "status": "PASS",
            "n_frac": n_cl,
            "positions_str": ";".join([f"{x:.1f}" for x in positions]),
            "Cf_str": ";".join([f"{c:.2e}" for c in Cf_list]),
            "kleak_str": ";".join([f"{k:.2e}" for k in kleak_list]),
            "elapsed_s": round(elapsed, 2),
            "npz_file": npz_filename,
            "error": "",
        }
    except Exception as e:
        return {
            "case_id": case_id,
            "status": "FAIL",
            "n_frac": n_cl,
            "positions_str": "",
            "Cf_str": "",
            "kleak_str": "",
            "elapsed_s": round(time_module.time() - t0, 2),
            "npz_file": "",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="MOC 多进程拉丁超立方采样批量仿真 (进液反演黄金数据集)")
    parser.add_argument("--n-samples", type=int, default=10, help=f"采样样本总数 (默认测试 10 组，推荐 {LHS_BATCH_CONFIG['default_n_samples']} 组)")
    parser.add_argument("--workers", type=int, default=LHS_BATCH_CONFIG["default_workers"], help="并发 worker 进程数 (针对 16 线程 CPU 默认设 14)")
    parser.add_argument("--tf", type=float, default=SIM_CONFIG["tf"], help=f"单 case 仿真截断时长 [s] (默认 {SIM_CONFIG['tf']}s)")
    parser.add_argument("--dt", type=float, default=SIM_CONFIG["dt"], help="仿真时间步长 [s] (默认 1ms)")
    parser.add_argument("--friction", type=str, default=LHS_BATCH_CONFIG["default_friction"], choices=["steady", "brunone"], help="摩阻模型")
    parser.add_argument("--out-dir", type=str, default=LHS_BATCH_CONFIG["output_dir"], help="数据输出根目录")
    parser.add_argument("--seed", type=int, default=LHS_BATCH_CONFIG["seed"], help="随机数种子")
    args = parser.parse_args()

    out_root = os.path.abspath(args.out_dir)
    out_data_dir = os.path.join(out_root, "data")
    os.makedirs(out_data_dir, exist_ok=True)
    
    print("=" * 72)
    print("水力压裂进液反演 —— MOC 拉丁超立方并行批量生成器")
    print("=" * 72)
    print(f"  样本总数 : {args.n_samples} 组")
    print(f"  并发进程 : {args.workers} Workers")
    print(f"  仿真时长 : tf = {args.tf} s (物理截断加速)")
    print(f"  摩阻模型 : {args.friction}")
    print(f"  输出路径 : {out_root}")
    print("=" * 72)
    
    # 1. 生成采样矩阵
    print("\n[Step 1] 正在生成拉丁超立方 (LHS) 物理参数矩阵...")
    samples = generate_lhs_params(args.n_samples, seed=args.seed)
    print(f"  成功生成 {len(samples)} 组参数矩阵。开始派发多进程计算任务...\n")
    
    # 2. 并发计算
    worker_args = [(samp, args.friction, args.tf, args.dt, out_data_dir) for samp in samples]
    t_start_all = time_module.time()
    
    results = []
    pass_count = 0
    fail_count = 0
    
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_simulate, worker_args)):
            results.append(res)
            cid = res["case_id"]
            if res["status"] == "PASS":
                pass_count += 1
                status_icon = "[PASS]"
            else:
                fail_count += 1
                status_icon = "[FAIL]"
            
            # 简单进度条汇报
            if (i + 1) % max(1, args.n_samples // 20) == 0 or (i + 1) == args.n_samples:
                pct = (i + 1) / args.n_samples * 100
                elapsed_all = time_module.time() - t_start_all
                spd = (i + 1) / elapsed_all
                eta = (args.n_samples - (i + 1)) / spd if spd > 0 else 0
                print(f"  [{pct:5.1f}%] 已跑完 {i+1:4d}/{args.n_samples} 组 | "
                      f"成功: {pass_count} 失败: {fail_count} | "
                      f"速度: {spd:.2f} case/s | 预计剩余: {eta/60:.1f} min | 最新: Case {cid:05d} ({res['elapsed_s']}s) {status_icon}")

    total_time = time_module.time() - t_start_all
    print("\n" + "=" * 72)
    print(f"批量仿真完成！总耗时: {total_time/60:.2f} min ({total_time:.1f} s)")
    print(f"成功: {pass_count} 组 | 失败: {fail_count} 组 | 平均速度: {args.n_samples/total_time:.2f} case/s")
    print("=" * 72)

    # 3. 汇总索引落盘 (CSV + JSON) -> PyTorch Dataset 极速索引
    results.sort(key=lambda x: x["case_id"])
    
    csv_path = os.path.join(out_root, "lhs_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "status", "n_frac", "positions_str", "Cf_str", "kleak_str", "elapsed_s", "npz_file", "error"
        ])
        writer.writeheader()
        writer.writerows(results)
        
    json_path = os.path.join(out_root, "lhs_metadata.json")
    meta_info = {
        "created_time": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples_total": args.n_samples,
        "n_pass": pass_count,
        "n_fail": fail_count,
        "tf_cutoff_s": args.tf,
        "dt_s": args.dt,
        "friction_model": args.friction,
        "workers": args.workers,
        "total_elapsed_min": round(total_time / 60, 2),
        "param_ranges": LHS_PARAM_RANGES,
        "samples_index": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
        
    print(f"\n[落盘验证] 总结报告已保存:")
    print(f"  - 索引表格 : {csv_path}")
    print(f"  - 完整元数据: {json_path}")
    print(f"  - NPZ 数据集: {out_data_dir}/ (共 {pass_count} 个独立波形文件)\n")


if __name__ == "__main__":
    main()
