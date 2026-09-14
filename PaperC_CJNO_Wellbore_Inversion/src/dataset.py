# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.dataset

水击波物理反演小样本预研数据集与数据管线 (PilotInversionDataset):
1. 加载 1k 案例物理数据集 (moc_v2_1k_dataset.h5)，支持 800 train / 100 val / 100 test 确定性切分；
2. 全局等距降采样时域波形至 N_time=4096 点，并提取一阶波前差分通道；
3. 在线/缓存提取 1D 空间深度倒谱特征并重采样至 N_ceps=1024 均匀空间网格；
4. 构造真值高斯连续流体进入贡献密度场 m_alpha(x) (N_grid=500) 用于 Wasserstein-1D 距离监督；
5. 封装支持变簇数 (Nc in [1..6]) 的掩码 (mask) 与批处理 collate 机制；
6. 自动序列化缓存至本地 .npz 文件，实现毫秒级快速加载与训练。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from moc_simulate.v2.batch.torch_dataset import MocWellboreDataset, split_dataset_indices
from moc_simulate.v2.signal.cepstrum_1d import compute_cepstrum_1d

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5_PATH = str(_PROJECT_ROOT / "data" / "datasets" / "moc_v2_1k_dataset.h5")
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[1] / "data")


class PilotInversionDataset(Dataset):
    """
    PaperC 物理反演预研专用数据集加载器。
    基于 MocWellboreDataset 原生载入并在内存中进行降采样、倒谱对齐与变长批处理封装。
    """

    MAX_CLUSTERS: int = 6
    L_DEFAULT: float = 5000.0
    CF_REF: float = 0.01  # 基准顺应性 Cf0 = 0.01 m^2

    def __init__(
        self,
        h5_path: str = DEFAULT_H5_PATH,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        n_time: int = 4096,
        n_ceps: int = 1024,
        n_grid: int = 500,
        sigma_m: float = 20.0,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        force_recompute: bool = False,
        load_raw_wave: bool = True,
    ):
        super().__init__()
        self.load_raw_wave = bool(load_raw_wave)
        self.h5_path = os.path.abspath(h5_path)
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"找不到指定的 HDF5 数据集文件: {self.h5_path}")

        self.split = str(split).strip().lower()
        self.split_ratios = split_ratios
        self.seed = seed
        self.n_time = int(n_time)
        self.n_ceps = int(n_ceps)
        self.n_grid = int(n_grid)
        self.sigma_m = float(sigma_m)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None

        # 空间查询网格 x in [0, L]
        self.grid_x = np.linspace(0, self.L_DEFAULT, self.n_grid, dtype=np.float32)
        self.dx_grid = self.L_DEFAULT / (self.n_grid - 1)

        # 检查/构建全量缓存
        all_data = self._load_or_build_cache(force_recompute=force_recompute)

        # 数据集索引划分
        n_total = len(all_data["n_frac"])
        splits = split_dataset_indices(n_total, self.split_ratios, self.seed)
        if self.split not in splits:
            raise ValueError(f"无效的 split: {self.split}, 可选: {list(splits.keys())}")
        self.indices = splits[self.split]
        self.length = len(self.indices)

        # 提取当前 split 的张量数组并转为 torch.Tensor
        idx = self.indices
        self.waveforms = torch.from_numpy(all_data["waveforms"][idx]).float()        # (N, 2, n_time)
        self.cepstrums = torch.from_numpy(all_data["cepstrums"][idx]).float()        # (N, 1, n_ceps)
        self.conds = torch.from_numpy(all_data["conds"][idx]).float()                # (N, 3)
        self.n_frac = torch.from_numpy(all_data["n_frac"][idx]).long()               # (N,)
        self.positions = torch.from_numpy(all_data["positions"][idx]).float()        # (N, max_nc) [m]
        self.norm_positions = torch.from_numpy(all_data["norm_positions"][idx]).float()  # (N, max_nc) [0, 1]
        self.masks = torch.from_numpy(all_data["masks"][idx]).bool()                 # (N, max_nc)
        self.alphas = torch.from_numpy(all_data["alphas"][idx]).float()              # (N, max_nc)
        self.cf = torch.from_numpy(all_data["cf"][idx]).float()                      # (N, max_nc)
        self.log_cf = torch.from_numpy(all_data["log_cf"][idx]).float()              # (N, max_nc)
        self.m_alpha_grid = torch.from_numpy(all_data["m_alpha_grid"][idx]).float()  # (N, n_grid)
        self.raw_indices = idx

        # 加载原生 1000 Hz 全速率波形 (60,001 点, dt=1.0ms, 消除离线降采样信息瓶颈)
        if self.load_raw_wave:
            raw_wave_arr = self._load_or_build_raw_wave_cache(force_recompute=force_recompute)
            self.raw_waveforms = torch.from_numpy(raw_wave_arr[idx]).float()  # (N, 2, 60001)
        else:
            self.raw_waveforms = None

    def _load_or_build_raw_wave_cache(self, force_recompute: bool = False) -> np.ndarray:
        """加载或构建 60,001 点原生 1000 Hz 波形缓存 (2, 60001)"""
        raw_cache_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            raw_cache_file = os.path.join(self.cache_dir, "cache_raw_1k_t60001.npz")

        if raw_cache_file and os.path.exists(raw_cache_file) and not force_recompute:
            data = np.load(raw_cache_file)
            return data["raw_waveforms"]

        # 直接从 H5 读取并标准化
        with h5py.File(self.h5_path, "r") as f:
            raw_heads = f["waveforms/wellhead_head"][:]  # (1000, 60001)

        h_mean = raw_heads.mean(axis=1, keepdims=True)
        h_std = raw_heads.std(axis=1, keepdims=True) + 1e-6
        h_norm = (raw_heads - h_mean) / h_std
        dh = np.diff(h_norm, axis=1, prepend=h_norm[:, :1])
        dh_std = dh.std(axis=1, keepdims=True) + 1e-6
        dh_norm = dh / dh_std
        raw_wave_arr = np.stack([h_norm, dh_norm], axis=1).astype(np.float32)

        if raw_cache_file:
            np.savez_compressed(raw_cache_file, raw_waveforms=raw_wave_arr)
        return raw_wave_arr

    def _load_or_build_cache(self, force_recompute: bool = False) -> Dict[str, np.ndarray]:
        """加载或重新生成预研特征缓存 .npz 文件"""
        cache_file = None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(
                self.cache_dir,
                f"cache_1k_t{self.n_time}_c{self.n_ceps}_g{self.n_grid}.npz"
            )

        if cache_file and os.path.exists(cache_file) and not force_recompute:
            data = np.load(cache_file)
            return {k: data[k] for k in data.files}

        # 缓存不存在或强制重新计算
        data_dict = self._process_all_from_h5()
        if cache_file:
            np.savez_compressed(cache_file, **data_dict)
        return data_dict

    def _process_all_from_h5(self) -> Dict[str, np.ndarray]:
        """使用 MocWellboreDataset (in_memory=True) 提取并预处理 1,000 个案例"""
        base_ds = MocWellboreDataset(
            h5_path=self.h5_path,
            split="all",
            split_ratios=self.split_ratios,
            seed=self.seed,
            in_memory=True,
            return_velocity=False,
        )
        self.base_dataset = base_ds

        raw_heads = base_ds.mem_head
        raw_times = base_ds.timestamps
        n_samples = len(raw_heads)
        n_raw_time = raw_heads.shape[1]

        n_frac_arr = base_ds.mem_n_frac
        pos_arr = base_ds.mem_positions
        cf_arr = base_ds.mem_Cf
        weights_arr = base_ds.mem_weights
        tc_arr = base_ds.mem_tc
        wavespeed_arr = base_ds.mem_wavespeed
        h_ext_arr = base_ds.mem_H_ext

        # 1. 降采样索引 (4096 点)
        sample_time_indices = np.linspace(0, n_raw_time - 1, self.n_time).astype(np.int64)

        # 2. 空间倒谱目标网格 (1024 点, 0~5000m)
        ceps_x_grid = np.linspace(0, self.L_DEFAULT, self.n_ceps, dtype=np.float32)

        # 准备输出容器
        waveforms = np.zeros((n_samples, 2, self.n_time), dtype=np.float32)
        cepstrums = np.zeros((n_samples, 1, self.n_ceps), dtype=np.float32)
        conds = np.zeros((n_samples, 3), dtype=np.float32)
        positions = np.zeros((n_samples, self.MAX_CLUSTERS), dtype=np.float32)
        norm_positions = np.zeros((n_samples, self.MAX_CLUSTERS), dtype=np.float32)
        masks = np.zeros((n_samples, self.MAX_CLUSTERS), dtype=bool)
        alphas = np.zeros((n_samples, self.MAX_CLUSTERS), dtype=np.float32)
        cf_out = np.zeros((n_samples, self.MAX_CLUSTERS), dtype=np.float32)
        log_cf_out = np.zeros((n_samples, self.MAX_CLUSTERS), dtype=np.float32)
        m_alpha_grid = np.zeros((n_samples, self.n_grid), dtype=np.float32)

        # 空间密度网格 x_grid
        grid_x = self.grid_x
        inv_sqrt_2pi_sigma = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigma_m)
        two_sigma_sq = 2.0 * (self.sigma_m ** 2)

        for i in range(n_samples):
            # A. 时域波形降采样与标准化
            h_raw = raw_heads[i]
            h_sub = h_raw[sample_time_indices]
            # 一阶波前差分
            dh_sub = np.diff(h_sub, prepend=h_sub[0]).astype(np.float32)

            # Per-sample Z-score 标准化
            h_mean, h_std = float(np.mean(h_sub)), float(np.std(h_sub) + 1e-6)
            dh_std = float(np.std(dh_sub) + 1e-6)

            waveforms[i, 0, :] = (h_sub - h_mean) / h_std
            waveforms[i, 1, :] = dh_sub / dh_std

            # B. 在线 1D 倒谱计算与空间网格插值
            a_val = float(wavespeed_arr[i])
            ceps_res = compute_cepstrum_1d(
                time=raw_times,
                head=h_raw,
                wavespeed=a_val,
                ts=1.0,
                max_distance=self.L_DEFAULT,
            )
            c_dist = ceps_res["distance"]
            c_vals = ceps_res["cepstrum"]
            # 线性插值对齐到固定空间网格
            ceps_interp = np.interp(ceps_x_grid, c_dist, c_vals).astype(np.float32)
            # 倒谱标准化 (去均值/方差)
            c_std = float(np.std(ceps_interp) + 1e-6)
            cepstrums[i, 0, :] = (ceps_interp - float(np.mean(ceps_interp))) / c_std

            # C. 工况条件归一化
            # tc: [0.001, 0.100] -> (tc - 0.05) / 0.03
            # a:  [1420, 1480]   -> (a - 1450) / 20.0
            # H:  [80, 120]      -> (H - 100) / 15.0
            conds[i, 0] = (tc_arr[i] - 0.05) / 0.03
            conds[i, 1] = (a_val - 1450.0) / 20.0
            conds[i, 2] = (h_ext_arr[i] - 100.0) / 15.0

            # D. 变长簇标签与掩码 (Nc in [1..6])
            nc = int(n_frac_arr[i])
            nc = min(nc, self.MAX_CLUSTERS)
            masks[i, :nc] = True

            act_pos = pos_arr[i, :nc]
            act_w = weights_arr[i, :nc]
            act_cf = cf_arr[i, :nc]

            # 确保 alpha 严格和为 1
            if act_w.sum() > 1e-6:
                act_w = act_w / act_w.sum()
            else:
                act_w = np.full(nc, 1.0 / nc, dtype=np.float32)

            positions[i, :nc] = act_pos
            norm_positions[i, :nc] = act_pos / self.L_DEFAULT
            alphas[i, :nc] = act_w
            cf_out[i, :nc] = act_cf
            log_cf_out[i, :nc] = np.log(np.maximum(act_cf, 1e-7) / self.CF_REF)

            # E. 构造真值高斯连续流体进入密度场 m_alpha(x)
            # m_alpha(x) = sum_j alpha_j * N(x; x_j, sigma)
            field = np.zeros(self.n_grid, dtype=np.float32)
            for j in range(nc):
                xj = act_pos[j]
                wj = act_w[j]
                gauss = inv_sqrt_2pi_sigma * np.exp(-((grid_x - xj) ** 2) / two_sigma_sq)
                field += wj * gauss

            # 归一化连续场积分积为 1.0 (Riemann sum)
            field_sum = np.sum(field) * self.dx_grid
            if field_sum > 1e-8:
                field = field / field_sum
            m_alpha_grid[i] = field

        return {
            "waveforms": waveforms,
            "cepstrums": cepstrums,
            "conds": conds,
            "n_frac": n_frac_arr,
            "positions": positions,
            "norm_positions": norm_positions,
            "masks": masks,
            "alphas": alphas,
            "cf": cf_out,
            "log_cf": log_cf_out,
            "m_alpha_grid": m_alpha_grid,
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "wave": self.waveforms[idx],               # (2, 4096)
            "cepstrum": self.cepstrums[idx],           # (1, 1024)
            "cond": self.conds[idx],                   # (3,)
            "n_frac": self.n_frac[idx],                # ()
            "positions": self.positions[idx],          # (6,) [m]
            "norm_positions": self.norm_positions[idx],# (6,) [0, 1]
            "mask": self.masks[idx],                   # (6,) bool
            "alpha": self.alphas[idx],                 # (6,)
            "cf": self.cf[idx],                        # (6,)
            "log_cf": self.log_cf[idx],                # (6,)
            "m_alpha_grid": self.m_alpha_grid[idx],    # (500,)
        }
        if self.raw_waveforms is not None:
            item["raw_wave"] = self.raw_waveforms[idx] # (2, 60001)
        return item

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
