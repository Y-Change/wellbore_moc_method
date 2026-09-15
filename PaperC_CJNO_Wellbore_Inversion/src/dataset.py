# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.dataset

水击波物理反演小样本预研数据集与数据管线 (PilotInversionDataset):
1. 加载物理正向稳态 HDF5（默认 moc_v2_pilot_steady_100.h5），按源文件样本数做确定性切分；
2. 全局等距降采样时域波形至 N_time=4096 点，并提取一阶波前差分通道；
3. 在线/缓存提取 1D 空间深度倒谱特征并重采样至 N_ceps=1024 均匀空间网格；
4. 构造真值高斯连续流体进入贡献密度场 m_alpha(x) (N_grid=500) 用于 Wasserstein-1D 距离监督；
5. 封装支持变簇数 (Nc in [1..6]) 的掩码 (mask) 与批处理 collate 机制；
6. 缓存文件名绑定 HDF5 路径、内容指纹与样本数，禁止跨数据源复用旧 npz。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from moc_simulate.v2.batch.torch_dataset import MocWellboreDataset, split_dataset_indices
from moc_simulate.v2.signal.cepstrum_1d import compute_cepstrum_1d

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5_PATH = str(_PROJECT_ROOT / "data" / "datasets" / "moc_v2_pilot_steady_100.h5")
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parents[1] / "data")
_H5_FALLBACKS = (
    str(_PROJECT_ROOT / "data" / "datasets" / "moc_v2_pilot_steady_100.h5"),
    str(_PROJECT_ROOT / "data" / "datasets" / "moc_v2_1k_dataset.h5"),
    str(_PROJECT_ROOT / "data" / "datasets" / "moc_v2_1k_dataset_legacy_nonsteady_v1.h5"),
)


def fingerprint_h5_source(h5_path: str) -> Dict[str, Any]:
    """
    生成与数据源绑定的缓存指纹。
    包含绝对路径、文件大小、mtime、HDF5 样本数/时域长度，以及文件头尾内容哈希，
    避免切换 HDF5 后仍命中旧的 cache_1k_*.npz。
    """
    abs_path = os.path.abspath(h5_path)
    st = os.stat(abs_path)
    size = int(st.st_size)
    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    with h5py.File(abs_path, "r") as f:
        head_ds = f["waveforms/wellhead_head"]
        n_samples = int(head_ds.shape[0])
        n_time_raw = int(head_ds.shape[1])
    with open(abs_path, "rb") as fh:
        head_bytes = fh.read(65536)
        if size > 65536:
            fh.seek(max(0, size - 65536))
            tail_bytes = fh.read(65536)
        else:
            tail_bytes = b""
    content_token = hashlib.sha256(head_bytes + tail_bytes).hexdigest()[:16]
    payload = "|".join(
        [
            abs_path.replace("\\", "/"),
            str(size),
            str(mtime_ns),
            str(n_samples),
            str(n_time_raw),
            content_token,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {
        "path": abs_path,
        "digest": digest,
        "n_samples": n_samples,
        "n_time_raw": n_time_raw,
        "size": size,
    }


def feature_cache_filename(
    cache_dir: str,
    n_time: int,
    n_ceps: int,
    n_grid: int,
    digest: str,
    n_samples: int,
) -> str:
    return os.path.join(
        cache_dir,
        f"cache_t{int(n_time)}_c{int(n_ceps)}_g{int(n_grid)}_{digest}_n{int(n_samples)}.npz",
    )


def raw_wave_cache_filename(cache_dir: str, digest: str, n_samples: int, n_time_raw: int) -> str:
    return os.path.join(
        cache_dir,
        f"cache_raw_t{int(n_time_raw)}_{digest}_n{int(n_samples)}.npz",
    )


def _resolve_h5_path(h5_path: str) -> str:
    requested = os.path.abspath(h5_path)
    if os.path.exists(requested):
        return requested
    default_abs = os.path.abspath(DEFAULT_H5_PATH)
    if requested != default_abs:
        raise FileNotFoundError(f"找不到指定的 HDF5 数据集文件: {requested}")
    for alt in _H5_FALLBACKS:
        if os.path.exists(alt):
            return os.path.abspath(alt)
    raise FileNotFoundError(f"找不到指定的 HDF5 数据集文件: {requested}")


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
        self.h5_path = _resolve_h5_path(h5_path)
        self.source_meta = fingerprint_h5_source(self.h5_path)
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
        if n_total != int(self.source_meta["n_samples"]):
            raise RuntimeError(
                f"特征缓存样本数 {n_total} 与 HDF5 源 {self.h5_path} 的 "
                f"{self.source_meta['n_samples']} 例不一致，已禁止使用污染缓存。"
            )
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

    def _feature_cache_path(self) -> Optional[str]:
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return feature_cache_filename(
            self.cache_dir,
            self.n_time,
            self.n_ceps,
            self.n_grid,
            self.source_meta["digest"],
            self.source_meta["n_samples"],
        )

    def _raw_cache_path(self) -> Optional[str]:
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return raw_wave_cache_filename(
            self.cache_dir,
            self.source_meta["digest"],
            self.source_meta["n_samples"],
            self.source_meta["n_time_raw"],
        )

    def _cache_matches_source(self, n_cached: int, digest: Optional[str] = None) -> bool:
        if int(n_cached) != int(self.source_meta["n_samples"]):
            return False
        if digest is not None and str(digest) != str(self.source_meta["digest"]):
            return False
        return True

    def _load_or_build_raw_wave_cache(self, force_recompute: bool = False) -> np.ndarray:
        """加载或构建与当前 HDF5 指纹绑定的原生波形缓存"""
        raw_cache_file = self._raw_cache_path()

        if raw_cache_file and os.path.exists(raw_cache_file) and not force_recompute:
            data = np.load(raw_cache_file)
            raw_wave_arr = data["raw_waveforms"]
            cached_digest = str(data["source_digest"]) if "source_digest" in data.files else None
            if self._cache_matches_source(raw_wave_arr.shape[0], cached_digest):
                return raw_wave_arr

        with h5py.File(self.h5_path, "r") as f:
            raw_heads = f["waveforms/wellhead_head"][:]

        h_mean = raw_heads.mean(axis=1, keepdims=True)
        h_std = raw_heads.std(axis=1, keepdims=True) + 1e-6
        h_norm = (raw_heads - h_mean) / h_std
        dh = np.diff(h_norm, axis=1, prepend=h_norm[:, :1])
        dh_std = dh.std(axis=1, keepdims=True) + 1e-6
        dh_norm = dh / dh_std
        raw_wave_arr = np.stack([h_norm, dh_norm], axis=1).astype(np.float32)

        if raw_cache_file:
            np.savez_compressed(
                raw_cache_file,
                raw_waveforms=raw_wave_arr,
                source_digest=np.array(self.source_meta["digest"]),
                source_n=np.array(self.source_meta["n_samples"]),
            )
        return raw_wave_arr

    def _load_or_build_cache(self, force_recompute: bool = False) -> Dict[str, np.ndarray]:
        """加载或重新生成与当前 HDF5 指纹绑定的预研特征缓存 .npz"""
        cache_file = self._feature_cache_path()

        if cache_file and os.path.exists(cache_file) and not force_recompute:
            data = np.load(cache_file)
            payload = {k: data[k] for k in data.files if not k.startswith("source_")}
            cached_digest = str(data["source_digest"]) if "source_digest" in data.files else None
            n_cached = int(payload["n_frac"].shape[0]) if "n_frac" in payload else -1
            if self._cache_matches_source(n_cached, cached_digest):
                return payload

        data_dict = self._process_all_from_h5()
        if cache_file:
            np.savez_compressed(
                cache_file,
                **data_dict,
                source_digest=np.array(self.source_meta["digest"]),
                source_n=np.array(self.source_meta["n_samples"]),
            )
        return data_dict

    def _process_all_from_h5(self) -> Dict[str, np.ndarray]:
        """使用 MocWellboreDataset (in_memory=True) 提取并预处理全部案例"""
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
