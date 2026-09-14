# -*- coding: utf-8 -*-
"""
moc_simulate.v2.batch.torch_dataset

面向深度学习智能反演任务的标准化 PyTorch 数据集加载器 (MocWellboreDataset)：
1. 原生继承 torch.utils.data.Dataset，无缝集成 DataLoader、DistributedSampler 与多卡训练；
2. 支持严格自洽的 train / val / test 随机划分 (基于确定性种子)；
3. 支持高效全内存常驻模式 (in_memory=True，微秒级样本吞吐) 与按需切片模式 (in_memory=False)；
4. 内置在线动态预处理算子：一阶波前时域差分、Z-Score / Min-Max 归一化；
5. 支持即时在线 1D 倒谱特征变换 (compute_cepstrum=True)，消除离线预计算的磁盘存储冗余。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

# 兼容 Windows 环境下 PyTorch 与 NumPy/SciPy MKL OpenMP 动态库共存
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from moc_simulate.v2.signal.cepstrum_1d import compute_cepstrum_1d


def split_dataset_indices(
    n_samples: int,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    根据比例划分训练集、验证集与测试集索引。

    参数:
        n_samples: 样本总数
        split_ratios: (train, val, test) 比例三元组，和必须为 1.0
        seed: 随机数种子

    返回:
        {"train": np.ndarray, "val": np.ndarray, "test": np.ndarray, "all": np.ndarray}
    """
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios 之和必须为 1.0，当前为: {sum(split_ratios)}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(round(n_samples * split_ratios[0]))
    n_val = int(round(n_samples * split_ratios[1]))

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "all": np.arange(n_samples, dtype=np.int64),
    }


def _safe_h5_slice(dset: Any, idx: np.ndarray) -> np.ndarray:
    """进程与 h5py 安全的切片读取，自动去重、升序读取并还原原始顺序与多重度"""
    if len(idx) == 0:
        return np.empty((0, *dset.shape[1:]), dtype=dset.dtype)
    unique_idx, inverse = np.unique(idx, return_inverse=True)
    unique_data = dset[unique_idx]
    return unique_data[inverse]


class MocWellboreDataset(Dataset):
    """
    水击瞬变流井口水头时程与多簇地质物理标签数据集。
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        in_memory: bool = True,
        diff_order: int = 0,
        normalize: Union[bool, str] = False,
        compute_cepstrum: bool = False,
        cepstrum_max_distance: float = 5000.0,
        return_channel_dim: bool = False,
        return_velocity: bool = False,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        初始化 MocWellboreDataset。

        参数:
            h5_path: HDF5 数据集文件绝对或相对路径
            split: 数据集划分子集 ('train', 'val', 'test', 'all')，若显式传入 indices 则忽略此项
            split_ratios: 训练/验证/测试比例 (train, val, test)，默认 (0.8, 0.1, 0.1)
            seed: 划分伪随机数种子，保障全生命周期确定性可复现
            in_memory: 是否一次性读入系统内存 (推荐 True，1000 个样本仅 ~230MB，微秒级切片)
            diff_order: 波形时间差分阶数 (0: 原始水头 H_wh(t); 1: 一阶差分; 2: 二阶差分，保持原长对齐)
            normalize: 是否执行标准化 (False, True/"zscore": 零均值单位方差; "minmax": 线性映射至 [0, 1])
            compute_cepstrum: 是否在线即时提取 1D 倒谱特征深度向量
            cepstrum_max_distance: 倒谱最大深度截断距离 [m]
            return_channel_dim: 是否返回 (1, N_time) 通道维度 (适合 1D CNN / U-Net)
            return_velocity: 是否额外加载并返回井口流速时程 (默认 False 以节约 ~230MB 内存)
            indices: 自定义指定读取的样本索引列表 (若非空则优先使用，方便交叉验证与自定义抽样)
            transform: 自定义即时特征增强变换回调函数
        """
        if not HAS_TORCH:
            raise RuntimeError("MocWellboreDataset 需要安装 PyTorch 库 (pip install torch)")
        if not HAS_H5PY:
            raise RuntimeError("MocWellboreDataset 需要安装 h5py 库 (pip install h5py)")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"找不到指定的 HDF5 数据集文件: {h5_path}")

        self.h5_path = os.path.abspath(h5_path)
        self.split = str(split).strip().lower()
        self.split_ratios = split_ratios
        self.seed = seed
        self.in_memory = in_memory
        self.diff_order = int(diff_order)
        self.normalize = normalize
        self.compute_cepstrum = compute_cepstrum
        self.cepstrum_max_distance = float(cepstrum_max_distance)
        self.return_channel_dim = bool(return_channel_dim)
        self.return_velocity = bool(return_velocity)
        self.transform = transform

        # 读取元数据并确立样本总数
        with h5py.File(self.h5_path, "r") as h5:
            self.total_samples = int(h5["waveforms"]["wellhead_head"].shape[0])
            self.n_time = int(h5["waveforms"]["wellhead_head"].shape[1])
            self.timestamps = np.asarray(h5["waveforms"]["timestamps"][:], dtype=np.float32)

            self.metadata: Dict[str, Any] = {}
            for k in h5.attrs:
                val = h5.attrs[k]
                if k == "metadata_json":
                    try:
                        self.metadata = json.loads(val)
                    except Exception:
                        self.metadata[k] = val
                else:
                    self.metadata[k] = val

            self.has_precomputed_cepstrum = "features" in h5 and "cepstrum" in h5["features"]
            if self.has_precomputed_cepstrum:
                self.distances = np.asarray(h5["features"]["distances"][:], dtype=np.float32)
            else:
                self.distances = None

        # 确定当前分面的样本索引集合
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
        else:
            valid_splits = ("train", "val", "test", "all")
            if self.split not in valid_splits:
                raise ValueError(f"无效的 split 选项: {self.split}，必须为 {valid_splits} 之一")
            split_dict = split_dataset_indices(self.total_samples, self.split_ratios, self.seed)
            self.indices = split_dict[self.split]

        self.length = len(self.indices)

        # 惰性文件句柄 (用于多进程/按需磁盘读取，反序列化时自动重置)
        self._h5_file: Optional[h5py.File] = None

        # 若开启常驻内存模式，则一次性预加载指定切片到内存中
        if self.in_memory:
            self._load_into_memory()

    def _load_into_memory(self):
        """将当前 split 索引对应的所有张量一次性载入内存"""
        with h5py.File(self.h5_path, "r") as h5:
            idx = self.indices
            self.mem_head = np.asarray(_safe_h5_slice(h5["waveforms"]["wellhead_head"], idx), dtype=np.float32)
            if self.return_velocity and "wellhead_velocity" in h5["waveforms"]:
                self.mem_vel = np.asarray(_safe_h5_slice(h5["waveforms"]["wellhead_velocity"], idx), dtype=np.float32)
            else:
                self.mem_vel = None

            grp_lab = h5["labels"]
            self.mem_n_frac = np.asarray(_safe_h5_slice(grp_lab["n_frac"], idx), dtype=np.int64)
            self.mem_positions = np.asarray(_safe_h5_slice(grp_lab["fracture_positions"], idx), dtype=np.float32)
            self.mem_Cf = np.asarray(_safe_h5_slice(grp_lab["fracture_Cf"], idx), dtype=np.float32)
            self.mem_kleak = np.asarray(_safe_h5_slice(grp_lab["fracture_kleak"], idx), dtype=np.float32)
            self.mem_weights = np.asarray(_safe_h5_slice(grp_lab["fracture_weights"], idx), dtype=np.float32)
            self.mem_Kp = np.asarray(_safe_h5_slice(grp_lab["fracture_Kp"], idx), dtype=np.float32)

            if "fracture_type_ids" in grp_lab:
                self.mem_type_ids = np.asarray(_safe_h5_slice(grp_lab["fracture_type_ids"], idx), dtype=np.int64)
            else:
                self.mem_type_ids = np.zeros_like(self.mem_positions, dtype=np.int64)

            self.mem_tc = np.asarray(_safe_h5_slice(grp_lab["pump_closure_tc"], idx), dtype=np.float32)
            self.mem_wavespeed = np.asarray(_safe_h5_slice(grp_lab["wavespeed"], idx), dtype=np.float32)
            self.mem_H_ext = np.asarray(_safe_h5_slice(grp_lab["H_ext"], idx), dtype=np.float32)

            if "initial_velocity" in grp_lab:
                self.mem_v0 = np.asarray(_safe_h5_slice(grp_lab["initial_velocity"], idx), dtype=np.float32)
            else:
                self.mem_v0 = np.ones(self.length, dtype=np.float32)

            if "initial_head" in grp_lab:
                self.mem_h0 = np.asarray(_safe_h5_slice(grp_lab["initial_head"], idx), dtype=np.float32)
            else:
                self.mem_h0 = np.full(self.length, 300.0, dtype=np.float32)

            if "has_fault" in grp_lab:
                self.mem_has_fault = np.asarray(_safe_h5_slice(grp_lab["has_fault"], idx), dtype=np.int64)
            else:
                self.mem_has_fault = np.zeros(self.length, dtype=np.int64)

            if "fault_cluster_idx" in grp_lab:
                self.mem_fault_idx = np.asarray(_safe_h5_slice(grp_lab["fault_cluster_idx"], idx), dtype=np.int64)
            else:
                self.mem_fault_idx = np.full(self.length, -1, dtype=np.int64)

            if self.has_precomputed_cepstrum:
                self.mem_cepstrum = np.asarray(_safe_h5_slice(h5["features"]["cepstrum"], idx), dtype=np.float32)
            else:
                self.mem_cepstrum = None

    def _get_h5(self) -> h5py.File:
        """获取进程安全的 HDF5 文件句柄"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getstate__(self) -> Dict[str, Any]:
        """支持 PyTorch DataLoader 多进程序列化 (num_workers > 0)，剔除不可序列化的 HDF5 句柄"""
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """反序列化时恢复状态并将句柄置为 None (待子进程初次访问时按需打开)"""
        self.__dict__.update(state)
        self._h5_file = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        提取单样本数据张量字典。
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(f"索引超出范围: {idx} (数据集长度: {self.length})")

        if self.in_memory:
            head = self.mem_head[idx].copy()
            vel = self.mem_vel[idx].copy() if self.mem_vel is not None else None
            n_frac = self.mem_n_frac[idx]
            pos = self.mem_positions[idx]
            cf = self.mem_Cf[idx]
            kleak = self.mem_kleak[idx]
            weights = self.mem_weights[idx]
            kp = self.mem_Kp[idx]
            type_ids = self.mem_type_ids[idx]
            tc = self.mem_tc[idx]
            a = self.mem_wavespeed[idx]
            h_ext = self.mem_H_ext[idx]
            v0 = self.mem_v0[idx]
            h0 = self.mem_h0[idx]
            has_fault = self.mem_has_fault[idx]
            fault_idx = self.mem_fault_idx[idx]
            precomputed_ceps = self.mem_cepstrum[idx] if self.mem_cepstrum is not None else None
        else:
            h5 = self._get_h5()
            real_idx = int(self.indices[idx])
            head = np.asarray(h5["waveforms"]["wellhead_head"][real_idx], dtype=np.float32)
            if self.return_velocity and "wellhead_velocity" in h5["waveforms"]:
                vel = np.asarray(h5["waveforms"]["wellhead_velocity"][real_idx], dtype=np.float32)
            else:
                vel = None

            grp_lab = h5["labels"]
            n_frac = int(grp_lab["n_frac"][real_idx])
            pos = np.asarray(grp_lab["fracture_positions"][real_idx], dtype=np.float32)
            cf = np.asarray(grp_lab["fracture_Cf"][real_idx], dtype=np.float32)
            kleak = np.asarray(grp_lab["fracture_kleak"][real_idx], dtype=np.float32)
            weights = np.asarray(grp_lab["fracture_weights"][real_idx], dtype=np.float32)
            kp = np.asarray(grp_lab["fracture_Kp"][real_idx], dtype=np.float32)
            type_ids = np.asarray(grp_lab["fracture_type_ids"][real_idx], dtype=np.int64) if "fracture_type_ids" in grp_lab else np.zeros_like(pos, dtype=np.int64)
            tc = float(grp_lab["pump_closure_tc"][real_idx])
            a = float(grp_lab["wavespeed"][real_idx])
            h_ext = float(grp_lab["H_ext"][real_idx])
            v0 = float(grp_lab["initial_velocity"][real_idx]) if "initial_velocity" in grp_lab else 1.0
            h0 = float(grp_lab["initial_head"][real_idx]) if "initial_head" in grp_lab else 300.0
            has_fault = int(grp_lab["has_fault"][real_idx]) if "has_fault" in grp_lab else 0
            fault_idx = int(grp_lab["fault_cluster_idx"][real_idx]) if "fault_cluster_idx" in grp_lab else -1
            precomputed_ceps = np.asarray(h5["features"]["cepstrum"][real_idx], dtype=np.float32) if self.has_precomputed_cepstrum else None

        # 1. 动态特征处理：波前时域差分 (diff_order >= 1，前向填充 0.0 保长)
        processed_wave = head.copy()
        if self.diff_order > 0:
            for _ in range(self.diff_order):
                diff = np.diff(processed_wave)
                processed_wave = np.concatenate(([0.0], diff)).astype(np.float32)

        # 2. 动态特征处理：标准化
        if self.normalize:
            norm_type = "zscore" if isinstance(self.normalize, bool) else str(self.normalize).strip().lower()
            if norm_type in ("zscore", "standard", "standardize"):
                mean = float(np.mean(processed_wave))
                std = float(np.std(processed_wave))
                processed_wave = (processed_wave - mean) / (std + 1e-8)
            elif norm_type in ("minmax", "min_max"):
                w_min = float(np.min(processed_wave))
                w_max = float(np.max(processed_wave))
                processed_wave = (processed_wave - w_min) / (w_max - w_min + 1e-8)

        # 3. 动态特征处理：在线 1D 倒谱提取
        cepstrum = None
        distances = None
        if self.compute_cepstrum:
            ceps_res = compute_cepstrum_1d(
                time=self.timestamps,
                head=head,  # 倒谱基于原始水头提取
                wavespeed=float(a),
                ts=1.0,
                max_distance=self.cepstrum_max_distance,
            )
            cepstrum = torch.from_numpy(ceps_res["cepstrum"]).float()
            distances = torch.from_numpy(ceps_res["distance"]).float()
        elif precomputed_ceps is not None:
            cepstrum = torch.from_numpy(precomputed_ceps).float()
            distances = torch.from_numpy(self.distances).float() if self.distances is not None else None

        # 构建 PyTorch Tensor
        wave_t = torch.from_numpy(processed_wave).float()
        if self.return_channel_dim:
            wave_t = wave_t.unsqueeze(0)  # (1, N_time)

        sample_dict = {
            "waveform": wave_t,
            "timestamps": torch.from_numpy(self.timestamps).float(),
            "n_frac": torch.tensor(n_frac, dtype=torch.long),
            "fracture_positions": torch.from_numpy(pos).float(),
            "fracture_Cf": torch.from_numpy(cf).float(),
            "fracture_kleak": torch.from_numpy(kleak).float(),
            "fracture_weights": torch.from_numpy(weights).float(),
            "fracture_Kp": torch.from_numpy(kp).float(),
            "fracture_type_ids": torch.from_numpy(type_ids).long(),
            "pump_closure_tc": torch.tensor(tc, dtype=torch.float32),
            "wavespeed": torch.tensor(a, dtype=torch.float32),
            "H_ext": torch.tensor(h_ext, dtype=torch.float32),
            "initial_velocity": torch.tensor(v0, dtype=torch.float32),
            "initial_head": torch.tensor(h0, dtype=torch.float32),
            "has_fault": torch.tensor(has_fault, dtype=torch.long),
            "fault_cluster_idx": torch.tensor(fault_idx, dtype=torch.long),
        }

        if vel is not None:
            sample_dict["wellhead_velocity"] = torch.from_numpy(vel).float()
        if cepstrum is not None:
            sample_dict["cepstrum"] = cepstrum
        if distances is not None:
            sample_dict["distances"] = distances

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        return sample_dict

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> DataLoader:
        """
        开箱即用便捷实例化关联的 PyTorch DataLoader。
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

    def close(self):
        """安全释放 HDF5 文件句柄"""
        h5 = getattr(self, "_h5_file", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
            self._h5_file = None

    def __del__(self):
        self.close()
