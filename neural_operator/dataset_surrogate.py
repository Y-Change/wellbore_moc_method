# -*- coding: utf-8 -*-
"""
neural_operator/dataset_surrogate.py — 神经算子代理模型 Dataset 与物理脉冲序列编码器

功能：
1. 批量遍历 output/lhs_dataset/data/ 下的 .npz 仿真真解；
2. 为每一组真解构建 4 通道物理脉冲序列编码 (Physically-Informed Impulse Train Encoding)；
3. 将原始 50,000 步 (dt=1ms, tf=50s) 样条插值对齐为标准 2 的幂次长度 (默认 N_TIME = 4096)，大幅优化 FFT 计算极速。
"""
from __future__ import annotations
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import sys
import os
_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, 'README.md')):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _p = os.path.dirname(_d)
    if _p == _d:
        break
    _d = _p

# 导入默认物理参数以计算反射时延
from moc_simulate.config import WELL_CONFIG, SIM_CONFIG


class FracturingMOCSurrogateDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "output/lhs_dataset/data",
        n_time_target: int = 4096,
        wavespeed: float = 1450.0,
        ts: float = 1.0,
        sigma_impulse_s: float = 0.15,  # 高斯脉冲半宽时间 [s] (体现波头滤失与频散平滑)
        split: str = "train",
        train_ratio: float = 0.85,
        seed: int = 42,
    ):
        super(FracturingMOCSurrogateDataset, self).__init__()
        self.data_dir = os.path.abspath(data_dir)
        self.n_time_target = n_time_target
        self.wavespeed = wavespeed
        self.ts = ts
        self.sigma = sigma_impulse_s
        
        all_files = sorted(glob.glob(os.path.join(self.data_dir, "case_*.npz")))
        if not all_files:
            raise FileNotFoundError(f"[错误] 未在 {self.data_dir} 下找到任何 case_*.npz 数据集文件！")
            
        # 划分训练集和测试/验证集；用局部 RandomState 保持历史 seed=42 划分不变。
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_files))
        n_train = int(len(all_files) * train_ratio)
        if split.lower() == "train":
            self.files = [all_files[i] for i in indices[:n_train]]
        else:
            self.files = [all_files[i] for i in indices[n_train:]]
            
        print(f"[{split.upper()} 集] 成功挂载 {len(self.files)} 组物理波形文件。")

    def __len__(self) -> int:
        return len(self.files)

    def _encode_impulse_trains(
        self,
        t_target: np.ndarray,
        positions: np.ndarray,
        Cf_list: np.ndarray,
        kleak_list: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在目标时间网格上生成各簇反射时刻 t_arr 的高斯脉冲序列"""
        n_points = len(t_target)
        impulse_cf = np.zeros(n_points, dtype=np.float32)
        impulse_kleak = np.zeros(n_points, dtype=np.float32)
        
        # 针对每一条裂缝，计算两倍井深往返反射时刻 t_arr = ts + 2 * xf / a
        for xf, cf, kl in zip(positions, Cf_list, kleak_list):
            t_arr = self.ts + (2.0 * xf) / self.wavespeed
            # 计算距离反射中轴的高斯权重 exp( - (t - t_arr)^2 / (2 * sigma^2) )
            gauss_kernel = np.exp(-((t_target - t_arr) ** 2) / (2.0 * self.sigma ** 2)).astype(np.float32)
            
            # 对数加权：将柔度与滤失数量级差异平滑化
            w_cf = float(np.log10(max(cf, 1.0e-12)) + 12.0)    # 偏移为正数 [3.3 ~ 6.5]
            w_kl = float(np.log10(max(kl, 1.0e-15)) + 15.0)    # 偏移为正数 [9.0 ~ 12.0]
            
            impulse_cf += w_cf * gauss_kernel
            impulse_kleak += w_kl * gauss_kernel
            
        return impulse_cf, impulse_kleak

    def _load_case(self, idx: int) -> Tuple[str, np.lib.npyio.NpzFile]:
        filepath = self.files[idx]
        npz = np.load(filepath, allow_pickle=False)
        required = ("t", "H_wh", "x_f", "Cf", "kleak")
        missing = [key for key in required if key not in npz]
        if missing:
            npz.close()
            raise ValueError(f"{filepath} 缺少字段: {missing}")

        t = np.asarray(npz["t"])
        H_wh = np.asarray(npz["H_wh"])
        x_f = np.asarray(npz["x_f"])
        Cf = np.asarray(npz["Cf"])
        kleak = np.asarray(npz["kleak"])
        if t.ndim != 1 or H_wh.ndim != 1 or len(t) != len(H_wh):
            npz.close()
            raise ValueError(f"{filepath} 的 t/H_wh 形状不一致")
        if len(t) < 2 or not np.all(np.diff(t) > 0):
            npz.close()
            raise ValueError(f"{filepath} 的时间轴必须严格递增")
        if not (len(x_f) == len(Cf) == len(kleak)):
            npz.close()
            raise ValueError(f"{filepath} 的 x_f/Cf/kleak 长度不一致")
        for name, array in (("t", t), ("H_wh", H_wh), ("x_f", x_f), ("Cf", Cf), ("kleak", kleak)):
            if not np.isfinite(array).all():
                npz.close()
                raise ValueError(f"{filepath} 的 {name} 含 NaN 或 Inf")
        return filepath, npz

    def get_case_metadata(self, idx: int) -> dict:
        """返回不参与 DataLoader 拼接的可审计样本元数据。"""
        filepath, npz = self._load_case(idx)
        try:
            t_raw = np.asarray(npz["t"], dtype=np.float64)
            tf = float(npz["tf"]) if "tf" in npz else float(t_raw[-1])
            case_name = os.path.splitext(os.path.basename(filepath))[0]
            friction = str(npz["friction"].item()) if "friction" in npz else "unknown"
            return {
                "case_id": case_name,
                "source_file": os.path.abspath(filepath),
                "tf": tf,
                "n_frac": int(npz["n_frac"]) if "n_frac" in npz else int(len(npz["x_f"])),
                "x_f": np.asarray(npz["x_f"], dtype=np.float32).copy(),
                "Cf": np.asarray(npz["Cf"], dtype=np.float32).copy(),
                "kleak": np.asarray(npz["kleak"], dtype=np.float32).copy(),
                "friction": friction,
                "time_axis": np.linspace(0.0, tf, self.n_time_target, dtype=np.float32),
                "wavespeed": float(self.wavespeed),
                "pump_shut_time": float(self.ts),
                "sigma_impulse_s": float(self.sigma),
            }
        finally:
            npz.close()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath, npz = self._load_case(idx)
        try:
            t_raw = npz["t"]
            H_wh_raw = npz["H_wh"]
            x_f = npz["x_f"]
            Cf = npz["Cf"]
            kleak = npz["kleak"]
            tf = float(npz["tf"]) if "tf" in npz else t_raw[-1]

            # 1. 构造标准对齐的目标时间网格 (例如 4096 点)
            t_target = np.linspace(0.0, tf, self.n_time_target, dtype=np.float32)

            # 2. 对原真解波形进行样条/线性插值对齐
            H_wh_target = np.interp(t_target, t_raw, H_wh_raw).astype(np.float32)

            # 3. 构造 4 通道物理脉冲输入
            ch0_time = (t_target / max(tf, 1.0)).astype(np.float32)
            ch1_pump = (t_target < self.ts).astype(np.float32)
            ch2_cf, ch3_kl = self._encode_impulse_trains(t_target, x_f, Cf, kleak)

            input_tensor = np.stack([ch0_time, ch1_pump, ch2_cf, ch3_kl], axis=0)
            target_tensor = np.expand_dims(H_wh_target, axis=0)
            return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)
        finally:
            npz.close()


def get_surrogate_dataloaders(
    data_dir: str = "output/lhs_dataset/data",
    batch_size: int = 32,
    num_workers: int = 4,
    n_time_target: int = 4096,
) -> Tuple[DataLoader, DataLoader]:
    """快捷构造训练集和验证集的 DataLoader"""
    train_ds = FracturingMOCSurrogateDataset(
        data_dir=data_dir, n_time_target=n_time_target, split="train", train_ratio=0.85
    )
    val_ds = FracturingMOCSurrogateDataset(
        data_dir=data_dir, n_time_target=n_time_target, split="val", train_ratio=0.85
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


if __name__ == "__main__":
    if sys.platform.startswith('win') and hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    # 本地冒烟测试：尝试读取咱们刚才跑出的 10 组测试集
    test_dir = "output/lhs_dataset/data"
    if os.path.exists(test_dir):
        print(f"正在测试读取 {test_dir} 下的数据...")
        ds = FracturingMOCSurrogateDataset(data_dir=test_dir, n_time_target=4096, split="train")
        inp, tgt = ds[0]
        print(f"读取样本 0 成功! 输入张量: {inp.shape}, 目标输出: {tgt.shape}")
        print("Dataset 测试全通过 [PASS]")
    else:
        print("[提示] 生产数据集尚在生成中，可随后进行 DataLoader 测试。")
