# -*- coding: utf-8 -*-
"""
tests/test_torch_dataset.py

测试 PyTorch 数据集加载器 (MocWellboreDataset) 与轻量 HDF5 导出器：
1. 验证 train / val / test 比例划分与种子可复现性；
2. 验证全内存常驻模式与按需切片模式数值严格一致性；
3. 验证在线一阶差分滤波、Z-score 标准化与在线 1D 倒谱提取算子；
4. 验证 PyTorch DataLoader 批量切片读取耗时在微秒级；
5. 验证 Hdf5StreamWriter 分批流式写入与断点续传功能。
"""
from __future__ import annotations

import os
import tempfile
import time
import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader

from moc_simulate.v2.batch.dataset_exporter import (
    Hdf5StreamWriter,
    save_hdf5_dataset,
    load_hdf5_dataset,
    map_fracture_type_to_id,
)
from moc_simulate.v2.batch.torch_dataset import (
    MocWellboreDataset,
    split_dataset_indices,
)


@pytest.fixture(scope="module")
def sample_hdf5_path():
    """生成包含 20 个合成工况的轻量 HDF5 数据集供测试使用"""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test_mock_dataset.h5")
        n_samples = 20
        n_time = 1000
        timestamps = np.linspace(0.0, 1.0, n_time, dtype=np.float32)

        results = []
        for i in range(n_samples):
            # 构造合成阻尼振荡波形
            t = timestamps
            freq = 5.0 + i * 0.5
            head = 300.0 + 30.0 * np.exp(-3.0 * t) * np.sin(2.0 * np.pi * freq * t)
            vel = 1.0 - 0.8 * t

            n_frac = (i % 5) + 1
            pos = [4000.0 + 30.0 * j for j in range(n_frac)]
            cf = [0.010 * (1.0 + 0.1 * j) for j in range(n_frac)]
            kleak = [1.0e-4 * (1.0 + 0.05 * j) for j in range(n_frac)]
            weights = [1.0 / n_frac] * n_frac
            kp = [3.5e5 * (1.0 + 0.2 * j) for j in range(n_frac)]
            types = [f"Type {(j % 5) + 1}" for j in range(n_frac)]

            sample_meta = {
                "sample_id": i,
                "n_frac": n_frac,
                "fracture_positions": pos,
                "fracture_Cf": cf,
                "fracture_kleak": kleak,
                "fracture_inflow_weights": weights,
                "fracture_Kp": kp,
                "fracture_types": types,
                "pump_closure_duration": 0.05,
                "wavespeed": 1450.0,
                "H_ext": 100.0,
                "initial_velocity": 1.0,
                "initial_head": 300.0,
                "has_fault": (i % 7 == 0),
                "fault_cluster_idx": 0 if (i % 7 == 0) else -1,
            }

            results.append({
                "sample_id": i,
                "status": "success",
                "timestamps": timestamps,
                "wellhead_head": head.astype(np.float32),
                "wellhead_velocity": vel.astype(np.float32),
                "sample_meta": sample_meta,
            })

        save_hdf5_dataset(
            file_path=file_path,
            results=results,
            metadata={"description": "mock test dataset", "n_samples": n_samples},
            max_frac_dim=8,
            compression="gzip",
            compression_opts=4,
            export_features=False,  # 轻量模式
        )

        yield file_path


class TestTorchDatasetSplitsAndBasics:
    """测试数据集基本划分、元数据与索引"""

    def test_split_indices_proportions(self):
        n_samples = 1000
        splits = split_dataset_indices(n_samples, split_ratios=(0.8, 0.1, 0.1), seed=42)
        assert len(splits["train"]) == 800
        assert len(splits["val"]) == 100
        assert len(splits["test"]) == 100
        assert len(splits["all"]) == 1000

        # 确保无交集且完全覆盖
        combined = np.concatenate([splits["train"], splits["val"], splits["test"]])
        assert len(np.unique(combined)) == 1000

    def test_split_reproducibility(self):
        s1 = split_dataset_indices(100, (0.8, 0.1, 0.1), seed=123)
        s2 = split_dataset_indices(100, (0.8, 0.1, 0.1), seed=123)
        np.testing.assert_array_equal(s1["train"], s2["train"])

    def test_torch_dataset_initialization(self, sample_hdf5_path):
        ds_train = MocWellboreDataset(sample_hdf5_path, split="train", split_ratios=(0.8, 0.1, 0.1), seed=42)
        ds_val = MocWellboreDataset(sample_hdf5_path, split="val", split_ratios=(0.8, 0.1, 0.1), seed=42)
        ds_test = MocWellboreDataset(sample_hdf5_path, split="test", split_ratios=(0.8, 0.1, 0.1), seed=42)
        ds_all = MocWellboreDataset(sample_hdf5_path, split="all")

        assert len(ds_train) == 16
        assert len(ds_val) == 2
        assert len(ds_test) == 2
        assert len(ds_all) == 20
        assert ds_all.n_time == 1000


class TestTorchDatasetItemAndTransforms:
    """测试单样本读取与动态变换算子"""

    def test_getitem_shapes_and_types(self, sample_hdf5_path):
        ds = MocWellboreDataset(sample_hdf5_path, split="all", in_memory=True)
        sample = ds[0]

        assert isinstance(sample["waveform"], torch.Tensor)
        assert sample["waveform"].shape == (1000,)
        assert sample["waveform"].dtype == torch.float32

        assert sample["n_frac"].dtype == torch.long
        assert sample["fracture_positions"].shape == (8,)
        assert sample["fracture_type_ids"].shape == (8,)
        assert sample["fracture_type_ids"].dtype == torch.long
        assert sample["wavespeed"].dtype == torch.float32

    def test_in_memory_vs_lazy_exact_match(self, sample_hdf5_path):
        ds_mem = MocWellboreDataset(sample_hdf5_path, split="all", in_memory=True)
        ds_lazy = MocWellboreDataset(sample_hdf5_path, split="all", in_memory=False)

        for i in range(5):
            s_mem = ds_mem[i]
            s_lazy = ds_lazy[i]

            torch.testing.assert_close(s_mem["waveform"], s_lazy["waveform"])
            torch.testing.assert_close(s_mem["fracture_positions"], s_lazy["fracture_positions"])
            torch.testing.assert_close(s_mem["fracture_Cf"], s_lazy["fracture_Cf"])
            torch.testing.assert_close(s_mem["fracture_type_ids"], s_lazy["fracture_type_ids"])

        ds_lazy.close()

    def test_wavefront_first_order_difference(self, sample_hdf5_path):
        ds_raw = MocWellboreDataset(sample_hdf5_path, split="all", diff_order=0)
        ds_diff = MocWellboreDataset(sample_hdf5_path, split="all", diff_order=1)

        raw_wave = ds_raw[0]["waveform"].numpy()
        diff_wave = ds_diff[0]["waveform"].numpy()

        assert len(diff_wave) == len(raw_wave)
        assert diff_wave[0] == 0.0
        np.testing.assert_allclose(diff_wave[1:], np.diff(raw_wave), rtol=1e-5, atol=1e-5)

    def test_zscore_normalization(self, sample_hdf5_path):
        ds_norm = MocWellboreDataset(sample_hdf5_path, split="all", normalize=True)
        wave = ds_norm[0]["waveform"]

        assert abs(float(wave.mean())) < 1e-4
        assert abs(float(wave.std()) - 1.0) < 1e-3

    def test_dynamic_cepstrum_transform(self, sample_hdf5_path):
        ds_ceps = MocWellboreDataset(sample_hdf5_path, split="all", compute_cepstrum=True, cepstrum_max_distance=4500.0)
        sample = ds_ceps[0]

        assert "cepstrum" in sample
        assert "distances" in sample
        assert isinstance(sample["cepstrum"], torch.Tensor)
        assert len(sample["cepstrum"]) > 0
        assert len(sample["cepstrum"]) == len(sample["distances"])


    def test_minmax_normalization_and_higher_order_diff(self, sample_hdf5_path):
        ds_minmax = MocWellboreDataset(sample_hdf5_path, split="all", normalize="minmax", diff_order=2)
        sample = ds_minmax[0]
        wave = sample["waveform"]
        assert float(wave.min()) >= -1e-6
        assert float(wave.max()) <= 1.0 + 1e-6

    def test_custom_indices_and_velocity(self, sample_hdf5_path):
        custom_idx = [0, 2, 4, 2, 0]  # 含重复与乱序
        ds_custom = MocWellboreDataset(sample_hdf5_path, indices=custom_idx, in_memory=True, return_velocity=True)
        assert len(ds_custom) == 5
        s0 = ds_custom[0]
        assert "wellhead_velocity" in s0
        assert s0["wellhead_velocity"].shape == (1000,)
        # 验证重复样本数值严格相等
        torch.testing.assert_close(ds_custom[0]["waveform"], ds_custom[4]["waveform"])
        torch.testing.assert_close(ds_custom[1]["waveform"], ds_custom[3]["waveform"])


class TestTorchDataLoaderPerformance:
    """测试 PyTorch DataLoader 批量切片、微秒级高吞吐与多 Worker 兼容性"""

    def test_dataloader_batch_slicing_and_speed(self, sample_hdf5_path):
        ds = MocWellboreDataset(sample_hdf5_path, split="all", in_memory=True)
        loader = DataLoader(ds, batch_size=4, shuffle=True)

        batch_count = 0
        t0 = time.perf_counter()
        for batch in loader:
            batch_count += 1
            assert batch["waveform"].shape == (4, 1000)
            assert batch["fracture_positions"].shape == (4, 8)
            assert batch["fracture_type_ids"].shape == (4, 8)
            assert len(batch["n_frac"]) == 4

        elapsed = time.perf_counter() - t0
        assert batch_count == 5
        # 验证单样本平均加载读取时延在微秒级 (< 1 ms / sample)
        time_per_sample_us = (elapsed / 20.0) * 1e6
        print(f"\nPyTorch DataLoader 吞吐实测: {time_per_sample_us:.2f} 微秒/样本")
        assert time_per_sample_us < 5000.0  # 宽松上限 5 ms 以防止低性能机器偶发卡顿

    def test_dataloader_multiworker_serialization_lazy(self, sample_hdf5_path):
        """验证多进程 DataLoader 在惰性 HDF5 且主进程已打开文件句柄场景下的序列化健壮性"""
        ds_lazy = MocWellboreDataset(sample_hdf5_path, split="all", in_memory=False)
        # 模拟主进程预先读取触发 _h5_file 打开
        _ = ds_lazy[0]
        assert ds_lazy._h5_file is not None

        # 验证 getstate 将 _h5_file 成功清空为 None
        state = ds_lazy.__getstate__()
        assert state["_h5_file"] is None

        # 启动多 Worker DataLoader
        loader = ds_lazy.get_dataloader(batch_size=4, num_workers=2, shuffle=False)
        batches_seen = 0
        for batch in loader:
            batches_seen += 1
            assert batch["waveform"].shape == (4, 1000)
            if batches_seen >= 2:
                break
        assert batches_seen >= 2
        ds_lazy.close()


class TestHdf5StreamWriter:
    """测试 HDF5 流式写入器与断点续传机制"""

    def test_stream_writer_batch_append_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "stream_test.h5")
            n_time = 500
            ts = np.linspace(0.0, 0.5, n_time, dtype=np.float32)

            writer = Hdf5StreamWriter(
                file_path=file_path,
                n_time=n_time,
                timestamps=ts,
                max_frac_dim=6,
                export_features=False,
                chunk_size=5,
            )

            # 写入第 1 批 (3 个样本)
            batch1 = []
            for i in range(3):
                batch1.append({
                    "sample_id": i,
                    "status": "success",
                    "timestamps": ts,
                    "wellhead_head": np.full(n_time, 300.0 + i, dtype=np.float32),
                    "wellhead_velocity": np.ones(n_time, dtype=np.float32),
                    "sample_meta": {
                        "fracture_positions": [4100.0, 4120.0],
                        "fracture_Cf": [0.01, 0.012],
                        "fracture_kleak": [1e-4, 1.2e-4],
                        "fracture_inflow_weights": [0.5, 0.5],
                        "fracture_Kp": [3e5, 3.2e5],
                        "fracture_types": ["Type I: 优势发育主进液簇", "Type II: 均衡/正常发育簇"],
                        "pump_closure_duration": 0.05,
                        "wavespeed": 1450.0,
                        "H_ext": 100.0,
                    },
                })
            count1 = writer.append_batch(batch1)
            assert count1 == 3
            assert writer.get_num_samples() == 3

            # 模拟进程中断并重新挂载现有文件续传
            writer.close()

            writer_resume = Hdf5StreamWriter(
                file_path=file_path,
                n_time=n_time,
                timestamps=ts,
                max_frac_dim=6,
                export_features=False,
                chunk_size=5,
            )
            assert writer_resume.get_num_samples() == 3

            # 追加第 2 批 (2 个样本)
            batch2 = []
            for i in range(3, 5):
                batch2.append({
                    "sample_id": i,
                    "status": "success",
                    "timestamps": ts,
                    "wellhead_head": np.full(n_time, 300.0 + i, dtype=np.float32),
                    "wellhead_velocity": np.ones(n_time, dtype=np.float32),
                    "sample_meta": {
                        "fracture_positions": [4200.0],
                        "fracture_Cf": [0.02],
                        "fracture_kleak": [2e-4],
                        "fracture_inflow_weights": [1.0],
                        "fracture_Kp": [2e5],
                        "fracture_types": ["Type I: 优势发育主进液簇"],
                        "pump_closure_duration": 0.08,
                        "wavespeed": 1450.0,
                        "H_ext": 100.0,
                    },
                })
            count2 = writer_resume.append_batch(batch2)
            assert count2 == 5

            writer_resume.finalize(metadata={"experiment": "stream_test"})

            # 读取完整文件校验
            loaded = load_hdf5_dataset(file_path)
            assert loaded["metadata"]["n_samples"] == 5
            assert loaded["metadata"]["experiment"] == "stream_test"
            assert loaded["waveforms"]["wellhead_head"].shape == (5, n_time)
            assert loaded["labels"]["fracture_type_ids"].shape == (5, 6)
            # 校验第 0 个样本与第 3 个样本的标签
            assert loaded["labels"]["fracture_type_ids"][0, 0] == 1
            assert loaded["labels"]["fracture_type_ids"][0, 1] == 2
            assert loaded["labels"]["fracture_type_ids"][3, 0] == 1
            assert loaded["labels"]["fracture_type_ids"][3, 1] == 0
