# -*- coding: utf-8 -*-
"""
moc_simulate.v2.batch.dataset_exporter

面向水击波智能反演任务的深度学习标准化数据集导出器：
以 HDF5 (.h5) 为原生主要存储格式，支持大容量存储、快速切片索引与 PyTorch DataLoader 原生直接读取。
支持轻量模式 (仅存波形与地质物理标签)、流式分批写入与断点续传。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# 裂缝类型与整型数值类别 ID 映射
FRACTURE_TYPE_NAME_TO_ID: Dict[str, int] = {
    "Type I": 1,
    "Type II": 2,
    "Type III": 3,
    "Type IV": 4,
    "Type V": 5,
}

FRACTURE_ID_TO_NAME: Dict[int, str] = {
    1: "Type I: 优势发育主进液簇",
    2: "Type II: 均衡/正常发育簇",
    3: "Type III: 受抑/欠发育弱进液簇",
    4: "Type IV: 砂堵闭合/未起裂死簇",
    5: "Type V: 沟通天然断层/强微裂缝簇",
}


def map_fracture_type_to_id(ftype_str: Optional[str]) -> int:
    """将裂缝类型文本描述映射为标准化类别整数 ID (1~5，0 代表填充无裂缝)"""
    if not ftype_str:
        return 0
    s = str(ftype_str).strip()
    if s.startswith("Type I:") or s == "Type I":
        return 1
    if s.startswith("Type II:") or s == "Type II":
        return 2
    if s.startswith("Type III:") or s == "Type III":
        return 3
    if s.startswith("Type IV:") or s == "Type IV":
        return 4
    if s.startswith("Type V:") or s == "Type V":
        return 5
    return 0


def save_hdf5_dataset(
    file_path: str,
    results: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    max_frac_dim: int = 8,
    compression: str = "gzip",
    compression_opts: int = 4,
    export_features: bool = True,
) -> str:
    """
    将正演批量仿真结果序列化打包存储为标准 HDF5 文件。

    结构组织:
        /waveforms/
            wellhead_head      : (N_samples, N_time)
            wellhead_velocity  : (N_samples, N_time)
            timestamps         : (N_time,)
        /features/ (可选，仅在 export_features=True 且存在倒谱特征时写入)
            cepstrum           : (N_samples, N_dist)
            distances          : (N_dist,)
        /labels/
            n_frac             : (N_samples,)
            fracture_positions : (N_samples, max_frac_dim)
            fracture_Cf        : (N_samples, max_frac_dim)
            fracture_kleak     : (N_samples, max_frac_dim)
            fracture_weights   : (N_samples, max_frac_dim)
            fracture_Kp        : (N_samples, max_frac_dim)
            fracture_type_ids  : (N_samples, max_frac_dim) [int32]
            fracture_types     : (N_samples, max_frac_dim) [string]
            pump_closure_tc    : (N_samples,)
            wavespeed          : (N_samples,)
            H_ext              : (N_samples,)
            initial_velocity   : (N_samples,)
            initial_head       : (N_samples,)
            has_fault          : (N_samples,) [int32]
            fault_cluster_idx  : (N_samples,) [int32]
    """
    if not HAS_H5PY:
        raise RuntimeError("保存 HDF5 数据集需要安装 h5py 库 (pip install h5py)")

    valid_results = [r for r in results if r.get("status") == "success"]
    if not valid_results:
        raise ValueError("传入的仿真结果列表中没有成功的样本可供导出。")

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    n_samples = len(valid_results)

    # 提取时间基准向量
    timestamps = np.asarray(valid_results[0]["timestamps"], dtype=np.float32)
    n_time = len(timestamps)

    # 检查是否包含倒谱特征
    has_valid_features = (
        export_features
        and any("cepstrum" in r and r["cepstrum"] is not None and len(r["cepstrum"]) > 0 for r in valid_results)
    )

    distances = np.array([], dtype=np.float32)
    n_dist = 0
    if has_valid_features:
        for r in valid_results:
            if "distances" in r and r["distances"] is not None and len(r["distances"]) > 0:
                distances = np.asarray(r["distances"], dtype=np.float32)
                n_dist = len(distances)
                break

    # 动态自适应最大裂缝数，避免截断大簇数仿真
    batch_max_frac = 0
    for r in valid_results:
        pos = r.get("sample_meta", {}).get("fracture_positions", [])
        if pos:
            batch_max_frac = max(batch_max_frac, len(pos))
    actual_max_frac_dim = max(max_frac_dim, batch_max_frac)

    # 初始化矩阵容器
    head_matrix = np.zeros((n_samples, n_time), dtype=np.float32)
    vel_matrix = np.zeros((n_samples, n_time), dtype=np.float32)
    ceps_matrix = np.zeros((n_samples, n_dist), dtype=np.float32) if has_valid_features else None

    n_frac_arr = np.zeros(n_samples, dtype=np.int32)
    pos_matrix = np.zeros((n_samples, actual_max_frac_dim), dtype=np.float32)
    cf_matrix = np.zeros((n_samples, actual_max_frac_dim), dtype=np.float32)
    kleak_matrix = np.zeros((n_samples, actual_max_frac_dim), dtype=np.float32)
    weights_matrix = np.zeros((n_samples, actual_max_frac_dim), dtype=np.float32)
    kp_matrix = np.zeros((n_samples, actual_max_frac_dim), dtype=np.float32)
    type_ids_matrix = np.zeros((n_samples, actual_max_frac_dim), dtype=np.int32)
    type_names_list: List[List[str]] = [[""] * actual_max_frac_dim for _ in range(n_samples)]

    tc_arr = np.zeros(n_samples, dtype=np.float32)
    wavespeed_arr = np.zeros(n_samples, dtype=np.float32)
    hext_arr = np.zeros(n_samples, dtype=np.float32)
    v0_arr = np.zeros(n_samples, dtype=np.float32)
    h0_arr = np.zeros(n_samples, dtype=np.float32)
    has_fault_arr = np.zeros(n_samples, dtype=np.int32)
    fault_idx_arr = np.full(n_samples, -1, dtype=np.int32)

    for i, r in enumerate(valid_results):
        wh_h = np.asarray(r["wellhead_head"], dtype=np.float32)
        len_h = min(n_time, len(wh_h))
        head_matrix[i, :len_h] = wh_h[:len_h]

        if r.get("wellhead_velocity") is not None:
            wh_v = np.asarray(r["wellhead_velocity"], dtype=np.float32)
            len_v = min(n_time, len(wh_v))
            vel_matrix[i, :len_v] = wh_v[:len_v]

        if has_valid_features and "cepstrum" in r and r["cepstrum"] is not None and len(r["cepstrum"]) > 0:
            c_len = min(n_dist, len(r["cepstrum"]))
            ceps_matrix[i, :c_len] = np.asarray(r["cepstrum"][:c_len], dtype=np.float32)

        meta = r.get("sample_meta", {})
        raw_pos = meta.get("fracture_positions", [])
        nf = min(actual_max_frac_dim, len(raw_pos))
        n_frac_arr[i] = len(raw_pos)

        if nf > 0:
            pos_matrix[i, :nf] = np.asarray(raw_pos[:nf], dtype=np.float32)
            if "fracture_Cf" in meta and meta["fracture_Cf"] is not None:
                cf_matrix[i, :nf] = np.asarray(meta["fracture_Cf"][:nf], dtype=np.float32)
            if "fracture_kleak" in meta and meta["fracture_kleak"] is not None:
                kleak_matrix[i, :nf] = np.asarray(meta["fracture_kleak"][:nf], dtype=np.float32)
            if "fracture_inflow_weights" in meta and meta["fracture_inflow_weights"] is not None:
                weights_matrix[i, :nf] = np.asarray(meta["fracture_inflow_weights"][:nf], dtype=np.float32)
            if "fracture_Kp" in meta and meta["fracture_Kp"] is not None:
                kp_matrix[i, :nf] = np.asarray(meta["fracture_Kp"][:nf], dtype=np.float32)

            ftypes = meta.get("fracture_types", [])
            for j in range(min(nf, len(ftypes))):
                type_names_list[i][j] = str(ftypes[j])
                type_ids_matrix[i, j] = map_fracture_type_to_id(ftypes[j])

        tc_arr[i] = float(meta.get("pump_closure_duration", 1.0))
        wavespeed_arr[i] = float(meta.get("wavespeed", 1450.0))
        hext_arr[i] = float(meta.get("H_ext", 100.0))
        v0_arr[i] = float(meta.get("initial_velocity", 1.0))
        h0_arr[i] = float(meta.get("initial_head", 300.0))
        has_fault_arr[i] = int(bool(meta.get("has_fault", False)))
        fault_idx_arr[i] = int(meta.get("fault_cluster_idx", -1))

    # 写入 HDF5
    str_dtype = h5py.string_dtype(encoding="utf-8")
    comp_kwargs = {"compression": compression, "shuffle": True}
    if compression == "gzip" and compression_opts is not None:
        comp_kwargs["compression_opts"] = compression_opts

    with h5py.File(file_path, "w") as h5:
        # 1. 波形
        grp_wave = h5.create_group("waveforms")
        grp_wave.create_dataset("wellhead_head", data=head_matrix, chunks=(1, n_time), **comp_kwargs)
        grp_wave.create_dataset("wellhead_velocity", data=vel_matrix, chunks=(1, n_time), **comp_kwargs)
        grp_wave.create_dataset("timestamps", data=timestamps)

        # 2. 特征 (仅在具有倒谱时保存)
        if has_valid_features and ceps_matrix is not None:
            grp_feat = h5.create_group("features")
            grp_feat.create_dataset("cepstrum", data=ceps_matrix, chunks=(1, n_dist), **comp_kwargs)
            grp_feat.create_dataset("distances", data=distances)

        # 3. 标签
        grp_lab = h5.create_group("labels")
        grp_lab.create_dataset("n_frac", data=n_frac_arr)
        grp_lab.create_dataset("fracture_positions", data=pos_matrix, **comp_kwargs)
        grp_lab.create_dataset("fracture_Cf", data=cf_matrix, **comp_kwargs)
        grp_lab.create_dataset("fracture_kleak", data=kleak_matrix, **comp_kwargs)
        grp_lab.create_dataset("fracture_weights", data=weights_matrix, **comp_kwargs)
        grp_lab.create_dataset("fracture_Kp", data=kp_matrix, **comp_kwargs)
        grp_lab.create_dataset("fracture_type_ids", data=type_ids_matrix, **comp_kwargs)
        grp_lab.create_dataset("fracture_types", data=np.array(type_names_list, dtype=object), dtype=str_dtype, **comp_kwargs)
        grp_lab.create_dataset("pump_closure_tc", data=tc_arr)
        grp_lab.create_dataset("wavespeed", data=wavespeed_arr)
        grp_lab.create_dataset("H_ext", data=hext_arr)
        grp_lab.create_dataset("initial_velocity", data=v0_arr)
        grp_lab.create_dataset("initial_head", data=h0_arr)
        grp_lab.create_dataset("has_fault", data=has_fault_arr)
        grp_lab.create_dataset("fault_cluster_idx", data=fault_idx_arr)

        # 4. 元数据属性
        h5.attrs["n_samples"] = n_samples
        h5.attrs["format_version"] = "2.0-moc"
        h5.attrs["lightweight"] = not has_valid_features
        h5.attrs["max_frac_dim"] = actual_max_frac_dim
        if metadata:
            h5.attrs["metadata_json"] = json.dumps(metadata)

    return os.path.abspath(file_path)


class Hdf5StreamWriter:
    """
    支持流式分批写入与断点续传的 HDF5 数据集写入器。
    采用 chunked + maxshape=(None, ...) 动态扩展存储，有效防止内存膨胀，并保障仿真中断时进度不丢失。
    """

    def __init__(
        self,
        file_path: str,
        n_time: int = 60001,
        timestamps: Optional[np.ndarray] = None,
        max_frac_dim: int = 8,
        compression: str = "gzip",
        compression_opts: int = 4,
        export_features: bool = False,
        n_dist: int = 0,
        distances: Optional[np.ndarray] = None,
        chunk_size: int = 50,
    ):
        if not HAS_H5PY:
            raise RuntimeError("Hdf5StreamWriter 需要安装 h5py 库 (pip install h5py)")

        self.file_path = os.path.abspath(file_path)
        self.n_time = n_time
        self.timestamps = timestamps
        self.max_frac_dim = max_frac_dim
        self.compression = compression
        self.compression_opts = compression_opts
        self.export_features = export_features
        self.n_dist = n_dist
        self.distances = distances
        self.chunk_size = chunk_size

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.h5: Optional[h5py.File] = None
        self._init_file()

    def _init_file(self):
        """初始化或打开现有 HDF5 文件并创建可变长数据集"""
        file_exists = os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0
        self.h5 = h5py.File(self.file_path, "a")

        str_dtype = h5py.string_dtype(encoding="utf-8")
        comp_kwargs = {"compression": self.compression, "shuffle": True}
        if self.compression == "gzip" and self.compression_opts is not None:
            comp_kwargs["compression_opts"] = self.compression_opts

        # 1. waveforms 组
        if "waveforms" not in self.h5:
            grp_wave = self.h5.create_group("waveforms")
            grp_wave.create_dataset(
                "wellhead_head",
                shape=(0, self.n_time),
                maxshape=(None, self.n_time),
                chunks=(1, self.n_time),
                dtype=np.float32,
                **comp_kwargs,
            )
            grp_wave.create_dataset(
                "wellhead_velocity",
                shape=(0, self.n_time),
                maxshape=(None, self.n_time),
                chunks=(1, self.n_time),
                dtype=np.float32,
                **comp_kwargs,
            )
            if self.timestamps is not None:
                grp_wave.create_dataset("timestamps", data=np.asarray(self.timestamps, dtype=np.float32))

        # 2. features 组 (仅在 export_features 时)
        if self.export_features and "features" not in self.h5:
            grp_feat = self.h5.create_group("features")
            grp_feat.create_dataset(
                "cepstrum",
                shape=(0, self.n_dist),
                maxshape=(None, self.n_dist),
                chunks=(1, self.n_dist),
                dtype=np.float32,
                **comp_kwargs,
            )
            if self.distances is not None:
                grp_feat.create_dataset("distances", data=np.asarray(self.distances, dtype=np.float32))

        # 3. labels 组
        if "labels" not in self.h5:
            grp_lab = self.h5.create_group("labels")
            grp_lab.create_dataset("n_frac", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.int32)
            grp_lab.create_dataset(
                "fracture_positions",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=np.float32,
                **comp_kwargs,
            )
            grp_lab.create_dataset(
                "fracture_Cf",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=np.float32,
                **comp_kwargs,
            )
            grp_lab.create_dataset(
                "fracture_kleak",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=np.float32,
                **comp_kwargs,
            )
            grp_lab.create_dataset(
                "fracture_weights",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=np.float32,
                **comp_kwargs,
            )
            grp_lab.create_dataset(
                "fracture_Kp",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=np.float32,
                **comp_kwargs,
            )
            grp_lab.create_dataset(
                "fracture_type_ids",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=np.int32,
                **comp_kwargs,
            )
            grp_lab.create_dataset(
                "fracture_types",
                shape=(0, self.max_frac_dim),
                maxshape=(None, self.max_frac_dim),
                chunks=(self.chunk_size, self.max_frac_dim),
                dtype=str_dtype,
                **comp_kwargs,
            )
            grp_lab.create_dataset("pump_closure_tc", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.float32)
            grp_lab.create_dataset("wavespeed", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.float32)
            grp_lab.create_dataset("H_ext", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.float32)
            grp_lab.create_dataset("initial_velocity", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.float32)
            grp_lab.create_dataset("initial_head", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.float32)
            grp_lab.create_dataset("has_fault", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.int32)
            grp_lab.create_dataset("fault_cluster_idx", shape=(0,), maxshape=(None,), chunks=(self.chunk_size,), dtype=np.int32)

        # 4. 基础属性
        self.h5.attrs["format_version"] = "2.0-moc"
        self.h5.attrs["lightweight"] = not self.export_features
        self.h5.attrs["max_frac_dim"] = self.max_frac_dim

    def get_num_samples(self) -> int:
        """获取当前文件中已成功写入的样本数"""
        if self.h5 is None:
            return 0
        return int(self.h5["waveforms"]["wellhead_head"].shape[0])

    def append_batch(self, results: List[Dict[str, Any]]) -> int:
        """
        向 HDF5 文件追加一批正演仿真样本，自动扩展数据集大小并立即刷盘。
        """
        valid_results = [r for r in results if r.get("status") == "success"]
        if not valid_results:
            return self.get_num_samples()

        cur_n = self.get_num_samples()
        batch_n = len(valid_results)
        new_n = cur_n + batch_n

        # 首次动态注入 timestamps (若初始化时未传入)
        if "timestamps" not in self.h5["waveforms"]:
            ts = np.asarray(valid_results[0]["timestamps"], dtype=np.float32)
            self.h5["waveforms"].create_dataset("timestamps", data=ts)

        # 提取当前批次数据
        head_batch = np.zeros((batch_n, self.n_time), dtype=np.float32)
        vel_batch = np.zeros((batch_n, self.n_time), dtype=np.float32)
        ceps_batch = np.zeros((batch_n, self.n_dist), dtype=np.float32) if self.export_features else None

        n_frac_batch = np.zeros(batch_n, dtype=np.int32)
        pos_batch = np.zeros((batch_n, self.max_frac_dim), dtype=np.float32)
        cf_batch = np.zeros((batch_n, self.max_frac_dim), dtype=np.float32)
        kleak_batch = np.zeros((batch_n, self.max_frac_dim), dtype=np.float32)
        weights_batch = np.zeros((batch_n, self.max_frac_dim), dtype=np.float32)
        kp_batch = np.zeros((batch_n, self.max_frac_dim), dtype=np.float32)
        type_ids_batch = np.zeros((batch_n, self.max_frac_dim), dtype=np.int32)
        type_names_batch: List[List[str]] = [[""] * self.max_frac_dim for _ in range(batch_n)]

        tc_batch = np.zeros(batch_n, dtype=np.float32)
        wavespeed_batch = np.zeros(batch_n, dtype=np.float32)
        hext_batch = np.zeros(batch_n, dtype=np.float32)
        v0_batch = np.zeros(batch_n, dtype=np.float32)
        h0_batch = np.zeros(batch_n, dtype=np.float32)
        has_fault_batch = np.zeros(batch_n, dtype=np.int32)
        fault_idx_batch = np.full(batch_n, -1, dtype=np.int32)

        for i, r in enumerate(valid_results):
            wh_h = np.asarray(r["wellhead_head"], dtype=np.float32)
            len_h = min(self.n_time, len(wh_h))
            head_batch[i, :len_h] = wh_h[:len_h]

            if r.get("wellhead_velocity") is not None:
                wh_v = np.asarray(r["wellhead_velocity"], dtype=np.float32)
                len_v = min(self.n_time, len(wh_v))
                vel_batch[i, :len_v] = wh_v[:len_v]

            if self.export_features and "cepstrum" in r and r["cepstrum"] is not None and len(r["cepstrum"]) > 0:
                c_len = min(self.n_dist, len(r["cepstrum"]))
                ceps_batch[i, :c_len] = np.asarray(r["cepstrum"][:c_len], dtype=np.float32)

            meta = r.get("sample_meta", {})
            raw_pos = meta.get("fracture_positions", [])
            nf = min(self.max_frac_dim, len(raw_pos))
            n_frac_batch[i] = len(raw_pos)

            if nf > 0:
                pos_batch[i, :nf] = np.asarray(raw_pos[:nf], dtype=np.float32)
                if "fracture_Cf" in meta and meta["fracture_Cf"] is not None:
                    cf_batch[i, :nf] = np.asarray(meta["fracture_Cf"][:nf], dtype=np.float32)
                if "fracture_kleak" in meta and meta["fracture_kleak"] is not None:
                    kleak_batch[i, :nf] = np.asarray(meta["fracture_kleak"][:nf], dtype=np.float32)
                if "fracture_inflow_weights" in meta and meta["fracture_inflow_weights"] is not None:
                    weights_batch[i, :nf] = np.asarray(meta["fracture_inflow_weights"][:nf], dtype=np.float32)
                if "fracture_Kp" in meta and meta["fracture_Kp"] is not None:
                    kp_batch[i, :nf] = np.asarray(meta["fracture_Kp"][:nf], dtype=np.float32)

                ftypes = meta.get("fracture_types", [])
                for j in range(min(nf, len(ftypes))):
                    type_names_batch[i][j] = str(ftypes[j])
                    type_ids_batch[i, j] = map_fracture_type_to_id(ftypes[j])

            tc_batch[i] = float(meta.get("pump_closure_duration", 1.0))
            wavespeed_batch[i] = float(meta.get("wavespeed", 1450.0))
            hext_batch[i] = float(meta.get("H_ext", 100.0))
            v0_batch[i] = float(meta.get("initial_velocity", 1.0))
            h0_batch[i] = float(meta.get("initial_head", 300.0))
            has_fault_batch[i] = int(bool(meta.get("has_fault", False)))
            fault_idx_batch[i] = int(meta.get("fault_cluster_idx", -1))

        # 扩展并填充数据集
        grp_wave = self.h5["waveforms"]
        grp_wave["wellhead_head"].resize((new_n, self.n_time))
        grp_wave["wellhead_head"][cur_n:new_n] = head_batch
        grp_wave["wellhead_velocity"].resize((new_n, self.n_time))
        grp_wave["wellhead_velocity"][cur_n:new_n] = vel_batch

        if self.export_features and "features" in self.h5 and ceps_batch is not None:
            grp_feat = self.h5["features"]
            grp_feat["cepstrum"].resize((new_n, self.n_dist))
            grp_feat["cepstrum"][cur_n:new_n] = ceps_batch

        grp_lab = self.h5["labels"]
        grp_lab["n_frac"].resize((new_n,))
        grp_lab["n_frac"][cur_n:new_n] = n_frac_batch
        grp_lab["fracture_positions"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_positions"][cur_n:new_n] = pos_batch
        grp_lab["fracture_Cf"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_Cf"][cur_n:new_n] = cf_batch
        grp_lab["fracture_kleak"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_kleak"][cur_n:new_n] = kleak_batch
        grp_lab["fracture_weights"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_weights"][cur_n:new_n] = weights_batch
        grp_lab["fracture_Kp"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_Kp"][cur_n:new_n] = kp_batch
        grp_lab["fracture_type_ids"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_type_ids"][cur_n:new_n] = type_ids_batch
        grp_lab["fracture_types"].resize((new_n, self.max_frac_dim))
        grp_lab["fracture_types"][cur_n:new_n] = np.array(type_names_batch, dtype=object)

        grp_lab["pump_closure_tc"].resize((new_n,))
        grp_lab["pump_closure_tc"][cur_n:new_n] = tc_batch
        grp_lab["wavespeed"].resize((new_n,))
        grp_lab["wavespeed"][cur_n:new_n] = wavespeed_batch
        grp_lab["H_ext"].resize((new_n,))
        grp_lab["H_ext"][cur_n:new_n] = hext_batch
        grp_lab["initial_velocity"].resize((new_n,))
        grp_lab["initial_velocity"][cur_n:new_n] = v0_batch
        grp_lab["initial_head"].resize((new_n,))
        grp_lab["initial_head"][cur_n:new_n] = h0_batch
        grp_lab["has_fault"].resize((new_n,))
        grp_lab["has_fault"][cur_n:new_n] = has_fault_batch
        grp_lab["fault_cluster_idx"].resize((new_n,))
        grp_lab["fault_cluster_idx"][cur_n:new_n] = fault_idx_batch

        self.h5.attrs["n_samples"] = new_n
        self.h5.flush()
        return new_n

    def finalize(self, metadata: Optional[Dict[str, Any]] = None):
        """写入元数据并关闭 HDF5 文件"""
        if self.h5 is not None:
            if metadata:
                self.h5.attrs["metadata_json"] = json.dumps(metadata)
            self.h5.flush()
            self.h5.close()
            self.h5 = None

    def close(self):
        """安全关闭文件句柄"""
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


def load_hdf5_dataset(file_path: str) -> Dict[str, Any]:
    """
    读取并解析 HDF5 数据集文件内容为内存字典。
    """
    if not HAS_H5PY:
        raise RuntimeError("读取 HDF5 数据集需要安装 h5py 库 (pip install h5py)")

    dataset: Dict[str, Any] = {
        "waveforms": {},
        "features": {},
        "labels": {},
        "metadata": {},
    }

    with h5py.File(file_path, "r") as h5:
        for k in h5.attrs:
            val = h5.attrs[k]
            if k == "metadata_json":
                dataset["metadata"] = json.loads(val)
            else:
                dataset["metadata"][k] = val

        for grp_name in ("waveforms", "features", "labels"):
            if grp_name in h5:
                grp = h5[grp_name]
                for ds_name in grp:
                    ds = grp[ds_name]
                    # 检查是否为字符串数据集
                    if h5py.check_string_dtype(ds.dtype) is not None:
                        dataset[grp_name][ds_name] = ds.asstr()[:]
                    else:
                        dataset[grp_name][ds_name] = ds[:]

    return dataset
