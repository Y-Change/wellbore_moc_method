# -*- coding: utf-8 -*-
"""
CJ-AlphaNet 数据集：停泵后 60 s 原生波形 + 1D/2D 倒谱，标签来自离线 npz。

- 不读取 HDF5 ``labels/wavespeed`` 作为网络条件
- 倒谱缓存键绑定 â 与 Kaiser/窗长设定，避免与旧 4096/1024 Pilot 缓存混用
- 2D 倒谱在簇深度 x_j 上取样后进入 batch（原生 41×Nq 仅在构缓存时短暂存在）
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from moc_simulate.v2.batch.torch_dataset import split_dataset_indices
from moc_simulate.v2.signal.cepstrum_1d import quefrency_to_distance, real_cepstrum
from PaperC_CJNO_Wellbore_Inversion.src.dataset import fingerprint_h5_source
from PaperC_CJNO_Wellbore_Inversion.src.label_physics import (
    A_HAT_MAX,
    A_HAT_MIN,
    L_DEFAULT,
    N_GRID_DEFAULT,
    POST_SHUT_DURATION_S,
    extract_post_shut_head,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PAPERC_DIR = Path(__file__).resolve().parents[1]

DEFAULT_H5_PATH = str(
    _PROJECT_ROOT
    / "data"
    / "datasets"
    / "moc_v2_physical_steady_1k_newa"
    / "moc_v2_physical_steady_1k_x4500_4950.h5"
)
DEFAULT_NPZ_PATH = str(
    _PROJECT_ROOT
    / "data"
    / "datasets"
    / "moc_v2_physical_steady_1k_newa"
    / "case_labels_alpha_y.npz"
)
DEFAULT_CACHE_DIR = str(_PAPERC_DIR / "data" / "alpha_cache")

MAX_CLUSTERS = 6
TS_DEFAULT = 1.0
A_HAT_COND_CENTER = 1400.0
A_HAT_COND_SCALE = 200.0
CEP2D_WINDOW_S = 20.0
CEP2D_HOP_S = 1.0
CEP_WINDOW = ("kaiser", 14.0)
N_TIME_DS = 4096
N_CEPS_DS = 1024
N_CEP2D_FRAMES = 41
NATIVE_CEPS_PAD = 8192

# 明确不把设计波速当输入
USES_DESIGN_WAVESPEED = False


def normalize_a_hat(a_hat: np.ndarray) -> np.ndarray:
    a = np.asarray(a_hat, dtype=np.float32)
    return ((a - A_HAT_COND_CENTER) / A_HAT_COND_SCALE).astype(np.float32)


def _sha16(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fingerprint_npz_labels(npz_path: str) -> Dict[str, Any]:
    abs_path = os.path.abspath(npz_path)
    st = os.stat(abs_path)
    data = np.load(abs_path)
    a_hat = np.asarray(data["a_hat"])
    pos = np.asarray(data["positions"])
    token = hashlib.sha256(a_hat.tobytes() + pos.tobytes()).hexdigest()[:16]
    digest = _sha16(
        "|".join(
            [
                abs_path.replace("\\", "/"),
                str(int(st.st_size)),
                token,
                str(int(a_hat.shape[0])),
            ]
        )
    )
    return {"path": abs_path, "digest": digest, "n_samples": int(a_hat.shape[0])}


def alpha_cache_dirname(
    cache_dir: str,
    h5_digest: str,
    npz_digest: str,
    ts: float,
    duration: float,
    window_beta: float,
    win_s: float,
    hop_s: float,
    L: float,
) -> str:
    key = (
        f"alpha_h{h5_digest}_z{npz_digest}_ts{ts:g}_dur{duration:g}"
        f"_kaiser{window_beta:g}_w{win_s:g}_h{hop_s:g}_L{L:g}_native"
    )
    return os.path.join(cache_dir, key)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _sample_cepstrum_at_x(
    distance: np.ndarray,
    values: np.ndarray,
    x_query: np.ndarray,
) -> np.ndarray:
    """values: (nq,) or (n_frames, nq) -> (nq_q,) or (n_frames, n_q)."""
    dist = np.asarray(distance, dtype=np.float64).reshape(-1)
    xq = np.asarray(x_query, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64)
    if dist.size < 2:
        if vals.ndim == 1:
            return np.zeros((xq.size,), dtype=np.float32)
        return np.zeros((vals.shape[0], xq.size), dtype=np.float32)
    order = np.argsort(dist)
    dist = dist[order]
    if vals.ndim == 1:
        y = np.interp(xq, dist, vals[order])
        return y.astype(np.float32)
    out = np.zeros((vals.shape[0], xq.size), dtype=np.float32)
    for i in range(vals.shape[0]):
        out[i] = np.interp(xq, dist, vals[i, order]).astype(np.float32)
    return out


def compute_native_cepstrum_1d(
    head_post: np.ndarray,
    dt: float,
    a_hat: float,
    L: float = L_DEFAULT,
    window=CEP_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """原生倒频率 → x=â τ/2，只留 x∈[0,L]，不做 1024 插值。"""
    fs = 1.0 / float(dt)
    ceps, q = real_cepstrum(np.asarray(head_post, dtype=np.float64), fs=fs, window=window)
    dist = quefrency_to_distance(q, float(a_hat))
    keep = dist <= float(L) + 1e-9
    return dist[keep].astype(np.float32), ceps[keep].astype(np.float64)


def compute_native_cepstrogram_2d(
    head_post: np.ndarray,
    t_post: np.ndarray,
    dt: float,
    a_hat: float,
    L: float = L_DEFAULT,
    window_len_s: float = CEP2D_WINDOW_S,
    hop_len_s: float = CEP2D_HOP_S,
    window=CEP_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    滑窗短时倒谱，不做全时程三次插值。
    返回 (time_centers, distances, cepstrogram[n_frames, nq]).
    """
    h = np.asarray(head_post, dtype=np.float64).reshape(-1)
    t = np.asarray(t_post, dtype=np.float64).reshape(-1)
    fs = 1.0 / float(dt)
    n = int(h.size)
    duration = float(t[-1] - t[0]) if t.size > 1 else float(n) * float(dt)
    win_samples = int(round(n * float(window_len_s) / max(duration, 1e-6)))
    win_samples = max(8, min(win_samples, n))
    n_frames_target = int(np.floor((duration - float(window_len_s)) / float(hop_len_s) + 1e-9)) + 1
    n_frames_target = max(1, n_frames_target)

    frames = []
    centers = []
    ref_dist = None
    for k in range(n_frames_target):
        t_start = k * float(hop_len_s)
        start = int(round(t_start / float(dt)))
        if start + win_samples > n:
            start = max(0, n - win_samples)
        seg = h[start : start + win_samples]
        mid = start + win_samples // 2
        centers.append(float(t[mid] if mid < t.size else t[-1]))
        ceps, q = real_cepstrum(seg, fs=fs, window=window)
        dist = quefrency_to_distance(q, float(a_hat))
        keep = dist <= float(L) + 1e-9
        dist = dist[keep]
        ceps = ceps[keep]
        if ref_dist is None:
            ref_dist = dist
        n_use = min(int(ref_dist.size), int(ceps.size))
        frames.append(ceps[:n_use])

    if not frames:
        ceps, q = real_cepstrum(h, fs=fs, window=window)
        dist = quefrency_to_distance(q, float(a_hat))
        keep = dist <= float(L) + 1e-9
        ref_dist = dist[keep]
        frames = [ceps[keep]]
        centers = [float(np.mean(t))]

    nq = int(min(len(f) for f in frames))
    mat = np.stack([f[:nq] for f in frames], axis=0).astype(np.float32)
    return (
        np.asarray(centers, dtype=np.float32),
        np.asarray(ref_dist[:nq], dtype=np.float32),
        mat,
    )


def _pad_1d(values: np.ndarray, n_pad: int) -> np.ndarray:
    out = np.zeros((n_pad,), dtype=np.float32)
    n = min(int(values.size), n_pad)
    if n > 0:
        out[:n] = np.asarray(values[:n], dtype=np.float32)
    return out


def build_alpha_feature_cache(
    h5_path: str,
    npz_path: str,
    cache_dir: str,
    ts: float = TS_DEFAULT,
    duration: float = POST_SHUT_DURATION_S,
    L: float = L_DEFAULT,
    force: bool = False,
) -> str:
    """从 h5 波形 + npz 的 â 构建倒谱/波形缓存，不改 h5。"""
    h5_meta = fingerprint_h5_source(h5_path)
    npz_meta = fingerprint_npz_labels(npz_path)
    out_dir = alpha_cache_dirname(
        cache_dir,
        h5_meta["digest"],
        npz_meta["digest"],
        ts=ts,
        duration=duration,
        window_beta=14.0,
        win_s=CEP2D_WINDOW_S,
        hop_s=CEP2D_HOP_S,
        L=L,
    )
    meta_path = os.path.join(out_dir, "cache_meta.json")
    wave_path = os.path.join(out_dir, "wave.npy")
    if os.path.isfile(meta_path) and os.path.isfile(wave_path) and not force:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("h5_digest") == h5_meta["digest"]
            and meta.get("npz_digest") == npz_meta["digest"]
            and int(meta.get("n_samples", -1)) == int(h5_meta["n_samples"])
        ):
            return out_dir

    os.makedirs(out_dir, exist_ok=True)
    labels = np.load(npz_path)
    a_hat_all = np.asarray(labels["a_hat"], dtype=np.float32)
    pos_all = np.asarray(labels["positions"], dtype=np.float32)
    mask_all = np.asarray(labels["mask_design"], dtype=np.int32)
    n_samples = int(a_hat_all.shape[0])
    if n_samples != int(h5_meta["n_samples"]):
        raise RuntimeError(
            f"npz 样本数 {n_samples} 与 HDF5 {h5_meta['n_samples']} 不一致"
        )

    with h5py.File(h5_path, "r") as h5:
        heads = np.asarray(h5["waveforms/wellhead_head"][:], dtype=np.float32)
        timestamps = np.asarray(h5["waveforms/timestamps"][:], dtype=np.float64)

    t0, h0 = extract_post_shut_head(timestamps, heads[0], ts=ts, duration=duration)
    n_time = int(h0.size)
    if t0.size >= 2:
        dt_ref = float(np.median(np.diff(t0)))
    else:
        dt_ref = 1.0e-3

    wave = np.zeros((n_samples, 2, n_time), dtype=np.float32)
    wave_4096 = np.zeros((n_samples, 2, N_TIME_DS), dtype=np.float32)
    cep1d_pad = np.zeros((n_samples, NATIVE_CEPS_PAD), dtype=np.float32)
    cep1d_nq = np.zeros((n_samples,), dtype=np.int32)
    cep1d_at_xj = np.zeros((n_samples, MAX_CLUSTERS), dtype=np.float32)
    cep2d_at_xj = np.zeros((n_samples, N_CEP2D_FRAMES, MAX_CLUSTERS), dtype=np.float32)
    cep1024 = np.zeros((n_samples, N_CEPS_DS), dtype=np.float32)
    n_frames_arr = np.zeros((n_samples,), dtype=np.int32)
    idx_4096 = np.linspace(0, n_time - 1, N_TIME_DS).astype(np.int64)
    x_grid_1024 = np.linspace(0.0, float(L), N_CEPS_DS, dtype=np.float64)

    print(f"[alpha-cache] 构建 {n_samples} 例停泵后波形/倒谱 -> {out_dir}")
    for i in range(n_samples):
        t_post, h_post = extract_post_shut_head(
            timestamps, heads[i], ts=ts, duration=duration
        )
        if int(h_post.size) != n_time:
            # 对齐到固定长度
            if h_post.size >= n_time:
                h_post = h_post[:n_time]
                t_post = t_post[:n_time]
            else:
                pad = n_time - int(h_post.size)
                h_post = np.pad(h_post, (0, pad), mode="edge")
                dt_i = float(np.median(np.diff(t_post))) if t_post.size > 1 else dt_ref
                extra = t_post[-1] + dt_i * np.arange(1, pad + 1)
                t_post = np.concatenate([t_post, extra])
        dt_i = float(np.median(np.diff(t_post))) if t_post.size > 1 else dt_ref
        dH = np.gradient(h_post.astype(np.float64), dt_i)
        wave[i, 0] = _zscore(h_post)
        wave[i, 1] = _zscore(dH)
        wave_4096[i] = wave[i][:, idx_4096]

        a_hat = float(np.clip(a_hat_all[i], A_HAT_MIN, A_HAT_MAX))
        dist1, ceps1 = compute_native_cepstrum_1d(h_post, dt_i, a_hat, L=L)
        ceps1_z = _zscore(ceps1)
        nq = int(min(ceps1_z.size, NATIVE_CEPS_PAD))
        cep1d_nq[i] = nq
        cep1d_pad[i, :nq] = ceps1_z[:nq]
        xj = pos_all[i]
        mj = mask_all[i] > 0
        sampled_1d = _sample_cepstrum_at_x(dist1[:nq], ceps1_z[:nq], xj)
        sampled_1d = np.where(mj, sampled_1d, 0.0).astype(np.float32)
        cep1d_at_xj[i] = sampled_1d
        cep1024[i] = _sample_cepstrum_at_x(dist1[:nq], ceps1_z[:nq], x_grid_1024)

        _tc, dist2, ceps2 = compute_native_cepstrogram_2d(
            h_post, t_post, dt_i, a_hat, L=L
        )
        # 每样本 z-score 整张 2D 谱，再在 x_j 取列
        c2 = ceps2.astype(np.float64)
        c2 = (c2 - float(np.mean(c2))) / (float(np.std(c2)) + 1e-6)
        n_fr = int(c2.shape[0])
        n_frames_arr[i] = n_fr
        sampled_2d = _sample_cepstrum_at_x(dist2, c2.astype(np.float32), xj)
        # sampled_2d: (n_fr, M)
        if n_fr >= N_CEP2D_FRAMES:
            cep2d_at_xj[i] = sampled_2d[:N_CEP2D_FRAMES]
        else:
            cep2d_at_xj[i, :n_fr] = sampled_2d
        cep2d_at_xj[i] = np.where(mj.reshape(1, -1), cep2d_at_xj[i], 0.0)

        if (i + 1) % 50 == 0 or i == 0 or i + 1 == n_samples:
            print(f"    样本 {i + 1}/{n_samples}  nq={nq}  frames={n_fr}")

    np.save(os.path.join(out_dir, "wave.npy"), wave)
    np.save(os.path.join(out_dir, "wave_4096.npy"), wave_4096)
    np.save(os.path.join(out_dir, "cepstrum_1d_native.npy"), cep1d_pad)
    np.save(os.path.join(out_dir, "cepstrum_1d_nq.npy"), cep1d_nq)
    np.save(os.path.join(out_dir, "cepstrum_1d_at_xj.npy"), cep1d_at_xj)
    np.save(os.path.join(out_dir, "cepstrum_2d_at_xj.npy"), cep2d_at_xj)
    np.save(os.path.join(out_dir, "cepstrum_1024.npy"), cep1024)
    np.save(os.path.join(out_dir, "cepstrum_2d_n_frames.npy"), n_frames_arr)

    meta = {
        "h5_path": os.path.abspath(h5_path),
        "npz_path": os.path.abspath(npz_path),
        "h5_digest": h5_meta["digest"],
        "npz_digest": npz_meta["digest"],
        "n_samples": n_samples,
        "n_time": n_time,
        "dt_ref": dt_ref,
        "ts": ts,
        "duration": duration,
        "L": L,
        "cep_window": ["kaiser", 14.0],
        "cep2d_window_s": CEP2D_WINDOW_S,
        "cep2d_hop_s": CEP2D_HOP_S,
        "n_cep2d_frames_target": N_CEP2D_FRAMES,
        "native_ceps_pad": NATIVE_CEPS_PAD,
        "uses_design_wavespeed": False,
        "a_hat_from": "case_labels_alpha_y.npz",
        "note": "x = a_hat * tau / 2; no interpolation to 1024 for AlphaNet path",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[alpha-cache] 完成 {out_dir}")
    return out_dir


class AlphaInversionDataset(Dataset):
    """
    newa 物理稳态 1k：800/100/100，seed=42。
    标签全部来自 case_labels_alpha_y.npz，波形来自 newa HDF5。
    """

    MAX_CLUSTERS = MAX_CLUSTERS
    L_DEFAULT = L_DEFAULT
    USES_DESIGN_WAVESPEED = False

    def __init__(
        self,
        h5_path: str = DEFAULT_H5_PATH,
        npz_path: str = DEFAULT_NPZ_PATH,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        force_recompute: bool = False,
        ts: float = TS_DEFAULT,
        duration: float = POST_SHUT_DURATION_S,
        L: float = L_DEFAULT,
    ):
        super().__init__()
        self.h5_path = os.path.abspath(h5_path)
        self.npz_path = os.path.abspath(npz_path)
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"找不到 HDF5: {self.h5_path}")
        if not os.path.isfile(self.npz_path):
            raise FileNotFoundError(f"找不到标签 npz: {self.npz_path}")

        self.split = str(split).strip().lower()
        self.split_ratios = split_ratios
        self.seed = int(seed)
        self.ts = float(ts)
        self.duration = float(duration)
        self.L = float(L)
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

        labels = np.load(self.npz_path)
        n_total = int(labels["n_frac"].shape[0])
        h5_meta = fingerprint_h5_source(self.h5_path)
        if n_total != int(h5_meta["n_samples"]):
            raise RuntimeError("npz 与 HDF5 样本数不一致")

        splits = split_dataset_indices(n_total, self.split_ratios, self.seed)
        if self.split not in splits:
            raise ValueError(f"无效 split: {self.split}")
        self.indices = np.asarray(splits[self.split], dtype=np.int64)
        self.length = int(self.indices.size)
        self.raw_indices = self.indices

        cache_path = build_alpha_feature_cache(
            self.h5_path,
            self.npz_path,
            self.cache_dir,
            ts=self.ts,
            duration=self.duration,
            L=self.L,
            force=force_recompute,
        )
        self.cache_path = cache_path
        with open(os.path.join(cache_path, "cache_meta.json"), "r", encoding="utf-8") as f:
            self.cache_meta = json.load(f)

        idx = self.indices
        mmap = True
        self.waveforms = torch.from_numpy(
            np.load(os.path.join(cache_path, "wave.npy"), mmap_mode="r")[idx].copy()
        ).float()
        self.wave_4096 = torch.from_numpy(
            np.load(os.path.join(cache_path, "wave_4096.npy"), mmap_mode="r")[idx].copy()
        ).float()
        self.cepstrum_1d_native = torch.from_numpy(
            np.load(os.path.join(cache_path, "cepstrum_1d_native.npy"), mmap_mode="r")[idx].copy()
        ).float()
        self.cepstrum_1d_nq = torch.from_numpy(
            np.load(os.path.join(cache_path, "cepstrum_1d_nq.npy"))[idx].copy()
        ).long()
        self.cepstrum_1d_at_xj = torch.from_numpy(
            np.load(os.path.join(cache_path, "cepstrum_1d_at_xj.npy"))[idx].copy()
        ).float()
        self.cepstrum_2d_at_xj = torch.from_numpy(
            np.load(os.path.join(cache_path, "cepstrum_2d_at_xj.npy"))[idx].copy()
        ).float()
        self.cepstrum_1024 = torch.from_numpy(
            np.load(os.path.join(cache_path, "cepstrum_1024.npy"), mmap_mode="r")[idx].copy()
        ).float()

        self.n_frac = torch.from_numpy(np.asarray(labels["n_frac"])[idx]).long()
        self.positions = torch.from_numpy(np.asarray(labels["positions"])[idx]).float()
        self.norm_positions = self.positions / float(self.L)
        self.mask_design = torch.from_numpy(np.asarray(labels["mask_design"])[idx] > 0)
        self.alpha = torch.from_numpy(np.asarray(labels["alpha"])[idx]).float()
        self.y_act = torch.from_numpy(np.asarray(labels["y_active"])[idx]).float()
        self.Y_eq = torch.from_numpy(np.asarray(labels["Y_eq"])[idx]).float()
        self.logY = torch.from_numpy(np.asarray(labels["log10_Y_over_Y0"])[idx]).float()
        self.m_alpha_grid = torch.from_numpy(np.asarray(labels["m_alpha_grid"])[idx]).float()
        self.a_hat = torch.from_numpy(np.asarray(labels["a_hat"])[idx]).float()
        self.T1 = torch.from_numpy(np.asarray(labels["T1"])[idx]).float()
        self.Y0 = torch.from_numpy(np.asarray(labels["Y0"])[idx]).float()
        self.x1 = torch.from_numpy(np.asarray(labels["x1"])[idx]).float()
        self.omega_star = torch.from_numpy(np.asarray(labels["omega_star"])[idx]).float()
        self.grid_x = torch.from_numpy(np.asarray(labels["grid_x"])).float()
        self.cond = torch.from_numpy(normalize_a_hat(np.asarray(labels["a_hat"])[idx])).float().unsqueeze(-1)
        self.n_time = int(self.waveforms.shape[-1])
        self.n_grid = int(self.m_alpha_grid.shape[-1]) if self.m_alpha_grid.ndim == 2 else N_GRID_DEFAULT
        _ = mmap

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        mask = self.mask_design[i]
        item = {
            "wave": self.waveforms[i],
            "wave_4096": self.wave_4096[i],
            "cepstrum_1d": self.cepstrum_1d_native[i].unsqueeze(0),
            "cepstrum_1d_nq": self.cepstrum_1d_nq[i],
            "cepstrum_1d_at_xj": self.cepstrum_1d_at_xj[i],
            "cepstrum_2d_at_xj": self.cepstrum_2d_at_xj[i],
            "cepstrum": self.cepstrum_1024[i].unsqueeze(0),
            "cond": self.cond[i],
            "n_frac": self.n_frac[i],
            "positions": self.positions[i],
            "norm_positions": self.norm_positions[i],
            "mask_design": mask,
            "mask": mask,
            "alpha": self.alpha[i],
            "y_act": self.y_act[i],
            "Y_eq": self.Y_eq[i],
            "logY": self.logY[i],
            "m_alpha_grid": self.m_alpha_grid[i],
            "a_hat": self.a_hat[i],
            "T1": self.T1[i],
            "Y0": self.Y0[i],
            "x1": self.x1[i],
            "omega_star": self.omega_star[i],
            "sample_id": torch.tensor(int(self.indices[i]), dtype=torch.long),
        }
        return item

    def get_dataloader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
