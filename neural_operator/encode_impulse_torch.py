# -*- coding: utf-8 -*-
"""
encode_impulse_torch.py — FNO 4 通道脉冲编码的可微 Torch 实现。

与 ``FracturingMOCSurrogateDataset._encode_impulse_trains`` / ``__getitem__``
逐元素对齐，使位置 ``x_f`` 可通过 autograd 反传。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[Sequence[float], np.ndarray, torch.Tensor]


def _as_1d_tensor(x: ArrayLike, *, device, dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype).reshape(-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=device, dtype=dtype).reshape(-1)


def encode_impulse_trains(
    t_target: torch.Tensor,
    x_f: torch.Tensor,
    Cf: torch.Tensor,
    kleak: torch.Tensor,
    *,
    ts: float = 1.0,
    wavespeed: float = 1450.0,
    sigma_impulse_s: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """可微高斯脉冲串（ch2 / ch3 内核）。

    Parameters
    ----------
    t_target : (T,)
    x_f, Cf, kleak : (n_frac,)  — ``x_f`` 可带 grad
    """
    t = t_target.reshape(-1)
    xf = x_f.reshape(-1)
    cf = Cf.reshape(-1)
    kl = kleak.reshape(-1)
    if not (xf.numel() == cf.numel() == kl.numel()):
        raise ValueError("x_f / Cf / kleak 长度须一致")

    t_arr = ts + (2.0 * xf) / float(wavespeed)
    gauss = torch.exp(
        -((t.unsqueeze(0) - t_arr.unsqueeze(1)) ** 2) / (2.0 * float(sigma_impulse_s) ** 2)
    )
    w_cf = torch.log10(torch.clamp(cf, min=1.0e-12)) + 12.0
    w_kl = torch.log10(torch.clamp(kl, min=1.0e-15)) + 15.0
    impulse_cf = (w_cf.unsqueeze(1) * gauss).sum(dim=0)
    impulse_kl = (w_kl.unsqueeze(1) * gauss).sum(dim=0)
    return impulse_cf, impulse_kl


def encode_fno_input(
    x_f: ArrayLike,
    Cf: ArrayLike,
    kleak: ArrayLike,
    *,
    n_time: int = 4096,
    tf: float = 50.0,
    ts: float = 1.0,
    wavespeed: float = 1450.0,
    sigma_impulse_s: float = 0.15,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造 FNO 输入 ``(1, 4, T)`` 与时间轴 ``(T,)``。

    通道：ch0=t/tf，ch1=(t<ts)，ch2=Cf 脉冲，ch3=kleak 脉冲。
    """
    if device is None:
        if isinstance(x_f, torch.Tensor):
            device = x_f.device
        else:
            device = torch.device("cpu")

    xf = _as_1d_tensor(x_f, device=device, dtype=dtype)
    cf = _as_1d_tensor(Cf, device=device, dtype=dtype)
    kl = _as_1d_tensor(kleak, device=device, dtype=dtype)
    t_target = torch.linspace(0.0, float(tf), int(n_time), device=device, dtype=dtype)

    ch0 = t_target / max(float(tf), 1.0)
    ch1 = (t_target < float(ts)).to(dtype=dtype)
    ch2, ch3 = encode_impulse_trains(
        t_target,
        xf,
        cf,
        kl,
        ts=float(ts),
        wavespeed=float(wavespeed),
        sigma_impulse_s=float(sigma_impulse_s),
    )
    x_in = torch.stack([ch0, ch1, ch2, ch3], dim=0).unsqueeze(0)  # (1,4,T)
    return x_in, t_target


def encode_fno_input_numpy(
    x_f: ArrayLike,
    Cf: ArrayLike,
    kleak: ArrayLike,
    *,
    n_time: int = 4096,
    tf: float = 50.0,
    ts: float = 1.0,
    wavespeed: float = 1450.0,
    sigma_impulse_s: float = 0.15,
) -> np.ndarray:
    """NumPy 对照（与 Dataset 一致），返回 ``(4, T)`` float32。"""
    from neural_operator.dataset_surrogate import FracturingMOCSurrogateDataset

    ds = object.__new__(FracturingMOCSurrogateDataset)
    ds.wavespeed = float(wavespeed)
    ds.ts = float(ts)
    ds.sigma = float(sigma_impulse_s)
    t_target = np.linspace(0.0, float(tf), int(n_time), dtype=np.float32)
    xf = np.asarray(x_f, dtype=np.float64).reshape(-1)
    cf = np.asarray(Cf, dtype=np.float64).reshape(-1)
    kl = np.asarray(kleak, dtype=np.float64).reshape(-1)
    ch2, ch3 = ds._encode_impulse_trains(t_target, xf, cf, kl)
    ch0 = (t_target / max(float(tf), 1.0)).astype(np.float32)
    ch1 = (t_target < float(ts)).astype(np.float32)
    return np.stack([ch0, ch1, ch2, ch3], axis=0)


def interp1d_torch(
    t_src: torch.Tensor,
    y_src: torch.Tensor,
    t_query: torch.Tensor,
) -> torch.Tensor:
    """可微分段线性插值。``t_src`` 须严格递增。"""
    t_src = t_src.reshape(-1)
    y_src = y_src.reshape(-1)
    tq = t_query.reshape(-1)
    idx = torch.searchsorted(t_src, tq, right=False)
    idx1 = torch.clamp(idx, 1, t_src.numel() - 1)
    idx0 = idx1 - 1
    t0, t1 = t_src[idx0], t_src[idx1]
    y0, y1 = y_src[idx0], y_src[idx1]
    denom = (t1 - t0).clamp_min(1e-12)
    w = (tq - t0) / denom
    return y0 + w * (y1 - y0)


def butterworth_lowpass_torch(
    y: torch.Tensor,
    dt: float,
    fc: float,
    order: int = 4,
) -> torch.Tensor:
    """零相位 Butterworth 低通（与 ``crb_core._butterworth_lowpass`` 同构造）。"""
    y = y.reshape(-1)
    n = int(y.numel())
    if n < 3 or fc <= 0:
        return y
    pad = min(n, max(64, int(round(4.0 / (float(fc) * float(dt))))))
    # NumPy: left = 2*y[0] - y[pad:0:-1]  → indices pad..1
    left = 2.0 * y[0] - torch.flip(y[1 : pad + 1], dims=(0,))
    # NumPy: right = 2*y[-1] - y[-2:-pad-2:-1] → last pad samples mirrored
    right = 2.0 * y[-1] - torch.flip(y[-pad - 1 : -1], dims=(0,))
    ext = torch.cat([left, y, right], dim=0)
    m = int(ext.numel())
    freqs = torch.fft.rfftfreq(m, d=float(dt), device=y.device, dtype=y.dtype)
    gain = 1.0 / torch.sqrt(1.0 + (freqs / float(fc)) ** (2 * int(order)))
    Y = torch.fft.rfft(ext)
    filt = torch.fft.irfft(Y * gain.to(dtype=Y.real.dtype), n=m)
    return filt[left.numel() : left.numel() + n]


def observe_torch(
    H_wh: torch.Tensor,
    t: torch.Tensor,
    *,
    ts: float,
    dt: float,
    fc_hz: float = 20.0,
    order: int = 4,
    t_end: Optional[float] = None,
) -> torch.Tensor:
    """可微观测：停泵后截取 + 可选低通。"""
    t = t.reshape(-1)
    H = H_wh.reshape(-1)
    t1 = float(t[-1] if t_end is None else t_end)
    mask = (t >= float(ts) - 1e-12) & (t <= t1 + 1e-12)
    seg = H[mask]
    if fc_hz is not None and fc_hz > 0:
        nyq = 0.5 / float(dt)
        if float(fc_hz) < nyq:
            seg = butterworth_lowpass_torch(seg, float(dt), float(fc_hz), int(order))
    return seg
