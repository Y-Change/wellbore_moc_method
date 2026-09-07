# -*- coding: utf-8 -*-
"""Hand-checkable ZVB recursion: NumPy vs Tensor, then full frozen kernel."""
from __future__ import annotations

import numpy as np
import torch

from kernel_loader import load_frozen_kernel
from stage1_bridge import mk


def phi1_t(h: torch.Tensor) -> torch.Tensor:
    small = h < 1e-6
    return torch.where(small, 1.0 - 0.5 * h, (1.0 - torch.exp(-h)) / h.clamp_min(1e-30))


def recurse_np(dV, E, Phi, w):
    z = np.zeros(E.shape[-1] if E.ndim == 1 else E.shape[0], dtype=np.float64)
    if E.ndim == 1:
        J = np.empty(dV.size)
        for i in range(dV.size):
            z = E * z + Phi * dV[i]
            J[i] = w * z.sum()
        return J, z
    raise ValueError("E must be 1d for unit test")


def recurse_t(dV, E, Phi, w):
    dV = torch.as_tensor(dV, dtype=torch.float64)
    E = torch.as_tensor(E, dtype=torch.float64)
    Phi = torch.as_tensor(Phi, dtype=torch.float64)
    z = torch.zeros_like(E)
    Js = []
    for i in range(dV.numel()):
        z = E * z + Phi * dV[i]
        Js.append(w * z.sum())
    return torch.stack(Js), z


def unit_one_term_hand():
    """One location, one term, two steps. Closed form."""
    E = np.array([0.5])
    Phi = np.array([2.0])
    w = 3.0
    dV = np.array([1.0, -0.25])
    # z1 = 0.5*0 + 2*1 = 2; J1 = 6
    # z2 = 0.5*2 + 2*(-0.25) = 0.5; J2 = 1.5
    J_np, z_np = recurse_np(dV, E, Phi, w)
    J_t, z_t = recurse_t(dV, E, Phi, w)
    expect = np.array([6.0, 1.5])
    return {
        "expect_J": expect.tolist(),
        "numpy_J": J_np.tolist(),
        "tensor_J": J_t.detach().cpu().numpy().tolist(),
        "numpy_z": z_np.tolist(),
        "tensor_z": z_t.detach().cpu().numpy().tolist(),
        "pass": bool(np.allclose(J_np, expect) and np.allclose(J_t.detach().cpu().numpy(), expect)
                     and np.allclose(z_np, z_t.detach().cpu().numpy())),
        "driver": "dV = V^{n+1}-V^n",
        "update": "z^{n+1}=E z^n + Phi dV",
        "Ju": "w * sum z",
        "z0": 0.0,
    }


def full_kernel_numpy_vs_tensor(n_steps=40, seed=0):
    frozen = load_frozen_kernel()
    rng = np.random.default_rng(seed)
    dV = rng.normal(0.0, 1e-3, size=n_steps)
    nu, D, dt = 1.0e-6, 0.1, 1e-3
    rk = mk.RecursiveKernel.build(frozen.zielke_m, frozen.zielke_n, nu, D, dt)
    J_np, z_np = recurse_np(dV, rk.E, rk.Phi, rk.w)
    J_t, z_t = recurse_t(dV, rk.E, rk.Phi, rk.w)
    Jt = J_t.detach().cpu().numpy()
    zt = z_t.detach().cpu().numpy()
    return {
        "M": int(rk.E.size),
        "max_abs_J": float(np.max(np.abs(J_np - Jt))),
        "max_abs_z": float(np.max(np.abs(z_np - zt))),
        "rel_J": float(np.linalg.norm(J_np - Jt) / max(np.linalg.norm(J_np), 1e-30)),
        "pass": bool(np.allclose(J_np, Jt) and np.allclose(z_np, zt)),
        "kernel_sha": frozen.yaml_sha256,
        "used_online_refit": False,
        "E_sample": rk.E[:3].tolist(),
        "w": float(rk.w),
        "dt": dt,
    }
