# -*- coding: utf-8 -*-
"""
路线 A：已知子波 h(t) 的 ℓ₁ 正则化反卷积。
"""
from __future__ import annotations

import os
import sys
import numpy as np

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

from analysis.sparse_deconv.solvers import fista_solve
from analysis.sparse_deconv.evaluation import extract_peaks


class L1Deconvolver:
    """已知子波 h(t) 的 ℓ₁ 正则化反卷积器"""
    
    def __init__(self, h: np.ndarray, fs: float, a_adj: float):
        self.h = h
        self.fs = fs
        self.a_adj = a_adj
    
    def _count_peaks(self, r: np.ndarray) -> int:
        """计算显著峰的数量"""
        r_max = np.max(np.abs(r)) if len(r) > 0 else 0.0
        peaks = extract_peaks(r, depth_axis=np.zeros_like(r), height_thr=max(1e-6, r_max * 1e-2))
        return len(peaks)
        
    def find_lambda_for_n_peaks(self, y: np.ndarray, n_target: int, h: np.ndarray = None,
                                lambda_range: tuple = (1e-6, 1e2),
                                max_search_iter: int = 30) -> float:
        """
        二分搜索 λ，使 FISTA 解 r 的非零峰数 = n_target
        更大 λ -> 越稀疏 -> 峰越少
        """
        if h is None:
            h = self.h
            
        lo, hi = lambda_range
        
        # 先快速验证边界
        r_hi = fista_solve(y, h, lam=hi, max_iter=200)
        if self._count_peaks(r_hi) >= n_target:
            return hi # 最大lambda也压不住，直接返回最大
            
        r_lo = fista_solve(y, h, lam=lo, max_iter=200)
        if self._count_peaks(r_lo) <= n_target:
            return lo # 最小lambda也检不出足够峰，返回最小
            
        best_lam = (lo + hi) / 2
        best_diff = np.inf
        
        for _ in range(max_search_iter):
            mid = (lo + hi) / 2
            r = fista_solve(y, h, lam=mid, max_iter=500)
            n_peaks = self._count_peaks(r)
            
            diff = abs(n_peaks - n_target)
            if diff < best_diff:
                best_diff = diff
                best_lam = mid
                
            if n_peaks == n_target:
                return mid
            elif n_peaks > n_target:
                lo = mid    # 峰太多，增大 λ
            else:
                hi = mid    # 峰太少，减小 λ
                
        return best_lam

    def solve(self, delta_y: np.ndarray, n_fracs: int, fixed_lambda: float = None) -> dict:
        """
        对差信号执行 ℓ₁ 反卷积。
        
        1. 二分搜索 λ，使非零峰数 = n_fracs
        2. FISTA 求解
        
        Returns
        -------
        dict:
            r           : 稀疏解 shape (N,)
            depth_axis  : 深度轴 [m]
            lambda_used : 使用的 λ 值
        """
        pulse_len = len(self.h) - 10
        y_trunc = delta_y[:pulse_len]
        h_trunc = self.h[:pulse_len]
        
        if fixed_lambda is not None:
            lam = fixed_lambda
        else:
            # 取导数，将病态的阶跃卷积转化为良态的脉冲卷积
            # 注意：必须使用 prepend=0，因为 h(t) 是在 t=0 瞬间发生的阶跃，
            # 若用 np.gradient 会丢失 t=0 的巨大脉冲！
            dy = np.diff(y_trunc, prepend=0)
            dh = np.diff(h_trunc, prepend=0)
            lam = self.find_lambda_for_n_peaks(dy, n_fracs, h=dh)
            
        dy = np.diff(y_trunc, prepend=0)
        dh = np.diff(h_trunc, prepend=0)
        r = fista_solve(dy, dh, lam=lam, max_iter=2000)
        
        n = len(r)
        t_axis = np.arange(n) / self.fs
        depth_axis = t_axis * self.a_adj / 2.0
        
        return {
            'r': r,
            'depth_axis': depth_axis,
            'lambda_used': lam,
        }
