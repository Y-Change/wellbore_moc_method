# -*- coding: utf-8 -*-
"""
路线 B：交替最小化盲稀疏反卷积（AM 算法）。
"""
from __future__ import annotations

import os
import sys
import numpy as np
from scipy.fft import fft, ifft

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

from analysis.sparse_deconv.solvers import fista_solve, pad_to_pow2
from analysis.sparse_deconv.evaluation import extract_peaks


class BlindSparseDeconvolver:
    """交替最小化盲稀疏反卷积，同时估计 h(t) 和 r(t)"""
    
    def __init__(self, fs: float, a_adj: float, L: float):
        self.fs = fs
        self.a_adj = a_adj
        self.L = L
        
    def _count_peaks(self, r: np.ndarray) -> int:
        r_max = np.max(np.abs(r)) if len(r) > 0 else 0.0
        peaks = extract_peaks(r, depth_axis=np.zeros_like(r), height_thr=max(1e-6, r_max * 1e-2))
        return len(peaks)
        
    def _find_lambda_for_n_peaks(self, delta_y: np.ndarray, h: np.ndarray, n_target: int, 
                                 lambda_range: tuple = (1e-6, 1e2),
                                 max_search_iter: int = 20) -> float:
        lo, hi = lambda_range
        
        # 边界验证
        r_hi = fista_solve(delta_y, h, lam=hi, max_iter=200)
        if self._count_peaks(r_hi) >= n_target:
            return hi
            
        r_lo = fista_solve(delta_y, h, lam=lo, max_iter=200)
        if self._count_peaks(r_lo) <= n_target:
            return lo
            
        best_lam = (lo + hi) / 2
        best_diff = np.inf
        
        for _ in range(max_search_iter):
            mid = (lo + hi) / 2
            r = fista_solve(delta_y, h, lam=mid, max_iter=300)
            n_peaks = self._count_peaks(r)
            
            diff = abs(n_peaks - n_target)
            if diff < best_diff:
                best_diff = diff
                best_lam = mid
                
            if n_peaks == n_target:
                return mid
            elif n_peaks > n_target:
                lo = mid
            else:
                hi = mid
                
        return best_lam

    def _solve_kernel_step(self, y: np.ndarray, r: np.ndarray, lambda_h: float) -> np.ndarray:
        """
        固定 r，更新 h。
        min_h ½‖y - r*h‖² + λ_h‖h‖²
        使用 Wiener 滤波闭式解（频域）
        
        h = IFFT( (R_conj * Y) / (R_conj * R + λ_h) )
        """
        N = pad_to_pow2(len(y) + len(r) - 1)
        # 为了避免频域循环卷积导致的卷绕误差，最好补零
        # 但是这里我们直接对等长的 y 和 r 求解
        N = len(y)
        
        Y_fft = fft(y, n=N)
        R_fft = fft(r, n=N)
        
        R_conj = np.conj(R_fft)
        
        H_fft = (R_conj * Y_fft) / (R_conj * R_fft + lambda_h)
        
        h_full = np.real(ifft(H_fft))
        
        # h(t) 应该具有短支撑，截断它
        # 截断长度：至少 2L/a 往返时间
        T_pulse = 2.0 * self.L / self.a_adj
        h_len = int(T_pulse * self.fs)
        
        h = h_full[:h_len]
        return h

    def solve(self, delta_y: np.ndarray, h_init: np.ndarray, n_fracs: int,
              max_outer_iter: int = 15, lambda_h: float = 1e-1, tol: float = 1e-4) -> dict:
        """
        AM 主循环
        """
        pulse_len = len(h_init) - 10
        y_trunc = delta_y[:pulse_len]
        h = h_init[:pulse_len].copy()
        
        # 归一化 h
        h = h / np.linalg.norm(h)
        
        r = np.zeros_like(y_trunc)
        h_history = [h.copy()]
        
        # 针对水击波是阶跃响应，取导数 (必须用 np.diff 捕获起点跃变)
        dy = np.diff(y_trunc, prepend=0)
        
        for k in range(max_outer_iter):
            dh = np.diff(h, prepend=0)
            
            # 1. 搜 lambda 并更新 r (使用导数域)
            lam_r = self._find_lambda_for_n_peaks(dy, dh, n_fracs)
            r_new = fista_solve(dy, dh, lam=lam_r, max_iter=800)
            
            # 2. 更新 h (也使用导数域更新 dh，然后积分回 h，或者直接在原域更新 h？)
            # 为了稳定，我们在原域用原信号更新 h
            h_new = self._solve_kernel_step(y_trunc, r_new, lambda_h)
            
            # 3. 归一化 h
            h_norm = np.linalg.norm(h_new)
            if h_norm > 0:
                h_new = h_new / h_norm
                
            # 检查收敛
            r_diff = np.linalg.norm(r_new - r) / (np.linalg.norm(r_new) + 1e-12)
            h_diff = np.linalg.norm(h_new - h) / (np.linalg.norm(h_new) + 1e-12)
            
            r = r_new
            h = h_new
            h_history.append(h.copy())
            
            if k > 0 and r_diff < tol and h_diff < tol:
                break
                
        n = len(r)
        t_axis = np.arange(n) / self.fs
        depth_axis = t_axis * self.a_adj / 2.0
        
        return {
            'r': r,
            'h': h,
            'depth_axis': depth_axis,
            'n_outer_iter': k + 1,
            'h_history': h_history,
        }
