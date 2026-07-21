# -*- coding: utf-8 -*-
"""
FISTA 稀疏反卷积求解器。
"""
from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """近端算子：软阈值"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def pad_to_pow2(n: int) -> int:
    """寻找大于等于 n 的最小 2 的幂次，用于 FFT 加速"""
    return 1 << (n - 1).bit_length()


def _fft_prep(h: np.ndarray, N: int):
    """
    预处理：将 h 补零到 N 并进行 FFT，供后续卷积和相关使用。
    
    Parameters
    ----------
    h : 子波，通常较短
    N : 目标长度（信号 y 的长度）
    
    Returns
    -------
    H_fft : h 补零到 N 后的 FFT
    H_conj_fft : H_fft 的复共轭（用于相关）
    """
    h_padded = np.zeros(N)
    # 将 h 放置在开头，这样卷积不产生时间偏移（假设 h 的起点是 t=0）
    h_padded[:len(h)] = h
    H_fft = fft(h_padded)
    return H_fft, np.conj(H_fft)


def fista_solve(y: np.ndarray, h: np.ndarray, lam: float,
                max_iter: int = 2000, tol: float = 1e-6) -> np.ndarray:
    """
    FISTA 快速近端梯度法求解：
        min_r  ½‖y - h*r‖² + λ‖r‖₁
    
    Parameters
    ----------
    y : 观测信号（差信号 Δy），shape (N,)
    h : 子波，shape (M,)  M << N
    lam : ℓ₁ 正则化参数
    max_iter : 最大迭代次数
    tol : 收敛容差
    
    Returns
    -------
    r : 稀疏反射系数序列，shape (N,)
    """
    N = len(y)
    
    # 预计算 h 的 FFT
    H_fft, H_conj_fft = _fft_prep(h, N)
    
    # y 的 FFT，用于计算残差
    Y_fft = fft(y)
    
    # Lipschitz 常数 (Lipschitz constant of the gradient)
    # L = 最大的特征值 (H^T H) = max(|H_fft|^2)
    L = np.max(np.abs(H_fft)**2)
    if L == 0:
        return np.zeros(N)
    
    step_size = 1.0 / L
    
    # 初始化
    r = np.zeros(N)
    p = np.zeros(N)
    t = 1.0
    
    r_old = np.zeros(N)
    
    for i in range(max_iter):
        # 计算梯度 ∇f(p) = H^T (H p - y)
        P_fft = fft(p)
        # H_fft * P_fft - Y_fft 就是 (Hp - y) 的 FFT
        residual_fft = H_fft * P_fft - Y_fft
        # 乘以 H_conj_fft 就是 H^T (Hp - y) 的 FFT
        grad_fft = H_conj_fft * residual_fft
        grad = np.real(ifft(grad_fft))
        
        # 梯度下降步
        r_new = p - step_size * grad
        
        # 软阈值（近端算子步）
        r_new = soft_threshold(r_new, lam * step_size)
        
        # 加速步
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        p = r_new + ((t - 1.0) / t_new) * (r_new - r)
        
        # 收敛性检查
        if i % 10 == 0:
            diff = np.linalg.norm(r_new - r)
            if diff / (np.linalg.norm(r_new) + 1e-12) < tol:
                r = r_new
                break
                
        r = r_new
        t = t_new
        
    return r
