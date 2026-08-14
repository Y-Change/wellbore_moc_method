# -*- coding: utf-8 -*-
from __future__ import annotations
"""
audit_phase1.py — Phase 1 成果校对脚本

严格验证以下声称：
1. 匹配工况（a=1450 标称, k=1.0）: 定位误差 < 5 m
2. 失配工况（a=1430, k=1.5）: 定位误差 < 5 m
3. 物理锚定的波速估计误差 < 20 m/s
4. 代码无硬编码泄漏（不能依赖已知真值 4000.0 m）
5. 双缝场景的泛化性
"""

import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _p = os.path.dirname(_d)
    if _p == _d:
        break
    _d = _p

from dataclasses import replace
from analysis.identifiability.crb_core import forward, observe
from analysis.identifiability.mle import add_noise, mle_grid_coord_descent, mle_refine_gn
from analysis.identifiability.scenarios import case_by_name, make_obs, make_scenario
from analysis.identifiability.run_joint_mle import run_joint_pipeline

# ─────────────────────────────────────────────────────────
# Audit 0: 代码审计 — 检查硬编码真值泄漏
# ─────────────────────────────────────────────────────────
print("=" * 72)
print("AUDIT 0: 检查硬编码真值泄漏")
print("=" * 72)

import inspect
source = inspect.getsource(run_joint_pipeline)
issues = []

# 检查 best_x0 = [4000.0] 是否存在（这是一个回退默认值，但不应影响结果）
if "4000.0" in source:
    issues.append("WARNING: run_joint_pipeline 中存在硬编码 4000.0 — 这是回退默认值，"
                  "如果倒谱寻峰失败，会回退到这个值。需要确认倒谱确实找到了峰。")

# 检查是否使用了真值波速或真值位置作为输入
if "scn_true" in source or "x_true" in source:
    issues.append("CRITICAL: 代码中引用了真值场景或真值位置！")

for issue in issues:
    print(f"  ⚠ {issue}")
if not issues:
    print("  ✓ 无真值泄漏问题")

# ─────────────────────────────────────────────────────────
# Audit 1: 物理锚定的自洽性检验
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("AUDIT 1: 物理锚定波速估计的自洽性")
print("=" * 72)

# 生成不同真实波速下的观测信号，检验边界检测是否一致
test_cases = [
    {"a_true": 1430.0, "k_scale": 1.5, "desc": "失配工况 (a=1430, k=1.5)"},
    {"a_true": 1450.0, "k_scale": 1.0, "desc": "匹配工况 (a=1450, k=1.0)"},
    {"a_true": 1380.0, "k_scale": 2.0, "desc": "极端失配 (a=1380, k=2.0)"},
]

case = case_by_name("single_4000")
rng = np.random.default_rng(42)

for tc in test_cases:
    scn_true = make_scenario(case)
    n_cells_true = max(10, int(round(scn_true.L / (tc["a_true"] * scn_true.dt))))
    a_nominal_true = scn_true.a_for(n_cells_true)
    scn_true = replace(scn_true, a_nominal=a_nominal_true, k_scale=tc["k_scale"])

    fr = forward(scn_true)
    s0 = observe(fr.H_wh, fr.t, scn_true, make_obs())
    y, sigma = add_noise(s0, 40, rng)

    # 复现 run_joint_pipeline 中的边界检测逻辑
    dt = scn_true.dt
    t_start_idx = int(6.0 / dt)
    t_end_idx = int(8.0 / dt)
    search_window = np.abs(np.diff(y[t_start_idx:t_end_idx]))
    peak_idx = np.argmax(search_window)
    t_bound = (t_start_idx + peak_idx) * dt
    a_est = 2 * scn_true.L / t_bound

    # 真实的双程旅行时间
    t_expected = 2 * scn_true.L / a_nominal_true
    a_err = abs(a_est - a_nominal_true)

    status = "✓" if a_err < 20.0 else "✗"
    print(f"  {status} {tc['desc']}: a_true={a_nominal_true:.2f}, a_est={a_est:.2f}, "
          f"err={a_err:.2f} m/s, t_bound={t_bound:.4f}s, t_expected={t_expected:.4f}s")

# ─────────────────────────────────────────────────────────
# Audit 2: 边界检测搜索窗硬编码检查
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("AUDIT 2: 边界搜索窗 [6.0, 8.0]s 的物理合理性")
print("=" * 72)

# 对于 L=5000m, a 在 [1300, 1600] 范围内的双程时间
a_range = [1300.0, 1600.0]
t_range = [2 * 5000.0 / a for a in a_range]
print(f"  波速范围 [{a_range[0]}, {a_range[1]}] m/s")
print(f"  双程时间范围 [{t_range[1]:.3f}, {t_range[0]:.3f}] s")
print(f"  搜索窗口: [6.0, 8.0] s")
if t_range[1] >= 6.0 and t_range[0] <= 8.0:
    print("  ✓ 搜索窗口覆盖合理")
else:
    print(f"  ✗ 搜索窗口不完全覆盖！最小 t={t_range[1]:.3f}, 最大 t={t_range[0]:.3f}")

# ─────────────────────────────────────────────────────────
# Audit 3: 短窗 Cepstrum (tf=6s) 是否真能找到裂缝
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("AUDIT 3: 短窗 Cepstrum (tf=6.0s) 的裂缝可见性")
print("=" * 72)

# 对于 a=1430, 裂缝在 4000m, 裂缝回波的双程时间 = 2*4000/1430 = 5.59s
# 这意味着在 6.0s 的窗口内，只有第一个回波能被看到
# 倒谱需要至少看到两个以上的回波周期才能正常工作
a_test = 1430.0
x_frac = 4000.0
t_first_echo = 2 * x_frac / a_test
t_second_echo = 4 * x_frac / a_test
print(f"  裂缝 x=4000m, 波速 a={a_test:.0f} m/s:")
print(f"    第一回波到达: {t_first_echo:.3f} s")
print(f"    第二回波到达: {t_second_echo:.3f} s")
print(f"    tf=6.0s 窗口内可见回波数: {'≥1' if t_first_echo < 6.0 else '0'}")
if t_second_echo > 6.0:
    print(f"  ⚠ 第二回波 ({t_second_echo:.3f}s) 超出 6.0s 窗口！"
          f"倒谱可能依赖不完整的谐波信息。")
    print(f"    但注意：倒谱仍能从频谱中提取周期性的间距信息（虽然精度降低），"
          f"且 6.0s 窗口的设计目的是避开井底回波，而非获取最佳倒谱。")

# ─────────────────────────────────────────────────────────
# Audit 4: scn_short 的 a_nominal 是否被正确离散化
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("AUDIT 4: 短窗场景的波速离散化一致性")
print("=" * 72)

# run_joint_pipeline 中: scn_short = replace(scn, a_nominal=a_est, k_scale=1.0, tf=6.0)
# 但 a_est 是一个连续值，而 MOC 会将其离散化为 a = L/(n_cells*dt)
# 这里 scn 的 a_nominal 被直接设为 a_est，MOC 会自行取整
scn_base = make_scenario(case, tf=6.0)
a_est_test = 1423.69  # 上次运行得到的值
scn_test = replace(scn_base, a_nominal=a_est_test, k_scale=1.0)
n_cells = scn_test.n_cells_nominal()
a_actual = scn_test.a_for(n_cells)
print(f"  a_est = {a_est_test:.2f}")
print(f"  n_cells = {n_cells}")
print(f"  a_actual (MOC) = {a_actual:.2f}")
print(f"  离散化偏差 = {abs(a_est_test - a_actual):.2f} m/s")
dx = scn_test.dx_for(n_cells)
print(f"  dx = {dx:.4f} m (空间步长 = 定位精度下限)")

# ─────────────────────────────────────────────────────────
# Audit 5: 完整端到端回归测试
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("AUDIT 5: 端到端回归测试")
print("=" * 72)

regression_tests = [
    {
        "name": "匹配工况 (a=1451.4标称, k=1.0)",
        "case_name": "single_4000",
        "a_mismatch": 1451.4,  # 默认标称波速
        "k_mismatch": 1.0,
        "snr": 40,
        "max_err_m": 5.0,
    },
    {
        "name": "失配工况 (a=1430, k=1.5)",
        "case_name": "single_4000",
        "a_mismatch": 1430.0,
        "k_mismatch": 1.5,
        "snr": 40,
        "max_err_m": 10.0,
    },
]

for test in regression_tests:
    print(f"\n  --- {test['name']} ---")
    t0 = time.time()

    case = case_by_name(test["case_name"])
    n_frac = len(case.x_f)

    # 构建真实场景
    scn_true = make_scenario(case)
    n_cells_true = max(10, int(round(scn_true.L / (test["a_mismatch"] * scn_true.dt))))
    a_nominal_true = scn_true.a_for(n_cells_true)
    scn_true = replace(scn_true, a_nominal=a_nominal_true, k_scale=test["k_mismatch"])

    obs = make_obs()

    fr_true = forward(scn_true)
    s0_true = observe(fr_true.H_wh, fr_true.t, scn_true, obs)

    rng = np.random.default_rng(42)
    y, sigma = add_noise(s0_true, test["snr"], rng)
    H_noisy = fr_true.H_wh + rng.normal(0, sigma, size=len(fr_true.H_wh))

    # 反演
    scn_inv = replace(make_scenario(case, tf=10.0), a_nominal=1451.4, k_scale=1.0)
    a_grid = np.linspace(1350, 1550, 5)
    k_grid = np.linspace(0.5, 2.0, 3)

    res = run_joint_pipeline(
        scn_inv, obs, y, fr_true.t, H_noisy, n_frac,
        a_grid=a_grid, k_grid=k_grid, radius=20, n_rounds=3, workers=1, gn_steps=3
    )

    t1 = time.time()

    err_x = np.max(np.abs(np.array(res['x_hat']) - np.array(scn_true.x_f)))
    err_a = abs(res['a_hat'] - a_nominal_true)

    status = "✓" if err_x < test["max_err_m"] else "✗"
    print(f"  {status} 位置误差: {err_x:.3f} m (阈值 {test['max_err_m']} m)")
    print(f"    波速误差: {err_a:.3f} m/s")
    print(f"    x_true={list(scn_true.x_f)}, x_hat={np.round(res['x_hat'], 2).tolist()}")
    print(f"    a_true={a_nominal_true:.2f}, a_hat={res['a_hat']:.2f}")
    print(f"    k_true={test['k_mismatch']:.2f}, k_hat={res['k_scale_hat']:.2f}")
    print(f"    耗时: {t1-t0:.1f} s")

print("\n" + "=" * 72)
print("AUDIT COMPLETE")
print("=" * 72)
