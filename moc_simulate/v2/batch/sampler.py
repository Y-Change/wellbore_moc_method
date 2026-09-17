# -*- coding: utf-8 -*-
"""
moc_simulate.v2.batch.sampler

基于真实工程地质与压裂工况范围的物理自洽拉丁超立方采样器 (Latin Hypercube Sampler, LHS)
对齐技术报告 2.6.3 与 2.6.4 节（以本文件实现为准）：
- 裂缝空间位置：首簇 x_start in [1000, 4500] m，多簇间距 10 ~ 50 m，末簇 <= 4850 m
- 物理联动默认 coupling_mode="physical"：Dirichlet 潜变量 r_j（E[r_j]=1）写物性，
  再由封闭趾端稳态正演 alpha_ss = q_j / Q_0；r_j 不是进液比
    C_f,j = C_f,base * r_j^0.85 * exp(eps_Cf)
    k_leak,j = k_leak,base * r_j^0.70 * exp(eps_leak)
    d_p,j = d_p,0 * (1 + delta_erode * r_j) -> K_p,j = 1 / (2*g*Cd^2*Ap^2)
- 稳态分流由 k_leak 与 K_p 决定，C_f 不进入稳态代数分流
- Type V：先验 p_fault 激活，k_leak 放大 5~10 倍至 5e-4~15e-4 m^{2.5}/s
- Type IV：r_j < 0.04*N_c（且 N_c>1）时硬截断 C_f < 0.001, k_leak < 0.1e-4, K_p > 5e7
- 类型标签由 classify_fracture_type 按实现 alpha 后验判定，不是联合采样盒子
- 提供 sample_preset_scenario 一键式现场工况生成接口（先装配物性再正演 alpha）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from moc_simulate.common.constants import G, H0_MAX_WORKING
from moc_simulate.v2.core.moc_mesh import MocGrid
from moc_simulate.v2.core.initial_field import InfeasibleSteadyStateError, solve_physical_steady_state

try:
    from scipy.stats import qmc
    HAS_QMC = True
except ImportError:
    HAS_QMC = False


# 5 大典型裂缝类型工程特征矩阵规范
FRACTURE_TYPE_SPECS: Dict[str, Dict[str, Any]] = {
    "Type I": {
        "name": "优势发育主进液簇 (Dominant / Runaway)",
        "w_range": (0.35, 0.55),
        "cf_range": (0.015, 0.025),
        "kleak_range": (1.8e-4, 3.5e-4),
        "dp_range": (0.0115, 0.0135),
        "kp_range": (1.5e5, 3.5e5),
        "description": "抢占排量失控狂奔，水力充分延伸，射孔严重冲蚀扩径，低流阻，大宏观反弹，倒谱尖锐高耸能量主峰",
    },
    "Type II": {
        "name": "均衡/正常发育簇 (Balanced / Average)",
        "w_range": (0.20, 0.30),
        "cf_range": (0.008, 0.014),
        "kleak_range": (0.8e-4, 1.6e-4),
        "dp_range": (0.010, 0.011),
        "kp_range": (4.5e5, 6.5e5),
        "description": "标准设计簇，应力阴影适度，射孔正常磨损，水锤振荡对称平稳，倒谱清晰亚米级检出率 100%",
    },
    "Type III": {
        "name": "受抑/欠发育弱进液簇 (Suppressed / Restricted)",
        "w_range": (0.05, 0.15),
        "cf_range": (0.002, 0.006),
        "kleak_range": (0.2e-4, 0.6e-4),
        "dp_range": (0.009, 0.010),
        "kp_range": (8.0e5, 1.8e6),
        "description": "遭受两侧主簇强应力阴影夹击，进液受限，孔眼未扩径高流阻，反弹微弱，倒谱特征峰低矮需波前锐化",
    },
    "Type IV": {
        "name": "砂堵闭合/未起裂死簇 (Screened-out / Inactive)",
        "w_range": (0.0, 0.04),
        "cf_range": (0.0001, 0.001),
        "kleak_range": (1.0e-6, 0.1e-4),
        "dp_range": (0.005, 0.008),
        "kp_range": (5.0e7, 1.0e8),
        "description": "射孔被高浓度砂柱堵死或起裂失败，断流盲管刚性全反射，无反弹动态，倒谱共振峰完全湮灭",
    },
    "Type V": {
        "name": "沟通天然断层/强微裂缝簇 (Fault-Intersected / Thief Zone)",
        "w_range": (0.15, 0.35),
        "cf_range": (0.005, 0.012),
        "kleak_range": (5.0e-4, 15.0e-4),
        "dp_range": (0.010, 0.012),
        "kp_range": (3.0e5, 6.0e5),
        "description": "水力裂缝切穿高导通天然断层或微裂缝带，超常强滤失(高5~10倍)，停泵后水头快速消退泄压，大反弹被拉平抹除，倒谱强阻尼衰减畸变",
    },
}


def classify_fracture_type(
    w: float,
    cf: float,
    kleak: float,
    kp: float,
    is_fault: bool = False,
) -> str:
    """
    根据已实现的稳态分流比与物性后验界定裂缝类型 (Type I ~ Type V)。
    不是按 FRACTURE_TYPE_SPECS 联合盒子装配。判定优先级：
    1. Type IV (砂堵死簇)：进液断流 (w < 0.04) 或流阻发散 (Kp >= 5.0e7) 或微小无缝 (Cf < 0.001)
    2. Type V (沟通天然断层)：显式断层标记 (is_fault=True) 或拟达西滤失异常放大 (kleak >= 4.5e-4 且非砂堵)
    3. Type I (优势发育簇)：高进液分流比 (w >= 0.35)
    4. Type III (受抑欠发育簇)：低进液分流比 (w <= 0.15)
    5. Type II (均衡正常发育簇)：常规基准 (0.15 < w < 0.35)
    """
    if w < 0.04 or kp >= 5.0e7 or cf < 0.001:
        return "Type IV: 砂堵闭合/未起裂死簇"
    if is_fault or kleak >= 4.5e-4:
        return "Type V: 沟通天然断层/强微裂缝簇"
    if w >= 0.35:
        return "Type I: 优势发育主进液簇"
    if w <= 0.15:
        return "Type III: 受抑/欠发育弱进液簇"
    return "Type II: 均衡/正常发育簇"


@dataclass
class LhsSamplingBounds:
    """LHS 物理参数空间取值范围定义与物理力学联动参数"""
    # 空间拓扑
    n_frac_min: int = 1
    n_frac_max: int = 6
    x_start_min: float = 4500.0
    x_start_max: float = 4900.0
    spacing_min: float = 10.0
    spacing_max: float = 50.0
    x_max: float = 4950.0  # 末簇最深空间位置上限 (保留盲端死水区)

    # 基准顺应性 Cf [m^2] (对数均匀)
    cf_min: float = 0.005
    cf_max: float = 0.030

    # 基准射孔参数
    np_holes_choices: tuple = (4, 6, 8, 12, 16)
    dp_perf_min: float = 0.008   # 8 mm
    dp_perf_max: float = 0.012   # 12 mm
    cd_perf_min: float = 0.60
    cd_perf_max: float = 0.75

    # 关泵斜坡动力学
    tc_min: float = 0.001          # 0.3 s
    tc_max: float = 0.1          # 2.0 s
    ramp_types: tuple = ("linear", "cosine")

    # 地层基准滤失与水头
    kleak_log_min: float = -5.0  # 1e-5
    kleak_log_max: float = -3.0  # 1e-3
    hext_min: float = 50.0
    hext_max: float = 200.0

    # 井筒物性扰动 (域随机化)
    wavespeed_min: float = 1400.0
    wavespeed_max: float = 1500.0
    v0_min: float = 0.8
    v0_max: float = 1.5

    # --- 物理力学联动机制配置 (耦合模式) ---
    coupling_mode: str = "physical"  # "physical" (物理第一性原理强联动) 或 "independent" (独立均匀扰动)

    # 物理联动模式基准参数取值区间 (对齐 2.6.4 节基准 Type II 均衡发育簇)
    cf_base_min: float = 0.008
    cf_base_max: float = 0.014
    kleak_base_min: float = 0.8e-4
    kleak_base_max: float = 1.4e-4

    # 相对发育指数 r_j 与顺应性/滤失的幂律（r_j 不是实现进液比）
    alpha_cf: float = 0.85        # Cf ~ r_j^0.85
    beta_leak: float = 0.70       # kleak ~ r_j^0.70
    sigma_cf: float = 0.10        # Cf 对数正态随机扰动标差
    sigma_leak: float = 0.15      # kleak 对数正态随机扰动标差

    # 射孔磨料冲蚀动力学参数
    delta_erode: float = 0.15     # 孔径冲蚀扩径增量系数: dp = dp0 * (1 + delta_erode * r_j)
    delta_cd: float = 0.05        # 孔流系数冲蚀钝化增量: cd = min(cd_max, cd0 + delta_cd * r_j)
    cd_max: float = 0.82

    # Type IV 砂堵死簇判定与截断
    screenout_w_threshold: float = 0.04
    screenout_cf_max: float = 0.001       # Cf < 0.001 m^2
    screenout_kleak_max: float = 0.1e-4   # kleak < 0.1e-4 m^2.5/s
    screenout_kp_min: float = 5.0e7       # Kp > 5.0e7 s^2/m^5

    # Type V 沟通天然断层/强微裂缝簇概率与强滤失放大倍数
    p_fault: float = 0.15                 # 压裂段沟通天然断层的先验概率
    fault_leak_multiplier_min: float = 5.0   # 滤失放大倍数下限 (5倍)
    fault_leak_multiplier_max: float = 10.0  # 滤失放大倍数上限 (10倍)
    fault_kleak_min: float = 5.0e-4          # 5.0e-4 m^2.5/s
    fault_kleak_max: float = 15.0e-4         # 15.0e-4 m^2.5/s
    fault_cf_min: float = 0.005              # 断层簇储能顺应性维持中等
    fault_cf_max: float = 0.012

    # 工程工作包络与拒绝采样
    h0_max: float = H0_MAX_WORKING           # 井口最大工作水头 [m]
    reject_attempts: int = 50                # 单样本内层拒绝采样最大次数


class LatinHypercubeSampler:
    """工程物理对齐的 LHS 样本生成器与物理力学联动采样基座"""

    def __init__(self, bounds: Optional[LhsSamplingBounds] = None, seed: int = 42):
        self.bounds = bounds or LhsSamplingBounds()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        生成 n_samples 个满足物理几何与水动力学自洽约束的正演仿真配置字典。
        """
        b = self.bounds
        samples: List[Dict[str, Any]] = []

        if HAS_QMC:
            sampler = qmc.LatinHypercube(d=8, seed=self.seed)
            u = sampler.random(n=n_samples)
        else:
            # 基础均匀分层抽样 fallback
            u = np.zeros((n_samples, 8))
            for dim in range(8):
                idx = self.rng.permutation(n_samples)
                u[:, dim] = (idx + self.rng.uniform(0.0, 1.0, size=n_samples)) / n_samples

        wellbore_d = 0.1397
        wellbore_area = float(np.pi * (wellbore_d ** 2) / 4.0)

        u_pool = [u[k] for k in range(n_samples)]
        n_outer_budget = n_samples * 30
        while len(samples) < n_samples:
            n_outer_budget -= 1
            if n_outer_budget < 0:
                raise RuntimeError(
                    f"无法在工程水头上限 H0_max={float(b.h0_max):.0f} m 内采满 {n_samples} 个"
                    f"满足水头/α 对比度/质量守恒约束的样本。单样本内层 "
                    f"{int(b.reject_attempts)} 次拒绝后已放弃该候选，禁止沿用未通过约束的最后一次结果。"
                )
            u_i = u_pool.pop(0) if u_pool else self.rng.random(8)
            i = len(samples)

            # 1. 簇数
            n_frac = int(self.rng.integers(b.n_frac_min, b.n_frac_max + 1))

            # 2. 空间几何排布 (起始位置 + 间距)
            x_start = float(b.x_start_min + u_i[0] * (b.x_start_max - b.x_start_min))
            positions = [x_start]
            cur_x = x_start
            for _ in range(n_frac - 1):
                sp = float(self.rng.uniform(b.spacing_min, b.spacing_max))
                cur_x += sp
                positions.append(cur_x)

            # 严格约束末簇深度不超过 x_max，保留死水区盲端
            if cur_x > b.x_max and n_frac > 1:
                excess = cur_x - b.x_max
                if x_start - excess >= b.x_start_min:
                    positions = [p - excess for p in positions]
                else:
                    span = cur_x - x_start
                    avail_span = max(b.spacing_min * (n_frac - 1), b.x_max - x_start)
                    scale = avail_span / span if span > 0 else 1.0
                    positions = [x_start + (p - x_start) * scale for p in positions]

            # 3. 关泵斜坡动力学
            tc = float(b.tc_min + u_i[2] * (b.tc_max - b.tc_min))
            ramp_type = str(self.rng.choice(b.ramp_types))

            # 4. 地层基准水头与井筒流体物性
            H_ext = float(b.hext_min + u_i[6] * (b.hext_max - b.hext_min))
            wavespeed = float(b.wavespeed_min + u_i[7] * (b.wavespeed_max - b.wavespeed_min))
            v0 = float(self.rng.uniform(b.v0_min, b.v0_max))
            total_q = v0 * wellbore_area

            # 5. 基准射孔与物性参数 (未受冲蚀基态)
            np_holes = int(self.rng.choice(b.np_holes_choices))
            dp_base = float(b.dp_perf_min + u_i[3] * (b.dp_perf_max - b.dp_perf_min))
            cd_base = float(b.cd_perf_min + u_i[4] * (b.cd_perf_max - b.cd_perf_min))

            # 基准顺应性与滤失系数
            if b.coupling_mode == "physical":
                cf_base = float(b.cf_base_min + u_i[1] * (b.cf_base_max - b.cf_base_min))
                kleak_base = float(b.kleak_base_min + u_i[5] * (b.kleak_base_max - b.kleak_base_min))
            else:
                cf_log = float(np.log10(b.cf_min) + u_i[1] * (np.log10(b.cf_max) - np.log10(b.cf_min)))
                cf_base = float(10.0 ** cf_log)
                kleak_log = float(b.kleak_log_min + u_i[5] * (b.kleak_log_max - b.kleak_log_min))
                kleak_base = float(10.0 ** kleak_log)

            grid = MocGrid.create(L=5000.0, wavespeed=wavespeed, dt=0.001, tf=20.0)
            frac_indices, sorted_pos, order = grid.map_fracture_positions(positions)

            # 6. 拒绝采样循环：先正向采样物性，再前向求解流量并过滤退化工况
            cf_list: List[float] = []
            kleak_list: List[float] = []
            kp_list: List[float] = []
            dp_list: List[float] = []
            cd_list: List[float] = []
            has_fault = False
            fault_cluster_idx = -1
            alpha_orig = np.zeros(n_frac, dtype=np.float64)
            q_orig = np.zeros(n_frac, dtype=np.float64)
            H0_realized = 300.0
            mass_residual = 0.0

            for attempt in range(int(b.reject_attempts)):
                cf_list = []
                kleak_list = []
                kp_list = []
                dp_list = []
                cd_list = []

                # Type V 天然断层沟通激活判定
                has_fault = False
                fault_cluster_idx = -1
                if b.p_fault > 0.0 and self.rng.uniform(0.0, 1.0) < b.p_fault and n_frac >= 1:
                    fault_cluster_idx = int(self.rng.integers(0, n_frac))
                    has_fault = True

                if b.coupling_mode == "physical":
                    alpha_dir = float(self.rng.choice([0.5, 1.0, 2.0]))
                    xi_arr = self.rng.dirichlet(np.full(n_frac, alpha_dir)) * float(n_frac)

                    for j in range(n_frac):
                        is_fault_cluster = (has_fault and j == fault_cluster_idx)
                        rj = float(xi_arr[j])

                        if rj < b.screenout_w_threshold * n_frac and n_frac > 1 and not is_fault_cluster:
                            # Type IV: 砂堵死簇
                            cf_j = float(self.rng.uniform(0.0002, b.screenout_cf_max))
                            kleak_j = float(self.rng.uniform(1.0e-6, b.screenout_kleak_max))
                            dp_j = float(dp_base * 0.70)
                            cd_j = float(cd_base * 0.80)
                            kp_j = float(self.rng.uniform(b.screenout_kp_min, 1.0e8))
                        elif is_fault_cluster:
                            # Type V: 沟通天然断层/强微裂缝簇
                            cf_norm = cf_base * (rj ** b.alpha_cf) * np.exp(float(self.rng.normal(0, b.sigma_cf)))
                            cf_j = float(np.clip(cf_norm, b.fault_cf_min, b.fault_cf_max))
                            mult = float(self.rng.uniform(b.fault_leak_multiplier_min, b.fault_leak_multiplier_max))
                            kleak_norm = kleak_base * (rj ** b.beta_leak) * np.exp(float(self.rng.normal(0, b.sigma_leak)))
                            kleak_j = float(np.clip(kleak_norm * mult, b.fault_kleak_min, b.fault_kleak_max))
                            dp_j = float(dp_base * (1.0 + b.delta_erode * rj))
                            cd_j = float(min(b.cd_max, cd_base + b.delta_cd * rj))
                            ap = np_holes * np.pi * (dp_j ** 2) / 4.0
                            kp_j = float(1.0 / (2.0 * G * (cd_j ** 2) * (ap ** 2)))
                        else:
                            # 正常发育簇：Type I, II, III
                            cf_j = float(cf_base * (rj ** b.alpha_cf) * np.exp(float(self.rng.normal(0, b.sigma_cf))))
                            kleak_j = float(kleak_base * (rj ** b.beta_leak) * np.exp(float(self.rng.normal(0, b.sigma_leak))))
                            dp_j = float(dp_base * (1.0 + b.delta_erode * rj))
                            cd_j = float(min(b.cd_max, cd_base + b.delta_cd * rj))
                            ap = np_holes * np.pi * (dp_j ** 2) / 4.0
                            kp_j = float(1.0 / (2.0 * G * (cd_j ** 2) * (ap ** 2)))

                        cf_list.append(cf_j)
                        kleak_list.append(kleak_j)
                        dp_list.append(dp_j)
                        cd_list.append(cd_j)
                        kp_list.append(kp_j)
                else:
                    for j in range(n_frac):
                        is_fault_cluster = (has_fault and j == fault_cluster_idx)
                        cf_j = float(cf_base * self.rng.uniform(0.8, 1.2))
                        kleak_j = float(kleak_base * self.rng.uniform(0.8, 1.2))
                        if is_fault_cluster:
                            cf_j = float(np.clip(cf_j, b.fault_cf_min, b.fault_cf_max))
                            mult = float(self.rng.uniform(b.fault_leak_multiplier_min, b.fault_leak_multiplier_max))
                            kleak_j = float(np.clip(kleak_j * mult, b.fault_kleak_min, b.fault_kleak_max))
                        dp_j = dp_base
                        cd_j = cd_base
                        ap = np_holes * np.pi * (dp_j ** 2) / 4.0
                        kp_j = float(1.0 / (2.0 * G * (cd_j ** 2) * (ap ** 2)))

                        cf_list.append(cf_j)
                        kleak_list.append(kleak_j)
                        dp_list.append(dp_j)
                        cd_list.append(cd_j)
                        kp_list.append(kp_j)

                # 正向物理稳态求解真实稳态分流 alpha_realized 与井口水头 H0_realized
                kleak_arr = np.array([kleak_list[k] for k in order], dtype=np.float64)
                kp_arr = np.array([kp_list[k] for k in order], dtype=np.float64)
                try:
                    (
                        H0_realized,
                        _,
                        _,
                        _,
                        _,
                        q_frac_ss,
                        alpha_realized,
                        mass_residual,
                    ) = solve_physical_steady_state(
                        L=5000.0,
                        N=grid.N,
                        dx=grid.dx,
                        D=wellbore_d,
                        area=wellbore_area,
                        nu=1.0e-6,
                        K_D=0.045e-3 / wellbore_d,
                        V0=v0,
                        g=G,
                        toe_bc="dead_end",
                        frac_indices=frac_indices,
                        frac_kleak_arr=kleak_arr,
                        frac_Kp_arr=kp_arr,
                        sorted_pos=sorted_pos,
                        H_ext=H_ext,
                        H0_max=float(b.h0_max),
                    )
                except InfeasibleSteadyStateError:
                    continue

                # 恢复为原始输入顺序
                alpha_orig = np.zeros(n_frac, dtype=np.float64)
                q_orig = np.zeros(n_frac, dtype=np.float64)
                for k, orig_idx in enumerate(order):
                    alpha_orig[orig_idx] = alpha_realized[k]
                    q_orig[orig_idx] = q_frac_ss[k]

                # 拒绝采样检验：排除全砂堵死簇或全均匀无对比度退化工况
                if n_frac >= 2:
                    spread = float(np.max(alpha_orig) - np.min(alpha_orig))
                    # 动态归一化阈值：标称 Nc=4 时对应 spread>=0.03, max>=0.08
                    if spread < (0.12 / n_frac):
                        continue
                    if np.max(alpha_orig) < (0.32 / n_frac):
                        continue
                if mass_residual > 1.0e-6:
                    continue
                if H0_realized <= H_ext + 10.0:
                    continue
                if H0_realized > float(b.h0_max):
                    continue

                break
            else:
                continue

            # 7. 根据正向求解得到的真实 alpha 计算裂缝类型
            types_list = [
                classify_fracture_type(
                    alpha_orig[j],
                    cf_list[j],
                    kleak_list[j],
                    kp_list[j],
                    is_fault=(has_fault and j == fault_cluster_idx),
                )
                for j in range(n_frac)
            ]

            samples.append({
                "sample_id": i,
                "n_frac": n_frac,
                "fracture_positions": positions,
                "fracture_Cf": cf_list,
                "fracture_kleak": kleak_list,
                "fracture_Kp": kp_list,
                "fracture_dp": dp_list,
                "fracture_cd": cd_list,
                "fracture_types": types_list,
                "has_fault": has_fault,
                "fault_cluster_idx": fault_cluster_idx,
                "pump_closure_duration": tc,
                "ramp_type": ramp_type,
                "perf_num_holes": np_holes,
                "perf_diameter": dp_base,
                "perf_cd": cd_base,
                "H_ext": H_ext,
                "initial_head": H0_realized,
                "H0_realized": H0_realized,
                "fracture_alpha_ss": alpha_orig.tolist(),
                "fracture_Q_ss": q_orig.tolist(),
                "steady_mass_residual": mass_residual,
                "wavespeed": wavespeed,
                "initial_velocity": v0,
            })

        return samples


# 导出别名统一命名风格
LhsSampler = LatinHypercubeSampler


def sample_preset_scenario(
    scenario_type: str,
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    一键式现场典型工况生成接口。

    支持预设工况类型:
    1. "perfect_uniform" / "uniform" / "完美均匀型": 各簇严格对称均分，标准 Type II 均衡发育簇
    2. "heel_dominant" / "跟部突进型": 近井跟部首簇绝对主导 (Type I)，中部正常 (Type II)，远井趾端受抑 (Type III)
    3. "toe_dominant" / "趾端优势型": 远井趾端优势延伸 (Type I)，跟部欠发育受抑 (Type III)
    4. "screenout_dead" / "单簇砂堵死簇型": 某一簇突发砂堵闭合 (Type IV, Kp > 5e7)，其余簇重新分流
    5. "fault_leaking" / "沟通断层强漏失型": 某一簇沟通天然断层 (Type V, kleak 扩大至 1e-3 m^2.5/s)

    参数:
        scenario_type: 工况英文代号或中文名称
        seed: 随机种子 (可选)
        kwargs: 允许自定义覆盖 wellbore_length, wavespeed, dt, tf, positions 等任意参数
    """
    # 基础物理几何默认值
    wellbore_length = float(kwargs.get("wellbore_length", 5000.0))
    wavespeed = float(kwargs.get("wavespeed", 1450.0))
    v0 = float(kwargs.get("initial_velocity", 1.0))
    H_ext = float(kwargs.get("H_ext", 100.0))
    tc = float(kwargs.get("pump_closure_duration", 1.0))
    ramp_type = str(kwargs.get("ramp_type", "linear"))
    dt = float(kwargs.get("dt", 0.001))
    tf = float(kwargs.get("tf", 20.0))

    wellbore_d = float(kwargs.get("wellbore_diameter", 0.1397))
    wellbore_area = float(np.pi * (wellbore_d ** 2) / 4.0)
    total_q = v0 * wellbore_area

    # 默认 4 簇经典间距 20m 拓扑: [4100, 4120, 4140, 4160] m
    default_positions = [4100.0, 4120.0, 4140.0, 4160.0]
    positions = list(kwargs.get("fracture_positions", default_positions))
    n_frac = len(positions)

    # 归一化工况键名
    key = scenario_type.strip().lower()

    if key in ("perfect_uniform", "uniform", "完美均匀型"):
        # 完美均分
        weights = [1.0 / n_frac] * n_frac
        cf_list = [0.010] * n_frac
        kleak_list = [1.0e-4] * n_frac
        dp_list = [0.0105] * n_frac
        cd_list = [0.65] * n_frac
        np_holes = 6
        kp_list = [
            float(1.0 / (2.0 * G * (cd_list[j] ** 2) * ((np_holes * np.pi * (dp_list[j] ** 2) / 4.0) ** 2)))
            for j in range(n_frac)
        ]
        has_fault = False
        fault_cluster_idx = -1
        types_list = ["Type II: 均衡/正常发育簇"] * n_frac

    elif key in ("heel_dominant", "跟部突进型"):
        # 跟部优势 (近井首簇 Type I，趾端末簇 Type III)
        np_holes = 6
        if n_frac == 1:
            weights = [1.0]
            cf_list = [0.020]
            kleak_list = [2.2e-4]
            dp_list = [0.0125]
            cd_list = [0.72]
        elif n_frac == 4:
            weights = [0.48, 0.24, 0.18, 0.10]
            cf_list = [0.020, 0.011, 0.009, 0.004]
            kleak_list = [2.2e-4, 1.1e-4, 0.9e-4, 0.4e-4]
            dp_list = [0.0125, 0.0105, 0.0100, 0.0092]
            cd_list = [0.72, 0.65, 0.65, 0.62]
        else:
            decay = np.linspace(1.0, 0.18, n_frac)
            weights = (decay / np.sum(decay)).tolist()
            w_avg = 1.0 / n_frac
            cf_list = [float(np.clip(0.010 * ((w / w_avg) ** 0.85), 0.003, 0.025)) for w in weights]
            kleak_list = [float(np.clip(1.0e-4 * ((w / w_avg) ** 0.70), 0.3e-4, 3.5e-4)) for w in weights]
            dp_list = [float(np.clip(0.0105 * (1.0 + 0.15 * (w / w_avg)), 0.009, 0.0135)) for w in weights]
            cd_list = [float(min(0.82, 0.62 + 0.05 * (w / w_avg))) for w in weights]

        kp_list = [
            float(1.0 / (2.0 * G * (cd_list[j] ** 2) * ((np_holes * np.pi * (dp_list[j] ** 2) / 4.0) ** 2)))
            for j in range(n_frac)
        ]
        has_fault = False
        fault_cluster_idx = -1
        types_list = [classify_fracture_type(weights[j], cf_list[j], kleak_list[j], kp_list[j]) for j in range(n_frac)]

    elif key in ("toe_dominant", "趾端优势型"):
        # 趾端优势 (远井末簇 Type I，跟部首簇 Type III)
        np_holes = 6
        if n_frac == 1:
            weights = [1.0]
            cf_list = [0.020]
            kleak_list = [2.2e-4]
            dp_list = [0.0125]
            cd_list = [0.72]
        elif n_frac == 4:
            weights = [0.10, 0.18, 0.24, 0.48]
            cf_list = [0.004, 0.009, 0.011, 0.020]
            kleak_list = [0.4e-4, 0.9e-4, 1.1e-4, 2.2e-4]
            dp_list = [0.0092, 0.0100, 0.0105, 0.0125]
            cd_list = [0.62, 0.65, 0.65, 0.72]
        else:
            decay = np.linspace(0.18, 1.0, n_frac)
            weights = (decay / np.sum(decay)).tolist()
            w_avg = 1.0 / n_frac
            cf_list = [float(np.clip(0.010 * ((w / w_avg) ** 0.85), 0.003, 0.025)) for w in weights]
            kleak_list = [float(np.clip(1.0e-4 * ((w / w_avg) ** 0.70), 0.3e-4, 3.5e-4)) for w in weights]
            dp_list = [float(np.clip(0.0105 * (1.0 + 0.15 * (w / w_avg)), 0.009, 0.0135)) for w in weights]
            cd_list = [float(min(0.82, 0.62 + 0.05 * (w / w_avg))) for w in weights]

        kp_list = [
            float(1.0 / (2.0 * G * (cd_list[j] ** 2) * ((np_holes * np.pi * (dp_list[j] ** 2) / 4.0) ** 2)))
            for j in range(n_frac)
        ]
        has_fault = False
        fault_cluster_idx = -1
        types_list = [classify_fracture_type(weights[j], cf_list[j], kleak_list[j], kp_list[j]) for j in range(n_frac)]

    elif key in ("screenout_dead", "单簇砂堵死簇型"):
        # 单簇砂堵 (某一簇闭合死簇 Type IV, 射孔流阻发散至 8.0e7)
        np_holes = 6
        if n_frac == 1:
            weights = [1.0]
            cf_list = [0.0005]
            kleak_list = [0.05e-4]
            dp_list = [0.0070]
            cd_list = [0.50]
            kp_list = [8.0e7]
        elif n_frac == 4:
            weights = [0.38, 0.35, 0.01, 0.26]
            cf_list = [0.016, 0.014, 0.0005, 0.010]
            kleak_list = [1.8e-4, 1.6e-4, 0.05e-4, 1.2e-4]
            dp_list = [0.0120, 0.0115, 0.0070, 0.0105]
            cd_list = [0.70, 0.68, 0.50, 0.65]
            kp_list = []
            for j in range(n_frac):
                if weights[j] <= 0.02:
                    kp_list.append(8.0e7)
                else:
                    ap = np_holes * np.pi * (dp_list[j] ** 2) / 4.0
                    kp_list.append(float(1.0 / (2.0 * G * (cd_list[j] ** 2) * (ap ** 2))))
        else:
            dead_idx = 1 if n_frac == 2 else min(2, n_frac - 1)
            rem_w = (1.0 - 0.01) / (n_frac - 1)
            weights = [rem_w] * n_frac
            weights[dead_idx] = 0.01

            cf_list = [0.012] * n_frac
            cf_list[dead_idx] = 0.0005
            kleak_list = [1.2e-4] * n_frac
            kleak_list[dead_idx] = 0.05e-4
            dp_list = [0.0105] * n_frac
            dp_list[dead_idx] = 0.0070
            cd_list = [0.65] * n_frac
            cd_list[dead_idx] = 0.50

            kp_list = []
            for j in range(n_frac):
                if j == dead_idx or weights[j] <= 0.02:
                    kp_list.append(8.0e7)
                else:
                    ap = np_holes * np.pi * (dp_list[j] ** 2) / 4.0
                    kp_list.append(float(1.0 / (2.0 * G * (cd_list[j] ** 2) * (ap ** 2))))

        has_fault = False
        fault_cluster_idx = -1
        types_list = [classify_fracture_type(weights[j], cf_list[j], kleak_list[j], kp_list[j]) for j in range(n_frac)]

    elif key in ("fault_leaking", "沟通断层强漏失型"):
        # 沟通天然断层/强微裂缝簇 (Type V: 滤失系数扩大至 1.0e-3 m^2.5/s)
        np_holes = 6
        if n_frac == 1:
            fault_cluster_idx = 0
            has_fault = True
            weights = [1.0]
            cf_list = [0.009]
            kleak_list = [1.0e-3]
            dp_list = [0.0110]
            cd_list = [0.68]
        elif n_frac == 4:
            fault_cluster_idx = 1
            has_fault = True
            weights = [0.25, 0.30, 0.25, 0.20]
            cf_list = [0.010, 0.009, 0.010, 0.008]
            kleak_list = [1.0e-4, 1.0e-3, 1.0e-4, 0.9e-4]
            dp_list = [0.0105, 0.0110, 0.0105, 0.0100]
            cd_list = [0.65, 0.68, 0.65, 0.64]
        else:
            fault_cluster_idx = 1 if n_frac >= 2 else 0
            has_fault = True
            w_fault = min(0.30, 1.0 / n_frac + 0.05) if n_frac >= 2 else 1.0
            rem_w = (1.0 - w_fault) / (n_frac - 1) if n_frac >= 2 else 0.0
            weights = [rem_w] * n_frac
            weights[fault_cluster_idx] = w_fault
            cf_list = [0.010] * n_frac
            cf_list[fault_cluster_idx] = 0.009
            kleak_list = [1.0e-4] * n_frac
            kleak_list[fault_cluster_idx] = 1.0e-3
            dp_list = [0.0105] * n_frac
            dp_list[fault_cluster_idx] = 0.0110
            cd_list = [0.65] * n_frac
            cd_list[fault_cluster_idx] = 0.68

        kp_list = [
            float(1.0 / (2.0 * G * (cd_list[j] ** 2) * ((np_holes * np.pi * (dp_list[j] ** 2) / 4.0) ** 2)))
            for j in range(n_frac)
        ]
        types_list = [
            classify_fracture_type(weights[j], cf_list[j], kleak_list[j], kp_list[j], is_fault=(j == fault_cluster_idx))
            for j in range(n_frac)
        ]
    else:
        raise ValueError(
            f"未知的预设工况类型 '{scenario_type}'。支持的类型包括: "
            "'perfect_uniform' (完美均匀型), 'heel_dominant' (跟部突进型), "
            "'toe_dominant' (趾端优势型), 'screenout_dead' (单簇砂堵死簇型), "
            "'fault_leaking' (沟通断层强漏失型)。"
        )

    # 前向物理稳态求解真实井口水头 H0* 与流量分布
    grid = MocGrid.create(L=wellbore_length, wavespeed=wavespeed, dt=dt, tf=tf)
    frac_indices, sorted_pos, order = grid.map_fracture_positions(positions)
    (
        H0_realized,
        _,
        _,
        _,
        _,
        q_frac_ss,
        alpha_realized,
        mass_residual,
    ) = solve_physical_steady_state(
        L=wellbore_length,
        N=grid.N,
        dx=grid.dx,
        D=wellbore_d,
        area=wellbore_area,
        nu=1.0e-6,
        K_D=0.045e-3 / wellbore_d,
        V0=v0,
        g=G,
        toe_bc="dead_end",
        frac_indices=frac_indices,
        frac_kleak_arr=np.array([kleak_list[k] for k in order], dtype=np.float64),
        frac_Kp_arr=np.array([kp_list[k] for k in order], dtype=np.float64),
        sorted_pos=sorted_pos,
        H_ext=H_ext,
        H0_max=np.inf,
    )

    alpha_orig = np.zeros(n_frac, dtype=np.float64)
    q_orig = np.zeros(n_frac, dtype=np.float64)
    for k, orig_idx in enumerate(order):
        alpha_orig[orig_idx] = alpha_realized[k]
        q_orig[orig_idx] = q_frac_ss[k]

    # 根据真实稳态分流更新分类标签
    types_list = [
        classify_fracture_type(
            alpha_orig[j],
            cf_list[j],
            kleak_list[j],
            kp_list[j],
            is_fault=(has_fault and j == fault_cluster_idx),
        )
        for j in range(n_frac)
    ]

    scenario_dict = {
        "sample_id": 0,
        "scenario_type": scenario_type,
        "n_frac": n_frac,
        "fracture_positions": positions,
        "fracture_Cf": cf_list,
        "fracture_kleak": kleak_list,
        "fracture_alpha_ss": alpha_orig.tolist(),
        "fracture_Q_ss": q_orig.tolist(),
        "fracture_Kp": kp_list,
        "fracture_dp": dp_list,
        "fracture_cd": cd_list,
        "fracture_types": types_list,
        "has_fault": has_fault,
        "fault_cluster_idx": fault_cluster_idx,
        "wellbore_length": wellbore_length,
        "wavespeed": wavespeed,
        "initial_velocity": v0,
        "pump_closure_duration": tc,
        "ramp_type": ramp_type,
        "perf_num_holes": np_holes,
        "perf_diameter": dp_list[0],
        "perf_cd": cd_list[0],
        "H_ext": H_ext,
        "initial_head": H0_realized,
        "H0_realized": H0_realized,
        "steady_mass_residual": mass_residual,
        "dt": dt,
        "tf": tf,
    }
    # 覆盖任何外部自定义参数
    for k, v in kwargs.items():
        if k not in ("scenario_type", "fracture_positions"):
            scenario_dict[k] = v

    return scenario_dict
