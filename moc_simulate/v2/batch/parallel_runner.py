# -*- coding: utf-8 -*-
"""
moc_simulate.v2.batch.parallel_runner

多进程高并发批量正演仿真调度引擎
支持动态异常捕获、任务进度汇报与结果聚合。
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from moc_simulate.v2.configs import MocV2Config
from moc_simulate.v2.core.solver import simulate_v2
from moc_simulate.v2.signal.cepstrum_1d import compute_cepstrum_1d


def _run_single_simulation(sample: Dict[str, Any]) -> Dict[str, Any]:
    """单样本仿真工作者函数 (顶层函数供多进程序列化 pickle)"""
    try:
        cfg = MocV2Config(
            wellbore_length=float(sample.get("wellbore_length", 5000.0)),
            wellbore_diameter=float(sample.get("wellbore_diameter", 0.1397)),
            wavespeed=float(sample.get("wavespeed", 1450.0)),
            friction_model=str(sample.get("friction_model", "brunone")),
            dt=float(sample.get("dt", 1.0e-3)),
            tf=float(sample.get("tf", 20.0)),  # 批处理默认 20s 覆盖主要水击混响
            pump_shut_time=float(sample.get("pump_shut_time", 1.0)),
            pump_closure_duration=float(sample.get("pump_closure_duration", 1.0)),
            ramp_type=str(sample.get("ramp_type", "linear")),
            initial_velocity=float(sample.get("initial_velocity", 1.0)),
            initial_head=float(sample.get("initial_head", 300.0)),
            toe_bc="dead_end",
            perf_num_holes=int(sample.get("perf_num_holes", 6)),
            perf_diameter=float(sample.get("perf_diameter", 0.01)),
            perf_cd=float(sample.get("perf_cd", 0.65)),
        )

        sim_res = simulate_v2(
            cfg=cfg,
            fracture_positions=sample.get("fracture_positions"),
            fracture_Cf=sample.get("fracture_Cf"),
            fracture_kleak=sample.get("fracture_kleak"),
            fracture_inflow_weights=sample.get("fracture_inflow_weights"),
            fracture_Kp=sample.get("fracture_Kp"),
            H_ext=float(sample.get("H_ext", 100.0)),
        )

        # 提取 1D 倒谱特征 (默认提取，轻量化模式下可通过 compute_cepstrum=False 跳过以节约算力)
        compute_ceps = bool(sample.get("compute_cepstrum", True))
        if compute_ceps:
            ceps_res = compute_cepstrum_1d(
                time=sim_res["timestamps"],
                head=sim_res["wellhead_head"],
                wavespeed=cfg.wavespeed,
                ts=cfg.pump_shut_time,
                max_distance=cfg.wellbore_length,
            )
            cepstrum = ceps_res["cepstrum"]
            distances = ceps_res["distance"]
        else:
            cepstrum = None
            distances = None

        is_lightweight = bool(sample.get("lightweight", False))
        return {
            "sample_id": sample.get("sample_id", 0),
            "status": "success",
            "timestamps": sim_res["timestamps"],
            "wellhead_head": sim_res["wellhead_head"],
            "wellhead_velocity": sim_res["wellhead_velocity"],
            "toe_head": None if is_lightweight else sim_res["toe_head"],
            "fracture_heads": None if is_lightweight else sim_res["fracture_heads"],
            "cepstrum": cepstrum,
            "distances": distances,
            "sample_meta": sample,
        }
    except Exception as e:
        return {
            "sample_id": sample.get("sample_id", -1),
            "status": "error",
            "error_msg": str(e),
            "sample_meta": sample,
        }


class BatchRunner:
    """高并发批处理运行调度器"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or max(1, (os.cpu_count() or 4) - 2)

    def run(
        self,
        samples: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        并发推进所有样本的正演仿真，并按 sample_id 顺序返回结果。
        """
        n_total = len(samples)
        if n_total == 0:
            return []

        results: List[Dict[str, Any]] = []

        # 单进程串行快速路径：避免多进程启动开销与跨平台 spawn 陷阱
        if self.max_workers == 1:
            for i, sample in enumerate(samples):
                res = _run_single_simulation(sample)
                results.append(res)
                if progress_callback:
                    progress_callback(i + 1, n_total)
            results.sort(key=lambda r: r.get("sample_id", 0))
            return results

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(_run_single_simulation, sample): sample.get("sample_id", i)
                for i, sample in enumerate(samples)
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_id):
                res = future.result()
                results.append(res)
                completed += 1
                if progress_callback:
                    progress_callback(completed, n_total)

        # 按 sample_id 恢复原始输入顺序
        results.sort(key=lambda r: r.get("sample_id", 0))
        return results
