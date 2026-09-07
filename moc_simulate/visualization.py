# -*- coding: utf-8 -*-
"""MOC 水击可视化的数据适配与导出。

本模块刻意不修改数值求解器。它以完整的 1 ms 节点时程和受限数量的
空间快照为输入，组织成供 Streamlit/Plotly 使用的稳定数据接口。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
import threading
from typing import Any, Mapping

import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure

from moc_simulate.config import FRACTURE_CONFIG, SIM_CONFIG, WELL_CONFIG
from moc_simulate.wellbore_moc import (
    G,
    MocConfig,
    darcy_friction_factor,
    simulate_wellbore,
)


MAX_DISPLAY_FRAMES = 1200
MAX_TRACE_POINTS = 6000
MAX_LIVE_FRAMES = 400

# Windows 工作站的科研图导出需明确选择中文字体，避免 PNG 中出现缺字方框。
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class VisualizationRunConfig:
    """网页可编辑的研究参数；其余参数沿用 ``moc_simulate.config``。"""

    total_time_s: float = float(SIM_CONFIG["tf"])
    shut_in_time_s: float = float(SIM_CONFIG["ts"])
    wavespeed_mps: float = float(WELL_CONFIG["wavespeed"])
    initial_velocity_mps: float = float(WELL_CONFIG["V0"])
    fracture_count: int = 4
    first_fracture_m: float = 4100.0
    fracture_spacing_m: float = 50.0
    fracture_compliance_m2: float = float(FRACTURE_CONFIG["Cf"])
    leakoff_coefficient: float = float(FRACTURE_CONFIG["kleak"])
    external_head_m: float = float(FRACTURE_CONFIG["H_ext"])
    # 为可重复的快速验证保留程序接口；网页不暴露该高级参数。
    time_step_s: float = float(SIM_CONFIG["dt"])
    max_display_frames: int = MAX_DISPLAY_FRAMES

    @property
    def fracture_positions_m(self) -> list[float]:
        return [
            self.first_fracture_m + i * self.fracture_spacing_m
            for i in range(self.fracture_count)
        ]


@dataclass
class ModelVisualResult:
    """一种摩阻模型的节点时程与降采样空间场。"""

    friction_model: str
    frame_t: np.ndarray
    x_grid: np.ndarray
    head_frames: np.ndarray
    velocity_frames: np.ndarray
    timestamps: np.ndarray
    wellhead_head: np.ndarray
    wellhead_velocity: np.ndarray
    toe_head: np.ndarray
    toe_velocity: np.ndarray
    fracture_heads: np.ndarray
    fracture_flows: np.ndarray
    fracture_indices: list[int]
    fracture_positions_m: list[float]
    arrival_times_s: list[float]
    toe_arrival_s: float
    grid_dx_m: float
    adjusted_wavespeed_mps: float
    joukowsky_head_m: float


@dataclass
class VisualizationPair:
    """相同物理工况下 steady 与 Brunone 的并列结果。"""

    run_config: VisualizationRunConfig
    steady: ModelVisualResult
    brunone: ModelVisualResult


@dataclass
class LiveModelSnapshot:
    """一个摩阻模型截至当前计算时刻的可视化数据。"""

    friction_model: str
    frame_t: np.ndarray
    head_frames: np.ndarray
    wellhead_t: np.ndarray
    wellhead_head: np.ndarray
    progress: float
    finished: bool


@dataclass
class LiveVisualizationSnapshot:
    """供网页定时刷新读取的线程安全快照。"""

    run_config: VisualizationRunConfig
    x_grid: np.ndarray
    initial_head_m: float
    pressure_scale_head_m: float
    steady: LiveModelSnapshot
    brunone: LiveModelSnapshot
    error: str | None

    @property
    def is_finished(self) -> bool:
        return self.steady.finished and self.brunone.finished

    @property
    def current_time_s(self) -> float:
        return float(max(self.steady.frame_t[-1], self.brunone.frame_t[-1]))


class LiveVisualizationSimulation:
    """将两个 MOC 求解器的过程帧安全地推送给交互页面。

    求解仍采用原始时间步；仅将有限数量的完整空间场保留为页面帧，从而避免
    显示层占用完整时空场内存。两个模型各自在线程中运行，页面可在其中任一
    模型完成前读取当前帧。
    """

    def __init__(
        self,
        run_config: VisualizationRunConfig | Mapping[str, Any],
        max_frames: int = MAX_LIVE_FRAMES,
    ) -> None:
        self.run_config = coerce_run_config(run_config)
        self._moc_cfg = _make_moc_config(self.run_config, "steady")
        self._max_frames = max(2, min(int(max_frames), MAX_DISPLAY_FRAMES))
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._error: str | None = None
        initial_head, _ = _initial_field(self._moc_cfg)
        self._frames = {"steady": [initial_head], "brunone": [initial_head.copy()]}
        self._frame_times = {"steady": [0.0], "brunone": [0.0]}
        self._wellhead_t = {"steady": [0.0], "brunone": [0.0]}
        self._wellhead_head = {
            "steady": [float(initial_head[0])],
            "brunone": [float(initial_head[0])],
        }
        self._progress = {"steady": 0.0, "brunone": 0.0}
        self._finished = {"steady": False, "brunone": False}

    def start(self) -> "LiveVisualizationSimulation":
        if self._started:
            return self
        self._started = True
        for friction_model in ("steady", "brunone"):
            worker = threading.Thread(
                target=self._run_worker,
                args=(friction_model,),
                name=f"moc-live-{friction_model}",
                daemon=True,
            )
            self._threads.append(worker)
            worker.start()
        return self

    def _record_frame(
        self,
        friction_model: str,
        time_s: float,
        head: np.ndarray,
        _velocity: np.ndarray,
        step: int,
        n_steps: int,
    ) -> None:
        with self._lock:
            if time_s <= self._frame_times[friction_model][-1]:
                return
            self._frame_times[friction_model].append(float(time_s))
            self._frames[friction_model].append(np.asarray(head, dtype=np.float64))
            self._wellhead_t[friction_model].append(float(time_s))
            self._wellhead_head[friction_model].append(float(head[0]))
            self._progress[friction_model] = float(step / n_steps) if n_steps else 1.0

    def _run_worker(self, friction_model: str) -> None:
        try:
            moc_cfg = _make_moc_config(self.run_config, friction_model)
            interval = max(1, int(np.ceil(moc_cfg.n_steps / (self._max_frames - 1))))
            simulate_wellbore(
                moc_cfg,
                fracture_positions=self.run_config.fracture_positions_m,
                fracture_Cf=[self.run_config.fracture_compliance_m2] * self.run_config.fracture_count,
                fracture_kleak=[self.run_config.leakoff_coefficient] * self.run_config.fracture_count,
                H_ext=self.run_config.external_head_m,
                store_full_field=False,
                progress_callback=lambda t, h, v, step, total: self._record_frame(
                    friction_model, t, h, v, step, total
                ),
                progress_interval_steps=interval,
            )
            with self._lock:
                self._progress[friction_model] = 1.0
                self._finished[friction_model] = True
        except Exception as exc:  # 页面应显示计算错误，不让后台线程静默失败。
            with self._lock:
                self._error = str(exc)
                self._finished[friction_model] = True

    def snapshot(self) -> LiveVisualizationSnapshot:
        """返回不可变数组快照；页面可在工作线程写入期间安全绘图。"""
        with self._lock:
            def model_snapshot(friction_model: str) -> LiveModelSnapshot:
                return LiveModelSnapshot(
                    friction_model=friction_model,
                    frame_t=np.asarray(self._frame_times[friction_model], dtype=np.float64),
                    head_frames=np.asarray(self._frames[friction_model], dtype=np.float64),
                    wellhead_t=np.asarray(self._wellhead_t[friction_model], dtype=np.float64),
                    wellhead_head=np.asarray(self._wellhead_head[friction_model], dtype=np.float64),
                    progress=float(self._progress[friction_model]),
                    finished=bool(self._finished[friction_model]),
                )

            return LiveVisualizationSnapshot(
                run_config=self.run_config,
                x_grid=np.linspace(0.0, self._moc_cfg.wellbore_length, self._moc_cfg.N + 1),
                initial_head_m=float(self._moc_cfg.initial_head),
                pressure_scale_head_m=max(
                    1.0, 1.2 * self._moc_cfg.a_adj * self._moc_cfg.initial_velocity / G
                ),
                steady=model_snapshot("steady"),
                brunone=model_snapshot("brunone"),
                error=self._error,
            )


def start_live_visualization(
    run_config: VisualizationRunConfig | Mapping[str, Any],
    max_frames: int = MAX_LIVE_FRAMES,
) -> LiveVisualizationSimulation:
    """启动可实时读取的 steady/Brunone 双模型仿真。"""
    return LiveVisualizationSimulation(run_config, max_frames=max_frames).start()


def _as_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数值。") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} 必须是有限数值。")
    return parsed


def coerce_run_config(run_config: VisualizationRunConfig | Mapping[str, Any]) -> VisualizationRunConfig:
    """接受数据类或网页字典，并在构造求解器前完成参数规范化。"""
    if isinstance(run_config, VisualizationRunConfig):
        cfg = run_config
    elif isinstance(run_config, Mapping):
        defaults = asdict(VisualizationRunConfig())
        unknown = set(run_config) - set(defaults)
        if unknown:
            raise ValueError(f"未知可视化参数: {', '.join(sorted(unknown))}")
        defaults.update(run_config)
        try:
            defaults["fracture_count"] = int(defaults["fracture_count"])
            defaults["max_display_frames"] = int(defaults["max_display_frames"])
        except (TypeError, ValueError) as exc:
            raise ValueError("裂缝数量和显示帧数必须为整数。") from exc
        for name in (
            "total_time_s", "shut_in_time_s", "wavespeed_mps",
            "initial_velocity_mps", "first_fracture_m", "fracture_spacing_m",
            "fracture_compliance_m2", "leakoff_coefficient", "external_head_m",
            "time_step_s",
        ):
            defaults[name] = _as_float(defaults[name], name)
        cfg = VisualizationRunConfig(**defaults)
    else:
        raise TypeError("run_config 必须是 VisualizationRunConfig 或参数字典。")
    validate_run_config(cfg)
    return cfg


def _make_moc_config(run_config: VisualizationRunConfig, friction_model: str) -> MocConfig:
    """以当前集中配置的非交互参数构建一个可视化算例。"""
    return MocConfig(
        wellbore_length=float(WELL_CONFIG["L"]),
        wellbore_diameter=float(WELL_CONFIG["wellbore_diameter"]),
        fluid_density=float(WELL_CONFIG["fluid_density"]),
        fluid_viscosity=float(WELL_CONFIG["fluid_viscosity"]),
        wavespeed=run_config.wavespeed_mps,
        roughness_height=float(WELL_CONFIG["roughness_height"]),
        friction_model=friction_model,
        dt=run_config.time_step_s,
        tf=run_config.total_time_s,
        wellhead_bc="velocity_step",
        pump_shut_time=run_config.shut_in_time_s,
        initial_velocity=run_config.initial_velocity_mps,
        initial_head=float(WELL_CONFIG["H0"]),
        theta=float(WELL_CONFIG["theta"]),
        # 与 leakoff_multi 的当前研究算例一致：趾端定水头边界。
        toe_bc="reservoir",
        toe_head=float(WELL_CONFIG["H0"]),
    )


def validate_run_config(run_config: VisualizationRunConfig) -> None:
    """验证网页参数，包括 MOC 网格对齐后的裂缝重合。"""
    numeric_positive = {
        "总仿真时长": run_config.total_time_s,
        "波速": run_config.wavespeed_mps,
        "初始流速": run_config.initial_velocity_mps,
        "首缝深度": run_config.first_fracture_m,
        "裂缝间距": run_config.fracture_spacing_m,
        "裂缝柔度 Cf": run_config.fracture_compliance_m2,
        "时间步长": run_config.time_step_s,
    }
    for label, value in numeric_positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} 必须大于 0。")
    if not np.isfinite(run_config.leakoff_coefficient) or run_config.leakoff_coefficient < 0:
        raise ValueError("滤失系数必须大于或等于 0。")
    if not np.isfinite(run_config.external_head_m):
        raise ValueError("地层孔隙压力水头必须为有限数值。")
    if not 1 <= run_config.fracture_count <= 8:
        raise ValueError("裂缝数量必须在 1 到 8 之间。")
    if run_config.shut_in_time_s < 0 or run_config.shut_in_time_s > run_config.total_time_s:
        raise ValueError("停泵时刻必须位于 0 到总仿真时长之间。")
    if not 2 <= run_config.max_display_frames <= MAX_DISPLAY_FRAMES:
        raise ValueError(f"显示帧数必须在 2 到 {MAX_DISPLAY_FRAMES} 之间。")

    probe = _make_moc_config(run_config, "steady")
    positions = run_config.fracture_positions_m
    if positions[-1] >= probe.wellbore_length:
        raise ValueError(
            f"最深裂缝 {positions[-1]:.2f} m 超出井筒长度 {probe.wellbore_length:.2f} m。"
        )
    indices = [max(1, min(probe.N - 1, round(position / probe.dx))) for position in positions]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "裂缝在当前 MOC 网格上对齐到同一节点；请增大裂缝间距或减小时间步长。"
        )


def _display_steps(cfg: MocConfig, max_frames: int) -> np.ndarray:
    frame_count = min(max_frames, cfg.n_steps + 1)
    return np.unique(np.rint(np.linspace(0, cfg.n_steps, frame_count)).astype(int))


def _initial_field(cfg: MocConfig) -> tuple[np.ndarray, np.ndarray]:
    """复刻求解器的 t=0 初始化，用于补齐不保存的第 0 帧。"""
    x_grid = np.linspace(0.0, cfg.wellbore_length, cfg.N + 1)
    head = np.full(cfg.N + 1, cfg.initial_head, dtype=np.float64)
    velocity = np.full(cfg.N + 1, cfg.initial_velocity, dtype=np.float64)
    if cfg.toe_bc == "dead_end":
        velocity[-1] = 0.0
        head[-1] = head[-2] + (cfg.a_adj / G) * velocity[-2]
    else:
        velocity[-1] = cfg.initial_velocity
        head[-1] = cfg.toe_head
        re0 = abs(cfg.initial_velocity) * cfg.wellbore_diameter / cfg.fluid_viscosity
        friction = darcy_friction_factor(
            re0, cfg.roughness_height / cfg.wellbore_diameter, cfg.friction_model
        )
        slope = friction * cfg.initial_velocity * abs(cfg.initial_velocity) / (
            2.0 * G * cfg.wellbore_diameter
        )
        head[:-1] = cfg.toe_head + slope * (cfg.wellbore_length - x_grid[:-1])
    return head, velocity


def _run_model(run_config: VisualizationRunConfig, friction_model: str) -> ModelVisualResult:
    moc_cfg = _make_moc_config(run_config, friction_model)
    steps = _display_steps(moc_cfg, run_config.max_display_frames)
    snapshot_times = (steps[steps > 0] * moc_cfg.dt_adj).tolist()
    raw = simulate_wellbore(
        moc_cfg,
        fracture_positions=run_config.fracture_positions_m,
        fracture_Cf=[run_config.fracture_compliance_m2] * run_config.fracture_count,
        fracture_kleak=[run_config.leakoff_coefficient] * run_config.fracture_count,
        H_ext=run_config.external_head_m,
        store_full_field=False,
        snapshot_times=snapshot_times,
    )
    initial_head, initial_velocity = _initial_field(moc_cfg)
    head_frames: list[np.ndarray] = []
    velocity_frames: list[np.ndarray] = []
    for step in steps:
        if step == 0:
            head_frames.append(initial_head)
            velocity_frames.append(initial_velocity)
            continue
        snapshot = raw["snapshots"].get(int(step))
        if snapshot is None:
            raise RuntimeError(f"未保存第 {step} 个显示快照。")
        head_frames.append(snapshot["H"])
        velocity_frames.append(snapshot["V"])

    fracture_indices = [int(i) for i in raw["fracture_indices"]]
    aligned_positions = [float(raw["x_grid"][i]) for i in fracture_indices]
    arrivals = [
        run_config.shut_in_time_s + 2.0 * position / moc_cfg.a_adj
        for position in aligned_positions
    ]
    return ModelVisualResult(
        friction_model=friction_model,
        frame_t=steps.astype(np.float64) * moc_cfg.dt_adj,
        x_grid=np.asarray(raw["x_grid"], dtype=np.float64),
        head_frames=np.asarray(head_frames, dtype=np.float64),
        velocity_frames=np.asarray(velocity_frames, dtype=np.float64),
        timestamps=np.asarray(raw["timestamps"], dtype=np.float64),
        wellhead_head=np.asarray(raw["wellhead_head"], dtype=np.float64),
        wellhead_velocity=np.asarray(raw["wellhead_velocity"], dtype=np.float64),
        toe_head=np.asarray(raw["toe_head"], dtype=np.float64),
        toe_velocity=np.asarray(raw["toe_velocity"], dtype=np.float64),
        fracture_heads=np.asarray(raw.get("fracture_heads", np.empty((len(raw["timestamps"]), 0))), dtype=np.float64),
        fracture_flows=np.asarray(raw.get("fracture_Qs", np.empty((len(raw["timestamps"]), 0))), dtype=np.float64),
        fracture_indices=fracture_indices,
        fracture_positions_m=aligned_positions,
        arrival_times_s=arrivals,
        toe_arrival_s=run_config.shut_in_time_s + 2.0 * moc_cfg.wellbore_length / moc_cfg.a_adj,
        grid_dx_m=float(moc_cfg.dx),
        adjusted_wavespeed_mps=float(moc_cfg.a_adj),
        joukowsky_head_m=float(moc_cfg.a_adj * moc_cfg.initial_velocity / G),
    )


def run_visualization_pair(
    run_config: VisualizationRunConfig | Mapping[str, Any],
) -> VisualizationPair:
    """运行同一工况下的 steady 与 Brunone MOC，并返回可视化数据。"""
    cfg = coerce_run_config(run_config)
    return VisualizationPair(
        run_config=cfg,
        steady=_run_model(cfg, "steady"),
        brunone=_run_model(cfg, "brunone"),
    )


def trace_indices(timestamps: np.ndarray, max_points: int = MAX_TRACE_POINTS) -> np.ndarray:
    """获取等距的时程绘图索引；原始节点时程不受影响。"""
    if len(timestamps) <= max_points:
        return np.arange(len(timestamps))
    return np.unique(np.rint(np.linspace(0, len(timestamps) - 1, max_points)).astype(int))


def pair_metadata(pair: VisualizationPair) -> dict[str, Any]:
    """生成不含数组的可复现参数与派生指标。"""
    def model_metadata(model: ModelVisualResult) -> dict[str, Any]:
        return {
            "friction_model": model.friction_model,
            "grid_dx_m": model.grid_dx_m,
            "adjusted_wavespeed_mps": model.adjusted_wavespeed_mps,
            "joukowsky_head_m": model.joukowsky_head_m,
            "fracture_indices": model.fracture_indices,
            "fracture_positions_m": model.fracture_positions_m,
            "arrival_times_s": model.arrival_times_s,
            "toe_arrival_s": model.toe_arrival_s,
            "frame_count": int(len(model.frame_t)),
            "frame_interval_s": float(np.median(np.diff(model.frame_t))) if len(model.frame_t) > 1 else 0.0,
        }

    return {
        "run_config": asdict(pair.run_config),
        "steady": model_metadata(pair.steady),
        "brunone": model_metadata(pair.brunone),
    }


def export_pair_json(pair: VisualizationPair) -> bytes:
    return json.dumps(pair_metadata(pair), ensure_ascii=False, indent=2).encode("utf-8")


def export_pair_npz(pair: VisualizationPair) -> bytes:
    """导出两种模型的完整节点时程与降采样时空场。"""
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        frame_t=pair.steady.frame_t,
        x_grid=pair.steady.x_grid,
        fracture_positions_m=np.asarray(pair.steady.fracture_positions_m),
        steady_head_frames=pair.steady.head_frames,
        steady_velocity_frames=pair.steady.velocity_frames,
        steady_timestamps=pair.steady.timestamps,
        steady_wellhead_head=pair.steady.wellhead_head,
        steady_wellhead_velocity=pair.steady.wellhead_velocity,
        steady_toe_head=pair.steady.toe_head,
        steady_toe_velocity=pair.steady.toe_velocity,
        steady_fracture_heads=pair.steady.fracture_heads,
        steady_fracture_flows=pair.steady.fracture_flows,
        brunone_head_frames=pair.brunone.head_frames,
        brunone_velocity_frames=pair.brunone.velocity_frames,
        brunone_timestamps=pair.brunone.timestamps,
        brunone_wellhead_head=pair.brunone.wellhead_head,
        brunone_wellhead_velocity=pair.brunone.wellhead_velocity,
        brunone_toe_head=pair.brunone.toe_head,
        brunone_toe_velocity=pair.brunone.toe_velocity,
        brunone_fracture_heads=pair.brunone.fracture_heads,
        brunone_fracture_flows=pair.brunone.fracture_flows,
        metadata_json=np.array(json.dumps(pair_metadata(pair), ensure_ascii=False)),
    )
    return buffer.getvalue()


def export_summary_png(pair: VisualizationPair, frame_index: int | None = None) -> bytes:
    """生成可用于汇报的四联静态摘要图，不写入工作区。"""
    steady, brunone = pair.steady, pair.brunone
    index = len(steady.frame_t) // 2 if frame_index is None else int(frame_index)
    index = max(0, min(index, len(steady.frame_t) - 1))
    fig = Figure(figsize=(15, 10), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_profile, ax_wh, ax_frac, ax_diff = fig.subplots(2, 2).ravel()

    ax_profile.plot(steady.x_grid, steady.head_frames[index], color="#0d5c63", label="稳态达西")
    ax_profile.plot(brunone.x_grid, brunone.head_frames[index], color="#c75b39", label="Brunone")
    for position in steady.fracture_positions_m:
        ax_profile.axvline(position, color="#d7a62f", lw=0.9, ls="--")
    ax_profile.set(title=f"井筒水头剖面，t={steady.frame_t[index]:.3f} s", xlabel="井深 x [m]", ylabel="水头 H [m]")
    ax_profile.grid(alpha=0.25); ax_profile.legend()

    ax_wh.plot(steady.timestamps, steady.wellhead_head, color="#0d5c63", lw=0.9, label="稳态达西")
    ax_wh.plot(brunone.timestamps, brunone.wellhead_head, color="#c75b39", lw=0.9, label="Brunone")
    ax_wh.axvline(pair.run_config.shut_in_time_s, color="#323842", ls=":", label="停泵")
    ax_wh.set(title="井口水头时程", xlabel="时间 t [s]", ylabel="水头 H [m]")
    ax_wh.grid(alpha=0.25); ax_wh.legend()

    for number, position in enumerate(steady.fracture_positions_m):
        ax_frac.plot(steady.timestamps, steady.fracture_heads[:, number], lw=0.8, label=f"F{number + 1} {position:.0f}m")
    ax_frac.set(title="稳态达西：裂缝节点水头", xlabel="时间 t [s]", ylabel="水头 H [m]")
    ax_frac.grid(alpha=0.25); ax_frac.legend(fontsize=7, ncol=2)

    delta = brunone.wellhead_head - steady.wellhead_head
    ax_diff.plot(steady.timestamps, delta, color="#6d4c9a", lw=0.9)
    ax_diff.axhline(0.0, color="#323842", lw=0.7)
    ax_diff.set(title="井口差异：Brunone − 稳态达西", xlabel="时间 t [s]", ylabel="ΔH [m]")
    ax_diff.grid(alpha=0.25)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    return buffer.getvalue()
