# -*- coding: utf-8 -*-
"""实时 MOC 全井筒压力观测页面。

启动：streamlit run moc_simulate/visualize_app.py
"""
from __future__ import annotations

from pathlib import Path
import sys

# Streamlit 将脚本所在目录置为导入根目录。补入项目根目录后，既可从项目根
# 目录启动，也可双击/从任意工作目录启动，而不会出现 ``moc_simulate`` 找不到。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from moc_simulate.config import FRACTURE_CONFIG, SIM_CONFIG, WELL_CONFIG
from moc_simulate.visualization import (
    LiveModelSnapshot,
    LiveVisualizationSnapshot,
    VisualizationRunConfig,
    start_live_visualization,
)


STEADY_COLOR = "#28c4d5"
BRUNONE_COLOR = "#f07a4d"
ACCENT_COLOR = "#e0b129"
PANEL_COLOR = "#111e28"
PLOT_COLOR = "#0d1821"
GRID_COLOR = "#2a3d4c"
TEXT_COLOR = "#dce8ec"
MUTED_COLOR = "#91a4af"
PRESSURE_COLORS = [
    [0.00, "#27235b"],
    [0.18, "#315fc7"],
    [0.38, "#28bfd0"],
    [0.50, "#59df9b"],
    [0.67, "#d8e93f"],
    [0.84, "#f77f2a"],
    [1.00, "#b6150f"],
]


st.set_page_config(page_title="MOC 全井筒压力观测", page_icon="◒", layout="wide")


def _inject_style() -> None:
    """建立与确认原型一致的深色科研仪表盘外观。"""
    st.markdown(
        """
        <style>
        :root {
            --surface: #071017;
            --panel: #111e28;
            --edge: #294150;
            --text: #dce8ec;
            --muted: #91a4af;
            --accent: #e0b129;
        }
        .stApp { background: var(--surface); color: var(--text); }
        .block-container { max-width: 2000px; padding-top: 1.7rem; padding-bottom: 2.6rem; }
        [data-testid="stSidebar"] { background: #101c25; border-right: 1px solid #1b303e; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stSidebar"] [data-baseweb="input"] {
            background: #1a2b37; border: 1px solid #335062; border-radius: 6px;
        }
        [data-testid="stSidebar"] input { color: #eaf2f3 !important; }
        [data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] { background: var(--accent); }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 4rem; border: 0; border-radius: 7px;
            background: #dfb126; color: #172026; font-size: 1.12rem; font-weight: 700;
        }
        [data-testid="stSidebar"] .stButton > button:hover { background: #f1c232; color: #101820; }
        h1, h2, h3, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            font-family: Bahnschrift, "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
            color: var(--text);
        }
        .dashboard-header {
            background: var(--panel); border-radius: 0 0 13px 13px; padding: 1.3rem 2rem 1.25rem;
            border-bottom: 1px solid #152c38; margin-bottom: 1.35rem;
        }
        .dashboard-header h1 { font-size: 1.72rem; margin: 0; font-weight: 640; }
        .dashboard-header p { color: var(--muted); margin: .35rem 0 0; font-size: 1rem; }
        .status-chip {
            display: inline-block; padding: .48rem 1.1rem; border-radius: 999px;
            background: #123f37; border: 1px solid #1c6c5c; color: #72e0bb; font-weight: 650;
        }
        .scope-note { color: var(--muted); font-size: .88rem; line-height: 1.6; }
        [data-testid="stPlotlyChart"] { background: var(--panel); border: 1px solid var(--edge); border-radius: 13px; padding: .5rem; }
        [data-testid="stAlert"] { background: #10202b; color: var(--text); border-color: #315061; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _run_inputs() -> VisualizationRunConfig:
    """读取限定的科研工况输入，不修改集中配置文件。"""
    st.sidebar.markdown("# MOC 水击波场")
    st.sidebar.caption("实时压力场对比 · 单工况")
    st.sidebar.markdown("### 仿真参数")
    total_time = st.sidebar.number_input(
        "总仿真时长 tf [s]", min_value=0.01, value=float(SIM_CONFIG["tf"]), step=1.0
    )
    shut_in = st.sidebar.number_input(
        "停泵时刻 ts [s]", min_value=0.0, value=float(SIM_CONFIG["ts"]), step=0.1
    )
    wavespeed = st.sidebar.number_input(
        "波速 a [m/s]", min_value=1.0, value=float(WELL_CONFIG["wavespeed"]), step=10.0
    )
    velocity = st.sidebar.number_input(
        "初始流速 V₀ [m/s]", min_value=0.001, value=float(WELL_CONFIG["V0"]),
        step=0.1, format="%.3f"
    )
    st.sidebar.markdown("### 裂缝边界")
    fracture_count = st.sidebar.slider("裂缝数量", min_value=1, max_value=8, value=4)
    first_fracture = st.sidebar.number_input("首缝深度 [m]", min_value=1.0, value=4100.0, step=10.0)
    spacing = st.sidebar.number_input("等间距 D [m]", min_value=0.01, value=50.0, step=5.0)
    compliance = st.sidebar.number_input(
        "裂缝柔度 Cf [m²]", min_value=1.0e-9, value=float(FRACTURE_CONFIG["Cf"]),
        step=1.0e-6, format="%.2e"
    )
    leakoff = st.sidebar.number_input(
        "滤失系数 kₗₑₐₖ", min_value=0.0, value=float(FRACTURE_CONFIG["kleak"]),
        step=1.0e-5, format="%.2e"
    )
    external_head = st.sidebar.number_input(
        "地层水头 H_ext [m]", value=float(FRACTURE_CONFIG["H_ext"]), step=10.0
    )
    st.sidebar.caption(f"求解时间步固定为 {SIM_CONFIG['dt']:.1e} s；趾端采用定水头边界。")
    return VisualizationRunConfig(
        total_time_s=float(total_time),
        shut_in_time_s=float(shut_in),
        wavespeed_mps=float(wavespeed),
        initial_velocity_mps=float(velocity),
        fracture_count=int(fracture_count),
        first_fracture_m=float(first_fracture),
        fracture_spacing_m=float(spacing),
        fracture_compliance_m2=float(compliance),
        leakoff_coefficient=float(leakoff),
        external_head_m=float(external_head),
    )


def _current_model_time(model: LiveModelSnapshot) -> float:
    return float(model.frame_t[-1]) if len(model.frame_t) else 0.0


def _pressure_cloud_figure(snapshot: LiveVisualizationSnapshot) -> go.Figure:
    """绘制已计算部分的两幅全井筒压力波动云图：横深度、纵时间。"""
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.075,
        subplot_titles=("稳态达西摩阻", "Brunone 非定常摩阻"),
    )
    for column, model in ((1, snapshot.steady), (2, snapshot.brunone)):
        # 显示相对于初始静压的波动，两个模型固定同一色标，才可直观比较振幅。
        pressure_change = model.head_frames - snapshot.initial_head_m
        figure.add_trace(
            go.Heatmap(
                x=snapshot.x_grid,
                y=model.frame_t,
                z=pressure_change,
                colorscale=PRESSURE_COLORS,
                zmin=-snapshot.pressure_scale_head_m,
                zmax=snapshot.pressure_scale_head_m,
                showscale=column == 2,
                colorbar={
                    "title": {"text": "ΔH [m]", "side": "right"},
                    "x": 1.01,
                    "thickness": 13,
                    "len": 0.82,
                    "tickfont": {"color": MUTED_COLOR, "size": 10},
                },
                hovertemplate="井深=%{x:.1f} m<br>时间=%{y:.4f} s<br>ΔH=%{z:.3f} m<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.add_hline(
            y=_current_model_time(model), line_color=ACCENT_COLOR, line_width=1.5,
            annotation_text="当前帧", annotation_position="top right",
            annotation_font_color=ACCENT_COLOR, row=1, col=column,
        )

    figure.update_xaxes(
        title_text="井深 x [m]", showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
        tickfont={"color": MUTED_COLOR}, title_font={"color": MUTED_COLOR},
    )
    figure.update_yaxes(
        autorange="reversed", showgrid=True, gridcolor=GRID_COLOR,
        zeroline=False, tickfont={"color": MUTED_COLOR}, title_font={"color": MUTED_COLOR},
    )
    figure.update_yaxes(title_text="模拟时间 t [s]", row=1, col=1)
    figure.update_yaxes(title_text="", row=1, col=2)
    figure.update_layout(
        height=520, margin={"l": 50, "r": 55, "t": 65, "b": 48},
        paper_bgcolor=PANEL_COLOR, plot_bgcolor=PLOT_COLOR,
        font={"family": "Bahnschrift, Microsoft YaHei, sans-serif", "color": TEXT_COLOR},
        title={"text": "压力波动 ΔH(x, t) · 井深 0–5000 m", "x": 0.02, "font": {"size": 16, "color": TEXT_COLOR}},
    )
    figure.update_annotations(font={"color": TEXT_COLOR, "size": 16})
    return figure


def _wellhead_figure(snapshot: LiveVisualizationSnapshot) -> go.Figure:
    """绘制两模型截至当前时刻的井口水头（压力）响应。"""
    figure = go.Figure()
    for model, color, label in (
        (snapshot.steady, STEADY_COLOR, "稳态达西"),
        (snapshot.brunone, BRUNONE_COLOR, "Brunone"),
    ):
        figure.add_trace(
            go.Scatter(
                x=model.wellhead_t,
                y=model.wellhead_head,
                mode="lines",
                name=label,
                line={"color": color, "width": 2.5},
                hovertemplate="时间=%{x:.4f} s<br>井口水头=%{y:.3f} m<extra>" + label + "</extra>",
            )
        )
    figure.add_vline(
        x=snapshot.run_config.shut_in_time_s, line_dash="dash", line_color=ACCENT_COLOR,
        line_width=1.4, annotation_text="停泵", annotation_font_color=ACCENT_COLOR,
    )
    figure.add_vline(
        x=snapshot.current_time_s, line_color=ACCENT_COLOR, line_width=1.5,
        annotation_text="当前帧", annotation_font_color=ACCENT_COLOR,
    )
    figure.update_layout(
        title={"text": "井口压力波动 H_wh(t)", "x": 0.02, "font": {"size": 18, "color": TEXT_COLOR}},
        height=370, margin={"l": 58, "r": 28, "t": 62, "b": 48},
        paper_bgcolor=PANEL_COLOR, plot_bgcolor=PLOT_COLOR,
        font={"family": "Bahnschrift, Microsoft YaHei, sans-serif", "color": TEXT_COLOR},
        legend={"orientation": "h", "x": 0.82, "y": 1.10, "font": {"color": TEXT_COLOR}},
        xaxis={
            "title": "模拟时间 t [s]", "range": [0.0, snapshot.run_config.total_time_s],
            "gridcolor": GRID_COLOR, "zeroline": False, "tickfont": {"color": MUTED_COLOR},
            "title_font": {"color": MUTED_COLOR},
        },
        yaxis={
            "title": "井口水头 H [m]", "gridcolor": GRID_COLOR, "zeroline": False,
            "tickfont": {"color": MUTED_COLOR}, "title_font": {"color": MUTED_COLOR},
        },
    )
    return figure


@st.fragment(run_every=0.35)
def _live_dashboard() -> None:
    """周期性重绘正在增长的云图；求解线程不阻塞页面响应。"""
    simulation = st.session_state.get("moc_live_simulation")
    if simulation is None:
        st.info("在左侧设置工况后点击“开始仿真”。页面将从 t = 0 起实时绘制两种摩阻模型的压力云图。")
        return

    snapshot = simulation.snapshot()
    if snapshot.error:
        st.error(f"仿真已停止：{snapshot.error}")
        return

    completed = snapshot.is_finished
    st.markdown(
        "<div class='dashboard-header'><h1>全井筒瞬态压力观测</h1>"
        f"<p>当前物理时刻 t = {snapshot.current_time_s:.4f} s　·　"
        f"稳态 {snapshot.steady.progress:.0%}　·　Brunone {snapshot.brunone.progress:.0%}</p>"
        f"<span class='status-chip'>{'完成 · 结果已保留' if completed else 'LIVE · 计算中'}</span></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_pressure_cloud_figure(snapshot), width="stretch", config={"displaylogo": False})
    st.plotly_chart(_wellhead_figure(snapshot), width="stretch", config={"displaylogo": False})
    if completed and not st.session_state.get("moc_live_completion_rerun", False):
        # fragment 只刷新主面板；完成后再触发一次全页重绘以恢复侧栏按钮。
        st.session_state.moc_live_completion_rerun = True
        st.rerun(scope="app")


def main() -> None:
    _inject_style()
    run_config = _run_inputs()
    existing = st.session_state.get("moc_live_simulation")
    running = existing is not None and not existing.snapshot().is_finished
    start = st.sidebar.button("开始仿真", type="primary", width="stretch", disabled=running)
    if running:
        st.sidebar.success("● 实时求解中\n\n页面每 0.35 秒刷新一次云图。")
    else:
        st.sidebar.markdown("<p class='scope-note'>只显示：<br>· 两个全井筒压力云图<br>· 井口压力波动曲线</p>", unsafe_allow_html=True)

    if start:
        try:
            # 每个模型至多保留 400 帧，MOC 原始步长和求解过程不变。
            st.session_state.moc_live_simulation = start_live_visualization(run_config)
            st.session_state.moc_live_completion_rerun = False
            st.rerun()
        except (TypeError, ValueError, RuntimeError) as exc:
            st.error(f"无法开始该工况：{exc}")

    _live_dashboard()


if __name__ == "__main__":
    main()
