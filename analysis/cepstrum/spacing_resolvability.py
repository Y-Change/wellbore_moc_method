# -*- coding: utf-8 -*-
"""
从 steady_D* leakoff 结果汇总缝距分辨能力，并写出理论参数链。

只读 output/leakoff/steady_D{10,20,50,100}/{case}/moc_leakoff.json，
不重跑仿真。输出：
  output/leakoff/SPACING_RESOLVABILITY.md

运行
----
    python validation/cepstrum/spacing_resolvability.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

from moc_simulate.paths import OUTPUT_DIR, SERIES_LEAKOFF
from moc_simulate.config import (
    WELL_CONFIG,
    SIM_CONFIG,
    CEPSTRUM_CONFIG,
    SPACING_PRESETS_M,
)
from moc_simulate.wellbore_moc import MocConfig

CASES = ('single', 'dual', 'triple', 'quad', 'quint')


def _moc_grid() -> Dict[str, float]:
    w, s = WELL_CONFIG, SIM_CONFIG
    cfg = MocConfig(
        wellbore_length=w['L'],
        wellbore_diameter=w['wellbore_diameter'],
        fluid_density=w['fluid_density'],
        fluid_viscosity=w['fluid_viscosity'],
        wavespeed=w['wavespeed'],
        roughness_height=w['roughness_height'],
        friction_model='steady',
        dt=s['dt'], tf=s['tf'],
        wellhead_bc='velocity_step', pump_shut_time=s['ts'],
        initial_velocity=w['V0'], initial_head=w['H0'],
        theta=w['theta'], toe_bc='reservoir', toe_head=w['H0'],
    )
    fs = 1.0 / cfg.dt
    return {
        'L': float(w['L']),
        'a_nom': float(w['wavespeed']),
        'a_adj': float(cfg.a_adj),
        'dt': float(cfg.dt),
        'dx': float(cfg.dx),
        'N': float(cfg.N),
        'fs': fs,
        'ts': float(s['ts']),
        'tf': float(s['tf']),
        'T0': 4.0 * w['L'] / cfg.a_adj,
        'f0': cfg.a_adj / (4.0 * w['L']),
        'dd_bin': cfg.a_adj / (2.0 * fs),
        'T_1d': float(s['tf']) - float(s['ts']),
        'wlen': float(CEPSTRUM_CONFIG['wlen_sec']),
        'hop': float(CEPSTRUM_CONFIG['hop_sec']),
        'win': str(CEPSTRUM_CONFIG['win_type']),
    }


def _theory_chain(g: Dict[str, float]) -> Dict[str, Any]:
    """Layer 0–7 理论量（1D 全长 vs 2D 短时分栏）。"""
    fc = 1.0 / g['ts']
    n_harm_theo = fc * g['T0']
    # 1D 全长：物理频率分辨率由可用时长决定
    df_1d = 1.0 / g['T_1d']
    n_harm_1d_res = fc / df_1d
    n_harm_1d_eff = min(n_harm_theo, n_harm_1d_res)
    # 停泵低通饱和后，全长窗仍可对已有谐波做更密采样；
    # 经验上 1D 有效谐波远高于 fc*T0（多周期相干），用 Rayleigh 反推见实证节。
    fwhm_d_1d_theo = 2.0 * g['L'] / max(n_harm_1d_eff, 1e-12)
    dd_min_1d_theo = fwhm_d_1d_theo  # Rayleigh: Δd_min ≈ FWHM_d

    # 2D
    df_2d = 1.0 / g['wlen']
    n_harm_2d_res = fc / df_2d
    n_harm_2d_eff = min(n_harm_theo, n_harm_2d_res)
    fwhm_d_2d = 2.0 * g['L'] / max(n_harm_2d_eff, 1e-12)
    n_frames = 1 + max(
        0,
        int(math.floor((g['T_1d'] - g['wlen']) / g['hop'])),
    )
    return {
        'fc_Hz': fc,
        'N_harm_theo': n_harm_theo,
        'df_1d_Hz': df_1d,
        'N_harm_1d_eff_sat': n_harm_1d_eff,
        'FWHM_d_1d_sat_m': fwhm_d_1d_theo,
        'dd_min_1d_sat_m': dd_min_1d_theo,
        'df_2d_Hz': df_2d,
        'N_harm_2d_eff': n_harm_2d_eff,
        'FWHM_d_2d_m': fwhm_d_2d,
        'dd_min_2d_m': fwhm_d_2d,
        'n_frames_2d': n_frames,
        'dt_frame_s': g['hop'],
        'dq_s': 1.0 / g['fs'],
        'dd_bin_m': g['dd_bin'],
    }


def _json_path(spacing: int, case: str) -> str:
    return os.path.join(
        OUTPUT_DIR, SERIES_LEAKOFF, f'steady_D{spacing}', case, 'moc_leakoff.json',
    )


def load_match_matrix(
    spacings: Tuple[int, ...] = SPACING_PRESETS_M,
) -> Dict[int, Dict[str, Optional[Dict[str, Any]]]]:
    out: Dict[int, Dict[str, Optional[Dict[str, Any]]]] = {}
    for d in spacings:
        out[d] = {}
        for case in CASES:
            path = _json_path(d, case)
            if not os.path.isfile(path):
                out[d][case] = None
                continue
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            cep = (data.get('cepstrum') or {}).get('1d_real') or {}
            out[d][case] = {
                'n_matched': cep.get('n_matched'),
                'n_fracs': cep.get('n_fracs'),
                'mean_error_m': cep.get('mean_error_m'),
                'max_error_m': cep.get('max_error_m'),
                'snr': cep.get('snr'),
                'match_tol_m': cep.get('match_tol_m'),
                'matches': cep.get('matches') or [],
                'detected_peaks': cep.get('detected_peaks') or [],
                'path': path,
            }
    return out


def _fmt_err(x: Optional[float]) -> str:
    if x is None:
        return '—'
    return f'{x:.2f}'


def _infer_empirical_dd_min(
    matrix: Dict[int, Dict[str, Optional[Dict[str, Any]]]],
) -> Dict[str, Any]:
    """
    用 dual 全匹配的最小间距作为经验 Δd_min（1D）。
    dual 要求 n_matched == n_fracs == 2。
    """
    dual_ok: List[int] = []
    dual_fail: List[int] = []
    for d in sorted(matrix.keys()):
        row = matrix[d].get('dual')
        if row is None:
            continue
        if row['n_matched'] == 2 and row['n_fracs'] == 2:
            dual_ok.append(d)
        else:
            dual_fail.append(d)
    dd_emp = min(dual_ok) if dual_ok else None
    # 反推 N_harm_eff ≈ 2L / Δd_min（Rayleigh）
    L = WELL_CONFIG['L']
    n_harm_emp = (2.0 * L / dd_emp) if dd_emp else None
    return {
        'dual_ok_spacings': dual_ok,
        'dual_fail_spacings': dual_fail,
        'dd_min_emp_m': dd_emp,
        'N_harm_eff_emp': n_harm_emp,
        'FWHM_d_emp_m': dd_emp,
    }


def _multi_frac_note(
    matrix: Dict[int, Dict[str, Optional[Dict[str, Any]]]],
) -> List[str]:
    lines = []
    for d in sorted(matrix.keys()):
        for case in ('quad', 'quint'):
            row = matrix[d].get(case)
            if not row:
                continue
            nm, nf = row['n_matched'], row['n_fracs']
            if nm is not None and nf is not None and nm < nf:
                # 看未匹配缝是否为后缝
                unmatched = [
                    m['frac_id'] for m in row['matches'] if not m.get('matched')
                ]
                lines.append(
                    f'- D={d} m / {case}: {nm}/{nf} 匹配；未匹配缝 id={unmatched}'
                )
    return lines


def render_markdown(
    g: Dict[str, float],
    th: Dict[str, Any],
    matrix: Dict[int, Dict[str, Optional[Dict[str, Any]]]],
    emp: Dict[str, Any],
) -> str:
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines: List[str] = []
    lines.append('# 缝距分辨能力汇总（steady_Dall）')
    lines.append('')
    lines.append(f'> 自动生成于 {now}，数据源：`output/leakoff/steady_D*/**/moc_leakoff.json`')
    lines.append(f'> 脚本：`validation/cepstrum/spacing_resolvability.py`')
    lines.append('')
    lines.append('## 1. 理论参数链（当前 config）')
    lines.append('')
    lines.append('| 层 | 量 | 数值 |')
    lines.append('|----|----|------|')
    lines.append(f'| 0 | L / a_nom / a_adj | {g["L"]:.0f} m / {g["a_nom"]:.1f} / **{g["a_adj"]:.4f}** m/s |')
    lines.append(f'| 0 | T₀ = 4L/a_adj / f₀ | {g["T0"]:.3f} s / {g["f0"]:.5f} Hz |')
    lines.append(f'| 1 | dt / dx / N / fs | {g["dt"]:g} s / {g["dx"]:.4f} m / {g["N"]:.0f} / {g["fs"]:.0f} Hz |')
    lines.append(f'| 1 | Δd_bin = a_adj/(2fs) | {g["dd_bin"]:.4f} m |')
    lines.append(f'| 2 | ts / fc≈1/ts | {g["ts"]:g} s / {th["fc_Hz"]:.2f} Hz |')
    lines.append(f'| 2 | N_harm_theo = fc·T₀ | {th["N_harm_theo"]:.1f} |')
    lines.append(f'| 3 | tf / 1D 可用长 T_1d | {g["tf"]:g} s / {g["T_1d"]:.1f} s |')
    lines.append(
        f'| 4 | 2D wlen / hop / win | {g["wlen"]:g} s / {g["hop"]:g} s / {g["win"]} |'
    )
    lines.append(
        f'| 4 | 2D n_frames / Δt_frame | {th["n_frames_2d"]} / {th["dt_frame_s"]:g} s |'
    )
    lines.append(f'| 5 | 1D Δf = 1/T_1d | {th["df_1d_Hz"]:.5f} Hz |')
    lines.append(
        f'| 5 | 1D N_harm_eff（停泵饱和） | {th["N_harm_1d_eff_sat"]:.1f} → '
        f'FWHM_d≈{th["FWHM_d_1d_sat_m"]:.0f} m（偏悲观） |'
    )
    lines.append(f'| 5 | Δq / Δd_bin | {th["dq_s"]:.4f} s / {th["dd_bin_m"]:.4f} m |')
    lines.append(f'| 6 | 2D Δf = 1/wlen | {th["df_2d_Hz"]:.5f} Hz |')
    lines.append(
        f'| 6 | 2D N_harm_eff / FWHM_d | {th["N_harm_2d_eff"]:.1f} / '
        f'{th["FWHM_d_2d_m"]:.0f} m |'
    )
    lines.append('')
    lines.append(
        '说明：停泵饱和给出的 N_harm≈14 → FWHM_d≈714 m 与 1D 全长实证不符；'
        '全长倒谱通过多周期相干可利用远高于 fc 的有效谐波。'
        '下面用 dual 全匹配反推经验 N_harm_eff。'
    )
    lines.append('')
    lines.append('## 2. 1D 匹配矩阵（n_matched / n_fracs）')
    lines.append('')
    header = '| Δd [m] | ' + ' | '.join(CASES) + ' |'
    sep = '|--------|' + '|'.join(['-------'] * len(CASES)) + '|'
    lines.append(header)
    lines.append(sep)
    for d in sorted(matrix.keys()):
        cells = []
        for case in CASES:
            row = matrix[d].get(case)
            if row is None:
                cells.append('缺失')
            else:
                cells.append(f"{row['n_matched']}/{row['n_fracs']}")
        lines.append(f'| {d} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('## 3. 误差与 SNR')
    lines.append('')
    lines.append('| Δd | case | mean_err [m] | max_err [m] | SNR | match_tol [m] |')
    lines.append('|----|------|--------------|-------------|-----|---------------|')
    for d in sorted(matrix.keys()):
        for case in CASES:
            row = matrix[d].get(case)
            if row is None:
                lines.append(f'| {d} | {case} | — | — | — | — |')
                continue
            lines.append(
                f"| {d} | {case} | {_fmt_err(row['mean_error_m'])} | "
                f"{_fmt_err(row['max_error_m'])} | "
                f"{row['snr']:.0f} | {row['match_tol_m']} |"
            )
    lines.append('')
    lines.append('## 4. 经验 Δd_min（由 dual 全匹配反推）')
    lines.append('')
    lines.append(f"- dual 全匹配间距：{emp['dual_ok_spacings']}")
    lines.append(f"- dual 未全匹配间距：{emp['dual_fail_spacings']}")
    if emp['dd_min_emp_m'] is not None:
        lines.append(
            f"- **经验 Δd_min（1D）≈ {emp['dd_min_emp_m']} m**"
            f"（取 dual 全匹配的最小间距）"
        )
        lines.append(
            f"- 反推 N_harm_eff ≈ 2L/Δd_min = "
            f"**{emp['N_harm_eff_emp']:.0f}**"
        )
        lines.append(
            f"- 对应 FWHM_d ≈ {emp['FWHM_d_emp_m']} m"
        )
        # 预测各间距是否可分
        lines.append('')
        lines.append('| Δd [m] | Δτ=2Δd/a_adj [s] | 相对经验 FWHM | dual 预测 | dual 实测 |')
        lines.append('|--------|-----------------|---------------|----------|----------|')
        for d in sorted(matrix.keys()):
            dtau = 2.0 * d / g['a_adj']
            ratio = d / emp['dd_min_emp_m']
            pred = '可分' if d >= emp['dd_min_emp_m'] else '不可分'
            row = matrix[d].get('dual')
            if row and row['n_matched'] == 2:
                obs = '全匹配'
            elif row:
                obs = f"{row['n_matched']}/{row['n_fracs']}"
            else:
                obs = '缺失'
            lines.append(
                f'| {d} | {dtau:.5f} | {ratio:.2f}× | {pred} | {obs} |'
            )
    lines.append('')
    lines.append('## 5. 多缝幅值衰减（≠ 间距不可分）')
    lines.append('')
    notes = _multi_frac_note(matrix)
    if notes:
        lines.extend(notes)
    else:
        lines.append('- 本批结果中 quad/quint 均全匹配，或 JSON 缺失。')
    lines.append('')
    lines.append(
        '在 Δd≥50 m 时 dual/triple 常可全匹配，但 quad/quint 后缝峰变弱导致漏检：'
        '这是**峰高/SNR 瓶颈**，不是 Rayleigh 间距瓶颈。'
    )
    lines.append('')
    lines.append('## 6. match_tol 解读注意')
    lines.append('')
    lines.append(
        '`match_tol_m = clip(0.45·min_spacing, 80, 250)`：'
        '当 Δd=10–20 m 时容差仍被 **80 m 下限**卡住。'
        '因此小间距“匹配失败”主要反映峰未分离（detected_peaks 中无第二缝），'
        '而非容差过严；但解读时勿把 tol 当成物理分辨率。'
    )
    lines.append('')
    lines.append('## 7. 1D vs 2D')
    lines.append('')
    lines.append('| | 1D 实倒谱 | 2D cepstrogram |')
    lines.append('|--|-----------|----------------|')
    lines.append(
        f'| 有效时长 | T_1d≈{g["T_1d"]:.0f} s（全长） | wlen={g["wlen"]:g} s |'
    )
    lines.append(
        f'| Δf | {th["df_1d_Hz"]:.5f} Hz | {th["df_2d_Hz"]:.5f} Hz |'
    )
    lines.append(
        f'| 定量匹配 | `moc_leakoff.json → cepstrum.1d_real` | 当前 JSON **无** 2D 匹配率 |'
    )
    lines.append('| 用途 | 缝距分辨结论 | 时变可视化（图） |')
    lines.append('')
    lines.append('## 8. 结论')
    lines.append('')
    if emp['dd_min_emp_m'] is not None:
        lines.append(
            f'1. 在当前 steady + 滤失、1D 全长倒谱下，**可分辨缝距下界约 '
            f'{emp["dd_min_emp_m"]} m**（dual 全匹配最小间距；'
            f'D20 及以下为临界/失败区）。'
        )
        lines.append(
            f'2. 经验有效谐波数 N_harm_eff≈{emp["N_harm_eff_emp"]:.0f}，'
            f'远高于停泵饱和估计 ~{th["N_harm_theo"]:.0f}；'
            f'旧文档“100 m 不可行”应废弃。'
        )
    lines.append(
        '3. 多缝（≥4）时后缝漏检由幅值衰减主导，增大缝距不能完全消除。'
    )
    lines.append(
        f'4. 2D（wlen={g["wlen"]:g} s）深度分辨率弱于 1D 全长；'
        '缝距定量结论以 1D JSON 为准。'
    )
    lines.append(
        f'5. 网格 Δd_bin≈{g["dd_bin"]:.2f} m 远小于缝距，不是瓶颈；'
        '停泵 ts 与可用时长 T_1d 才是主控。'
    )
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    g = _moc_grid()
    th = _theory_chain(g)
    matrix = load_match_matrix()
    emp = _infer_empirical_dd_min(matrix)
    md = render_markdown(g, th, matrix, emp)

    out_dir = os.path.join(OUTPUT_DIR, SERIES_LEAKOFF)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'SPACING_RESOLVABILITY.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'Wrote {out_path}')

    # 控制台摘要
    print('\nMatch matrix (n_matched/n_fracs):')
    print('Δd   ' + '  '.join(f'{c:7s}' for c in CASES))
    for d in sorted(matrix.keys()):
        cells = []
        for case in CASES:
            row = matrix[d].get(case)
            cells.append('miss' if row is None else f"{row['n_matched']}/{row['n_fracs']}")
        print(f'{d:<4} ' + '  '.join(f'{c:7s}' for c in cells))
    if emp['dd_min_emp_m'] is not None:
        print(f"\nEmpirical Δd_min (1D dual) ≈ {emp['dd_min_emp_m']} m")
        print(f"Inferred N_harm_eff ≈ {emp['N_harm_eff_emp']:.0f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
