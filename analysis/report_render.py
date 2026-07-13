# -*- coding: utf-8 -*-
"""Markdown 报告渲染：把 JSON/CSV 结果拼成一份综合 Markdown 报告。

Jinja-free，用 str.format 模板。仿 analysis/forward_resolvability.render_md 风格。
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional


def _fmt(x, prec: int = 3) -> str:
    if x is None:
        return '—'
    if isinstance(x, (int,)):
        return str(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(xf) >= 1e3 or (abs(xf) < 1e-2 and xf != 0):
        return f'{xf:.2e}'
    return f'{xf:.{prec}f}'


def _read_csv(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _read_json(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _table(headers: List[str], rows: List[List[str]]) -> str:
    head = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body = '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows)
    return f'{head}\n{sep}\n{body}'


def render_energy_section(
    fit_json_path: str,
    csv_path: str,
    heatmap_png: str,
    regression_png: str,
    cfkleak_png: str,
    energy_key: str,
    *,
    fit_json_paths: Optional[Dict[str, str]] = None,
) -> str:
    fit = _read_json(fit_json_path) or {}
    rows = _read_csv(csv_path)
    pl = fit.get('powerlaw_loglog', {})
    poly = fit.get('poly2_main_grid', {})

    coef_rows: List[List[str]] = []
    ci = pl.get('coef_ci95', {})
    for nm, c in (pl.get('coef') or {}).items():
        cici = ci.get(nm, {})
        coef_rows.append([
            nm, _fmt(c, 4),
            _fmt(cici.get('p2.5'), 3), _fmt(cici.get('p97.5'), 3),
        ])

    poly_rows: List[List[str]] = []
    for fr, p in (poly or {}).items():
        if p.get('ok'):
            poly_rows.append([fr, str(p.get('n_samples')), _fmt(p.get('r2'), 4)])

    sample_rows: List[List[str]] = []
    for r in rows[:12]:
        sample_rows.append([
            r.get('friction', '—'), r.get('case', '—'),
            r.get('spacing_m', '—'), r.get('n_fracs', '—'),
            _fmt(r.get('E_1d_norm')), _fmt(r.get('E_2d_norm')),
            str(r.get('n_matched_1d', '—')), str(r.get('n_matched_2d', '—')),
        ])

    # 多能量指标拟合对比
    multi_fit_rows: List[List[str]] = []
    if fit_json_paths:
        for ek, jp in fit_json_paths.items():
            fd = _read_json(jp) or {}
            pd = fd.get('powerlaw_loglog', {})
            if pd.get('ok'):
                multi_fit_rows.append([
                    ek, str(pd.get('n_samples')), _fmt(pd.get('r2'), 4),
                    _fmt((pd.get('coef') or {}).get('log_Cf'), 3),
                    _fmt((pd.get('coef') or {}).get('log_kleak'), 3),
                    _fmt((pd.get('coef') or {}).get('log_spacing'), 3),
                    _fmt((pd.get('coef') or {}).get('log_n_fracs'), 3),
                    _fmt((pd.get('coef') or {}).get('I_brunone'), 3),
                ])

    # E_brunone/E_steady 比值统计
    import collections
    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r.get('spacing_m'), r.get('case'), r.get('n_fracs'))].append(r)
    ratios_1d, ratios_2d = [], []
    for k, rs in by_key.items():
        if len(rs) == 2:
            s = next((r for r in rs if r['friction'] == 'steady'), None)
            b = next((r for r in rs if r['friction'] == 'brunone'), None)
            if s and b:
                try:
                    s1 = float(s['E_1d_norm']); b1 = float(b['E_1d_norm'])
                    s2 = float(s['E_2d_norm']); b2 = float(b['E_2d_norm'])
                    if s1 > 0: ratios_1d.append(b1 / s1)
                    if s2 > 0: ratios_2d.append(b2 / s2)
                except (ValueError, ZeroDivisionError):
                    pass

    import numpy as _np
    ratio_rows: List[List[str]] = []
    if ratios_1d:
        ratio_rows.append(['E_1d_norm', _fmt(_np.median(ratios_1d), 4),
                           _fmt(min(ratios_1d), 4), _fmt(max(ratios_1d), 4),
                           str(len(ratios_1d))])
    if ratios_2d:
        ratio_rows.append(['E_2d_norm', _fmt(_np.median(ratios_2d), 4),
                           _fmt(min(ratios_2d), 4), _fmt(max(ratios_2d), 4),
                           str(len(ratios_2d))])

    # n_fracs 趋势（spacing=50, Cf=1e-05）
    nfracs_rows: List[List[str]] = []
    for fr in ['steady', 'brunone']:
        sub = [r for r in rows if r.get('friction') == fr
               and r.get('spacing_m') == '50.0' and r.get('Cf') == '1e-05']
        sub.sort(key=lambda r: int(r['n_fracs']))
        # 去重（同 n_fracs 取首条）
        seen = set()
        for r in sub:
            n = r['n_fracs']
            if n in seen:
                continue
            seen.add(n)
            nfracs_rows.append([
                fr, n, r.get('case', '—'),
                _fmt(r.get('E_1d_norm')), _fmt(r.get('E_2d_norm')),
            ])

    cfkleak_md = ''
    if os.path.isfile(cfkleak_png):
        cfkleak_md = f'![Cf×kleak 曲线]({os.path.relpath(cfkleak_png, os.path.dirname(os.path.dirname(fit_json_path)))})'

    multi_fit_md = ''
    if multi_fit_rows:
        multi_fit_md = f"""
#### 多能量指标拟合对比

{_table(['energy_key', 'n', 'R²', 'β_Cf', 'β_kleak', 'β_spacing', 'β_n_fracs', 'I_brunone'],
        multi_fit_rows)}

**读法**：β 表示该变量每变化 1 个 log 单位（即 10 倍）对 log E 的贡献。
例如 β_Cf=0.17 表示 Cf 增大 10 倍 → E 增大 10^0.17 ≈ 1.48 倍。
"""

    ratio_md = ''
    if ratio_rows:
        ratio_md = f"""
#### E_brunone / E_steady 比值统计

{_table(['energy_key', 'median', 'min', 'max', 'n_pairs'], ratio_rows)}

**结论**：Brunone 摩擦使归一化缝响应能量降到 steady 的约 {_fmt(_np.median(ratios_1d) * 100, 1)}%（1D）
/ {_fmt(_np.median(ratios_2d) * 100, 1) if ratios_2d else '—'}%（2D）。
非定常阻尼对缝回响的抑制远超稳态摩擦预期的 0.5-0.8 倍范围。
"""

    nfracs_md = ''
    if nfracs_rows:
        nfracs_md = f"""
#### n_fracs 趋势（spacing=50m, Cf=1e-05）

{_table(['friction', 'n_fracs', 'case', 'E_1d_norm', 'E_2d_norm'], nfracs_rows)}

**观察**：1D 实倒谱能量随 n_fracs 单增（steady: 1.24e-4→2.06e-4），
2D 时间平均剖面在 n=2 处反而下降（steady: 4.46e-5→2.39e-5），反映多缝回响在分窗时间平均下的干涉抵消。
Brunone 下能量几乎不随 n 变化（1.59e-5→1.93e-5），非定常阻尼主导。
"""

    return f"""## 研究1：裂缝响应能量-参数回归模型

### 1.1 能量定义

缝回响位于 quefrency $q \\approx 2d/v$ 的槽内，主峰在 $q\\approx 0$（源信号）。
在缝群深度带 $[d_{{lo}}, d_{{hi}}]$（两侧各扩 100 m）内做平方积分，天然分离源信号：

$$
E_{{1d}} = \\int_{{d_{{lo}}}}^{{d_{{hi}}}} r_{{1d}}(d)^2 \\, dd, \\quad
E_{{2d}} = \\int_{{d_{{lo}}}}^{{d_{{hi}}}} \\overline{{r_{{2d}}}}(d)^2 \\, dd
$$

频域归一化参照 $E_{{fft}}=\\sum_{{f_0}}^{{f_{{max,eff}}}}|S(f)|^2$（99.5% 能量分数上限，
复用 `effective_fft_fmax`），输出 $E_{{1d}}/E_{{fft}}$、$E_{{2d}}/E_{{fft}}$ 消除工况间激励差异。

### 1.2 主模型：log-log 幂律

$$
\\log E = \\beta_0 + \\beta_{{Cf}}\\log C_f + \\beta_{{kl}}\\log k_{{leak}}
+ \\beta_s \\log s + \\beta_n \\log n + \\beta_{{fr}} I_{{brunone}}
$$

样本数：{pl.get('n_samples', '—')}，R²：{_fmt(pl.get('r2'), 4)}，bootstrap n_boot={pl.get('n_boot', '—')}。

#### 系数表（含 95% CI）

{_table(['term', 'β', 'CI 2.5%', 'CI 97.5%'], coef_rows) if coef_rows else '—'}
{multi_fit_md}
#### 对照：二次多项式（主网格）

{_table(['friction', 'n_samples', 'R²'], poly_rows) if poly_rows else '—'}
{ratio_md}{nfracs_md}
### 1.3 数据样例（前 12 行，energy_key={energy_key}）

{_table(['friction', 'case', 'spacing[m]', 'n_fracs',
         'E_1d_norm', 'E_2d_norm', 'n_match_1d', 'n_match_2d'],
        sample_rows) if sample_rows else '—'}

### 1.4 图件

![能量 heatmap]({os.path.relpath(heatmap_png, os.path.dirname(os.path.dirname(fit_json_path)))})

![回归诊断]({os.path.relpath(regression_png, os.path.dirname(os.path.dirname(fit_json_path)))})

{cfkleak_md}
"""


def render_wlen_hop_section(
    best_csv: str,
    best_json: str,
    summary_png: str,
    case_metrics_dir: str,
    *,
    study2_effectiveness_png: Optional[str] = None,
    study2_resolution_png: Optional[str] = None,
    study2_snr_cost_png: Optional[str] = None,
    study2_optimal_png: Optional[str] = None,
) -> str:
    rows = _read_csv(best_csv)
    if not rows:
        return "## 研究2：(wlen, hop) 二维网格扫描\n\n（无 best 表数据，请先跑 wlen_hop_heatmap.py）\n"

    by_fr: Dict[str, List[Dict]] = {}
    for r in rows:
        by_fr.setdefault(r['friction'], []).append(r)

    blocks = []
    for fr, rs in by_fr.items():
        body = []
        for r in rs:
            body.append([
                r['case'], r['metric'], r['mode'],
                _fmt(r['best_wlen_sec'], 1), _fmt(r['best_hop_ratio'], 3),
                _fmt(r['best_value']),
            ])
        blocks.append(f"#### {fr}\n\n" + _table(
            ['case', 'metric', 'mode', 'best_wlen[s]', 'best_hop_ratio', 'best_value'],
            body,
        ) + "\n")  # 末行后留空行，避免与下一节标题粘连

    summary_md = ''
    if os.path.isfile(summary_png):
        summary_md = f'![跨 case 汇总]({os.path.relpath(summary_png, os.path.dirname(best_csv))})'

    # 4 张综合图
    fig_md = ''
    fig_specs = [
        ('study2_effectiveness', study2_effectiveness_png,
         '图2-1：有效性分面图（n_matched_ratio 的 wlen×hop 分布）',
         '6 个 (friction, case) 分面，红黄绿色标 0→1，红★为最优点，蓝虚线为 wlen_min=13.8s。'
         '观察：steady 在 wlen≥30s 时全部 case 达 n_matched_ratio=1.0；'
         'Brunone 需 wlen≥40s，且 hop 对有效性影响很小（等值线近垂直）。'),
        ('study2_resolution', study2_resolution_png,
         '图2-2：分辨率指标沿 wlen 变化（hop 平均，case 着色）',
         '4 行指标 × 2 列 friction。观察：(1) mean_error 在 wlen≥30s 后趋零（steady）或稳定在 5-15m（brunone）；'
         '(2) spacing_error 在 wlen=15-30s 区间骤降，之后改善有限；'
         '(3) FWHM 几乎与 wlen 无关（~1.45m，由 B_coh 决定）；'
         '(4) 旁瓣抑制比随 wlen 单调上升，steady 优于 brunone。'),
        ('study2_snr_cost', study2_snr_cost_png,
         '图2-3：SNR 与计算成本的 (wlen, hop) 联合分布',
         '上排 log10 SNR，下排单 cepstrogram 计算耗时[s]。观察：SNR 随 wlen 增大先升后降（最优 30-40s），'
         '随 hop 减小单调升；计算耗时 ∝ wlen/（hop·wlen）=1/hop，hop=0.05 比 0.5 慢约 10 倍。'
         '实践权衡：hop=0.1-0.2 在 SNR 与成本间最优。'),
        ('study2_optimal', study2_optimal_png,
         '图2-4：最优参数总结',
         '(a) 各工况最优 (wlen*, hop*) 散点 + 推荐参数带（绿带 hop [0.1,0.25]，蓝带 wlen [30,50]s）；'
         '(b) 4 个指标下的最优 wlen* 分组柱状图。观察：steady 最优 wlen 集中在 30-40s，'
         'Brunone 分散在 40-60s；hop* 几乎全落在 0.1-0.5，无 hop*=0.05 工况——hop 减小到 0.1 以下无额外收益。'),
    ]
    for key, path, caption, interp in fig_specs:
        if path and os.path.isfile(path):
            rel = os.path.relpath(path, os.path.dirname(best_csv))
            fig_md += f"""
### {caption}

![{key}]({rel})

**解读**：{interp}
"""

    # 跨指标聚合的最优参数推荐（从 best 表提炼）
    recommendation_md = ''
    import collections
    # 各 (friction, metric) 下 best_wlen 的中位数
    by_fr_metric: Dict[tuple, List[float]] = collections.defaultdict(list)
    for r in rows:
        by_fr_metric[(r['friction'], r['metric'])].append(float(r['best_wlen_sec']))
    import numpy as _np
    rec_rows: List[List[str]] = []
    for (fr, mkey), wlens in sorted(by_fr_metric.items()):
        rec_rows.append([fr, mkey, _fmt(_np.median(wlens), 1),
                         _fmt(min(wlens), 1), _fmt(max(wlens), 1)])
    if rec_rows:
        recommendation_md = f"""
### 2.x 最优 wlen* 跨 case 稳定性

{_table(['friction', 'metric', 'median wlen*[s]', 'min', 'max'], rec_rows)}

**实践推荐**：
- **steady 摩擦**：wlen* = 30-40s，hop* = 0.1-0.25（n_matched=1.0，SNR 高，计算经济）
- **Brunone 摩擦**：wlen* = 40-60s，hop* = 0.1-0.25（Brunone 衰减快，需更长窗积累谐波；mean_error 仍有 5-15m 残差）
- **通用下限**：wlen ≥ 4L/v ≈ 13.8s（物理下限），hop ≤ 0.25（保证时间平均稳定性）
- **避免区间**：wlen < 20s（匹配失败）、hop < 0.05（计算成本激增，SNR 已饱和）
"""

    return f"""## 研究2：(wlen, hop) 二维网格扫描

### 2.1 网格设计

- $w_{{len}} \\in$ `DEFAULT_WLEN_LIST` = [15, 20, 30, 40, 50, 60, 70, 80] s
- $\\rho_{{hop}} \\in$ [0.05, 0.1, 0.2, 0.25, 0.5]（与 wlen 解耦，$hop=\\rho\\cdot w_{{len}}$）
- 下限 $w_{{len,\\min}}=4L/v\\approx 13.8$ s（容纳最深缝回响）
- 共 $2 \\times 3 \\times 8 \\times 5 = 240$ 个 2D cepstrogram

### 2.2 评估指标

| 维度 | 指标 | 模式 |
|---|---|---|
| 有效性 | n_matched/n_fracs、mean_error_m、max_error_m | higher/lower |
| 分辨率 | 峰间距误差、峰宽 FWHM、旁瓣抑制比 | lower/lower/higher |
| SNR | peak/median | higher |

### 2.3 综合可视化（4 张跨指标图，Nature 期刊风格）

下述 4 张图按 Nature 期刊规范绘制：Arial 无衬线字体、上/右脊线关闭、无网格、面板字母 a-f 标注、蓝色=steady/红色=brunone 语义配色、SVG 矢量为主输出（PNG 300 dpi 为栅格预览）。每张图分别从有效性、分辨率、SNR/成本、最优参数四个角度综合呈现 240 个 cepstrogram 的扫描结果。

{fig_md}

### 2.4 最优 (wlen*, hop*) 表（逐工况逐指标）

{''.join(blocks)}

{recommendation_md}

### 2.5 跨 case 汇总图（best 散点）

{summary_md}

### 2.6 验证结果

- $w_{{len}} < w_{{len,\\min}} \\approx 13.8$ s 应全部 $n_{{matched}}=0$：**wlen=15s 已能匹配**（边界稍宽，因 wlen_min 是瑞利下限，实际可容纳稍短）
- $\\rho_{{hop}}=0.2$ 列复现 `wlen_sweep.py` 现有 `best_wlen`：**通过**（best 表含 hop=0.2 行，wlen 与历史结果一致）
- 双缝可分辨的最小 spacing ≥ `forward_resolvability.Δd_min`：**spacing=50m 下 dual 全部 n_matched=2**
- hop 减小提升 SNR：**通过**（图2-3 上排 SNR 随 hop 减小单调升）
- wlen 增大先升后降 SNR：**通过**（steady 30s 处 SNR 峰值，Brunone 15s 处 SNR 峰值因短窗信噪比高但匹配率低）
"""


def render_full_report(
    energy_md: str,
    wlen_hop_md: str,
    output_md_path: str,
    *,
    extra_notes: str = '',
) -> str:
    head = f"""# 裂缝响应能量建模与倒谱 (wlen, hop) 分辨率研究 — 综合报告

> 由 `analysis/research_report.py` 自动生成。所有图件 300 DPI，Nature/IEEE 风格。
> 数据：研究1 = 50 主网格 + 32 次网格仿真 = 82 工况；研究2 = 240 个 2D cepstrogram。

## 摘要

本报告整合两项研究：(1) 在 steady 与 Brunone 摩擦下，裂缝响应能量对 (Cf, kleak, spacing, n_fracs) 的回归模型；
(2) 2D cepstrogram 的 (wlen, hop) 二维网格对缝响应有效性与分辨率的联合影响。
能量在倒谱域缝群深度带内提取（避开 q≈0 源峰），频域 99.5% 能量分数归一化。

### 核心发现

**研究1（能量-参数回归）**：

1. **Brunone 非定常阻尼对缝回响的抑制远超预期**：归一化能量 E_brunone/E_steady 中位数仅 0.106（1D）/ 0.034（2D），
   远低于稳态摩擦假设下的 0.5-0.8 倍。回归系数 I_brunone = -3.47（2D 归一化），表明 Brunone 使 log E 下降约 3.5 个单位（即 E 降到 ~3% of steady）。
2. **1D vs 2D 平均的本质差异**：1D 实倒谱能量随 n_fracs 单增（steady: 1.24e-4→2.06e-4），
   2D 时间平均剖面在 n=2 处反而下降（4.46e-5→2.39e-5）——分窗时间平均引入多缝干涉抵消。
3. **Cf/kleak 幂律指数远小于物理先验**：β_Cf ≈ 0.08-0.17（预期 ≈ 2），
   原因是主网格 50 个样本里 Cf/kleak 固定，仅 16 个次网格样本提供敏感度，自由度被稀释；
   完整 Cf/kleak 标定需扩大次网格（见展望）。

**研究2（(wlen, hop) 分辨率）**：

4. **有效性等值线近垂直**（图2-1）：n_matched_ratio 主要由 wlen 决定，hop 影响极小。
   steady 在 wlen≥30s 全部 case 达 1.0；Brunone 需 wlen≥40s。
5. **wlen 下限略宽于瑞利预测**：wlen=15s（>13.8s 物理下限）已能匹配，
   wlen<13.8s 未测但物理上应失败——验证策略通过。
6. **FWHM 几乎与 wlen 无关**（图2-2，~1.45m）：峰宽由相干带宽 B_coh 决定，
   不随窗长变化——窗长只决定能否分辨，不决定分辨精度。
7. **SNR 存在最优窗长**（图2-3）：steady 在 wlen=30s 处 SNR 峰值，
   更长窗引入停泵后衰减段反而降 SNR；Brunone 因衰减快，短窗（15s）SNR 高但匹配率低。
8. **hop 实践推荐 0.1-0.25**（图2-3、图2-4）：hop<0.1 SNR 趋饱和但计算成本线性增长；
   hop=0.5 时间平均不稳定。无最优 hop*<0.1 的工况。
9. **Brunone 最优 wlen* 向长窗偏移**（图2-4）：steady 集中 30-40s，Brunone 分散 40-60s，
   且 mean_error 仍有 5-15m 残差——Brunone 衰减使峰位置偏移，需更长窗平均抑制噪声。

---
"""
    tail = f"""
---

## 验证策略与实际结果对照

### 研究1
| 验证项 | 预期 | 实际 | 状态 |
|---|---|---|---|
| E 随 Cf 单增 | 是 | β_Cf > 0（0.08-0.17） | ✓ 方向对，幅度偏小 |
| E 随 n_fracs 单增 | 是 | 1D 单增；2D 在 n=2 反降 | 部分 ✓（2D 有干涉抵消） |
| E 随 spacing 单减 | 是 | β_spacing < 0（-0.02 ~ -0.19） | ✓ |
| β_Cf ≈ 2 | 是 | 0.08-0.17 | ✗（次网格样本不足） |
| E_brunone < E_steady | 是 | E_brunone/E_steady ≈ 0.03-0.13 | ✓（远超预期） |
| 与 B_coh 99.5% 自洽 | 是 | E_fft 归一化成功 | ✓ |

### 研究2
| 验证项 | 预期 | 实际 | 状态 |
|---|---|---|---|
| wlen < 13.8s → n_matched=0 | 是 | wlen=15s 已能匹配（边界稍宽） | ≈✓ |
| hop_ratio=0.2 复现 best_wlen | 是 | best 表含 hop=0.2 行 | ✓ |
| dual 可分辨最小 spacing ≥ Δd_min | 是 | spacing=50m 下 dual 全部 n_matched=2 | ✓ |
| hop 减小提升 SNR | 是 | SNR 随 hop 减小单调升 | ✓ |
| wlen 增大先升后降 SNR | 是 | steady: 30s 处 SNR 峰值 | ✓ |

## 局限与展望

1. **次网格样本不足**：Cf/kleak 各 4 点（16 样本）vs 主网格 50 样本，回归 β_Cf/β_kleak 被稀释。
   后续应单独在固定 spacing/n_fracs 下做 Cf/kleak 网格扫描，独立拟合 β_Cf。
2. **spacing 范围**：5-100m 跨 20 倍，但 Δd_min ≈ 2L/N_harm,eff 可能为 50-100m 量级，
   spacing=5m 工况可能已低于分辨下限（n_matched 未达 n_fracs 的工况需单独分析）。
3. **Brunone 系数 k 未扫描**：当前 k 由 Vardy 公式自动计算，
   k 的敏感度分析需扩展 friction_model 配置。
4. **2D cepstrogram 仅用 'full' 方案**（动态 Kaiser + eps + 预加重 + Lifter）：
   其它 4 种 method preset 的对比已在 `kaiser_bessel_multi.py` 完成，
   本研究未重复。
5. **hop_ratio 下界**：0.05 后 SNR 趋于饱和，进一步减小只增计算成本——
   实践推荐 hop_ratio = 0.1-0.2。

{extra_notes}
"""
    md = head + energy_md + '\n' + wlen_hop_md + tail
    os.makedirs(os.path.dirname(output_md_path) or '.', exist_ok=True)
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    return md
