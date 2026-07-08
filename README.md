# Wellbore MOC Method

井筒-裂缝系统 **自研轻量 MOC 水击仿真器**（滑溜水 / 裂缝柔度+滤失 / Brunone 摩阻）及验证脚本。

## 依赖

```bash
pip install -r requirements.txt
```

## 目录结构

```
wellbore_moc_method/
├── wellbore_moc.py              # MOC 核心求解器
├── cepstrum_mocdata.py          # 倒谱分析库
├── paths.py                     # 分级 output 路径工厂
├── validation/                  # 验证脚本（推荐入口）
│   ├── config.py                # 集中配置：井参数/仿真参数/缝形态/摩阻模型
│   ├── leakoff_multi.py         # 统一 leakoff 验证（steady/brunone × 单~五缝）
│   ├── step01_joukowsky.py      # Step 1 Joukowsky 解析解验证
│   ├── step03_fracture.py       # Step 3 单缝反射验证
│   ├── step03b_brunone.py       # Step 3b Brunone 摩阻验证
│   └── cepstrum/                # 倒谱方法对比与窗长扫描
│       ├── _kb_core.py          # Kaiser-Bessel / AR 倒谱核心算法
│       ├── kaiser_bessel_multi.py  # 多缝 KB/AR 方案对比（--friction steady|brunone）
│       └── wlen_sweep.py        # 窗长扫描分析（--friction steady|brunone）
├── analysis/
│   └── window_comparison.py
├── legacy/                      # 旧版兼容 wrapper（见 legacy/README.md）
│   └── wrappers/
├── docs/
│   └── 计划及完成进度.md
└── output/                      # 三级分级输出（见 output/README.md）
```

日常请使用 `validation/`、`analysis/` 入口。旧命令见 `legacy/wrappers/`。

## 运行

在本目录下执行，例如：

```bash
# 统一 leakoff 验证（推荐）
python validation/leakoff_multi.py --friction steady --case all
python validation/leakoff_multi.py --friction brunone --case dual
python validation/leakoff_multi.py --case single              # 默认 steady

# 倒谱方法对比
python validation/cepstrum/kaiser_bessel_multi.py --friction steady --case all
python validation/cepstrum/kaiser_bessel_multi.py --friction brunone --case quad

# 窗长扫描
python validation/cepstrum/wlen_sweep.py --friction steady --case all
python validation/cepstrum/wlen_sweep.py --friction brunone --case dual --no-grid

# Step 验证
python validation/step01_joukowsky.py
python validation/step03b_brunone.py

# 旧版兼容（legacy/wrappers/）
python legacy/wrappers/validate_moc_test_b.py
python legacy/wrappers/validate_moc_test_b_multi_Kaiser-Bessel.py --case all
```

## 输出路径

结果写入 `output/{系列}/{friction}/{case}/`，例如：

| 脚本 | 输出目录 |
|------|----------|
| `leakoff_multi.py` | `output/leakoff/{steady\|brunone}/{case}/` |
| `kaiser_bessel_multi.py` | `output/cepstrum/kaiser_bessel/{steady\|brunone}/{case}/` |
| `wlen_sweep.py` | `output/cepstrum/wlen_sweep/{steady\|brunone}/{case}/` |
| `step01_joukowsky.py` | `output/step01_joukowsky/` |
| `step03b_brunone.py` | `output/step03b_brunone/` |

每个 case 目录下生成：
- `moc_leakoff.png` — 2×2 MOC 验证图（时域/差信号/缝节点 H+Q）
- `cepstrum_standard.png` — 标准四联倒谱图（时域/FFT/1D倒谱/2D倒谱）
- `moc_leakoff.json` — PASS/FAIL 判定与指标

## 集中配置

所有物理参数、仿真参数、缝形态、摩阻模型集中在 [`validation/config.py`](validation/config.py)：

```python
WELL_CONFIG       # L, diameter, density, viscosity, wavespeed, roughness, V0, H0, theta
SIM_CONFIG        # ts, dt, tf
FRACTURE_CONFIG   # Cf, kleak, H_ext
CASES             # single/dual/triple/quad/quint 的 x_f_list
FRICTION_PARAMS   # steady/brunone 的 friction_model, judgment4, stab_factor
```

修改该文件即可全局调整所有验证的物理与仿真设定，无需改各脚本。

## 缝形态配置

| Case | 缝数 | 缝深 [m] |
|------|------|----------|
| single | 1 | 4300 |
| dual | 2 | 4300, 4600 |
| triple | 3 | 4100, 4300, 4500 |
| quad | 4 | 4100, 4300, 4500, 4700 |
| quint | 5 | 3700, 3900, 4100, 4300, 4500 |

## 验证项

- Joukowsky 解析解（Step 1）
- 单缝 / 双缝反射（Step 3–4a）
- 稳态 vs Brunone 摩阻
- 滤失 + 摩阻（leakoff_multi：steady/brunone × single→quint）
- Kaiser-Bessel / AR 倒谱方法对比（kaiser_bessel_multi）
- 窗长对倒谱裂缝识别影响（wlen_sweep）

详细进度见 [`docs/计划及完成进度.md`](docs/计划及完成进度.md)。

## 脚本与旧版 wrapper 对照

| 旧 wrapper（`legacy/wrappers/`） | 新入口 | 输出目录 |
|----------------------------------|--------|----------|
| `validate_moc_joukowsky.py` | `validation/step01_joukowsky.py` | `output/step01_joukowsky/` |
| `validate_moc_test_b.py` | `validation/leakoff_multi.py --case single` | `output/leakoff/steady/single/` |
| `validate_moc_test_b_dual.py` | `validation/leakoff_multi.py --case dual` | `output/leakoff/steady/dual/` |
| `validate_moc_test_b_Kaiser-Bessel.py` | `validation/cepstrum/kaiser_bessel_multi.py --case single` | `output/cepstrum/kaiser_bessel/steady/single/` |
| `validate_moc_test_b_multi_Kaiser-Bessel.py` | `validation/cepstrum/kaiser_bessel_multi.py` | `output/cepstrum/kaiser_bessel/{friction}/{case}/` |
| `plot_window_comparison.py` | `analysis/window_comparison.py` | `output/analysis/window_comparison/` |
