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
├── cepstrum_mocdata.py          # 倒谱分析库（1D/2D + 时间平均剖面）
├── paths.py                     # 分级 output 路径工厂
├── validation/                  # 验证脚本（推荐入口）
│   ├── config.py                # 集中配置：井参数/仿真参数/缝形态/摩阻+间距
│   ├── leakoff_multi.py         # 统一 leakoff 验证（steady|brunone|*_D* × 单~五缝）
│   ├── step01_joukowsky.py      # Step 1 Joukowsky 解析解验证
│   ├── step03_fracture.py       # Step 3 单缝反射验证
│   ├── step03b_brunone.py       # Step 3b Brunone 摩阻验证
│   └── cepstrum/                # 倒谱方法对比与窗长扫描
│       ├── _kb_core.py          # Kaiser-Bessel / AR 倒谱核心算法
│       ├── kaiser_bessel_multi.py  # 多缝 KB 方案对比（--friction steady|brunone）
│       ├── wlen_sweep.py        # 窗长扫描分析（--friction steady|brunone）
│       └── spacing_resolvability.py  # steady_D* 缝距分辨能力汇总
├── analysis/
│   └── window_comparison.py
├── legacy/                      # 旧版兼容 wrapper（见 legacy/README.md）
│   └── wrappers/
├── docs/
│   ├── PARAMETER_CHAIN.md       # 停泵→网格→倒谱→缝距区分参数链
│   └── 计划及完成进度.md
└── output/                      # 分级输出（见 output/README.md）
```

日常请使用 `validation/`、`analysis/` 入口。旧命令见 `legacy/wrappers/`。

## 运行

在本目录下执行，例如：

```bash
# 统一 leakoff 验证（推荐）
python validation/leakoff_multi.py --friction steady --case all
python validation/leakoff_multi.py --friction brunone --case dual
python validation/leakoff_multi.py --case single              # 默认 steady

# 等间距扫描（首缝 4100 m，D=10/20/50/100）
python validation/leakoff_multi.py --friction steady_D10 --case all
python validation/leakoff_multi.py --friction steady_D100 --case dual
python validation/leakoff_multi.py --friction brunone_D50 --case all

# 一次跑完 SPACING_PRESETS_M 全部间距
python validation/leakoff_multi.py --friction steady_Dall --case all
python validation/leakoff_multi.py --friction brunone_Dall --case dual

# 倒谱方法对比
python validation/cepstrum/kaiser_bessel_multi.py --friction steady --case all
python validation/cepstrum/kaiser_bessel_multi.py --friction brunone --case quad

# 窗长扫描
python validation/cepstrum/wlen_sweep.py --friction steady --case all
python validation/cepstrum/wlen_sweep.py --friction brunone --case dual --no-grid

# 缝距分辨能力汇总（只读 steady_D* 的 moc_leakoff.json，不重跑仿真）
python validation/cepstrum/spacing_resolvability.py

# 1D 实倒谱过程可视化（默认 steady_D50/quad）
python analysis/cepstrum_1d_pipeline.py

# B_coh / N_harm,eff / Δd_min 正推（默认 steady_D50/single）
python analysis/forward_resolvability.py
python analysis/forward_resolvability.py --dr-db 80

# Step 验证
python validation/step01_joukowsky.py
python validation/step03b_brunone.py

# 旧版兼容（legacy/wrappers/）
python legacy/wrappers/validate_moc_test_b.py
python legacy/wrappers/validate_moc_test_b_multi_Kaiser-Bessel.py --case all
```



## 输出路径

结果写入 `output/{系列}/{friction}/{case}/`，例如：


| 脚本                       | 输出目录                                                           |
| ------------------------ | -------------------------------------------------------------- |
| `leakoff_multi.py`            | `output/leakoff/{steady|brunone|steady_D*|brunone_D*}/{case}/` |
| `kaiser_bessel_multi.py`      | `output/cepstrum/kaiser_bessel/{steady|brunone}/{case}/`       |
| `wlen_sweep.py`               | `output/cepstrum/wlen_sweep/{steady|brunone}/{case}/`          |
| `spacing_resolvability.py`    | `output/leakoff/SPACING_RESOLVABILITY.md`                      |
| `step01_joukowsky.py`         | `output/step01_joukowsky/`                                     |
| `step03b_brunone.py`          | `output/step03b_brunone/`                                      |


`leakoff_multi` 每个 case 目录下生成：


| 文件                      | 内容                                            |
| ----------------------- | --------------------------------------------- |
| `moc_leakoff.png`       | 2×2 MOC 验证图（时域 / 差信号 / 缝节点 H+Q）               |
| `cepstrum_standard.png` | 倒谱五联图（时域 / FFT / 1D 实倒谱 / 2D 倒谱 / 时间平均 1D 剖面） |
| `moc_timeseries.csv`    | 井口与各缝口水头、流量时程（`t,H_wh,Q_wh,H_f1,Q_f1,...`）    |
| `moc_leakoff.json`      | PASS/FAIL 判定与指标（含 1D 倒谱缝深匹配）                  |




## 集中配置

所有物理参数、仿真参数、缝形态、摩阻与间距键集中在 `[validation/config.py](validation/config.py)`：

```python
WELL_CONFIG         # L, diameter, density, viscosity, wavespeed, roughness, V0, H0, theta
SIM_CONFIG          # ts, dt, tf
FRACTURE_CONFIG     # Cf, kleak, H_ext
CEPSTRUM_CONFIG     # wlen_sec, hop_sec, win_type（rect/hamming/hanning/kaiser/gauss）
FRAC_FIRST_M        # 首缝深度（默认 4100 m）
SPACING_PRESETS_M   # 等间距预设 (10, 20, 50, 100)
build_cases(D)      # 按间距 D 生成 single~quint 的 x_f_list
CASES               # 默认 = build_cases(50)，供 steady / brunone 使用
FRICTION_PARAMS     # steady / brunone / steady_D* / brunone_D*
                    #   friction_model, judgment4, stab_factor
                    #   spacing_m（D* 键专用，驱动 build_cases）
```

修改该文件即可全局调整物理与仿真设定。`--friction steady_D50` 等键会写入对应输出子目录，并按 `spacing_m` 生成缝深。

## 缝形态配置

默认（`CASES = build_cases(50)`，首缝 4100 m，间距 50 m）：


| Case   | 缝数  | 缝深 [m]                       |
| ------ | --- | ---------------------------- |
| single | 1   | 4100                         |
| dual   | 2   | 4100, 4150                   |
| triple | 3   | 4100, 4150, 4200             |
| quad   | 4   | 4100, 4150, 4200, 4250       |
| quint  | 5   | 4100, 4150, 4200, 4250, 4300 |


等间距变体（`--friction {steady\|brunone}_D{N}`，N∈{10,20,50,100}）：


| 键                              | 输出目录                       | dual 示例        |
| ------------------------------ | -------------------------- | -------------- |
| `steady_D10` / `brunone_D10`   | `output/leakoff/..._D10/`  | `[4100, 4110]` |
| `steady_D20` / `brunone_D20`   | `output/leakoff/..._D20/`  | `[4100, 4120]` |
| `steady_D50` / `brunone_D50`   | `output/leakoff/..._D50/`  | `[4100, 4150]` |
| `steady_D100` / `brunone_D100` | `output/leakoff/..._D100/` | `[4100, 4200]` |


批量一次跑完全部间距：`--friction steady_Dall` 或 `brunone_Dall`（展开为上表四个 D 键）。

## 验证项

- Joukowsky 解析解（Step 1）
- 单缝 / 双缝反射（Step 3–4a）
- 稳态 vs Brunone 摩阻
- 滤失 + 摩阻（leakoff_multi：steady/brunone × single→quint）
- 缝间距扫描（leakoff_multi：`*_D10/20/50/100`）
- 缝距分辨能力汇总（spacing_resolvability → `SPACING_RESOLVABILITY.md`）
- Kaiser-Bessel 倒谱方法对比（kaiser_bessel_multi）
- 窗长对倒谱裂缝识别影响（wlen_sweep）

参数链推导（停泵→网格→频/倒频→Δd_min）见 [`docs/PARAMETER_CHAIN.md`](docs/PARAMETER_CHAIN.md)。  
详细进度见 [`docs/计划及完成进度.md`](docs/计划及完成进度.md)。

## 脚本与旧版 wrapper 对照


| 旧 wrapper（`legacy/wrappers/`）                | 新入口                                                        | 输出目录                                               |
| -------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------- |
| `validate_moc_joukowsky.py`                  | `validation/step01_joukowsky.py`                           | `output/step01_joukowsky/`                         |
| `validate_moc_test_b.py`                     | `validation/leakoff_multi.py --case single`                | `output/leakoff/steady/single/`                    |
| `validate_moc_test_b_dual.py`                | `validation/leakoff_multi.py --case dual`                  | `output/leakoff/steady/dual/`                      |
| `validate_moc_test_b_Kaiser-Bessel.py`       | `validation/cepstrum/kaiser_bessel_multi.py --case single` | `output/cepstrum/kaiser_bessel/steady/single/`     |
| `validate_moc_test_b_multi_Kaiser-Bessel.py` | `validation/cepstrum/kaiser_bessel_multi.py`               | `output/cepstrum/kaiser_bessel/{friction}/{case}/` |
| `plot_window_comparison.py`                  | `analysis/window_comparison.py`                            | `output/analysis/window_comparison/`               |


