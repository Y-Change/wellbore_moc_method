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
│   ├── step01_joukowsky.py … step04a_*.py
│   ├── steady_leakoff/          # 稳态摩阻 + 滤失：单/双/三/四/五缝 MOC + 标准倒谱
│   └── cepstrum/                # Kaiser-Bessel 倒谱方法对比
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
# 新路径（推荐）
python validation/steady_leakoff/single.py
python validation/cepstrum/kaiser_bessel_multi.py --case triple

# 旧版兼容（legacy/wrappers/）
python legacy/wrappers/validate_moc_test_b.py
python legacy/wrappers/validate_moc_test_b_multi_Kaiser-Bessel.py --case all
```

结果写入 `output/{系列}/{用例}/`，例如：

- `output/steady_leakoff/single/moc_leakoff.png`
- `output/cepstrum/kaiser_bessel/dual/compare_2d.png`
- `output/step01_joukowsky/validation.json`

## 脚本与输出命名对照

| 旧 wrapper（`legacy/wrappers/`） | 新路径 | 输出目录 |
|----------------------------------|--------|----------|
| `validate_moc_joukowsky.py` | `validation/step01_joukowsky.py` | `output/step01_joukowsky/` |
| `validate_moc_test_b.py` | `validation/steady_leakoff/single.py` | `output/steady_leakoff/single/` |
| `validate_moc_test_b_dual.py` | `validation/steady_leakoff/dual.py` | `output/steady_leakoff/dual/` |
| `validate_moc_test_b_Kaiser-Bessel.py` | `validation/cepstrum/kaiser_bessel_single.py` | `output/cepstrum/kaiser_bessel/single/` |
| `validate_moc_test_b_multi_Kaiser-Bessel.py` | `validation/cepstrum/kaiser_bessel_multi.py` | `output/cepstrum/kaiser_bessel/{case}/` |
| `plot_window_comparison.py` | `analysis/window_comparison.py` | `output/analysis/window_comparison/` |

## 验证项

- Joukowsky 解析解（Step 1）
- 单缝 / 双缝反射（Step 3–4a）
- 稳态 vs Brunone 摩阻
- 滤失 + 稳态摩阻（steady_leakoff：single → quint）
- Kaiser-Bessel / AR 倒谱方法对比

详细进度见 [`docs/计划及完成进度.md`](docs/计划及完成进度.md)。
