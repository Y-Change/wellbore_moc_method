# legacy — 旧版兼容入口与归档

本目录存放分级结构迁移后**不再推荐**但仍可运行的旧入口，以及说明文档。

## wrappers/

原根目录 `validate_moc_*.py`、`plot_window_comparison.py` 等 **thin wrapper**（仅转发到 `validation/` / `analysis/`）。

```bash
# 仍可从项目根目录运行，例如：
python legacy/wrappers/validate_moc_test_b.py
python legacy/wrappers/validate_moc_test_b_multi_Kaiser-Bessel.py --case all
```

**推荐**改用新路径：

| legacy wrapper | 新入口 |
|----------------|--------|
| `wrappers/validate_moc_joukowsky.py` | `validation/step01_joukowsky.py` |
| `wrappers/validate_moc_test_b.py` | `validation/test_b/single.py` |
| `wrappers/validate_moc_test_b_Kaiser-Bessel.py` | `validation/cepstrum/kaiser_bessel_single.py` |
| `wrappers/plot_window_comparison.py` | `analysis/window_comparison.py` |

## 旧版 output

迁移前扁平命名产物（如 `test_b_fracture_leakoff.png`）已移至 `output/_legacy/`。

新运行结果写入 `output/{系列}/{用例}/`，见 [`output/README.md`](../output/README.md)。
