# steady_leakoff — 稳态摩阻 + 裂缝滤失验证

本系列验证 MOC 求解器在 **steady 达西摩阻** 与 **裂缝滤失** (`k_leak > 0`) 条件下的多缝稳定性与倒谱可分辨性。

## 物理配置

- `friction_model='steady'`（无 Brunone 非定常摩阻）
- `fracture_kleak > 0`，`H_ext` 驱动滤失
- 缝数用例：`single` / `dual` / `triple` / `quad` / `quint`

## 运行

```bash
python validation/steady_leakoff/single.py
python validation/steady_leakoff/dual.py
# …
```

兼容旧命令：`python legacy/wrappers/validate_moc_test_b.py`

## 输出

`output/steady_leakoff/{case}/`

| 文件 | 内容 |
|------|------|
| `moc_leakoff.png` | 2×2 MOC 验证图 |
| `moc_leakoff.json` | PASS/FAIL 判定 |
| `cepstrum_standard.png` | 标准四联倒谱图 |
