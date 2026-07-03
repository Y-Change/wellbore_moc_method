# Wellbore MOC Method

井筒-裂缝系统 **自研轻量 MOC 水击仿真器**（滑溜水 / 裂缝柔度+滤失 / Brunone 摩阻）及验证脚本。

## 依赖

```bash
pip install -r requirements.txt
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `wellbore_moc.py` | MOC 核心求解器 |
| `cepstrum_mocdata.py` | 井口水头倒谱分析（1D + 2D） |
| `validate_moc_*.py` | 物理验证脚本 |
| `paths.py` | 输出目录 `output/` |
| `计划及完成进度.md` | 方案与进度文档 |

## 运行

在本目录下执行，例如：

```bash
python validate_moc_test_b.py
python validate_moc_joukowsky.py
```

结果写入 `output/`（PNG + JSON）。

## 验证项

- Joukowsky 解析解
- 单缝 / 双缝反射
- 稳态 vs Brunone 摩阻
- 滤失 + 稳态摩阻（测试 B）
