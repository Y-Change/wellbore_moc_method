# output 分级目录说明

采用三级模型：`output / {验证系列} / {用例} / {产物}.{ext}`

## Step 物理验证（无 L2 用例）

| 目录 | 产物 |
|------|------|
| `step01_joukowsky/` | `validation.png`, `validation.json` |
| `step03_fracture/` | 同上 |
| `step03b_brunone/` | 同上 |
| `step04a_dual_fracture/` | 同上 |
| `step04a_friction_only/` | 同上 |

## 测试 B — MOC + 标准倒谱

目录：`test_b/{single|dual|triple|quad|quint}/`

| 文件 | 内容 |
|------|------|
| `moc_leakoff.png` | 2×2 水击 / 差信号 / 缝节点图 |
| `moc_leakoff.json` | PASS/FAIL 判定与 metrics |
| `cepstrum_standard.png` | 四联标准倒谱图 |

## 倒谱方法对比 — Kaiser-Bessel

目录：`cepstrum/kaiser_bessel/{single|dual|triple|quad|quint}/`

| 文件 | 内容 |
|------|------|
| `compare_2d.png` | 6 方法 2D 热力图对比 |
| `profile_overlay.png` | 1D 时间平均剖面叠加 |
| `profile_grid.png` | 6 方法 1D 分图 |
| `metrics.json` | 各方法缝深匹配结果 |

## 分析

目录：`analysis/window_comparison/`

| 文件 | 内容 |
|------|------|
| `compare_2d.png` | 窗函数 2D 倒谱对比 |
| `cepstrum_{win_type}.png` | 各窗函数单独图 |

## 旧文件

迁移前扁平 output（如 `test_b_fracture_leakoff.png`）已归档至 `output/_legacy/`。

新运行结果写入 `output/{系列}/{用例}/`。
