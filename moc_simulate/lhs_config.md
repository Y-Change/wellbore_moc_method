# LHS 批量仿真参数与数据结构说明 (lhs_config.md)

本文件旨在说明和规范 `run_lhs_batch_simulate.py` 在执行水力压裂井后进液诊断（集总柔度与分布滤失反演）课题时的数据生成规范与 PyTorch 接口定义。

---

## 1. 物理参数采样区间 (Latin Hypercube Sampling Ranges)

为保证我们构建的训练集充分覆盖压裂现场各种地层（页岩、硬质砂岩、多簇缝网叠加），脚本在 `LHS_PARAM_RANGES` 中设定了如下科学对数均匀/均匀空间：

| 物理参数 | 符号 | 采样空间类型 | 采样上下限 | 现场物理意义与对标 |
| :--- | :---: | :---: | :---: | :--- |
| **压裂簇数** | $N_{cl}$ | 离散均匀分布 | $1 \sim 6$ 簇 | 模拟单缝改造至水平井段多簇穿孔压裂 |
| **缝网分布深界** | $Z_{start}\sim Z_{end}$ | 连续均匀分布 | $3500 \text{ m} \sim 4800 \text{ m}$ | 定位在中深层压裂作业井段 (总井深 $5000\text{ m}$) |
| **簇间最小间距** | $\Delta d_{min}$ | 硬约束下限 | $\ge 15.0 \text{ m}$ | 防止物理重叠，保证水击回波具备基础可辨识度 |
| **集总柔度** | $C_f$ | **对数均匀分布** | $10^{-8.7} \sim 10^{-5.5} \text{ m}^3/\text{Pa}$ | 对标现场 $0.01 \sim 1.0 \text{ bbl/psi}$ 储能效应 |
| **分布滤失系数** | $k_{leak}$ | **对数均匀分布** | $10^{-6.0} \sim 10^{-3.0} \text{ m}^2\cdot\text{s/kg}$ | 对标超低渗页岩到中渗缝网的基质渗漏耗散 |

---

## 2. 三重提速策略与硬件优化绑定

1. **多进程并发优化 (`--workers 14`)**
   针对您的 **i5-12600KF (10核16线程 + 32G RAM)**，脚本默认开启 **14 个 Worker 进程**。保留 2 个线程给系统后台与 I/O 响应，使 CPU 利用率平稳维持在 88%~92%，达到极速不卡顿。
2. **物理截断时间 (`--tf 30.0`)**
   默认从原本的 $100\text{ s}$ 缩短为 $30\text{ s}$。水击波在 $5000\text{ m}$ 井中的单次往返约 $6.9\text{ s}$，$30\text{ s}$ 足以记录前 $4$ 次完整的压裂缝网反射混响，仿真单向提速 70%！
3. **预期产出时间**
   * 快速小测 (`--n-samples 10`)：耗时约 **5 ~ 8 秒**。
   * 黄金训练集 (`--n-samples 1500`)：耗时约 **45 ~ 55 分钟**。

---

## 3. 数据输出目录布局与数据格式

当脚本运行完毕后，会在指定的输出路径（默认 `output/lhs_dataset/`）生成如下整齐的结构：

```text
output/lhs_dataset/
  ├── lhs_summary.csv        # 汇总索引表 (快速查询全部样本参数与执行状态)
  ├── lhs_metadata.json      # 完整全局元数据与运行参数记录
  └── data/
       ├── case_00000.npz    # 单个样本的压缩波形与标签
       ├── case_00001.npz
       ├── ...
       └── case_01499.npz
```

### 3.1 为什么选用 `.npz` 替代纯文本 CSV 或大型 HDF5？
* **零外挂依赖**：不需要额外安装 `h5py` 或配置复杂的数据库，Python 内置 `numpy.savez_compressed` 直接极速读写。
* **PyTorch DataLoader 极速索引**：每个 case 为独立小文件（约十几 KB），训练神经算子和条件扩散模型时，PyTorch Dataset 按需秒级读入，完全不受 I/O 阻塞！

### 3.2 单个 `.npz` 内部的结构化键值定义

可以通过 `data = np.load("case_00000.npz")` 查看：

| Key 名称 | 数据类型/形状 | 说明 (AI 反演特征与标签) |
| :--- | :---: | :--- |
| `t` | `np.ndarray`, shape `(30000,)` | 时间步序列 $[0.000, 0.001, \dots, 30.000]\text{ s}$ |
| `H_wh` | `np.ndarray`, shape `(30000,)` | **核心观测特征 $y(t)$**：井口水头时序信号 $[\text{m}]$ |
| `Q_wh` | `np.ndarray`, shape `(30000,)` | 辅助观测特征：井口流量时序信号 $[\text{m}^3/\text{s}]$ |
| `x_f` | `np.ndarray`, shape `(n_frac,)` | **反演真实标签 $p_i$**：各压裂簇位置 $[\text{m}]$ |
| `Cf` | `np.ndarray`, shape `(n_frac,)` | **反演真实标签 $C_f$**：各簇集总柔度 $[\text{m}^3/\text{Pa}]$ |
| `kleak` | `np.ndarray`, shape `(n_frac,)` | **反演真实标签 $q_L$**：各簇分布滤失 $[\text{m}^2\cdot\text{s/kg}]$ |
| `n_frac` | scalar `int` | 压裂簇总条数 $N_{cl}$ |
| `friction`| scalar `str` | 摩阻模型 (如 `"brunone"`) |

---

## 4. PyTorch 神经算子与扩散去噪器 Dataset 示例对接代码

后续在编写 Phase 3 训练代码时，只需用以下最精简的模板即可将上述数据喂入我们的**相幅倒谱双流条件编码器**：

```python
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class FracturingWaterHammerDataset(Dataset):
    def __init__(self, data_dir="output/lhs_dataset/data"):
        self.files = sorted(glob.glob(os.path.join(data_dir, "case_*.npz")))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        npz = np.load(self.files[idx])
        # 1. 输入时序特征 y(t): [1, T] -> 供倒谱与小波双流网络提取特征
        signal_H = torch.tensor(npz["H_wh"], dtype=torch.float32).unsqueeze(0)
        
        # 2. 输出真实参数标签 X_0: 这里演示打包 n_frac, Cf, kleak
        # 实际训练时可按需求填充至固定维度 max_n_frac=6 进行对齐
        cf_target = torch.tensor(npz["Cf"], dtype=torch.float32)
        kleak_target = torch.tensor(npz["kleak"], dtype=torch.float32)
        
        return signal_H, cf_target, kleak_target
```
