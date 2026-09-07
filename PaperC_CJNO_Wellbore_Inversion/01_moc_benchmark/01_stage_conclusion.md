# 阶段一结论：MOC 高保真基准与能量收支

版本：2026-09-05  
依据：`攻关执行方案_v2.md` §2–§3  
验收表：`tables/table_moc_gate.tex`  
指标 JSON：`data/goldens/metrics.json`  
图：`figures/Fig_2_moc_convergence.pdf`（边注含 run_id / digest）

---

## 1. 本阶段做了什么

按 §3.1 重写并锁死了确定性正演链，全部新产出只写入 `PaperC_CJNO_Wellbore_Inversion/`：

| 模块 | 合同 |
|---|---|
| `moc_solver.py` | 完整 C± 相容式（保留 ±B Q_P）；Darcy / Zielke–VB 递归 / 直接卷积 / Brunone 四分支分列；端口能量与节点质量审计 |
| `node_newton.py` | 逐簇阻尼 Newton；孔眼二次项 + 独立迂曲项；惯性指数梯形；储容 Crank–Nicolson（刚性格切换指数积分器）；无量纲 ‖r‖_∞ < 10^{-8} |
| `memory_kernel.py` | 分段完整 Zielke W + Vardy–Brown；M 项指数递归；正性约束；直接卷积参考 |
| `grid.py` | 段内均匀、簇节点精确落位；Cr_j ≤ Cr_target；短段全局缩 dt |
| `validate.py` / `test_units.py` | §3.3 双档 Cr 验收门 + R1 量纲/特征结构单测 |
| `generate_dataset.py` | maximin-LHS pilot；正式 Sobol 在全部门未 PASS 前拒绝启动 |

生产数值：Nx_ref = 1024，Cr = 1（验收基准），Cr = 0.8 只作插值差异说明。记忆核生产路径 M = 12（见 §4 修复记录）。Brunone 锁定 `vitkovsky2000`（见 §4）。

---

## 2. 量化验收（Cr = 1 为硬门）

| 项目 | 实测 | 阈值 | 状态 |
|---|---|---|---|
| 单元锁（C±、量纲、Z_b、Brunone 特征速、核正性、落网） | 40 passed | 0 fail | PASS |
| Joukowsky 峰值 / 周期 | 0 / 1.6×10^{-16} | < 10^{-6} | PASS |
| 单 shunt Γ | 6.9×10^{-10} | < 1% | PASS |
| 被动 RCI 频响峰值 | 0.070% | < 1% | PASS |
| Bergant 2001 装置 Zielke 包络（递归 vs 直接卷积） | 0.084% | < 5% | PASS |
| Bergant 首峰 vs aV_0/g（9 ms 关阀） | 2.31% | < 5% | PASS |
| TMM 共振频率 | 0.048%（匹配 5 峰） | < 0.5% | PASS |
| Richardson 阶（单管 Darcy、井口峰值） | 2.49 | ≥ 1.8 | PASS |
| 三网格峰值差 / 首波到时差 | 0.0093% / 0.015% | < 0.5% / < 0.2% | PASS |
| 每步节点质量残差 | 1.0×10^{-10} | < 10^{-6} | PASS |
| 端口能量不平衡 | 8.9×10^{-8} | < 10^{-4} | PASS |
| 记忆核拟合 / J_u L2 / 相位 | 1.14% / 0.82% / 0.38° | < 2% / < 1% / < 1° | PASS |
| 波形级递归 vs 直接卷积 L2 / 相位 | 0.049% / 0.11° | < 1% / < 1° | PASS |
| s_min = 10 m，70 Hz 每波长点数（Nx=2048） | 10.0 | ≥ 10 | PASS |
| s_min 到时残差 vs 3.5 ms 门 | 0.087 ms | < 3.5 ms | PASS |
| Newton 收敛率（pilot 2000） | 1.000（1896 例求解，0 失败；104 例因稳态井口压力越界拒绝） | > 99.9% | PASS |
| Joukowsky Cr = 0.8 | 峰值 2.8×10^{-15}，周期机器精度 | < 10^{-3} | PASS |
| Brunone（单独） | W_u = +4.93×10^{-2} J ≥ 0；包络另报 | 不与 Zielke 混合 | REPORT + 符号 PASS |

Cr = 0.8 插值档：shunt 面积比 Γ 误差 0.53%（峰比因弥散不可用）、Richardson 阶 0.51、全局体积审计 4.4×10^{-3}，一律 **REPORT**，与 Ghidaoui & Karney (1994) 的插值耗散同量级说明一致；高频/到时验收以 Cr = 1 为准。

记忆核 wall-clock（Bergant 装置，Nx=256，8 个往返）：直接卷积 28.4 s，递归核 0.12 s。一维 J_u 历史两者均约 0.01 s。复杂度分析不替代该实测表。

---

## 3. 论文 §3 方法学初稿（可直接改写入主稿）

The wellbore is advanced by the method of characteristics on a segment-wise mesh that places every perforation cluster on a node. Along each characteristic the compatibility relations are kept in the complete two-unknown form

$$C^+:\ H_P+B Q_P+\tfrac12 R_P Q_P|Q_P|=C_P,\qquad
C^-:\ H_P-B Q_P-\tfrac12 R_P Q_P|Q_P|=C_M,\qquad B=a/(gA),$$

so that the new head and discharge are obtained simultaneously; dropping the \(\pm B Q_P\) terms makes the pair inconsistent (unit-tested). When \(C_r=a\Delta t/\Delta x<1\) the feet are obtained by linear space-line interpolation. All high-frequency and arrival claims are accepted on the interpolation-free \(C_r=1\) grid and merely differenced at \(C_r=0.8\), following the dissipation estimate of Ghidaoui & Karney (1994).

Each cluster is a damped Newton solve of mass jump, continuity of the wellbore pressure trace, per-perforation orifice and tortuosity laws, near-well inertia/resistance and fracture storage/leak-off. Storage is integrated by Crank–Nicolson, with an exponential integrator substituted when \(G_l\Delta t/C_f>1\). Leak-off that has entered the storage ODE is not subtracted again from the wellbore mass jump. The small-signal port impedance is the parallel storage topology \(Z_b(s)=R_{\mathrm{perf}}+R_f+sI_f+1/(sC_f+G_l)\).

Frequency-dependent wall shear is treated as two distinct models. The Zielke / Vardy–Brown path uses the piecewise-complete weighting function (small-\(\tau\) square-root series, large-\(\tau\) exponential series) as a direct-convolution reference, and an \(M=12\) positive exponential-sum recursion in production; \(M=10\) misses the 2% fit gate on \(\tau\in[10^{-8},10^2]\) and is not used. The Brunone instantaneous-acceleration path is a separate branch. On the present upstream-valve geometry the written form \(k(V_t-a\,\mathrm{sgn}(VV_x)V_x)\) injects energy; the locked discrete form is the Vítkovský (2000) sign correction, which yields non-negative unsteady work. The two models are never mixed into a single error number.

Mechanical energy excludes the memory state. Passivity of the recursive kernel is argued by positive-realness of the transfer function under \(w_l,\alpha_l,\beta_l>0\), and is checked on the closed-valve window after shut-in. Port energy imbalance on the eight-cluster reference well is \(8.9\times10^{-8}\) at \(C_r=1\).

---

## 4. 修复记录（§3.4 顺序：量纲 → CFL/插值 → 节点闭合 → 记忆核）

失败 case 未删除；下列为定位与处置。

1. **量纲（R1）**  
   PyYAML 1.1 把 `4.0e7`（指数无正负号）读成字符串，导致 `p_res / rho_g` 类型错误。`config_io.py` 增加科学计数法强制转换，并把 `physics.yaml` / `friction_brunone.yaml` 改为 `4.0e+7` 写法。单元测试锁定 `kappa, I_f, R_f, p_res` 为 float。

2. **CFL / 插值**  
   梯形摩阻曾用 \(f(Q^n)\) 滞后线性化，井口峰值 Richardson 阶只有 1.59。改为对新时刻流量做一次 Picard 校正（井口边界直接用已知 \(Q_P\)）。峰值阶升至 2.49。Cr = 0.8 的峰值 Γ 估计改用脉冲面积比；峰比因插值抹平只作诊断。

3. **节点闭合**  
   “每步质量残差”按节点 \(Q^--Q^+-\sum q\) 验收（\(10^{-10}\)），不以梯形井筒体积审计代替。全局体积审计 Cr = 1 为 \(1.7\times10^{-4}\)，记入 REPORT。

4. **记忆核**  
   M = 10 在 \(\tau\in[10^{-8},10^2]\) 上 Zielke 相对误差 2.44%、幂律 3.19%，低于 2% 门。生产路径改为 M = 12（0.82% / 1.14%），正性约束保持。Bergant 层流包络以同一装置上的直接卷积为目标，不把 Brunone 误差并入该行。

5. **Brunone 符号（R7）**  
   方案书写形式在上游阀构型下累计非定常功为负。按失败→定位→记录，锁定 `vitkovsky2000`；`plan_v2` 保留为对照臂。该结论仅对固定 ±a 模板 + 滞后源项的该变体成立。

Zielke 1968 分段级数在公布分界 \(\tau=0.02\) 处有约 10% 固有跳跃；参考路径保持分段完整 W，不做 C^0 拼接。

---

## 5. 簇落网与 s_min = 10 m

连续簇位在均匀 \(\Delta x\approx4\) m 网格上的量化可达 \(\pm2\) m，往返到时误差最大约 5.7 ms，超过 §4.3 的 3.5 ms 门。现行网格把每簇放在节点上，量化误差为 0。Nx_ref = 1024 时 70 Hz 每波长仅 4 点；Nx_ref = 2048（2× 加密，兼作阶段三 challenge 真值档）达到 10 点/波长。生产网格与 2× 网格的到时差 0.087 ms，对照 3.5 ms 门。

---

## 6. Pilot 与正式扩充纪律

- 生成器：`generate_dataset.py --pilot`，maximin-LHS，seed 20260905，24 维，8 个候选设计取最大最小间距。  
- 正式 Owen-scrambled Sobol（20k / 3k / 3k / 4k）在 `goldens/metrics.json` 仍有 FAIL 或 Newton 门未 PASS 时被 CLI 拒绝（§3.3）。  
- 失败例写入 `manifest_v1.json` 的 `failures`，不删除（R4）。  
- 切分按井 × 簇布局 × 激励事件的 `group_id`，禁止同一 case 的时间窗跨 split（R8）。

覆盖与 Newton 率（已冻结）：`data/pilot_2000/coverage.json` 与 `data/manifests/manifest_v1.json`。

- 抽样 2000 组，几何全部可构造；求解后 104 例因 \(p_{\mathrm{wh}}\) 越界拒绝（记录在 manifest，不删除）。
- 1896 例进入 Newton：失败 0、错误 0，案例收敛率 **100%**。
- \(N=4\ldots12\) 各约 200 例；最小间距 10.53 m（训练域下界 12 m 经 jitter 后的实现值，仍 ≥ 10 m 拒绝线）。
- 生成器 `frozen=true`。正式 Owen-scrambled Sobol + 8 类 OOD 采样器已实现：`--formal` 无确认开关拒绝启动 30k MOC；`--formal --design-only` 写出完整 20k/3k/3k/4k 设计与 R8 切分；`--formal --smoke` 小规模仿真走通 schema；满量仿真必须 `--formal --confirm-30k`。

---

## 7. 审稿人预置问答（阶段一）

**Q. 为何不用商业瞬变流软件做真值？**  
A. 节点 RCI 拓扑、迂曲独立加性项、记忆核预历史 \(z_l(0)\) 和逐项残差量纲需要白盒控制。商业软件不提供这些残差。频域侧以传递矩阵共振交叉验证（Cr = 1 匹配误差 0.048%）。

**Q. \(C_r=1\) 无插值是否掩盖了插值耗散？**  
A. 全部高频/到时门双档报告。Cr = 1 为验收基准；Cr = 0.8 的 Richardson 阶降至 0.51、shunt 峰比失效，与 Ghidaoui & Karney 的插值耗散估计一致，写进差异说明而不是并入 Cr = 1 误差。

**Q. 为何生产核是 M = 12 而不是方案正文的 M = 10？**  
A. M = 10 未过 < 2% 拟合门（Zielke 2.44%）。按 §3.4 记忆核修复项改 M = 12，并在 `friction_zielke.yaml` 的 `repair_record` 冻结。误差分解中拟合误差、卷积截断与网络误差将分列（阶段二）。

**Q. Brunone 与 Zielke 能否报一个“非定常摩阻误差”？**  
A. 不能。二者机制不同；验收表分列；混合出数触发 R3。

---

## 8. 尚未做、且按方案不得提前做的事

- 正式 30k Sobol / OOD **仿真**：生成器与 OOD 采样器已落地（`data/sobol_30000/design_formal.json`），但满量 MOC 未跑（磁盘与核时约 15× pilot）。启动命令：`python generate_dataset.py --formal --confirm-30k`。  
- 阶段二 CJ-NO、阶段三 FIM / 反演、商业软件对照：不在本阶段范围。  
- Bergant 2001 Fig. 3 实验曲线的独立数字化点尚未入库；本阶段用同一装置上的分段完整 Zielke 直接卷积作包络真值（该模型在原文中与层流实验一致）。若后续取得数字化点，只替换包络目标、不改求解器。

---

## 9. 复现命令

```text
python run_all_reproducible.py --stage 1 --seed 42 --device cpu --smoke
python 01_moc_benchmark/code/validate.py
python 01_moc_benchmark/code/generate_dataset.py --pilot --workers 8
python 01_moc_benchmark/code/test_generator.py
python 01_moc_benchmark/code/generate_dataset.py --formal                 # 无确认则拒绝（exit 3）
python 01_moc_benchmark/code/generate_dataset.py --formal --design-only   # 30k 设计，无 MOC
python 01_moc_benchmark/code/generate_dataset.py --formal --smoke         # 小规模 Sobol+OOD 仿真
python 01_moc_benchmark/code/generate_dataset.py --formal --confirm-30k   # 满量 MOC，勿默认执行
```
