# zvb_tensor_validation

run_id: `r4_P3_zvb_20260906_231822_8a284b86`

- 单位置单核项手算：J=[6, 1.5] PASS
- 完整冻结核 NumPy vs Tensor：rel_J ~ 1e-16 PASS
- 接入同一 `ProductionMOC.step`：z 更新后写 `Ju`，下一步 C± 使用
- 零激励 / 短脉冲 / 首案短窗 / 4 案长窗 Nx64：prod vs twin 井口 pert ~1e-15
- Darcy vs ZVB 同网格 Nx64 短窗物理差 pert L2=1.107e-2（不是 Nx32 vs 64）
- 未走拟核 / 在线 refit
