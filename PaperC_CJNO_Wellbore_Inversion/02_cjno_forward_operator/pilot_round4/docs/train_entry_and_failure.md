# train_entry_and_failure

run_id: `r4_P4_train_20260906_232156_c0d8e7d4`

- `reset()` 不恢复外接 pK
- `TrainEntry` 与直接 unroll 对 pK 的梯度一致：-16.101768663382444
- K（none）全步长 rel：1.59e-5 / 4.84e-5 / 1.16e-3，最佳过 1e-4
- Gl 最佳 8.91e-6 PASS；Cf 最佳 4.86e-3 FAIL（小步 FD=0）
- ZVB 上 K 三步长均 <1e-4 PASS
- guard/tikhonov 轨迹按步写入，不是固定 `branch_switch_separated=True`
- `max_iter=0` 原求解器 `ok=False`，optimizer/scheduler/global_step/动态状态不推进；无 monkeypatch
- 正常 max_iter=100、302 步失败率 0（只验证处理，不是分布失败率）
