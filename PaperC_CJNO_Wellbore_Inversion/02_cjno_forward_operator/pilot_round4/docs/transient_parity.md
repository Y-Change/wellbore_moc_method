# transient_parity

run_id: `r4_P1_transient_20260906_231503_6d5cfa65`

24 条非静默瞬态全部 PASS（4 案 × none/Darcy × 短窗 Nx64 + 短窗 Nx512 + 长窗 Nx64）。

- 短窗覆盖阀动作及首簇反射回井口；长窗为 Round3 登记窗口。
- 比较井口扰动与全部簇的 q、pc、F、左右 H/Q。Newton 均被调用。
- 全簇扁平数组 + offsets、真实初态、时间、左右迹、空间状态与 ZVB 记忆（none/Darcy 时 z=0）写入各 run 的 npz。
- 同源近零误差是实现一致性，不是独立物理证明。
- 未用 15/20 步当瞬态终点。
