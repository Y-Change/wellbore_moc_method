# causality_and_convergence

两张独立表。阈值未改。粗网格 FAIL 保留。

时序 run_id: `r4_P2_causality_20260906_232440_43c7df5c`  
精度 run_id: `r4_P2_align_20260906_232814_15f0cf88`

详见 `round4_decision.md` 第 2 节。

- Nx512 四案实测 Cr0=1，不是普遍保证。
- 单程 4/4 PASS；往返 3/4 PASS（00543 FAIL，误差 0.01035 s，门 0.00930 s）。
- 最细参考加密 <0.2% 后，生产 Nx512 vs 参考 1024 的 0.5% 门 4/4 PASS。
