# Stage-2 pilot preliminary

预研目录。**不宣布**阶段一正式验收、H1、正式 OOD 或全场精度。

正式训练解释器：`D:\Anaconda\envs\torch24\python.exe`（torch 2.4.0 + CUDA）。

结论：`pilot_stage_conclusion.md`  
数据准入：`data_audit/`  
节点方程：`docs/node_equations.md`  
缓存约定：`docs/cache_format.md`

```text
set PYTHON=D:\Anaconda\envs\torch24\python.exe
cd code
%PYTHON% test_interface.py
%PYTHON% test_dataset.py
%PYTHON% test_checkpoint.py
%PYTHON% test_memory_moc.py
%PYTHON% audit_pilot.py
%PYTHON% train.py --mode hard --n-train 256 --seed 42 --tag hard_n256_torch24
%PYTHON% train.py --mode soft --n-train 256 --seed 42 --tag soft_n256_torch24
%PYTHON% train.py --mode hard --n-train 64 --seed 42 --tag hard_n64_torch24
%PYTHON% train.py --mode soft --n-train 64 --seed 42 --tag soft_n64_torch24
%PYTHON% train.py --mode fno --n-train 256 --seed 42 --tag fno_n256_torch24
%PYTHON% eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val
%PYTHON% eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split test
%PYTHON% plot_waveforms.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val
%PYTHON% write_pilot_report.py
```

256 跑完后可用 `code/run_remaining_torch24.ps1` 接嵌套 n、FNO、eval 与报告。

禁止：`--confirm-30k`；改写 Stage-1 goldens / pilot manifests。
