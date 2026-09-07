$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$py = "D:\Anaconda\envs\torch24\python.exe"
Set-Location $PSScriptRoot

# After official 256 hard/soft have finished. Do not start if GPU train is still running.
& $py train.py --mode hard --n-train 64 --seed 42 --tag hard_n64_torch24
& $py train.py --mode soft --n-train 64 --seed 42 --tag soft_n64_torch24
& $py train.py --mode fno --n-train 256 --seed 42 --tag fno_n256_torch24
& $py eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val
& $py eval.py --ckpt ../checkpoints/soft_n256_torch24/seed42_best.pt --split val
& $py eval.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split test
& $py plot_waveforms.py --ckpt ../checkpoints/hard_n256_torch24/seed42_best.pt --split val
& $py plot_waveforms.py --ckpt ../checkpoints/soft_n256_torch24/seed42_best.pt --split val
& $py write_pilot_report.py
