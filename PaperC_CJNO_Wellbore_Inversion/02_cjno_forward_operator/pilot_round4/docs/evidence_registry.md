# evidence_registry

协议：只认显式 `run_id` 目录。不读 `latest/`。子进程失败、未完成或协议不匹配不得引用旧 PASS。

## Round 3 C 运行归属（只只读审计）

| run_id | 代码摘要前缀 | 观察 | 文件 | 判定 |
|---|---|---|---|---|
| `r3_C_gates_20260906_223058_4eb48517` | 358e4a6a | 终端 killed，259 s | stdout 到 22:52，wall 1291 s，metrics 仅汇总字段 | **CONFLICTING_EVIDENCE**。不得当完整 C 门 |
| `r3_C_gates_20260906_223531_c5f1025e` | 82890a82 | 观察时仍在首案长窗 ZVB nx=64 | stdout 有 refine + zvb 启动后长间隔 | **TIMEOUT_OR_KILLED_THEN_PARTIAL_CONTINUE**。不得当 PASS |
| `r3_C_gates_20260906_223918_f538d6f9` | 542f26d4 | 终端 succeeded，226 s | `C done` | **COMPLETED**。唯一正常结束的 C-gates |

共享 `pilot_round3/manifests/production_rollout_gates.json`：**来源不确定**（多进程 + 事后补丁）。`same_grid_discrete_pass` / `continuum_0p5pct_gate` 不猜测补全，Round4 不以该文件过门。

## Round 4 绑定

每个 run 目录含 `run_meta.json`（code/stage1 sha256）、`resolved_config.json`、`metrics.json`。冻结核 sha256 见 P0。四案 source sha256 写入各 run 配置。
