import os, time

handoffs = [
    r'.agents/explorer_p3_theory/handoff.md',
    r'.agents/explorer_p3_codebase/handoff.md',
    r'.agents/explorer_p3_data/handoff.md',
    r'.agents/worker_m1_m2_dis/handoff.md',
    r'.agents/worker_m3_benchmark/handoff.md',
    r'.agents/worker_m4_report/handoff.md',
    r'.agents/reviewer_1_p3/handoff.md',
    r'.agents/reviewer_2_p3/handoff.md',
    r'.agents/challenger_1_p3/handoff.md',
    r'.agents/challenger_2_p3/handoff.md',
    r'.agents/auditor_p3/handoff.md',
    r'.agents/orchestrator_5/handoff.md'
]

for h in handoffs:
    if os.path.exists(h):
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(h)))
        size = os.path.getsize(h)
        print(f'{mtime} | {size:>6} B | {h}')
    else:
        print(f'MISSING | {h}')
