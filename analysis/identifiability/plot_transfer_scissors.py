# -*- coding: utf-8 -*-
"""plot_transfer_scissors.py — Paper C 迁移剪刀差表图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary",
        default="output/analysis/identifiability/transfer/main/transfer_summary.json",
    )
    p.add_argument(
        "--out-dir",
        default="output/analysis/identifiability/transfer/main/figures",
    )
    args = p.parse_args()
    summ = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = []
    brunone_f1 = []
    steady_f1 = []
    cs = summ["classical_summary"]
    for m in ("cepstrum", "p0"):
        methods.append(m)
        brunone_f1.append(cs[m]["brunone"]["f1_mean"])
        steady_f1.append(cs[m]["steady"]["f1_mean"])
    pn = summ.get("phasenet") or {}
    if pn.get("brunone_ref_f1") is not None and pn.get("steady_test_f1_new_thr") is not None:
        methods.append("PhaseNet")
        brunone_f1.append(pn["brunone_ref_f1"])
        steady_f1.append(pn["steady_test_f1_new_thr"])

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w / 2, brunone_f1, w, label="brunone (train/dict)", color="#4C72B0")
    ax.bar(x + w / 2, steady_f1, w, label="steady (paired)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("F1 @ 10 m")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title("Paired friction-mismatch scissors")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_scissors_f1.png", dpi=150)
    fig.savefig(out / "fig_scissors_f1.svg")
    fig.savefig(out / "fig_scissors_f1.pdf")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
