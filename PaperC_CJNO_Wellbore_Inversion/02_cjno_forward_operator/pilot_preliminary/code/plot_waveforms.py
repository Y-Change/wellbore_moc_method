# -*- coding: utf-8 -*-
"""Val waveform overlays. Caption must carry run_id. CPU-only to avoid GPU clash."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cjno import PilotPrototype
from dataset import PilotDataset, collate_batch
from paths import CONFIG_DIR, FIG_DIR
from train import batch_to_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-cases", type=int, default=4)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load((CONFIG_DIR / "train_forward_pilot.yaml").read_text(encoding="utf-8"))
    model = PilotPrototype(mode=blob.get("mode", "hard"), d=cfg["model"]["d"],
                           fno_d=cfg["model"]["fno_d"], fno_modes=cfg["model"]["fno_modes"])
    model.load_state_dict(blob["model"])
    model.to("cpu").eval()
    ds = PilotDataset(args.split, window_periods=cfg["data"]["window_periods"],
                      wellhead_decim=cfg["data"]["wellhead_decim"],
                      node_max_len=cfg["data"]["node_max_len"])
    n = min(args.n_cases, len(ds))
    fig, axes = plt.subplots(n, 1, figsize=(6.2, 1.7 * n), sharex=False)
    if n == 1:
        axes = [axes]
    run_id = Path(args.ckpt).parent.name
    with torch.no_grad():
        for i in range(n):
            s = ds[i]
            batch = batch_to_torch(collate_batch([s], blob["norm"]), "cpu")
            hat = model(batch, apply_hard_times=0)["p_pert_hat"][0].numpy()
            t = s.t - s.t[0]
            ax = axes[i]
            ax.plot(t, s.p_pert, lw=1.0, label="MOC pert")
            ax.plot(t, hat, lw=1.0, label="pred pert")
            ax.set_ylabel("Pa")
            ax.set_title(f"{s.case_id}  N={s.N}", fontsize=8)
            if i == 0:
                ax.legend(fontsize=7)
    axes[-1].set_xlabel("t after shut-in window start [s]")
    fig.suptitle(f"{args.split} overlays  mode={blob.get('mode')} seed={blob.get('seed')}", fontsize=10)
    fig.text(0.01, 0.01, f"ckpt={Path(args.ckpt).as_posix()}  tag={run_id}", fontsize=6)
    fig.tight_layout()
    out = FIG_DIR / f"Fig_pilot_waveforms_{Path(args.ckpt).parent.name}_{args.split}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
