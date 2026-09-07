# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import N_MAX, Round2Dataset, collate_predict, load_round1_manifest, nested_ids, nested_order
from models import QueryFourierMLP
from paths import CKPT_DIR, CONFIG_DIR, FIG_DIR, TABLE_DIR, ensure_dirs


def _load_overfit(n):
    files = sorted(TABLE_DIR.glob(f"overfit_n{n}_*.json"))
    # prefer PASS for n=1, else latest
    chosen = files[-1]
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") == "PASS":
            chosen = f
    return json.loads(chosen.read_text(encoding="utf-8")), chosen


def main():
    ensure_dirs()
    recs = {}
    for n in (1, 8, 64):
        recs[n], _ = _load_overfit(n)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4))
    for n, c in recs.items():
        h = c["history"]
        ax.plot([r["epoch"] for r in h], [r["train_pert_mean"] for r in h],
                label=f"n={n} best={c['best']['pert_l2_mean']:.3f} {c['status']}")
    ax.axhline(0.01, ls="--", c="0.4", lw=0.8, label="1% (n=1 gate)")
    ax.axhline(0.02, ls=":", c="0.4", lw=0.8, label="2% (n=8/64 gate)")
    ax.set_xlabel("epoch (train-selected, no val early-stop)")
    ax.set_ylabel("train perturbation L2")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Round2 C1 overfit — QueryFourierMLP, seed=42")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig_r2_overfit_curves.png", dpi=160)
    fig.savefig(FIG_DIR / "Fig_r2_overfit_curves.pdf")
    plt.close(fig)

    # n=1 waveform from official PASS ckpt
    cfg = yaml.safe_load((CONFIG_DIR / "train_round2.yaml").read_text(encoding="utf-8"))
    man = load_round1_manifest()
    order = nested_order(man, int(cfg["subset_seed"]))
    ds = Round2Dataset("train", cfg, man, case_ids=nested_ids(order, 1))
    ckpt = Path(recs[1]["ckpt_best"])
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = QueryFourierMLP(8 + 3 * N_MAX, d=cfg["model"]["d"],
                            n_harmonics=cfg["model"]["n_harmonics"],
                            n_layers=cfg["model"]["n_layers"])
    model.load_state_dict(blob["model"])
    model.eval()
    b = ds[0]
    pb = collate_predict([b], blob["norm"])
    with torch.no_grad():
        y = model.predict(torch.from_numpy(pb["feat"]), torch.from_numpy(pb["tau"]),
                          float(blob["norm"]["p_pert_scale"]))["p_pert"][0].numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
    ax.plot(b.query.t_rel, b.targets.p_pert / 1e6, lw=1.2, label="MOC resampled")
    ax.plot(b.query.t_rel, y[:b.query.t.size] / 1e6, lw=1.0, label="QueryFourierMLP best")
    ax.set_xlabel(r"$t-t_s$ (s)")
    ax.set_ylabel("wellhead $p$ perturbation (MPa)")
    ax.set_title(f"{b.physical.case_id}  relL2={recs[1]['best']['pert_l2_mean']:.4f}  "
                 f"run {recs[1].get('run_id', recs[1]['ckpt_best'][-20:-8])}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig_r2_n1_waveform.png", dpi=160)
    fig.savefig(FIG_DIR / "Fig_r2_n1_waveform.pdf")
    plt.close(fig)
    print("wrote figures")


if __name__ == "__main__":
    raise SystemExit(main())
