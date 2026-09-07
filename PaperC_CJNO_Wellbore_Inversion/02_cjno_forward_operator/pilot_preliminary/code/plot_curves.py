# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIG_DIR, TABLE_DIR, ensure_dirs


def plot_history(json_path: Path, out_png: Path, run_id: str = ""):
    rec = json.loads(json_path.read_text(encoding="utf-8"))
    hist = rec.get("history", [])
    if not hist:
        return
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if "train_L" in hist[0]:
        ax.plot(ep, [h["train_L"] for h in hist], label="train L")
    ax.plot(ep, [h["val_pert_l2"] for h in hist], label="val pert L2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.legend()
    ax.set_title(f"{rec.get('tag','')} seed={rec.get('seed')}")
    fig.text(0.01, 0.01, f"run_id={run_id or rec.get('tag')}", fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ab_compare(hard_json: Path, soft_json: Path, out_png: Path):
    h = json.loads(hard_json.read_text(encoding="utf-8"))
    s = json.loads(soft_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    if h.get("history"):
        ax.plot([r["epoch"] for r in h["history"]],
                [r["val_pert_l2"] for r in h["history"]],
                label=f"hard val  run={h.get('run_id', h.get('tag'))[:24]}")
    if s.get("history"):
        ax.plot([r["epoch"] for r in s["history"]],
                [r["val_pert_l2"] for r in s["history"]],
                label=f"soft val  run={s.get('run_id', s.get('tag'))[:24]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val pert L2 (16-case subset)")
    ax.legend(fontsize=7)
    ax.set_title("A1b hard vs soft (same PilotPrototype shell)")
    fig.text(0.01, 0.01, f"hard={h.get('run_id', h.get('tag'))}  soft={s.get('run_id', s.get('tag'))}",
             fontsize=6)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_n_curve(rows, out_png: Path, run_note: str = ""):
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for mode in sorted({r["mode"] for r in rows}):
        sub = [r for r in rows if r["mode"] == mode]
        sub = sorted(sub, key=lambda r: r["n"])
        ax.plot([r["n"] for r in sub], [r["val_pert"] for r in sub], "o-", label=mode)
    ax.set_xlabel("n_train")
    ax.set_ylabel("best val pert L2")
    ax.legend()
    ax.set_title("nested-n learning curve (not official HPO)")
    fig.text(0.01, 0.01, run_note, fontsize=6)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ensure_dirs()
    for p in TABLE_DIR.glob("*_seed42.json"):
        plot_history(p, FIG_DIR / f"{p.stem}.png")
    hard = TABLE_DIR / "hard_n256_torch24_seed42.json"
    soft = TABLE_DIR / "soft_n256_torch24_seed42.json"
    if hard.exists() and soft.exists():
        plot_ab_compare(hard, soft, FIG_DIR / "Fig_pilot_hard_vs_soft_n256.png")


if __name__ == "__main__":
    main()
