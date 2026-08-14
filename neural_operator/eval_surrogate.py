# -*- coding: utf-8 -*-
"""
eval_surrogate.py — FNO 正演代理精度门槛（相对 L2 / RMSE）。

验收（方法段入口，非反演门槛）：
  在 held-out 集上报告 mean/median 相对 L2、RMSE，并导出叠图。
  「FNO-MLE 与 MOC-MLE 位置差 < CRB」是下一步；本脚本只做波形精度。

用法
----
    python -m neural_operator.eval_surrogate \\
        --ckpt output/models/fno_surrogate_best.pt \\
        --data-dir output/lhs_dataset_2000/data \\
        --out-dir output/analysis/identifiability/fno_accuracy/main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _p = os.path.dirname(_d)
    if _p == _d:
        break
    _d = _p

from neural_operator.dataset_surrogate import FracturingMOCSurrogateDataset
from neural_operator.fno_1d import FNO1dSurrogate


def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(pred - target)
    den = np.linalg.norm(target) + eps
    return float(num / den)


def main():
    p = argparse.ArgumentParser(description="FNO surrogate accuracy gate")
    p.add_argument("--ckpt", default="output/models/fno_surrogate_best.pt")
    p.add_argument("--data-dir", default="output/lhs_dataset_2000/data")
    p.add_argument("--out-dir", default="output/analysis/identifiability/fno_accuracy/main")
    p.add_argument("--n-time", type=int, default=4096)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--modes", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-batches", type=int, default=0, help="0=全验证集")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(args.data_dir)

    ds = FracturingMOCSurrogateDataset(
        data_dir=args.data_dir,
        n_time_target=args.n_time,
        split="val",
        train_ratio=0.85,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = FNO1dSurrogate(
        in_channels=4,
        out_channels=1,
        width=args.width,
        modes=args.modes,
        num_layers=args.layers,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rels, rmses, mses = [], [], []
    t0 = time.time()
    n_seen = 0
    first_batch = None
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            # timing one batch later
            p_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            for i in range(p_np.shape[0]):
                pi = p_np[i, 0]
                yi = y_np[i, 0]
                rels.append(relative_l2(pi, yi))
                err = pi - yi
                rmses.append(float(np.sqrt(np.mean(err ** 2))))
                mses.append(float(np.mean(err ** 2)))
            if first_batch is None:
                first_batch = (y_np[:4], p_np[:4])
            n_seen += p_np.shape[0]
            if args.max_batches and (bi + 1) >= args.max_batches:
                break

    # timing
    x0, y0 = next(iter(loader))
    x0 = x0[:1].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(x0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd_ms = (time.time() - t1) / 20.0 * 1000.0

    rels = np.asarray(rels, dtype=float)
    rmses = np.asarray(rmses, dtype=float)
    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "data_dir": os.path.abspath(args.data_dir),
        "n_val_files": len(ds.files),
        "n_evaluated": int(len(rels)),
        "n_time": args.n_time,
        "device": str(device),
        "rel_l2_mean": float(np.mean(rels)),
        "rel_l2_median": float(np.median(rels)),
        "rel_l2_p90": float(np.percentile(rels, 90)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_median": float(np.median(rmses)),
        "mse_mean": float(np.mean(mses)),
        "forward_ms_batch1": float(t_fwd_ms),
        "wall_s": float(time.time() - t0),
        "gate_note": (
            "Waveform accuracy audit only. Inversion gate "
            "(FNO-MLE vs MOC-MLE position error < CRB) is separate."
        ),
        # Soft advisory thresholds for discussion (not hard pass/fail of science claim)
        "advisory_rel_l2_ok": bool(np.median(rels) < 0.05),
    }
    (out / "fno_accuracy.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        out / "per_sample_metrics.npz",
        rel_l2=rels,
        rmse=rmses,
    )

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        yt, pt = first_batch
        t = np.linspace(0, 1, yt.shape[-1])  # normalized; true tf unknown per-plot
        for i, ax in enumerate(axes.ravel()):
            if i >= yt.shape[0]:
                ax.axis("off")
                continue
            ax.plot(t, yt[i, 0], "k-", lw=1.2, label="MOC")
            ax.plot(t, pt[i, 0], "r--", lw=1.0, label="FNO")
            ax.set_title(f"val sample {i}  relL2={relative_l2(pt[i,0], yt[i,0]):.3f}")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        fig.suptitle("FNO vs MOC (normalized time axis)")
        fig.tight_layout()
        fig.savefig(out / "fno_vs_moc_overlay.png", dpi=150)
        fig.savefig(out / "fno_vs_moc_overlay.svg")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(rels, bins=30, color="steelblue", edgecolor="k", alpha=0.85)
        ax.axvline(summary["rel_l2_median"], color="r", ls="--", label="median")
        ax.set_xlabel("relative L2")
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rel_l2_hist.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
