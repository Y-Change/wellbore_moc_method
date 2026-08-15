# -*- coding: utf-8 -*-
"""补全 decay 网格的单缝工况：steady/brunone × x1∈{2000..4500} × 全部 D。

单缝几何与间距无关：每个 (摩阻, x1) 只正演一次，波形 npz 与倒谱峰复制到全部 D。
"""
from __future__ import annotations

import os
import shutil
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.isfile(os.path.join(_d, "README.md")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        raise RuntimeError("Cannot find wellbore_moc_method root")
    _d = _parent

from moc_simulate.config import FRACTURE_CONFIG, FRICTION_PARAMS
from moc_simulate.paths import SERIES_DECAY_REGRESSION, output_path
from analysis.decay_analysis.decay_regression import (
    append_csv,
    load_completed_cases,
    run_one,
)


X1_LIST = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0]
SPACING_LIST = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
FRICTIONS = ["steady", "brunone"]
N_FRAC = 1
CANONICAL_SP = 10.0


def _npz_path(friction_key: str, x1: float, spacing_m: float) -> str:
    raw_dir = output_path(SERIES_DECAY_REGRESSION, "01_simulated_waves", "")
    os.makedirs(raw_dir, exist_ok=True)
    return os.path.join(
        raw_dir,
        f"{friction_key}_x1_{int(x1)}_sp_{int(spacing_m)}_n_{N_FRAC}.npz",
    )


def main() -> None:
    csv_path = output_path(
        SERIES_DECAY_REGRESSION, "03_extracted_peaks_csv", "decay_table.csv"
    )
    completed = load_completed_cases(csv_path)
    n_sim = 0
    n_csv = 0
    n_npz_copy = 0

    for fr in FRICTIONS:
        model = FRICTION_PARAMS[fr]["friction_model"]
        for x1 in X1_LIST:
            need_sp = [
                sp
                for sp in SPACING_LIST
                if (model, x1, sp, N_FRAC) not in completed
            ]
            if not need_sp:
                print(f"skip {fr} x1={int(x1)} n=1 (CSV 已齐)")
                continue

            canon = _npz_path(fr, x1, CANONICAL_SP)
            existing = [
                _npz_path(fr, x1, sp)
                for sp in SPACING_LIST
                if os.path.isfile(_npz_path(fr, x1, sp))
            ]
            src_npz = canon if os.path.isfile(canon) else (existing[0] if existing else None)

            if src_npz is None:
                print(f"SIM {fr} x1={int(x1)} n=1 (canonical D={int(CANONICAL_SP)})")
                rows = run_one(fr, x1, CANONICAL_SP, N_FRAC)
                n_sim += 1
                src_npz = canon
            else:
                print(f"reuse waveform {os.path.basename(src_npz)}")
                if os.path.abspath(src_npz) != os.path.abspath(canon):
                    if not os.path.isfile(canon):
                        shutil.copy2(src_npz, canon)
                        n_npz_copy += 1
                    src_npz = canon
                rows = run_one(fr, x1, CANONICAL_SP, N_FRAC)

            if not os.path.isfile(src_npz):
                raise FileNotFoundError(src_npz)

            peak = rows[0]

            for sp in SPACING_LIST:
                dst = _npz_path(fr, x1, sp)
                if os.path.abspath(src_npz) != os.path.abspath(dst):
                    if not os.path.isfile(dst):
                        shutil.copy2(src_npz, dst)
                        n_npz_copy += 1
                if (model, x1, sp, N_FRAC) in completed:
                    continue
                row = dict(peak)
                row["spacing_m"] = sp
                row["Cf"] = float(FRACTURE_CONFIG["Cf"])
                row["kleak"] = float(FRACTURE_CONFIG["kleak"])
                append_csv([row], csv_path)
                completed.add((model, x1, sp, N_FRAC))
                n_csv += 1

    print(
        f"Done. unique_MOC={n_sim} csv_rows_added={n_csv} npz_copied={n_npz_copy}"
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
