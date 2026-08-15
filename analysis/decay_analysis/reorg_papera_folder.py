# -*- coding: utf-8 -*-
"""把 PaperA 归档从脚本原目录名改成论文块结构。"""
from __future__ import annotations

import shutil
from pathlib import Path

PAPERA = next(
    p
    for p in Path(r"e:\water_hammer_research\wellbore_moc_method\output").iterdir()
    if p.is_dir() and p.name.startswith("PaperA")
)


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def move_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        for child in src.iterdir():
            target = dst / child.name
            if child.is_dir() and target.exists():
                move_tree(child, target)
            else:
                shutil.move(str(child), str(target))
        try:
            src.rmdir()
        except OSError:
            pass
        return
    shutil.move(str(src), str(dst))


def move_files(src: Path, dst: Path, suffix_ok=None) -> None:
    if not src.exists():
        return
    ensure(dst)
    for f in src.iterdir():
        if not f.is_file():
            continue
        if suffix_ok and f.suffix.lower() not in suffix_ok:
            continue
        shutil.move(str(f), str(dst / f.name))


def rmdir_if_empty(p: Path) -> None:
    if p.is_dir() and not any(p.iterdir()):
        p.rmdir()


def main() -> None:
    print("PAPERA", PAPERA)

    geo = ensure(PAPERA / "01_几何网格")
    attr = ensure(PAPERA / "02_裂缝属性_CfKleak")
    leak = PAPERA / "03_leakoff验证"
    ener = ensure(PAPERA / "04_能量回归")
    bru = ensure(PAPERA / "05_Brunone常数k")

    old_decay = PAPERA / "decay_regression"
    if old_decay.exists():
        move_tree(old_decay / "01_simulated_waves", geo / "波形_npz")
        move_tree(old_decay / "03_extracted_peaks_csv", geo / "峰值表")
        fig = ensure(geo / "过程图")
        for name in [
            "02_peak_tracking_zooms",
            "04_collapse_and_scaling",
            "04_collapse_and_scaling_pidx",
            "05_friction_comparisons",
            "05_friction_comparisons_pidx",
            "06_first_frac_energy",
            "07_new_fits_comparisons",
            "paperA_stats",
        ]:
            src = old_decay / name
            if src.exists():
                move_tree(src, fig / name)

        ck = old_decay / "cf_kleak_study"
        if ck.exists():
            move_tree(ck / "waves", attr / "波形_npz")
            move_tree(ck / "plots", attr / "图")
            csv = ck / "cf_kleak_table.csv"
            if csv.exists():
                shutil.move(str(csv), str(ensure(attr) / "cf_kleak_table.csv"))
            rmdir_if_empty(ck)
        rmdir_if_empty(old_decay)

    old_leak = PAPERA / "leakoff"
    if old_leak.exists():
        if leak.exists():
            move_tree(old_leak, leak)
        else:
            old_leak.rename(leak)

    old_e = PAPERA / "energy_regression"
    if old_e.exists():
        move_tree(old_e / "steady", ener / "工况" / "steady")
        move_tree(old_e / "brunone", ener / "工况" / "brunone")
        move_files(old_e, ener / "图", {".png", ".svg"})
        move_files(old_e, ener / "拟合", {".json"})
        et = old_e / "energy_table.csv"
        if et.exists():
            shutil.move(str(et), str(ener / "energy_table.csv"))
        rmdir_if_empty(old_e)

    old_b = PAPERA / "brunone_spacing_effect"
    if old_b.exists():
        cases = ensure(bru / "工况")
        for child in list(old_b.iterdir()):
            if child.is_dir() and child.name.startswith("D"):
                move_tree(child, cases / child.name)
            elif child.is_dir() and child.name == "audit_v2":
                move_tree(child, bru / "audit_v2")
            elif child.is_file() and child.suffix.lower() in {".png", ".svg"}:
                ensure(bru / "图")
                shutil.move(str(child), str(bru / "图" / child.name))
            elif child.is_file() and child.suffix.lower() == ".csv":
                shutil.move(str(child), str(bru / child.name))
        rmdir_if_empty(old_b)

    print("done top-level:")
    for c in sorted(PAPERA.iterdir(), key=lambda x: x.name):
        print(" ", ("D" if c.is_dir() else "F"), c.name)


if __name__ == "__main__":
    main()
