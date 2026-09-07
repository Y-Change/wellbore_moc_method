# -*- coding: utf-8 -*-
"""Locks for the formal Sobol generator and 8 OOD samplers (no 30k MOC)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_dataset as gd
from config_io import load_yaml, validate_against_schema
from paths import CODE_DIR


class _Suite:
    def __init__(self):
        self.n_ok = 0
        self.n_fail = 0
        self.failures = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.n_ok += 1
            print(f"  PASS  {name}")
        else:
            self.n_fail += 1
            self.failures.append((name, detail))
            print(f"  FAIL  {name}  {detail}")


def test_sobol_owen(s: _Suite) -> None:
    u = gd.sobol_owen(32, len(gd.TRAIN_KEYS), seed=42)
    s.check("Sobol shape", u.shape == (32, len(gd.TRAIN_KEYS)), str(u.shape))
    s.check("Sobol in unit cube", float(u.min()) >= 0.0 and float(u.max()) <= 1.0)
    v = gd.sobol_owen(32, len(gd.TRAIN_KEYS), seed=42)
    s.check("Sobol deterministic", np.allclose(u, v))
    w = gd.sobol_owen(32, len(gd.TRAIN_KEYS), seed=43)
    s.check("Sobol seed changes design", not np.allclose(u, w))


def test_group_id_and_split(s: _Suite) -> None:
    phy = load_yaml("physics.yaml")
    cases = gd.draw_n_cases(40, seed=42, domain=phy["sampling_domain"])
    g1 = gd.make_group_id(cases[0])
    g2 = gd.make_group_id(cases[0])
    s.check("group_id stable", g1 == g2)
    splits, counts = gd.assign_group_splits(cases, 24, 8, 8, seed=99)
    s.check("split counts sum", sum(counts.values()) == 40, str(counts))
    rows = [{"group_id": c["group_id"], "split": spl} for c, spl in zip(cases, splits)]
    s.check("R8 leakage empty", gd.split_leakage(rows) == [])


def test_ood_constraints(s: _Suite) -> None:
    phy = load_yaml("physics.yaml")
    domain = phy["sampling_domain"]
    all_ood = []
    for typ in gd.OOD_TYPES:
        drawn = gd.draw_n_cases(6, seed=20260905 + hash(typ) % 1000, domain=domain, ood_type=typ)
        all_ood.extend(drawn)
        s.check(f"{typ} n=6", len(drawn) == 6)
    rep = gd.ood_constraint_report(all_ood, domain)
    for typ, rec in rep.items():
        s.check(f"{typ} constraint", rec["ok"], json.dumps(rec, default=str))


def test_case_schema(s: _Suite) -> None:
    phy = load_yaml("physics.yaml")
    c = gd.draw_n_cases(1, seed=7, domain=phy["sampling_domain"])[0]
    rec = gd.case_schema_record("id_00000", c, "train", 7,
                                {"Nx_ref": 256, "Cr_target": 1.0, "friction_model": "zvb_rec"})
    try:
        validate_against_schema(rec, "case.schema.json")
        s.check("case schema", True)
    except Exception as exc:
        s.check("case schema", False, str(exc))


def test_cli_refuse_without_confirm(s: _Suite) -> None:
    cmd = [sys.executable, str(CODE_DIR / "generate_dataset.py"), "--formal"]
    p = subprocess.run(cmd, cwd=str(CODE_DIR), capture_output=True, text=True)
    s.check("--formal without confirm exits 3", p.returncode == 3, f"rc={p.returncode}\n{p.stdout}")
    s.check("refuse mentions --confirm-30k", "--confirm-30k" in p.stdout)


def test_observation_spec(s: _Suite) -> None:
    import observation as om
    data_cfg = load_yaml("data.yaml")
    spec = om.observation_spec(data_cfg)
    s.check("obs fs", spec["sampling_rate_hz"] == 1000.0)
    t = np.linspace(0, 0.2, 400)
    p = np.sin(2 * np.pi * 20 * t)
    t_o, p_o = om.apply_observation_chain(t, p, spec)
    s.check("obs resampled", t_o.size > 10 and p_o.size == t_o.size)
    fe = om.end_to_end_f_eff_hz(spec)
    s.check("f_eff >= 70 Hz", fe >= 70.0, f"f_eff={fe}")


def run_all() -> dict:
    s = _Suite()
    for name, fn in (
        ("Sobol Owen", test_sobol_owen),
        ("group/split", test_group_id_and_split),
        ("OOD constraints", test_ood_constraints),
        ("case schema", test_case_schema),
        ("CLI refuse", test_cli_refuse_without_confirm),
        ("observation chain", test_observation_spec),
    ):
        print(f"[{name}]")
        try:
            fn(s)
        except Exception as exc:
            s.n_fail += 1
            s.failures.append((name, str(exc)))
            print(f"  FAIL  {name}  exception: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{s.n_ok} passed, {s.n_fail} failed")
    return {"n_ok": s.n_ok, "n_fail" : s.n_fail, "failures": s.failures}


if __name__ == "__main__":
    out = run_all()
    sys.exit(0 if out["n_fail"] == 0 else 1)
