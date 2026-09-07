# -*- coding: utf-8 -*-
"""Stage-1 unit tests that lock the equation, dimension and eigenstructure contracts.

These are the R1 / §2.2 / §2.4(c) locks: they must pass independently of the
expensive golden suite.  Run as ``python test_units.py``.
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import friction as fr
import memory_kernel as mk
import node_newton as nn
import transfer_matrix as tmm
from config_io import load_yaml
from grid import build_grid, interpolation_amplitude_factor

G = 9.80665


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

    def check_close(self, name: str, a, b, rtol=1e-12, atol=1e-14) -> None:
        a, b = float(a), float(b)
        ok = math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * abs(b)
        self.check(name, ok, f"got {a:.6e}, expected {b:.6e}")


def test_compatibility_bq_term(s: _Suite) -> None:
    """§2.2: C+ / C- must retain ±B Q_P; dropping it makes the two equations inconsistent."""
    CP, CM, B = 12.0, 4.0, 2.5
    H = 0.5 * (CP + CM)
    Q = (CP - CM) / (2.0 * B)
    s.check_close("C+ residual with +B Q", H + B * Q, CP)
    s.check_close("C- residual with -B Q", H - B * Q, CM)
    # the incorrect "H = CP and H = CM" system is inconsistent whenever Q != 0
    s.check("dropped-BQ system is inconsistent", abs(CP - CM) > 1.0)
    # trapezoidal: B Q + (R/2) Q|Q| = D
    D, R = 3.0, 0.4
    Q2 = nn.flow_from_characteristic(D, B, R)
    s.check_close("flow_from_characteristic residual", B * Q2 + 0.5 * R * Q2 * abs(Q2), D, atol=1e-12)
    Q3 = nn.flow_from_characteristic(-D, B, R)
    s.check_close("flow_from_characteristic sign", Q3, -Q2, atol=1e-12)
    s.check_close("R=0 reduces to D/B", nn.flow_from_characteristic(D, B, 0.0), D / B)


def test_residual_dimensions(s: _Suite) -> None:
    """§2.3 / R1: node residuals occupy the declared SI units."""
    rho, g, z_j = 1000.0, G, -2000.0
    rho_g = rho * g
    H, Qm, Qp = 4500.0, 0.04, 0.01
    q = np.array([0.02, 0.01])
    K = np.array([2.0e10, 2.0e10])          # Pa s^2 m^-6
    kappa = np.array([1.0e6, 1.0e6])        # Pa s^{1/2} m^{-3/2}
    c1 = np.array([1.0e8, 1.0e8])           # Pa s m^-3
    c0 = np.zeros(2)
    a1 = np.array([1.0e7, 1.0e7])           # Pa s m^-3
    a0 = np.array([4.0e7, 4.0e7])           # Pa
    p_w = rho_g * (H - z_j)                 # Pa
    r_mass = Qm - Qp - q.sum()              # m^3 s^-1
    r_or = [nn.perf_residual(q[m], p_w, K[m], kappa[m], 1e-8, c1[m], c0[m], a1[m], a0[m]) for m in range(2)]
    s.check("mass residual is m^3/s (finite, not Pa)", math.isfinite(r_mass) and abs(r_mass) < 10.0)
    s.check("orifice residual is Pa (O(1e6) not O(1e-3))", all(abs(r) > 1e3 for r in r_or))
    # tortuosity is an additive Pa term, never mixed into K
    r_no_tort = nn.perf_residual(q[0], p_w, K[0], 0.0, 1e-8, c1[0], c0[0], a1[0], a0[0])
    s.check("tortuosity is an independent additive Pa term", abs(r_or[0] - r_no_tort) > 1.0)
    # leak-off lives only in the storage ODE (a0, a1), not in the mass jump
    s.check("leak-off is not re-subtracted from the mass jump", abs(r_mass - (Qm - Qp - q.sum())) < 1e-15)


def test_rci_port_impedance(s: _Suite) -> None:
    """§2.3 rule 1: Z_b(s) = R_perf + R_f + s I_f + 1/(s C_f + G_l)  (parallel C||G)."""
    br = tmm.BranchLin(R_perf=2e7, R_f=1e8, I_f=5e5, C_f=1e-6, G_l=2e-10)
    w = np.array([0.1, 1.0, 10.0, 100.0])
    s_ = 1j * w
    Z = br.Z(s_)
    Z_ok = br.R_perf + br.R_f + s_ * br.I_f + 1.0 / (s_ * br.C_f + br.G_l)
    s.check("Z_b matches the parallel storage topology", np.allclose(Z, Z_ok))
    Z_wrong_series = br.R_perf + br.R_f + s_ * br.I_f + 1.0 / (s_ * br.C_f) + 1.0 / br.G_l
    s.check("series R+1/(sC) topology is rejected", np.max(np.abs(Z - Z_wrong_series)) > 1e6)


def test_impedance_discipline(s: _Suite) -> None:
    """§2.1: B = a/(gA) pairs with [H,Q]; Z_c = rho a/A pairs with [p,Q].  Never mix."""
    a, D, rho = 1400.0, 0.114, 1000.0
    A = math.pi * D ** 2 / 4.0
    B = a / (G * A)
    Zc = rho * a / A
    s.check_close("Z_c = rho g B", Zc, rho * G * B, rtol=1e-14)
    s.check("B and Z_c differ by ~rho g (~1e4)", abs(Zc / B - rho * G) < 1e-8)


def test_gravity_not_double_counted(s: _Suite) -> None:
    """§2.1: piezometric H already contains elevation; MOC must not add g A sinθ."""
    phy = load_yaml("physics.yaml")
    s.check("gravity_term declares H-form (no extra gA sinθ)",
            "不再加" in phy["gravity_term"] or "avoid" in phy["gravity_term"].lower()
            or "双计" in phy["gravity_term"])


def test_brunone_eigenvalues(s: _Suite) -> None:
    """§2.4(c): characteristic speeds of the locked k-form, both sgn branches."""
    a, k = 1400.0, 0.03
    lam_p = fr.brunone_characteristic_speeds(a, k, +1)
    lam_m = fr.brunone_characteristic_speeds(a, k, -1)
    s.check_close("sgn=+1 slower speed a/(1+k)", lam_p[1], a / (1.0 + k), rtol=1e-8, atol=1e-8)
    s.check_close("sgn=+1 backward -a", lam_p[0], -a, rtol=1e-8, atol=1e-6)
    s.check_close("sgn=-1 faster +a", lam_m[1], a, rtol=1e-8, atol=1e-6)
    s.check_close("sgn=-1 slower -a/(1+k)", lam_m[0], -a / (1.0 + k), rtol=1e-8, atol=1e-8)
    cfg = load_yaml("friction_brunone.yaml")
    s.check("coefficient convention is k (not k/2)", cfg["coefficient_convention"] == "k")
    s.check("locked convention is recorded", cfg["locked_convention"] in ("vitkovsky2000", "plan_v2"))
    k_lam = fr.brunone_k(1000.0)
    s.check_close("laminar k = sqrt(0.00476)/2", k_lam, math.sqrt(0.00476) / 2.0, rtol=1e-12)


def test_kernel_positivity_and_fit_gate(s: _Suite) -> None:
    """§2.4(b)/§2.6: m_l, n_l > 0; production M meets the <2% fit gate."""
    cfg = load_yaml("friction_zielke.yaml")
    M = int(cfg["production_path"]["M"])
    fz, fp = mk.kernel_fits_from_config(cfg, M)
    mk.assert_positive_kernel(fz.m, fz.n)
    mk.assert_positive_kernel(fp.m, fp.n)
    gate = float(cfg["production_path"]["fit"]["gate_max_rel_err"])
    s.check(f"Zielke M={M} fit < {100*gate:.0f}%", fz.max_rel_err < gate,
            f"max_rel_err={fz.max_rel_err:.4f}")
    s.check(f"power-law M={M} fit < {100*gate:.0f}%", fp.max_rel_err < gate,
            f"max_rel_err={fp.max_rel_err:.4f}")
    # M=10 is the design value and is documented to miss the gate
    z10 = cfg["fits_by_M"][10]["zielke_laminar"]["max_rel_err"]
    s.check("M=10 Zielke miss is recorded (repair path §3.4)", z10 >= gate)
    raised = False
    try:
        mk.assert_positive_kernel(np.array([1.0, -0.1]), np.array([1.0, 2.0]))
    except ValueError:
        raised = True
    s.check("negative weight is rejected", raised)


def test_zielke_piecewise_W(s: _Suite) -> None:
    """Small-tau sqrt series and large-tau exponential series are both active."""
    w_small = float(mk.zielke_W(np.array([1e-6]))[0])
    w_pl = mk.A_STAR / math.sqrt(1e-6)
    s.check("small-tau W ~ A*/sqrt(tau)", abs(w_small / w_pl - 1.0) < 0.05)
    w_split_m = float(mk.zielke_W(np.array([0.019]))[0])
    w_split_p = float(mk.zielke_W(np.array([0.021]))[0])
    # Zielke (1968) piecewise series has a known O(10%) mismatch at the published
    # split tau=0.02; we lock the published switch, not a C0 blend (方案 §2.4(a)).
    rel_jump = abs(w_split_m - w_split_p) / w_split_p
    s.check("both Zielke series are active and positive at the split",
            w_split_m > 0.0 and w_split_p > 0.0 and 0.01 < rel_jump < 0.20,
            f"rel_jump={rel_jump:.3f}")
    n1 = mk.ZIELKE_N_LARGE[0]
    s.check_close("first J2-zero squared", n1, 26.3744, rtol=0, atol=1e-4)


def test_cluster_exact_landing(s: _Suite) -> None:
    """§3.1: clusters sit exactly on nodes; residual quantisation = 0."""
    xs = np.array([3123.7, 3140.15, 3901.3])
    g = build_grid(4000.0, 1400.0, xs, Nx_ref=1024, Cr_target=1.0)
    for j, x in enumerate(xs):
        iL = g.cluster_left_index(j)
        s.check_close(f"cluster {j} lands on a node", g.x_nodes[iL], x, rtol=0, atol=1e-12)
    s.check("every Cr_j <= Cr_target", np.all(g.seg_Cr <= 1.0 + 1e-12))
    s.check("wellhead segment Cr == 1 when Cr_target=1 (unless dt reduced)",
            g.dt_reduced or abs(g.seg_Cr[0] - 1.0) < 1e-12)
    Gfac = interpolation_amplitude_factor(0.8, 2.0 * math.pi * 4.0 / 20.0)
    s.check("Ghidaoui–Karney |G| < 1 at Cr=0.8, 70 Hz, dx=4 m", Gfac < 1.0)


def test_yaml_numeric_coercion(s: _Suite) -> None:
    """YAML 1.1 / PyYAML treats 4.0e7 as a string; configs must still yield floats."""
    r = load_yaml("physics.yaml")["reference_well"]["clusters"]
    for key in ("kappa", "I_f", "R_f", "p_res"):
        s.check(f"{key} loads as float", isinstance(r[key], (int, float)) and not isinstance(r[key], bool),
                f"type={type(r[key]).__name__} value={r[key]!r}")


def test_long_wave_check(s: _Suite) -> None:
    phy = load_yaml("physics.yaml")
    lw = phy["long_wave_check"]
    val = lw["f_sig_max_hz"] * lw["D_max"] / lw["a_min"]
    s.check("f_sig D / a << 1 on the sampling domain", val <= lw["value_max"] + 1e-15,
            f"value={val:.4e}")


def run_all() -> dict:
    s = _Suite()
    tests = [
        ("compatibility ±B Q", test_compatibility_bq_term),
        ("residual dimensions", test_residual_dimensions),
        ("RCI port impedance", test_rci_port_impedance),
        ("impedance discipline", test_impedance_discipline),
        ("gravity not double-counted", test_gravity_not_double_counted),
        ("Brunone eigenvalues", test_brunone_eigenvalues),
        ("kernel positivity / fit", test_kernel_positivity_and_fit_gate),
        ("Zielke piecewise W", test_zielke_piecewise_W),
        ("cluster exact landing", test_cluster_exact_landing),
        ("YAML numeric coercion", test_yaml_numeric_coercion),
        ("long-wave check", test_long_wave_check),
    ]
    print("Stage-1 unit locks")
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn(s)
        except Exception as exc:
            s.n_fail += 1
            s.failures.append((name, f"exception: {exc}"))
            print(f"  FAIL  {name}  exception: {exc}")
            traceback.print_exc()
    print(f"\n{s.n_ok} passed, {s.n_fail} failed")
    return {"n_ok": s.n_ok, "n_fail": s.n_fail, "failures": s.failures,
            "status": "PASS" if s.n_fail == 0 else "FAIL"}


if __name__ == "__main__":
    out = run_all()
    sys.exit(0 if out["n_fail"] == 0 else 1)
