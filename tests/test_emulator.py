"""
Smoke + consistency tests for the DDM halo emulator.

Run as a script (prints PASS/FAIL):
    python tests/test_emulator.py
or under pytest:
    pytest tests/

Requires the model files in ../models/ (semianalytic.npz, ddm_emulator.json).
Tests that need a file that isn't present are skipped with a clear message.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ddm_emulator import DDMDensity, GPEmulator

MODELS = Path(__file__).resolve().parent.parent / "models"
NPZ = MODELS / "semianalytic.npz"
GPJSON = MODELS / "ddm_emulator.json"


def _load():
    if not NPZ.exists():
        raise FileNotFoundError(f"missing {NPZ} -- run export_semianalytic.py first")
    emu = DDMDensity(NPZ)
    gp = GPEmulator(GPJSON, NPZ) if GPJSON.exists() else None
    return emu, gp


def test_positive_and_finite():
    emu, _ = _load()
    r = np.geomspace(*emu.r_domain, 60)
    for tau in np.linspace(*emu.tau_domain, 5):
        for v in np.linspace(*emu.v_domain, 5):
            rho = emu(r, tau, v, check_domain=False)
            assert np.all(np.isfinite(rho)) and np.all(rho > 0), (tau, v)


def test_scalar_and_array_r():
    emu, _ = _load()
    lo, hi = emu.r_domain
    assert np.isscalar(float(emu(np.array([10.0]), 2.0, 0.3)[0]))
    out = emu(np.geomspace(lo, hi, 12), 2.0, 0.3)
    assert out.shape == (12,)


def test_closed_form_matches_gp():
    emu, gp = _load()
    if gp is None:
        print("SKIP closed-form-vs-GP: ddm_emulator.json not present")
        return
    rng = np.random.default_rng(0)
    r = np.geomspace(*emu.r_domain, 50)
    errs = []
    for _ in range(100):
        tau = rng.uniform(*emu.tau_domain)
        v = rng.uniform(*emu.v_domain)
        a = emu(r, tau, v, check_domain=False)
        b = gp(r, tau, v, check_domain=False)
        errs.append(np.median(np.abs(a / b - 1.0)))
    med = float(np.median(errs))
    print(f"closed-form vs GP: median |Δρ/ρ| = {med*100:.3f}%")
    assert med < 0.01, f"closed form and GP disagree by {med*100:.2f}% (>1%)"


def test_domain_edges_no_false_warning():
    emu, _ = _load()
    r = np.geomspace(*emu.r_domain, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for v in emu.v_domain:
            emu(r, emu.tau_domain[0], v)
            emu(r, emu.tau_domain[1], v)


def test_out_of_domain_warns():
    emu, _ = _load()
    r = np.geomspace(*emu.r_domain, 20)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        emu(r, emu.tau_domain[1], emu.v_domain[1] + 0.1)
        assert any("v_kick" in str(x.message) for x in w)


def test_ratio_reasonable():
    emu, _ = _load()
    r = np.geomspace(*emu.r_domain, 60)
    ratio = emu(r, 2.0, 0.3, check_domain=False) / emu.null_density(r)
    # outer halo should be close to the control; nothing should blow up
    assert 0.7 < ratio[-1] < 1.3, ratio[-1]
    assert np.all(ratio > 0) and np.all(ratio < 5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")