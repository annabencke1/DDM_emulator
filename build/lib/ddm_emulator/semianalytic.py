"""
Standalone closed-form evaluator for the decaying-dark-matter halo density
emulator of Morales, Bencke & Kamionkowski (2026).

    rho_DDM(r; tau, v_kick) = rho_null(r) * exp[ ybar(r) + sum_k c_k(tau,v) e_k(r) ]

with the four amplitudes given by low-order polynomials
    c_k(tau, v_kick) = sum_{i+j<=degree} a^(k)_ij * tau^i * v_kick^j.

Everything (the fixed radial functions ybar(r), e_k(r), rho_null(r) and the
polynomial coefficients) is read from the semianalytic.npz shipped with the
release. The only dependency is numpy.

Units: tau in dynamical times, v_kick in units of the virial velocity, r in kpc.
"""
from __future__ import annotations

import warnings

import numpy as np


def _outside(value, domain, tol=1e-6):
    """True if value is outside [lo, hi], allowing a small tolerance at the edges
    so that exact-boundary inputs (e.g. v_kick=0.1) don't trip on round-off."""
    lo, hi = domain
    pad = tol * max(1.0, hi - lo)
    return value < lo - pad or value > hi + pad


class DDMDensity:
    """Evaluate rho_DDM(r; tau, v_kick) from the semi-analytic tables."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.r_grid = np.asarray(d["r_grid"], float)          # (Nr,)
        self.ybar = np.asarray(d["ybar"], float)              # (Nr,)
        self.e_modes = np.asarray(d["e_modes"], float)        # (K, Nr)
        self.rho_null = np.asarray(d["rho_null"], float)      # (Nr,)
        self.coeffs = np.asarray(d["poly_coeffs"], float)     # (n_terms, K)
        self.powers = np.asarray(d["poly_powers"], int)       # (n_terms, 2): (i_tau, j_v)
        self.tau_domain = tuple(round(float(x), 6) for x in d["tau_domain"])
        self.v_domain = tuple(round(float(x), 6) for x in d["v_domain"])
        self.r_domain = (round(float(self.r_grid.min()), 4),
                         round(float(self.r_grid.max()), 4))

    # -- pieces --------------------------------------------------------------
    def coefficients(self, tau: float, v_kick: float) -> np.ndarray:
        """The four PCA amplitudes c_k at (tau, v_kick)."""
        mono = np.array([tau ** int(i) * v_kick ** int(j)
                         for i, j in self.powers])
        return mono @ self.coeffs

    def log_modification(self, r, tau: float, v_kick: float) -> np.ndarray:
        """ln(rho_DDM/rho_null) at radius r."""
        c = self.coefficients(tau, v_kick)
        y_grid = self.ybar + c @ self.e_modes
        return np.interp(np.log(np.asarray(r, float)),
                         np.log(self.r_grid), y_grid)

    def null_density(self, r) -> np.ndarray:
        """The evolved non-decaying control density rho_null(r)."""
        return np.exp(np.interp(np.log(np.asarray(r, float)),
                                np.log(self.r_grid), np.log(self.rho_null)))

    # -- main entry point ----------------------------------------------------
    def __call__(self, r, tau: float, v_kick: float, check_domain: bool = True):
        """
        rho_DDM(r; tau, v_kick) in Msun/kpc^3.

        tau     : decay lifetime in dynamical times
        v_kick  : kick velocity in units of the virial velocity
        r       : radius (scalar or array) in kpc
        """
        if check_domain:
            self._warn_domain(tau, v_kick, r)
        return self.null_density(r) * np.exp(self.log_modification(r, tau, v_kick))

    def _warn_domain(self, tau, v_kick, r):
        r = np.atleast_1d(np.asarray(r, float))
        if _outside(tau, self.tau_domain):
            warnings.warn(f"tau={tau} outside validated range {self.tau_domain}.")
        if _outside(v_kick, self.v_domain):
            warnings.warn(f"v_kick={v_kick} outside validated range {self.v_domain}.")
        if _outside(r.min(), self.r_domain) or _outside(r.max(), self.r_domain):
            warnings.warn(f"r outside validated range {self.r_domain} kpc.")


if __name__ == "__main__":
    import sys
    emu = DDMDensity(sys.argv[1] if len(sys.argv) > 1 else "models/semianalytic.npz")
    r = np.geomspace(*emu.r_domain, 8)
    print("rho_DDM(tau=2, v_kick=0.1):")
    for ri, rhoi in zip(r, emu(r, tau=2.0, v_kick=0.1)):
        print(f"  r={ri:8.2f} kpc   rho={rhoi:.4e} Msun/kpc^3")