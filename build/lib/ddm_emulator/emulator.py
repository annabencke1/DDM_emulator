"""
Full PCA + Gaussian-process emulator for the decaying-dark-matter halo density
of Bencke, Kamionkowski, Peter & Morales (2026).

    rho_DDM(r; tau, v_kick) = rho_null(r) * exp[ ybar(r) + sum_k c_k(tau,v) e_k(r) ]

This is the *exact* trained model: the four amplitudes c_k come from the trained
Gaussian processes, rather than the polynomial approximation in semianalytic.py.
Use this if you want the emulator exactly as fitted; use DDMDensity
(semianalytic.py) if you want a lighter, dependency-free closed form.

Inference is pure numpy -- the trained GP alpha-vectors are stored in the model
JSON, so no scikit-learn is needed to evaluate. scipy is imported only if the
Matern smoothness nu is not one of {0.5, 1.5, 2.5}.

Units: tau in dynamical times, v_kick in units of the virial velocity, r in kpc.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np


def _outside(value, domain, tol=1e-6):
    """True if value is outside [lo, hi], allowing a small tolerance at the edges
    so that exact-boundary inputs (e.g. v_kick=0.1) don't trip on round-off."""
    lo, hi = domain
    pad = tol * max(1.0, hi - lo)
    return value < lo - pad or value > hi + pad


def matern_kernel(x1, x2, length_scale, nu, amplitude):
    """Anisotropic Matern covariance between input points x1 and x2."""
    x1 = np.atleast_2d(np.asarray(x1, float))
    x2 = np.atleast_2d(np.asarray(x2, float))
    length_scale = np.asarray(length_scale, float)
    diff = (x1[:, None, :] - x2[None, :, :]) / length_scale[None, None, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    if nu == 0.5:
        base = np.exp(-r)
    elif nu == 1.5:
        c = np.sqrt(3.0) * r
        base = (1.0 + c) * np.exp(-c)
    elif nu == 2.5:
        c = np.sqrt(5.0) * r
        base = (1.0 + c + c ** 2 / 3.0) * np.exp(-c)
    else:
        from scipy.special import gamma, kv  # only needed for exotic nu
        scaled = np.sqrt(2.0 * nu) * np.maximum(r, 1e-14)
        base = (2.0 ** (1.0 - nu) / gamma(nu)) * scaled ** nu * kv(nu, scaled)
        base[r == 0.0] = 1.0
    return amplitude * base


class GPEmulator:
    """Exact trained PCA + Gaussian-process emulator for rho_DDM(r; tau, v_kick).

    Parameters
    ----------
    model_json : path to the trained emulator JSON (PCA + GP state).
    npz_path   : path to semianalytic.npz -- supplies the fixed control profile
                 rho_null(r) and the validated (tau, v, r) domain, keeping this
                 path numerically identical to the closed form.
    """

    def __init__(self, model_json, npz_path):
        payload = json.loads(Path(model_json).read_text(encoding="utf-8"))
        self.payload = payload
        self.radial_grid = np.asarray(payload["radial_grid_kpc"], float)
        self.weights = np.asarray(payload["radial_feature_weights"], float)
        self.pca_mean = np.asarray(
            payload["pca"]["mean_weighted_log_suppression"], float)
        self.pca_components = np.asarray(payload["pca"]["components"], float)
        self.nu = float(payload["config"]["matern_nu"])
        self.input_transform = str(payload.get("input_transform", "log"))

        s = payload["input_scaling"]
        self.log_tau_center = float(s["log_tau_center"])
        self.log_tau_scale = float(s["log_tau_scale"])
        self.log_v_center = float(s["log_v_center"])
        self.log_v_scale = float(s["log_v_scale"])

        self.gps = []
        for gp in payload["gaussian_processes"]:
            theta = np.asarray(gp["kernel_theta"], float)
            self.gps.append({
                "amplitude": float(np.exp(theta[0])),
                "length_scale": np.exp(theta[1:3]),
                "x_train": np.asarray(gp["x_train"], float),
                "alpha": np.asarray(gp["alpha_vector"], float),
                "y_train_mean": float(gp["y_train_mean"]),
                "y_train_std": float(gp["y_train_std"]),
            })

        # Fixed control profile + validated domain, shared with the closed form.
        d = np.load(npz_path)
        self._null_r = np.asarray(d["r_grid"], float)
        self._null_rho = np.asarray(d["rho_null"], float)
        self.tau_domain = tuple(round(float(x), 6) for x in d["tau_domain"])
        self.v_domain = tuple(round(float(x), 6) for x in d["v_domain"])
        self.r_domain = (round(float(self._null_r.min()), 4),
                         round(float(self._null_r.max()), 4))

    # -- pieces --------------------------------------------------------------
    def _scale_inputs(self, tau, v):
        tx = np.log if self.input_transform == "log" else (lambda z: z)
        return np.array([[
            (tx(float(tau)) - self.log_tau_center) / self.log_tau_scale,
            (tx(float(v)) - self.log_v_center) / self.log_v_scale,
        ]], float)

    def coefficients(self, tau, v_kick):
        """The four PCA amplitudes c_k from the trained GPs."""
        x = self._scale_inputs(tau, v_kick)
        out = []
        for gp in self.gps:
            k = matern_kernel(x, gp["x_train"], gp["length_scale"],
                              self.nu, gp["amplitude"])[0]
            out.append((k @ gp["alpha"]) * gp["y_train_std"] + gp["y_train_mean"])
        return np.asarray(out, float)

    def log_modification(self, r, tau, v_kick):
        """ln(rho_DDM/rho_null) at radius r."""
        weighted = self.pca_mean + self.coefficients(tau, v_kick) @ self.pca_components
        y_grid = weighted / np.sqrt(self.weights)
        return np.interp(np.log(np.asarray(r, float)),
                         np.log(self.radial_grid), y_grid)

    def null_density(self, r):
        """The evolved non-decaying control density rho_null(r)."""
        return np.exp(np.interp(np.log(np.asarray(r, float)),
                                np.log(self._null_r), np.log(self._null_rho)))

    # -- main entry point ----------------------------------------------------
    def __call__(self, r, tau, v_kick, check_domain=True):
        """rho_DDM(r; tau, v_kick) in Msun/kpc^3, from the exact trained GP."""
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
    gp = GPEmulator(sys.argv[1], sys.argv[2])  # model.json  semianalytic.npz
    r = np.geomspace(*gp.r_domain, 8)
    print("GP rho_DDM(tau=2, v_kick=0.3):")
    for ri, rhoi in zip(r, gp(r, tau=2.0, v_kick=0.3)):
        print(f"  r={ri:8.2f} kpc   rho={rhoi:.4e} Msun/kpc^3")