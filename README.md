# DDM halo density emulator

A fast, closed-form emulator for the relaxed density profile of a self-gravitating
halo whose dark matter decays, from **Bencke, Peter, Kamionkowski, & Morales (2026)**,
*Dynamics of self-gravitating halos with decaying dark matter*.

Given a decay lifetime `τ` and a kick velocity `v_kick`, the emulator returns the
spherically-averaged density profile `ρ_DDM(r)` of the relaxed halo, reconstructed as

```
ρ_DDM(r; τ, v_kick) = ρ_null(r) · exp[ ȳ(r) + Σ_k c_k(τ, v_kick) e_k(r) ]
```

where `ρ_null(r)` is the evolved non-decaying control profile, `ȳ(r)` and the four
radial modes `e_k(r)` are fixed curves, and the amplitudes `c_k(τ, v_kick)` are
degree-3 polynomials. This closed form reproduces the full PCA + Gaussian-process
emulator to a **median ~0.8%** in density; the emulator itself matches the
simulations to **~1.5%**.

## Install

```bash
pip install .            # numpy + matplotlib (for the example notebook)
```

The emulators themselves run on numpy alone at inference — if you only need
`DDMDensity`/`GPEmulator` and not the notebook, copying `ddm_emulator/` and the
files in `models/` into your project is enough.

## Quickstart

Two evaluators are provided. The **closed form** (`DDMDensity`) uses the polynomial
fit; the **exact trained model** (`GPEmulator`) uses the Gaussian processes. They
share the same rho_null and agree to ~0.8%. `tau` is in dynamical times; `v_kick`
is in units of the virial velocity.

```python
import numpy as np
from ddm_emulator import DDMDensity, GPEmulator

r = np.geomspace(1.0, 190.0, 50)  # radius in kpc
# closed-form polynomial evaluator (numpy only)
emu = DDMDensity("models/semianalytic.npz")
rho = emu(r, tau=2.0, v_kick=0.2)  # density in Msun/kpc^3
# exact trained Gaussian-process evaluator
gp = GPEmulator("models/ddm_emulator.json", "models/semianalytic.npz")
rho_gp = gp(r, tau=2.0, v_kick=0.2)
# optional pieces
coeffs = emu.coefficients(tau=2.0, v_kick=0.2)  # the four PCA amplitudes
rho_null = emu.null_density(r)  # evolved control profile
```

## Domain of validity

The emulator interpolates *within* the simulated grid. Outside it, results are
extrapolation (a warning is raised).

| quantity | range |
|---|---|
| decay lifetime `τ` | 0.5 – 10 dynamical times |
| kick velocity `v_kick` | 0.1 – 0.5 `v_virial` |
| radius `r` | ~0.55 – 190 kpc |

Below `v_kick ≈ 0.1 v_virial` the kick energy is negligible against the local
binding energy and the modification is consistent with zero (ρ_DDM → ρ_null); the
emulator is not fit there by design.

All profiles are for an isolated isotropic Hernquist halo with
`M = 1e12 M_sun`, `r_s = 23 kpc`, evolved to `200 t_dyn ≈ 10 Gyr`.
`t_dyn = 52.01 Myr`, `v_virial = 432 km/s`.

## What's in the box

```
ddm_emulator/
  semianalytic.py   closed-form polynomial   (DDMDensity)
  emulator.py       exact trained PCA + GP   (GPEmulator)
models/
  semianalytic.npz  radial functions ȳ, e_k, ρ_null + polynomial coefficients
  semianalytic.json human-readable version of the same
  ddm_emulator.json trained PCA + GP state
examples/
  quickstart.ipynb  load, predict, plot the ratio sweep
```

## Citation

If you use this emulator, please cite the paper and the archived release:

```bibtex
@article{ddm_halos_2026,
  author  = {Bencke, Anna and Kamionkowski, Marc and Peter, Annika and Morales, Juniper},
  title   = {Dynamics of self-gravitating halos with decaying dark matter},
  year    = {2026},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
}
```

## License

MIT.