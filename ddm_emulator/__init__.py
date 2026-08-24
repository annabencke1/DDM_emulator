"""Decaying-dark-matter halo density emulator.

Two evaluators, both numpy-only at inference:

    from ddm_emulator import DDMDensity          # closed-form polynomial
    emu = DDMDensity("models/semianalytic.npz")
    rho = emu(r, tau=2.0, v_kick=0.1)

    from ddm_emulator import GPEmulator           # exact trained Gaussian process
    gp = GPEmulator("models/ddm_emulator.json", "models/semianalytic.npz")
    rho = gp(r, tau=2.0, v_kick=0.1)
"""
from .emulator import GPEmulator
from .semianalytic import DDMDensity

__all__ = ["DDMDensity", "GPEmulator"]
__version__ = "0.1.0"