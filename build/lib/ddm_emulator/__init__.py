"""Decaying-dark-matter halo density emulator.

Standalone closed-form evaluator (numpy only):

    from ddm_emulator import DDMDensity
    emu = DDMDensity("models/semianalytic.npz")
    rho = emu(r, tau=2.0, v_kick=0.1)
"""
from .semianalytic import DDMDensity

__all__ = ["DDMDensity"]
__version__ = "0.1.0"