"""Deformation-grid design (factorial sweep).

The factorial spans Rs × k × Wk × orientations × realisations.  Random
orientations are drawn from the identity-keyed seed policy so the grid is
reproducible and order-independent.  ``build_deformation_grid`` realises Wk by
solving for the shear ``gamma`` that yields the target Wk at each (Rs, k), the
sequential design of an earlier phase, and is kept for the tests; the benchmark
uses the steady-flow deformations of ``runner.build_configs``.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import brentq

from .deformation.tensors import deformation_from_targets, strain_ellipsoid, vorticity_number

__all__ = ["build_deformation_grid", "gamma_for_Wk", "DEFAULT_GRID"]

#: The factorial of the paper: 5 Rs x 5 k x 5 Wk, one orientation, eight synthetic rocks.
DEFAULT_GRID = {
    "Rs": [1.2, 1.5, 2.0, 3.0, 5.0],
    "k": [0.2, 0.5, 1.0, 2.0, 5.0],
    "Wk": [0.0, 0.50, 0.71, 0.87, 1.0],
    "n_orient": 1,
    "n_realisations": 8,
}


def gamma_for_Wk(Wk: float, Rs: float, k: float) -> float:
    """Shear strain ``gamma`` giving kinematic vorticity ``Wk`` at (Rs, k).

    Wk = 0 → pure shear (gamma=0); Wk → 1 → simple-shear-dominated (gamma → ∞,
    capped).  Solved from ``vorticity_number(gamma, Rs, k)``.
    """
    if Wk <= 0.0:
        return 0.0
    if Wk >= 1.0:
        return 1e6                          # simple-shear end-member (gamma -> inf)
    try:
        return float(brentq(lambda g: vorticity_number(g, Rs, k) - Wk, 1e-9, 1e6))
    except Exception:
        return 0.0


def build_deformation_grid(Rs=None, k=None, Wk=None, n_orient=1, n_realisations=1,
                           master_seed: int | None = None):
    """Build a list of (label, F, StrainEllipsoid) over the factorial grid.

    The reported StrainEllipsoid is the *true* finite strain from F (which, for
    Wk>0, differs from the nominal Rs/k).  Random orientations are seeded per
    (combo, realisation, orientation) via the seed policy.
    """
    from .repro import child_rng, DEFAULT_MASTER_SEED
    seed = DEFAULT_MASTER_SEED if master_seed is None else master_seed
    Rs = Rs or DEFAULT_GRID["Rs"]; k = k or DEFAULT_GRID["k"]; Wk = Wk or DEFAULT_GRID["Wk"]

    out = []
    for ci, (rs, kk, wk) in enumerate(itertools.product(Rs, k, Wk)):
        gamma = gamma_for_Wk(wk, rs, kk)
        for realis in range(n_realisations):
            for o in range(n_orient):
                orng = child_rng(seed, ci, realis, o) if n_orient > 1 else None
                F = deformation_from_targets(Rs=rs, k=kk, gamma=gamma, rng=orng)
                label = f"Rs{rs}_k{kk}_Wk{wk}_r{realis}_o{o}"
                out.append((label, F, strain_ellipsoid(F)))
    return out
