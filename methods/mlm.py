"""M4 — Maximum-Likelihood-style void estimation (after Kumar et al. 2014).

The central void is modelled as an ellipsoidal exclusion zone: the ellipsoid that
contains as few near-origin Fry vectors as possible while being as large as
possible (a penalised objective), solved by differential evolution.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from ..schema import GrainTable
from .fry import fry_vectors
from .ellipsoid import StrainEstimate, ellipsoid_to_strain

__all__ = ["estimate_strain_mlm"]


def _euler_to_R(a, b, c) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a); cb, sb = np.cos(b), np.sin(b); cc, sc = np.cos(c), np.sin(c)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def estimate_strain_mlm(table: GrainTable, max_pairs: int = 150_000, weight_inside: float = 50.0,
                        maxiter: int = 60, seed: int = 0, rng=None) -> StrainEstimate:
    """MLM-style strain estimate via penalised void fitting (global optimiser)."""
    vecs = fry_vectors(table.centroids, max_pairs=max_pairs, rng=rng)
    if len(vecs) < 30:
        raise ValueError("too few grains for MLM analysis")
    norms = np.linalg.norm(vecs, axis=1)
    scale = np.median(norms)
    near = vecs[norms <= 3 * scale]
    lo, hi = 0.1 * scale, 2.5 * scale

    def objective(params):
        a, b, c, e1, e2, e3 = params
        R = _euler_to_R(e1, e2, e3)
        Ainv = R @ np.diag([1 / a**2, 1 / b**2, 1 / c**2]) @ R.T
        q = np.einsum("ij,nj,ni->n", Ainv, near, near)
        return weight_inside * np.mean(q < 1.0) - np.log(a * b * c)

    bounds = [(lo, hi)] * 3 + [(0, np.pi)] * 3
    result = differential_evolution(objective, bounds, maxiter=maxiter, tol=1e-3,
                                    seed=seed, polish=True, updating="deferred")
    a, b, c, e1, e2, e3 = result.x
    R = _euler_to_R(e1, e2, e3)
    semi = np.array([a, b, c])
    order = np.argsort(semi)[::-1]
    return ellipsoid_to_strain(semi[order], method="M4_mlm", axes=R[:, order],
                               extra={"objective": float(result.fun)})
