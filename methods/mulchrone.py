"""M17 / M18 — Mulchrone's Delaunay-neighbour estimators, as published, extended to 3-D.

Both use Delaunay-neighbour centre-to-centre vectors normalised by the summed equivalent
radii (Erslev 1988), with edges touching convex-hull vertices removed ("rogue values may be
eliminated by subtracting points which form part of the convex hull", Mulchrone 2003 p. 691).

M18 — Mulchrone (2003).  Linear least squares through all neighbour vectors of the polar
ellipse equation, ``1/d² = C − A cos2θ − B sin2θ`` (Eqs 8–10); in 3-D ``1/d² = uᵀ A u`` with
``A`` symmetric (6 unknowns), semi-axes ``eig(A)^{-1/2}``.  (The paper's M3 instead fits the
closest vector per direction — the void shell — so it is not Mulchrone's estimator.)
Mulchrone's selection factor (keep pairs with D/(rₐ+r_b) ≤ sf ≈ 1.0–1.3, radii along the pair
line; after Erslev & Ge 1990) is a contact criterion for PACKED aggregates; dispersed spinel
has no touching pairs, so the regression here uses all hull-trimmed neighbours.  On a
separated hard-core pattern this dilutes the strain signal (Rs 2 → ≈1.3; axes still correct).

M17 — Mulchrone (2013) "fitting the void".  Asymmetric least squares: minimise
``Σ wₖ (f(uₖ) − ρₖ)²`` with ``f(u) = (uᵀAu)^{-1/2}`` the trial-ellipsoid radius and
``wₖ = z`` when the point lies inside the trial ellipsoid, else 1, with the "asymmetry
coefficient" ``z = 1000`` (pp. 25–29), so the ellipsoid is pushed against the inner edge of
the neighbour cloud.  Mulchrone notes it "will provide lower bounds on the finite strain
ratio" (p. 30) and assumes an anticlustered starting pattern.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, ellipsoid_to_strain

__all__ = ["delaunay_neighbour_vectors", "estimate_strain_dtm_ls", "estimate_strain_void_fit"]

_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def delaunay_neighbour_vectors(table: GrainTable, normalise: bool = True) -> np.ndarray:
    """Symmetric (±) Delaunay-neighbour vectors, hull-incident edges removed, optionally
    divided by the summed equivalent radii."""
    C = np.asarray(table.centroids, dtype=float)
    if len(C) < 20:
        raise ValueError("too few grains for Delaunay-neighbour analysis")
    s = Delaunay(C).simplices
    e = np.concatenate([s[:, list(p)] for p in _EDGES])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    hull = np.zeros(len(C), dtype=bool)
    hull[ConvexHull(C).vertices] = True
    e = e[~hull[e[:, 0]] & ~hull[e[:, 1]]]
    V = C[e[:, 1]] - C[e[:, 0]]
    if normalise:
        if table.volume is None:
            raise ValueError("normalisation needs grain volumes")
        r = 0.5 * table.equivalent_diameters()
        V = V / (r[e[:, 0]] + r[e[:, 1]])[:, None]
    V = V[np.linalg.norm(V, axis=1) > 0]
    return np.vstack([V, -V])


def _design(U):
    x, y, z = U.T
    return np.column_stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z])


def _sym(c):
    return np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])


def estimate_strain_dtm_ls(table: GrainTable) -> StrainEstimate:
    """M18 — Mulchrone (2003) least-squares regression ``1/d² = uᵀAu`` on normalised,
    hull-trimmed Delaunay-neighbour vectors."""
    V = delaunay_neighbour_vectors(table)
    d = np.linalg.norm(V, axis=1)
    coef, *_ = np.linalg.lstsq(_design(V / d[:, None]), 1.0 / d ** 2, rcond=None)
    w, Q = np.linalg.eigh(_sym(coef))
    if np.any(w <= 0):
        raise RuntimeError("M18: regression gave a non-positive-definite quadratic form")
    semi = 1.0 / np.sqrt(w)
    o = np.argsort(semi)[::-1]
    return ellipsoid_to_strain(semi[o], method="M18_dtm_ls", axes=Q[:, o], extra={"n_vectors": len(V) // 2})


def estimate_strain_void_fit(table: GrainTable, z: float = 1000.0) -> StrainEstimate:
    """M17 — Mulchrone (2013) asymmetric least-squares void fit (polar form), in 3-D."""
    V = delaunay_neighbour_vectors(table)
    rho = np.linalg.norm(V, axis=1)
    U = V / rho[:, None]
    D = _design(U)

    def unpack(p):                                    # log-eigen parameterisation keeps A SPD
        L = _sym(p)
        w, Q = np.linalg.eigh(L)
        return (Q * np.exp(-w)) @ Q.T                 # A = exp(−L): semi-axes exp(w/2)

    def g(p):
        A = unpack(p)
        f = 1.0 / np.sqrt(D @ np.array([A[0, 0], A[1, 1], A[2, 2], A[0, 1], A[0, 2], A[1, 2]]))
        r = f - rho
        return float(np.sum(np.where(r > 0, z, 1.0) * r * r))

    # starts: an isotropic sphere at a low radial quantile, and the M18-type regression shape
    starts = [np.r_[np.full(3, 2.0 * np.log(np.quantile(rho, 0.05))), np.zeros(3)]]
    try:
        coef, *_ = np.linalg.lstsq(D, 1.0 / rho ** 2, rcond=None)
        w, Q = np.linalg.eigh(_sym(coef))
        if np.all(w > 0):
            L = (Q * -np.log(w)) @ Q.T                                    # A = exp(−L)
            # rescale so the geometric-mean semi-axis exp(tr L / 6) equals the 5 % radial quantile
            L += (2.0 * np.log(np.quantile(rho, 0.05)) - np.trace(L) / 3.0) * np.eye(3)
            starts.append(np.array([L[0, 0], L[1, 1], L[2, 2], L[0, 1], L[0, 2], L[1, 2]]))
    except np.linalg.LinAlgError:
        pass
    x0 = min(starts, key=g)
    res = minimize(g, x0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-9, "maxfev": 4000})
    w, Q = np.linalg.eigh(_sym(res.x))
    o = np.argsort(w)[::-1]
    return ellipsoid_to_strain(np.exp(w[o] / 2.0), method="M17_void_fit", axes=Q[:, o],
                               extra={"n_vectors": len(V) // 2, "objective": float(res.fun),
                                      "converged": bool(res.success)})
