"""M19 — star-length-distribution (SLD) shape fabric of the grain phase, after Ketcham &
Ryan (2004); the star-length/star-volume family is what CT fabric studies of spinel report
(Chatzaras et al. 2016; Bernard et al. 2019).

Star length ``s(ω)`` = mean distance, along direction ω, from random points inside the grain
phase to the phase boundary (Ketcham & Ryan 2004, Eq. 1).  For disjoint ellipsoidal grains it
is computed analytically: a point ``p`` inside grain ``(c, M)`` reaches the boundary at the
positive root ``t`` of ``(p − c + tω)ᵀ M⁻¹ (p − c + tω) = 1`` (our derivation).  For one
ellipsoid ``E = A·Ball`` the mean interior star length is exactly proportional to the radius
function ``r(ω) = |A⁻¹ω|⁻¹``, so a least-squares ellipsoid ``1/s(ω)² = ωᵀ Q ω`` gives the
fabric ellipsoid; its axial ratio is reported as Rs (an UNCALIBRATED strain proxy — "P′
magnitude may not coincide with the strain magnitude", Chatzaras et al. 2016 p. 4899).
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, ellipsoid_to_strain, fibonacci_sphere

__all__ = ["estimate_strain_star_length"]


def estimate_strain_star_length(table: GrainTable, n_points: int = 6000, n_dirs: int = 130,
                                rng=None) -> StrainEstimate:
    if not table.has_shapes or table.volume is None or table.n < 10:
        raise ValueError("star-length fabric needs grain shapes and volumes")
    rng = np.random.default_rng(0) if rng is None else rng
    M = table.shape_tensors()                                     # R diag(a²) Rᵀ, (N,3,3)
    ok = np.isfinite(np.linalg.det(M)) & (np.linalg.det(M) > 0)
    idx_ok = np.nonzero(ok)[0]
    vol = table.volume[idx_ok]
    g = idx_ok[rng.choice(len(idx_ok), size=n_points, p=vol / vol.sum())]    # volume-weighted grains
    w, V = np.linalg.eigh(M[g])
    half = (V * np.sqrt(w)[:, None, :])                          # A = V diag(√w): E = A·Ball
    u = rng.normal(size=(n_points, 3))
    u *= (rng.uniform(size=(n_points, 1)) ** (1.0 / 3.0)) / np.linalg.norm(u, axis=1, keepdims=True)
    q = np.einsum("nij,nj->ni", half, u)                          # p − c, uniform inside grain
    Minv = (V / w[:, None, :]) @ np.swapaxes(V, 1, 2)
    dirs = fibonacci_sphere(n_dirs)
    dirs = dirs @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T     # random rotation of the direction set
    a = np.einsum("di,nij,dj->nd", dirs, Minv, dirs)              # ωᵀM⁻¹ω
    b = np.einsum("ni,nij,dj->nd", q, Minv, dirs)                 # qᵀM⁻¹ω
    c = np.einsum("ni,nij,nj->n", q, Minv, q)[:, None] - 1.0      # qᵀM⁻¹q − 1 (< 0 inside)
    t = (-b + np.sqrt(np.clip(b * b - a * c, 0.0, None))) / a     # positive root
    s = t.mean(0)                                                 # star length per direction
    x, y, zc = dirs.T
    D = np.column_stack([x * x, y * y, zc * zc, 2 * x * y, 2 * x * zc, 2 * y * zc])
    coef, *_ = np.linalg.lstsq(D, 1.0 / s ** 2, rcond=None)
    Q = np.array([[coef[0], coef[3], coef[4]], [coef[3], coef[1], coef[5]], [coef[4], coef[5], coef[2]]])
    ew, EV = np.linalg.eigh(Q)
    if np.any(ew <= 0):
        raise RuntimeError("M19: star-length ellipsoid fit not positive definite")
    semi = 1.0 / np.sqrt(ew)
    o = np.argsort(semi)[::-1]
    return ellipsoid_to_strain(semi[o], method="M19_star_length", axes=EV[:, o],
                               extra={"n_points": n_points, "n_dirs": n_dirs})
