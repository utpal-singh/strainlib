"""M10 — normalized Fry method (Erslev 1988), in 3-D.

Each centre-to-centre vector is divided by the sum of the two grains' equivalent radii,
``v_ij / (r_i + r_j)``, removing the blurring of the vacancy field caused by variable
grain size (for touching objects the normalized distance is 1).  The central-void boundary
of the normalized cloud (closest vector per direction) is then fitted by an ellipsoid, as
in M1.  Erslev's rationale is packed aggregates; spinel in peridotite is dispersed, so the
method is included as the literature baseline rather than an expected improvement.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, fit_ellipsoid, ellipsoid_to_strain, void_shell_points
from .fry import FRY_PAIRS

__all__ = ["estimate_strain_nfry", "short_normalized_pairs"]


def short_normalized_pairs(C, r, nn_mult: float = 4.0):
    """Ordered pairs (i, j), both directions, whose NORMALIZED distance |c_i − c_j|/(r_i + r_j)
    is at most ``D_cut`` = ``nn_mult`` × the median normalized distance to the nearest
    neighbour, found exactly (a qualifying pair lies within D_cut·(r_i + max r) of grain i).
    → (i, j, D_cut)."""
    C = np.asarray(C, dtype=float)
    r = np.asarray(r, dtype=float)
    tree = cKDTree(C)
    d, nn = tree.query(C, k=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dn = d[:, 1] / (r + r[nn[:, 1]])
    d_cut = float(nn_mult * np.median(dn[np.isfinite(dn)]))
    balls = tree.query_ball_point(C, d_cut * (r + r.max()), return_sorted=False)
    i = np.repeat(np.arange(len(C)), [len(b) for b in balls])
    j = np.concatenate([np.asarray(b, dtype=int) for b in balls]) if len(i) else np.empty(0, int)
    keep = i < j
    i, j = i[keep], j[keep]
    denom = r[i] + r[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (denom > 0) & (np.linalg.norm(C[i] - C[j], axis=1) <= d_cut * denom)
    i, j = i[ok], j[ok]
    return np.concatenate([i, j]), np.concatenate([j, i]), d_cut


def estimate_strain_nfry(table: GrainTable, n_dirs: int = 150, cone_deg: float = 15.0,
                         max_pairs: int = 200_000, rng=None, pairs: str = "short",
                         nn_mult: float = 4.0) -> StrainEstimate:
    """``pairs="short"`` (default) uses every pair whose normalized distance is within
    ``nn_mult`` × the median normalized nearest-neighbour distance
    (:func:`short_normalized_pairs`); ``pairs="sample"`` is the pre-2026-10-17 behaviour, a
    random sample of ``max_pairs`` ordered pairs (audit 2026-10-17, E3)."""
    if pairs not in FRY_PAIRS:
        raise ValueError(f"pairs must be one of {FRY_PAIRS}, got {pairs!r}")
    if table.volume is None:
        raise ValueError("normalized Fry needs grain volumes")
    C = np.asarray(table.centroids, dtype=float)
    n = len(C)
    if n < 30:
        raise ValueError("too few grains for normalized Fry analysis")
    r = 0.5 * table.equivalent_diameters()
    extra = {}
    if pairs == "short":
        i, j, extra["d_cut"] = short_normalized_pairs(C, r, nn_mult)
    elif n * (n - 1) <= max_pairs:
        i, j = np.nonzero(~np.eye(n, dtype=bool))
    else:
        rng = np.random.default_rng() if rng is None else rng
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        same = i == j
        while np.any(same):
            j[same] = rng.integers(0, n, size=int(same.sum()))
            same = i == j
    denom = r[i] + r[j]
    good = denom > 0
    V = (C[i[good]] - C[j[good]]) / denom[good, None]
    if len(V) < 30:
        raise ValueError("too few pairs for normalized Fry analysis")
    shell = void_shell_points(V, n_dirs=n_dirs, cone_deg=cone_deg)
    semi, axes = fit_ellipsoid(shell)
    if np.any(np.isnan(semi)):
        raise RuntimeError("normalized Fry ellipsoid fit failed (non-positive-definite)")
    return ellipsoid_to_strain(semi, method="M10_nfry", axes=axes,
                               extra={"n_shell": len(shell), "n_vectors": len(V), **extra})
