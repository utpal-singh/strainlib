"""M1 — raw 3D Fry method (Fry 1979)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, fit_ellipsoid, ellipsoid_to_strain, void_shell_points

__all__ = ["fry_vectors", "short_fry_vectors", "short_pair_cutoff", "estimate_strain_fry", "FRY_PAIRS"]

#: pair modes of the void-shell estimators (M1, M10): every pair within a short cutoff
#: (default since 2026-10-17), or the random sample of ``max_pairs`` ordered pairs used before
FRY_PAIRS = ("short", "sample")


def fry_vectors(centroids, max_pairs: int | None = 400_000, rng=None) -> np.ndarray:
    """Inter-centroid difference vectors (the Fry cloud).

    For large populations the full ``n(n-1)`` set is never materialised; instead
    ``max_pairs`` random ordered pairs are sampled — essential for real CT data
    (grain counts can reach 1e5+).  This is the full-lag cloud used by the outer-edge and
    percentile estimators (M2, M9); the void-shell estimators use :func:`short_fry_vectors`.
    """
    C = np.asarray(centroids, dtype=float)
    n = len(C)
    if n < 3:
        return np.empty((0, 3))
    total = n * (n - 1)
    if max_pairs is None or total <= max_pairs:
        diff = C[:, None, :] - C[None, :, :]
        return diff[~np.eye(n, dtype=bool)]
    rng = np.random.default_rng() if rng is None else rng
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(same.sum()))
        same = i == j
    return C[i] - C[j]


def short_pair_cutoff(centroids, nn_mult: float = 4.0) -> float:
    """Default cutoff of :func:`short_fry_vectors`: ``nn_mult`` × the median nearest-neighbour
    distance, several times the radius of the central void."""
    C = np.asarray(centroids, dtype=float)
    return float(nn_mult * np.median(cKDTree(C).query(C, k=2)[0][:, 1]))


def short_fry_vectors(centroids, r_cut: float | None = None, nn_mult: float = 4.0) -> np.ndarray:
    """Every inter-centroid vector shorter than ``r_cut``, in both directions (±v).

    The central void of the Fry cloud is defined by the rare close pairs, so a void-shell
    estimator needs all of them; a random sample of ordered pairs (:func:`fry_vectors` with
    ``max_pairs``) discards most of them once n(n−1) exceeds the cap (audit 2026-10-17, E3).
    ``r_cut`` defaults to :func:`short_pair_cutoff`."""
    C = np.asarray(centroids, dtype=float)
    if len(C) < 3:
        return np.empty((0, 3))
    if r_cut is None:
        r_cut = short_pair_cutoff(C, nn_mult)
    ij = cKDTree(C).query_pairs(r_cut, output_type="ndarray")
    V = C[ij[:, 1]] - C[ij[:, 0]]
    return np.vstack([V, -V])


def estimate_strain_fry(table: GrainTable, n_dirs: int = 250, cone_deg: float = 15.0,
                        max_pairs: int = 400_000, rng=None, pairs: str = "short",
                        nn_mult: float = 4.0) -> StrainEstimate:
    """Raw 3D Fry: fit an ellipsoid to the central-void boundary shell.

    ``pairs="short"`` (default) uses every pair within ``nn_mult`` × the median
    nearest-neighbour distance; ``pairs="sample"`` is the pre-2026-10-17 behaviour, a random
    sample of ``max_pairs`` ordered pairs (``max_pairs`` and ``rng`` apply only to it)."""
    if pairs not in FRY_PAIRS:
        raise ValueError(f"pairs must be one of {FRY_PAIRS}, got {pairs!r}")
    extra = {}
    if pairs == "short":
        r_cut = short_pair_cutoff(table.centroids, nn_mult) if len(table.centroids) >= 3 else 0.0
        vecs = short_fry_vectors(table.centroids, r_cut=r_cut)
        extra["r_cut"] = r_cut
    else:
        vecs = fry_vectors(table.centroids, max_pairs=max_pairs, rng=rng)
    if len(vecs) < 30:
        raise ValueError("too few grains for Fry analysis")
    shell = void_shell_points(vecs, n_dirs=n_dirs, cone_deg=cone_deg)
    semi, axes = fit_ellipsoid(shell)
    if np.any(np.isnan(semi)):
        raise RuntimeError("Fry ellipsoid fit failed (non-positive-definite)")
    return ellipsoid_to_strain(semi, method="M1_fry", axes=axes,
                               extra={"n_shell": len(shell), "n_vectors": len(vecs), **extra})
