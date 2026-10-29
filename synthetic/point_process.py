"""3D Thomas cluster process with Matérn hard-core thinning.

The Thomas process (Thomas 1949; Illian et al. 2008) models spinels that cluster
around unobserved parent sites (e.g. olivine grain triple junctions); Matérn-II
hard-core thinning enforces a minimum inter-grain distance (Andersen & Hahn 2016).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["thomas_process", "matern_hardcore_thin", "generate_centroids"]


def thomas_process(box, kappa, mu, sigma, rng=None) -> np.ndarray:
    """Daughter points of a 3D Thomas cluster process, clipped to ``box``.

    ``kappa`` parent intensity (per unit volume), ``mu`` mean daughters/parent,
    ``sigma`` isotropic Gaussian daughter scatter.
    """
    rng = np.random.default_rng() if rng is None else rng
    Lx, Ly, Lz = box
    pad = 3.0 * sigma
    ext_volume = (Lx + 2*pad) * (Ly + 2*pad) * (Lz + 2*pad)
    n_parents = rng.poisson(kappa * ext_volume)
    if n_parents == 0:
        return np.empty((0, 3))
    parents = rng.uniform([-pad, -pad, -pad], [Lx+pad, Ly+pad, Lz+pad], size=(n_parents, 3))
    counts = rng.poisson(mu, size=n_parents)
    total = int(counts.sum())
    if total == 0:
        return np.empty((0, 3))
    pts = np.repeat(parents, counts, axis=0) + rng.normal(0.0, sigma, size=(total, 3))
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] <= Lx) & (pts[:, 1] >= 0) & (pts[:, 1] <= Ly)
              & (pts[:, 2] >= 0) & (pts[:, 2] <= Lz))
    return pts[inside]


def matern_hardcore_thin(points, hardcore, rng=None) -> np.ndarray:
    """Random sequential (Matérn type III) hard-core thinning: visit the points in random order
    and retain one only if no already-accepted point lies within ``hardcore``."""
    rng = np.random.default_rng() if rng is None else rng
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0 or hardcore <= 0:
        return points
    order = rng.permutation(n)
    pairs = cKDTree(points).query_pairs(r=hardcore, output_type="ndarray")
    conflict = {i: set() for i in range(n)}
    for i, j in pairs:
        conflict[i].add(j); conflict[j].add(i)
    accepted: set[int] = set()
    for idx in order:
        if accepted.isdisjoint(conflict[idx]):
            accepted.add(int(idx))
    return points[np.array(sorted(accepted), dtype=int)]


def generate_centroids(box, kappa, mu, sigma, hardcore=0.0, rng=None) -> np.ndarray:
    """Thomas process followed by optional hard-core thinning."""
    rng = np.random.default_rng() if rng is None else rng
    pts = thomas_process(box, kappa, mu, sigma, rng=rng)
    if hardcore > 0:
        pts = matern_hardcore_thin(pts, hardcore, rng=rng)
    return pts
