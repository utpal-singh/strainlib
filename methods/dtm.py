"""M3 — 3D Delaunay Triangulation Method (Mulchrone 2003, extended to 3D).

Restricting the Fry cloud to Delaunay nearest-neighbour pairs sharpens the void
boundary by excluding long-range clustering noise.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, fit_ellipsoid, ellipsoid_to_strain, void_shell_points

__all__ = ["dtm_vectors", "estimate_strain_dtm"]


def dtm_vectors(centroids) -> np.ndarray:
    """Fry vectors restricted to Delaunay nearest-neighbour pairs (symmetric cloud)."""
    C = np.asarray(centroids, dtype=float)
    if len(C) < 5:
        return np.empty((0, 3))
    tri = Delaunay(C)
    edges = set()
    for simplex in tri.simplices:
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = simplex[a], simplex[b]
                edges.add((min(i, j), max(i, j)))
    edges = np.array(list(edges))
    d = C[edges[:, 0]] - C[edges[:, 1]]
    return np.vstack([d, -d])


def estimate_strain_dtm(table: GrainTable, n_dirs: int = 250, cone_deg: float = 18.0) -> StrainEstimate:
    """3D DTM: void-shell ellipsoid fit on Delaunay-edge Fry vectors."""
    vecs = dtm_vectors(table.centroids)
    if len(vecs) < 30:
        raise ValueError("too few grains for DTM analysis")
    shell = void_shell_points(vecs, n_dirs=n_dirs, cone_deg=cone_deg)
    semi, axes = fit_ellipsoid(shell)
    if np.any(np.isnan(semi)):
        raise RuntimeError("DTM ellipsoid fit failed")
    return ellipsoid_to_strain(semi, method="M3_dtm", axes=axes, extra={"n_edges": len(vecs) // 2})
