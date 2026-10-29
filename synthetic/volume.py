"""Rasterise a GrainTable into a synthetic X-ray CT gray volume.

Needed by M8 (gray-level deviation), which operates on voxel gray values rather
than a segmented centroid cloud.  Each grain is drawn as its equivalent ellipsoid
(bright spinel) in a dark matrix, with additive Gaussian noise.
"""
from __future__ import annotations

import numpy as np

__all__ = ["rasterize_grains"]


def rasterize_grains(table, voxel_size, pad=2.0, matrix=0.2, spinel=1.0,
                     noise_std=0.05, rng=None):
    """Rasterise grains into a 3D gray volume → (volume, origin, voxel_size).

    ``origin`` is the world coordinate of voxel (0,0,0).  Requires grain shapes.
    """
    rng = np.random.default_rng() if rng is None else rng
    if not table.has_shapes:
        raise ValueError("rasterize_grains requires grain shapes")
    C = table.centroids
    semi = table.semi_axes()
    R = table.eigenvectors
    lo = C.min(0) - pad
    hi = C.max(0) + pad
    dims = tuple(int(np.ceil((h - l) / voxel_size)) + 1 for l, h in zip(lo, hi))
    vol = np.full(dims, float(matrix))

    for i in range(table.n):
        a = semi[i]
        Minv = R[i] @ np.diag(1.0 / np.clip(a, 1e-6, None) ** 2) @ R[i].T   # (x-c)ᵀMinv(x-c)<=1
        ext = a.max()
        vlo = np.clip(np.floor((C[i] - ext - lo) / voxel_size).astype(int), 0, np.array(dims) - 1)
        vhi = np.clip(np.ceil((C[i] + ext - lo) / voxel_size).astype(int), 0, np.array(dims) - 1)
        if np.any(vhi < vlo):
            continue
        gx = np.arange(vlo[0], vhi[0] + 1)
        gy = np.arange(vlo[1], vhi[1] + 1)
        gz = np.arange(vlo[2], vhi[2] + 1)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        coords = np.stack([lo[0] + X * voxel_size, lo[1] + Y * voxel_size, lo[2] + Z * voxel_size], -1)
        d = coords - C[i]
        q = np.einsum("...j,jk,...k->...", d, Minv, d)
        inside = q <= 1.0
        vol[X[inside], Y[inside], Z[inside]] = spinel

    if noise_std > 0:
        vol = vol + rng.normal(0.0, noise_std, size=dims)
    return vol, lo, voxel_size
