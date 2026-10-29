"""M8 — Gray-Level Deviation (GLD) fabric tensor (after Lenaerts et al. 2014).

Operates on the CT gray volume, not a segmented cloud.  Along lines of length ``L``
in many directions, the standard deviation of gray values is computed and averaged
over sample points.  A fabric aligned along X (lineation) crosses fewer spinel–matrix
boundaries along X → low deviation along X and high along Z; the directional-deviation
ellipsoid is therefore the *inverse* of the strain ellipsoid, and is inverted to
recover it.  Applied only where gray values are available (Decision D3: validated on
a subset).
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, fit_ellipsoid, ellipsoid_to_strain, fibonacci_sphere

__all__ = ["gld_directional_deviation", "estimate_strain_gld"]


def gld_directional_deviation(volume, voxel_size, n_dirs=120, line_len_vox=12.0,
                              n_line=13, n_samples=1500, rng=None):
    """Mean gray-level std along ``n_dirs`` directions → (sigma_bar (n_dirs,), dirs (n_dirs,3))."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng() if rng is None else rng
    dims = np.array(volume.shape)
    dirs = fibonacci_sphere(n_dirs)
    half = line_len_vox / 2.0
    t = np.linspace(-half, half, n_line)                        # sample offsets (voxels)
    # sample points safely inside the volume (avoid the line running off the edge)
    margin = half + 1
    lo = np.full(3, margin)
    hi = dims - margin
    ok = np.all(hi > lo)
    if not ok:
        raise ValueError("volume too small for the requested line length")
    pts = rng.uniform(lo, hi, size=(n_samples, 3))              # voxel coords
    sigma = np.empty(n_dirs)
    for d in range(n_dirs):
        step = dirs[d]                                          # unit dir in voxel space
        # (n_samples, n_line, 3) line coordinates
        coords = pts[:, None, :] + t[None, :, None] * step[None, None, :]
        vals = map_coordinates(volume, coords.reshape(-1, 3).T, order=1, mode="nearest")
        vals = vals.reshape(n_samples, n_line)
        sigma[d] = np.mean(np.std(vals, axis=1))
    return sigma, dirs


def estimate_strain_gld(table: GrainTable, voxel_size: float | None = None,
                        gray_volume=None, n_dirs=120, rng=None) -> StrainEstimate:
    """GLD strain estimate.  If ``gray_volume`` is not given, the table is rasterised
    at ``voxel_size`` (default ≈ 1/4 of the median grain diameter)."""
    from ..synthetic.volume import rasterize_grains
    rng = np.random.default_rng() if rng is None else rng
    if gray_volume is None:
        if voxel_size is None:
            voxel_size = float(np.median(table.equivalent_diameters()) / 4.0)
        vol, _, voxel_size = rasterize_grains(table, voxel_size, rng=rng)
    else:
        vol = gray_volume
        if voxel_size is None:
            voxel_size = 1.0
    sigma, dirs = gld_directional_deviation(vol, voxel_size, n_dirs=n_dirs, rng=rng)
    # fit an ellipsoid to the directional deviation, then invert (deviation ellipsoid
    # is the inverse of the strain ellipsoid)
    semi_dev, axes = fit_ellipsoid((sigma[:, None] * dirs))
    if np.any(np.isnan(semi_dev)):
        raise RuntimeError("GLD ellipsoid fit failed")
    semi_strain = 1.0 / semi_dev                                # invert
    order = np.argsort(semi_strain)[::-1]
    return ellipsoid_to_strain(semi_strain[order], method="M8_gld", axes=axes[:, order],
                               extra={"sigma_range": (float(sigma.min()), float(sigma.max()))})
