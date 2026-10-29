"""M6 / M7 — conventional 2D strain analysis on three orthogonal sections.

A thin central slab is taken perpendicular to each coordinate axis and its grains
projected to the plane; a 2D strain-ellipse ratio is recovered from the in-plane
Fry cloud, and the three ratios are combined into a 3D ellipsoid (Ramsay & Huber
1983).  Because the (unknown) principal strain axes are generally *not* aligned
with the coordinate sections, the reconstruction carries a geometric error — this
is precisely the "why existing 2D methods fail on 3D data" benchmark.

M6 uses the **raw 2D-Fry** void boundary (nearest-in-cone); M7 uses the **2D
continuous-function** void (percentile-in-cone) — genuinely distinct estimators.
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, fit_ellipse_2d, ellipsoid_to_strain, void_shell_2d, cone_cosines

__all__ = ["estimate_strain_sections"]


def _slab_indices(coord, frac):
    lo, hi = np.min(coord), np.max(coord)
    mid = 0.5 * (lo + hi)
    return np.abs(coord - mid) <= 0.5 * frac * (hi - lo)


def _fry_vectors_2d(P, max_pairs=200_000, rng=None):
    P = np.asarray(P, dtype=float)
    n = len(P)
    if n < 10:
        raise ValueError("too few points in section for 2D analysis")
    if n * (n - 1) <= max_pairs:
        diff = P[:, None, :] - P[None, :, :]
        return diff[~np.eye(n, dtype=bool)]
    rng = np.random.default_rng() if rng is None else rng
    i = rng.integers(0, n, size=max_pairs); j = rng.integers(0, n, size=max_pairs)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(same.sum())); same = i == j
    return P[i] - P[j]


def _section_ratio(P, mode, rng, n_dirs=180, cone_deg=12.0, void_percentile=8.0, min_in_cone=5):
    """2D axial ratio (a/b) of the Fry-void ellipse in one section."""
    vecs = _fry_vectors_2d(P, rng=rng)
    if mode == "fry":                                 # M6: raw nearest-in-cone void
        shell = void_shell_2d(vecs, n_dirs=n_dirs, cone_deg=cone_deg)
    elif mode == "cfm":                               # M7: continuous percentile void
        norms = np.linalg.norm(vecs, axis=1)
        good = norms > 1e-9
        vecs, norms = vecs[good], norms[good]
        units = vecs / norms[:, None]
        ang = np.linspace(0, np.pi, n_dirs, endpoint=False)
        dirs = np.column_stack([np.cos(ang), np.sin(ang)])
        cos_thresh = np.cos(np.deg2rad(cone_deg))
        shell = []
        for cos, b0 in cone_cosines(units, dirs):
            cos = np.abs(cos)
            for j in range(cos.shape[1]):
                ic = cos[:, j] >= cos_thresh
                if ic.sum() < min_in_cone:
                    continue
                r = np.percentile(norms[ic], void_percentile)
                shell.append(r * dirs[b0 + j]); shell.append(-r * dirs[b0 + j])
        shell = np.asarray(shell)
    else:
        raise ValueError("mode must be 'fry' or 'cfm'")
    semi, _ = fit_ellipse_2d(shell)
    return float(semi[0] / semi[1])


def estimate_strain_sections(table: GrainTable, mode: str = "fry", slab_frac: float = 0.5,
                             method_name: str | None = None, rng=None) -> StrainEstimate:
    """2D-section reconstruction of the 3D strain ellipsoid (M6 fry / M7 cfm)."""
    if method_name is None:
        method_name = "M6_fry2d" if mode == "fry" else "M7_cfm2d"
    C = table.centroids
    Rxy = _section_ratio(C[_slab_indices(C[:, 2], slab_frac)][:, [0, 1]], mode, rng)   # thin-z slab
    Rxz = _section_ratio(C[_slab_indices(C[:, 1], slab_frac)][:, [0, 2]], mode, rng)   # thin-y slab
    Ryz = _section_ratio(C[_slab_indices(C[:, 0], slab_frac)][:, [1, 2]], mode, rng)   # thin-x slab
    # reconstruct principal stretches from X/Z=Rxz, Y/Z=Ryz (Z=1), then volume-normalise
    t = (Rxz * Ryz) ** (-1.0 / 3.0)
    semi = np.sort(np.array([Rxz * t, Ryz * t, t]))[::-1]
    return ellipsoid_to_strain(semi, method=method_name, axes=None,
                               extra={"Rxy": Rxy, "Rxz": Rxz, "Ryz": Ryz})
