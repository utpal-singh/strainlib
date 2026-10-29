"""Spatial point-pattern statistics (real-data characterisation + synthetic calibration).

Measures clustering of the real spinel cloud and fits the Thomas-process parameters
that drive the synthetic generator (Illian et al. 2008; Baddeley et al. 2015).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

__all__ = ["pair_correlation", "clark_evans_index", "fit_thomas_pcf", "bounding_box", "PCF_EDGE"]


def bounding_box(points: np.ndarray) -> tuple[float, float, float]:
    """Axis-aligned extent (Lx, Ly, Lz) of a point cloud."""
    P = np.asarray(points, dtype=float)
    return tuple(float(x) for x in (P.max(0) - P.min(0)))


PCF_EDGE = ("translation", "none")


def pair_correlation(points, box, r_max=None, dr=None, edge="translation"):
    """Estimate the isotropic 3D pair-correlation function g(r) → (r_centres, g).

    Bin estimator, g(r)=1 random, >1 clustered.  ``box`` = (Lx, Ly, Lz) is the observation
    window, an axis-aligned box that contains the points (only its side lengths matter).

    ``edge="translation"`` (default) applies the translation edge correction (Illian et al.
    2008; Baddeley et al. 2015): each ordered pair with separation v is weighted by
    |W|/C_W(v), C_W(v) = Π(L_i − |v_i|) the box's set covariance, and the squared intensity
    is estimated by n(n−1)/|W|².  This is unbiased for g at every r < min(L_i).

    ``edge="none"`` is the uncorrected estimator used before 2026-10-17 (kept for
    comparison): shells reaching past the box walls are counted as if fully inside, so g
    is biased low, increasingly with r (CSR in the no21 box: 0.92 at 2.7 mm, 0.72 at
    10.7 mm), and a Thomas fit to it is biased (μ −13 %, κ +19 %, σ −4 %).
    """
    if edge not in PCF_EDGE:
        raise ValueError(f"edge must be one of {PCF_EDGE}, got {edge!r}")
    P = np.asarray(points, dtype=float)
    L = np.asarray(box, dtype=float)
    volume = float(np.prod(L))
    n = len(P)
    if n < 10:
        raise ValueError("need at least 10 points for g(r)")
    lam = n / volume
    if r_max is None:
        r_max = 0.25 * min(box)
    if dr is None:
        dr = r_max / 30.0
    edges = np.arange(0, r_max + dr, dr)
    centres = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    tree = cKDTree(P)
    if edge == "none":
        cum = np.array([tree.count_neighbors(tree, r) for r in edges], dtype=float)
        counts = np.diff(cum)                          # ordered i!=j pairs per shell
        expected = lam * shell_vol * n
    else:
        ij = tree.query_pairs(edges[-1], output_type="ndarray")
        v = P[ij[:, 1]] - P[ij[:, 0]]
        cw = np.prod(L - np.abs(v), axis=1)            # set covariance C_W(v) of the box
        keep = cw > 0
        w = 2.0 * volume / cw[keep]                    # both orders of each unordered pair
        counts, _ = np.histogram(np.linalg.norm(v[keep], axis=1), bins=edges, weights=w)
        expected = (n * (n - 1) / volume) * shell_vol  # λ²|W| × shell volume, λ² = n(n−1)/|W|²
    g = np.divide(counts, expected, out=np.zeros_like(counts), where=expected > 0)
    return centres, g


def clark_evans_index(points, box) -> float:
    """Clark-Evans nearest-neighbour index (R<1 clustered, =1 random, >1 regular)."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    lam = n / (box[0] * box[1] * box[2])
    d, _ = cKDTree(P).query(P, k=2)
    expected = 0.55396 / (lam ** (1.0 / 3.0))          # E[d] for 3D Poisson
    return float(np.mean(d[:, 1]) / expected)


def fit_thomas_pcf(r, g, lam) -> dict:
    """Fit the 3-D Thomas pair-correlation ``g(r) = 1 + exp(−r²/4σ²) / (κ·(4πσ²)^{3/2})``
    → (kappa, sigma, mu=lam/kappa).

    Neyman–Scott form ``g = 1 + h(r)/κ``, with ``h`` the density of the difference of
    two iid N(0, σ²I₃) offspring displacements, i.e. N(0, 2σ²I₃) (Illian et al. 2008;
    Baddeley et al. 2015).  (The previous ``1/(4πσ²κr²)`` term had units of mm⁻¹ — not a
    valid pcf — and drove the fitted σ to ~11 mm, i.e. effectively unclustered rocks.)"""
    r = np.asarray(r, dtype=float); g = np.asarray(g, dtype=float)
    mask = r > 0
    r, g = r[mask], g[mask]

    def model(params):
        kappa, sigma = params
        return 1.0 + np.exp(-(r**2) / (4 * sigma**2)) / (kappa * (4 * np.pi * sigma**2) ** 1.5) - g

    p0 = [max(lam / 20.0, 1e-9), max(r[np.argmax(g)], r[1])]
    res = least_squares(model, p0, bounds=([1e-12, 1e-6], [lam, r.max()]), max_nfev=2000)
    kappa, sigma = res.x
    return {"kappa": float(kappa), "sigma": float(sigma),
            "mu": float(lam / kappa), "cost": float(res.cost)}
