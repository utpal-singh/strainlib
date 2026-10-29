"""M2 — 3-D continuous-function method (CFM).

Two implementations:

* :func:`estimate_strain_cfm` : a low PERCENTILE of the in-cone
  Fry-vector norms taken as the void-boundary radius.  This is NOT the method of
  Waldron & Wallace (2007).  Its percentile is taken over all pairs in the sample, so
  inside a finite scan window it sits at ~p^{1/3} of the window radius (≈0.43 R for
  p = 8 %) and reads the window outline rather than the strain
  (``scripts/diagnostics/verify_bug_fixes.py``).  Kept for comparability.
* :func:`estimate_strain_cfm_ww` — the continuous-function method as defined by
  Waldron & Wallace (2007, §3.2.2), generalised to 3-D.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from ..schema import GrainTable
from .fry import fry_vectors
from .ellipsoid import StrainEstimate, fit_ellipsoid, ellipsoid_to_strain, fibonacci_sphere, cone_cosines

__all__ = ["estimate_strain_cfm", "estimate_strain_cfm_ww"]


def estimate_strain_cfm(table: GrainTable, n_dirs: int = 300, cone_deg: float = 20.0,
                        void_percentile: float = 8.0, min_in_cone: int = 5,
                        max_pairs: int = 400_000, rng=None) -> StrainEstimate:
    """3D CFM strain estimate (percentile-of-in-cone void boundary)."""
    vecs = fry_vectors(table.centroids, max_pairs=max_pairs, rng=rng)
    if len(vecs) < 30:
        raise ValueError("too few grains for CFM analysis")
    norms = np.linalg.norm(vecs, axis=1)
    good = norms > 1e-9
    vecs, norms = vecs[good], norms[good]
    units = vecs / norms[:, None]
    dirs = fibonacci_sphere(n_dirs)
    cos_thresh = np.cos(np.deg2rad(cone_deg))
    boundary = []
    for cos, b0 in cone_cosines(units, dirs):
        for j in range(cos.shape[1]):
            in_cone = cos[:, j] >= cos_thresh
            if in_cone.sum() < min_in_cone:
                continue
            boundary.append(np.percentile(norms[in_cone], void_percentile) * dirs[b0 + j])
    boundary = np.asarray(boundary)
    if len(boundary) < 12:
        raise RuntimeError("CFM: too few directional boundary points")
    semi, axes = fit_ellipsoid(boundary)
    if np.any(np.isnan(semi)):
        raise RuntimeError("CFM ellipsoid fit failed")
    return ellipsoid_to_strain(semi, method="M2_cfm", axes=axes, extra={"n_boundary": len(boundary)})


# --------------------------------------------------------------------------- #
def _sym_traceless(a):
    a11, a22, a12, a13, a23 = a
    return np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, -a11 - a22]])


def estimate_strain_cfm_ww(table: GrainTable, window=None, edge: str = "outer", k: float = 3.0,
                           f: float | None = None, f_window: float = 0.6, max_pairs: int = 400_000,
                           rng=None) -> StrainEstimate:
    """3-D continuous-function method (Waldron & Wallace 2007, Eqs 4–7), edge-corrected.

    A trial ellipsoid ``{v : vᵀB⁻¹v = s²}`` (det B = 1) is scored against the Fry cloud
    within the limiting radius ``f``.  Each Fry vector v at distance r has relative
    deviation ``x = r/r_e − 1 = √(vᵀB⁻¹v)/s − 1`` (Eq. 4) and contributes

        z = w_W(v) · x · exp(−(k x)²) / r²,        Z = Σ z                     (Eqs 5–7)

    ``/r²`` is the 3-D form of W&W's ``/r`` (the number of Fry points per unit radius of
    an unclustered pattern grows as r² in 3-D, not r).  ``w_W(v) = |W| / C_W(v)`` is the
    translation edge correction of the scan window W (Baddeley et al. 2015, Eq. 7.19),
    which removes the "decline towards plot edge" of W&W's Fig. 3 — without it a
    finite window biases the fit toward the window's own shape.

    ``edge="inner"`` maximises Z: the steepest RISING density edge, i.e. the vacancy
    field of anticlustered markers — W&W's original target.  ``edge="outer"`` maximises
    −Z: the steepest FALLING edge, which in a CLUSTERED pattern is the flank of the
    central cluster peak.  W&W considered outer edges but set them aside because in
    anticlustered samples they are diffuse (§3.2); in a clustered sample (Clark–Evans
    R < 1, e.g. xenolith no21, R = 0.48) the cluster flank is the best-sampled feature,
    and at cluster scales well above grain size it is carried passively by the matrix
    even when the grains themselves are rigid.

    For a statistically isotropic initial pattern strained by F, every direction of the
    unstrained frame shares the same optimal boundary radius, so the true ellipsoid
    B = FFᵀ maximises Z for either edge.  ``k = 3`` is W&W's recommended value.
    ``f`` defaults to ``f_window`` × the window's smallest half-extent; trial ellipsoids
    are confined so the kernel (|x| ≤ 2.5/k) stays inside ``f`` — ``extra['constrained']``
    flags an optimum pinned at that limit (strain too large for the window).
    """
    if edge not in ("inner", "outer"):
        raise ValueError("edge must be 'inner' or 'outer'")
    from ..observation import ScanWindow                      # lazy: avoids an import cycle
    P = np.asarray(table.centroids, dtype=float)
    n = len(P)
    if n < 30:
        raise ValueError("too few grains for CFM analysis")
    if window is None and (table.meta or {}).get("window"):   # set by observation.observe()
        window = ScanWindow(**table.meta["window"])
    if window is None:                                        # no scan geometry supplied
        window = ScanWindow(tuple(np.ptp(P, axis=0) / 2.0), "box")
    tree = cKDTree(P)
    nn = float(np.median(tree.query(P, k=2)[0][:, 1]))
    f = float(f_window * min(window.half)) if f is None else float(f)
    V = P[(ij := tree.query_pairs(f, output_type="ndarray"))[:, 1]] - P[ij[:, 0]]
    r2 = np.einsum("ij,ij->i", V, V)
    V, r2 = V[r2 > 1e-18], r2[r2 > 1e-18]
    if len(V) > max_pairs:
        rng = np.random.default_rng(0) if rng is None else rng
        sel = rng.choice(len(V), max_pairs, replace=False)
        V, r2 = V[sel], r2[sel]
    if len(V) < 50:
        raise RuntimeError("CFM-WW: too few Fry vectors within the limiting radius")
    wz = window.volume / np.clip(window.set_covariance(V), 1e-12 * window.volume, None) / r2
    sign = 1.0 if edge == "inner" else -1.0
    reach = 1.0 + 2.5 / k                                     # kernel support in units of r_e
    s_max = f / reach

    def Z(th):
        S = _sym_traceless(th[:5])
        w, Q = np.linalg.eigh(S)
        Binv = (Q * np.exp(-w)) @ Q.T
        x = np.sqrt(np.einsum("ij,jk,ik->i", V, Binv, V)) / np.exp(th[5]) - 1.0
        return float(np.sum(wz * x * np.exp(-(k * x) ** 2))), w

    # 1) isotropic scan over the trial size (robust global start)
    s_grid = np.geomspace(0.05 * nn, s_max, 60)
    z_iso = np.array([sign * Z(np.r_[np.zeros(5), np.log(s)])[0] for s in s_grid])
    s0 = float(s_grid[int(np.argmax(z_iso))])
    scale = max(abs(float(z_iso.max())), 1e-12)

    def obj(th):
        z, w = Z(th)
        excess = np.exp(th[5]) * np.exp(w.max() / 2.0) / s_max - 1.0   # longest semi-axis vs limit
        return -sign * z / scale + (1e3 * excess ** 2 if excess > 0 else 0.0)

    # 2) shape starts: isotropic, and the log-shape of the short-pair covariance (×1, ×3)
    near = r2 <= (2.0 * s0) ** 2
    starts = [np.zeros(5)]
    if near.sum() > 20:
        wc, Qc = np.linalg.eigh(np.cov(V[near].T))
        lw = np.log(np.clip(wc, 1e-12, None)); lw -= lw.mean()
        for g in (1.0, 3.0):
            L = (Qc * (g * lw)) @ Qc.T
            starts.append(np.array([L[0, 0], L[1, 1], L[0, 1], L[0, 2], L[1, 2]]))
    bounds = [(-3.0, 3.0)] * 5 + [(np.log(0.02 * nn), np.log(s_max))]
    best = None
    for a0 in starts:
        res = minimize(obj, np.r_[a0, np.log(s0)], method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    S = _sym_traceless(best.x[:5])
    w, Q = np.linalg.eigh(S)
    o = np.argsort(w)[::-1]
    s = float(np.exp(best.x[5]))
    semi = s * np.exp(w[o] / 2.0)
    return ellipsoid_to_strain(semi, method=f"M2_cfm_ww_{edge}", axes=Q[:, o],
                               extra={"s": s, "Z": -best.fun * scale * sign, "n_pairs": int(len(V)),
                                      "f": f, "k": k, "edge": edge, "nn": nn,
                                      "constrained": bool(semi[0] * reach > f * 0.999),
                                      "converged": bool(best.success)})
