"""M12 / M13 / M14 — isotropy-restoration strain estimators (new; branch strain-inversion).

Principle
---------
Every centre-to-centre method rests on one assumption (Fry 1979): the undeformed marker
pattern was statistically isotropic.  If ``F`` is the volume-constant finite strain and
``B = F Fᵀ``, the retrodeformed ("whitened") pattern ``B^{-1/2} x`` is therefore isotropic
again — whatever its clustering, hard-core spacing or radial pair profile.  Instead of
reading one feature of the Fry plot (a void edge, a percentile, a density gradient),
these estimators solve for the ``B`` that restores isotropy of the LOCAL neighbourhood
structure, which is where a clustered pattern carries its signal.  They are a statistical,
3-D, window-corrected form of the iterative "unstraining" of McNaught (1994), as reviewed
by Waldron & Wallace (2007).

M12 (pairs).  Estimating equations ``T_h(B) = I/3``: ``T_h`` is the orientation tensor of
the whitened pair vectors ``w = B^{-1/2} v``, weighted by a Gaussian kernel of scale ``h``
in ``|w|`` and by the window's translation edge correction ``|W| / C_W(v)`` (Baddeley et al.
2015, Eq. 7.19).  At the true ``B`` the whitened pair cloud is isotropic, the kernel is
isotropic in whitened space and the edge weights undo the fixed observation window, so
``E[T_h] = I/3`` for every ``h`` — an unbiased estimating equation that needs no model of
the clustering (contrast R1, which assumes Gaussian clusters).  Two scales (1 and 2 median
nearest-neighbour distances) give 10 equations for the 5 unknowns (det B = 1), solved by
least squares.

M13 (self-consistent Delaunay).  Mulchrone's (2003) Delaunay neighbours, made
affine-consistent.  A Delaunay triangulation is not invariant under strain (empty
circumspheres are not preserved), so the edge set M3 uses is biased toward isotropy and its
void-edge reading is filled in by clustering.  Here the triangulation is recomputed on the
whitened points each iteration and ``B`` is updated until the whitened unit edge vectors have
an isotropic orientation tensor.  Edges touching points within a boundary margin are dropped
(minus-sampling), so the window cannot truncate neighbourhoods.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import Delaunay, cKDTree

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, ellipsoid_to_strain

__all__ = ["estimate_strain_isotropy", "estimate_strain_scdtm", "estimate_strain_sepl"]

_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _sym_traceless(a):
    a11, a22, a12, a13, a23 = a
    return np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, -a11 - a22]])


def _iso_residual(T):
    """5 independent components of ``3T − I`` (T has unit trace)."""
    return 3.0 * np.array([T[0, 0] - 1.0 / 3.0, T[1, 1] - 1.0 / 3.0, T[0, 1], T[0, 2], T[1, 2]])


def _estimate(w, Q, method, extra):
    o = np.argsort(w)[::-1]
    return ellipsoid_to_strain(np.exp(w[o] / 2.0), method=method, axes=Q[:, o], extra=extra)


# --------------------------------------------------------------------------- M12
def estimate_strain_isotropy(table: GrainTable, window=None, h_nn=(1.0, 2.0), support: float = 3.5,
                             r_window: float = 1.0, max_pairs: int = 300_000, rng=None) -> StrainEstimate:
    """Isotropy restoration on kernel-weighted, edge-corrected pair orientations (M12)."""
    from ..observation import window_for
    W, _ = window_for(table, window)
    P = np.asarray(table.centroids, dtype=float)
    if len(P) < 30:
        raise ValueError("too few grains for isotropy restoration")
    tree = cKDTree(P)
    nn = float(np.median(tree.query(P, k=2)[0][:, 1]))
    R = float(r_window * min(W.half))
    ij = tree.query_pairs(R, output_type="ndarray")
    V = P[ij[:, 1]] - P[ij[:, 0]]
    if len(V) > max_pairs:
        rng = np.random.default_rng(0) if rng is None else rng
        V = V[rng.choice(len(V), max_pairs, replace=False)]
    if len(V) < 100:
        raise RuntimeError("M12: too few pairs")
    omega = W.volume / np.clip(W.set_covariance(V), 1e-9 * W.volume, None)
    hs = np.asarray(h_nn, dtype=float) * nn

    def residuals(theta):
        w, Q = np.linalg.eigh(_sym_traceless(theta))
        Wv = V @ ((Q * np.exp(-w / 2.0)) @ Q.T).T
        r2 = np.einsum("ij,ij->i", Wv, Wv)
        out = []
        for h in hs:
            m = (r2 <= (support * h) ** 2) & (r2 > 0)
            if m.sum() < 30:
                out.append(np.full(5, 3.0))
                continue
            k = omega[m] * np.exp(-r2[m] / (2.0 * h * h))
            U = Wv[m] / np.sqrt(r2[m])[:, None]
            out.append(_iso_residual((U.T * k) @ U / k.sum()))
        return np.concatenate(out)

    # starts: isotropic, and the log-shape of the short-pair covariance (and its double)
    short = np.einsum("ij,ij->i", V, V) <= (support * hs[0]) ** 2
    starts = [np.zeros(5)]
    if short.sum() > 30:
        wc, Qc = np.linalg.eigh(np.cov(V[short].T))
        lw = np.log(np.clip(wc, 1e-12, None)); lw -= lw.mean()
        for g in (1.0, 2.0):
            L = (Qc * (g * lw)) @ Qc.T
            starts.append(np.array([L[0, 0], L[1, 1], L[0, 1], L[0, 2], L[1, 2]]))
    x0 = min(starts, key=lambda s: float(np.sum(residuals(s) ** 2)))      # best start, one solve
    best = least_squares(residuals, x0, method="trf", diff_step=1e-3, max_nfev=200, bounds=(-4.0, 4.0))
    w, Q = np.linalg.eigh(_sym_traceless(best.x))
    stretch_max = float(np.exp(w.max() / 2.0))
    return _estimate(w, Q, "M12_isotropy",
                     {"cost": float(best.cost), "nn": nn, "h": hs.tolist(), "n_pairs": int(len(V)),
                      "constrained": bool(support * hs.max() * stretch_max > R),
                      "converged": bool(best.success)})


# --------------------------------------------------------------------------- M13
def _delaunay_pairs(X):
    s = Delaunay(X).simplices
    e = np.concatenate([s[:, list(p)] for p in _EDGES])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def estimate_strain_scdtm(table: GrainTable, window=None, margin_nn: float = 3.0, len_q: float = 0.95,
                          gain: float = 2.5, max_iter: int = 40, tol: float = 2e-3) -> StrainEstimate:
    """Self-consistent (affine-covariant) Delaunay isotropy restoration (M13)."""
    from ..observation import window_for, boundary_distance
    W, centre = window_for(table, window)
    P = np.asarray(table.centroids, dtype=float)
    n = len(P)
    if n < 30:
        raise ValueError("too few grains for the self-consistent Delaunay method")
    nn = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))
    interior = boundary_distance(W, P - centre) >= margin_nn * nn
    if interior.sum() < 30:
        raise RuntimeError("M13: too few interior grains")
    logB = np.zeros((3, 3))
    history, last_norm, g = [], np.inf, gain
    converged = False
    for it in range(max_iter):
        wb, Qb = np.linalg.eigh(logB)
        Xw = P @ ((Qb * np.exp(-wb / 2.0)) @ Qb.T).T
        e = _delaunay_pairs(Xw)
        e = e[interior[e[:, 0]] & interior[e[:, 1]]]
        Vw = Xw[e[:, 1]] - Xw[e[:, 0]]
        L = np.linalg.norm(Vw, axis=1)
        keep = (L > 0) & (L <= np.quantile(L, len_q))
        U = Vw[keep] / L[keep, None]
        T = U.T @ U / len(U)
        wt, Qt = np.linalg.eigh(T)
        lt = np.log(np.clip(wt, 1e-12, None)); lt -= lt.mean()
        norm = float(np.linalg.norm(lt))
        history.append(logB.copy())
        if norm < tol:
            converged = True
            break
        if norm > last_norm:                               # overshoot → damp the gain
            g *= 0.5
        last_norm = norm
        Bh = (Qb * np.exp(wb / 2.0)) @ Qb.T
        Bnew = Bh @ ((Qt * np.exp(g * lt)) @ Qt.T) @ Bh
        wn, Qn = np.linalg.eigh(0.5 * (Bnew + Bnew.T))
        ln = np.log(np.clip(wn, 1e-12, None)); ln -= ln.mean()
        logB = (Qn * ln) @ Qn.T
    if not converged:                                      # fluctuating around the root
        logB = np.mean(history[-10:], axis=0)
    w, Q = np.linalg.eigh(logB)
    return _estimate(w, Q, "M13_scdtm", {"n_iter": it + 1, "converged": converged, "final_gain": g,
                                         "n_interior": int(interior.sum()), "nn": nn})


# --------------------------------------------------------------------------- M14
def estimate_strain_sepl(table: GrainTable, window=None, n_bins: int = 14, stretch_cap: float = 3.0,
                         n_dirs: int = 160, max_pairs: int = 300_000, rng=None) -> StrainEstimate:
    """Semiparametric elliptical pair likelihood (M14).

    Pair separations ``v`` observed in window ``W`` have density ∝ ``C_W(v) · g₀(‖B^{-1/2}v‖)``
    for ANY isotropic radial profile ``g₀`` (clusters, hard core, both).  With ``g₀``
    piecewise-constant over whitened-radius bins, its maximum-likelihood value profiles out
    in closed form, leaving the composite profile log-likelihood

        ℓ(B) = Σ_k n_k(B) · log( (n_k(B)/N) / (E_k(B)/E) ),   N = Σ n_k,  E = Σ E_k

    with ``n_k`` the number of whitened pair separations in bin ``k`` and
    ``E_k(B) = ∫_bin ∫_S² C_W(B^{1/2} ρ u) ρ² dΩ dρ`` (quadrature; det B = 1).  ``ℓ`` is
    largest when the whitened radial pair profile is sharpest, which for an initially
    isotropic pattern happens at the true strain.  No cluster model is assumed (contrast R1);
    a Poisson pattern gives a flat likelihood, correctly.  Bins span whitened radii up to
    ``R / stretch_cap`` so the whitened ball stays inside the observed pair set for principal
    stretches up to ``stretch_cap``.
    """
    from scipy.optimize import minimize
    from ..observation import window_for
    from .ellipsoid import fibonacci_sphere
    W, _ = window_for(table, window)
    P = np.asarray(table.centroids, dtype=float)
    if len(P) < 30:
        raise ValueError("too few grains for the pair likelihood")
    tree = cKDTree(P)
    nn = float(np.median(tree.query(P, k=2)[0][:, 1]))
    R = float(min(W.half))
    ij = tree.query_pairs(R, output_type="ndarray")
    V = P[ij[:, 1]] - P[ij[:, 0]]
    if len(V) > max_pairs:
        rng = np.random.default_rng(0) if rng is None else rng
        V = V[rng.choice(len(V), max_pairs, replace=False)]
    rho_max = R / stretch_cap
    edges = np.r_[0.0, np.geomspace(0.15 * nn, rho_max, n_bins)]
    gl_x, gl_w = np.polynomial.legendre.leggauss(3)
    lo, hi = edges[:-1], edges[1:]
    rho_q = (0.5 * (hi - lo)[:, None] * gl_x[None, :] + 0.5 * (hi + lo)[:, None])        # (K,3)
    w_q = 0.5 * (hi - lo)[:, None] * gl_w[None, :] * rho_q ** 2                           # ρ² dρ
    U = fibonacci_sphere(n_dirs)
    dOmega = 4.0 * np.pi / n_dirs

    def neg_profile_ll(theta):
        w, Q = np.linalg.eigh(_sym_traceless(theta))
        if np.exp(w.max() / 2.0) > stretch_cap:
            return 1e12
        Wv = V @ ((Q * np.exp(-w / 2.0)) @ Q.T).T
        rho = np.sqrt(np.einsum("ij,ij->i", Wv, Wv))
        n_k = np.histogram(rho[rho < rho_max], bins=edges)[0].astype(float)
        Bh = (Q * np.exp(w / 2.0)) @ Q.T
        pts = (rho_q.reshape(-1, 1, 1) * (U @ Bh.T)[None, :, :]).reshape(-1, 3)          # ρ·B^{1/2}u
        C = W.set_covariance(pts).reshape(-1, n_dirs).sum(1) * dOmega
        E_k = (C.reshape(rho_q.shape) * w_q).sum(1)
        m = n_k > 0
        return -float(np.sum(n_k[m] * np.log(n_k[m] / E_k[m])) - n_k.sum() * np.log(n_k.sum() / E_k.sum()))

    starts = [np.zeros(5)]
    short = np.einsum("ij,ij->i", V, V) <= (3.0 * nn) ** 2
    if short.sum() > 30:
        wc, Qc = np.linalg.eigh(np.cov(V[short].T))
        lw = np.log(np.clip(wc, 1e-12, None)); lw -= lw.mean()
        for g in (1.0, 2.0):
            L = (Qc * (g * lw)) @ Qc.T
            starts.append(np.array([L[0, 0], L[1, 1], L[0, 1], L[0, 2], L[1, 2]]))
    x0 = min(starts, key=neg_profile_ll)                                   # best start, one search
    best = minimize(neg_profile_ll, x0, method="Powell", options={"xtol": 3e-3, "ftol": 1e-7, "maxfev": 700})
    w, Q = np.linalg.eigh(_sym_traceless(best.x))
    return _estimate(w, Q, "M14_sepl", {"neg_ll": float(best.fun), "nn": nn, "rho_max": rho_max,
                                        "n_pairs": int(len(V)),
                                        "constrained": bool(np.exp(w.max() / 2.0) > 0.98 * stretch_cap),
                                        "converged": bool(best.success)})
