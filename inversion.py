"""Strain by INVERSION — experimental (branch: strain-inversion).

Instead of a direct forward estimator (Fry void, orientation tensor, ...), pose
strain recovery as an inverse problem: find the deformation that best explains the
OBSERVED pattern under an explicit generative model that INCLUDES the three things
that break the forward estimators — spatial CLUSTERING, the FIXED scan window, and
incomplete DETECTION (``scripts/diagnostics/verify_bug_fixes.py`` shows why the
forward family fails: 3-D Fry is blind to strain in clustered rocks, and the 3-D CFM
reads the window outline).

Where the strain information is
-------------------------------
Inside a fixed window the pattern at lags much larger than the clusters is Poisson
and the window is not strained, so large-lag summaries carry almost no strain.  The
information sits at the CLUSTER scale: an affinely strained Thomas process is an
anisotropic Thomas process whose offspring scatter is N(0, σ²B), with B = F Fᵀ the
strain ellipsoid itself.

Routes
------
* :func:`anisotropic_thomas_fit` — model-based, no simulation: composite (pair)
  likelihood of the anisotropic Thomas pair-correlation, window handled exactly via
  its set covariance C_W.  Invariant to independent thinning (MCAR, resolution).
* :func:`simulate_bank` + :func:`abc_posterior` — simulation-based inference
  (rejection ABC): the forward pipeline is the simulator, observed through the SAME
  :func:`strainlib.observation.observe` model as the benchmark, so clustering, window
  and detection biases sit in the data and the simulations alike.  Summaries:
  :func:`short_range_summary` (cluster-scale) and :func:`default_summary`
  (quantile shells, large-lag — kept for comparison).
* :func:`second_moment_strain` — baseline; under-performs on clustered/windowed data.

Kinematic vorticity Wk is NOT identifiable from the finite-strain pattern of passive
grain centres (the same ellipsoid is reached at any Wk); posteriors for Wk should be
reported against the prior, never as a recovered value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import qmc

__all__ = ["second_moment_strain", "pair_anisotropy", "anisotropic_thomas_fit",
           "default_summary", "short_range_summary", "simulate_observation", "simulate_bank",
           "abc_posterior", "abc_strain_inversion", "ABCResult"]


# --------------------------------------------------------------- baseline
def _pairs(pts, max_pairs=120000, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    n = len(pts)
    i = rng.integers(0, n, max_pairs); j = rng.integers(0, n, max_pairs); m = i != j
    V = pts[i[m]] - pts[j[m]]
    return V, np.linalg.norm(V, axis=1)


def pair_anisotropy(pts, qlo=0.2, qhi=0.8, max_pairs=120000, rng=None):
    """Eigenvalues (desc) of the covariance of pair-difference vectors in the
    [qlo, qhi] norm-quantile shell.  Under affine strain of an isotropic pattern this
    covariance is ~F Fᵀ; clustering/window bias it toward isotropy (hence the baseline
    underestimates)."""
    V, nr = _pairs(pts, max_pairs, rng)
    lo, hi = np.quantile(nr, qlo), np.quantile(nr, qhi)
    sel = (nr >= lo) & (nr <= hi)
    C = (V[sel].T @ V[sel]) / max(int(sel.sum()), 1)
    w, Q = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return w[o], Q[:, o]


def second_moment_strain(pts, **kw) -> dict:
    """BASELINE inversion (documented to under-perform on clustered/windowed data)."""
    w, Q = pair_anisotropy(pts, **kw)
    semi = np.sqrt(np.clip(w, 1e-12, None))
    return {"Rs": float(semi[0] / semi[2]), "axes": Q, "note": "baseline; under-performs raw Fry on the hard case"}


# --------------------------------------------------------------- model-based route
def _sym_traceless(s):
    s11, s22, s12, s13, s23 = s
    return np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, -s11 - s22]])


def _ellipsoid_from_log(w):
    """Rs and Flinn k from the eigenvalues ``w`` of log B (stretches = e^{w/2}).

    Flinn k = (R_xy − 1)/(R_yz − 1) with R_xy = e^{(w₀−w₁)/2}, R_yz = e^{(w₁−w₂)/2}, the same
    definition as :func:`strainlib.methods.ellipsoid.ellipsoid_to_strain` and the truth
    (:func:`strainlib.deformation.tensors.strain_ellipsoid`).  Before 2026-10-17 this returned
    (w₀−w₁)/(w₁−w₂) = ln R_xy / ln R_yz, which is Ramsay's K, not Flinn k (audit error E2);
    :func:`_ramsay_K_from_log` keeps that quantity under its own name."""
    w = np.sort(w)[::-1]
    Rs = float(np.exp((w[0] - w[2]) / 2.0))
    Rxy, Ryz = np.exp((w[0] - w[1]) / 2.0), np.exp((w[1] - w[2]) / 2.0)
    k = float((Rxy - 1.0) / (Ryz - 1.0)) if abs(Ryz - 1.0) > 1e-9 else np.inf
    return Rs, k


def _ramsay_K_from_log(w):
    """Ramsay's K = ln R_xy / ln R_yz = (w₀−w₁)/(w₁−w₂) from the eigenvalues of log B."""
    w = np.sort(w)[::-1]
    return float((w[0] - w[1]) / (w[1] - w[2])) if w[1] - w[2] > 1e-12 else np.inf


def anisotropic_thomas_fit(pts, window, r_max=None, n_mc=4096, n_starts=4, seed=0) -> dict:
    """Strain ellipsoid by composite (pair) likelihood of a strained Thomas process.

    A volume-constant affine strain F maps an isotropic Thomas process (parent
    intensity κ, offspring scatter σ) to an anisotropic one with offspring scatter
    N(0, σ²B), B = FFᵀ (left Cauchy–Green tensor — the strain ellipsoid), so

        g(v) = 1 + c · exp(−vᵀB⁻¹v / 4σ²),      c = 1 / (κ (4πσ²)^{3/2}).

    Observed pair separations with |v| < r_max inside the fixed window W have density
    ∝ C_W(v) g(v), with C_W the window's set covariance.  The log composite likelihood
    Σ log g(v_ij) − N log ∫_{|v|<r_max} C_W g  is maximised over (B with det B = 1, σ, c):
    the clustering (σ, c) and the window are MODELLED, not ignored.  Independent
    thinning leaves g unchanged, so MCAR/resolution loss does not bias the estimate.

    The normaliser uses ∫ C_W g = I₀ + c (4πσ²)^{3/2} E[C_W(√2σB^{1/2}z)·1{|·|<r_max}],
    z ~ N(0, I) with fixed quasi-random draws (smooth in the parameters).
    """
    P = np.asarray(pts, dtype=float)
    n = len(P)
    if n < 30:
        raise ValueError("too few points for the pair likelihood")
    tree = cKDTree(P)
    nn = float(np.median(tree.query(P, k=2)[0][:, 1]))
    if r_max is None:
        r_max = float(min(10.0 * nn, 0.3 * 2.0 * min(window.half)))
    ij = tree.query_pairs(r_max, output_type="ndarray")
    V = P[ij[:, 1]] - P[ij[:, 0]]
    N = len(V)
    if N < 50:
        raise ValueError("too few pairs within r_max")

    # I0 = ∫_{|v|<r_max} C_W(v) dv  (QMC over the ball)
    U = qmc.Sobol(3, scramble=True, seed=seed).random(2 ** 17) * 2.0 - 1.0
    U = U[np.sum(U ** 2, axis=1) <= 1.0] * r_max
    I0 = (4.0 / 3.0) * np.pi * r_max ** 3 * float(np.mean(window.set_covariance(U)))
    Z = qmc.MultivariateNormalQMC(np.zeros(3), seed=seed).random(n_mc)

    def unpack(th):
        S = _sym_traceless(th[:5])
        w, Q = np.linalg.eigh(S)
        return w, Q, float(np.exp(th[5])), float(np.exp(th[6]))

    def nll(th):
        w, Q, sig, c = unpack(th)
        Binv = (Q * np.exp(-w)) @ Q.T
        q = np.einsum("ij,jk,ik->i", V, Binv, V)
        ll_pairs = np.sum(np.log1p(c * np.exp(-q / (4.0 * sig * sig))))
        Bh = (Q * np.exp(w / 2.0)) @ Q.T
        Vm = np.sqrt(2.0) * sig * Z @ Bh.T
        inside = np.sum(Vm ** 2, axis=1) <= r_max ** 2
        cl = c * (4.0 * np.pi * sig * sig) ** 1.5 * float(np.mean(window.set_covariance(Vm) * inside))
        return -(ll_pairs - N * np.log(I0 + cl))

    # starts: short-pair covariance shape (det-normalised), isotropic, and perturbations
    short = np.linalg.norm(V, axis=1) <= 3.0 * nn
    C = np.cov(V[short].T) if short.sum() > 20 else np.eye(3)
    wc, Qc = np.linalg.eigh(C)
    Lc = (Qc * (np.log(np.clip(wc, 1e-9, None)) - np.mean(np.log(np.clip(wc, 1e-9, None))))) @ Qc.T
    base = [np.array([Lc[0, 0], Lc[1, 1], Lc[0, 1], Lc[0, 2], Lc[1, 2]]), np.zeros(5)]
    rng = np.random.default_rng(seed)
    starts = base + [2.0 * base[0] + rng.normal(0, 0.3, 5) for _ in range(max(0, n_starts - 2))]
    bounds = [(-4.0, 4.0)] * 5 + [(np.log(0.05 * nn), np.log(r_max / 2.0)), (np.log(1e-3), np.log(1e7))]
    best = None
    for s0 in starts:
        th0 = np.concatenate([s0, [np.log(nn), np.log(10.0)]])
        res = minimize(nll, th0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    w, Q, sig, c = unpack(best.x)
    o = np.argsort(w)[::-1]
    Rs, k = _ellipsoid_from_log(w)
    return {"Rs": Rs, "k": k, "K_ramsay": _ramsay_K_from_log(w), "axes": Q[:, o], "log_eigs": w[o],
            "sigma": sig, "c": c,
            "kappa": 1.0 / (c * (4.0 * np.pi * sig * sig) ** 1.5),
            "r_max": r_max, "n_pairs": N, "nll": float(best.fun), "converged": bool(best.success)}


# --------------------------------------------------------------- simulation-based route
@dataclass
class ABCResult:
    Rs: float; Rs_ci: tuple; k: float; k_ci: tuple; Wk: float; Wk_sd: float
    n_accept: int; posterior: np.ndarray = field(repr=False)


def default_summary(pts, shells=((0.15, 0.45), (0.45, 0.75), (0.75, 0.95)),
                    max_pairs=80000, rng=None):
    """Large-lag anisotropy: log eigen-ratios of the pair covariance in quantile shells
    of ALL pair distances.  Inside a fixed window these lags are dominated by the
    (unstrained) window; kept as a comparison for :func:`short_range_summary`."""
    V, nr = _pairs(pts, max_pairs, rng)
    feats = []
    for lo, hi in shells:
        a, b = np.quantile(nr, lo), np.quantile(nr, hi)
        sel = (nr >= a) & (nr <= b)
        C = (V[sel].T @ V[sel]) / max(int(sel.sum()), 1)
        w = np.sort(np.linalg.eigvalsh(C))[::-1]
        feats += [np.log(w[0] / w[2]), np.log(w[1] / w[2])]
    return np.asarray(feats, float)


def short_range_summary(pts, edges=(0.0, 1.5, 3.0, 6.0, 12.0), rng=None):
    """Cluster-scale anisotropy: for lag shells in units of the median nearest-neighbour
    distance, the log eigen-ratios of the unit-vector orientation tensor of ALL pairs in
    the shell, plus log pair-count ratios between shells (clustering strength)."""
    P = np.asarray(pts, dtype=float)
    tree = cKDTree(P)
    nn = float(np.median(tree.query(P, k=2)[0][:, 1]))
    e = np.asarray(edges) * nn
    ij = tree.query_pairs(e[-1], output_type="ndarray")
    V = P[ij[:, 1]] - P[ij[:, 0]]
    d = np.linalg.norm(V, axis=1)
    feats, counts = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        sel = (d > lo) & (d <= hi)
        counts.append(max(int(sel.sum()), 1))
        if sel.sum() < 10:
            feats += [0.0, 0.0]
            continue
        U = V[sel] / d[sel, None]
        w = np.clip(np.sort(np.linalg.eigvalsh(U.T @ U / sel.sum()))[::-1], 1e-9, None)
        feats += [np.log(w[0] / w[2]), np.log(w[1] / w[2])]
    shell_vol = np.diff(e ** 3)
    dens = np.log(np.asarray(counts) / shell_vol)
    return np.asarray(feats + list(np.diff(dens)), float)


def simulate_observation(params, template_params, window, p_detect, rng, rotate=None):
    """Forward-model one (Rs, k, Wk): parent rock sized to the window's pre-image,
    deform, image through the FIXED ``window`` (:func:`strainlib.observation.observe`),
    then MCAR detection.  ``rotate`` (default: only for non-spherical windows) applies
    a uniformly random orientation — irrelevant when window and summaries are both
    rotation-invariant.  Returns (observed GrainTable, F)."""
    from .synthetic import build_synthetic_rock
    from .synthetic.shapes import random_rotation
    from .deformation.tensors import steady_state_deformation
    from .detection import mcar_thin
    from .observation import observe, parent_box
    Rs, k, Wk = params
    F = steady_state_deformation(Rs, k, Wk)
    if rotate is None:
        rotate = not (window.shape == "ellipsoid" and len(set(window.half)) == 1)
    if rotate:
        Q = random_rotation(rng)
        F = Q @ F @ Q.T
    pad = 3.0 * float(template_params.get("sigma", 0.0))
    rock = build_synthetic_rock(rng=rng, **dict(template_params, box=parent_box(window, [F], pad)))
    obs = observe(rock, F, window)
    if p_detect < 1.0:
        obs = mcar_thin(obs, p_detect, rng=rng)
    return obs, F


def simulate_bank(n_sim, template_params, window, *, p_detect=0.6, priors=None, seed=0,
                  summaries=None, progress=False) -> dict:
    """Reference table for rejection ABC, computed ONCE and reusable for any number of
    observations and summaries.  Priors: Rs ~ U, k ~ log-uniform, Wk ~ U.  Stores the
    nominal (Rs, k, Wk) AND the finite-ellipsoid (Rs, k) of each simulated F."""
    from .deformation.tensors import strain_ellipsoid
    rng = np.random.default_rng(seed)
    pri = priors or {"Rs": (1.0, 6.0), "k": (0.2, 5.0), "Wk": (0.0, 1.0)}
    summaries = summaries or {"short": short_range_summary, "default": default_summary}
    theta, S = [], {name: [] for name in summaries}
    failures = {}
    for i in range(n_sim):
        th = (rng.uniform(*pri["Rs"]), float(np.exp(rng.uniform(*np.log(pri["k"])))),
              rng.uniform(*pri["Wk"]))
        try:
            obs, F = simulate_observation(th, template_params, window, p_detect, rng)
            s = {name: f(obs.centroids, rng=rng) for name, f in summaries.items()}
        except Exception as exc:                           # noqa: BLE001
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
            continue
        if not all(np.all(np.isfinite(v)) for v in s.values()):
            failures["non-finite summary"] = failures.get("non-finite summary", 0) + 1
            continue
        se = strain_ellipsoid(F)
        theta.append((*th, se.Rs, se.k))
        for name in summaries:
            S[name].append(s[name])
        if progress and (i + 1) % max(1, n_sim // 10) == 0:
            print(f"  [bank] {i + 1}/{n_sim}", flush=True)
    return {"theta": np.asarray(theta), "cols": ("Rs_nom", "k_nom", "Wk", "Rs", "k"),
            "S": {name: np.asarray(v) for name, v in S.items()}, "priors": pri, "window": window,
            "p_detect": p_detect, "failures": failures}


def abc_posterior(bank, s_obs, summary="short", keep=0.05) -> ABCResult:
    """Rejection-ABC posterior: accept the ``keep`` fraction of simulations whose
    standardised summaries are closest to ``s_obs``.  Rs and k refer to the FINITE
    strain ellipsoid."""
    S = bank["S"][summary]
    scale = np.std(S, axis=0)
    scale[scale == 0] = 1.0
    dist = np.linalg.norm((S - np.asarray(s_obs)) / scale, axis=1)
    acc = dist <= np.quantile(dist, keep)
    post = bank["theta"][acc]
    Rs, k, Wk = post[:, 3], post[:, 4], post[:, 2]
    return ABCResult(Rs=float(np.median(Rs)), Rs_ci=tuple(np.percentile(Rs, [2.5, 97.5])),
                     k=float(np.exp(np.median(np.log(k)))), k_ci=tuple(np.percentile(k, [2.5, 97.5])),
                     Wk=float(np.mean(Wk)), Wk_sd=float(np.std(Wk)), n_accept=int(acc.sum()),
                     posterior=post)


def abc_strain_inversion(obs_pts, template_params, window, *, p_detect=0.6, n_sim=2000,
                         keep=0.05, priors=None, seed=0, summary="short") -> ABCResult:
    """One-shot convenience wrapper: build a bank, then the posterior for ``obs_pts``."""
    fns = {"short": short_range_summary, "default": default_summary}
    bank = simulate_bank(n_sim, template_params, window, p_detect=p_detect, priors=priors,
                         seed=seed, summaries={summary: fns[summary]})
    s_obs = fns[summary](np.asarray(obs_pts), rng=np.random.default_rng(seed + 1))
    return abc_posterior(bank, s_obs, summary, keep)
