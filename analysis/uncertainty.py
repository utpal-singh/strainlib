"""Phase 5 — uncertainty & internal validation.

Part 1 (this file): the **rock-level cluster bootstrap** and the **detection-rate
sensitivity** sweep, both computed from the finished Phase-2 estimate table.

The rock (``realisation``, n=8) is the only resampled unit: the 125-cell Rs×k×Wk
grid is a *fixed design factor* present in every rock (a crossed design), so its
cell-to-cell variation is part of the estimand "expected performance over the
design", not sampling noise to inflate.  A rock drawn twice in a replicate is
relabelled to a fresh cluster id so the paired-config and per-rock-slope logic
still sees distinct clusters.  The headline CIs are percentile intervals (BCa is
available for scalar statistics via :func:`bootstrap_statistic`); with only 8
clusters intervals are wide — that honesty is the point.

The rankings, P(win) and m* are over the centre family, with the reference R1 ranked
but never counted as a winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as _st

from . import phase3
from .phase3 import leave_one_rock_out, _rmse

__all__ = [
    "BootstrapCI", "cluster_bootstrap_frames", "percentile_interval", "bca_interval",
    "bootstrap_statistic", "bootstrap_mstar_ranking", "bootstrap_calibration",
    "bootstrap_invariance", "detection_sensitivity", "run_bootstrap",
]


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    se: float
    ci_method: str
    n_boot: int
    n_clusters: int
    note: str = ""

    def as_dict(self):
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "se": self.se,
                "ci_method": self.ci_method, "n_boot": self.n_boot,
                "n_clusters": self.n_clusters, "note": self.note}


# --------------------------------------------------------------- resampling

def _relabel_draw(df: pd.DataFrame, drawn: np.ndarray, col: str = "realisation") -> pd.DataFrame:
    """Concatenate the drawn clusters, giving each drawn copy a fresh unique id so a
    cluster sampled k times becomes k independent clusters (not one merged group)."""
    parts = []
    for new_id, rid in enumerate(drawn):
        g = df[df[col] == rid].copy()
        g[col] = new_id
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def cluster_bootstrap_frames(df: pd.DataFrame, n_boot: int, seed: int, col: str = "realisation"):
    """Yield ``n_boot`` cluster-bootstrap replicate frames (rocks resampled with
    replacement, relabelled to fresh ids)."""
    rng = np.random.default_rng(seed)
    rocks = np.sort(df[col].unique())
    n = len(rocks)
    for _ in range(n_boot):
        drawn = rng.choice(rocks, size=n, replace=True)
        yield _relabel_draw(df, drawn, col)


# --------------------------------------------------------------- intervals

def percentile_interval(boot: np.ndarray, alpha: float = 0.05):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) < 2:
        return np.nan, np.nan
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def bca_interval(theta_hat: float, boot: np.ndarray, jack: np.ndarray, alpha: float = 0.05):
    """Bias-corrected & accelerated interval; falls back to percentile when the
    acceleration/bias terms are undefined (returns ``fell_back`` flag)."""
    b = np.asarray(boot, float); b = b[np.isfinite(b)]
    if len(b) < 2:
        return np.nan, np.nan, {"fell_back": True}
    prop = (np.sum(b < theta_hat) + 0.5) / (len(b) + 1)
    z0 = _st.norm.ppf(np.clip(prop, 1e-6, 1 - 1e-6))
    j = np.asarray(jack, float); j = j[np.isfinite(j)]
    a = np.nan
    if len(j) >= 2:
        jbar = j.mean(); num = np.sum((jbar - j) ** 3); den = 6.0 * (np.sum((jbar - j) ** 2)) ** 1.5
        a = num / den if den > 0 else np.nan
    if not (np.isfinite(z0) and np.isfinite(a)):
        lo, hi = percentile_interval(b, alpha)
        return lo, hi, {"z0": z0, "a": a, "fell_back": True}
    zl, zu = _st.norm.ppf(alpha / 2), _st.norm.ppf(1 - alpha / 2)
    a1 = _st.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = _st.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    return float(np.quantile(b, a1)), float(np.quantile(b, a2)), {"z0": float(z0), "a": float(a), "fell_back": False}


def bootstrap_statistic(df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float], *,
                        n_boot: int = 2000, seed: int = 12345, ci_method: str = "bca",
                        alpha: float = 0.05) -> BootstrapCI:
    """Cluster-bootstrap CI for a scalar statistic ``stat_fn(df) -> float``."""
    theta = float(stat_fn(df))
    boot = np.array([_safe(stat_fn, rep) for rep in cluster_bootstrap_frames(df, n_boot, seed)])
    boot = boot[np.isfinite(boot)]
    note = ""
    if ci_method == "bca":
        jack = np.array([_safe(stat_fn, df[df.realisation.isin(tr)]) for tr, _ in leave_one_rock_out(df)])
        lo, hi, info = bca_interval(theta, boot, jack, alpha)
        if info.get("fell_back"):
            note = "bca->percentile (degenerate jackknife/z0)"
    else:
        lo, hi = percentile_interval(boot, alpha)
    se = float(np.std(boot, ddof=1)) if len(boot) > 1 else np.nan
    return BootstrapCI(theta, lo, hi, se, ci_method, len(boot), df["realisation"].nunique(), note)


def _safe(fn, df):
    try:
        v = fn(df)
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


# ------------------------------------------------------- headline bootstraps

def _per_method_relrmse(df: pd.DataFrame, p_hat: float, objective: str, rs_lo: float, rs_hi: float):
    """Per-method rel-RMSE on the paired window (select_mstar's ranking table), m* and the
    strict (lowest-RMSE) winner — both among the selectable, non-reference methods."""
    r = phase3.select_mstar(df, p_hat=p_hat, objective=objective, rs_lo=rs_lo, rs_hi=rs_hi)
    rk = r["ranking"].set_index("method")
    return rk["rmse"], r["mstar"], rk.loc[~rk["reference"], "rmse"].idxmin()


def bootstrap_mstar_ranking(df: pd.DataFrame, *, p_hat: float = 0.60, objective: str = "relative",
                            rs_lo: float = 1.5, rs_hi: float = 5.0, n_boot: int = 2000,
                            seed: int = 12345) -> dict:
    """Per-method rel-RMSE CIs + P(method wins) from the cluster bootstrap."""
    point, mstar0, strict0 = _per_method_relrmse(df, p_hat, objective, rs_lo, rs_hi)
    methods = list(point.index)
    reference = set(phase3.REFERENCE)
    boot = {m: [] for m in methods}
    win_strict = {m: 0 for m in methods}
    win_tie = {m: 0 for m in methods}
    n_ok = 0
    for rep in cluster_bootstrap_frames(df, n_boot, seed):
        try:
            rk, mstar, strict = _per_method_relrmse(rep, p_hat, objective, rs_lo, rs_hi)
        except Exception:
            continue
        n_ok += 1
        for m in methods:
            if m in rk.index:
                boot[m].append(rk[m])
        win_strict[strict] = win_strict.get(strict, 0) + 1
        win_tie[mstar] = win_tie.get(mstar, 0) + 1
    rows = []
    for m in methods:
        b = np.array(boot[m], float)
        lo, hi = percentile_interval(b)
        rows.append({"method": m, "rel_rmse": float(point[m]), "lo": lo, "hi": hi,
                     "se": float(np.std(b, ddof=1)) if len(b) > 1 else np.nan,
                     "reference": m in reference,
                     "p_win_strict": win_strict.get(m, 0) / n_ok if n_ok else np.nan,
                     "p_win_tierule": win_tie.get(m, 0) / n_ok if n_ok else np.nan})
    return {"mstar": mstar0, "n_boot_ok": n_ok, "table": pd.DataFrame(rows).sort_values("rel_rmse").reset_index(drop=True)}


def bootstrap_calibration(df: pd.DataFrame, mstar: str, *, p_hat: float = 0.60, degree: int = 2,
                          n_boot: int = 2000, seed: int = 12345) -> dict:
    """CIs on the deployable inverse-calibration polynomial coefficients (Rs_true~poly(Rs_est))."""
    scen = f"mcar_p{p_hat:.2f}"

    def coefs(d):
        m = d[(d.method == mstar) & (d.scenario == scen) & d.fit_ok]
        if len(m) < degree + 2:
            return None
        return np.polyfit(m.Rs_est, m.Rs_true, degree)

    c0 = coefs(df)
    boot = [coefs(rep) for rep in cluster_bootstrap_frames(df, n_boot, seed)]
    boot = np.array([c for c in boot if c is not None])
    out = []
    for i in range(degree + 1):
        lo, hi = percentile_interval(boot[:, i]) if len(boot) else (np.nan, np.nan)
        out.append({"coef": f"c{degree - i}", "point": float(c0[i]), "lo": lo, "hi": hi,
                    "se": float(np.std(boot[:, i], ddof=1)) if len(boot) > 1 else np.nan})
    return {"mstar": mstar, "degree": degree, "coefs": pd.DataFrame(out), "n_boot_ok": len(boot)}


def bootstrap_invariance(df: pd.DataFrame, *, margin: float = 0.05, n_boot: int = 2000,
                         seed: int = 12345) -> dict:
    """Cluster-bootstrap CIs on the per-method MCAR-invariance mean slope + bootstrap
    support P(|mean slope| supports equivalence)."""
    point = phase3.test_mcar_invariance(df, margin=margin)["slope_tests"].set_index("method")
    methods = list(point.index)
    boot = {m: [] for m in methods}
    for rep in cluster_bootstrap_frames(df, n_boot, seed):
        try:
            t = phase3.test_mcar_invariance(rep, margin=margin)["slope_tests"].set_index("method")
        except Exception:
            continue
        for m in methods:
            if m in t.index:
                boot[m].append(float(t.loc[m, "mean_slope"]))
    rows = []
    for m in methods:
        b = np.array(boot[m], float); b = b[np.isfinite(b)]
        lo, hi = percentile_interval(b)
        rows.append({"method": m, "mean_slope": float(point.loc[m, "mean_slope"]), "lo": lo, "hi": hi,
                     "p_within_margin": float(np.mean(np.abs(b) < margin)) if len(b) else np.nan,
                     "invariance_point": bool(point.loc[m, "invariance_supported"])})
    return {"margin": margin, "table": pd.DataFrame(rows)}


# ------------------------------------------------------- detection sensitivity

def detection_sensitivity(df: pd.DataFrame, *, p_levels=(0.25, 0.50, 0.60, 0.75, 1.0),
                          p_hat: float = 0.60, sigma_p: float = 0.06, objective: str = "relative",
                          rs_lo: float = 1.5, rs_hi: float = 5.0, degree: int = 2) -> dict:
    """Re-select m* and re-fit the inverse calibration at each detection level p, to
    show the conclusions are insensitive to the assumed field rate p̂ over p̂±σ.
    p=1.0 maps to the stored ``ideal`` scenario."""
    rows = []
    for p in p_levels:
        scen = "ideal" if p >= 0.999 else f"mcar_p{p:.2f}"
        if scen not in df.scenario.unique():
            continue
        ms = phase3.select_mstar(df, p_hat=p_hat, objective=objective, rs_lo=rs_lo, rs_hi=rs_hi,
                                 scenario=scen)
        rk = ms["ranking"].sort_values("rmse")
        rk = rk[~rk["reference"]].reset_index(drop=True)          # R1 never wins
        best, runner = rk.iloc[0], rk.iloc[1]
        # deployable inverse calibration of the winner at this scenario
        w = df[(df.method == best["method"]) & (df.scenario == scen) & df.fit_ok]
        w = w[phase3.in_rs_band(w.Rs_true, rs_lo, rs_hi)]
        pct = np.nan
        if len(w) > degree + 2:
            coef = np.poly1d(np.polyfit(w.Rs_est, w.Rs_true, degree))
            before = _rmse(w.Rs_est - w.Rs_true)
            after = _rmse(coef(w.Rs_est) - w.Rs_true)
            pct = 100 * (before - after) / before if before else np.nan
        rows.append({"p": p, "scenario": scen, "mstar": best["method"], "mstar_rel_rmse": float(best["rmse"]),
                     "mstar_bias": float(best["bias"]), "runner_up": runner["method"],
                     "margin_to_flip": float((runner["rmse"] - best["rmse"]) / best["rmse"]),
                     "calib_pct_improve": pct})
    tab = pd.DataFrame(rows)
    band = tab[(tab.p >= p_hat - sigma_p) & (tab.p <= p_hat + sigma_p)]
    stable = bool(tab.mstar.nunique() == 1)
    return {"table": tab, "p_hat": p_hat, "sigma_p": sigma_p, "mstar_stable": stable,
            "mstar_over_band": sorted(band.mstar.unique().tolist()) if len(band) else [],
            "mstar_over_all_p": sorted(tab.mstar.unique().tolist())}


def run_bootstrap(results_dir: str, *, n_boot: int = 2000, p_hat: float = 0.60, seed: int = 12345) -> dict:
    """Convenience: load results, run the rock-level bootstrap headlines + detection
    sensitivity, return a headline dict (does not write files)."""
    df = phase3.load_results(results_dir, methods=phase3.CENTRE_FAMILY)
    ranking = bootstrap_mstar_ranking(df, p_hat=p_hat, n_boot=n_boot, seed=seed)
    calib = bootstrap_calibration(df, ranking["mstar"], p_hat=p_hat, n_boot=n_boot, seed=seed)
    inv = bootstrap_invariance(df, n_boot=n_boot, seed=seed)
    sens = detection_sensitivity(df, p_hat=p_hat)
    return {"ranking": ranking, "calibration": calib, "invariance": inv,
            "detection_sensitivity": sens}
