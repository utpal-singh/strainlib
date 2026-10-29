"""Ranking statistics, method selection and calibration.

Consumes the benchmark estimate table (``results/benchmark/realisation=*.parquet``)
and produces every headline number for the paper from a results file.

This module was hardened against a 29-finding adversarial statistics review; the
non-obvious corrections (all in the analysis layer — the raw estimate table is
untouched) are:

* **Deployable calibration.**  The bias-correction is fit and
  scored as a function of the *observable* ``Rs_est`` (an inverse calibration
  ``Rs_true ≈ g(Rs_est)``), never of the ground-truth ``Rs_true`` — the latter is
  exactly the unknown on a real xenolith, so a truth-conditioned correction is an
  oracle that cannot be run in practice.  The oracle is still reported, explicitly
  labelled as an upper bound.
* **Variance-stabilised comparisons.**  Rs error grows ~linearly with Rs, so the
  invariance test uses the log-ratio ``ln(Rs_est/Rs_true)`` /
  relative error, and m\\* is ranked on *relative* squared error so the choice is
  not dominated by the Rs=5 cells.
* **Honest units of replication.**  The same 125 deformations are applied to every
  rock, so rows are massively pseudo-replicated.  The MCAR-invariance test and all
  calibration uncertainties are computed with the **rock (realisation) as the
  independent unit** (per-rock slopes; leave-one-rock-out CV; rock bootstrap), not
  per estimate row.  Invariance is judged by an **equivalence test** (the slope CI
  inside a margin), not by failing to reject zero.
* **No survivorship.**  m\\* is compared on the set of configs *all* candidate
  methods fit (paired), and each method's fit-failure rate is reported and gates
  the recommendation, so a method cannot win by declining the hard cases.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from ..methods import CENTRE_METHODS, REFERENCE_METHODS

__all__ = [
    "load_results", "add_derived", "rs_band", "in_rs_band", "RS_TOL", "scenario_family", "mcar_p",
    "accuracy_surface", "fit_failure_rates", "select_mstar",
    "fit_bias_correction", "test_mcar_invariance",
    "train_test_split_realisations", "leave_one_rock_out", "run_phase3",
]

#: the analysis covers the centre family only; R1 knows the generator's cluster model, so it is
#: ranked as a reference but can never be selected as m*
CENTRE_FAMILY, REFERENCE = CENTRE_METHODS, REFERENCE_METHODS
_MCAR_LEVELS = ["ideal", "mcar_p0.75", "mcar_p0.50", "mcar_p0.25", "mcar_p0.60"]

# Validity gates.  RS_TRUE_MAX excludes cells whose *ground truth* is degenerate — in
# sequential Fossen-Tikoff, Wk->1 forces gamma->inf (the solver caps at 1e6) and
# strain_ellipsoid then returns Rs_true ~ 1e11; those cells are an invalid experiment,
# not a method error.  RS_EST_IMPLAUSIBLE reclassifies a *finite but absurd* estimate
# (or non-finite) as a numerical breakdown = a method failure (folded into the fail
# rate, not silently dropped).  It is set high so that merely-bad-but-real estimates
# (the degradation the paper measures) are kept.
RS_TRUE_MAX = 50.0
RS_EST_IMPLAUSIBLE = 1.0e3

#: The realised Rs_true of a design cell differs from its nominal Rs by ~1e-12, on either side.
#: Band edges are compared with this tolerance so that a band such as [1.5, 5] keeps *every*
#: cell of its edge levels; an exact comparison kept a float-noise subset (24 of the 200 Rs=1.5
#: cells, 112 of the 200 Rs=5 cells, 120 of the 200 Rs=2 cells for "Rs >= 2").
RS_TOL = 1e-9


def in_rs_band(rs, lo: float, hi: float = np.inf, tol: float = RS_TOL) -> np.ndarray:
    """Boolean mask ``lo <= rs <= hi``, both edges widened by ``tol`` (see ``RS_TOL``)."""
    rs = np.asarray(rs, dtype=float)
    return (rs >= lo - tol) & (rs <= hi + tol)


# --------------------------------------------------------------------------- IO

def clean_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cells with degenerate ground truth and reclassify numerically-broken
    estimates as failures.  Records the counts on ``df.attrs`` for reporting."""
    df = df.copy()
    if "fail" not in df.columns:
        df["fail"] = pd.NA
    bad_truth = ~np.isfinite(df["Rs_true"]) | (df["Rs_true"] > RS_TRUE_MAX)
    n_bad_truth = int(bad_truth.sum())
    df = df[~bad_truth].copy()
    impl = df["fit_ok"] & (~np.isfinite(df["Rs_est"]) | (df["Rs_est"] > RS_EST_IMPLAUSIBLE)
                           | (df["Rs_est"] <= 0))
    n_impl = int(impl.sum())
    df.loc[impl, "fit_ok"] = False
    df.loc[impl, "fail"] = df.loc[impl, "fail"].where(df.loc[impl, "fail"].notna(), "implausible_Rs")
    df.attrs["n_excluded_bad_truth"] = n_bad_truth
    df.attrs["n_reclassified_implausible"] = n_impl
    return df


def load_results(results_dir: str, methods=None) -> pd.DataFrame:
    """Concatenate every ``benchmark/realisation=*.parquet`` under ``results_dir``
    (including split ``realisation=RR.partIofN.parquet`` files), optionally keep only
    ``methods``, apply the validity gates, then add derived error columns."""
    bench = os.path.join(results_dir, "benchmark")
    files = sorted(glob.glob(os.path.join(bench, "realisation=*.parquet")))
    if not files:
        raise FileNotFoundError(f"no realisation parquet in {bench!r}")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if methods is not None:
        df = df[df["method"].isin(list(methods))].reset_index(drop=True)
    cleaned = clean_estimates(df)
    out = add_derived(cleaned)
    out.attrs.update(cleaned.attrs)
    return out


# --------------------------------------------------------- derived quantities

def rs_band(rs) -> pd.Categorical:
    """Coarse Rs bands with **numeric-range** labels (not nominal-grid values, which
    are wrong once Wk>0 pushes the realised Rs_true off the nominal Rs)."""
    rs = np.asarray(rs, dtype=float)
    labels = np.where(rs < 1.75, "[1.0,1.75)",
                      np.where(rs < 3.5, "[1.75,3.5)", "[3.5,inf)"))
    return pd.Categorical(labels, categories=["[1.0,1.75)", "[1.75,3.5)", "[3.5,inf)"], ordered=True)


def scenario_family(scen: str) -> str:
    if scen in ("ideal", "mcar+res", "resolution"):
        return scen
    if scen.startswith("mcar_p"):
        return "mcar"
    if scen.startswith("mnar_"):
        return "mnar"
    return "other"


def mcar_p(scen: str) -> float:
    """Retention probability p for an ``mcar_pX.XX`` scenario; ideal -> 1.0; else NaN."""
    if scen == "ideal":
        return 1.0
    m = re.fullmatch(r"mcar_p(\d+\.\d+)", scen)
    return float(m.group(1)) if m else np.nan


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed / relative / log-ratio Rs error, Rs-band (by realised and nominal
    Rs), scenario family and MCAR p."""
    df = df.copy()
    if "fail" not in df.columns:
        df["fail"] = pd.NA
    err = df["Rs_est"] - df["Rs_true"]
    df["signed_err_Rs"] = err
    df["sq_err_Rs"] = err ** 2
    df["rel_signed_err_Rs"] = err / df["Rs_true"]
    df["rel_sq_err_Rs"] = df["rel_signed_err_Rs"] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(df["Rs_est"].to_numpy() / df["Rs_true"].to_numpy())
    df["log_ratio_Rs"] = lr                       # variance-stabilised, NaN if Rs_est<=0
    df["Rs_band"] = rs_band(df["Rs_true"])        # by realised (true) Rs — correct for error stats
    df["Rs_band_nom"] = rs_band(df["Rs_nom"])     # by design factor — balanced, for modelling
    df["scenario_family"] = df["scenario"].map(scenario_family)
    df["p_mcar"] = df["scenario"].map(mcar_p)
    return df


# ------------------------------------------------------------------ surfaces

def _agg_errors(g: pd.DataFrame) -> pd.Series:
    ok = g[g["fit_ok"]]
    n_ok = len(ok)
    se = ok["signed_err_Rs"].to_numpy()
    rse = ok["rel_signed_err_Rs"].to_numpy()
    k_defined = int(np.isfinite(ok["abs_err_k"].to_numpy()).sum()) if n_ok else 0
    theta = ok["theta_X_deg"].to_numpy() if n_ok else np.array([])
    n_oriented = int(np.isfinite(theta).sum())
    return pd.Series({
        "n": len(g), "n_ok": n_ok,
        "fit_fail_rate": 1.0 - n_ok / len(g) if len(g) else np.nan,
        "rmse_Rs": float(np.sqrt(np.mean(se ** 2))) if n_ok else np.nan,
        "bias_Rs": float(np.mean(se)) if n_ok else np.nan,
        "rel_rmse_Rs": float(np.sqrt(np.mean(rse ** 2))) if n_ok else np.nan,
        "rel_bias_Rs": float(np.mean(rse)) if n_ok else np.nan,
        "mae_Rs": float(np.mean(np.abs(se))) if n_ok else np.nan,
        "median_relerr_Rs": float(ok["rel_err_Rs"].median()) if n_ok else np.nan,
        "mae_k": float(ok["abs_err_k"].mean(skipna=True)) if n_ok else np.nan,
        "n_k_defined": k_defined, "n_k_undefined": n_ok - k_defined,
        "median_theta_X": float(np.nanmedian(theta)) if n_oriented else np.nan,
        "median_theta_Z": float(np.nanmedian(ok["theta_Z_deg"].to_numpy())) if n_oriented else np.nan,
        "n_oriented": n_oriented,
        "mean_runtime_s": float(ok["runtime_s"].mean()) if ("runtime_s" in ok and n_ok) else np.nan,
    })


def accuracy_surface(df: pd.DataFrame, by=("method", "scenario", "Rs_band")) -> pd.DataFrame:
    """RMSE / bias (absolute & relative) / MAE-k / orientation-error surface."""
    return (df.groupby(list(by), observed=True, dropna=False)
              .apply(_agg_errors, include_groups=False).reset_index())


def fit_failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(method, scenario) fit-failure rate and the failure-type breakdown."""
    rows = []
    for (method, scen), g in df.groupby(["method", "scenario"], observed=True):
        fails = g[~g["fit_ok"]]
        types = fails["fail"].value_counts(dropna=True).to_dict() if len(fails) else {}
        rows.append({"method": method, "scenario": scen, "n": len(g), "n_fail": len(fails),
                     "fail_rate": len(fails) / len(g) if len(g) else np.nan, "fail_types": types})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- method selection

def _paired_config_keys(sub: pd.DataFrame, methods) -> set:
    """(realisation, deformation) keys for which *every* candidate method fit_ok —
    so cross-method comparison is on a common config set (no survivorship)."""
    ok = sub[sub["fit_ok"]]
    counts = (ok.groupby(["realisation", "deformation"])["method"]
                .nunique())
    keys = counts[counts == len(set(methods))].index
    return set(map(tuple, keys.to_list()))


def select_mstar(df: pd.DataFrame, p_hat: float = 0.60, rs_lo: float = 1.5, rs_hi: float = 5.0,
                 tie_frac: float = 0.05, objective: str = "relative",
                 fail_rate_tol: float = 0.02, scenario: str | None = None,
                 reference=REFERENCE) -> dict:
    """m\\* = argmin error at the p_hat MCAR scenario over Rs in [rs_lo, rs_hi].

    ``objective='relative'`` ranks on relative RMSE (default; scale-free, so the
    choice is not dominated by the Rs=5 cells); ``'absolute'`` uses Rs-unit RMSE.
    The comparison is on the **paired** config set all methods fit, the 5%-tie band
    is on **RMSE** (the interpretable scale) broken toward lower |bias|, and each
    method's fit-failure rate is reported — a method whose failure rate exceeds the
    winner's by more than ``fail_rate_tol`` is flagged (cannot win by abstaining).
    ``scenario`` overrides the p_hat-derived scenario name (e.g. ``"ideal"`` for the
    p=1 endpoint) — used by the  detection-rate sensitivity sweep phase.
    ``reference`` methods are ranked (flagged in the table) but never selected: R1 fits
    the synthetic generator's own cluster model, which no real rock provides."""
    scen = scenario if scenario is not None else f"mcar_p{p_hat:.2f}"
    window = df[(df["scenario"] == scen) & in_rs_band(df["Rs_true"], rs_lo, rs_hi)]
    if window.empty:
        raise ValueError(f"no rows for scenario {scen!r} in Rs[{rs_lo},{rs_hi}]")
    methods = sorted(window["method"].unique())
    keys = _paired_config_keys(window, methods)
    if not keys:
        raise ValueError("no config fit by all candidate methods (cannot pair)")
    key = list(zip(window["realisation"], window["deformation"]))
    in_paired = np.array([k in keys for k in key])
    paired = window[in_paired & window["fit_ok"].to_numpy()]
    sqcol = "rel_sq_err_Rs" if objective == "relative" else "sq_err_Rs"
    bcol = "rel_signed_err_Rs" if objective == "relative" else "signed_err_Rs"
    fails = (window.groupby("method")["fit_ok"].apply(lambda s: 1.0 - s.mean()))
    tab = (paired.groupby("method")
                 .agg(mse=(sqcol, "mean"), bias=(bcol, "mean"),
                      abs_rmse=("sq_err_Rs", lambda s: float(np.sqrt(s.mean()))),
                      abs_bias=("signed_err_Rs", "mean"), n_paired=(sqcol, "size"))
                 .reset_index())
    tab["rmse"] = np.sqrt(tab["mse"])
    tab["fail_rate"] = tab["method"].map(fails).astype(float)
    tab = tab.sort_values("rmse").reset_index(drop=True)
    tab["reference"] = tab["method"].isin(list(reference))
    selectable = tab[~tab["reference"]]
    if selectable.empty:
        raise ValueError("only reference methods present — nothing to select")
    best_rmse = selectable["rmse"].min()
    contenders = selectable[selectable["rmse"] <= best_rmse * (1.0 + tie_frac)].copy()
    contenders["abs_of_bias"] = contenders["bias"].abs()
    mstar_row = contenders.sort_values("abs_of_bias").iloc[0]
    mstar = mstar_row["method"]
    winner_fail = float(mstar_row["fail_rate"])
    fail_flags = tab[(tab["fail_rate"] > winner_fail + fail_rate_tol)]["method"].tolist()
    # per-scenario robustness ranking (relative rmse), so the single-scenario choice is auditable
    robust = (df[in_rs_band(df["Rs_true"], rs_lo, rs_hi) & df["fit_ok"]]
                .groupby(["scenario", "method"])
                .agg(rel_rmse=("rel_sq_err_Rs", lambda s: float(np.sqrt(s.mean()))))
                .reset_index())
    return {"mstar": mstar, "scenario": scen, "objective": objective, "rs_range": (rs_lo, rs_hi),
            "ranking": tab, "tie_contenders": contenders.sort_values("abs_of_bias").reset_index(drop=True),
            "winner_fail_rate": winner_fail, "methods_worse_failrate": fail_flags,
            "n_paired_configs": len(keys), "robustness_by_scenario": robust}


# ------------------------------------------------------------- calibration

def train_test_split_realisations(df: pd.DataFrame, test_frac: float = 0.25):
    """Deterministic single split of realisation indices into train / held-out test
    (kept for backwards use; prefer :func:`leave_one_rock_out` for reported CV)."""
    realis = np.sort(df["realisation"].unique())
    n_test = max(1, int(round(len(realis) * test_frac)))
    test = set(realis[-n_test:].tolist())
    return [r for r in realis if r not in test], sorted(test)


def leave_one_rock_out(df: pd.DataFrame):
    """Yield (train_realisations, [test_realisation]) folds — the rock is the
    independent unit, so this is the honest cross-validation granularity."""
    realis = list(np.sort(df["realisation"].unique()))
    for r in realis:
        yield [x for x in realis if x != r], [r]


def _rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def fit_bias_correction(df: pd.DataFrame, mstar: str, p_hat: float = 0.60,
                        degree: int = 2, n_boot: int = 500, seed: int = 12345) -> dict:
    """**Deployable** inverse calibration for m\\* at the p_hat scenario: fit
    ``Rs_true ≈ poly(Rs_est)`` (a function of the *observable* estimate) and score
    it by leave-one-rock-out CV — held-out RMSE of ``poly(Rs_est) - Rs_true`` vs the
    raw ``Rs_est - Rs_true``.  Coefficient uncertainty from a rock-level bootstrap.
    A truth-conditioned reference ``signed_err ~ poly(Rs_true)`` is also reported as
    a diagnostic — it is *not* deployable (uses the unknown) and, being a plain bias
    subtraction rather than the MMSE predictor, is not an RMSE bound."""
    scen = f"mcar_p{p_hat:.2f}"
    base = df[(df["method"] == mstar) & (df["scenario"] == scen) & df["fit_ok"]].copy()
    if base["realisation"].nunique() < 3:
        return {"ok": False, "error": "need >=3 rocks for leave-one-out bias-correction"}
    before, after, oracle, per_fold = [], [], [], []
    for train, test in leave_one_rock_out(base):
        tr = base[base["realisation"].isin(train)]
        te = base[base["realisation"].isin(test)]
        if len(tr) < degree + 2 or te.empty:
            continue
        inv = np.poly1d(np.polyfit(tr["Rs_est"], tr["Rs_true"], degree))     # deployable
        orc = np.poly1d(np.polyfit(tr["Rs_true"], tr["signed_err_Rs"], degree))  # oracle
        eb = te["Rs_est"].to_numpy() - te["Rs_true"].to_numpy()
        ea = inv(te["Rs_est"].to_numpy()) - te["Rs_true"].to_numpy()
        eo = (te["Rs_est"].to_numpy() - orc(te["Rs_true"].to_numpy())) - te["Rs_true"].to_numpy()
        before += eb.tolist(); after += ea.tolist(); oracle += eo.tolist()
        per_fold.append({"test_rock": int(test[0]), "rmse_before": _rmse(eb),
                         "rmse_after": _rmse(ea), "rmse_truth_conditioned_ref": _rmse(eo)})
    if not per_fold:
        return {"ok": False, "error": "insufficient train/test rows for bias-correction"}
    pf = pd.DataFrame(per_fold)
    # rock bootstrap for deployable-coefficient uncertainty
    rng = np.random.default_rng(seed)
    rocks = np.sort(base["realisation"].unique())
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(rocks, size=len(rocks), replace=True)
        bs = pd.concat([base[base["realisation"] == r] for r in pick], ignore_index=True)
        if len(bs) >= degree + 2:
            boot.append(np.polyfit(bs["Rs_est"], bs["Rs_true"], degree))
    boot = np.array(boot)
    coef_se = boot.std(axis=0, ddof=1).tolist() if len(boot) > 2 else None
    return {"ok": True, "mstar": mstar, "scenario": scen, "degree": degree,
            "deployable_model": "Rs_true ~ poly(Rs_est)",
            "cv_rmse_before": _rmse(before), "cv_rmse_after": _rmse(after),
            "cv_rmse_truth_conditioned_ref": _rmse(oracle),
            "cv_rmse_after_se": float(pf["rmse_after"].std(ddof=1)) if len(pf) > 1 else np.nan,
            "coef_bootstrap_se": coef_se, "n_folds": len(pf), "per_fold": pf.to_dict("records")}


# ------------------------------------------------------- MCAR invariance

def test_mcar_invariance(df: pd.DataFrame, methods=None, margin: float = 0.05,
                         use_relative: bool = True) -> dict:
    """Test the prediction that MCAR (independent) thinning leaves estimators
    unbiased, treating the **rock as the independent unit** (no pseudoreplication):

    * per rock, per method, regress mean bias on the deletion fraction (1-p) across
      {ideal + 4 MCAR levels}; the per-rock slopes are the independent replicates;
    * report the mean slope with a rock-level CI (t, dof=n_rock-1);
    * judge invariance by an **equivalence test** — the slope CI must lie inside
      ±``margin`` (relative Rs bias per unit deletion).  "Not significant" alone is
      *not* treated as evidence of invariance;
    * also report RMSE growth with deletion (variance, not just bias).

    ``use_relative`` uses relative Rs bias (variance-comparable across Rs)."""
    bcol = "rel_signed_err_Rs" if use_relative else "signed_err_Rs"
    lv = df[df["scenario"].isin(_MCAR_LEVELS) & df["fit_ok"]].copy()
    lv = lv[np.isfinite(lv["p_mcar"])]
    lv["deletion"] = 1.0 - lv["p_mcar"]
    # collapse to one number per (method, rock, level) — the honest granularity
    cell = (lv.groupby(["method", "realisation", "p_mcar", "deletion"])
              .agg(bias=(bcol, "mean"), sq=("rel_sq_err_Rs" if use_relative else "sq_err_Rs", "mean"))
              .reset_index())
    per_level = (cell.groupby(["method", "p_mcar", "deletion"])
                     .agg(bias=("bias", "mean"), bias_se=("bias", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
                          rmse=("sq", lambda s: float(np.sqrt(s.mean()))), n_rock=("bias", "size"))
                     .reset_index())
    tests = []
    from scipy import stats as _st
    for m, gm in cell.groupby("method"):
        slopes = []
        for r, gr in gm.groupby("realisation"):
            if gr["deletion"].nunique() >= 2:
                s = np.polyfit(gr["deletion"], gr["bias"], 1)[0]
                slopes.append(s)
        slopes = np.asarray(slopes, dtype=float)
        if len(slopes) < 2:
            continue
        mean_s = float(slopes.mean()); se = float(slopes.std(ddof=1) / np.sqrt(len(slopes)))
        dof = len(slopes) - 1
        tcrit = float(_st.t.ppf(0.975, dof)) if dof > 0 else np.nan   # two-sided 5% (nonzero test + reported CI)
        tcrit_eq = float(_st.t.ppf(0.95, dof)) if dof > 0 else np.nan  # TOST at alpha=0.05 => 90% CI
        ci = (mean_s - tcrit * se, mean_s + tcrit * se)
        ci_eq = (mean_s - tcrit_eq * se, mean_s + tcrit_eq * se)
        sig_nonzero = bool(np.isfinite(tcrit) and abs(mean_s) > tcrit * se)
        equivalent = bool(np.isfinite(tcrit_eq) and ci_eq[0] > -margin and ci_eq[1] < margin)
        tests.append({"method": m, "mean_slope": mean_s, "slope_se": se,
                      "ci_low": ci[0], "ci_high": ci[1], "n_rock": len(slopes),
                      "sig_different_from_0": sig_nonzero, "invariance_supported": equivalent,
                      "equiv_alpha": 0.05, "is_point_pattern": m in (methods or CENTRE_FAMILY)})
    return {"margin": margin, "response": bcol, "per_level": per_level,
            "slope_tests": pd.DataFrame(tests)}


# ---------------------------------------------------------------- orchestrator

def run_phase3(results_dir: str, out_dir: str | None = None, p_hat: float = 0.60,
               objective: str = "relative", methods=CENTRE_FAMILY) -> dict:
    """Ranking statistics -> CSV/JSON in ``out_dir`` (default ``<results_dir>/ranking``).
    ``methods`` restricts the analysis (and so the paired m* comparison) to those ids —
    by default the centre family, so shape-estimator rows in a stored benchmark are
    ignored; ``None`` analyses every method present."""
    import json
    df = load_results(results_dir, methods=methods)
    if out_dir is None:
        out_dir = os.path.join(results_dir, "ranking")
    os.makedirs(out_dir, exist_ok=True)

    surf_ms = accuracy_surface(df, by=("method", "scenario"))
    fails = fit_failure_rates(df)
    # rank under BOTH objectives — 'relative' avoids the Rs=5 domination but mirror-biases
    # toward low Rs, so report both and flag disagreement rather than silently pick one.
    ms = select_mstar(df, p_hat=p_hat, objective=objective)
    other = "absolute" if objective == "relative" else "relative"
    ms_other = select_mstar(df, p_hat=p_hat, objective=other)
    mstar_agrees = bool(ms["mstar"] == ms_other["mstar"])
    bias = fit_bias_correction(df, ms["mstar"], p_hat=p_hat)

    surf_ms.to_csv(os.path.join(out_dir, "accuracy_by_method_scenario.csv"), index=False)
    fails.to_csv(os.path.join(out_dir, "fit_failure_rates.csv"), index=False)
    ms["ranking"].to_csv(os.path.join(out_dir, f"mstar_ranking_{objective}.csv"), index=False)
    ms_other["ranking"].to_csv(os.path.join(out_dir, f"mstar_ranking_{other}.csv"), index=False)
    ms["robustness_by_scenario"].to_csv(os.path.join(out_dir, "mstar_robustness_by_scenario.csv"), index=False)

    headline = {
        "n_rows": int(len(df)), "n_realisations": int(df["realisation"].nunique()),
        "methods": sorted(df["method"].unique()), "scenarios": sorted(df["scenario"].unique()),
        "n_excluded_bad_truth": int(df.attrs.get("n_excluded_bad_truth", 0)),
        "n_reclassified_implausible": int(df.attrs.get("n_reclassified_implausible", 0)),
        "overall_fit_fail_rate": float(1.0 - df["fit_ok"].mean()),
        "mstar": ms["mstar"], "mstar_scenario": ms["scenario"], "mstar_objective": objective,
        "mstar_ranking": ms["ranking"].to_dict("records"),
        "mstar_alt_objective": other, "mstar_alt": ms_other["mstar"],
        "mstar_alt_ranking": ms_other["ranking"].to_dict("records"),
        "mstar_objectives_agree": mstar_agrees,
        "mstar_n_paired_configs": ms["n_paired_configs"],
        "mstar_methods_worse_failrate": ms["methods_worse_failrate"],
        "mstar_reference_methods": sorted(set(REFERENCE) & set(df["method"].unique())),
        "bias_correction": bias,
    }
    with open(os.path.join(out_dir, "ranking_headline.json"), "w", encoding="utf-8") as fh:
        json.dump(headline, fh, indent=1, default=str)
    return headline
