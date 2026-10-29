"""Error metrics and the benchmarking harness.

Reproducibility: every stochastic unit of work draws from ``repro.child_rng``
keyed by config *identity* (deformation, scenario, method labels) — so results
are independent of evaluation order, subsetting, and parallelism for a fixed
``master_seed``.
"""
from __future__ import annotations

import inspect
import numpy as np

from ..deformation.tensors import StrainEllipsoid
from ..methods import METHODS
from ..methods.ellipsoid import StrainEstimate

__all__ = ["error_metrics", "benchmark"]


def _axis_angle(a, b):
    c = np.clip(abs(np.dot(a, b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def error_metrics(est: StrainEstimate, truth: StrainEllipsoid) -> dict:
    """Per-estimate error metrics against ground truth."""
    err = {
        "Rs_est": est.Rs, "Rs_true": truth.Rs,
        "rel_err_Rs": abs(est.Rs - truth.Rs) / truth.Rs,
        "k_est": est.k, "k_true": truth.k,
        "abs_err_k": abs(est.k - truth.k) if np.isfinite(est.k) and np.isfinite(truth.k) else np.nan,
    }
    if est.axes is not None:
        err["theta_X_deg"] = _axis_angle(est.axes[:, 0], truth.axes[:, 0])
        err["theta_Z_deg"] = _axis_angle(est.axes[:, 2], truth.axes[:, 2])
        # Hencky tensor of the estimate (Brandon 1995): lets estimates be pooled as TENSORS and
        # errors measured as distances in strain-magnitude space (analysis/magnitude_space.py)
        try:
            from .magnitude_space import hencky_from_estimate
            H = hencky_from_estimate(est)
            for name, (i, j) in {"H_xx": (0, 0), "H_yy": (1, 1), "H_zz": (2, 2),
                                 "H_xy": (0, 1), "H_xz": (0, 2), "H_yz": (1, 2)}.items():
                err[name] = float(H[i, j])
        except Exception:                                      # noqa: BLE001 — never fail an estimate
            pass
    else:
        err["theta_X_deg"] = err["theta_Z_deg"] = np.nan
    return err


def _accepts_rng(func) -> bool:
    try:
        return "rng" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def benchmark(rock, deformations, detection_fn, method_ids=None, method_kwargs=None,
              master_seed=None):
    """Run the benchmark → results DataFrame.

    ``deformations``: list of (label, F, StrainEllipsoid).  ``detection_fn(table, rng)``
    → {scenario_label: GrainTable}.  Streams are keyed by config identity so the run
    is order-/subset-/parallelism-independent for a fixed ``master_seed``.
    """
    import pandas as pd
    from ..deformation.tensors import apply_to_grains
    from ..repro import child_rng, DEFAULT_MASTER_SEED

    if master_seed is None:
        master_seed = DEFAULT_MASTER_SEED
    method_ids = method_ids or list(METHODS)
    method_kwargs = method_kwargs or {}
    accepts = {mid: _accepts_rng(METHODS[mid]) for mid in method_ids}

    rows = []
    for def_label, F, truth in deformations:
        deformed = apply_to_grains(rock, F)
        scenarios = detection_fn(deformed, child_rng(master_seed, def_label))
        for scen_label, observed in scenarios.items():
            for mid in method_ids:
                row = {"deformation": def_label, "scenario": scen_label, "method": mid,
                       "n_grains": observed.n}
                try:
                    kw = dict(method_kwargs.get(mid, {}))
                    if accepts[mid]:
                        kw.setdefault("rng", child_rng(master_seed, def_label, scen_label, mid))
                    est = METHODS[mid](observed, **kw)
                    row.update(error_metrics(est, truth))
                    row["fit_ok"] = True
                except Exception as exc:  # noqa: BLE001
                    row["fit_ok"] = False
                    row["fail"] = f"{type(exc).__name__}"
                rows.append(row)
    return pd.DataFrame(rows)
