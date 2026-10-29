"""Calibrate the synthetic generator to a real CT sample.

Fits the Thomas-process clustering (from g(r)), the log-normal volume and
grain-aspect-ratio distributions, and the hard-core distance, so a synthetic
rock statistically matches a given real :class:`GrainTable`.  This is the
data-driven step: the synthetic benchmark spans the conditions of the
rocks the method is applied to.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..schema import GrainTable
from ..analysis.spatial_stats import pair_correlation, fit_thomas_pcf, bounding_box
from .build import build_synthetic_rock

__all__ = ["calibrate_from_grains", "build_like"]


def calibrate_from_grains(gt: GrainTable, r_max: float | None = None,
                          pcf_edge: str = "translation") -> dict:
    """Fit synthetic-generator parameters to a real GrainTable.

    ``pcf_edge`` is passed to :func:`pair_correlation`: ``"translation"`` (edge-corrected
    g(r), the default since 2026-10-17) or ``"none"`` (the uncorrected g(r) behind the
    calibration of the 2026-09 benchmark, kept to reproduce it)."""
    C = gt.centroids
    box = bounding_box(C)
    lam = gt.n / (box[0] * box[1] * box[2])

    # spatial clustering: Thomas fit to g(r)
    try:
        r, g = pair_correlation(C, box, r_max=r_max, edge=pcf_edge)
        thom = fit_thomas_pcf(r, g, lam)
    except Exception:
        thom = {"kappa": lam / 10.0, "sigma": 0.1 * min(box), "mu": 10.0}

    # hard-core: robust minimum nearest-neighbour distance
    d, _ = cKDTree(C).query(C, k=2)
    hardcore = float(np.percentile(d[:, 1], 1))

    # log-normal volume
    v = gt.volume[gt.volume > 0]
    logv = np.log(v)

    # grain aspect ratios from equivalent-ellipsoid semi-axes
    a = gt.semi_axes()
    q1 = np.clip(a[:, 0] / a[:, 1], 1.0, None)
    q2 = np.clip(a[:, 1] / a[:, 2], 1.0, None)
    lq1, lq2 = np.log(q1), np.log(q2)

    return {
        "box": box,
        "kappa": thom["kappa"], "mu": thom["mu"], "sigma": thom["sigma"],
        "hardcore": hardcore,
        "log_volume_mean": float(logv.mean()), "log_volume_std": float(logv.std()),
        "q1_logmean": float(lq1.mean()), "q1_logstd": float(max(lq1.std(), 1e-3)),
        "q2_logmean": float(lq2.mean()), "q2_logstd": float(max(lq2.std(), 1e-3)),
        "sample_id": gt.sample_id, "locality": gt.locality, "length_unit": gt.length_unit,
    }


def build_like(gt: GrainTable, rng=None, **overrides) -> GrainTable:
    """Build a synthetic rock statistically matched to ``gt``."""
    params = calibrate_from_grains(gt)
    params.update(overrides)
    out = build_synthetic_rock(rng=rng, **params)
    out.meta["calibrated_to"] = gt.sample_id
    return out
