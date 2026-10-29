"""Strain-analysis methods + the ``METHODS`` registry.

The paper's scope is the **centre family**: estimators that read only grain centroids.
M1 Fry · M2 CFM · M3 DTM · M4 MLM · M6 2D-Fry×3 · M7 2D-CFM×3 (+ the branch additions
M9, M9w, R1, M10, M12, M14, M17, M18 in ``METHODS_ALL``).  (M8 GLD reads a gray volume
— see methods/gld.py.)  Estimators that read grain *shapes* (M5 SPO and its variants
M11, M15, M16, M19) are kept in the library as ``SHAPE_METHODS`` but are not registered
for the benchmark or any analysis.
Every callable takes a :class:`~strainlib.schema.GrainTable` and returns a
:class:`StrainEstimate`; those that sample accept an ``rng`` for reproducibility.
"""
from __future__ import annotations

import numpy as np

from .ellipsoid import (StrainEstimate, fit_ellipsoid, fit_ellipse_2d, ellipsoid_to_strain,
                        fibonacci_sphere, void_shell_points, void_shell_2d, stretches_from_Rs_k)
from .fry import estimate_strain_fry, fry_vectors
from .cfm import estimate_strain_cfm, estimate_strain_cfm_ww
from .thomas_cl import estimate_strain_thomas_cl
from .dtm import estimate_strain_dtm, dtm_vectors
from .mlm import estimate_strain_mlm
from .spo import estimate_fabric_spo, orientation_tensor, woodcock_parameters
from .sections2d import estimate_strain_sections
from .gld import estimate_strain_gld, gld_directional_deviation

__all__ = ["StrainEstimate", "fit_ellipsoid", "fit_ellipse_2d", "ellipsoid_to_strain",
           "fibonacci_sphere", "void_shell_points", "void_shell_2d", "stretches_from_Rs_k",
           "fry_vectors", "dtm_vectors", "orientation_tensor", "woodcock_parameters",
           "estimate_strain_fry", "estimate_strain_cfm", "estimate_strain_cfm_ww",
           "estimate_strain_thomas_cl", "estimate_strain_dtm",
           "estimate_strain_mlm", "estimate_fabric_spo", "estimate_strain_sections",
           "estimate_strain_gld", "gld_directional_deviation",
           "METHODS", "METHODS_ALL", "SHAPE_METHODS", "CENTRE_METHODS", "REFERENCE_METHODS",
           "run_method", "bootstrap_strain"]


def _m6_fry2d(table, rng=None, **kw):
    return estimate_strain_sections(table, mode="fry", method_name="M6_fry2d", rng=rng, **kw)


def _m7_cfm2d(table, rng=None, **kw):
    return estimate_strain_sections(table, mode="cfm", method_name="M7_cfm2d", rng=rng, **kw)


def _m9w_cfm_ww(table, rng=None, **kw):
    """M9 with the limiting radius at the full smallest half-window (f_window = 1.0).
    Sensitivity variant, added after the smoke test showed M9 (f_window = 0.6) pinned at
    its size limit for Rs = 5 in the no21-sized window; everything else identical."""
    kw.setdefault("f_window", 1.0)
    return estimate_strain_cfm_ww(table, rng=rng, **kw)


#: centre methods of the original factorial (M8 GLD is excluded here — it needs a
#: rasterised gray volume and is run only on a subset, Decision D3).
METHODS = {
    "M1_fry": estimate_strain_fry,
    "M2_cfm": estimate_strain_cfm,
    "M3_dtm": estimate_strain_dtm,
    "M4_mlm": estimate_strain_mlm,
    "M6_fry2d": _m6_fry2d,
    "M7_cfm2d": _m7_cfm2d,
}

#: estimators that read grain shapes, not positions: out of the paper's scope (the synthetic
#: grains are passive ellipsoids, so these are handed the answer by construction).  Kept for
#: explicit use; never run by the benchmark or the analyses.
SHAPE_METHODS = {"M5_spo": estimate_fabric_spo}

#: all eight methods (M8 GLD included) — for the subset validation of M8 — plus, on the
#: strain-inversion branch, M9 (the published Waldron & Wallace 2007 CFM, outer/cluster
#: edge, window-corrected) and R1 (anisotropic-Thomas likelihood reference).
METHODS_ALL = {**METHODS, "M8_gld": estimate_strain_gld,
               "M9_cfm_ww": estimate_strain_cfm_ww, "M9w_cfm_ww": _m9w_cfm_ww,
               "R1_thomas_cl": estimate_strain_thomas_cl}


def _register_extra():
    """Second-pass estimators (literature + new), imported lazily-safe at module load:
    M10 normalized Fry (Erslev 1988); M11 affine-invariant mean shape tensor (3-D Yamaji-type
    SPO, calibration-free); M12 isotropy restoration; M14 semiparametric elliptical pair
    likelihood.  (M13, self-consistent Delaunay, was dropped: not identifiable at n ~ 10³.)"""
    from .nfry import estimate_strain_nfry
    from .shape_mean import estimate_strain_spo_rmean, estimate_strain_spo_hencky, estimate_strain_spo_yamaji
    from .isotropy import estimate_strain_isotropy, estimate_strain_sepl
    from .mulchrone import estimate_strain_dtm_ls, estimate_strain_void_fit
    from .star_fabric import estimate_strain_star_length
    METHODS_ALL.update({"M10_nfry": estimate_strain_nfry,
                        "M12_isotropy": estimate_strain_isotropy, "M14_sepl": estimate_strain_sepl})
    # knowledge-base survey (literature methods as published, 3-D): Mulchrone (2013) void fit,
    # Mulchrone (2003) Delaunay regression
    METHODS_ALL.update({"M17_void_fit": estimate_strain_void_fit, "M18_dtm_ls": estimate_strain_dtm_ls})
    # shape estimators of the same passes — Karcher mean, Hencky mean (Brandon 1995), Yamaji
    # (2008) hyperbolic mean, Ketcham & Ryan (2004) star-length fabric — out of scope
    SHAPE_METHODS.update({"M11_spo_rmean": estimate_strain_spo_rmean, "M15_spo_hencky": estimate_strain_spo_hencky,
                          "M16_spo_yamaji": estimate_strain_spo_yamaji, "M19_star_length": estimate_strain_star_length})


_register_extra()

#: the 13 centre estimators of the corrected benchmark, in presentation order
CENTRE_METHODS = ("M1_fry", "M2_cfm", "M3_dtm", "M6_fry2d", "M7_cfm2d", "M9_cfm_ww", "M9w_cfm_ww",
                  "R1_thomas_cl", "M10_nfry", "M12_isotropy", "M14_sepl", "M17_void_fit", "M18_dtm_ls")
#: fits the synthetic generator's own cluster model: an optimistic reference, ranked but never
#: selected as m*
REFERENCE_METHODS = ("R1_thomas_cl",)


def run_method(method_id: str, table, **kwargs) -> StrainEstimate:
    if method_id not in METHODS_ALL:
        raise KeyError(f"unknown method '{method_id}'. Available: {list(METHODS_ALL)}")
    return METHODS_ALL[method_id](table, **kwargs)


def bootstrap_strain(method_id: str, table, n_boot: int = 200, rng=None, **kwargs) -> dict:
    """Bootstrap a method over resampled grains → Rs/k summary stats.

    Pass an explicit ``rng`` (e.g. ``repro.child_rng(...)``) for reproducible CIs.
    """
    rng = np.random.default_rng() if rng is None else rng
    func = METHODS[method_id]
    Rs, k = [], []
    for _ in range(n_boot):
        boot = table.subset(rng.integers(0, table.n, size=table.n))
        try:
            est = func(boot, **kwargs)
        except Exception:
            continue
        if np.isfinite(est.Rs):
            Rs.append(est.Rs)
        if np.isfinite(est.k):
            k.append(est.k)
    Rs, k = np.array(Rs), np.array(k)
    return {"Rs_median": float(np.median(Rs)) if Rs.size else np.nan,
            "Rs_ci": (float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))) if Rs.size else (np.nan, np.nan),
            "k_median": float(np.median(k)) if k.size else np.nan,
            "n_success": int(Rs.size)}
