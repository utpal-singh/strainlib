"""R1 — anisotropic-Thomas composite likelihood, as a benchmark method.

A thin :class:`StrainEstimate` wrapper around
:func:`strainlib.inversion.anisotropic_thomas_fit` so the model-based inversion can run
inside the factorial benchmark next to the classical estimators.  It is a REFERENCE
(the generator's own cluster model, so an optimistic ceiling on synthetic data), not
one of the paper's M1–M8 methods.
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, ellipsoid_to_strain

__all__ = ["estimate_strain_thomas_cl"]


def estimate_strain_thomas_cl(table: GrainTable, window=None) -> StrainEstimate:
    """Strain ellipsoid by anisotropic-Thomas pair likelihood.  The scan window is taken
    from ``window``, else ``table.meta['window']`` (set by :func:`observation.observe`),
    else the centroid bounding box."""
    from ..inversion import anisotropic_thomas_fit
    from ..observation import ScanWindow
    P = np.asarray(table.centroids, dtype=float)
    if window is None and (table.meta or {}).get("window"):
        window = ScanWindow(**table.meta["window"])
    if window is None:
        window = ScanWindow(tuple(np.ptp(P, axis=0) / 2.0), "box")
    fit = anisotropic_thomas_fit(P, window, seed=0)
    stretches = np.exp(np.asarray(fit["log_eigs"]) / 2.0)          # det B = 1 → unit product
    # Rs and Flinn k from the stretches, as for every other estimator (before 2026-10-17 the
    # reported k was Ramsay's K = ln Rxy / ln Ryz, scored against Flinn k: audit error E2)
    return ellipsoid_to_strain(stretches, method="R1_thomas_cl", axes=fit["axes"],
                               extra={"sigma": fit["sigma"], "kappa": fit["kappa"], "n_pairs": fit["n_pairs"],
                                      "converged": fit["converged"], "K_ramsay": fit["K_ramsay"]})
