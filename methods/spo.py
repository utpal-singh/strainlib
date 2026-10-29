"""M5 — Shape Preferred Orientation via the orientation tensor (Woodcock 1977).

The orientation tensor of grain long-axis vectors gives the fabric directions
(lineation = X, foliation pole = Z) and intensity (Woodcock C, Γ).  The strain
magnitude Rs is obtained from the mean grain aspect ratio through a calibration
exponent ``alpha`` that is fit empirically on the synthetic experiments
(:func:`strainlib.analysis.calibrate_spo_alpha`) — it is a fit parameter,
not a placeholder.
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, stretches_from_Rs_k

__all__ = ["orientation_tensor", "woodcock_parameters", "estimate_fabric_spo"]


def orientation_tensor(unit_vectors):
    """Second-order orientation tensor → (eigenvalues desc summing to 1, eigenvectors)."""
    U = np.asarray(unit_vectors, dtype=float)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    T = (U[:, :, None] * U[:, None, :]).mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(T)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def woodcock_parameters(eigvals):
    """Woodcock (1977) fabric strength C = ln(τ1/τ3) and shape Γ."""
    t1, t2, t3 = np.clip(eigvals, 1e-12, None)
    C = float(np.log(t1 / t3))
    denom = np.log(t2 / t3)
    Gamma = np.inf if abs(denom) < 1e-12 else float(np.log(t1 / t2) / denom)
    return C, Gamma


def estimate_fabric_spo(table: GrainTable, alpha: float = 1.0) -> StrainEstimate:
    """SPO fabric/strain estimate from grain long axes.

    ``Rs = mean_aspect_ratio ** alpha`` (geometric-mean a/c); ``k`` from the fabric
    shape Γ; principal directions from the orientation-tensor eigenvectors.
    """
    if not table.has_shapes:
        raise ValueError("SPO requires grain shapes (eigenvalues & eigenvectors)")
    eigvals, eigvecs = orientation_tensor(table.long_axis_vectors())
    C, Gamma = woodcock_parameters(eigvals)

    a = table.semi_axes()
    ar = a[:, 0] / a[:, 2]                                 # a/c per grain
    mean_ar = float(np.exp(np.mean(np.log(np.clip(ar, 1e-9, None)))))
    Rs = mean_ar ** alpha
    k = Gamma
    s = stretches_from_Rs_k(Rs, k)
    return StrainEstimate(
        method="M5_spo", Rs=Rs, k=k, stretches=s, axes=eigvecs,
        extra={"woodcock_C": C, "woodcock_Gamma": Gamma,
               "orientation_eigs": eigvals, "mean_aspect_ratio": mean_ar, "alpha": alpha},
    )
