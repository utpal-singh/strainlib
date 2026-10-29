"""Strain-magnitude space: invariants, tensor averaging and error decomposition.

After Brandon (1995), *Analysis of geologic strain data in strain-magnitude space*,
J. Struct. Geol. 17, 1375–1385.  Strain-magnitude space has the natural (logarithmic)
principal strains as orthogonal axes, so

* total-strain magnitude  ``E_s = ‖(Eₓ, E_y, E_z)‖``,
* volume strain           ``E_v = Eₓ + E_y + E_z = ln S_v`` (the octahedral normal),
* deviatoric strain       ``E_d = ‖E′‖``, ``E′ = E − E_v/3`` (the octahedral/deviatoric section),

and ``E_s² = E_v²/3 + E_d²``.  The Flinn and Nadai diagrams are sections of this space;
Brandon prefers the Nadai diagram as the undistorted view of the deviatoric section
(Appendix B gives its log–log construction, :func:`nadai_coords`).

Two results from the paper drive the functions here.

1. **Average strain tensors, not strain magnitudes** (Appendix A).  Each measurement is
   converted to its Hencky tensor ``H = T diag(Eₓ, E_y, E_z) Tᵀ`` and the average is taken
   component by component; eigen-decomposition of ``AVE(H)`` gives the average principal
   strains and directions.  Volume strain averages as ``AVE(E_v)``, i.e. the logarithmic
   mean of volume stretches.  Brandon's numerical tests: unbiased when the average
   deformation is coaxial (even if individual samples are not); when it is non-coaxial the
   principal directions stay accurate but the stretches are biased, "usually by no more than
   10–20 %" (p. 1384), since the averaging ignores the rotation R in F = V R.
2. **Scalar averaging inflates deviatoric magnitude**: "the deviatoric strain magnitude
   indicated by the tensor average will always lie on the low side of the distribution unless
   the individual strain tensors all have the same orientation" (p. 1382) — the vector-vs-
   magnitude averaging analogy.  :func:`average_hencky` is therefore the correct way to pool
   estimates over rocks, replicate scans or scenarios.

:func:`tensor_error` reports estimation error as a distance in this space, which keeps
magnitude, shape and orientation in one invariant measure and avoids the unbounded Flinn-k
error (k → ∞ as the intermediate axis approaches either extreme).
"""
from __future__ import annotations

import numpy as np

__all__ = ["hencky", "hencky_from_estimate", "average_hencky", "strain_from_hencky",
           "magnitude_coords", "nadai_coords", "tensor_error"]


def hencky(stretches, axes) -> np.ndarray:
    """Hencky (natural) strain tensor ``H = T diag(ln S) Tᵀ`` from principal stretches
    (descending) and their directions (columns of ``axes``)."""
    S = np.asarray(stretches, dtype=float)
    T = np.asarray(axes, dtype=float)
    if S.shape != (3,) or T.shape != (3, 3):
        raise ValueError("stretches must be (3,) and axes (3, 3) with principal directions as columns")
    return (T * np.log(S)) @ T.T


def hencky_from_estimate(est) -> np.ndarray:
    """Hencky tensor of a :class:`~strainlib.methods.ellipsoid.StrainEstimate` (or any object
    with ``stretches`` and ``axes``)."""
    if est.axes is None:
        raise ValueError(f"{getattr(est, 'method', 'estimate')} reports no principal directions")
    return hencky(est.stretches, est.axes)


def average_hencky(Hs) -> np.ndarray:
    """Brandon's tensor average: the component-by-component mean of Hencky tensors."""
    H = np.asarray(Hs, dtype=float)
    if H.ndim != 3 or H.shape[1:] != (3, 3):
        raise ValueError("expected a stack of (3, 3) Hencky tensors")
    return H.mean(0)


def strain_from_hencky(H):
    """→ (stretches desc, axes with directions as columns).  Inverse of :func:`hencky`."""
    w, V = np.linalg.eigh(np.asarray(H, dtype=float))
    o = np.argsort(w)[::-1]
    return np.exp(w[o]), V[:, o]


def magnitude_coords(H) -> dict:
    """Strain-magnitude-space invariants of a Hencky tensor: total ``E_s``, volume ``E_v``,
    deviatoric ``E_d``, Lode's parameter ``nu`` and the Nadai (deviatoric-section) angle in
    degrees, plus the axial ratios ``Rxz``/``Ryz`` and ``S_v``."""
    E = np.sort(np.linalg.eigvalsh(np.asarray(H, dtype=float)))[::-1]
    Ev = float(E.sum())
    Ed_vec = E - Ev / 3.0
    Ed = float(np.linalg.norm(Ed_vec))
    denom = E[0] - E[2]
    nu = float((2.0 * E[1] - E[0] - E[2]) / denom) if denom > 1e-12 else 0.0
    return {"E_s": float(np.linalg.norm(E)), "E_v": Ev, "E_d": Ed, "lode_nu": nu,
            "nadai_deg": float(np.degrees(np.arctan(nu / np.sqrt(3.0)))),
            "Rxz": float(np.exp(E[0] - E[2])), "Ryz": float(np.exp(E[1] - E[2])), "S_v": float(np.exp(Ev))}


def nadai_coords(H):
    """Brandon's Appendix-B coordinates for the log–log Nadai diagram: ``(Rxz, S'_y)`` with
    ``S'_y = exp(E'_y)`` the deviatoric intermediate stretch.  Plot on equal-length log axes;
    the accessible sector's borders are ``E_Rxz = −3E'_y`` and ``E_Rxz = 3E'_y``."""
    E = np.sort(np.linalg.eigvalsh(np.asarray(H, dtype=float)))[::-1]
    return float(np.exp(E[0] - E[2])), float(np.exp(E[1] - E.sum() / 3.0))


def tensor_error(H_est, H_true) -> dict:
    """Estimation error as a distance in strain-magnitude space.

    ``dE_total`` = ‖H_est − H_true‖_F (natural-strain units, orientation included);
    ``dE_d`` = error in deviatoric magnitude (signed: positive = overestimate);
    ``dE_v`` = volume-strain error; ``d_nadai_deg`` = difference in Nadai angle (strain SHAPE,
    bounded, unlike Flinn k); ``theta_X_deg`` = angle between the two maximum-stretch axes.
    """
    He, Ht = np.asarray(H_est, dtype=float), np.asarray(H_true, dtype=float)
    ce, ct = magnitude_coords(He), magnitude_coords(Ht)
    _, Ve = strain_from_hencky(He)
    _, Vt = strain_from_hencky(Ht)
    return {"dE_total": float(np.linalg.norm(He - Ht)),
            "dE_d": ce["E_d"] - ct["E_d"], "dE_v": ce["E_v"] - ct["E_v"],
            "d_nadai_deg": ce["nadai_deg"] - ct["nadai_deg"],
            "theta_X_deg": float(np.degrees(np.arccos(min(1.0, abs(float(Ve[:, 0] @ Vt[:, 0])))))),
            "E_d_est": ce["E_d"], "E_d_true": ct["E_d"]}
