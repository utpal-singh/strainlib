"""M11 — calibration-free shape-fabric strain: the affine-invariant (Karcher) mean of the
grain shape tensors.  Branch strain-inversion.

Principle
---------
A passive grain's shape tensor transforms by congruence, ``M → F M Fᵀ``.  On the manifold
of symmetric positive-definite matrices the affine-invariant Riemannian distance
``d(A, B) = ‖log(A^{-1/2} B A^{-1/2})‖_F`` is invariant under every congruence, so the
Karcher (Fréchet) mean — the minimiser of Σ d²(X, Mᵢ) — is equivariant:
``mean(F Mᵢ Fᵀ) = F · mean(Mᵢ) · Fᵀ`` (Moakher 2005).  If the undeformed grains had
statistically isotropic orientations, their volume-normalised mean is the identity, so the
mean after deformation is exactly ``B = F Fᵀ`` — independent of the grains' initial
aspect-ratio distribution.  No empirical calibration law is needed (contrast M5, whose
``Rs = c·ā^α`` must be fitted).  In 2-D this is the hyperbolic-geometry mean of strain
ellipses (Yamaji 2008); here it is taken in 3-D.

Assumptions: grains deform as passive markers (in the synthetic benchmark they do by
construction; rigid or competent spinel does not), and their initial orientations were
isotropic.  Random, resolution and beam-shadow detection losses are independent of grain
orientation and do not bias the mean.
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .ellipsoid import StrainEstimate, ellipsoid_to_strain

__all__ = ["karcher_mean_spd", "estimate_strain_spo_rmean", "estimate_strain_spo_hencky",
           "estimate_strain_spo_yamaji"]


def _sym_fn(M, fn):
    w, V = np.linalg.eigh(M)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def karcher_mean_spd(Ms, tol: float = 1e-10, max_iter: int = 100):
    """Affine-invariant Karcher mean of a stack of SPD matrices ``(n, d, d)``.
    Fixed point ``X ← X^{1/2} exp(mean log(X^{-1/2} Mᵢ X^{-1/2})) X^{1/2}``, started at
    the log-Euclidean mean.  Returns ``(X, n_iter)``."""
    Ms = np.asarray(Ms, dtype=float)
    X = _sym_fn(_sym_fn(Ms, np.log).mean(0), np.exp)
    for it in range(1, max_iter + 1):
        w, V = np.linalg.eigh(X)
        Xh = (V * np.sqrt(w)) @ V.T
        Xih = (V / np.sqrt(w)) @ V.T
        S = _sym_fn(Xih @ Ms @ Xih, np.log).mean(0)
        X = Xh @ _sym_fn(S, np.exp) @ Xh
        X = 0.5 * (X + X.T)
        if np.linalg.norm(S) < tol:
            return X, it
    return X, max_iter


def estimate_strain_spo_rmean(table: GrainTable) -> StrainEstimate:
    """Strain ellipsoid = affine-invariant mean of volume-normalised grain shape tensors."""
    if not table.has_shapes or table.n < 10:
        raise ValueError("M11 needs grain shapes for at least 10 grains")
    M = table.shape_tensors()
    det = np.linalg.det(M)
    ok = np.isfinite(det) & (det > 0)
    if ok.sum() < 10:
        raise ValueError("too few non-degenerate grain shapes")
    M = M[ok] / np.cbrt(det[ok])[:, None, None]
    X, n_iter = karcher_mean_spd(M)
    w, V = np.linalg.eigh(X)
    o = np.argsort(w)[::-1]
    return ellipsoid_to_strain(np.sqrt(w[o]), method="M11_spo_rmean", axes=V[:, o],
                               extra={"n_grains": int(ok.sum()), "n_iter": n_iter})


# --------------------------------------------------------------------------- M15 / M16
def _unit_shape_tensors(table: GrainTable):
    if not table.has_shapes or table.n < 10:
        raise ValueError("shape-tensor means need grain shapes for at least 10 grains")
    M = table.shape_tensors()
    det = np.linalg.det(M)
    ok = np.isfinite(det) & (det > 0)
    if ok.sum() < 10:
        raise ValueError("too few non-degenerate grain shapes")
    return M[ok] / np.cbrt(det[ok])[:, None, None], int(ok.sum())


def estimate_strain_spo_hencky(table: GrainTable) -> StrainEstimate:
    """M15 — Hencky (logarithmic) tensor average of grain shapes (Brandon 1995, App. A;
    applied to CT-measured spinel by Chatzaras et al. 2021).

    Per grain ``Hᵢ = Rᵢ diag(ln(a/s), ln(b/s), ln(c/s)) Rᵢᵀ`` with ``s = (abc)^{1/3}``; the
    component-by-component mean ``H̄`` gives principal stretches ``exp(eig H̄)``.  This is the
    log-Euclidean mean: unlike M11/M16 it is not exactly equivariant under ``M → F M Fᵀ`` when
    the grains are not coaxial with the strain, which Brandon notes as a bias source."""
    M, n = _unit_shape_tensors(table)
    w, V = np.linalg.eigh(M)
    H = 0.5 * (V * np.log(w)[:, None, :]) @ np.swapaxes(V, 1, 2)      # = R diag(ln semi/s) Rᵀ
    e, Q = np.linalg.eigh(H.mean(0))
    o = np.argsort(e)[::-1]
    return ellipsoid_to_strain(np.exp(e[o]), method="M15_spo_hencky", axes=Q[:, o], extra={"n_grains": n})


def estimate_strain_spo_yamaji(table: GrainTable) -> StrainEstimate:
    """M16 — determinant-normalised sum of unit shape matrices: Yamaji's (2008, Eqs 20–21)
    hyperbolic vector mean of strain ellipses, whose point on the unit hyperboloid is the
    unit-area shape matrix, generalised to 3-D.  Linear in the shape tensors, hence exactly
    equivariant under ``M → F M Fᵀ``; for isotropically oriented passive grains the mean is
    ``∝ B = F Fᵀ`` ("perfectly accurate, provided that pre-strain points had the centroid at C",
    Yamaji 2008 p. 1458)."""
    M, n = _unit_shape_tensors(table)
    S = M.sum(0)
    B = S / np.cbrt(np.linalg.det(S))
    w, V = np.linalg.eigh(0.5 * (B + B.T))
    o = np.argsort(w)[::-1]
    return ellipsoid_to_strain(np.sqrt(w[o]), method="M16_spo_yamaji", axes=V[:, o], extra={"n_grains": n})
