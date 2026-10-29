"""Deformation gradient tensors and the finite-strain ellipsoid.

Fossen & Tikoff (1993, 1997) deformation-matrix framework: pure shear, simple
shear and their combination (sequential and simultaneous), plus extraction of
the finite-strain ellipsoid from the left Cauchy-Green tensor ``B = F Fᵀ``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm

from ..schema import GrainTable
from ..synthetic.shapes import random_rotation

__all__ = ["StrainEllipsoid", "pure_shear", "simple_shear", "velocity_gradient",
           "simultaneous_deformation", "sequential_deformation", "progressive_deformation",
           "steady_state_deformation", "deformation_from_targets", "strain_ellipsoid",
           "vorticity_number", "apply_to_grains", "apply_to_points"]

_PLANE_IDX = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}


@dataclass
class StrainEllipsoid:
    stretches: np.ndarray    # (3,) λ1>=λ2>=λ3
    axes: np.ndarray         # (3,3) columns = principal directions (X, Y, Z)
    F: np.ndarray | None = None

    @property
    def Rs(self) -> float:
        return float(self.stretches[0] / self.stretches[2])

    @property
    def Rxy(self) -> float:
        return float(self.stretches[0] / self.stretches[1])

    @property
    def Ryz(self) -> float:
        return float(self.stretches[1] / self.stretches[2])

    @property
    def k(self) -> float:
        denom = self.Ryz - 1.0
        return np.inf if abs(denom) < 1e-9 else float((self.Rxy - 1.0) / denom)

    def __repr__(self) -> str:  # pragma: no cover
        return f"StrainEllipsoid(Rs={self.Rs:.3f}, k={self.k:.3f})"


def _axial_ratios_from_Rs_k(Rs, k):
    if not np.isfinite(k):
        return Rs, 1.0
    if abs(k) < 1e-9:
        return 1.0, Rs
    a, b, c = k, (1.0 - k), -Rs                       # k*Ryz² + (1-k)*Ryz - Rs = 0
    Ryz = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
    return float(Rs / Ryz), float(Ryz)


def pure_shear(Rs, k) -> np.ndarray:
    """Volume-constant coaxial deformation gradient (diagonal F)."""
    Rxy, Ryz = _axial_ratios_from_Rs_k(Rs, k)
    t = (Ryz / Rxy) ** (1.0 / 3.0)
    return np.diag([Rxy * t, t, t / Ryz])


def simple_shear(gamma, plane="xz") -> np.ndarray:
    F = np.eye(3)
    F[_PLANE_IDX[plane]] = gamma
    return F


def velocity_gradient(Rs, k, gamma, plane="xz") -> np.ndarray:
    """Constant velocity gradient L for combined pure+simple shear (log stretches on
    the diagonal, shear rate off-diagonal); ``exp(L)`` is the simultaneous matrix."""
    L = np.diag(np.log(np.diag(pure_shear(Rs, k)))).astype(float)
    L[_PLANE_IDX[plane]] = gamma
    return L


def simultaneous_deformation(Rs, k, gamma, plane="xz") -> np.ndarray:
    """Simultaneous (steady-state) deformation D = exp(L), Fossen & Tikoff (1993)."""
    return expm(velocity_gradient(Rs, k, gamma, plane))


def sequential_deformation(Rs, k, gamma, plane="xz") -> np.ndarray:
    """Bulk sequential deformation F = F_ss @ F_ps."""
    return simple_shear(gamma, plane) @ pure_shear(Rs, k)


def steady_state_deformation(Rs, k, Wk, plane="xz") -> np.ndarray:
    """Steady-state (simultaneous) deformation ``D = exp(s·L)`` with kinematic
    vorticity ``Wk ∈ [0, 1]`` and target **finite** strain ratio ``Rs`` (X/Z of the
    finite ellipsoid).

    Reparameterises the (Rs, Wk) grid the physically-correct way for continuous
    flow: a velocity-gradient direction ``L`` fixes Wk — ``Wk = γ/√(γ²+2Σεᵢ²)`` (see
    :func:`vorticity_number`) is invariant to the scale ``s`` — so the pure-shear
    log-stretch shape ``εᵢ`` is set by k and ``γ = Wk·√(2Σεᵢ²)/√(1−Wk²)``; the scalar
    ``s`` is then solved so the finite Rs hits the target.  This is exact for every k
    (not only plane strain) and finite for **all** Wk — including the simple-shear
    end-member Wk=1 (no pure-shear component, so ``L`` is nilpotent and ``exp`` stays
    finite) — unlike targeting γ at a fixed pure-shear Rs, where Wk→1 forces γ→∞.
    At Wk=1 the deformation is plane-strain simple shear (k has no effect), as it
    must be.  (Previously ``γ=2·Wk`` inverted a factor-of-4 error in the vorticity
    formula, so nominal Wk 0.5/0.71/0.87 were really ≈0.76/0.90/0.96.)
    """
    from scipy.optimize import brentq
    Wk = float(np.clip(Wk, 0.0, 1.0))
    if Rs <= 1.0 + 1e-9:
        return np.eye(3)
    if Wk >= 1.0 - 1e-12:                             # simple shear: no pure-shear part
        eps, gamma_dir = np.zeros(3), 1.0
    else:
        eps = np.log(np.diag(pure_shear(np.e, k)))    # pure-shear log-stretch shape for this k
        gamma_dir = Wk * np.sqrt(2.0 * float(np.sum(eps**2))) / np.sqrt(1.0 - Wk * Wk)
    L_dir = np.diag(eps).astype(float)
    L_dir[_PLANE_IDX[plane]] = gamma_dir
    D = lambda s: expm(s * L_dir)
    f = lambda s: strain_ellipsoid(D(s)).Rs - Rs
    s_hi = 1.0
    while f(s_hi) < 0.0 and s_hi < 1.0e6:
        s_hi *= 2.0
    s = brentq(f, 1e-9, s_hi)
    return D(s)


def progressive_deformation(Rs, k, gamma, n_steps=1, plane="xz", mode="sequential") -> np.ndarray:
    """Deformation accumulated as ``n_steps`` equal increments (Fossen & Tikoff 1997
    minimum-strain path in the sequential→simultaneous limit)."""
    if mode == "simultaneous":
        inc = expm(velocity_gradient(Rs, k, gamma, plane) / n_steps)
    elif mode == "sequential":
        F_ps = pure_shear(Rs, k)
        inc = simple_shear(gamma / n_steps, plane) @ np.diag(np.diag(F_ps) ** (1.0 / n_steps))
    else:
        raise ValueError("mode must be 'sequential' or 'simultaneous'")
    F = np.eye(3)
    for _ in range(n_steps):
        F = inc @ F
    return F


def deformation_from_targets(Rs=1.0, k=1.0, gamma=0.0, plane="xz", mode="sequential",
                             Q=None, rng=None) -> np.ndarray:
    """Deformation gradient from target (Rs, k) pure shear + optional simple shear
    ``gamma`` + optional rigid rotation ``Q`` (random if ``rng`` given).  When
    gamma>0 the finite strain differs from nominal — read it from :func:`strain_ellipsoid`."""
    if gamma == 0.0:
        F = pure_shear(Rs, k)
    elif mode == "simultaneous":
        F = simultaneous_deformation(Rs, k, gamma, plane)
    else:
        F = sequential_deformation(Rs, k, gamma, plane)
    if Q is None and rng is not None:
        Q = random_rotation(rng)
    if Q is not None:
        F = Q @ F @ Q.T
    return F


def strain_ellipsoid(F) -> StrainEllipsoid:
    """Finite-strain ellipsoid from F via the left Cauchy-Green tensor B = F Fᵀ."""
    F = np.asarray(F, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(F @ F.T)
    order = np.argsort(eigvals)[::-1]
    stretches = np.sqrt(np.clip(eigvals[order], 1e-12, None))
    return StrainEllipsoid(stretches=stretches, axes=eigvecs[:, order], F=F)


def vorticity_number(gamma, Rs, k) -> float:
    """Kinematic vorticity number ``Wk = |ω| / (√2·‖D‖)`` for a constant-volume flow
    whose velocity gradient carries pure-shear log-stretches εᵢ on the diagonal and
    simple-shear rate ``gamma`` off-diagonal (Truesdell 1953; Fossen & Tikoff 1993;
    Xypolias 2010).  Here vorticity ``ω = γ`` and ``‖D‖² = Σεᵢ² + γ²/2``, so
    ``Wk = γ / √(γ² + 2Σεᵢ²)`` — which is ``γ/√(γ²+(ε₁−ε₃)²)`` for plane strain (k=1).
    (The previous ``4(ε₁−ε₃)²`` denominator was a factor-of-4 error.)"""
    eps = np.log(np.diag(pure_shear(Rs, k)))
    denom = np.sqrt(gamma**2 + 2.0 * float(np.sum(eps**2)))
    return 0.0 if denom == 0 else float(abs(gamma) / denom)


# --------------------------------------------------------------------------- #
def apply_to_points(points, F) -> np.ndarray:
    """Affine transform of coordinates: x' = F x."""
    return (np.asarray(F) @ np.asarray(points).T).T


def _shape_tensor_to_axes(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    semi = np.sqrt(np.clip(eigvals[order], 1e-12, None))
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    return semi, eigvecs


def apply_to_grains(table: GrainTable, F) -> GrainTable:
    """Apply deformation gradient ``F`` to a whole (CT-native) GrainTable.

    Centroids transform affinely; grain shape tensors transform as ``M' = F M Fᵀ``,
    re-extracted into new semi-axes (stored as ``eigenvalues = semi²``) and
    ``eigenvectors``.  Volume follows det F (``V' = |det F| V``).  Per-grain detection
    info and sample scalars propagate (deformation preserves grain count + order).
    """
    F = np.asarray(F, dtype=float)
    new_c = apply_to_points(table.centroids, F)
    eig, evec, vol = table.eigenvalues, table.eigenvectors, table.volume
    if table.has_shapes and table.n:
        M = table.shape_tensors()                     # (N,3,3), uses semi_axes+eigenvectors
        Mdef = F @ M @ F.T                            # broadcast F over the stack
        # batched eigendecomposition of the whole stack (eigh: ascending -> reverse to desc)
        w, V = np.linalg.eigh(Mdef)                   # w (N,3) asc, V (N,3,3)
        semi = np.sqrt(np.clip(w[:, ::-1], 1e-12, None))     # (N,3) desc
        R = V[:, :, ::-1].copy()                      # columns reordered desc
        flip = np.linalg.det(R) < 0                   # enforce right-handed frames
        R[flip, :, -1] *= -1
        eig = semi ** 2                               # sqrt(eig)=new semi
        evec = R
        vol = (4.0 / 3.0) * np.pi * semi[:, 0] * semi[:, 1] * semi[:, 2]
    return GrainTable(
        centroids=new_c, volume=vol, eigenvalues=eig, eigenvectors=evec, ids=table.ids,
        meta={**table.meta, "deformed": True},
        detected=table.detected, detect_prob=table.detect_prob, **table._scalar_kwargs(),
    )
