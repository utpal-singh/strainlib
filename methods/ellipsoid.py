"""Shared ellipsoid/ellipse fitting and the StrainEstimate result type.

Used by every strain-analysis method (M1–M8).  Geometry only — no dependence on
the CT fields, so it is reused unchanged for 2D sections and 3D clouds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

__all__ = ["StrainEstimate", "fit_ellipsoid", "fit_ellipse_2d", "ellipsoid_to_strain",
           "fibonacci_sphere", "void_shell_points", "stretches_from_Rs_k", "void_shell_2d",
           "cone_cosines"]


@dataclass
class StrainEstimate:
    """Result of a strain-analysis method."""
    method: str
    Rs: float
    k: float
    stretches: np.ndarray            # (3,) normalised principal stretches desc
    axes: np.ndarray | None = None   # (3,3) columns = principal directions
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"StrainEstimate({self.method}, Rs={self.Rs:.3f}, k={self.k:.3f})"


def fit_ellipsoid(points: np.ndarray):
    """Least-squares fit of a centred ellipsoid xᵀAx=1 → (semi_axes desc (3,), axes (3,3))."""
    P = np.asarray(points, dtype=float)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    D = np.column_stack([x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z])
    coef, *_ = np.linalg.lstsq(D, np.ones(P.shape[0]), rcond=None)
    A = np.array([[coef[0], coef[3], coef[4]], [coef[3], coef[1], coef[5]], [coef[4], coef[5], coef[2]]])
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.where(eigvals <= 0, np.nan, eigvals)
    semi = 1.0 / np.sqrt(eigvals)
    order = np.argsort(semi)[::-1]
    return semi[order], eigvecs[:, order]


def fit_ellipse_2d(points: np.ndarray):
    """Least-squares fit of a centred 2D ellipse xᵀAx=1 → (semi desc (2,), axes (2,2))."""
    P = np.asarray(points, dtype=float)
    x, y = P[:, 0], P[:, 1]
    D = np.column_stack([x*x, y*y, 2*x*y])
    coef, *_ = np.linalg.lstsq(D, np.ones(P.shape[0]), rcond=None)
    A = np.array([[coef[0], coef[2]], [coef[2], coef[1]]])
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.where(eigvals <= 0, np.nan, eigvals)
    semi = 1.0 / np.sqrt(eigvals)
    order = np.argsort(semi)[::-1]
    return semi[order], eigvecs[:, order]


def ellipsoid_to_strain(semi_axes, method, axes=None, extra=None) -> StrainEstimate:
    """Convert fitted ellipsoid semi-axes into a StrainEstimate (volume-normalised)."""
    s = np.asarray(semi_axes, dtype=float)
    s = s / np.cbrt(np.prod(s))
    Rs = float(s[0] / s[2]); Ryz = s[1] / s[2]; Rxy = s[0] / s[1]
    k = np.inf if abs(Ryz - 1) < 1e-9 else float((Rxy - 1) / (Ryz - 1))
    return StrainEstimate(method=method, Rs=Rs, k=k, stretches=s, axes=axes, extra=extra or {})


def stretches_from_Rs_k(Rs: float, k: float) -> np.ndarray:
    """Normalised principal stretches (desc, unit product) from Rs=X/Z and Flinn k."""
    if not np.isfinite(k):
        Rxy, Ryz = float(Rs), 1.0
    elif abs(k) < 1e-9:
        Rxy, Ryz = 1.0, float(Rs)
    else:
        a, b, c = k, (1.0 - k), -Rs
        Ryz = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
        Rxy = Rs / Ryz if Ryz else Rs
    s = np.array([Rs, Ryz, 1.0], dtype=float)         # X=Rs, Y=Ryz, Z=1 (since Z=1)
    return s / np.cbrt(np.prod(s))


def void_shell_2d(vectors: np.ndarray, n_dirs: int = 180, cone_deg: float = 12.0) -> np.ndarray:
    """2D analogue of :func:`void_shell_points`: nearest-to-origin in-cone vector per
    direction (the raw 2D-Fry void boundary)."""
    V = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(V, axis=1)
    good = norms > 1e-12
    V, norms = V[good], norms[good]
    units = V / norms[:, None]
    ang = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = np.column_stack([np.cos(ang), np.sin(ang)])
    cos_thresh = np.cos(np.deg2rad(cone_deg))
    out = []
    for cos, _ in cone_cosines(units, dirs):
        for j in range(cos.shape[1]):
            in_cone = cos[:, j] >= cos_thresh
            if not np.any(in_cone):
                continue
            idx = np.where(in_cone)[0]
            out.append(V[idx[np.argmin(norms[idx])]])
    return np.asarray(out)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` approximately-uniform unit vectors on the sphere, (n, 3)."""
    i = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    theta = phi * i
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def void_shell_points(vectors: np.ndarray, n_dirs: int = 200, cone_deg: float = 15.0) -> np.ndarray:
    """Inner boundary (central-void surface) of a Fry vector cloud: for each of
    ``n_dirs`` directions, the closest-to-origin vector within ``cone_deg``."""
    V = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(V, axis=1)
    good = norms > 1e-12
    V, norms = V[good], norms[good]
    units = V / norms[:, None]
    dirs = fibonacci_sphere(n_dirs)
    cos_thresh = np.cos(np.deg2rad(cone_deg))
    boundary = []
    for cos, _ in cone_cosines(units, dirs):
        for j in range(cos.shape[1]):
            in_cone = cos[:, j] >= cos_thresh
            if not np.any(in_cone):
                continue
            idx = np.where(in_cone)[0]
            boundary.append(V[idx[np.argmin(norms[idx])]])
    return np.asarray(boundary)


def cone_cosines(units: np.ndarray, dirs: np.ndarray, block: int = 16):
    """Yield ``(units @ dirs[b0:b0+block].T, b0)`` block by block.  Identical values to
    the full ``units @ dirs.T`` matrix, but peak memory is ``block/len(dirs)`` of it
    (a 200 000-vector × 200-direction matrix is 320 MB; 16-direction blocks are 26 MB),
    which lets many benchmark processes run in parallel."""
    for b0 in range(0, len(dirs), block):
        yield units @ dirs[b0:b0 + block].T, b0
