"""Grain shape and orientation assignment.

Uniform-random rotations via Shoemake (1992); log-normal volumes and aspect
ratios, calibrated to a real grain table (near-equant only with the default
log-means of 0).
"""
from __future__ import annotations

import numpy as np

__all__ = ["random_rotation", "random_rotations", "axes_from_volume_ratios", "assign_shapes"]


def random_rotation(rng: np.random.Generator | None = None) -> np.ndarray:
    """A single uniformly-random rotation matrix (Shoemake 1992)."""
    return random_rotations(1, rng=rng)[0]


def random_rotations(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """``n`` uniformly-random rotation matrices (Shoemake 1992 quaternions), (n,3,3)."""
    rng = np.random.default_rng() if rng is None else rng
    u1, u2, u3 = rng.uniform(size=(3, n))
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    w, x, y, z = q4, q1, q2, q3
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2); R[:, 0, 1] = 2 * (x*y - z*w); R[:, 0, 2] = 2 * (x*z + y*w)
    R[:, 1, 0] = 2 * (x*y + z*w); R[:, 1, 1] = 1 - 2 * (x**2 + z**2); R[:, 1, 2] = 2 * (y*z - x*w)
    R[:, 2, 0] = 2 * (x*z - y*w); R[:, 2, 1] = 2 * (y*z + x*w); R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def axes_from_volume_ratios(volume, q1, q2) -> np.ndarray:
    """Semi-axes (a≥b≥c) from volume and aspect ratios ``q1=a/b``, ``q2=b/c``.

    Ellipsoid volume ``V = 4/3 π a b c``.  Returns (N, 3) sorted descending.
    """
    volume = np.asarray(volume, dtype=float)
    q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
    c = (volume / ((4.0 / 3.0) * np.pi * q1 * q2**2)) ** (1.0 / 3.0)
    b = q2 * c
    a = q1 * b
    return np.sort(np.stack([a, b, c], axis=1), axis=1)[:, ::-1]


def assign_shapes(n, log_volume_mean, log_volume_std, q1_logmean=0.0, q1_logstd=0.15,
                  q2_logmean=0.0, q2_logstd=0.15, rng=None):
    """Draw (volumes, semi-axes (n,3), orientations (n,3,3)) for ``n`` grains.

    Volumes log-normal; aspect ratios ``q1, q2`` log-normal (default log-means 0 →
    near-equant; calibrated to NO21 they are 0.57 and 0.40); orientations isotropic.
    """
    rng = np.random.default_rng() if rng is None else rng
    volumes = rng.lognormal(mean=log_volume_mean, sigma=log_volume_std, size=n)
    q1 = rng.lognormal(mean=q1_logmean, sigma=q1_logstd, size=n)
    q2 = rng.lognormal(mean=q2_logmean, sigma=q2_logstd, size=n)
    axes = axes_from_volume_ratios(volumes, q1, q2)
    return volumes, axes, random_rotations(n, rng=rng)
