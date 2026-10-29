"""Build a synthetic rock as a CT-native :class:`GrainTable`.

Spatial distribution: Thomas cluster process + Matérn (type III) hard-core.  Grain
volumes and shapes: log-normal, calibrated to the real grain table, with random
orientations (no initial shape fabric).  The output carries the
same fields a real CT export would (via ``eigenvalues``/``eigenvectors``), so it
is interchangeable with real data throughout the pipeline.
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable
from .point_process import generate_centroids
from .shapes import assign_shapes

__all__ = ["build_synthetic_rock"]


def build_synthetic_rock(
    box=(60.0, 60.0, 60.0), kappa=0.01, mu=8.0, sigma=2.0, hardcore=0.0,
    log_volume_mean=None, log_volume_std=0.6, mean_grain_diameter=0.4,
    q1_logmean=0.0, q1_logstd=0.15, q2_logmean=0.0, q2_logstd=0.15,
    sample_id=None, locality=None, length_unit="mm",
    rng: np.random.Generator | None = None, centroids=None,
) -> GrainTable:
    """Generate a complete synthetic rock (CT-native GrainTable).

    ``log_volume_mean`` sets the volume scale; if ``None`` it is derived from
    ``mean_grain_diameter`` (in ``length_unit``).  Grain shape is stored as
    ``eigenvalues = semi_axes**2`` + ``eigenvectors`` so ``semi_axes()`` reproduces
    the assigned axes and volume exactly.

    ``centroids`` (N, 3) places the grains at given centres instead of drawing them
    from the Thomas/Matérn process (the shape/volume model is unchanged); the
    point-process parameters are then recorded as ``None``.
    """
    rng = np.random.default_rng() if rng is None else rng
    if centroids is None:
        centroids = generate_centroids(box, kappa, mu, sigma, hardcore=hardcore, rng=rng)
        pp = {"kappa": kappa, "mu": mu, "sigma": sigma, "hardcore": hardcore}
    else:
        centroids = np.asarray(centroids, dtype=float).reshape(-1, 3)
        pp = {"kappa": None, "mu": None, "sigma": None, "hardcore": None}
    n = len(centroids)
    base_meta = {"synthetic": True, "box": tuple(float(x) for x in box), **pp}
    if n == 0:
        return GrainTable(centroids=np.empty((0, 3)), sample_id=sample_id,
                          locality=locality, length_unit=length_unit, meta=base_meta)
    if log_volume_mean is None:
        mean_vol = (4.0 / 3.0) * np.pi * (mean_grain_diameter / 2.0) ** 3
        log_volume_mean = np.log(mean_vol) - 0.5 * log_volume_std**2

    volumes, axes, orient = assign_shapes(
        n, log_volume_mean=log_volume_mean, log_volume_std=log_volume_std,
        q1_logmean=q1_logmean, q1_logstd=q1_logstd,
        q2_logmean=q2_logmean, q2_logstd=q2_logstd, rng=rng)

    return GrainTable(
        centroids=centroids, volume=volumes,
        eigenvalues=axes ** 2,          # sqrt(eig)=semi; semi_axes() rescales to volume
        eigenvectors=orient,
        sample_id=sample_id, locality=locality, length_unit=length_unit, meta=base_meta,
    )
