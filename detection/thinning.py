"""Detection / thinning models simulating CT acquisition loss.

* MCAR : independent Bernoulli thinning (Daley & Vere-Jones 2003).
* MNAR : Beer–Lambert cone-shadow (attenuation by upstream grains), beam ∥ X and Z.
* Resolution truncation: grains below ~2 voxels are unresolvable.

Each model populates per-grain ``detect_prob`` / ``detected`` and the sample
``p_hat`` on the returned table, so the downstream uncertainty budget can be
detection-aware (MNAR bootstrap).
"""
from __future__ import annotations

import numpy as np

from ..schema import GrainTable

__all__ = ["mcar_thin", "mnar_beer_lambert", "resolution_cut", "detection_scenarios"]


def mcar_thin(table: GrainTable, p: float, rng: np.random.Generator | None = None) -> GrainTable:
    """Retain each grain independently with probability ``p`` (MCAR)."""
    rng = np.random.default_rng() if rng is None else rng
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    keep = rng.uniform(size=table.n) < p
    out = table.subset(keep)
    out.meta = {**table.meta, "thinning": "mcar", "p": p}
    out.detect_prob = np.full(out.n, float(p))
    out.detected = np.ones(out.n, dtype=bool)
    out.p_hat = float(p)
    return out


def mnar_beer_lambert(table: GrainTable, beam_dir=(0.0, 0.0, 1.0), mu_eff: float = 0.01,
                      cone_half_angle_deg: float = 5.0, rng=None) -> GrainTable:
    """Spatially-biased (MNAR) thinning by a Beer–Lambert cone-shadow model.

    ``p_i = exp(-mu_eff · Σ_j 2 r_j)`` over grains ``j`` upstream of ``i`` (towards the
    source) within a narrow cone about ``beam_dir``; ``r_j`` from grain volume.
    """
    rng = np.random.default_rng() if rng is None else rng
    if table.volume is None:
        raise ValueError("volume is required for the Beer-Lambert model")
    n = table.n
    if n == 0:
        return table.copy()
    beam = np.asarray(beam_dir, dtype=float); beam /= np.linalg.norm(beam)
    C = table.centroids
    radii = (3.0 * table.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    cos_thresh = np.cos(np.deg2rad(cone_half_angle_deg))
    p_detect = np.ones(n)
    for i in range(n):
        d = C[i] - C
        along = d @ beam
        upstream = along > 1e-9
        if not np.any(upstream):
            continue
        norms = np.linalg.norm(d, axis=1); norms[norms == 0] = np.inf
        in_cone = upstream & (along / norms >= cos_thresh)
        p_detect[i] = np.exp(-np.sum(2.0 * radii[in_cone]) * mu_eff)
    keep = rng.uniform(size=n) < p_detect
    out = table.subset(keep)
    out.meta = {**table.meta, "thinning": "mnar", "mu_eff": mu_eff, "beam": tuple(beam)}
    out.detect_prob = p_detect[keep]
    out.detected = np.ones(out.n, dtype=bool)
    out.p_hat = float(np.mean(p_detect))
    return out


def resolution_cut(table: GrainTable, voxel_size: float, min_voxels: float = 2.0) -> GrainTable:
    """Remove grains with equivalent diameter below ``min_voxels`` voxels."""
    if table.volume is None:
        raise ValueError("volume is required for resolution_cut")
    keep = table.equivalent_diameters() >= min_voxels * voxel_size
    out = table.subset(keep)
    out.meta = {**table.meta, "resolution_voxel": voxel_size}
    out.detect_prob = np.ones(out.n)
    out.detected = np.ones(out.n, dtype=bool)
    out.p_hat = float(np.mean(keep)) if out.n > 0 else None    # 0 is out of the (0,1] schema range
    return out


def detection_scenarios(table: GrainTable, p_hat: float = 0.6, voxel_size: float | None = None,
                        rng=None) -> dict:
    """The standard 13-scenario battery for one deformed rock → {label: GrainTable}.

    ideal(1) · MCAR p∈{0.75,0.50,0.25,p̂}(4) · MNAR beam∥X {low,mid,high}(3) ·
    MNAR beam∥Z {low,mid,high}(3) · resolution(1) · MCAR+resolution(1).
    (Resolution scenarios require ``voxel_size``; MNAR requires volumes.)
    """
    rng = np.random.default_rng() if rng is None else rng
    out = {"ideal": table.copy()}
    for p in (0.75, 0.50, 0.25, p_hat):
        out[f"mcar_p{p:.2f}"] = mcar_thin(table, p, rng=rng)
    if table.volume is not None:
        for axis, beam in (("x", (1, 0, 0)), ("z", (0, 0, 1))):
            for mu_eff in (0.005, 0.01, 0.02):
                out[f"mnar_{axis}_{mu_eff}"] = mnar_beer_lambert(table, beam_dir=beam, mu_eff=mu_eff, rng=rng)
        if voxel_size is not None:
            out["resolution"] = resolution_cut(table, voxel_size)
            out["mcar+res"] = resolution_cut(mcar_thin(table, p_hat, rng=rng), voxel_size)
    return out
