"""Observation model — a CT scan images a FIXED volume; the window never deforms.

Why this module exists (validity bug 1)
---------------------------------------
Pair-difference vectors observed inside a window W are sampled in proportion to its
set covariance ``C_W(v) = |W ∩ (W − v)|`` (Baddeley, Rubak & Turner 2015, Eq. 7.19),
so an estimator that reads large lags — e.g. a percentile "void" boundary — inherits
the shape of W.  The benchmark deformed the whole synthetic rock, outline
included, so W carried the imposed strain and the 3-D CFM looked accurate for the
wrong reason: on grains with NO strain seen through an Rs = 3 ellipsoidal window it
reports Rs ≈ 4, and through a fixed window it reports ≈ 1.7 for a true Rs = 3
(``scripts/diagnostics/verify_bug_fixes.py``).  A real scan window is set by the
sample and the scanner, not by the strain being measured.

Protocol — deform, THEN crop to a fixed lab-frame window
--------------------------------------------------------
1. build a parent rock large enough that the pre-image ``F⁻¹(W)`` fits inside it
   (:func:`parent_box`);
2. map centroids by ``F`` about the parent centre and keep those inside ``W``;
3. deform only the kept grains (shapes, volumes) — identical to deforming the whole
   parent and then cropping, at a fraction of the cost.

Both the factorial benchmark (:mod:`strainlib.runner`) and the simulator of the
inversion (:mod:`strainlib.inversion`) observe through this one function, so the
inversion's forward model is exactly the benchmark's observation model.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .schema import GrainTable
from .deformation.tensors import apply_to_grains

__all__ = ["ScanWindow", "scan_window_like", "parent_box", "observe", "window_for"]


@dataclass(frozen=True)
class ScanWindow:
    """Fixed lab-frame scan window centred on the origin.

    ``shape="box"``: axis-aligned box with half-extents ``half``;
    ``shape="ellipsoid"``: axis-aligned ellipsoid with semi-axes ``half`` (a sphere
    when all three are equal)."""
    half: tuple
    shape: str = "box"

    def __post_init__(self):
        h = np.asarray(self.half, dtype=float)
        if h.shape != (3,) or np.any(h <= 0):
            raise ValueError("half must be three positive half-extents")
        if self.shape not in ("box", "ellipsoid"):
            raise ValueError("shape must be 'box' or 'ellipsoid'")
        object.__setattr__(self, "half", tuple(float(x) for x in h))

    def contains(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        h = np.asarray(self.half)
        if self.shape == "box":
            return np.all(np.abs(X) <= h, axis=1)
        return np.sum((X / h) ** 2, axis=1) <= 1.0

    @property
    def volume(self) -> float:
        p = float(np.prod(self.half))
        return 8.0 * p if self.shape == "box" else 4.0 / 3.0 * np.pi * p

    def set_covariance(self, V) -> np.ndarray:
        """``C_W(v) = |W ∩ (W − v)|`` — the volume in which a pair with separation ``v``
        can be observed (Baddeley et al. 2015, Eq. 7.19).  Box: ``Π max(0, 2hᵢ − |vᵢ|)``.
        Ellipsoid ``W = H·B₁``: ``det H · C_B₁(‖H⁻¹v‖)``, with the unit-ball lens volume
        ``C_B₁(d) = π/12 (4 + d)(2 − d)²`` for ``d ≤ 2``."""
        V = np.atleast_2d(np.asarray(V, dtype=float))
        h = np.asarray(self.half)
        if self.shape == "box":
            return np.prod(np.clip(2.0 * h - np.abs(V), 0.0, None), axis=1)
        d = np.minimum(np.linalg.norm(V / h, axis=1), 2.0)
        return float(np.prod(h)) * np.pi / 12.0 * (4.0 + d) * (2.0 - d) ** 2

    def preimage_half_extents(self, F) -> np.ndarray:
        """Axis-aligned half-extents of ``F⁻¹(W)``: ``|F⁻¹|·h`` for a box, the row norms
        of ``F⁻¹·diag(h)`` for an ellipsoid."""
        A = np.linalg.inv(np.asarray(F, dtype=float))
        h = np.asarray(self.half)
        if self.shape == "box":
            return np.abs(A) @ h
        return np.linalg.norm(A * h[None, :], axis=1)


def scan_window_like(box, shape="box") -> ScanWindow:
    """Window the size of a real scan with extent ``box`` (e.g. calibrated
    ``params['box']``): that box, or for ``shape='sphere'`` the sphere of equal volume
    (orientation-neutral — isolates strain from window anisotropy)."""
    b = np.asarray(box, dtype=float)
    if shape == "box":
        return ScanWindow(tuple(b / 2.0), "box")
    if shape == "sphere":
        r = float((3.0 * np.prod(b) / (4.0 * np.pi)) ** (1.0 / 3.0))
        return ScanWindow((r, r, r), "ellipsoid")
    raise ValueError("shape must be 'box' or 'sphere'")


def parent_box(window: ScanWindow, Fs, pad: float = 0.0) -> tuple:
    """Parent-rock box such that ``F⁻¹(W)``, centred in the parent, fits for every ``F``
    in ``Fs``, with ``pad`` to spare on each side."""
    ext = np.max([window.preimage_half_extents(F) for F in Fs], axis=0)
    return tuple(float(x) for x in 2.0 * (ext + pad))


def observe(rock: GrainTable, F, window: ScanWindow) -> GrainTable:
    """Deform ``rock`` by ``F`` about its box centre and image it through the fixed
    ``window``.  Returned centroids are in the window frame (window centre = origin).

    Raises if the parent is too small, i.e. if part of ``F⁻¹(W)`` lies outside it —
    the window would then contain empty space shaped by the deformed outline, which is
    exactly the artifact this module exists to prevent."""
    F = np.asarray(F, dtype=float)
    box = rock.meta.get("box") if rock.meta else None
    if box is not None:                                  # synthetic parent spans [0, box]
        half = np.asarray(box, dtype=float) / 2.0
        centre = half
    else:
        lo, hi = rock.centroids.min(0), rock.centroids.max(0)
        half, centre = (hi - lo) / 2.0, (hi + lo) / 2.0
    need = window.preimage_half_extents(F)
    if np.any(need > half + 1e-9):
        raise ValueError(f"parent rock too small for this F: pre-image half-extents "
                         f"{np.round(need, 2)} exceed parent half-extents {np.round(half, 2)}; "
                         f"size it with parent_box()")
    X = (rock.centroids - centre) @ F.T
    keep = window.contains(X)
    out = apply_to_grains(rock.subset(keep), F)
    out = dataclasses.replace(out, centroids=X[keep])
    out.meta = {**(out.meta or {}), "window": {"shape": window.shape, "half": window.half}}
    return out


def window_for(table: GrainTable, window: ScanWindow | None = None):
    """The scan window an observation was imaged through, and its centre in the table's
    coordinates: the explicit ``window`` (centre 0), else ``table.meta['window']`` as set
    by :func:`observe` (centre 0), else the centroid bounding box (centre = box middle) —
    the fallback for real data exported without a sample mask."""
    if window is not None:
        return window, np.zeros(3)
    meta = table.meta or {}
    if meta.get("window"):
        return ScanWindow(**meta["window"]), np.zeros(3)
    C = np.asarray(table.centroids, dtype=float)
    lo, hi = C.min(0), C.max(0)
    return ScanWindow(tuple((hi - lo) / 2.0), "box"), (hi + lo) / 2.0


def boundary_distance(window: ScanWindow, X) -> np.ndarray:
    """Distance from each point (window frame) to the window boundary, ≥ 0 inside.
    Exact for a box; for an ellipsoid the radial-scaling approximation
    (1 − ‖x/h‖)·min(h), which never overstates the true distance."""
    X = np.asarray(X, dtype=float)
    h = np.asarray(window.half)
    if window.shape == "box":
        return np.min(h - np.abs(X), axis=1)
    return (1.0 - np.linalg.norm(X / h, axis=1)) * float(h.min())
