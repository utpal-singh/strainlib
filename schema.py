"""The GrainTable data contract — CT-native.

`GrainTable` is the single interface between every pipeline stage: the synthetic
generator and the real X-ray CT segmentation both emit a `GrainTable`, and every
analysis method consumes one.  The field set is built directly around the columns
exported by the CT shape-analysis (Avizo/Dragonfly-style) so that **real and
synthetic tables are interchangeable** — the whole method can be validated on
synthetic data and then applied unchanged to a real xenolith.

Per-grain arrays (length ``N``)
-------------------------------
Required:
* ``centroids``    : (N, 3) float — grain barycentre (BaryCenter X/Y/Z), ``length_unit``.

Core shape (optional, but needed for shape/SPO methods):
* ``volume``       : (N,)   float — Volume3d, ``length_unit**3``.
* ``eigenvalues``  : (N, 3) float — voxel-covariance variances ``λ1 ≥ λ2 ≥ λ3`` (EigenVal1..3).
* ``eigenvectors`` : (N, 3, 3) float — columns are the long/intermediate/short axis unit
                     vectors (EigenVec1, EigenVec2, EigenVec3), in the ``frame`` coordinate system.

Raw CT extras (optional, carried through for fidelity/QC):
* ``extent_min`` / ``extent_max`` : (N, 3) — principal-axis bounding box (ExtentMin/Max 1..3).
* ``bin_moments``  : (N, 6) — second moments [2x, 2y, 2z, xy, xz, yz] (BinMom*).
* ``voxel_face_area`` (N,), ``border_voxel_count`` (N,), ``grey_mass`` (N,) — VoxelFaceArea,
                     BorderVoxelCount, GreyMass.
* ``ids``          : (N,) int — stable grain id (the CT ``index``).

Detection (populated by the detection models / imputed for real data):
* ``detected``     : (N,) bool — grain survived detection.
* ``detect_prob``  : (N,) float ∈ [0, 1] — modelled per-grain retention probability ``p_i``.

Sample-level scalars
--------------------
``p_hat`` (detection rate), ``sample_id``, ``locality`` (provenance, for multi-site Paper B),
``frame`` (default ``"CT"``), ``length_unit`` (default ``"mm"``), ``schema_version``, ``meta``.

Anisotropy = 1 − λ3/λ1, Elongation = λ2/λ1, Flatness = λ3/λ2 are provided as methods
(computed from ``eigenvalues``); if the CT export supplies them they are stored and returned as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import numpy as np

__all__ = ["GrainTable", "SCHEMA_VERSION"]

SCHEMA_VERSION = "2.0.0-ct"


@dataclass
class GrainTable:
    centroids: np.ndarray                          # (N, 3) BaryCenter
    volume: np.ndarray | None = None               # (N,)  Volume3d
    eigenvalues: np.ndarray | None = None          # (N, 3) EigenVal1..3 (desc)
    eigenvectors: np.ndarray | None = None         # (N, 3, 3) cols = EigenVec1..3
    extent_min: np.ndarray | None = None           # (N, 3) ExtentMin1..3
    extent_max: np.ndarray | None = None           # (N, 3) ExtentMax1..3
    bin_moments: np.ndarray | None = None          # (N, 6) [2x,2y,2z,xy,xz,yz]
    voxel_face_area: np.ndarray | None = None      # (N,)
    border_voxel_count: np.ndarray | None = None   # (N,)
    grey_mass: np.ndarray | None = None            # (N,)
    ids: np.ndarray | None = None                  # (N,) CT index
    # stored CT shape scalars (else computed from eigenvalues)
    anisotropy_stored: np.ndarray | None = None
    elongation_stored: np.ndarray | None = None
    flatness_stored: np.ndarray | None = None
    # detection
    detected: np.ndarray | None = None
    detect_prob: np.ndarray | None = None
    # sample-level scalars
    p_hat: float | None = None
    sample_id: str | None = None
    locality: str | None = None
    frame: str = "CT"
    length_unit: str = "mm"
    schema_version: str = SCHEMA_VERSION
    meta: dict = field(default_factory=dict)

    _PERGRAIN_1D = ("volume", "voxel_face_area", "border_voxel_count", "grey_mass",
                    "anisotropy_stored", "elongation_stored", "flatness_stored",
                    "detected", "detect_prob")
    _PERGRAIN_ND = {"eigenvalues": (3,), "eigenvectors": (3, 3), "extent_min": (3,),
                    "extent_max": (3,), "bin_moments": (6,)}

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.centroids = np.asarray(self.centroids, dtype=float)
        if self.centroids.ndim != 2 or self.centroids.shape[1] != 3:
            raise ValueError("centroids must have shape (N, 3)")
        n = self.centroids.shape[0]
        if self.ids is None:
            self.ids = np.arange(n, dtype=int)
        else:
            self.ids = np.asarray(self.ids, dtype=int).reshape(n)
        for name in self._PERGRAIN_1D:
            v = getattr(self, name)
            if v is not None:
                dt = bool if name == "detected" else float
                setattr(self, name, np.asarray(v, dtype=dt).reshape(n))
        for name, shp in self._PERGRAIN_ND.items():
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, np.asarray(v, dtype=float).reshape((n,) + shp))
        if self.detect_prob is not None and (np.any(self.detect_prob < 0) or np.any(self.detect_prob > 1)):
            raise ValueError("detect_prob must lie in [0, 1]")
        if self.p_hat is not None and not (0.0 < float(self.p_hat) <= 1.0):
            raise ValueError("p_hat must be in (0, 1]")

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return self.centroids.shape[0]

    def __len__(self) -> int:
        return self.n

    @property
    def has_shapes(self) -> bool:
        return self.eigenvalues is not None and self.eigenvectors is not None

    @property
    def has_detection(self) -> bool:
        return self.detect_prob is not None or self.detected is not None

    # ---- derived shape descriptors ----------------------------------- #
    def _need_eig(self):
        if self.eigenvalues is None:
            raise ValueError("eigenvalues are required for this operation")

    def anisotropy(self) -> np.ndarray:
        if self.anisotropy_stored is not None:
            return self.anisotropy_stored
        self._need_eig()
        ev = self.eigenvalues
        return 1.0 - ev[:, 2] / ev[:, 0]

    def elongation(self) -> np.ndarray:
        if self.elongation_stored is not None:
            return self.elongation_stored
        self._need_eig()
        ev = self.eigenvalues
        return ev[:, 1] / ev[:, 0]

    def flatness(self) -> np.ndarray:
        if self.flatness_stored is not None:
            return self.flatness_stored
        self._need_eig()
        ev = self.eigenvalues
        return ev[:, 2] / ev[:, 1]

    def semi_axes(self) -> np.ndarray:
        """Equivalent-ellipsoid semi-axes ``a ≥ b ≥ c``, (N, 3).

        Shape ratios come from ``sqrt(eigenvalues)``; the absolute scale is set so
        that ``4/3 π a b c`` equals ``volume`` when volumes are present (otherwise the
        raw ``sqrt(eigenvalues)`` are returned — shape only, arbitrary scale).
        """
        self._need_eig()
        r = np.sqrt(np.clip(self.eigenvalues, 0.0, None))          # (N,3), r1>=r2>=r3
        if self.volume is None:
            return r
        prod = r[:, 0] * r[:, 1] * r[:, 2]
        ok = prod > 0                                              # non-degenerate grains
        s = np.zeros(self.n)
        s[ok] = (self.volume[ok] / (4.0 / 3.0 * np.pi * prod[ok])) ** (1.0 / 3.0)
        a = s[:, None] * r
        if np.any(~ok):
            # degenerate covariance (a near-zero eigenvalue -> planar/single-voxel grain):
            # fall back to the equal-volume sphere so 4/3 pi a b c == volume for every grain
            R_eq = (3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            a[~ok] = R_eq[~ok, None]
        return a

    def long_axis_vectors(self) -> np.ndarray:
        self._need_eig(); return self.eigenvectors[:, :, 0]

    def intermediate_axis_vectors(self) -> np.ndarray:
        self._need_eig(); return self.eigenvectors[:, :, 1]

    def short_axis_vectors(self) -> np.ndarray:
        self._need_eig(); return self.eigenvectors[:, :, 2]

    def shape_tensors(self) -> np.ndarray:
        """Per-grain shape tensors ``M = R diag(a², b², c²) Rᵀ``, (N, 3, 3)."""
        self._need_eig()
        R = self.eigenvectors
        a = self.semi_axes()
        D = np.zeros((self.n, 3, 3))
        D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = a[:, 0] ** 2, a[:, 1] ** 2, a[:, 2] ** 2
        return R @ D @ np.transpose(R, (0, 2, 1))

    def equivalent_diameters(self) -> np.ndarray:
        if self.volume is None:
            raise ValueError("volume is required for equivalent_diameters")
        return 2.0 * (3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    # ------------------------------------------------------------------ #
    def _scalar_kwargs(self) -> dict:
        return dict(p_hat=self.p_hat, sample_id=self.sample_id, locality=self.locality,
                    frame=self.frame, length_unit=self.length_unit, schema_version=self.schema_version)

    def subset(self, mask: np.ndarray) -> "GrainTable":
        """New GrainTable keeping grains where ``mask`` is True (scalars propagate)."""
        mask = np.asarray(mask)
        def m(name):
            v = getattr(self, name)
            return None if v is None else v[mask]
        return GrainTable(
            centroids=self.centroids[mask],
            **{k: m(k) for k in ("volume", "eigenvalues", "eigenvectors", "extent_min",
                                 "extent_max", "bin_moments", "voxel_face_area",
                                 "border_voxel_count", "grey_mass", "ids",
                                 "anisotropy_stored", "elongation_stored", "flatness_stored",
                                 "detected", "detect_prob")},
            meta=dict(self.meta), **self._scalar_kwargs(),
        )

    def copy(self) -> "GrainTable":
        return self.subset(np.ones(self.n, dtype=bool))

    def with_detection(self, detect_prob=None, detected=None, p_hat=None) -> "GrainTable":
        out = self.copy()
        if detect_prob is not None:
            out.detect_prob = np.asarray(detect_prob, dtype=float).reshape(out.n)
        if detected is not None:
            out.detected = np.asarray(detected, dtype=bool).reshape(out.n)
        if p_hat is not None:
            out.p_hat = float(p_hat)
        return out

    # ------------------------------------------------------------------ #
    def to_dataframe(self):
        """Flatten to a CT-style DataFrame (one row per grain)."""
        import pandas as pd
        d = {"index": self.ids,
             "BaryCenterX": self.centroids[:, 0], "BaryCenterY": self.centroids[:, 1],
             "BaryCenterZ": self.centroids[:, 2]}
        if self.volume is not None: d["Volume3d"] = self.volume
        if self.eigenvalues is not None:
            for i in range(3): d[f"EigenVal{i+1}"] = self.eigenvalues[:, i]
        if self.eigenvectors is not None:
            for i in range(3):
                for ax, j in zip("XYZ", range(3)):
                    d[f"EigenVec{i+1}{ax}"] = self.eigenvectors[:, j, i]
        for name, col in (("detected", "detected"), ("detect_prob", "detect_prob"),
                          ("grey_mass", "GreyMass"), ("border_voxel_count", "BorderVoxelCount"),
                          ("voxel_face_area", "VoxelFaceArea")):
            v = getattr(self, name)
            if v is not None: d[col] = v
        df = pd.DataFrame(d)
        df.attrs.update(self._scalar_kwargs())
        return df

    # ------------------------------------------------------------------ #
    @staticmethod
    def _json_safe(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: GrainTable._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [GrainTable._json_safe(v) for v in obj]
        return obj

    _NPZ_ARRAYS = ("centroids", "volume", "eigenvalues", "eigenvectors", "extent_min",
                   "extent_max", "bin_moments", "voxel_face_area", "border_voxel_count",
                   "grey_mass", "ids", "anisotropy_stored", "elongation_stored",
                   "flatness_stored", "detected", "detect_prob")

    def save(self, path: str) -> None:
        """Save to a compressed ``.npz`` (authoritative round-trip)."""
        empty = np.array([])
        arrs = {k: (empty if getattr(self, k) is None else getattr(self, k)) for k in self._NPZ_ARRAYS}
        scalars = {**self._scalar_kwargs(), "meta": self._json_safe(self.meta)}
        np.savez_compressed(path, _scalars=np.array(json.dumps(scalars)), **arrs)

    @classmethod
    def load(cls, path: str) -> "GrainTable":
        d = np.load(path, allow_pickle=False)
        def opt(name):
            if name not in d: return None
            a = d[name]
            return None if a.size == 0 else a
        sc = json.loads(str(d["_scalars"])) if "_scalars" in d else {}
        meta = sc.pop("meta", {}) or {}
        kw = {k: opt(k) for k in cls._NPZ_ARRAYS if k != "centroids"}
        if kw.get("detected") is not None:
            kw["detected"] = d["detected"].astype(bool)
        return cls(centroids=d["centroids"], meta=meta,
                   p_hat=sc.get("p_hat"), sample_id=sc.get("sample_id"), locality=sc.get("locality"),
                   frame=sc.get("frame", "CT"), length_unit=sc.get("length_unit", "mm"),
                   schema_version=sc.get("schema_version", SCHEMA_VERSION), **kw)

    def __repr__(self) -> str:  # pragma: no cover
        loc = f", locality={self.locality!r}" if self.locality else ""
        sid = f", sample={self.sample_id!r}" if self.sample_id else ""
        return (f"GrainTable(n={self.n}, shapes={'yes' if self.has_shapes else 'no'}, "
                f"unit={self.length_unit}{sid}{loc}, p_hat={self.p_hat})")
