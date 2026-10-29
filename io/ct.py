"""Load real X-ray CT spinel shape-analysis exports into a :class:`GrainTable`.

Handles the Avizo/Dragonfly-style CSV: a title line, then a header row with the
35 columns (Volume3d, BaryCenter*, Anisotropy, Elongation, Flatness, EigenVal1-3,
EigenVec{1,2,3}{X,Y,Z}, ExtentMin/Max1-3, BinMom*, VoxelFaceArea, BorderVoxelCount,
GreyMass, index).  Column matching is whitespace/unit-insensitive.
"""
from __future__ import annotations

import glob
import os
import re
import numpy as np

from ..schema import GrainTable

__all__ = ["load_ct_csv", "load_ct_directory", "infer_locality", "CT_COLUMNS"]

# canonical (normalised) CT column keys, for reference / QC
CT_COLUMNS = [
    "volume3d", "barycenterx", "barycentery", "barycenterz", "anisotropy", "elongation",
    "flatness", "eigenval1", "eigenval2", "eigenval3",
    "eigenvec1x", "eigenvec1y", "eigenvec1z", "eigenvec2x", "eigenvec2y", "eigenvec2z",
    "eigenvec3x", "eigenvec3y", "eigenvec3z",
    "extentmin1", "extentmin2", "extentmin3", "extentmax1", "extentmax2", "extentmax3",
    "binmom2x", "binmom2y", "binmom2z", "binmomxy", "binmomxz", "binmomyz",
    "voxelfacearea", "bordervoxelcount", "greymass", "index",
]

# filename-prefix -> locality (SE-Australia Newer Volcanics Province maars)
_LOCALITY = {"gn": "Gnotuk", "noo": "Noorat", "no": "Noorat", "sh": "Mount Shadwell"}

#: samples excluded from the working set (user decision).
EXCLUDE = {"wbgm12"}


def _norm(col: str) -> str:
    """Normalise a column name: drop unit suffixes + whitespace, lowercase."""
    c = re.sub(r"\(.*?\)", "", str(col))            # strip "(mm^3)" etc.
    return re.sub(r"[^a-z0-9]", "", c.lower())


def infer_locality(name: str) -> str | None:
    key = name.lower()
    for pre in ("noo", "gn", "no", "sh"):
        if key.startswith(pre):
            return _LOCALITY.get(pre)
    return None


def load_ct_csv(path: str, sample_id: str | None = None,
                locality: str | None = None, length_unit: str = "mm") -> GrainTable:
    """Load one CT export CSV into a :class:`GrainTable`."""
    import pandas as pd
    # detect the header row: the real header is the first line that contains "Volume3d"
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        head = [fh.readline() for _ in range(5)]
    skip = next((i for i, ln in enumerate(head) if "volume3d" in ln.lower()), 1)
    df = pd.read_csv(path, skiprows=skip)
    cols = {_norm(c): c for c in df.columns}

    def col(key, required=False):
        c = cols.get(key)
        if c is None:
            if required:
                raise ValueError(f"{os.path.basename(path)}: missing required column '{key}'")
            return None
        return df[c].to_numpy(dtype=float)

    centroids = np.column_stack([col("barycenterx", True), col("barycentery", True), col("barycenterz", True)])
    n = len(centroids)

    def stack(keys):
        arrs = [col(k) for k in keys]
        return None if any(a is None for a in arrs) else np.column_stack(arrs)

    eigenvalues = stack(["eigenval1", "eigenval2", "eigenval3"])
    eigenvectors = None
    evk = [[f"eigenvec{i}{ax}" for ax in "xyz"] for i in (1, 2, 3)]
    if all(cols.get(k) for row in evk for k in row):
        # (N,3,3): [:, :, i] = EigenVec(i+1) = [X,Y,Z]
        eigenvectors = np.stack([np.column_stack([col(k) for k in row]) for row in evk], axis=2)

    idx = col("index")
    ids = idx.astype(int) if idx is not None else None
    name = sample_id or os.path.splitext(os.path.basename(path))[0]

    return GrainTable(
        centroids=centroids,
        volume=col("volume3d"),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        extent_min=stack(["extentmin1", "extentmin2", "extentmin3"]),
        extent_max=stack(["extentmax1", "extentmax2", "extentmax3"]),
        bin_moments=stack(["binmom2x", "binmom2y", "binmom2z", "binmomxy", "binmomxz", "binmomyz"]),
        voxel_face_area=col("voxelfacearea"),
        border_voxel_count=col("bordervoxelcount"),
        grey_mass=col("greymass"),
        ids=ids,
        anisotropy_stored=col("anisotropy"),
        elongation_stored=col("elongation"),
        flatness_stored=col("flatness"),
        sample_id=name,
        locality=locality if locality is not None else infer_locality(name),
        length_unit=length_unit,
        meta={"source_csv": os.path.basename(path), "n_columns": len(df.columns)},
    )


def load_ct_directory(directory: str, pattern: str = "*.csv",
                      include_excluded: bool = False) -> dict:
    """Load every CT CSV in a directory -> {sample_id: GrainTable}.

    Skips files with no Volume3d column (non-CT csvs) and samples in :data:`EXCLUDE`
    unless ``include_excluded`` is set.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(directory, pattern))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name in EXCLUDE and not include_excluded:
            continue
        try:
            gt = load_ct_csv(p)
            out[gt.sample_id] = gt
        except ValueError:
            continue
    return out
