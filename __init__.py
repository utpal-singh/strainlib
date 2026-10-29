"""strainlib — synthetic-validated 3D finite-strain analysis from CT-imaged marker grains.

The CT-native :class:`GrainTable` (``strainlib.schema``) is the single contract between
every stage; ``strainlib.io`` ingests real X-ray CT shape-analysis exports; ``strainlib.repro``
provides the deterministic seed policy + run manifest.

Pipeline subpackages:
* :mod:`strainlib.synthetic`    — Thomas + Matérn hard-core rock, calibrated to real samples
* :mod:`strainlib.deformation`  — Fossen–Tikoff deformation + finite-strain ellipsoid
* :mod:`strainlib.detection`    — CT detection loss (MCAR / MNAR / resolution)
* :mod:`strainlib.methods`      — strain-analysis methods (M1–M8; built per docs/PAPER_A_PLAN.md)
* :mod:`strainlib.analysis`     — spatial statistics (+ benchmarking once methods exist)
* :mod:`strainlib.experiment`   — the deformation-grid design
"""
from __future__ import annotations

from . import schema, repro, io, synthetic, deformation, detection, methods, analysis, experiment
from .schema import GrainTable, SCHEMA_VERSION

__version__ = "0.1.0"

__all__ = ["schema", "repro", "io", "synthetic", "deformation", "detection", "methods",
           "analysis", "experiment", "GrainTable", "SCHEMA_VERSION", "__version__"]
