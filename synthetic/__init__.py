"""Synthetic rock generation (Thomas + Matérn hard-core; log-normal shapes)."""
from __future__ import annotations

from .point_process import thomas_process, matern_hardcore_thin, generate_centroids
from .shapes import random_rotation, random_rotations, axes_from_volume_ratios, assign_shapes
from .build import build_synthetic_rock
from .calibrate import calibrate_from_grains, build_like
from .volume import rasterize_grains

__all__ = ["thomas_process", "matern_hardcore_thin", "generate_centroids",
           "random_rotation", "random_rotations", "axes_from_volume_ratios", "assign_shapes",
           "build_synthetic_rock", "calibrate_from_grains", "build_like", "rasterize_grains"]
