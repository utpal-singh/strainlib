"""Fossen–Tikoff deformation gradients and the finite-strain ellipsoid."""
from __future__ import annotations

from .tensors import (StrainEllipsoid, pure_shear, simple_shear, velocity_gradient,
                      simultaneous_deformation, sequential_deformation, progressive_deformation,
                      deformation_from_targets, strain_ellipsoid, vorticity_number,
                      apply_to_grains, apply_to_points)

__all__ = ["StrainEllipsoid", "pure_shear", "simple_shear", "velocity_gradient",
           "simultaneous_deformation", "sequential_deformation", "progressive_deformation",
           "deformation_from_targets", "strain_ellipsoid", "vorticity_number",
           "apply_to_grains", "apply_to_points"]
