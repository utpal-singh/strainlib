"""CT detection-loss models (MCAR, MNAR Beer–Lambert, resolution cut)."""
from __future__ import annotations

from .thinning import mcar_thin, mnar_beer_lambert, resolution_cut, detection_scenarios

__all__ = ["mcar_thin", "mnar_beer_lambert", "resolution_cut", "detection_scenarios"]
