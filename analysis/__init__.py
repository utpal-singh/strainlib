"""Analysis: spatial statistics (and, once methods exist, metrics/benchmark)."""
from __future__ import annotations

from .spatial_stats import pair_correlation, clark_evans_index, fit_thomas_pcf, bounding_box
from .metrics import error_metrics, benchmark
from .phase3 import (load_results, accuracy_surface, fit_failure_rates,
                     select_mstar, fit_bias_correction,
                     test_mcar_invariance, run_phase3)

__all__ = ["pair_correlation", "clark_evans_index", "fit_thomas_pcf", "bounding_box",
           "error_metrics", "benchmark",
           "load_results", "accuracy_surface", "fit_failure_rates",
           "select_mstar", "fit_bias_correction",
           "test_mcar_invariance", "run_phase3"]
