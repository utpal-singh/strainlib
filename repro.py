"""Reproducibility utilities.

A single deterministic seed policy for the whole pipeline, plus a run-manifest
builder so every cached result and every figure traces back to the exact code,
environment, and seeds that produced it.

Seed policy
-----------
One integer ``master_seed`` defines a run.  Every stochastic unit of work draws
its generator from :func:`child_rng(master_seed, *key)`, where ``key`` identifies
the work (ints or strings).  Because the child generator is derived from a
:class:`numpy.random.SeedSequence` built from ``[master_seed, *key]``, it is
**independent of evaluation order and of parallelism** — the same (master_seed, key)
always yields the same stream, serial or across joblib workers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as _im

import numpy as np

__all__ = ["child_seed_sequence", "child_rng", "config_hash", "git_sha", "git_dirty",
           "package_versions", "run_manifest", "DEFAULT_MASTER_SEED"]

DEFAULT_MASTER_SEED = 20240601


def _key_to_int(k) -> int:
    """Map a work-item key component (int or str) to a non-negative int.

    Strings (e.g. a deformation/sample label) are hashed, so work can be keyed by
    stable *identity* rather than list position — making the stream independent of
    ordering/subsetting.
    """
    if isinstance(k, (int, np.integer)):
        return int(k)
    return int.from_bytes(hashlib.sha256(str(k).encode("utf-8")).digest()[:8], "big")


def child_seed_sequence(master_seed: int, *key) -> np.random.SeedSequence:
    """Deterministic child :class:`SeedSequence` for a unit of work (ints or strings)."""
    return np.random.SeedSequence([int(master_seed)] + [_key_to_int(k) for k in key])


def child_rng(master_seed: int, *key) -> np.random.Generator:
    """Deterministic :class:`numpy.random.Generator` for a unit of work."""
    return np.random.default_rng(child_seed_sequence(master_seed, *key))


def _to_native(obj):
    """Coerce numpy scalars/arrays to native Python for stable, order-safe JSON."""
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj


def config_hash(config: dict) -> str:
    """Stable 16-hex-char hash of a config dict (numpy-safe, order-insensitive)."""
    blob = json.dumps(_to_native(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _git(*args) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


def git_sha(short: bool = False) -> str | None:
    return _git("rev-parse", *(["--short"] if short else []), "HEAD") or None


def git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    return None if status is None else (status != "")


def _thread_env() -> dict:
    import os
    return {v: os.environ.get(v) for v in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def package_versions(names: list[str] | None = None) -> dict:
    if names is None:
        names = ["numpy", "scipy", "pandas", "joblib", "statsmodels", "pyarrow", "scikit-image"]
    out = {}
    for name in names:
        try:
            out[name] = _im.version(name)
        except Exception:
            out[name] = "not-installed"
    return out


def run_manifest(master_seed: int, design: dict | None = None, grid_size: int | None = None,
                 n_scenarios: int | None = None, methods: list | None = None,
                 extra: dict | None = None) -> dict:
    """Provenance manifest tracing a result to code+env+design+seed."""
    from . import __version__ as strainlib_version
    m = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": int(master_seed),
        "git_sha": git_sha(short=False),
        "git_dirty": git_dirty(),
        "strainlib_version": strainlib_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "threads": _thread_env(),
        "grid_size": grid_size,
        "n_scenarios": n_scenarios,
        "methods": methods,
        "design": _to_native(design) if design is not None else None,
        "design_hash": config_hash(design) if design is not None else None,
    }
    if extra:
        m["extra"] = _to_native(extra)
    return m
