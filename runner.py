"""
Executes the forward–inverse loop over the (Rs × k × Wk × orientation) deformation
grid × the detection-scenario battery × the methods, for ``n_realisations``
independent synthetic rocks calibrated to a real sample.  Parallelised with joblib
(chunked so each task builds its rock once — no large data shipped), cached to
Parquet per realisation (resumable), with a provenance manifest.

Every configuration is imaged through a FIXED lab-frame scan window the size of the
calibration sample (:mod:`strainlib.observation`): a parent rock large enough for the
whole design is deformed, then cropped.  (Before this, the whole rock was deformed and
its outline — which carries the imposed strain — was visible to the estimators.)
"""
from __future__ import annotations

import itertools
import os
import time
import json
import numpy as np
import pandas as pd

from .synthetic import build_synthetic_rock
from .synthetic.shapes import random_rotation
from .deformation.tensors import steady_state_deformation, strain_ellipsoid
from .detection import detection_scenarios
from .observation import observe, parent_box, scan_window_like
from .methods import METHODS, METHODS_ALL
from .analysis.metrics import error_metrics, _accepts_rng
from .experiment import DEFAULT_GRID
from .repro import child_rng, DEFAULT_MASTER_SEED, run_manifest

__all__ = ["build_configs", "run_factorial", "estimate_run_size"]


def build_configs(Rs, k, Wk, n_orient, master_seed):
    """Deformation configs: list of (label, F, truth, meta) with nominal Rs/k/Wk/orient.

    Uses the **steady-state (simultaneous) Fossen-Tikoff** deformation: Wk sets the
    velocity-gradient style and the finite Rs is solved for, so the target finite
    strain is hit exactly at every Wk — including the pure-simple-shear end-member
    Wk=1, which diverges under the older sequential/gamma targeting.  Orientations
    (only when n_orient > 1; the paper uses one) are seeded by the combination index
    and the orientation index, and shared across realisations."""
    out = []
    for ci, (rs, kk, wk) in enumerate(itertools.product(Rs, k, Wk)):
        D0 = steady_state_deformation(rs, kk, wk)
        for o in range(n_orient):
            if n_orient > 1:
                Q = random_rotation(child_rng(master_seed, "orient", ci, o))
                F = Q @ D0 @ Q.T
            else:
                F = D0
            out.append((f"Rs{rs}_k{kk}_Wk{wk}_o{o}", F, strain_ellipsoid(F),
                        {"Rs_nom": rs, "k_nom": kk, "Wk": wk, "orient": o}))
    return out


def _eval_config(rock, cfg, realisation, p_hat, voxel_size, method_ids, method_kwargs, master_seed,
                 window, scenario_filter=None):
    label, F, truth, meta = cfg
    deformed = observe(rock, F, window)                 # deform the parent, crop to the fixed window
    scenarios = detection_scenarios(deformed, p_hat=p_hat, voxel_size=voxel_size,
                                    rng=child_rng(master_seed, "det", realisation, label))
    if scenario_filter is not None:                     # all 13 are drawn (same streams), only these scored
        scenarios = {k: v for k, v in scenarios.items() if k in scenario_filter}
    rows = []
    for scen_label, obs in scenarios.items():
        for mid in method_ids:
            row = {"realisation": realisation, "deformation": label, "scenario": scen_label,
                   "method": mid, "Rs_nom": meta["Rs_nom"], "k_nom": meta["k_nom"],
                   "Wk": meta["Wk"], "orient": meta["orient"], "n_grains": obs.n,
                   "Rs_true": truth.Rs, "k_true": truth.k}
            t0 = time.perf_counter()
            try:
                kw = dict(method_kwargs.get(mid, {}))
                if _accepts_rng(METHODS_ALL[mid]):
                    kw.setdefault("rng", child_rng(master_seed, realisation, label, scen_label, mid))
                est = METHODS_ALL[mid](obs, **kw)
                row.update(error_metrics(est, truth))
                row["fit_ok"] = True
                if "constrained" in est.extra:                # M9: fit pinned at the window limit
                    row["constrained"] = bool(est.extra["constrained"])
            except Exception as exc:  # noqa: BLE001
                row["fit_ok"] = False
                row["fail"] = type(exc).__name__
            row["runtime_s"] = time.perf_counter() - t0
            rows.append(row)
    return rows


def _run_chunk(params, realisation, chunk, p_hat, voxel_size, method_ids, method_kwargs,
               master_seed, chunk_idx=0, n_chunks=1, window=None, scenario_filter=None):
    """``params`` are the PARENT-rock parameters (box sized by :func:`parent_box`)."""
    rock = build_synthetic_rock(rng=child_rng(master_seed, "rock", realisation), **params)
    rows = []
    for j, cfg in enumerate(chunk):
        t0 = time.perf_counter()
        rows.extend(_eval_config(rock, cfg, realisation, p_hat, voxel_size,
                                 method_ids, method_kwargs, master_seed, window, scenario_filter))
        print(f"[r{realisation}] chunk {chunk_idx+1}/{n_chunks} config {j+1}/{len(chunk)} "
              f"{cfg[0]} {time.perf_counter()-t0:.1f}s", flush=True)
    return rows


def estimate_run_size(Rs, k, Wk, n_orient, n_realisations, n_scenarios=13, n_methods=7):
    n_configs = len(Rs) * len(k) * len(Wk) * n_orient
    return {"n_configs_per_realisation": n_configs,
            "n_config_realisation": n_configs * n_realisations,
            "n_estimates": n_configs * n_realisations * n_scenarios * n_methods}


def run_factorial(params, out_dir, Rs=None, k=None, Wk=None, n_orient=None, n_realisations=None,
                  method_ids=None, method_kwargs=None, p_hat=0.6, voxel_size=None,
                  master_seed=DEFAULT_MASTER_SEED, n_jobs=-1, chunk_size=None, limit=None,
                  realis_list=None, window_shape="box", part=None, tag=None, scenarios=None,
                  manifest_extra=None):
    """Run the factorial → per-realisation Parquet in ``out_dir`` + manifest. Resumable.

    ``realis_list`` runs only the given realisation indices (for launching several
    independent single-realisation processes in parallel — the reliable alternative
    to in-process joblib workers when the parent is detached on Windows).

    ``window_shape``: ``"box"`` images each configuration through a window with the
    calibration sample's extent ``params['box']`` (realistic, anisotropic);
    ``"sphere"`` uses the sphere of equal volume (orientation-neutral).

    ``part=(i, n)`` runs every n-th configuration starting at i (0-based) and writes
    ``realisation=RR.part{i+1}of{n}.parquet``, so one realisation can be spread over n
    processes.  The parent rock is sized from the FULL design and every random stream is
    keyed by config identity, so the union of the parts equals the unsplit run.

    ``tag`` appends ``.{tag}`` to output names, so a later pass with extra methods writes
    alongside an earlier one; it images the identical observations (same seeds), so the
    rows of both passes are paired config by config.

    ``scenarios`` (list of labels) scores only those detection scenarios; all 13 are still
    drawn from the same streams, so the kept observations equal those of a full run.
    ``manifest_extra`` (dict) is added to the manifest's ``extra``.
    """
    from joblib import Parallel, delayed
    Rs = Rs or DEFAULT_GRID["Rs"]; k = k or DEFAULT_GRID["k"]; Wk = Wk or DEFAULT_GRID["Wk"]
    n_orient = n_orient or DEFAULT_GRID["n_orient"]
    n_realisations = n_realisations or DEFAULT_GRID["n_realisations"]
    method_ids = method_ids or list(METHODS)
    method_kwargs = method_kwargs or {}
    os.makedirs(os.path.join(out_dir, "benchmark"), exist_ok=True)

    configs = build_configs(Rs, k, Wk, n_orient, master_seed)
    # fixed scan window = the calibration sample's extent; the parent rock covers F⁻¹(W)
    # for EVERY configuration of the design (pad: the Thomas generator's own 3σ parent
    # margin) — sized before any part/limit slicing so all parts share one parent rock
    window = scan_window_like(params["box"], window_shape)
    pbox = parent_box(window, [c[1] for c in configs], pad=3.0 * float(params.get("sigma", 0.0)))
    parent = dict(params, box=pbox)
    suffix = ""
    if part is not None:
        pi, pn = int(part[0]), int(part[1])
        configs = configs[pi::pn]
        suffix = f".part{pi + 1}of{pn}"
    if tag:
        suffix += f".{tag}"
    if limit:
        configs = configs[:limit]
    if chunk_size is None:
        chunk_size = max(1, len(configs) // 8)
    chunks = [configs[i:i + chunk_size] for i in range(0, len(configs), chunk_size)]

    t0 = time.perf_counter()
    written = []
    realisations = list(realis_list) if realis_list is not None else list(range(n_realisations))
    print(f"[runner] {len(configs)} configs x realisations {realisations}, "
          f"n_jobs={n_jobs} ({'serial' if n_jobs == 1 else 'parallel'})", flush=True)
    print(f"[runner] fixed {window_shape} window {tuple(round(2 * h, 1) for h in window.half)}; "
          f"parent rock {tuple(round(b, 1) for b in pbox)}", flush=True)
    for r in realisations:
        path = os.path.join(out_dir, "benchmark", f"realisation={r:02d}{suffix}.parquet")
        if os.path.exists(path):                       # resumable
            print(f"[runner] realisation {r}: cached, skip", flush=True)
            written.append(path); continue
        tr = time.perf_counter()
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_run_chunk)(parent, r, ch, p_hat, voxel_size, method_ids, method_kwargs,
                                master_seed, ci, len(chunks), window, scenarios)
            for ci, ch in enumerate(chunks))
        df = pd.DataFrame([row for chunk_rows in results for row in chunk_rows])
        df.to_parquet(path, index=False)
        written.append(path)
        print(f"[runner] realisation {r+1}/{n_realisations} done: {len(df)} rows, "
              f"{time.perf_counter()-tr:.0f}s (elapsed {(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    wall = time.perf_counter() - t0

    manifest = run_manifest(master_seed,
                            design={"Rs": Rs, "k": k, "Wk": Wk, "n_orient": n_orient,
                                    "n_realisations": n_realisations, "p_hat": p_hat},
                            grid_size=len(configs), n_scenarios=len(scenarios) if scenarios else 13,
                            methods=method_ids,
                            extra={"wall_seconds": wall, "n_jobs": n_jobs,
                                   "calibrated_to": params.get("sample_id"),
                                   "calibration": {k: params.get(k) for k in ("kappa", "mu", "sigma", "hardcore")},
                                   "scenarios": list(scenarios) if scenarios else "all",
                                   "method_kwargs": method_kwargs,
                                   "scan_window": {"shape": window.shape, "half": window.half},
                                   "parent_box": pbox, "part": part, "realisations": realisations,
                                   **(manifest_extra or {})})
    # one manifest per process when the run is split, so parallel writers never collide
    name = ("manifest.json" if realis_list is None and part is None and not tag else
            f"manifest_r{'-'.join(map(str, realisations))}{suffix}.json")
    os.makedirs(os.path.join(out_dir, "run_manifests"), exist_ok=True)
    with open(os.path.join(out_dir, "run_manifests", name), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return {"parquet": written, "wall_seconds": wall, "manifest": manifest}
