"""factorial CLI (run as a package module so joblib/loky can re-import it
safely on Windows spawn):

    python -m strainlib.run --orient 1 --realis 8 --njobs 8
    python -m strainlib.run --subset               # M4 + M8 on a small grid

D5: the fast methods run over the full grid (tuned); M4 MLM (~43 s/call) and M8 GLD
run on a representative subset.  Every method set is centre-family only (grain
centroids); the shape estimators (M5, M11, M15, M16, M19) are out of the paper's scope —
see :mod:`strainlib.methods`.
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from . import io
from .synthetic import calibrate_from_grains
from .runner import run_factorial, estimate_run_size
from .experiment import DEFAULT_GRID

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: the original paper's centre methods (its sixth, M5 SPO, reads grain shapes and is dropped)
FAST = ["M1_fry", "M2_cfm", "M3_dtm", "M6_fry2d", "M7_cfm2d"]
SLOW = ["M4_mlm", "M8_gld"]
#: strain-inversion branch: the paper's fast set + published CFM (outer edge; pre-declared and
#: wide-limit variant) + anisotropic-Thomas likelihood reference
CORRECTED = FAST + ["M9_cfm_ww", "M9w_cfm_ww", "R1_thomas_cl"]


def _method_set(name):
    """Named method sets defined after launch (kept out of CORRECTED so a pass that is
    already running keeps its method list): 'extra' = the second-pass estimators."""
    if name in ("extra", "all", "kb", "full"):
        from .methods import METHODS_ALL
        base = {"extra": EXTRA, "all": CORRECTED + EXTRA, "kb": KB, "full": CORRECTED + EXTRA + KB}[name]
        return [m for m in base if m in METHODS_ALL]
    return None


#: second pass — literature (KB) and new estimators, run on the identical observations
EXTRA = ["M10_nfry", "M12_isotropy", "M14_sepl"]
#: knowledge-base survey pass — literature estimators as published, in 3-D
KB = ["M17_void_fit", "M18_dtm_ls"]
#: M1 (and M10, untuned) use every short pair by default since 2026-10-17 (audit E3), so M1's
#: max_pairs applies only with pairs="sample"; M2's full-lag random sample is unchanged
TUNED = {"M1_fry": {"n_dirs": 150, "max_pairs": 200_000},
         "M2_cfm": {"n_dirs": 200, "max_pairs": 200_000}}


def main(argv=None):
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="no21")
    ap.add_argument("--orient", type=int, default=DEFAULT_GRID["n_orient"])
    ap.add_argument("--realis", type=int, default=DEFAULT_GRID["n_realisations"])
    ap.add_argument("--njobs", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_ROOT, "results"))
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap configs/realisation (smoke test)")
    ap.add_argument("--only-realis", type=int, default=None, help="run just this realisation index")
    ap.add_argument("--grainscale", type=float, default=1.0, help="scale grain count (kappa) for speed")
    ap.add_argument("--methods", default="fast",
                    help="'fast' (paper M1-M3, M6, M7), 'corrected' (fast + M9_cfm_ww, M9w_cfm_ww, "
                         "R1_thomas_cl), 'extra', 'kb', 'all', 'full' (all 13 centre methods) "
                         "or a comma-separated list of method ids")
    ap.add_argument("--part", default=None, help="i/n: run every n-th config from i (1-based), e.g. 1/2")
    ap.add_argument("--tag", default=None, help="output-name tag for an additional pass (e.g. 'extra')")
    ap.add_argument("--pcf-edge", default="translation", choices=["translation", "none"],
                    help="g(r) edge correction of the Thomas calibration; 'none' reproduces the "
                         "uncorrected calibration used before 17 October 2026")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated detection scenarios to score (default: all 13), e.g. mcar_p0.60")
    args = ap.parse_args(argv)
    scenarios = args.scenarios.split(",") if args.scenarios else None
    method_ids = ({"fast": FAST, "corrected": CORRECTED}.get(args.methods, None)
                  or _method_set(args.methods) or args.methods.split(","))
    part = None
    if args.part:
        pi, pn = (int(x) for x in args.part.split("/"))
        part = (pi - 1, pn)

    real = io.load_ct_csv(os.path.join(_ROOT, "data", "xenoliths_ct", f"{args.sample}.csv"))
    params = calibrate_from_grains(real, pcf_edge=args.pcf_edge)
    params["kappa"] *= args.grainscale
    voxel = float(np.percentile(real.equivalent_diameters(), 2) * 0.6)
    realis_list = [args.only_realis] if args.only_realis is not None else None

    if args.subset:
        Rs, k, Wk, orient, realis = [1.5, 3.0, 5.0], [0.5, 1.0, 2.0], [0.0, 0.71], 2, 3
        size = estimate_run_size(Rs, k, Wk, orient, realis, n_methods=len(SLOW))
        print(f"SUBSET slow-method run ({args.sample}): {size['n_config_realisation']} config-realis, {SLOW}")
        res = run_factorial(params, os.path.join(args.out, "subset_slow"), Rs=Rs, k=k, Wk=Wk,
                            n_orient=orient, n_realisations=realis, method_ids=SLOW,
                            voxel_size=voxel, n_jobs=args.njobs)
    else:
        Rs, k, Wk = DEFAULT_GRID["Rs"], DEFAULT_GRID["k"], DEFAULT_GRID["Wk"]
        size = estimate_run_size(Rs, k, Wk, args.orient, args.realis, n_methods=len(method_ids))
        print(f"factorial ({args.sample}): Rs{len(Rs)}xk{len(k)}xWk{len(Wk)}x"
              f"orient{args.orient}xrealis{args.realis} = {size['n_config_realisation']:,} config-realis, "
              f"{size['n_estimates']:,} estimates, {method_ids}, part={args.part}")
        res = run_factorial(params, args.out, Rs=Rs, k=k, Wk=Wk, n_orient=args.orient,
                            n_realisations=args.realis, method_ids=method_ids, method_kwargs=TUNED,
                            voxel_size=voxel, n_jobs=args.njobs, limit=args.limit,
                            realis_list=realis_list, part=part, tag=args.tag, scenarios=scenarios,
                            manifest_extra={"pcf_edge": args.pcf_edge, "grainscale": args.grainscale,
                                            "sample": args.sample})
    print(f"DONE: {len(res['parquet'])} parquet files, wall={res['wall_seconds']/60:.1f} min -> {args.out}")
    print(f"manifest: git_sha={res['manifest']['git_sha']}, seed={res['manifest']['master_seed']}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
