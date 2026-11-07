#!/usr/bin/env python3
"""OffCEM-scale version of plots 04 + 10 (quadrant 1): one world, several estimation clusterings; per clustering: eps_max from logs, eps sweep (closed form, plot 04),
then beliefs B3 / B5 / B7 at eps_max and one grid level below (merged LP, plot 10, no recentring). Wall-clock time per clustering is recorded.

    cd "Test notebooks/scripts/offcem_synthetic"
    conda run -n offcem python scale_experiment.py --K 1000 --n 3000 --users 1000 --C 50 --clusterings kmeans agglomerative spectral --tag offcem_default
    -> ../../results_offcem_global/scale_<tag>_seed<seed>.json (saved after every solve; re-running skips finished items)   then: python plot_scale.py --tag offcem_default

Setup = the earlier setup of plot 04: original exclusions (redrawn per round), lambda = inf, m_hat cross-fitted by ROUND (--folds round; --folds user makes the merged LP exact), eta = rho, ridge centre, range [-30, 30].
DGP clusters are the repo's 'original' softmax-sampled clustering (C clusters); the estimation clustering is fitted on the action one-hot features and is one of
kmeans | agglomerative (Ward; 'hierarchical' is the same thing) | spectral | dgp (oracle labels). Beliefs as in b5_b7_experiment.py (L for B3, z-margin variants).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import argparse
import json
import time
import warnings
import numpy as np
from sklearn.metrics import adjusted_rand_score

import b3_lipschitz_experiment as B
import b5_b7_experiment as X
import eps_sweep_dgp as S
import make_slide_figures as F
import partial_ope_global as G
import clustering as CL

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results_offcem_global")
LO, HI = B.LO, B.HI


def labels(W, method, seed):
    if method == "dgp":
        return np.asarray(W["cl"]).astype(int)
    f = {"kmeans": CL.cluster_kmeans, "agglomerative": CL.cluster_agglomerative, "hierarchical": CL.cluster_agglomerative, "spectral": CL.cluster_spectral}[method]
    lab = np.asarray(f(np.asarray(W["action_x"], float), W["C"], random_state=seed)).astype(int)
    _, lab = np.unique(lab, return_inverse=True)            # ids 0..C'-1 (spectral may leave a cluster empty)
    return lab


def dump(d, path):
    json.dump(d, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def run(args):
    out = os.path.join(RES, f"scale_{args.tag}_seed{args.seed}.json")
    d = json.load(open(out)) if os.path.exists(out) else {}
    cfg = dict(K=args.K, n=args.n, users=args.users, C=args.C, def_frac=args.def_frac, exclusion="round_random", sigma=args.sigma, beta=-0.1, eps_policy=0.2, clustering="original", seed=args.seed)
    t0 = time.perf_counter()
    W = G.make_world(G.WorldCfg(**cfg))
    d["world"] = dict(cfg, folds=args.folds, build_seconds=time.perf_counter() - t0, n_rounds=int(W["n"]), n_users=int(len(np.unique(W["user_of_round"]))), dgp_clusters=int(W["C"]))
    d.setdefault("clusterings", {})
    print(f"world built in {d['world']['build_seconds']:.1f}s: {args.K} actions, {args.n} rounds, {d['world']['n_users']} users, {W['C']} DGP clusters", flush=True)
    for method in args.clusterings:
        tc = time.perf_counter()
        r = d["clusterings"].setdefault(method, {})
        r.setdefault("runs", {})
        lab = labels(W, method, args.seed)
        r["ari_vs_dgp"] = float(adjusted_rand_score(W["cl"], lab)); r["n_clusters"] = int(lab.max()) + 1
        P = B.prep(args.seed, user_folds=(args.folds == "user"), W=W, lab=lab)
        Wp, M = P["W"], P["M"]
        cal = P["cal"]
        if args.noise_sd is not None:                     # repeats too sparse for a data-only sigma: take the noise sd as known (an assumption; recorded in the JSON)
            cal["sigma_noise_estimated"] = cal["sigma_noise"]; cal["sigma_noise"] = float(args.noise_sd); cal["sigma_source"] = "given"
            cal["tau2_raw"] = cal["mean_half_sq_contrast"] - args.noise_sd ** 2; cal["tau"] = float(np.sqrt(max(cal["tau2_raw"], 0.0))); cal["eps_max"] = cal["z"] * cal["tau"]
            P["sigma"] = float(args.noise_sd)
        else:
            cal["sigma_source"] = "repeats"
        emax = cal["eps_max"]
        if not np.isfinite(emax):
            raise SystemExit(f"[{method}] eps_max is NaN: the noise sd could not be estimated ({cal['n_repeat_dof']} repeat dof from repeated (user, action) logged pairs, need >= 10). "
                             "Use fewer users or more rounds (the small world had 10 rounds per user), or pass --noise-sd <value> to take the noise sd as known.")
        grid = [float(x) for x in (np.linspace(1.0, emax, args.npts) if emax > 1.0 else [emax])]
        r.update(calibration=P["cal"], eta_hat=P["eta"], grid=grid, eps_max=emax, V=P["V"], V_plus_true=P["Vp_true"], V0_true=P["V0_true"], lp_mass=P["lp_mass"], supported=P["sup"],
                 n_merged_vars=int(P["nV"]), n_pairs=int(P["n_pairs"]), sigma_hat=P["sigma"], prep_seconds=time.perf_counter() - tc)
        print(f"[{method}] ARI {r['ari_vs_dgp']:.2f}  eps_max {emax:.2f}  grid {np.round(grid, 2).tolist()}  vars {P['nV']}  prep {r['prep_seconds']:.0f}s", flush=True)
        # ---- plot-04 sweep (closed form, per supported estimator)
        if "sweep" not in r:
            ts = time.perf_counter(); sw = {}
            for sp in B.SUPS:
                res = G.evaluate(Wp, M, {**F.SPLIT_SPEC, "overlap_lambda": None, "supported": sp, "eps_lc": [2 * g for g in grid]})
                sw[sp] = [dict(eps=g, ok=x["status"] == "optimal", status=x["status"], lo=x.get("with_se", [None, None])[0], hi=x.get("with_se", [None, None])[1],
                               lp_lo=x.get("unsupported_interval", [None, None])[0], lp_hi=x.get("unsupported_interval", [None, None])[1], covers_V=x.get("contains_V_with_se")) for g, x in zip(grid, res["results"])]
            r["sweep"] = sw; r["sweep_seconds"] = time.perf_counter() - ts; dump(d, out)
            print(f"[{method}] sweep {r['sweep_seconds']:.0f}s  DR at eps_max " + (f"[{sw['dr'][-1]['lo']:.2f}, {sw['dr'][-1]['hi']:.2f}]" if sw["dr"][-1]["ok"] else sw["dr"][-1]["status"]), flush=True)
        # ---- beliefs at eps_max and one level below
        rhos = grid[-2:] if len(grid) > 1 else grid
        qv = P["qu"][P["v_user"], P["v_act"]]
        mean, n = (P["ref_log"][0] + P["ref_log"][1]) / 2, P["ref_n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            hz = args.z * P["sigma"] / np.sqrt(n)
        P["ref_log_z"] = (np.where(n > 0, mean - hz, np.nan), np.where(n > 0, mean + hz, np.nan))
        lb3, ub3, av3 = B.box(P, args.L, P["ref_log"]); lb3z, ub3z, _ = B.box(P, args.L, P["ref_log_z"])
        b5 = {m: X.b5_bounds(P, m) for m in ("range", "q5")}
        b7 = {"B7_ord": X.b7_rows(P, None), f"B7_z{args.z:g}": X.b7_rows(P, args.z)}
        w = P["coef_v"]
        r["beliefs_info"] = dict(L=args.L, z=args.z, b3_refs_per_var_median=float(np.median(av3)), b3_share_vars_with_ref=float((av3 >= 1).mean()),
                                 b3_true_outside_mass=float((w * ((qv < lb3 - 1e-9) | (qv > ub3 + 1e-9))).sum() / w.sum()),
                                 b3z_true_outside_mass=float((w * ((qv < lb3z - 1e-9) | (qv > ub3z + 1e-9))).sum() / w.sum()),
                                 b5={m: dict(v[2], true_outside_mass=float((w * ((qv < v[0] - 1e-9) | (qv > v[1] + 1e-9))).sum() / w.sum())) for m, v in b5.items()},
                                 b7={k: dict(v[3], true_violation_share_rows=float((qv[v[0][0]] > qv[v[0][1]] + 1e-9).mean()),
                                             true_outside_mass=float((w * ((qv < v[1] - 1e-9) | (qv > v[2] + 1e-9))).sum() / w.sum())) for k, v in b7.items()})
        for rho in rhos:
            P["eta"] = rho
            lb5r, ub5r, _ = b5["range"]; lb5q, ub5q, _ = b5["q5"]
            domo, lb7o, ub7o, _ = b7["B7_ord"]; domz, lb7z, ub7z, _ = b7[f"B7_z{args.z:g}"]
            items = [("baseline", np.full(P["nV"], LO), np.full(P["nV"], HI), False, None), ("B3", lb3, ub3, True, None), (f"B3_z{args.z:g}", lb3z, ub3z, True, None),
                     ("B5_range", lb5r, ub5r, False, None), ("B5_q5", lb5q, ub5q, False, None), ("B7_ord", lb7o, ub7o, False, domo), (f"B7_z{args.z:g}", lb7z, ub7z, False, domz),
                     (f"B5_q5+B7_z{args.z:g}", np.maximum(lb5q, lb7z), np.minimum(ub5q, ub7z), False, domz),
                     (f"B3z+B5_q5+B7_z{args.z:g}", np.maximum(np.maximum(lb3z, lb5q), lb7z), np.minimum(np.minimum(ub3z, ub5q), ub7z), True, domz)]
            for name, lb, ub, coup, dm in items:
                if args.beliefs and name not in args.beliefs and name != "baseline":
                    continue
                key = f"{rho}|{name}"
                if key in r["runs"] and not args.redo:
                    continue
                lo0, up0, info, secs = X.solve_both(P, rho, lb, ub, L=args.L, coupling=coup, dom=dm, tl=args.time_limit, max_it=args.max_it)
                rr = B.rec(P, rho, args.L if coup else None, name, lo0, up0, info, secs)
                if lo0 is not None:
                    rr["box_tightening"] = dict(share_lb=float((lb > LO + 1e-9).mean()), share_ub=float((ub < HI - 1e-9).mean()))
                r["runs"][key] = rr; r["elapsed_seconds"] = time.perf_counter() - tc; dump(d, out)
                print(f"[{method}] rho {rho:.2f} {name:22s} {secs:7.1f}s " + (f"status {info['status']} bad {info.get('n_bad')}" if lo0 is None else
                      f"LP [{lo0:.2f}, {up0:.2f}] w {up0 - lo0:.2f}  DR [{rr['dr']['lo']:.2f}, {rr['dr']['hi']:.2f}] {'V in' if rr['dr']['contains_V'] else 'V OUT'}  cuts {info['cuts']}"), flush=True)
        r["elapsed_seconds"] = time.perf_counter() - tc; r["done"] = True; dump(d, out)
        print(f"[{method}] DONE  wall clock {r['elapsed_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--K", type=int, default=1000); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--users", type=int, default=200); ap.add_argument("--C", type=int, default=50)
    ap.add_argument("--def-frac", type=float, default=0.5); ap.add_argument("--sigma", type=float, default=3.0, help="DGP reward noise sd")
    ap.add_argument("--noise-sd", type=float, default=None, help="take the noise sd as KNOWN for calibration and reference boxes (when repeated logged pairs are too sparse to estimate it)")
    ap.add_argument("--clusterings", nargs="+", default=["kmeans", "agglomerative", "spectral"], choices=("kmeans", "agglomerative", "hierarchical", "spectral", "dgp"))
    ap.add_argument("--npts", type=int, default=5, help="eps grid points in [1, eps_max] (plot 04 used 5)")
    ap.add_argument("--L", type=float, default=3.0); ap.add_argument("--z", type=float, default=3.5)
    ap.add_argument("--beliefs", nargs="*", default=None, help="subset of belief names to run (default all)")
    ap.add_argument("--time-limit", type=float, default=3600.0); ap.add_argument("--max-it", type=int, default=8)
    ap.add_argument("--folds", default="round", choices=("round", "user"), help="cross-fitting folds for m_hat: round (plot-04 setup; merged LP can be infeasible when fold models disagree) or user (merged LP = closed form)")
    ap.add_argument("--tag", default="offcem_default"); ap.add_argument("--redo", action="store_true")
    run(ap.parse_args())
