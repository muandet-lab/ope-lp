#!/usr/bin/env python3
"""Why is B7 (favourite dominance + logged lower bounds) infeasible at scale under some clusterings?
Rebuilds the scale world and the clipped two-step centre, then checks each user's favourite against (i) its B7 logged lower bound and (ii) the dominance
rows, using the widest window the LC + level constraints allow at rho = eta = eps_max: [f + g - 2 rho, f + g + 2 rho] (a necessary condition; the
shared cell level makes the true joint conflict at least this large). Reports per clustering: how many users conflict, the gap sizes, and the smallest
soft tolerance t that would clear the per-user gaps. No LP solves.

    conda run -n offcem python scripts/offcem_synthetic/diag_b7_scale.py --clusterings dgp kmeans agglomerative spectral > results_offcem_global/logs/diag_b7_scale.txt
"""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import partial_ope_global as G, b3_lipschitz_experiment as B, b5_b7_experiment as X, scale_experiment as SE

ap = argparse.ArgumentParser(); ap.add_argument("--clusterings", nargs="+", default=["kmeans", "spectral"]); ap.add_argument("--K", type=int, default=1000); ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--users", type=int, default=200); ap.add_argument("--C", type=int, default=50); ap.add_argument("--seed", type=int, default=7000); ap.add_argument("--lam", type=float, default=100.0); ap.add_argument("--z", type=float, default=3.5)
a = ap.parse_args()
W = G.make_world(G.WorldCfg(K=a.K, n=a.n, users=a.users, C=a.C, def_frac=0.5, exclusion="round_random", sigma=3.0, beta=-0.1, eps_policy=0.2, clustering="original", seed=a.seed))
for method in a.clusterings:
    lab = SE.labels(W, method, a.seed)
    P = B.prep(a.seed, user_folds=True, W=W, lab=lab, model="two_step", lam=a.lam, clip="cluster_logged"); P["gh_cell"] = P["gh"]
    rho = P["cal"]["eps_max"]; eta = rho; nV = P["nV"]; mp, gh = P["mp"], P["gh"]
    coo = P["A_base"].tocoo(); n_pairs = P["n_pairs"]
    sel = (coo.row < n_pairs) & (coo.col < nV); v_of = np.zeros(n_pairs, int); v_of[coo.row[sel]] = coo.col[sel]     # LP row r (first block) -> its variable
    lo_w = np.full(nV, -np.inf); hi_w = np.full(nV, np.inf)
    np.maximum.at(lo_w, v_of, mp + gh - rho - eta); np.minimum.at(hi_w, v_of, mp + gh + rho + eta)          # intersection over the variable's rows (one per logged-round cell)
    lo_w = np.maximum(lo_w, P["LO"]); hi_w = np.minimum(hi_w, P["HI"])
    csum = np.bincount(v_of, weights=mp + gh, minlength=nV); ccnt = np.bincount(v_of, minlength=nV); centre = csum / np.maximum(ccnt, 1)   # mean LC centre per variable
    qv = P["qu"][P["v_user"], P["v_act"]]
    (va, vs), lb7, ub7, info = X.b7_rows(P, a.z)
    print(f"\n=== {method}: eps_max = {rho:.2f} (rho = eta), LP vars {nV}, dominance rows {len(va)}, users with favourite as LP variable {info['users_star_is_variable']}", flush=True)
    stars = np.unique(vs); has_lb = lb7[stars] > P["LO"]
    gap_lb = lb7[stars] - hi_w[stars]
    print(f"   (i) logged lower bounds on favourites: {has_lb.sum()} of {len(stars)}; lb median {np.median(lb7[stars][has_lb]):.2f}, max {lb7[stars][has_lb].max():.2f}; favourite window tops median {np.median(hi_w[stars]):.2f}")
    print(f"       conflicts (lb > window top): {(gap_lb > 1e-9).sum()} users; gap p50 {np.median(gap_lb[gap_lb > 1e-9]) if (gap_lb > 1e-9).any() else 0:.2f}, max {gap_lb.max():.2f}")
    viol = lo_w[va] - hi_w[vs]; bad = viol > 1e-9
    print(f"   (ii) dominance rows violated by the windows alone: {bad.sum()} of {len(va)} over {len(np.unique(P['v_user'][vs[bad]]))} users; worst {viol.max():.2f}; p90 among violating {np.percentile(viol[bad], 90) if bad.any() else 0:.2f}")
    need = np.zeros(len(stars))
    for i, s in enumerate(stars):
        m = vs == s
        need[i] = max(lo_w[va[m]].max() if m.any() else -np.inf, lb7[s]) - hi_w[s]
    print(f"   per-user requirement (max over rows and lb, minus window top): > 0 for {(need > 1e-9).sum()} users; p50 {np.median(need):.2f}; p90 {np.percentile(need, 90):.2f}; max {need.max():.2f}  => smallest uniform t clearing all users ~ {max(need.max(), 0):.1f}")
    worst = stars[np.argsort(-need)[:5]]
    print("   five worst favourites (centre f+g, true q, window top, lb7, max other-action window bottom):", [(round(centre[s], 1), round(qv[s], 1), round(hi_w[s], 1), round(lb7[s], 1), round(lo_w[va[vs == s]].max(), 1)) for s in worst])
    print(f"   favourites: centre f+g median {np.median(centre[stars]):.2f}, true q median {np.median(qv[stars]):.2f}, q - centre median {np.median(qv[stars] - centre[stars]):.2f}; rows per favourite variable median {np.median(ccnt[stars]):.0f}")
    print(f"   truth check at eps_max: favourites with true q outside their window: {((qv[stars] < lo_w[stars] - 1e-9) | (qv[stars] > hi_w[stars] + 1e-9)).sum()} of {len(stars)}", flush=True)
