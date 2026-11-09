#!/usr/bin/env python3
"""B3: Lipschitz smoothness across users for the same action, added to the unsupported-pairs LP.   Seed 7000, K-means clusters (K = 10), original exclusions, lambda = inf,
user-level cross-fitting, eta FIXED at eta_hat (7.38).   B1 and B2 are NOT used here.

    cd "Test notebooks/scripts/offcem_synthetic"
    conda run -n offcem python b3_lipschitz_experiment.py --rhos 5.85 6.27 7.47 --Ls 1.0 1.35 2.0 3.0 4.4 --modes refs oracle_logged oracle_all          # box constraints only
    conda run -n offcem python b3_lipschitz_experiment.py --rhos 5.85 --Ls 1.35 2.0 --modes coupling --time-limit 120                                      # + unsupported-to-unsupported coupling
    -> ../../results_offcem_global/b3_lipschitz_seed7000.json (appended per invocation, saved after every solve)

Belief:  |mu_{u,a} - mu_{v,a}| <= L d(x_u, x_v) for users u != v and the same action a;  d = Euclidean distance between STANDARDISED user covariates (100 users). Distances are fixed, so every
instance is two linear inequalities. Applied irrespective of the learned cluster assignments.
 (4) supported-to-unsupported: for every unsupported (u,a) and every other user v with a LOGGED reward for a, the reference interval [q_lo(v,a), q_hi(v,a)] (logged mean +- 1.96 sigma_hat / sqrt(n),
     sigma_hat from repeated logged rounds) gives   q_lo - L d <= mu_ua <= q_hi + L d.   These only change variable bounds; all references are intersected with range, LC and band.
     Positive support does not guarantee an observed reward: reference availability is reported. The cluster-only prediction is NOT used as a reference.
 (5) unsupported-to-unsupported (mode 'coupling' = references + coupling): |mu_ua - mu_va| <= L d for the k nearest users per (action) first, then a cutting-plane loop adds every violated pair
     (checked over ALL user pairs of the same action) and re-solves until the full check passes.
Merging: under original exclusions each round is its own unit, but x and q(x, a) are identical across a user's rounds (verified), so mu is ONE variable per (user, action). The LC pair constraint stays per
round (its own level b_{round, cluster}). The merged baseline reproduces the closed-form baseline (checked below).
Modes: refs (estimated logged references), oracle_logged (exact true q at the same logged (v,a) pairs: isolates reference ACCURACY), oracle_all (exact true q for every other user: isolates AVAILABILITY / what
smoothness alone could give), coupling (refs + coupling). Oracle modes use true rewards and are diagnostics only. 'Contains' flags are empirical containment, not assumption checks.
DGP note: q(x,a) is linear in x with an action-specific gradient (OBP linear_reward_function, degree 1), so it is continuous and Lipschitz; there are no threshold jumps to model.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import time
import warnings
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from sklearn.cluster import KMeans

import make_slide_figures as F
import partial_ope_global as G
import eps_sweep_dgp as S
import eta_separation_experiment as E
import extra_beliefs_experiment as X

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results_offcem_global", "b3_lipschitz_seed7000.json")
SUPS = ("dr", "ips", "offcem")
LO, HI = -30.0, 30.0
KNN = 3


def prep(seed, user_folds=True, W=None, lab=None, model="cluster_level", lam=None, limits=None, calibrate=None, sigma_ref=None, alpha=1.0, clip=None):
    """W = None: the small world of plots 03-10 (SPLIT_WORLD, original exclusions) with K-means labels. Otherwise a prebuilt world and an action-label vector (any clustering).
    lam: overlap threshold defining the LP region (None = pi0 == 0 only). limits: outcome range (default [-30, 30]). calibrate: eps_max calibration function (default eps_sweep_dgp.calibrate_user_cells).
    sigma_ref: noise sd used for logged reference intervals (default: the calibration's sigma_noise)."""
    lo_, hi_ = (LO, HI) if limits is None else (float(limits[0]), float(limits[1]))
    if W is None:
        W = G.make_world(G.WorldCfg(**{**F.SPLIT_WORLD, "seed": seed, "exclusion": "round_random"}))
    km = KMeans(n_clusters=W["C"], n_init=10, random_state=0).fit_predict(np.asarray(W["action_x"], float)).astype(int) if lab is None else np.asarray(lab).astype(int)
    Wp = {**W, "cl": km, "C": int(km.max()) + 1}
    M = G.get_model(Wp, model, "ridge", 5, alpha, 1, user_folds=user_folds)
    if limits is not None:
        M = np.clip(M, lo_, hi_)                        # the centre is a prediction of a mean in the outcome range (e.g. a probability)
    clip_info = None
    if clip == "cluster_logged":                        # option (4): clip the centre to the range of LOGGED rewards of the action's cluster (data only; removes extreme predictions)
        clr = Wp["cl"][W["act"]]; y = np.asarray(W["y"], float)
        ymin = np.array([y[clr == c].min() if (clr == c).any() else lo_ for c in range(Wp["C"])]); ymax = np.array([y[clr == c].max() if (clr == c).any() else hi_ for c in range(Wp["C"])])
        lo_c, hi_c = ymin[Wp["cl"]][None, :], ymax[Wp["cl"]][None, :]
        M0 = M; M = np.clip(M, lo_c, hi_c)
        clip_info = dict(rule="cluster_logged", share_clipped=float((M != M0).mean()), mean_abs_change=float(np.abs(M - M0).mean()), max_abs_change=float(np.abs(M - M0).max()))
    cal = (calibrate or S.calibrate_user_cells)(Wp, M)
    et = E.eta_hat(Wp, M, cal)
    st = X.structures(Wp, M, lam, alpha)
    user = np.asarray(W["user_of_round"]); _, inv_round = np.unique(user, return_inverse=True)     # round -> user index
    nu, K, C = inv_round.max() + 1, W["K"], Wp["C"]
    nU = len(W["X_unit"]); inv = np.zeros(nU, int); inv[W["uid"]] = inv_round                      # unit -> user index (round_random: unit = round; user_fixed: unit = user)
    first = np.array([np.flatnonzero(inv == k)[0] for k in range(nu)])                              # one unit per user
    Xu, qu = W["X_unit"][first], W["q_unit"][first]
    Zs = (Xu - Xu.mean(0)) / Xu.std(0)
    D = np.sqrt(((Zs[:, None] - Zs[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    P = dict(W=Wp, M=M, cal=cal, eta=et["eta_hat"], st=st, inv=inv, nu=nu, K=K, C=C, D=D, qu=qu, sigma=(cal["sigma_noise"] if sigma_ref is None else float(sigma_ref)), seed=seed, lam=lam, LO=lo_, HI=hi_, limits=[lo_, hi_], alpha=alpha, clip_info=clip_info)
    U, ident, wu, pi = st["U"], st["ident"], st["wu"], st["pi"]
    pu, pa = np.nonzero(U); pc = km[pa]; keep = ident[pu, pc]
    coef_all = wu[pu] * pi[pu, pa]
    P["const_lo"], P["const_up"] = float((coef_all[~keep] * lo_).sum()), float((coef_all[~keep] * hi_).sum())
    iu, ia, ic, coef = pu[keep], pa[keep], pc[keep], coef_all[keep]
    vkeys, vid = np.unique(inv[iu] * K + ia, return_inverse=True)
    nV = len(vkeys); P.update(nV=nV, v_user=vkeys // K, v_act=vkeys % K, vid=vid)
    ckeys, cid = np.unique(iu * C + ic, return_inverse=True)
    Kc = len(ckeys); cu, cc = ckeys // C, ckeys % C
    P.update(Kc=Kc, mp=M[iu, ia], coef_v=np.bincount(vid, coef, nV), gh=st["G"][cu, cc], gh_se=st["S"][cu, cc], cell_rows=st["rows_per_cluster"][cc], n_pairs=len(iu), n_nonident=int((~keep).sum()))
    rr = np.arange(len(iu))
    P["A_base"] = coo_matrix((np.concatenate([np.ones(len(iu)), -np.ones(len(iu)), -np.ones(len(iu)), np.ones(len(iu))]),
                              (np.concatenate([rr, rr, rr + len(iu), rr + len(iu)]), np.concatenate([vid, nV + cid, vid, nV + cid]))), shape=(2 * len(iu), nV + Kc)).tocsr()
    VID = np.full((nu, K), -1); VID[P["v_user"], P["v_act"]] = np.arange(nV); P["VID"] = VID
    # logged references: mean reward of (user, action) pairs with logged rounds
    act, y = W["act"], W["y"]
    s, n = np.zeros((nu, K)), np.zeros((nu, K))
    np.add.at(s, (inv_round, act), y); np.add.at(n, (inv_round, act), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n; half = 1.96 * P["sigma"] / np.sqrt(n)
    P["ref_log"] = (np.where(n > 0, mean - half, np.nan), np.where(n > 0, mean + half, np.nan)); P["ref_n"] = n
    P["ref_orc_logged"] = (np.where(n > 0, qu, np.nan),) * 2
    P["ref_orc_all"] = (qu.copy(),) * 2
    # supported estimates and SEs (unchanged by the belief)
    sup = {}
    for sp in SUPS:
        r = G.evaluate(Wp, M, {**F.SPLIT_SPEC, "overlap_lambda": lam, "limits": [lo_, hi_], "ridge_alpha": alpha, "supported": sp, "eps_lc": [2 * 5.85 * (hi_ - lo_) / 60.0], "eta_value": et["eta_hat"]})
        d = r["data"]
        sup[sp] = dict(estimate=d["supported_estimate"], supported_se=d["supported_se"], total_se=d["total_se"], error=d["supported_error"])
        P.update(V=d["oracle_V"], Vp_true=d["oracle_supported_part"], V0_true=d["oracle_unsupported_part"], lp_mass=d["unsupported_policy_mass"])
    P["sup"] = sup
    return P


def box(P, L, ref):
    """Per-variable bounds from the reference boxes of OTHER users with the same action, stretched by L d. Loops over actions (memory O(users^2), not users^2 x actions)."""
    Rlo, Rhi = ref
    D, VID = P["D"], P["VID"]
    lo_, hi_ = P.get("LO", LO), P.get("HI", HI)
    lb, ub, avail = np.full(P["nV"], lo_), np.full(P["nV"], hi_), np.zeros(P["nV"], int)
    for a in range(P["K"]):
        us = np.flatnonzero(VID[:, a] >= 0)
        if not len(us):
            continue
        refs = np.flatnonzero(~np.isnan(Rlo[:, a]))
        vid = VID[us, a]
        avail[vid] = len(refs) - np.isin(us, refs)
        if not len(refs):
            continue
        Ds = L * D[np.ix_(us, refs)]                      # inf on the diagonal (a user is not its own reference)
        lb[vid] = np.maximum((Rlo[refs, a][None] - Ds).max(1), lo_); ub[vid] = np.minimum((Rhi[refs, a][None] + Ds).min(1), hi_)
    return lb, ub, avail


def knn_pairs(P):
    D, VID, nu, K = P["D"], P["VID"], P["nu"], P["K"]
    out = set()
    for a in range(K):
        idx = np.flatnonzero(VID[:, a] >= 0)
        if len(idx) < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        nb = np.argsort(sub, axis=1)[:, :KNN]
        for r in range(len(idx)):
            for c in nb[r]:
                if np.isfinite(sub[r, c]):
                    v1, v2 = VID[idx[r], a], VID[idx[c], a]
                    out.add((min(v1, v2), max(v1, v2)))
    e = np.array(sorted(out)); return e[:, 0], e[:, 1]


def violated(P, mu, L, tol=1e-6, cap=40000):
    MU = np.full((P["nu"], P["K"]), np.nan); MU[P["v_user"], P["v_act"]] = mu
    vs = []
    for a in range(P["K"]):
        idx = np.flatnonzero(~np.isnan(MU[:, a]))
        if len(idx) < 2:
            continue
        col = MU[idx, a]
        viol = np.abs(col[:, None] - col[None, :]) - L * P["D"][np.ix_(idx, idx)]
        i, j = np.where(np.triu(viol > tol, 1))
        for ii, jj in zip(i, j):
            vs.append((viol[ii, jj], P["VID"][idx[ii], a], P["VID"][idx[jj], a]))
    vs.sort(reverse=True)
    return vs[:cap], len(vs)


def lp(P, rho, lb, ub, L=None, pairs=None, tl=120.0, sign=1.0):
    nV, Kc = P["nV"], P["Kc"]
    lbs = np.concatenate([lb, P["gh"] - P["eta"]]); ubs = np.concatenate([ub, P["gh"] + P["eta"]])
    b = np.concatenate([P["mp"] + rho, -P["mp"] + rho])
    A = P["A_base"]
    if pairs is not None and len(pairs[0]):
        v1, v2 = pairs; e = len(v1); r = np.arange(e)
        Ac = coo_matrix((np.concatenate([np.ones(e), -np.ones(e), -np.ones(e), np.ones(e)]), (np.concatenate([r, r, r + e, r + e]), np.concatenate([v1, v2, v1, v2]))), shape=(2 * e, nV + Kc)).tocsr()
        A = vstack([A, Ac]).tocsr(); b = np.concatenate([b, L * P["D"][P["v_user"][v1], P["v_user"][v2]], L * P["D"][P["v_user"][v1], P["v_user"][v2]]])
    scale = P["coef_v"].max(); c = np.zeros(nV + Kc); c[:nV] = P["coef_v"] / scale
    r = linprog(sign * c, A_ub=A, b_ub=b, bounds=np.column_stack([lbs, ubs]), method="highs", options=dict(time_limit=tl, presolve=True))
    return r, scale


def solve_both(P, rho, lb, ub, L=None, coupling=False, tl=120.0, max_it=8):
    t0 = time.perf_counter()
    info = dict(iterations=0, cuts=0, full_check_passed=None, status="optimal")
    if (lb > ub + 1e-9).any():
        return None, None, dict(info, status="infeasible_bounds", n_bad=int((lb > ub + 1e-9).sum())), time.perf_counter() - t0
    res = []
    for sign in (1.0, -1.0):
        pairs = knn_pairs(P) if coupling else None
        for it in range(max_it if coupling else 1):
            r, scale = lp(P, rho, lb, ub, L, pairs, tl, sign)
            info["iterations"] = max(info["iterations"], it + 1)
            if not r.success:
                info["status"] = "infeasible" if r.status == 2 else f"solver_status_{r.status}"
                return None, None, info, time.perf_counter() - t0
            if not coupling:
                break
            vs, nviol = violated(P, r.x[:P["nV"]], L)
            if nviol == 0:
                info["full_check_passed"] = True if info["full_check_passed"] is None else info["full_check_passed"]
                break
            add = (np.array([v[1] for v in vs]), np.array([v[2] for v in vs]))
            pairs = (np.concatenate([pairs[0], add[0]]), np.concatenate([pairs[1], add[1]])); info["cuts"] = max(info["cuts"], len(pairs[0]))
            if it == max_it - 1:
                info["full_check_passed"] = False
        res.append(r.fun * scale)
    return res[0] + P["const_lo"], -res[1] + P["const_up"], info, time.perf_counter() - t0


def oracle_checks(P, L, lb, ub, ref_mode):
    qv = P["qu"][P["v_user"], P["v_act"]]
    bad = (qv < lb - 1e-9) | (qv > ub + 1e-9)
    w = P["coef_v"]
    out = dict(box_violation_share_mass=float((w * bad).sum() / w.sum()), box_violation_share_count=float(bad.mean()))
    if ref_mode == "refs":
        Rlo, Rhi = P["ref_log"]; m = ~np.isnan(Rlo)
        out["reference_interval_contains_truth"] = float(((P["qu"] >= Rlo - 1e-9) & (P["qu"] <= Rhi + 1e-9))[m].mean())
    return out


def lip_global(P, L):
    """Oracle: share of same-action user pairs (all pairs) whose TRUE means violate |dq| <= L d."""
    qu, D = P["qu"], P["D"]
    iu = np.triu_indices(P["nu"], 1)
    dq = np.abs(qu[iu[0]] - qu[iu[1]]); thr = L * D[iu][:, None]
    return float((dq > thr + 1e-9).mean())


def run(args):
    global OUT
    if args.out:
        OUT = args.out
    P = prep(args.seed, user_folds=(args.folds == "user"))
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d.setdefault("runs", {})
    nu, K = P["nu"], P["K"]
    s = np.abs(P["qu"][:, None, :] - P["qu"][None, :, :]) / np.where(np.isfinite(P["D"]), P["D"], np.inf)[:, :, None]
    iu = np.triu_indices(nu, 1); sl = s[iu[0], iu[1], :]
    Rlo, Rhi = P["ref_log"]
    avail = ((~np.isnan(Rlo)).sum(0)[None, :] - (~np.isnan(Rlo)))[P["v_user"], P["v_act"]]
    nearest = np.array([np.min(np.where(np.isnan(Rlo[:, a]) | (np.arange(nu) == u), np.inf, P["D"][u])) for u, a in zip(P["v_user"][:3000], P["v_act"][:3000])])
    d["setup"] = dict(seed=P["seed"], clusters="kmeans K=10", eta_hat=P["eta"], rho_max=P["cal"]["eps_max"], sigma_hat=P["sigma"], V=P["V"], V_plus_true=P["Vp_true"], V0_true=P["V0_true"], lp_mass=P["lp_mass"],
                      supported=P["sup"], n_users=int(nu), n_merged_vars=int(P["nV"]), n_pairs=int(P["n_pairs"]), n_nonident_pairs=P["n_nonident"],
                      distance=dict(median=float(np.median(P["D"][np.isfinite(P["D"])])), nearest_neighbour_median=float(np.median(P["D"].min(1)))),
                      oracle_slope=dict(median=float(np.median(sl)), p95=float(np.percentile(sl, 95)), p99=float(np.percentile(sl, 99)), max=float(sl.max()), per_action_max_median=float(np.median(sl.max(0)))),
                      reference_availability=dict(logged_user_action_pairs=int((P["ref_n"] > 0).sum()), pairs_with_2plus=int((P["ref_n"] > 1).sum()), refs_per_action_mean=float((P["ref_n"] > 0).sum(0).mean()),
                                                  unsupported_vars=int(P["nV"]), with_ge1_other_user_ref_share=float((avail >= 1).mean()), refs_per_var_median=float(np.median(avail)),
                                                  nearest_ref_distance_median=float(np.median(nearest[np.isfinite(nearest)]))))
    print("setup:", json.dumps({k: v for k, v in d["setup"].items() if k in ("distance", "oracle_slope", "reference_availability", "n_merged_vars", "n_pairs", "n_nonident_pairs")}), flush=True)
    d["setup"]["eta_mode"] = args.eta; d["setup"]["folds"] = args.folds
    for rho in args.rhos:
        if args.eta == "rho":
            P["eta"] = rho  # band allowance tied to the LC radius (the earlier setup), instead of the logged-data estimate eta_hat
        # baseline (merged) + closed-form check
        lb0, ub0 = np.full(P["nV"], LO), np.full(P["nV"], HI)
        if f"{rho}|baseline" not in d["runs"]:
            lo0, up0, info, secs = solve_both(P, rho, lb0, ub0, tl=args.time_limit)
            cf = G.evaluate(P["W"], P["M"], {**F.SPLIT_SPEC, "overlap_lambda": None, "supported": "dr", "eps_lc": [2 * rho], "eta_value": P["eta"]})["results"][0]["unsupported_interval"]
            d["runs"][f"{rho}|baseline"] = rec(P, rho, None, "baseline", lo0, up0, info, secs, closed_form=cf)
            print(f"rho {rho} baseline (merged LP) {secs:5.1f}s [{lo0:.4f}, {up0:.4f}]  closed form [{cf[0]:.4f}, {cf[1]:.4f}]", flush=True)
            json.dump(d, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        for L in args.Ls:
            for mode in args.modes:
                key = f"{rho}|{L}|{mode}"
                if key in d["runs"] and not args.redo:
                    continue
                ref = {"refs": P["ref_log"], "coupling": P["ref_log"], "oracle_logged": P["ref_orc_logged"], "oracle_all": P["ref_orc_all"]}[mode]
                lb, ub, av = box(P, L, ref)
                lo0, up0, info, secs = solve_both(P, rho, lb, ub, L=L, coupling=(mode == "coupling"), tl=args.time_limit, max_it=args.max_it)
                r = rec(P, rho, L, mode, lo0, up0, info, secs)
                r["oracle"] = oracle_checks(P, L, lb, ub, "refs" if mode in ("refs", "coupling") else mode)
                r["oracle"]["lipschitz_violation_all_user_pairs_true_means"] = lip_global(P, L)
                r["box_tightening"] = dict(share_vars_lb_tighter=float((lb > LO + 1e-9).mean()), share_vars_ub_tighter=float((ub < HI - 1e-9).mean()))
                d["runs"][key] = r
                json.dump(d, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
                print(f"rho {rho} L {L} {mode:13s} {secs:6.1f}s " + (f"status {info['status']} bad_vars {info.get('n_bad')}" if lo0 is None else f"LP-only [{lo0:.2f}, {up0:.2f}] w {up0 - lo0:.2f}  V0 {'in' if r['lp_contains_V0'] else 'OUT'}  iters {info['iterations']} cuts {info['cuts']} box-viol {r['oracle']['box_violation_share_mass']:.2f}"), flush=True)


def rec(P, rho, L, mode, lo0, up0, info, secs, closed_form=None):
    r = dict(rho=rho, L=L, mode=mode, seconds=secs, **info)
    if closed_form is not None:
        r["closed_form"] = list(closed_form)
    if lo0 is not None:
        r.update(lp_lo=lo0, lp_hi=up0, lp_contains_V0=bool(lo0 <= P["V0_true"] <= up0))
        for sp in SUPS:
            se, est = P["sup"][sp]["total_se"], P["sup"][sp]["estimate"]
            r[sp] = dict(lo=est + lo0 - 1.96 * se, hi=est + up0 + 1.96 * se, contains_V=bool(est + lo0 - 1.96 * se <= P["V"] <= est + up0 + 1.96 * se))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.85, 6.27, 7.47])
    ap.add_argument("--Ls", type=float, nargs="+", default=[1.0, 1.35, 2.0, 3.0, 4.4])
    ap.add_argument("--modes", nargs="+", default=["refs"], choices=("refs", "oracle_logged", "oracle_all", "coupling"))
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--max-it", type=int, default=8)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--eta", default="hat", choices=("hat", "rho"), help="band allowance: 'hat' = eta_hat estimated from logged data (default), 'rho' = eta set equal to each rho")
    ap.add_argument("--folds", default="user", choices=("user", "round"), help="cross-fitting folds for m_hat: by user (default, the fixed setup) or by round (the earlier setup of plots 03/04)")
    ap.add_argument("--out", default=None, help="output JSON (default: results_offcem_global/b3_lipschitz_seed7000.json)")
    run(ap.parse_args())
