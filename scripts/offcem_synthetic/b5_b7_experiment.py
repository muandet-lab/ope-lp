#!/usr/bin/env python3
"""B5 (group-specific mean bounds) and B7 (favourite-action mean dominance) added to the merged unsupported-pairs LP, on the EARLIER setup of plot 04
(K-means K = 10, original exclusions, lambda = inf, seed 7000, by-ROUND folds, eta = rho), with and without importance-weighted recentring of the cell levels.

    cd "Test notebooks/scripts/offcem_synthetic"
    conda run -n offcem python b5_b7_experiment.py --rhos 5.85 7.47 --L 3.0
    -> ../../results_offcem_global/b5_b7_seed7000_roundfolds.json

Beliefs (all on the merged mu, one variable per (user, action) in identified cells):
 B3  |mu_ua - mu_va| <= L d(x_u, x_v): logged reference boxes + unsupported-to-unsupported coupling (b3_lipschitz_experiment).
 B5  L_c <= mu_ua <= U_c for every action a in K-means cluster c. From logs only: per logged (user, action) pair the box ybar +- 1.96 sigma_hat / sqrt(n);
     'range' = [min of box lows, max of box highs] over the cluster's logged pairs; 'q5' = [5th pct of box lows, 95th pct of box highs] (trimmed, less conservative).
 B7  mu_{u,a*(u)} >= mu_{u,a} for every other action a, a*(u) = argmax_a pi(a | x_u) (the target's favourite; the policy is known, no truth used).
     If (u,a) is logged instead of a variable, mu_ua is replaced by its logged box: mu_{u,a*} >= ybar_ua - 1.96 sigma/sqrt(n). If a* is itself logged (not a variable),
     the constraint becomes an upper bound mu_ua <= ybar_{u,a*} + 1.96 sigma/sqrt(n) on the other variables (dominance applied both ways, as stated).
 recentre  shift every identified cell level g_hat_{u,c} by one constant so that sum_u w_u pi_U(c|x_u) g_hat = mean over logged rounds of pi_U(c|x)/pi0(c|x) (y - m_hat)
     (the OffCEM-style cluster-weighted residual; partial_ope_global.evaluate(recentre="cluster_ips")). Logged data only.
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

import b3_lipschitz_experiment as B
import make_slide_figures as F
import partial_ope_global as G

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results_offcem_global", "b5_b7_seed7000_roundfolds.json")
LO, HI = B.LO, B.HI


def recentre_delta(P, lam_c=100.0):
    W, st, M = P["W"], P["st"], P["M"]
    cl, C = W["cl"], W["C"]
    uid, act, y = W["uid"], W["act"], W["y"]
    pi0, pi, U, Gh, ident, wu = W["pi0_unit"], st["pi"], st["U"], st["G"], st["ident"], st["wu"]
    resid, clr = y - M[uid, act], cl[act]
    pi0_c = np.stack([pi0[:, cl == c].sum(1) for c in range(C)], 1)
    pm_cell = np.stack([(pi[:, cl == c] * U[:, cl == c]).sum(1) for c in range(C)], 1)
    ok = pi0_c[uid, clr] > 0
    wt = np.minimum(np.where(ok, pm_cell[uid, clr] / np.where(ok, pi0_c[uid, clr], 1.0), 0.0), lam_c)
    wid = wu[:, None] * pm_cell * ident
    delta = float((np.mean(wt * resid) - (wid * np.where(ident, Gh, 0.0)).sum()) / wid.sum())
    return delta, dict(moment=float(np.mean(wt * resid)), level_sum=float((wid * np.where(ident, Gh, 0.0)).sum()), max_weight=float(wt.max()))


def b5_bounds(P, mode):
    """Per-cluster [L_c, U_c] from logged (user, action) boxes; returned per variable."""
    Rlo, Rhi = P["ref_log"]; cl = P["W"]["cl"]; C = P["C"]
    Lc, Uc = np.full(C, LO), np.full(C, HI); nref = np.zeros(C, int)
    for c in range(C):
        cols = np.flatnonzero(cl == c)
        lo, hi = Rlo[:, cols], Rhi[:, cols]; m = ~np.isnan(lo)
        nref[c] = int(m.sum())
        if m.sum() == 0:
            continue
        if mode == "range":
            Lc[c], Uc[c] = max(lo[m].min(), LO), min(hi[m].max(), HI)
        else:
            Lc[c], Uc[c] = max(np.percentile(lo[m], 5), LO), min(np.percentile(hi[m], 95), HI)
    return Lc[cl[P["v_act"]]], Uc[cl[P["v_act"]]], dict(L_c=Lc.tolist(), U_c=Uc.tolist(), n_logged_pairs=nref.tolist())


def b7_rows(P, z=None):
    """Favourite-action dominance. Returns (rows for A_ub x <= 0 as (v_a, v_star) pairs, extra lb per var, extra ub per var, info).
    z = None: ordering among LP variables only. z > 0: also use logged pairs with margin z sigma / sqrt(n) (z = 3.5 ~ multiplicity-corrected for ~10 logged pairs per user)."""
    W, st = P["W"], P["st"]
    pi = st["pi"]; inv = P["inv"]; VID = P["VID"]; nu, K = P["nu"], P["K"]
    Rlo, Rhi = P["ref_log"]; n_ref = P["ref_n"]; mean = (Rlo + Rhi) / 2
    first = np.array([np.flatnonzero(inv == k)[0] for k in range(nu)])
    astar = pi[first].argmax(1)
    lb_x, ub_x = np.full(P["nV"], LO), np.full(P["nV"], HI)
    pa, ps = [], []
    n_star_var = n_star_logged_only = n_star_none = 0
    for u in range(nu):
        a_s = astar[u]; vs = VID[u, a_s]
        others = np.flatnonzero((VID[u] >= 0) & (np.arange(K) != a_s))
        if vs >= 0:
            n_star_var += 1
            pa.append(VID[u, others]); ps.append(np.full(len(others), vs))
            logged = np.flatnonzero(~np.isnan(Rlo[u]) & (np.arange(K) != a_s))
            if z is not None and len(logged):
                lb_x[vs] = max(lb_x[vs], (mean[u, logged] - z * P["sigma"] / np.sqrt(n_ref[u, logged])).max())
        elif z is not None and not np.isnan(Rhi[u, a_s]):
            n_star_logged_only += 1
            ub_x[VID[u, others]] = np.minimum(ub_x[VID[u, others]], mean[u, a_s] + z * P["sigma"] / np.sqrt(n_ref[u, a_s]))
        else:
            n_star_none += 1
    pa, ps = np.concatenate(pa), np.concatenate(ps)
    return (pa, ps), lb_x, ub_x, dict(n_rows=int(len(pa)), users_star_is_variable=n_star_var, users_star_logged_only=n_star_logged_only, users_star_unavailable=n_star_none,
                                      vars_with_lb_from_logged=int((lb_x > LO).sum()), vars_with_ub_from_star=int((ub_x < HI).sum()))


def lp(P, rho, lb, ub, L, pairs, dom, tl, sign):
    nV, Kc = P["nV"], P["Kc"]
    lbs = np.concatenate([lb, P["gh"] - P["eta"]]); ubs = np.concatenate([ub, P["gh"] + P["eta"]])
    b = np.concatenate([P["mp"] + rho, -P["mp"] + rho]); blocks = [P["A_base"]]
    if pairs is not None and len(pairs[0]):
        v1, v2 = pairs; e = len(v1); r = np.arange(e)
        blocks.append(coo_matrix((np.concatenate([np.ones(e), -np.ones(e), -np.ones(e), np.ones(e)]), (np.concatenate([r, r, r + e, r + e]), np.concatenate([v1, v2, v1, v2]))), shape=(2 * e, nV + Kc)).tocsr())
        dd = L * P["D"][P["v_user"][v1], P["v_user"][v2]]; b = np.concatenate([b, dd, dd])
    if dom is not None:
        va, vs = dom; e = len(va); r = np.arange(e)
        blocks.append(coo_matrix((np.concatenate([np.ones(e), -np.ones(e)]), (np.concatenate([r, r]), np.concatenate([va, vs]))), shape=(e, nV + Kc)).tocsr())
        b = np.concatenate([b, np.zeros(e)])
    A = vstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
    scale = P["coef_v"].max(); c = np.zeros(nV + Kc); c[:nV] = P["coef_v"] / scale
    r = linprog(sign * c, A_ub=A, b_ub=b, bounds=np.column_stack([lbs, ubs]), method="highs", options=dict(time_limit=tl, presolve=True))
    return r, scale


def solve_both(P, rho, lb, ub, L=None, coupling=False, dom=None, tl=300.0, max_it=8):
    t0 = time.perf_counter(); info = dict(iterations=0, cuts=0, full_check_passed=None, status="optimal")
    if (lb > ub + 1e-9).any():
        return None, None, dict(info, status="infeasible_bounds", n_bad=int((lb > ub + 1e-9).sum())), time.perf_counter() - t0
    res = []
    for sign in (1.0, -1.0):
        pairs = B.knn_pairs(P) if coupling else None
        for it in range(max_it if coupling else 1):
            r, scale = lp(P, rho, lb, ub, L, pairs, dom, tl, sign)
            info["iterations"] = max(info["iterations"], it + 1)
            if not r.success:
                info["status"] = "infeasible" if r.status == 2 else f"solver_status_{r.status}"
                return None, None, info, time.perf_counter() - t0
            if not coupling:
                break
            vs, nviol = B.violated(P, r.x[:P["nV"]], L)
            if nviol == 0:
                info["full_check_passed"] = True if info["full_check_passed"] is None else info["full_check_passed"]
                break
            add = (np.array([v[1] for v in vs]), np.array([v[2] for v in vs]))
            pairs = (np.concatenate([pairs[0], add[0]]), np.concatenate([pairs[1], add[1]])); info["cuts"] = max(info["cuts"], len(pairs[0]))
            if it == max_it - 1:
                info["full_check_passed"] = False
        res.append(r.fun * scale)
    return res[0] + P["const_lo"], -res[1] + P["const_up"], info, time.perf_counter() - t0


def run(args):
    P = B.prep(args.seed, user_folds=False)
    gh0 = P["gh"].copy()
    mean, n = (P["ref_log"][0] + P["ref_log"][1]) / 2, P["ref_n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        hz = args.b7_z * P["sigma"] / np.sqrt(n)
    P["ref_log_z"] = (np.where(n > 0, mean - hz, np.nan), np.where(n > 0, mean + hz, np.nan))
    delta, rinfo = recentre_delta(P)
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d.setdefault("runs", {})
    d["setup"] = dict(seed=P["seed"], clusters="kmeans K=10", folds="round", eta_mode="rho", rho_max=P["cal"]["eps_max"], sigma_hat=P["sigma"], V=P["V"], V_plus_true=P["Vp_true"], V0_true=P["V0_true"],
                      lp_mass=P["lp_mass"], supported=P["sup"], n_users=int(P["nu"]), n_merged_vars=int(P["nV"]), recentre_delta=delta, recentre=rinfo, L=args.L, b5_modes=args.b5_modes)
    qv = P["qu"][P["v_user"], P["v_act"]]
    configs = {}
    for mode in args.b5_modes:
        lb5, ub5, i5 = b5_bounds(P, mode)
        i5["true_mean_outside_share_mass"] = float((P["coef_v"] * ((qv < lb5 - 1e-9) | (qv > ub5 + 1e-9))).sum() / P["coef_v"].sum())
        configs[f"B5_{mode}"] = (lb5, ub5, i5)
    b7 = {}
    for name, z in (("B7_ord", None), (f"B7_z{args.b7_z:g}", args.b7_z)):
        dom, lb7, ub7, i7 = b7_rows(P, z)
        # oracle diagnostics of the dominance belief (true means): share of rows where q_{u,a} > q_{u,a*}; mass whose true mean is outside the logged-derived bounds
        i7["true_violation_share_rows"] = float((qv[dom[0]] > qv[dom[1]] + 1e-9).mean())
        i7["true_mean_outside_bounds_share_mass"] = float((P["coef_v"] * ((qv < lb7 - 1e-9) | (qv > ub7 + 1e-9))).sum() / P["coef_v"].sum())
        b7[name] = (dom, lb7, ub7, i7)
    d["setup"]["b5"] = {k: v[2] for k, v in configs.items()}; d["setup"]["b7"] = {k: v[3] for k, v in b7.items()}
    print("recentre delta", round(delta, 3), rinfo, "\nB5", {k: v[2] for k, v in configs.items()}, "\nB7", d["setup"]["b7"], flush=True)
    for rec_on in (False, True):
        P["gh"] = gh0 + (delta if rec_on else 0.0)
        tag = "recentre" if rec_on else "norecentre"
        for rho in args.rhos:
            P["eta"] = rho
            cf = G.evaluate(P["W"], P["M"], {**F.SPLIT_SPEC, "overlap_lambda": None, "supported": "dr", "eps_lc": [2 * rho], "eta_value": rho, **({"recentre": "cluster_ips"} if rec_on else {})})["results"][0]["unsupported_interval"]
            lb0, ub0 = np.full(P["nV"], LO), np.full(P["nV"], HI)
            lb3, ub3, _ = B.box(P, args.L, P["ref_log"])
            lb3z, ub3z, _ = B.box(P, args.L, P["ref_log_z"])      # same boxes with the z-margin (needed for feasibility together with B7)
            items = [("baseline", lb0, ub0, False, None)]
            items.append(("B3", lb3, ub3, True, None)); items.append((f"B3_z{args.b7_z:g}", lb3z, ub3z, True, None))
            for k, (lb5, ub5, _) in configs.items():
                items.append((k, lb5, ub5, False, None))
            for k7, (dom, lb7, ub7, _) in b7.items():
                items.append((k7, lb7, ub7, False, dom))
            for k7, (dom, lb7, ub7, _) in b7.items():
                for k, (lb5, ub5, _) in configs.items():
                    items.append((f"B3+{k}+{k7}", np.maximum(np.maximum(lb3, lb5), lb7), np.minimum(np.minimum(ub3, ub5), ub7), True, dom))
                    items.append((f"B3z+{k}+{k7}", np.maximum(np.maximum(lb3z, lb5), lb7), np.minimum(np.minimum(ub3z, ub5), ub7), True, dom))
                    items.append((f"{k}+{k7}", np.maximum(lb5, lb7), np.minimum(ub5, ub7), False, dom))
            for name, lb, ub, coup, dm in items:
                key = f"{tag}|{rho}|{name}"
                if key in d["runs"] and not args.redo:
                    continue
                lo0, up0, info, secs = solve_both(P, rho, lb, ub, L=args.L, coupling=coup, dom=dm, tl=args.time_limit, max_it=args.max_it)
                r = B.rec(P, rho, args.L if coup else None, name, lo0, up0, info, secs, closed_form=cf if name == "baseline" else None)
                r["recentre"] = rec_on
                if lo0 is not None:
                    r["box_tightening"] = dict(share_lb=float((lb > LO + 1e-9).mean()), share_ub=float((ub < HI - 1e-9).mean()))
                d["runs"][key] = r
                json.dump(d, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                print(f"{tag:10s} rho {rho} {name:16s} {secs:6.1f}s " + (f"status {info['status']} bad {info.get('n_bad')}" if lo0 is None else
                      f"LP [{lo0:.2f}, {up0:.2f}] w {up0 - lo0:.2f}  V0 {'in' if r['lp_contains_V0'] else 'OUT'}  DR [{r['dr']['lo']:.2f}, {r['dr']['hi']:.2f}] {'V in' if r['dr']['contains_V'] else 'V OUT'}  cuts {info['cuts']}"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.85, 7.47])
    ap.add_argument("--L", type=float, default=3.0)
    ap.add_argument("--b5-modes", nargs="+", default=["range", "q5"], choices=("range", "q5"))
    ap.add_argument("--b7-z", type=float, default=3.5, help="margin multiplier for logged pairs in B7 (ordering-only B7 is always run too)")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--max-it", type=int, default=8)
    ap.add_argument("--redo", action="store_true")
    run(ap.parse_args())
