#!/usr/bin/env python3
"""Extra belief statements (linear constraints) on top of the unsupported-pairs LP: K-means clusters, original exclusions, lambda = inf, seed 7000, rho (displayed eps) in {5.85, 7.47}.

    cd "Test notebooks/scripts/offcem_synthetic"
    conda run -n offcem python extra_beliefs_experiment.py --seed 7000 --rhos 5.85 7.47          # one world; writes ../../results_offcem_global/extra_beliefs_seed7000.json

Base LP (unchanged): variables mu_ua (unsupported pairs) and a cell level b_uc;   LC |mu - m_hat - b| <= rho,   correction band |b - b_hat| <= eta,   range l <= mu <= u.
m_hat is cross-fitted by USER; eta is held fixed at eta_hat (eta_separation_experiment.eta_hat, from logged data) and the eta = rho variant is kept as a comparison.

Added beliefs, all in the finite-LP form of Section 3.2 (constraint (3): sum_s theta_s * E[Y(t)|x] <= c_s over a finite set R3_s; L1 beliefs use auxiliary variables):
 B1  AVERAGE LC (average-error budget):   sum_p w_p |mu_p - m_p - b_p| <= abar * sum_p w_p,   w_p = unit weight * target prob.  abar = 1.25 * sqrt(2/pi) * tau_hat
     (E|offset| of a Gaussian offset with sd tau_hat, from logged within-cell contrasts, +25% margin). Assumes offsets are roughly Gaussian and that unsupported actions deviate like logged ones.
 B2  AGGREGATE MEAN (per cluster c, a constraint of the form (3) with theta = w, R3 = unsupported pairs of the cluster):
        | sum_{p in c} w_p (mu_p - m_p - bhat_p) | <= delta_c * sum_{p in c} w_p
     "the model error of the unsupported pairs, relative to the fitted level, averages out".  Two strengths:  CALIBRATED: delta_c = Z * sd_dev / sqrt(n_eff,c) with
     sd_dev^2 = s2 - sigma^2 (logged-data variance of offset + level deviation), n_eff,c = (sum_u W_uc)^2 / sum_u W_uc^2 over REAL users (independence across users, zero mean: assumptions);
     the target mass sits on few users, so n_eff is only about 14 and delta_c is 2.2 to 3.7.   STRICT: delta_c = 1.0 (a hand-set, tighter belief).
 B1 + B2 (calibrated) combined.
Both B1 and B2 narrow the bound only because they are BELIEFS about the unsupported region; they cannot be checked from the logged data. Bounds = V+ estimate (IPS / DR / OffCEM) + LP interval,
widened by 1.96 SE. True rewards are used only for 'contains' flags and the oracle checks of each belief.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import math
import time
import warnings
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans

import make_slide_figures as F
import partial_ope_global as G
import eps_sweep_dgp as S
import eta_separation_experiment as E

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results_offcem_global")
SUPS = ("dr", "ips", "offcem")
Z = 2.0


def structures(W, M):
    """Pieces evaluate() builds internally (lambda = inf, ridge centre, aggregate noise), reproduced so the extra LP sees the same b_hat / identified cells."""
    cl, C, n = W["cl"], W["C"], W["n"]
    pi, pi0, wu = W["pi_unit"], W["pi0_unit"], W["counts"] / W["n"]
    uid, act, y = W["uid"], W["act"], W["y"]
    U = pi0 == 0
    nU = pi.shape[0]
    resid, clr = y - M[uid, act], cl[act]
    pi0_c = np.stack([pi0[:, cl == c].sum(1) for c in range(C)], 1)
    pi_c = np.stack([pi[:, cl == c].sum(1) for c in range(C)], 1)
    pm_cell = np.stack([(pi[:, cl == c] * U[:, cl == c]).sum(1) for c in range(C)], 1)
    support_viol = (pi_c > 0) & (pi0_c <= 0)
    Gh, _, agg_se = G.level_ridge(W["X_rows"], clr, resid, W["X_unit"], C, 1.0, 1, agg_w=wu[:, None] * pm_cell * ~support_viol)
    ident = ~support_viol & ~np.isnan(Gh)
    return dict(U=U, G=Gh, ident=ident, wu=wu, pi=pi, agg_se=agg_se, nU=nU)


def solve(st, M, W, rho, eta_fix, beliefs, l=-30.0, u=30.0):
    """Min / max of V0 = sum_u wu sum_a pi mu over the base LP + the requested beliefs. Returns (lo0, up0, seconds)."""
    t0 = time.perf_counter()
    cl, C = W["cl"], W["C"]
    U, Gh, ident, wu, pi = st["U"], st["G"], st["ident"], st["wu"], st["pi"]
    pu, pa = np.nonzero(U)
    pc = cl[pa]
    coef_all = wu[pu] * pi[pu, pa]
    keep = ident[pu, pc]
    # pairs in non-identified cells keep only [l, u] (as the closed form does)
    const_lo = float((coef_all[~keep] * l).sum()); const_up = float((coef_all[~keep] * u).sum())
    iu, ia, ic, coef = pu[keep], pa[keep], pc[keep], coef_all[keep]
    P = len(iu)
    ckeys, cinv = np.unique(iu * C + ic, return_inverse=True)
    K = len(ckeys)
    cu, cc = ckeys // C, ckeys % C
    eta = rho if eta_fix is None else eta_fix
    use_t = "B1" in beliefs
    nvar = P + K + (P if use_t else 0)
    lb, ub = np.zeros(nvar), np.zeros(nvar)
    lb[:P], ub[:P] = l, u
    lb[P:P + K], ub[P:P + K] = Gh[cu, cc] - eta, Gh[cu, cc] + eta
    mp = M[iu, ia]
    rr = np.arange(P)
    rows, cols, vals, rhs = [], [], [], []
    if use_t:
        lb[P + K:], ub[P + K:] = 0.0, rho
        t = P + K + rr
        rows += [rr, rr, rr, rr + P, rr + P, rr + P]; cols += [rr, P + cinv, t, rr, P + cinv, t]
        vals += [np.ones(P), -np.ones(P), -np.ones(P), -np.ones(P), np.ones(P), -np.ones(P)]
        rhs += [mp, -mp]
    else:
        rows += [rr, rr, rr + P, rr + P]; cols += [rr, P + cinv, rr, P + cinv]
        vals += [np.ones(P), -np.ones(P), -np.ones(P), np.ones(P)]
        rhs += [mp + rho, -mp + rho]
    nrows = 2 * P
    extra_rows = []
    if use_t:
        extra_rows.append((np.full(P, nrows), P + K + rr, coef, beliefs["B1"] * coef.sum()))
        nrows += 1
    if "B2" in beliefs:
        dlt = beliefs["B2"]                       # vector over clusters
        base = mp + Gh[iu, ic]                    # m_hat + b_hat per pair
        for c in range(C):
            sel = np.flatnonzero(ic == c)
            if not len(sel):
                continue
            wc = coef[sel]
            rhs_c = float((wc * base[sel]).sum())
            tot = float(wc.sum()) * dlt[c]
            extra_rows.append((np.full(len(sel), nrows), sel, wc, rhs_c + tot)); nrows += 1
            extra_rows.append((np.full(len(sel), nrows), sel, -wc, -rhs_c + tot)); nrows += 1
    b_extra = []
    for r_, c_, v_, bb in extra_rows:
        rows.append(r_); cols.append(c_); vals.append(v_); b_extra.append(bb)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrows, nvar)).tocsr()
    b = np.concatenate(rhs + [np.asarray(b_extra)])
    scale = coef.max()
    c_obj = np.zeros(nvar); c_obj[:P] = coef / scale
    out = []
    for sign in (1.0, -1.0):
        r = linprog(sign * c_obj, A_ub=A, b_ub=b, bounds=np.column_stack([lb, ub]), method="highs", options=dict(presolve=True))
        if not r.success:
            return None, None, time.perf_counter() - t0, r.status
        out.append(r.fun * scale)
    return out[0] + const_lo, -out[1] + const_up, time.perf_counter() - t0, 0


def belief_calibration(W, M, cal, eta_info, st):
    tau = cal["tau"]
    abar = 1.25 * math.sqrt(2 / math.pi) * tau
    sd_dev = math.sqrt(max(0.0, eta_info["s2_dof_corrected"] - cal["sigma_noise"] ** 2))
    C, cl = W["C"], W["cl"]
    U, pi, wu, uid = st["U"], st["pi"], st["wu"], W["uid"]
    user = np.asarray(W["user_of_round"])               # real user of each unit (round)
    ids, inv = np.unique(user, return_inverse=True)
    delta_tight = []
    for c in range(C):
        Om = wu * (pi[:, cl == c] * U[:, cl == c]).sum(1)         # per unit
        Ou = np.bincount(inv, weights=Om, minlength=len(ids))     # per real user
        neff = (Ou.sum() ** 2) / max((Ou ** 2).sum(), 1e-300)
        delta_tight.append(Z * sd_dev / math.sqrt(max(neff, 1.0)))
    return dict(abar=abar, sd_dev=sd_dev, delta_tight=delta_tight, delta_loose=1.0)


def oracle_checks(W, M, st, rho, eta, beliefs):
    """True-mean checks of each belief on the unsupported pairs (evaluation only): B1 average deviation from the best level, B2 per-cluster aggregate deviation from m_hat + b_hat."""
    cl, C = W["cl"], W["C"]
    U, Gh, ident, wu, pi, q = st["U"], st["G"], st["ident"], st["wu"], st["pi"], W["q_unit"]
    out = {}
    r = q - M
    num = den = 0.0
    agg = {}
    for c in range(C):
        cols = np.flatnonzero(cl == c)
        Uc, rc, pc = U[:, cols], r[:, cols], pi[:, cols] * U[:, cols] * wu[:, None]
        sel = ident[:, c] & Uc.any(1)
        gh = np.where(ident[:, c], Gh[:, c], 0.0)
        mid = (np.where(Uc, rc, -np.inf).max(1) + np.where(Uc, rc, np.inf).min(1)) / 2
        mid = np.where(sel, mid, 0.0)
        dev = np.abs(rc - mid[:, None]) * Uc
        num += float((pc * dev)[sel].sum()); den += float(pc[sel].sum())
        sw = pc[sel].sum()
        agg[c] = float((pc[sel] * (rc[sel] - gh[sel][:, None])).sum() / max(sw, 1e-300))
    out["true_avg_abs_dev_from_midrange"] = num / max(den, 1e-300)
    out["true_cluster_agg_dev"] = agg
    return out


def one(seed, rhos):
    world = {**F.SPLIT_WORLD, "seed": seed, "exclusion": "round_random"}
    W = G.make_world(G.WorldCfg(**world))
    km = KMeans(n_clusters=W["C"], n_init=10, random_state=0).fit_predict(np.asarray(W["action_x"], float)).astype(int)
    Wp = {**W, "cl": km, "C": int(km.max()) + 1}
    M = G.get_model(Wp, "cluster_level", "ridge", 5, 1.0, 1, user_folds=True)
    cal = S.calibrate_user_cells(Wp, M)
    et = E.eta_hat(Wp, M, cal)
    st = structures(Wp, M)
    bc = belief_calibration(Wp, M, cal, et, st)
    out = dict(seed=seed, world=world, clusters="kmeans K=10", rho_max=cal["eps_max"], tau_hat=cal["tau"], sigma_hat=cal["sigma_noise"], eta_hat=et["eta_hat"], calibration=bc, rhos=rhos, runs={})
    sup = {}
    for sp in SUPS:        # supported estimate and SE from the standard evaluate (lambda = inf)
        r = G.evaluate(Wp, M, {**F.SPLIT_SPEC, "overlap_lambda": None, "supported": sp, "eps_lc": [2 * rhos[0]], "eta_value": et["eta_hat"]})
        d = r["data"]
        sup[sp] = dict(estimate=d["supported_estimate"], supported_se=d["supported_se"], total_se=d["total_se"], error=d["supported_error"])
        out.update(V=d["oracle_V"], V_plus_true=d["oracle_supported_part"], V0_true=d["oracle_unsupported_part"], lp_mass=d["unsupported_policy_mass"], agg_se=d["ghat_agg_se"])
    out["supported"] = sup
    variants = {"baseline": {}, "+B1 average LC": {"B1": bc["abar"]}, "+B2 calibrated": {"B2": bc["delta_tight"]},
                "+B2 strict (delta=1)": {"B2": [bc["delta_loose"]] * Wp["C"]}, "+B1 +B2 calibrated": {"B1": bc["abar"], "B2": bc["delta_tight"]}}
    path = os.path.join(RES, f"extra_beliefs_seed{seed}.json")
    out["oracle_beliefs"] = oracle_checks(Wp, M, st, rhos[0], et["eta_hat"], None)
    for base, ef in (("eta fixed", et["eta_hat"]), ("eta=rho", None)):       # main (eta fixed) first; the eta=rho comparison afterwards
        for rho in rhos:
            for vn, bel in variants.items():
                lo0, up0, secs, status = solve(st, M, Wp, rho, ef, bel)
                rec = dict(rho=rho, base=base, variant=vn, status=status, seconds=secs)
                if lo0 is not None:
                    rec.update(lp_lo=lo0, lp_hi=up0, lp_contains_V0=bool(lo0 <= out["V0_true"] <= up0))
                    for sp in SUPS:
                        se = sup[sp]["total_se"]; est = sup[sp]["estimate"]
                        rec[sp] = dict(lo=est + lo0 - 1.96 * se, hi=est + up0 + 1.96 * se, contains_V=bool(est + lo0 - 1.96 * se <= out["V"] <= est + up0 + 1.96 * se))
                out["runs"][f"{rho}|{base}|{vn}"] = rec
                json.dump(out, open(path, "w"), indent=1)          # incremental save: a partial run is still usable
                print(f"rho {rho} {base:9s} {vn:20s} {secs:6.1f}s  " + ("infeasible" if lo0 is None else f"LP-only [{lo0:.2f}, {up0:.2f}] width {up0-lo0:.2f}  V0 {'in' if rec['lp_contains_V0'] else 'OUT'}"), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--rhos", type=float, nargs="+", default=[5.85, 7.47])
    a = ap.parse_args()
    res = one(a.seed, a.rhos)
    json.dump(res, open(os.path.join(RES, f"extra_beliefs_seed{a.seed}.json"), "w"), indent=1)
