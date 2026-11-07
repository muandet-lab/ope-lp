#!/usr/bin/env python3
"""Single-seed check of two settings of the unsupported-pairs LP (original exclusions, seed 7000, lambda = inf and 10, DGP clusters vs the existing K-means partition).

    cd "Test notebooks/scripts/offcem_synthetic"
    conda run -n offcem python eta_separation_experiment.py --seed 7000          # ~20 s; writes ../../results_offcem_global/eta_sep_seed7000.{json,md}

(1) USER-LEVEL CROSS-FITTING.  fit_cluster_level_units(user_folds=True): under round_random a unit is a round, so by-unit folds put the rounds of one user into different
    folds and gave them different m_hat. Here the fold is the ACTUAL USER (W['user_of_round']); the round-specific exclusion masks, LP indexing and propensities are untouched.
    m_hat depends on x and the cluster only, so all rounds of a user get the same prediction for a cluster (verified below). Residual fits (the level ridge b_hat inside the LP),
    the supported estimators and the calibration are all recomputed from the new m_hat.

(2) SEPARATE CORRECTION UNCERTAINTY FROM LOCAL CORRECTNESS.   LC:  |mu_ua - m_hat_ua - b_uc| <= rho  (swept).   Correction band:  |b_uc - b_hat_uc| <= eta.
    eta variants: 'eta=rho' (the earlier default, eta moves with rho) and 'eta=eta_hat' (FIXED over the whole rho sweep, identical for lambda = inf and 10 because it uses logged data only).
    How eta_hat is obtained (logged data only, per partition):
      e_i = y_i - m_hat(round i, a_i);   b_hat = per-cluster ridge of e on x (the LP's own level fit, in sample);   s2 = SSR / dof (dof-corrected, pooled over clusters);
      E[s2] = Var(b - b_hat) + tau^2 + sigma^2   [e_i = b + o + noise_i];   Var_b = max(0, s2 - tau_hat^2 - sigma_hat^2),   eta_hat = Z * sqrt(Var_b),  Z = 2.
      (A user-level OUT-OF-FOLD level fit was tried first and is REJECTED: nested cross-fitting makes it anti-correlated with the residuals, corr ~ -0.7, which inflates eta_hat to ~12.)
    Assumptions this needs: (a) b, offsets o and noise are independent, so the variances add; (b) sigma^2 (repeat-round variance) and tau^2 (within-cell contrasts) are well estimated,
    since they are subtracted; (c) the deviation of the cell level from the smooth ridge is roughly Gaussian and zero-mean-plus-bias, Z = 2 ~ 95%; (d) the unsupported cells, which are never logged,
    deviate like the logged cells (not testable); (e) the dof correction makes the in-sample variance approximately unbiased for the out-of-sample one; the LP's own b_hat sampling error also enters separately through the 1.96 * aggregate SE. eta is NOT a confidence band for b_hat; it is a belief about how far the true level sits from the smooth fit.
rho_max (also Z * tau_hat) is recalibrated per partition after the fix. All variants run on one common rho grid = the previous sweep values + the new endpoints.
True rewards are used ONLY for the oracle checks and for 'contains' flags. 'Contains' (empirical value containment) is not the same thing as 'assumptions hold' (the oracle checks).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import make_slide_figures as F
import partial_ope_global as G
import eps_sweep_dgp as S

warnings.filterwarnings("ignore")
Z = S.Z
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results_offcem_global")
SUPS = ("dr", "ips", "offcem")
LAMS = (None, 10.0)


def eta_hat(W, M, cal, folds=5):
    """eta_hat = Z * sqrt(max(0, s2 - tau^2 - sigma^2)), s2 = dof-corrected residual variance of the per-cluster ridge level fitted to e = y - m_hat on x (the same ridge as the LP's b_hat:
    standardised linear features, alpha = 1, intercept unpenalised), pooled over clusters as sum(SSR_c) / sum(dof_c), dof_c = n_c - trace(hat_c).  Var(e - b_hat) = Var(b - b_hat) + tau^2 + sigma^2.
    The USER-LEVEL OUT-OF-FOLD version (b_oof, folds by user) is also computed and reported as a REJECTED diagnostic: with m_hat cross-fitted on the same user folds, the nested out-of-fold residuals are
    anti-correlated with their own level fit (corr(e, b_oof) ~ -0.7), which inflates E[d^2] (about 50 vs e^2 about 25) and would put eta_hat near 12; that is a cross-fitting artifact, not level error."""
    user, act, y, C = np.asarray(W["user_of_round"]), W["act"], W["y"], W["C"]
    clr = W["cl"][act]
    e = y - M[W["uid"], act]
    X = W["X_rows"]
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Pall = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    pen = np.eye(Pall.shape[1]); pen[0, 0] = 0.0
    ssr = dof = 0.0
    for c in range(C):
        rows = np.flatnonzero(clr == c)
        if len(rows) < 2:
            continue
        P = Pall[rows]
        A = np.linalg.inv(P.T @ P + pen)
        r = e[rows] - P @ (A @ P.T @ e[rows])
        ssr += float((r ** 2).sum()); dof += max(len(rows) - float(np.trace(P @ A @ P.T)), 1.0)
    s2 = ssr / dof
    var_b = max(0.0, s2 - max(cal["tau2_raw"], 0.0) - cal["sigma_noise"] ** 2)
    # rejected diagnostic: user-level out-of-fold level
    ids, inv = np.unique(user, return_inverse=True)
    fold = (np.random.RandomState(W["cfg"].seed).permutation(len(ids)) % folds)[inv]
    bo = np.full(len(y), np.nan)
    for f in range(folds):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Gp, _ = G.level_ridge(X[tr], clr[tr], e[tr], X[te], C, 1.0, 1)
        bo[te] = Gp[np.arange(len(te)), clr[te]]
    ok = np.isfinite(bo)
    d2o = float(np.mean((e[ok] - bo[ok]) ** 2))
    return dict(s2_dof_corrected=s2, var_b=var_b, eta_hat=Z * float(np.sqrt(var_b)), mean_e2=float(np.mean(e ** 2)),
                rejected_oof=dict(mean_d2=d2o, corr_e_boof=float(np.corrcoef(e[ok], bo[ok])[0, 1]),
                                  eta_if_used=Z * float(np.sqrt(max(0.0, d2o - max(cal["tau2_raw"], 0.0) - cal["sigma_noise"] ** 2)))))


def user_consistency(W, M):
    """Max over (user, cluster) of the spread of the model's prediction across that user's rounds (cluster-level model => must be 0)."""
    user, C, cl = np.asarray(W["user_of_round"]), W["C"], W["cl"]
    rep = [int(np.flatnonzero(cl == c)[0]) for c in range(C)]
    worst, same_x = 0.0, 0.0
    for u in np.unique(user):
        idx = np.flatnonzero(user == u)
        if len(idx) < 2:
            continue
        P = M[np.ix_(idx, rep)]
        worst = max(worst, float((P.max(0) - P.min(0)).max()))
        same_x = max(same_x, float(np.abs(W["X_unit"][idx] - W["X_unit"][idx[0]]).max()))
    return dict(max_prediction_spread_within_user_cluster=worst, max_context_spread_within_user=same_x)


def build_grid(extra):
    old = []
    for f in ("eps_sweep_dgp_round_random_seed7000.json", "eps_sweep_dgp_round_random_seed7000_learned.json"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            old += json.load(open(p))["grid"]
    g = []
    for v in sorted(old + list(extra)):
        if not g or v - g[-1] > 0.015:
            g.append(float(v))
    return g, sorted(old)


def run_variant(Wp, M, lam, sup, grid, eta_fixed):
    spec = {**F.SPLIT_SPEC, "overlap_lambda": lam, "supported": sup, "eps_lc": [2 * g for g in grid]}
    if eta_fixed is not None:
        spec["eta_value"] = float(eta_fixed)
    r = G.evaluate(Wp, M, spec)
    d = r["data"]
    rows = []
    for g, x in zip(grid, r["results"]):
        ok = x["status"] == "optimal"
        rows.append(dict(rho=g, ok=ok, infeasible_cells=x["infeasible_cells"],
                         lp_lo=x["unsupported_interval"][0], lp_hi=x["unsupported_interval"][1],
                         lp_ci_lo=x["unsupported_interval_ci"][0] if ok else None, lp_ci_hi=x["unsupported_interval_ci"][1] if ok else None,
                         lp_contains_V0=bool(x["contains_V0"]) if ok else None, lp_ci_contains_V0=bool(x["contains_V0_ci"]) if ok else None,
                         full_lo=x["lower"] if ok else None, full_hi=x["upper"] if ok else None,
                         full_se_lo=x["with_se"][0] if ok else None, full_se_hi=x["with_se"][1] if ok else None,
                         contains_V=bool(x["contains_V"]) if ok else None, contains_V_with_se=bool(x["contains_V_with_se"]) if ok else None,
                         lc_violation=x["truth_spread_violation_share"], band_violation=x["truth_band_violation_share"],
                         joint_violation=x["truth_joint_violation_share"], range_violation=x["truth_range_violation_share"],
                         all_three_violation=x["truth_violation_share"]))
    return dict(rows=rows, supported_estimate=d["supported_estimate"], supported_se=d["supported_se"], ghat_agg_se=d["ghat_agg_se"], total_se=d["total_se"],
                supported_error=d["supported_error"], V=d["oracle_V"], V_plus_true=d["oracle_supported_part"], V0_true=d["oracle_unsupported_part"],
                lp_mass=d["unsupported_policy_mass"])


def one_seed(seed):
    world = {**F.SPLIT_WORLD, "seed": seed, "exclusion": "round_random"}
    W = G.make_world(G.WorldCfg(**world))
    km = KMeans(n_clusters=W["C"], n_init=10, random_state=0).fit_predict(np.asarray(W["action_x"], float)).astype(int)
    parts = {"dgp": W["cl"].copy(), "learned": km}
    out = dict(seed=seed, world=world, partitions={}, ari_learned_vs_dgp=float(adjusted_rand_score(parts["dgp"], km)))
    prep = {}
    for name, lab in parts.items():
        Wp = {**W, "cl": lab, "C": int(lab.max()) + 1}
        M_old = G.get_model(Wp, "cluster_level", "ridge", 5, 1.0, 1, user_folds=False)
        M_new = G.get_model(Wp, "cluster_level", "ridge", 5, 1.0, 1, user_folds=True)
        info = {}
        for tag, M in (("before", M_old), ("after", M_new)):
            cal = S.calibrate_user_cells(Wp, M)
            info[tag] = dict(calibration=cal, eta=eta_hat(Wp, M, cal), consistency=user_consistency(Wp, M))
        prep[name] = (Wp, M_new, info)
        out["partitions"][name] = dict(info=info)
    grid, old = build_grid([prep[n][2]["after"]["calibration"]["eps_max"] for n in prep])
    out["grid"], out["previous_grid_values"] = grid, old
    for name, (Wp, M, info) in prep.items():
        eh = info["after"]["eta"]["eta_hat"]
        out["partitions"][name]["eta_hat"] = eh
        out["partitions"][name]["rho_max"] = info["after"]["calibration"]["eps_max"]
        runs = {}
        for lam in LAMS:
            key = "inf" if lam is None else str(int(lam))
            runs[key] = {}
            for variant, ef in (("eta=rho", None), ("eta=eta_hat", eh)):
                runs[key][variant] = {sp: run_variant(Wp, M, lam, sp, grid, ef) for sp in SUPS}
        out["partitions"][name]["runs"] = runs
    return out


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def interval(lo, hi, hit):
    return "n/a" if lo is None else f"[{lo:.2f}, {hi:.2f}] {'✓' if hit else '✗'}"


def report(res):
    L = []
    L.append(f"# eta separation, user-level cross-fitting — seed {res['seed']}, original exclusions (round_random)\n")
    L.append("Displayed rho = per-action LC radius (stored eps_LC = 2 rho). ✓ / ✗ = the interval contains / misses its true value (EMPIRICAL containment, not an assumption check).\n")
    L.append(f"K-means vs DGP partition: ARI = {res['ari_learned_vs_dgp']:.3f}.  Common rho grid: " + ", ".join(f"{g:.2f}" for g in res["grid"]) + "\n")
    L.append("## 1. Cross-fitting fix: noise, contrast statistic, rho_max, eta_hat (before = by-round folds, after = by-user folds)\n")
    L.append("| partition | | sigma_hat | repeat dof | pairs | mean contrast E[(e_i-e_j)^2/2] | tau_hat^2 | rho_max = 2 tau_hat | s2 (dof-corr.) | eta_hat | rejected OOF eta | max pred. spread within (user, cluster) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, P in res["partitions"].items():
        for tag in ("before", "after"):
            c, e, k = P["info"][tag]["calibration"], P["info"][tag]["eta"], P["info"][tag]["consistency"]
            L.append(f"| {name} | {tag} | {c['sigma_noise']:.2f} | {c['n_repeat_dof']} | {c['n_pairs']} | {c['mean_half_sq_contrast']:.2f} | {c['tau2_raw']:.2f} | {c['eps_max']:.2f} | {e['s2_dof_corrected']:.2f} | {e['eta_hat']:.2f} | {e['rejected_oof']['eta_if_used']:.2f} (corr {e['rejected_oof']['corr_e_boof']:+.2f}) | {k['max_prediction_spread_within_user_cluster']:.3g} |")
    L.append("")
    for name, P in res["partitions"].items():
        for lam in ("inf", "10"):
            first = P["runs"][lam]["eta=rho"]["dr"]
            L.append(f"## {name} clusters, lambda = {'∞' if lam == 'inf' else lam}  (true V = {first['V']:.3f}, true V⁺ = {first['V_plus_true']:.3f}, true V⁰ = {first['V0_true']:.3f}, LP mass {first['lp_mass']:.2f}; rho_max = {P['rho_max']:.2f}, eta_hat = {P['eta_hat']:.2f})\n")
            L.append("Supported estimates (V⁺): " + "; ".join(
                f"{sp.upper()} {P['runs'][lam]['eta=rho'][sp]['supported_estimate']:.3f} (se {P['runs'][lam]['eta=rho'][sp]['supported_se']:.2f}, error {P['runs'][lam]['eta=rho'][sp]['supported_error']:+.3f})" for sp in SUPS)
                     + f"; aggregate level SE {first['ghat_agg_se']:.2f}; total SE (DR/IPS/OffCEM) " + "/".join(f"{P['runs'][lam]['eta=rho'][sp]['total_se']:.2f}" for sp in SUPS) + "\n")
            for variant in ("eta=rho", "eta=eta_hat"):
                eta_txt = "eta = rho (moves with the sweep)" if variant == "eta=rho" else f"eta fixed at {P['eta_hat']:.2f}"
                L.append(f"### {eta_txt}\n")
                L.append("| rho | LP-only V⁰ [lo, hi] | LP-only +1.96 agg SE | full DR | full IPS | full OffCEM | width (DR) | oracle: LC viol. | band viol. | joint viol. |")
                L.append("|---|---|---|---|---|---|---|---|---|---|")
                R = P["runs"][lam][variant]
                for k, g in enumerate(res["grid"]):
                    a = R["dr"]["rows"][k]
                    mark = " ←ρ_max" if abs(g - P["rho_max"]) < 0.015 else ""
                    L.append(f"| {g:.2f}{mark} | {interval(a['lp_lo'], a['lp_hi'], a['lp_contains_V0'])} | {interval(a['lp_ci_lo'], a['lp_ci_hi'], a['lp_ci_contains_V0'])} | "
                             + " | ".join(interval(R[sp]['rows'][k]['full_se_lo'], R[sp]['rows'][k]['full_se_hi'], R[sp]['rows'][k]['contains_V_with_se']) for sp in SUPS)
                             + f" | {fmt(None if a['full_se_lo'] is None else a['full_se_hi'] - a['full_se_lo'])} | {a['lc_violation']:.2f} | {a['band_violation']:.2f} | {a['joint_violation']:.2f} |")
                L.append("")
            L.append("Width added by eta = rho vs fixed eta_hat (DR full-value width, eta=rho minus fixed): " + ", ".join(
                f"{g:.2f}: {((P['runs'][lam]['eta=rho']['dr']['rows'][k]['full_se_hi'] - P['runs'][lam]['eta=rho']['dr']['rows'][k]['full_se_lo']) - (P['runs'][lam]['eta=eta_hat']['dr']['rows'][k]['full_se_hi'] - P['runs'][lam]['eta=eta_hat']['dr']['rows'][k]['full_se_lo'])):+.2f}"
                for k, g in enumerate(res["grid"])) + "\n")
    L.append("Oracle columns are mass-weighted shares of unsupported-pair cells whose TRUE residuals violate: LC (range of q - m_hat > 2 rho), the correction band (midrange of q - m_hat further than eta from b_hat), or the joint feasibility of both (no level b exists that satisfies both).\n")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    res = one_seed(a.seed)
    json.dump(res, open(os.path.join(RES, f"eta_sep_seed{a.seed}.json"), "w"), indent=1)
    md = report(res)
    open(os.path.join(RES, f"eta_sep_seed{a.seed}.md"), "w").write(md)
    print(md[:6000])
