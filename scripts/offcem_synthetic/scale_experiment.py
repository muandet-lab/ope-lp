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
import real_world as RW

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results_offcem_global")


def labels(W, method, seed):
    if method in ("dgp", "random"):          # dgp = the synthetic generator's clusters; random = the paper's uniform random clusters stored in the real world
        return np.asarray(W["cl"]).astype(int)
    f = {"kmeans": CL.cluster_kmeans, "agglomerative": CL.cluster_agglomerative, "hierarchical": CL.cluster_agglomerative, "spectral": CL.cluster_spectral}[method]
    lab = np.asarray(f(np.asarray(W["action_x"], float), W["C"], random_state=seed)).astype(int)
    _, lab = np.unique(lab, return_inverse=True)            # ids 0..C'-1 (spectral may leave a cluster empty)
    return lab


def estimate_L(P, q=0.95, min_pairs=50):
    """Logged-data Lipschitz constant: over all pairs of users (u, v) that both logged the same action a, the noise-corrected slope
    s_uv,a^2 = max(0, (ybar_ua - ybar_va)^2 - sigma_hat^2 (1/n_ua + 1/n_va)) / d_uv^2 ; L_hat = q-quantile of s (default 95th). Uses only logged means, counts, sigma_hat and
    standardised-context distances. No truth. Reports the pair count; fewer than min_pairs -> status 'insufficient' (L_hat still returned for transparency)."""
    Rlo, Rhi = P["ref_log"]; mean = P["ref_mean_raw"] if "ref_mean_raw" in P else (Rlo + Rhi) / 2; n = P["ref_n"]; D = P["D"]; s2 = P["sigma"] ** 2
    slopes = []
    for a in range(P["K"]):
        us = np.flatnonzero(n[:, a] > 0)
        if len(us) < 2:
            continue
        i, j = np.triu_indices(len(us), 1); ui, uj = us[i], us[j]
        num = np.maximum((mean[ui, a] - mean[uj, a]) ** 2 - s2 * (1.0 / n[ui, a] + 1.0 / n[uj, a]), 0.0)
        slopes.append(np.sqrt(num) / D[ui, uj])
    slopes = np.concatenate(slopes) if slopes else np.array([])
    out = dict(n_pairs=int(len(slopes)), quantile=q, status="ok" if len(slopes) >= min_pairs else "insufficient")
    if len(slopes):
        out.update(L_hat=float(np.quantile(slopes, q)), median=float(np.median(slopes)), p75=float(np.quantile(slopes, 0.75)), max=float(slopes.max()), share_zero_after_noise_correction=float((slopes == 0).mean()))
    return out


def truth_checks(P, rho, eta):
    """Truth-only diagnostics (after the fact): share of LP mass whose true mean violates LC |q - m_hat - b_hat| <= rho (b at its estimate), and whether the TRUE mean vector is feasible
    for some admissible b per (round, cluster) cell: need max(q - m_hat) - min(q - m_hat) <= 2 rho within the cell and [max - rho, min + rho] to intersect [g_hat - eta, g_hat + eta]."""
    A = P["A_base"].tocoo(); nV, npairs = P["nV"], P["n_pairs"]
    m = (A.row < npairs) & (A.col >= nV); cid = np.zeros(npairs, int); cid[A.row[m]] = A.col[m] - nV
    qp = P["qu"][P["v_user"], P["v_act"]][P["vid"]]; r = qp - P["mp"]                     # per pair: true mean minus model prediction
    w = P["coef_v"][P["vid"]]
    lc_viol = float((w * (np.abs(r - P["gh"][cid]) > rho + 1e-9)).sum() / w.sum())
    hi = np.full(P["Kc"], -np.inf); lo = np.full(P["Kc"], np.inf); np.maximum.at(hi, cid, r); np.minimum.at(lo, cid, r)
    b_lo, b_hi = np.maximum(hi - rho, P["gh_cell"] - eta), np.minimum(lo + rho, P["gh_cell"] + eta)
    bad = b_lo > b_hi + 1e-9
    return dict(lc_violation_share_mass=lc_viol, true_vector_feasible=bool(not bad.any()), cells_infeasible_for_truth=int(bad.sum()), cells=int(P["Kc"]),
                cells_spread_over_2rho=int(((hi - lo) > 2 * rho + 1e-9).sum()))


def dump(d, path):
    json.dump(d, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def run(args):
    out = os.path.join(RES, f"scale_{args.tag}_seed{args.seed}.json")
    d = json.load(open(out)) if os.path.exists(out) else {}
    t0 = time.perf_counter()
    if args.real:
        rc = RW.RealCfg(dataset=args.real, n=args.n, users=args.users, beta=args.beta, eps_policy=args.eps_target, lam=args.lam, C=args.C, seed=args.seed, logger=args.logger)
        W = RW.make_real_world(rc, args.data_dir); cfg = {k: v for k, v in rc.__dict__.items()}
        limits, lam, calibrate, sigma_ref = [0.0, 1.0], args.lam, RW.calibrate_binomial, 0.5      # Bernoulli: range [0,1]; reference intervals use the sd bound 0.5
    else:
        cfg = dict(K=args.K, n=args.n, users=args.users, C=args.C, def_frac=args.def_frac, exclusion="round_random", sigma=args.sigma, beta=-0.1, eps_policy=0.2, clustering="original", seed=args.seed)
        W = G.make_world(G.WorldCfg(**cfg)); limits, lam, calibrate, sigma_ref = [-30.0, 30.0], args.lam, None, None
    d["world"] = dict(cfg, folds=args.folds, model=args.model, limits=limits, lam=lam, eps_min=args.eps_min, build_seconds=time.perf_counter() - t0, n_rounds=int(W["n"]), n_users=int(len(np.unique(W["user_of_round"]))), dgp_clusters=int(W["C"]))
    d.setdefault("clusterings", {})
    print(f"world built in {d['world']['build_seconds']:.1f}s: {W['K']} actions, {args.n} rounds, {d['world']['n_users']} users, {W['C']} DGP clusters", flush=True)
    for method in args.clusterings:
        tc = time.perf_counter()
        r = d["clusterings"].setdefault(method, {})
        r.setdefault("runs", {})
        lab = labels(W, method, args.seed)
        r["ari_vs_dgp"] = float(adjusted_rand_score(W["cl"], lab)); r["n_clusters"] = int(lab.max()) + 1
        if args.alpha == "cv":                           # ridge penalty by out-of-fold squared error of the cross-fitted centre on the logged rounds (logged data only)
            Wl = {**W, "cl": lab, "C": int(lab.max()) + 1}; oof = {}
            for a_ in (0.1, 1.0, 10.0, 100.0, 1000.0):
                Ma = G.get_model(Wl, args.model, "ridge", 5, a_, 1, user_folds=(args.folds == "user"))
                if limits is not None:
                    Ma = np.clip(Ma, limits[0], limits[1])
                oof[a_] = float(((W["y"] - Ma[W["uid"], W["act"]]) ** 2).mean())
            if max(oof.values()) - min(oof.values()) < 1e-12:   # the centre ignores alpha (two_step): alpha then only sets the level ridge; take the strongest penalty rather than the first candidate
                alpha = max(oof); r["ridge_alpha_cv"] = dict(oof_mse=oof, chosen=alpha, degenerate=True, note="centre does not depend on alpha; chosen for the level ridge")
            else:
                alpha = min(oof, key=oof.get); r["ridge_alpha_cv"] = dict(oof_mse=oof, chosen=alpha)
            print(f"[{method}] ridge alpha by out-of-fold MSE: {oof} -> {alpha}", flush=True)
        else:
            alpha = float(args.alpha)
        P = B.prep(args.seed, user_folds=(args.folds == "user"), W=W, lab=lab, model=args.model, lam=lam, limits=limits, calibrate=calibrate, sigma_ref=sigma_ref, alpha=alpha, clip=args.clip_model, shrink=(None if args.h_shrink is None else ("cv" if args.h_shrink == "cv" else float(args.h_shrink))), g_model=args.g_model)
        if P["W"].get("two_step_info"):
            r["two_step_info"] = P["W"]["two_step_info"]; print(f"[{method}] two-step correction weight s: {P['W']['two_step_info']}", flush=True)
        if P.get("clip_info"):
            r["clip_info"] = P["clip_info"]; print(f"[{method}] centre clipped to the cluster's logged reward range: {P['clip_info']}", flush=True)
        Wp, M = P["W"], P["M"]; LO, HI = P["LO"], P["HI"]
        if args.clip_model == "outcome_cell":             # clip the LC centre m_hat + g_hat of each pair into the outcome range by shifting the cell's level estimate (a mean of a bounded outcome lies in the range)
            cell_id = np.repeat(np.arange(P["Kc"]), 0)
            A = P["A_base"].tocsr(); rows = A[:P["n_pairs"]]; cells = np.asarray(rows.indices.reshape(P["n_pairs"], 2)[:, 1]) - P["nV"]
            c = P["mp"] + P["gh"]; shift = np.clip(c, LO, HI) - c
            cell_shift = np.zeros(P["Kc"]); np.add.at(cell_shift, cells, shift); cnt = np.bincount(cells, minlength=P["Kc"]); cell_shift /= np.maximum(cnt, 1)
            P["gh"] = P["gh"] + cell_shift[cells] if len(P["gh"]) == P["n_pairs"] else P["gh"] + cell_shift
            r["clip_info"] = dict(rule="outcome_cell", share_pairs_shifted=float((np.abs(shift) > 1e-12).mean()), mean_abs_shift=float(np.abs(shift).mean()), max_abs_shift=float(np.abs(shift).max()))
            print(f"[{method}] LC centre clipped into the outcome range: {r['clip_info']}", flush=True)
        P["gh_cell"] = P["gh"]                            # P["gh"] is already one level estimate per (round, cluster) cell
        if args.ref_z is not None:                        # one reference rule for every logged interval (B3 references, B5 endpoints); B7 logged bounds use --z
            mean_, n_ = (P["ref_log"][0] + P["ref_log"][1]) / 2, P["ref_n"]
            with np.errstate(invalid="ignore", divide="ignore"):
                hz_ = args.ref_z * P["sigma"] / np.sqrt(n_)
            P["ref_log"] = (np.where(n_ > 0, mean_ - hz_, np.nan), np.where(n_ > 0, mean_ + hz_, np.nan))
        cal = P["cal"]
        if args.noise_sd is not None:                     # repeats too sparse for a data-only sigma: take the noise sd as known (an assumption; recorded in the JSON)
            cal["sigma_noise_estimated"] = cal["sigma_noise"]; cal["sigma_noise"] = float(args.noise_sd); cal["sigma_source"] = "given"
            cal["tau2_raw"] = cal["mean_half_sq_contrast"] - args.noise_sd ** 2; cal["tau"] = float(np.sqrt(max(cal["tau2_raw"], 0.0))); cal["eps_max"] = cal["z"] * cal["tau"]
            P["sigma"] = float(args.noise_sd)
        else:
            cal["sigma_source"] = "repeats"
        emax = cal["eps_max"]
        if np.isfinite(emax) and emax <= args.eps_min:
            print(f"[{method}] eps_max = {emax:.4f} <= eps_min = {args.eps_min} (tau2_raw {cal['tau2_raw']:.4f}: contrasts at or below the noise level). Grid collapses to the single point eps_min.", flush=True)
            r["eps_max_below_eps_min"] = True
        if not np.isfinite(emax):
            raise SystemExit(f"[{method}] eps_max is NaN: the noise sd could not be estimated ({cal['n_repeat_dof']} repeat dof from repeated (user, action) logged pairs, need >= 10). "
                             "Use fewer users or more rounds (the small world had 10 rounds per user), or pass --noise-sd <value> to take the noise sd as known.")
        grid = [float(x) for x in np.linspace(args.eps_min, emax, args.npts)] if emax > args.eps_min else [float(args.eps_min)]
        if args.eps_grid:
            grid = [float(x) for x in args.eps_grid]   # forced grid: ignores eps_max; beliefs run at every grid point
        r.update(calibration=P["cal"], eta_hat=P["eta"], level_se=dict(median=float(np.median(P["gh_se"])), p90=float(np.percentile(P["gh_se"], 90)), max=float(P["gh_se"].max())), rows_per_cluster=dict(min=int(P["st"]["rows_per_cluster"].min()), median=float(np.median(P["st"]["rows_per_cluster"]))), grid=grid, eps_max=emax, V=P["V"], V_plus_true=P["Vp_true"], V0_true=P["V0_true"], lp_mass=P["lp_mass"], supported=P["sup"],
                 n_merged_vars=int(P["nV"]), n_pairs=int(P["n_pairs"]), sigma_hat=P["sigma"], prep_seconds=time.perf_counter() - tc)
        print(f"[{method}] ARI {r['ari_vs_dgp']:.2f}  eps_max {emax:.2f}  grid {np.round(grid, 2).tolist()}  vars {P['nV']}  prep {r['prep_seconds']:.0f}s", flush=True)
        # ---- plot-04 sweep (closed form, per supported estimator)
        if "sweep" not in r:
            ts = time.perf_counter(); sw = {}
            for sp in B.SUPS:
                res = G.evaluate(Wp, M, {**F.SPLIT_SPEC, "overlap_lambda": lam, "limits": limits, "ridge_alpha": alpha, "supported": sp, "eps_lc": [2 * g for g in grid]})
                sw[sp] = [dict(eps=g, ok=x["status"] == "optimal", status=x["status"], lo=x.get("with_se", [None, None])[0], hi=x.get("with_se", [None, None])[1],
                               lp_lo=x.get("unsupported_interval", [None, None])[0], lp_hi=x.get("unsupported_interval", [None, None])[1], covers_V=x.get("contains_V_with_se")) for g, x in zip(grid, res["results"])]
            r["sweep"] = sw; r["sweep_seconds"] = time.perf_counter() - ts; dump(d, out)
            print(f"[{method}] sweep {r['sweep_seconds']:.0f}s  DR at eps_max " + (f"[{sw['dr'][-1]['lo']:.2f}, {sw['dr'][-1]['hi']:.2f}]" if sw["dr"][-1]["ok"] else sw["dr"][-1]["status"]), flush=True)
        # ---- beliefs at eps_max and one level below
        rhos = grid if args.eps_grid else (grid[-2:] if len(grid) > 1 else grid)
        qv = P["qu"][P["v_user"], P["v_act"]]
        mean, n = (P["ref_log"][0] + P["ref_log"][1]) / 2, P["ref_n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            hz = args.z * P["sigma"] / np.sqrt(n)
        P["ref_log_z"] = (np.where(n > 0, mean - hz, np.nan), np.where(n > 0, mean + hz, np.nan))
        P["ref_mean_raw"] = (P["ref_log"][0] + P["ref_log"][1]) / 2        # logged means (before any interval replacement) for the L estimate
        if args.binomial_refs:                            # Bernoulli outcomes: exact Clopper-Pearson intervals at the two-sided level implied by z replace mean +- z sigma / sqrt(n) everywhere a logged reference is used
            from scipy.stats import beta as _beta, norm as _norm
            n_ = P["ref_n"]; mean_ = (P["ref_log"][0] + P["ref_log"][1]) / 2; s_ = np.rint(np.where(n_ > 0, mean_ * n_, 0.0))
            def cp(zz):
                a = 2 * _norm.sf(zz)
                with np.errstate(invalid="ignore"):
                    lo_ = np.where(s_ > 0, _beta.ppf(a / 2, s_, n_ - s_ + 1), 0.0); hi_ = np.where(s_ < n_, _beta.ppf(1 - a / 2, s_ + 1, n_ - s_), 1.0)
                return np.where(n_ > 0, lo_, np.nan), np.where(n_ > 0, hi_, np.nan)
            P["ref_log"] = cp(1.96); P["ref_log_z"] = cp(args.z); P["ref_lbub_z"] = cp(args.z)
            r["binomial_refs"] = dict(level_default=1.96, level_z=args.z, pairs_logged=int((n_ > 0).sum()), median_n=float(np.median(n_[n_ > 0])))
            print(f"[{method}] logged references: Clopper-Pearson (two-sided z = 1.96 default / z = {args.z:g}); {r['binomial_refs']}", flush=True)
        if args.b7_pool is not None:                      # pooled B7 anchor (dominance => favourite >= logger's average reward for the user): Hoeffding on the user's logged rewards, range [LO, HI]
            invr = P["inv"][Wp["uid"]]; cnt = np.bincount(invr, minlength=P["nu"]); ysum = np.bincount(invr, weights=np.asarray(Wp["y"], float), minlength=P["nu"])
            ybar = ysum / np.maximum(cnt, 1); lbpool = ybar - (HI - LO) * np.sqrt(np.log(1 / args.b7_pool) / (2 * np.maximum(cnt, 1)))
            P["b7_pool_lb"] = np.where(cnt > 0, lbpool, LO)
            r["b7_pool"] = dict(delta=args.b7_pool, mean_lb=float(P["b7_pool_lb"].mean()), users_positive=int((P["b7_pool_lb"] > LO).sum()), rounds_per_user_median=float(np.median(cnt)))
            print(f"[{method}] B7 pooled anchor (Hoeffding, delta {args.b7_pool}): {r['b7_pool']}", flush=True)
        Linfo = estimate_L(P) if args.L == "auto" else dict(status="given", L_hat=float(args.L))
        L_use = float(Linfo["L_hat"]); r["L_calibration"] = Linfo
        print(f"[{method}] L: {Linfo}", flush=True)
        lb3, ub3, av3 = B.box(P, L_use, P["ref_log"]); lb3z, ub3z, _ = B.box(P, L_use, P["ref_log_z"])
        b5 = {m: X.b5_bounds(P, m) for m in ("range", "q5")}
        b7 = {"B7_ord": X.b7_rows(P, None), f"B7_z{args.z:g}": X.b7_rows(P, args.z)}
        w = P["coef_v"]
        r["beliefs_info"] = dict(L=L_use, L_source=("logged-data 95th pct of noise-corrected same-action slopes" if args.L == "auto" else "given"), z=args.z, ref_z=args.ref_z, b3_refs_per_var_median=float(np.median(av3)), b3_share_vars_with_ref=float((av3 >= 1).mean()),
                                 b3_true_outside_mass=float((w * ((qv < lb3 - 1e-9) | (qv > ub3 + 1e-9))).sum() / w.sum()),
                                 b3z_true_outside_mass=float((w * ((qv < lb3z - 1e-9) | (qv > ub3z + 1e-9))).sum() / w.sum()),
                                 b5={m: dict(v[2], true_outside_mass=float((w * ((qv < v[0] - 1e-9) | (qv > v[1] + 1e-9))).sum() / w.sum())) for m, v in b5.items()},
                                 b7={k: dict(v[3], true_violation_share_rows=float((qv[v[0][0]] > qv[v[0][1]] + 1e-9).mean()),
                                             true_outside_mass=float((w * ((qv < v[1] - 1e-9) | (qv > v[2] + 1e-9))).sum() / w.sum())) for k, v in b7.items()})
        for rho in rhos:
            P["eta"] = (float(args.eta_fixed) if args.eta_fixed is not None else rho) if args.eta_se_z is None else np.maximum(rho, args.eta_se_z * P["gh_se"])   # per-cell band: at least rho, at least z x the level ridge's own SE
            if args.eta_se_z is not None:
                r.setdefault("eta_cell", {})[str(rho)] = dict(z=args.eta_se_z, median=float(np.median(P["eta"])), p90=float(np.percentile(P["eta"], 90)), max=float(P["eta"].max()), share_above_rho=float((P["eta"] > rho + 1e-12).mean()))
            lb5r, ub5r, _ = b5["range"]; lb5q, ub5q, _ = b5["q5"]
            domo, lb7o, ub7o, _ = b7["B7_ord"]; domz, lb7z, ub7z, _ = b7[f"B7_z{args.z:g}"]
            r.setdefault("truth_checks", {})[str(rho)] = truth_checks(P, rho, P["eta"])
            items = [("baseline", np.full(P["nV"], LO), np.full(P["nV"], HI), False, None), ("B3", lb3, ub3, True, None)] + ([] if args.ref_z is not None else [(f"B3_z{args.z:g}", lb3z, ub3z, True, None)]) + [
                     ("B5_range", lb5r, ub5r, False, None), ("B5_q5", lb5q, ub5q, False, None), ("B7_ord", lb7o, ub7o, False, domo), (f"B7_z{args.z:g}", lb7z, ub7z, False, domz),
                     (f"B5_q5+B7_z{args.z:g}", np.maximum(lb5q, lb7z), np.minimum(ub5q, ub7z), False, domz),
                     ("B3z+B5_q5", np.maximum(lb3z, lb5q), np.minimum(ub3z, ub5q), True, None),
                     (f"B3z+B5_q5+B7_z{args.z:g}", np.maximum(np.maximum(lb3z, lb5q), lb7z), np.minimum(np.minimum(ub3z, ub5q), ub7z), True, domz)]
            for name, lb, ub, coup, dm in items:
                if args.beliefs and name not in args.beliefs and name != "baseline":
                    continue
                key = f"{rho}|{name}"
                if key in r["runs"] and not args.redo:
                    continue
                lo0, up0, info, secs = X.solve_both(P, rho, lb, ub, L=L_use, coupling=coup, dom=dm, tl=args.time_limit, max_it=args.max_it)
                if lo0 is None and args.soft_t and name != "baseline":      # option (5): relax the extra beliefs by t, smallest feasible t in the list
                    tried = {"0": info["status"]}
                    for t_ in args.soft_t:
                        lo0, up0, info, secs = X.solve_both(P, rho, lb, ub, L=L_use, coupling=coup, dom=dm, tl=args.time_limit, max_it=args.max_it, t=t_)
                        tried[f"{t_:g}"] = info["status"]
                        if lo0 is not None:
                            break
                    info = dict(info, soft_t_tried=tried, min_feasible_t=(float(t_) if lo0 is not None else None))
                rr = B.rec(P, rho, L_use if coup else None, name, lo0, up0, info, secs)
                rr["n_constraints"] = dict(lc_rows=int(2 * P["n_pairs"]), coupling_cuts=int(info["cuts"]), dominance_rows=int(len(dm[0])) if dm is not None else 0, vars=int(P["nV"] + P["Kc"]))
                base_ = r["runs"].get(f"{rho}|baseline")
                if lo0 is not None and base_ is not None and base_.get("lp_lo") is not None:
                    rr["vs_baseline"] = dict(lo_change=lo0 - base_["lp_lo"], hi_change=up0 - base_["lp_hi"], never_wider=bool(lo0 >= base_["lp_lo"] - 1e-6 and up0 <= base_["lp_hi"] + 1e-6))
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
    ap.add_argument("--clusterings", nargs="+", default=["kmeans", "agglomerative", "spectral"], choices=("kmeans", "agglomerative", "hierarchical", "spectral", "dgp", "random"))
    ap.add_argument("--real", default=None, choices=("eurlex", "wiki"), help="real-data arm: the paper's extreme-classification bandit (needs --data-dir with <name>/{train,test}.txt)")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "..", "data"))
    ap.add_argument("--lam", type=float, default=None, help="overlap threshold: pairs with pi/pi0 > lam go to the LP (None = only pi0 = 0, i.e. lambda = inf)")
    ap.add_argument("--beta", type=float, default=30.0, help="real arm: logging softmax inverse temperature (paper: 30)")
    ap.add_argument("--logger", default="paper", choices=("paper", "true_q"), help="real arm logging policy: paper = softmax(beta * ridge r_hat) (paper real-data code), true_q = softmax(beta * true q) (paper synthetic-style)")
    ap.add_argument("--eps-target", type=float, default=0.1, help="real arm: target eps-greedy exploration (paper: 0.1)")
    ap.add_argument("--alpha", default="1.0", help="ridge penalty for the cluster-level centre and the level ridge: a number (plots 03-10 used 1.0) or 'cv' = chosen by out-of-fold MSE on logged rounds")
    ap.add_argument("--eta-se-z", type=float, default=None, help="per-cell level band eta_cell = max(rho, z * SE of the cell's level ridge estimate); default: eta = rho for every cell")
    ap.add_argument("--eps-min", type=float, default=1.0, help="lowest displayed eps of the sweep grid (synthetic: 1.0; real [0,1] outcomes: e.g. 0.02)")
    ap.add_argument("--eps-grid", type=float, nargs="+", default=None, help="force the displayed eps grid (e.g. 1 2 3); eps_max is still computed and recorded but not used; beliefs run at every grid point")
    ap.add_argument("--npts", type=int, default=5, help="eps grid points in [1, eps_max] (plot 04 used 5)")
    ap.add_argument("--L", default="3.0", help="Lipschitz constant for B3: a number, or 'auto' = estimated from logged data (estimate_L), frozen before any bound")
    ap.add_argument("--z", type=float, default=3.5, help="margin multiplier for B7 logged lower bounds (and the B3_z variant)")
    ap.add_argument("--ref-z", type=float, default=None, help="if set, ONE margin multiplier for every logged reference interval (B3 references and B5 endpoints); the separate B3_z variant is then skipped")
    ap.add_argument("--beliefs", nargs="*", default=None, help="subset of belief names to run (default all)")
    ap.add_argument("--time-limit", type=float, default=3600.0); ap.add_argument("--max-it", type=int, default=8)
    ap.add_argument("--h-shrink", default=None, help="two_step only: weight s on the pairwise correction h_hat (f = g_hat_s + s h_hat, g refit on y - s h); a number, or 'cv' = s in {0,.25,.5,.75,1} by out-of-fold MSE on the logged rounds")
    ap.add_argument("--g-model", default="mlp", choices=("mlp", "ridge"), help="two_step only: first stage g_hat(x,c): paper's MLP (OBP RegressionModel) or one ridge per cluster on the user context (alpha = --alpha)")
    ap.add_argument("--b7-pool", type=float, default=None, help="B7 logged variant: add the pooled anchor mu(u,a*) >= Hoeffding lower bound (level delta) of the user's mean logged reward (valid under favourite dominance)")
    ap.add_argument("--eta-fixed", type=float, default=None, help="hold the level band eta at this value while rho sweeps (default eta = rho)")
    ap.add_argument("--binomial-refs", action="store_true", help="Bernoulli outcomes: Clopper-Pearson intervals (level from z) for every logged reference (B3 refs, B5 endpoints, B7 bounds) instead of mean +- z sigma / sqrt(n)")
    ap.add_argument("--clip-model", default=None, choices=(None, "cluster_logged", "outcome_cell"), help="option (4): cluster_logged = clip m_hat(u,a) to [min, max] of the logged rewards of a's cluster; outcome_cell = shift each cell's level so that the LC centre m_hat + g_hat lies in the outcome range")
    ap.add_argument("--soft-t", type=float, nargs="*", default=None, help="option (5): when an extra belief is infeasible, relax every extra-belief constraint by t (reference bounds +-t, coupling L d + t, dominance <= t) for t in this ascending list and keep the smallest feasible t")
    ap.add_argument("--model", default="cluster_level", choices=("cluster_level", "two_step"), help="reward model m_hat: cluster-level ridge (plots 03-10) or OffCEM's two-step f_hat = g_hat(x,c) + h_hat(x,a) (paper code)")
    ap.add_argument("--folds", default="round", choices=("round", "user"), help="cross-fitting folds for m_hat: round (plot-04 setup; merged LP can be infeasible when fold models disagree) or user (merged LP = closed form)")
    ap.add_argument("--tag", default="offcem_default"); ap.add_argument("--redo", action="store_true")
    run(ap.parse_args())
