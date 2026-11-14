#!/usr/bin/env python3
"""Convergence of the estimated LP bounds on V^2 to the population bounds (paper Sec. 3.3 / experiment (iii)).

Fixed small world (convergence.fixed_world, seed 7000), K-means labels (10 clusters, action features), lambda = 100, frozen reward model m~ (convergence.freeze_model:
two-step, ridge first stage, trained once on an independent log of 10,000 rounds). For every n and repetition: draw a log (convergence.sample_log), estimate
everything from it (level centres b^, LP weights, PB thresholds, eps^ = calibrated eps_max, L^), and solve six items with one b per (user, cluster)
(prep(cells='user')): base (13)-(15), + Lipschitz, + PB_sev, + PB_sev+ext, + Lipschitz + PB_sev, + Lipschitz + PB_sev+ext. eta = eps.
Two eps modes:  'calibrated'  eps = eps^_n, L = L^_n, compared with the population LP at eps*, L*   (everything estimated)
                'fixed'       eps = eps*, L = L*, compared with the same population LP              (only b^, weights, thresholds estimated)
Target: the LP bound on V^2 only (no supported part, no SE widening). Population inputs and bounds: population.py.

    cd scripts/offcem_synthetic
    conda run -n offcem python convergence_experiment.py --ns 1000 3000 10000 30000 100000 --reps 20 --workers 4
    -> ../../results_offcem_global/convergence/convergence_seed7000.json (saved after every log; re-running skips finished logs)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np

import b3_lipschitz_experiment as B
import b5_b7_experiment as X
import convergence as CV
import population as PO
from scale_experiment import labels, estimate_L, prob_belief_box

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "results_offcem_global", "convergence")
LO, HI, LAM = -30.0, 30.0, 100.0
PBA = SimpleNamespace(pb_sev_q=.05, pb_poor_q=.5, pb_alpha=.2, pb_ext_q=.01, pb_ext_p=.02, pb_cell=1.0)
ITEMS = [("base", None, False), ("Lip", None, True), ("PB_sev", "PB_sev", False), ("PB_sev+ext", "PB_sev+ext", False),
         ("Lip+PB_sev", "PB_sev", True), ("Lip+PB_sev+ext", "PB_sev+ext", True)]
_G = {}


def setup(seed, n_train, train_seed, n_masks):
    """World, labels, frozen model (cached on disk) and every population quantity."""
    os.makedirs(OUTDIR, exist_ok=True)
    world = CV.fixed_world(seed)
    lab = labels({**CV.sample_log(world, 1000, 5), "C": world["cfg"].C}, "kmeans", seed)
    f_m = os.path.join(OUTDIR, f"mtilde_seed{seed}_train{n_train}_{train_seed}.npy"); f_i = f_m.replace(".npy", ".json")
    if os.path.exists(f_m):
        table, minfo = np.load(f_m), json.load(open(f_i))
    else:
        table, minfo = CV.freeze_model(world, lab, n_train=n_train, train_seed=train_seed)
        np.save(f_m, table); json.dump(minfo, open(f_i, "w"))
    Epi0, Preg, pi = PO.mask_moments(world, lam=LAM, n_masks=n_masks)
    w = PO.lp_weights(pi, Preg); Bst = PO.level_centres(world, table, lab, Epi0)
    th = PO.thresholds(world, Epi0); es = PO.eps_star(world, table, lab, Epi0); Ls = PO.L_star(world)
    pb = prob_belief_box(None, LO, HI, PBA, thresholds=th)
    PP = PO.population_P(world, table, lab, w, Bst, es["eps_max"])
    pop = dict(eps_star=es["eps_max"], tau_star=es["tau"], L_star=Ls, thresholds=th, pb_lower={k: pb[k][0] for k in ("PB_sev", "PB_sev+ext")}, lp_mass=float(w.sum()), bounds={})
    for name, pbk, coup in ITEMS:
        lb = np.full(PP["nV"], LO if pbk is None else pb[pbk][0])
        lo, up, info, secs = X.solve_both(PP, es["eps_max"], lb, np.full(PP["nV"], HI), L=Ls if coup else None, coupling=coup, tl=1e9)
        pop["bounds"][name] = dict(lo=lo, hi=up, status=info["status"], seconds=secs)
    return world, lab, table, minfo, pop


def _init(seed, n_train, train_seed):
    world = CV.fixed_world(seed)
    _G.update(world=world, lab=labels({**CV.sample_log(world, 1000, 5), "C": world["cfg"].C}, "kmeans", seed),
              table=np.load(os.path.join(OUTDIR, f"mtilde_seed{seed}_train{n_train}_{train_seed}.npy")))


def one_log(task):
    n, rep, log_seed, eps_star, L_star = task
    world, lab, table = _G["world"], _G["lab"], _G["table"]
    t0 = time.perf_counter()
    W = CV.sample_log(world, n, log_seed)
    P = B.prep(log_seed, user_folds=True, W=W, lab=lab, model="two_step", lam=LAM, limits=[LO, HI], M_fixed=CV.expand(table, W), cells="user", g_model="ridge")
    P["gh_cell"] = P["gh"]; P["ref_mean_raw"] = (P["ref_log"][0] + P["ref_log"][1]) / 2
    eps_hat = float(P["cal"]["eps_max"]); Linfo = estimate_L(P); L_hat = float(Linfo["L_hat"])
    pb = prob_belief_box(P["W"]["y"], LO, HI, PBA)
    rec = dict(n=n, rep=rep, log_seed=log_seed, eps_hat=eps_hat, L_hat=L_hat, L_status=Linfo["status"], thresholds=pb["thresholds"],
               pb_lower={k: pb[k][0] for k in ("PB_sev", "PB_sev+ext")}, nV=int(P["nV"]), Kc=int(P["Kc"]), rows=int(2 * P["n_pairs"]),
               nonident_mass=float((P["const_up"] - P["const_lo"]) / (HI - LO)), lp_mass_ident=float(P["coef_v"].sum()), prep_seconds=time.perf_counter() - t0, modes={})
    for mode, eps, L in (("calibrated", eps_hat, L_hat), ("fixed", eps_star, L_star)):
        P["eta"] = eps; res = {}
        for name, pbk, coup in ITEMS:
            lb = np.full(P["nV"], LO if pbk is None else pb[pbk][0])
            lo, up, info, secs = X.solve_both(P, eps, lb, np.full(P["nV"], HI), L=L if coup else None, coupling=coup, tl=1e9)
            res[name] = dict(lo=lo, hi=up, status=info["status"], seconds=secs, cuts=info.get("cuts"))
        rec["modes"][mode] = dict(eps=eps, L=L, items=res)
    rec["seconds"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000); ap.add_argument("--ns", type=int, nargs="+", default=[1000, 3000, 10000, 30000, 100000])
    ap.add_argument("--reps", type=int, default=20); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=10000); ap.add_argument("--train-seed", type=int, default=1); ap.add_argument("--n-masks", type=int, default=50000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = os.path.join(OUTDIR, f"convergence{a.tag}_seed{a.seed}.json")
    t0 = time.perf_counter()
    world, lab, table, minfo, pop = setup(a.seed, a.n_train, a.train_seed, a.n_masks)
    d = json.load(open(out)) if os.path.exists(out) else {}
    d.update(config=dict(seed=a.seed, world=CV.SMALL_WORLD, clusters="kmeans", C=world["cfg"].C, lam=LAM, limits=[LO, HI], model="two_step, ridge first stage, frozen",
                         frozen_model=minfo, n_masks=a.n_masks, pb=vars(PBA), eta="= eps", cells="user"), population=pop, setup_seconds=time.perf_counter() - t0)
    d.setdefault("logs", {})
    json.dump(d, open(out, "w"), indent=1)
    print(f"population: eps* {pop['eps_star']:.3f}  L* {pop['L_star']:.3f}  " + "  ".join(f"{k} [{v['lo']:.2f}, {v['hi']:.2f}]" if v["lo"] is not None else f"{k} infeasible" for k, v in pop["bounds"].items()), flush=True)
    tasks = [(n, r, 100000 * (i + 1) + r, pop["eps_star"], pop["L_star"]) for i, n in enumerate(a.ns) for r in range(a.reps) if f"{n}|{r}" not in d["logs"]]
    tasks.sort(key=lambda t: -t[0])                    # largest logs first
    print(f"{len(tasks)} logs to run", flush=True)
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.seed, a.n_train, a.train_seed)) as pool:
        for rec in pool.imap_unordered(one_log, tasks):
            d["logs"][f"{rec['n']}|{rec['rep']}"] = rec; json.dump(d, open(out, "w"), indent=1)
            c = rec["modes"]["calibrated"]["items"]
            print(f"n {rec['n']:6d} rep {rec['rep']:2d}  {rec['seconds']:5.0f}s  eps^ {rec['eps_hat']:.2f}  L^ {rec['L_hat']:.2f}  base " +
                  (f"[{c['base']['lo']:.2f}, {c['base']['hi']:.2f}]" if c["base"]["lo"] is not None else "infeasible") +
                  "  infeasible items (cal/fixed): " + str(sum(v["lo"] is None for v in c.values())) + "/" + str(sum(v["lo"] is None for v in rec["modes"]["fixed"]["items"].values())), flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
