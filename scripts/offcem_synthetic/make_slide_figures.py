#!/usr/bin/env python3
"""Figures for the OffCEM slides, in the style of Dist_OPE.pptx (default matplotlib look, thick vertical bounds with round
end markers, dashed blue IPW line, dashed black true value, light y grid).

    python make_slide_figures.py --fixed ../../results_offcem_global/joint_experiment_50.json \
        --rounds ../../results_offcem_global/joint_experiment_round_random.json --out ../../slides_figs
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import partial_ope_global as G
from concurrent.futures import ProcessPoolExecutor
from obp.ope import DoublyRobust as OBP_DR, InverseProbabilityWeighting as OBP_IPW
from ope import OffCEM

warnings.filterwarnings("ignore")
TAB = plt.rcParams["axes.prop_cycle"].by_key()["color"]      # tab:blue, orange, green, red, purple, brown, pink ...
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, PINK = TAB[:7]
CYAN = TAB[9]
EST_COLOR = {"IPW": BLUE, "DR": CYAN, "OffCEM": PINK}
EPS_COLOR = {None: ORANGE, 2.0: GREEN, 4.0: RED, 6.0: PURPLE, 8.0: BROWN}


def estimators(W, M):
    """IPW and DR from OBP, OffCEM from the repo's ope.OffCEM (cluster weights, is_clustering=True); DR and OffCEM use the
    same cross-fitted cluster-level reward model M as the LP. No weight cap. Cross-checked against the closed form."""
    n, K, C = W["n"], W["K"], W["C"]
    uid, act, y = W["uid"], W["act"], W["y"]
    pi, pi0, cl = W["pi_unit"][uid], W["pi0_unit"][uid], W["cl"]
    Mr = M[uid]
    ad, qh = pi[:, :, None], Mr[:, :, None]
    out = {"IPW": float(OBP_IPW().estimate_policy_value(reward=y, action=act, action_dist=ad, pscore=W["pscore"])),
           "DR": float(OBP_DR().estimate_policy_value(reward=y, action=act, action_dist=ad, pscore=W["pscore"], estimated_rewards_by_reg_model=qh))}
    onehot = np.eye(C)[cl]
    pea = np.broadcast_to(onehot, (n, K, C)).astype(float)
    oc = OffCEM(n_actions=K, estimator_name="OffCEM", is_clustering=True)
    out["OffCEM"] = float(oc.estimate_policy_value(reward=y, action=act, action_embed=np.zeros((n, 1), int), pi_b=pi0[:, :, None], action_dist=ad,
                                                    estimated_rewards_by_reg_model=qh, p_e_a=pea))
    pc, p0c = pi @ onehot, pi0 @ onehot
    w = pc[np.arange(n), cl[act]] / p0c[np.arange(n), cl[act]]
    cf = float(((pi * Mr).sum(1) + w * (y - Mr[np.arange(n), act])).mean())
    assert abs(cf - out["OffCEM"]) < 1e-8, (cf, out["OffCEM"])
    return out


def world_estimates(args):
    seed, mode = args
    W = G.make_world(G.WorldCfg(seed=seed, exclusion=mode))
    M = G.get_model(W, "cluster_level")
    V = float(((W["counts"] / W["n"])[:, None] * W["pi_unit"] * W["q_unit"]).sum())
    return {"seed": seed, "V": V, **estimators(W, M)}


def all_estimates(cache, n_fixed, n_rounds, workers=6):
    cache = Path(cache)
    if cache.exists():
        d = json.load(open(cache))
        if len(d["user_fixed"]) >= n_fixed and len(d["round_random"]) >= n_rounds:
            return d
    jobs = [(7000 + i, "user_fixed") for i in range(n_fixed)] + [(7000 + i, "round_random") for i in range(n_rounds)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(world_estimates, jobs))
    d = {"user_fixed": res[:n_fixed], "round_random": res[n_fixed:]}
    cache.write_text(json.dumps(d, indent=1))
    return d


def ips_estimate(W):
    """Plain IPW estimate on the logged rounds (no cap): only logged pairs enter, so the excluded mass is missed."""
    w = W["pi_unit"][W["uid"], W["act"]] / W["pscore"]
    return float((w * W["y"]).mean())


def label(e):
    return "No local\ncorrectness" if e is None else f"\u03b5 = {e:g}"


def bar(ax, x, lo, hi, color, solid=True, lw=4.5):
    ax.plot([x, x], [lo, hi], color=color, lw=lw, ls="-" if solid else (0, (2.2, 1.6)), solid_capstyle="butt", zorder=2)
    for y in (lo, hi):
        ax.plot([x], [y], "o", ms=9, color=color, mfc=color if solid else "white", mew=2, zorder=3)


def single_world(run, est, fname, title, eps_list):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    rows = {r["eps_lc"]: r for r in run["results"]}
    keep = [e for e in eps_list if e in rows and rows[e]["status"] == "optimal"]
    for i, e in enumerate(keep):
        bar(ax, i, rows[e]["lower"], rows[e]["upper"], EPS_COLOR[e])
    for k in ("IPW", "DR", "OffCEM"):
        ax.axhline(est[k], color=EST_COLOR[k], ls="--", lw=2, label=f"{k} estimate = {est[k]:.2f}")
    ax.axhline(run["V"], color="black", ls="--", lw=2, label=f"True value = {run['V']:.2f}")
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([label(e) for e in keep])
    ax.set_xlim(-0.6, len(keep) - 0.4)
    ax.set_ylabel("Policy value")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.32)
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def relative(runs_by_mode, fname, title, eps_list, est_err=None):
    """Median (lower - V) and (upper - V) over worlds, per eps. runs_by_mode: [(name, runs, solid)]."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    n_mode = len(runs_by_mode)
    off = np.linspace(-0.16, 0.16, n_mode) if n_mode > 1 else [0.0]
    ticks = []
    for i, e in enumerate(eps_list):
        cov_txt = []
        for (name, runs, solid), dx in zip(runs_by_mode, off):
            rs = [r for run in runs for r in run["results"] if r["eps_lc"] == e and r["status"] == "optimal"]
            vs = [run["V"] for run in runs for r in run["results"] if r["eps_lc"] == e and r["status"] == "optimal"]
            lo = np.median([r["lower"] - v for r, v in zip(rs, vs)])
            hi = np.median([r["upper"] - v for r, v in zip(rs, vs)])
            bar(ax, i + dx, lo, hi, EPS_COLOR[e], solid=solid, lw=4.0 if n_mode > 1 else 4.5)
            cov_txt.append(f"{np.mean([r['covers'] for r in rs]):.0%}")
        ticks.append(f"{label(e)}\ncovers {' / '.join(cov_txt)}")
    for k, v in (est_err or {}).items():
        ax.axhline(v, color=EST_COLOR[k], ls="--", lw=2, label=f"{k} (median error = {v:+.2f})")
    ax.axhline(0.0, color="black", ls="--", lw=2, label="True value (0)")
    ax.set_xticks(range(len(eps_list)))
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_xlim(-0.6, len(eps_list) - 0.4)
    ax.set_ylabel("Bound minus true value (median)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.32)
    ax.legend(loc="upper center", ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def dgp_figures(out):
    W = G.make_world(G.WorldCfg(seed=7000))
    q, pi, pi0, cl = W["q_unit"], W["pi_unit"], W["pi0_unit"], W["cl"]
    u = next(i for i in range(len(q)) if pi0[i, q[i].argmax()] == 0)          # a user whose best action is excluded
    order = np.argsort(cl, kind="stable")
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for c in range(W["C"]):
        idx = np.flatnonzero(order >= 0)
        a = order[cl[order] == c]
        ex = pi0[u, a] == 0
        x = np.flatnonzero(cl[order] == c)
        ax.scatter(x[~ex], q[u, a][~ex], s=14, color=TAB[c % 10])
        ax.scatter(x[ex], q[u, a][ex], s=14, facecolors="white", edgecolors=TAB[c % 10], linewidths=1.0)
    ax.scatter([], [], s=14, color="gray", label="logged-able")
    ax.scatter([], [], s=14, facecolors="white", edgecolors="gray", label="excluded")
    ax.set_xlabel("Actions (grouped by cluster)")
    ax.set_ylabel("Mean reward")
    ax.set_title("One user: mean reward per action", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "dgp_actions.png", dpi=100)
    plt.close(fig)
    top = np.argsort(-q[u])[:8]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    x = np.arange(len(top))
    ax.bar(x - 0.2, pi[u, top], 0.4, color=BLUE, label="Target policy")
    ax.bar(x + 0.2, pi0[u, top], 0.4, color=ORANGE, label="Logging policy")
    for xi, a in zip(x, top):
        if pi0[u, a] == 0:
            ax.text(xi + 0.2, 0.01, "x", ha="center", color=ORANGE, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i + 1}" for i in x])
    ax.set_xlabel("Actions ranked by mean reward (x = excluded)")
    ax.set_ylabel("Probability")
    ax.set_title("Same user: who plays the best actions", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dgp_policies.png", dpi=100)
    plt.close(fig)
    return {"user": int(u), "target_mass_best": float(pi[u, top[0]]), "logging_mass_best": float(pi0[u, top[0]])}


SPLIT_EPS = [2.0, 4.0, 6.0, 8.0]
SPLIT_SPEC = dict(model="cluster_level", cluster_model="ridge", centre="ridge", noise="aggregate", overlap_lambda=10.0, limits=[-30.0, 30.0],
                  eps_lc=SPLIT_EPS, ips_z=1.96, lp="closed", eta=0.0, add_rho=True, repair=False)
SPLIT_WORLD = dict(K=200, n=1000, users=100, C=10, def_frac=0.5, exclusion="user_fixed", sigma=3.0, beta=-0.1, eps_policy=0.2, clustering="original")


def split_run(args):
    """Split setup (partial_ope_global.py): LP only on the unsupported pairs (and weights above lambda); supported part from DR or IPS;
    aggregate confidence interval for the shared level; interval = supported estimate + LP interval, widened by 1.96 SE."""
    seed, sup = args
    out = G.run_setting({**SPLIT_SPEC, "supported": sup, "world": {**SPLIT_WORLD, "seed": seed}})
    d = out["data"]
    rows = {}
    for r in out["results"]:
        e = float(r["eps_lc"])
        ok = r["status"] == "optimal"
        rows[e] = dict(ok=ok, lo=r["with_se"][0] if ok else None, hi=r["with_se"][1] if ok else None,
                       covers=bool(r["contains_V_with_se"]) if ok else False)
    return {"seed": seed, "sup": sup, "V": d["oracle_V"], "supported_estimate": d["supported_estimate"], "supported_se": d["total_se"],
            "unsupported_mass": d["unsupported_policy_mass"], "rows": rows}


def all_split(cache, n, workers=6):
    cache = Path(cache)
    if cache.exists():
        d = json.load(open(cache))
        if len(d["dr"]) >= n:
            return d
    jobs = [(7000 + i, sup) for sup in ("dr", "ips") for i in range(n)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(split_run, jobs))
    d = {"dr": res[:n], "ips": res[n:]}
    cache.write_text(json.dumps(d, indent=1))
    return d


def split_stats(rows):
    out = {}
    for e in SPLIT_EPS:
        k = str(e)
        rs = [r["rows"][k] if k in r["rows"] else r["rows"][e] for r in rows]
        okw = [x for x in rs if x["ok"]]
        out[e] = dict(feasible=len(okw) / len(rs), coverage=float(np.mean([x["covers"] for x in rs])),
                      width=float(np.median([x["hi"] - x["lo"] for x in okw])) if okw else None)
    return out


def split_single(dr, ips, est, fname, title="OPE estimates and partial-identification bounds"):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, e in enumerate(SPLIT_EPS):
        a, b = dr["rows"][str(e)], ips["rows"][str(e)]
        if a["ok"]:
            bar(ax, i - 0.14, a["lo"], a["hi"], EPS_COLOR[e], solid=True, lw=4.0)
        if b["ok"]:
            bar(ax, i + 0.14, b["lo"], b["hi"], EPS_COLOR[e], solid=False, lw=4.0)
    for k in ("IPW", "DR", "OffCEM"):
        ax.axhline(est[k], color=EST_COLOR[k], ls="--", lw=2)
    ax.axhline(dr["V"], color="black", ls="--", lw=2)
    ax.set_xticks(range(len(SPLIT_EPS)))
    ax.set_xticklabels([f"\u03b5 = {e:g}" for e in SPLIT_EPS])
    ax.set_xlim(-0.6, len(SPLIT_EPS) - 0.4)
    ax.set_ylabel("Policy value")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.3)
    handles = [Line2D([0], [0], color=EST_COLOR[k], ls="--", lw=2, label=f"{k} estimate = {est[k]:.2f}") for k in ("IPW", "DR", "OffCEM")]
    handles += [Line2D([0], [0], color="black", ls="--", lw=2, label=f"True value = {dr['V']:.2f}"),
                Line2D([0], [0], color="gray", lw=4, marker="o", ms=8, label="DR + LP bound"),
                Line2D([0], [0], color="gray", lw=4, ls=(0, (2.2, 1.6)), marker="o", ms=8, mfc="white", label="IPW + LP bound")]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def med_err(rows):
    return {k: float(np.median([r[k] - r["V"] for r in rows])) for k in ("IPW", "DR", "OffCEM")}


def mean_abs_err(rows):
    return {k: float(np.mean([abs(r[k] - r["V"]) for r in rows])) for k in ("IPW", "DR", "OffCEM")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed", default="../../results_offcem_global/joint_experiment_50.json")
    ap.add_argument("--rounds", default="../../results_offcem_global/joint_experiment_round_random.json")
    ap.add_argument("--out", default="../../slides_figs")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"dgp": dgp_figures(out)}
    fixed = json.load(open(a.fixed))["runs"]["A cluster moment (base)"]
    rr = None
    if Path(a.rounds).exists():
        rr = json.load(open(a.rounds))["runs"]
        rr = rr[next(iter(rr))]
    est = all_estimates(out / "estimates.json", len(fixed), len(rr) if rr else 0, a.workers)
    title = "OPE estimates and partial-identification bounds"
    ef, er = est["user_fixed"][:len(fixed)], est["round_random"]
    single_world(fixed[0], ef[0], out / "fixed_single.png", title, [2.0, 4.0, 6.0, 8.0])
    relative([("fixed", fixed, True)], out / "fixed_relative.png", f"Bounds relative to the truth ({len(fixed)} worlds)", [2.0, 4.0, 6.0, 8.0],
             est_err=med_err(ef))
    info["fixed_estimators"] = {"worlds": len(ef), "median_error": med_err(ef), "mean_abs_error": mean_abs_err(ef), "seed7000": ef[0]}
    if rr:
        n = len(rr)
        single_world(rr[0], er[0], out / "rounds_single.png", title, [2.0, 4.0, 8.0])
        relative([("original", rr, True), ("fixed", fixed[:n], False)], out / "rounds_relative.png",
                 f"Original (solid) vs fixed (dashed); lines = original rule", [2.0, 4.0, 8.0], est_err=med_err(er))
        info["round_random_worlds"] = n
        info["round_estimators"] = {"worlds": n, "median_error": med_err(er), "mean_abs_error": mean_abs_err(er), "seed7000": er[0]}
        info["fixed_estimators_same_seeds"] = {"median_error": med_err(ef[:n]), "mean_abs_error": mean_abs_err(ef[:n])}
    sp = all_split(out / "split_bounds.json", len(fixed), a.workers)
    split_single(sp["dr"][0], sp["ips"][0], ef[0], out / "split_single.png")
    info["split"] = {"worlds": len(sp["dr"]), "dr": split_stats(sp["dr"]), "ips": split_stats(sp["ips"]),
                     "seed7000": {k: sp[k][0] for k in ("dr", "ips")}}
    (out / "figures_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
