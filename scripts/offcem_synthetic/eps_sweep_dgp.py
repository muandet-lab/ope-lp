#!/usr/bin/env python3
"""Unsupported-pairs LP, DGP clusters, lambda = inf: estimate a global eps_max from the LOGGED data, then sweep eps over [1, eps_max].

    cd "Test notebooks/scripts/offcem_synthetic"
    conda run -n offcem python eps_sweep_dgp.py --seed 7000 --excl round_random --npts 5     # one world, ~2 s
    (outputs: ../../results_offcem_global/eps_sweep_dgp_<excl>_seed<seed>.json and ../../slides_figs/eps_sweep_dgp_<excl>_seed<seed>.{png,pdf,_hires.png})

Convention: displayed eps = rho = the per-action radius (stored eps_LC = 2 * rho), as in all earlier plots. eps_max is the pooled calibrated rho.
eps_max estimate (logged data only, DGP clusters), same recipe as cluster_tolerance_experiment.calibrate:
  e_i = y_i - m_hat(round i, a_i)                      residual of a logged round (m_hat cross-fitted by unit)
  sigma^2 = pooled within-(user, action) variance of y over repeated logged rounds
  contrasts = pairs of logged rounds of DIFFERENT actions in the same (USER, cluster) cell:  E[(e_i - e_j)^2 / 2] = tau^2 + sigma^2
  tau^2 = max(0, mean contrast - sigma^2);   eps_max = Z * tau, Z = 2   (Gaussian-offset assumption, not a guarantee)
With round_random exclusions every round is its own unit, so a cell cannot be (round, cluster); the contrast cell is (user, cluster), the user being the round's
context owner (W['user_of_round']); rounds of one user share x, so their m_hat differs only through the cross-fitting folds.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
import json
import warnings
import numpy as np

import make_slide_figures as F
import partial_ope_global as G

warnings.filterwarnings("ignore")
Z = 2.0
SUPS = ("dr", "ips", "offcem")


def calibrate_user_cells(W, M):
    user, act, y = np.asarray(W["user_of_round"]), W["act"], W["y"]
    C = W["C"]
    clr = W["cl"][act]
    e = y - M[W["uid"], act]                             # M is indexed by unit (= round under round_random, = user under user_fixed)
    key = user.astype(np.int64) * 100000 + act
    ss, dof = 0.0, 0
    for k in np.unique(key):
        yk = y[key == k]
        if len(yk) > 1:
            ss += float(((yk - yk.mean()) ** 2).sum()); dof += len(yk) - 1
    sigma2 = ss / dof if dof >= 10 else float("nan")
    cell = user.astype(np.int64) * C + clr
    order = np.argsort(cell, kind="stable")
    cs, ces, cas = cell[order], e[order], act[order]
    halves = []
    for idx in np.split(np.arange(len(cs)), np.flatnonzero(np.diff(cs)) + 1):
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        keep = cas[idx][i] != cas[idx][j]
        halves.append(((ces[idx][i] - ces[idx][j]) ** 2 / 2.0)[keep])
    halves = np.concatenate(halves)
    raw = float(halves.mean()) - sigma2
    tau = float(np.sqrt(max(raw, 0.0)))
    return dict(sigma_noise=float(np.sqrt(sigma2)), n_repeat_dof=int(dof), n_pairs=int(len(halves)), mean_half_sq_contrast=float(halves.mean()),
                tau2_raw=raw, tau=tau, eps_max=Z * tau, z=Z)


def run(seed, excl, npts, lam=None, clusters="dgp"):
    W = G.make_world(G.WorldCfg(**{**F.SPLIT_WORLD, "seed": seed, "exclusion": excl}))
    if clusters == "learned":     # K-means (K = C = 10, domain knowledge) on the action features only; world data untouched
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        lab = KMeans(n_clusters=W["C"], n_init=10, random_state=0).fit_predict(np.asarray(W["action_x"], float)).astype(int)
        ari = float(adjusted_rand_score(W["cl"], lab)); W["cl"], W["C"] = lab, int(lab.max()) + 1
    else:
        ari = 1.0
    M = G.get_model(W, "cluster_level", "ridge", 5, 1.0, 1)
    cal = calibrate_user_cells(W, M)
    emax = cal["eps_max"]
    grid = [float(x) for x in (np.linspace(1.0, emax, npts) if emax > 1.0 else [emax])]
    out = dict(seed=seed, excl=excl, lam=("inf" if lam is None else lam), clusters=clusters, ari_vs_dgp=ari, calibration=cal, grid=grid, eps_max=emax, sup={})
    for sp in SUPS:
        spec = {**F.SPLIT_SPEC, "overlap_lambda": lam, "supported": sp, "eps_lc": [2 * g for g in grid]}
        r = G.evaluate(W, M, spec)
        d = r["data"]
        rows = []
        for g, x in zip(grid, r["results"]):
            ok = x["status"] == "optimal"
            rows.append(dict(eps=g, ok=ok, lo=x["with_se"][0] if ok else None, hi=x["with_se"][1] if ok else None, lp_lo=x["unsupported_interval"][0] if ok else None,
                             lp_hi=x["unsupported_interval"][1] if ok else None, covers_V=bool(x["contains_V_with_se"]) if ok else None,
                             truth_spread_violation_share=x["truth_spread_violation_share"]))
        out["sup"][sp] = dict(rows=rows, supported_estimate=d["supported_estimate"], total_se=d["total_se"])
        out.update(V=d["oracle_V"], V_plus_true=d["oracle_supported_part"], lp_mass=d["unsupported_policy_mass"])
    return out


def plot(res, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import textwrap
    grid, V, Vp = res["grid"], res["V"], res["V_plus_true"]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(grid)))
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    vals = [V, Vp]
    for k, (g, c) in enumerate(zip(grid, cols)):
        for dx, sp, mk, fill in ((-0.22, "dr", "o", True), (0.0, "ips", "o", False), (0.22, "offcem", "D", True)):
            r = res["sup"][sp]["rows"][k]
            ax.plot([k + dx] * 2, [r["lo"], r["hi"]], color=c, lw=3, ls={"dr": "-", "ips": (0, (2.2, 1.6)), "offcem": (0, (0.8, 1.0))}[sp])
            ax.plot([k + dx] * 2, [r["lo"], r["hi"]], ls="", marker=mk, ms=9, mfc=c if fill else "white", mec=c, mew=2)
            vals += [r["lo"], r["hi"]]
    est = {"ips": ("IPS", "#1f77b4"), "dr": ("DR", "#17becf"), "offcem": ("OffCEM", "#e377c2")}
    hs = []
    for sp, (nm, col) in est.items():
        v = res["sup"][sp]["supported_estimate"]; vals.append(v)
        ax.axhline(v, color=col, ls="--", lw=2)
        hs.append(Line2D([0], [0], color=col, ls="--", lw=2, label=f"{nm} on supported part V⁺ = {v:.2f}"))
    ax.axhline(V, color="black", ls=(0, (1, 1.6)), lw=2)
    ax.axhline(Vp, color="black", ls="--", lw=2)
    hs += [Line2D([0], [0], color="black", ls="--", lw=2, label=f"True supported part V⁺ = {Vp:.2f}"),
           Line2D([0], [0], color="black", ls=(0, (1, 1.6)), lw=2, label=f"True full value V = {V:.2f}"),
           Line2D([0], [0], color="gray", lw=3, marker="o", ms=9, label="DR + LP bound"),
           Line2D([0], [0], color="gray", lw=3, ls=(0, (2.2, 1.6)), marker="o", ms=9, mfc="white", mew=2, label="IPS + LP bound"),
           Line2D([0], [0], color="gray", lw=3, ls=(0, (0.8, 1.0)), marker="D", ms=8, label="OffCEM + LP bound")]
    pad = 0.06 * (max(vals) - min(vals))
    ax.set_ylim(min(vals) - pad, max(vals) + pad)
    ax.set_xticks(range(len(grid)))
    ax.set_xticklabels([f"ε = {g:.2f}" + ("\n(ε_max)" if k == len(grid) - 1 else "") for k, g in enumerate(grid)], fontsize=12)
    ax.set_xlim(-0.6, len(grid) - 0.4)
    ax.set_ylabel("Policy value", fontsize=13)
    ax.grid(axis="y", color="#ddd", lw=0.6)
    cal = res["calibration"]
    ax.set_title(f"{'DGP clusters' if res['clusters'] == 'dgp' else 'K-means clusters (K = 10)'}, {'original exclusions (redrawn per round)' if res['excl'] == 'round_random' else 'fixed exclusions'}, λ = {'∞' if res['lam'] == 'inf' else int(res['lam'])}:\nε swept over [1, ε_max = {res['eps_max']:.2f}]", fontsize=13)
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=10.5)
    note = (f"Seed {res['seed']}, one world; clusters: {res['clusters']} (ARI vs DGP {res['ari_vs_dgp']:.2f}). ε_max = Z·τ̂ with Z = 2, τ̂ = {cal['tau']:.2f} estimated from {cal['n_pairs']} within-(user, cluster) logged contrasts, "
            f"noise σ̂ = {cal['sigma_noise']:.2f} ({cal['n_repeat_dof']} repeat dof); logged data only. Bars = supported estimate + LP interval, widened by 1.96 SE; "
            f"λ = {res['lam']}, unsupported mass {res['lp_mass']:.2f}; ridge centre, no recentring, range [-30, 30]. Dashed lines = supported part V⁺ only. Displayed ε = stored ε_LC / 2.")
    fig.text(0.02, 0.008, "\n".join(textwrap.fill(p, 110) for p in note.split("\n")), fontsize=8.5, color="#555", va="bottom")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.90, bottom=0.34)
    fig.savefig(path_stem + ".png", dpi=100); fig.savefig(path_stem + ".pdf"); fig.savefig(path_stem + "_hires.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--excl", default="round_random", choices=("round_random", "user_fixed"))
    ap.add_argument("--npts", type=int, default=5)
    ap.add_argument("--clusters", default="dgp", choices=("dgp", "learned"))
    ap.add_argument("--lam", default="inf", help="weight cap lambda or inf")
    a = ap.parse_args()
    lam = None if a.lam == "inf" else float(a.lam)
    res = run(a.seed, a.excl, a.npts, lam, a.clusters)
    name = f"eps_sweep_dgp_{a.excl}_seed{a.seed}" + ("" if a.clusters == "dgp" else "_learned") + ("" if lam is None else f"_lam{a.lam}")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    jp = os.path.join(base, "results_offcem_global", name + ".json")
    json.dump(res, open(jp, "w"), indent=1)
    plot(res, os.path.join(base, "slides_figs", name))
    c = res["calibration"]
    print(f"seed {a.seed} {a.excl}: sigma {c['sigma_noise']:.2f} (dof {c['n_repeat_dof']}), pairs {c['n_pairs']}, tau {c['tau']:.2f}, eps_max {res['eps_max']:.2f}; V {res['V']:.3f}, V+ true {res['V_plus_true']:.3f}, mass {res['lp_mass']:.2f}")
    for sp in SUPS:
        s = res["sup"][sp]
        print(f"  {sp}: V+ estimate {s['supported_estimate']:.3f} (se {s['total_se']:.2f})")
        for r in s["rows"]:
            print(f"     eps {r['eps']:.2f}: [{r['lo']:.2f}, {r['hi']:.2f}] width {r['hi']-r['lo']:.2f} covers V {r['covers_V']}  violation_share {r['truth_spread_violation_share']:.2f}")
