#!/usr/bin/env python3
"""B3 (references + coupling) on the EARLIER setup of plot 04: K-means clusters, original exclusions, lambda = inf, seed 7000, by-ROUND folds, eta = rho.
Colours follow plot 04 (viridis position of each rho in the plot-04 grid).

    cd "Test notebooks/scripts/offcem_synthetic" && conda run -n offcem python plot_b3_round_folds.py
    -> ../../slides_figs/09_lamInf_kmeans_B3coupling_earlierSetup_byRoundFolds_etaRho_eps5.85-7.47.{png,pdf,_hires.png}

Input: results_offcem_global/b3_lipschitz_seed7000_eta_rho_roundfolds.json (b3_lipschitz_experiment.py --eta rho --folds round).
First x = the plot-04 bar (per-round closed form). Second x = the same beliefs with mu merged to one variable per (user, action) (the LP that B3 is built on). Then + B3 at L = 3, 3.5, 4.4.
"""
import json
import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EST = {"ips": ("IPS", "#1f77b4"), "dr": ("DR", "#17becf"), "offcem": ("OffCEM", "#e377c2")}
GRID04 = [1.0, 2.62, 4.23, 5.85, 7.47]
COLS04 = plt.cm.viridis(np.linspace(0.05, 0.85, len(GRID04)))


def main():
    d = json.load(open(os.path.join(BASE, "results_offcem_global", "b3_lipschitz_seed7000_eta_rho_roundfolds.json")))
    s = d["setup"]; V, Vp = s["V"], s["V_plus_true"]
    rhos, Ls = [5.85, 7.47], (3.0, 3.5, 4.4)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8), sharey=True)
    allv = [V, Vp] + [v["estimate"] for v in s["supported"].values()]
    for ax, rho in zip(axes, rhos):
        c = COLS04[GRID04.index(rho)]
        base = d["runs"][f"{rho}|baseline"]
        items = [("plot 04 bar\n(per-round\nclosed form)", None), ("baseline\n(μ merged\nper user-action)", base)] + [(f"+ B3 + coupling\nL = {L:g}", d["runs"][f"{rho}|{L}|coupling"]) for L in Ls]
        for k, (lab, r) in enumerate(items):
            for dx, sp, mk, fill, ls in ((-0.24, "dr", "o", True, "-"), (0.0, "ips", "o", False, (0, (2.2, 1.6))), (0.24, "offcem", "D", True, (0, (0.8, 1.0)))):
                if r is None:  # closed form + supported estimate +- 1.96 total SE (what plot 04 shows)
                    cf = base["closed_form"]; sup = s["supported"][sp]; z = 1.96 * sup["total_se"]
                    lo, hi = sup["estimate"] + cf[0] - z, sup["estimate"] + cf[1] + z
                else:
                    lo, hi = r[sp]["lo"], r[sp]["hi"]
                ax.plot([k + dx] * 2, [lo, hi], color=c, lw=3, ls=ls)
                ax.plot([k + dx] * 2, [lo, hi], ls="", marker=mk, ms=9, mfc=c if fill else "white", mec=c, mew=2)
                allv += [lo, hi]
                if sp == "dr":
                    wdr, botdr = hi - lo, lo
            bot = min(allv[-6], allv[-4], allv[-2])
            ax.text(k, bot - 0.45, f"w {wdr:.1f}", ha="center", va="top", fontsize=9.5, color=c)
        for sp, (nm, col) in EST.items():
            ax.axhline(s["supported"][sp]["estimate"], color=col, ls="--", lw=2)
        ax.axhline(Vp, color="black", ls="--", lw=2); ax.axhline(V, color="black", ls=(0, (1, 1.6)), lw=2)
        ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=10)
        ax.set_xlim(-0.6, len(items) - 0.4); ax.grid(axis="y", color="#ddd", lw=0.6)
        ax.set_title(f"ε = ρ = η = {rho}" + ("  (ε_max)" if rho == 7.47 else ""), fontsize=13)
    pad = 0.09 * (max(allv) - min(allv))
    axes[0].set_ylim(min(allv) - pad, max(allv) + pad); axes[0].set_ylabel("Policy value", fontsize=13)
    hs = [Line2D([0], [0], color=col, ls="--", lw=2, label=f"{nm} on supported part V⁺ = {s['supported'][sp]['estimate']:.2f}") for sp, (nm, col) in EST.items()]
    hs += [Line2D([0], [0], color="black", ls="--", lw=2, label=f"True supported part V⁺ = {Vp:.2f}"), Line2D([0], [0], color="black", ls=(0, (1, 1.6)), lw=2, label=f"True full value V = {V:.2f}"),
           Line2D([0], [0], color="gray", lw=3, marker="o", ms=9, label="DR + LP bound"), Line2D([0], [0], color="gray", lw=3, ls=(0, (2.2, 1.6)), marker="o", ms=9, mfc="white", mew=2, label="IPS + LP bound"),
           Line2D([0], [0], color="gray", lw=3, ls=(0, (0.8, 1.0)), marker="D", ms=8, label="OffCEM + LP bound")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.10))
    fig.suptitle("B3 (Lipschitz across users + coupling) on the earlier setup of plot 04 — K-means clusters (K = 10), original exclusions, λ = ∞, seed 7000, by-round folds, η = ρ", fontsize=13.5, y=0.985)
    note = ("One world; same world, model (m̂ cross-fitted by ROUND) and supported estimates as plot 04; η = ρ as in plot 04. 'w' = width of the DR bar. First x reproduces the plot-04 bar (per-round closed form). "
            "Second x: the same beliefs with μ_{u,a} merged into one variable per (user, action), which is what B3 is added to; with by-round folds m̂ differs across a user's rounds, so merging is slightly tighter than the per-round closed form "
            f"(with by-user folds the two coincide). B3: |μ_(u,a) − μ_(v,a)| ≤ L·d(x_u, x_v), logged reference boxes (mean ± 1.96·σ̂/√n, σ̂ = {s['sigma_hat']:.2f}) plus unsupported-to-unsupported coupling (full pair check passed). "
            f"Bars = supported estimate + LP, widened by 1.96 SE; LP mass {s['lp_mass']:.2f}. Dashed lines = supported part V⁺ only. Displayed ε = ρ = stored ε_LC / 2. Source: b3_lipschitz_seed7000_eta_rho_roundfolds.json.")
    fig.text(0.012, 0.006, "\n".join(textwrap.fill(p, 195) for p in note.split("\n")), fontsize=8.5, color="#555", va="bottom")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.30, wspace=0.05)
    stem = os.path.join(BASE, "slides_figs", "09_lamInf_kmeans_B3coupling_earlierSetup_byRoundFolds_etaRho_eps5.85-7.47")
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); fig.savefig(stem + "_hires.png", dpi=250)
    print("wrote", stem)


if __name__ == "__main__":
    main()
