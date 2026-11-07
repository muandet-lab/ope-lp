#!/usr/bin/env python3
"""Plot 10: baseline (plot-04 setup) + B3 (Lipschitz, coupled) + B5 (group mean bounds) + B7 (favourite-action dominance), with and without recentring.
K-means K = 10, original exclusions, lambda = inf, seed 7000, by-round folds, eta = rho; eps 5.85 and 7.47. Colour scheme: viridis as in plot 03 (one colour per belief set).

    cd "Test notebooks/scripts/offcem_synthetic" && conda run -n offcem python plot_b5_b7.py
    -> ../../slides_figs/10_lamInf_kmeans_B3-B5-B7_earlierSetup_byRoundFolds_etaRho_recentre_eps5.85-7.47.{png,pdf,_hires.png}
Input: results_offcem_global/b5_b7_seed7000_roundfolds.json (b5_b7_experiment.py).
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
ITEMS = [("plot 04 bar\n(closed form)", "plot04"), ("baseline\n(μ merged)", "baseline"), ("+ B3\nL = 3", "B3"), ("+ B5\nrange", "B5_range"), ("+ B5\n5–95 %", "B5_q5"),
         ("+ B7\nordering", "B7_ord"), ("+ B7\n+ logged\n(z = 3.5)", "B7_z3.5"), ("+ B3 (z = 3.5)\n+ B5 (5–95 %)\n+ B7 (z = 3.5)", "B3z+B5_q5+B7_z3.5")]
COLS = plt.cm.viridis(np.linspace(0.05, 0.85, len(ITEMS)))   # plot-03 scheme


def main():
    d = json.load(open(os.path.join(BASE, "results_offcem_global", "b5_b7_seed7000_roundfolds.json")))
    s = d["setup"]; V, Vp = s["V"], s["V_plus_true"]
    rhos = [5.85, 7.47]
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5), sharey=True)
    allv = [V, Vp] + [v["estimate"] for v in s["supported"].values()]
    for i, tag in enumerate(("norecentre", "recentre")):
        for j, rho in enumerate(rhos):
            ax = axes[i, j]
            base = d["runs"][f"{tag}|{rho}|baseline"]
            for k, (lab, key) in enumerate(ITEMS):
                c = COLS[k]
                r = base if key == "plot04" else d["runs"].get(f"{tag}|{rho}|{key}")
                if r is None or r.get("lp_lo") is None and key != "plot04":
                    ax.text(k, 0.0, "infeasible", rotation=90, ha="center", va="center", fontsize=10, color=c); continue
                lows = []
                for dx, sp, mk, fill, ls in ((-0.24, "dr", "o", True, "-"), (0.0, "ips", "o", False, (0, (2.2, 1.6))), (0.24, "offcem", "D", True, (0, (0.8, 1.0)))):
                    if key == "plot04":
                        cf = base["closed_form"]; sup = s["supported"][sp]; z = 1.96 * sup["total_se"]
                        lo, hi = sup["estimate"] + cf[0] - z, sup["estimate"] + cf[1] + z
                    else:
                        lo, hi = r[sp]["lo"], r[sp]["hi"]
                    ax.plot([k + dx] * 2, [lo, hi], color=c, lw=3, ls=ls)
                    ax.plot([k + dx] * 2, [lo, hi], ls="", marker=mk, ms=9, mfc=c if fill else "white", mec=c, mew=2)
                    allv += [lo, hi]; lows.append(lo)
                    if sp == "dr":
                        wdr = hi - lo
                ax.text(k, min(lows) - 0.45, f"w {wdr:.1f}", ha="center", va="top", fontsize=9.5, color=c)
            for sp, (nm, col) in EST.items():
                ax.axhline(s["supported"][sp]["estimate"], color=col, ls="--", lw=2)
            ax.axhline(Vp, color="black", ls="--", lw=2); ax.axhline(V, color="black", ls=(0, (1, 1.6)), lw=2)
            ax.set_xticks(range(len(ITEMS))); ax.set_xticklabels([t[0] for t in ITEMS], fontsize=9.5)
            ax.set_xlim(-0.6, len(ITEMS) - 0.4); ax.grid(axis="y", color="#ddd", lw=0.6)
            ax.set_title(f"ε = ρ = η = {rho}" + ("  (ε_max)" if rho == 7.47 else "") + ("  — no recentring" if tag == "norecentre" else f"  — levels recentred (shift {s['recentre_delta']:+.2f})"), fontsize=12.5)
    pad = 0.08 * (max(allv) - min(allv))
    axes[0, 0].set_ylim(min(allv) - pad, max(allv) + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel("Policy value", fontsize=12.5)
    hs = [Line2D([0], [0], color=col, ls="--", lw=2, label=f"{nm} on supported part V⁺ = {s['supported'][sp]['estimate']:.2f}") for sp, (nm, col) in EST.items()]
    hs += [Line2D([0], [0], color="black", ls="--", lw=2, label=f"True supported part V⁺ = {Vp:.2f}"), Line2D([0], [0], color="black", ls=(0, (1, 1.6)), lw=2, label=f"True full value V = {V:.2f}"),
           Line2D([0], [0], color="gray", lw=3, marker="o", ms=9, label="DR + LP bound"), Line2D([0], [0], color="gray", lw=3, ls=(0, (2.2, 1.6)), marker="o", ms=9, mfc="white", mew=2, label="IPS + LP bound"),
           Line2D([0], [0], color="gray", lw=3, ls=(0, (0.8, 1.0)), marker="D", ms=8, label="OffCEM + LP bound")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.115))
    fig.suptitle("B3 (Lipschitz + coupling), B5 (group mean bounds), B7 (favourite-action dominance) on the plot-04 setup\nK-means (K = 10), original exclusions, λ = ∞, seed 7000, by-round folds, η = ρ", fontsize=13, y=0.99)
    b5 = s["b5"]; b7 = s["b7"]
    note = ("One world; same world, model (m̂ cross-fitted by round) and supported estimates as plots 04 and 09. 'w' = width of the DR bar. Bars = supported estimate + LP, widened by 1.96 SE; dashed lines = supported part V⁺ only. "
            "First x = plot-04 bar (per-round closed form); second x = same beliefs with μ merged to one variable per (user, action). "
            f"B3: |μ_ua − μ_va| ≤ 3·d(x_u, x_v) with logged reference boxes (mean ± 1.96 σ̂/√n) + unsupported-to-unsupported coupling. "
            f"B5: L_c ≤ μ_ua ≤ U_c per K-means cluster from the cluster's logged (user, action) boxes: 'range' = [min low, max high] (true means outside: {100 * b5['B5_range']['true_mean_outside_share_mass']:.0f}% of LP mass), "
            f"'5–95 %' = [5th pct of lows, 95th pct of highs] (outside: {100 * b5['B5_q5']['true_mean_outside_share_mass']:.0f}%). "
            f"B7: μ_(u,a*) ≥ μ_(u,a) for all a, a* = argmax π(·|x_u) (the target's favourite; a* is an LP variable for {b7['B7_ord']['users_star_is_variable']} of {s['n_users']} users); 'ordering' = LP variables only, "
            f"'+ logged' also μ_(u,a*) ≥ ȳ_ua − 3.5 σ̂/√n for the user's logged actions (z = 3.5 ≈ multiplicity-corrected over ~10 logged actions; z = 1.96 was infeasible: the model under-predicts a* for two users by more than ρ). "
            "The combined bar uses z = 3.5 for the B3 boxes as well (with 1.96 one user's a* box contradicts the ordering). Recentring: all identified cell levels ĝ shifted by one constant so their policy-weighted sum equals "
            f"the logged cluster-weighted residual π_U(c|x)/π0(c|x)·(y − m̂) (OffCEM-style; data only); on this world the shift is {s['recentre_delta']:+.2f}, i.e. downward. "
            "Oracle check of B7 ordering: 0 of the dominance rows are violated by the true means. Displayed ε = ρ = stored ε_LC / 2. Source: b5_b7_seed7000_roundfolds.json.")
    fig.text(0.012, 0.006, "\n".join(textwrap.fill(p, 215) for p in note.split("\n")), fontsize=8.3, color="#555", va="bottom")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.925, bottom=0.235, wspace=0.05, hspace=0.32)
    stem = os.path.join(BASE, "slides_figs", "10_lamInf_kmeans_B3-B5-B7_earlierSetup_byRoundFolds_etaRho_recentre_eps5.85-7.47")
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); fig.savefig(stem + "_hires.png", dpi=250)
    print("wrote", stem)


if __name__ == "__main__":
    main()
