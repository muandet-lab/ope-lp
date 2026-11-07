#!/usr/bin/env python3
"""eta = rho vs eta = eta_hat (7.38), baseline and baseline + B3 (references + coupling) at rho = 5.85 and 7.47.
K-means clusters, original exclusions, lambda = inf, seed 7000, user-level cross-fitting.

    cd "Test notebooks/scripts/offcem_synthetic" && conda run -n offcem python plot_b3_eta_rho.py
    -> ../../slides_figs/08_lamInf_kmeans_B3coupling_etaRho-vs-etaHat_eps5.85-7.47.{png,pdf,_hires.png}

Inputs: results_offcem_global/b3_lipschitz_seed7000.json (eta_hat) and b3_lipschitz_seed7000_eta_rho.json (eta = rho, from b3_lipschitz_experiment.py --eta rho).
"""
import json
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EST = {"ips": ("IPS", "#1f77b4"), "dr": ("DR", "#17becf"), "offcem": ("OffCEM", "#e377c2")}
SETS = [("hat", "b3_lipschitz_seed7000.json", "#7f7f7f", -0.21), ("rho", "b3_lipschitz_seed7000_eta_rho.json", "#2ca02c", 0.21)]


def main():
    D = {k: json.load(open(os.path.join(BASE, "results_offcem_global", f))) for k, f, _, _ in SETS}
    s = D["hat"]["setup"]; V, Vp = s["V"], s["V_plus_true"]
    rhos, Ls = [5.85, 7.47], (3.0, 3.5, 4.4)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharey=True)
    allv = [V, Vp] + [v["estimate"] for v in s["supported"].values()]
    for ax, rho in zip(axes, rhos):
        items = [("baseline", None, "baseline")] + [("coupling", L, f"+ B3 + coupling\nL = {L:g}") for L in Ls]
        for k, (mode, L, lab) in enumerate(items):
            for key, _, col, off in SETS:
                r = D[key]["runs"].get(f"{rho}|baseline" if mode == "baseline" else f"{rho}|{L}|{mode}")
                if r is None or "dr" not in r:
                    ax.text(k + off, 0.5 * (min(allv) + max(allv)), "not run", ha="center", fontsize=9, color=col); continue
                for dx, sp, mk, fill, ls in ((-0.12, "dr", "o", True, "-"), (0.0, "ips", "o", False, (0, (2.2, 1.6))), (0.12, "offcem", "D", True, (0, (0.8, 1.0)))):
                    lo, hi = r[sp]["lo"], r[sp]["hi"]
                    ax.plot([k + off + dx] * 2, [lo, hi], color=col, lw=2.6, ls=ls)
                    ax.plot([k + off + dx] * 2, [lo, hi], ls="", marker=mk, ms=8, mfc=col if fill else "white", mec=col, mew=2)
                    allv += [lo, hi]
                w = r["dr"]["hi"] - r["dr"]["lo"]; bot = min(r[sp]["lo"] for sp in EST)
                ax.text(k + off, bot - 0.45, f"w {w:.1f}", ha="center", va="top", fontsize=9, color=col)
        for sp, (nm, col) in EST.items():
            ax.axhline(s["supported"][sp]["estimate"], color=col, ls="--", lw=1.8)
        ax.axhline(Vp, color="black", ls="--", lw=1.8); ax.axhline(V, color="black", ls=(0, (1, 1.6)), lw=2)
        ax.set_xticks(range(len(items))); ax.set_xticklabels([i[2] for i in items], fontsize=10)
        ax.set_xlim(-0.6, len(items) - 0.4); ax.grid(axis="y", color="#e0e0e0", lw=0.6)
        ax.set_title(f"ε = ρ = {rho}   (grey: η = η̂ = {s['eta_hat']:.2f};  green: η = ρ = {rho})", fontsize=12.5)
    pad = 0.09 * (max(allv) - min(allv))
    axes[0].set_ylim(min(allv) - pad, max(allv) + pad); axes[0].set_ylabel("Policy value", fontsize=13)
    hs = [Line2D([0], [0], color=col, ls="--", lw=2, label=f"{nm} on supported part V⁺ = {s['supported'][sp]['estimate']:.2f}") for sp, (nm, col) in EST.items()]
    hs += [Line2D([0], [0], color="black", ls="--", lw=2, label=f"True supported part V⁺ = {Vp:.2f}"), Line2D([0], [0], color="black", ls=(0, (1, 1.6)), lw=2, label=f"True full value V = {V:.2f}"),
           Line2D([0], [0], color="#7f7f7f", lw=3, label=f"η fixed at η̂ = {s['eta_hat']:.2f} (plot 06)"), Line2D([0], [0], color="#2ca02c", lw=3, label="η = ρ (new run)"),
           Line2D([0], [0], color="gray", lw=3, marker="o", ms=8, label="DR + LP bound"), Line2D([0], [0], color="gray", lw=3, ls=(0, (2.2, 1.6)), marker="o", ms=8, mfc="white", mew=2, label="IPS + LP bound"),
           Line2D([0], [0], color="gray", lw=3, ls=(0, (0.8, 1.0)), marker="D", ms=7, label="OffCEM + LP bound")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.07))
    fig.suptitle("η = ρ vs η = η̂, with and without B3 (Lipschitz across users + coupling) — K-means clusters, original exclusions, λ = ∞, seed 7000", fontsize=14, y=0.985)
    note = ("One world. Same LP as plot 06 (range [-30, 30], LC with ρ, band |b − b̂| ≤ η, m̂ cross-fitted by user); only η differs between grey and green. B3: |μ_{u,a} − μ_{v,a}| ≤ L·d(x_u, x_v) with logged "
            f"reference boxes (mean ± 1.96·σ̂/√n, σ̂ = {s['sigma_hat']:.2f}) plus unsupported-to-unsupported coupling (full pair check passed). 'w' = width of the DR bar. Bars = supported estimate + LP, widened by 1.96 SE; LP mass {s['lp_mass']:.2f}. "
            "Dashed lines = supported part V⁺ only. Displayed ε = ρ = stored ε_LC / 2. Sources: b3_lipschitz_seed7000.json, b3_lipschitz_seed7000_eta_rho.json.")
    fig.text(0.012, 0.006, "\n".join(textwrap.fill(p, 200) for p in note.split("\n")), fontsize=8.5, color="#555", va="bottom")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.25, wspace=0.05)
    stem = os.path.join(BASE, "slides_figs", "08_lamInf_kmeans_B3coupling_etaRho-vs-etaHat_eps5.85-7.47")
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); fig.savefig(stem + "_hires.png", dpi=250)
    print("wrote", stem)


if __name__ == "__main__":
    main()
