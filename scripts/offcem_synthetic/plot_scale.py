#!/usr/bin/env python3
"""Plots for scale_experiment.py: per clustering a plot-04-style eps sweep and a plot-10-style belief panel (eps_max and one level below), plus one combined belief figure (rows = clusterings).

    cd "Test notebooks/scripts/offcem_synthetic" && conda run -n offcem python plot_scale.py --tag offcem_default [--seed 7000]
    -> ../../slides_figs/scale/<tag>_seed<seed>_<clustering>_epsSweep.{png,pdf}, ..._beliefs.{png,pdf}, ..._beliefs_all.{png,pdf}
"""
import argparse
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
STY = ((-0.24, "dr", "o", True, "-"), (0.0, "ips", "o", False, (0, (2.2, 1.6))), (0.24, "offcem", "D", True, (0, (0.8, 1.0))))
NAMES = {"kmeans": "K-means", "agglomerative": "Agglomerative (Ward)", "hierarchical": "Agglomerative (Ward)", "spectral": "Spectral", "dgp": "DGP clusters", "random": "Paper's random clusters"}


def world_line(w):
    if w.get("dataset"):
        return f"{w['dataset']} ({'paper logger' if w.get('logger', 'paper') == 'paper' else 'softmax(β·q) logger'}, β = {w['beta']:g}, target ε = {w['eps_policy']:g}), weak overlap λ = {w['lam']:g}"
    return "original exclusions, λ = " + ("∞" if w.get("lam") is None else f"{w['lam']:g}")


def legend_handles(r):
    hs = [Line2D([0], [0], color=col, ls="--", lw=2, label=f"{nm} on supported part V⁺ = {r['supported'][sp]['estimate']:.2f}") for sp, (nm, col) in EST.items()]
    hs += [Line2D([0], [0], color="black", ls="--", lw=2, label=f"True supported part V⁺ = {r['V_plus_true']:.2f}"), Line2D([0], [0], color="black", ls=(0, (1, 1.6)), lw=2, label=f"True full value V = {r['V']:.2f}"),
           Line2D([0], [0], color="gray", lw=3, marker="o", ms=9, label="DR + LP bound"), Line2D([0], [0], color="gray", lw=3, ls=(0, (2.2, 1.6)), marker="o", ms=9, mfc="white", mew=2, label="IPS + LP bound"),
           Line2D([0], [0], color="gray", lw=3, ls=(0, (0.8, 1.0)), marker="D", ms=8, label="OffCEM + LP bound")]
    return hs


def lines(ax, r):
    for sp, (nm, col) in EST.items():
        ax.axhline(r["supported"][sp]["estimate"], color=col, ls="--", lw=2)
    ax.axhline(r["V_plus_true"], color="black", ls="--", lw=2); ax.axhline(r["V"], color="black", ls=(0, (1, 1.6)), lw=2)
    ax.grid(axis="y", color="#ddd", lw=0.6)


def sweep_plot(d, method, stem):
    r = d["clusterings"][method]; w = d["world"]; grid = r["grid"]
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(grid)))
    fig, ax = plt.subplots(figsize=(9.2, 7.2)); vals = [r["V"], r["V_plus_true"]]
    for k, (g, c) in enumerate(zip(grid, cols)):
        for dx, sp, mk, fill, ls in STY:
            x = r["sweep"][sp][k]
            if not x["ok"] or x["lo"] is None:
                continue
            ax.plot([k + dx] * 2, [x["lo"], x["hi"]], color=c, lw=3, ls=ls); ax.plot([k + dx] * 2, [x["lo"], x["hi"]], ls="", marker=mk, ms=9, mfc=c if fill else "white", mec=c, mew=2)
            vals += [x["lo"], x["hi"]]
    lines(ax, r); vals += [v["estimate"] for v in r["supported"].values()]
    pad = 0.06 * (max(vals) - min(vals)); ax.set_ylim(min(vals) - pad, max(vals) + pad)
    ax.set_xticks(range(len(grid))); ax.set_xticklabels([f"ε = {g:.2f}" + ("\n(ε_max)" if k == len(grid) - 1 else "") for k, g in enumerate(grid)], fontsize=12); ax.set_xlim(-0.6, len(grid) - 0.4)
    ax.set_ylabel("Policy value", fontsize=13)
    ax.set_title(f"{NAMES.get(method, method)} (C = {r['n_clusters']}, ARI vs {'paper random' if w.get('dataset') else 'DGP'} {r['ari_vs_dgp']:.2f}), {world_line(w)}: ε swept over [{w.get('eps_min', 1.0):g}, ε_max = {r['eps_max']:.2f}]\n"
                 f"{w['K']} actions, {w['n']} rounds, {w['n_users']} users, seed {w['seed']}", fontsize=12)
    ax.legend(handles=legend_handles(r), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=10.5)
    cal = r["calibration"]
    note = (f"One world. ε_max = Z·τ̂ with Z = 2, τ̂ = {cal['tau']:.2f} from {cal['n_pairs']} within-(user, cluster) logged contrasts, noise σ̂ = {cal['sigma_noise']:.2f} ({'taken as known' if cal.get('sigma_source') == 'given' else str(cal['n_repeat_dof']) + ' repeat dof'}). "
            f"Bars = supported estimate + closed-form LP interval, widened by 1.96 SE; m̂ = {'OffCEM two-step f̂' if w.get('model') == 'two_step' else 'cluster-level ridge'} cross-fitted by {w.get('folds', 'round')}, η = ρ, ridge centre, no recentring, range [-30, 30]; unsupported mass {r['lp_mass']:.2f}. "
            f"Dashed lines = supported part V⁺ only. Displayed ε = stored ε_LC / 2. Wall clock for this clustering: {r.get('elapsed_seconds', 0) / 60:.1f} min.")
    fig.text(0.02, 0.008, "\n".join(textwrap.fill(p, 110) for p in note.split("\n")), fontsize=8.5, color="#555", va="bottom")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.89, bottom=0.34)
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); plt.close(fig)


def lp_only_sweep_plot(d, methods, stem):
    """Unsupported part only: closed-form LP interval for V0 per eps, against the true V0. No supported estimates, no SE widening."""
    methods = [m for m in methods if "sweep" in d["clusterings"].get(m, {})]
    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 6.4), sharey=True, squeeze=False); axes = axes[0]
    w = d["world"]; vals = []
    for ax, m in zip(axes, methods):
        r = d["clusterings"][m]; grid = r["grid"]; cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(grid)))
        for k, (g, c) in enumerate(zip(grid, cols)):
            x = r["sweep"]["dr"][k]
            if not x["ok"] or x["lp_lo"] is None:
                ax.text(k, r["V0_true"], "infeasible", rotation=90, ha="center", va="center", fontsize=9, color=c); continue
            lo, hi = x["lp_lo"], x["lp_hi"]
            ax.plot([k, k], [lo, hi], color=c, lw=4); ax.plot([k, k], [lo, hi], ls="", marker="o", ms=9, color=c)
            ax.annotate(f"w {hi - lo:.2f}" if hi - lo < 2 else f"w {hi - lo:.1f}", (k, lo), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=9.5, color=c); vals += [lo, hi]
        ax.axhline(r["V0_true"], color="black", ls=(0, (1, 1.6)), lw=2); vals.append(r["V0_true"])
        ax.set_xticks(range(len(grid))); ax.set_xticklabels([f"ε = {g:.2f}" + ("\n(ε_max)" if k == len(grid) - 1 else "") for k, g in enumerate(grid)], fontsize=10.5)
        ax.set_xlim(-0.6, len(grid) - 0.4); ax.grid(axis="y", color="#ddd", lw=0.6)
        ax.set_title(f"{NAMES.get(m, m)} (C = {r['n_clusters']}, ARI vs {'paper random' if w.get('dataset') else 'DGP'} {r['ari_vs_dgp']:.2f})\nε_max = {r['eps_max']:.2f}, unsupported mass {r['lp_mass']:.2f}", fontsize=12)
    pad = 0.08 * (max(vals) - min(vals)); axes[0].set_ylim(min(vals) - pad, max(vals) + pad); axes[0].set_ylabel("Unsupported part V⁰ (policy value on excluded actions)", fontsize=12)
    r0 = d["clusterings"][methods[0]]
    hs = [Line2D([0], [0], color="black", ls=(0, (1, 1.6)), lw=2, label=f"True unsupported part V⁰ = {r0['V0_true']:.2f}"), Line2D([0], [0], color="gray", lw=4, marker="o", ms=9, label="LP bounds on V⁰ (min / max of the LP)")]
    fig.legend(handles=hs, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.10))
    fig.suptitle(f"LP bounds on the unsupported part only — {w['K']} actions, {w['n']} rounds, {w['n_users']} users, seed {w['seed']}; {world_line(w)}", fontsize=13, y=0.985)
    note = (f"Beliefs: outcome range {w.get('limits', [-30, 30])}; relaxed local correctness |μ − m̂ − b| ≤ ε around the cluster-level model m̂ ({'OffCEM two-step f̂' if w.get('model') == 'two_step' else 'ridge'}, cross-fitted by {w.get('folds', 'round')}); level band |b − ĝ| ≤ η with η = ε. "
            "ε_max = 2·τ̂ from within-(user, cluster) logged contrasts, logged data only. No supported-region estimate and no sampling-error widening enter these bars: they are the LP optimum values. Displayed ε = ρ = stored ε_LC / 2.")
    fig.text(0.012, 0.006, "\n".join(textwrap.fill(q, 190) for q in note.split("\n")), fontsize=8.5, color="#555", va="bottom")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.26, wspace=0.06)
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); plt.close(fig)


def belief_items(r):
    z = r["beliefs_info"]["z"]; L = r["beliefs_info"]["L"]; rz = r["beliefs_info"].get("ref_z")
    items = [("plot-04 bar\n(closed form)", "closed"), ("baseline\n(μ merged)", "baseline"), (f"+ B3\nL = {L:.2f}" + (f"\n(z = {rz:g})" if rz else ""), "B3")]
    if not rz:
        items.append((f"+ B3\nz = {z:g}", f"B3_z{z:g}"))
    items += [("+ B5\nrange", "B5_range"), ("+ B5\n5–95 %", "B5_q5"), ("+ B7\nordering", "B7_ord"), (f"+ B7\n+ logged\n(z = {z:g})", f"B7_z{z:g}"), (f"+ B5 5–95 %\n+ B7 z = {z:g}", f"B5_q5+B7_z{z:g}")]
    items.append((f"+ B3 + B5 5–95 %\n+ B7 z = {z:g}" if rz else f"+ B3 z = {z:g}\n+ B5 5–95 %\n+ B7 z = {z:g}", f"B3z+B5_q5+B7_z{z:g}"))
    return items


def belief_axes(ax, r, rho, items, cols, allv):
    k0 = r["grid"].index(rho)
    for k, (lab, key) in enumerate(items):
        c = cols[k]; lows = []
        rr = r["runs"].get(f"{rho}|{key}")
        if key != "closed" and (rr is None or rr.get("lp_lo") is None):
            ax.text(k, np.mean(allv), "infeasible" if rr else "not run", rotation=90, ha="center", va="center", fontsize=9.5, color=c); continue
        for dx, sp, mk, fill, ls in STY:
            if key == "closed":
                x = r["sweep"][sp][k0]; lo, hi = x["lo"], x["hi"]
            else:
                lo, hi = rr[sp]["lo"], rr[sp]["hi"]
            ax.plot([k + dx] * 2, [lo, hi], color=c, lw=3, ls=ls); ax.plot([k + dx] * 2, [lo, hi], ls="", marker=mk, ms=8, mfc=c if fill else "white", mec=c, mew=2)
            allv += [lo, hi]; lows.append(lo)
            if sp == "dr":
                wdr = hi - lo
        ax.annotate(f"w {wdr:.2f}" if wdr < 2 else f"w {wdr:.1f}", (k, min(lows)), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=9, color=c)
    lines(ax, r)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([t[0] for t in items], fontsize=8); ax.set_xlim(-0.6, len(items) - 0.4)


def belief_note(d, r):
    bi = r["beliefs_info"]; w = d["world"]
    return (f"One world ({w['K']} actions, {w['n']} rounds, {w['n_users']} users, {w['dgp_clusters']} DGP clusters, seed {w['seed']}); original exclusions, λ = ∞, m̂ = {'OffCEM two-step f̂ (ĝ + ĥ)' if w.get('model') == 'two_step' else 'cluster-level ridge'} cross-fitted by {w.get('folds', 'round')}, η = ρ, no recentring. "
            "'w' = width of the DR bar. Bars = supported estimate + LP, widened by 1.96 SE; dashed lines = supported part V⁺ only. First x = closed form (plot-04 bar); second = same beliefs with μ merged per (user, action). "
            f"B3: |μ_ua − μ_va| ≤ {bi['L']:g}·d(x_u, x_v), logged reference boxes (mean ± z σ̂/√n, z = 1.96 or {bi['z']:g}) + coupling; median {bi['b3_refs_per_var_median']:.0f} references per variable, "
            f"{100 * bi['b3_share_vars_with_ref']:.0f}% of variables have one; true means outside the boxes: {100 * bi['b3_true_outside_mass']:.0f}% (z = 1.96) / {100 * bi['b3z_true_outside_mass']:.0f}% (z = {bi['z']:g}) of LP mass. "
            f"B5: per-cluster [L_c, U_c] from the cluster's logged boxes, range (true outside {100 * bi['b5']['range']['true_outside_mass']:.0f}%) or 5–95 % (outside {100 * bi['b5']['q5']['true_outside_mass']:.0f}%). "
            f"B7: μ_(u,a*) ≥ μ_(u,a), a* = target's favourite (an LP variable for {bi['b7']['B7_ord']['users_star_is_variable']} of {w['n_users']} users); 'ordering' = LP variables only, '+ logged' adds μ_(u,a*) ≥ ȳ_ua − z σ̂/√n; "
            f"oracle: {100 * bi['b7']['B7_ord']['true_violation_share_rows']:.1f}% of ordering rows violated by true means. Displayed ε = stored ε_LC / 2.")


def belief_plot(d, method, stem):
    r = d["clusterings"][method]; items = belief_items(r); cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(items)))
    rhos = r["grid"][-2:]
    fig, axes = plt.subplots(1, len(rhos), figsize=(9 * len(rhos), 8), sharey=True); axes = np.atleast_1d(axes)
    allv = [r["V"], r["V_plus_true"]] + [v["estimate"] for v in r["supported"].values()]
    for ax, rho in zip(axes, rhos):
        belief_axes(ax, r, rho, items, cols, allv)
        ax.set_title(f"ε = ρ = η = {rho:.2f}" + ("  (ε_max)" if rho == r["grid"][-1] else "  (one level below ε_max)"), fontsize=12.5)
    pad = 0.08 * (max(allv) - min(allv)); axes[0].set_ylim(min(allv) - pad, max(allv) + pad); axes[0].set_ylabel("Policy value", fontsize=12.5)
    fig.legend(handles=legend_handles(r), loc="lower center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.13))
    fig.suptitle(f"Beliefs B3 / B5 / B7 — {NAMES[method]} (C = {r['n_clusters']}, ARI vs DGP {r['ari_vs_dgp']:.2f}), wall clock {r.get('elapsed_seconds', 0) / 60:.0f} min", fontsize=13.5, y=0.985)
    fig.text(0.012, 0.006, "\n".join(textwrap.fill(p, 200) for p in belief_note(d, r).split("\n")), fontsize=8.3, color="#555", va="bottom")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.30, wspace=0.05)
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); plt.close(fig)


def belief_plot_all(d, methods, stem):
    methods = [m for m in methods if "beliefs_info" in d["clusterings"].get(m, {})]
    fig, axes = plt.subplots(len(methods), 2, figsize=(18, 5.6 * len(methods) + 2.5), sharey="row", squeeze=False)
    for i, m in enumerate(methods):
        r = d["clusterings"][m]; items = belief_items(r); cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(items)))
        allv = [r["V"], r["V_plus_true"]] + [v["estimate"] for v in r["supported"].values()]
        for ax, rho in zip(axes[i], r["grid"][-2:]):
            belief_axes(ax, r, rho, items, cols, allv)
            ax.set_title(f"{NAMES[m]} (ARI {r['ari_vs_dgp']:.2f}, {r.get('elapsed_seconds', 0) / 60:.0f} min) — ε = {rho:.2f}" + ("  (ε_max)" if rho == r["grid"][-1] else ""), fontsize=11.5)
        pad = 0.08 * (max(allv) - min(allv)); axes[i, 0].set_ylim(min(allv) - pad, max(allv) + pad); axes[i, 0].set_ylabel("Policy value", fontsize=12)
    r0 = d["clusterings"][methods[0]]
    fig.legend(handles=legend_handles(r0)[3:], loc="lower center", ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.012))
    w = d["world"]
    fig.suptitle(f"Beliefs at ε_max and one level below, per estimation clustering — {w['K']} actions, {w['n']} rounds, {w['n_users']} users, seed {w['seed']}; {world_line(w)}", fontsize=13.5, y=0.995)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.955, bottom=0.10, wspace=0.05, hspace=0.5)
    fig.savefig(stem + ".png", dpi=100); fig.savefig(stem + ".pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="offcem_default"); ap.add_argument("--seed", type=int, default=7000); a = ap.parse_args()
    # one file per run, or several per-clustering files sharing the tag (run_scale_gcp.sh writes scale_<tag>_<clustering>_seed<seed>.json): merge their 'clusterings'
    import glob
    files = sorted(glob.glob(os.path.join(BASE, "results_offcem_global", f"scale_{a.tag}_seed{a.seed}.json")) + glob.glob(os.path.join(BASE, "results_offcem_global", f"scale_{a.tag}_*_seed{a.seed}.json")))
    if not files:
        raise SystemExit(f"no results for tag {a.tag} seed {a.seed}")
    d = {"clusterings": {}}
    for f in files:
        dd = json.load(open(f)); d["world"] = dd["world"]; d["clusterings"].update(dd["clusterings"])
    print("merged", [os.path.basename(f) for f in files])
    od = os.path.join(BASE, "slides_figs", "scale"); os.makedirs(od, exist_ok=True)
    pre = os.path.join(od, f"{a.tag}_seed{a.seed}")
    for m, r in d["clusterings"].items():
        if "sweep" in r:
            sweep_plot(d, m, f"{pre}_{m}_epsSweep")
        if "beliefs_info" in r:
            belief_plot(d, m, f"{pre}_{m}_beliefs")
    belief_plot_all(d, list(d["clusterings"]), f"{pre}_beliefs_all")
    lp_only_sweep_plot(d, list(d["clusterings"]), f"{pre}_lpOnly_epsSweep")
    print("wrote plots under", od)


if __name__ == "__main__":
    main()
