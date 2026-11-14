#!/usr/bin/env python3
"""Robustness table over worlds / seeds (run_multiseed.sh). Per arm, constraint set and eps level (synthetic: floor(eps_max) x {1/2, 3/4, 1}; real: eps_max x {1/2, 3/4, 1}):
  coverage   k / worlds whose OffCEM bar (supported estimate + LP, +- 1.96 SE) contains V, among worlds where the set is feasible as stated (hard)
  width      mean +- sd of the bar width (hard-feasible worlds), and of the width relative to the same world's base (13)-(15) bar
  infeasible worlds where the set is infeasible as stated (no slack is used)
Synthetic arms also: the S1 / S2 sweep (base at eps = 1, 2, 3, 4): coverage and width per eps.
Rows: (13) range only, (13)+(14) (level band off), base (13)-(15), then the beliefs. Relative bars (lo - V, hi - V) are written to the JSON for the figure.

    python summarize_multiseed.py [--dir ../../results_offcem_global/multiseed] [--prefix ms]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ARMS = {"sw": (("dgp", "kmeans"), ["range", "noband", "baseline", "B3c", "PB_sev", "PB_sev+ext", "B3c+PB_sev", "B3c+PB_sev+ext"]),
        "full": (("dgp", "kmeans"), ["range", "noband", "baseline", "B3c", "PB_sev", "PB_sev+ext", "B3c+PB_sev", "B3c+PB_sev+ext"]),
        "eurlex": (("kmeans",), ["range", "noband", "baseline", "B3c", "PB_fail0.4", "PB_fail0.3", "B3c+PB_fail0.4", "B3c+PB_fail0.3"]),
        "wiki": (("kmeans",), ["range", "noband", "baseline", "B3c", "PB_fail0.4", "PB_fail0.3", "B3c+PB_fail0.4", "B3c+PB_fail0.3"])}
LABEL = {"range": "(13) range", "noband": "+ (14), band off", "baseline": "base (13)-(15)", "B3c": "+ Lipschitz", "PB_sev": "+ severe rare", "PB_sev+ext": "+ severe + extreme rare",
         "B3c+PB_sev": "+ Lip + severe", "B3c+PB_sev+ext": "+ Lip + severe + extreme", "PB_fail0.4": "+ cap P(Y=0)<=0.4", "PB_fail0.3": "+ cap P(Y=0)<=0.3",
         "B3c+PB_fail0.4": "+ Lip + cap 0.4", "B3c+PB_fail0.3": "+ Lip + cap 0.3"}
FRAC = ["1/2", "3/4", "1"]
LEVEL = {"sw": "floor(eps_max)", "full": "floor(eps_max)", "eurlex": "eps_max", "wiki": "eps_max"}


def load(d, prefix, arm, kind):
    out = {}
    for f in glob.glob(os.path.join(d, f"scale_{prefix}_{arm}_{kind}_s*_seed*.json")):
        s = int(re.search(r"_s(\d+)_seed", f).group(1)); out[s] = json.load(open(f))["clusterings"]
    return out


def cell(world_runs, item, pos):
    """(bar dict or None, relaxed t or None, eps) for one world; item from main / noband / range runs."""
    kind, r = world_runs[item]
    if r is None:
        return None
    grid = r["grid"]; e = grid[0] if kind == "range" else (grid[pos] if pos < len(grid) else None)
    if e is None:
        return None
    rr = r["runs"].get(f"{e}|{'baseline' if kind != 'main' else item}")
    if rr is None:
        return None
    t = rr.get("min_feasible_t") if rr.get("t") else None
    return rr.get("offcem"), t, e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results_offcem_global", "multiseed"))
    ap.add_argument("--prefix", default="ms"); a = ap.parse_args()
    summary = {}
    for arm, (clus, rows) in ARMS.items():
        main_, nob, rng = load(a.dir, a.prefix, arm, "main"), load(a.dir, a.prefix, arm, "noband"), load(a.dir, a.prefix, arm, "range")
        if not main_:
            continue
        for cl in clus:
            seeds = sorted(s for s in main_ if cl in main_[s])
            if not seeds:
                continue
            key = f"{arm}|{cl}"; summary[key] = {}
            print(f"\n=== {arm} / {cl}: {len(seeds)} worlds {seeds}   V per world: " + ", ".join(f"{main_[s][cl]['V']:.2f}" for s in seeds)
                  + "   eps_max: " + ", ".join(f"{main_[s][cl]['eps_max']:.2f}" for s in seeds))
            for pos in range(3):
                print(f"  eps = {FRAC[pos]} x {LEVEL[arm]}" + " " * 10 + "coverage   width mean +- sd    rel. to base     infeasible")
                for item in rows:
                    per, rel = [], []
                    for s in seeds:
                        wr = {"range": ("range", rng.get(s, {}).get(cl)), "noband": ("noband", nob.get(s, {}).get(cl))}
                        wr.update({k: ("main", main_[s][cl]) for k in rows if k not in ("range", "noband")})
                        c = cell(wr, item, pos); b = cell(wr, "baseline", pos)
                        if c is None:
                            continue
                        bar, t, e = c; V = main_[s][cl]["V"]
                        hard = bar is not None and bar.get("lo") is not None and t is None
                        rec = dict(seed=s, eps=e, V=V, hard=hard, t=t, lo=None if bar is None else bar.get("lo"), hi=None if bar is None else bar.get("hi"))
                        if bar is not None and bar.get("lo") is not None:
                            rec.update(rel_lo=bar["lo"] - V, rel_hi=bar["hi"] - V, contains=bool(bar["contains_V"]), width=bar["hi"] - bar["lo"])
                            if hard and b is not None and b[0] is not None and b[0].get("lo") is not None:
                                rel.append(rec["width"] / (b[0]["hi"] - b[0]["lo"]))
                        per.append(rec)
                    hd = [p for p in per if p["hard"]]; inf = [p for p in per if not p["hard"]]; ts = [p["t"] for p in inf if p["t"] is not None]
                    w = [p["width"] for p in hd]
                    summary[key].setdefault(FRAC[pos], {})[item] = per
                    print(f"    {LABEL[item]:28s} {sum(p['contains'] for p in hd):>2d}/{len(hd):<2d}     "
                          + (f"{np.mean(w):6.2f} +- {np.std(w, ddof=1) if len(w) > 1 else 0:5.2f}    " if w else " " * 20)
                          + (f"{np.mean(rel):5.2f} +- {np.std(rel, ddof=1) if len(rel) > 1 else 0:4.2f}" if rel else " " * 12)
                          + f"     {len(inf)}/{len(per)}" + (f"        {np.median(ts):g}, {max(ts):g}" if ts else ""))
    for arm in ("sw", "full"):                          # S1 / S2 sweeps across worlds
        sw = load(a.dir, a.prefix, arm, "sweep")
        for cl in ("dgp", "kmeans"):
            seeds = sorted(s for s in sw if cl in sw[s])
            if not seeds:
                continue
            print(f"\n=== {arm} / {cl}: S{'1' if cl == 'dgp' else '2'} sweep, base (13)-(15), {len(seeds)} worlds")
            key = f"{arm}|{cl}|sweep"; summary[key] = {}
            for e in (1.0, 2.0, 3.0, 4.0):
                per = []
                for s in seeds:
                    r = sw[s][cl]; o = (r["runs"].get(f"{e}|baseline") or {}).get("offcem")
                    if o and o.get("lo") is not None:
                        per.append(dict(seed=s, V=r["V"], lo=o["lo"], hi=o["hi"], rel_lo=o["lo"] - r["V"], rel_hi=o["hi"] - r["V"], contains=bool(o["contains_V"]), width=o["hi"] - o["lo"]))
                    else:
                        per.append(dict(seed=s, V=r["V"], lo=None, hi=None))
                ok = [p for p in per if p["lo"] is not None]; w = [p["width"] for p in ok]
                summary[key][f"{e:g}"] = per
                print(f"    eps = {e:g}:  contains V in {sum(p['contains'] for p in ok)}/{len(ok)} worlds   width {np.mean(w):.2f} +- {np.std(w, ddof=1) if len(w) > 1 else 0:.2f}   infeasible {len(per) - len(ok)}")
    out = os.path.join(a.dir, f"summary_{a.prefix}.json"); json.dump(summary, open(out, "w"), indent=1); print("\nwrote", out)


if __name__ == "__main__":
    main()
