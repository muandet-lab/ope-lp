# ope-lp — partial-identification bounds for off-policy evaluation with unsupported actions

Linear-programme bounds on the unsupported part of a policy's value (V = V⁺ + V⁰), on OffCEM-style synthetic worlds, with the beliefs
B3 (Lipschitz across users), B5 (group-specific mean bounds) and B7 (favourite-action dominance). See `docs/` for the full experiment log
(`OFFCEM_JOINT_LP_EXPERIMENTS.md`, sections 7–8 are the B3/B5/B7 work) and the LP specification (`GLOBAL_JOINT_LP_SPEC.md`).

## Layout
```
scripts/offcem_synthetic/   all code (self-contained: world generator dataset.py, policy.py, clustering.py are local copies)
  scale_experiment.py       OffCEM-scale driver: world -> estimation clustering -> eps_max from logs -> eps sweep (closed form) -> beliefs at eps_max and one level below
  plot_scale.py             plots for scale_experiment.py (merges per-clustering result files)
  run_scale_gcp.sh          VM launcher: one process per clustering (kmeans, agglomerative, spectral)
  b3_lipschitz_experiment.py, b5_b7_experiment.py, eps_sweep_dgp.py   small-world (seed 7000) experiments behind plots 04, 09, 10
  plot_b3_round_folds.py, plot_b3_eta_rho.py, plot_b5_b7.py          their plots
  partial_ope_global.py     world, reward models, cell levels, closed-form LP, supported estimators (IPS / DR / OffCEM)
results_offcem_global/      JSON results (small-world results included; scale_*.json are git-ignored, copy them back from the VM)
slides_figs/                plots (README_plots.md is the index; plots 04, 09, 10 included)
docs/                       experiment log, LP spec, setup notes, known issues
```

## Environment
```
conda env create -f environment.yml      # python 3.9, OBP 0.5.5, scipy (HiGHS via scipy.optimize.linprog)
```

## Run
Small-world check (about 1 min):
```
cd scripts/offcem_synthetic
conda run -n offcem python scale_experiment.py --K 200 --n 1000 --users 100 --C 10 --clusterings kmeans agglomerative --tag smoke
conda run -n offcem python plot_scale.py --tag smoke
```
OffCEM default scale (1000 actions, 3000 rounds, 1000 users, 50 clusters), three clusterings in parallel, inside tmux on the VM:
```
cd scripts/offcem_synthetic && bash run_scale_gcp.sh offcem_default 7000
```
Results land in `results_offcem_global/scale_offcem_default_<clustering>_seed7000.json` (saved after every solve; re-running resumes) and
`results_offcem_global/logs/`. Then `python plot_scale.py --tag offcem_default` writes `slides_figs/scale/`.

Flags worth knowing: `--folds round|user` (by-round folds are the plot-04 setup, but the merged LP can be infeasible at scale when fold models disagree;
by-user folds make the merged LP equal to the closed form), `--beliefs <names>` to run a subset, `--L`, `--z`, `--time-limit`, `--npts`.

## Setup conventions
Original exclusions (the logging policy is barred from a random half of the actions, redrawn every round), λ = ∞ (only π0 = 0 pairs go to the LP),
ridge cluster-level reward model cross-fitted by round, η = ρ, outcome range [−30, 30], displayed ε = ρ = stored ε_LC / 2. ε_max is Z·τ̂ (Z = 2) from
within-(user, cluster) logged contrasts. Bars = supported estimate + LP interval ± 1.96 SE. Beliefs are described in the docstrings of
`b5_b7_experiment.py` and `b3_lipschitz_experiment.py`.
