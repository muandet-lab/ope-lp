# Plot index (slides_figs)

Every plot has three files: `.png`, `.pdf`, `_hires.png`. All are ONE world (seed 7000) unless noted. Displayed epsilon = rho = stored eps_LC / 2.
Name pattern: `NN_<lambda>_<clusters>_<epsilon values>_<exclusions>_<what>`.
"origExcl" = original exclusions (redrawn every round); "fixedExcl" = one excluded set per user. "earlierSetup" = before user-level cross-fitting and the separate eta.
"userCV" = model cross-fitted with folds by user. "etaFixed" = eta held at the logged-data estimate (7.38 for K-means, 8.23 for DGP clusters).

| File (without extension) | What it is | Old name |
|---|---|---|
| 01_noClusters_allPairsJointLP_eps1-4_fixedExcl | no-cluster test, all-pairs joint LP: C = 1 world (left) and clusters ignored (right) | no_clusters_two_panel |
| 02_lamInf_dgpClusters_eps1-4_origExcl_DR-IPS-OffCEM_Vplus-lines | lambda = inf, DGP clusters, eps 1-4: DR / IPS / OffCEM + LP bars, dashed lines = supported part V+ only | unsupported_lam_inf_original |
| 03_lamInf_dgpClusters_epsSweep1-5.33_origExcl_earlierSetup | lambda = inf, DGP clusters, eps swept 1 to estimated eps_max (5.33); by-round folds, eta = rho | eps_sweep_dgp_round_random_seed7000 |
| 04_lamInf_kmeans_epsSweep1-7.47_origExcl_earlierSetup | lambda = inf, K-means clusters, eps swept 1 to 7.47; earlier setup | eps_sweep_dgp_round_random_seed7000_learned |
| 05_lamInf_kmeans_BASELINE_origExcl_userCV_etaFixed | K-means baseline after the fixes (user-level cross-fitting, eta fixed 7.38), ten eps values; this is the baseline for B3 | baseline_learned_laminf_seed7000 |
| 06_lamInf_kmeans_BASELINE-vs-B3_eps5.85-7.47 | K-means: baseline vs baseline + B3 (Lipschitz across users, same action) at L = 3.0, 3.5, 4.4; eps 5.85 and 7.47 | b3_vs_baseline_rho585_747_seed7000 |
| 07_lamInf_kmeans_B3_L3-4.4_refs-vs-coupling_eps5.85-6.27-7.47 | larger B3 plot: reference boxes alone vs plus coupling, three eps values | b3_lipschitz_kmeans_laminf_seed7000 |

`three_plots/` holds three copies of eps 1-4 DGP-cluster plots (same names without the number prefix):
`lamInf_..._Vplus-lines` (= 02), `lam10_..._DR-IPS_FULLvalue-lines` (the original two-panel plot, DR and IPS bars only; old name `unsupported_two_panel`),
`lam10_..._DR-IPS-OffCEM_Vplus-lines` (old name `unsupported_lam10_vplus_two_panel`).

Note: the scripts that generate these plots still write the OLD names (`make_standard_figures.py`, `eps_sweep_dgp.py`, `plot_baseline_bars.py`, `plot_eta_separation.py`, `plot_b3.py`, `plot_b3_simple.py`). Re-running one recreates the old file name next to the renamed one.


All lambda = 10 plots were moved to `lam10/` (its own numbering, see `lam10/README_plots.md`). The two lambda = 10 copies in `three_plots/` were left where they are.

| 08_lamInf_kmeans_B3coupling_etaRho-vs-etaHat_eps5.85-7.47 | K-means, λ = ∞, seed 7000: η = ρ (green) vs η fixed at η̂ = 7.38 (grey), baseline and + B3 refs + coupling at L = 3, 3.5, 4.4; eps 5.85 and 7.47. Script `plot_b3_eta_rho.py`; data `b3_lipschitz_seed7000_eta_rho.json` (from `b3_lipschitz_experiment.py --eta rho --out ...`) | (new, 2026-11-07) |
| 09_lamInf_kmeans_B3coupling_earlierSetup_byRoundFolds_etaRho_eps5.85-7.47 | B3 refs + coupling on the plot-04 setup (K-means, λ = ∞, by-round folds, η = ρ), eps 5.85 and 7.47; first bar reproduces plot 04. Script `plot_b3_round_folds.py`; data `b3_lipschitz_seed7000_eta_rho_roundfolds.json` (`b3_lipschitz_experiment.py --eta rho --folds round --out ...`) | (new, 2026-11-07) |
| 10_lamInf_kmeans_B3-B5-B7_earlierSetup_byRoundFolds_etaRho_recentre_eps5.85-7.47 | Plot-04 setup (K-means, λ = ∞, by-round folds, η = ρ), eps 5.85 and 7.47, 2 × 2: rows = no recentring / levels recentred (importance-weighted residual, shift −0.42 on this world); x = plot-04 bar, merged baseline, + B3 (L = 3, coupled), + B5 group mean bounds (range / 5–95 %), + B7 favourite-action dominance (ordering only / + logged lower bounds z = 3.5), + all three. Viridis per belief set (plot-03 scheme). Script `plot_b5_b7.py`; data `b5_b7_seed7000_roundfolds.json` (`b5_b7_experiment.py`) | (new, 2026-11-07) |
