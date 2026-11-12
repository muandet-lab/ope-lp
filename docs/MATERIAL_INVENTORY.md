# Material inventory for the paper (2026-11-12)

Everything we have, by paper section, with file names, what each shows, and what is still missing. Plots are in `Test notebooks/slides_figs/clusters_x_lambda/` (S-series, M-series) or `OffCEM_bias_analysis/images/` (bias analysis); result files are in `ope-lp/results_offcem_global/`; settings are in `docs/FINAL_FORMULATION.md`; the experiments draft is `docs/EXPERIMENTS_DRAFT.md`.

## 1. Motivation: OffCEM's point estimate depends on the clustering

| Claim | Plot | Evidence |
|---|---|---|
| Partition mismatch gives a bias floor that does not decay with n | M1 (`M1_offcem_bias2_vs_n_dgp-wfss-kmeans_DR-DM`), M2 (relative MSE), M3 (estimate vs n with V) | DGP clusters and DR fall with n; WFSS and K-means stay at bias^2 ~ 27 up to n = 100,000; all estimators under-estimate V = 6.05 |
| Theorem 3.3's bias term drives it | `theorem_bias_vs_chi2_clean.png` | bias term vs within-cluster chi^2 divergence; five learned partitions on one curve, matched far below |
| Relative to DR | `exp2_offcem_dr_ratio_vs_chi2.png` | MSE_OffCEM / MSE_DR > 1 for every learned partition as the policy shift grows |
| Clustering algorithm is outcome-agnostic; bias follows local-correctness error | `offcem_bias_predictors_clean.png` | all learned partitions at LC error 18-20, bias 3.8; matched at 8, 0.6 |
| Bias grows with policy shift | `offcem_bias2_vs_lambda_clean.png` | learned partitions climb to 14, matched flat |
| No principled K | `kmeans_size_sweep_bias2_with_coverage.png`, `kmeans_initialization_bias2_with_coverage.png`, `clustering_results/mismatched_comparison.png` | finer K helps only at large n and stays far above matched |
| Whole-cluster support failure (pi0(c|x) = 0) | none | not exercised by our generator; one small experiment would cover it |

## 2. Method as instantiated

- Point estimators (OffCEM, DR, IPS) on the supported pairs, lambda = 100; the finite LP of the CLeaR draft on the rest (0.88-0.90 of the target's mass on synthetic data; exactly each user's favourite label on real data).
- Six constraints: (1) range, (2) relaxed local correctness around OffCEM's two-step model, (3) level band, (4) Lipschitz smoothness, (5) cluster-specific bounds, (6) favourite-treatment dominance. All plug-ins from logs (eps_max, eta = eps, L, reference intervals, cluster percentiles); the two-step model is clipped to the cluster's logged reward range; constraints 4-6 are imposed exactly and relaxed by a reported t only on infeasibility; eps is reported at three levels around the calibrated eps_max.
- Compatibility with the draft: all constraints are linear in the means, so the LP is the draft's with the trivial outcome partition (no p^B); the corrections b are an auxiliary lifting of relaxed local correctness.
- Two implementations of 4 and 6: with logged reference intervals and logged anchors (used in S3, S4, S5, S6), and exactly as written in the paper without them (S3b, S3c).

## 3. Results

### Synthetic, small world (200 actions, 1,000 rounds, 100 users, 10 clusters; V = 5.95)
| Plot | Content |
|---|---|
| S1 | DGP clusters, eps 1-4: bars contain V from eps = 1 |
| S2, S2b | K-means, eps 1-4 and 1-9: V inside from eps = 3; eps_max = 9.9 vs 7.7 for DGP |
| S3 | K-means, cumulative 1-3 / 1-4 / 1-5 / 1-6 at eps 4, 7, 9 with logged anchors: [-15.4, 17.6] -> [-0.2, 11.2] at eps 9, hard |
| S3b, S3c | same with the six constraints exactly as written (no anchors), K-means and DGP: [-15.4, 17.6] -> [-3.6, 11.3] at eps 9 (K-means), [-9.6, 16.9] -> [-0.8, 9.7] at eps 7 (DGP), hard; t = 16 at eps <= 4 |
| 2x2 (numbers only, `scale_sw_rangeonly_seed7000.json`) | range only [-27, 27]; + LC [-15.4, 17.6]; + Lipschitz [-12.3, 16.4]; both [-10.9, 13.8] (K-means, eps 9); chain without LC ends at [-0.8, 12.3] |

### Synthetic, OffCEM scale (1,000 actions, 8,000 rounds, 200 users, 50 clusters; V = 7.48)
| Plot | Content |
|---|---|
| S4c | four clusterings, cumulative 1-6 at floor(eps_max) -1 / floor / +1, no time limit: every bar contains V; t from 16 at the lowest level to 0-8 at the highest; B3 cutting planes capped at 3 rounds |
| S4d-S4g | the same, one panel per clustering |
| S4, S4b | earlier passes (900 s limit, ladder {2, 8}); keep as record of the infeasibility that the longer ladder resolved |
| 2x2 at scale | pending: `rangeonly_lam100_twostep_*` VM runs (command in the 2026-11-11 notes) |

### Real data (EUR-Lex 4K, Wiki10-31K; 300 users, 8,000 rounds, true-q logger beta = 8)
| Plot | Content |
|---|---|
| S5b | EUR-Lex, K-means: cumulative with Clopper-Pearson references, L from logs, pooled B7 anchor: [0.02, 0.92] -> [0.11, 0.92] at eps_max 0.45, all hard; V = 0.674 inside |
| S6 | Wiki10-31K, K-means: [0.00, 0.96] -> [0.08, 0.96], all hard; V = 0.682 inside |
| z = 3.5 check, L sweep, eta sweep, pooled B7 (`ex_eurlex_*`, `final_*_z35`) | z irrelevant; B3 inert for any L; eta = 0.1 excludes the truth; pooled anchor is the only informative belief |
| range-only on real data | identical to the baseline: the range is already the bound |

### Diagnostics kept in the result files (appendix)
- Truth checks: share of LP mass on which local correctness is violated at eps_max (30-50% everywhere), feasibility of the true mean vector, per-cell counts.
- Calibration details: tau^2, noise term, pairs, repeat degrees of freedom; L estimate with percentiles; clipping shares; soft-t tried per item; cutting-plane rounds and full-check flag.
- Scale B7 diagnosis (`diag_b7_scale.py`): 2-9 users per clustering whose favourite the two-step model puts 20-30 below the truth.

## 4. Missing for the current draft
1. Convergence with n (claim iii): no runs. Design: fixed world, n in {1k, 3k, 10k, 30k, 100k}, population endpoints from the true means, estimated bounds at calibrated and fixed eps.
2. Scale row of the 2x2 (range only, + Lipschitz): VM pass pending.
3. Whole-cluster support failure: not exercised.
4. Repeated-seed coverage rates: all S-plots are one world (seed 7000); the 50-seed closed-form runs exist only for the early ridge setup.
5. A figure for the "fragile assumption" claim (i): eps sweep with the LC-violation share as a second axis; data exist in the integer-grid files, no plot yet.

## 5. Caveats that must appear in the text
- Favourite-treatment dominance is true by construction (eps-greedy target on q) on both data types; a general target needs a stated regret margin (the reported t).
- The level band and relaxed local correctness always enter together; eta = eps.
- Clipping is part of the reward-model estimate, stated once with the share of predictions affected.
- At scale, Lipschitz is enforced by cutting planes capped at 3 rounds; the full Lipschitz check did not always pass.
- Real-data bars are wide because the logger shows each favourite at most once; the logger itself is the paper's synthetic-style logger, not its ridge-softmax logger, for the reason recorded in `REAL_DATA_DESIGN.md`.
