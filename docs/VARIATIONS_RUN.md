# Every variation of the LP-on-OffCEM setup run so far (as of 2026-11-09)

Grouped by component. Each line says what the variant is and where it was run (small world = 200 actions / 1000 rounds / 100 users / 10 clusters, seed 7000; scale = 1000 / 8000 / 200 / 50, VM; EUR-Lex = 3956 labels / 8000 rounds / 300 users / 100 clusters).

## 1. Centre m_hat (the model the LP is built around; also the model term of the DR / OffCEM supported estimators)
| Variant | Where |
|---|---|
| Cluster-level ridge g_hat(x, c), alpha = 1 | small world (plots 01-12 of clusters_x_lambda), scale ridge arm, lambda sweep arm (running) |
| Cluster-level ridge, alpha by out-of-fold MSE (chose 1000) | EUR-Lex |
| Paper's two-step f_hat = g_hat_MLP + h_hat (pairwise net, per-action heads) | small world (13-15), scale two-step arm, EUR-Lex |
| Two-step clipped to the cluster's logged reward range (option 4) | small world (16-18, 20, 21, 22, 22b) |
| Two-step with CV weight s on h_hat (shrinkage; s = 0 chosen on synthetic, 0-0.5 on EUR-Lex) | small world, EUR-Lex -- dropped |
| Two-step with ridge first stage (--g-model ridge) | code only, run aborted -- dropped |
| Oracle / oracle-shift / oracle-noise / zero centres | early small-world diagnostics (partial_ope_global) |

## 2. Overlap threshold lambda (which pairs the LP bounds)
| lambda | Where |
|---|---|
| inf (only pi0 = 0 pairs) | small world (03-06, 13, 16), scale ridge + two-step arms |
| 10 | small world (01, 02, 07, 08, 14, 17, 19, 20), EUR-Lex (all runs), scale lambda sweep (running) |
| 100 | small world (09, 10, 15, 18, 21, 22, 22b), EUR-Lex, scale lambda sweep (running) |
| 1000 | small world (11, 12) -- identical to inf on that world |

## 3. Clusters used for estimation
| Variant | Where |
|---|---|
| DGP's own clusters | small world, scale (dgp arm, relaunched), EUR-Lex: paper's uniform random clusters |
| K-means on action features (synthetic) / label co-occurrence embedding (EUR-Lex) | everywhere |
| Agglomerative (Ward), Spectral | scale, EUR-Lex |
| Hierarchical (= agglomerative alias) | scale launcher only |
| No clusters (one cluster) | early small-world "no-cluster" plots |

## 4. Local-correctness tolerance eps (displayed eps = rho = eps_LC / 2)
| Variant | Where |
|---|---|
| Forced grid 1-4 (plot-04 style) | small world 01-20 |
| Forced grid 1, 2, 3 | EUR-Lex (box collapses to [0, 1]) |
| Forced K-means grid {1, 1.9, 2.7, 3.6, 4.4} for DGP clusters at scale | scale dgp arm |
| Data-calibrated grid [eps_min, eps_max], 5 points, eps_max = 2 tau_hat from within-cluster logged contrasts | scale, EUR-Lex, small world 21, 22b |
| Integer grid 1 ... floor(eps_max) | small world 22 |
| Calibration variants: sigma from repeats (default); sigma given (--noise-sd); Bernoulli plug-in m(1-m) noise term (EUR-Lex); conservative bound 0.25 (reported) | scale / EUR-Lex |
| eps_max <= eps_min: abort (old) -> collapse to {eps_min} (new) | EUR-Lex, scale dgp |

## 5. Level band eta (|b_cell - g_hat_cell| <= eta)
| Variant | Where |
|---|---|
| eta = rho ("earlier setup", default) | nearly everywhere |
| eta = eta_hat (logged-data recipe, ~6-8 on synthetic) | early small-world plot 08, reported at scale (6.1-6.2, not used) |
| eta fixed at logged estimate (plot 05 "etaFixed") | early small world |
| Per-cell eta = max(rho, z * SE of the level ridge), z = 2 | EUR-Lex (b12), small world calib_etacell check |

## 6. Cross-fitting folds for m_hat
| Variant | Where |
|---|---|
| By round (plot-04 setup) | early small world; merged LP can be infeasible |
| By user (merged LP exact) | everything since the scale runs |

## 7. LP form
| Variant | Where |
|---|---|
| Closed form per (round, cluster) cell (plots 01-04 data, split_offcem*.json) | small world sweeps |
| Merged LP: one mu per (user, action), b per cell | all belief runs |
| Recentring of the LP by cluster-IPS bias estimate (recentre = cluster_ips / recentre_delta) | early plot 10 and split runs (recentre=ON rows) -- off since |
| Repair mode (cells with no feasible beliefs keep the outcome range) | closed form only |
| Outcome range [-30, 30] (synthetic) / [0, 1] (EUR-Lex) | all |

## 8. B3 -- Lipschitz across users for the same action
| Variant | Where |
|---|---|
| Coupling rows |mu(u,a) - mu(v,a)| <= L d(x_u, x_v) via cutting planes (kNN seed pairs), L = 3 | all belief runs |
| Reference boxes from logged (user, action) means +- z sigma / sqrt(n): z = 1.96 ("B3") | all; infeasible_bounds at scale |
| Reference boxes z = 3.5 ("B3_z3.5") | all since scale |
| One --ref-z for all reference intervals (skips the separate B3_z item) | option, used at scale |
| L = 3 given; L = 4.4; L estimated from logs (95th pct of noise-corrected same-action slopes; ~0.06 on EUR-Lex) | early plot 07 (3 vs 4.4), --L auto option |
| Oracle references (true q, logged pairs only / all pairs) | early plot 07 diagnostics |
| Soft: boxes +- t, coupling L d + t | small world 19-22 |

## 9. B5 -- group (cluster) mean bounds
| Variant | Where |
|---|---|
| Range: [min lower endpoint, max upper endpoint] of the cluster's logged reference boxes ("B5_range") | all belief runs |
| 5-95 %: [5th pct of lower endpoints, 95th pct of upper] ("B5_q5") | all belief runs |
| Endpoints at z = 1.96 (default) or z = 3.5 (--ref-z) | both used |
| Soft: box +- t | small world 19-22 |

## 10. B7 -- favourite-action dominance
| Variant | Where |
|---|---|
| Ordering among LP variables: mu(u,a) <= mu(u,a*), a* = argmax pi(.|x_u) ("B7_ord") | all belief runs |
| + logged lower bounds mu(u,a*) >= ybar(u,a) - z sigma / sqrt(n) for logged a, z = 1.96 (plot 10) and z = 3.5 ("B7_z3.5") | plot 10 (1.96), all since (3.5) |
| Combinations: B5_q5 + B7_z; B3z + B5_q5 + B7_z; (plot 10: B3 + B5_q5 + B7 logged) | all belief runs |
| Soft: dominance <= t, lower bounds - t | small world 19-22 |

## 11. Supported-region estimators (bars = estimate + LP, +- 1.96 SE)
| Variant | Where |
|---|---|
| IPS (weights <= lambda), DR (action-level weight + centre), OffCEM (cluster weight pi_S(c|x)/pi0(c|x), cap 100, + centre) | all |
| Full-value estimates as dashed lines (old plot 01) vs supported-part V+ lines (all since) | plots |

## 12. Logging / DGP variants
| Variant | Where |
|---|---|
| Synthetic: softmax(beta q) logger, beta = -0.1, deficient actions redrawn per round ("original exclusions") | all synthetic |
| Synthetic: fixed exclusions per user | early two-panel plots only |
| EUR-Lex: paper logger softmax(30 r_hat) (0.2% positives logged) and lower beta | EUR-Lex first contact |
| EUR-Lex: true-q softmax logger, beta = 12 and beta = 8 | EUR-Lex |

## 13. Feasibility repairs tried for the two-step centre
| Variant | Outcome |
|---|---|
| Calibrated eps grid instead of forced 1-4 | baseline feasible at top levels; beliefs still infeasible |
| Per-cell eta (z = 2) | no change to belief feasibility |
| Clipping the centre to the cluster's logged range (option 4) | baseline feasible at all eps; beliefs still infeasible alone |
| Soft beliefs t <= 8 (option 5) alone | infeasible; t = 16-32 needed, beliefs relaxed away |
| Clipping + soft beliefs | feasible: t = 0 at eps_max (K-means), t <= 4 one level below; B5 caps the top, B7 lifts the floor (plots 21, 22) |
| Shrinkage of h_hat by CV | works but changes the centre -- dropped by decision |
| Ridge first stage | not run -- dropped by decision |
