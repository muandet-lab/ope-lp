# Final formulation: one LP on top of OffCEM, same structure on synthetic and real data (2026-11-10)

Objective: finite, valid bounds on the target policy's value when OffCEM's point estimate is unreliable (e.g. a different clustering at estimation than in the data).
Point estimators (IPS / DR / OffCEM) are used only on the supported pairs; every unsupported pair is bounded by the LP of the CLeaR draft (range, local correctness, level band, plus linear beliefs).
Everything below is computed from logged data; the true expected rewards enter only the truth checks stored in the result files.

## Fixed across all settings
| Component | Rule |
|---|---|
| Centre m_hat | OffCEM's two-step model f_hat = g_hat + h_hat (paper's pairwise model), cross-fitted by user; also the model term of the DR / OffCEM supported estimators |
| Centre screening | never outside the logged / possible range: synthetic = clip f_hat(u,a) to [min, max] of logged rewards in a's estimation cluster; bounded outcomes = clip f_hat to the outcome range and intersect the level band with {b : f_hat + b in range} |
| Overlap threshold | lambda = 100: pairs with pi0 = 0 or pi / pi0 > 100 go to the LP |
| LP | merged: one mu per (user, action), one level b per (user, cluster) cell; eta = rho; constraints 1-3 of the slide (range, relaxed local correctness, level band) |
| eps | eps_max = 2 tau_hat from within-cluster logged contrasts (noise term: repeat variance on continuous rewards, m(1-m) plug-in on Bernoulli); reported grid = three levels around eps_max: floor(eps_max) -1 / floor / +1 on the +-30 reward scale, eps_max x {0.75, 1, 1.25} on [0, 1] |
| Beliefs, cumulative | 1-3 baseline; 1-4 + Lipschitzness (B3); 1-5 + cluster-specific bounds (B5, 5th-95th pct of logged reference endpoints); 1-6 + favourite-action dominance (B7: ordering + logged lower bounds + pooled Hoeffding anchor on the user's mean logged reward, delta = 0.05) |
| Reference intervals | logged mean +- 3.5 sigma_hat / sqrt(n) on continuous rewards; exact Clopper-Pearson at the level implied by z = 3.5 on Bernoulli rewards (z = 1.96 gives the same bars on both real datasets) |
| L for B3 | estimated from logs: 95th percentile of noise-corrected same-action slopes across user pairs (synthetic small world 2.97, EUR-Lex 0.061, Wiki10-31K 0.135); the earlier given L = 3 coincides with the estimate |
| Softness | hard first; if infeasible, every extra-belief constraint relaxed by t from a fixed ladder, smallest feasible t reported with the bar: {1, 2, 4, 8, 16} on +-30, {0.05, 0.1, 0.2, 0.4, 0.8} on [0, 1]; local correctness and the level band are never relaxed |
| Solver | HiGHS, no time limit; B3 cutting planes capped at 3 rounds at scale (reported) |
| Reporting | OffCEM + LP bars (DR / IPS available), t labels, 'infeasible' where no t works, eps_max and the grid rule in the footnote; truth checks kept in the JSON |

## What the real data add, and why
- Bernoulli rewards: outcome range [0, 1], Clopper-Pearson references, m(1-m) noise term, relative eps grid. Same formulation, outcome-appropriate plug-ins.
- The pooled B7 anchor mu(u,a*) >= Hoeffding lower bound of the user's mean logged reward is valid under dominance everywhere; it is the only belief with information on real data (one logged observation per pair otherwise) and is useless on +-30 rewards (bound ~ -23), so it is included everywhere and matters only on [0, 1].
- Logger: the paper's EUR-Lex logger (softmax over a ridge prediction, beta 30) logs a true positive in 0.2% of rounds; we use the paper's synthetic-style logger softmax(beta q) with beta = 8 (16% positives on EUR-Lex, 6% on Wiki10-31K). Stated as a design choice.

## Status per setting (OffCEM bars, V inside?)
| Setting | baseline | 1-4 | 1-5 | 1-6 | notes |
|---|---|---|---|---|---|
| Small world, DGP clusters (eps 4-6) | yes | yes | yes | yes, t <= 4 | S1/S3 |
| Small world, K-means (eps 7-9) | yes | yes, t <= 1 | yes, t <= 1 | yes, t <= 1 | combined [-1.4, 12.0] vs baseline [-15.4, 17.6] at eps 9 |
| Scale, 4 clusterings | yes | pending (final2 VM runs) | pending | pending | B5 alone fine; B7 infeasible under K-means / agglomerative in the previous pass (centre 20-30 below truth on 2-9 favourites); favourite-specific eps is the open fix |
| EUR-Lex 4K, 4 clusterings | yes, [0.01, 0.95] | same | same | [0.10, 0.92-0.95] | all hard-feasible; LC violated on ~47% of LP mass at eps_max (centre under-predicts favourites by 0.32); beliefs starved (median 1 logged observation per pair) |
| Wiki10-31K, 4 clusterings | yes, [0.00, 0.96] | same | same | [0.08, 0.96] | same picture, 6% logged positives |

## Open items
1. Scale B7 under K-means / agglomerative: favourite-specific eps from logged favourite residuals (same calibration recipe restricted to logged favourites), or drop LC on favourites with a direct B7 anchor. Decide after the final2 runs.
2. Favourite-action dominance is true by construction for an eps-greedy target on q (both data types); for a general target it needs a stated regret margin (the soft t) or the weaker pooled form.
3. Real data: the LP width is set by the logging density, not by the clustering; a logger that revisits favourites is the only route to informative real-data bars.
