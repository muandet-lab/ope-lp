# Small-world pipeline, end to end (as run for S1-S3, S3b, S3c; 2026-11-12)

Every quantity below is computed by `scripts/offcem_synthetic/scale_experiment.py` and the modules it calls (`partial_ope_global.py` = G, `b3_lipschitz_experiment.py` = B, `b5_b7_experiment.py` = X, `extra_beliefs_experiment.py`, `eps_sweep_dgp.py`, `eta_separation_experiment.py`). Function names are given so the formulas can be checked against the code. The scale and real-data runs use the same pipeline with the differences listed at the end.

## 1. World (G.make_world, G.finish_world)
- OffCEM's synthetic generator: K = 200 treatments, N = 100 users with x ~ N(0, I_10), C = 10 generating clusters from the treatments' categorical embeddings, n = 1,000 logged rounds, seed 7000.
- Expected reward q(x, t) = g(x, c_t) + h_{c_t}(x, t); reward y ~ N(q, 3^2).
- Logging policy pi0 = softmax(-0.1 q) with 50% of treatments excluded **per round** (`exclusion = round_random`), renormalised; the excluded set is redrawn every round, so a treatment with pi0 = 0 in one round is logged in others.
- Target policy pi = 0.2-greedy on the true q (`G.gen_eps_greedy`); its favourite t*(x) = argmax_t q(x, t) by construction.
- Stored per world: user context X_unit (N x 10), true q_unit (N x K), pi0_unit, pi_unit (N x K, per-round pi0 collapsed to the user's average), logged rounds (uid, act, y, user_of_round), treatment features action_x (K x 40, one-hot embedding), generating clusters cl.
- True values: V = mean over users of sum_t pi(t|x) q(x, t) = 5.95; V+ and V0 = the same sums restricted to supported / unsupported pairs (0.30 and 5.65 at lambda = 100).

## 2. Estimation clusters (scale_experiment.labels)
- `dgp`: the generating clusters. `kmeans`: K-means (10 clusters, n_init 10, seed 7000) on action_x. Agglomerative / spectral likewise. The world data never change; only the partition used by the model and the LP cells.

## 3. Reward model m_hat (G.get_model -> G.fit_two_step_units)
- OffCEM's two-step model f = g + h as in the paper's code: h from `ope.train_pairwise_model` (pairwise net over observed same-user same-cluster pairs, per-treatment heads, 30 epochs), g from an OBP RegressionModel MLP(50, 50, 50) on the residual y - h with the cluster as the action.
- Cross-fitted with 5 folds **by user** (`--folds user`): the model that predicts for user u never saw u's rounds. (User folds make the merged LP exact; round folds gave fold disagreement.)
- Clipping (`--clip-model cluster_logged`, B.prep): m_hat(u, t) is clipped to [min, max] of the rewards logged in t's estimation cluster, over all users. Small world: 9-10% of predictions changed, mean change 0.3-0.4, max 19. Also clipped to the outcome range [-30, 30].
- The same m_hat is the LP centre, the model term of the DR / OffCEM supported estimates, and the input to the eps calibration.

## 4. Support split and supported part (X.structures, G.evaluate)
- U(u, t) = [pi0(t|u) = 0] or [pi(t|u) > lambda pi0(t|u)], lambda = 100. Unsupported mass = sum over users of their weight wu times sum_{t in U} pi(t|u) = 0.88.
- Supported estimates on the complement (strongly supported rounds, weight <= lambda): IPS = mean of w y 1{supported}; DR = mean of [sum_t pi(t|x) m_hat 1{not U} + w 1{supported} (y - m_hat)]; OffCEM = the same with the cluster weight pi_S(c|x) / pi0(c|x), pi_S = target mass on the supported treatments of the cluster, capped at 100. Standard error Vp_se = sd of the per-round terms / sqrt(n).
- Total SE for the bars: se_tot = sqrt(Vp_se^2 + agg_se^2), where agg_se is the sampling SE of the aggregated level estimate sum_u,c w_uc b_hat_uc from the level ridge (`G.level_ridge`, agg_w = wu x unsupported target mass of the cell).
- Bar = supported estimate + [LP lower, LP upper] widened by 1.96 se_tot on each side (B.rec).

## 5. LP variables (B.prep)
- One pair row per (user u, unsupported treatment t) with cluster c = cl[t]; its objective coefficient is coef = wu(u) pi(t|u), wu = share of logged rounds belonging to u.
- A cell (u, c) is *identified* if its cluster has logging mass for u (pi0(c|u) > 0 or pi(c|u) = 0) and the level ridge returned a value. Pairs in non-identified cells are not variables: they contribute coef x LO to the lower bound and coef x HI to the upper bound (const_lo, const_up) -- the range only.
- Variables: mu_v, one per identified (user, treatment) pair, nV = 19,907 on the small world; b_k, one per identified (user, cluster) cell, Kc cells. Several logged rounds of the same user map to the same mu (merged LP), which is why user folds are needed.
- Objective: minimise (lower) and maximise (upper) sum_v coef_v mu_v, scaled by max coef for conditioning; the two solves give [LP lower, LP upper] on V0 after adding const_lo / const_up.

## 6. Constraints and how every constant is computed
### (1) Range
LO <= mu_v <= HI with [LO, HI] = [-30, 30] (outcome range given).

### (2) Relaxed local correctness (rows of A_base)
For every pair row r = (u, t) in cell k: mu_v - b_k <= m_hat(u, t) + rho and -mu_v + b_k <= -m_hat(u, t) + rho. rho = displayed eps. Two rows per pair, 2 n_pairs rows in total.

**rho grid.** eps_max = Z tau_hat, Z = 2 (`eps_sweep_dgp.calibrate_user_cells`):
- residual e_i = y_i - m_hat(u_i, t_i) for every logged round;
- noise variance sigma^2 = pooled sample variance of y over (user, treatment) pairs logged more than once (41 repeat dof on the small world; NaN if < 10);
- for every pair of logged rounds of the same user in the same estimation cluster with different treatments, half the squared residual contrast (e_i - e_j)^2 / 2; its expectation is tau^2 + sigma^2 because the shared correction cancels;
- tau_hat^2 = mean of those halves - sigma^2, clipped at 0; eps_max = 2 tau_hat (7.69 DGP, 9.86 K-means).
- Reported grid: `--eps-grid 1 ... 9` for S1-S3 (integer sweep), `--eps-around-max` = floor(eps_max) - 1 / floor / + 1 for the final formulation.

### (3) Level band (bounds on b_k)
b_hat_k - eta <= b_k <= b_hat_k + eta, eta = rho. b_hat_k = G[u, c] from `G.level_ridge`: per estimation cluster c, a ridge regression (alpha = 1, intercept unpenalised, standardised linear features of the user context) of the logged residuals e on x over the rounds whose treatment lies in c, evaluated at user u's context. Its standard error S[u, c] is stored (used only by the per-cell-eta variant and the diagnostics). `eta_hat` from `eta_separation_experiment` is computed and recorded but not used (eta = rho throughout).

### (4) Lipschitz smoothness (X.lp rows + B.knn_pairs / B.violated cutting planes)
|mu_v - mu_w| <= L d(x_u, x_u') for pairs of variables with the same treatment and different users, imposed as two rows each. d = Euclidean distance between standardised user contexts (B.prep: z-scored X_unit, D matrix). Cutting planes: start from each variable's 3 nearest same-treatment neighbours (KNN = 3), solve, add every violated pair (B.violated, up to 40,000 per round), repeat up to `--max-it` rounds (8 on the small world, 3 at scale); `full_check_passed` records whether no violation remained.
- L: `--L auto` = 95th percentile over all user pairs that both logged treatment t of sqrt(max(0, (ybar_ut - ybar_u't)^2 - sigma^2 (1/n_ut + 1/n_u't))) / d(u, u') (scale_experiment.estimate_L) = 2.97 on the small world; earlier runs used L = 3 given.
- *With logged references* (S3, S4, S5): additionally mu_v in [lo(u', t) - L d, hi(u', t) + L d] intersected over every other user u' who logged t, where [lo, hi] = ybar +- z sigma / sqrt(n), z = 3.5 (B.box). *As written in the paper* (S3b, S3c, items `B3c*`): coupling only, no boxes.

### (5) Cluster-specific bounds (X.b5_bounds, mode q5)
ell_c <= mu_v <= u_c for every variable whose treatment lies in c; ell_c = 5th percentile of the lower endpoints and u_c = 95th percentile of the upper endpoints of the logged reference intervals ybar +- 1.96 sigma / sqrt(n) over all logged (user, treatment) pairs with treatment in c. (The `range` variant, min / max, was dropped.)

### (6) Favourite-treatment dominance (X.b7_rows)
mu_v <= mu_{v*} for every variable v = (u, t) of user u whose favourite t*(u) = argmax_t pi(t|u) is itself an LP variable v*; one row per (u, t). On the small world every user's favourite is unsupported. *With logged anchors* (S3-S6, z = 3.5): also mu_{v*} >= max over logged t of (ybar_ut - z sigma / sqrt(n_ut)), and on real data the pooled Hoeffding anchor mu_{v*} >= ybar_u - (HI - LO) sqrt(log(1/delta) / (2 n_u)), delta = 0.05. *As written* (S3b, S3c, `B7_ord`): ordering rows only.

### Cumulative items
`baseline` = (1)-(3); `B3c` = + (4); `B3c+B5_q5` = + (5); `B3c+B5_q5+B7_ord` = + (6). The anchored versions are `B3_z3.5`, `B3z+B5_q5`, `B3z+B5_q5+B7_z3.5`.

## 7. Soft relaxation (X.lp argument t, scale_experiment ladder)
Each item is solved with t = 0. If HiGHS reports infeasible, t steps through `--soft-t 1 2 4 8 16`: reference / cluster bounds become lb - t, ub + t (within the range), coupling becomes L d + t, dominance becomes mu_v - mu_v* <= t. The first feasible t is reported (`min_feasible_t`, `soft_t_tried`). Rows (1)-(3) are never relaxed.

## 8. Solver
scipy.optimize.linprog, method HiGHS, presolve on, `--time-limit 0` = no limit. Two solves per item (lower, upper); with coupling, up to max_it re-solves per direction.

## 9. Diagnostics stored with every item (not used by the LP)
- `lp_contains_V0`, per-estimator bars and `contains_V`, `cuts`, `iterations`, `full_check_passed`, `box_tightening`, `n_constraints`, `min_feasible_t`, `soft_t_tried`, seconds.
- Per rho: `truth_checks` = share of unsupported target mass whose true mean lies outside its local-correctness window, whether the true mean vector is feasible, number of infeasible cells.
- Per clustering: calibration dict (sigma, tau, pairs, dof), L estimate, clip_info, beliefs_info (true mass outside each belief's bounds).

## 10. Differences at scale and on real data
- Scale: K = 1,000, n = 8,000, N = 200, C = 50; same pipeline; B3 cutting planes capped at 3 rounds; 14 min per baseline solve, 45-80 min per Lipschitz solve.
- Real data: Bernoulli rewards, range [0, 1]; sigma^2 term in the calibration = plug-in m_hat (1 - m_hat); reference intervals = Clopper-Pearson at the level implied by z; clipping = `outcome_cell` (m_hat clipped to [0, 1] and the level band intersected with {b : m_hat + b in [0, 1]}); eps grid eps_max x {0.75, 1, 1.25}; soft ladder {0.05, 0.1, 0.2, 0.4, 0.8}; level ridge alpha = 1000 on the 100 TruncatedSVD / PCA features; one LP variable per user (the favourite label) at lambda = 100.
