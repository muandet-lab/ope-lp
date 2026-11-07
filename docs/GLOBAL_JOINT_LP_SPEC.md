# The global joint LP on the OffCEM DGP: full specification

Status: 2026-11-02. Everything below describes `Test notebooks/scripts/offcem_synthetic/partial_ope_global.py` as it exists now
(plus `sanity_checks.py` and `plot_bounds_vs_truth.py`). It is written so that someone could re-implement the method from this document alone.
Companion documents: `OFFCEM_PARTIAL_OPE_SETUP.md` (history and results narrative), `results_offcem_global/SANITY_REPORT.md` (all tables),
`PARTIAL_OPE_KNOWN_ISSUES.md` (older baseline).

Tags: **[code]** read from the implementation; **[derived]** follows mathematically; **[measured]** a number from one of my runs (small worlds,
K=200, n=1000, 100 users, 10 clusters unless stated); **[assumption]** something the validity of the bound depends on; **[gap]** something that is
missing, approximate or not yet justified.

---

## 1. Goal and decomposition

Estimate bounds on the value of an evaluation policy pi from logged data collected under pi0, when pi puts mass on actions that logging does not
cover well. With contexts x and actions a,

    V(pi) = (1/n) sum_i sum_a pi(a|x_i) q(x_i,a),        q(x,a) = E[Y | x, a].

The average is over the n logged contexts (the "sample-context" value; the oracle in all experiments is computed the same way). Pairs (x,a) are split
into a set **U** ("LP pairs") and its complement ("strong pairs"):

    V = V+ + V0
    V+ = (1/n) sum_i sum_{a not in U_i} pi(a|x_i) q(x_i,a)      estimated by IPS/DR (section 6)
    V0 = (1/n) sum_i sum_{a in U_i}     pi(a|x_i) q(x_i,a)      bounded by the LP (sections 4-5)

U = {pi0(a|x) = 0}  (hard no-overlap)  union  {pi(a|x)/pi0(a|x) > lambda}  (weak overlap; only if `overlap_lambda` is set). The final interval is

    [ V+hat + LO0 - z*se_total ,  V+hat + UP0 + z*se_total ],       z = 1.96 by default,

where [LO0, UP0] is the LP interval for V0 and se_total combines the sampling error of V+hat and (in aggregate mode) of the estimated cell levels.

## 2. The data-generating process (what the method is tested on)

The reward model is the OffCEM synthetic DGP of the ICML 2023 repo (`dataset.py`, unchanged): 10-dim standard-normal contexts from a fixed user pool,
K actions with 10 categorical attributes of 5 levels (one-hot, first level dropped), C clusters from `compute_clusters` (default `original`: a random
linear projection flattened by a softmax of temperature 10 and then *sampled*; `kmeans` is also available), expected reward
q(x,a) = g(x, cluster(a)) + h(x,a) (OBP degree-3 polynomial plus four threshold indicators for g; per-cluster random bilinear-plus-linear terms for h),
observed reward y = q + N(0, sigma^2), sigma = 3 by default. Measured: sd(q) 3.6, sd(g) 3.5, sd(h) 1.0.

### 2.1 Logging policy and the exclusion mechanism ([code] `make_world`)
Logging: pi0(a|x) = softmax(beta * q(x,.)) restricted to the allowed actions, beta = -0.1 by default (the repo's logits are q for continuous rewards). Two ways to
create the no-overlap set, with D = int(K * def_frac) excluded actions:

* **`user_fixed` (default).** The dataset is called with `n_deficient_actions=0` and only its contexts, expected rewards, cluster ids and `user_idx` are
  used (its own logged actions and rewards are discarded). Then, with `RandomState(seed+1000)`: for each user, D excluded actions are drawn uniformly without
  replacement, once for the whole world; pi0(.|u) = softmax(beta q_u) over the non-excluded actions; each round's action is drawn from pi0(.|user) by inverse
  CDF with one uniform per round; y = q(u,a) + sigma * N(0,1); pscore = pi0(a|u). Consequence: a pair (user, excluded action) is never observable.
  The LP's unit is the **user**; its unknown means are shared by all rounds of that user.
* **`round_random` (the dataset's own mechanism).** `n_deficient_actions=D`; the dataset draws, independently for every round, D excluded actions by Gumbel top-k.
  The LP's unit is the **round**. [measured] 4.9% of the "unsupported" (round, action) pairs (2.3% of the policy mass) are logged by the same user in another round,
  so the true mean q(x_u,a) is partly observable; and the marginal propensity (averaged over the random exclusion) is positive for every action, so this setting is not a permanent
  no-overlap problem.

### 2.2 Evaluation policy ([code] `finish_world`)
epsilon-greedy on the TRUE q (`policy.gen_eps_greedy`): with k=1, pi(a|x) = (1-eps)*1{a = argmax q} + eps/K, normalised; eps_policy = 0.2 by default. So pi > 0 on every action
and about 0.8 of its mass sits on one action per user. [measured] unsupported mass is about def_frac (0.11 / 0.50 / 0.90 for def_frac 0.1 / 0.5 / 0.9).

### 2.3 World object ([code])
`finish_world` returns per-round arrays (`uid`, `act`, `y`, `pscore`, `X_rows`) and per-unit arrays (`X_unit`, `q_unit` (units x K), `pi0_unit`, `pi_unit`,
`counts` = rounds per unit), the cluster map `cl` (K,), `C`, and `user_of_round`. Units with zero rounds are dropped and ids remapped. Unit weight wu = counts/n.
`world_from_cache` rebuilds a round-level world from the earlier `run_partial_ope_lc.py` cache (used only to reproduce old results).

## 3. Quantities given to the LP

For a unit u (user or round) with context x_u, weight wu_u = n_u/n:

* unsupported-or-weak set U_u (section 1), a boolean mask over actions;
* a fixed reward model m_ua (section 3.1), a "cell level" estimate ghat_uc and its allowance (section 3.2);
* outcome limits [l, u].

A **cell** is (unit u, cluster c) with at least one action of cluster c in U_u. Cluster of an action: `cl[a]`.

### 3.1 The reward model m ([code] `get_model`)
* **`cluster_level` (default).** m(x_u, a) = Mc[u, cl[a]]: a regression on (x, cluster) with **no per-action term**, constant over the actions of a cluster. Cross-fitted
  **by unit**: `foldu = RandomState(seed).permutation(nUnits) % folds` (5 folds); for held-out units of fold f the model is fitted on the rounds of all other units. So unit u's rewards never enter m(x_u, .)
  (tested, check W5). Two backends:
  * `ridge` (default): per cluster, `Ridge(alpha=1)` on `StandardScaler`-ed `PolynomialFeatures(degree=1, include_bias=False)` of x, using the training rounds whose logged action is in that cluster;
    clusters with fewer than 2 training rounds predict the training mean of y.
  * `mlp`: one pooled MLPRegressor (50,50,50), `max_iter=300`, `random_state=seed+f`, inputs = standardised x and the one-hot cluster; predicts all clusters for the held-out units.
* **Diagnostic models using the true q (never usable on real data):** `oracle` (m = q), `oracle_shift` (m = q - model_noise*tanh(x W_c)[cluster], a smooth cluster-constant error, `W_c ~ N(0,1)/sqrt(d)` from `RandomState(seed+1)`),
  `oracle_noise` (m = q + N(0, model_noise^2) iid per pair), `zero` (m = 0).
* The paper's two-stage pairwise model is NOT used by this program (it was too slow at default size and, at the small size, its within-cell error spread was about 12 versus 3 for `cluster_level`).

### 3.2 The cell level estimate ghat and its allowance ([code] `level_ridge`, `level_knn`)
Residuals of every logged round: r_j = y_j - m(unit_j, a_j), assigned to the cluster of the logged action. For each cluster c, and for every unit context x_u:

* **ridge centre (default).** Ridge regression of r on `[1, standardised poly(x)]` (degree 1, alpha 1, intercept unpenalised) fitted on the rounds of cluster c; ghat_uc = prediction at x_u.
  Standard error per cell: SE_uc = sigma_c * sqrt(phi_u' A_c phi_u), A_c = (P'P + penalty)^-1, sigma_c^2 = RSS / max(n_c - tr(P A_c P'), 1). Clusters with fewer than 2 rounds give NaN (the cell is "unidentified").
* **kNN centre.** Mean of the k = 20 nearest logged residuals (in context) of cluster c; SE = sd/sqrt(k).

The allowance eta_uc (section 4, belief 3) has up to four parts ([code] in `evaluate`):

    eta_uc = eta_fixed  +  [a_budget if eta_from_avg]  +  noise_uc  +  [rho if add_rho]
    noise_uc = se_mult * SE_uc  (per-cell mode, se_mult = 2 by default)   or   0  (aggregate mode)
    rho = eps_LC / 2

Cluster support at the context: cluster c is supported at u iff pi0(c|x_u) = sum_{a in c} pi0(a|x_u) > 0. A cell is **identified** iff its cluster is supported at u and ghat_uc is not NaN. Support is judged
on the hard pi0 only, not on lambda. Unidentified cells carry only the outcome range (belief 1).

## 4. The beliefs (the constraint set), exactly as coded

Unknowns: mu_ua = E[Y(a)|x_u] for every (u,a) in U; g_uc for every identified cell; optionally t_ua (average-budget auxiliaries).

1. **Outcome range.** l <= mu_ua <= u, for all pairs in U. Default l = floor(min y), u = ceil(max y) over the logged sample (a working assumption; the true q is outside this range in most small worlds, see 9). Justified limits can be passed (`limits`).
2. **Relaxed local correctness (LC).** For identified cells: |mu_ua - m_ua - g_uc| <= rho = eps_LC/2 for every unsupported action a of the cell.
   Equivalent statement [derived]: all pairs of unsupported actions a,b of the cell satisfy |(mu_a - mu_b) - (m_a - m_b)| <= eps_LC (a set of residuals has range <= eps_LC iff all lie within eps_LC/2 of their mid-range, which is g).
   eps_LC = 0 is the paper's Assumption 3.1 restricted to the LP pairs. Because the model m is constant inside a cluster, m_a - m_b = 0, so the belief says that the true means of the unsupported actions of a cell differ by at most eps_LC.
3. **Level close to its estimate.** |g_uc - ghat_uc| <= eta_uc, identified cells only.
4. **Optional average-error budget** (`avg_eps`): t_ua >= |mu_ua - m_ua - g_uc|, 0 <= t_ua <= rho, and sum_p w_p t_p <= avg_eps * sum_p w_p over identified pairs, w_p = wu_u * pi(a|x_u). This couples all cells in one row; it is the only constraint that makes the joint LP non-separable.

Not in the LP (deliberately, relative to the earlier 2A/2B/2C stages): event-probability beliefs, the ratio constraint, Lipschitz smoothness in x, any coupling across rounds/users other than the optional budget.

## 5. The program and how it is solved

Objective: minimise (maximise) sum_u wu_u sum_{a in U_u} pi(a|x_u) mu_ua. The unsupported-part interval is [LO0, UP0]; the total is V+hat + [LO0, UP0] before the sampling-error widening.

### 5.1 Single global LP ([code] `solve_global_lp`)
* **Variables** (layout): first the P unsupported means mu (one per pair in U), then Kc cell levels g (one per identified cell), then (average mode only) one t per identified pair. Pairs in unidentified cells keep only bounds.
* **Bounds:** mu in [l,u]; g_k in [ghat_k - eta_k, ghat_k + eta_k]; t in [0, min(rho, 1e6)].
* **Rows** (sparse, COO to CSR), per identified pair p in cell k:
  without the budget: mu_p - g_k <= m_p + rho and -mu_p + g_k <= -m_p + rho;
  with the budget: mu_p - g_k - t_p <= m_p and -mu_p + g_k - t_p <= -m_p, plus one row sum_p (coef_p/scale) t_p <= avg_eps * sum_p coef_p / scale.
* **Objective scaling:** coefficients coef_p = wu_u * pi(a|x_u), divided by coef.max() (HiGHS reported status 4 on tiny coefficients in the earlier orthant work), the optimum multiplied back.
* **Solver:** `scipy.optimize.linprog(method="highs")`, primal/dual feasibility tolerances 1e-9, called twice (min, and min of the negated objective). Status 2 is reported as `infeasible`; other failures as `solver_failure`; if there are no identified pairs the optimum is the [l,u] corner and no solver is called.
* [measured] 11,000 variables solve in 0.2 s, 110,000 in about 4 s.

### 5.2 The same optimum in closed form ([code] `cell_bounds`, [derived])
Without the budget the LP is block diagonal over cells, and each block is one-dimensional in g. For a cell with unsupported set U_c and rho = eps_LC/2:

    G_lo = max( ghat - eta ,  max_{a in U_c} (l - m_a - rho) ),      G_hi = min( ghat + eta ,  min_{a in U_c} (u - m_a + rho) )
    cell infeasible  iff  G_lo > G_hi + 1e-9
    lower = sum_a pi_a * max( l , m_a + G_lo - rho ),                upper = sum_a pi_a * min( u , m_a + G_hi + rho )
    unidentified cell:  lower = (sum_a pi_a) * l,  upper = (sum_a pi_a) * u

[measured] the two solvers agree to <= 3e-7 (typically 1e-13) on 400 random toy problems including infeasible ones, and on 11 worlds; infeasibility statuses always match.
The closed form is used for all sweeps (it is fast); `--lp both` runs both and reports the discrepancy.

### 5.3 Infeasibility and repair
An empty cell means no g satisfies the beliefs (limits, rho, ghat, eta). Default: the whole result is reported as `infeasible` with the count of cells and the largest gap, never clipped.
With `repair=True`, infeasible cells keep only [l,u] (as if unidentified), the result is marked `repaired`, and the count is reported; this is valid but wider. A cell can be infeasible even for huge eta if the limits and rho conflict by themselves.

### 5.4 Aggregate noise mode ([code] `noise="aggregate"`, `level_ridge(agg_w=...)`)
Per-cell mode puts the estimation noise of ghat into eta, so worst cases add up across cells. Aggregate mode runs the LP at the estimated levels (noise_uc = 0) and instead adds one sampling error for the whole contribution of the estimated levels:

    a_c = sum_u W_uc phi_u,  W_uc = wu_u * (sum_{a in U_uc} pi(a|x_u)) * 1{cluster supported at u};     Var_hat = sum_c sigma_c^2 a_c' A_c a_c
    agg_se = sqrt(Var_hat),        se_total = sqrt( se(V+hat)^2 + agg_se^2 )

[derived] to first order V0 moves by W_uc per unit of ghat_uc when the limits do not bind; clusters use disjoint logged rows, so their ghat errors are independent. Requires the ridge centre. The ridge shrinkage bias is not included.

## 6. The identified part V+ ([code] `evaluate`)

With known propensities, w_j = pi(a_j|x_j)/pscore_j for each logged round and `strong_j = (unit_j, a_j) not in U`:

* **IPS:** terms_j = y_j * w_j * strong_j.
* **DR (default):** terms_j = sum_{a not in U_u} pi(a|x_u) m_ua + w_j * strong_j * (y_j - m_{u_j a_j}).
* V+hat = mean of terms; its standard error = std(terms, ddof=1)/sqrt(n) (rounds treated as i.i.d.).

Truncation by indicator (not by clipping w) keeps the estimate unbiased for the strong pairs: E[w 1{w<=lambda} y] = sum over strong a of pi q. With `overlap_lambda = None`, U is only the hard no-overlap set and the terms are the usual IPS/DR restricted to pi0 > 0.
The logged rounds dropped as weak still feed the residuals used for ghat. DR is unbiased for any model that did not use the round's own reward (cross-fitted `cluster_level`, the oracle models).

## 7. Outputs ([code] `evaluate`, JSON from `run_setting`)

* `data`: unsupported mass (LP pairs), oracle V, V+ and V0, outcome limits, true q range, supported estimates (IPS, DR, their errors and standard errors), max weight (strong and raw), logged rounds dropped as weak, number of identified cells, unidentified mass, `ghat_agg_se`, `total_se`, noise mode, and the no-assumption bounds V+hat + mass*[l,u].
* `validation_oracle` (never used to fit): weighted spread of the true model error inside cells (median/p95/max), error of ghat against the true cell mid-range, absolute model error, optionally the true mean absolute deviation from the best common level.
* per eps_LC in the list (a number or `oracle:Q`, the mass-weighted Q-quantile of the TRUE within-cell spread over identified cells, a diagnostic): status, infeasible cells, `unsupported_interval` [LO0, UP0], `lower`/`upper` (= V+hat + LO0/UP0), `width`, `with_se` (the headline interval, widened by z*se_total), coverage flags (`contains_V0`, `contains_V`, `contains_V_with_se`, `contains_V0_ci`), `truth_violation_share` (policy mass in cells whose TRUE means break at least one belief, split into range and spread parts), the global-LP result and gap if requested, and the budget LP result if a budget is set.

## 8. Assumptions the validity rests on

**Standing assumptions (outside the constraints):**
1. **Known propensities.** pscore and pi are exact; both are inputs to the weights. Estimated propensities are not handled.
2. **Unconfoundedness / consistency.** The reward depends on (x,a) only through q; the logging policy depends on the covariates x (here through q), not on unobserved variables.
3. **i.i.d. rounds, stable reward mechanism, no interference.** Used for the standard errors; in `user_fixed` the user pool is fixed and the rounds are independent draws from it.
4. **Sample-context target.** V is the average over the observed n contexts (not over the user population).
5. **Fixed, known clustering.** The cluster map used for estimation is the one used to generate the rewards (matched). The mismatched case is not tested here.
6. **Model fitted without the unit's own data** (cross-fitting by unit) so DR is unbiased and residuals are honest.

**Assumptions encoded as constraints (the "beliefs"):**
7. **Outcome limits** contain every true conditional mean of the LP pairs.
8. **Relaxed LC with tolerance eps_LC** on the unsupported actions of each cell. The method does not estimate eps_LC; in the experiments it is set from the true spread (`oracle:Q`), which is a diagnostic.
9. **Level belief:** the common level g of the cell is within eta of the estimate ghat. Justification of the ingredients:
   * noise: se_mult * SE per cell (per-cell mode) or the aggregate confidence interval (aggregate mode);
   * the **rho term**: ghat averages the logged residuals of the same cluster at nearby contexts, and logged actions belong to the cell too. If LC holds with eps_LC over ALL actions of the cell (logged and unsupported), every residual lies within rho of the cell's mid-range, so ghat can sit rho away from the level g that the LP's actions share. The LP itself only constrains the unsupported actions, so this extension to logged actions is an assumption, not something the program enforces [gap];
   * **bias from borrowing across contexts** (the level g(x,c) varies with x; ghat is a regression, linear in standardised x by default) is not modelled except through the optional `eta` (a fixed allowance) [gap].
10. **Cluster support at the context.** Cells whose cluster has pi0(c|x_u) = 0 get only the outcome range; this is a feature of the program, not an assumption, but it makes the bound loose (e.g. 12.5% of the unsupported mass at def_frac 0.9 with 10 clusters).

**Assumptions about the sampling interval:**
11. **Normal approximation with z = 1.96** for the sum of the V+hat error and (aggregate mode) the ghat error, combined in quadrature, i.e. treated as independent. They share the residuals y - m, so independence is approximate [gap]. The aggregate variance formula assumes the limits do not bind, ignores ridge shrinkage bias and treats the ridge noise level sigma_c as the same for all contexts [gap].
12. **Weak-overlap threshold lambda** is a user choice (default None); a larger set U lowers the variance of V+hat and raises the mass the LP must bound.

**Not assumed anywhere:** monotone or sign restrictions, Lipschitz continuity, a parametric reward form.

## 9. What has been measured (small worlds; details in SANITY_REPORT.md)

* **Part A (mechanics), 44 passed, 0 failed, 1 informational:** global LP == closed form; sign symmetry; intervals nest as rho and eta grow; huge slack gives V+hat + mass*[l,u]; exact LC with the true level and zero slack gives a point; the oracle model with noiseless rewards and eps_LC = 0 collapses exactly onto V (width 0; with unidentified cells the width equals their mass times (u-l)); probabilities and mass sums; user_fixed exclusions are constant and never logged; no cross-fitting leakage; invariance to round order, cluster relabelling, action relabelling; determinism; reproduction of the earlier saved result to 1e-16; edge cases (no unsupported pairs rejected; tiny world; singleton clusters; more users than rounds; greedy and uniform targets; def_frac 0.95). The one informational item: the default outcome limits (sample range of y) exclude true means in most worlds.
* **Part B (351 runs, 36 setups, per-cell eta):** the LP alone covers V0 in every feasible run at eps_LC = 4 and at the oracle 95% spread (one exception: no noise allowance, 0.9), while the truth breaks a belief in 8-30% of cell mass, so aggregate coverage does not validate the beliefs.
* **Part D (identified part, no cap):** over 200 worlds IPS/DR are unbiased (mean error +0.08 +- 0.13 and +0.06 +- 0.12) but heavy-tailed (5-95% range about [-2, +3.8], rms 1.7; error averages -2.1 when the greedy action is never logged); standard errors cover about 77-79% at +-2 se.
* **Part E (overlap threshold):** coverage of the full interval 1.00 at 10%/50%/90% excluded for lambda from 2 to 30 (uncapped: 0.28/0.93/1.00); LP mass 0.82-0.98, widths 12-19.
* **Part F (aggregate noise, lambda 10, repair on, 40 worlds per row):**

  | setting | per-cell eta: coverage / width | aggregate: coverage / width |
  |---|---|---|
  | target eps 0.2, 10% excluded | 1.00 / 12.4 | 0.95 / 7.5 |
  | target eps 0.2, 50% excluded | 1.00 / 14.6 | 1.00 / 9.2 |
  | target eps 0.2, 90% excluded | 1.00 / 18.6 | 1.00 / 13.5 |
  | target eps 0.5, 50% excluded | 1.00 / 12.2 | 1.00 / 7.6 |
  | target eps 1.0, 50% excluded | 1.00 / 7.8 | 1.00 / 4.7 |

  Dropping the rho term gave coverage 0.00-0.07 in most settings; the average-error belief with a budget from the oracle mean deviation failed (0.00 at 3x and at 10x for 10%/50% excluded).
* All of the above is at eps_LC = the oracle 95% quantile of the true spread and the sample-range outcome limits.

## 10. Known gaps and limitations

1. **eps_LC is not estimated** (oracle diagnostic only). With sigma = 3 and about 3-5 observations per action, the spread of h (sd 1) is not identifiable from the data.
2. **Outcome limits** default to the sample range, which excludes true means in 14 of 20 worlds [measured]; wide limits ([-30,30]) cost nothing in width here.
3. **Command-line interface is incomplete.** `main()` exposes `--overlap-lambda`, `--eps-lc`, `--eta`, `--se-mult`, `--no-add-rho`, `--avg-eps`, `--lp`, centre/model options, but NOT `noise=aggregate`, `repair`, `eta_from_avg`, `avg_eps_mult`, `diag_mad`; those are only reachable through `run_setting(spec)` (as `sanity_checks.py` does). `--avg-eps` with an explicit number works.
4. **The level belief only constrains unsupported actions**; the "rho term" extends it to logged actions by assumption (item 9 above).
5. **Smoothness bias of ghat** is not modelled; no data-driven allowance for it.
6. **Combined standard error** assumes independence of the V+ and ghat errors and normality; heavy tails are reduced by lambda but not removed (the uncapped version covered only 77-79% at 2 se).
7. **Coverage numbers have granularity 1/40** (40 worlds per row), single base configuration, small worlds only; **no default-size run** (1000 actions, 3000 rounds, 50 clusters). Part B reuses the same world seeds across setups, so rows differing only in LP settings share their supported-part errors.
8. **The truth-violation diagnostic** checks the beliefs on unsupported actions only (the cell level interval and the range), not the full pairwise LC over logged actions.
9. **Cell-level beliefs are often false while aggregate coverage holds** (per-cell slack is conservative in the sum), so coverage is not evidence that the beliefs are true.
10. **Known propensities**; no estimation error in pscore or pi.
11. **The average-error belief** has no data-based budget; the oracle-based one failed.
12. **Targets with concentrated mass** (eps 0.2) force most of the policy mass into the LP once weights are capped, so valid bounds are wide (7-13 around values of 0-9).

## 11. File map and how to run

| File (in `scripts/offcem_synthetic/`) | Role |
|---|---|
| `partial_ope_global.py` | world generation, models, cell levels, closed form, global LP, estimators, validation (this document) |
| `sanity_checks.py` | `--part A` mechanics (45 records), `B` 36 setups x seeds, `C` average budget, `D` identified part, `E` overlap threshold, `F` aggregate noise and variants |
| `plot_bounds_vs_truth.py` | figure `results_offcem_global/bounds_vs_truth.png` (aggregate versus per-cell, three exclusion levels) |
| `dataset.py`, `policy.py`, `clustering.py`, `ope.py`, ... | the repo's OffCEM code, unchanged |
| `run_partial_ope_2d.py`, `run_partial_ope_lc.py` | earlier versions (independent boxes; per-round cells with the paper's pairwise model) |

Run from `Test notebooks/scripts/offcem_synthetic` in the `offcem` environment (Python 3.9, OBP 0.5.5). Short runs only; large jobs are for the user to launch.

```bash
# CLI: global LP and closed form on a small world, weights capped at 10
conda run -n offcem python partial_ope_global.py --def-frac 0.5 --exclusion user_fixed --model cluster_level --centre ridge \
    --supported dr --overlap-lambda 10 --eps-lc 4 oracle:0.95 --lp both --output ../../results_offcem_global/example.json

# aggregate noise + repair (options not on the CLI): save as a script and run it from this directory
cat > /tmp/agg_example.py <<'PY'
import json
import partial_ope_global as G
out = G.run_setting({"world": dict(K=200, n=1000, users=100, C=10, def_frac=0.5, exclusion="user_fixed", seed=7001),
    "model": "cluster_level", "cluster_model": "ridge", "centre": "ridge", "supported": "dr", "overlap_lambda": 10.0,
    "noise": "aggregate", "add_rho": True, "repair": True, "eps_lc": ["oracle:0.95"], "lp": "both"})
print(json.dumps(G.clean(out["results"][0]), indent=1)[:1500])
PY
PYTHONPATH=. conda run -n offcem python /tmp/agg_example.py

# the sanity suites (each takes about a minute on 6-7 workers)
conda run -n offcem python sanity_checks.py --part A --out ../../results_offcem_global
conda run -n offcem python sanity_checks.py --part F --workers 7 --out ../../results_offcem_global
```

Default-size world (not yet run; the cluster-level model has no pairwise training so it should be fast; the global LP has about 100k variables for 200 users x 500 excluded actions):

```bash
conda run -n offcem python partial_ope_global.py --K 1000 --n 3000 --users 200 --C 50 --def-frac 0.5 --overlap-lambda 10 \
    --eps-lc oracle:0.95 --lp both --output ../../results_offcem_global/default_size.json
```

## 12. Glossary of symbols

| Symbol | Meaning |
|---|---|
| u, unit | user (`user_fixed`) or round (`round_random`); wu = rounds of the unit / n |
| U | unsupported (pi0 = 0) or weakly supported (pi/pi0 > lambda) pairs; the LP bounds them |
| mu_ua | unknown mean reward of action a at the unit's context |
| m_ua | fitted model prediction (constant over the actions of a cluster for `cluster_level`) |
| g_uc, ghat_uc | shared cell level (LP variable) and its regression estimate |
| eps_LC, rho | LC tolerance on pairwise differences; rho = eps_LC/2, the radius around g |
| eta | allowance on |g - ghat| |
| l, u | outcome limits |
| lambda | weak-overlap threshold on pi/pi0 |
| V+, V0 | identified and LP-bounded parts of V |
