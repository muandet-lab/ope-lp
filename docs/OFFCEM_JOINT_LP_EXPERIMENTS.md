# OffCEM partial-identification: joint LP, experiments and slides (2–3 Nov 2026)

Everything lives in `Test notebooks/scripts/offcem_synthetic/`. The baseline scripts in `Test notebooks/scripts` are untouched.
Run with `conda run -n offcem python ...` (Python 3.9, OBP 0.5.5) from `offcem_synthetic/`.
Older docs: `GLOBAL_JOINT_LP_SPEC.md` (per-cell global LP, not updated for the joint LP), `OFFCEM_PARTIAL_OPE_SETUP.md`, `PARTIAL_OPE_KNOWN_ISSUES.md`, `results_offcem_global/SANITY_REPORT.md`.

## 1. Goal
Bound V(π) = V⁺ + V⁰, where V⁰ is the part on pairs the logged data cannot cover (π0 = 0; also π > λ·π0 in the split version).
Bounds should be tight and contain the true value. Outcome range, relaxed local correctness (ε) and logged-data constraints are the assumptions.

## 2. The world (DGP)
- Users: X ~ N(0, I_d), 100 users, d = 10. 1000 logged rounds.
- Actions: 200, in 10 clusters (`clustering="original"`).
- Reward: q(x,t) = shared cluster part g_c(x) + action-specific part (random per-action coefficients, so within-cluster spread is mostly unlearnable). Y = q + N(0, 3²).
- Logging policy: softmax-style exp(−0.1 q), with 50% of actions zeroed.
- Eval policy: 0.8 on the best action, 0.2 spread over the rest (`eps_policy`).
- Exclusions: `user_fixed` (one excluded set per user, the default for these experiments) vs `round_random` (the original setting, redrawn per round; the LP has 210,000 variables, about 105 s per ε).

## 3. Methods
### 3.1 Joint LP (`partial_ope_joint.py`)
Variables μ for ALL pairs plus a free shared level g_uc per user–cluster cell. Constraints:
- Range: ℓ ≤ μ ≤ u, fixed [−30, 30] (an assumption).
- Relaxed local correctness: |μ − m − g| ≤ ρ = ε/2 for every pair in a cell (so the spread inside a cell is at most ε).
- Data moments: b_j − r_j ≤ Σ A_j,ua (μ_ua − m_ua) ≤ b_j + r_j, with logged terms t_ij = f_j(u_i,a_i)(y_i − m_i).
- Moment families: strong (π/π0, capped), cluster (π(c|x)/π0(c|x), cap lam_c = 100, the default), cluster_by, logging_by.
- Simultaneous radii: sup-t multiplier bootstrap (B = 4000), Bonferroni, pointwise or none.
- HiGHS min and max of the policy value. No repair step, no average-error budget.
- Oracle checks (separate from coverage): range_ok, LC (max true spread ≤ ε), confidence set contains truth. `theory_violation` fires only if all of these hold and the LP still misses the truth.
- Model: cluster-level ridge, cross-fitted by unit. Action-attribute and interaction models were tried and do not reduce spread.

### 3.2 Split version (`partial_ope_global.py`)
LP only on unsupported pairs (and weights above λ = 10). Supported part from DR (default) or IPS, with truncation. Per-cell closed form with ridge centre ĝ, η = ρ and ρ = ε/2. Result is V⁺ + LP interval, widened by 1.96·SE (aggregate ridge noise).

### 3.3 Estimators used in plots
IPW and DR via OBP. OffCEM via the repo's `ope.OffCEM(is_clustering=True)` (DM + π(c|x)/π0(c|x)·(y − m)). DR and OffCEM use the cross-fitted cluster-level model, no cap.

## 4. Experiments and findings
Logs: `results_offcem_global/joint_experiment*.log`.

### 4.1 Mechanics
- With oracle model and exact moments the interval collapses to V exactly (width 0). 0 theory violations over 150+ runs.
- An early width of 32 came from lam_c = 10 dropping 57% of the cluster mass; fixed with lam_c = 100.

### 4.2 Joint LP over 50 worlds (configs A, F, G; `joint_experiment_50`)
Base config A (cluster moment), median width and coverage:

| ε | LC true | coverage | width |
|---|---|---|---|
| 2 | 0% | 96% | 5.74 |
| 4 | 0% | 100% | 9.18 |
| 6 | 2% | 100% | 12.63 |
| 8 | 90% | 100% | 16.07 |
| none | 98% | 100% | 59.42 |

- The confidence set contains the truth in about 94% of worlds. No theory violations.
- Width is about linear in ε. The real driver is ε, not the moment family (configs B–E add strong, logging_by, Bonferroni and a cap of 30, with only small changes).
- ε ≈ 8 is the justified level (LC true in about 90% of worlds): width about 16 vs about 59 with no LC.
- F (90% of actions excluded) is wider (12.6 to 22.6); G (less concentrated target) is narrower (4.5 to 13.2).
- Coverage at ε < 8 holds empirically (aggregate averaging) but is not guaranteed: cell-wise LC is false there.

### 4.3 ε from data
A data-only proxy gives about 13.4. The true within-cluster spread has median about 3 (q95 about 4.9, max about 7). ε is not identifiable from data, and the ε grid was picked after looking at the truth.

### 4.4 Fixed vs original exclusions
Fixing exclusions per user helped validity, not tightness. `round_random` widths are about the same as `user_fixed`. Only 8 worlds were run for `round_random`. OffCEM is the most stable point estimator, especially under `round_random`.

### 4.5 Split version, and the ε = 2 coverage issue (3 Nov)
Over 50 worlds, the original split setup had coverage of 10% at ε = 2 and 100% at ε ≥ 4. Median widths 4.7, 8.3, 11.9, 15.5 for ε = 2, 4, 6, 8. About 90% of the mass is in the LP part, so the DR-based and IPW-based bars nearly coincide.

Cause: the ridge centre ĝ is smooth in x and sits near 0. The target puts its mass on the top action of each cluster, which is unsupported, and the cluster model under-predicts it by about 3.3 per unit mass (about 2.95 on V). With ε = 2 the half-width is only ε·mass = 1.8, so the true value is out of reach.

Fix: `recentre="cluster_ips"` in `partial_ope_global.py`. A constant shift of the cell levels so that their policy-weighted sum equals the cluster-weighted residual π_U(c|x)/π0(c|x)·(y − m) (capped at `lam_c`, default 100). The SE uses the joint error of the supported estimate plus the correction. Off by default.

| ε | coverage before → after | median width before → after |
|---|---|---|
| 2 | 10% → 96% | 4.7 → 5.9 |
| 4 | 100% → 100% | 8.3 → 9.5 |
| 6 | 100% → 100% | 11.9 → 13.1 |
| 8 | 100% → 100% | 15.5 → 16.7 |

The correction recovers about 1.5 of the 2.95 gap. The rest is selection within a cluster that logged data cannot show, so the 96% is empirical (48 of 50 worlds), not a guarantee. Without the SE widening, coverage at ε = 2 is 68%. DR- and IPS-supported versions agree to two decimals.

## 5. Slides
`Test notebooks/OffCEM_slides.pptx` (5 slides, built by `build_slides.py`, figures by `make_slide_figures.py`, outputs in `slides_figs/`):
1. OffCEM setup. 2. Joint LP. 3. Results, fixed exclusions. 4. Results, original exclusions. 5. Split version bounds.
Slides 3–5 show dashed lines for IPW, DR, OffCEM and the true value. The deck was validated but not rendered, so equations and text fit are unchecked. Slide 5 still shows the pre-fix split results (the ε = 2 bar misses the truth). The user's own `Dist_OPE.pptx` (slides 22–26) reuses these plots; its notes on 22–26 are stale.

## 6. Open items
- ε and the outcome range [−30, 30] are assumptions.
- Slide 5, its notes and `slides_figs/split_bounds.json` need regenerating with `recentre="cluster_ips"` (not done).
- `GLOBAL_JOINT_LP_SPEC.md` does not describe the joint LP.
- `round_random` has only 8 worlds.
- Slide wording: the code uses ρ = ε/2 per pair, while slide 23 writes |μ − m̂ − b| ≤ ε.

## 7. Update 2026-11-07: B3 with η = ρ, and on the earlier (by-round) setup — plots 08 and 09

Question: how much does the Lipschitz belief B3 (references + coupling) buy when the level band η is tied to ρ instead of fixed at η̂, and on the setup of plot 04 (by-round folds)? One world, seed 7000, K-means (K = 10), original exclusions, λ = ∞, ρ ∈ {5.85, 7.47}, L ∈ {3, 3.5, 4.4}.
`b3_lipschitz_experiment.py` gained `--eta {hat,rho}`, `--folds {user,round}`, `--out` (defaults unchanged). Plots: `plot_b3_eta_rho.py` → `slides_figs/08_...`, `plot_b3_round_folds.py` → `slides_figs/09_...`. Data: `results_offcem_global/b3_lipschitz_seed7000_eta_rho.json`, `..._eta_rho_roundfolds.json`.

DR full-value widths (all bars contain V = 5.95):

| setup | ρ | baseline | +B3 L=3 | L=3.5 | L=4.4 |
|---|---|---|---|---|---|
| by-user folds, η = η̂ = 7.38 (plot 06) | 5.85 | 13.5 | 10.9 | 11.9 | 13.0 |
| by-user folds, η = ρ (plot 08) | 5.85 | 12.0 | 10.2 | 11.0 | 11.7 |
| by-round folds, η = ρ (plot 09; plot-04 bar 12.0, merged 11.2) | 5.85 | 11.2 | 9.8 | 10.5 | 11.0 |
| by-user folds, η = η̂ (plot 06) | 7.47 | 15.1 | 11.5 | 12.7 | 14.1 |
| by-user folds, η = ρ (plot 08) | 7.47 | 15.2 | 11.5 | 12.8 | 14.2 |
| by-round folds, η = ρ (plot 09; plot-04 bar 15.1, merged 14.4) | 7.47 | 14.4 | 11.3 | 12.4 | 13.7 |

Findings: (1) η = ρ vs η̂ only matters at ρ = 5.85 (about 1–1.5 narrower); at 7.47 the two η values coincide. (2) B3 removes 1–2 units at ρ = 5.85 and 1–4 at ρ = 7.47, independently of η and of the fold scheme; the gains add up rather than interact. (3) Smallest usable L is 3 (reference boxes contradict below), which is already the 99th percentile of the true slopes, so B3 cannot be pushed further. (4) Coupling (unsupported-to-unsupported) changes nothing beyond the reference boxes (≤ 0.03). (5) With by-round folds, merging μ to one variable per (user, action) is itself ~0.8 tighter than the per-round closed form; plot 09 shows that step separately. All of this is one world; no multi-seed coverage.

## 8. Update 2026-11-07 (later): B5 group mean bounds and B7 favourite-action dominance, with and without recentring — plot 10

Setup: the plot-04 setup (K-means K = 10, original exclusions, λ = ∞, seed 7000, by-round folds, η = ρ), ε = 5.85 and 7.47. Script `scripts/offcem_synthetic/b5_b7_experiment.py`
(reuses `b3_lipschitz_experiment.prep`), data `results_offcem_global/b5_b7_seed7000_roundfolds.json`, plot `slides_figs/10_...`. All LPs are the merged LP (one μ per user-action), HiGHS.

Beliefs added to baseline (range + relaxed LC + level band):
- **B3** |μ_ua − μ_va| ≤ 3·d(x_u, x_v), logged reference boxes (mean ± 1.96 σ̂/√n) + coupling (as plot 09).
- **B5** L_c ≤ μ_ua ≤ U_c per K-means cluster, from the cluster's logged (user, action) boxes. `range` = [min low, max high]; `q5` = [5th pct of lows, 95th pct of highs].
  Oracle check: true means outside the bounds = 2 % of LP mass (range), 7 % (q5).
- **B7** μ_(u,a*) ≥ μ_(u,a) for every other action a, a* = argmax_a π(a|x_u) (the target's favourite; the policy is known, no truth used). a* is an LP variable for 99 of 100 users.
  `ord` = ordering among LP variables only. `z3.5` = also μ_(u,a*) ≥ ȳ_ua − 3.5 σ̂/√n for the user's logged actions. With z = 1.96 the LP is infeasible: for two users the model
  under-predicts a* by more than ρ (e.g. true q(u,a*) = 17.7 vs LC ceiling 6.2), and the max over ~10 noisy logged values overshoots. Oracle: 0 dominance rows violated by true means.
- **Recentring** (`partial_ope_global.evaluate(recentre="cluster_ips")`): one constant shift of all identified cell levels so their policy-weighted sum equals the logged
  cluster-weighted residual π_U(c|x)/π0(c|x)(y − m̂). On this world the shift is **−0.42** (downward).

DR bars (supported DR 0.96 + LP, ± 1.96 SE), no recentring:

| ε | baseline (merged) | + B3 L=3 | + B5 range | + B5 q5 | + B7 ord | + B7 z3.5 | + B3(z3.5) + B5 q5 + B7 z3.5 |
|---|---|---|---|---|---|---|---|
| 5.85 | [−3.95, 7.25] w 11.2 | [−3.10, 6.72] w 9.8 | [−3.95, 7.24] w 11.2 | [−3.64, 6.92] w 10.6 | [−2.84, 7.17] w 10.0 | [−0.31, 7.17] w 7.5 | [0.23, 6.76] w 6.5 |
| 7.47 | [−5.54, 8.83] w 14.4 | [−3.74, 7.59] w 11.3 | [−5.41, 8.63] w 14.0 | [−4.32, 7.51] w 11.8 | [−4.42, 8.76] w 13.2 | [−0.66, 8.76] w 9.4 | [0.12, 7.28] w 7.2 |

With recentring every bar moves down by ≈ 0.2 (the −0.42 level shift, partly absorbed by the band); B7 z3.5 at ε = 5.85 becomes infeasible (the lowered LC ceiling on a* contradicts a logged lower bound).

Findings:
1. **B7 is the only belief that shifts the interval.** Ordering alone raises the DR floor by 1.1 at ε = 5.85; adding the user's own logged rewards as lower bounds on a* raises it by 3.6
   (ε = 5.85) and 4.9 (ε = 7.47). The ceiling is unchanged (7.17 vs 7.25): one-sided, as intended. V = 5.95 stays inside.
2. **B5 does almost nothing with an honest margin.** The cluster's logged range (±1.96 σ̂/√n per pair) spans roughly [−19, 19]; it trims 0.0–0.3. The trimmed 5–95 % version trims 0.6–2.6 but
   already excludes 7 % of the true mass.
3. **B3 and B7 conflict at the 1.96 margin.** One user's intersected reference box on a* (top 5.53) sits below the true mean (6.18) and below another action's box bottom, so B3 refs + ordering
   is infeasible. With z = 3.5 on the B3 boxes the combination is feasible and gives the narrowest bar: [0.23, 6.76] at ε = 5.85 (w 6.5, vs 11.2 baseline, vs 12.0 plot 04).
4. **Recentring does not help on this world.** The importance-weighted residual moment is −0.18, below the level sum, so the shift is downward. The 50-world gain in section 4.5 was under the
   split (user-fixed) exclusions with by-user folds; under by-round folds and original exclusions on seed 7000 it goes the other way. Single-world evidence only.
5. The infeasibilities are informative: they show that the LC window (±ρ around m̂ + b) is violated on exactly the favourite actions, which is where the target's mass sits and why the
   bars are centred low. B7 with logged evidence is a data-only way of pulling those cells up.
