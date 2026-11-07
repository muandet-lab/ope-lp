# Partial-identification OPE on the OffCEM DGP: setup and method

Status as of 2026-11-01. Companion files: `PARTIAL_OPE_EXPERIMENT_LOG.md` (earlier orthant / vector-treatment work),
`PARTIAL_OPE_KNOWN_ISSUES.md` (issues of that baseline), `icml2023-offcem-expts/experimental_setup.md` (plain-English OffCEM guide).
Code: `Test notebooks/scripts/offcem_synthetic/` (a verbatim copy of `icml2023-offcem-expts/src/synthetic/`, plus two new scripts).
Results: `Test notebooks/results_offcem_2d/`, `Test notebooks/results_offcem_lc/`. Every run so far is one seed.

Tags used below: **[code]** read from the scripts, **[paper]** from Saito, Ren & Joachims (ICML 2023, arXiv:2305.08062),
**[measured]** a number I computed in this project, **[inferred]** my reasoning, not tested.

---

## 1. What problem we are solving

Off-policy evaluation (OPE): estimate V(π) = E_x Σ_a π(a|x) q(x,a) for a new policy π from data logged under π₀.

Standard identification needs common support: π(a|x) > 0 ⇒ π₀(a|x) > 0. When it fails, the part of V(π) that sits on
unsupported pairs (x,a) with π₀(a|x) = 0 is not identified. We split

    V(π) = V⁺ + V⁰
    V⁺ = (1/n) Σ_i Σ_{a: π₀(a|x_i)>0} π(a|x_i) q(x_i,a)       identified (IPS/DR with known propensities)
    V⁰ = (1/n) Σ_{(i,a)∈U} π(a|x_i) z_ia,   z_ia = E[Y(a)|x_i]   not identified, U = {(i,a): π₀(a|x_i)=0}

(sample-context version; the oracle is the same average over the n logged contexts). OffCEM sidesteps this for a point estimate by
assuming local correctness; we instead ask for **bounds on V⁰** under assumptions written as **linear constraints on the unknown means z**
(the "γ-block" of the Partial-OPE slides, 20 Oct 2026). Goal stated by the user: bounds that are **tight and contain the true V(π)**.

## 2. The OffCEM data-generating process [code: `dataset.py`, `policy.py`, `conf/setting/default.yaml`]

Defaults: dim_context 10, n_actions 1000, 10 categorical action attributes with 5 values each, 50 clusters, n_val_data 3000 rounds,
n_users 200, continuous reward with σ = 3, logging β = −0.1, target ε = 0.2, n_def_actions 0 (no deficient actions by default).

* **Users.** A fixed pool of n_users contexts x ~ N(0, I₁₀); each logged round samples a user uniformly.
* **Actions.** Each action has a 10-dim categorical embedding (5 levels per dim, sampled from a softmax of N(0, 4²) logits), one-hot encoded
  with the first level dropped.
* **Clusters φ(a).** `compute_clusters`; the default `original` method is a random linear projection of the features, flattened by a
  softmax at temperature 10, then *sampled*. It is stochastic and only weakly feature-dependent (the repo found ~0 alignment with K-means);
  do not describe it as geometrically coherent.
* **Expected reward** q(x,a) = g(x, φ(a)) + h(x,a)  (the paper's conjunct effect model; the split itself is only a decomposition).
  * cluster effect g(x,c): OBP `polynomial_reward_function` (degree 3 in x and the cluster one-hot) + four threshold indicators.
    [measured] The indicators fire with probabilities ≈ 0.62, 0.50, 1e-5, 0.99, and are added identically to every cluster, so they shift
    all actions of a user equally and do not change either policy.
  * residual effect h(x,a): OBP `linear_reward_function`, seeded `random_state + c` per cluster, so coefficients differ across clusters:
    xᵀM_c·onehot_a + θ_{x,c}ᵀx + θ_{a,c}ᵀonehot_a.
* **Observed reward** y = q + N(0, 3²).
* **Logging policy** π₀(a|x) = softmax(β·q(x,·)) with β = −0.1 (slightly suboptimal, near-uniform).
* **Deficient actions** (our no-overlap mechanism): with `n_deficient_actions = D`, in every round a random set of K−D actions
  (Gumbel top-k, independent of x) gets π₀ = softmax over only those; the other D actions have **π₀ = 0 in that round**. We set D = def_frac·K.
* **Target policy** π: ε-greedy on the true q with ε = 0.2, i.e. mass 0.8 + 0.2/K on the best action, 0.2/K on every other. So π > 0
  everywhere and the unsupported mass is ≈ def_frac·(0.2 + 0.8·P(best action deficient)) [measured: 0.47–0.50 at def_frac 0.5; 0.10 at 0.1; 0.89 at 0.9].

Two generation paths exist: `obtain_batch_bandit_feedback` (stateful RNG; **each call draws a new world**, so `main_*.py` scores seeds against a
"true value" from a separate test call; [measured] two calls give different action features, clusters and contexts) and
`generate_fixed_world` + `sample_logged_stream` (reproducible worlds, but it **raises if `n_deficient_actions > 0`**).
We use the first, with the oracle taken from the *same* call.

Measured size of the pieces (K=1000, n=3000, 50 clusters): sd(q) 3.6, sd(g) 3.5, **sd(h) 1.0**, noise 3.0.
Spread of the true h inside a (round, cluster) cell: median 3.1, p95 4.8.

## 3. The paper's assumption: local correctness [paper, Assumption 3.1]

A regression model f̂ and clustering φ satisfy local correctness (LC) if for all x and all a, b with φ(a) = φ(b):

    q(x,a) − q(x,b) = f̂(x,a) − f̂(x,b).

Equivalently (Appendix C, Eq. 12): there is g(x,c) with q(x,a) − f̂(x,a) = g(x, φ(a)); the model error is constant inside a cluster at each x.
It needs no accuracy in absolute reward level and no relation across clusters. Under common cluster support (Assumption 2.2 with E = C:
π(c|x) > 0 ⇒ π₀(c|x) > 0) OffCEM is unbiased (Prop. 3.2). Theorem 3.3 gives its bias when LC fails: a sum over within-cluster pairs of
π₀(a|x,c)π₀(b|x,c)·(Δq − Δf̂)·(π(b)/π₀(b) − π(a)/π₀(a)). There is **no tolerance version** in the paper; the ε-relaxation is ours.
The other conditions: i.i.d. contexts, known π₀ and π, a fixed clustering; no action-level support and no "no direct effect" assumption.

## 4. Our formulation

### 4.1 Unknowns and constraints (script `run_partial_ope_lc.py`)

Per unsupported pair (i,a) ∈ U: unknown mean μ_ia (= z_ia). Per cell (round i, cluster c) containing an unsupported action: auxiliary g_ic.
Fixed inputs: a fitted model m_ia, an estimated cell level ĝ_ic with allowance η_ic, outcome limits [ℓ,u].

    ℓ ≤ μ_ia ≤ u                                                         outcome range
    |(μ_ia − μ_ib) − (m_ia − m_ib)| ≤ ε_LC      (φ(a) = φ(b), same i)    relaxed LC; ε_LC = 0 is the paper's condition
    |g_ic − ĝ_ic| ≤ η_ic                                                  estimated cell level

Pairwise LC over a cell is exactly equivalent to ∃ g_ic with |μ_ia − m_ia − g_ic| ≤ ε_LC/2 for all a in the cell (g = mid-range of the
residuals), so one auxiliary variable replaces all action pairs. Supported actions in the cell enter only through ĝ (their means are not
observed individually).

The program: minimise (maximise) V⁺ + (1/n) Σ_U π(a|x_i) μ_ia. Cells are independent and each is one-dimensional in g, so the optimum is
closed-form (ρ = ε_LC/2):

    G_lo = max(ĝ − η, max_a(ℓ − m_a − ρ)),   G_hi = min(ĝ + η, min_a(u − m_a + ρ))
    G_lo > G_hi  →  cell empty → INFEASIBLE (reported, never clipped)
    lower = Σ_a π_a max(ℓ, m_a + G_lo − ρ),   upper = Σ_a π_a min(u, m_a + G_hi + ρ)

[measured] closed form vs HiGHS on random cells: max abs difference ≤ 3e-7 in every run.
The earlier script `run_partial_ope_2d.py` is the special case "independent boxes |z − c| ≤ ε around c = m + ĝ" (stage "2D" of the earlier
2A–2D numbering: **2D is a label, not "two-dimensional"**); it needs no solver either.

### 4.2 Ingredients and the choices behind them

| Ingredient | Options in the code | Notes |
|---|---|---|
| Reward model m | `two_stage` (paper f̂ = ĥ + ĝ, trained in `ope.py`), `one_step`, **`cluster_level`**, `oracle`, `oracle_shift`, `oracle_noise` | `cluster_level` = cross-fitted regression on (x, cluster), **no per-action term**; `oracle*` use the true q and exist only as sanity checks |
| Cell level ĝ_ic | `--centre knn` (k nearest logged residuals in context, same cluster) or `ridge` (per-cluster ridge on polynomial features, prediction standard error) | the paper never estimates g this way; its IW term absorbs it |
| Allowance η_ic | `--eta` + `--se-mult`·SE(ĝ) (+ ρ with `--eta-add-rho`) | SE covers estimation noise only, not smoothness bias |
| Supported part V⁺ | `--supported dr` (default) or `ips` | DR is unbiased for V⁺ for any model that does not use the round's own reward |
| Outcome limits | default sample min/max of y; `--outcome-lower/upper` | a working assumption; true q already falls outside it |
| Cluster support | checked at x_i: π₀(c|x_i) > 0 whenever π(c|x_i) > 0 | unsupported cells get only [ℓ,u] and are counted |

### 4.3 What the oracle sanity models show [measured, small world]

* `oracle` (m = q): ε_LC = 0 is feasible; the pipeline works. The interval misses V only through the IPS error.
* `oracle_shift` (m = q − smooth cluster-constant shift): LC exact, level wrong; ε_LC = 2 already covers V.
* `oracle_noise` (m = q + iid noise): needs ε_LC ≈ the cell spread to become feasible.
A first version of `oracle_shift` drew an independent shift per round, which no estimator over contexts could recover; it is now smooth in x.

## 5. Results so far (single seed, small world: 200 actions, 1000 rounds, 100 users, 10 clusters, def_frac 0.5, oracle V = 3.316)

| Model / settings | Within-cell error spread (median) | Outcome |
|---|---|---|
| `two_stage` (paper pairwise ĥ) | 12.3 | infeasible for ε_LC ≤ 4; feasible from 8 ([−1.04, 3.33]) but 84% of true cells violate the beliefs |
| `cluster_level` (MLP), ridge ĝ | **3.08** | covers V from ε_LC = 4: [−0.56, 4.20], width 4.8, 34% cells violate; with `--eta-add-rho` [−1.50, 5.11], width 6.6, 21% violate |
| `oracle` | 0 | ε_LC = 0 feasible: [1.64, 2.85] |

Other measurements:
* Supported part: IPS 0.77, DR 0.59 (ridge) / 1.43 (MLP) against the oracle supported part 1.805; standard errors 0.8–1.05 (DR) vs 0.99 (IPS), max weight 136.
  At the default size the IPS se was 1.83 with max weight 634 [measured, no training].
* A kNN→ridge swap for ĝ did **not** change the trained-model failure (ĝ error 2.59 vs 2.59): the bottleneck was the model's within-cell spread.
* Default-size runs of the *trained* pairwise model did not finish within 10+ minutes and were stopped. `cluster_level` needs no pairwise training;
  its default-size run has not been done.

## 6. What limits tightness and validity [measured/inferred]

1. **Spread of h.** Valid ε_LC must cover the true within-cluster spread of h (median 3.1, p95 4.8). A per-action model that overfits (≈3–5 observations
   per action) adds error and inflates the spread to 12; a constant-in-cluster model leaves only h. The cluster-level error cancels in differences
   (it is absorbed by g, but only up to the allowance η, so ĝ still has to be estimated).
2. **η dominates the width.** Per-cell noise in ĝ is treated as simultaneously worst-case in every cell, so widths add up. Estimation noise should instead average
   out roughly like 1/√n in the total [inferred]. Proposed next step: keep ε_LC (and ρ) as deterministic assumption slack and put the sampling error of the
   aggregate into one confidence interval, together with the supported-part error. Not implemented.
3. **Supported part.** IPS/DR error (se ≈ 1–1.8) is as large as the unsupported width floor (mass × ε_LC ≈ 1.5 at def_frac 0.5). Heavy weights (max 136–634)
   limit what DR can do; clipping would add bias.
4. **Width scales with the unsupported mass** (floor ≈ mass·ε_LC: 0.3 at def_frac 0.1, 1.5 at 0.5, 2.7 at 0.9 [measured]). This is a property of the problem.
5. **ε_LC is not identifiable from this data.** With noise 3 and ~3–5 observations per action the spread of h (sd 1) cannot be estimated. Validity rests on an assumed
   ε_LC. In this synthetic world we can set it from the oracle diagnostic (labelled as such) and check coverage across seeds; on real data it needs external justification.
6. **Remaining misspecification:** outcome limits from the sample; known propensities; kNN/ridge borrowing across contexts (no smoothness assumption is made in the LP);
   in-sample two_stage/one_step models make the DR part slightly biased; cluster support is only checked at x_i.

## 7. Differences from the paper's use of LC

| | Paper | Ours |
|---|---|---|
| Role | exact condition for unbiasedness of a point estimator | linear constraint set for an interval |
| Tolerance | none | ε_LC (and η for ĝ) |
| Cell level g(x,c) | never estimated (cluster-level IW corrects it) | estimated (kNN/ridge) and bounded by η |
| Unsupported actions | handled by Σπ f̂ plus cluster-level IW | one unknown mean per pair |
| Policy weighting of the violation | Theorem 3.3 weights by π₀π₀ and policy ratios | worst-case ε_LC, no weighting |
| Cluster-level overlap | assumed | checked; cells without it get [ℓ,u] |

## 8. Scripts

| File | Purpose |
|---|---|
| `offcem_synthetic/dataset.py`, `policy.py`, `clustering.py`, `ope.py`, `main_*.py`, `run_*.py` | the repo's OffCEM code, copied unchanged |
| `offcem_synthetic/run_partial_ope_2d.py` | independent-box version (band around m + kNN residual) |
| `offcem_synthetic/run_partial_ope_lc.py` | pairwise-LC / g-form program above; `--cache` stores data and fitted models |

## 9. How to run (do not launch long jobs from here; the user runs them)

Run from `Test notebooks/scripts/offcem_synthetic`, environment `offcem` (Python 3.9, OBP 0.5.5).

Small world, cluster-level model (seconds):

    conda run -n offcem python run_partial_ope_lc.py --n-actions 200 --n-rounds 1000 --n-users 100 --n-clusters 10 \
      --def-frac 0.5 --model cluster_level --centre ridge --supported dr --eta-add-rho --eps-lc 2 4 5 6 8 \
      --cache ../../results_offcem_lc/cache_small.npz --output ../../results_offcem_lc/small_cluster_level.json

Default size, cluster-level model (no pairwise training; not yet run):

    conda run -n offcem python run_partial_ope_lc.py --model cluster_level --centre ridge --supported dr --eta-add-rho \
      --eps-lc 2 4 5 6 8 --cache ../../results_offcem_lc/cache_default.npz --output ../../results_offcem_lc/default_cluster_level.json

Paper two-stage model at default size is slow (> 10 minutes with 3000 rounds × 1000 actions); `--cache` avoids retraining when changing ε, η or the centre.

## 10. Open items
1. Separate assumption slack from sampling error (aggregate confidence interval instead of per-cell η).
2. Multi-seed coverage at the default size, and a sweep over def_frac and ε_LC (the user runs these).
3. Justified outcome limits instead of the sample range.
4. Check how the within-cell spread and the width change with n/K.
5. Whether the pairwise-difference form should also couple unsupported actions with a logged action b at the same x (currently only through ĝ).
6. Fold the OffCEM-DGP issues (per-call world, near-inert indicators, reward-range description in `experimental_setup.md`) into `PARTIAL_OPE_KNOWN_ISSUES.md`.

---

## 11. Update 2026-11-02: one global LP, user-fixed exclusions, sanity checks

New files in `scripts/offcem_synthetic/`: `partial_ope_global.py` (the program below) and `sanity_checks.py` (parts A-D). Results: `results_offcem_global/` (`SANITY_REPORT.md` has every table; the JSON files hold the raw numbers). All runs are small worlds (K=200, n=1000, 100 users, 10 clusters), not the default size.

### 11.1 The global LP
* **One sparse LP** holds every unsupported mean mu_ua and every shared cell level g_uc as variables; the cluster-level beliefs are ordinary rows (`--lp global`). HiGHS solves it once per bound: 11,000 variables in 0.2 s, 110,000 in about 4 s.
* The earlier per-cell closed form is the exact solution of this LP while it is block diagonal. They agree to 1e-12 on 400 random toy problems (including infeasible ones and cells without cluster support) and on 11 worlds (`--lp both` reports the gap).
* A global LP can also carry rows that couple cells. Implemented: `--avg-eps`, an average local-correctness budget `sum w |mu - m - g| <= avg_eps * sum w` (auxiliary variables t >= |.|, plus a per-pair cap). Nested and equal to the per-pair program when `avg_eps >= rho` (toy test T6).

### 11.2 The four fix ideas
1. **Model differences.** The diagnostics report the TRUE within-cell spread of the model error; `--eps-lc oracle:Q` sets eps_LC to its mass-weighted Q-quantile (labelled oracle; it cannot be estimated from the data). With the cluster-level model the median spread is about 2.8.
2. **Shared level.** `ghat` by ridge or kNN; allowance `eta = --eta + --se-mult * SE (+ eps_LC/2 with the default `--add-rho`)`. Part B: dropping the noise term (`se_mult=0`) gives V coverage 0.4 with 52% of cell mass violating; dropping the rho term gives 0.6.
3. **Supported part.** DR with a model cross-fitted BY UNIT (no unit's reward enters its own prediction; leakage test W5), IPS also reported. See 11.4: the estimator is unbiased but heavy-tailed.
4. **Exclusions.** `--exclusion user_fixed` draws one excluded set per user for the whole world, and the unknowns are shared by the rounds of a user (identical bounds to one unknown per round; test W6). In the dataset's per-round exclusions 4.9% of 'unsupported' (round, action) pairs (2.3% of the policy mass) are logged by the same user in another round, so numerically the effect is small here; conceptually the per-round world is not a permanent no-overlap problem.

### 11.3 What the checks establish (Part A: 44 pass, 0 fail, 1 informational)
Exact identities: global LP == closed form; sign symmetry; nesting in rho and eta; huge slack gives V+ + mass*[l,u]; exact LC with the true level and zero slack gives a point; oracle model + noiseless reward + eps_LC = 0 collapses onto V (width 0); probabilities, propensities and mass sum correctly; user-fixed exclusions are constant and never logged; cross-fitting has no leakage; the results are invariant to round order, cluster relabelling and action relabelling; identical runs are identical; the earlier saved result on the cached world is reproduced to 1e-16; edge cases (no unsupported pairs, tiny world, singleton clusters, more users than rounds, greedy/uniform targets, def_frac 0.95) run and agree.
At def_frac 0.9 with 10 clusters, 12.5% of the unsupported policy mass sits in cells whose cluster has no support at x; there only [l, u] applies, so the bounds cannot be tight, by design.

### 11.4 What Parts B-D show (small worlds)
* **The LP alone (V0) is robust.** Across 36 setups it covers the true unsupported part in every feasible run at eps_LC = 4 and at the oracle 95% spread (exception: no noise allowance, 0.9). Cell-level beliefs are nonetheless violated by the truth in 8-30% of cell mass (and 94% for a model whose errors are far larger than eps_LC), so aggregate coverage does not validate the beliefs: per-cell slack is conservative in the sum.
* **The LP is wide.** Median full width about 15 x unsupported mass at eps_LC = 4 (7.4 at mass 0.5). With the average-error belief and the level allowance tied to it, 4.2 (3 x oracle MAD) over 12 worlds, V0 covered 12/12; the budget is oracle-calibrated.
* **The identified part is what breaks total coverage.** IPS and DR are unbiased (200 worlds: mean error +0.08 +- 0.13 and +0.06 +- 0.12), but heavy-tailed: median -0.15, 5-95% range about [-2, +3.8], rms 1.7. In the 5.5% of worlds where the greedy action is never logged the error averages -2.1. Per-run standard errors cover only 77-79% of errors at +-2 se. With eps_policy = 0.2 and def_frac 0.1 the rms is 3.2 (IPS) / 2.6 (DR) against an LP width of 1.4; with eps_policy = 1.0 it is 0.07-0.14 and well calibrated. DR helps only a little.
* **Outcome limits.** The sample range of y excludes true means in 14 of 20 runs; a justified wide range ([-30, 30]) costs nothing here (width 7.53 -> 7.56) and removes the exclusions.
* **The reward model matters little for the LP** once `ghat` is estimated by ridge (a zero model gives similar V0 coverage and slightly narrower width); it matters for the supported-part DR.

### 11.5 Updated open items
1. Make the identified part reliable: a target with less concentrated weights, or an estimator robust to rare large weights, and an interval that is honest about heavy tails. This now limits total coverage more than the LP does.
2. Replace the per-cell worst-case allowance with an assumption-slack-plus-aggregate-confidence-interval split, or the average-belief form (11.1); both need a defensible budget that the data cannot identify.
3. Default-size multi-seed run (the user runs it; commands in `SANITY_REPORT.md` header and section 9): `python partial_ope_global.py` and `python sanity_checks.py --part B` with larger `BASE_W` are cheap; the paper's pairwise model is not used in the new program.
4. Justified outcome limits instead of the sample range.

---

## 12. Update 2026-11-02 (later): bounds that hold, and how much tighter they can get

Code: `partial_ope_global.py` (new options `--overlap-lambda`, `noise=aggregate`, `repair`, `eta_from_avg`, `avg_eps_mult`), `sanity_checks.py` parts E and F, `plot_bounds_vs_truth.py` (figure `results_offcem_global/bounds_vs_truth.png`). Part A was rerun after these changes: 44 passed, 0 failed, 1 informational (a counter bug that skipped numpy booleans was fixed; the saved JSON had always shown 0 failures).

### 12.1 Why total coverage failed, and the fix
The target puts 0.8 of its mass on one action per user that logging almost never takes (weight about 100), so the identified part was heavy-tailed (IPS/DR coverage 28% at 10% excluded). **Overlap threshold:** pairs with pi/pi0 > lambda are handed to the LP; IPS/DR keeps only pairs with weight <= lambda (an indicator truncation, so it stays unbiased for that part). Result (Part E, 40 worlds per row, eps_LC = oracle 95% spread): coverage of the full interval rises from 0.28 / 0.93 / 1.00 to **1.00 / 1.00 / 1.00** at 10% / 50% / 90% excluded, for any lambda from 2 to 30; the identified-part rms error at 10% excluded falls from 3.05 to 0.02. Cost: the LP carries 0.82-0.98 of the policy mass, so widths are 12-19 (values are 0-9).

### 12.2 Tighter: aggregate confidence interval for the shared-level noise (Part F)
Instead of a worst-case slack 2 SE in every cell, the LP runs at the estimated cell levels with only the deterministic slack eps_LC/2, and the standard error of the total contribution of the estimated levels (closed form, ridge covariance, clusters use disjoint rows) is added in quadrature to the supported-part standard error and applied once. Cells with no feasible beliefs are repaired by keeping only the outcome limits (counted; they occur in a few percent of runs).

| setting | A per-cell: coverage / width | B aggregate: coverage / width |
|---|---|---|
| target eps 0.2, 10% excluded | 1.00 / 12.4 | 0.95 / 7.5 |
| target eps 0.2, 50% excluded | 1.00 / 14.6 | 1.00 / 9.2 |
| target eps 0.2, 90% excluded | 1.00 / 18.6 | 1.00 / 13.5 |
| target eps 0.5, 50% excluded | 1.00 / 12.2 | 1.00 / 7.6 |
| target eps 1.0, 50% excluded | 1.00 / 7.8 | 1.00 / 4.7 |

About 35-40% narrower at the same coverage (40 worlds per row, so coverage is accurate to a few points). Variants that did NOT work: dropping the eps_LC/2 term (coverage 0.00-0.07 except at 90% excluded or target eps 1.0); the average-error belief with a budget set from the oracle mean absolute deviation (coverage 0.00 at 3x, and still 0.00 at 10x for 10% and 50% excluded; at 10x it reaches 0.97-1.00 elsewhere with widths of 8-10, no gain over B). The oracle mean deviation is degenerate when one action dominates a cell, so this belief is not usable as calibrated here.

### 12.3 What is and is not established
* Valid at about the 95% level in these small worlds when eps_LC is the oracle 95% quantile of the true spread. That quantity cannot be estimated from the data, so validity depends on an assumed eps_LC; the outcome limits are still the sample range (they exclude true means in most worlds; wide justified limits cost nothing here).
* Still wide: roughly 7-13 around values of 0-9 when the target is concentrated (eps 0.2). A target with less concentrated weights shrinks the LP mass (eps 1.0: width 4.7).
* Not yet done: default-size runs, estimating eps_LC from data, a justified outcome range, the average belief with a data-based budget.
