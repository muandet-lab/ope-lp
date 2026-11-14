# Real-data experiments: EUR-Lex 4K and Wiki10-31K

How the real-data experiments are built and what they gave. The arm is the OffCEM paper's extreme-classification bandit, with the unsupported-region LP on top.

Last updated 2026-11-13. Sources:
- code: `scripts/offcem_synthetic/real_world.py`, `real_dataset.py`, `scale_experiment.py --real ...`, `b3_lipschitz_experiment.prep`;
- result files under `results_offcem_global/` (listed in §11).

Every number below was read from those files or recomputed on the data.

> **Read §2 first.** Until 2026-11-13 the real-data rewards were generated with a bug that made them depend on the random number used to draw the action. Every real-data result before that date is invalid: the 2026-11-10 "final" runs, and the tables in the first version of this file. §8–§10 are the corrected results.

---

## 1. Pipeline in one view

```
Extreme-classification dataset (train/test docs, sparse features, multi-hot labels)
  -> paper's loader: drop rare labels -> actions; PCA/SVD(100) contexts; q(x,a) = sigmoid(label transform with per-action noise)
  -> fixed pool of 300 users (training docs); 8,000 rounds, one user drawn uniformly per round
  -> logging policy pi0 = softmax(8 q)                       [design choice; paper: softmax(30 r_hat)]
  -> Bernoulli rewards Y ~ Bern(q(x, a)), independent of the action draw   [fixed 2026-11-13, §2]
  -> target policy pi = 0.1-greedy on the true q              [paper]
  -> estimation clusters (100): paper's random clusters, or K-means / agglomerative / spectral on label co-occurrence
  -> split (user, action) pairs: supported if pi/pi0 <= lambda = 100, else LP region (= each user's favourite)
  -> supported part: OffCEM point estimate +- 1.96 SE
  -> LP region: min / max of sum pi(a|x) p(x) mu_{x,a} under range, relaxed local correctness, level band, extra beliefs
  -> bar = point estimate + [LP min, LP max], widened by 1.96 SE; compared with the true V
```

---

## 2. The reward-generation bug (found and fixed 2026-11-13)

**The bug.** In `real_world.make_real_world`, the `true_q` logger drew the actions with `sample_action_fast(..., random_state=seed + 7)` and the rewards with `np.random.RandomState(seed + 7).binomial(...)`. The two streams started from the same seed, so each round's reward reused the uniform number that had picked its action.

**What it did.**
- The reward was a function of where that uniform fell in the cumulative π0, not an independent Bernoulli(q). Measured: corr(u, y) = 0.69, P(y = 1 | u > 0.7) = 0.96, P(y = 1 | u < 0.3) = 0.08.
- Two rounds of the same user with nearby action indices got correlated rewards. Within-cell contrasts therefore fell *below* the Bernoulli noise floor. With K-means clusters, D(y) = P(y_i ≠ y_j)/2 was 0.177, while independent rewards can't go below 0.197.
- That shrank the calibrated ε_max, most visibly with learned clusters, where it collapsed.

**The fix.** The reward stream now uses `seed + 8`. Actions are unchanged; only the rewards are redrawn.

**Check after the fix** (EUR-Lex, true q):

| | Value |
|---|---|
| corr(u, y) | −0.006 |
| D(y), all four clusterings | 0.209–0.218 |
| True mean noise q(1 − q) | 0.206 |
| D with the true q as the centre | 0.203–0.213 |

So the contrasts now sit where independent Bernoulli rewards put them.

**Scope:** every run with `--logger true_q`, on both datasets. The paper's own logger (`--logger paper`) wasn't affected, but it isn't used (§4).

---

## 3. Data

| | EUR-Lex 4K | Wiki10-31K |
|---|---|---|
| Source | Extreme Classification Repository (Bhatia et al.), BoW version | same |
| Files | `data/eurlex/{train,test}.txt` | `data/wiki/{train,test}.txt` (zip in `data/wiki_dl/`) |
| Train / test docs | 15,539 / 3,809 | 14,146 / 6,616 |
| Features | 5,000 (densified) | 101,938 (kept sparse) |
| Labels in the file | 3,993 | 30,938 |
| Rare-label filter (paper) | ≥ 1 occurrence over train + test | ≥ 9 occurrences |
| **Actions K** | **3,956** | **5,501** |
| Positive labels per user (median) | 6 | 14 |

Format: a header line `n_docs n_features n_labels`, then one line per document `label,label,... feat:value ...`.

EUR-Lex was obtained with:
```
gdown --fuzzy "https://drive.google.com/file/d/0B3lPMIHmG6vGU0VTR1pCejFpWjg/view?usp=sharing&resourcekey=0-SurjZ4z_5Tr38jENzf2Iwg"
```
then `Eurlex/eurlex_{train,test}.txt` renamed to `data/eurlex/{train,test}.txt`. The data folder is git-ignored.

---

## 4. Bandit construction

### The paper's loader (`real_dataset.ExtremeBanditDataset`)

`real_dataset.py` is a copy of the paper's `src/real/dataset.py`. It adds `data_dir` and keeps wide feature matrices sparse.

1. **Actions.** Load both splits and drop rare labels. The remaining labels are the actions.
2. **Per-action noise.** η_a ~ U(0, 0.2), drawn once per action.
3. **Users.** Sample 8,000 training documents and keep the first 300 as the fixed user pool.
4. **Contexts.** Reduce to 100 dimensions and standardise: PCA for EUR-Lex, TruncatedSVD for Wiki (the dense matrix would be about 17 GB).
5. **Expected reward.** q(x, a) = sigmoid( y_{x,a}(1 − η_a) + (1 − y_{x,a})(η_a − 1) ), with y_{x,a} the label. This gives about 0.27–0.31 when the label is absent and about 0.69–0.73 when present; the observed range is 0.269–0.731.
6. **Rounds.** 8,000 rounds, each drawing a user uniformly (about 27 per user). The action comes from π0 and the reward is Y ~ Bernoulli(q).
7. **Units are users.** π0 and π depend only on the user. No action is excluded (π0 > 0 everywhere).

### Logging policy: `--logger true_q --beta 8` (a design choice)

π0 = softmax(8 q), a softmax over the true q, in the style of the paper's synthetic generator.

The paper's real-data logger, softmax(30 r̂) with r̂ a ridge fitted on 2,000 test contexts, logs a true positive in only 0.2% of rounds. Even uniform logging gives 0.1%, because each document has a few positives among thousands of labels.

At β = 8, recomputed on the run configuration:

| | EUR-Lex | Wiki10-31K |
|---|---|---|
| Logged rounds on a true positive | 3.5% | 6.2% |
| Users whose favourite was ever logged | 17% | 14% |

The value "16%" in `FINAL_FORMULATION.md` is the β = 12 figure.

### Target policy (the paper's)

ε-greedy on the true q, ε = 0.1: 0.9 on the user's best action and 0.1 uniform over all K actions.

| | V | V⁺ (supported) | V⁰ (LP part) |
|---|---|---|---|
| EUR-Lex | 0.674 | 0.029 | 0.646 |
| Wiki10-31K | 0.682 | 0.029 | 0.653 |

---

## 5. Estimation clusterings (C = 100)

| Name | Construction |
|---|---|
| `random` | The paper's: each action assigned uniformly at random to one of 100 clusters. |
| `kmeans`, `agglomerative` (Ward), `spectral` | Fitted on a label co-occurrence embedding: TruncatedSVD(50) of the transposed *training* label matrix, rows L2-normalised. No rewards and no test data are used. |

The detailed progressions in §9–§10 use **K-means**. On EUR-Lex all four clusterings were also run (§8).

---

## 6. Supported vs unsupported pairs (`--lam 100`)

A pair goes to the LP if π(a | x)/π0(a | x) > λ.

**At λ = 100,** the LP holds exactly each user's favourite action: 300 variables and 0.90 of the policy mass, on both datasets. OffCEM's estimate of V⁺ is 0.030 ± 0.020 (EUR-Lex) and 0.030 ± 0.029 (Wiki), against a true 0.029.

**λ sweep** (EUR-Lex, K-means, ridge first stage):

| λ | LP variables | LP mass | OffCEM V⁺ | Bars |
|---|---|---|---|---|
| 10, 30, 100 | 300 | 0.90 | 0.030 ± 0.020 | identical at every ε |
| 300, 1000 | 2 | 0.005 | **0.508 ± 0.078** (true 0.673) | **all miss V**: [0.43, 0.59] |

The favourites have π/π0 between 100 and 300. Above that, they count as supported, OffCEM must estimate them with importance weights of 100–300, and it under-estimates by about 0.17. **λ = 100 is the setting.**

---

## 7. Reward model and LP

### Reward model (centre of local correctness)

| Item | Setting |
|---|---|
| Main model | `--model two_step --g-model ridge`: OffCEM's f̂ = ĝ(x, c_a) + ĥ(x, a), with ĝ one ridge per cluster on the user context and ĥ OffCEM's pairwise model (s = 1). |
| Comparison | `--g-model mlp`: the paper's first stage, OBP's MLP(50, 50, 50) with the cluster as the action. |
| Cross-fitting | by user, 5 folds |
| Ridge penalty | `--alpha cv`: out-of-fold MSE over {0.1, 1, 10, 100, 1000}. The error falls monotonically on both datasets, so α = 1000 is chosen (the edge of the grid). The same α is used for b̂. |
| Centre screening | `--clip-model outcome_cell`: each cell's level estimate is shifted so that m̂ + b̂ lies in [0, 1]. About 10–15% of pairs are shifted with the ridge, 31–34% with the MLP. |

**A second fix (2026-11-13):** the `--alpha cv` step didn't pass `--g-model`, so it always tuned the MLP version and fell back to α = 1000. It now tunes the model that is actually used. The chosen α is unchanged (1000), so no result moved. Without `--alpha cv`, the default α = 1 makes b̂ unstable on EUR-Lex: the bar SE grows tenfold (0.0095 → 0.0999) and bars leave [0, 1].

### LP variables and objective
- μ_{x,a}: one per unsupported (user, action).
- b_{x,c}: one per (user, cluster) cell.
- Objective: minimise and maximise Σ π(a | x) p(x) μ_{x,a}.

### Constraints

| | Constraint | Setting |
|---|---|---|
| (13) range | 0 ≤ μ_{x,a} ≤ 1 | Binary Y, so μ = P(Y = 1 \| x). |
| (14) relaxed local correctness | \|μ_{x,a} − m̂_{x,a} − b_{x,c_a}\| ≤ ε | ε = ε_max × {½, ¾, 1} (`--eps-fracs 0.5 0.75 1`) |
| (15) level band | \|b_{x,c} − b̂_{x,c}\| ≤ η | η = ε; b̂ is a ridge of the logged residuals on the user context, per cluster |
| Lipschitz (`B3c`, coupling only) | \|μ_{x,a} − μ_{x′,a}\| ≤ L̂·d(x, x′) | L̂ from the logs (`--L auto`): the 95th percentile of noise-corrected same-action slopes |
| Failure cap (`PB_fail<α>`) | P(Y(a) = 0 \| x) ≤ α, i.e. μ_{x,a} ≥ 1 − α | α ∈ {0.3, 0.4, 0.5, 0.6} (`--pb-fail-alpha`), alone and with Lipschitz |

**About the failure cap.** It replaces the synthetic severity beliefs `PB_sev` / `PB_ext`. With binary rewards every logged quantile is 0 or 1, so there is no separate "severe" or "extreme" loss to constrain. On binary outcomes a probability belief is exactly a bound on the mean, so the cap demonstrates how a probability belief moves the bounds, not a distributional effect. At λ = 100 it applies to the favourites only, since they are the LP variables.

**ε calibration** (`real_world.calibrate_binomial`). Within each (user, cluster) cell, take pairs of logged rounds with different actions, with residuals e = y − m̂. Then:
- D = mean of (e_i − e_j)²/2, and the noise term is the plug-in mean of (v_i + v_j)/2 with v = m̂(1 − m̂), clipped to [0, 0.25];
- τ̂² = D − noise, and ε_max = 2τ̂.

The noise term comes from the model, not the data (there are almost no repeated pairs), so it depends on the first stage (§8).

**Not used:** cluster-specific bounds (17), dominance (18), and slack. Infeasible items are recorded as infeasible.

**Solver:** HiGHS, no time limit. With 300 variables a solve takes under a second, and a full run takes 2–5 minutes per clustering.

---

## 8. Calibration results (after the fix)

### ε_max

| Dataset | Clustering | Contrast pairs | Ridge first stage: D | noise | **ε_max** | MLP first stage: D | noise | **ε_max** |
|---|---|---|---|---|---|---|---|---|
| EUR-Lex | random | 1,049 | 0.270 | 0.161 | **0.659** | 0.245 | 0.085 | **0.802** |
| | K-means | 1,411 | 0.271 | 0.157 | **0.674** | 0.246 | 0.076 | **0.823** |
| | agglomerative | 1,426 | 0.282 | 0.155 | **0.712** | 0.260 | 0.079 | **0.853** |
| | spectral | 1,553 | 0.284 | 0.146 | **0.743** | 0.252 | 0.075 | **0.841** |
| Wiki10-31K | K-means | 1,275 | 0.269 | 0.168 | **0.636** | — | — | — |

**Before the fix** (invalid, for reference), EUR-Lex ε_max with the ridge first stage was 0.40, 0.09, 0.18 and 0.19 (random, K-means, agglomerative, spectral). With the MLP first stage it was 0.64, 0.45, 0.48 and 0.49.

**Caveat on the noise term.** The true mean noise on the logged rounds is 0.206, and with the true q as the centre τ² is only 0.005, giving ε ≈ 0.14. Both plug-ins under-estimate the noise (ridge 0.15–0.17, MLP 0.08), so both calibrated ε_max values are inflated, the MLP's more. This inflation is part of why the base bars are wide enough to contain V. Any write-up should say so.

### Other calibrated quantities

| | EUR-Lex | Wiki10-31K |
|---|---|---|
| L̂ (`B3c`) | 1.010 (0.061 before the fix) | 0.373 |
| Lipschitz coupling pairs (users sharing an unsupported favourite) | 217 (50 shared labels) | 3,670 (33 shared labels) |
| Ridge penalty (CV) | 1000 | 1000 |
| Policy mass where the cap is false: α = 0.3 | 7.5% | 2.8% |
| α = 0.4 | 0.5% | 0% |
| α = 0.5, 0.6 | 0.5% | 0% |

The favourites have q ≈ 0.69–0.73, so their true failure probability is 0.27–0.31.

---

## 9. EUR-Lex 4K, K-means: full progression

Ridge first stage, λ = 100, ε_max = 0.674. OffCEM bars ± 1.96 SE, with V = 0.674. Ratios are W / W(base). Every feasible interval contains V.

| Constraints | ε = 0.34 (½ ε_max) | ε = 0.51 (¾ ε_max) | ε = 0.67 (ε_max) |
|---|---|---|---|
| (13) range | [0.01, 0.95], 1.10 | [0.01, 0.95], 1.00 | [0.01, 0.95], 1.00 |
| + (14), b free | [0.01, 0.95], 1.10 | [0.01, 0.95], 1.00 | [0.01, 0.95], 1.00 |
| base (13)–(15) | [0.02, 0.87], 1.00 | [0.01, 0.95], 1.00 | [0.01, 0.95], 1.00 |
| + Lipschitz | [0.02, 0.87], 1.00 | [0.01, 0.95], 1.00 | [0.01, 0.95], 1.00 |
| + cap α = 0.4 | [0.55, 0.87], **0.38** | [0.55, 0.95], **0.43** | [0.55, 0.95], **0.43** |
| + Lipschitz + cap 0.4 | [0.55, 0.87], 0.38 | [0.55, 0.95], 0.43 | [0.55, 0.95], 0.43 |
| + cap α = 0.3 | infeasible | [0.64, 0.95], **0.33** | [0.64, 0.95], **0.33** |
| + Lipschitz + cap 0.3 | infeasible | [0.64, 0.95], 0.33 | [0.64, 0.95], 0.33 |
| + cap α = 0.5 / 0.6 | [0.46, 0.87] / [0.37, 0.87] | [0.46, 0.95] / [0.37, 0.95] | same |

Local correctness is false on 59%, 30% and 15% of the policy mass at the three ε values. The true means satisfy the base at ¾ ε_max and above.

**ε up to 2 × ε_max** (`--eps-fracs 0.5 0.75 1 1.5 2`): at ε = 1.01 and 1.35 the bars are identical to ε_max (base [0.01, 0.95], cap 0.4 [0.55, 0.95], cap 0.3 [0.64, 0.95]). From ¾ ε_max on, the base is already the full range.

**The other clusterings** (ridge first stage, same items) behave the same:
- the base reaches [0.01, 0.95] at ¾ ε_max and above;
- cap 0.4 gives [0.55, 0.95], cap 0.3 gives [0.64, 0.95];
- with random clusters, cap 0.3 is infeasible at ½ ε_max; with agglomerative and spectral, every cap is feasible at every ε.

**MLP first stage, K-means:**
- ε_max = 0.823, grid 0.41 / 0.62 / 0.82;
- base [0.03, 0.89] at ½ ε_max, then [0.01, 0.95];
- every cap feasible at every ε, including α = 0.3 at ½ ε_max: [0.64, 0.89];
- local correctness false on 55%, 43% and 0%.

---

## 10. Wiki10-31K, K-means: full progression

Ridge first stage, λ = 100, ε_max = 0.636. OffCEM bars ± 1.96 SE, with V = 0.682. Every feasible interval contains V.

| Constraints | ε = 0.32 (½ ε_max) | ε = 0.48 (¾ ε_max) | ε = 0.64 (ε_max) |
|---|---|---|---|
| (13) range | [0.00, 0.96], 1.14 | [0.00, 0.96], 1.00 | [0.00, 0.96], 1.00 |
| + (14), b free | [0.00, 0.96], 1.14 | [0.00, 0.96], 1.00 | [0.00, 0.96], 1.00 |
| base (13)–(15) | [0.04, 0.88], 1.00 | [0.00, 0.96], 1.00 | [0.00, 0.96], 1.00 |
| + Lipschitz | [0.04, 0.88], 1.00 | [0.00, 0.96], 1.00 | [0.00, 0.96], 1.00 |
| + cap α = 0.4 | [0.54, 0.88], **0.40** | [0.54, 0.96], **0.43** | [0.54, 0.96], **0.44** |
| + Lipschitz + cap 0.4 | [0.54, 0.88], 0.40 | [0.54, 0.96], 0.43 | [0.54, 0.96], 0.44 |
| + cap α = 0.3 | infeasible | [0.63, 0.96], **0.34** | [0.63, 0.96], **0.34** |
| + Lipschitz + cap 0.3 | infeasible | [0.63, 0.96], 0.34 | [0.63, 0.96], 0.34 |
| + cap α = 0.5 / 0.6 | [0.45, 0.88] / [0.36, 0.88] | [0.45, 0.96] / [0.36, 0.96] | same |

Local correctness is false on 52%, 31% and 15% of the policy mass at the three ε values.

### EUR-Lex and Wiki side by side

| | EUR-Lex 4K | Wiki10-31K |
|---|---|---|
| ε_max | 0.674 | 0.636 |
| Base at ½ ε_max / ε_max | [0.02, 0.87] / [0.01, 0.95] | [0.04, 0.88] / [0.00, 0.96] |
| + cap 0.4 at ε_max | [0.55, 0.95], 0.43 | [0.54, 0.96], 0.44 |
| + cap 0.3 at ε_max | [0.64, 0.95], 0.33 | [0.63, 0.96], 0.34 |
| Cap 0.3 at ½ ε_max | infeasible | infeasible |
| Lipschitz effect | none | ≤ 0.002 |

---

## 11. Reading and caveats

**What the progressions show:**
- **The base only binds at ½ ε_max.** From ¾ ε_max on, (13)–(15) give the full range: the outcome range is narrow relative to the band.
- **(14) without (15) adds nothing,** as on the synthetic data, because a free b absorbs any μ.
- **Lipschitz adds nothing.** Only users sharing a favourite label are coupled: 217 pairs from 50 labels on EUR-Lex, 3,670 pairs from 33 labels on Wiki.
- **The failure cap does all the tightening.** It lifts the lower end from about 0.01 to 0.55 (α = 0.4) or 0.64 (α = 0.3), giving 0.33–0.44 of the base width.
- **One infeasible cell on both datasets:** the strict cap α = 0.3 at ½ ε_max. There, local correctness forces some favourites below 0.7. Neither λ nor a larger ε_max removes it, since the grid always includes ½ ε_max. α = 0.4 is feasible everywhere and false on at most 0.5% of the mass, so it is the recommended main setting.

**Caveats:**
1. **The logger is not the paper's** (§4). With the paper's logger the logs contain essentially no positives.
2. **The calibrated ε_max is inflated** by a model-based noise term below the true 0.206 (§8). Coverage of V partly relies on that inflation. A model-free noise estimate would give ε ≈ 0.14, and narrow bars that may miss V.
3. **Local correctness is false on 15–59% of the LP mass** across the grid. The bars contain V partly because the outcome range is narrow.
4. **Only K-means was run on Wiki,** and only the ridge first stage. The MLP first stage was run on EUR-Lex.
5. **One seed** (7000) and one draw of the data.

---

## 12. Commands and files

Run from `scripts/offcem_synthetic/`. These are the exact commands used on 2026-11-13; the code includes the reward-seed fix and the CV fix.

**Main progression** (EUR-Lex; for Wiki use `--real wiki`):
```bash
conda run -n offcem python scale_experiment.py --real eurlex --data-dir ../../data --logger true_q --beta 8 --eps-target 0.1 \
  --n 8000 --users 300 --C 100 --lam 100 --folds user --model two_step --g-model ridge --alpha cv --clip-model outcome_cell \
  --binomial-refs --L auto --eps-fracs 0.5 0.75 1 --eps-min 0.02 --time-limit 0 --pb-fail-alpha 0.3 0.4 0.5 0.6 \
  --clusterings kmeans \
  --beliefs B3c PB_fail0.3 PB_fail0.4 PB_fail0.5 PB_fail0.6 B3c+PB_fail0.3 B3c+PB_fail0.4 B3c+PB_fail0.5 B3c+PB_fail0.6 \
  --tag eurlex_ridgeg_fail_seedfix
```
The EUR-Lex run of this tag used `--clusterings random kmeans agglomerative spectral`.

**Variants:**
- **(13)+(14) row** (level band off): add `--eta-fixed 1e6 --eps-grid <the three grid values> --beliefs baseline`.
- **(13) row** (range only): add `--eta-fixed 1e6 --eps-grid 1e6 --beliefs baseline`.
- **λ sweep:** `--lam 10 | 30 | 300 | 1000`.
- **Doubled grid:** `--eps-fracs 0.5 0.75 1 1.5 2`.
- **MLP first stage:** `--g-model mlp`.

**New flags (2026-11-13):**
- `--eps-fracs`: ε grid as fractions of the calibrated ε_max.
- `--pb-fail-alpha`: failure caps on binary outcomes, items `PB_fail<α>` and `B3c+PB_fail<α>`.
- The JSON now records `b3c_possible_pairs` and `prob_beliefs_info.fail_cap`.

**Result files** (`results_offcem_global/`, each with a `.log` of the same name):

| File | Contents |
|---|---|
| `scale_eurlex_ridgeg_fail_seedfix_seed7000.json` | EUR-Lex, ridge first stage, 4 clusterings (§8–§9) |
| `scale_eurlex_mlp_fail_seedfix_seed7000.json` | EUR-Lex, MLP first stage, 4 clusterings |
| `scale_eurlex_ridgeg_km_noband_seed7000.json`, `scale_eurlex_ridgeg_km_rangeonly_seed7000.json` | EUR-Lex (13)+(14) and (13) rows |
| `scale_eurlex_ridgeg_km_lam{10,30,300,1000}_seed7000.json` | λ sweep |
| `scale_eurlex_ridgeg_km_lam100_2x_seed7000.json` | ε up to 2 × ε_max |
| `scale_wiki_ridgeg_km_seed7000.json`, `scale_wiki_ridgeg_km_noband_seed7000.json`, `scale_wiki_ridgeg_km_rangeonly_seed7000.json` | Wiki progression (§10) |

**Superseded (before the reward fix, invalid):**
- `scale_final_{eurlex,wiki}_lam100_twostep[_z35]_seed7000.json` (2026-11-10, old belief set B3/B5/B7);
- `scale_eurlex_ridgeg_fail[_cvfix]_seed7000.json`;
- `scale_eurlex_mlp_fail_seed7000.json`.

---

## 13. Differences from the synthetic arm

| | Synthetic (current) | Real data (current) |
|---|---|---|
| World | OffCEM generator, Gaussian reward σ = 3, 50% of actions excluded per round | extreme-classification labels, Bernoulli reward, no exclusions |
| Outcome range | [−30, 30] | [0, 1] |
| Logger | softmax(−0.1 q) with exclusions | softmax(8 q) |
| LP region (λ = 100) | π0 = 0 or π/π0 > 100: 20k (small) / 200k (full) variables | each user's favourite: 300 variables |
| Model | two-step, ridge first stage, clipped to the cluster's logged range | two-step, ridge first stage, centre shifted into [0, 1] per cell |
| ε calibration | contrasts minus repeat variance (model-free) | contrasts minus m̂(1 − m̂) (model-based) |
| ε grid | floor(ε_max) − 2, − 1, floor | ε_max × {½, ¾, 1} |
| Extra beliefs | Lipschitz coupling, `PB_sev`, `PB_sev+ext` (severity beliefs on a multi-level outcome) | Lipschitz coupling, failure cap P(Y = 0) ≤ α (exactly a mean bound on binary Y) |
| Ridge penalty | 1.0 | 1000 (CV) |
