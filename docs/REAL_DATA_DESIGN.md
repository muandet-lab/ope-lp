# Real-data arm: LP bounds on the paper's extreme-classification bandit (design note, 2026-11-08)

Purpose: run the unsupported-region LP on the OffCEM paper's real-data setup (Sec. 4.2) with as few changes as possible, and decide the points
that the synthetic setup never had to decide. Nothing here is implemented yet; the decisions marked **DECIDE** need a call before coding.

## 1. The paper's real-data pipeline as the released code runs it (`icml2023-offcem/src/real`)

| Piece | Paper code |
|---|---|
| Data | EUR-Lex 4K (3,956 labels after dropping rare ones) or Wiki10-31K (30,938). Extreme Classification Repository format (`train.txt`, `test.txt`), features sparse, labels multi-hot. |
| Contexts | Raw features → PCA to 100 dims → standardised. |
| Users | **Fixed pool of 300 contexts** sampled from the training split; each round draws a user uniformly from the pool. So contexts repeat (2000 rounds ≈ 6.7 per user; 8000 ≈ 27). Same structure as our synthetic world. |
| Expected reward | q(x, a) = sigmoid( y_a (1 − η_a) + (1 − y_a)(η_a − 1) ), y_a ∈ {0,1} the label, η_a ~ U(0, 0.2) per action. So q is two-valued per action: ≈ 0.27–0.31 if the label is absent, ≈ 0.69–0.73 if present. |
| Reward | Bernoulli(q). Range [0, 1]; noise variance q(1 − q) ≤ 0.25. |
| Logging policy | softmax(β · r̂(x, a)), β = 30, where r̂ is a ridge regression (per action) fitted on 2,000 **test** contexts to predict q. A learned, peaked logger; every action has π0 > 0. |
| Target policy | ε-greedy on the TRUE q, ε = 0.1: 0.9 on the top-1 true label, 0.1 uniform. |
| Clusters | 100 clusters, **uniform random** assignment of actions (`train_clustering`), the paper's stated heuristic. |
| Reward model | Two-step: pairwise network + MLP(30,30,30) cluster regression; fitted in-sample. |
| Truth | V computed on the same 300 users' q with the target's action distribution — already on the logged population. |
| Metric | normalised MSE over 100 seeds, n_val ∈ {1000, 2000, 4000, 8000}. No deficient actions. |

## 2. What carries over unchanged
- Fixed users with repeats → the within-(user, cluster) contrast calibration of ε_max and the per-user beliefs B5/B7 apply as they are.
- Truth on the logged users → containment checks are exact (q is known for every pair from the label matrix).
- Merged μ per (user, action), per-round level b, closed-form baseline, B5, B7 ordering and logged bounds, the ε sweep and the plot scripts.
- Cluster-level ridge centre ĝ (OffCEM's cluster effect), cross-fitted by user. The two-step ĥ is expected to be noise again (≈ 2000–8000 rounds over 3,956 actions); it can be run as a comparison arm as at scale.

## 3. What has to change

### 3.1 The unsupported region is defined by weak overlap, not by π0 = 0  — **DECIDE λ**
No action is excluded, so R² = {π0 = 0} is empty. The LP region must be R² = {π/π0 > λ}: pairs the logging policy would show at least λ times less often than the target.
With β = 30 and r̂ ∈ (0.3, 0.7), logging probabilities span about five orders of magnitude, so a large share of the target's 0.1 uniform mass sits on actions with π0 ≈ 1e-5
(weights ≈ 10^3), and the favourite (0.9 mass) is supported only when the logger's ridge also ranks it near the top. Proposal: λ = 10 as the main setting, λ ∈ {3, 10, 30} as a sweep; report
LP mass for each. This is exactly the λ mechanism the synthetic results pointed to, and here it is the definition rather than an option. The decomposition V = V⁺ + V⁰ and the draft's Theorem 1 are unchanged.

### 3.2 Noise variance without repeats  — **DECIDE plug-in**
Bernoulli noise variance is q(1 − q), unknown per pair. For the contrast calibration D = τ² + E[noise var], use the plug-in m̂(1 − m̂) from the cross-fitted centre (per logged round), averaged over the
contrast pairs. Alternative: the bound 0.25, which over-subtracts and shrinks ε_max (since q is near 0.3 or 0.7, q(1−q) ≈ 0.21, so the bound is close anyway). Proposal: plug-in, report both.
Reference intervals for B5/B7 become Wilson or Wald binomial intervals on ȳ_ua with the same z = 3.5 rule; σ̂ is no longer estimated from repeats.

### 3.3 Outcome range
[0, 1] replaces [−30, 30]. This alone makes the baseline much tighter than in the synthetic world; the LC and band windows are clipped to [0, 1].

### 3.4 Rounds  — **DECIDE n_val**
The paper's default is 2,000 (≈ 6.7 rounds per user over 100 clusters → ≈ 0.07 rounds per (user, cluster) cell, far too sparse for contrasts). Its sweep goes to 8,000 (27 per user, ≈ 0.27 per cell,
≈ 300 users × 100 cells × 0.27²/2 ≈ 1,100 contrasts). Proposal: 8,000 as the main setting (inside the paper's sweep), and report the contrast count; if it is below ~1,000 the ε_max estimate is fragile and we say so.
Alternative: fewer clusters (the paper's 100 is a heuristic), e.g. 30, which multiplies contrasts by ≈ 11 — but changes the paper's setting.

### 3.5 Clusterings to compare
There is no DGP clustering in real data. The contrast is: (a) the paper's uniform random clusters (the baseline, by construction uninformative); (b) clusters learned from label co-occurrence in the
training split (SVD of the label matrix columns → K-means, 100 clusters), which groups labels that tend to appear on the same documents and is the natural "informed" clustering; (c) optionally
agglomerative / spectral on the same label embeddings. Expectation: (b) gives a smaller ε_max and tighter bounds than (a), which is the real-data version of the DGP-vs-learned contrast.

### 3.6 Favourite action under multi-label truth
Documents carry several positive labels (all with q ≈ 0.7); the target's top-1 is one of them (ties broken by rank order). B7's ordering μ(u, a*) ≥ μ(u, a) is then true with equality for the other
positive labels — still valid, no margin, as before.

### 3.7 Size and cost
EUR-Lex: 300 users × 3,956 actions ≈ 1.2 M (user, action) pairs; under λ = 10 most of them are in the LP. Closed-form baseline: seconds. B5: per-variable bounds, seconds. B7: couples actions within a
user only, so without B3 the LP separates into 300 per-user problems of ≤ 3,956 variables: minutes. **B3 is dropped** (cross-user coupling; inert at scale and would make a single 1.2 M-variable LP).
Wiki10-31K (31 k actions, 9 M pairs) is feasible for the closed form and per-user B7 but slow; start with EUR-Lex only.

## 4. Data
Not in the workspace. EUR-Lex 4K from the Extreme Classification Repository (Bhatia et al.), files `train.txt` / `test.txt` in the repository's sparse format; the paper's loader expects
`<repo>/data/eurlex/{train,test}.txt`. Place under `ope-lp/data/eurlex/` (git-ignored). Wiki10-31K likewise under `data/wiki/`.

## 5. Implementation plan (after the DECIDE items)
1. `real_world.py`: wrap the paper's `ExtremeBanditDataset` into our world dict (X_unit per user, q_unit, pi0_unit, pi_unit, logged rounds, action labels) with a λ-defined U; keep the paper's
   PCA/standardisation, reward transform, logger (β = 30) and target (ε = 0.1) untouched.
2. Calibration: binomial plug-in for the noise term; binomial reference intervals.
3. Driver flag `--real eurlex --lam 10 --clusterings random cooccur` reusing `scale_experiment.py`'s sweep, beliefs (B5, B7) and plots.
4. Report: ε_max, LP mass, baseline and B7 bars per clustering, containment of V and of V⁰, with the paper's OffCEM point estimate alongside.

## 6. Status (2026-11-08, evening): implemented and plumbing-tested
Decisions taken: λ = 10 (weak overlap defines the LP region), plug-in Bernoulli noise term for calibration, 8,000 rounds, reference intervals with the Bernoulli sd bound 0.5 and z = 3.5.
Code: `real_dataset.py` (paper's loader + `data_dir`), `real_world.py` (world wrapper, binomial calibration, label-co-occurrence action embedding, fake-data generator),
`scale_experiment.py --real eurlex --lam 10 --alpha cv --eps-min 0.02 ...`. Two additions forced by the real data and applied generally:
- `--alpha cv`: the ridge penalty (cluster-level centre and level ridge) chosen by out-of-fold squared error on the logged rounds. With 100 PCA dimensions and ~100 rows per cluster the
  default α = 1 extrapolated to predictions outside [0, 1] (−1.3 to 1.7) and made every LP infeasible; α = 1000 was chosen on the fake data. Synthetic runs keep α = 1 unless asked.
- the centre m̂ is clipped to the outcome range (a prediction of a mean in [0, 1]).
Tested end to end on a fake extreme-classification dataset (`real_world.make_fake_xc`, same text format): λ = 10 puts essentially each user's favourite action into the LP (203 variables for 100
users), the paper's random clusters and K-means both run, B5/B7 feasible, plots render. Fake-data numbers are not results. The unit→user mapping in `b3_lipschitz_experiment.prep` was fixed for
user-level worlds in the process (no effect on round-level worlds).
Run on EUR-Lex (files under `ope-lp/data/eurlex/{train,test}.txt`):
    python scale_experiment.py --real eurlex --n 8000 --users 300 --C 100 --lam 10 --folds user --alpha cv --eps-min 0.02 --clusterings random kmeans agglomerative spectral --beliefs B5_range B5_q5 B7_ord B7_z3.5 B5_q5+B7_z3.5 --tag eurlex_lam10
