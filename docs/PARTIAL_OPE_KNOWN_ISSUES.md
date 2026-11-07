# Partial-identification OPE: known issues at the current baseline

Baseline = the scripts in `Test notebooks/scripts/` and the runs recorded in `PARTIAL_OPE_EXPERIMENT_LOG.md` (all seed 0, joint LP, degree 2, `k = 0`).
Source tags: **[log]** from the experiment log, **[code]** from reading the scripts, **[slides]** from the 20 Oct 2026 update (`project-Partial-OPE-262010.pdf`).
Nothing here has been fixed; this is a record only.

---

## A. Validity of the results

1. **Coverage rests on a generous ε, not on an accurate centre.** [log]
   The 2D width is about 2ε(1 − γ)·(unsupported share), so it is set by ε = 1.0 and the weights, not by the data. Median |centre error| is 0.17–0.24, p95 0.46–0.71, max up to 1.75. "Bound contains the oracle in all 36 runs" is therefore partly a consequence of ε being large.
2. **ε and L are configured, not calibrated.** [log, code]
   `epsilon_source: explicit configuration, not oracle calibration`. No sweep over ε or L exists, so the coverage-vs-width trade-off is unknown.
3. **Only one seed per configuration.** [log]
   Coverage across seeds is untested.
4. **Only 2D is informative.** [log]
   2A and 2B are essentially vacuous (width 2–11). 2C works for dX = 2 with a low unsupported share, and loses power as covariate dimension and unsupported share grow (dX = 5, dT = 5: width 4.5–6.1). At d = 10 in the orthant world 2C is about equal to 2A.
5. **The belief block is false for most rows.** [log]
   `true_parameter_violations["2A"].cumulative_any_violated` is 0.66–0.92, mainly `pC ≥ lower` (0.50–0.61 of rows) and `pA ≤ upper` (0.27). Cause: sd-3 noise in unlogged cells makes the fitted belief interval too narrow or mis-centred. Coverage survives only because 2C/2D constraints dominate.
6. **2B (ratio constraint) carries no information in any multi-treatment world.** [log]
   With k = 1 about 9% of rows have true k·pA > pC and 2B is infeasible, so `--k 0` was adopted everywhere from phase 1. This is a choice, not a fix, and the ratio constraint is untested there.
7. **Envelope violations of the true mean at dX = 2.** [log]
   4–13% of rows in binary (2,2) and three-level (2,1), unexplained beyond sparse references in a thin region.
8. **Cubic outcome model fails at d = 10** (2D infeasible with 622 treated rows). [log]
   Degree 2 is the working default; the cause is variance and extrapolation, not misspecification.

## B. Assumptions baked in

9. **Policy probabilities are assumed known** (IPW weights are exact). [log, code]
   The supported-part test therefore checks variance only. Effective sample fractions were 0.39–0.81. Estimated propensities are untested.
10. **Outcome limits are the sample min and max.** [log, code]
    `l = floor(min y)`, `u = ceil(max y)`, although the noise is Gaussian. The event thresholds −3 and 0 are hardcoded into every row constraint, which is a three-cell discretisation of the slides' general events B_(h,t,x). [slides]
11. **The multi-treatment baseline for beliefs is the logged action at x_i.** [code]
    This is valid only if adding treatment components is non-harmful.
12. **The outcome model is additive across treatment components.** [code]
    That is true in this DGP and assumed in the LP; it may not hold elsewhere.
13. **The orthant LP only supports nested deterministic policies.** [code]
    `dgp.generate_data` and `separate_lp.prepare_problem` assert this; the unsupported mask is not a general support detector.
14. **Stochastic logging inflates noise by construction.** [code]
    In `dgp_multival.py`, randomised logged rows (t ≠ L0) have sd 3. This is part of why 2A–2C are wider under stochastic logging.

## C. Coverage of the experiment space

15. **Binary dT = 10 not run; three-level dT = 10 infeasible** (3^10 actions, no logged reference for most). [log]
16. **Three-level (5,5) at 50k is the heaviest run** (175k LP variables). [log]
17. **Unsupported share explodes with dT** (dT = 5: 66% binary, 88% three-level). [log]

## D. Script and CLI footguns

18. **Defaults disagree with how runs were actually made.** [code]
    `multi_lp.py`, `multival_lp.py` and `everywhere_lp.py` default to `--k 1.0`; the log's runs all used `--k 0`. `separate_lp.py` defaults to `--degree 3` (the setting that failed at d = 10), while the multi scripts default to 2. Commands must pass `--k 0 --degree 2` explicitly.
19. **Bare `--separate-stages` silently skips the separate LPs** (`nargs="*"`). [log, code]
    This caused the earlier `KeyError: '2D'` in the summary script. List the stages (`2C 2D`) to run them.
20. **Belief-repair knobs are not exposed in the multi CLIs.** [code]
    `good_margin` and `bad_margin` exist in `LPConfig` and in `separate_lp.py`'s flags, but not in `multi_lp.py`, `multival_lp.py` or `everywhere_lp.py`. Fixing issue 5 via margins needs a CLI change.
21. **Shell pitfalls.** [log]
    `conda run` does not forward stdin, and zsh does not word-split unquoted variables (matters in shell loops).
22. **Minor.** [code]
    - `plot_degree_compare.py` hardcodes the d = 2/5/10 result paths.
    - `data_checks` in `multival_lp.py` and `everywhere_lp.py` call `summarize` repeatedly.
    - The `multi_lp.py` docstring says "deterministic policies" although `prepare` now supports stochastic ones.
    - `diagnose_2c_2d.py` and `diagnose_models.py` are orthant-only and are not listed in the log's script table.
    - `diagnose_models.py` contains an oracle-form benchmark model (`param3_BENCHMARK`); it is labelled benchmark-only and must not be used as a method.

## E. Gap between the scripts and the slides' remaining tasks [slides]

23. No estimation procedure with theoretical guarantees; the LP is a plug-in on fitted nuisances.
24. No partial-ignorability, "relaxed assumptions of other point estimators" or knowledge-graph constraints.
25. No real-data or sparse high-dimensional use case.
26. Continuous Y is handled only through the three-cell discretisation (issue 10).
27. Technical novelty is not yet articulated.

## F. Documentation discrepancy

28. The log cites an "8-case plan, page 13 of `project-Partial-OPE-262010.pdf`". The 15-page PDF has no such plan; slide 13 (numbered 150) is the Remaining Tasks list. The plan may come from another version or from conversation.
