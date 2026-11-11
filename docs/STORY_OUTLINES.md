# Candidate story outlines (2026-11-11)

Plots referenced are in `Test notebooks/slides_figs/clusters_x_lambda/` (S1-S3 small world; S4c-S4g scale; S5b EUR-Lex; S6 Wiki10-31K; M1-M3 OffCEM motivation). Numbers are OffCEM + LP bars at floor(eps_max) unless stated.

## Outline A: "Bounds where OffCEM's point estimate cannot be trusted" (the direct story)

1. OffCEM is only as good as its clusters (M1 / M3). With the generating clusters its bias vanishes with n; with any learned clustering it sits at a floor of about 5 on V = 6, at every n. A decision-maker has only learned clusterings.
2. Keep OffCEM, add a finite LP for what it cannot see. Point estimators on the supported pairs; every pair the logger under-covers (lambda = 100) goes to the LP of the CLeaR draft: range, relaxed local correctness around OffCEM's own two-step model, level band. All plug-ins from logs (eps_max, eta, the clipping range).
3. The LP delivers valid bounds under wrong clusterings (S1, S2): DGP clusters cover V from eps = 1; K-means covers from eps = 3 and the data say eps_max = 9.9 -- the wrong clustering costs tolerance, visibly.
4. Beliefs narrow the bound (S3): Lipschitzness, cluster-specific bounds, favourite-action dominance, each a linear belief with a logged-data plug-in; cumulative bar at eps = 9 goes from [-15, 18] to [-0.2, 11.2], hard-feasible.
5. At OffCEM scale the same recipe runs for four clusterings (S4c): every bar contains V; the tolerance t that each belief needs is printed, and it is small where the centre is good (spectral, DGP) and large (16) where OffCEM's model is 20-30 off on a few favourites (K-means, agglomerative). The LP reports what the point estimate hides.
6. Real data (S5b, S6): same formulation, Bernoulli plug-ins; valid for every clustering, hard-feasible, and wide -- because the logger shows each favourite once. The width is a property of the logging design, and the LP says so.
7. Takeaway: one LP, one belief set, one calibration recipe; the output is a bound plus a tolerance report, not a number that is silently wrong.

Risks: the "beliefs narrow" claim is strongest on the small world; at scale B7 widens the bar for two clusterings (t = 16). Say it.

## Outline B: "Local correctness is a belief; here is what happens when you treat it as one" (the diagnostic story)

1. OffCEM assumes local correctness (M2 / bias-vs-LC plot): learned clusterings violate it by 2-3x and the bias follows.
2. Relax it: |mu - m_hat - b| <= eps with eps calibrated from within-cluster logged contrasts. eps_max is the data's own statement of how far local correctness holds: 4 for the ridge on true clusters, 6-10 for learned clusterings and the two-step model.
3. The LP turns the relaxed belief into a bound (S1-S3), and the bound's width is the price of the violation.
4. Beliefs that are cheap to state and checkable on logs buy the width back (S3), and the soft tolerance t is the second diagnostic: it measures how far the logged evidence is from the model's centre.
5. Scale (S4c): t reveals the centre's tail errors per clustering; real data (S5b/S6): t = 0 everywhere, width from logging density. Two failure modes, both read off the same two numbers (eps_max, t) without the truth.
6. Takeaway: the LP is an audit of OffCEM's assumptions with a bound attached.

Risks: more conceptual; needs the truth-check numbers (LC violated on 30-50% of mass) in the appendix to be convincing.

## Outline C: "What the data support" (the decision-maker story; shortest)

1. The question a decision-maker asks: what range of values is consistent with my logs and the beliefs I am willing to state?
2. Method in one slide: OffCEM for the supported part, finite LP for the rest, beliefs 1-6, every constant from the logs, three eps levels around the calibrated eps_max.
3. One figure per setting, same layout, read left to right as beliefs are added: small world K-means (S3), scale four clusterings (S4c), EUR-Lex (S5b), Wiki (S6).
4. Reading rule for every figure: V is inside every bar; the bar narrows as beliefs are added; a label t > 0 means the belief had to be loosened by t to agree with the model; wide bars on real data mean the logs have one observation per pair.
5. Takeaway: the same pipeline, unchanged, across three data regimes; its outputs are bounds and tolerances the user can act on.

Risks: hides the mechanics; reviewers will ask why eps_max differs across clusterings and why t = 16 -- keep Outline A's slides 5 and 6 as backup.

## Outline D: "Two regimes" (synthetic vs real as the spine)

1. Setup and method (as A.1-A.2).
2. Regime 1, cluster structure is real (synthetic): local correctness nearly holds, eps_max is moderate, beliefs B5 and B7 bite, bounds are informative (S3, S4c). Failure mode: the two-step centre's tail on favourites, exposed by t.
3. Regime 2, cluster structure is a convenience (extreme classification): no clustering is right for the favourite label; the two-step model repairs the centre; beliefs are valid but starved; bounds are wide and honest (S5b, S6). Failure mode: logging density.
4. The LP's diagnostics (eps_max, t, infeasibility) separate the regimes from the logs alone.
5. Takeaway and what would change each regime's bounds: a better centre in regime 1, a logger that revisits favourites in regime 2.

Risks: regime 2 is the weaker half; needs the pooled-B7 result (floor 0.00 -> 0.10) as its one positive.

## Pieces every outline needs
- A one-line statement that favourite-action dominance is true by construction for an eps-greedy target on q and needs a regret margin (the soft t) for a general target.
- The eps grid rule (floor(eps_max) -1 / floor / +1; x0.75 / x1 / x1.25 on [0, 1]) and that no eps is chosen with the truth.
- The B3 cap at scale (3 cutting-plane rounds, 576,000 cuts) and that the reference boxes carry B3's effect.
- Truth checks in the appendix: LC violated on 30-50% of LP mass at eps_max under every clustering and both real datasets; the bars cover V by calibrated width, the beliefs narrow them without excluding V.

My recommendation: Outline A as the spine, with B's two diagnostics (eps_max, t) as the connective tissue on slides 5-6, and C's one-figure-per-setting layout for the results section.
