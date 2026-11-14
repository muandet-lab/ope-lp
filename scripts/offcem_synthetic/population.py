#!/usr/bin/env python3
"""Population counterparts of every estimated LP input, for the convergence experiment (convergence.py). Everything is computed from the fixed world's truth
and the frozen model m~; no logged data. Each function mirrors one estimator of the main pipeline at n -> infinity (user-uniform rounds, independent masks):

  mask_moments      E_mask[pi0_round(a|u)] and P_mask((u,a) in the LP region), by Monte Carlo over simulated exclusion masks (no rewards). The kept half is a
                    uniformly random subset (generator: Gumbel top-k); the LP region of a round is pi0_round(a) = 0 or pi(a|u) > lam * pi0_round(a)
                    (extra_beliefs_experiment.structures).
  lp_weights        coef_v of b3_lipschitz_experiment.prep = (1/n) sum over rounds of pi(a|u) 1[(round, a) in region], merged per (user, action)
                    -> (1/N_users) pi(a|u) P_mask(region). (Cells with every action of the cluster excluded in a round are non-identified; probability
                    ~ 0.5^cluster size, ignored.)
  level_centres     partial_ope_global.level_ridge: per cluster, ridge (penalty alpha, intercept unpenalised) of the residuals y - m~ on standardised context
                    features over the cluster's logged rows. Fixed alpha / n -> 0, so the limit is the weighted least-squares fit, row weight
                    (1/N_users) E[pi0](a|u); standardisation by the user-uniform mean / sd of the contexts (rows are user-uniform).
  thresholds        logged-reward quantiles -> quantiles of the mixture sum_u (1/N_users) sum_a E[pi0](a|u) N(q(u,a), sigma^2), by root-finding.
  eps_star          eps_sweep_dgp.calibrate_user_cells: mean of (e_i - e_j)^2 / 2 over pairs of rounds in one (user, cluster) cell with different actions, minus
                    the repeat variance sigma^2. Limit: tau*^2 = sum over users (equal weight), clusters and a != b of p_a p_b (r_a - r_b)^2 / 2 divided by the
                    same weights, r = q - m~, p = E[pi0](.|u); eps* = Z tau* (Z of the calibration, 2).
  L_star            scale_experiment.estimate_L: 95th percentile, over actions and DISTINCT user pairs that both logged the action (each pair once per action),
                    of noise-corrected slopes |ybar_u - ybar_v| / d(u, v). Every (user, action) has positive logging probability, so the limit is the
                    unweighted 95th percentile of |q(u,a) - q(v,a)| / d(u, v) over all user pairs and actions; d = distance of standardised user contexts (prep).
  population_P      the LP of prep(cells='user') with all of the above: one mu per (user, action), one b per (user, cluster), one local-correctness row per
                    variable; solved with b5_b7_experiment.solve_both like the estimated LPs.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.stats import norm

import partial_ope_global as G
import eps_sweep_dgp as S


def mask_moments(world, lam=100.0, n_masks=50000, seed=0, chunk=5000):
    cfg, q, K = world["cfg"], world["q"], world["K"]
    nu = world["users"]; n_keep = K - int(K * cfg.def_frac)
    pi = G.gen_eps_greedy(expected_reward=q, eps=cfg.eps_policy)[:, :, 0]
    Epi0, Preg = np.zeros((nu, K)), np.zeros((nu, K)); rng = np.random.RandomState(seed)
    for u in range(nu):
        e = np.exp(cfg.beta * q[u] - (cfg.beta * q[u]).max())
        for s in range(0, n_masks, chunk):
            m = min(chunk, n_masks - s)
            kept = np.zeros((m, K), bool); kept[np.repeat(np.arange(m), n_keep), np.argsort(rng.uniform(size=(m, K)), axis=1)[:, :n_keep].ravel()] = True
            w = np.where(kept, e[None, :], 0.0); p0 = w / w.sum(1, keepdims=True)
            Epi0[u] += p0.sum(0); Preg[u] += ((p0 == 0) | (pi[u][None, :] > lam * p0)).sum(0)
    return Epi0 / n_masks, Preg / n_masks, pi


def lp_weights(pi, Preg):
    return pi * Preg / pi.shape[0]


def level_centres(world, mtilde, lab, Epi0):
    X, q = world["X"], world["q"]; nu = world["users"]; C = int(lab.max()) + 1
    mu, sd = X.mean(0), X.std(0) + 1e-12                       # rows are user-uniform: row mean / sd -> user mean / sd
    Phi = np.hstack([np.ones((nu, 1)), (X - mu) / sd])
    r = q - mtilde; B = np.full((nu, C), np.nan)
    for c in range(C):
        a = np.flatnonzero(lab == c); w = Epi0[:, a].sum(1) / nu   # weight of user u's rows in cluster c
        rbar = (Epi0[:, a] * r[:, a]).sum(1) / np.maximum(Epi0[:, a].sum(1), 1e-300)   # weighted mean residual of user u in c (same features for all its actions)
        A = (Phi * w[:, None]).T @ Phi; beta = np.linalg.solve(A, (Phi * (w * rbar)[:, None]).sum(0))
        B[:, c] = Phi @ beta
    return B


def thresholds(world, Epi0, qs=(("sev", 0.05), ("poor", 0.5), ("ext", 0.01))):
    q, sig, nu = world["q"], world["cfg"].sigma, world["users"]
    w = (Epi0 / nu).ravel(); m = q.ravel(); keep = w > 0; w, m = w[keep] / w[keep].sum(), m[keep]
    F = lambda t: float((w * norm.cdf((t - m) / sig)).sum())
    lo, hi = m.min() - 10 * sig, m.max() + 10 * sig
    return {k: brentq(lambda t: F(t) - p, lo, hi, xtol=1e-10) for k, p in qs}


def eps_star(world, mtilde, lab, Epi0):
    r = world["q"] - mtilde; num = den = 0.0
    for c in range(int(lab.max()) + 1):
        a = np.flatnonzero(lab == c); p, rc = Epi0[:, a], r[:, a]
        s0, s1, s2 = p.sum(1), (p * rc).sum(1), (p * rc ** 2).sum(1)
        # sum_{a != b} p_a p_b (r_a - r_b)^2 / 2 = s0 s2 - s1^2 ; sum_{a != b} p_a p_b = s0^2 - sum p^2
        num += float((s0 * s2 - s1 ** 2).sum()); den += float((s0 ** 2 - (p ** 2).sum(1)).sum())
    tau = float(np.sqrt(num / den))
    return dict(tau=tau, eps_max=S.Z * tau, z=S.Z)


def user_distance(world):
    X = world["X"]; Zs = (X - X.mean(0)) / X.std(0)
    D = np.sqrt(((Zs[:, None] - Zs[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    return D


def L_star(world, q_level=0.95):
    q, D = world["q"], user_distance(world); nu = world["users"]
    i, j = np.triu_indices(nu, 1)
    slopes = (np.abs(q[i] - q[j]) / D[i, j][:, None]).ravel()
    return float(np.quantile(slopes, q_level))


def population_P(world, mtilde, lab, w, B, eps, LO=-30.0, HI=30.0):
    """prep-style dict for b5_b7_experiment.lp / solve_both (cells per (user, cluster), one row per variable)."""
    nu, K = world["users"], world["K"]; C = int(lab.max()) + 1
    vu, va = np.nonzero(w > 0); nV = len(vu)
    ckeys, cid = np.unique(vu * C + lab[va], return_inverse=True); Kc = len(ckeys)
    r = np.arange(nV)
    A = coo_matrix((np.concatenate([np.ones(nV), -np.ones(nV), -np.ones(nV), np.ones(nV)]),
                    (np.concatenate([r, r, r + nV, r + nV]), np.concatenate([r, nV + cid, r, nV + cid]))), shape=(2 * nV, nV + Kc)).tocsr()
    VID = np.full((nu, K), -1); VID[vu, va] = r
    return dict(nV=nV, Kc=Kc, LO=LO, HI=HI, mp=mtilde[vu, va], gh=B[ckeys // C, ckeys % C], eta=eps, A_base=A, coef_v=w[vu, va], const_lo=0.0, const_up=0.0,
                v_user=vu, v_act=va, VID=VID, D=user_distance(world), nu=nu, K=K)
