#!/usr/bin/env python3
"""Partial-identification bounds on the OffCEM DGP as ONE global linear program.

Run from this directory (flat imports), env `offcem`:
    python partial_ope_global.py --K 200 --n 1000 --users 100 --C 10 --def-frac 0.5 --exclusion user_fixed \
        --model cluster_level --cluster-model ridge --centre ridge --eps-lc 4 oracle:0.95 --lp both \
        --output ../../results_offcem_global/example.json

WHAT IS NEW relative to run_partial_ope_lc.py
 1. ONE GLOBAL LP.  All unknown means mu_p (one per unsupported pair p=(unit,action)) and all shared cell levels g_k
    (one per (unit, cluster) cell) are variables of a single sparse LP; the cluster-level beliefs are ordinary rows.
    It is solved once for the lower and once for the upper bound by HiGHS (--lp global).  The per-cell closed form
    used before (--lp closed) is just the exact solution of this LP when it is block diagonal; --lp both runs both and
    reports the discrepancy.  Because the LP is global it can also carry constraints that couple cells, which the closed
    form cannot: --avg-eps adds an AVERAGE local-correctness budget (see below).
 2. EXCLUSIONS FIXED PER USER (--exclusion user_fixed).  In the original dataset the deficient set is redrawn every
    round, so a pair (user, action) that is "unsupported" in one round is often logged in another round of the same
    user and is then identified.  user_fixed draws one excluded set per user for the whole world, so unsupported
    means are genuinely unobservable and the unknowns are shared by all rounds of a user ("unit" = user).
    --exclusion round_random reproduces the dataset's own mechanism (unit = round).
 3. SUPPORTED PART with a model fitted without the unit's own data (cross-fitting by unit), DR by default, plus its
    sampling standard error; the interval is also reported widened by --ips-z standard errors.
 4. ORACLE DIAGNOSTICS (never used to fit): true within-cell spread of the model error, error of the estimated cell
    level, share of cells whose TRUE means violate the beliefs, and --eps-lc oracle:Q which sets eps_LC to the
    mass-weighted Q-quantile of the true spread (a labelled diagnostic, not an estimator).

PROGRAM (unit u has weight n_u/n; U_u = actions with pi0(a|x_u) = 0; cell (u,c) = unsupported actions of cluster c)
   variables   mu_ua (a in U_u),  g_uc,  [t_ua if --avg-eps]
   minimise / maximise   V+ + sum_u (n_u/n) sum_{a in U_u} pi(a|x_u) mu_ua
   l <= mu_ua <= u                                                   outcome range
   |mu_ua - m_ua - g_uc| <= rho = eps_LC/2                           relaxed pairwise LC (equivalent to
                                                                     |(mu_a-mu_b)-(m_a-m_b)| <= eps_LC inside the cell)
   |g_uc - ghat_uc| <= eta_uc                                        shared cell level close to its estimate
   [--avg-eps]  t_ua >= |mu_ua - m_ua - g_uc|, t_ua <= rho,  sum w_ua t_ua <= avg_eps * sum w_ua   (w = n_u pi / n)
Cells whose cluster has no support at x_u (pi0(c|x_u) = 0) or no logged residual keep only the outcome range.
An empty feasible set is reported as infeasible, never clipped.

Oracle arrays (q, true spreads) are used only to build the world, the policy and the validation block.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from obp.dataset import linear_reward_function
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures

from dataset import SyntheticBanditDataset
from policy import gen_eps_greedy

warnings.filterwarnings("ignore", category=ConvergenceWarning)
HIGHS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}


# ----------------------------------------------------------------------------------------------- utilities
def wquantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if not x.size:
        return float("nan")
    o = np.argsort(x)
    x, w = x[o], w[o]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.quantile(x, q))
    return float(x[min(np.searchsorted(cum / cum[-1], q), len(x) - 1)])


def qsum(a, w=None):
    a = np.asarray(a, float)
    if not a.size:
        return None
    if w is None:
        return {"median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(a.max())}
    return {"median": wquantile(a, w, .5), "p95": wquantile(a, w, .95), "max": float(a.max())}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


# ----------------------------------------------------------------------------------------------- the world
@dataclass
class WorldCfg:
    K: int = 200                 # actions
    n: int = 1000                # logged rounds
    users: int = 100
    C: int = 10                  # clusters
    def_frac: float = 0.5        # fraction of actions with pi0 = 0
    exclusion: str = "user_fixed"  # user_fixed | round_random
    sigma: float = 3.0           # reward noise sd
    beta: float = -0.1           # logging inverse temperature
    eps_policy: float = 0.2      # epsilon of the epsilon-greedy target policy
    clustering: str = "original"
    seed: int = 12345


def finish_world(cfg, uid, act, y, pscore, X_unit, q_unit, pi0_unit, cl, n, user=None):
    counts = np.bincount(uid, minlength=len(X_unit))
    keep = counts > 0
    remap = np.cumsum(keep) - 1
    pi_unit = gen_eps_greedy(expected_reward=q_unit[keep], eps=cfg.eps_policy)[:, :, 0]
    cl = np.asarray(cl).astype(int)
    assert np.array_equal(np.unique(cl), np.arange(cl.max() + 1)), "cluster ids must be 0..C-1"
    return {"cfg": cfg, "n": n, "uid": remap[uid], "act": act, "y": y.astype(float), "pscore": pscore,
            "X_unit": X_unit[keep], "X_rows": X_unit[keep][remap[uid]], "q_unit": q_unit[keep],
            "pi0_unit": pi0_unit[keep], "pi_unit": pi_unit, "counts": counts[keep], "cl": cl, "C": int(cl.max()) + 1,
            "K": q_unit.shape[1], "user_of_round": np.asarray(uid if user is None else user)}


def make_world(cfg: WorldCfg):
    D = int(cfg.K * cfg.def_frac)
    ds = SyntheticBanditDataset(
        n_actions=cfg.K, dim_context=10, beta=cfg.beta, reward_type="continuous", n_cat_per_dim=5,
        latent_param_mat_dim=5, n_cat_dim=10, n_deficient_actions=D if cfg.exclusion == "round_random" else 0,
        reward_function=linear_reward_function, reward_std=cfg.sigma, random_state=cfg.seed)
    data = ds.obtain_batch_bandit_feedback(n_rounds=cfg.n, n_users=cfg.users, n_clusters=cfg.C,
                                           clustering_method=cfg.clustering)
    cl = data["cluster_indices"]
    if cfg.exclusion == "round_random":
        W = finish_world(cfg, np.arange(cfg.n), data["action"], data["reward"], data["pscore"], data["context"],
                         data["expected_reward"], data["pi_b"][:, :, 0], cl, cfg.n, user=data["user_idx"])
        W["action_x"] = np.asarray(data["action_context_one_hot"], float)
        return W
    if cfg.exclusion != "user_fixed":
        raise ValueError("exclusion must be user_fixed or round_random")
    # one excluded set per user for the whole world; actions and rewards are redrawn under it
    q_users, uidx = data["fixed_expected_rewards"], data["user_idx"]
    rng = np.random.RandomState(cfg.seed + 1000)
    excl = np.zeros((cfg.users, cfg.K), bool)
    for u in range(cfg.users):
        excl[u, rng.choice(cfg.K, D, replace=False)] = True
    z = cfg.beta * q_users
    p0 = np.exp(z - z.max(axis=1, keepdims=True)) * (~excl)
    p0 /= p0.sum(axis=1, keepdims=True)
    act = (p0[uidx].cumsum(axis=1) > rng.uniform(size=(cfg.n, 1))).argmax(axis=1)
    y = q_users[uidx, act] + cfg.sigma * rng.normal(size=cfg.n)
    W = finish_world(cfg, uidx, act, y, p0[uidx, act], data["fixed_user_contexts"], q_users, p0, cl, cfg.n)
    W["action_x"] = np.asarray(data["action_context_one_hot"], float)   # observed action attributes (used by action-level reward models)
    return W


def world_from_cache(path, eps_policy=0.2):
    """Round-level world rebuilt from run_partial_ope_lc.py's cache (reproduction test)."""
    z = np.load(path, allow_pickle=False)
    n = len(z["y"])
    cfg = WorldCfg(K=z["q"].shape[1], n=n, users=n, C=int(z["cluster_idx"].max()) + 1, exclusion="round_random",
                   eps_policy=eps_policy)
    return finish_world(cfg, np.arange(n), z["act"], z["y"], z["pscore"], z["context"], z["q"], z["pi_b"],
                        z["cluster_idx"], n)


# ----------------------------------------------------------------------------------------------- reward models
def fit_cluster_level_units(W, kind="ridge", folds=5, seed=12345, alpha=1.0, degree=1, user_folds=False):
    """Mc[u,c] ~ E[Y | x_u, cluster c], no per-action term, cross-fitted BY UNIT: unit u's rewards never enter Mc[u,.].
    user_folds=True: folds are assigned by the ACTUAL USER (W['user_of_round'] under round_random, where a unit is a round), so all rounds of one user share a fold
    and, since the model depends on x only, get identical predictions. Default False keeps the earlier by-unit folds."""
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    Xr, uid, y, nU, C = W["X_rows"], W["uid"], W["y"], len(W["X_unit"]), W["C"]
    clr = W["cl"][W["act"]]
    if user_folds:
        unit_user = np.asarray(W["user_of_round"]) if W["cfg"].exclusion == "round_random" else np.arange(nU)
        ids, inv = np.unique(unit_user, return_inverse=True)
        foldu = (np.random.RandomState(seed).permutation(len(ids)) % folds)[inv]
    else:
        foldu = np.random.RandomState(seed).permutation(nU) % folds
    fold_rows = foldu[uid]
    Mc, eye = np.zeros((nU, C)), np.eye(C)
    for f in range(folds):
        tr, te = np.flatnonzero(fold_rows != f), np.flatnonzero(foldu == f)
        if not len(te):
            continue
        if kind == "mlp":
            sc = StandardScaler().fit(Xr[tr])
            Z = np.hstack([sc.transform(Xr[tr]), eye[clr[tr]]])
            net = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=300, random_state=seed + f).fit(Z, y[tr])
            Xt = sc.transform(W["X_unit"][te])
            for c in range(C):
                Mc[te, c] = net.predict(np.hstack([Xt, np.tile(eye[c], (len(te), 1))]))
        else:
            poly = PolynomialFeatures(degree, include_bias=False).fit(Xr[:1])
            sc = StandardScaler().fit(poly.transform(Xr[tr]))
            Pr, Pu = sc.transform(poly.transform(Xr)), sc.transform(poly.transform(W["X_unit"][te]))
            for c in range(C):
                rows = tr[clr[tr] == c]
                Mc[te, c] = y[tr].mean() if len(rows) < 2 else Ridge(alpha=alpha).fit(Pr[rows], y[rows]).predict(Pu)
    return Mc


def fit_two_step_units(W, folds=5, seed=12345, user_folds=False, epochs=30):
    """OffCEM's two-step reward model f_hat(x, a) = g_hat(x, c_a) + h_hat(x, a) (Saito et al. 2023, Sec. 3.2), as implemented in the paper's code (ope.train_pairwise_model +
    OBP RegressionModel MLP(50,50,50) on the residual with the cluster as the action), cross-fitted by unit (or by user). Pairwise data: observed action pairs of the same user in the
    same (estimation) cluster; reward per (user, action) = mean logged y. Returns f_hat as (units x K)."""
    import torch
    import ope as OPE
    from obp.ope import RegressionModel
    from sklearn.neural_network import MLPRegressor as MLP
    uid, act, y, nU, C, cl, K = W["uid"], W["act"], W["y"], len(W["X_unit"]), W["C"], np.asarray(W["cl"]), W["K"]
    user_r = np.asarray(W["user_of_round"]) if W["cfg"].exclusion == "round_random" else uid
    users, inv_r = np.unique(user_r, return_inverse=True); nu = len(users)
    first = np.array([np.flatnonzero(inv_r == k)[0] for k in range(nu)])
    Xu = W["X_unit"][uid[first]] if W["cfg"].exclusion == "round_random" else W["X_unit"]      # one context per user
    unit_user = inv_r[np.array([np.flatnonzero(uid == u)[0] for u in range(nU)])] if W["cfg"].exclusion == "round_random" else np.arange(nU)
    if user_folds:
        foldu = (np.random.RandomState(seed).permutation(nu) % folds)[unit_user]
    else:
        foldu = np.random.RandomState(seed).permutation(nU) % folds
    fold_rows = foldu[uid]
    cluster_idx_mat = np.tile(cl, (nu, 1))
    M = np.zeros((nU, K))
    for f in range(folds):
        tr, te = np.flatnonzero(fold_rows != f), np.flatnonzero(foldu == f)
        if not len(te):
            continue
        obs, s_, n_ = np.zeros((nu, K), int), np.zeros((nu, K)), np.zeros((nu, K))
        np.add.at(s_, (inv_r[tr], act[tr]), y[tr]); np.add.at(n_, (inv_r[tr], act[tr]), 1); obs[n_ > 0] = 1
        with np.errstate(invalid="ignore", divide="ignore"):
            reward_mat = np.where(n_ > 0, s_ / n_, 0.0)
        bd = dict(n_users=nu, fixed_user_contexts=Xu, n_actions=K, action_context_one_hot=np.asarray(W["action_x"], float), obs_mat=obs, reward_mat=reward_mat,
                  context=Xu[inv_r[tr]], user_idx=inv_r[tr], action=act[tr], reward=y[tr].astype(float))
        torch.manual_seed(seed + f)
        h_hat = OPE.train_pairwise_model(bd, cluster_idx_mat, C, prediction_context=Xu, num_epochs=epochs, random_state=seed + f)   # (nu x K)
        resid = y[tr] - h_hat[inv_r[tr], act[tr]]
        reg = RegressionModel(n_actions=C, action_context=np.eye(C), base_model=MLP(hidden_layer_sizes=(50, 50, 50), random_state=seed + f))
        reg.fit(context=Xu[inv_r[tr]], action=cl[act[tr]], reward=resid)
        g_hat = reg.predict(context=Xu)[:, :, 0]                                                                                        # (nu x C)
        f_hat = h_hat + g_hat[:, cl]
        M[te] = f_hat[unit_user[te]]
    return M


def get_model(W, name, cluster_model="ridge", folds=5, alpha=1.0, degree=1, model_noise=2.0, user_folds=False):
    q, cl, C, seed = W["q_unit"], W["cl"], W["C"], W["cfg"].seed
    rng = np.random.RandomState(seed + 1)
    if name == "cluster_level":
        return fit_cluster_level_units(W, cluster_model, folds, seed, alpha, degree, user_folds)[:, cl]
    if name == "two_step":
        return fit_two_step_units(W, folds, seed, user_folds)
    if name == "oracle":
        return q.copy()
    if name == "oracle_shift":
        Wc = rng.normal(size=(W["X_unit"].shape[1], C)) / math.sqrt(W["X_unit"].shape[1])
        return q - (model_noise * np.tanh(W["X_unit"] @ Wc))[:, cl]
    if name == "oracle_noise":
        return q + rng.normal(0.0, model_noise, size=q.shape)
    if name == "zero":
        return np.zeros_like(q)
    raise ValueError(f"unknown model {name}")


# ----------------------------------------------------------------------------------------------- cell levels
def level_knn(Xr, clr, resid, Xu, C, k):
    G, S = np.full((len(Xu), C), np.nan), np.full((len(Xu), C), np.nan)
    for c in range(C):
        rows = np.flatnonzero(clr == c)
        if not len(rows):
            continue
        kk = min(k, len(rows))
        idx = NearestNeighbors(n_neighbors=kk).fit(Xr[rows]).kneighbors(Xu, return_distance=False)
        v = resid[rows][idx]
        G[:, c] = v.mean(axis=1)
        S[:, c] = (v.std(axis=1, ddof=1) if kk > 1 else 0.0) / math.sqrt(kk)
    return G, S


def level_ridge(Xr, clr, resid, Xu, C, alpha, degree, agg_w=None):
    poly = PolynomialFeatures(degree, include_bias=False).fit(Xr[:1])
    Pr = poly.transform(Xr)
    mu, sd = Pr.mean(axis=0), Pr.std(axis=0) + 1e-12
    Pr = np.hstack([np.ones((len(Xr), 1)), (Pr - mu) / sd])
    Pu = np.hstack([np.ones((len(Xu), 1)), (poly.transform(Xu) - mu) / sd])
    pen = np.eye(Pr.shape[1]) * alpha
    pen[0, 0] = 0.0
    G, S = np.full((len(Xu), C), np.nan), np.full((len(Xu), C), np.nan)
    agg_var = 0.0
    for c in range(C):
        rows = np.flatnonzero(clr == c)
        if len(rows) < 2:
            continue
        P = Pr[rows]
        A = np.linalg.inv(P.T @ P + pen)
        beta = A @ P.T @ resid[rows]
        dof = max(len(rows) - np.trace(P @ A @ P.T), 1.0)
        sigma = math.sqrt(float(((resid[rows] - P @ beta) ** 2).sum()) / dof)
        G[:, c] = Pu @ beta
        S[:, c] = sigma * np.sqrt(np.einsum("ij,jk,ik->i", Pu, A, Pu))
        if agg_w is not None:     # Var(sum_u w_uc ghat_uc) = sigma_c^2 a' A a with a = sum_u w_uc phi(x_u); clusters use disjoint rows
            a_vec = agg_w[:, c] @ Pu
            agg_var += sigma ** 2 * float(a_vec @ A @ a_vec)
    return (G, S) if agg_w is None else (G, S, math.sqrt(agg_var))


# ----------------------------------------------------------------------------------------------- LP engines
def cell_bounds(m, pi, U, cl, ghat, eta, rho, l, u, ident, C):
    """Exact optimum of every (unit, cluster) cell (closed form). Arrays: (units, C). Values are sum_a pi_a mu_a (no unit weight)."""
    nU = m.shape[0]
    shp = (nU, C)
    lower, upper, gap = np.zeros(shp), np.zeros(shp), np.zeros(shp)
    feas, has_u, mass = np.ones(shp, bool), np.zeros(shp, bool), np.zeros(shp)
    for c in range(C):
        cols = np.flatnonzero(cl == c)
        if not len(cols):
            continue
        Uc, mc, pc = U[:, cols], m[:, cols], pi[:, cols]
        rho_c = float(rho) if np.ndim(rho) == 0 else float(rho[c])       # per-cluster tolerance when rho is a vector
        hu = Uc.any(axis=1)
        glo = np.where(Uc, l - mc - rho_c, -np.inf).max(axis=1)
        ghi = np.where(Uc, u - mc + rho_c, np.inf).min(axis=1)
        gh = np.where(ident[:, c], ghat[:, c], 0.0)
        Glo, Ghi = np.maximum(gh - eta[:, c], glo), np.minimum(gh + eta[:, c], ghi)
        ok = Glo <= Ghi + 1e-9
        pm = (pc * Uc).sum(axis=1)
        lo = np.where(Uc, pc * np.maximum(l, mc + Glo[:, None] - rho_c), 0.0).sum(axis=1)
        hi = np.where(Uc, pc * np.minimum(u, mc + Ghi[:, None] + rho_c), 0.0).sum(axis=1)
        lower[:, c] = np.where(ident[:, c], lo, pm * l)
        upper[:, c] = np.where(ident[:, c], hi, pm * u)
        feas[:, c] = np.where(hu & ident[:, c], ok, True)
        gap[:, c] = np.where(hu & ident[:, c], np.maximum(0.0, Glo - Ghi), 0.0)
        has_u[:, c], mass[:, c] = hu, pm
    return lower, upper, feas, gap, has_u, mass


def solve_global_lp(m, pi, U, cl, ghat, eta, rho, l, u, ident, wu, avg_eps=None):
    """ONE sparse LP over all unsupported means and cell levels. Returns the unsupported-part interval sum_u wu sum_a pi mu."""
    t0 = time.perf_counter()
    C = int(cl.max()) + 1
    pu, pa = np.nonzero(U)
    P = len(pu)
    if not P:
        raise ValueError("no unsupported pairs")
    pc = cl[pa]
    coef = wu[pu] * pi[pu, pa]
    scale = coef.max()
    ip = np.flatnonzero(ident[pu, pc])
    Pi = len(ip)
    ckeys, cinv = np.unique(pu[ip] * C + pc[ip], return_inverse=True)
    Kc = len(ckeys)
    cu, cc = ckeys // C, ckeys % C
    use_t = avg_eps is not None
    nvar = P + Kc + (Pi if use_t else 0)
    bounds = np.zeros((nvar, 2))
    bounds[:P] = [l, u]
    bounds[P:P + Kc, 0] = ghat[cu, cc] - eta[cu, cc]
    bounds[P:P + Kc, 1] = ghat[cu, cc] + eta[cu, cc]
    rr = np.arange(Pi)
    mp = m[pu[ip], pa[ip]]
    rows, cols, vals, rhs = [], [], [], []
    rows += [rr, rr, rr + Pi, rr + Pi]
    cols += [ip, P + cinv, ip, P + cinv]
    vals += [np.ones(Pi), -np.ones(Pi), -np.ones(Pi), np.ones(Pi)]
    if use_t:
        tcol = P + Kc + rr
        bounds[P + Kc:, 0], bounds[P + Kc:, 1] = 0.0, min(rho, 1e6)
        rows += [rr, rr + Pi]; cols += [tcol, tcol]; vals += [-np.ones(Pi), -np.ones(Pi)]
        rhs = [mp, -mp]
        rows.append(np.full(Pi, 2 * Pi)); cols.append(tcol); vals.append(coef[ip] / scale)
        b_extra = [avg_eps * coef[ip].sum() / scale]
        nrows = 2 * Pi + 1
    else:
        rhs = [mp + rho, -mp + rho]
        b_extra, nrows = [], 2 * Pi
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrows, nvar)).tocsr()
    b = np.concatenate(rhs + [np.asarray(b_extra)]) if b_extra else np.concatenate(rhs)
    c_obj = np.zeros(nvar)
    c_obj[:P] = coef / scale
    out = {"n_vars": int(nvar), "n_rows": int(nrows), "nnz": int(A.nnz)}
    res = []
    for sign in (1.0, -1.0):
        r = linprog(sign * c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs", options=HIGHS) if Pi else None
        if Pi == 0:   # nothing constrained: every pair keeps only [l, u]
            res.append(sign * (coef / scale * (l if sign > 0 else u)).sum())
            continue
        if not r.success:
            out.update(status="infeasible" if r.status == 2 else "solver_failure", solver_status=int(r.status),
                       lower0=None, upper0=None, seconds=time.perf_counter() - t0)
            return out
        res.append(r.fun)
    out.update(status="optimal", lower0=res[0] * scale, upper0=-res[1] * scale, seconds=time.perf_counter() - t0)
    return out


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(W, M, spec):
    """All quantities for one world and one model matrix M (units, K). Oracle arrays only enter the validation block."""
    cl, C, counts, n = W["cl"], W["C"], W["counts"], W["n"]
    q, pi, pi0 = W["q_unit"], W["pi_unit"], W["pi0_unit"]
    uid, act, y, pscore = W["uid"], W["act"], W["y"], W["pscore"]
    nU, K = q.shape
    wu = counts / n
    U_hard = pi0 == 0
    lam = spec.get("overlap_lambda")
    # weakly supported pairs (importance weight pi/pi0 above lambda) are handed to the LP together with the unsupported ones;
    # IPS/DR then only uses pairs with weight <= lambda, so its terms are bounded and its sampling error is well behaved
    U = U_hard | (pi > lam * pi0) if lam is not None else U_hard
    if not U.any():
        raise ValueError("no unsupported pairs")
    mass = float((wu[:, None] * pi * U).sum())
    oracle_V = float((wu[:, None] * pi * q).sum())
    oracle_Vp = float((wu[:, None] * pi * q * ~U).sum())
    oracle_V0 = oracle_V - oracle_Vp

    # identified part: IPS and DR (model term over SUPPORTED actions only), known propensities
    w = pi[uid, act] / pscore
    strong = ~U[uid, act]                       # logged pair is strongly supported (weight <= lambda)
    t_ips = y * w * strong
    t_dr = (pi * M * ~U).sum(axis=1)[uid] + w * strong * (y - M[uid, act])
    terms = t_dr if spec.get("supported", "dr") == "dr" else t_ips
    if spec.get("supported") == "offcem":
        # OffCEM on the supported part: model term over supported pairs + cluster-weighted correction pi_S(c|x)/pi0(c|x) * (y - m),
        # pi_S = target mass on the supported (non-U) pairs of the cluster. With the unsupported correction (recentre) the two sum to full OffCEM.
        cl_, C_ = W["cl"], W["C"]
        pS_c = np.stack([(pi * ~U)[:, cl_ == c].sum(axis=1) for c in range(C_)], axis=1)
        p0_c = np.stack([pi0[:, cl_ == c].sum(axis=1) for c in range(C_)], axis=1)
        c_i = cl_[act]
        okc = p0_c[uid, c_i] > 0
        w_s = np.minimum(np.where(okc, pS_c[uid, c_i] / np.where(okc, p0_c[uid, c_i], 1.0), 0.0), spec.get("lam_c", 100.0))
        terms = (pi * M * ~U).sum(axis=1)[uid] + w_s * (y - M[uid, act])
    Vp, Vp_se = float(terms.mean()), float(terms.std(ddof=1) / math.sqrt(n))

    # cell levels and support
    resid = y - M[uid, act]
    clr = cl[act]
    pi0_c, pi_c = np.zeros((nU, C)), np.zeros((nU, C))
    pm_cell = np.zeros((nU, C))
    for c in range(C):
        pi0_c[:, c], pi_c[:, c] = pi0[:, cl == c].sum(axis=1), pi[:, cl == c].sum(axis=1)
        pm_cell[:, c] = (pi[:, cl == c] * U[:, cl == c]).sum(axis=1)
    support_viol = (pi_c > 0) & (pi0_c <= 0)
    agg_mode = spec.get("noise", "per_cell") == "aggregate"
    agg_se = 0.0
    if spec.get("centre", "ridge") == "knn":
        if agg_mode:
            raise ValueError("aggregate noise needs the ridge centre")
        G, S = level_knn(W["X_rows"], clr, resid, W["X_unit"], C, spec.get("neighbors", 20))
    elif agg_mode:
        G, S, agg_se = level_ridge(W["X_rows"], clr, resid, W["X_unit"], C, spec.get("ridge_alpha", 1.0), spec.get("ridge_degree", 1),
                                   agg_w=wu[:, None] * pm_cell * ~support_viol)
    else:
        G, S = level_ridge(W["X_rows"], clr, resid, W["X_unit"], C, spec.get("ridge_alpha", 1.0), spec.get("ridge_degree", 1))
    ident = ~support_viol & ~np.isnan(G)
    cor_terms = None
    if spec.get("recentre") == "cluster_ips":
        # The ridge level is smooth in x, so it misses the (unit, cluster) residuals where the target puts its mass. Recentre the
        # levels so their policy-weighted sum equals the cluster-weighted residual moment pi_U(c|x)/pi0(c|x) * (y - m), the
        # OffCEM-style correction (data only, capped at lam_c). Only a constant shift: the per-cell spread beliefs are unchanged.
        ok_c = pi0_c[uid, clr] > 0
        wt = np.minimum(np.where(ok_c, pm_cell[uid, clr] / np.where(ok_c, pi0_c[uid, clr], 1.0), 0.0), spec.get("lam_c", 100.0))
        cor_terms = wt * resid
        wid = wu[:, None] * pm_cell * ident
        if wid.sum() > 0:
            G = G + np.where(ident, (cor_terms.mean() - (wid * np.where(ident, G, 0.0)).sum()) / wid.sum(), 0.0)
    l = float(math.floor(y.min())) if spec.get("limits") is None else float(spec["limits"][0])
    u = float(math.ceil(y.max())) if spec.get("limits") is None else float(spec["limits"][1])

    # oracle structure of the model error inside identified cells that contain unsupported mass
    r = q - M
    sp, mid, cmass, ci, cc_ = [], [], [], [], []
    for c in range(C):
        cols = cl == c
        Uc, rc = U[:, cols], r[:, cols]
        hu = Uc.any(axis=1) & ident[:, c]
        rmax, rmin = np.where(Uc, rc, -np.inf).max(axis=1), np.where(Uc, rc, np.inf).min(axis=1)
        sp.append((rmax - rmin)[hu]); mid.append(((rmax + rmin) / 2)[hu])
        cmass.append((wu[:, None] * pi[:, cols] * Uc).sum(axis=1)[hu])
        ci.append(np.flatnonzero(hu)); cc_.append(np.full(hu.sum(), c))
    sp, mid, cmass, ci, cc_ = (np.concatenate(v) for v in (sp, mid, cmass, ci, cc_))
    g_err = np.abs(G[ci, cc_] - mid)
    out = {"world": asdict(W["cfg"]), "units": int(nU), "unsupported_pairs": int(U.sum()),
           "data": {"unsupported_policy_mass": mass, "oracle_V": oracle_V, "oracle_supported_part": oracle_Vp,
                    "oracle_unsupported_part": oracle_V0, "outcome_limits": [l, u],
                    "true_q_range": [float(q.min()), float(q.max())],
                    "supported_estimator": spec.get("supported", "dr"), "supported_estimate": Vp, "supported_se": Vp_se,
                    "supported_error": Vp - oracle_Vp, "ips_estimate": float(t_ips.mean()), "dr_estimate": float(t_dr.mean()),
                    "ips_error": float(t_ips.mean()) - oracle_Vp, "dr_error": float(t_dr.mean()) - oracle_Vp,
                    "ips_se": float(t_ips.std(ddof=1) / math.sqrt(n)), "dr_se": float(t_dr.std(ddof=1) / math.sqrt(n)),
                    "max_ipw_weight": float(w[strong].max()) if strong.any() else 0.0, "max_ipw_weight_raw": float(w.max()),
                    "overlap_lambda": lam, "hard_unsupported_mass": float((wu[:, None] * pi * U_hard).sum()),
                    "logged_rounds_dropped_as_weak": int((~strong).sum()), "cells": int(len(ci)),
                    "cells_unidentified_policy_mass": float((wu[:, None] * np.stack(
                        [(pi[:, cl == c] * U[:, cl == c]).sum(axis=1) for c in range(C)], axis=1) * ~ident).sum()),
                    "no_assumption_bounds": [Vp + mass * l, Vp + mass * u]},
           "validation_oracle": {"spread_unsupported_in_cell": qsum(sp, cmass), "ghat_error_vs_midrange": qsum(g_err, cmass),
                                 "model_error_abs": qsum(np.abs(r[U])), "n_cells": int(len(ci)),
                                 **({"true_mean_abs_dev_best_g": true_mad_best_g(r, pi, U, cl, ident, wu, C)} if spec.get("diag_mad") else {})},
           "results": []}

    a_budget = spec.get("avg_eps")
    if spec.get("avg_eps_mult"):       # ORACLE-calibrated average-error budget: kappa * true mean abs deviation (diagnostic)
        a_budget = spec["avg_eps_mult"] * true_mad_best_g(r, pi, U, cl, ident, wu, C)
        out["validation_oracle"]["avg_budget"] = a_budget
    z = spec.get("ips_z", 1.96)
    se_tot = math.sqrt(Vp_se ** 2 + agg_se ** 2)
    if cor_terms is not None:      # same rounds feed the supported part and the correction: use the joint sampling error
        agg_se = float(cor_terms.std(ddof=1) / math.sqrt(n))
        se_tot = float((terms + cor_terms).std(ddof=1) / math.sqrt(n))
    out["data"].update(ghat_agg_se=agg_se, total_se=se_tot, noise_mode="aggregate" if agg_mode else "per_cell")
    eps_list = []
    for e in spec.get("eps_lc", [4.0]):
        eps_list.append((str(e), wquantile(sp, cmass, float(e.split(":")[1])) if isinstance(e, str) and e.startswith("oracle:") else float(e)))
    for label, eps_lc in eps_list:
        rho = eps_lc / 2.0
        # rho_cluster (optional, length C): per-cluster tolerance; eps_lc is then only a label. eta_value (optional): a fixed eta, decoupled from rho
        rho_vec = np.full(C, rho) if spec.get("rho_cluster") is None else np.asarray(spec["rho_cluster"], float)
        noise_term = np.zeros_like(S) if agg_mode else spec.get("se_mult", 2.0) * np.where(np.isnan(S), 0.0, S)   # aggregate mode: noise goes into the final CI
        eta = spec.get("eta", 0.0) + (a_budget if spec.get("eta_from_avg") else 0.0) + noise_term + (rho_vec[None, :] if spec.get("add_rho", True) else 0.0)
        if spec.get("eta_value") is not None:
            eta = np.full_like(S, float(spec["eta_value"])) + noise_term
        res = {"eps_lc_label": label, "eps_lc": eps_lc}
        lo, up, feas, gap, has_u, _ = cell_bounds(M, pi, U, cl, G, eta, rho_vec, l, u, ident, C)
        n_inf = int((~feas).sum())
        pm_cell = np.stack([(pi[:, cl == c] * U[:, cl == c]).sum(axis=1) for c in range(C)], axis=1)
        lo0 = float((wu[:, None] * np.where(feas, lo, pm_cell * l)).sum())
        up0 = float((wu[:, None] * np.where(feas, up, pm_cell * u)).sum())
        res.update(infeasible_cells=n_inf, status="infeasible" if n_inf else "optimal", max_gap=float(gap.max()))
        # truth check: do the TRUE means satisfy the three beliefs in every identified cell?
        bad_mass, tot_mass, bad_range, bad_spread, bad_band, bad_joint = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for c in range(C):
            cols = cl == c
            Uc, rc, qc = U[:, cols], r[:, cols], q[:, cols]
            sel = ident[:, c] & Uc.any(axis=1)
            rmax, rmin = np.where(Uc, rc, -np.inf).max(axis=1), np.where(Uc, rc, np.inf).min(axis=1)
            gh = np.where(ident[:, c], G[:, c], 0.0)
            ok_g = np.maximum(rmax - rho_vec[c], gh - eta[:, c]) <= np.minimum(rmin + rho_vec[c], gh + eta[:, c]) + 1e-9
            in_range = (np.where(Uc, qc, l) >= l - 1e-9).all(axis=1) & (np.where(Uc, qc, u) <= u + 1e-9).all(axis=1)
            pm = wu * pm_cell[:, c]
            bad_mass += float((pm * (sel & ~(ok_g & in_range))).sum()); tot_mass += float((pm * sel).sum())
            bad_band += float((pm * (sel & (np.abs((rmax + rmin) / 2 - gh) > eta[:, c] + 1e-9))).sum()); bad_joint += float((pm * (sel & ~ok_g)).sum())
            bad_range += float((pm * (sel & ~in_range)).sum()); bad_spread += float((pm * (sel & (rmax - rmin > 2 * rho_vec[c] + 1e-9))).sum())
        res.update(truth_violation_share=bad_mass / max(tot_mass, 1e-300), truth_range_violation_share=bad_range / max(tot_mass, 1e-300),
                   truth_spread_violation_share=bad_spread / max(tot_mass, 1e-300),
                   truth_band_violation_share=bad_band / max(tot_mass, 1e-300), truth_joint_violation_share=bad_joint / max(tot_mass, 1e-300))
        res.update(unsupported_interval=[lo0, up0], contains_V0=bool(lo0 <= oracle_V0 <= up0) if not n_inf else None)
        if spec.get("repair") and n_inf:
            res["repaired"] = True       # cells with no feasible beliefs keep only the outcome limits (valid, wider); counted in infeasible_cells
        if (not n_inf) or spec.get("repair"):
            res.update(lower=Vp + lo0, upper=Vp + up0, width=up0 - lo0, contains_V=bool(Vp + lo0 <= oracle_V <= Vp + up0),
                       with_se=[Vp + lo0 - z * se_tot, Vp + up0 + z * se_tot],
                       contains_V_with_se=bool(Vp + lo0 - z * se_tot <= oracle_V <= Vp + up0 + z * se_tot),
                       unsupported_interval_ci=[lo0 - z * agg_se, up0 + z * agg_se],
                       contains_V0_ci=bool(lo0 - z * agg_se <= oracle_V0 <= up0 + z * agg_se))
        mode = spec.get("lp", "closed")
        if mode in ("global", "both"):
            g = solve_global_lp(M, pi, U, cl, G, eta, rho, l, u, ident, wu)
            res["global_lp"] = g
            if g["status"] == "optimal" and not n_inf:
                res["global_vs_closed_max_abs_diff"] = float(max(abs(g["lower0"] - lo0), abs(g["upper0"] - up0)))
            res["global_status_matches_closed"] = bool((g["status"] == "infeasible") == bool(n_inf))
        if a_budget is not None:
            g = solve_global_lp(M, pi, U, cl, G, eta, rho, l, u, ident, wu, avg_eps=float(a_budget))
            res["avg_lp"] = g
            if g["status"] == "optimal":
                res["avg_lp"]["interval"] = [Vp + g["lower0"], Vp + g["upper0"]]
                res["avg_lp"]["with_se"] = [Vp + g["lower0"] - z * se_tot, Vp + g["upper0"] + z * se_tot]
                res["avg_lp"]["contains_V_with_se"] = bool(Vp + g["lower0"] - z * se_tot <= oracle_V <= Vp + g["upper0"] + z * se_tot)
                res["avg_lp"]["contains_V"] = bool(Vp + g["lower0"] <= oracle_V <= Vp + g["upper0"])
                res["avg_lp"]["contains_V0"] = bool(g["lower0"] <= oracle_V0 <= g["upper0"])
        out["results"].append(res)
    return out


def true_mad_best_g(r, pi, U, cl, ident, wu, C):
    """ORACLE diagnostic: policy-mass-weighted mean |r - g*| of the true model error r = q - m, with g* the weighted median of r
    in each cell (the best common level). This is the scale of an average-error budget (--avg-eps) the truth would satisfy."""
    num = den = 0.0
    for c in range(C):
        cols = np.flatnonzero(cl == c)
        Uc, rc, wc = U[:, cols], r[:, cols], pi[:, cols] * U[:, cols]
        for u in np.flatnonzero(ident[:, c] & Uc.any(axis=1)):
            msk = Uc[u]
            x, w = rc[u, msk], wc[u, msk]
            num += wu[u] * float((w * np.abs(x - wquantile(x, w, .5))).sum())
            den += wu[u] * float(w.sum())
    return num / max(den, 1e-300)


def run_setting(spec):
    """Build the world, the model and evaluate. spec: dict(world=WorldCfg fields, model=..., ...)."""
    t0 = time.perf_counter()
    W = world_from_cache(spec["cache"], spec.get("world", {}).get("eps_policy", 0.2)) if spec.get("cache") else make_world(WorldCfg(**spec.get("world", {})))
    if spec.get("cluster_labels") is not None:   # alternative partition of the same actions (DGP / learned / pooled); world data untouched
        W["cl"] = np.asarray(spec["cluster_labels"], int)
        W["C"] = int(W["cl"].max()) + 1
    if spec.get("ignore_clusters"):      # same world, but estimation sees ONE cluster (cluster-free test)
        W["cl"], W["C"] = np.zeros_like(W["cl"]), 1
    M = get_model(W, spec.get("model", "cluster_level"), spec.get("cluster_model", "ridge"), spec.get("folds", 5),
                  spec.get("ridge_alpha", 1.0), spec.get("ridge_degree", 1), spec.get("model_noise", 2.0), bool(spec.get("user_folds")))
    out = evaluate(W, M, spec)
    out["spec"] = {k: v for k, v in spec.items() if k != "world"}
    out["seconds"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, d, t in (("K", 200, int), ("n", 1000, int), ("users", 100, int), ("C", 10, int), ("def-frac", 0.5, float),
                    ("sigma", 3.0, float), ("beta", -0.1, float), ("eps-policy", 0.2, float), ("seed", 12345, int)):
        ap.add_argument("--" + f, type=t, default=d)
    ap.add_argument("--exclusion", choices=("user_fixed", "round_random"), default="user_fixed")
    ap.add_argument("--clustering", default="original")
    ap.add_argument("--model", choices=("cluster_level", "oracle", "oracle_shift", "oracle_noise", "zero"), default="cluster_level")
    ap.add_argument("--cluster-model", choices=("ridge", "mlp"), default="ridge")
    ap.add_argument("--model-noise", type=float, default=2.0)
    ap.add_argument("--centre", choices=("ridge", "knn"), default="ridge")
    ap.add_argument("--neighbors", type=int, default=20)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--ridge-degree", type=int, default=1)
    ap.add_argument("--supported", choices=("dr", "ips"), default="dr")
    ap.add_argument("--overlap-lambda", type=float, help="treat pairs with importance weight pi/pi0 > lambda as unsupported (LP) instead of IPS/DR")
    ap.add_argument("--eps-lc", nargs="+", default=["4"], help="numbers, or oracle:Q (mass-weighted Q-quantile of the TRUE spread; diagnostic)")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--se-mult", type=float, default=2.0)
    ap.add_argument("--no-add-rho", action="store_true")
    ap.add_argument("--avg-eps", type=float, help="average local-correctness budget (global coupling row)")
    ap.add_argument("--lp", choices=("closed", "global", "both"), default="closed")
    ap.add_argument("--outcome-lower", type=float); ap.add_argument("--outcome-upper", type=float)
    ap.add_argument("--ips-z", type=float, default=1.96)
    ap.add_argument("--output")
    a = ap.parse_args()
    spec = {"world": dict(K=a.K, n=a.n, users=a.users, C=a.C, def_frac=a.def_frac, exclusion=a.exclusion, sigma=a.sigma,
                          beta=a.beta, eps_policy=a.eps_policy, clustering=a.clustering, seed=a.seed),
            "model": a.model, "cluster_model": a.cluster_model, "model_noise": a.model_noise, "centre": a.centre,
            "neighbors": a.neighbors, "ridge_alpha": a.ridge_alpha, "ridge_degree": a.ridge_degree, "supported": a.supported, "overlap_lambda": a.overlap_lambda,
            "eps_lc": [e if e.startswith("oracle:") else float(e) for e in a.eps_lc], "eta": a.eta, "se_mult": a.se_mult,
            "add_rho": not a.no_add_rho, "avg_eps": a.avg_eps, "lp": a.lp, "ips_z": a.ips_z,
            "limits": None if a.outcome_lower is None or a.outcome_upper is None else [a.outcome_lower, a.outcome_upper]}
    out = run_setting(spec)
    d = out["data"]
    print(f"{a.exclusion} K={a.K} n={a.n} users={a.users} C={a.C} def={a.def_frac} model={a.model} | oracle V={d['oracle_V']:.3f} "
          f"(V+={d['oracle_supported_part']:.3f}, V0={d['oracle_unsupported_part']:.3f}) unsupported mass={d['unsupported_policy_mass']:.3f}")
    print(f"supported: IPS {d['ips_estimate']:.3f}±{d['ips_se']:.2f}  DR {d['dr_estimate']:.3f}±{d['dr_se']:.2f}  (oracle {d['oracle_supported_part']:.3f})  "
          f"spread(unsupported,cell) {out['validation_oracle']['spread_unsupported_in_cell']}")
    for r in out["results"]:
        if r["status"] == "infeasible":
            print(f"eps_LC={r['eps_lc_label']}({r['eps_lc']:.2f}) INFEASIBLE in {r['infeasible_cells']} cells; truth violates beliefs {r['truth_violation_share']:.1%}")
        else:
            print(f"eps_LC={r['eps_lc_label']}({r['eps_lc']:.2f}) [{r['lower']:.3f}, {r['upper']:.3f}] width={r['width']:.3f} covers_V={r['contains_V']} "
                  f"(+se {r['contains_V_with_se']}) covers_V0={r['contains_V0']} truth violates {r['truth_violation_share']:.1%}")
        if "global_vs_closed_max_abs_diff" in r:
            print(f"    global LP ({r['global_lp']['n_vars']} vars, {r['global_lp']['seconds']:.1f}s) vs closed form: max abs diff {r['global_vs_closed_max_abs_diff']:.2e}")
        if "avg_lp" in r and r["avg_lp"]["status"] == "optimal":
            print(f"    avg-eps={a.avg_eps}: [{r['avg_lp']['interval'][0]:.3f}, {r['avg_lp']['interval'][1]:.3f}] covers_V={r['avg_lp']['contains_V']}")
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(clean(out), indent=2) + "\n")


if __name__ == "__main__":
    main()
