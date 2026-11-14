#!/usr/bin/env python3
"""Convergence experiment (paper Sec. 3.3 / experiment (iii)): estimated LP bounds on V^2 converge to the population bounds as the log grows.

This module holds the pieces that are independent of the LP:
  fixed_world(seed)          the small world (200 actions, 100 users, 10 DGP clusters, sigma 3, 50% per-round exclusions, logger softmax(-0.1 q), 0.2-greedy target),
                             built once by the OffCEM generator; only its world part is kept (user contexts, true q, clusters, action features).
  sample_log(world, n, seed) a fresh log of n rounds from that fixed world, with the generator's round mechanics (user uniform; a uniformly random half of the
                             actions kept per round; pi0 = softmax(beta q) over the kept half; action ~ pi0; reward ~ N(q, sigma^2)), returned in the world-dict
                             format of partial_ope_global.finish_world (units = rounds). The generator draws world and log from one random stream, so the
                             world cannot be held fixed by reseeding it.
  freeze_model(world, ...)   the frozen reward model m~: OffCEM two-step (ridge first stage + pairwise network), trained once on an independent training log and
                             clipped to that log's per-cluster reward range; returned as a (users x K) table.
  expand(table, W)           the table as the (units x K) array prep() takes as M_fixed.
"""
import numpy as np

import partial_ope_global as G

SMALL_WORLD = dict(K=200, users=100, C=10, def_frac=0.5, exclusion="round_random", sigma=3.0, beta=-0.1, eps_policy=0.2, clustering="original")


def fixed_world(seed=7000, n_build=20000):
    """World part of the generator's small world. n_build is large so that every user appears; the build log itself is discarded."""
    cfg = G.WorldCfg(**SMALL_WORLD, n=n_build, seed=seed)
    W0 = G.make_world(cfg)
    user = np.asarray(W0["user_of_round"]); nu = int(user.max()) + 1
    assert len(np.unique(user)) == nu == cfg.users, "every user must appear in the build log"
    first = np.array([np.flatnonzero(user == u)[0] for u in range(nu)])
    uid = np.asarray(W0["uid"])
    return dict(cfg=cfg, X=np.asarray(W0["X_unit"])[uid[first]], q=np.asarray(W0["q_unit"])[uid[first]], cl=np.asarray(W0["cl"]).astype(int),
                action_x=np.asarray(W0["action_x"], float), K=int(W0["K"]), users=nu, seed=seed)


def sample_log(world, n, seed):
    """n logged rounds from the fixed world, with the generator's round mechanics. Returns a finish_world dict (units = rounds)."""
    cfg, K, q, X = world["cfg"], world["K"], world["q"], world["X"]
    rng = np.random.RandomState(seed)
    user = rng.randint(world["users"], size=n)
    n_keep = K - int(K * cfg.def_frac)
    kept = np.argsort(rng.gumbel(size=(n, K)), axis=1)[:, ::-1][:, :n_keep]          # uniformly random kept half, as the generator (Gumbel top-k)
    logits = cfg.beta * q[user]
    pi0 = np.zeros((n, K)); rows = np.repeat(np.arange(n), n_keep)
    z = logits[rows, kept.ravel()].reshape(n, n_keep); z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
    pi0[rows, kept.ravel()] = z.ravel()
    u = rng.uniform(size=(n, 1)); act = np.minimum((pi0.cumsum(1) < u).sum(1), K - 1)  # inverse-CDF action draw
    y = rng.normal(loc=q[user, act], scale=cfg.sigma)
    W = G.finish_world(cfg, np.arange(n), act, y, pi0[np.arange(n), act], X[user], q[user], pi0, world["cl"], n, user=user)
    W["action_x"] = world["action_x"]
    return W


def freeze_model(world, lab, n_train=10000, train_seed=1, alpha=1.0, g_model="ridge"):
    """Frozen m~ (users x K): two-step model fitted on an independent training log (folds by user, as in the main runs; every prediction is still a fixed
    function of the training log only), then clipped to the training log's per-cluster logged reward range. Deterministic given the seeds."""
    import torch
    torch.manual_seed(train_seed); torch.use_deterministic_algorithms(True)
    Wt = sample_log(world, n_train, train_seed)
    Wl = {**Wt, "cl": np.asarray(lab).astype(int), "C": int(np.max(lab)) + 1}
    M = G.get_model(Wl, "two_step", "ridge", 5, alpha, 1, user_folds=True, g_model=g_model)
    user = np.asarray(Wt["user_of_round"]); uid = np.asarray(Wt["uid"])
    first = np.array([np.flatnonzero(user == k)[0] for k in range(world["users"])])
    table = M[uid[first]]
    spread = max(float(np.abs(M[uid[np.flatnonzero(user == k)]] - table[k]).max()) for k in range(world["users"]))
    clr = Wl["cl"][Wt["act"]]; y = np.asarray(Wt["y"], float); C = Wl["C"]
    ymin = np.array([y[clr == c].min() for c in range(C)]); ymax = np.array([y[clr == c].max() for c in range(C)])
    clipped = np.clip(table, ymin[Wl["cl"]][None, :], ymax[Wl["cl"]][None, :])
    info = dict(n_train=n_train, train_seed=train_seed, g_model=g_model, alpha=alpha, max_spread_within_user=spread,
                share_clipped=float((clipped != table).mean()), clip_range=[ymin.tolist(), ymax.tolist()])
    return clipped, info


def expand(table, W):
    """(users x K) table -> (units x K) array for prep(M_fixed=...); units are rounds."""
    return np.asarray(table)[_unit_user(W)]


def _unit_user(W):
    uid, user = np.asarray(W["uid"]), np.asarray(W["user_of_round"])
    out = np.empty(len(W["X_unit"]), int); out[uid] = user
    return out
