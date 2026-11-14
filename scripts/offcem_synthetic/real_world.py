#!/usr/bin/env python3
"""Real-data arm (docs/REAL_DATA_DESIGN.md): the OffCEM paper's extreme-classification bandit (EUR-Lex 4K / Wiki10-31K, src/real/dataset.py, copied here as real_dataset.py)
wrapped into the world dict used by the LP pipeline.

Paper pipeline kept as is: fixed user pool sampled from the training split, PCA(100) + standardised contexts, q(x,a) = sigmoid(label-based score with per-action noise eta_a ~ U(0, 0.2)),
Bernoulli rewards, logging policy softmax(beta * r_hat) with r_hat a ridge fitted on 2,000 test contexts (beta = 30), target eps-greedy on the true q (eps = 0.1), 100 uniform random clusters.
Unsupported region: weak overlap pi/pi0 > lambda (no action is ever excluded by the paper's logger). Outcome range [0, 1]. Units = users (pi0 is identical across a user's rounds).
Action features for learned clusterings: truncated SVD of the training label matrix columns (label co-occurrence), so clusters group labels that appear on the same documents.
Calibration: the same within-(user, cluster) residual contrasts, with the Bernoulli noise term plugged in from the cross-fitted centre (m_hat (1 - m_hat)) instead of repeated observations.
"""
import os
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import TruncatedSVD

import partial_ope_global as G
import real_dataset as RD


@dataclass
class RealCfg:
    dataset: str = "eurlex"
    n: int = 8000
    users: int = 300
    beta: float = 30.0
    eps_policy: float = 0.1
    lam: float = 10.0
    n_components: int = 100
    max_reward_noise: float = 0.2
    emb_dim: int = 50
    logger: str = "paper"              # paper = softmax(beta * r_hat), r_hat the paper's ridge fitted on test labels; true_q = softmax(beta * q) on the true expected reward (as in the paper's SYNTHETIC generator)
    exclusion: str = "user_fixed"      # units = users; no per-round exclusions exist in the paper's real-data generator
    seed: int = 7000
    K: int = 0
    C: int = 100


def make_real_world(cfg: RealCfg, data_dir):
    ds = RD.ExtremeBanditDataset(dataset_name=cfg.dataset, max_reward_noise=cfg.max_reward_noise, n_components=cfg.n_components, random_state=cfg.seed, data_dir=data_dir)
    bd = ds.obtain_batch_bandit_feedback(n_rounds=cfg.n, n_users=cfg.users, beta=cfg.beta)
    Xu, q = bd["fixed_user_contexts"][:cfg.users], bd["expected_reward"][:cfg.users]  # the paper's loader draws one training doc per round and uses the first n_users as the fixed pool
    if cfg.logger == "paper":
        pi0 = ds.compute_pi_b(Xu, q, beta=cfg.beta)                                 # per user; identical across the user's rounds
    else:                                                                           # true_q logger: redraw actions and rewards under softmax(beta * q), everything else unchanged
        from obp.utils import softmax, sample_action_fast
        pi0 = softmax(cfg.beta * q)
        rng = np.random.RandomState(cfg.seed + 8)                                    # reward stream: must differ from the action-draw seed below (seed + 7 for both made y a function of the action uniform)
        acts = sample_action_fast(pi0[bd["user_idx"]], random_state=cfg.seed + 7)
        bd = {**bd, "action": acts, "reward": rng.binomial(n=1, p=q[bd["user_idx"], acts]), "pscore": pi0[bd["user_idx"], acts]}
    rng = np.random.RandomState(cfg.seed + 11)
    cl = rng.choice(cfg.C, size=ds.n_actions)                                       # the paper's uniform random clusters (train_clustering)
    _, cl = np.unique(cl, return_inverse=True)
    cfg.K = int(ds.n_actions)
    W = G.finish_world(cfg, bd["user_idx"], bd["action"], bd["reward"].astype(float), bd["pscore"], Xu, q, pi0, cl, cfg.n)
    # label co-occurrence embedding of the actions from the TRAINING split (no rewards, no test data)
    lab = ds.train_label.astype(np.float32)
    svd = TruncatedSVD(n_components=min(cfg.emb_dim, lab.shape[1] - 1), random_state=cfg.seed).fit(lab.T)
    emb = svd.transform(lab.T); W["action_x"] = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    W["n_train_docs"], W["n_test_docs"] = int(ds.n_train), int(ds.n_test)
    return W


def calibrate_binomial(W, M, bound=0.25):
    """eps_max from within-(user, cluster) logged contrasts of different actions, as eps_sweep_dgp.calibrate_user_cells, with the Bernoulli noise term plugged in:
    E[(e_i - e_j)^2 / 2] = tau^2 + (v_i + v_j) / 2,  v_i = m_hat_i (1 - m_hat_i) clipped to [0, bound]. Also reports tau with the conservative bound v = 0.25."""
    user, act, y = np.asarray(W["user_of_round"]), W["act"], W["y"]
    C = W["C"]; clr = W["cl"][act]
    mh = np.clip(M[W["uid"], act], 0.0, 1.0); e = y - mh; v = np.clip(mh * (1 - mh), 0.0, bound)
    cell = user.astype(np.int64) * C + clr
    order = np.argsort(cell, kind="stable"); cs, ces, cas, cvs = cell[order], e[order], act[order], v[order]
    halves, noise = [], []
    for idx in np.split(np.arange(len(cs)), np.flatnonzero(np.diff(cs)) + 1):
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1); keep = cas[idx][i] != cas[idx][j]
        halves.append(((ces[idx][i] - ces[idx][j]) ** 2 / 2.0)[keep]); noise.append(((cvs[idx][i] + cvs[idx][j]) / 2.0)[keep])
    halves = np.concatenate(halves) if halves else np.array([]); noise = np.concatenate(noise) if noise else np.array([])
    if not len(halves):
        return dict(sigma_noise=float(np.sqrt(v.mean())), n_repeat_dof=0, n_pairs=0, mean_half_sq_contrast=float("nan"), tau2_raw=float("nan"), tau=float("nan"), eps_max=float("nan"), z=2.0, noise_term="plugin")
    D = float(halves.mean()); raw = D - float(noise.mean()); tau = float(np.sqrt(max(raw, 0.0)))
    raw_b = D - bound
    key = user.astype(np.int64) * 100000 + act; _, cnt = np.unique(key, return_counts=True)
    return dict(sigma_noise=float(np.sqrt(v.mean())), n_repeat_dof=int((cnt - 1).sum()), n_pairs=int(len(halves)), mean_half_sq_contrast=D, noise_term_mean=float(noise.mean()),
                tau2_raw=raw, tau=tau, eps_max=2.0 * tau, z=2.0, noise_term="plugin m_hat(1-m_hat)", tau_with_bound_025=float(np.sqrt(max(raw_b, 0.0))), eps_max_with_bound_025=2.0 * float(np.sqrt(max(raw_b, 0.0))))


def make_fake_xc(root, n_train=600, n_test=300, n_feat=60, n_label=400, n_topics=20, seed=0):
    """A small dataset in the Extreme Classification Repository text format, with topic structure so that labels co-occur and contexts predict them. For plumbing tests only."""
    rng = np.random.RandomState(seed); d = os.path.join(root, "eurlex"); os.makedirs(d, exist_ok=True)
    topic_of_label = rng.randint(n_topics, size=n_label); topic_feat = rng.normal(size=(n_topics, n_feat))
    for split, n in (("train", n_train), ("test", n_test)):
        lines = [f"{n} {n_feat} {n_label}"]
        for _ in range(n):
            t = rng.choice(n_topics, size=2, replace=False); x = topic_feat[t].sum(0) + rng.normal(scale=0.8, size=n_feat)
            labs = np.flatnonzero(np.isin(topic_of_label, t) & (rng.uniform(size=n_label) < 0.35))
            if not len(labs):
                labs = np.array([rng.randint(n_label)])
            nz = np.flatnonzero(np.abs(x) > 0.3)
            lines.append(",".join(map(str, labs)) + " " + " ".join(f"{k}:{x[k]:.4f}" for k in nz))
        open(os.path.join(d, f"{split}.txt"), "w").write("\n".join(lines) + "\n")
    return root
