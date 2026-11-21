"""Robust-Q-FTRL, Algorithm 2 in the paper (Timon: run, Ali: solve_exact).

Interface proposal; Timon and Ali may change the arguments as long as both
functions return a ``PolicyMixture``.

- ``run``: the actual algorithm. Collects data only through the simulator
  (Algorithm 1: for every agent i and pair (s, a_i), N samples while the other
  agents play their current policy). Total cost ``H * K * N * S * sum(A_i)``.
- ``solve_exact``: the same FTRL updates, but with exact expectations on a
  given model instead of samples, and no bonus. Needed by the plug-in baseline
  and for the sanity check "gap -> 0 on the true model".

Loop order (Algorithm 2, lines 3-5): ``for h = H..1: for k = 1..K: for i = 1..n``.
Every time step runs all K iterations against the fixed V[h+1], then sets
``V[h] = sum_k alpha^K_k <pi^k_h, q^k_h>`` (Eq. 25) before moving to h - 1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .evaluate import _sigmas, mixed_kernel
from .games import MarkovGame
from .policies import PolicyMixture
from .robust import tv_worst_case
from .simulator import GenerativeModel


def learning_rates(K: int, H: int, c_alpha: float = 24.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learning rates of Eqs. (21)-(22) and the output weights of Eq. (24).

    Returns ``(alpha, eta, weights)``, each of shape ``(K,)`` and indexed k = 1..K as 0..K-1:

    - ``alpha[k-1] = c log K / (k - 1 + c log K)``; alpha_1 = 1 (also for K = 1, where log K = 0).
    - ``eta[k-1] = eta_{k+1} = sqrt(log K / (alpha_k H))``: the rate for the policy pi^{k+1}
      computed after iteration k.
    - ``weights[k-1] = alpha^K_k = alpha_k prod_{j=k+1}^K (1 - alpha_j)``; they sum to 1
      because alpha_1 = 1.
    """
    if K < 1:
        raise ValueError(f"K = {K}, need at least one iteration.")
    log_K = np.log(K)
    k = np.arange(1, K + 1)
    alpha = np.ones(K) if K == 1 else c_alpha * log_K / (k - 1 + c_alpha * log_K)
    eta = np.sqrt(log_K / (alpha * H))
    # prod_{j=k+1}^K (1 - alpha_j): reversed cumulative product, shifted by one.
    tail = np.append(np.cumprod((1.0 - alpha[1:])[::-1])[::-1], 1.0)
    weights = alpha * tail
    return alpha, eta, weights


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def run(
    sim: GenerativeModel,
    sigma: float | Sequence[float],
    K: int,
    N: int,
    *,
    c_alpha: float = 24.0,
    c_b: float = 1.0,
    delta: float = 0.1,
    seed: int | None = None,
) -> PolicyMixture:
    """Algorithm 2 with learning rates Eqs. (21)-(22) and bonus Eq. (23).

    Args:
        sim: generative model; the only access to the game. Cost ``H * K * N * S * sum(A_i)``.
        sigma: TV radius, one number or one per agent.
        K: FTRL iterations per time step.
        N: samples per (agent i, state s, own action a_i) and iteration (Algorithm 1).
        c_alpha: learning-rate constant of Eq. (21) (the paper requires >= 24).
        c_b: constant of the bonus Eq. (23); 0 switches the bonus off.
        delta: failure probability inside the log term of the bonus.
        seed: seed for the actions the other agents draw in Algorithm 1 (the next states
            come from the simulator's own random stream).

    Returns:
        Mixture over the K plans with weights alpha^K_k (Eq. 24).
    """
    return _run(sim, sigma, K, N, c_alpha=c_alpha, c_b=c_b, delta=delta, seed=seed)[0]


def _query_plan(spec, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[slice]]:
    """Fixed layout of the queries of one call of Algorithm 1 for all agents at once.

    Agent i's block holds the queries ordered as (s, a_i, t), ``S * A_i * N`` of them.
    Returns the state of every query, the own action of every query, the agent that owns
    it, and one slice per agent into the batch.
    """
    S = spec.S
    states, own, owner, blocks, start = [], [], [], [], 0
    for i, A_i in enumerate(spec.A):
        s_idx, a_idx, _ = np.indices((S, A_i, N)).reshape(3, -1)
        states.append(s_idx)
        own.append(a_idx)
        owner.append(np.full(s_idx.size, i))
        blocks.append(slice(start, start + s_idx.size))
        start += s_idx.size
    return np.concatenate(states), np.concatenate(own), np.concatenate(owner), blocks


def _n_sample_estimation(
    sim: GenerativeModel,
    pi_h: list[np.ndarray],
    h: int,
    N: int,
    rng: np.random.Generator,
    plan: tuple[np.ndarray, np.ndarray, np.ndarray, list[slice]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Algorithm 1 for every agent i, in one simulator batch.

    For every query of agent i's block, every other agent j draws its action from
    ``pi_h[j](. | s)`` (Eq. 16), agent i plays its fixed own action. Returns per agent
    the empirical reward ``r_i`` of shape ``(S, A_i)`` and kernel ``P_i`` of shape
    ``(S, A_i, S)`` (line 6 of Algorithm 1).
    """
    spec = sim.spec
    S, n = spec.S, spec.n
    states, own, owner, blocks = plan
    M = states.size

    actions = np.empty((M, n), dtype=int)
    for j in range(n):
        cdf = np.cumsum(pi_h[j], axis=-1)[states]       # (M, A_j)
        cdf[:, -1] = 1.0
        actions[:, j] = (rng.random(M)[:, None] < cdf).argmax(axis=1)
    mine = owner[:, None] == np.arange(n)[None, :]      # a_i(s, a_i) = a_i for the owner
    actions[mine] = own

    s_next, r = sim.sample_batch(h, states, actions)

    r_hat, P_hat = [], []
    for i, A_i in enumerate(spec.A):
        b = blocks[i]
        r_hat.append(r[b, i].reshape(S, A_i, N).mean(axis=-1))
        counts = np.zeros((S * A_i, S))
        np.add.at(counts, (np.arange(S * A_i).repeat(N), s_next[b]), 1.0)
        P_hat.append(counts.reshape(S, A_i, S) / N)
    return r_hat, P_hat


def _run(
    sim: GenerativeModel,
    sigma: float | Sequence[float],
    K: int,
    N: int,
    *,
    c_alpha: float = 24.0,
    c_b: float = 1.0,
    delta: float = 0.1,
    seed: int | None = None,
) -> tuple[PolicyMixture, np.ndarray]:
    """``run`` that also returns the value estimates ``V`` of shape ``(n, H + 1, S)``."""
    if N < 1:
        raise ValueError(f"N = {N}, need at least one sample per (s, a_i).")
    spec = sim.spec
    n, S, H = spec.n, spec.S, spec.H
    sig = _sigmas(sigma, n)
    alpha, eta, weights = learning_rates(K, H, c_alpha)
    rng = np.random.default_rng(seed)
    plan = _query_plan(spec, N)
    # Prefactor of the bonus, Eq. (23): c_b * sqrt(log^3(K S sum_i A_i / delta) / (K H)).
    bonus_scale = c_b * np.sqrt(np.log(K * S * spec.sum_A / delta) ** 3 / (K * H))

    pis = [np.empty((K, H, S, A_i)) for A_i in spec.A]       # pis[i][k, h] = pi^{k+1}_{i,h}
    V = np.zeros((n, H + 1, S))                               # V[:, H] = 0
    for h in reversed(range(H)):                              # line 3
        pi_h = [np.full((S, A_i), 1.0 / A_i) for A_i in spec.A]   # pi^1: uniform
        Q = [np.zeros((S, A_i)) for A_i in spec.A]                # Q^0 = 0
        V_acc = np.zeros((n, S))     # sum_k alpha^K_k <pi^k(.|s), q^k(s,.)>
        var_acc = np.zeros((n, S))   # sum_k alpha^K_k Var_{pi^k(.|s)}(q^k(s,.))
        for k in range(K):                                    # line 4
            for i in range(n):
                pis[i][k, h] = pi_h[i]
            # Lines 5-7 for all agents under the same pi^k_h: one batch of S * sum_i A_i * N queries.
            r_hat, P_hat = _n_sample_estimation(sim, pi_h, h, N, rng, plan)
            for i in range(n):
                q = r_hat[i] + tv_worst_case(P_hat[i], V[i, h + 1], sig[i])   # (S, A_i), Eqs. (18)/(19)
                mean_q = np.sum(pi_h[i] * q, axis=-1)
                V_acc[i] += weights[k] * mean_q
                var_acc[i] += weights[k] * (np.sum(pi_h[i] * q**2, axis=-1) - mean_q**2)
                Q[i] = (1.0 - alpha[k]) * Q[i] + alpha[k] * q                  # line 8
                pi_h[i] = _softmax(eta[k] * Q[i])                              # pi^{k+1}, Eq. (22)
        beta = bonus_scale * np.sqrt(np.maximum(var_acc, 0.0) + H)            # Eq. (23)
        V[:, h] = np.minimum(V_acc + beta, H - h)                              # Eq. (25), 0-based h
    return PolicyMixture(pis=pis, weights=weights), V


def _solve_exact(
    game: MarkovGame, sigma: float | Sequence[float], K: int, c_alpha: float
) -> tuple[PolicyMixture, np.ndarray]:
    """``solve_exact`` that also returns the value estimates ``V`` of shape ``(n, H + 1, S)``."""
    spec = game.spec
    n, S, H = spec.n, spec.S, spec.H
    sig = _sigmas(sigma, n)
    alpha, eta, weights = learning_rates(K, H, c_alpha)

    pis = [np.empty((K, H, S, A_i)) for A_i in spec.A]       # pis[i][k, h] = pi^{k+1}_{i,h}
    V = np.zeros((n, H + 1, S))
    for h in reversed(range(H)):
        pi_h = [np.full((S, A_i), 1.0 / A_i) for A_i in spec.A]   # pi^1: uniform
        Q = [np.zeros((S, A_i)) for A_i in spec.A]                # Q^0 = 0
        V_acc = np.zeros((n, S))                                  # sum_k alpha^K_k <pi^k, q^k>
        for k in range(K):
            for i in range(n):
                pis[i][k, h] = pi_h[i]
            # q^k for all agents under the same pi^k_h (line 5 loops over i with pi^k fixed).
            # mixed_kernel reads only pi[j][h]; pad the time axis so it can index h.
            pi_full = [np.broadcast_to(p, (H, S, p.shape[-1])) for p in pi_h]
            q = []
            for i in range(n):
                Pbar, rbar = mixed_kernel(game, pi_full, i, h)
                q.append(rbar + tv_worst_case(Pbar, V[i, h + 1], sig[i]))   # (S, A_i), Eq. (18)
            for i in range(n):
                V_acc[i] += weights[k] * np.sum(pi_h[i] * q[i], axis=-1)
                Q[i] = (1.0 - alpha[k]) * Q[i] + alpha[k] * q[i]
                pi_h[i] = _softmax(eta[k] * Q[i])                          # pi^{k+1}, Eq. (22)
        V[:, h] = np.minimum(V_acc, H - h)                                 # Eq. (25) without bonus
    return PolicyMixture(pis=pis, weights=weights), V


def solve_exact(game: MarkovGame, sigma: float | Sequence[float], K: int, *, c_alpha: float = 24.0) -> PolicyMixture:
    """The same updates with exact expectations on ``game``, without bonus.

    Line 6 of Algorithm 2 (N-sample estimation) is replaced by the exact mixed kernel
    (Eq. 35) of the current product policy. The bonus beta (Eq. 23) is 0 because there
    is no estimation error.

    Returns:
        Mixture over the K plans with weights alpha^K_k (Eq. 24); plan k at time h is pi^k_h.
    """
    return _solve_exact(game, sigma, K, c_alpha)[0]
