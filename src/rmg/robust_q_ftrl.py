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

    Returns:
        Mixture over the K plans with weights alpha^K_k (Eq. 24).
    """
    raise NotImplementedError("Timon: Robust-Q-FTRL")


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
