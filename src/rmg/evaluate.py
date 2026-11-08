"""Exact gap evaluation by robust backward induction (Ali).

Reads the true model of the game. An algorithm must never call these functions
while learning, only for evaluation afterwards.

Two values per agent i, time h and state s (with V[H] = 0):

- follow:  ``V_follow[i, h, s] = sum_{a_i} pi_i(a_i) * Q[i, h, s, a_i]``
- deviate: ``V_best[i, h, s] = max_{a_i} Q*[i, h, s, a_i]``

with ``Q = r^{pi_-i}(s, a_i) + tv_worst_case(Pbar^{pi_-i}(s, a_i), V_next, sigma_i)``,
where V_follow continues with V_follow[h+1] and V_best with V_best[h+1].

Gap = ``max_{i, s} (V_best[i, 0, s] - V_follow[i, 0, s])``.

For a mixture (TA feedback point 2):

- gap_out: the worst case sees the drawn plan k (one ball per plan). This is
  the recursion in the proof of Theorem 2 (Eqs. 87-88), i.e. the quantity
  Theorem 2 controls.
- gap_in: the mixture is treated as one correlated policy with one ball around
  the mixture (Eq. 11). The follower has seen their recommendation (plan
  weights conditioned on a_i); the deviator sees nothing.

For K = 1, gap_out = gap_in = gap_product.

``sigma`` is a number (same for all agents) or a sequence of length n.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .games import MarkovGame
from .policies import PolicyMixture, ProductPolicy
from .robust import tv_worst_case


def _sigmas(sigma: float | Sequence[float], n: int) -> np.ndarray:
    """Radius per agent, shape ``(n,)``; a single number applies to every agent."""
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)) if np.ndim(sigma) == 0 else np.asarray(sigma, dtype=float)
    if sig.shape != (n,):
        raise ValueError(f"sigma has shape {sig.shape}, expected a number or a sequence of length n = {n}.")
    if ((sig < 0.0) | (sig > 1.0)).any():
        raise ValueError(f"sigma = {sig} is not in [0, 1].")
    return sig


def _average_out_others(X: np.ndarray, pis: list[np.ndarray], i: int, h: int) -> np.ndarray:
    """Contracts the action axes of all agents j != i in ``X`` with every plan's ``pi^k_j(. | h, s)``.

    ``X`` has shape ``(S, A_1, ..., A_n, *trailing)``; ``pis[j]`` has shape ``(K, H, S, A_j)``.
    The result has shape ``(K, S, A_i, *trailing)``, one averaged array per plan.

    Method: build the joint weights of the others ``W[k, s, a_-i] = prod_{j != i} pi^k_j(a_j | s)``
    (agents in ascending order, C order), move agent i's axis behind the others in ``X``,
    and contract over a_-i with one batched matrix product per state. About 8x faster
    than contracting one agent at a time (n = 8, K = 2000).
    """
    n = len(pis)
    K, S = pis[0].shape[0], X.shape[0]
    trailing = X.shape[1 + n:]
    A_i = X.shape[1 + i]
    W = np.ones((K, S, 1))
    for j in range(n):
        if j != i:
            W = (W[..., None] * pis[j][:, h][:, :, None, :]).reshape(K, S, -1)
    X = np.moveaxis(X, 1 + i, n)                            # (S, A_-i..., A_i, *trailing)
    X = X.reshape(S, W.shape[2], A_i * int(np.prod(trailing, dtype=int)))
    out = np.matmul(W.transpose(1, 0, 2), X)                # (S, K, A_i * T)
    return out.transpose(1, 0, 2).reshape((K, S, A_i) + trailing)


def _mixed_kernels(game: MarkovGame, pis: list[np.ndarray], i: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """``mixed_kernel`` for all K plans at once: shapes ``(K, S, A_i, S)`` and ``(K, S, A_i)``."""
    return _average_out_others(game.P[h], pis, i, h), _average_out_others(game.R[i, h], pis, i, h)


def mixed_kernel(game: MarkovGame, pi: ProductPolicy, i: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected nominal kernel from agent i's point of view (Eq. 35).

    Returns:
        ``(Pbar, rbar)`` with ``Pbar.shape == (S, A_i, S)`` and ``rbar.shape == (S, A_i)``:
        ``Pbar[s, a_i, :] = sum_{a_-i} pi_-i(a_-i | s) * P[h, s, a_i, a_-i, :]`` and
        ``rbar[s, a_i]    = sum_{a_-i} pi_-i(a_-i | s) * R[i, h, s, a_i, a_-i]``.
    """
    Pbar, rbar = _mixed_kernels(game, [p[None] for p in pi], i, h)
    return Pbar[0], rbar[0]


def _values_out(game: MarkovGame, mixture: PolicyMixture, sigma: float | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Follow and deviate values with one ball per plan (Eqs. 87-88).

    With Q^k(s, a_i) = rbar^k(s, a_i) + worst case over the ball around Pbar^k(s, a_i):

    - ``V_follow[h, s] = sum_k alpha_k sum_{a_i} pi^k_i(a_i | s) Q^k(s, a_i)``  (V_follow[h+1] inside)
    - ``V_best[h, s]   = max_{a_i} sum_k alpha_k Q^k(s, a_i)``                   (V_best[h+1] inside)

    The max is outside the sum over k: the deviator commits to an action without
    seeing the drawn plan (coarse correlated equilibrium).
    """
    spec = game.spec
    sig = _sigmas(sigma, spec.n)
    alpha = mixture.weights
    V_follow = np.zeros((spec.n, spec.H + 1, spec.S))
    V_best = np.zeros((spec.n, spec.H + 1, spec.S))
    for h in reversed(range(spec.H)):
        for i in range(spec.n):
            Pk, rk = _mixed_kernels(game, mixture.pis, i, h)                # (K, S, A_i, S), (K, S, A_i)
            Q_follow = rk + tv_worst_case(Pk, V_follow[i, h + 1], sig[i])   # (K, S, A_i)
            Q_best = rk + tv_worst_case(Pk, V_best[i, h + 1], sig[i])
            V_follow[i, h] = np.einsum("k,ksa,ksa->s", alpha, mixture.pis[i][:, h], Q_follow)
            V_best[i, h] = np.einsum("k,ksa->sa", alpha, Q_best).max(axis=-1)
    return V_follow, V_best


def _values_in(game: MarkovGame, mixture: PolicyMixture, sigma: float | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Follow and deviate values with one ball around the mixture (Eq. 11).

    - Follower: has seen the recommendation a_i, so the others play the plans weighted
      by ``alpha_k pi^k_i(a_i | s)`` (renormalised). One ball around that kernel.
    - Deviator: sees nothing, so the others play the plans weighted by ``alpha_k``.
      One ball around that kernel.
    """
    spec = game.spec
    sig = _sigmas(sigma, spec.n)
    alpha = mixture.weights
    V_follow = np.zeros((spec.n, spec.H + 1, spec.S))
    V_best = np.zeros((spec.n, spec.H + 1, spec.S))
    for h in reversed(range(spec.H)):
        for i in range(spec.n):
            Pk, rk = _mixed_kernels(game, mixture.pis, i, h)

            P_dev = np.einsum("k,ksaz->saz", alpha, Pk)
            r_dev = np.einsum("k,ksa->sa", alpha, rk)
            V_best[i, h] = (r_dev + tv_worst_case(P_dev, V_best[i, h + 1], sig[i])).max(axis=-1)

            joint = alpha[:, None, None] * mixture.pis[i][:, h]            # (K, S, A_i): P(plan k, a_i | s)
            xi_i = joint.sum(axis=0)                                        # (S, A_i): P(a_i | s)
            # P(plan k | a_i, s); for a never-recommended a_i the value is irrelevant (weight 0), use alpha.
            cond = np.where(xi_i > 0, joint / np.where(xi_i > 0, xi_i, 1.0), alpha[:, None, None])
            P_fol = np.einsum("ksa,ksaz->saz", cond, Pk)
            r_fol = np.einsum("ksa,ksa->sa", cond, rk)
            V_follow[i, h] = np.sum(xi_i * (r_fol + tv_worst_case(P_fol, V_follow[i, h + 1], sig[i])), axis=-1)
    return V_follow, V_best


def _gap(V_follow: np.ndarray, V_best: np.ndarray) -> float:
    """``max_{i, s} (V_best[i, 0, s] - V_follow[i, 0, s])``."""
    return float(np.max(V_best[:, 0] - V_follow[:, 0]))


def robust_values(game: MarkovGame, pi: ProductPolicy, sigma: float | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Follow and deviate values for a product policy.

    Backward induction from ``V[H] = 0``. At every (i, h) the uncertainty ball of radius
    sigma_i sits around agent i's mixed kernel (the fictitious RMG). V_follow plays
    pi_i; V_best plays the best action at every (h, s), i.e. the robust best response
    to pi_-i. This is the K = 1 case of the mixture recursion.

    Returns:
        ``(V_follow, V_best)``, both of shape ``(n, H + 1, S)``; the last time step is 0.
    """
    return _values_out(game, PolicyMixture.from_product(pi), sigma)


def gap_product(game: MarkovGame, pi: ProductPolicy, sigma: float | Sequence[float]) -> float:
    """Gap of a single product policy (what the plug-in baseline returns)."""
    return _gap(*robust_values(game, pi, sigma))


def gap_out(game: MarkovGame, mixture: PolicyMixture, sigma: float | Sequence[float]) -> float:
    """Gap of a mixture, averaging outside the worst case (controlled by Theorem 2)."""
    return _gap(*_values_out(game, mixture, sigma))


def gap_in(game: MarkovGame, mixture: PolicyMixture, sigma: float | Sequence[float]) -> float:
    """Gap of a mixture treated as one correlated policy (one ball around the mixture)."""
    return _gap(*_values_in(game, mixture, sigma))
