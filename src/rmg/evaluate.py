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


def mixed_kernel(game: MarkovGame, pi: ProductPolicy, i: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected nominal kernel from agent i's point of view (Eq. 35).

    Returns:
        ``(Pbar, rbar)`` with ``Pbar.shape == (S, A_i, S)`` and ``rbar.shape == (S, A_i)``:
        ``Pbar[s, a_i, :] = sum_{a_-i} pi_-i(a_-i | s) * P[h, s, a_i, a_-i, :]`` and
        ``rbar[s, a_i]    = sum_{a_-i} pi_-i(a_-i | s) * R[i, h, s, a_i, a_-i]``.
    """
    raise NotImplementedError("Ali: expected kernel")


def robust_values(game: MarkovGame, pi: ProductPolicy, sigma: float | Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Follow and deviate values for a product policy.

    Returns:
        ``(V_follow, V_best)``, both of shape ``(n, H + 1, S)``; the last time step is 0.
    """
    raise NotImplementedError("Ali: backward induction for a product policy")


def gap_product(game: MarkovGame, pi: ProductPolicy, sigma: float | Sequence[float]) -> float:
    """Gap of a single product policy (what the plug-in baseline returns)."""
    raise NotImplementedError("Ali: gap of a product policy")


def gap_out(game: MarkovGame, mixture: PolicyMixture, sigma: float | Sequence[float]) -> float:
    """Gap of a mixture, averaging outside the worst case (controlled by Theorem 2)."""
    raise NotImplementedError("Ali: gap_out")


def gap_in(game: MarkovGame, mixture: PolicyMixture, sigma: float | Sequence[float]) -> float:
    """Gap of a mixture treated as one correlated policy (one ball around the mixture)."""
    raise NotImplementedError("Ali: gap_in")
