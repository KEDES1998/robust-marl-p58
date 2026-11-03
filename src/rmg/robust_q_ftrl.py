"""Robust-Q-FTRL, Algorithm 2 in the paper (Timon: run, Ali: solve_exact).

Interface proposal; Timon and Ali may change the arguments as long as both
functions return a ``PolicyMixture``.

- ``run``: the actual algorithm. Collects data only through the simulator
  (Algorithm 1: for every agent i and pair (s, a_i), N samples while the other
  agents play their current policy). Total cost ``H * K * N * S * sum(A_i)``.
- ``solve_exact``: the same FTRL updates, but with exact expectations on a
  given model instead of samples, and no bonus. Needed by the plug-in baseline
  and for the sanity check "gap -> 0 on the true model".
"""

from __future__ import annotations

from typing import Sequence

from .games import MarkovGame
from .policies import PolicyMixture
from .simulator import GenerativeModel


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


def solve_exact(game: MarkovGame, sigma: float | Sequence[float], K: int, *, c_alpha: float = 24.0) -> PolicyMixture:
    """The same updates with exact expectations on ``game``, without bonus."""
    raise NotImplementedError("Ali: exact solver")
