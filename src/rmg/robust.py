"""Worst-case expectation over a total-variation ball.

Computes ``inf { P . V : TV(P, p) <= sigma }`` with ``TV(P, p) = 0.5 * ||P - p||_1``.

Two methods that must give exactly the same result:

- Greedy (Kevin): remove at most sigma probability mass from the next
  states with the highest V and put it on the next state with the lowest V.
- Dual form, Eq. (19) (Timon, used inside the algorithm):
  ``max_alpha { p . [V]_alpha - sigma * (alpha - min_s [V]_alpha(s)) }``
  with alpha between min V and max V and ``[V]_alpha = min(V, alpha)``.

``test_greedy_equals_dual`` compares both on random vectors.
"""

from __future__ import annotations

import numpy as np


def tv_worst_case(p: np.ndarray, V: np.ndarray, sigma: float) -> float | np.ndarray:
    """Greedy solution (Kevin).

    Args:
        p: nominal distribution(s), shape ``(..., S)``; every row sums to 1.
        V: values of the next states, shape ``(S,)``.
        sigma: radius in [0, 1].

    Returns:
        Worst-case expectation, shape ``p.shape[:-1]`` (a scalar for a single row).

    Example: ``V = (0, 5, 10)``, ``p = (0.2, 0.3, 0.5)``, ``sigma = 0.3`` gives 3.5.
    """
    raise NotImplementedError("Kevin: TV worst case (greedy)")


def tv_worst_case_dual(p: np.ndarray, V: np.ndarray, sigma: float) -> float | np.ndarray:
    """Dual form, Eq. (19) (Timon). Same inputs and outputs as ``tv_worst_case``."""
    raise NotImplementedError("Timon: TV worst case (dual form)")
