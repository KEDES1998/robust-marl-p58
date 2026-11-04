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

    Idea: every unit of mass moved from a state with value V(s) to the state with
    the lowest value lowers the expectation by V(s) - min V. The worst case therefore
    removes mass from the highest-valued states first, until sigma is used up:

        result = p . V - sum_s removed(s) * (V(s) - min V)

    Mass removed from the lowest-valued state itself costs nothing, so the formula
    also covers the case where sigma exceeds the mass on all other states.
    """
    p = np.asarray(p, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim != 1 or p.shape[-1] != V.shape[0]:
        raise ValueError(f"p has shape {p.shape}, V has shape {V.shape}; expected (..., S) and (S,).")
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma = {sigma} is not in [0, 1].")

    order = np.argsort(-V)                 # next states from highest to lowest value
    p_sorted = p[..., order]
    gain = V[order] - V.min()              # how much moving one unit of mass away from each state costs
    mass_before = np.cumsum(p_sorted, axis=-1) - p_sorted   # mass on states with higher value
    removed = np.clip(sigma - mass_before, 0.0, p_sorted)    # what is still left of sigma, at most p(s)

    result = p @ V - np.sum(removed * gain, axis=-1)
    return float(result) if result.ndim == 0 else result


def tv_worst_case_dual(p: np.ndarray, V: np.ndarray, sigma: float) -> float | np.ndarray:
    """Dual form, Eq. (19) (Timon). Same inputs and outputs as ``tv_worst_case``."""
    raise NotImplementedError("Timon: TV worst case (dual form)")
