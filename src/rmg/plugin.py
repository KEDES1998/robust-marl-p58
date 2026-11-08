"""Plug-in baseline (TA feedback point 3, Kevin).

Procedure:

1. Query every joint action at every (h, s) m times and estimate a model
   (P_hat, R_hat) from the answers. Cost: ``H * S * prod(A_i) * m`` samples.
2. Solve the same robust game on this model with the same solver as
   Robust-Q-FTRL in exact mode (``robust_q_ftrl.solve_exact``, Ali), without further samples.

This way the two methods differ only in how they collect data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .games import GameSpec, MarkovGame
from .policies import PolicyMixture
from .robust_q_ftrl import solve_exact
from .simulator import GenerativeModel


def _joint_actions(spec: GameSpec) -> np.ndarray:
    """All joint actions as rows, shape ``(prod(A_i), n)``.

    The order is the same as numpy's C order over the action axes of P and R
    (the last agent's action changes fastest), so a reshape to ``(..., A_1, ..., A_n)``
    puts every estimate back in the right place.
    """
    return np.indices(spec.A).reshape(spec.n, -1).T


def estimate_model(sim: GenerativeModel, m: int) -> MarkovGame:
    """Estimates (P_hat, R_hat) from m samples per (h, s, joint action).

    - ``P_hat[h, s, a, s'] = (number of answers equal to s') / m``
    - ``R_hat[i, h, s, a]`` = mean of the m reward answers (equal to R, since rewards
      are deterministic).

    One batch per time step h, ordered as (state, joint action, repetition), so the
    answers can be reshaped to ``(S, prod(A_i), m)``.
    """
    if m < 1:
        raise ValueError(f"m = {m}, need at least one sample per (h, s, joint action).")
    sp = sim.spec
    joint = _joint_actions(sp)                     # (J, n) with J = prod(A_i)
    J = joint.shape[0]

    s_query = np.repeat(np.arange(sp.S), J * m)                  # s = 0 ... 0, 1 ... 1, ...
    a_query = np.tile(np.repeat(joint, m, axis=0), (sp.S, 1))    # every joint action m times, per state

    P_hat = np.zeros((sp.H, sp.S, J, sp.S))
    R_hat = np.zeros((sp.n, sp.H, sp.S, J))
    for h in range(sp.H):
        s_next, r = sim.sample_batch(h, s_query, a_query)
        s_next = s_next.reshape(sp.S, J, m)
        P_hat[h] = (s_next[..., None] == np.arange(sp.S)).mean(axis=2)   # relative frequencies
        R_hat[:, h] = r.reshape(sp.S, J, m, sp.n).mean(axis=2).transpose(2, 0, 1)

    game = MarkovGame(
        spec=sp,
        P=P_hat.reshape(sp.H, sp.S, *sp.A, sp.S),
        R=R_hat.reshape(sp.n, sp.H, sp.S, *sp.A),
        name=f"plug-in estimate m={m}",
        meta={"method": "plugin", "m": m},
    )
    game.validate()
    return game


def run_plugin(
    sim: GenerativeModel, m: int, sigma: float | Sequence[float], K: int, *, c_alpha: float = 24.0
) -> PolicyMixture:
    """Estimates the model, then solves it with ``robust_q_ftrl.solve_exact``.

    ``c_alpha`` is the learning-rate constant of Eq. (21), passed on to the solver. Use the
    same value as for Robust-Q-FTRL so that the two methods differ only in how they collect
    data. The default is the paper's value 24; with K = 2000 it leaves a solver error of
    about 0.1 on F1 games (n = 2, ..., 4), so the experiment config sets a smaller value.

    Afterwards the sample counter reads ``H * S * prod(A_i) * m``.
    """
    model = estimate_model(sim, m)
    return solve_exact(model, sigma, K, c_alpha=c_alpha)
