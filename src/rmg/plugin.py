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

from .games import MarkovGame
from .policies import PolicyMixture
from .simulator import GenerativeModel


def estimate_model(sim: GenerativeModel, m: int) -> MarkovGame:
    """Estimates (P_hat, R_hat) from m samples per (h, s, joint action)."""
    raise NotImplementedError("Kevin: plug-in, model estimation")


def run_plugin(sim: GenerativeModel, m: int, sigma: float | Sequence[float], K: int) -> PolicyMixture:
    """Estimates the model, then solves it with ``robust_q_ftrl.solve_exact``.

    Afterwards the sample counter reads ``H * S * prod(A_i) * m``.
    """
    raise NotImplementedError("Kevin: plug-in, full run")
