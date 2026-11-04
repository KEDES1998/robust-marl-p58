"""Generative model (Eq. 14 in the paper) with a sample counter (Kevin).

Ground rule: algorithms see the world only through ``sample`` and ``sample_batch``.
Every answer increments the counter by 1. The counter is N_all in the plots.
Only the evaluation reads the true model (``MarkovGame.P`` and ``.R``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .games import GameSpec, MarkovGame


class GenerativeModel:
    """Answers queries "what happens at (h, s) under joint action a?".

    Args:
        game: the game to simulate. Stored privately.
        seed: seed for the sampling noise; separate from the seed that generates the game.
    """

    def __init__(self, game: MarkovGame, seed: int | None = None) -> None:
        self._game = game
        self._rng = np.random.default_rng(seed)
        self._n_samples = 0

    @property
    def spec(self) -> GameSpec:
        """Public description (n, S, H, A)."""
        return self._game.spec

    @property
    def n_samples(self) -> int:
        """Number of queries answered so far."""
        return self._n_samples

    def reset_counter(self) -> None:
        self._n_samples = 0

    def sample(self, h: int, s: int, a: Sequence[int]) -> tuple[int, np.ndarray]:
        """One query. Counter + 1.

        Returns:
            ``(s_next, r)`` with ``s_next ~ P[h, s, a, :]`` and ``r = R[:, h, s, a]`` (shape ``(n,)``).
        """
        raise NotImplementedError("Kevin: simulator, single query")

    def sample_batch(self, h: int, s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Many queries at once (used by Algorithm 1). Counter + m.

        Args:
            h: time step.
            s: states, shape ``(m,)``.
            a: joint actions, shape ``(m, n)``.

        Returns:
            ``(s_next, r)`` with shapes ``(m,)`` and ``(m, n)``.
        """
        raise NotImplementedError("Kevin: simulator, batch of queries")
