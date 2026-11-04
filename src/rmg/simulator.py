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

        Implemented as a batch of size 1, so both methods share one code path and
        draw from the same random stream.

        Returns:
            ``(s_next, r)`` with ``s_next ~ P[h, s, a, :]`` and ``r = R[:, h, s, a]`` (shape ``(n,)``).
        """
        s_next, r = self.sample_batch(h, np.array([s]), np.array([a]))
        return int(s_next[0]), r[0]

    def sample_batch(self, h: int, s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Many queries at once (used by Algorithm 1). Counter + m.

        Next states are drawn by inverse transform sampling: the interval [0, 1) is
        cut into pieces of length P[h, s, a, s'] (the cumulative sums mark the cuts),
        and a uniform random number u falls into the piece of s' with probability
        exactly P[h, s, a, s'].

        Args:
            h: time step.
            s: states, shape ``(m,)``.
            a: joint actions, shape ``(m, n)``.

        Returns:
            ``(s_next, r)`` with shapes ``(m,)`` and ``(m, n)``.
        """
        s = np.asarray(s, dtype=int)
        a = np.asarray(a, dtype=int)
        m = s.shape[0]
        self._check_query(h, s, a)

        idx = (h, s, *a.T)                          # one index per query: (h, s_j, a_1j, ..., a_nj)
        cdf = np.cumsum(self._game.P[idx], axis=1)  # shape (m, S)
        cdf[:, -1] = 1.0                            # guard against rounding, e.g. 0.9999999
        u = self._rng.random(m)
        s_next = (u[:, None] < cdf).argmax(axis=1)  # first s' whose cut lies above u

        r = np.moveaxis(self._game.R, 0, -1)[idx]   # shape (m, n); rewards are deterministic

        self._n_samples += m
        return s_next, r

    def _check_query(self, h: int, s: np.ndarray, a: np.ndarray) -> None:
        """Rejects out-of-range indices (negative ones would silently wrap around in numpy)."""
        sp = self.spec
        if s.ndim != 1 or a.shape != (s.shape[0], sp.n):
            raise ValueError(f"s has shape {s.shape}, a has shape {a.shape}; expected (m,) and (m, {sp.n}).")
        if not 0 <= h < sp.H:
            raise ValueError(f"h = {h} is not in 0, ..., {sp.H - 1}.")
        if s.size and (s.min() < 0 or s.max() >= sp.S):
            raise ValueError(f"A state is not in 0, ..., {sp.S - 1}.")
        if a.size and ((a < 0).any() or (a >= np.array(sp.A)).any()):
            raise ValueError(f"An action is out of range; A = {sp.A}.")
