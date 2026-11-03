"""Robust Markov games: reproduction of Shi et al. (2024), arXiv 2409.20067.

Modules and owners:

- ``games``          games and game families F1/F2          (Kevin; F2: anyone)
- ``policies``       policy and mixture formats             (shared)
- ``simulator``      generative model with sample counter   (Kevin)
- ``robust``         TV worst case                          (Kevin greedy, Timon dual form)
- ``evaluate``       exact gap evaluation                   (Ali)
- ``plugin``         plug-in baseline                       (Kevin)
- ``robust_q_ftrl``  Robust-Q-FTRL (Algorithm 2)            (Timon run, Ali solve_exact)
"""

from .games import GameSpec, MarkovGame
from .policies import PolicyMixture, ProductPolicy
from .simulator import GenerativeModel

__all__ = ["GameSpec", "MarkovGame", "ProductPolicy", "PolicyMixture", "GenerativeModel"]
