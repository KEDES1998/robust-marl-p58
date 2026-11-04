"""Policy formats (shared by all modules).

- Product policy (one recommendation): list of n arrays, ``pi[i].shape == (H, S, A_i)``.
  ``pi[i][h, s, a_i]`` is the probability that agent i plays a_i in state s at
  time h. Agents randomize independently of each other.

- Mixture (the output distribution xi-hat in the paper): K product policies with
  weights. Stored stacked per agent: ``pis[i].shape == (K, H, S, A_i)``,
  ``weights.shape == (K,)``. As in the paper (Eq. 24), the same weight alpha^K_k
  applies at every time step h; a plan is drawn independently at each h.

A single product policy is a mixture with K = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import GameSpec

ProductPolicy = list[np.ndarray]


def validate_product_policy(pi: ProductPolicy, spec: GameSpec, atol: float = 1e-8) -> None:
    """Checks shapes and that every distribution sums to 1."""
    if len(pi) != spec.n:
        raise ValueError(f"Policy has {len(pi)} agents, expected {spec.n}.")
    for i, p in enumerate(pi):
        expected = (spec.H, spec.S, spec.A[i])
        if p.shape != expected:
            raise ValueError(f"pi[{i}] has shape {p.shape}, expected {expected}.")
        if (p < -atol).any() or not np.allclose(p.sum(axis=-1), 1.0, atol=atol):
            raise ValueError(f"pi[{i}] does not contain valid distributions.")


def uniform_policy(spec: GameSpec) -> ProductPolicy:
    """Every agent plays every action with equal probability (initial policy of Algorithm 2)."""
    return [np.full((spec.H, spec.S, A_i), 1.0 / A_i) for A_i in spec.A]


def deterministic_policy(spec: GameSpec, actions: tuple[int, ...]) -> ProductPolicy:
    """Agent i always plays ``actions[i]``, in every state and at every time step."""
    pi = []
    for A_i, a in zip(spec.A, actions):
        p = np.zeros((spec.H, spec.S, A_i))
        p[..., a] = 1.0
        pi.append(p)
    return pi


@dataclass
class PolicyMixture:
    """K product policies with weights (the output of Robust-Q-FTRL)."""

    pis: list[np.ndarray]
    weights: np.ndarray

    @property
    def K(self) -> int:
        return len(self.weights)

    @classmethod
    def from_product(cls, pi: ProductPolicy) -> "PolicyMixture":
        """Wraps a single product policy as a mixture with K = 1."""
        return cls(pis=[p[None, ...] for p in pi], weights=np.ones(1))

    @classmethod
    def from_list(cls, policies: list[ProductPolicy], weights: np.ndarray) -> "PolicyMixture":
        """Builds a mixture from a list of product policies."""
        n = len(policies[0])
        pis = [np.stack([pol[i] for pol in policies]) for i in range(n)]
        return cls(pis=pis, weights=np.asarray(weights, dtype=float))

    def plan(self, k: int) -> ProductPolicy:
        """Plan number k as a product policy."""
        return [p[k] for p in self.pis]

    def validate(self, spec: GameSpec, atol: float = 1e-8) -> None:
        if self.weights.ndim != 1 or (self.weights < -atol).any() or not np.isclose(self.weights.sum(), 1.0, atol=atol):
            raise ValueError("weights must be a distribution of shape (K,).")
        for k in range(self.K):
            validate_product_policy(self.plan(k), spec, atol=atol)
