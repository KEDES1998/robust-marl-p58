"""Games: finite-horizon Markov games with n agents (Kevin; F2: anyone).

Conventions (binding for all modules):

- Indices are 0-based: time step ``h = 0, ..., H-1`` (1, ..., H in the paper),
  state ``s = 0, ..., S-1``, action of agent i ``a_i = 0, ..., A_i - 1``.
- A joint action is a tuple ``(a_1, ..., a_n)`` and indexes the arrays
  directly: ``P[h, s, a_1, ..., a_n, s_next]``.
- Rewards are deterministic and lie in [0, 1] (as in the paper).

Game families (TA feedback point 1: only n changes, several instances per n):

- F1 ``random_game``: unstructured, every joint action has its own distributions.
- F2 ``aggregative_game``: other agents act only through the action histogram.
  The same parameters yield a game for every n (paired comparison).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import numpy as np


@dataclass(frozen=True)
class GameSpec:
    """Public description of a game. Algorithms are allowed to see this."""

    n: int
    S: int
    H: int
    A: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.A) != self.n:
            raise ValueError(f"A has {len(self.A)} entries, expected n = {self.n}.")

    @property
    def n_joint(self) -> int:
        """Number of joint actions, the product of the A_i."""
        return prod(self.A)

    @property
    def sum_A(self) -> int:
        """Sum of the A_i; Theorem 2 says Robust-Q-FTRL scales with this."""
        return sum(self.A)


@dataclass
class MarkovGame:
    """A game including the true model. Only the evaluation may read ``P`` and ``R`` directly.

    Attributes:
        spec: public description (n, S, H, A).
        P: transitions, shape ``(H, S, A_1, ..., A_n, S)``; every last axis sums to 1.
        R: rewards, shape ``(n, H, S, A_1, ..., A_n)``, values in [0, 1].
        name: free-form label, e.g. ``"F1 n=4 seed=3"``.
        meta: parameters the game was generated with (for reproducibility).
    """

    spec: GameSpec
    P: np.ndarray
    R: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def validate(self, atol: float = 1e-8) -> None:
        """Checks shapes, probabilities and the reward range."""
        sp = self.spec
        p_shape = (sp.H, sp.S, *sp.A, sp.S)
        r_shape = (sp.n, sp.H, sp.S, *sp.A)
        if self.P.shape != p_shape:
            raise ValueError(f"P has shape {self.P.shape}, expected {p_shape}.")
        if self.R.shape != r_shape:
            raise ValueError(f"R has shape {self.R.shape}, expected {r_shape}.")
        if (self.P < -atol).any():
            raise ValueError("P contains negative probabilities.")
        if not np.allclose(self.P.sum(axis=-1), 1.0, atol=atol):
            raise ValueError("At least one row of P does not sum to 1.")
        if (self.R < -atol).any() or (self.R > 1 + atol).any():
            raise ValueError("R is not in [0, 1].")


def random_game(n: int, S: int, H: int, A: int | tuple[int, ...], seed: int, beta: float = 1.0) -> MarkovGame:
    """Game family F1: unstructured random game.

    Recipe (identical for every n; ``seed`` alone determines the instance):

    - Every state has a quality ``g(s)`` spread evenly over [0, 1], e.g. 0, 0.5, 1 for S = 3.
      Without it, good and bad states barely differ and the worst case has
      nothing to exploit.
    - ``R[i, h, s, a] = 0.5 * g(s) + 0.5 * u`` with ``u ~ Unif[0, 1]`` per entry.
    - ``P[h, s, a, :] ~ Dirichlet(beta * 1)``.

    Args:
        A: number of actions per agent; a single int means "the same for all".
        beta: Dirichlet concentration; small = peaked distributions, large = flat ones.
    """
    A = (A,) * n if isinstance(A, int) else tuple(A)
    spec = GameSpec(n=n, S=S, H=H, A=A)
    rng = np.random.default_rng(seed)

    # Transitions: one Dirichlet draw per (h, s, joint action); last axis = next state.
    P = rng.dirichlet(beta * np.ones(S), size=(H, S, *A))

    # Rewards: half state quality (same for everyone), half noise (per agent and entry).
    g = np.linspace(0.0, 1.0, S)
    g_broadcast = g.reshape((1, 1, S) + (1,) * n)           # aligns with axis s of R
    R = 0.5 * g_broadcast + 0.5 * rng.uniform(size=(n, H, S, *A))

    game = MarkovGame(
        spec=spec,
        P=P,
        R=R,
        name=f"F1 n={n} seed={seed}",
        meta={"family": "F1", "n": n, "S": S, "H": H, "A": A, "seed": seed, "beta": beta},
    )
    game.validate()
    return game


@dataclass
class AggregativeParams:
    """Parameters of F2, independent of n.

    Attributes (A = number of actions, the same for all agents):
        theta: shape ``(H, S, S)``, baseline transition logits.
        W: shape ``(H, S, S, A)``, effect of the action histogram on transitions.
        c: shape ``(S, A)``, own-action part of the reward.
        d: shape ``(S, A)``, effect of the histogram on the reward.
        g: shape ``(S,)``, state quality.
    """

    S: int
    H: int
    A: int
    theta: np.ndarray
    W: np.ndarray
    c: np.ndarray
    d: np.ndarray
    g: np.ndarray
    seed: int


def sample_aggregative_params(S: int, H: int, A: int, seed: int) -> AggregativeParams:
    """Draws one F2 instance. The same instance can be used for every n."""
    raise NotImplementedError("optional, anyone: game family F2 (parameters)")


def aggregative_game(n: int, params: AggregativeParams) -> MarkovGame:
    """Game family F2: other agents act only through the action histogram.

    With ``phi(a)[b] = share of agents playing action b`` (length A, whatever n is):

    - ``P[h, s, a, :] = softmax(theta[h, s] + W[h, s] @ phi(a))``
    - ``R[i, h, s, a] = 0.5 * g(s) + 0.5 * sigmoid(c[s, a_i] + d[s] @ phi(a))``
    """
    raise NotImplementedError("optional, anyone: game family F2 (game for a given n)")


TOY_P_GOOD = np.array([[0.9, 0.5], [0.5, 0.1]])
"""Probability of reaching the good state in ``toy_public_goods_game``.

Row = action of agent 0, column = action of agent 1; 0 = cooperate, 1 = defect.
"""


def toy_public_goods_game() -> MarkovGame:
    """Two-agent public goods game with hand-computable gaps, used as a known-answer test.

    States: 0 = start, 1 = good, 2 = bad. Horizon H = 2. Actions: 0 = cooperate, 1 = defect.

    - h = 0 in start: defecting pays 0.2 immediately, cooperating pays 0. The next state
      is good with probability ``TOY_P_GOOD[a_0, a_1]``, otherwise bad.
    - h = 1 in good: every agent gets 1. Every other reward is 0.
    - Good and bad are absorbing; nothing there depends on the actions.

    Each agent would rather let the other one cooperate. Expected gaps (see
    ``tests/test_evaluate.py``): both cooperate 0, both defect 0.2; with sigma = 0.2:
    both cooperate 0, both defect 0.1.
    """
    spec = GameSpec(n=2, S=3, H=2, A=(2, 2))
    P = np.zeros((2, 3, 2, 2, 3))
    R = np.zeros((2, 2, 3, 2, 2))
    for a0 in range(2):
        for a1 in range(2):
            p = TOY_P_GOOD[a0, a1]
            P[0, 0, a0, a1] = [0.0, p, 1.0 - p]
            R[0, 0, 0, a0, a1] = 0.2 * a0
            R[1, 0, 0, a0, a1] = 0.2 * a1
    for h in range(2):
        P[h, 1, ..., 1] = 1.0
        P[h, 2, ..., 2] = 1.0
    P[1, 0, ..., 0] = 1.0
    R[:, 1, 1] = 1.0
    game = MarkovGame(spec=spec, P=P, R=R, name="toy public goods game")
    game.validate()
    return game
