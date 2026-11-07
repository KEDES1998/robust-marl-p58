"""Known-answer tests for the gap evaluation on ``toy_public_goods_game``.

All expected values can be computed by hand: one decision at h = 0, then the
reward of the reached state at h = 1.
"""

import numpy as np
import pytest

from rmg.evaluate import gap_in, gap_out, gap_product, robust_values
from rmg.games import toy_public_goods_game
from rmg.policies import PolicyMixture, deterministic_policy

COOPERATE, DEFECT = 0, 1


def agent1_mixes(game, p_cooperate):
    """Agent 0 always cooperates, agent 1 cooperates with probability p_cooperate."""
    pi = deterministic_policy(game.spec, (COOPERATE, COOPERATE))
    pi[1][...] = [p_cooperate, 1.0 - p_cooperate]
    return pi


CASES = [
    ("both cooperate", (COOPERATE, COOPERATE), 0.0, 0.0),
    ("both defect", (DEFECT, DEFECT), 0.0, 0.2),
    ("both cooperate, sigma 0.2", (COOPERATE, COOPERATE), 0.2, 0.0),
    ("both defect, sigma 0.2", (DEFECT, DEFECT), 0.2, 0.1),
]


@pytest.mark.parametrize("name, actions, sigma, expected", CASES, ids=[c[0] for c in CASES])
def test_deterministic_policies(name, actions, sigma, expected):
    game = toy_public_goods_game()
    pi = deterministic_policy(game.spec, actions)
    assert gap_product(game, pi, sigma) == pytest.approx(expected, abs=1e-9)


def test_mixed_policy():
    """Agent 0 cooperates, agent 1 mixes 50/50, sigma 0.2: agent 1 gains 0.1 by always cooperating."""
    game = toy_public_goods_game()
    assert gap_product(game, agent1_mixes(game, 0.5), 0.2) == pytest.approx(0.1, abs=1e-9)


def test_values_both_cooperate():
    game = toy_public_goods_game()
    V_follow, V_best = robust_values(game, deterministic_policy(game.spec, (COOPERATE, COOPERATE)), 0.0)
    assert V_follow.shape == (2, 3, 3)
    assert V_follow[0, 0, 0] == pytest.approx(0.9)
    assert V_best[0, 0, 0] == pytest.approx(0.9)
    assert np.allclose(V_follow[:, 2], 0.0)


@pytest.mark.parametrize("sigma", [0.0, 0.2])
def test_single_plan_all_gaps_agree(sigma):
    game = toy_public_goods_game()
    pi = agent1_mixes(game, 0.5)
    mix = PolicyMixture.from_product(pi)
    g = gap_product(game, pi, sigma)
    assert gap_out(game, mix, sigma) == pytest.approx(g, abs=1e-9)
    assert gap_in(game, mix, sigma) == pytest.approx(g, abs=1e-9)


@pytest.mark.parametrize("sigma, expected", [(0.0, 0.1), (0.2, 0.05)])
def test_mixture_of_two_plans(sigma, expected):
    """Mixture of "both cooperate" and "both defect", 50 % each."""
    game = toy_public_goods_game()
    plans = [deterministic_policy(game.spec, (COOPERATE, COOPERATE)), deterministic_policy(game.spec, (DEFECT, DEFECT))]
    mix = PolicyMixture.from_list(plans, np.array([0.5, 0.5]))
    assert gap_out(game, mix, sigma) == pytest.approx(expected, abs=1e-9)
    assert gap_in(game, mix, sigma) == pytest.approx(expected, abs=1e-9)


def test_mixed_kernel_matches_brute_force():
    """Eq. (35) by explicit summation over all a_-i, with different A_i."""
    import itertools

    from rmg.evaluate import mixed_kernel
    from rmg.games import random_game

    game = random_game(n=3, S=3, H=2, A=(2, 3, 2), seed=0)
    rng = np.random.default_rng(1)
    pi = [rng.dirichlet(np.ones(A_i), size=(2, 3)) for A_i in game.spec.A]
    h = 1
    for i in range(3):
        Pbar, rbar = mixed_kernel(game, pi, i, h)
        assert Pbar.shape == (3, game.spec.A[i], 3)
        assert rbar.shape == (3, game.spec.A[i])
        assert np.allclose(Pbar.sum(axis=-1), 1.0)
        for s in range(3):
            P_ref = np.zeros((game.spec.A[i], 3))
            r_ref = np.zeros(game.spec.A[i])
            for a in itertools.product(*(range(A_j) for A_j in game.spec.A)):
                w = np.prod([pi[j][h, s, a[j]] for j in range(3) if j != i])
                P_ref[a[i]] += w * game.P[(h, s) + a]
                r_ref[a[i]] += w * game.R[(i, h, s) + a]
            assert np.allclose(Pbar[s], P_ref)
            assert np.allclose(rbar[s], r_ref)


@pytest.mark.parametrize("sigma, follow, best", [(0.0, 0.3, 0.5), (0.2, 0.2, 0.3)])
def test_values_both_defect(sigma, follow, best):
    """Values from the table in Protocol.md, Section 1.6."""
    game = toy_public_goods_game()
    V_follow, V_best = robust_values(game, deterministic_policy(game.spec, (DEFECT, DEFECT)), sigma)
    assert V_follow[:, 0, 0] == pytest.approx([follow, follow])
    assert V_best[:, 0, 0] == pytest.approx([best, best])


def test_per_agent_sigma():
    """A sequence of radii applies sigma_i to agent i only."""
    game = toy_public_goods_game()
    pi = deterministic_policy(game.spec, (DEFECT, DEFECT))
    V_follow, _ = robust_values(game, pi, [0.0, 0.2])
    assert V_follow[:, 0, 0] == pytest.approx([0.3, 0.2])
    with pytest.raises(ValueError):
        robust_values(game, pi, [0.1, 0.2, 0.3])


def test_gap_out_differs_from_gap_in():
    """Same joint distribution, two descriptions: where the ball sits changes the gap.

    Agent 0 cooperates, agent 1 cooperates or defects with 50 % each, sigma = 0.6.
    With next-state values (1, 0) the worst case of p_good is max(p_good - 0.6, 0).

    Mixture of the plans CC and CD (gap_out, one ball per plan), agent 0:
      follow  = 0.5 * max(0.9 - 0.6, 0) + 0.5 * max(0.5 - 0.6, 0)        = 0.15
      defect  = 0.2 + 0.5 * max(0.5 - 0.6, 0) + 0.5 * max(0.1 - 0.6, 0)  = 0.2   -> gap 0.05
    One ball around the mixture (gap_in), agent 0 (its recommendation C says nothing):
      follow  = max(0.7 - 0.6, 0)                                         = 0.1
      defect  = 0.2 + max(0.3 - 0.6, 0)                                   = 0.2   -> gap 0.1
    Agent 1 sees its own plan, so both versions give follow 0.25, best 0.3 -> gap 0.05.

    gap_in equals gap_product of the equivalent product policy (agent 1 mixing 50/50).
    """
    game = toy_public_goods_game()
    plans = [deterministic_policy(game.spec, (COOPERATE, COOPERATE)), deterministic_policy(game.spec, (COOPERATE, DEFECT))]
    mix = PolicyMixture.from_list(plans, np.array([0.5, 0.5]))
    assert gap_out(game, mix, 0.6) == pytest.approx(0.05, abs=1e-9)
    assert gap_in(game, mix, 0.6) == pytest.approx(0.1, abs=1e-9)
    assert gap_product(game, agent1_mixes(game, 0.5), 0.6) == pytest.approx(0.1, abs=1e-9)


def _random_mixture(spec, K, seed):
    rng = np.random.default_rng(seed)
    plans = [[rng.dirichlet(np.ones(A_i), size=(spec.H, spec.S)) for A_i in spec.A] for _ in range(K)]
    return PolicyMixture.from_list(plans, rng.dirichlet(np.ones(K)))


@pytest.mark.parametrize("seed", range(5))
def test_mixture_properties_random_game(seed):
    """Multi-step checks on F1 games with different A_i.

    - sigma = 0: the worst case is linear, so gap_out == gap_in.
    - sigma > 0: the worst case is convex in p (mixing the per-plan optima stays inside
      the ball around the mixed kernel) and monotone in V, so every value with one ball
      around the mixture is <= the value with one ball per plan.
    - K identical plans: all three gaps agree.
    """
    from rmg.evaluate import _values_in, _values_out
    from rmg.games import random_game

    game = random_game(n=3, S=4, H=3, A=(2, 3, 2), seed=seed)
    mix = _random_mixture(game.spec, K=5, seed=seed)
    assert gap_out(game, mix, 0.0) == pytest.approx(gap_in(game, mix, 0.0), abs=1e-12)

    F_out, B_out = _values_out(game, mix, 0.3)
    F_in, B_in = _values_in(game, mix, 0.3)
    assert (F_in <= F_out + 1e-12).all()
    assert (B_in <= B_out + 1e-12).all()

    pi = mix.plan(0)
    same = PolicyMixture.from_list([pi] * 4, np.full(4, 0.25))
    g = gap_product(game, pi, 0.3)
    assert gap_out(game, same, 0.3) == pytest.approx(g, abs=1e-12)
    assert gap_in(game, same, 0.3) == pytest.approx(g, abs=1e-12)
