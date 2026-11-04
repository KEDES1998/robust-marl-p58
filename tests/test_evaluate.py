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
