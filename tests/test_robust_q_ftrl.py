"""Tests for the exact solver (Algorithm 2 with exact expectations, no bonus)."""

import numpy as np
import pytest

from rmg.evaluate import _values_out, gap_out
from rmg.games import random_game, toy_public_goods_game
from rmg.robust_q_ftrl import _solve_exact, learning_rates, solve_exact


@pytest.mark.parametrize("K", [1, 2, 10, 2000])
def test_learning_rates(K):
    alpha, eta, weights = learning_rates(K, H=3)
    assert alpha[0] == 1.0
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert np.all(np.diff(weights) >= -1e-15)      # later iterations weigh more
    assert np.isfinite(eta).all()


def test_values_are_the_follower_values_of_the_output():
    """With exact q, Eq. (25) without bonus is Eq. (88a): the follower value of gap_out."""
    game = random_game(n=3, S=4, H=3, A=(2, 3, 2), seed=0)
    for sigma in (0.0, 0.3, [0.1, 0.2, 0.5]):
        mixture, V = _solve_exact(game, sigma, K=50, c_alpha=24.0)
        mixture.validate(game.spec)
        V_follow, _ = _values_out(game, mixture, sigma)
        assert np.allclose(V, V_follow, atol=1e-12)


@pytest.mark.parametrize("game", [toy_public_goods_game(), random_game(n=3, S=3, H=3, A=2, seed=0)], ids=["toy", "F1"])
def test_gap_decreases_with_K(game):
    gaps = [gap_out(game, solve_exact(game, 0.3, K), 0.3) for K in (10, 100, 1000)]
    assert gaps[0] > gaps[1] > gaps[2] >= 0.0
