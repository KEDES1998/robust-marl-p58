import numpy as np
import pytest

from rmg.games import (
    GameSpec,
    aggregative_game,
    random_game,
    sample_aggregative_params,
    toy_public_goods_game,
)


def test_spec_counts():
    spec = GameSpec(n=3, S=4, H=2, A=(2, 3, 2))
    assert spec.n_joint == 12
    assert spec.sum_A == 7


def test_toy_game_is_valid():
    game = toy_public_goods_game()
    assert game.P.shape == (2, 3, 2, 2, 3)
    assert game.P[0, 0, 0, 0, 1] == pytest.approx(0.9)
    assert game.R[0, 0, 0, 1, 0] == pytest.approx(0.2)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_random_game_valid(n):
    random_game(n=n, S=3, H=3, A=2, seed=0).validate()


def test_random_game_seed_reproducible():
    g1 = random_game(n=3, S=3, H=3, A=2, seed=7)
    g2 = random_game(n=3, S=3, H=3, A=2, seed=7)
    g3 = random_game(n=3, S=3, H=3, A=2, seed=8)
    assert np.array_equal(g1.P, g2.P) and np.array_equal(g1.R, g2.R)
    assert not np.array_equal(g1.P, g3.P)


def test_aggregative_same_params_for_all_n():
    params = sample_aggregative_params(S=3, H=3, A=2, seed=0)
    for n in range(2, 7):
        game = aggregative_game(n, params)
        game.validate()
        assert game.spec.n == n and game.spec.S == 3
