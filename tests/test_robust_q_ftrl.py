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


# ---------------------------------------------------------------------------
# run: Algorithm 2 with N-sample estimation (Algorithm 1)
# ---------------------------------------------------------------------------

from rmg.evaluate import mixed_kernel  # noqa: E402
from rmg.robust_q_ftrl import _n_sample_estimation, _query_plan, _run, run  # noqa: E402
from rmg.simulator import GenerativeModel  # noqa: E402


@pytest.mark.parametrize("K, N", [(1, 1), (5, 3)])
def test_run_sample_count_is_sum_not_product(K, N):
    """Cost H * K * N * S * sum_i A_i, never the number of joint actions."""
    game = random_game(n=3, S=4, H=2, A=(2, 3, 2), seed=0)
    sim = GenerativeModel(game, seed=0)
    mixture = run(sim, 0.2, K, N, c_alpha=1.0, seed=0)
    mixture.validate(game.spec)
    assert sim.n_samples == 2 * K * N * 4 * (2 + 3 + 2)


def test_n_sample_estimation_matches_mixed_kernel():
    """Algorithm 1 is unbiased for the mixed kernel of Eq. (35)."""
    game = random_game(n=3, S=3, H=2, A=(2, 3, 2), seed=1)
    rng = np.random.default_rng(0)
    pi_h = [rng.dirichlet(np.ones(A_i), size=3) for A_i in game.spec.A]
    sim = GenerativeModel(game, seed=0)
    r_hat, P_hat = _n_sample_estimation(sim, pi_h, 1, 20000, rng, _query_plan(game.spec, 20000))
    pi_full = [np.broadcast_to(p, (2, *p.shape)) for p in pi_h]
    for i in range(3):
        Pbar, rbar = mixed_kernel(game, pi_full, i, 1)
        assert np.abs(P_hat[i] - Pbar).max() < 0.02
        assert np.abs(r_hat[i] - rbar).max() < 0.02


def test_run_approaches_exact_solver_for_large_N():
    game = toy_public_goods_game()
    _, V_exact = _solve_exact(game, 0.2, K=30, c_alpha=1.0)
    _, V = _run(GenerativeModel(game, seed=0), 0.2, K=30, N=5000, c_alpha=1.0, c_b=0.0, seed=0)
    assert np.abs(V - V_exact).max() < 0.03


def test_run_is_reproducible():
    game = random_game(n=2, S=3, H=2, A=2, seed=0)
    a = run(GenerativeModel(game, seed=3), 0.3, 10, 2, seed=4)
    b = run(GenerativeModel(game, seed=3), 0.3, 10, 2, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a.pis, b.pis))


def test_bonus_is_optimistic_and_capped():
    """The bonus only raises V (Eq. 25); a huge c_b caps V at H - h."""
    game = random_game(n=2, S=3, H=3, A=2, seed=0)
    _, V0 = _run(GenerativeModel(game, seed=0), 0.3, 20, 2, c_b=0.0, seed=0)
    _, Vbig = _run(GenerativeModel(game, seed=0), 0.3, 20, 2, c_b=1e6, seed=0)
    caps = np.array([3, 2, 1, 0], dtype=float)[:, None]
    assert np.all(V0 <= caps + 1e-12)
    assert np.allclose(Vbig, np.broadcast_to(caps, Vbig.shape))

