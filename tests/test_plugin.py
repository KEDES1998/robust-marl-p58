import numpy as np
import pytest

from rmg.games import random_game, toy_public_goods_game
from rmg.plugin import estimate_model, run_plugin
from rmg.simulator import GenerativeModel


def test_counter_is_H_S_joint_m():
    """The plug-in pays for every joint action: H * S * prod(A_i) * m samples."""
    game = toy_public_goods_game()
    sim = GenerativeModel(game, seed=0)
    estimate_model(sim, m=3)
    sp = game.spec
    assert sim.n_samples == sp.H * sp.S * sp.n_joint * 3 == 72


def test_estimate_is_a_valid_game_with_exact_rewards():
    game = random_game(n=3, S=3, H=2, A=(2, 3, 2), seed=0)
    est = estimate_model(GenerativeModel(game, seed=1), m=5)
    est.validate()
    assert est.spec == game.spec
    assert np.allclose(est.R, game.R)


def test_deterministic_transitions_are_estimated_exactly():
    """In the toy game, good and bad are absorbing: every answer is the same state."""
    game = toy_public_goods_game()
    est = estimate_model(GenerativeModel(game, seed=0), m=4)
    assert np.array_equal(est.P[:, 1:], game.P[:, 1:])


def test_estimate_converges_to_true_kernel():
    """Standard error per entry is at most sqrt(0.25 / m) = 0.008 for m = 4000."""
    game = random_game(n=2, S=3, H=2, A=2, seed=3)
    est = estimate_model(GenerativeModel(game, seed=4), m=4000)
    assert np.abs(est.P - game.P).max() < 0.05


def test_needs_at_least_one_sample():
    with pytest.raises(ValueError):
        estimate_model(GenerativeModel(toy_public_goods_game(), seed=0), m=0)


def test_run_plugin_counter():
    """Skipped until ``solve_exact`` exists."""
    game = toy_public_goods_game()
    sim = GenerativeModel(game, seed=0)
    mixture = run_plugin(sim, m=2, sigma=0.2, K=10)
    mixture.validate(game.spec)
    assert sim.n_samples == game.spec.H * game.spec.S * game.spec.n_joint * 2


def test_run_plugin_passes_c_alpha():
    """``c_alpha`` reaches the solver: same samples, same result as solving the estimate by hand."""
    from rmg.robust_q_ftrl import solve_exact

    game = random_game(n=2, S=3, H=2, A=2, seed=0)
    mix = run_plugin(GenerativeModel(game, seed=5), m=3, sigma=0.3, K=50, c_alpha=1.0)
    ref = solve_exact(estimate_model(GenerativeModel(game, seed=5), m=3), 0.3, 50, c_alpha=1.0)
    assert np.allclose(mix.weights, ref.weights)
    assert all(np.allclose(p, q) for p, q in zip(mix.pis, ref.pis))
    default = run_plugin(GenerativeModel(game, seed=5), m=3, sigma=0.3, K=50)
    assert not np.allclose(mix.weights, default.weights)
