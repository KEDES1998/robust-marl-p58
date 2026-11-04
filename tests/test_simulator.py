import numpy as np
import pytest

from rmg.games import toy_public_goods_game
from rmg.simulator import GenerativeModel


def test_counter_counts_every_answer():
    sim = GenerativeModel(toy_public_goods_game(), seed=0)
    assert sim.n_samples == 0
    sim.sample(0, 0, (0, 0))
    assert sim.n_samples == 1
    sim.sample_batch(0, np.zeros(100, dtype=int), np.zeros((100, 2), dtype=int))
    assert sim.n_samples == 101
    sim.reset_counter()
    assert sim.n_samples == 0


def test_rewards_are_returned():
    sim = GenerativeModel(toy_public_goods_game(), seed=0)
    _, r = sim.sample(0, 0, (1, 0))
    assert r == pytest.approx([0.2, 0.0])


def test_frequencies_match_true_kernel():
    sim = GenerativeModel(toy_public_goods_game(), seed=1)
    m = 20_000
    s_next, r = sim.sample_batch(0, np.zeros(m, dtype=int), np.zeros((m, 2), dtype=int))
    assert s_next.shape == (m,) and r.shape == (m, 2)
    assert np.mean(s_next == 1) == pytest.approx(0.9, abs=0.01)


def test_same_seed_same_answers():
    a = np.zeros((50, 2), dtype=int)
    s = np.zeros(50, dtype=int)
    x1, _ = GenerativeModel(toy_public_goods_game(), seed=3).sample_batch(0, s, a)
    x2, _ = GenerativeModel(toy_public_goods_game(), seed=3).sample_batch(0, s, a)
    assert np.array_equal(x1, x2)


def test_rejects_out_of_range_queries():
    """Negative indices would silently wrap around in numpy, so the simulator refuses them."""
    sim = GenerativeModel(toy_public_goods_game(), seed=0)
    for h, s, a in [(2, 0, (0, 0)), (0, -1, (0, 0)), (0, 3, (0, 0)), (0, 0, (0, 2)), (0, 0, (0,))]:
        with pytest.raises(ValueError):
            sim.sample(h, s, a)
    assert sim.n_samples == 0
