import numpy as np
import pytest

from rmg.robust import tv_worst_case, tv_worst_case_dual


def test_worked_example():
    p = np.array([0.2, 0.3, 0.5])
    V = np.array([0.0, 5.0, 10.0])
    assert tv_worst_case(p, V, 0.3) == pytest.approx(3.5)


def test_mass_runs_out():
    p = np.array([0.0, 0.1, 0.9])
    V = np.array([0.0, 1.0, 0.0])
    assert tv_worst_case(p, V, 0.2) == pytest.approx(0.0)


def test_sigma_zero_is_plain_expectation():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5))
    V = rng.uniform(0, 3, size=5)
    assert tv_worst_case(p, V, 0.0) == pytest.approx(p @ V)


def test_sigma_one_is_minimum():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(5))
    V = rng.uniform(0, 3, size=5)
    assert tv_worst_case(p, V, 1.0) == pytest.approx(V.min())


def test_batch_shape():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(4), size=(3, 2))
    V = rng.uniform(0, 1, size=4)
    out = np.asarray(tv_worst_case(p, V, 0.25))
    assert out.shape == (3, 2)
    assert out[1, 0] == pytest.approx(tv_worst_case(p[1, 0], V, 0.25))


def test_greedy_equals_dual():
    rng = np.random.default_rng(3)
    for _ in range(1000):
        S = rng.integers(2, 7)
        p = rng.dirichlet(np.ones(S))
        V = rng.uniform(0, 5, size=S)
        sigma = rng.uniform(0, 1)
        assert tv_worst_case(p, V, sigma) == pytest.approx(tv_worst_case_dual(p, V, sigma), abs=1e-9)
