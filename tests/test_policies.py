import numpy as np

from rmg.games import toy_public_goods_game
from rmg.policies import PolicyMixture, deterministic_policy, uniform_policy, validate_product_policy


def test_uniform_and_deterministic_are_valid():
    spec = toy_public_goods_game().spec
    validate_product_policy(uniform_policy(spec), spec)
    pi = deterministic_policy(spec, (0, 1))
    validate_product_policy(pi, spec)
    assert pi[1][0, 0].tolist() == [0.0, 1.0]


def test_mixture_roundtrip():
    spec = toy_public_goods_game().spec
    plans = [deterministic_policy(spec, (0, 0)), uniform_policy(spec)]
    mix = PolicyMixture.from_list(plans, np.array([0.3, 0.7]))
    mix.validate(spec)
    assert mix.K == 2 and mix.pis[0].shape == (2, spec.H, spec.S, 2)
    assert np.array_equal(mix.plan(1)[0], plans[1][0])
    assert PolicyMixture.from_product(plans[0]).K == 1
