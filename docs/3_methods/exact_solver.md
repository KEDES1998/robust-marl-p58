# Exact solver

The FTRL loop of Algorithm 2 with exact expectations on a given model instead of samples.
Code: `robust_q_ftrl.solve_exact(game, sigma, K, c_alpha=)`.

It is used twice:

- on the **true model**: the solver floor (method `exact`). With the same K and c_α, no
  sampling method gets below this gap;
- on the **estimated model**: the second half of the [plug-in baseline](plugin_baseline.md).

[Robust-Q-FTRL](robust_q_ftrl.md) runs the same loop. The only difference is where r̄ and P̄ come from.

## The FTRL loop (Algorithm 2)

```
V[:, H] = 0
for h = H-1, ..., 0:                                  backward in time
    pi^1 = uniform, Q^0 = 0
    for k = 1, ..., K:
        r̄_i, P̄_i = mixed_kernel under pi^k for every agent i       Eq. (35), exact
        q_i   = r̄_i + tv_worst_case(P̄_i, V_i[h+1], sigma_i)         Eq. (18)
        Q_i   = (1 - alpha_k) Q_i + alpha_k q_i                     line 8
        pi_i^{k+1} = softmax(eta_{k+1} Q_i)                          Eq. (22)
    V_i[h] = min( Σ_k alpha^K_k <pi_i^k, q_i^k>, H - h )             Eq. (25), no bonus
output: the K plans with weights alpha^K_k                          Eq. (24)
```

The mixed kernel P̄ is agent i's view of the transition, averaged over what the other agents
play. The uncertainty ball sits around it.
There is no bonus, because there is no estimation error.

## Learning rates

`learning_rates(K, H, c_alpha)`:

```
alpha_k   = c_α log K / (k − 1 + c_α log K)          Eq. (21); alpha_1 = 1
eta_{k+1} = sqrt( log K / (alpha_k H) )              Eq. (22)
alpha^K_k = alpha_k ∏_{j>k} (1 − alpha_j)            Eq. (24); sums to 1
```

A larger c_α keeps more weight on the early plans, which were computed while the policies
were still near uniform. With the paper's c_α = 24, the floor at K = 2000 is 0.10–0.12. That
is why we use c_α = 1 ([deviations.md](../5_deviations/deviations.md)).

## Solver floor on F1

`gap` on the true model, K = 2000, c_α = 1, 10 instances per n:

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| mean | 0.022 | 0.024 | 0.025 | 0.029 | 0.030 | 0.031 | 0.029 |
| max | 0.028 | 0.031 | 0.032 | 0.033 | 0.035 | 0.035 | 0.034 |

ε = 0.1 lies clearly above the floor for every n.
