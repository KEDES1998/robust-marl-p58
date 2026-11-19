# Metrics

## Robust coarse correlated equilibrium

A method returns a mixture ξ̂ of K product policies (plans) with weights α^K_k. At every
time step, a shared random draw picks plan k and every agent plays its part of plan k.
The shared draw is what makes the equilibrium **correlated**.

ξ̂ is an **ε-robust CCE** if no agent can gain more than ε by ignoring the recommendation
and committing to its own policy in advance, with values evaluated under the worst case of
the uncertainty set. "Coarse" means the deviator decides without seeing the drawn plan.

## The gap

Robust backward induction from V[H] = 0, for every agent i, time h and state s:

- **V_follow[i, h, s]**: agent i follows the mixture;
- **V_best[i, h, s]**: agent i plays its best action at every (h, s) against the others' part of the mixture (robust best response).

```
gap = max_{i, s} ( V_best[i, 0, s] − V_follow[i, 0, s] )
```

This is the gain of the agent with the most to win, from its best starting state. ξ̂ is an
ε-robust CCE iff gap ≤ ε. The gap is computed exactly on the true model.

## N_all(ε)

N_all(ε) is the number of simulator queries a
method needs before its gap is ≤ ε.

### Cost formulas

The sample count is deterministic given the budget:

| Method | n_samples | Sweep setting (S = H = 3, A = 2) |
|---|---|---|
| plug-in | H · S · ∏ᵢ Aᵢ · m | 9 · 2ⁿ · m |
| Robust-Q-FTRL | H · K · N · S · Σᵢ Aᵢ | 18 · K · N · n |
| exact | 0 | 0 |
