# Plug-in baseline

Estimate the full model from samples, then solve the estimated game with the
[exact solver](exact_solver.md). Code: `plugin.run_plugin(sim, m, sigma, K, c_alpha=)`.

## Procedure

1. **Estimate** (`estimate_model(sim, m)`): query every joint action m times at every (h, s),
   one simulator batch per h.

   ```
   P̂(s' | h, s, a) = (number of answers equal to s') / m
   R̂ = R                                (rewards are deterministic)
   ```

2. **Solve** (`solve_exact(model, sigma, K, c_alpha)`): the exact solver on (P̂, R̂), with
   K = 2000 iterations and no further samples.

Using the same solver as Robust-Q-FTRL means the two methods differ only in how they
collect data).

## Cost

```
n_samples = H · S · ∏ᵢ Aᵢ · m          (sweep setting: 9 · 2ⁿ · m)
```

The baseline queries every joint action, so its cost grows exponentially in n. It can't go
below H · S · ∏ᵢ Aᵢ samples (m = 1), even on a trivial game.
