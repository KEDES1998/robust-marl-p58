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
collect data (TA feedback point 3).

## Cost

```
n_samples = H · S · ∏ᵢ Aᵢ · m          (sweep setting: 9 · 2ⁿ · m)
```

The baseline queries every joint action, so its cost grows exponentially in n. It can't go
below H · S · ∏ᵢ Aᵢ samples (m = 1), even on a trivial game.

## Accuracy

Each P̂ entry is a relative frequency of m trials, so

```
E[P̂] = P,   Var(P̂) = P(1 − P) / m ≤ 1 / (4m)
```

The standard error is at most 1/(2√m), about 0.044 at m = 128. The reward estimate is exact,
so the error comes only from P̂.

## Results on F1 (ε = 0.1)

The samples needed per (h, s, joint action) stay at about m* ≈ 8 for every n, so N_all
doubles with every extra agent. The slope of log₂ N_all against n is 1.01 for n = 2–5.
At n = 8, m* drops to 4 because F1 becomes trivial ([F1_triviality.md](../2_environment/F1_triviality.md)).
