# Methods

The two methods compared in the experiment, plus the exact solver they share. Both return a
`PolicyMixture` and are evaluated with `gap_out` ([metrics.md](metrics.md)).

| | Plug-in baseline | Robust-Q-FTRL | Exact solver |
|---|---|---|---|
| Code | `plugin.run_plugin(sim, m, sigma, K, c_alpha=)` | `robust_q_ftrl.run(sim, sigma, K, N, c_alpha=, c_b=, delta=, seed=)` | `robust_q_ftrl.solve_exact(game, sigma, K, c_alpha=)` |
| Data | m samples per (h, s, **joint action**) | N samples per (h, k, i, s, **own action**) | none, reads the model |
| Cost | H · S · ∏ᵢ Aᵢ · m | H · K · N · S · Σᵢ Aᵢ | 0 |
| Budget knobs | m (K_solver fixed at 2000) | N, K | K |
| Output | mixture of K plans | mixture of K plans | mixture of K plans |
| Owner | Kevin | Timon | Ali |

All three use the same FTRL loop. They differ only in where the one-step Q-values q come from.
That is TA feedback point 3: the comparison isolates how the data is collected.

## The shared FTRL loop (Algorithm 2)

```
V[:, H] = 0
for h = H-1, ..., 0:                                  backward in time
    pi^1 = uniform, Q^0 = 0
    for k = 1, ..., K:
        get r̄_i, P̄_i under pi^k for every agent i       <- the only difference
        q_i   = r̄_i + tv_worst_case(P̄_i, V_i[h+1], sigma_i)       Eqs. (18)/(19)
        Q_i   = (1 - alpha_k) Q_i + alpha_k q_i                     line 8
        pi_i^{k+1} = softmax(eta_{k+1} Q_i)                          Eq. (22)
    V_i[h] = min( Σ_k alpha^K_k <pi_i^k, q_i^k> + bonus_i, H - h )   Eq. (25)
output: the K plans with weights alpha^K_k                          Eq. (24)
```

Learning rates (`learning_rates(K, H, c_alpha)`):

```
alpha_k  = c_α log K / (k − 1 + c_α log K)          (Eq. 21; alpha_1 = 1)
eta_{k+1} = sqrt( log K / (alpha_k H) )             (Eq. 22)
alpha^K_k = alpha_k ∏_{j>k} (1 − alpha_j)           (Eq. 24; sums to 1)
```

A larger c_α keeps more weight on the early plans, which were computed while the policies
were still near uniform. That is why c_α = 24 leaves a high solver floor
([deviations.md](deviations.md)).

With K = 1, log K = 0, so η = 0 and the output is the uniform policy with weight 1:
**K = 1 learns nothing**. It is used as a triviality marker ([F1_triviality.md](F1_triviality.md)).

## Exact solver: `solve_exact`

r̄ᵢ and P̄ᵢ come from `mixed_kernel` (Eq. 35) of the current product policy, computed
exactly on the given model. There's no bonus, since there's no estimation error. Two uses:

- on the **true model**: the solver floor (method `exact` in `run_scaling.py`);
- on the **estimated model**: the second half of the plug-in baseline.

## Plug-in baseline: `run_plugin`

1. `estimate_model(sim, m)`: query every joint action m times at every (h, s), one batch per h.
   P̂(s' | h, s, a) = count(s') / m. Since rewards are deterministic, R̂ = R exactly.
2. `solve_exact(model, sigma, K, c_alpha)` on (P̂, R̂). It needs no further samples.

The standard error of each P̂ entry is at most 1/(2√m), about 0.044 at m = 128. Expected
behaviour is exponential growth in n through ∏ᵢ Aᵢ, and it can't go below H·S·∏ᵢ Aᵢ
samples (m = 1) even on a trivial game.

## Robust-Q-FTRL: `run`

At every (h, k), **Algorithm 1** (`_n_sample_estimation`) runs for all agents at once:

- for agent i and every (s, aᵢ), N queries in which i plays aᵢ and every other agent j draws
  aⱼ ~ π^k_{j,h}(· | s) (Eq. 16);
- r̂ᵢ(s, aᵢ) is the mean reward and P̂ᵢ(· | s, aᵢ) the empirical next-state frequencies. Their
  expectation is exactly the mixed kernel (`test_n_sample_estimation_matches_mixed_kernel`).

All S · N · Σᵢ Aᵢ queries of one (h, k) go into one `sample_batch` call
(the layout comes from `_query_plan`). This is valid because every agent samples under the
same π^k.

**Bonus** (Eq. 23), added to V before the cap H − h:

```
beta_{i,h}(s) = c_b · sqrt( log³(K S Σᵢ Aᵢ / δ) / (K H) · Σ_k alpha^K_k ( Var_{π^k}(q^k(s, ·)) + H ) )
```

It is switched off in the experiments (`c_b: 0.0`), because with c_b = 1 it exceeds the
value range ([deviations.md](deviations.md)).

Checks (`tests/test_robust_q_ftrl.py`): with N = 5000 and c_b = 0, V matches `solve_exact`
within 0.03 on the toy game; a huge c_b pushes V to the cap. On F1 (n = 3, K = 256), the
gap approaches the exact solver's as N grows (0.095 at N = 64 vs 0.096).

## What the sweeps found (F1, ε = 0.1)

From Protocol §3.5–3.7:

- **N = 1 suffices.** N = 4 lowers the gap by at most 0.01 at 4× the cost. The gap is limited
  by K (optimisation), not by the samples per iteration.
- **The K needed doesn't grow with n:** K = 256 at n = 2–4, then less. Robust-Q-FTRL's cost
  is linear in n, while plug-in's doubles per agent.
- **Constant factor:** for n ≤ 4 Robust-Q-FTRL needs 26–69× more samples than plug-in. The
  plug-in method gets its 2000 solver iterations free; Robust-Q-FTRL pays for every iteration.
  The extrapolated crossover is n ≈ 10, which F1 can't show because it is trivial there.
