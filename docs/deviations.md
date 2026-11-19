# Deviations from the paper

Everything in our implementation or experiments that differs from Shi et al. (2024) or that
the paper leaves open. Each entry: what the paper says, what we do, why, and where it's
documented. The report's method and setup sections should cover every entry marked **report**.

## Algorithm

### 1. Learning-rate constant c_α = 1 instead of ≥ 24 (**report**)

- **Paper:** c_α ≥ 24 (text above Eq. 21). Theorem 2 is proven under this condition.
- **We:** `c_alpha: 1.0` in both configs, used by all methods (plug-in solver, exact, Robust-Q-FTRL).
  The function defaults stay 24.
- **Why:** with c_α = 24 the early, near-uniform plans keep too much weight. Even on the true
  model the gap stays above ε:

  | c_α | K = 200 | K = 2000 | K = 10000 |
  |---|---|---|---|
  | 24 | 0.12–0.22 | 0.10–0.12 | 0.06 |
  | 4 | 0.11–0.15 | 0.05–0.06 | 0.016–0.024 |
  | 1 | 0.08–0.10 | 0.02–0.03 | 0.004–0.012 |

  With c_α = 24 the experiment would measure the solver, not the sample complexity: the plug-in
  gap didn't fall with more samples.
- **Consequence:** our Robust-Q-FTRL runs fall outside Theorem 2's condition.

### 2. Bonus switched off, c_b = 0 (**report**)

- **Paper:** β_{i,h} in Eq. (23) with "some absolute constant" c_b.
- **We:** `c_b: 0.0` in `scaling_n_f1_rq.yaml`. The function default stays 1.0.
- **Why:** with c_b = 1 the bonus is about 2.1 (K = 256), larger than the value range. Every V
  is capped at H − h, V_{h+1} becomes constant over states, and the worst case has nothing
  to exploit. The gap gets worse (0.126 vs 0.094 at n = 2). The bonus is a proof device for
  the high-probability bound.

### 3. 0-based time and the value cap

- **Paper:** h = 1, …, H, cap H − h + 1 in Eq. (25).
- **We:** h = 0, …, H − 1, cap `H - h`. These are the same values, only re-indexed.
- **Where:** [notation.md](notation.md).

### 4. One simulator batch per (h, k) for all agents

- **Paper:** Algorithm 2, line 5 loops over agents i.
- **We:** all agents' queries of one (h, k) go into one `sample_batch`. This is equivalent,
  because all agents sample under the same π^k and the update doesn't change π^k. Every
  query is still counted.
- **Where:** Protocol §3.3.

### 5. Exact solver for the baseline

- **Paper:** no baseline of this form.
- **We:** the plug-in baseline solves the estimated game with `solve_exact`, the same FTRL
  updates with exact expectations (TA feedback point 3).
- **Where:** [methods.md](methods.md).

## Evaluation

### 6. Reported gap = `gap_out` (**report**)

- **Paper:** defines the robust CCE for a correlated policy (Eq. 11). The proof of Theorem 2
  controls the recursion with one ball per plan (Eqs. 87–88).
- **We:** we report `gap_out` (the quantity the theorem controls) and also log `gap_in`
  (Eq. 11). They differ by at most 0.011 and never change N_all at ε = 0.1.
- **Where:** [metrics.md](metrics.md), Protocol §1.9.

### 7. Gap over all start states

- **We:** gap = max over agents **and start states s** at h = 0. There is no initial-state
  distribution, so this is the strictest choice.

## Experiment design

### 8. Plug-in rewards are exact (**report**)

Rewards are deterministic, so the plug-in estimate R̂ = R. Robust-Q-FTRL's reward estimate
averages over the others' sampled actions and is noisy. This is a small advantage for the
baseline. Where: Protocol §1.5, [simulator.md](simulator.md).

### 9. Synthetic game family F1 (**report**)

- **Paper:** no experiments with specific games.
- **We:** F1, a random game with Dirichlet(1) transitions and rewards w·g(s) + (1 − w)·u,
  w = 0.5, S = 3, H = 3, A = 2, n = 2–8 ([game_F1.md](game_F1.md)).
- **Limitation:** F1 becomes trivial from n ≈ 5–6 at ε = 0.1. The informative range is
  n = 2–4 (ε = 0.1) or n ≤ 6 (ε = 0.05) ([F1_triviality.md](F1_triviality.md)).

### 10. Fixed σ = 0.3 for all agents

The theory allows agent-specific σᵢ, and the code accepts a sequence. All runs use one σ.
The σ sweep (does a larger σ need fewer samples?) hasn't been run yet.

### 11. N_all from a doubling grid and a success rate (**report**)

- **Paper:** a high-probability bound (1 − δ).
- **We:** N_all(ε) = the smallest grid budget where ≥ 90 % of 10 sampling seeds reach gap ≤ ε,
  aggregated over 10 instances by geometric mean ([metrics.md](metrics.md)). N_all is
  therefore a grid point, and the grid minimum when the smallest budget already succeeds.

### 12. Robust-Q-FTRL grid: N ∈ {1, 4}, K ∈ {1, 32, …, 2048}

`scaling_n_f1_rq.yaml` uses a smaller N grid than `scaling_n.yaml` (N up to 32), because
N = 1 already suffices (Protocol §3.5). K = 1 is in the grid only as a triviality marker,
since it outputs the uniform policy.

### 13. Solver iterations are free for the baseline

The plug-in method runs K = 2000 solver iterations on its estimated model at no sample cost.
Robust-Q-FTRL pays S·N·Σᵢ Aᵢ samples for every iteration. This explains most of the constant
factor between the two methods (Protocol §3.7, caveat 2).
