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
- **Where:** [notation.md](../1_introduction/notation.md).
