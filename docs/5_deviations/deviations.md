# Deviations from the paper

Everything in our implementation or experiments that differs from Shi et al. (2024) or that
the paper leaves open. Each entry: what the paper says, what we do, why, and where it's
documented. The report's method and setup sections should cover every entry marked **report**.

## Algorithm

### 1. Learning-rate constant c_α = 1 instead of ≥ 24 (**report**)

- **Paper:** c_α ≥ 24 (text above Eq. 21). Theorem 2 is proven under this condition.
- **We:** `c_alpha: 1.0` in `scaling_n.yaml`, used by all methods (plug-in solver, exact,
  Robust-Q-FTRL). The function defaults stay 24.
- **Why:** with c_α = 24 the solver floor (exact solver on the true model, no samples) stays
  above ε on our K grid. Mean floor on F1, n = 3, 10 instances
  ([c_alpha_floor.md](../4_experiments/c_alpha_floor.md)):

  | c_α | K = 1000 | K = 4000 | K = 8000 | K = 16000 | K = 32000 |
  |---|---|---|---|---|---|
  | 24 | 0.145 | 0.095 | 0.066 | 0.044 | 0.029 |
  | 4 | 0.080 | 0.035 | 0.022 | 0.013 | 0.007 |
  | 1 | 0.039 | 0.014 | 0.008 | 0.005 | 0.003 |
  | 0.5 | 0.027 | 0.009 | 0.005 | 0.003 | 0.002 |

  With c_α = 24, ε = 0.05 needs K ≥ 16000 on every instance; our grid stops at K = 2048. The
  experiment would measure the solver, not the sample complexity: in an early test the plug-in
  gap didn't fall with more samples. The cause is not weight on early plans (the first half of
  the iterations gets weight < 10⁻⁴⁰ for c_α = 24) but a short averaging window and a small
  step size η (13 effective plans and η_K = 4.0 at K = 1000, versus 269 and 18.3 for c_α = 1).
- **Consequence:** our Robust-Q-FTRL runs fall outside Theorem 2's condition. A faithful run
  would need K ≥ 16000, about 8× the per-agent budget.

### 2. Bonus switched off, c_b = 0 (**report**)

- **Paper:** β_{i,h} in Eq. (23) with "some absolute constant" c_b.
- **We:** `c_b: 0.0` in `scaling_n.yaml`. The function default stays 1.0.
- **Why:** with c_b = 1 the bonus is about 2.1 (K = 256), larger than the value range. Every V
  is capped at H − h, V_{h+1} becomes constant over states, and the worst case has nothing
  to exploit. The gap gets worse (0.126 vs 0.094 at n = 2). The bonus is a proof device for
  the high-probability bound.

### 3. 0-based time and the value cap

- **Paper:** h = 1, …, H, cap H − h + 1 in Eq. (25).
- **We:** h = 0, …, H − 1, cap `H - h`. These are the same values, only re-indexed.
- **Where:** [notation.md](../1_introduction/notation.md).
