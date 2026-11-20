# Solver floor versus c_α and K

## Question

The paper needs c_α ≥ 24 for Theorem 2; our sweeps use c_α = 1
([deviations.md](../5_deviations/deviations.md)). How large does the solver floor get with
the paper's value, and how many FTRL iterations K would it need to reach the targets?

The floor is the gap of the [exact solver](../3_methods/exact_solver.md) on the true model
(no samples). No sampling method with the same K and c_α gets below it.

## Design

- Game family F1, setting A (β = 1, w = 0.5), n = 3, S = 3, H = 3, A = 2, σ = 0.3.
- 10 game instances.
- c_α ∈ {0.5, 1, 4, 24}, K ∈ {1000, 4000, 8000, 16000, 32000}: 200 runs of `solve_exact`,
  each evaluated with `gap_out` and `gap_in`.
- Target ε = 0.05 (only used for the share column).

Config `experiments/configs/c_alpha_floor.yaml`, script `experiments/c_alpha_solver_floor.py`,
results `results/c_alpha_floor/20261120-082743/` (`runs.csv`, `weights.csv`, `floor.png`):

```bash
uv run python experiments/c_alpha_solver_floor.py experiments/configs/c_alpha_floor.yaml --workers 8 --plot
```

Wall time 1.5 min with 8 processes (12 min compute; about 9.5 s per run at K = 32000).

## Results

### Floor (mean gap_out over 10 instances, max in brackets)

| c_α | K = 1000 | K = 4000 | K = 8000 | K = 16000 | K = 32000 |
|---|---|---|---|---|---|
| 0.5 | 0.027 (0.033) | 0.009 (0.012) | 0.005 (0.006) | 0.003 (0.004) | 0.002 (0.003) |
| 1 | 0.039 (0.044) | 0.014 (0.019) | 0.008 (0.011) | 0.005 (0.006) | 0.003 (0.005) |
| 4 | 0.080 (0.089) | 0.035 (0.042) | 0.022 (0.029) | 0.013 (0.018) | 0.007 (0.010) |
| 24 | 0.145 (0.171) | 0.095 (0.106) | 0.066 (0.074) | 0.044 (0.050) | 0.029 (0.036) |

Share of instances with floor ≤ 0.05: c_α = 0.5 and 1 at every K; c_α = 4 from K = 4000;
c_α = 24 only from K = 16000.

### Output weights α^K_k (no game needed)

| c_α | K | effective number of plans (1 / Σ w²) | weight on k ≤ K/2 | η_K |
|---|---|---|---|---|
| 0.5 | 1000 | 496 | 0.092 | 25.9 |
| 1 | 1000 | 269 | 0.0085 | 18.3 |
| 4 | 1000 | 72 | 7·10⁻⁹ | 9.2 |
| 24 | 1000 | 13 | 2·10⁻⁴⁵ | 4.0 |
| 24 | 32000 | 258 | 3·10⁻⁷⁵ | 21.2 |

## What the numbers show

- **The floor falls monotonically with smaller c_α and larger K.** For every K, c_α = 24 is
  about 4–10× worse than c_α = 1. With c_α = 24 the floor at K = 32000 (0.029) is about what
  c_α = 1 reaches at K = 1000 (0.039) to 2000 (≈ 0.024 in the scaling sweep).
- **With c_α = 24, ε = 0.05 needs K ≥ 16000** on every instance; ε = 0.1 needs K ≈ 4000–8000.
  Our scaling grid stops at K = 2048, so with the paper's constant Robust-Q-FTRL could not
  reach ε = 0.05 even with infinitely many samples, and every iteration costs
  18·N·n samples.
- **Mechanism.** A larger c_α keeps α_k close to 1 for longer, so the output mixture averages
  over few, late plans (13 effective plans at K = 1000 for c_α = 24, versus 269 for c_α = 1)
  and the step size η_K is smaller (4.0 versus 18.3). The early plans get essentially no
  weight for any c_α ≥ 1, so the large floor is not caused by early, near-uniform plans in
  the mixture. It points to the small step size and the short averaging window instead:
  c_α = 24 at K = 32000 has about the same number of effective plans (258) and step size
  (21.2) as c_α = 1 at K = 1000 (269, 18.3), and a similar floor (0.029 versus 0.039).
- **c_α = 0.5 is better still** (floor 0.027 at K = 1000). The sweeps keep c_α = 1, whose
  floor at K = 2000 (0.022–0.032 mean) is already well below both targets.

## Consequence

c_α = 1 is needed to keep the solver error small compared with ε on our K grid. A run that
satisfies Theorem 2's condition (c_α = 24) would need K ≥ 16000, an 8× larger budget per agent.
