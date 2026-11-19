# Experiment C1: sample complexity versus the number of agents

## Question

Theorem 2 says Robust-Q-FTRL finds an ε-robust CCE with a number of samples that grows with
Σᵢ Aᵢ, not ∏ᵢ Aᵢ. We can't check the constants (H⁶, ε⁻⁴), but we can check the **shape in n**:

> When only the number of agents n changes, does the number of samples needed to reach
> gap ≤ ε grow polynomially (Robust-Q-FTRL) or exponentially (a method that queries every
> joint action)?

Two sub-questions from the README: does N = 1 sample per iteration suffice in practice, and
does a larger σ need fewer samples (not run yet, see [planned.md](planned.md))?

## Hypotheses

| Method | Expected N_all(ε) |
|---|---|
| [plug-in baseline](../3_methods/plugin_baseline.md) | ∝ Aⁿ = 2ⁿ |
| [Robust-Q-FTRL](../3_methods/robust_q_ftrl.md) | ∝ n, if the K and N needed don't grow with n |

## Design

**Held fixed:** game family F1 ([game_F1.md](../2_environment/game_F1.md)), S = 3 states,
H = 3 steps, A = 2 actions per agent, σ = 0.3, c_α = 1, c_b = 0, success rate 90 %.

**Varied:** n = 2, 3, …, 8.

**Repetitions:**
- **10 game instances per n:** independent random games, so results aren't tied to one game;
- **10 sampling seeds per instance and budget:** the randomness of the samples.

**Budgets** (doubling grids):
- plug-in: m = 1, 2, 4, …, 128 samples per joint action, solved with K = 2000
- Robust-Q-FTRL: N ∈ {1, 4}, K ∈ {1, 32, 64, …, 2048}
- exact solver on the true model (K = 2000, no samples): the floor no method can beat.

**Targets:** ε = 0.1 (main) and ε = 0.05.

## Procedure

1. For each n and instance, generate the game.
2. Run every method with every budget and every sampling seed. Count the simulator queries,
   and compute the gap of the output exactly on the true model.
3. Per instance, N_all(ε) = the cheapest budget at which at least 9 of 10 seeds reach
   gap ≤ ε ([metrics.md](../1_introduction/metrics.md)).
4. Across the 10 instances per n: geometric mean.

## Results

### Solver floor

The mean gap of the exact solver on the true model rises slightly with n, from 0.022 (n = 2)
to 0.031 (n = 7). The maximum is 0.035. Both targets are reachable for every n.

### N_all(ε)

Geometric mean over 10 instances, with the typical budget in brackets. All 70 instances were
solved by both methods at both ε.

| n | plug-in, ε = 0.1 | Robust-Q-FTRL, ε = 0.1 | plug-in, ε = 0.05 | Robust-Q-FTRL, ε = 0.05 |
|---|---|---|---|---|
| 2 | 177 (m = 8) | 12,161 (K = 256) | 709 (m = 32) | 59,886 (K = 1024) |
| 3 | 501 (m = 8) | 18,241 (K = 256) | 1,871 (m = 32) | 63,518 (K = 1024) |
| 4 | 1,003 (m = 8) | 26,067 (K = 256) | 2,469 (m = 16) | 104,267 (K = 1024) |
| 5 | 1,746 (m = 8) | 5,760 (K = 32) | 6,984 (m = 32) | 211,728 (K = 2048) |
| 6 | 3,743 (m = 8) | 3,456 (K = 32) | 10,586 (m = 16) | 179,657 (K = 2048) |
| 7 | 7,486 (m = 8) | 4,032 (K = 32) | 22,692 (m = 16) | 120,384 (K = 2048) |
| 8 | 9,216 (m = 4) | 4,608 (K = 32) | 17,198 (m = 8) | 6,080 (K = 32) |

These columns come from the sweep without K = 1. With K = 1 in the grid, Robust-Q-FTRL drops
to about 18·n samples where the uniform policy already suffices ([triviality_check.md](triviality_check.md)).
All Robust-Q-FTRL budgets have N = 1.

### What the numbers show

- **Plug-in grows exponentially.** It needs about m ≈ 8 samples per joint action for every n,
  so its cost doubles per agent. The semi-log slope is 0.96 over all n (×1.94 per agent) and
  1.01 for n = 2–5.
- **Robust-Q-FTRL's per-agent budget doesn't grow with n.** At ε = 0.1 it needs K = 256 for
  n = 2–4, then less. Its cost therefore grows only through Σᵢ Aᵢ = 2n: ×2.1 from n = 2 to 4,
  while plug-in grows ×5.7.
- **N = 1 suffices.** N = 4 lowers the gap by at most 0.01 at 4× the cost. The gap is limited by
  the number of FTRL iterations K, not by the samples per iteration.
- **Robust-Q-FTRL is far more expensive in absolute terms.** For n ≤ 4 it needs 26–69× more
  samples at ε = 0.1 (34–84× at ε = 0.05). Plug-in gets its 2000 solver iterations free on the
  estimated model, while Robust-Q-FTRL pays for every iteration.

## Interpretation

**The shape of the claim holds; the absolute advantage can't be shown on F1.**

- Consistent with Theorem 2: the per-agent budget is independent of n, and the cost has no 2ⁿ term.
- From n ≈ 5 on (ε = 0.1), F1 is trivial: the uniform policy already has gap ≤ ε. Robust-Q-FTRL
  then succeeds at its smallest budget, and its slope fit (−0.41) is meaningless. Only
  **n = 2–4** at ε = 0.1, and n ≤ 6 at ε = 0.05, are informative ([F1_triviality.md](../2_environment/F1_triviality.md)).
- Extrapolating the informative range (plug-in ≈ 9·8·2ⁿ, Robust-Q-FTRL ≈ 18·256·n at ε = 0.1),
  the curves would cross at about n ≈ 10. On F1 this can't be measured.

## Threats to validity

- **Few informative points:** three values of n at ε = 0.1 are thin evidence for a growth rate.
- **Outside Theorem 2's conditions:** c_α = 1 instead of ≥ 24, and c_b = 0 ([deviations.md](../5_deviations/deviations.md)).
- **Baseline advantages:** exact rewards and free solver iterations favour plug-in on constants,
  though not on the growth rate.
- **Grid resolution:** N_all is always a grid point. Doubling grids can overstate the cost by up to 2×.
