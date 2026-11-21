# Why F1 becomes trivial for large n

## What "trivial" means here

The baseline is the **uniform policy**: every agent plays each action with equal probability.
It costs zero samples, and it is the starting policy of Robust-Q-FTRL (Algorithm 2).

F1 is trivial at n when the uniform policy already has `gap ≤ ε` on most instances. Then:

- any method succeeds with the smallest budget on the grid, so N_all(ε) is the grid
  minimum, not a measured requirement;
- a method that learned nothing would still look successful.

## The mechanism

### 1. An agent only sees averages

Fix agent i and let the others play π₋ᵢ. What matters for agent i at (h, s) is its
*marginal* reward and kernel (Eq. 35 in the paper):

```
r̄ᵢ(s, aᵢ) = Σ_{a₋ᵢ} π₋ᵢ(a₋ᵢ) · R[i, h, s, aᵢ, a₋ᵢ]
P̄(· | s, aᵢ) = Σ_{a₋ᵢ} π₋ᵢ(a₋ᵢ) · P[h, s, aᵢ, a₋ᵢ, ·]
```

Under the uniform policy, each sum is a plain average of M = 2ⁿ⁻¹ entries.

### 2. In F1 those entries are independent

Every `R[i, h, s, a]` and every `P[h, s, a, :]` is drawn independently for each
full joint action a. Nothing in the recipe depends on aᵢ alone: there is no own-action
term. Agent i's two actions therefore pick out two disjoint sets of M independent draws,
and r̄ᵢ(s, 0) and r̄ᵢ(s, 1) are averages of two independent samples of the same distribution.

### 3. Averages of M independent draws differ by ~ 1/√M

The formulas below are for setting A (β = 1, w = 0.5); setting B only changes the constants
(see Measured).

**Rewards.** The action-dependent part is (1 − w)·u with u ~ Unif[0, 1], whose variance is
(1 − w)²/12 = 1/48. The g(s) part is the same for both actions and cancels. So

```
Var[ r̄ᵢ(s,0) − r̄ᵢ(s,1) ] = 2 · 1/(48 M) = 1/(24 M)
E | r̄ᵢ(s,0) − r̄ᵢ(s,1) | = √(2/π) / √(24 · 2ⁿ⁻¹)
```

**Transitions.** Each component of a Dirichlet(1,1,1) row has variance 1/18. That gives

```
E TV( P̄(·|s,0), P̄(·|s,1) ) ≈ ½ · 3 · √(2/π) · √(1/(9M)) ≈ 0.40 / √(2ⁿ⁻¹)
```

Both shrink like **2^{-(n-1)/2}**, i.e. by a factor of √2 ≈ 1.41 per extra agent.

### 4. A small influence means a small gap

Under the uniform policy, agent i's best-response gain at one step is
`max(Q₀, Q₁) − (Q₀ + Q₁)/2 = |Q₀ − Q₁| / 2`. The difference |Q₀ − Q₁| is bounded by the
reward difference plus the kernel difference times the spread of next-state values. The
robust worst case shifts mass by σ toward the lowest-valued state, but it does that on
*both* averaged kernels, so it does not restore the difference. Summed over H steps, the
gap of the uniform policy is O(H · 2^{-(n-1)/2}) and inherits the √2-per-agent decay.

The reward weight w does not help either. g(s) is identical for both actions and
cancels, and its only effect is through P̄, which is averaged out as in step 3.

## Measured

10 instances per n, agent 0, others uniform, all (h, s), gap with σ = 0.3, produced by
[`experiments/f1_triviality.py`](../../experiments/f1_triviality.py) (reproduced on
2026-11-20; the numbers for setting A match Protocol.md, Section 1.9). The script reads β and w
from the config; `scaling_n.yaml` currently holds setting B:

```bash
uv run python experiments/f1_triviality.py experiments/configs/scaling_n.yaml --eps 0.1 0.05
```

**Setting A** (β = 1, w = 0.5; the formulas above):

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| mean \|r̄(s,0) − r̄(s,1)\| | 0.114 | 0.077 | 0.054 | 0.037 | 0.030 | 0.021 | 0.015 | 0.011 |
| predicted, √(2/π)/√(24·2ⁿ⁻¹) | 0.115 | 0.081 | 0.058 | 0.041 | 0.029 | 0.020 | 0.014 | 0.010 |
| mean TV(P̄(·\|s,0), P̄(·\|s,1)) | 0.298 | 0.202 | 0.137 | 0.093 | 0.069 | 0.052 | 0.037 | 0.024 |
| predicted, 0.40/√(2ⁿ⁻¹) | 0.28 | 0.20 | 0.14 | 0.10 | 0.071 | 0.050 | 0.035 | 0.025 |
| mean gap of the uniform policy | 0.269 | 0.207 | 0.163 | 0.110 | 0.081 | 0.060 | 0.042 | 0.033 |
| share with gap ≤ 0.1 | 0 | 0 | 0 | 0.3 | 0.9 | 1.0 | 1.0 | 1.0 |
| share with gap ≤ 0.05 | 0 | 0 | 0 | 0 | 0 | 0.1 | 0.9 | 1.0 |

**Setting B** (β = 0.1, w = 0). The predictions use the general variances: reward noise
(1 − w)²/12 = 1/12, and each Dirichlet(β·1_S) component has variance (S − 1)/(S²(Sβ + 1)) ≈ 0.17
for S = 3, β = 0.1 (versus 1/18 ≈ 0.056 for β = 1):

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| mean \|r̄(s,0) − r̄(s,1)\| | 0.223 | 0.162 | 0.117 | 0.078 | 0.054 | 0.038 | 0.030 | 0.019 |
| predicted | 0.230 | 0.163 | 0.115 | 0.081 | 0.058 | 0.041 | 0.029 | 0.020 |
| mean TV(P̄(·\|s,0), P̄(·\|s,1)) | 0.471 | 0.351 | 0.240 | 0.179 | 0.123 | 0.089 | 0.060 | 0.043 |
| predicted | 0.495 | 0.350 | 0.247 | 0.175 | 0.124 | 0.087 | 0.062 | 0.044 |
| mean gap of the uniform policy | 0.444 | 0.385 | 0.274 | 0.179 | 0.135 | 0.097 | 0.068 | 0.055 |
| share with gap ≤ 0.1 | 0 | 0 | 0 | 0 | 0 | 0.8 | 1.0 | 1.0 |
| share with gap ≤ 0.05 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1 | 0.4 |

In both settings the gap ratio from one n to the next is about 0.74, close to 1/√2 ≈ 0.71.
Setting B does not stop the decay; it only starts from a gap about 1.65× higher.

## Where the threshold lies

From the measured decay, gap_uniform(n) ≈ g₂ · 2^{-(n-2)/2} with g₂ the gap at n = 2
(0.27 in setting A, 0.44 in setting B). Setting this ≤ ε gives

```
n* ≈ 2 + 2 · log₂(g₂ / ε)
```

| Setting | ε = 0.1 | ε = 0.05 |
|---|---|---|
| A (g₂ = 0.27) | n* ≈ 4.9: trivial from **n = 5–6** (90 % at n = 6) | n* ≈ 6.9: trivial from **n = 7–8** |
| B (g₂ = 0.44) | n* ≈ 6.3: trivial from **n = 7** (80 % at n = 7) | n* ≈ 8.3: trivial from **n ≈ 9** (40 % at n = 9) |

Halving ε pushes the threshold out by only **two agents**; a larger single-agent influence
(setting B) pushes it out by about 1.4 more. Below the threshold (A: n = 2–4, B: n = 2–6 for
ε = 0.1), F1 is a meaningful test. Above it, N_all(ε) says nothing about learning.

## Consequences for the experiments

- **Plug-in:** it still pays H·S·2ⁿ samples (one per joint action) even when the game is
  trivial, so its N_all keeps doubling. That cost is real, but not because the problem got harder.
- **Robust-Q-FTRL:** it starts from the uniform policy, so above the threshold (A: n ≈ 5,
  B: n ≈ 7 at ε = 0.1) the smallest K on the grid already succeeds. Its advantage at large n on F1 is an artifact of the game.
- **Interpretation:** in F1, changing n also changes how much a single agent matters.
