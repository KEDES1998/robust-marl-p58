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

10 instances per n, agent 0, others uniform, all (h, s), gap with σ = 0.3. These are the
current code's results (reproduced on 2026-11-19; the numbers match Protocol.md, Section 1.9),
produced by [`experiments/f1_triviality.py`](../../experiments/f1_triviality.py):

```bash
uv run python experiments/f1_triviality.py experiments/configs/scaling_n.yaml
```


| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| mean \|r̄(s,0) − r̄(s,1)\| | 0.114 | 0.077 | 0.054 | 0.037 | 0.030 | 0.021 | 0.015 |
| predicted, √(2/π)/√(24·2ⁿ⁻¹) | 0.115 | 0.081 | 0.058 | 0.041 | 0.029 | 0.020 | 0.014 |
| mean TV(P̄(·\|s,0), P̄(·\|s,1)) | 0.298 | 0.202 | 0.137 | 0.093 | 0.069 | 0.052 | 0.037 |
| predicted, 0.40/√(2ⁿ⁻¹) | 0.28 | 0.20 | 0.14 | 0.10 | 0.071 | 0.050 | 0.035 |
| mean gap of the uniform policy | 0.269 | 0.207 | 0.163 | 0.110 | 0.081 | 0.060 | 0.042 |
| share with gap ≤ 0.1 | 0 | 0 | 0 | 0.3 | 0.9 | 1.0 | 1.0 |
| share with gap ≤ 0.05 | 0 | 0 | 0 | 0 | 0 | 0.1 | 0.9 |

The gap ratio from one n to the next is about 0.73, close to 1/√2 ≈ 0.71.

## Where the threshold lies

From the measured decay, gap_uniform(n) ≈ 0.27 · 2^{-(n-2)/2}. Setting this ≤ ε gives

```
n* ≈ 2 + 2 · log₂(0.27 / ε)
```

- ε = 0.1  → n* ≈ 4.9: trivial from **n = 5–6** (90 % of instances at n = 6);
- ε = 0.05 → n* ≈ 6.9: trivial from **n = 7–8**.

Halving ε pushes the threshold out by only **two agents**. Below the threshold (n = 2–4 for
ε = 0.1), F1 is a meaningful test. Above it, N_all(ε) says nothing about learning.

## Consequences for the experiments

- **Plug-in:** it still pays H·S·2ⁿ samples (one per joint action) even when the game is
  trivial, so its N_all keeps doubling. That cost is real, but not because the problem got harder.
- **Robust-Q-FTRL:** it starts from the uniform policy, so from n ≈ 5 the smallest K on the
  grid already succeeds. Its advantage at large n on F1 is an artifact of the game.
- **Interpretation:** in F1, changing n also changes how much a single agent matters.
