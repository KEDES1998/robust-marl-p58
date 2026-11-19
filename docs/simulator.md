# Simulator and sample counting

Every method sees the game only through the generative model, and every answer
costs one sample. This counter is N_all, so all methods are measured with the same ruler.
Code: [`src/rmg/simulator.py`](../src/rmg/simulator.py).

## Access rules

| Who | May use |
|---|---|
| methods (`run_plugin`, `robust_q_ftrl.run`) | `sim.spec` (n, S, H, A) and `sim.sample` / `sim.sample_batch` |
| evaluation (`evaluate.py`) and `solve_exact` on the true model | `game.P`, `game.R` directly, and only after learning |

`GenerativeModel` keeps the game private (`sim._game`). Nothing stops a method from
reading it, so this is a convention, checked in code review.

## The query (Eq. 14)

```python
s_next, r = sim.sample(h, s, a)               # a = joint action, tuple of length n
s_next, r = sim.sample_batch(h, s, a)         # s: (m,), a: (m, n) -> s_next: (m,), r: (m, n)
```

- `s_next ~ P[h, s, a, :]` is drawn by inverse transform sampling (cumulative sums, one
  uniform number per query; the last cut is set to exactly 1 against rounding).
- `r = R[:, h, s, a]` gives the rewards of **all n agents**. Rewards are deterministic, so
  the reward part carries no noise.
- `sample` is a batch of size 1, so there is one code path and one random stream.
- Out-of-range indices raise a `ValueError` and are **not** counted. (numpy would
  silently read index −1 as the last state.)

## The counter

`sim.n_samples` goes up by 1 per answered query, i.e. by m for a batch of size m.
`reset_counter()` sets it to 0. `run_scaling.py` creates a fresh simulator per run and
writes the final count to the CSV column `n_samples`.

### What each method pays

| Method | Queries | Total |
|---|---|---|
| plug-in | m per (h, s, joint action) | H · S · ∏ᵢ Aᵢ · m |
| Robust-Q-FTRL | N per (h, iteration k, agent i, s, aᵢ) | H · K · N · S · Σᵢ Aᵢ |
| exact | none | 0 |

Checked by tests: `test_main_writes_rows_with_the_right_counters` (`test_run_scaling.py`)
checks H·S·2ⁿ·m for plug-in, and `test_run_sample_count_is_sum_not_product`
(`test_robust_q_ftrl.py`) uses A = (2, 3, 2), which gives Σ = 7, not ∏ = 12.

What one query is worth differs between the two methods:

- **Plug-in** asks about a specific joint action. The reward answer is exact, so the
  estimate R̂ = R and only P̂ carries error.
- **Robust-Q-FTRL** fixes only agent i's own action. The others draw their actions from their
  current policies, so one query is a sample of agent i's averaged kernel. The reward
  estimate averages over the others' sampled actions and is noisy. This is a small advantage
  for the baseline that the report should mention ([deviations.md](deviations.md)).
- Robust-Q-FTRL batches all agents' queries of one (h, k) into one `sample_batch` call.
  Each query is still counted once, so batching doesn't change the cost.

## Two seeds

| Seed | Controls | Where |
|---|---|---|
| instance seed | the game (P, R) | `random_game(..., seed=instance_seed)` |
| sampling seed | the next-state noise | `GenerativeModel(game, seed=sampling_seed)` |

Robust-Q-FTRL has a third random stream, the other agents' actions in Algorithm 1
(`run(..., seed=...)`). `run_scaling.py` passes the sampling seed for both, so a run is
reproducible from (instance seed, sampling seed) alone.
