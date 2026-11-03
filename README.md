# robust-marl-p58

Reproducibility project for the Reinforcement Learning course (UZH, fall 2026) on
Shi et al., *Breaking the Curse of Multiagency in Robust Multi-Agent Reinforcement Learning*
([arXiv 2409.20067](https://arxiv.org/abs/2409.20067)).

**Claim under test (C1, Theorem 2):** Robust-Q-FTRL finds an ε-robust CCE with a number of
samples that grows with the sum of the action spaces Σ A_i rather than their product Π A_i.
What we can test is whether the number of iterations K and samples per iteration N needed
for a given ε grow exponentially with the number of agents n. Sub-questions: does N = 1
suffice in practice, and does a larger σ need fewer samples?

## Division of work

| Person | Code | Overleaf |
|---|---|---|
| Tiago | statistics and plots: N_all(ε), confidence intervals, log-log and semi-log plots | report setup, introduction, related work; definition of gap_in and gap_out (TA point 2), together with Ali |
| Kevin | environment: game family F1, simulator with sample counter, TV worst case (greedy); plug-in baseline and its runs; repo setup and interface | environment and baseline (TA points 1 and 3) |
| Timon | Robust-Q-FTRL (`run`), TV worst case (dual form, Eq. 19); Robust-Q-FTRL runs | method section on the algorithm |
| Ali | gap evaluation (`gap_product`, `gap_out`, `gap_in`), exact solver (`solve_exact`) | definition of gap_in and gap_out (TA point 2), together with Tiago |
| anyone | game family F2, only if time allows | report and poster are written jointly |

**Dependency:** the plug-in baseline needs the gap evaluation and the exact solver. Ali delivers
both by **Sat 7 Nov** so that baseline numbers are ready for the TA update.

**Dates:** Sun 8 Nov results and Overleaf text for the update · Mon 9 Nov update to the TA ·
Tue 10 Nov project session · 21 Nov main results for C1, report without abstract/results/conclusion ·
12 Dec reproducibility package, report and poster · 15 Dec poster session.

## Structure and owners

| File | Content | Owner |
|---|---|---|
| `src/rmg/games.py` | games, game families F1 and F2, a small toy game for tests | Kevin (F2: anyone) |
| `src/rmg/policies.py` | policy and mixture formats | shared |
| `src/rmg/simulator.py` | generative model with sample counter | Kevin |
| `src/rmg/robust.py` | TV worst case: greedy and dual form Eq. (19) | Kevin (greedy), Timon (dual) |
| `src/rmg/evaluate.py` | exact gap evaluation: gap_product, gap_out, gap_in | Ali |
| `src/rmg/plugin.py` | plug-in baseline | Kevin |
| `src/rmg/robust_q_ftrl.py` | Robust-Q-FTRL (Algorithm 2) and exact solver | Timon (`run`), Ali (`solve_exact`) |
| `experiments/` | configs and experiment scripts | runs: Kevin (baseline), Timon (Robust-Q-FTRL); statistics and plots: Tiago |
| `tests/` | tests, including known-answer tests for the gap evaluation | all |
| `results/` | raw data and plots, not in the repo | – |

## Interface (binding for everyone)

- **0-based indices:** time `h = 0, …, H-1` (1, …, H in the paper), states `s = 0, …, S-1`, actions `a_i = 0, …, A_i-1`.
- **Game** `MarkovGame`: `P.shape == (H, S, A_1, …, A_n, S)`, `R.shape == (n, H, S, A_1, …, A_n)`, rewards in [0, 1].
  A joint action is a tuple and indexes directly: `P[h, s, a_1, …, a_n]`.
- **Simulator** `GenerativeModel`: `sample(h, s, a)` and `sample_batch(h, s, a)` return the next state and the rewards;
  every answer counts towards `n_samples`. Algorithms only get the simulator and `sim.spec`, never the game itself.
- **Policy:** list of n arrays, `pi[i].shape == (H, S, A_i)`.
- **Mixture** `PolicyMixture`: `pis[i].shape == (K, H, S, A_i)`, `weights.shape == (K,)`, same weights at every h (Eq. 24).
- **Evaluation:** `gap_out(game, mixture, sigma)` is the quantity Theorem 2 controls. `sigma` is a number or a sequence of length n.

Please agree on changes to these formats in the group first.

## Tests

`pytest` shows what is done (passed), what is still missing (skipped: "not implemented yet")
and what is wrong (failed). Whoever implements a function makes its tests pass.

## Reproducibility

- Two separate seeds: one generates the game, one drives the sampling noise.
- Every experiment has a config in `experiments/configs/`; results are stored together with the config and seeds.
- Everything we try, including what does not work, also goes into the Overleaf.
