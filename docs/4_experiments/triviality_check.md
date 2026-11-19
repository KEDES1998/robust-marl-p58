# Triviality check: where does F1 need no learning?

## Question

Does a method's N_all(ε) actually reflect learning, or is the game already solved by a
policy that learned nothing?

## Why we ask

In the first plug-in sweep, the gap at n = 8 was **lower** than at n = 7 for every budget.
F1 draws every joint action independently, so one agent's influence averages out as n grows
([F1_triviality.md](../2_environment/F1_triviality.md)). If the **uniform policy** (every
agent plays each action with probability 1/2, zero samples) already has gap ≤ ε, every
method succeeds at its smallest budget and N_all says nothing about learning.

## Design

Two independent measurements on the same games as [the main experiment](scaling_in_n.md)
(F1, n = 2–8, 10 instances, σ = 0.3):

1. **Direct:** the gap of the uniform policy on the true model, plus agent 0's influence (the
   difference between its two actions in averaged reward and in averaged transition), with
   the other agents uniform. These are compared with the law-of-large-numbers prediction.
2. **Inside the sweep:** Robust-Q-FTRL with K = 1. With one iteration it outputs exactly the
   uniform policy, while the samples are still drawn and counted. An instance solved at
   K = 1 needs no learning.

## Results

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| gap of the uniform policy (mean) | 0.269 | 0.207 | 0.163 | 0.110 | 0.081 | 0.060 | 0.042 |
| solved at K = 1, ε = 0.1 | 0/10 | 0/10 | 0/10 | 3/10 | 9/10 | 10/10 | 10/10 |
| solved at K = 1, ε = 0.05 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 | 9/10 |

- The two measurements agree exactly: the K = 1 gaps equal the uniform-policy gaps.

## Consequence

| ε | Informative range | Trivial from |
|---|---|---|
| 0.1 | n = 2–4 | n ≈ 5–6 |
| 0.05 | n = 2–6 | n ≈ 7–8 |
