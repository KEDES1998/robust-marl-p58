# Simulator

Every method sees the game only through the generative model, and every answer
costs one sample. This counter is N_all, so all methods are measured with the same ruler.

## The query

```python
s_next, r = sim.sample(h, s, a)               # a = joint action, tuple of length n
s_next, r = sim.sample_batch(h, s, a)         # s: (m,), a: (m, n) -> s_next: (m,), r: (m, n)
```

- `s_next ~ P[h, s, a, :]` is drawn from the distribution.
- `r = R[:, h, s, a]` gives the rewards of all n agents. Rewards are deterministic, so the reward part carries no 
noise.

## The counter

`sim.n_samples` goes up by 1 per answered query, i.e. by m for a batch of size m.

## Two seeds

| Seed | Controls |
|---|---|---|
| instance seed | the game (P, R) |
| sampling seed | the next-state noise |
