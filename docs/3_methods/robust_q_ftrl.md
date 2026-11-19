# Robust-Q-FTRL

The paper's method: the FTRL loop of the [exact solver](exact_solver.md), with the exact mixed
kernel replaced by estimates from the simulator (Algorithm 1) and a bonus added to V.

## Algorithm 1: N-sample estimation

At every (h, k), for every agent i and every (s, aᵢ):

- N queries in which agent i plays aᵢ and every other agent j draws aⱼ ~ π^k_{j,h}(· | s) (Eq. 16);
- r̂ᵢ(s, aᵢ) = the mean reward, P̂ᵢ(· | s, aᵢ) = the empirical next-state frequencies.

Their expectation is exactly the mixed kernel of the exact solver. Agent i never fixes a joint
action, so its table has S × Aᵢ entries instead of S × ∏ⱼ Aⱼ. This is where the sum in
Theorem 2 comes from.

Unlike the plug-in baseline, the reward estimate is noisy: it averages over the others'
sampled actions.

## Loop

```
for h = H-1, ..., 0:
    pi^1 = uniform, Q^0 = 0
    for k = 1, ..., K:
        r̂_i, P̂_i = Algorithm 1 under pi^k
        q_i = r̂_i + tv_worst_case(P̂_i, V_i[h+1], sigma_i)          Eqs. (18)/(19)
        Q_i, pi_i^{k+1}: as in the exact solver
    V_i[h] = min( Σ_k alpha^K_k <pi_i^k, q_i^k> + bonus_i, H - h )   Eq. (25)
```

**Bonus** (Eq. 23):

```
beta_{i,h}(s) = c_b · sqrt( log³(K S Σᵢ Aᵢ / δ) / (K H) · Σ_k alpha^K_k ( Var_{π^k}(q^k(s, ·)) + H ) )
```

It is switched off in the experiments (c_b = 0), because with c_b = 1 it exceeds the value
range ([deviations.md](../5_deviations/deviations.md)).

## Cost

```
n_samples = H · K · N · S · Σᵢ Aᵢ        (sweep setting: 18 · K · N · n)
```

Two random streams: the other agents' actions use the `seed` argument, and the next states
come from the simulator. `run_scaling.py` passes the sampling seed for both, so a run is
reproducible.
