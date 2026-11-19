# Notation

## Game

| Symbol | Meaning | Code | Shape / type |
|---|---|---|---|
| n, S, H, Aᵢ | sizes | `GameSpec(n, S, H, A)` | `A` is a tuple of length n |
| ∏ᵢ Aᵢ | number of joint actions | `spec.n_joint` | int |
| Σᵢ Aᵢ | sum of action counts (Theorem 2) | `spec.sum_A` | int |
| P_h(s' \| s, a) | nominal transition | `game.P` | `(H, S, A_1, …, A_n, S)` |
| r_{i,h}(s, a) | reward of agent i, in [0, 1] | `game.R` | `(n, H, S, A_1, …, A_n)` |
| g(s) | state quality (F1 only) | `np.linspace(0, 1, S)` | `(S,)` |
| w | weight of g(s) in the reward (F1) | `random_game(..., w=0.5)` | float in [0, 1] |
| β (Dirichlet) | concentration of the F1 transitions | `random_game(..., beta=1.0)` | float > 0 |

## Robustness

| Symbol | Meaning | Code |
|---|---|---|
| σᵢ | TV radius of agent i | `sigma`: one number or a sequence of length n |
| TV(P, p) | total variation, ½‖P − p‖₁ | inside `tv_worst_case` |
| inf_{P ∈ 𝒰^σ(p)} P·V | worst-case expectation | `tv_worst_case(p, V, sigma)` (greedy), `tv_worst_case_dual` (Eq. 19) |
| P̄^{π₋ᵢ}_h(· \| s, aᵢ) | agent i's averaged kernel (Eq. 35) | `mixed_kernel(game, pi, i, h)[0]`, shape `(S, A_i, S)` |
| r̄^{π₋ᵢ}_{i,h}(s, aᵢ) | agent i's averaged reward | `mixed_kernel(game, pi, i, h)[1]`, shape `(S, A_i)` |

## Policies

| Symbol | Meaning | Code | Shape |
|---|---|---|---|
| π_{i,h}(aᵢ \| s) | product policy, agent i | `pi[i][h, s, a_i]` | list of n arrays `(H, S, A_i)` |
| π^k | plan k (iteration k of FTRL) | `mixture.plan(k)` | product policy |
| ξ̂ (paper's output distribution) | mixture of K plans | `PolicyMixture(pis, weights)` | `pis[i]`: `(K, H, S, A_i)`, `weights`: `(K,)` |
| α^K_k | weight of plan k (Eq. 24) | `mixture.weights[k-1]` | sums to 1 |


## Algorithm

| Symbol | Meaning | Code |
|---|---|---|
| K | FTRL iterations per time step | `K` |
| N | samples per (i, s, aᵢ) and iteration (Algorithm 1) | `N` |
| m | samples per (h, s, joint action) in the plug-in method | `m` |
| α_k | learning rate (Eq. 21) | `learning_rates(K, H, c_alpha)[0]` |
| η_{k+1} | softmax temperature (Eq. 22) | `learning_rates(...)[1]` |
| c_α | constant in α_k (paper: ≥ 24) | `c_alpha` |
| β_{i,h}(s) | bonus (Eq. 23) | computed in `_run` |
| c_b, δ | bonus constant and failure probability | `c_b`, `delta` |
| q^k_{i,h}(s, aᵢ) | one-step robust Q estimate (Eqs. 18/19) | `q` in `_run` |
| Q^k_{i,h} | FTRL accumulator (Algorithm 2, line 8) | `Q` in `_run` |
| V_{i,h}(s) | value estimate (Eq. 25) | `V`, shape `(n, H+1, S)` |

## Evaluation and experiment

| Symbol | Meaning | Code |
|---|---|---|
| V_follow, V_best | value of following, value of the best deviation | `robust_values`, shapes `(n, H+1, S)` |
| gap | max_{i,s}(V_best − V_follow) at h = 0 | `gap_product`, `gap_out`, `gap_in` |
| ε | target accuracy | `epsilon` in the config |
| N_all(ε) | samples needed to reach gap ≤ ε | `n_samples` column + `analyze_scaling.py` |
