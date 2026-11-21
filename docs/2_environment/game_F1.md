# Game environment

It is an unstructured random finite-horizon Markov game with n agents. Every joint
action has its own transition distribution and its own rewards, with no shared
structure across agents.

## Structure

| Symbol | Meaning | Index range (0-based) |
|---|---|---|
| n | number of agents | i = 0, …, n-1 |
| S | number of states | s = 0, …, S-1 |
| H | horizon | h = 0, …, H-1 (1, …, H in the paper) |
| A_i | actions of agent i | a_i = 0, …, A_i - 1 |

- **Joint action:** a = (a_1, …, a_n), which indexes the arrays directly. There are ∏ A_i joint actions.
- **Transitions:** `P[h, s, a_1, …, a_n, s']` has shape `(H, S, A_1, …, A_n, S)`, and each last axis sums to 1. The transitions depend on h (non-stationary).
- **Rewards:** `R[i, h, s, a_1, …, a_n]` has shape `(n, H, S, A_1, …, A_n)`. Rewards are deterministic and lie in [0, 1]. Each agent has its own reward, so the game is general-sum.
- **Termination:** V[H] = 0, with no terminal reward.

## Generation recipe

`random_game(n, S, H, A, seed, beta=1.0, w=0.5)` uses `rng = np.random.default_rng(seed)`:

1. **Transitions:** `P[h, s, a, :] ~ Dirichlet(beta · 1_S)`, drawn independently for every (h, s, a).
2. **State quality:** `g = linspace(0, 1, S)`. For S = 3 this gives g = (0, 0.5, 1).
3. **Rewards:** `R[i, h, s, a] = w · g(s) + (1 − w) · u`, where `u ~ Unif[0, 1]` independently for every (i, h, s, a).

The shared g(s) part makes some states clearly better than others for every agent. The noise part makes each agent's incentives different. Small β gives peaked (nearly deterministic) transitions, large β flat ones.

