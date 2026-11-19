# Uncertainty sets

What the adversary may change, where the uncertainty ball sits, and what σ does in practice.

## The set

Every agent i has a radius σᵢ ∈ [0, 1] (in the experiments the same σ = 0.3 for all). For
a nominal next-state distribution p, the uncertainty set is the total-variation ball

```
𝒰^σ(p) = { P ∈ Δ(𝒮) : TV(P, p) = ½ ‖P − p‖₁ ≤ σ }
```

TV is the probability mass that is moved away from p. So the adversary may move **at
most σ of the mass** to other next states. σ = 0 is the nominal game. σ = 1 allows any
distribution.

### Rectangularity

The ball is chosen **separately for every (i, h, s, aᵢ)**. The adversary picks a separate
worst case at each of them and does not have to be consistent across states, time steps or
agents. This is what lets the robust Bellman step be solved one entry at a time:

```
q_{i,h}(s, aᵢ) = r̄_{i,h}(s, aᵢ) + inf_{P ∈ 𝒰^{σᵢ}(P̄(· | s, aᵢ))} P · V_{i,h+1}
```

### Not a ball around the true model

The ball does **not** sit around the true joint kernel P_h(· | s, a). Each agent gets its
own ball around its own **averaged** kernel. It averages over what the other agents play
(Eq. 35, `mixed_kernel`):

```
P̄^{π₋ᵢ}_h(s' | s, aᵢ) = Σ_{a₋ᵢ} π₋ᵢ(a₋ᵢ | s) · P_h(s' | s, aᵢ, a₋ᵢ)
```

This is the paper's "fictitious RMG". It is the reason Robust-Q-FTRL only has to learn a
table of size S × Aᵢ per agent instead of S × ∏ⱼ Aⱼ.
