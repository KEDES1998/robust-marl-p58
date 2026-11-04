# Protocol

Working protocol of the group: one section per member. Each section records what that
member's part does, why it matters, its current status and the open points. Section 1 also
gives an overview of the whole project, written so that anyone can follow it without
having read the paper in detail.

Last updated: 4 Nov 2026 (state of commit `3dc00ed`).

---

## 1. Kevin: environment, simulator, TV worst case (greedy), plug-in baseline

### 1.1 Current status (commit `3dc00ed`, 4 Nov 2026)

Kevin's parts are implemented and tested:

- `games.py`: `random_game` (game family F1)
- `simulator.py`: `GenerativeModel.sample` and `GenerativeModel.sample_batch`
- `robust.py`: `tv_worst_case` (greedy)
- `plugin.py`: `estimate_model`; `run_plugin` is written but calls `solve_exact`, so it only
  runs once Ali's exact solver exists

Implemented before (skeleton): `GameSpec`, `MarkovGame.validate`, `toy_public_goods_game` and
everything in `policies.py`.

Still stubs (`NotImplementedError`):

- Timon: `tv_worst_case_dual`, `robust_q_ftrl.run`
- Ali: `mixed_kernel`, `robust_values`, `gap_product`, `gap_out`, `gap_in`, `solve_exact`
- optional (anyone): `sample_aggregative_params`, `aggregative_game`
- `experiments/run_scaling.py`

`pytest`: 37 tests, 24 passed, 13 skipped, 0 failed. The skipped tests wait for Ali (11,
including `test_run_plugin_counter`), Timon's dual form (1) and F2 (1).

### 1.2 Log

| Date | Commit | What happened | `pytest` |
|---|---|---|---|
| 3 Nov | `3c5543f` to `5726f5d` | project structure, README with division of work, interface, experiment config draft, function stubs | – |
| 4 Nov | `ff8833f`, `b1a7635` | data formats (`games.py`, `policies.py`, `simulator.py` skeleton), 29 tests | 4 passed, 25 skipped |
| 4 Nov | `6c877bc` | removed empty `PLAN.md` and `CLAUDE.md` | 4 passed, 25 skipped |
| 4 Nov | `3dc00ed` | F1, simulator, TV worst case (greedy), plug-in estimation, 8 new tests | 24 passed, 13 skipped |

### 1.3 The idea of the project

**The problem (curse of multiagency).** With $n$ agents and $A$ actions each there are
$A^n$ joint actions. A method that queries every joint action in the simulator needs about

$$H \cdot S \cdot A^n \cdot m$$

samples, with $m$ samples per (time step, state, joint action). For $n = 8$ and $A = 2$ that
is already 256 joint actions per state, and the number doubles with every additional agent.

**The claim (Theorem 2, Eq. 30).** Robust-Q-FTRL finds an $\varepsilon$-robust coarse
correlated equilibrium (CCE) with

$$\tilde O\left(\frac{S H^6 \sum_{i} A_i}{\varepsilon^4}\,\min\left\lbrace H, \frac{1}{\min_i \sigma_i}\right\rbrace\right)$$

samples. The number of agents enters through the **sum** $\sum_i A_i$ instead of the product
$\prod_i A_i$. For $n = 8$ and $A = 2$: 16 instead of 256.

**Why this can work.** In Algorithm 1, agent $i$ fixes only its own action $a_i$; all other
agents draw their actions from their current policies. Agent $i$ therefore only learns the
average outcome of $a_i$ given how the others currently behave. Its table has $S \times A_i$
entries instead of $S \times \prod_j A_j$. The uncertainty set of the robust game is placed
around exactly this averaged transition (the "fictitious RMG" of the paper). This is why
`evaluate.py` contains `mixed_kernel` (Eq. 35): it computes this averaged transition from the
point of view of one agent.

**What "robust" means here.** Every agent evaluates its options as if an adversary could move
up to $\sigma$ of the probability mass of every transition towards the worst next states.
A robust equilibrium has to hold under this worst case, not only under the nominal model.

**What we can test.** We cannot verify the constants ($H^6$, $\varepsilon^4$). We can test
the shape in $n$: keep everything fixed, change only the number of agents $n$, and measure
$N_{\text{all}}(\varepsilon)$, the number of samples needed until the equilibrium gap is at
most $\varepsilon$. Sub-questions from the README: does $N = 1$ suffice in practice, and does
a larger $\sigma$ need fewer samples?

**How to read the plots.** The choice of axes follows directly from the two cost formulas:

- Plug-in: $N_{\text{all}} = HSm \cdot A^n$, so
  $\log N_{\text{all}} = \log(HSm) + n \log A$.
  This is a straight line in a **semi-log** plot (log $y$, linear $n$) with slope $\log A$.
- Robust-Q-FTRL with $A_i = A$: $N_{\text{all}} = HKNS \cdot nA$, so
  $\log N_{\text{all}} = \log(HKNSA) + \log n$.
  This is a straight line in a **log-log** plot with slope 1, provided the $K$ and $N$ needed
  for a given $\varepsilon$ do not themselves grow quickly with $n$. Whether they do is
  exactly what the experiment measures.

### 1.4 Walkthrough of the code

Data flow:

```
game (true P, R)
  -> simulator (counts every query)
  -> method: plug-in baseline | Robust-Q-FTRL
  -> mixture of K plans
  -> evaluation gap_out (the only place that reads the true P, R)
  -> plot N_all(eps) against n
```

#### `games.py`: the game (Kevin; F2 optional)

- `GameSpec` is the public description: $n$, $S$, $H$, $A$. It computes `n_joint`
  $= \prod_i A_i$ and `sum_A` $= \sum_i A_i$, the two numbers the claim is about.
- `MarkovGame` is the public description plus the true model $P$, $R$.
  `P.shape == (H, S, A_1, ..., A_n, S)`: at time $h$ in state $s$, when the agents jointly play
  $(a_1, \dots, a_n)$, the probability of each next state. The curse of multiagency is already
  visible in this shape: the array has $\prod_i A_i$ entries per $(h, s)$.
  `R.shape == (n, H, S, A_1, ..., A_n)`, values in $[0, 1]$.

Game family F1 (`random_game`) uses the same recipe for every $n$:

- every state gets a quality $g(s)$ spread evenly over $[0, 1]$ (for $S = 3$: $0, 0.5, 1$),
- $R_i(h, s, a) = 0.5\, g(s) + 0.5\, u$ with $u \sim \text{Unif}[0, 1]$,
- $P(\cdot \mid h, s, a) \sim \text{Dirichlet}(\beta \mathbf{1})$; small $\beta$ gives peaked
  distributions, large $\beta$ flat ones.

Why $g(s)$: without clearly good and clearly bad states, moving probability mass to the worst
state changes almost nothing, and the effect of robustness would be invisible in the experiment.

Game family F2 (optional): the other agents act only through the action histogram $\phi(a)$,
i.e. the share of agents playing each action. The same parameters then define a game for every
$n$, which gives a paired comparison across $n$ (TA feedback point 1).

`toy_public_goods_game`: a two-agent game whose gaps can be computed by hand (Section 1.6).
It is the known-answer test for the evaluation.

#### `simulator.py`: the generative model (Kevin)

- Algorithms see the world only through `sample(h, s, a)` and `sample_batch(h, s, a)` and the
  public `sim.spec`, never through the game itself.
- Every answer increments the counter `n_samples` by 1. This counter is $N_{\text{all}}$ in the
  plots, so all methods are measured with the same ruler.
- Two separate seeds: one generates the game, one drives the sampling noise. The same game can
  thus be repeated with different noise.

#### The two methods

**Plug-in baseline (`plugin.py`, Kevin).** Query every joint action $m$ times at every
$(h, s)$ and estimate

$$\hat P(s' \mid h, s, a) = \frac{\text{number of answers equal to } s'}{m}.$$

Then solve the robust game on $(\hat P, \hat R)$ with `solve_exact`, the same solver that
Robust-Q-FTRL uses in exact mode, without further samples. Cost: $H S \prod_i A_i \, m$.
Using the same solver is TA feedback point 3: the two methods then differ only in how they
collect data, which makes the comparison fair. Expected behaviour: exponential growth in $n$.

**Robust-Q-FTRL (`robust_q_ftrl.py`, Timon: `run`, Ali: `solve_exact`).** For every agent $i$
and every pair $(s, a_i)$: $N$ samples while the other agents play their current policies.
Cost: $H K N S \sum_i A_i$.

#### `policies.py`: what the methods return (shared)

- Product policy: every agent randomises independently, `pi[i][h, s, a_i]`
  $= \pi_i(a_i \mid h, s)$.
- Mixture: $K$ product policies ("plans") with weights $\alpha_k$. A shared random draw picks
  plan $k$, then every agent plays plan $k$. This shared draw is what makes the equilibrium
  *correlated*. Robust-Q-FTRL outputs such a mixture over its iterations (Eq. 24); the same
  weights apply at every time step, and a plan is drawn independently at each $h$.
- A single product policy is a mixture with $K = 1$.

#### `robust.py`: worst case over a total-variation ball (Kevin: greedy, Timon: dual form)

The quantity is

$$\inf \left\lbrace P \cdot V \;:\; \tfrac12 \Vert P - p \Vert_1 \le \sigma \right\rbrace .$$

The total-variation distance $\tfrac12 \Vert P - p \Vert_1$ equals the probability mass that is
moved away from $p$. So the worst case may move at most $\sigma$ of mass.

**Greedy solution.** To make the expectation as small as possible, take mass from the next
states with the highest $V$ and put it on the next state with the lowest $V$. Every unit of
moved mass then lowers the expectation by the largest possible amount.

Worked example (docstring and `test_worked_example`): $V = (0, 5, 10)$, $p = (0.2, 0.3, 0.5)$,
$\sigma = 0.3$.

- Nominal expectation: $0 \cdot 0.2 + 5 \cdot 0.3 + 10 \cdot 0.5 = 6.5$.
- Move $0.3$ from the state with $V = 10$ to the state with $V = 0$: $P = (0.5, 0.3, 0.2)$,
  expectation $0 + 1.5 + 2 = 3.5$.

If the best state holds less than $\sigma$ ("mass runs out", `test_mass_runs_out`), all of its
mass is moved and the remainder is taken from the second-best state, and so on.

Edge cases checked by the tests: $\sigma = 0$ gives the plain expectation $p \cdot V$;
$\sigma = 1$ gives $\min_s V(s)$.

**Dual form (Eq. 19).** Timon's version must give exactly the same number;
`test_greedy_equals_dual` compares both on 1000 random vectors. The two implementations
check each other.

#### `evaluate.py`: exact gap evaluation (Ali)

Robust backward induction from $h = H$ (with $V_H = 0$) down to $h = 0$, applying the TV worst
case at every step. For every agent $i$, time $h$ and state $s$ two values are computed:

- $V_{\text{follow}}$: agent $i$ follows the policy,
- $V_{\text{best}}$: agent $i$ picks its best action at every step (best response).

The gap is

$$\text{gap} = \max_{i, s} \big( V_{\text{best}}[i, 0, s] - V_{\text{follow}}[i, 0, s] \big),$$

i.e. how much the agent with the most to gain could win by deviating. A policy is an
$\varepsilon$-robust CCE if $\text{gap} \le \varepsilon$.

For a mixture there are two versions (TA feedback point 2):

- `gap_out`: the worst case sees the drawn plan $k$ (one uncertainty ball per plan). This is
  the recursion in the proof of Theorem 2 (Eqs. 87 to 88), i.e. the quantity the theorem
  controls, and the one we report.
- `gap_in`: the mixture is treated as one correlated policy with one ball around the mixture
  (Eq. 11).

For $K = 1$ all three (`gap_product`, `gap_out`, `gap_in`) coincide.
Only the evaluation may read the true $P$ and $R$, and only after learning.

#### `tests/`

`conftest.py` turns a `NotImplementedError` into "skipped" instead of "failed". `pytest`
therefore always shows what is done (passed), what is missing (skipped) and what is wrong
(failed). Whoever implements a function makes its tests pass.

#### `experiments/`

Draft config `scaling_n.yaml`: family F1, $n = 2, \dots, 8$, $A = 2$, $S = 3$, $H = 3$,
$\sigma = 0.3$, $\varepsilon = 0.1$, 10 game instances per $n$, 10 sampling seeds per instance
and budget. $N_{\text{all}}(\varepsilon)$ is the smallest budget at which at least 90 % of the
seeds reach $\text{gap} \le \varepsilon$. Budgets double: plug-in $m = 1, 2, 4, \dots, 128$;
Robust-Q-FTRL $N = 1, \dots, 32$ and $K = 64, \dots, 2048$. `run_scaling.py` is still a
placeholder; its docstring describes the protocol (per instance, then mean and 95 % confidence
interval per $n$ and method, raw data as CSV in `results/`).

### 1.5 Implementation details and checks (commit `3dc00ed`)

#### Game family F1: `random_game`

- `A` may be one number (same for every agent) or a tuple $(A_1, \dots, A_n)$.
- One random generator per `seed`: first one Dirichlet draw per $(h, s, \text{joint action})$
  for $P$, then the uniform noise for $R$. The quality is $g = (0, \tfrac{1}{S-1}, \dots, 1)$.
- `meta` stores all parameters (family, $n$, $S$, $H$, $A$, seed, $\beta$); the name reads
  e.g. `F1 n=3 seed=0`.

Checks:

- Expected mean reward per state: $\mathbb E[R \mid s] = 0.5\, g(s) + 0.25$, i.e.
  $0.25, 0.5, 0.75$ for $S = 3$. Measured ($n = 3$, $H = 2$, $A = (2, 3, 2)$, seed 0):
  $0.27, 0.55, 0.73$.
- Effect of $\beta$ ($S = 5$; a flat distribution would give $0.2$): the largest probability
  per row is on average $0.83$ for $\beta = 0.1$, $0.46$ for $\beta = 1$ and $0.27$ for
  $\beta = 10$.

#### Simulator: `sample` and `sample_batch`

**Drawing the next state (inverse transform sampling).** Cut the interval $[0, 1)$ into pieces
of length $P(s' \mid h, s, a)$; the cumulative sums mark the cuts. Draw
$u \sim \text{Unif}[0, 1)$ and return the state whose piece contains $u$. Since $u$ is uniform,
it lands in the piece of $s'$ with probability equal to the length of that piece, i.e. exactly
$P(s' \mid h, s, a)$. Example: $P = (0, 0.9, 0.1)$ gives the cuts $(0, 0.9, 1)$; $u = 0.3$
returns state 1, $u = 0.95$ returns state 2. A state with probability 0 has a piece of length
0 and is never returned. The last cut is set to exactly 1 to guard against rounding.

Design decisions:

- `sample` is a batch of size 1, so there is only one code path and both methods draw from the
  same random stream.
- Out-of-range indices raise a `ValueError` and leave the counter unchanged. Reason: numpy reads
  index $-1$ as "the last state", so a bug would otherwise silently query a wrong state.
- Rewards are returned as copies; a caller cannot change the game by accident.

Checks: 200,000 queries on a peaked kernel ($\beta = 0.1$, $S = 6$) reproduce $P$ to about
three decimals; a state with probability $6 \cdot 10^{-6}$ was drawn twice (expected 1.3).

#### TV worst case: `tv_worst_case` (greedy)

**Derivation.** Suppose the amount $\text{removed}(s)$ is taken from every state $s$ and
everything is put on the worst state ($V_{\min} = \min_s V(s)$, unit vector $e_{\min}$):

$$P = p - \text{removed} + \Big(\textstyle\sum_s \text{removed}(s)\Big)\, e_{\min} .$$

Multiplying by $V$:

$$P \cdot V = p \cdot V - \sum_s \text{removed}(s)\, V(s) + \sum_s \text{removed}(s)\, V_{\min} .$$

Combining the two sums:

$$P \cdot V = p \cdot V - \sum_s \text{removed}(s)\, \big(V(s) - V_{\min}\big) .$$

Every unit of mass moved away from $s$ lowers the expectation by $V(s) - V_{\min}$ (called
`gain` in the code). With a budget of $\sigma$ and at most $p(s)$ available from each state,
this is a fractional knapsack problem: taking mass from the states with the largest gain first
is optimal.

**Implementation.** Sort the states by $V$ in descending order. For each state, the mass of all
states before it is already used up ("mass before $s$", `mass_before` in the code), so

$$\text{removed}(s) = \min\Big\lbrace p(s),\; \max\big(\sigma - \text{mass before } s,\; 0\big)\Big\rbrace ,$$

and the result is $p \cdot V - \sum_s \text{removed}(s)\,(V(s) - V_{\min})$. If $\sigma$ exceeds
the mass on all better states, the rest is taken from a worst state, where the gain is 0; so
$\sigma = 1$ needs no special case.

The function is vectorised over leading axes: `p` may have shape `(..., S)` with `V` of shape
`(S,)`, and the result has shape `p.shape[:-1]`.

Check: compared with a general linear-programming solver (`scipy.optimize.linprog`) on 2000
random cases, including ties in $V$: largest difference $1.2 \cdot 10^{-7}$, the tolerance of
the solver. (The comparison script is not part of the repository.)

#### Plug-in estimation: `estimate_model` and `run_plugin`

**Estimator.** $\hat P(s' \mid h, s, a) = \frac{\text{number of answers equal to } s'}{m}$.
Each answer is a yes/no trial "is it $s'$?" with success probability $P(s')$, so

$$\mathbb E\big[\hat P\big] = P, \qquad \text{Var}\big(\hat P\big) = \frac{P(1 - P)}{m} \le \frac{1}{4m},$$

because $P(1 - P)$ is largest at $P = \tfrac12$. The standard error is at most
$\frac{1}{2\sqrt m}$, about $0.044$ for $m = 128$.

**Rewards.** Rewards are deterministic, so $\hat R = R$ exactly and the error of the plug-in
method comes only from $\hat P$. In Robust-Q-FTRL the reward estimate averages over the sampled
actions of the other agents and is therefore noisy. This is a small advantage for the baseline
that should be mentioned in the report.

**Ordering.** One batch per time step, ordered as (state, joint action, repetition). Joint
actions are enumerated with `np.indices` in numpy's C order (the last agent's action changes
fastest), which matches the action axes of `P` and `R`; a reshape puts every estimate back in
its place. Checked with $A = (2, 3, 2)$ and $m = 3000$: $\max |\hat P - P| = 0.025$.

**Cost.** Exactly $H \cdot S \cdot \prod_i A_i \cdot m$ samples. For the largest configuration
($n = 8$, $A = 2$, $S = 3$, $H = 3$, $m = 128$): 294,912 samples in about 0.1 s.

**`run_plugin`** is two lines: estimate the model, then call `solve_exact(model, sigma, K)`.
If the signature of `solve_exact` changes, only this call has to be adapted.

#### New tests (8)

- `tests/test_plugin.py`: sample counter (72 samples for the toy game with $m = 3$), valid game
  with exact rewards, absorbing states estimated exactly, convergence for $m = 4000$
  (all entries within $0.05$), error for $m = 0$, counter of `run_plugin` (skipped until
  `solve_exact` exists).
- `tests/test_simulator.py`: out-of-range queries are rejected and not counted.
- `tests/test_games.py`: F1 with different action counts $A = (2, 3, 2)$.

#### Notes for the others

- Ali: `tv_worst_case` accepts `p` of shape `(..., S)`. Passing the expected kernel of shape
  `(S, A_i, S)` together with `V_next` of shape `(S,)` returns all worst-case values of shape
  `(S, A_i)` in one call. `sigma` is one number per call; for agent-specific radii, call it once
  per agent.
- Timon: `test_greedy_equals_dual` runs as soon as `tv_worst_case_dual` exists. The greedy
  version is already checked against a linear-programming solver, so a mismatch points to the
  dual form.
- Everyone: `sample` returns `(int, array of shape (n,))`, `sample_batch` returns arrays of shape
  `(m,)` and `(m, n)`; invalid indices raise a `ValueError`.

### 1.6 Known-answer example worked by hand: `toy_public_goods_game`

Two agents, actions cooperate (C) or defect (D), one decision at $h = 0$. Defecting pays
$0.2$ immediately, cooperating pays $0$. Afterwards the game moves to the good state (value 1)
with probability $p_{\text{good}}$, otherwise to the bad state (value 0):

| $p_{\text{good}}$ | other plays C | other plays D |
|---|---|---|
| I play C | 0.9 | 0.5 |
| I play D | 0.5 | 0.1 |

Because the next-state values are $(1, 0)$, the value of an action is
immediate reward $+\, p_{\text{good}}$. With radius $\sigma$, the worst case moves $\sigma$
from the good to the bad state, which gives immediate reward
$+ \max(p_{\text{good}} - \sigma, 0)$.

| Case | Follow | Deviate | Gap |
|---|---|---|---|
| both defect, $\sigma = 0$ | $0.2 + 0.1 = 0.3$ | C: $0 + 0.5 = 0.5$ | $0.2$ |
| both defect, $\sigma = 0.2$ | $0.2 + \max(0.1 - 0.2, 0) = 0.2$ | C: $0 + 0.3 = 0.3$ | $0.1$ |
| both cooperate, $\sigma = 0$ | $0.9$ | D: $0.2 + 0.5 = 0.7$ | $0$ |
| both cooperate, $\sigma = 0.2$ | $0.7$ | D: $0.2 + 0.3 = 0.5$ | $0$ |

These are exactly the values in `tests/test_evaluate.py`.

Why this matters: under "both defect", the worst case costs the deviating agent the full
$0.2$, but the following agent only $0.1$, because there was only $0.1$ of mass on the good
state to take away. The gap therefore depends on $\sigma$. A method has to solve the robust
game itself; solving the nominal game and hoping it is also robust is not enough.

### 1.7 Thirty-second summary

The paper claims that a robust equilibrium of a multi-agent game can be found with a number of
samples that grows with the sum of the action-space sizes instead of their product, because
each agent only queries its own actions while the other agents sample along. We test this by
changing only the number of agents $n$ and measuring how many samples are needed until the
equilibrium gap drops below $\varepsilon$. As a baseline we use a plug-in method that queries
all joint actions but solves the estimated game with the same solver, so that only the data
collection differs. Exponential growth shows up as a straight line in a semi-log plot,
polynomial growth as a straight line in a log-log plot.

### 1.8 Open points (Kevin)

Done in `3dc00ed`: F1, simulator, TV worst case (greedy), `estimate_model`, code of
`run_plugin`.

- `run_plugin` waits for Ali's `solve_exact` and the gap evaluation (planned by Sat 7 Nov);
  then `test_run_plugin_counter` should pass.
- Plug-in part of `experiments/run_scaling.py` and the plug-in sweep ($n = 2, \dots, 8$,
  $m = 1, \dots, 128$) for the TA update (results Sun 8 Nov, update Mon 9 Nov). Agree on the
  CSV format with Timon (Robust-Q-FTRL runs) and Tiago (statistics and plots).
- Overleaf: environment and baseline (TA feedback points 1 and 3), including the remark that
  the plug-in rewards are exact (Section 1.5).
- Optional: game family F2, if time allows.

---

## 2. Tiago: statistics and plots

Responsibilities (from the README): $N_{\text{all}}(\varepsilon)$, confidence intervals,
log-log and semi-log plots; Overleaf: report setup, introduction, related work, definition of
gap_in and gap_out (together with Ali).

_To be filled in by Tiago._

---

## 3. Timon: Robust-Q-FTRL

Responsibilities (from the README): Robust-Q-FTRL (`run`), TV worst case in dual form (Eq. 19),
Robust-Q-FTRL runs; Overleaf: method section on the algorithm.

_To be filled in by Timon._

---

## 4. Ali: gap evaluation and exact solver

Responsibilities (from the README): gap evaluation (`gap_product`, `gap_out`, `gap_in`),
exact solver (`solve_exact`); Overleaf: definition of gap_in and gap_out (together with Tiago).

_To be filled in by Ali._
