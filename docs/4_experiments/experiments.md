# Running and reproducing the experiments

Scripts, configs, output layout, and the exact command for every result.
Use the project environment: `.venv/Scripts/python.exe` on Windows, or `python` inside
an activated venv. Python ≥ 3.10; dependencies are in `requirements.txt` (numpy, pandas,
matplotlib, pyyaml, pytest).

## Pipeline

```
config (YAML)
  -> run_scaling.py        one CSV row per (instance, method, budget, sampling seed)
  -> analyze_scaling.py    N_all(eps) per n and method, slope fits, plot
```

`f1_triviality.py` is separate: it measures how F1 behaves under the uniform policy.

## Scripts

### `experiments/run_scaling.py`: run the sweep

```bash
python experiments/run_scaling.py <config> [--methods exact plugin robust_q_ftrl] [--n 2 3 ...] [--instances I] [--seeds S] [--workers W] [--out DIR]
```

- `--methods`: default `exact plugin`. `robust_q_ftrl` fails fast if `run` isn't implemented.
- `--n`, `--instances`, `--seeds` override the config. `--workers` runs that many processes,
  each with one BLAS thread.
- Jobs run largest n first. Each job regenerates its game from (n, instance seed), so every
  method, seed and budget sees the same game.
- **Note:** Protocol §3.6 uses `--K 1` to run only the K = 1 grid point. The run of 19 Nov
  recorded that override, but the current `run_scaling.py` has no `--K` option, and the test
  `test_K1_returns_the_uniform_policy` named there isn't in `tests/` either. Both were probably
  never committed. Until they are, repeat that run with `K: [1]` in a copy of the config.

### `experiments/analyze_scaling.py`: N_all and plot

```bash
python experiments/analyze_scaling.py <runs.csv> [<runs.csv> ...] [--eps 0.1] [--success-rate 0.9] [--plot out.png]
```

The CSVs are concatenated, so a plug-in sweep and a Robust-Q-FTRL sweep can be analysed
together. It prints per (method, n): solved/total, geometric mean, median, min, max, the
typical budget, and `solved_by_K1`. Then the semi-log and log-log slopes and the solver
floor if `exact` rows are present. `--plot` writes a semi-log and a log-log panel. The
definitions are in [metrics.md](../1_introduction/metrics.md).

### `experiments/f1_triviality.py`: influence of one agent in F1

```bash
python experiments/f1_triviality.py <config> [--n ...] [--instances I] [--eps 0.1 0.05] [--out table.csv]
```

Measured and predicted reward and transition differences for agent 0, the uniform-policy
gap, and the share of instances under each ε. Needs A = 2. Background:
[F1_triviality.md](../2_environment/F1_triviality.md).

## Configs (`experiments/configs/`)

| Key | Meaning | `scaling_n.yaml` | `scaling_n_f1_rq.yaml` |
|---|---|---|---|
| `name` | output folder name | `scaling_n` | `scaling_n_f1_rq` |
| `family` | `F1` (`F2` not implemented) | F1 | F1 |
| `n` | numbers of agents | 2–8 | 2–8 |
| `A`, `S`, `H` | actions per agent, states, horizon | 2, 3, 3 | 2, 3, 3 |
| `sigma` | TV radius | 0.3 | 0.3 |
| `epsilon` | target gap (summary print only) | 0.1 | 0.1 |
| `n_instances` | games per n (seeds 0…I−1) | 10 | 10 |
| `n_sampling_seeds` | repetitions per game and budget | 10 | 10 |
| `success_rate` | share of seeds that must succeed | 0.9 | 0.9 |
| `c_alpha` | learning-rate constant, all methods | 1.0 | 1.0 |
| `methods.exact.K` | solver iterations, floor | 2000 | 2000 |
| `methods.plugin.m` | samples per joint action | 1, 2, …, 128 | 1, 2, …, 128 |
| `methods.plugin.K_solver` | solver iterations | 2000 | 2000 |
| `methods.robust_q_ftrl.N` | samples per (s, aᵢ) | 1, 2, …, 32 | 1, 4 |
| `methods.robust_q_ftrl.K` | FTRL iterations | 64, …, 2048 | 1, 32, …, 2048 |
| `methods.robust_q_ftrl.c_b` | bonus constant (default 1.0 if missing) | – | 0.0 |

`run_scaling.py` doesn't read `beta` or `w`: F1 games use the defaults β = 1 and w = 0.5.
`f1_triviality.py` reads them if present.

## Output

```
results/<name>/<YYYYMMDD-HHMMSS>/
  runs.csv          one row per run, written and flushed row by row (an interrupted run keeps its rows)
  <config>.yaml     copy of the config
  run_info.yaml     command-line overrides, start/end time, number of runs, Python and numpy versions
  *.png             plots, if analyze_scaling.py was pointed there
```

CSV columns:

| Column | Meaning |
|---|---|
| `family, n, S, H, A, sigma` | game and robustness setting |
| `instance_seed` | game seed |
| `sampling_seed` | simulator seed (empty for `exact`) |
| `method` | `exact`, `plugin` or `robust_q_ftrl` |
| `m` | plug-in only |
| `N` | Robust-Q-FTRL only |
| `K` | solver or FTRL iterations |
| `c_alpha` | learning-rate constant |
| `n_samples` | simulator queries (0 for `exact`) |
| `gap_out`, `gap_in` | gaps on the true model ([metrics.md](../1_introduction/metrics.md)) |
| `runtime_s` | wall time of the run including evaluation |

`c_b` is passed to the solver but **not** written to the CSV. It is only recorded in the
config copy.

## Reproducing the results

| Result | Command | Output | Runtime |
|---|---|---|---|
| Solver floor | `python experiments/run_scaling.py experiments/configs/scaling_n.yaml --methods exact` | `results/scaling_n/20261108-132828` | ~4 min |
| Plug-in sweep (Protocol §1.9) | `python experiments/run_scaling.py experiments/configs/scaling_n.yaml --workers 8` | `results/scaling_n/20261108-133521` | 1 h 27 min, 8 processes |
| Robust-Q-FTRL sweep (§3.5) | `python experiments/run_scaling.py experiments/configs/scaling_n_f1_rq.yaml --methods robust_q_ftrl --workers 8` | `results/scaling_n_f1_rq/20261114-114125` | 1 h 32 min, 8 processes |
| K = 1 sweep (§3.6) | the same with only `K: [1]` (see the note on `--K` above) | `results/scaling_n_f1_rq/20261119-101535` | 6 s |
| N_all tables and plot | `python experiments/analyze_scaling.py results/scaling_n/20261108-133521/runs.csv results/scaling_n_f1_rq/20261114-114125/runs.csv results/scaling_n_f1_rq/20261119-101535/runs.csv --eps 0.1 --plot n_all.png` | printed, plus the plot | seconds |
| F1 triviality table | `python experiments/f1_triviality.py experiments/configs/scaling_n.yaml` | printed | seconds |

The 14 Nov sweep ran with K starting at 32. The config now also contains K = 1, so a fresh full
run of `scaling_n_f1_rq.yaml` produces both sweeps in one folder.

Quick check that everything works (about a minute):

```bash
python experiments/run_scaling.py experiments/configs/scaling_n.yaml --n 2 3 --instances 2 --seeds 2
```

## Tests

```bash
python -m pytest -q
```

`conftest.py` turns `NotImplementedError` into "skipped", so the summary shows what's done,
missing and broken. Currently one test is skipped: F2.

## Which results are in git

`results/scaling_n/…` and `results/scaling_n_f1_rq/20261114-114125` are committed.
`results/scaling_n_f1_rq/20261119-*` are not yet. The `run_scaling.py` docstring says raw
data is "not committed". Decide on one rule before the reproducibility package (12 Dec).
