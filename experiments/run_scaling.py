"""Experiment C1: N_all(eps) versus the number of agents n.

Runs: Kevin (plug-in baseline, solver floor), Timon (Robust-Q-FTRL). Statistics and plots: Tiago.

Usage:

    # everything in the config for the methods exact and plugin (default)
    python experiments/run_scaling.py experiments/configs/scaling_n.yaml --workers 4

    # quick test run: two values of n, 2 instances, 2 sampling seeds
    python experiments/run_scaling.py experiments/configs/scaling_n.yaml --n 2 3 --instances 2 --seeds 2

    # solver floor only (step before the sweep: is epsilon clearly above it?)
    python experiments/run_scaling.py experiments/configs/scaling_n.yaml --methods exact

Protocol:

1. For every n and every instance (game seed), generate a game of the chosen family.
2. For every method and budget, repeat the run with n_sampling_seeds seeds and
   record the sample counter and gap_out on the true model.
3. Per instance: N_all(eps) = smallest budget at which at least success_rate of
   the seeds reach gap <= eps.
4. Across instances: mean and 95 % confidence interval per n and method.
5. Raw data as CSV in results/ (not committed); log-log and semi-log plots.

This script does steps 1 and 2 and writes one CSV row per run (columns: ``FIELDS``).
Steps 3 to 5 work on that CSV (Tiago).

Methods (one value of ``method`` per row):

- ``exact``: ``solve_exact`` on the true model, no samples. The floor of the solver:
  neither method can get below it with the same K and c_alpha. One row per instance.
- ``plugin``: ``run_plugin`` with m samples per (h, s, joint action), solved with
  ``solve_exact`` (K_solver, c_alpha).
- ``robust_q_ftrl``: ``robust_q_ftrl.run`` with N samples per (s, a_i) and K iterations
  (Timon). Only starts once ``run`` is implemented.

Output: ``results/<name>/<timestamp>/`` with ``runs.csv`` (written row by row, so an
interrupted run keeps what it has), a copy of the config and ``run_info.yaml`` (command-line
overrides, versions, start and end time).
"""

from __future__ import annotations

import os

# One BLAS thread per process: the matrices are small, and with --workers the processes
# would otherwise compete for the same cores. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import multiprocessing as mp
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmg import robust_q_ftrl  # noqa: E402
from rmg.evaluate import gap_in, gap_out  # noqa: E402
from rmg.games import MarkovGame, random_game  # noqa: E402
from rmg.plugin import run_plugin  # noqa: E402
from rmg.robust_q_ftrl import solve_exact  # noqa: E402
from rmg.simulator import GenerativeModel  # noqa: E402

METHODS = ("exact", "plugin", "robust_q_ftrl")

FIELDS = [
    "family", "n", "S", "H", "A", "sigma",
    "instance_seed", "sampling_seed",
    "method", "m", "N", "K", "c_alpha",
    "n_samples", "gap_out", "gap_in", "runtime_s",
]
"""One row per run. Columns that do not apply to a method stay empty
(``m`` only for plugin, ``N`` only for robust_q_ftrl, ``sampling_seed`` not for exact)."""


# ---------------------------------------------------------------------------
# Games and jobs
# ---------------------------------------------------------------------------


def make_game(cfg: dict, n: int, instance_seed: int) -> MarkovGame:
    """The game of the configured family for n agents; ``instance_seed`` alone fixes the instance."""
    if cfg["family"] == "F1":
        return random_game(n=n, S=cfg["S"], H=cfg["H"], A=cfg["A"], seed=instance_seed)
    if cfg["family"] == "F2":
        raise NotImplementedError("game family F2 (aggregative_game) is not implemented yet")
    raise ValueError(f"unknown family {cfg['family']!r}, expected F1 or F2")


def build_jobs(cfg: dict, methods: list[str]) -> list[dict]:
    """All runs as small dicts, most expensive first (large n), for an even load across workers."""
    jobs = []
    base = {key: cfg[key] for key in ("family", "S", "H", "A", "sigma", "c_alpha")}
    for n in cfg["n"]:
        for inst in range(cfg["n_instances"]):
            common = dict(base, n=n, instance_seed=inst)
            if "exact" in methods:
                jobs.append(dict(common, method="exact", K=cfg["methods"]["exact"]["K"]))
            for seed in range(cfg["n_sampling_seeds"]):
                if "plugin" in methods:
                    plug = cfg["methods"]["plugin"]
                    for m in plug["m"]:
                        jobs.append(dict(common, method="plugin", sampling_seed=seed, m=m, K=plug["K_solver"]))
                if "robust_q_ftrl" in methods:
                    rq = cfg["methods"]["robust_q_ftrl"]
                    for N in rq["N"]:
                        for K in rq["K"]:
                            jobs.append(dict(common, method="robust_q_ftrl", sampling_seed=seed, N=N, K=K))
    jobs.sort(key=lambda job: -job["n"])
    return jobs


def run_job(job: dict) -> dict:
    """Runs one job and returns its CSV row. Only the evaluation reads the true model."""
    start = time.perf_counter()
    cfg = {"family": job["family"], "S": job["S"], "H": job["H"], "A": job["A"]}
    game = make_game(cfg, job["n"], job["instance_seed"])
    sigma, c_alpha = job["sigma"], job["c_alpha"]

    if job["method"] == "exact":
        n_samples = 0
        mixture = solve_exact(game, sigma, job["K"], c_alpha=c_alpha)
    elif job["method"] == "plugin":
        sim = GenerativeModel(game, seed=job["sampling_seed"])
        mixture = run_plugin(sim, job["m"], sigma, job["K"], c_alpha=c_alpha)
        n_samples = sim.n_samples
    elif job["method"] == "robust_q_ftrl":
        sim = GenerativeModel(game, seed=job["sampling_seed"])
        mixture = robust_q_ftrl.run(sim, sigma, job["K"], job["N"], c_alpha=c_alpha, seed=job["sampling_seed"])
        n_samples = sim.n_samples
    else:
        raise ValueError(f"unknown method {job['method']!r}")

    row = {key: job.get(key, "") for key in FIELDS}
    row["A"] = job["A"] if np.ndim(job["A"]) == 0 else "x".join(map(str, job["A"]))
    row.update(
        n_samples=n_samples,
        gap_out=gap_out(game, mixture, sigma),
        gap_in=gap_in(game, mixture, sigma),
        runtime_s=round(time.perf_counter() - start, 3),
    )
    return row


def check_robust_q_ftrl_ready() -> None:
    """Fails fast, before the sweep starts, if ``robust_q_ftrl.run`` is still a stub."""
    from rmg.games import toy_public_goods_game

    try:
        robust_q_ftrl.run(GenerativeModel(toy_public_goods_game(), seed=0), 0.2, K=1, N=1, seed=0)
    except NotImplementedError as exc:
        sys.exit(f"Method robust_q_ftrl cannot run yet: {exc}. Leave it out of --methods.")


# ---------------------------------------------------------------------------
# Running and summary
# ---------------------------------------------------------------------------


def run_all(jobs: list[dict], csv_path: Path, workers: int) -> list[dict]:
    """Runs the jobs (in parallel with workers > 1) and writes each row as soon as it is done."""
    rows = []
    total, start = len(jobs), time.perf_counter()
    last_report = start
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None
        try:
            results = pool.imap_unordered(run_job, jobs, chunksize=1) if pool else map(run_job, jobs)
            for done, row in enumerate(results, start=1):
                writer.writerow(row)
                f.flush()
                rows.append(row)
                now = time.perf_counter()
                if now - last_report >= 30 or done == total:          # progress at most every 30 s
                    last_report, elapsed = now, now - start
                    eta = elapsed / done * (total - done)
                    print(f"  {done:>6} / {total} runs  ({elapsed / 60:5.1f} min, about {eta / 60:5.1f} min left)", flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
    return rows


def print_summary(rows: list[dict], epsilon: float) -> None:
    """Quick look only (mean over instances and seeds); the real statistics are Tiago's part."""
    print(f"\nSummary: mean gap_out and share of runs with gap_out <= {epsilon}")
    for n in sorted({r["n"] for r in rows}):
        exact = [r["gap_out"] for r in rows if r["n"] == n and r["method"] == "exact"]
        if exact:
            print(f"n = {n}: solver floor (exact) mean {np.mean(exact):.4f}, max {np.max(exact):.4f}")
        for method, budget in (("plugin", ("m",)), ("robust_q_ftrl", ("N", "K"))):
            sub = [r for r in rows if r["n"] == n and r["method"] == method]
            for key in sorted({tuple(r[b] for b in budget) for r in sub}):
                gaps = np.array([r["gap_out"] for r in sub if tuple(r[b] for b in budget) == key])
                label = ", ".join(f"{b} = {v}" for b, v in zip(budget, key))
                samples = next(r["n_samples"] for r in sub if tuple(r[b] for b in budget) == key)
                print(f"  {method:<13} {label:<16} samples {samples:>9}   gap_out {gaps.mean():.4f}   "
                      f"<= eps in {np.mean(gaps <= epsilon):4.0%} of {len(gaps)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to the YAML config")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["exact", "plugin"],
                        help="methods to run (default: exact plugin)")
    parser.add_argument("--n", nargs="+", type=int, help="override the values of n")
    parser.add_argument("--instances", type=int, help="override n_instances")
    parser.add_argument("--seeds", type=int, help="override n_sampling_seeds")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    parser.add_argument("--out", help="output directory (default results/<name>/<timestamp>)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    if args.n:
        cfg["n"] = args.n
    if args.instances is not None:
        cfg["n_instances"] = args.instances
    if args.seeds is not None:
        cfg["n_sampling_seeds"] = args.seeds

    if "robust_q_ftrl" in args.methods:
        check_robust_q_ftrl_ready()
    make_game(cfg, cfg["n"][0], 0)  # fails here, not in a worker, if the family is not available

    started = datetime.now()
    out = Path(args.out) if args.out else ROOT / "results" / cfg["name"] / started.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out / config_path.name)

    jobs = build_jobs(cfg, args.methods)
    print(f"{len(jobs)} runs ({', '.join(args.methods)}), n = {cfg['n']}, {cfg['n_instances']} instances, "
          f"{cfg['n_sampling_seeds']} sampling seeds, {args.workers} worker(s) -> {out}")
    rows = run_all(jobs, out / "runs.csv", args.workers)

    info = {
        "config": str(config_path),
        "overrides": {"methods": args.methods, "n": args.n, "instances": args.instances,
                      "seeds": args.seeds, "workers": args.workers},
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "n_runs": len(rows),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out / "run_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    print_summary(rows, cfg["epsilon"])
    print(f"\nRaw data: {out / 'runs.csv'}")


if __name__ == "__main__":
    main()
