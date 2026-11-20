"""Experiment: effect of c_alpha on the solver floor of the exact solver.

The learning rate alpha_k = c_alpha log K / (k - 1 + c_alpha log K) (Eq. 21) sets how fast
the FTRL loop forgets and which of the K plans the output mixture keeps (weights alpha^K_k,
Eq. 24). A larger c_alpha puts the weight on fewer, later plans. The paper needs
c_alpha >= 24 for Theorem 2; the scaling sweeps use c_alpha = 1. This experiment measures
what that choice does to the solver floor: ``solve_exact`` on the true model (no samples),
evaluated with ``gap_out``.

Usage:

    python experiments/c_alpha_solver_floor.py experiments/configs/c_alpha_floor.yaml --workers 8

    # other n, c_alpha or K than in the config
    python experiments/c_alpha_solver_floor.py experiments/configs/c_alpha_floor.yaml --n 2 4 --c-alpha 1 24 --K 1000

Protocol:

1. For every n and instance (game seed), generate the game of the configured family
   (with ``beta`` and ``w`` for F1, as in ``run_scaling.py``).
2. For every c_alpha and K, run ``solve_exact`` and record gap_out and gap_in on the true model.
3. Summary per (n, c_alpha, K): mean and max over the instances and the share with
   gap <= epsilon; per (c_alpha, K) the weights alpha^K_k: effective number of plans
   (1 / sum of squared weights) and the weight on the first half of the iterations.

Output: ``results/<name>/<timestamp>/`` with ``runs.csv`` (one row per run, written row by
row), ``weights.csv``, a copy of the config, ``run_info.yaml`` and, with ``--plot``,
``floor.png`` (mean floor versus K per c_alpha, 95 % bootstrap CI over the instances).
"""

from __future__ import annotations

import os

# One BLAS thread per process (see run_scaling.py). Must be set before numpy is imported.
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
sys.path.insert(0, str(ROOT / "experiments"))

from rmg.evaluate import gap_in, gap_out  # noqa: E402
from rmg.robust_q_ftrl import learning_rates, solve_exact  # noqa: E402
from run_scaling import GAME_DEFAULTS, make_game  # noqa: E402

FIELDS = [
    "family", "n", "S", "H", "A", "beta", "w", "sigma",
    "instance_seed", "c_alpha", "K",
    "gap_out", "gap_in", "runtime_s",
]
"""One row per run (one game instance, one c_alpha, one K)."""

WEIGHT_FIELDS = ["c_alpha", "K", "effective_plans", "weight_first_half", "eta_last"]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def build_jobs(cfg: dict) -> list[dict]:
    """All runs as small dicts, most expensive first (large n, large K)."""
    base = {key: cfg[key] for key in ("family", "S", "H", "A", "sigma")}
    base.update({key: cfg.get(key, default) for key, default in GAME_DEFAULTS.items()})
    jobs = [
        dict(base, n=n, instance_seed=inst, c_alpha=float(c), K=int(K))
        for n in cfg["n"]
        for inst in range(cfg["n_instances"])
        for c in cfg["c_alpha"]
        for K in cfg["K"]
    ]
    jobs.sort(key=lambda job: (-job["n"], -job["K"]))
    return jobs


def run_job(job: dict) -> dict:
    """Solves one game exactly and returns its CSV row."""
    start = time.perf_counter()
    game = make_game(job, job["n"], job["instance_seed"])
    mixture = solve_exact(game, job["sigma"], job["K"], c_alpha=job["c_alpha"])
    row = {key: job.get(key, "") for key in FIELDS}
    row["A"] = job["A"] if np.ndim(job["A"]) == 0 else "x".join(map(str, job["A"]))
    row.update(
        gap_out=gap_out(game, mixture, job["sigma"]),
        gap_in=gap_in(game, mixture, job["sigma"]),
        runtime_s=round(time.perf_counter() - start, 3),
    )
    return row


def weight_table(c_alphas: list[float], Ks: list[int], H: int) -> list[dict]:
    """How many plans the output mixture keeps, per (c_alpha, K). Needs no game."""
    rows = []
    for c in c_alphas:
        for K in Ks:
            _, eta, w = learning_rates(int(K), H, float(c))
            rows.append({
                "c_alpha": float(c), "K": int(K),
                "effective_plans": float(1.0 / np.sum(w**2)),
                "weight_first_half": float(w[: int(K) // 2].sum()),
                "eta_last": float(eta[-1]),
            })
    return rows


# ---------------------------------------------------------------------------
# Running, summary and plot
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
                if now - last_report >= 30 or done == total:
                    last_report, elapsed = now, now - start
                    eta = elapsed / done * (total - done)
                    print(f"  {done:>6} / {total} runs  ({elapsed / 60:5.1f} min, about {eta / 60:5.1f} min left)", flush=True)
        finally:
            if pool:
                pool.close()
                pool.join()
    return rows


def print_summary(rows: list[dict], weights: list[dict], epsilon: float) -> None:
    print("\nWeights alpha^K_k of the output mixture:")
    print(f"  {'c_alpha':>7} {'K':>6} {'effective plans':>16} {'weight on k <= K/2':>19} {'eta_K':>7}")
    for w in weights:
        print(f"  {w['c_alpha']:>7g} {w['K']:>6} {w['effective_plans']:>16.0f} {w['weight_first_half']:>19.1e} {w['eta_last']:>7.2f}")

    print(f"\nSolver floor gap_out: mean (max) over instances, share <= {epsilon}")
    for n in sorted({r["n"] for r in rows}):
        sub = [r for r in rows if r["n"] == n]
        Ks = sorted({r["K"] for r in sub})
        print(f"n = {n}")
        print(f"  {'c_alpha':>7} " + "".join(f"{'K = ' + str(K):>22}" for K in Ks))
        for c in sorted({r["c_alpha"] for r in sub}):
            cells = []
            for K in Ks:
                g = np.array([r["gap_out"] for r in sub if r["c_alpha"] == c and r["K"] == K])
                cells.append(f"{g.mean():.3f} ({g.max():.3f}) {np.mean(g <= epsilon):4.0%}")
            print(f"  {c:>7g} " + "".join(f"{cell:>22}" for cell in cells))


def plot(rows: list[dict], epsilon: float, path: Path, n_boot: int = 2000, seed: int = 0) -> None:
    """Mean floor versus K (log-log), one line per c_alpha and n, with 95 % bootstrap CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analyze_scaling import BACKGROUND, GRID, INK, mean_ci

    rng = np.random.default_rng(seed)
    ns = sorted({r["n"] for r in rows})
    c_alphas = sorted({r["c_alpha"] for r in rows})
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(c_alphas)))
    markers = ["o", "s", "^", "D", "v"]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(7.75, 5.9))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    for c, color in zip(c_alphas, colors):
        for j, n in enumerate(ns):
            Ks = sorted({r["K"] for r in rows if r["c_alpha"] == c and r["n"] == n})
            stats = [mean_ci(np.array([r["gap_out"] for r in rows if r["c_alpha"] == c and r["n"] == n and r["K"] == K]),
                             n_boot, rng) for K in Ks]
            mean, lo, hi = map(np.array, zip(*stats))
            label = rf"$c_\alpha = {c:g}$" + (f", n = {n}" if len(ns) > 1 else "")
            ax.errorbar(Ks, mean, yerr=[mean - lo, hi - mean], color=color, marker=markers[j % len(markers)],
                        ms=9, lw=2.2, capsize=5, elinewidth=1.3, capthick=1.3, zorder=3, label=label)
    ax.axhline(epsilon, color="#b23a33", ls=(0, (5, 4)), lw=1.3, zorder=2)
    ax.text(0.99, epsilon, rf"$\varepsilon = {epsilon:g}$", color="#b23a33", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    Ks_all = sorted({r["K"] for r in rows})
    ax.set_xticks(Ks_all)
    ax.set_xticklabels([str(K) for K in Ks_all])
    ax.minorticks_off()
    ax.set_xlabel("FTRL iterations $K$", labelpad=10, fontsize=16)
    ax.set_ylabel("solver floor (gap on the true model)", labelpad=8, fontsize=16)
    ax.grid(True, which="major", color=GRID, lw=1.0, zorder=1)
    ax.tick_params(colors=INK, which="both", length=0, pad=8)
    for side, spine in ax.spines.items():
        spine.set_color(INK if side in ("left", "bottom") else GRID)
        spine.set_linewidth(1.2 if side in ("left", "bottom") else 0.8)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(INK)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK, handlelength=2.6, borderaxespad=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to the YAML config (e.g. experiments/configs/c_alpha_floor.yaml)")
    parser.add_argument("--n", nargs="+", type=int, help="override the values of n")
    parser.add_argument("--instances", type=int, help="override n_instances")
    parser.add_argument("--c-alpha", nargs="+", type=float, help="override the values of c_alpha")
    parser.add_argument("--K", nargs="+", type=int, help="override the values of K")
    parser.add_argument("--epsilon", type=float, help="override epsilon (summary and plot only)")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    parser.add_argument("--plot", action="store_true", help="also write floor.png to the output directory")
    parser.add_argument("--out", help="output directory (default results/<name>/<timestamp>)")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    if args.n:
        cfg["n"] = args.n
    if args.instances is not None:
        cfg["n_instances"] = args.instances
    if args.c_alpha:
        cfg["c_alpha"] = args.c_alpha
    if args.K:
        cfg["K"] = args.K
    if args.epsilon is not None:
        cfg["epsilon"] = args.epsilon
    for key, default in GAME_DEFAULTS.items():
        cfg.setdefault(key, default)

    make_game(cfg, cfg["n"][0], 0)  # fails here, not in a worker, if the family is not available

    started = datetime.now()
    out = Path(args.out) if args.out else ROOT / "results" / cfg["name"] / started.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out / config_path.name)

    weights = weight_table(cfg["c_alpha"], cfg["K"], cfg["H"])
    with open(out / "weights.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WEIGHT_FIELDS)
        writer.writeheader()
        writer.writerows(weights)

    jobs = build_jobs(cfg)
    print(f"{len(jobs)} runs, n = {cfg['n']}, c_alpha = {cfg['c_alpha']}, K = {cfg['K']}, "
          f"{cfg['n_instances']} instances, {args.workers} worker(s) -> {out}")
    rows = run_all(jobs, out / "runs.csv", args.workers)

    info = {
        "config": str(config_path),
        "overrides": {"n": args.n, "instances": args.instances, "c_alpha": args.c_alpha, "K": args.K,
                      "epsilon": args.epsilon, "workers": args.workers},
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "n_runs": len(rows),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out / "run_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    print_summary(rows, weights, cfg["epsilon"])
    if args.plot:
        plot(rows, cfg["epsilon"], out / "floor.png")
        print(f"\nplot: {out / 'floor.png'}")
    print(f"\nRaw data: {out / 'runs.csv'}")


if __name__ == "__main__":
    main()
