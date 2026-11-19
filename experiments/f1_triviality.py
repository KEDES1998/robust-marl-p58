"""Experiment: how fast does a single agent's influence vanish in F1 as n grows?

Background and derivation: docs/F1_triviality.md.

For every n and instance it measures, with all other agents playing uniformly:

- the reward difference |r̄_0(s, 0) - r̄_0(s, 1)| of agent 0 between its two actions,
- the TV distance between agent 0's marginal kernels P̄(· | s, 0) and P̄(· | s, 1),
- the gap of the uniform product policy (zero samples) on the true model,

and compares the first two with the law-of-large-numbers prediction (~ 1/sqrt(M), where
M = number of joint actions of the other agents). Averages are over all (h, s) and instances.

Usage:

    python experiments/f1_triviality.py experiments/configs/scaling_n.yaml
    python experiments/f1_triviality.py experiments/configs/scaling_n.yaml --n 2 3 4 --instances 5 --eps 0.1 0.05

Reads n, S, H, A, sigma, n_instances and epsilon from the config (beta and w if present,
otherwise the defaults 1.0 and 0.5). Prints a table; ``--out`` also writes it as CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from math import pi, prod, sqrt
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmg.evaluate import gap_product  # noqa: E402
from rmg.games import random_game  # noqa: E402
from rmg.policies import uniform_policy  # noqa: E402


def agent0_influence(game) -> tuple[float, float]:
    """Mean |r̄(s,0) - r̄(s,1)| and mean TV(P̄(·|s,0), P̄(·|s,1)) of agent 0, others uniform."""
    n = game.spec.n
    others = tuple(range(3, 3 + n - 1))           # axes a_1, ..., a_{n-1} after (h, s, a_0)
    P_bar = game.P.mean(axis=others)              # (H, S, A_0, S)
    r_bar = game.R[0].mean(axis=others)           # (H, S, A_0)
    dr = np.abs(r_bar[:, :, 0] - r_bar[:, :, 1]).mean()
    tv = 0.5 * np.abs(P_bar[:, :, 0] - P_bar[:, :, 1]).sum(axis=-1).mean()
    return float(dr), float(tv)


def predicted(A: tuple[int, ...], S: int, beta: float, w: float) -> tuple[float, float]:
    """LLN prediction for agent 0 under uniform others (independent entries per joint action).

    Reward: (1 - w) u with Var u = 1/12; difference of two means of M draws, E|.| = sqrt(2/pi) sd.
    TV: each Dirichlet(beta 1_S) component has variance (S - 1) / (S^2 (S beta + 1)).
    """
    M = prod(A[1:])
    dr = (1 - w) * sqrt(2 / pi) * sqrt(2 / (12 * M))
    var_p = (S - 1) / (S**2 * (S * beta + 1))
    tv = 0.5 * S * sqrt(2 / pi) * sqrt(2 * var_p / M)
    return dr, tv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to the YAML config (e.g. experiments/configs/scaling_n.yaml)")
    parser.add_argument("--n", nargs="+", type=int, help="override the values of n")
    parser.add_argument("--instances", type=int, help="override n_instances")
    parser.add_argument("--eps", nargs="+", type=float, help="thresholds for the share column (default: epsilon and epsilon/2)")
    parser.add_argument("--out", help="also write the table as CSV to this path")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    ns = args.n or cfg["n"]
    n_inst = args.instances if args.instances is not None else cfg["n_instances"]
    eps_list = args.eps or [cfg["epsilon"], cfg["epsilon"] / 2]
    S, H, sigma = cfg["S"], cfg["H"], cfg["sigma"]
    beta, w = cfg.get("beta", 1.0), cfg.get("w", 0.5)
    if not isinstance(cfg["A"], int) or cfg["A"] != 2:
        sys.exit("this experiment compares the two actions of agent 0 and needs A = 2")

    print(f"F1, S = {S}, H = {H}, A = 2, beta = {beta}, w = {w}, sigma = {sigma}, {n_inst} instances per n\n")
    eps_cols = [f"share gap<={e:g}" for e in eps_list]
    header = ["n", "|dr|", "|dr| pred", "TV", "TV pred", "gap uniform", *eps_cols]
    print("  ".join(f"{h:>12}" for h in header))

    rows = []
    for n in ns:
        drs, tvs, gaps = [], [], []
        for seed in range(n_inst):
            game = random_game(n=n, S=S, H=H, A=2, seed=seed, beta=beta, w=w)
            dr, tv = agent0_influence(game)
            drs.append(dr)
            tvs.append(tv)
            gaps.append(gap_product(game, uniform_policy(game.spec), sigma))
        gaps = np.array(gaps)
        dr_pred, tv_pred = predicted(game.spec.A, S, beta, w)
        row = {"n": n, "|dr|": np.mean(drs), "|dr| pred": dr_pred, "TV": np.mean(tvs), "TV pred": tv_pred,
               "gap uniform": gaps.mean(), **{c: np.mean(gaps <= e) for c, e in zip(eps_cols, eps_list)}}
        rows.append(row)
        print("  ".join(f"{row[h]:>12}" if h == "n" else f"{row[h]:>12.3f}" for h in header))

    gap_means = np.array([r["gap uniform"] for r in rows])
    if len(rows) > 1:
        ratio = np.exp(np.mean(np.diff(np.log(gap_means)) / np.diff(ns)))
        print(f"\nmean gap ratio per extra agent: {ratio:.2f}  (1/sqrt(2) = {1 / sqrt(2):.2f})")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"table written to {out}")


if __name__ == "__main__":
    main()
