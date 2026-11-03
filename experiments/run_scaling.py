"""Experiment C1: N_all(eps) versus the number of agents n.

Runs: Kevin (plug-in baseline), Timon (Robust-Q-FTRL). Statistics and plots: Tiago.

Usage (once implemented):

    python experiments/run_scaling.py experiments/configs/scaling_n.yaml

Protocol:

1. For every n and every instance (game seed), generate a game of the chosen family.
2. For every method and budget, repeat the run with n_sampling_seeds seeds and
   record the sample counter and gap_out on the true model.
3. Per instance: N_all(eps) = smallest budget at which at least success_rate of
   the seeds reach gap <= eps.
4. Across instances: mean and 95 % confidence interval per n and method.
5. Raw data as CSV in results/ (not committed); log-log and semi-log plots.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to the YAML config")
    parser.parse_args()
    raise NotImplementedError("Experiment script follows once games, evaluation and methods are in place.")


if __name__ == "__main__":
    main()
