# Documentation

Reference for our reproduction of Shi et al. (2024), *Breaking the Curse of Multiagency in
Robust Multi-Agent RL*, claim C1: Robust-Q-FTRL needs a number of samples that grows with
Σᵢ Aᵢ instead of ∏ᵢ Aᵢ.

## 1. Introduction

- [Notation](1_introduction/notation.md): paper symbols, code names, array shapes, 0-based indexing
- [Metrics](1_introduction/metrics.md): robust CCE, the gap, N_all(ε), cost formulas
- [Uncertainty sets](1_introduction/uncertainty_sets.md): the TV ball, where it sits

## 2. Environment

- [Game family F1](2_environment/game_F1.md): random game, generation recipe, sweep setting
- [Why F1 becomes trivial for large n](2_environment/F1_triviality.md): mechanism, measurements, threshold
- [Simulator and sample counting](2_environment/simulator.md): access rules, what one sample is, seeds

## 3. Methods

- [Plug-in baseline](3_methods/plugin_baseline.md): estimate the full model, then solve it
- [Exact solver](3_methods/exact_solver.md): the shared FTRL loop, learning rates, solver floor
- [Robust-Q-FTRL](3_methods/robust_q_ftrl.md): N-sample estimation, bonus, results

## 4. Experiments

- [Running and reproducing the experiments](4_experiments/experiments.md): scripts, configs, output, commands

## 5. Deviations from the paper

- [Deviations from the paper](5_deviations/deviations.md): c_α, c_b, metric, game, baseline advantages
