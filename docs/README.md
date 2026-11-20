# Documentation

Reference for our reproduction of Shi et al. (2024), *Breaking the Curse of Multiagency in
Robust Multi-Agent RL*, claim C1: Robust-Q-FTRL needs a number of samples that grows with
Σᵢ Aᵢ instead of ∏ᵢ Aᵢ.

## 1. Introduction

- [Notation](1_introduction/notation.md)
- [Metrics](1_introduction/metrics.md)
- [Uncertainty sets](1_introduction/uncertainty_sets.md)

## 2. Environment

- [Game family F1](2_environment/game_F1.md)
- [Why F1 becomes trivial for large n](2_environment/F1_triviality.md)
- [Simulator](2_environment/simulator.md)

## 3. Methods

- [Exact solver](3_methods/exact_solver.md)
- [Plug-in baseline](3_methods/plugin_baseline.md)
- [Robust-Q-FTRL](3_methods/robust_q_ftrl.md)

## 4. Experiments

- [C1: sample complexity versus the number of agents](4_experiments/scaling_in_n.md)
- [Triviality check: where F1 needs no learning](4_experiments/triviality_check.md)
- [Solver floor versus c_α and K](4_experiments/c_alpha_floor.md)

## 5. Deviations from the paper

- [Deviations from the paper](5_deviations/deviations.md)
