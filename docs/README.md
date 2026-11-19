# Documentation

Reference for our reproduction of Shi et al. (2024), *Breaking the Curse of Multiagency in
Robust Multi-Agent RL*, claim C1: Robust-Q-FTRL needs a number of samples that grows with
Σᵢ Aᵢ instead of ∏ᵢ Aᵢ.

## 1. Introduction

- [Notation](notation.md): paper symbols, code names, array shapes, 0-based indexing
- [Metrics](metrics.md): robust CCE, `gap_out` / `gap_in`, solver floor, N_all(ε), cost formulas
- [Uncertainty sets](uncertainty_sets.md): the TV ball, where it sits, what σ does

## 2. Environment

- [Game family F1](game_F1.md): random game, generation recipe, sweep setting
- [Why F1 becomes trivial for large n](F1_triviality.md): mechanism, measurements, threshold
- [Simulator and sample counting](simulator.md): access rules, what one sample is, seeds

## 3. Methods

- [Plug-in baseline](plugin_baseline.md)
- [Exact solver](exact_solver.md)
- [Robust-Q-FTRL and the exact solver](methods.md)

## 4. Experiments

- [Running and reproducing the experiments](experiments.md): scripts, configs, output, commands

# 5. Deviations from the paper
- [Deviations from the paper](deviations.md): c_α, c_b, metric, game, baseline advantages