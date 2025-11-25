#!/usr/bin/env python
# coding: utf-8

"""
Differential Evolution Optimization for DCI/DIS Scoring
Hybrid: best1bin + first-to-improvement
Output feeds back into Phase 4.2
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# -------------------------------------------------------
# Load play-level metrics
# -------------------------------------------------------

INPUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel.parquet"
OUTPUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_optimized.parquet"

print("[INFO] Loading metrics...")
df = pd.read_parquet(INPUT_PATH)

DCI = df["DCI"].values
DIS = df["DIS"].values

# -------------------------------------------------------
# Differential Evolution – Objective Function
# -------------------------------------------------------

def objective(params):
    """
    params = [w1, w2, w3, w4, alpha]
    w1: weight on mean(DCI)
    w2: weight on mean(DIS)
    w3: penalty on var(DCI)
    w4: penalty on var(DIS)
    alpha: shape factor (exponential smoothing)
    """

    w1, w2, w3, w4, alpha = params

    # Exponential smoothing
    DCI_adj = np.exp(-alpha * (1 - DCI))
    DIS_adj = np.exp(-alpha * (1 - DIS))

    # Global score
    score = (
        w1 * np.mean(DCI_adj)
        + w2 * np.mean(DIS_adj)
        - w3 * np.var(DCI_adj)
        - w4 * np.var(DIS_adj)
    )

    # Penalize instability
    penalty = 0.05 * (np.std(DCI_adj) + np.std(DIS_adj))

    return -(score - penalty)  # negative for minimization

# -------------------------------------------------------
# Hybrid Differential Evolution Settings
# -------------------------------------------------------

bounds = [
    (0.0, 2.0),  # w1
    (0.0, 2.0),  # w2
    (0.0, 1.0),  # w3
    (0.0, 1.0),  # w4
    (0.1, 3.0)   # alpha
]

# Custom strategy – “first-to-best” hybrid
def custom_strategy(pop, scores, mutation=0.8, recomb=0.7):
    """
    Implements first-to-improvement:
    Evaluate mutations until one improves; accept immediately.
    Replaces standard DE mutation.
    """
    new_pop = pop.copy()
    best_idx = np.argmin(scores)
    best = pop[best_idx]

    for i in range(len(pop)):
        for attempt in range(5):  # 5 attempts per candidate
            r1, r2 = np.random.choice(len(pop), 2, replace=False)
            mutant = pop[i] + mutation * (best - pop[r1] + pop[r2] - pop[i])

            # crossover
            cross_mask = np.random.rand(len(bounds)) < recomb
            trial = np.where(cross_mask, mutant, pop[i])

            # enforce bounds
            for d in range(len(bounds)):
                lo, hi = bounds[d]
                trial[d] = np.clip(trial[d], lo, hi)

            trial_score = objective(trial)

            if trial_score < scores[i]:  # improvement
                new_pop[i] = trial
                scores[i] = trial_score
                break

    return new_pop, scores

# -------------------------------------------------------
# Run DE (wrapper)
# -------------------------------------------------------

class HybridDE:
    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, max_iter=60, pop_size=16):
        dim = len(self.bounds)

        # Initialize
        pop = np.random.rand(pop_size, dim)
        for i in range(dim):
            lo, hi = bounds[i]
            pop[:, i] = lo + pop[:, i] * (hi - lo)

        scores = np.array([objective(ind) for ind in pop])

        for it in range(max_iter):
            pop, scores = custom_strategy(pop, scores)

            if it % 5 == 0:
                print(f"[DE] Iter {it} | best score: {scores.min():.6f}")

        best_idx = np.argmin(scores)
        return pop[best_idx], scores[best_idx]

# -------------------------------------------------------
# Run optimization
# -------------------------------------------------------

de = HybridDE(bounds)
best_params, best_score = de.run()

print("\n=== BEST PARAMETERS FOUND ===")
print(best_params)
print("Score:", best_score)

w1, w2, w3, w4, alpha = best_params

# -------------------------------------------------------
# Generate optimized metrics and save
# -------------------------------------------------------

DCI_adj = np.exp(-alpha * (1 - DCI))
DIS_adj = np.exp(-alpha * (1 - DIS))

df["DCI_optimized"] = DCI_adj
df["DIS_optimized"] = DIS_adj
df["DCI_DIS_final"] = (
    w1 * DCI_adj
    + w2 * DIS_adj
    - w3 * np.var(DCI_adj)
    - w4 * np.var(DIS_adj)
)

df.to_parquet(OUTPUT_PATH, index=False)

print("\nOptimized metrics saved to:")
print(OUTPUT_PATH)
