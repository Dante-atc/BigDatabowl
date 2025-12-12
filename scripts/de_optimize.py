#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5 — Metric Calibration (Differential Evolution)
=====================================================

This module implements a Hybrid Differential Evolution algorithm to scientifically
calibrate the hyperparameters of the Defensive Coverage Index (DCI).

Optimization Strategy:
    Unlike traditional supervised learning, this optimizer maximizes the 
    NEGATIVE CORRELATION between our derived Defensive Score and the 
    offensive Expected Points Added (EPA).

    Hypothesis: High Defensive Score -> Low EPA.

Scope:
    - Filters strictly for Pass Plays (Dropbacks).
    - Tunes decay rates (alpha) and weights (beta, gamma).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

BASE_PROJECT_PATH = "/lustre/proyectos/p037"
DATA_ROOT = f"{BASE_PROJECT_PATH}/datasets/raw/114239_nfl_competition_files_published_analytics_final"
INPUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPPLEMENTARY_PATH = f"{DATA_ROOT}/supplementary_data.csv"
OUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_optimized.parquet"
PARAM_PATH = "/lustre/home/dante/compartido/metrics/de_best_params.json"

# -------------------------------------------------------
# DATA LOADING & FILTERING
# -------------------------------------------------------

print("[INFO] Loading baseline metrics...")
df = pd.read_parquet(INPUT_PATH)
df.rename(columns={"gameId": "game_id", "playId": "play_id"}, inplace=True)

print(f"[INFO] Loading ground truth labels from: {SUPPLEMENTARY_PATH}")
supp = pd.read_csv(SUPPLEMENTARY_PATH, low_memory=False)
supp.rename(columns={"gameId": "game_id", "playId": "play_id", 
                     "passResult": "pass_result", "expectedPointsAdded": "epa"}, inplace=True)

# --- CRITICAL: FILTER FOR PASS PLAYS ONLY ---
print(f"[INFO] Filtering for PASS PLAYS only...")
valid_results = ['C', 'I', 'S', 'IN']
supp = supp[supp['pass_result'].isin(valid_results)].copy()
supp['epa'] = supp['epa'].fillna(0.0)

# Merge
df = df.merge(supp[["game_id", "play_id", "epa", "pass_result"]], on=["game_id", "play_id"], how="inner")
print(f"   -> Optimization dataset ready. Size: {len(df)} plays.")

# Extract Arrays for Performance
raw_dist = df["distance_to_ideal"].values
spacing = df["spacing_proxy"].values
integrity = df["integrity_proxy"].values
epa = df["epa"].values

# -------------------------------------------------------
# OPTIMIZATION LOGIC
# -------------------------------------------------------

def objective(params):
    """Correlation Objective Function (Minimize this)."""
    alpha, beta, gamma = params
    
    # DCI Calculation
    dci_opt = np.exp(-alpha * raw_dist)

    # DIS Calculation
    dis_raw = (beta * spacing + gamma * integrity)
    dis_opt = dis_raw / (np.max(dis_raw) + 1e-9)

    # Global Score
    global_score = (dci_opt + dis_opt) / 2.0

    # Pearson Correlation (We want this to be negative)
    corr_epa, _ = pearsonr(global_score, epa)
    return corr_epa

# Differential Evolution Setup
bounds = [
    (0.1, 8.0),   # alpha (Decay)
    (0.1, 5.0),   # beta  (Spacing)
    (0.1, 5.0),   # gamma (Integrity)
]

def run_differential_evolution(pop_size=40, max_iter=100):
    dim = len(bounds)
    pop = np.random.rand(pop_size, dim)
    # Scale population to bounds
    for i in range(dim):
        pop[:, i] = pop[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    
    scores = np.array([objective(p) for p in pop])
    best_idx = np.argmin(scores)
    
    print(f"[DE] Initial Best Correlation: {scores[best_idx]:.4f}")

    for it in range(max_iter):
        for j in range(pop_size):
            # Mutation: Select 3 distinct random vectors
            idxs = [x for x in range(pop_size) if x != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            
            # Create mutant
            mutant = a + 0.8 * (b - c)
            # Clip to bounds
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Selection
            trial_score = objective(mutant)
            if trial_score < scores[j]:
                pop[j] = mutant
                scores[j] = trial_score
        
        if it % 20 == 0:
            best_curr = np.min(scores)
            print(f"[DE] Iter {it} — Best EPA Correlation: {best_curr:.6f}")
    
    best_idx = np.argmin(scores)
    return pop[best_idx], scores[best_idx]


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] Starting Differential Evolution Optimization...")
    best_params, best_score = run_differential_evolution()

    alpha, beta, gamma = best_params
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Optimal Parameters: Alpha={alpha:.4f}, Beta={beta:.4f}, Gamma={gamma:.4f}")
    print(f"Best Correlation with EPA: {best_score:.4f}")

    # Save Final Data
    df["dci_opt"] = np.exp(-alpha * raw_dist)
    dis_raw = (beta * spacing + gamma * integrity)
    df["dis_opt"] = dis_raw / (np.max(dis_raw) + 1e-9)

    df.to_parquet(OUT_PATH, index=False)
    
    with open(PARAM_PATH, "w") as f:
        json.dump({"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma)}, f, indent=4)

    print(f"[INFO] Optimized metrics saved to: {OUT_PATH}")