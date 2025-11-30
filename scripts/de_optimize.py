#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DE Optimization for Defensive Coverage Index (DCI) & Defensive Integrity Score (DIS)
===================================================================================

This module implements a Hybrid Differential Evolution (DE) algorithm to calibrate
the hyperparameters of the proposed defensive metrics.

Objective:
    To find the optimal weights (alpha, beta, gamma) that maximize the predictive
    power of the metrics regarding defensive success.

Optimization Strategy:
    Instead of maximizing the raw score magnitude (which leads to trivial solutions),
    we maximize the NEGATIVE CORRELATION between our Global Defensive Score and
    the offensive Expected Points Added (EPA).
    
    Hypothesis: High Defensive Score -> Low EPA (Strong Negative Correlation).

CRITICAL UPDATE:
    - Filters ground truth data to include ONLY Pass Plays (Dropbacks).
    - Removes Run plays, Kneels, and Spikes to prevent noise in the optimization.

Inputs:
    - metrics_playlevel_baseline.parquet : Baseline geometric proxies.
    - supplementary_data.csv             : Official BDB metadata (Ground Truth labels).

Outputs:
    - metrics_playlevel_optimized.parquet : Final dataframe with calibrated DCI/DIS.
    - de_best_params.json                 : Optimal hyperparameters found.

"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

# Define project root paths
BASE_PROJECT_PATH = "/lustre/proyectos/p037"
DATA_ROOT = f"{BASE_PROJECT_PATH}/datasets/raw/114239_nfl_competition_files_published_analytics_final"

# Input paths
INPUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPPLEMENTARY_PATH = f"{DATA_ROOT}/supplementary_data.csv"

# Output paths
OUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_optimized.parquet"
PARAM_PATH = "/lustre/home/dante/compartido/metrics/de_best_params.json"

# -------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -------------------------------------------------------

print("[INFO] Loading baseline metrics...")
df = pd.read_parquet(INPUT_PATH)

print(f"[DEBUG] Columns in baseline metrics: {df.columns.tolist()}")

# Standardize column names (Handle inconsistencies between steps)
rename_map = {
    "gameId": "game_id",
    "playId": "play_id",
    "frameId": "frame_id",
    "nflId": "nfl_id"
}
df.rename(columns=rename_map, inplace=True)

# Validation
if "game_id" not in df.columns:
    raise KeyError(f"CRITICAL: 'game_id' not found in dataframe. Columns are: {df.columns.tolist()}")

print(f"   -> Loaded {len(df)} plays from baseline metrics.")

print(f"[INFO] Loading ground truth labels from: {SUPPLEMENTARY_PATH}")
supp = pd.read_csv(SUPPLEMENTARY_PATH, low_memory=False)

# Standardize supplementary column names
cols_map = {
    "gameId": "game_id", 
    "playId": "play_id", 
    "passResult": "pass_result", 
    "expectedPointsAdded": "expected_points_added"
}
supp.rename(columns=cols_map, inplace=True)

# Ensure snake_case keys exist
if "expectedPointsAdded" in supp.columns and "expected_points_added" not in supp.columns:
     supp.rename(columns={"expectedPointsAdded": "expected_points_added"}, inplace=True)
if "passResult" in supp.columns and "pass_result" not in supp.columns:
     supp.rename(columns={"passResult": "pass_result"}, inplace=True)


# --- CRITICAL FIX: FILTER FOR PASS PLAYS ONLY ---
# Run plays do not have a "coverage" structure in the same way. Including them introduces
# massive noise because the metric will penalize defenders for breaking formation to tackle.
# We filter for valid pass results: Complete (C), Incomplete (I), Sack (S), Interception (IN).

print(f"[INFO] Filtering for PASS PLAYS only...")
initial_len = len(supp)
supp = supp.dropna(subset=['pass_result']) 

valid_results = ['C', 'I', 'S', 'IN']
supp = supp[supp['pass_result'].isin(valid_results)]

print(f"   -> Retained {len(supp)} pass plays (Dropped {initial_len - len(supp)} non-pass plays).")
# ------------------------------------------------


# Filter required columns for optimization
required_cols = ["game_id", "play_id", "expected_points_added", "pass_result"]
try:
    supp = supp[required_cols].copy()
except KeyError as e:
    print(f"[ERROR] Missing columns in supplementary data. Available: {supp.columns}")
    raise e

# Merge: Combine calculated proxies with Ground Truth labels
# Use inner join to ensure we only optimize on plays where we have valid labels AND are pass plays
print("[INFO] Merging metrics with ground truth labels...")
df = df.merge(supp, on=["game_id", "play_id"], how="inner")

# Convert pass_result to numeric binary for analysis
# 1 = Complete (Bad for defense), 0 = Incomplete/Int (Good for defense)
df["is_complete"] = df["pass_result"].apply(lambda x: 1 if x == 'C' else 0)

# Extract numpy arrays for high-performance optimization
raw_dist = df["distance_to_ideal"].values
spacing = df["spacing_proxy"].values
integrity = df["integrity_proxy"].values
epa = df["expected_points_added"].values

# Handle NaNs in EPA (rare edge cases in BDB data)
epa = np.nan_to_num(epa, nan=0.0)

print(f"[INFO] Optimization dataset ready. Size: {len(df)} plays.")

# -------------------------------------------------------
# DE OBJECTIVE FUNCTION
# -------------------------------------------------------

def objective(params):
    """
    Evaluates the quality of the hyperparameters based on EPA correlation.

    Args:
        params (list): [alpha, beta, gamma]
        
        alpha (float): Decay rate for Geometric Coverage (DCI).
        beta (float):  Weight for Spacing Cohesion in DIS.
        gamma (float): Weight for Tactic Integrity/Clarity in DIS.

    Returns:
        float: Pearson correlation between Global Score and EPA.
               DE minimizes this value. Ideally, we want negative correlation.
    """
    alpha, beta, gamma = params

    # 1. Recalculate DCI (Geometric Coverage Index)
    # Formula: DCI = exp(-alpha * distance_to_ideal)
    dci_opt = np.exp(-alpha * raw_dist)

    # 2. Recalculate DIS (Defensive Integrity Score)
    # Formula: Weighted average of spacing and tactic integrity
    dis_raw = (beta * spacing + gamma * integrity)
    
    # Min-Max normalization to keep DIS within [0, 1] range relative to the batch
    dis_max = np.max(dis_raw) + 1e-9
    dis_opt = dis_raw / dis_max

    # 3. Compute Global Defensive Score
    global_score = (dci_opt + dis_opt) / 2.0

    # 4. Fitness Calculation: Pearson Correlation with EPA
    # Goal: High Defensive Score should correlate with Low (Negative) EPA.
    corr_epa, _ = pearsonr(global_score, epa)
    
    return corr_epa

# -------------------------------------------------------
# HYBRID DIFFERENTIAL EVOLUTION (DE) IMPLEMENTATION
# -------------------------------------------------------

# Bounds for hyperparameters:
bounds = [
    (0.1, 8.0),   # alpha
    (0.1, 5.0),   # beta
    (0.1, 5.0),   # gamma
]

def first_to_best_mutation(pop, scores, mutation=0.8, recomb=0.7):
    """
    Applies 'current-to-best/1' mutation strategy with greedy acceptance.
    """
    new_pop = pop.copy()
    best_idx = np.argmin(scores)
    best = pop[best_idx]
    dim = len(bounds)
    pop_len = len(pop)

    for i in range(pop_len):
        # Limited attempts for computational efficiency
        for _ in range(3):
            r1, r2 = np.random.choice(pop_len, 2, replace=False)
            
            # Mutation vector
            mutant = pop[i] + mutation * (best - pop[r1] + pop[r2] - pop[i])
            
            # Crossover
            cross_mask = np.random.rand(dim) < recomb
            trial = np.where(cross_mask, mutant, pop[i])

            # Enforce bounds
            for d in range(dim):
                lo, hi = bounds[d]
                trial[d] = np.clip(trial[d], lo, hi)

            trial_score = objective(trial)

            # Greedy selection
            if trial_score < scores[i]:
                new_pop[i] = trial
                scores[i] = trial_score
                break 

    return new_pop, scores

class HybridDE:
    def __init__(self, bounds):
        self.bounds = bounds

    def run(self, pop_size=40, max_iter=100):
        """
        Executes the optimization loop.
        """
        dim = len(self.bounds)
        pop = np.zeros((pop_size, dim))

        # Population Initialization
        for d in range(dim):
            lo, hi = self.bounds[d]
            pop[:, d] = np.random.uniform(lo, hi, pop_size)

        # Initial Evaluation
        scores = np.array([objective(ind) for ind in pop])
        print(f"[DE] Initial Best Correlation with EPA: {scores.min():.4f}")

        # Evolution Loop
        for it in range(max_iter):
            pop, scores = first_to_best_mutation(pop, scores)
            
            if it % 10 == 0:
                print(f"[DE] Iter {it} — Best EPA Correlation: {scores.min():.6f}")

        best_idx = np.argmin(scores)
        return pop[best_idx], scores[best_idx]

# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] Starting Differential Evolution Optimization...")
    
    # Initialize and run optimizer
    optimizer = HybridDE(bounds)
    best_params, best_score = optimizer.run(pop_size=50, max_iter=150)

    alpha_best, beta_best, gamma_best = best_params

    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Optimal Parameters Found:")
    print(f"   Alpha (Geometry Decay): {alpha_best:.4f}")
    print(f"   Beta  (Spacing Weight): {beta_best:.4f}")
    print(f"   Gamma (Integrity Weight): {gamma_best:.4f}")
    print(f"Best Correlation with EPA: {best_score:.4f}")
    print("(Note: Lower/More negative indicates better predictive power for defense)")

    # -------------------------------------------------------
    # REBUILD FINAL METRICS
    # -------------------------------------------------------
    
    print("[INFO] Recomputing final metrics with optimal parameters...")

    # Final DCI Calculation
    dci_final = np.exp(-alpha_best * raw_dist)

    # Final DIS Calculation
    dis_raw_final = (beta_best * spacing + gamma_best * integrity)
    dis_final = dis_raw_final / (np.max(dis_raw_final) + 1e-9)

    # Prepare Output DataFrame
    df_out = df.copy()
    df_out["dci_opt"] = dci_final
    df_out["dis_opt"] = dis_final

    # Select relevant columns for analytics and visualization
    cols_to_save = [
        "game_id", "play_id", 
        "cluster_id", "distance_to_ideal", "integrity_proxy", # Metadata
        "dci_opt", "dis_opt",  # Optimized Metrics
        "expected_points_added", "pass_result" # Ground Truth
    ]
    
    # Safety check for columns existence
    existing_cols = [c for c in cols_to_save if c in df_out.columns]
    
    # Save optimized metrics
    df_out[existing_cols].to_parquet(OUT_PATH, index=False)

    # Save best parameters for reproducibility
    with open(PARAM_PATH, "w") as f:
        json.dump({
            "alpha": float(alpha_best),
            "beta": float(beta_best),
            "gamma": float(gamma_best),
            "epa_correlation": float(best_score)
        }, f, indent=4)

    print("\n   Optimization Phase Completed Successfully.")
    print(f"    Optimized metrics saved to: {OUT_PATH}")
    print(f"    Best parameters saved to:   {PARAM_PATH}")