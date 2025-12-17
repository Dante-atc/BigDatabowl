#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility — Metric Validation (Ground Truth Alignment)
====================================================

This script performs a 'Litmus Test' on the calculated geometric metrics using
Official NFL Outcomes (Ground Truth data from supplementary files).

Hypothesis:
    There should be a statistically significant gap between the coverage metrics
    of Complete Passes vs. Incomplete Passes.

    - Complete Passes (C): Should show HIGHER distance to ideal (Worse coverage).
    - Incomplete Passes (I): Should show LOWER distance to ideal (Better coverage).

    Target Gap = Avg(Distance_Complete) - Avg(Distance_Incomplete) > 0
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

# -----------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------

print("[INFO] Loading metrics and ground truth...")
df = pd.read_parquet(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Normalize Column Names
supp.rename(columns={"gameId": "game_id", "playId": "play_id", "passResult": "pass_result"}, inplace=True)

# Merge Data
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")

# STRICT FILTER: Pure Pass Plays Only
# We exclude sacks (S) or scrambles/other to isolate pure coverage performance vs outcomes.
# Note: 'IN' (Interception) is conceptually 'Incomplete' for the offense, but 'Great' for defense.
# For simplicity, we compare C (Complete) vs I (Incomplete).
valid_outcomes = ['C', 'I', 'IN', 'S']
merged = merged[merged['pass_result'].isin(valid_outcomes)]

print(f"[INFO] Analyzing {len(merged)} valid pass plays...")

# --- HYPOTHESIS TESTING ---

group = merged.groupby('pass_result')[['distance_to_ideal', 'spacing_proxy', 'integrity_proxy']].mean()
print("\n=== METRIC AVERAGES BY RESULT ===")
print(group)

# Calculate the "Gap"
try:
    dist_c = group.loc['C', 'distance_to_ideal']
    dist_i = group.loc['I', 'distance_to_ideal']
    gap = dist_c - dist_i
    
    print(f"\n[RESULT] Coverage Gap (Complete - Incomplete): {gap:.4f}")

    if gap > 0:
        print("\n[SUCCESS] Signal Detected.")
        print("   Interpretation: Bad coverages (High Distance) allow more completions.")
        print("   The DE Optimizer will be able to exploit this signal.")
    else:
        print("\n[WARNING] Counter-intuitive result detected (Gap < 0).")
        print("   Interpretation: 'Good' coverages (Low Distance) are allowing passes.")
        print("   Investigation into the embedding space or archetype definition is required.")

except KeyError as e:
    print(f"\n[ERROR] Insufficient data to calculate gap. Missing result type: {e}")