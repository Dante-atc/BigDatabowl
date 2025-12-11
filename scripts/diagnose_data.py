#!/usr/bin/env python
# coding: utf-8

"""
CORRELATION ANALYSIS (EPA VS METRICS)
=====================================

SUMMARY
-------
This script performs a statistical validation of the custom defensive metrics 
(Distance to Ideal, Spacing, Integrity) by correlating them against the 
NFL's official "Expected Points Added" (EPA).

EPA is the "Ground Truth" for value in football. If the custom metrics are valid, 
they should show a statistically significant correlation with EPA.

OUTPUT
------
- Prints descriptive statistics for EPA and custom metrics.
- Prints Pearson correlation coefficients (r) to the console.
- Displays a sample of the merged dataset for manual inspection.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------

print("--- LOADING DATA ---")
df = pd.read_parquet(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Normalize column names in supplementary data
rename_map = {
    "gameId": "game_id", 
    "playId": "play_id", 
    "passResult": "pass_result", 
    "expectedPointsAdded": "expected_points_added"
}
supp.rename(columns=rename_map, inplace=True)

# Merge Metrics with Ground Truth
print(f"Metrics rows: {len(df)}")
print(f"Supp rows:    {len(supp)}")
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")
print(f"Merged rows:  {len(merged)}")

print("\n--- EPA CHECK (GROUND TRUTH) ---")
print(merged["expected_points_added"].describe())
print(f"Nulls in EPA: {merged['expected_points_added'].isna().sum()}")
print(f"Exact zeros in EPA: {(merged['expected_points_added'] == 0).sum()}")

print("\n--- DISTANCE CHECK (YOUR METRIC) ---")
print(merged["distance_to_ideal"].describe())

print("\n--- RAW CORRELATIONS ---")
# Check direct correlation (Pearson's r) without complex transformations
corrs = {}
targets = ["distance_to_ideal", "spacing_proxy", "integrity_proxy"]

for t in targets:
    # Clean Nulls to ensure statistical validity
    clean = merged[[t, "expected_points_added"]].dropna()
    
    # Calculate Pearson correlation
    r, _ = pearsonr(clean[t], clean["expected_points_added"])
    corrs[t] = r
    print(f"{t} vs EPA: {r:.4f}")

print("\n--- DATA SAMPLE ---")
print(merged[["game_id", "play_id", "distance_to_ideal", "expected_points_added", "pass_result"]].head(10))
