"""
Data Diagnosis & Correlation Check
==================================

This script performs a quick sanity check on the calculated metrics against the
Ground Truth (EPA) labels. It checks for data integrity issues (NaNs, Zeros)
and computes raw Pearson correlations to validate the predictive signal.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Paths
METRICS_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final/supplementary_data.csv"

print("--- LOADING DATA ---")
df = pd.read_parquet(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Standardize names
rename_map = {
    "gameId": "game_id", 
    "playId": "play_id", 
    "passResult": "pass_result", 
    "expectedPointsAdded": "expected_points_added"
}
supp.rename(columns=rename_map, inplace=True)

# Merge
print(f"Metrics rows: {len(df)}")
print(f"Suppl rows:   {len(supp)}")
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")
print(f"Merged rows:  {len(merged)}")

print("\n--- EPA DISTRIBUTION CHECK (Ground Truth) ---")
print(merged["expected_points_added"].describe())
print(f"NaNs in EPA: {merged['expected_points_added'].isna().sum()}")
print(f"Exact Zeros in EPA: {(merged['expected_points_added'] == 0).sum()}")

print("\n--- DISTANCE METRIC CHECK (Your Metric) ---")
print(merged["distance_to_ideal"].describe())

print("\n--- RAW CORRELATION ANALYSIS ---")
# Direct Pearson correlation check
corrs = {}
targets = ["distance_to_ideal", "spacing_proxy", "integrity_proxy"]

for t in targets:
    # Drop NaNs for valid calculation
    clean = merged[[t, "expected_points_added"]].dropna()
    if not clean.empty:
        r, _ = pearsonr(clean[t], clean["expected_points_added"])
        corrs[t] = r
        print(f"{t} vs EPA: {r:.4f}")
    else:
        print(f"{t} vs EPA: Insufficient Data")

print("\n--- DATA SAMPLE ---")
cols_preview = ["game_id", "play_id", "distance_to_ideal", "expected_points_added", "pass_result"]
print(merged[cols_preview].head(10))