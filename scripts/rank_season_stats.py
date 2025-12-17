#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 8: SEASON DEFENSE RANKING
===============================

SUMMARY
-------
Generates a season-level leaderboard aggregating the two key defensive metrics:
Defensive Coverage Index (DCI) and Defensive Integrity Score (DIS).

It normalizes raw scores to a 0-100 scale and computes a composite 'Total Score'.

INPUTS
------
1. metrics_playlevel_supervised.parquet
2. supplementary_data.csv

OUTPUTS
-------
1. season_defense_rankings.csv (Saved in metrics folder)
"""

import pandas as pd
import numpy as np
import os

# -------------------------------------------------------
# 1. PATH CONFIGURATION
# -------------------------------------------------------
# Base directories
METRICS_DIR = "/lustre/home/dante/compartido/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

# Input files
METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")

# Output file
OUTPUT_PATH = os.path.join(METRICS_DIR, "season_defense_rankings.csv")

# -------------------------------------------------------
# 2. DATA LOADING
# -------------------------------------------------------
print("[INFO] Loading metrics and data from Lustre...", flush=True)

try:
    df_metrics = pd.read_parquet(METRICS_PATH)
    print(f"    -> Metrics loaded: {len(df_metrics)} plays.")
except Exception as e:
    print(f"[ERROR] Could not load metrics file at: {METRICS_PATH}")
    print(f"        Error: {e}")
    exit()

try:
    df_supp = pd.read_csv(SUPP_PATH, low_memory=False)
    print(f"    -> Supplementary data loaded.")
except Exception as e:
    print(f"[ERROR] Could not find supplementary_data.csv at: {SUPP_PATH}")
    print(f"        Error: {e}")
    exit()

# -------------------------------------------------------
# 3. CLEANING AND MERGING
# -------------------------------------------------------
print("[INFO] Processing tables...", flush=True)

rename_map = {
    'gameId': 'game_id', 
    'playId': 'play_id', 
    'defensiveTeam': 'defensive_team'
}
df_supp.rename(columns=rename_map, inplace=True)

if 'defensive_team' not in df_supp.columns:
    print("[ERROR] 'defensive_team' column missing in supplementary data.")
    exit()

merged_df = df_metrics.merge(
    df_supp[['game_id', 'play_id', 'defensive_team']], 
    on=['game_id', 'play_id'], 
    how='inner'
)
print(f"[INFO] Total merged plays ready for analysis: {len(merged_df)}")

# -------------------------------------------------------
# 4. RANKING CALCULATION
# -------------------------------------------------------

# Identify correct DIS column
if 'dis_final' in merged_df.columns:
    dis_col = 'dis_final'
elif 'dis_opt' in merged_df.columns:
    dis_col = 'dis_opt'
elif 'integrity_proxy' in merged_df.columns:
    dis_col = 'integrity_proxy'
    print("[WARN] Using 'integrity_proxy' as DIS.")
else:
    print("[ERROR] No Structural Integrity column found.")
    exit()

# Group by Defense
season_stats = merged_df.groupby('defensive_team')[['dci_supervised', dis_col]].mean().reset_index()
season_stats.columns = ['Team', 'DCI_Raw', 'DIS_Raw']

# Normalization (0-100)
def normalize_series(series):
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v: return 50.0
    return ((series - min_v) / (max_v - min_v)) * 100

season_stats['DCI_Score'] = normalize_series(season_stats['DCI_Raw'])
season_stats['DIS_Score'] = normalize_series(season_stats['DIS_Raw'])

# Composite Score (50/50 split)
W_DCI = 0.5
W_DIS = 0.5
season_stats['Total_Score'] = (season_stats['DCI_Score'] * W_DCI) + (season_stats['DIS_Score'] * W_DIS)

# Sort descending
season_stats = season_stats.sort_values(by='Total_Score', ascending=False).reset_index(drop=True)

# -------------------------------------------------------
# 5. SAVING OUTPUT
# -------------------------------------------------------
print(f"[INFO] Saving ranking to: {OUTPUT_PATH}")
try:
    season_stats.to_csv(OUTPUT_PATH, index=False)
    print("    -> Save successful.")
except Exception as e:
    print(f"[ERROR] Could not save CSV file: {e}")

# -------------------------------------------------------
# 6. CONSOLE REPORT
# -------------------------------------------------------
pd.set_option('display.float_format', '{:.1f}'.format)
pd.set_option('display.max_rows', 50)

print("\n" + "="*75)
print("NFL SEASON DEFENSIVE RANKINGS (DCI + DIS)")
print("="*75)
print("   DCI: Physical Coverage Tightness")
print("   DIS: Structural Integrity & Discipline")
print("-" * 75)

print("TOP 10 DEFENSES:")
print(season_stats.head(10).to_string(index=False))

print("\n" + "-" * 75)
print("BOTTOM 5 DEFENSES:")
print(season_stats.tail(5).sort_values(by='Total_Score', ascending=True).to_string(index=False))

print("\n[INFO] Execution completed.")