#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 7: TEAM RANKING REPORT
============================

SUMMARY
-------
Generates a Defensive Coverage Index (DCI) and Defensive Interception Score (DIS) 
ranking report for NFL teams.

INPUTS
------
1. metrics_playlevel_supervised.parquet
2. supplementary_data.csv

OUTPUTS
-------
1. defensive_rankings.parquet (Saved in metrics folder)
"""

import pandas as pd
import os

# -------------------------------------------------------
# 1. PATH CONFIGURATION
# -------------------------------------------------------
# Base directories
METRICS_DIR = "/lustre/proyectos/p037/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

# Input files
METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")

# Output file
OUTPUT_PATH = os.path.join(METRICS_DIR, "defensive_rankings.parquet")

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
    'defensiveTeam': 'defensive_team',
    'possessionTeam': 'possession_team',
    'week': 'week',
    'homeTeamAbbr': 'home_team',
    'visitorTeamAbbr': 'away_team'
}
df_supp.rename(columns=rename_map, inplace=True)

cols_to_use = ['game_id', 'play_id', 'defensive_team', 'possession_team']
if 'week' in df_supp.columns: cols_to_use.append('week')
if 'home_team' in df_supp.columns: cols_to_use.append('home_team')
if 'away_team' in df_supp.columns: cols_to_use.append('away_team')

df_supp_clean = df_supp[cols_to_use].copy()

merged_df = df_metrics.merge(df_supp_clean, on=['game_id', 'play_id'], how='inner')
print(f"[INFO] Total merged plays ready for analysis: {len(merged_df)}")

# -------------------------------------------------------
# 4. RANKING CALCULATION
# -------------------------------------------------------

def get_opponent(row):
    if 'home_team' in row and pd.notna(row['home_team']):
        if row['defensive_team'] == row['home_team']:
            return row['away_team']
        return row['home_team']
    return row['possession_team']

group_cols = ['game_id', 'defensive_team']
if 'week' in merged_df.columns:
    group_cols.insert(1, 'week')

# Aggregating both DCI and DIS
team_stats = merged_df.groupby(group_cols)[['dci_supervised', 'dis_final']].mean().reset_index()

meta_cols = group_cols + ['home_team', 'away_team', 'possession_team']
available_meta = [c for c in meta_cols if c in merged_df.columns]

team_stats = team_stats.merge(
    merged_df[available_meta].drop_duplicates(subset=group_cols),
    on=group_cols,
    how='left'
)

team_stats['opponent'] = team_stats.apply(get_opponent, axis=1)

# Renaming columns for the final report
if 'week' in team_stats.columns:
    final_ranking = team_stats[['week', 'defensive_team', 'opponent', 'dci_supervised', 'dis_final']]
    final_ranking.columns = ['Week', 'Defensive_Team', 'Opponent', 'Avg_DCI', 'Avg_DIS']
else:
    final_ranking = team_stats[['defensive_team', 'opponent', 'dci_supervised', 'dis_final']]
    final_ranking.columns = ['Defensive_Team', 'Opponent', 'Avg_DCI', 'Avg_DIS']

# Sort by DCI initially
final_ranking = final_ranking.sort_values(by='Avg_DCI', ascending=False).reset_index(drop=True)

# -------------------------------------------------------
# 5. SAVING OUTPUT
# -------------------------------------------------------
print(f"[INFO] Saving ranking to: {OUTPUT_PATH}")
try:
    final_ranking.to_parquet(OUTPUT_PATH, index=False)
    print("    -> Save successful.")
except Exception as e:
    print(f"[ERROR] Could not save parquet file: {e}")

# -------------------------------------------------------
# 6. CONSOLE REPORT
# -------------------------------------------------------

print("\n" + "="*60)
print("TOP 10 BEST DEFENSIVE PERFORMANCES (BY DCI)")
print("="*60)
print(final_ranking.head(10))

print("\n" + "="*60)
print("TOP 10 BEST DEFENSIVE PERFORMANCES (BY DIS)")
print("="*60)
print(final_ranking.sort_values(by='Avg_DIS', ascending=False).head(10))

print("\n" + "="*60)
print("SEASON RANKING (OVERALL AVERAGE)")
print("="*60)
season_ranking = final_ranking.groupby('Defensive_Team')[['Avg_DCI', 'Avg_DIS']].mean().reset_index()
season_ranking = season_ranking.sort_values(by='Avg_DCI', ascending=False).reset_index(drop=True)
print(season_ranking.head(10))

print("\n" + "="*60)
print("HEAD OF SAVED FILE")
print("="*60)
print(final_ranking.head())

print("\n[INFO] Execution completed.")
