#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 10: STATISTICAL IMPACT ANALYSIS
=====================================

SUMMARY
-------
Performs rigorous statistical validation of the DCI and DIS metrics.
It answers the question: "Do these metrics actually correlate with preventing big plays?"

METHODOLOGY
-----------
1. Logistic Regression: Measures the impact of DCI/DIS on the probability of 
   allowing an 'Explosive Play' (EPA >= 2.0).
2. Quantile Regression: Measures how coverage quality lowers the "ceiling" of 
   offensive production (reducing variance).

OUTPUTS
-------
- Statistical summary printed to console (Odds Ratios, Coefficients).
- explosive_play_reduction.png (Bar chart).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------
# 1. CONFIGURATION (SERVER / YUCA)
# -------------------------------------------------------
METRICS_DIR = "/lustre/home/dante/compartido/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")
PLOT_OUTPUT = os.path.join(METRICS_DIR, "explosive_play_reduction.png")

# -------------------------------------------------------
# 2. DATA LOADING
# -------------------------------------------------------
print("[INFO] Loading metrics and metadata...", flush=True)
try:
    df_metrics = pd.read_parquet(METRICS_PATH)
    df_supp = pd.read_csv(SUPP_PATH, low_memory=False)
except Exception as e:
    print(f"[ERROR] Data load failed: {e}")
    exit()

# -------------------------------------------------------
# 3. PREPROCESSING
# -------------------------------------------------------
print("[INFO] Cleaning and merging...", flush=True)

# Map known columns
rename_map = {
    'gameId': 'game_id',
    'playId': 'play_id',
    'defensiveTeam': 'defensive_team',
    'possessionTeam': 'possession_team',
    'down': 'down',
    'quarter': 'qtr',
    'preSnapHomeScore': 'home_score',
    'preSnapVisitorScore': 'away_score'
}
df_supp.rename(columns=rename_map, inplace=True)

# Find 'Yards To Go' dynamically
possible_yd_cols = ['yardsToGo', 'YardsToGo', 'yards_to_go', 'ydstogo']
found_yd_col = None
for col in possible_yd_cols:
    if col in df_supp.columns:
        found_yd_col = col
        break

if found_yd_col:
    df_supp.rename(columns={found_yd_col: 'ydstogo'}, inplace=True)
else:
    df_supp['ydstogo'] = 10 

# Prepare Control Columns
control_cols = ['game_id', 'play_id', 'defensive_team', 'possession_team', 'ydstogo']
if 'down' in df_supp.columns: control_cols.append('down')
if 'home_score' in df_supp.columns and 'away_score' in df_supp.columns:
    df_supp['score_diff'] = df_supp['home_score'] - df_supp['away_score']
    control_cols.append('score_diff')

# Merge
df = df_metrics.merge(df_supp[control_cols], on=['game_id', 'play_id'], how='inner')

# Filter for Passes Only
valid_pass_codes = ['C', 'I', 'S', 'IN', 'COMPLETE', 'INCOMPLETE', 'INTERCEPTION', 'SACK']
if 'pass_result' in df.columns:
    df = df[df['pass_result'].isin(valid_pass_codes)].copy()

print(f"[INFO] Final analysis set: {len(df)} plays.")

# -------------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------------
# Target: Explosive Play (EPA >= 2.0)
df['is_explosive'] = (df['epa'] >= 2.0).astype(int)

# Standardize metrics (Z-score) for regression interpretation
df['dci_z'] = (df['dci_supervised'] - df['dci_supervised'].mean()) / df['dci_supervised'].std()
df['dis_z'] = (df['dis_final'] - df['dis_final'].mean()) / df['dis_final'].std()

# Drop rows with NaNs in critical columns
df.dropna(subset=['dci_supervised', 'dis_final', 'epa', 'ydstogo'], inplace=True)

# -------------------------------------------------------
# 5. EXPERIMENT 1: LOGISTIC REGRESSION
# -------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENT 1: LOGISTIC REGRESSION (Explosive Plays)")
print("="*60)

formula_logit = "is_explosive ~ dci_z + dis_z + ydstogo"
if 'down' in df.columns: formula_logit += " + C(down)"
if 'score_diff' in df.columns: formula_logit += " + score_diff"

try:
    model_logit = smf.logit(formula_logit, data=df).fit(disp=0)
    print(model_logit.summary())

    # Odds Ratios
    params = model_logit.params
    conf = model_logit.conf_int()
    conf['OR'] = np.exp(params)
    conf.columns = ['2.5%', '97.5%', 'Odds_Ratio']
    
    print("\n--- EFFECT SIZES (Odds Ratios) ---")
    print(conf.loc[['dci_z', 'dis_z']])
    print("(Note: OR < 1.0 means the metric SUCCEEDS in reducing explosive plays)")

except Exception as e:
    print(f"[ERROR] Logistic Regression failed: {e}")

# -------------------------------------------------------
# 6. EXPERIMENT 2: QUANTILE REGRESSION
# -------------------------------------------------------
print("\n" + "="*60)
print("EXPERIMENT 2: QUANTILE REGRESSION (Offensive Ceiling)")
print("="*60)

formula_quant = "epa ~ dci_z + dis_z + ydstogo"
if 'down' in df.columns: formula_quant += " + C(down)"
if 'score_diff' in df.columns: formula_quant += " + score_diff"

try:
    mod_quant = smf.quantreg(formula_quant, df)
    # Assessing the 90th percentile (The "Ceiling" of offense)
    res_quant = mod_quant.fit(q=0.9) 
    
    print(res_quant.summary())
    print("\n[INTERPRETATION] Check 'dci_z' coefficient.")
    print("If Negative: Better coverage significantly lowers the offense's best plays.")

except Exception as e:
    print(f"[ERROR] Quantile Regression failed: {e}")

# -------------------------------------------------------
# 7. VISUALIZATION
# -------------------------------------------------------
print("\n[INFO] Generating impact plot...", flush=True)
try:
    df['dci_quartile'] = pd.qcut(df['dci_supervised'], 4, labels=["Q1 (Loose)", "Q2", "Q3", "Q4 (Tight)"])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='dci_quartile', y='is_explosive', data=df, palette="Blues", errorbar=('ci', 95))
    plt.title("Probability of Explosive Play (EPA >= 2.0) by Coverage Quality")
    plt.ylabel("Explosive Play Probability")
    plt.xlabel("Defensive Coverage Index (DCI Quartiles)")
    
    plt.savefig(PLOT_OUTPUT, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to {PLOT_OUTPUT}")
    
except Exception as e:
    print(f"[WARN] Could not generate plot: {e}")