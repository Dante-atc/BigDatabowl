#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 5 — Supervised DCI Calibration & Evaluation
=================================================

This module trains a Gradient Boosting Classifier to refine the raw geometric
Defensive Coverage Index (DCI). By treating 'defensive success' as a supervised
target (pass incomplete, interception, sack), we can calibrate the metric to
account for game context.

Key Improvements:
1.  Context Integration: Incorporates Down, Distance, and Defenders in the Box.
2.  Feature Interaction: Introduces ratios (e.g., Integrity per Distance Unit).
3.  Model: Uses HistGradientBoostingClassifier for efficiency and native categorical support.
4.  Output: Produces a 'Supervised DCI' probability score correlated with EPA.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import os
import joblib

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

BASE_DIR = "/lustre/home/dante/compartido"
RAW_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

METRICS_PATH = f"{BASE_DIR}/metrics/metrics_playlevel_baseline.parquet"
SUPP_PATH = f"{RAW_DIR}/supplementary_data.csv"
OUT_PATH = f"{BASE_DIR}/metrics/metrics_playlevel_supervised.parquet"
MODEL_OUT = f"{BASE_DIR}/models/dci_calibrator.pkl"

SEED = 42

# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------

print("[INFO] Loading baseline metrics...")
df = pd.read_parquet(METRICS_PATH)

print("[INFO] Loading ground truth labels (supplementary data)...")
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# --- STANDARDIZATION & RENAMING ---
cols_map = {
    "gameId": "game_id", 
    "playId": "play_id", 
    "passResult": "pass_result",
    "expectedPointsAdded": "epa",
    "expected_points_added": "epa",
    "down": "down",
    "yardsToGo": "yards_to_go",
    "defendersInTheBox": "defenders_in_the_box"
}
supp.rename(columns=cols_map, inplace=True)

# Merge Datasets
merged = df.merge(supp, on=["game_id", "play_id"], how="inner")

# Filter: Focus strictly on Pass Plays (Dropbacks)
valid_pass_types = ['C', 'I', 'S', 'IN']
pass_df = merged[merged['pass_result'].isin(valid_pass_types)].copy()

print(f"[INFO] Dataset filtered. Analyzing {len(pass_df)} valid pass plays.")

# Define Target (1 = Good Defense / 0 = Completed Pass)
pass_df['defensive_success'] = pass_df['pass_result'].apply(lambda x: 0 if x == 'C' else 1)

# -----------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------

print("[INFO] Engineering contextual features...")

# 1. Impute missing values in context features
pass_df['down'] = pass_df['down'].fillna(1).astype(int)
pass_df['yards_to_go'] = pass_df['yards_to_go'].fillna(10).astype(int)
pass_df['defenders_in_the_box'] = pass_df['defenders_in_the_box'].fillna(6).astype(int)

# 2. Feature Interactions (Ratios)
# "Integrity per Distance Unit": Does strict structural integrity compensate for loose spacing?
pass_df['integrity_dist_ratio'] = pass_df['integrity_proxy'] / (pass_df['distance_to_ideal'] + 1e-6)

# 3. Categorical Handling
pass_df['cluster_id'] = pass_df['cluster_id'].astype('category')

features = [
    'distance_to_ideal', 
    'distance_to_second', 
    'spacing_proxy', 
    'integrity_proxy',
    'integrity_dist_ratio',  
    'down',                  
    'yards_to_go',           
    'defenders_in_the_box',         
    'cluster_id'             # Handled natively by HistGradientBoosting
]

X = pass_df[features]
y = pass_df['defensive_success'].values

# -----------------------------------------------------------
# MODEL TRAINING 
# -----------------------------------------------------------

print("[INFO] Training HistGradientBoostingClassifier...")

clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_depth=6,
    l2_regularization=0.1,
    random_state=SEED,
    categorical_features='from_dtype' 
)

# Generate Out-of-Fold Predictions (Calibrated Scores)
print("[INFO] Generating calibrated DCI scores via 5-Fold Cross-Validation...")
dci_probs = cross_val_predict(
    clf, X, y, cv=5, method='predict_proba'
)[:, 1]

# Fit final model on full dataset
clf.fit(X, y)

# -----------------------------------------------------------
# EVALUATION & EXPORT
# -----------------------------------------------------------

auc_score = roc_auc_score(y, dci_probs)
print(f"\n=== MODEL PERFORMANCE ===")
print(f"   AUC: {auc_score:.4f}")

# Correlation Analysis
pass_df['dci_supervised'] = dci_probs
corr_epa = pass_df[['dci_supervised', 'epa']].corr().iloc[0, 1]

print(f"   DCI vs EPA Correlation: {corr_epa:.4f}")
print(f"   (Note: Strong negative correlation implies valid defensive signal)")

# Save Final Metrics
pass_df['dis_final'] = pass_df['integrity_proxy']

output_cols = [
    'game_id', 'play_id', 
    'dci_supervised', 
    'dis_final', 
    'epa', 
    'pass_result', 
    'cluster_id'
]

pass_df[output_cols].to_parquet(OUT_PATH, index=False)
print(f"\n[INFO] Final Optimized Metrics saved to: {OUT_PATH}")

# Save Model Artifact
joblib.dump(clf, MODEL_OUT)
print(f"[INFO] Saved DCI Calibrator model to: {MODEL_OUT}")