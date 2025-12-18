#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 11: METRIC VALIDATION - EPA CORRELATION
=============================================

SUMMARY
-------
Generates visual validation of the defensive metrics by plotting their 
correlation with Expected Points Added (EPA).

It creates scatter plots overlaid with linear regression trend lines to 
demonstrate the negative correlation between defensive quality (DCI/DIS) 
and offensive success (EPA).

METHODOLOGY
-----------
1. Loads play-level metrics.
2. Filters out extreme EPA outliers (rare turnovers/touchdowns) for visual clarity.
3. Plots DCI vs. EPA and DIS vs. EPA with:
   - Scatter points (individual plays).
   - Regression line (trend).
   - Pearson correlation coefficient (r).

INPUTS
------
1. metrics_playlevel_supervised.parquet

OUTPUTS
-------
1. epa_correlation_regplot.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------
METRICS_DIR = "/lustre/proyectos/p037/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
# Supplementary path defined for consistency, though not strictly used in this specific visualization
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")

OUTPUT_FILE = "epa_correlation_regplot.png"

# -------------------------------------------------------
# 2. DATA LOADING
# -------------------------------------------------------
print("[INFO] Loading metrics data...", flush=True)

if not os.path.exists(METRICS_PATH):
    print(f"[ERROR] Metrics file not found at: {METRICS_PATH}")
    exit()

try:
    df = pd.read_parquet(METRICS_PATH)
    print(f"    -> Loaded {len(df)} rows.")
except Exception as e:
    print(f"[ERROR] Failed to load parquet: {e}")
    exit()

# -------------------------------------------------------
# 3. PREPROCESSING
# -------------------------------------------------------
print("[INFO] Cleaning data for visualization...", flush=True)

# Clean data: Drop rows where any key metric is NaN
df_clean = df.dropna(subset=['dci_supervised', 'dis_final', 'epa']).copy()

# Optional: Filter out extreme EPA outliers (rare turnovers/touchdowns) for a cleaner view
# Keeping EPA between -5 and 5 covers 99% of normal plays
original_len = len(df_clean)
df_clean = df_clean[(df_clean['epa'] > -5) & (df_clean['epa'] < 5)]
print(f"    -> Filtered {original_len - len(df_clean)} outliers (EPA < -5 or > 5).")

# -------------------------------------------------------
# 4. HELPER FUNCTION FOR REGRESSION
# -------------------------------------------------------
def plot_regression(ax, x_data, y_data, color_scatter, color_line, title, xlabel):
    # 1. Plot the Scatter (Raw Data)
    # Alpha is extremely low (0.15) because we likely have thousands of dots
    ax.scatter(x_data, y_data, alpha=0.15, c=color_scatter, s=15, edgecolors='none', label='Individual Plays')

    # 2. Calculate Regression Line (y = mx + b)
    # Polyfit returns [slope, intercept] for degree 1
    slope, intercept = np.polyfit(x_data, y_data, 1)
    
    # Create a line range based on min/max X
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_pred = slope * x_range + intercept
    
    # 3. Plot the Regression Line
    ax.plot(x_range, y_pred, color=color_line, linewidth=3, linestyle='--', label=f'Trend (Slope: {slope:.3f})')

    # 4. Calculate Correlation (Pearson r)
    correlation = x_data.corr(y_data)
    
    # 5. Styling
    ax.set_title(f"{title}\nCorrelation (r): {correlation:.3f}", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Offensive EPA (Expected Points Added)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add a horizontal line at 0 EPA (League Average)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

# -------------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------------
print("[INFO] Generating regression plots...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# --- Plot 1: DCI vs EPA ---
plot_regression(
    axes[0], 
    df_clean['dci_supervised'], 
    df_clean['epa'], 
    color_scatter='#4c72b0', # Blue dots
    color_line='darkblue',   # Dark Blue line
    title="Defensive Coverage (DCI) Impact on EPA",
    xlabel="DCI Score (Higher = Tighter Coverage)"
)

# --- Plot 2: DIS vs EPA ---
plot_regression(
    axes[1], 
    df_clean['dis_final'], 
    df_clean['epa'], 
    color_scatter='#c44e52', # Red dots
    color_line='darkred',    # Dark Red line
    title="Defensive Integrity (DIS) Impact on EPA",
    xlabel="DIS Score (Higher = Better Integrity)"
)

# -------------------------------------------------------
# 6. SAVE AND SHOW
# -------------------------------------------------------
plt.suptitle('Statistical Validation: Does Better Defense Lower Offensive EPA?', fontsize=18, y=1.02)
plt.tight_layout()

print(f"[INFO] Saving plot to: {OUTPUT_FILE}")
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
print("[SUCCESS] Execution completed.")

# plt.show() # Commented out for headless server environment
