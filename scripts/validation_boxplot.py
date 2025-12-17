# coding: utf-8

"""
Defensive Metrics Validation & Visualization
=============================================

This script validates learned defensive metrics against real play outcomes
in NFL tracking data. It compares model-derived defensive scores—such as
Defensive Coverage Index (DCI) and Defensive Integrity Score (DIS)—across
different pass results using clean, publication-ready visualizations.

The primary objective is to assess whether these metrics behave consistently
and monotonically with respect to offensive success (e.g., completions vs.
sacks or interceptions), providing qualitative and quantitative validation
for downstream analysis or reporting.

Workflow:
1. Load play-level supervised metrics from a parquet file.
2. Filter the dataset to relevant passing outcomes.
3. Map raw outcome codes to human-readable labels.
4. Generate comparative boxplots for DCI and DIS by outcome.
5. Export a high-resolution figure suitable for academic papers or Kaggle.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------
# Base directories
METRICS_DIR = "/lustre/proyectos/p037/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

# Input files
METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")

# Visualization style (publication-friendly)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------
# 2. DATA LOADING & PREPARATION
# -------------------------------------------------------
print("Loading data...")

try:
    df = pd.read_parquet(METRICS_PATH)
except FileNotFoundError:
    print(f"File not found at {METRICS_PATH}. Please check the path.")
    # Optional fallback for testing/debugging
    # data = {
    #     'dci_supervised': [0.4, 0.4, 0.2, 0.3, 0.3],
    #     'dis_final': [0.4, 0.3, 0.8, 0.1, 0.4],
    #     'pass_result': ['I', 'C', 'C', 'C', 'C']
    # }
    # df = pd.DataFrame(data)
    exit()

# Retain only common passing outcomes to avoid noise from rare events
# Update codes if dataset conventions differ
target_outcomes = ['C', 'I', 'S', 'IN']
df_clean = df[df['pass_result'].isin(target_outcomes)].copy()

# Map abbreviated outcome codes to descriptive labels
label_map = {
    'C': 'Complete',
    'I': 'Incomplete',
    'S': 'Sack',
    'IN': 'Interception'
}
df_clean['Outcome'] = df_clean['pass_result'].map(label_map)

# Explicit ordering to reflect offensive success → failure
order = ['Complete', 'Incomplete', 'Sack', 'Interception']

# -------------------------------------------------------
# 3. VISUALIZATION
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

# --- Panel A: Defensive Coverage Index (DCI) ---
sns.boxplot(
    data=df_clean,
    x='Outcome',
    y='dci_supervised',
    order=order,
    palette="Blues_d",
    ax=axes[0],
    showfliers=False
)
axes[0].set_title(
    'Defensive Coverage Index (DCI) by Outcome',
    fontweight='bold',
    pad=15
)
axes[0].set_ylabel('DCI Score (Higher = Tighter Coverage)')
axes[0].set_xlabel('')

# --- Panel B: Defensive Integrity Score (DIS) ---
sns.boxplot(
    data=df_clean,
    x='Outcome',
    y='dis_final',
    order=order,
    palette="Reds_d",
    ax=axes[1],
    showfliers=False
)
axes[1].set_title(
    'Defensive Integrity Score (DIS) by Outcome',
    fontweight='bold',
    pad=15
)
axes[1].set_ylabel('DIS Score (Higher = Better Integrity)')
axes[1].set_xlabel('')

# -------------------------------------------------------
# 4. EXPORT & FINALIZATION
# -------------------------------------------------------
plt.suptitle(
    'Validation of Defensive Metrics Against Play Outcomes',
    fontsize=20,
    y=1.02
)
plt.tight_layout()

# Save high-resolution figure (Kaggle / paper ready)
output_file = "metric_validation_boxplot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {output_file}")

plt.show()
