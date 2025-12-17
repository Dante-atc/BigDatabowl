#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 9: ADVANCED DEFENSIVE LANDSCAPE
=====================================

SUMMARY
-------
Generates a competition-quality visualization map to showcase the model-derived 
defensive metrics in a 2D space.

VISUAL ELEMENTS
---------------
1. Efficiency Isoquants: Contour lines of equal composite score (DCI + DIS).
2. Pareto Frontier: The "unbeatable" boundary where no team is better in both metrics.
3. Highlight Region: A convex hull envelope emphasizing the top-tier teams.
4. Volume Sizing: Point sizes reflect the sample size (number of plays).

INPUTS
------
1. metrics_playlevel_supervised.parquet
2. supplementary_data.csv

OUTPUTS
-------
1. defensive_performance_advanced.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# -------------------------------------------------------
# 1. CONFIGURATION (SERVER / YUCA)
# -------------------------------------------------------
METRICS_DIR = "/lustre/home/dante/compartido/metrics"
RAW_DATA_DIR = "/lustre/proyectos/p037/datasets/raw/114239_nfl_competition_files_published_analytics_final"

METRICS_PATH = os.path.join(METRICS_DIR, "metrics_playlevel_supervised.parquet")
SUPP_PATH = os.path.join(RAW_DATA_DIR, "supplementary_data.csv")
OUTPUT_FILE = os.path.join(METRICS_DIR, "defensive_performance_advanced.png")

# Visualization Parameters
TOP_N_LABELS = 10
TOP_TIER_HULL = 8          # Number of teams defining the "Elite Envelope"
WEIGHT_DCI = 0.5
WEIGHT_DIS = 0.5

# Styling
N_CONTOUR_LEVELS = 5       # Cleaner look
CONTOUR_ALPHA = 0.12
LABEL_FONT_SIZE = 10
LABEL_WEIGHT = "semibold"

# -------------------------------------------------------
# 2. DATA LOADING & PROCESSING
# -------------------------------------------------------
print("[INFO] Loading data...", flush=True)

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")
if not os.path.exists(SUPP_PATH):
    raise FileNotFoundError(f"Supplementary file not found: {SUPP_PATH}")

df_metrics = pd.read_parquet(METRICS_PATH)
df_supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Standardize columns
df_supp.rename(columns={
    "gameId": "game_id",
    "playId": "play_id",
    "defensiveTeam": "defensive_team"
}, inplace=True)

print("[INFO] Merging datasets...", flush=True)
merged_df = df_metrics.merge(
    df_supp[["game_id", "play_id", "defensive_team"]],
    on=["game_id", "play_id"],
    how="inner"
)

# Aggregation: Mean metrics + Unique play count (for robustness)
team_stats = merged_df.groupby("defensive_team").agg(
    dci_supervised=("dci_supervised", "mean"),
    dis_final=("dis_final", "mean"),
    play_count=("play_id", "nunique") 
).reset_index()

# -------------------------------------------------------
# 3. SCORING (Z-SCORES + COMPOSITE)
# -------------------------------------------------------
# We use Z-scores to normalize axes for the isoquants
mu_dci = team_stats["dci_supervised"].mean()
mu_dis = team_stats["dis_final"].mean()
sig_dci = team_stats["dci_supervised"].std(ddof=0) or 1.0
sig_dis = team_stats["dis_final"].std(ddof=0) or 1.0

team_stats["dci_z"] = (team_stats["dci_supervised"] - mu_dci) / sig_dci
team_stats["dis_z"] = (team_stats["dis_final"] - mu_dis) / sig_dis

# Composite "Elite Score"
team_stats["elite_score"] = (WEIGHT_DCI * team_stats["dci_z"]) + (WEIGHT_DIS * team_stats["dis_z"])

# Rank for labeling
team_stats = team_stats.sort_values("elite_score", ascending=False).reset_index(drop=True)
team_stats["rank"] = team_stats.index + 1

# -------------------------------------------------------
# 4. PARETO FRONTIER LOGIC
# -------------------------------------------------------
def get_pareto_frontier(df, x_col, y_col):
    """Identifies the 'dominating' set of points (Maximize X and Y)."""
    sorted_df = df.sort_values(x_col, ascending=False)
    pareto = []
    max_y = -np.inf

    for row in sorted_df.itertuples():
        yv = getattr(row, y_col)
        if yv >= max_y:
            pareto.append((getattr(row, x_col), yv))
            max_y = yv

    return sorted(pareto, key=lambda t: t[0])

pareto_points = get_pareto_frontier(team_stats, "dci_supervised", "dis_final")
pareto_x, pareto_y = zip(*pareto_points) if pareto_points else ([], [])

# -------------------------------------------------------
# 5. PLOTTING
# -------------------------------------------------------
print("[INFO] Generating advanced plot...", flush=True)
plt.figure(figsize=(14, 11))
ax = plt.gca()

# --- A. Isoquants (Background Contours) ---
x_min, x_max = team_stats["dci_supervised"].min(), team_stats["dci_supervised"].max()
y_min, y_max = team_stats["dis_final"].min(), team_stats["dis_final"].max()

pad_x = (x_max - x_min) * 0.12 if x_max != x_min else 0.01
pad_y = (y_max - y_min) * 0.12 if y_max != y_min else 0.01

xi = np.linspace(x_min - pad_x, x_max + pad_x, 120)
yi = np.linspace(y_min - pad_y, y_max + pad_y, 120)
X, Y = np.meshgrid(xi, yi)

Z_dci = (X - mu_dci) / sig_dci
Z_dis = (Y - mu_dis) / sig_dis
Z_score = (WEIGHT_DCI * Z_dci) + (WEIGHT_DIS * Z_dis)

levels = np.linspace(Z_score.min(), Z_score.max(), N_CONTOUR_LEVELS)
plt.contour(
    X, Y, Z_score,
    levels=levels,
    colors="gray",
    alpha=CONTOUR_ALPHA,
    linestyles="dashed",
    zorder=1
)

# --- B. Highlight Region (Top Tier Hull) ---
top_tier = team_stats.head(TOP_TIER_HULL)[["dci_supervised", "dis_final"]].values
if len(top_tier) >= 3:
    hull = ConvexHull(top_tier)
    hull_points = top_tier[hull.vertices]
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)

    plt.fill(hull_points[:, 0], hull_points[:, 1],
             color="gold", alpha=0.10, label=f"Top-{TOP_TIER_HULL} Highlight Region", zorder=0)
    plt.plot(hull_points[:, 0], hull_points[:, 1],
             color="gold", alpha=0.35, linestyle="--", zorder=2)

# --- C. Main Scatter (Size = Play Volume) ---
sizes = (team_stats["play_count"] / team_stats["play_count"].max()) * 380 + 90

sc = plt.scatter(
    team_stats["dci_supervised"],
    team_stats["dis_final"],
    c=team_stats["elite_score"],
    s=sizes,
    cmap="viridis",
    alpha=0.85,
    edgecolors="white",
    linewidth=1.2,
    zorder=3
)

cbar = plt.colorbar(sc, pad=0.02)
cbar.set_label("Composite Elite Score (Z-Score Sum)", rotation=270, labelpad=18, fontweight="bold")

# --- D. Pareto Frontier Line ---
if pareto_x:
    plt.plot(pareto_x, pareto_y,
             color="crimson", linestyle="-", linewidth=2.5, alpha=0.60, zorder=2,
             label="Pareto Frontier")

# --- E. League Averages ---
plt.axvline(mu_dci, color="black", linestyle=":", alpha=0.35, zorder=1)
plt.axhline(mu_dis, color="black", linestyle=":", alpha=0.35, zorder=1)

plt.text(x_max + pad_x * 0.02, mu_dis, "League Avg DIS",
         ha="right", va="bottom", fontsize=9, alpha=0.55)
plt.text(mu_dci, y_max + pad_y * 0.02, "League Avg DCI",
         ha="left", va="top", fontsize=9, alpha=0.55, rotation=90)

# --- F. Smart Labels (Cyclic Offsets) ---
top_teams_list = team_stats.head(TOP_N_LABELS).reset_index(drop=True)
offset_cycle = [(15, 15), (15, -15), (-15, 15), (-15, -15), (0, 22), (0, -22)]

for i, row in top_teams_list.iterrows():
    dx, dy = offset_cycle[i % len(offset_cycle)]
    label_txt = f"{row['defensive_team']}\n#{int(row['rank'])}"

    plt.annotate(
        label_txt,
        (row["dci_supervised"], row["dis_final"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=LABEL_FONT_SIZE,
        fontweight=LABEL_WEIGHT,
        ha="center",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.92, lw=0.5),
        arrowprops=dict(arrowstyle="-", color="black", alpha=0.25, lw=1.0)
    )

# -------------------------------------------------------
# 6. FINAL LAYOUT & SAVE
# -------------------------------------------------------
plt.title("Defensive Landscape: The Elite Frontier", fontsize=20, fontweight="bold", pad=15)
plt.xlabel("Defensive Coverage Index (DCI) → Higher is Tighter", fontsize=13, fontweight="bold")
plt.ylabel("Defensive Integrity Score (DIS) → Higher is Better", fontsize=13, fontweight="bold")

info_text = (
    "○ Point Size: Sample Size (Plays)\n"
    "-- Dashed Lines: Efficiency Isoquants\n"
    "▬ Red Line: Pareto Frontier\n"
    "Gold Region: Elite Tier Envelope"
)
plt.text(
    x_min - pad_x * 0.05, y_max + pad_y * 0.02,
    info_text,
    fontsize=9,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", fc="whitesmoke", ec="none", alpha=0.85)
)

plt.grid(True, linestyle="--", alpha=0.12)
plt.xlim(x_min - pad_x, x_max + pad_x * 1.25)
plt.ylim(y_min - pad_y, y_max + pad_y)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"[SUCCESS] Advanced plot saved to: {OUTPUT_FILE}")