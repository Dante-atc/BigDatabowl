#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defensive Landscape: Elite Frontier — Local Runner
===================================================
Reads local CSV files and generates a def_elite-style scatter plot.

Inputs:
    - metrics_playlevel_supervised.csv  (workspace root)
    - supplementary_data.csv            (workspace root)

Output:
    - imgs/def_elite_new.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.spatial import ConvexHull

# -------------------------------------------------------
# PATHS  (relative to BigDatabowl/ root)
# -------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "metrics_playlevel_supervised.csv")
SUPP_PATH    = os.path.join(BASE_DIR, "supplementary_data.csv")
OUT_PATH     = os.path.join(BASE_DIR, "imgs", "def_elite_new.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# -------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------
TOP_N_LABELS  = 12
TOP_TIER_HULL =  8
WEIGHT_DCI    = 0.5
WEIGHT_DIS    = 0.5
DIS_LAMBDA    = 1.0   # kept for provenance; data already contains computed values

DARK_BG   = "#0d0d0d"
GOLD      = "#FFD700"
ACCENT    = "#E63946"
TEXT_CLR  = "#F0F0F0"
GRID_CLR  = "#2e2e2e"

# -------------------------------------------------------
# DATA
# -------------------------------------------------------
print("[INFO] Loading metrics …")
df_m = pd.read_csv(METRICS_PATH)

print("[INFO] Loading supplementary data …")
supp = pd.read_csv(SUPP_PATH, low_memory=False)
supp = supp[["game_id", "play_id", "defensive_team"]].drop_duplicates()

merged = df_m.merge(supp, on=["game_id", "play_id"], how="inner")

# -------------------------------------------------------
# TEAM AGGREGATION
# -------------------------------------------------------
team_stats = (
    merged
    .groupby("defensive_team")
    .agg(
        dci=("dci_supervised", "mean"),
        dis=("dis_final",       "mean"),
        n  =("play_id",         "nunique"),
    )
    .reset_index()
)

mu_dci, sig_dci = team_stats["dci"].mean(), team_stats["dci"].std(ddof=0) or 1.0
mu_dis, sig_dis = team_stats["dis"].mean(), team_stats["dis"].std(ddof=0) or 1.0

team_stats["dci_z"] = (team_stats["dci"] - mu_dci) / sig_dci
team_stats["dis_z"] = (team_stats["dis"] - mu_dis) / sig_dis
team_stats["elite"] = WEIGHT_DCI * team_stats["dci_z"] + WEIGHT_DIS * team_stats["dis_z"]
team_stats = team_stats.sort_values("elite", ascending=False).reset_index(drop=True)
team_stats["rank"] = team_stats.index + 1

print(f"[INFO] {len(team_stats)} teams aggregated.")

# -------------------------------------------------------
# PARETO FRONTIER
# -------------------------------------------------------
def pareto_frontier(df):
    s = df.sort_values("dci", ascending=False)
    pts, best_dis = [], -np.inf
    for row in s.itertuples():
        if row.dis >= best_dis:
            pts.append((row.dci, row.dis))
            best_dis = row.dis
    return sorted(pts, key=lambda t: t[0])

pf = pareto_frontier(team_stats)
p_x, p_y = zip(*pf) if pf else ([], [])

# -------------------------------------------------------
# FIGURE SETUP
# -------------------------------------------------------
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "text.color":   TEXT_CLR,
    "axes.labelcolor": TEXT_CLR,
    "xtick.color":  TEXT_CLR,
    "ytick.color":  TEXT_CLR,
})

fig, ax = plt.subplots(figsize=(15, 12))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

x, y = team_stats["dci"].values, team_stats["dis"].values
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
px = (x_max - x_min) * 0.13
py = (y_max - y_min) * 0.13

# -------------------------------------------------------
# A. ISOQUANT CONTOURS
# -------------------------------------------------------
xi = np.linspace(x_min - px, x_max + px, 200)
yi = np.linspace(y_min - py, y_max + py, 200)
Xg, Yg = np.meshgrid(xi, yi)
Z = WEIGHT_DCI * (Xg - mu_dci) / sig_dci + WEIGHT_DIS * (Yg - mu_dis) / sig_dis
levels = np.linspace(Z.min(), Z.max(), 9)

ct = ax.contour(Xg, Yg, Z, levels=levels, colors="#8888aa", alpha=0.18,
                linestyles="dashed", linewidths=0.7, zorder=1)

# -------------------------------------------------------
# B. ELITE-TIER CONVEX HULL
# -------------------------------------------------------
top_pts = team_stats.head(TOP_TIER_HULL)[["dci", "dis"]].values
if len(top_pts) >= 3:
    hull = ConvexHull(top_pts)
    verts = top_pts[hull.vertices]
    verts = np.vstack([verts, verts[0]])
    ax.fill(verts[:, 0], verts[:, 1], color=GOLD, alpha=0.07, zorder=0)
    ax.plot(verts[:, 0], verts[:, 1], color=GOLD, alpha=0.4,
            linestyle="--", linewidth=1.8, zorder=2)

# -------------------------------------------------------
# C. SCATTER
# -------------------------------------------------------
sizes   = (team_stats["n"] / team_stats["n"].max()) * 420 + 80
colors  = team_stats["elite"].values

sc = ax.scatter(x, y, c=colors, s=sizes, cmap="plasma",
                alpha=0.88, edgecolors="#333333", linewidth=0.8,
                zorder=5, vmin=colors.min(), vmax=colors.max())

cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75)
cbar.set_label("Composite Elite Score (Z)", rotation=270, labelpad=20,
               fontsize=11, fontweight="bold", color=TEXT_CLR)
cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_CLR)

# -------------------------------------------------------
# D. PARETO FRONTIER LINE
# -------------------------------------------------------
if p_x:
    ax.plot(p_x, p_y, color=ACCENT, linewidth=2.8, alpha=0.75,
            zorder=4, label="Pareto Frontier", solid_capstyle="round")

# -------------------------------------------------------
# E. QUADRANT DIVIDERS + LABELS
# -------------------------------------------------------
ax.axvline(mu_dci, color="#555566", linestyle=":", alpha=0.5, linewidth=1.2, zorder=2)
ax.axhline(mu_dis, color="#555566", linestyle=":", alpha=0.5, linewidth=1.2, zorder=2)

quad_kw = dict(fontsize=10, alpha=0.35, color=TEXT_CLR, style="italic", zorder=3)
ax.text(x_max + px * 0.05, y_max - py * 0.1,  "Tight &\nDisciplined", ha="right", va="top",    **quad_kw)
ax.text(x_min - px * 0.05, y_max - py * 0.1,  "Soft but\nDisciplined", ha="left",  va="top",    **quad_kw)
ax.text(x_max + px * 0.05, y_min + py * 0.1,  "Tight but\nChaotic",   ha="right", va="bottom", **quad_kw)
ax.text(x_min - px * 0.05, y_min + py * 0.1,  "Soft &\nChaotic",      ha="left",  va="bottom", **quad_kw)

# -------------------------------------------------------
# F. TEAM LABELS (top N)
# -------------------------------------------------------
top_df = team_stats.head(TOP_N_LABELS)
offset_cycle = [(14, 14), (14, -16), (-14, 14), (-14, -16),
                (0, 22),  (0, -22),  (20, 5),   (-20, 5),
                (16, -10),(-16, -10),(10, 20),   (-10, 20)]

for idx, row in top_df.iterrows():
    dx, dy = offset_cycle[idx % len(offset_cycle)]
    lbl = f"{row['defensive_team']}\n#{int(row['rank'])}"
    ax.annotate(
        lbl,
        (row["dci"], row["dis"]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=9.5, fontweight="semibold", color="#ffffff", zorder=10,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#111111", ec=GOLD,
                  alpha=0.88, linewidth=0.9),
        arrowprops=dict(arrowstyle="-", color="#888888", alpha=0.4, lw=0.9),
    )

# Also label bottom-tier outliers for context
bot_df = team_stats.tail(3)
for idx2, row in bot_df.iterrows():
    ax.annotate(
        row["defensive_team"],
        (row["dci"], row["dis"]),
        xytext=(0, -18), textcoords="offset points",
        fontsize=8.5, color="#aaaaaa", ha="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#111111", ec="#555555",
                  alpha=0.75, linewidth=0.7),
        arrowprops=dict(arrowstyle="-", color="#555555", alpha=0.3, lw=0.7),
    )

# -------------------------------------------------------
# G. TITLES / LABELS / LEGEND
# -------------------------------------------------------
ax.set_title(
    "Defensive Landscape: The Elite Frontier",
    fontsize=22, fontweight="bold", color=TEXT_CLR, pad=18,
    path_effects=[pe.withStroke(linewidth=3, foreground=DARK_BG)],
)
ax.set_xlabel("Defensive Coverage Index (DCI)  →  Higher = Tighter Coverage",
              fontsize=13, fontweight="bold", color=TEXT_CLR, labelpad=10)
ax.set_ylabel("Defensive Integrity Score (DIS)  →  Higher = More Disciplined",
              fontsize=13, fontweight="bold", color=TEXT_CLR, labelpad=10)

legend_txt = (
    "● Size: Sample size (plays)\n"
    "-- Dashed: Efficiency Isoquants\n"
    "━ Red: Pareto Frontier\n"
    "◆ Gold: Elite Tier Envelope"
)
ax.text(x_min - px * 0.05, y_max + py * 0.92, legend_txt,
        fontsize=9, va="top", color="#aaaaaa",
        bbox=dict(boxstyle="round", fc="#1a1a1a", ec="#333333", alpha=0.9))

ax.legend(loc="lower right", framealpha=0.25, edgecolor="#444444",
          labelcolor=TEXT_CLR, fontsize=10)

# -------------------------------------------------------
# H. GRID / SPINES
# -------------------------------------------------------
ax.grid(True, linestyle="--", alpha=0.12, color=GRID_CLR)
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

ax.set_xlim(x_min - px, x_max + px * 1.3)
ax.set_ylim(y_min - py, y_max + py * 1.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
print(f"[SUCCESS] Plot saved → {OUT_PATH}")
