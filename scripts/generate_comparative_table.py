#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparative Metrics Table: Yards vs DCI vs DIS
===============================================
Generates a publication-ready comparative analysis showing that
DCI/DIS provide information beyond simple yardage metrics.

Outputs:
  - Console tables
  - CSV / LaTeX files
  - Comparative figure (correlation heatmap + bar chart)

Inputs:
  - metrics_playlevel_supervised.csv
  - supplementary_data.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "metrics_playlevel_supervised.csv")
SUPP_PATH = os.path.join(BASE_DIR, "supplementary_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "paper_tables")
IMG_DIR = os.path.join(BASE_DIR, "imgs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
print("[INFO] Loading data...")
df_m = pd.read_csv(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Drop pass_result from supp to avoid _x/_y conflict (metrics CSV already has it)
supp_cols = [c for c in supp.columns if c != "pass_result"]
merged = df_m.merge(supp[supp_cols], on=["game_id", "play_id"], how="inner")
merged = merged.dropna(subset=["dci_supervised", "dis_final", "epa", "yards_gained"])
print(f"[INFO] Analysis dataset: {len(merged)} plays")

RESULT_MAP = {"C": "Complete", "I": "Incomplete", "S": "Sack", "IN": "Interception"}
merged["outcome"] = merged["pass_result"].map(RESULT_MAP)

# -------------------------------------------------------
# 1. PLAY-LEVEL CORRELATION ANALYSIS
# -------------------------------------------------------
print(f"\n{'='*70}")
print("1. PLAY-LEVEL CORRELATIONS WITH EPA")
print(f"{'='*70}\n")

metrics_to_compare = {
    "yards_gained": "Yards Gained",
    "dci_supervised": "DCI (Coverage Index)",
    "dis_final": "DIS (Integrity Score)",
}

corr_rows = []
for col, label in metrics_to_compare.items():
    r, p = stats.pearsonr(merged[col], merged["epa"])
    rho, p_sp = stats.spearmanr(merged[col], merged["epa"])
    corr_rows.append({
        "Metric": label,
        "Pearson r": f"{r:.4f}",
        "p-value (Pearson)": f"{p:.2e}",
        "Spearman rho": f"{rho:.4f}",
        "p-value (Spearman)": f"{p_sp:.2e}",
    })

corr_df = pd.DataFrame(corr_rows)
print(corr_df.to_string(index=False))
corr_df.to_csv(os.path.join(OUT_DIR, "correlation_comparison.csv"), index=False)
corr_df.to_latex(os.path.join(OUT_DIR, "correlation_comparison.tex"), index=False, escape=True)

# -------------------------------------------------------
# 2. PREDICTIVE POWER: DEFENSIVE SUCCESS
# -------------------------------------------------------
print(f"\n{'='*70}")
print("2. PREDICTIVE POWER FOR DEFENSIVE SUCCESS (EPA <= 0)")
print(f"{'='*70}\n")

merged["def_success"] = (merged["epa"] <= 0).astype(int)

pred_rows = []
for col, label in metrics_to_compare.items():
    # Point-biserial correlation
    r_pb, p_pb = stats.pointbiserialr(merged["def_success"], merged[col])

    # Mean difference
    success = merged[merged["def_success"] == 1][col].mean()
    failure = merged[merged["def_success"] == 0][col].mean()
    diff = success - failure

    # Cohen's d
    pooled_std = np.sqrt(
        (merged[merged["def_success"] == 1][col].var()
         + merged[merged["def_success"] == 0][col].var()) / 2
    )
    cohens_d = diff / pooled_std if pooled_std > 0 else 0

    pred_rows.append({
        "Metric": label,
        "Mean (Success)": f"{success:.3f}",
        "Mean (Failure)": f"{failure:.3f}",
        "Difference": f"{diff:+.3f}",
        "Cohen's d": f"{cohens_d:.3f}",
        "Point-Biserial r": f"{r_pb:.4f}",
        "p-value": f"{p_pb:.2e}",
    })

pred_df = pd.DataFrame(pred_rows)
print(pred_df.to_string(index=False))
pred_df.to_csv(os.path.join(OUT_DIR, "predictive_power.csv"), index=False)
pred_df.to_latex(os.path.join(OUT_DIR, "predictive_power.tex"), index=False, escape=True)

# -------------------------------------------------------
# 3. BY-OUTCOME BREAKDOWN
# -------------------------------------------------------
print(f"\n{'='*70}")
print("3. METRIC MEANS BY PASS OUTCOME")
print(f"{'='*70}\n")

outcome_order = ["Complete", "Incomplete", "Sack", "Interception"]
outcome_df = merged[merged["outcome"].isin(outcome_order)]

breakdown_rows = []
for outcome in outcome_order:
    subset = outcome_df[outcome_df["outcome"] == outcome]
    breakdown_rows.append({
        "Outcome": outcome,
        "N": len(subset),
        "Avg Yards": f"{subset['yards_gained'].mean():.1f}",
        "Avg DCI": f"{subset['dci_supervised'].mean():.3f}",
        "Avg DIS": f"{subset['dis_final'].mean():.3f}",
        "Avg EPA": f"{subset['epa'].mean():.2f}",
        "Std Yards": f"{subset['yards_gained'].std():.1f}",
        "Std DCI": f"{subset['dci_supervised'].std():.3f}",
        "Std DIS": f"{subset['dis_final'].std():.3f}",
    })

breakdown_df = pd.DataFrame(breakdown_rows)
print(breakdown_df.to_string(index=False))
breakdown_df.to_csv(os.path.join(OUT_DIR, "outcome_breakdown.csv"), index=False)
breakdown_df.to_latex(os.path.join(OUT_DIR, "outcome_breakdown.tex"), index=False, escape=True)

# -------------------------------------------------------
# 4. TEAM-LEVEL: YARDS ALLOWED vs DCI/DIS RANK
# -------------------------------------------------------
print(f"\n{'='*70}")
print("4. TEAM-LEVEL: YARDS ALLOWED vs DCI/DIS RANKINGS")
print(f"{'='*70}\n")

team_stats = (
    merged.groupby("defensive_team")
    .agg(
        plays=("play_id", "count"),
        avg_yards_allowed=("yards_gained", "mean"),
        avg_dci=("dci_supervised", "mean"),
        avg_dis=("dis_final", "mean"),
        avg_epa=("epa", "mean"),
        explosive_rate=("epa", lambda x: (x >= 2.0).mean()),
    )
    .reset_index()
)

# Rankings (lower yards/EPA = better defense; higher DCI/DIS = better)
team_stats["rank_yards"] = team_stats["avg_yards_allowed"].rank(ascending=True).astype(int)
team_stats["rank_dci"] = team_stats["avg_dci"].rank(ascending=False).astype(int)
team_stats["rank_dis"] = team_stats["avg_dis"].rank(ascending=False).astype(int)
team_stats["rank_epa"] = team_stats["avg_epa"].rank(ascending=True).astype(int)

# Composite rank
team_stats["composite_rank"] = (
    (team_stats["rank_yards"] + team_stats["rank_dci"] + team_stats["rank_dis"]) / 3.0
)
team_stats = team_stats.sort_values("composite_rank")

display_cols = [
    "defensive_team", "plays", "avg_yards_allowed", "avg_dci", "avg_dis",
    "avg_epa", "rank_yards", "rank_dci", "rank_dis", "rank_epa"
]
team_display = team_stats[display_cols].copy()
team_display.columns = [
    "Team", "Plays", "Avg Yards", "Avg DCI", "Avg DIS",
    "Avg EPA", "Rank (Yards)", "Rank (DCI)", "Rank (DIS)", "Rank (EPA)"
]

# Format numeric columns
for c in ["Avg Yards", "Avg EPA"]:
    team_display[c] = team_display[c].map(lambda x: f"{x:.2f}")
for c in ["Avg DCI", "Avg DIS"]:
    team_display[c] = team_display[c].map(lambda x: f"{x:.3f}")

print(team_display.to_string(index=False))
team_display.to_csv(os.path.join(OUT_DIR, "team_rankings_comparison.csv"), index=False)
team_display.to_latex(os.path.join(OUT_DIR, "team_rankings_comparison.tex"), index=False, escape=True)

# -------------------------------------------------------
# 5. RANK CORRELATION: Do DCI/DIS rankings agree with yards?
# -------------------------------------------------------
print(f"\n{'='*70}")
print("5. RANK AGREEMENT (Spearman) BETWEEN METRICS")
print(f"{'='*70}\n")

rank_cols = {"rank_yards": "Yards Allowed", "rank_dci": "DCI", "rank_dis": "DIS", "rank_epa": "EPA"}
rank_matrix = []
for c1, l1 in rank_cols.items():
    row_data = {"Metric": l1}
    for c2, l2 in rank_cols.items():
        rho, _ = stats.spearmanr(team_stats[c1], team_stats[c2])
        row_data[l2] = f"{rho:.3f}"
    rank_matrix.append(row_data)

rank_df = pd.DataFrame(rank_matrix)
print(rank_df.to_string(index=False))
rank_df.to_csv(os.path.join(OUT_DIR, "rank_agreement_matrix.csv"), index=False)

# -------------------------------------------------------
# 6. FIGURE: Comparative visualization
# -------------------------------------------------------
print(f"\n[INFO] Generating comparative figure...")

DARK_BG = "#0d0d0d"
TEXT_CLR = "#F0F0F0"
GOLD = "#FFD700"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_CLR,
    "axes.labelcolor": TEXT_CLR,
    "xtick.color": TEXT_CLR,
    "ytick.color": TEXT_CLR,
})

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor(DARK_BG)

# Panel A: Correlation bars
ax = axes[0]
ax.set_facecolor(DARK_BG)
metric_labels = ["Yards\nGained", "DCI\n(Coverage)", "DIS\n(Integrity)"]
pearson_vals = [
    float(corr_rows[0]["Pearson r"]),
    float(corr_rows[1]["Pearson r"]),
    float(corr_rows[2]["Pearson r"]),
]
bar_colors = ["#4c72b0", "#2ca02c", "#d62728"]
bars = ax.bar(metric_labels, [abs(v) for v in pearson_vals], color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, pearson_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"r={val:.3f}", ha="center", va="bottom", fontsize=11, color=TEXT_CLR, fontweight="bold")
ax.set_ylabel("|Pearson r| with EPA", fontsize=12, fontweight="bold")
ax.set_title("Correlation Strength with EPA", fontsize=14, fontweight="bold", color=TEXT_CLR, pad=12)
ax.set_ylim(0, max(abs(v) for v in pearson_vals) * 1.25)
ax.grid(axis="y", alpha=0.15, color="#333")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")

# Panel B: Mean by outcome
ax2 = axes[1]
ax2.set_facecolor(DARK_BG)
x_pos = np.arange(len(outcome_order))
width = 0.25

# Extract raw means
yards_means = [outcome_df[outcome_df["outcome"] == o]["yards_gained"].mean() for o in outcome_order]
dci_means = [outcome_df[outcome_df["outcome"] == o]["dci_supervised"].mean() for o in outcome_order]
dis_means = [outcome_df[outcome_df["outcome"] == o]["dis_final"].mean() for o in outcome_order]

# Normalize all to 0-1 for comparison
def norm01(vals):
    mn, mx = min(vals), max(vals)
    return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

ax2.bar(x_pos - width, norm01(yards_means), width, label="Yards (norm)", color="#4c72b0", alpha=0.85)
ax2.bar(x_pos, norm01(dci_means), width, label="DCI (norm)", color="#2ca02c", alpha=0.85)
ax2.bar(x_pos + width, norm01(dis_means), width, label="DIS (norm)", color="#d62728", alpha=0.85)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(outcome_order, fontsize=10)
ax2.set_ylabel("Normalized Mean (0-1)", fontsize=12, fontweight="bold")
ax2.set_title("Metric Sensitivity by Outcome", fontsize=14, fontweight="bold", color=TEXT_CLR, pad=12)
ax2.legend(fontsize=9, loc="upper right", framealpha=0.3, edgecolor="#444", labelcolor=TEXT_CLR)
ax2.grid(axis="y", alpha=0.15, color="#333")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")

# Panel C: Rank disagreement scatter (team level)
ax3 = axes[2]
ax3.set_facecolor(DARK_BG)
sc = ax3.scatter(
    team_stats["rank_yards"], team_stats["rank_dci"],
    c=team_stats["rank_epa"], cmap="plasma", s=120,
    edgecolors="white", linewidth=0.8, alpha=0.9, zorder=5
)

# Perfect agreement line
ax3.plot([1, 32], [1, 32], "--", color="#555", alpha=0.5, linewidth=1.5, label="Perfect Agreement")

# Label outliers (rank diff > 10)
for _, row in team_stats.iterrows():
    diff = abs(row["rank_yards"] - row["rank_dci"])
    if diff >= 10:
        ax3.annotate(
            row["defensive_team"],
            (row["rank_yards"], row["rank_dci"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, color=GOLD, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111", ec=GOLD, alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="#888", alpha=0.5),
        )

cbar = fig.colorbar(sc, ax=ax3, pad=0.02, shrink=0.85)
cbar.set_label("EPA Rank", rotation=270, labelpad=15, fontsize=11, color=TEXT_CLR)
cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_CLR)

ax3.set_xlabel("Rank by Yards Allowed", fontsize=12, fontweight="bold")
ax3.set_ylabel("Rank by DCI", fontsize=12, fontweight="bold")
ax3.set_title("Yards vs DCI: Rank Disagreement", fontsize=14, fontweight="bold", color=TEXT_CLR, pad=12)
ax3.legend(fontsize=9, loc="upper left", framealpha=0.3, edgecolor="#444", labelcolor=TEXT_CLR)
ax3.grid(alpha=0.12, color="#333")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333")

plt.tight_layout()
fig_path = os.path.join(IMG_DIR, "yards_vs_dci_dis_comparison.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"[SUCCESS] Figure saved: {fig_path}")

print(f"\n[DONE] All comparative tables saved to: {OUT_DIR}/")
