#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regenerate All Paper Figures (Local Data)
==========================================
A unified script that generates all publication-ready figures from local
CSV/parquet data. Replaces the need to run individual scripts that
reference HPC paths.

Figures generated:
  1. Defensive Landscape: Elite Frontier (DCI vs DIS scatter)
  2. EPA Correlation: DCI/DIS vs EPA regression plots
  3. Validation Boxplots: DCI/DIS by pass outcome
  4. Explosive Play Analysis: Probability by DCI quartile

Inputs:
  - metrics_playlevel_supervised.csv
  - supplementary_data.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy import stats

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "metrics_playlevel_supervised.csv")
SUPP_PATH = os.path.join(BASE_DIR, "supplementary_data.csv")
IMG_DIR = os.path.join(BASE_DIR, "imgs")
os.makedirs(IMG_DIR, exist_ok=True)

# -------------------------------------------------------
# STYLE CONFIGURATION
# -------------------------------------------------------
DARK_BG = "#0d0d0d"
GOLD = "#FFD700"
ACCENT = "#E63946"
TEXT_CLR = "#F0F0F0"
GRID_CLR = "#2e2e2e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_CLR,
    "axes.labelcolor": TEXT_CLR,
    "xtick.color": TEXT_CLR,
    "ytick.color": TEXT_CLR,
})

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
print("[INFO] Loading data...")
df_m = pd.read_csv(METRICS_PATH)
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Merge for team-level analysis
merged = df_m.merge(
    supp[["game_id", "play_id", "defensive_team", "yards_gained", "down",
          "yards_to_go", "defenders_in_the_box", "offense_formation",
          "team_coverage_type"]].drop_duplicates(),
    on=["game_id", "play_id"], how="inner"
)
print(f"[INFO] Loaded {len(df_m)} metrics, merged {len(merged)} plays with context.")


# =======================================================
# FIGURE 1: DEFENSIVE LANDSCAPE — ELITE FRONTIER
# =======================================================
def generate_defensive_landscape():
    print("\n[FIG 1] Generating Defensive Landscape...")

    WEIGHT_DCI = 0.5
    WEIGHT_DIS = 0.5
    TOP_N_LABELS = 12
    TOP_TIER_HULL = 8

    team_stats = (
        merged.groupby("defensive_team")
        .agg(dci=("dci_supervised", "mean"), dis=("dis_final", "mean"), n=("play_id", "nunique"))
        .reset_index()
    )

    mu_dci, sig_dci = team_stats["dci"].mean(), team_stats["dci"].std(ddof=0) or 1.0
    mu_dis, sig_dis = team_stats["dis"].mean(), team_stats["dis"].std(ddof=0) or 1.0

    team_stats["dci_z"] = (team_stats["dci"] - mu_dci) / sig_dci
    team_stats["dis_z"] = (team_stats["dis"] - mu_dis) / sig_dis
    team_stats["elite"] = WEIGHT_DCI * team_stats["dci_z"] + WEIGHT_DIS * team_stats["dis_z"]
    team_stats = team_stats.sort_values("elite", ascending=False).reset_index(drop=True)
    team_stats["rank"] = team_stats.index + 1

    # Pareto frontier
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

    fig, ax = plt.subplots(figsize=(15, 12))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    x, y = team_stats["dci"].values, team_stats["dis"].values
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    px = (x_max - x_min) * 0.13
    py = (y_max - y_min) * 0.13

    # Isoquant contours
    xi = np.linspace(x_min - px, x_max + px, 200)
    yi = np.linspace(y_min - py, y_max + py, 200)
    Xg, Yg = np.meshgrid(xi, yi)
    Z = WEIGHT_DCI * (Xg - mu_dci) / sig_dci + WEIGHT_DIS * (Yg - mu_dis) / sig_dis
    levels = np.linspace(Z.min(), Z.max(), 9)
    ax.contour(Xg, Yg, Z, levels=levels, colors="#8888aa", alpha=0.18,
               linestyles="dashed", linewidths=0.7, zorder=1)

    # Elite-tier convex hull
    top_pts = team_stats.head(TOP_TIER_HULL)[["dci", "dis"]].values
    if len(top_pts) >= 3:
        hull = ConvexHull(top_pts)
        verts = top_pts[hull.vertices]
        verts = np.vstack([verts, verts[0]])
        ax.fill(verts[:, 0], verts[:, 1], color=GOLD, alpha=0.07, zorder=0)
        ax.plot(verts[:, 0], verts[:, 1], color=GOLD, alpha=0.4,
                linestyle="--", linewidth=1.8, zorder=2)

    # Scatter
    sizes = (team_stats["n"] / team_stats["n"].max()) * 420 + 80
    colors = team_stats["elite"].values
    sc = ax.scatter(x, y, c=colors, s=sizes, cmap="plasma",
                    alpha=0.88, edgecolors="#333333", linewidth=0.8,
                    zorder=5, vmin=colors.min(), vmax=colors.max())
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.75)
    cbar.set_label("Composite Elite Score (Z)", rotation=270, labelpad=20,
                   fontsize=11, fontweight="bold", color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_CLR)

    # Pareto frontier
    if p_x:
        ax.plot(p_x, p_y, color=ACCENT, linewidth=2.8, alpha=0.75,
                zorder=4, label="Pareto Frontier", solid_capstyle="round")

    # Quadrant dividers
    ax.axvline(mu_dci, color="#555566", linestyle=":", alpha=0.5, linewidth=1.2, zorder=2)
    ax.axhline(mu_dis, color="#555566", linestyle=":", alpha=0.5, linewidth=1.2, zorder=2)

    quad_kw = dict(fontsize=10, alpha=0.35, color=TEXT_CLR, style="italic", zorder=3)
    ax.text(x_max + px * 0.05, y_max - py * 0.1, "Tight &\nDisciplined", ha="right", va="top", **quad_kw)
    ax.text(x_min - px * 0.05, y_max - py * 0.1, "Soft but\nDisciplined", ha="left", va="top", **quad_kw)
    ax.text(x_max + px * 0.05, y_min + py * 0.1, "Tight but\nChaotic", ha="right", va="bottom", **quad_kw)
    ax.text(x_min - px * 0.05, y_min + py * 0.1, "Soft &\nChaotic", ha="left", va="bottom", **quad_kw)

    # Team labels
    top_df = team_stats.head(TOP_N_LABELS)
    offset_cycle = [(14, 14), (14, -16), (-14, 14), (-14, -16),
                    (0, 22), (0, -22), (20, 5), (-20, 5),
                    (16, -10), (-16, -10), (10, 20), (-10, 20)]
    for idx, row in top_df.iterrows():
        dx, dy = offset_cycle[idx % len(offset_cycle)]
        lbl = f"{row['defensive_team']}\n#{int(row['rank'])}"
        ax.annotate(lbl, (row["dci"], row["dis"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, fontweight="semibold", color="#ffffff", zorder=10, ha="center",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#111111", ec=GOLD, alpha=0.88, linewidth=0.9),
                    arrowprops=dict(arrowstyle="-", color="#888888", alpha=0.4, lw=0.9))

    bot_df = team_stats.tail(3)
    for _, row in bot_df.iterrows():
        ax.annotate(row["defensive_team"], (row["dci"], row["dis"]),
                    xytext=(0, -18), textcoords="offset points",
                    fontsize=8.5, color="#aaaaaa", ha="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111111", ec="#555555", alpha=0.75, linewidth=0.7),
                    arrowprops=dict(arrowstyle="-", color="#555555", alpha=0.3, lw=0.7))

    ax.set_title("Defensive Landscape: The Elite Frontier",
                 fontsize=22, fontweight="bold", color=TEXT_CLR, pad=18,
                 path_effects=[pe.withStroke(linewidth=3, foreground=DARK_BG)])
    ax.set_xlabel("Defensive Coverage Index (DCI)  \u2192  Higher = Tighter Coverage",
                  fontsize=13, fontweight="bold", color=TEXT_CLR, labelpad=10)
    ax.set_ylabel("Defensive Integrity Score (DIS)  \u2192  Higher = More Disciplined",
                  fontsize=13, fontweight="bold", color=TEXT_CLR, labelpad=10)

    legend_txt = (
        "\u25cf Size: Sample size (plays)\n"
        "-- Dashed: Efficiency Isoquants\n"
        "\u2501 Red: Pareto Frontier\n"
        "\u25c6 Gold: Elite Tier Envelope"
    )
    ax.text(x_min - px * 0.05, y_max + py * 0.92, legend_txt,
            fontsize=9, va="top", color="#aaaaaa",
            bbox=dict(boxstyle="round", fc="#1a1a1a", ec="#333333", alpha=0.9))
    ax.legend(loc="lower right", framealpha=0.25, edgecolor="#444444",
              labelcolor=TEXT_CLR, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.12, color=GRID_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_xlim(x_min - px, x_max + px * 1.3)
    ax.set_ylim(y_min - py, y_max + py * 1.3)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "def_elite_new.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  -> Saved: {out}")

    # Also save team stats CSV
    team_stats.to_csv(os.path.join(IMG_DIR, "def_elite_actual_team_stats.csv"), index=False)


# =======================================================
# FIGURE 2: EPA CORRELATION REGRESSION PLOTS
# =======================================================
def generate_epa_correlation():
    print("\n[FIG 2] Generating EPA Correlation Plots...")

    df_clean = df_m.dropna(subset=["dci_supervised", "dis_final", "epa"]).copy()
    df_clean = df_clean[(df_clean["epa"] > -5) & (df_clean["epa"] < 5)]

    def plot_regression(ax, x_data, y_data, color_scatter, color_line, title, xlabel):
        ax.scatter(x_data, y_data, alpha=0.15, c=color_scatter, s=15, edgecolors="none")
        slope, intercept = np.polyfit(x_data, y_data, 1)
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_pred = slope * x_range + intercept
        ax.plot(x_range, y_pred, color=color_line, linewidth=3, linestyle="--",
                label=f"Trend (Slope: {slope:.3f})")
        correlation = x_data.corr(y_data)
        ax.set_title(f"{title}\nCorrelation (r): {correlation:.3f}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Offensive EPA (Expected Points Added)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    plot_regression(axes[0], df_clean["dci_supervised"], df_clean["epa"],
                    "#4c72b0", "darkblue", "Defensive Coverage (DCI) Impact on EPA",
                    "DCI Score (Higher = Tighter Coverage)")
    plot_regression(axes[1], df_clean["dis_final"], df_clean["epa"],
                    "#c44e52", "darkred", "Defensive Integrity (DIS) Impact on EPA",
                    "DIS Score (Higher = Better Integrity)")

    plt.suptitle("Statistical Validation: Does Better Defense Lower Offensive EPA?", fontsize=18, y=1.02)
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "epa_correlation_regplot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")


# =======================================================
# FIGURE 3: VALIDATION BOXPLOTS
# =======================================================
def generate_validation_boxplots():
    print("\n[FIG 3] Generating Validation Boxplots...")

    sns.set_theme(style="whitegrid", context="talk")

    target_outcomes = ["C", "I", "S", "IN"]
    df_clean = df_m[df_m["pass_result"].isin(target_outcomes)].copy()
    label_map = {"C": "Complete", "I": "Incomplete", "S": "Sack", "IN": "Interception"}
    df_clean["Outcome"] = df_clean["pass_result"].map(label_map)
    order = ["Complete", "Incomplete", "Sack", "Interception"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    sns.boxplot(data=df_clean, x="Outcome", y="dci_supervised", order=order,
                palette="Blues_d", ax=axes[0], showfliers=False)
    axes[0].set_title("Defensive Coverage Index (DCI) by Outcome", fontweight="bold", pad=15)
    axes[0].set_ylabel("DCI Score (Higher = Tighter Coverage)")
    axes[0].set_xlabel("")

    sns.boxplot(data=df_clean, x="Outcome", y="dis_final", order=order,
                palette="Reds_d", ax=axes[1], showfliers=False)
    axes[1].set_title("Defensive Integrity Score (DIS) by Outcome", fontweight="bold", pad=15)
    axes[1].set_ylabel("DIS Score (Higher = Better Integrity)")
    axes[1].set_xlabel("")

    plt.suptitle("Validation of Defensive Metrics Against Play Outcomes", fontsize=20, y=1.02)
    plt.tight_layout()
    out = os.path.join(IMG_DIR, "metric_validation_boxplot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")

    # Reset style for subsequent plots
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": TEXT_CLR,
        "axes.labelcolor": TEXT_CLR,
        "xtick.color": TEXT_CLR,
        "ytick.color": TEXT_CLR,
    })


# =======================================================
# FIGURE 4: EXPLOSIVE PLAY REDUCTION
# =======================================================
def generate_explosive_play_analysis():
    print("\n[FIG 4] Generating Explosive Play Analysis...")

    df_analysis = merged.dropna(subset=["dci_supervised", "dis_final", "epa", "yards_to_go"]).copy()
    df_analysis["is_explosive"] = (df_analysis["epa"] >= 2.0).astype(int)

    # DCI quartiles
    df_analysis["dci_quartile"] = pd.qcut(
        df_analysis["dci_supervised"], 4,
        labels=["Q1 (Loose)", "Q2", "Q3", "Q4 (Tight)"]
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="dci_quartile", y="is_explosive", data=df_analysis,
                palette="Blues", errorbar=("ci", 95), ax=ax)
    ax.set_title("Probability of Explosive Play (EPA >= 2.0) by Coverage Quality",
                 fontweight="bold", pad=15)
    ax.set_ylabel("Explosive Play Probability")
    ax.set_xlabel("Defensive Coverage Index (DCI Quartiles)")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "explosive_play_reduction.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out}")

    # Also do DIS quartile version
    df_analysis["dis_quartile"] = pd.qcut(
        df_analysis["dis_final"], 4,
        labels=["Q1 (Chaotic)", "Q2", "Q3", "Q4 (Disciplined)"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="dis_quartile", y="is_explosive", data=df_analysis,
                palette="Reds", errorbar=("ci", 95), ax=ax)
    ax.set_title("Probability of Explosive Play (EPA >= 2.0) by Structural Integrity",
                 fontweight="bold", pad=15)
    ax.set_ylabel("Explosive Play Probability")
    ax.set_xlabel("Defensive Integrity Score (DIS Quartiles)")

    plt.tight_layout()
    out_dis = os.path.join(IMG_DIR, "explosive_play_reduction_dis.png")
    plt.savefig(out_dis, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_dis}")


# =======================================================
# FIGURE 5: DCI vs DIS SCATTER (play-level density)
# =======================================================
def generate_dci_dis_density():
    print("\n[FIG 5] Generating DCI vs DIS Density Plot...")

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    df_clean = df_m.dropna(subset=["dci_supervised", "dis_final", "epa"]).copy()

    sc = ax.scatter(
        df_clean["dci_supervised"], df_clean["dis_final"],
        c=df_clean["epa"], cmap="RdYlGn_r", s=8, alpha=0.35,
        edgecolors="none", vmin=-3, vmax=3
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("EPA (Red = High Offense, Green = Strong Defense)",
                   rotation=270, labelpad=20, fontsize=11, color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_CLR)

    ax.set_xlabel("DCI (Coverage Tightness)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("DIS (Structural Integrity)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("Play-Level DCI vs DIS Colored by EPA",
                 fontsize=18, fontweight="bold", color=TEXT_CLR, pad=15,
                 path_effects=[pe.withStroke(linewidth=3, foreground=DARK_BG)])
    ax.grid(True, alpha=0.12, color=GRID_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "dci_dis_scatter_epa.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  -> Saved: {out}")


# =======================================================
# RUN ALL
# =======================================================
if __name__ == "__main__":
    print(f"{'='*60}")
    print("REGENERATING ALL PAPER FIGURES")
    print(f"{'='*60}")

    generate_defensive_landscape()
    generate_epa_correlation()
    generate_validation_boxplots()
    generate_explosive_play_analysis()
    generate_dci_dis_density()

    print(f"\n{'='*60}")
    print(f"ALL FIGURES SAVED TO: {IMG_DIR}/")
    print(f"{'='*60}")
    print("Files generated:")
    for f in sorted(os.listdir(IMG_DIR)):
        if f.endswith(".png"):
            fpath = os.path.join(IMG_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")
