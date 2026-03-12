#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Case Study Tables for Paper
============================
Generates paper-ready tables from play-level metrics data for specific
case study games (the same ones used in animations).

Outputs LaTeX and console tables for:
  1. Per-play DCI/DIS breakdown for case study games
  2. Per-team aggregated stats for case study matchups
  3. Highlighted key plays with contextual info (down, distance, formation, result)

Inputs:
  - metrics_playlevel_supervised.csv
  - supplementary_data.csv
"""

import os
import pandas as pd
import numpy as np

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "metrics_playlevel_supervised.csv")
SUPP_PATH = os.path.join(BASE_DIR, "supplementary_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "paper_tables")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# CASE STUDY GAMES (same as animation scripts)
# -------------------------------------------------------
CASE_STUDIES = {
    "ravens_49ers_christmas": {
        "game_id": 2023122502,
        "title": "Baltimore Ravens @ San Francisco 49ers (Dec 25, 2023)",
        "defense_team": "SF",
        "offense_team": "BAL",
    },
    "vikings_49ers": {
        "game_id": 2023102200,
        "title": "Minnesota Vikings @ San Francisco 49ers (Oct 23, 2023)",
        "defense_team": "SF",
        "offense_team": "MIN",
    },
    "seahawks_giants": {
        "game_id": 2023100200,
        "title": "Seattle Seahawks @ New York Giants (Oct 2, 2023)",
        "defense_team": "NYG",
        "offense_team": "SEA",
    },
}

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------
print("[INFO] Loading metrics...")
df_m = pd.read_csv(METRICS_PATH)

print("[INFO] Loading supplementary data...")
supp = pd.read_csv(SUPP_PATH, low_memory=False)

# Drop pass_result from supp to avoid _x/_y conflict (metrics CSV already has it)
supp_cols = [c for c in supp.columns if c != "pass_result"]
merged = df_m.merge(supp[supp_cols], on=["game_id", "play_id"], how="inner")
print(f"[INFO] Merged dataset: {len(merged)} plays")

# -------------------------------------------------------
# LABEL HELPERS
# -------------------------------------------------------
RESULT_MAP = {"C": "Complete", "I": "Incomplete", "S": "Sack", "IN": "Interception"}


def dci_quality(v):
    if v >= 0.40:
        return "Tight"
    if v >= 0.30:
        return "Moderate"
    return "Loose"


def dis_quality(v):
    if v >= 0.60:
        return "Disciplined"
    if v >= 0.30:
        return "Mixed"
    return "Chaotic"


# -------------------------------------------------------
# GENERATE TABLES PER CASE STUDY
# -------------------------------------------------------
all_case_rows = []

for key, info in CASE_STUDIES.items():
    gid = info["game_id"]
    game_df = merged[merged["game_id"] == gid].copy()

    if game_df.empty:
        # Try fuzzy match on game_id (some datasets encode differently)
        close = merged[merged["game_id"].astype(str).str.startswith(str(gid)[:8])]
        if not close.empty:
            game_df = close.copy()
            gid = game_df["game_id"].iloc[0]
            print(f"[WARN] Exact game_id {info['game_id']} not found, using {gid}")

    if game_df.empty:
        print(f"[WARN] No data for {info['title']} (game_id={gid}). Skipping.")
        continue

    print(f"\n{'='*70}")
    print(f"CASE STUDY: {info['title']}")
    print(f"{'='*70}")
    print(f"Plays found: {len(game_df)}")

    # Sort by play_id (chronological)
    game_df = game_df.sort_values("play_id")

    # Build table columns
    cols_for_table = []
    for _, row in game_df.iterrows():
        cols_for_table.append({
            "Play": int(row["play_id"]),
            "Qtr": int(row["quarter"]) if pd.notna(row.get("quarter")) else "-",
            "Down": int(row["down"]) if pd.notna(row.get("down")) else "-",
            "Dist": int(row["yards_to_go"]) if pd.notna(row.get("yards_to_go")) else "-",
            "Formation": row.get("offense_formation", "-") if pd.notna(row.get("offense_formation")) else "-",
            "Result": RESULT_MAP.get(row["pass_result"], row["pass_result"]),
            "Yards": int(row["yards_gained"]) if pd.notna(row.get("yards_gained")) else "-",
            "EPA": f"{row['epa']:.2f}" if pd.notna(row.get("epa")) else "-",
            "DCI": f"{row['dci_supervised']:.3f}",
            "DIS": f"{row['dis_final']:.3f}",
            "DCI Rating": dci_quality(row["dci_supervised"]),
            "DIS Rating": dis_quality(row["dis_final"]),
            "Coverage": row.get("team_coverage_type", "-") if pd.notna(row.get("team_coverage_type")) else "-",
        })

    table_df = pd.DataFrame(cols_for_table)

    # Print console table
    print(f"\n{table_df.to_string(index=False)}\n")

    # Summary stats
    print(f"  Avg DCI: {game_df['dci_supervised'].mean():.3f} ({dci_quality(game_df['dci_supervised'].mean())})")
    print(f"  Avg DIS: {game_df['dis_final'].mean():.3f} ({dis_quality(game_df['dis_final'].mean())})")
    print(f"  Avg EPA: {game_df['epa'].mean():.2f}")
    print(f"  Pass Results: {game_df['pass_result'].value_counts().to_dict()}")

    # Save CSV
    csv_out = os.path.join(OUT_DIR, f"case_study_{key}.csv")
    table_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # Save LaTeX
    latex_out = os.path.join(OUT_DIR, f"case_study_{key}.tex")
    # Select columns for LaTeX (compact)
    latex_cols = ["Play", "Qtr", "Down", "Dist", "Result", "Yards", "EPA", "DCI", "DIS", "Coverage"]
    latex_df = table_df[latex_cols]
    latex_str = latex_df.to_latex(index=False, column_format="r" * len(latex_cols), escape=True)
    with open(latex_out, "w") as f:
        f.write(f"% Case Study: {info['title']}\n")
        f.write(latex_str)
    print(f"  Saved: {latex_out}")

    # Collect per-game summary for cross-study comparison
    all_case_rows.append({
        "Game": info["title"],
        "Plays": len(game_df),
        "Avg DCI": f"{game_df['dci_supervised'].mean():.3f}",
        "Avg DIS": f"{game_df['dis_final'].mean():.3f}",
        "Avg EPA": f"{game_df['epa'].mean():.2f}",
        "Completions": int((game_df["pass_result"] == "C").sum()),
        "Incompletes": int((game_df["pass_result"] == "I").sum()),
        "Sacks": int((game_df["pass_result"] == "S").sum()),
        "INTs": int((game_df["pass_result"] == "IN").sum()),
    })

# -------------------------------------------------------
# CROSS-STUDY SUMMARY TABLE
# -------------------------------------------------------
if all_case_rows:
    print(f"\n{'='*70}")
    print("CROSS-STUDY COMPARISON")
    print(f"{'='*70}\n")
    summary_df = pd.DataFrame(all_case_rows)
    print(summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(OUT_DIR, "case_study_summary.csv"), index=False)
    summary_df.to_latex(os.path.join(OUT_DIR, "case_study_summary.tex"), index=False, escape=True)
    print(f"\nSaved summary to: {OUT_DIR}/case_study_summary.*")

# -------------------------------------------------------
# HIGHLIGHTED KEY PLAYS TABLE
# (Extreme DCI/DIS values — notable breakdowns or lockdowns)
# -------------------------------------------------------
print(f"\n{'='*70}")
print("KEY PLAYS: Top 5 Defensive Lockdowns & Top 5 Breakdowns")
print(f"{'='*70}\n")

# Filter to case study games only
case_game_ids = [info["game_id"] for info in CASE_STUDIES.values()]
case_plays = merged[merged["game_id"].isin(case_game_ids)].copy()

if not case_plays.empty:
    # Composite defensive quality
    case_plays["def_quality"] = (
        (case_plays["dci_supervised"] - case_plays["dci_supervised"].mean()) / case_plays["dci_supervised"].std()
        + (case_plays["dis_final"] - case_plays["dis_final"].mean()) / case_plays["dis_final"].std()
    ) / 2.0

    # Best defensive plays
    top_def = case_plays.nlargest(5, "def_quality")
    print("TOP 5 DEFENSIVE LOCKDOWNS:")
    for _, r in top_def.iterrows():
        desc = r.get("play_description", "")
        desc_short = (desc[:80] + "...") if isinstance(desc, str) and len(desc) > 80 else desc
        print(f"  Game {int(r['game_id'])} Play {int(r['play_id'])}: "
              f"DCI={r['dci_supervised']:.3f} DIS={r['dis_final']:.3f} "
              f"EPA={r['epa']:.2f} Result={r['pass_result']}")
        if isinstance(desc_short, str):
            print(f"    {desc_short}")

    print()

    # Worst defensive plays (breakdowns)
    bot_def = case_plays.nsmallest(5, "def_quality")
    print("TOP 5 DEFENSIVE BREAKDOWNS:")
    for _, r in bot_def.iterrows():
        desc = r.get("play_description", "")
        desc_short = (desc[:80] + "...") if isinstance(desc, str) and len(desc) > 80 else desc
        print(f"  Game {int(r['game_id'])} Play {int(r['play_id'])}: "
              f"DCI={r['dci_supervised']:.3f} DIS={r['dis_final']:.3f} "
              f"EPA={r['epa']:.2f} Result={r['pass_result']}")
        if isinstance(desc_short, str):
            print(f"    {desc_short}")

    # Save key plays
    key_cols = ["game_id", "play_id", "dci_supervised", "dis_final", "epa", "pass_result",
                "yards_gained", "down", "yards_to_go", "offense_formation", "team_coverage_type"]
    key_cols_available = [c for c in key_cols if c in case_plays.columns]
    key_plays = pd.concat([top_def[key_cols_available], bot_def[key_cols_available]])
    key_plays.to_csv(os.path.join(OUT_DIR, "key_plays_highlights.csv"), index=False)
    print(f"\nSaved key plays to: {OUT_DIR}/key_plays_highlights.csv")

print(f"\n[DONE] All case study tables generated in: {OUT_DIR}/")
