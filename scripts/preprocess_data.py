#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 0 — Raw Data Preprocessing (ETL)
======================================

SUMMARY
-------
This script serves as the primary ETL pipeline for the raw NFL tracking data. 
It ingests split CSV files, standardizes physical units, normalizes field 
direction (Left->Right), and merges game metadata.

Output:
    A single compressed Parquet file ('plays_processed.parquet') optimized for 
    loading into the Self-Supervised Learning (SSL) pipeline.

Functions:
    - flip_play_direction: Standardizes coordinates to a single orientation.
    - normalize_physical_columns: Converts mixed units (e.g., "6-2") to standard integers.
    - merge_input_output: Combines disjoint tracking files.
    - compress_play: Filters essential columns for the backbone model.
"""

import os
import glob
import pandas as pd
import numpy as np

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

# Root Project Path
BASE_PROJECT_PATH = "/lustre/proyectos/p037"

# Raw Data Directory
ACTUAL_DATA_ROOT = f"{BASE_PROJECT_PATH}/datasets/raw/114239_nfl_competition_files_published_analytics_final"
TRAIN_PATH = f"{ACTUAL_DATA_ROOT}/train"
SUPPLEMENTARY = f"{ACTUAL_DATA_ROOT}/supplementary_data.csv"

# Output Directory
OUTPUT_DIR = f"{BASE_PROJECT_PATH}/datasets/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def flip_play_direction(df):
    """
    Standardizes field direction.
    If play_direction == "left", transforms coordinates so the offense always moves "right".
    """
    left_mask = df["play_direction"] == "left"

    # Flip X (120-yard field standard) & Y (53.3 yards width)
    df.loc[left_mask, "x"] = 120 - df.loc[left_mask, "x"]
    df.loc[left_mask, "y"] = 53.3 - df.loc[left_mask, "y"]

    # Normalize orientation (o) and direction (dir)
    df.loc[left_mask, "o"] = (df.loc[left_mask, "o"] + 180) % 360
    df.loc[left_mask, "dir"] = (df.loc[left_mask, "dir"] + 180) % 360

    df["play_direction"] = "right"
    return df


def normalize_physical_columns(df):
    """
    Converts physical attributes to standard numerical units.
    Example: '6-2' (ft-in) -> 74 (inches).
    """
    if "player_height" in df.columns and df["player_height"].dtype == object:
        df["player_height"] = df["player_height"].apply(
            lambda x: int(x.split("-")[0]) * 12 + int(x.split("-")[1]) 
            if isinstance(x, str) and '-' in x else np.nan
        )
    return df


def merge_input_output(input_df, output_df):
    """
    Merges disjoint input/output tracking files into a single timeline.
    """
    return pd.concat([input_df, output_df], axis=0).sort_values(
        ["game_id", "play_id", "nfl_id", "frame_id"]
    )


def compress_play(df):
    """
    Selects only the columns required for the SSL backbone model to save memory.
    """
    keep_cols = [
        "game_id", "play_id", "frame_id", "nfl_id", "player_position",
        "player_side", "x", "y", "s", "a", "o", "dir"
    ]

    # Filter only existing columns
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()
    df = df.sort_values(["frame_id", "nfl_id"])
    return df


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------

def main():
    print("[INFO] Loading supplementary metadata...", flush=True)
    supp = pd.read_csv(SUPPLEMENTARY, low_memory=False)

    print("[INFO] Searching for tracking files...", flush=True)
    # Pattern matching for specific week files
    input_files = sorted(glob.glob(f"{TRAIN_PATH}/input_2023_w??.csv"))
    output_files = sorted(glob.glob(f"{TRAIN_PATH}/output_2023_w??.csv"))
    
    print(f"       Found {len(input_files)} input files and {len(output_files)} output files.", flush=True)

    if not input_files or not output_files:
        print("[ERROR] No input or output files found. Check paths.")
        print(f"       Search Path: {TRAIN_PATH}")
        return

    all_plays = []

    for in_path, out_path in zip(input_files, output_files):
        print(f"[INFO] Processing pair: {os.path.basename(in_path)} | {os.path.basename(out_path)}...", flush=True)
        
        input_df = pd.read_csv(in_path)
        output_df = pd.read_csv(out_path)

        # 1. Normalize Units
        input_df = normalize_physical_columns(input_df)

        # 2. Merge Timelines
        merged = merge_input_output(input_df, output_df)
        
        # 3. Fill 'play_direction' (constant per play)
        merged['play_direction'] = merged.groupby('play_id')['play_direction'].ffill().bfill()
        
        # 4. Standardize Direction (Left -> Right)
        merged = flip_play_direction(merged)

        # 5. Merge Metadata (Validate Many-to-One relationship)
        merged = merged.merge(
            supp,
            on=["game_id", "play_id"],
            how="left",
            validate="m:1"
        )

        # 6. Compress and Store
        for (g, p), play_df in merged.groupby(["game_id", "play_id"]):
            processed = compress_play(play_df)
            all_plays.append(processed)

    if not all_plays:
        print("[WARN] No plays were processed.")
        return

    print(f"[INFO] Total plays processed: {len(all_plays)}", flush=True)

    # Export to Parquet
    out_file = f"{OUTPUT_DIR}/plays_processed.parquet"
    print(f"[INFO] Saving compressed dataset...", flush=True)
    pd.concat(all_plays).to_parquet(out_file)

    print(f"[SUCCESS] Dataset saved at: {out_file}", flush=True)


if __name__ == "__main__":
    main()