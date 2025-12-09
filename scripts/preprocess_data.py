#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
"""
NFL TRACKING DATA PIPELINE
==========================

SUMMARY
-------
This script acts as the primary ETL pipeline for raw NFL tracking data. 
It ingests split CSV files (input/output), normalizes physical units, 
standardizes field direction to a single orientation (Left->Right), 
and merges game metadata.

The output is a single compressed Parquet file optimized for 
Self-Supervised Learning (SSL) models.

FUNCTIONS
---------
- flip_play_direction: Standardizes coordinates so all plays move Left->Right.
- normalize_physical_columns: Converts "ft-in" height strings to integer inches.
- merge_input_output: Combines split tracking files into a continuous timeline.
- compress_play: Filters columns to keep only those needed for the model backbone.
- main: Orchestrates file loading, processing loops, and final export.
"""

# ============================================================
# CONFIG
# ============================================================

# Pointing to the project root
BASE_PROJECT_PATH = "/lustre/proyectos/p037"

# This is the root path of the actual data found
ACTUAL_DATA_ROOT = f"{BASE_PROJECT_PATH}/datasets/raw/114239_nfl_competition_files_published_analytics_final"

# Update all paths to use it
TRAIN_PATH = f"{ACTUAL_DATA_ROOT}/train"
SUPPLEMENTARY = f"{ACTUAL_DATA_ROOT}/supplementary_data.csv"

# Destination remains the same
OUTPUT_DIR = f"{BASE_PROJECT_PATH}/datasets/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def flip_play_direction(df):
    """
    Standardizes field direction:
    If play_direction == "left", flip coordinates so EVERYTHING points "right".
    """
    left_mask = df["play_direction"] == "left"

    # Flip x: 120-yard field
    df.loc[left_mask, "x"] = 120 - df.loc[left_mask, "x"]
    df.loc[left_mask, "y"] = 53.3 - df.loc[left_mask, "y"]

    # Normalize orientation/direction
    df.loc[left_mask, "o"] = (df.loc[left_mask, "o"] + 180) % 360
    df.loc[left_mask, "dir"] = (df.loc[left_mask, "dir"] + 180) % 360

    df["play_direction"] = "right"
    return df


def normalize_physical_columns(df):
    """
    Normalizes physical columns and converts mixed formats (like height ft-in -> inches).
    """
    # Convert player_height from ft-in to total inches
    if "player_height" in df.columns and df["player_height"].dtype == object:
        df["player_height"] = df["player_height"].apply(
            lambda x: int(x.split("-")[0]) * 12 + int(x.split("-")[1]) if isinstance(x, str) and '-' in x else np.nan
        )
    return df


def merge_input_output(input_df, output_df):
    """
    Joins input and output into a single continuous frame-by-frame dataframe.
    """
    return pd.concat([input_df, output_df], axis=0).sort_values(
        ["game_id", "play_id", "nfl_id", "frame_id"]
    )


def compress_play(df):
    """
    Groups a full play and returns:
    - Frames tensor
    - Play metadata
    """
    # Keep only columns required for SSL backbone
    keep_cols = [
        "game_id", "play_id", "frame_id", "nfl_id", "player_position",
        "player_side", "x", "y", "s", "a", "o", "dir"
    ]

    # Filter only existing columns
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()

    df = df.sort_values(["frame_id", "nfl_id"])
    return df

# ============================================================
# PIPELINE
# ============================================================

def main():
    print("Loading supplementary data...", flush=True)
    
    # Added low_memory=False to silence DtypeWarning
    supp = pd.read_csv(SUPPLEMENTARY, low_memory=False)

    print("Searching for tracking files...", flush=True)
    
    # --- CHANGE HERE: Using 'w??.csv' to be more specific ---
    input_files = sorted(glob.glob(f"{TRAIN_PATH}/input_2023_w??.csv"))
    output_files = sorted(glob.glob(f"{TRAIN_PATH}/output_2023_w??.csv"))
    
    print(f"--- DEBUG: Found {len(input_files)} input and {len(output_files)} output files ---", flush=True)

    if not input_files or not output_files:
        print("Error: No input or output files found. Check paths.", flush=True)
        print(f"Searching in (input): {TRAIN_PATH}", flush=True)
        print(f"Searching in (output): {TRAIN_PATH}", flush=True)
        return

    print(f"Found {len(input_files)} input files and {len(output_files)} output files.", flush=True)
    all_plays = []

    for in_path, out_path in zip(input_files, output_files):

        print(f"Processing {in_path} and {out_path}...", flush=True)
        input_df = pd.read_csv(in_path)
        output_df = pd.read_csv(out_path)

        # --- LOGIC CHANGE ---
        
        # 1. Fix height, etc. in input
        input_df = normalize_physical_columns(input_df)

        # 2. Merge them. 'play_direction' will be NaN for output rows
        merged = merge_input_output(input_df, output_df)
        
        # 3. Fill 'play_direction' for the whole play
        # (Assuming it is constant per 'play_id')
        merged['play_direction'] = merged.groupby('play_id')['play_direction'].ffill().bfill()
        
        # 4. NOW, flip the entire dataframe
        merged = flip_play_direction(merged)

        # Merge with supplementary metadata (game, EPA, etc.)
        merged = merged.merge(
            supp,
            on=["game_id", "play_id"],
            how="left",
            validate="m:1"  # <-- CHANGE HERE!
        )

        # Group play
        for (g, p), play_df in merged.groupby(["game_id", "play_id"]):
            processed = compress_play(play_df)
            all_plays.append(processed)

    if not all_plays:
        print("No plays were processed.", flush=True)
        return

    print(f"Total plays processed: {len(all_plays)}", flush=True)

    # Save all plays as parquet list (fast, compressed)
    out_file = f"{OUTPUT_DIR}/plays_processed.parquet"
    pd.concat(all_plays).to_parquet(out_file)

    print(f" Processed dataset saved at: {out_file}", flush=True)


if __name__ == "__main__":
    main()
