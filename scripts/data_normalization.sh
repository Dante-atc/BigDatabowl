#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================
RAW_PATH = "/lustre/home/dante/compartido/datasets/raw/114239_nfl_competition_files_published_analytics_final"
TRAIN_PATH = f"{RAW_PATH}/train"
SUPPLEMENTARY = f"{RAW_PATH}/supplementary_data.csv"

OUTPUT_DIR = "/lustre/home/dante/compartido/datasets/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def flip_play_direction(df):
    """
    Unifica la dirección del campo:
    Si play_direction == "left", flippear coordenadas para que TODO apunte a "right".
    """
    left_mask = df["play_direction"] == "left"

    # Flip x: campo de 120 yardas
    df.loc[left_mask, "x"] = 120 - df.loc[left_mask, "x"]
    df.loc[left_mask, "y"] = 53.3 - df.loc[left_mask, "y"]

    # Normalizar orientación
    df.loc[left_mask, "o"] = (df.loc[left_mask, "o"] + 180) % 360
    df.loc[left_mask, "dir"] = (df.loc[left_mask, "dir"] + 180) % 360

    df["play_direction"] = "right"
    return df


def normalize_physical_columns(df):
    """
    Normaliza columnas físicas y convierte dobles formatos (como altura ft-in → pulgadas)
    """
    # Convert player_height from ft-in to total inches
    if df["player_height"].dtype == object:
        df["player_height"] = df["player_height"].apply(
            lambda x: int(x.split("-")[0]) * 12 + int(x.split("-")[1]) if isinstance(x, str) else np.nan
        )

    # Normalize speeds, accelerations? (NO aqui, SSL aprende raw)
    return df


def merge_input_output(input_df, output_df):
    """
    Une input y output en un único dataframe continuo frame-by-frame.
    """
    return pd.concat([input_df, output_df], axis=0).sort_values(
        ["game_id", "play_id", "nfl_id", "frame_id"]
    )


def compress_play(df):
    """
    Agrupa un play completo y devuelve:
    - Tensor de frames
    - Metadata de jugada
    """
    # Keep only columns required for SSL backbone
    keep_cols = [
        "game_id", "play_id", "frame_id", "nfl_id", "player_position",
        "player_side", "x", "y", "s", "a", "o", "dir"
    ]

    df = df[keep_cols].copy()
    df = df.sort_values(["frame_id", "nfl_id"])

    return df


# ============================================================
# PIPELINE
# ============================================================

def main():
    print("Loading supplementary data...")
    supp = pd.read_csv(SUPPLEMENTARY)

    print("Loading input tracking data...")
    input_files = sorted(glob.glob(f"{TRAIN_PATH}/input_2023_w*.csv"))
    output_files = sorted(glob.glob(f"{RAW_PATH}/output_2023_w*.csv"))

    all_plays = []

    for in_path, out_path in zip(input_files, output_files):

        print(f"Processing {in_path} ...")
        input_df = pd.read_csv(in_path)
        output_df = pd.read_csv(out_path)

        # Flip direction to unify field orientation
        input_df = flip_play_direction(input_df)
        output_df = flip_play_direction(output_df)

        # Fix physical cols (height etc)
        input_df = normalize_physical_columns(input_df)

        # Merge input + output frames
        merged = merge_input_output(input_df, output_df)

        # Merge with supplementary metadata (game, EPA, etc.)
        merged = merged.merge(
            supp,
            on=["game_id", "play_id"],
            how="left",
            validate="many_to-one"
        )

        # Group play
        for (g, p), play_df in merged.groupby(["game_id", "play_id"]):
            processed = compress_play(play_df)
            all_plays.append(processed)

    print(f"Total plays processed: {len(all_plays)}")

    # Save all plays as parquet list (fast, compressed)
    out_file = f"{OUTPUT_DIR}/plays_processed.parquet"
    pd.concat(all_plays).to_parquet(out_file)

    print(f"✅ Saved processed dataset to: {out_file}")


if __name__ == "__main__":
    main()
