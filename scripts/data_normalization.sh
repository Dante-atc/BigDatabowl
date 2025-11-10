#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

# --- CAMBIO AQUÍ: Usamos la ruta real del proyecto ---
BASE_PROJECT_PATH = "/lustre/proyectos/p037"

# Asumimos que la carpeta '114239...' está dentro de 'raw'
RAW_DATA_ROOT = f"{BASE_PROJECT_PATH}/datasets/raw/114239_nfl_competition_files_published_analytics_final"

TRAIN_PATH = f"{RAW_DATA_ROOT}/train"
SUPPLEMENTARY = f"{RAW_DATA_ROOT}/supplementary_data.csv"

OUTPUT_DIR = f"{BASE_PROJECT_PATH}/datasets/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# (Tu código de helpers va aquí... flip_play_direction, etc.)
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
    if "player_height" in df.columns and df["player_height"].dtype == object:
        df["player_height"] = df["player_height"].apply(
            lambda x: int(x.split("-")[0]) * 12 + int(x.split("-")[1]) if isinstance(x, str) and '-' in x else np.nan
        )
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

    # Filtra solo columnas que existen
    existing_cols = [col for col in keep_cols if col in df.columns]
    df = df[existing_cols].copy()

    df = df.sort_values(["frame_id", "nfl_id"])
    return df

# ============================================================
# PIPELINE
# =================================G===========================

def main():
    print("Cargando supplementary data...")
    supp = pd.read_csv(SUPPLEMENTARY)

    print("Buscando archivos de tracking...")
    # Tus globs originales
    input_files = sorted(glob.glob(f"{TRAIN_PATH}/input_2023_w*.csv"))
    output_files = sorted(glob.glob(f"{RAW_DATA_ROOT}/output_2023_w*.csv")) # Ojo: este está en RAW_DATA_ROOT

    if not input_files or not output_files:
        print("Error: No se encontraron archivos de input u output. Revisa las rutas.")
        print(f"Buscando en: {TRAIN_PATH}")
        print(f"Buscando en: {RAW_DATA_ROOT}")
        return

    print(f"Encontrados {len(input_files)} archivos de input y {len(output_files)} de output.")
    all_plays = []

    for in_path, out_path in zip(input_files, output_files):

        print(f"Procesando {in_path} y {out_path}...")
        input_df = pd.read_csv(in_path)
        output_df = pd.read_csv(out_path)

        # Flip direction to unify field orientation
        input_df = flip_play_direction(input_df)
        output_df = flip_play_direction(output_df)

        # Fix physical cols (height etc)
        # Solo los inputs suelen tener 'player_height'
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

    if not all_plays:
        print("No se procesaron jugadas.")
        return

    print(f"Total de jugadas procesadas: {len(all_plays)}")

    # Save all plays as parquet list (fast, compressed)
    out_file = f"{OUTPUT_DIR}/plays_processed.parquet"
    pd.concat(all_plays).to_parquet(out_file)

    print(f"✅ Dataset procesado guardado en: {out_file}")


if __name__ == "__main__":
    main()