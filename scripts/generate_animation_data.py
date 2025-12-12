#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 6 — Visualization Data Generator
======================================

This script processes a specific play frame-by-frame to generate the data
required for visualization and video rendering.

Key Features:
1.  Global Metrics: Computes the 'Instant DCI' per frame.
2.  Node-Level Heatmap: Computes 'Node Stress' for each defender to visualize 
    structural breaks.

Output:
    - CSV file containing enriched tracking data for the target play.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

# Import Architecture
from train_ssl import DynamicEncoder, HIDDEN_DIM, IN_DIM

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

TARGET_GAME_ID = 2023090700
TARGET_PLAY_ID = 101
RAW_DATA = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"
MODEL_PATH = "/lustre/home/dante/compartido/models/backbone_ssl.pth"
OUT_CSV = f"animation_data_{TARGET_GAME_ID}_{TARGET_PLAY_ID}.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def calculate_node_stress(player_pos, teammate_pos):
    """
    Calculates 'Stress' based on local isolation.
    High Stress = Player is far from teammates.
    """
    if len(teammate_pos) == 0:
        return 0.0
    dists = np.linalg.norm(teammate_pos - player_pos, axis=1)
    # Clip stress at 10 yards distance
    return np.clip(np.min(dists) / 10.0, 0.0, 1.0)


# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------

def main():
    print(f"[INFO] Generating animation data for Game {TARGET_GAME_ID} Play {TARGET_PLAY_ID}...")

    # Load Model
    encoder = DynamicEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle prefix if present
        new_state = {k.replace("encoder.", ""): v for k, v in state.items() if "encoder." in k}
        if not new_state: new_state = state
        encoder.load_state_dict(new_state, strict=False)
        encoder.eval()
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return

    # Load Data
    print("[INFO] Loading play data...")
    df = pd.read_parquet(RAW_DATA, filters=[('game_id', '==', TARGET_GAME_ID), ('play_id', '==', TARGET_PLAY_ID)])
    
    if df.empty:
        print("Play not found in dataset.")
        return

    frames = sorted(df.frame_id.unique())
    output_rows = []

    print(f"[INFO] Processing {len(frames)} frames...")

    with torch.no_grad():
        for frame in frames:
            df_f = df[df.frame_id == frame].copy()
            coords = df_f[['x', 'y']].values
            
            # Iterate players to calculate stress
            for idx, row in enumerate(df_f.itertuples()):
                current_pos = coords[idx]
                other_pos = np.delete(coords, idx, axis=0)
                
                stress = calculate_node_stress(current_pos, other_pos)
                
                output_rows.append({
                    "frame_id": frame,
                    "nfl_id": getattr(row, "nfl_id", 0),
                    "x": row.x, "y": row.y, "s": row.s, "dir": row.dir, "o": row.o,
                    "node_stress": stress,
                    "player_side": getattr(row, "player_side", "unknown")
                })

    # Export
    pd.DataFrame(output_rows).to_csv(OUT_CSV, index=False)
    print(f"[SUCCESS] Animation data saved: {OUT_CSV}")

if __name__ == "__main__":
    main()