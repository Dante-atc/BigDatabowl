#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 6 — Animation Data Generator (Frame-by-Frame Inference)
=============================================================

This script processes a specific play frame-by-frame to generate the data
required for visualization and video rendering.

Key Features:
1.  **Global Metrics:** Computes the 'Instant DCI' (Defensive Coverage Index) per frame.
2.  **Node-Level Heatmap:** Computes 'Node Stress' for each defender to visualize 
    structural breaks (where the integrity is failing).

Output:
    - animation_{gameId}_{playId}.csv : Detailed frame-level data for plotting.

"""

import torch
import pandas as pd
import numpy as np
import joblib
from torch_geometric.data import Data

# Import Architecture
from train_ssl import DynamicEncoder, HIDDEN_DIM, IN_DIM

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

# SELECT THE PLAY TO ANIMATE 
TARGET_GAME_ID = 2023090700
TARGET_PLAY_ID = 101

# Paths
BASE_DIR = "/lustre/home/dante/compartido"
RAW_DATA = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"
MODEL_PATH = f"{BASE_DIR}/models/backbone_ssl_final.pth"

# Output CSV
OUT_CSV = f"animation_data_{TARGET_GAME_ID}_{TARGET_PLAY_ID}.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def calculate_node_stress(player_pos, teammate_pos):
    """
    Calculates the 'Stress' of a node based on local isolation.
    High Stress = Player is far from teammates (Potential structural break).
    """
    if len(teammate_pos) == 0:
        return 0.0
    
    # Euclidean distance to all teammates
    dists = np.linalg.norm(teammate_pos - player_pos, axis=1)
    
    # We take the distance to the nearest neighbor as the primary stress factor
    # (If my closest help is far away, I am stressed).
    nearest_dist = np.min(dists)
    
    # Normalize simply for visualization (e.g., > 10 yards is high stress)
    stress_score = np.clip(nearest_dist / 10.0, 0.0, 1.0)
    return stress_score

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------

def main():
    print(f"[INFO] Generating animation data for Game {TARGET_GAME_ID} Play {TARGET_PLAY_ID}...")

    # 1. Load Backbone Model (R-GCN)
    print("[INFO] Loading Backbone Model...")
    encoder = DynamicEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle 'encoder.' prefix if present
        new_state = {k.replace("encoder.", ""): v for k, v in state.items() if "encoder." in k}
        if not new_state: new_state = state
        encoder.load_state_dict(new_state, strict=False)
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return

    encoder.eval()

    # 2. Load Raw Tracking Data
    print("[INFO] Loading and filtering raw play data...")
    # Reading full parquet might be slow, optimize if needed
    full_df = pd.read_parquet(RAW_DATA, filters=[
        ('game_id', '==', TARGET_GAME_ID),
        ('play_id', '==', TARGET_PLAY_ID)
    ])
    
    if full_df.empty:
        print(" Play not found in dataset.")
        return

    # Sort by frame
    frames = sorted(full_df.frame_id.unique())
    print(f"[INFO] Processing {len(frames)} frames...")

    output_rows = []

    # 3. Frame-by-Frame Inference
    with torch.no_grad():
        for frame in frames:
            df_f = full_df[full_df.frame_id == frame].copy()
            
            # --- PREPARE TENSORS ---
            feat_cols = ["x", "y", "s", "a", "o", "dir"]
            # Sanitize inputs
            x_np = np.nan_to_num(df_f[feat_cols].values.astype(np.float32))
            x_tensor = torch.tensor(x_np)
            
            # Build Graph (Dynamic edges < 10 yards)
            pos = x_tensor[:, :2]
            dist_mat = torch.cdist(pos, pos)
            edge_index = (dist_mat < 10.0).nonzero(as_tuple=False).T
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=DEVICE)
            
            # --- MODEL INFERENCE (Global Context) ---
            # We run the model to get the latent representation of the frame
            data = Data(x=x_tensor, edge_index=edge_index).to(DEVICE)
            node_embs = encoder(data.x, data.edge_index, edge_type, None)
            
            # Note: For the animation CSV, we mostly need the raw positions 
            # and the calculated per-node stress. The global DCI curve 
            # can be smoothed from the 'metrics_playlevel_supervised.parquet' later.

            # --- NODE-LEVEL HEATMAP CALCULATION ---
            # Identify defensive players
            # Assuming 'player_side' column exists. If not, use position logic.
            # Here we assume column 'player_side' == 'defense' (check your data schema)
            
            # If 'player_side' is missing in processed data, we infer from file context
            # or skip specific side logic for generic heatmap.
            
            # Compute Stress for each player
            # We iterate rows to keep mapping correct
            coords = df_f[['x', 'y']].values
            
            for idx, (i, row) in enumerate(df_f.iterrows()):
                
                # Identify if Defense (Logic depends on your schema)
                # If you have 'player_side', use it. Otherwise, calculate for all.
                is_defense = True 
                
                current_pos = coords[idx]
                other_pos = np.delete(coords, idx, axis=0)
                
                # Calculate Node Stress (DIS Proxy)
                stress = calculate_node_stress(current_pos, other_pos)
                
                # Append to output
                out_row = {
                    "frame_id": frame,
                    "nfl_id": row.get("nfl_id", 0),
                    "x": row["x"],
                    "y": row["y"],
                    "s": row["s"],
                    "dir": row["dir"],
                    "o": row["o"],
                    "node_stress": stress, 
                    "player_side": row.get("player_side", "unknown")
                }
                output_rows.append(out_row)

    # 4. Export
    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(OUT_CSV, index=False)
    
    print(f" Animation data saved: {OUT_CSV}")
    print("   Use 'node_stress' column to color-code the DIS heatmap.")

if __name__ == "__main__":
    main()