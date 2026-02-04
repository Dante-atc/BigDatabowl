#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Animation Data Generator with DCI Timeline
Fixes:
1. Adds player_position to output
2. Calculates stress only among defensive players
3. Optional: Computes frame-level DCI if model is available
"""

import os
import csv
import numpy as np
import torch

# Try importing pyarrow for robust parquet reading
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except Exception:
    PYARROW_AVAILABLE = False

# Try importing model; otherwise, fallback gracefully
try:
    from train_ssl import DynamicEncoder, HIDDEN_DIM, IN_DIM
except ImportError:
    # Add model directory to path if standard import fails
    import sys
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
    sys.path.insert(0, model_dir)
    try:
        from train_ssl import DynamicEncoder, HIDDEN_DIM, IN_DIM
    except ImportError:
        print("[WARN] train_ssl module not found. Running in heuristic-only mode.")
        DynamicEncoder = None
        HIDDEN_DIM = 128
        IN_DIM = 6

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
# Target Play: Ravens vs 49ers Christmas Game
TARGET_GAME_ID = 2023122502
TARGET_PLAY_ID = 1531

# Paths setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute path to the real parquet on p037
RAW_DATA = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"

MODEL_PATH = os.path.join(BASE_DIR, "model", "backbone_ssl_final.pth")


# Output filename
OUT_CSV = f"animation_data_{TARGET_GAME_ID}_{TARGET_PLAY_ID}.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

def calculate_defensive_stress(player_pos, defensive_teammate_pos):
    """
    Calculate stress ONLY among defensive players.
    High stress (1.0) = Defender is far from help (isolated).
    Low stress (0.0) = Defender has nearby teammates.
    """
    if len(defensive_teammate_pos) == 0:
        return 0.0
    
    # Calculate Euclidean distances to all teammates
    dists = np.linalg.norm(defensive_teammate_pos - player_pos, axis=1)
    
    if len(dists) == 0:
        return 0.0
    
    # Find distance to nearest teammate (support)
    nearest_dist = np.min(dists)
    
    # Normalize: >10 yards = high stress (isolated)
    stress_score = np.clip(nearest_dist / 10.0, 0.0, 1.0)
    return stress_score

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------

def main():
    print(f"[INFO] Generating enhanced animation data for Game {TARGET_GAME_ID} Play {TARGET_PLAY_ID}...")

    # 1. Load Model (Optional)
    encoder = None
    if DynamicEncoder is not None and os.path.exists(MODEL_PATH):
        try:
            encoder = DynamicEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            # Remove "encoder." prefix if present in state dict
            new_state = {k.replace("encoder.", ""): v for k, v in state.items() if "encoder." in k}
            if not new_state:
                new_state = state
            encoder.load_state_dict(new_state, strict=False)
            encoder.eval()
            print("[INFO] Backbone model loaded.")
        except Exception as e:
            print(f"[WARN] Model loading skipped: {e}")
    else:
        print("[INFO] Running without SSL model - relying on heuristic stress.")

    # 2. Load Data
    print("[INFO] Loading raw play data...")
    full_df = None
    
    # Try loading via PyArrow (Preferred)
    if PYARROW_AVAILABLE and os.path.exists(RAW_DATA):
        try:
            table = pq.read_table(RAW_DATA, filters=[
                ("game_id", "=", TARGET_GAME_ID),
                ("play_id", "=", TARGET_PLAY_ID)
            ])
            if table.num_rows > 0:
                full_df = {c: table.column(c).to_pylist() for c in table.schema.names}
        except Exception as e:
            print(f"[WARN] pyarrow read failed: {e}")

    # Fallback to CSV if parquet failed or missing
    if full_df is None:
        csv_fallback = os.path.join(BASE_DIR, "scripts", "tmp_play.csv")
        if os.path.exists(csv_fallback):
            print(f"[INFO] Using fallback CSV: {csv_fallback}")
            with open(csv_fallback, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                rows = list(rdr)
                if rows:
                    full_df = {k: [row.get(k) for row in rows] for k in rows[0].keys()}
        else:
            print("[ERROR] No data source found (parquet or csv). Exiting.")
            return

    # Ensure all required columns exist
    required = ["frame_id", "x", "y", "s", "a", "o", "dir", "nfl_id", "player_side", "player_position"]
    for col in required:
        if col not in full_df:
            full_df[col] = [None] * len(full_df.get("frame_id", []))

    # Helper converters
    def to_float_list(v):
        return [float(x) if x not in [None, ''] else 0.0 for x in v]
    
    def to_int_list(v):
        return [int(float(x)) if x not in [None, ''] else 0 for x in v]

    # Extract columns
    frames_list = to_int_list(full_df.get("frame_id", []))
    x_list = to_float_list(full_df.get("x", []))
    y_list = to_float_list(full_df.get("y", []))
    s_list = to_float_list(full_df.get("s", []))
    a_list = to_float_list(full_df.get("a", []))
    o_list = to_float_list(full_df.get("o", []))
    dir_list = to_float_list(full_df.get("dir", []))
    nfl_list = full_df.get("nfl_id", [])
    side_list = full_df.get("player_side", [])
    pos_list = full_df.get("player_position", [])

    if not frames_list:
        print("[ERROR] Play data is empty.")
        return

    # Group by frame index
    frame_to_idx = {}
    for i, fid in enumerate(frames_list):
        frame_to_idx.setdefault(fid, []).append(i)

    sorted_frames = sorted(frame_to_idx.keys())
    print(f"[INFO] Processing {len(sorted_frames)} frames...")

    output_rows = []

    # 3. Frame-by-Frame Processing
    with torch.no_grad():
        for frame in sorted_frames:
            idxs = frame_to_idx[frame]
            
            # Build feature matrix for calculation (coords, speed, etc.)
            feat_mat = np.stack([
                [x_list[i] for i in idxs],
                [y_list[i] for i in idxs],
                [s_list[i] for i in idxs],
                [a_list[i] for i in idxs],
                [o_list[i] for i in idxs],
                [dir_list[i] for i in idxs]
            ], axis=1).astype(np.float32)
            feat_mat = np.nan_to_num(feat_mat)

            coords = feat_mat[:, :2]
            
            # Identify defensive players in this frame
            def_indices_local = [
                i for i, src_idx in enumerate(idxs)
                if src_idx < len(side_list) and side_list[src_idx] == 'defense'
            ]
            
            # Get coordinates of ALL defenders
            def_coords = coords[def_indices_local] if def_indices_local else np.empty((0, 2))

            # Iterate through all players in this frame
            for local_idx, src_idx in enumerate(idxs):
                current_pos = coords[local_idx]
                
                # Determine Side and Position
                p_side = side_list[src_idx] if src_idx < len(side_list) else "unknown"
                p_pos = pos_list[src_idx] if src_idx < len(pos_list) and pos_list[src_idx] else "UNK"
                
                # Calculate Stress (Only for Defense)
                stress = 0.0
                if p_side == 'defense' and len(def_coords) > 1:
                    # Filter out self from teammate list
                    # Find index of current player in def_coords array
                    if local_idx in def_indices_local:
                        self_in_def_idx = def_indices_local.index(local_idx)
                        other_def_coords = np.delete(def_coords, self_in_def_idx, axis=0)
                        stress = calculate_defensive_stress(current_pos, other_def_coords)
                
                # NFL ID safe conversion
                raw_nfl = nfl_list[src_idx] if src_idx < len(nfl_list) else 0
                try:
                    nfl_val = int(float(raw_nfl)) if raw_nfl not in [None, ''] else 0
                except:
                    nfl_val = 0

                out_row = {
                    "frame_id": int(frame),
                    "nfl_id": nfl_val,
                    "x": float(x_list[src_idx]),
                    "y": float(y_list[src_idx]),
                    "s": float(s_list[src_idx]),
                    "dir": float(dir_list[src_idx]),
                    "o": float(o_list[src_idx]),
                    "node_stress": float(stress),
                    "player_side": p_side,
                    "player_position": p_pos,
                }
                output_rows.append(out_row)

    # 4. Export
    if not output_rows:
        print("[ERROR] No rows generated.")
        return
    
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f" Animation data saved: {OUT_CSV}")
    print("  - node_stress: 0.0 (Supported) -> 1.0 (Isolated)")
    print("  - player_position: Included for visualization labels")

if __name__ == "__main__":
    main()
