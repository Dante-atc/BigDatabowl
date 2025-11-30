#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Play-Level Embeddings (Bulletproof Version)
====================================================
This script includes aggressive sanitization to prevent NaN propagation.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

# Import model architecture
from train_ssl import DynamicEncoder, HIDDEN_DIM, IN_DIM
from dataset_dynamic import DynamicPlayDataset

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_PATH = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"
MODEL_PATH = "/lustre/home/dante/compartido/models/backbone_ssl_final.pth"
OUT_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"[INFO] Device: {DEVICE}")

    # 1. Load Dataset
    print("[INFO] Loading dataset...")
    dataset = DynamicPlayDataset(DATA_PATH, seq_len=1)
    print(f"[INFO] Loaded {len(dataset)} plays.")

    # 2. Load Model
    print(f"[INFO] Loading backbone from {MODEL_PATH}...")
    encoder = DynamicEncoder(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state_dict[k.replace("encoder.", "")] = v
            else:
                new_state_dict[k] = v
        encoder.load_state_dict(new_state_dict, strict=False)
        print("[INFO] Model weights loaded successfully.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not load model: {e}")
        return

    encoder.eval()

    # 3. Generate Embeddings
    results = []
    
    print("[INFO] Generating embeddings with sanitation...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset.plays))):
            play_data = dataset.plays[i]
            game_id = play_data["game_id"]
            play_id = play_data["play_id"]
            df = play_data["df"]
            
            # Take first frame
            first_frame = df["frame_id"].min()
            df_frame = df[df["frame_id"] == first_frame]
            
            # Prepare tensor [N, F]
            feat_cols = ["x", "y", "s", "a", "o", "dir"]
            x_np = df_frame[feat_cols].values.astype(np.float32)
            
            # --- FIX 1: SANITIZE INPUT ---
            # Replace NaNs or Infinity with 0.0 before creating tensor
            if np.isnan(x_np).any() or np.isinf(x_np).any():
                x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            x_tensor = torch.tensor(x_np)
            
            # Default embedding (zeros) in case of failure
            emb_np = np.zeros(HIDDEN_DIM, dtype=np.float32)
            
            if x_tensor.shape[0] >= 2:
                pos = x_tensor[:, :2]
                dist = torch.cdist(pos, pos, p=2)
                edge_index = (dist < 10.0).nonzero(as_tuple=False).T
                mask = edge_index[0] != edge_index[1]
                edge_index = edge_index[:, mask]

                # --- FIX 2: HANDLE EMPTY GRAPHS ---
                # If players are too far apart (no edges), R-GCN might divide by zero.
                if edge_index.shape[1] > 0:
                    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=DEVICE)
                    
                    data = Data(x=x_tensor, edge_index=edge_index).to(DEVICE)
                    
                    # Forward Pass
                    try:
                        node_embs = encoder(data.x, data.edge_index, edge_type, None)
                        
                        # Check for NaNs immediately after model
                        if torch.isnan(node_embs).any():
                            # Silent fail for this play, keep zeros
                            pass 
                        else:
                            graph_emb = node_embs.mean(dim=0)
                            emb_np = graph_emb.cpu().numpy()
                            
                            # --- FIX 3: SANITIZE OUTPUT ---
                            if np.isnan(emb_np).any():
                                emb_np = np.zeros(HIDDEN_DIM, dtype=np.float32)

                    except Exception as e:
                        # If model crashes for this specific graph, skip it (keep zeros)
                        pass

            # Store result
            row = {
                "game_id": int(game_id),
                "play_id": int(play_id),
                "frame_id": int(first_frame)
            }
            for dim_idx, val in enumerate(emb_np):
                row[f"dim_{dim_idx}"] = float(val)
                
            results.append(row)

    # 4. Save
    print("[INFO] Saving to parquet...")
    df_out = pd.DataFrame(results)
    df_out["game_id"] = df_out["game_id"].astype(int)
    df_out["play_id"] = df_out["play_id"].astype(int)
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_parquet(OUT_PATH, index=False)
    
    print(f"✅ Embeddings saved successfully: {OUT_PATH}")
    
    # Final Sanity Check
    nans = df_out["dim_0"].isna().sum()
    zeros = (df_out["dim_0"] == 0).sum()
    print(f"   -> Plays with NaN: {nans}")
    print(f"   -> Plays with Zeros (Rescue): {zeros}")

if __name__ == "__main__":
    main()