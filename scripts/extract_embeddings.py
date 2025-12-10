#!/usr/bin/env python
# coding: utf-8

"""
EMBEDDING EXTRACTION (SSL BACKBONE)
===================================

SUMMARY
-------
This script executes the inference phase using the pre-trained Self-Supervised 
Learning (SSL) backbone. It processes the dataset in batches, converts raw 
tracking data into graph structures, and extracts a fixed-size embedding vector 
for each play.

This is the primary feature extraction step before clustering.

DEPENDENCIES
------------
- train_ssl.py: Contains the model architecture (DynamicEncoder).
- dataset_dynamic.py: Contains the dataloader and graph construction logic.

OUTPUT
------
A Parquet file containing the play_id and the high-dimensional embedding vector
for every play.
"""

import sys
# Add source directory to path to allow importing local modules
sys.path.append("/lustre/home/dante/BigDataBowl/src")

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
BACKBONE_PATH = "/lustre/home/dante/compartido/models/backbone_ssl_final.pth"
OUTPUT_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"

# Create output directory if it does not exist
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load classes from train_ssl.py
# (Ensure these files are in the python path)
from train_ssl import DynamicEncoder, HIDDEN_DIM, build_graphs_from_batch, dataloader, device

# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------

def main():
    # 1. Load Model
    print(f"[INFO] Loading backbone from {BACKBONE_PATH} ...")
    
    # Initialize architecture
    # in_dim=6 corresponds to [x, y, s, a, o, dir]
    encoder = DynamicEncoder(in_dim=6, hidden_dim=HIDDEN_DIM).to(device)
    
    # Load weights
    try:
        state = torch.load(BACKBONE_PATH, map_location=device)
        encoder.load_state_dict(state)
        encoder.eval()
        print("[INFO] Backbone successfully loaded in eval mode.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        return

    # 2. Extract embeddings per play
    embeddings = []
    play_ids = []

    print(f"[INFO] Starting inference on device: {device}")

    with torch.no_grad():
        # Iterate through the dataloader (which yields batches of play data)
        for batch in tqdm(dataloader, desc="Extracting embeddings", ncols=100):
            X_t_batch, X_tp1_batch = batch
            
            # Convert raw batch tensors into Graph objects
            graphs_t = build_graphs_from_batch(X_t_batch)

            for g in graphs_t:
                g = g.to(device)
                
                # Forward pass through the encoder
                # h dimensions: [num_nodes, hidden_dim]
                h = encoder(g.x, g.edge_index, getattr(g, "edge_type", None), None)
                
                # Pooling: Average all node embeddings to get one Play Embedding
                emb = h.mean(dim=0).detach().cpu().numpy()

                # Retrieve metadata (play_id) attached to the graph object
                pid = getattr(g, "play_id", None)
                
                play_ids.append(pid)
                embeddings.append(emb)

    # 3. Save results
    print("[INFO] Stacking results...")
    embeddings_np = np.stack(embeddings)
    
    df = pd.DataFrame(embeddings_np, columns=[f"dim_{i}" for i in range(embeddings_np.shape[1])])
    df["play_id"] = play_ids

    # Reorder columns to have play_id first (optional, for readability)
    cols = ["play_id"] + [c for c in df.columns if c != "play_id"]
    df = df[cols]

    print(f"[INFO] Saving to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    
    print(f"[INFO] Embeddings saved successfully.")
    print(f"[INFO] Total processed plays: {len(df)}")

if __name__ == "__main__":
    main()
