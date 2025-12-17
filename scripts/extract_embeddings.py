#!/usr/bin/env python
# coding: utf-8

"""
Phase 2 — Embedding Extraction
==============================

This module utilizes the Self-Supervised Learning (SSL) backbone trained in Phase 1
to transform raw player tracking data into dense vector representations (embeddings).

Process:
    1. Loads the pre-trained 'DynamicEncoder' model weights.
    2. Iterates through the play dataset using the dynamic graph builder.
    3. Extracts the latent feature vector (embedding) for each play.
    4. Exports the embeddings to a Parquet file for downstream tasks.

Dependencies:
    - train_ssl.py: Model architecture definition.
    - dataset_dynamic.py: Data loading and graph construction logic.
"""

import sys
# Adjust path to include local source modules if necessary
sys.path.append(os.getcwd())

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

BACKBONE_PATH = "/lustre/home/dante/compartido/models/backbone_ssl.pth"
OUTPUT_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Import Architecture & Data Helpers
try:
    from train_ssl import DynamicEncoder, HIDDEN_DIM, build_graphs_from_batch, dataloader, device
except ImportError as e:
    raise ImportError("Could not import modules from train_ssl.py. Check your python path.") from e


# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------

print(f"[INFO] Loading backbone model from: {BACKBONE_PATH}")
encoder = DynamicEncoder(in_dim=6, hidden_dim=HIDDEN_DIM).to(device)

if os.path.exists(BACKBONE_PATH):
    state = torch.load(BACKBONE_PATH, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()
    print("[INFO] Backbone successfully loaded in EVAL mode.")
else:
    raise FileNotFoundError(f"Model checkpoint not found at {BACKBONE_PATH}")


# -----------------------------------------------------------
# EXTRACTION LOOP
# -----------------------------------------------------------

embeddings = []
play_ids = []

print("[INFO] Starting embedding extraction...")

with torch.no_grad():
    # Loop over the dataloader
    for batch in tqdm(dataloader, desc="Extracting embeddings", ncols=100):
        # Skip invalid batches (filtered by collate_fn)
        if batch[0] is None:
            continue
            
        X_t_batch, X_tp1_batch = batch
        
        # Build graph representations from raw tensors
        graphs_t = build_graphs_from_batch(X_t_batch)

        for g in graphs_t:
            g = g.to(device)
            
            # Forward pass to get node embeddings
            # We assume edge_type is handled inside the model or graph builder
            h = encoder(g.x, g.edge_index, getattr(g, "edge_type", None), None)
            
            # Mean Pooling: Aggregating node features to get a single vector per play
            emb = h.mean(dim=0).detach().cpu().numpy()

            # Metadata extraction
            pid = getattr(g, "play_id", None)
            
            play_ids.append(pid)
            embeddings.append(emb)


# -----------------------------------------------------------
# EXPORT results
# -----------------------------------------------------------

if len(embeddings) == 0:
    print("[WARN] No embeddings were extracted. Check your dataset path.")
else:
    embeddings = np.stack(embeddings)

    # Create DataFrame with named dimensions
    cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=cols)
    df["play_id"] = play_ids

    # Save to Parquet
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n[INFO] Extraction complete.")
    print(f"       Total processed plays: {len(df)}")
    print(f"       Embeddings saved to:   {OUTPUT_PATH}")