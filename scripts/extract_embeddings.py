#!/usr/bin/env python
# coding: utf-8

"""
Extracts play embeddings using SSL trained backbone.
Needs:
 - train_ssl.py (model DynamicEncoder)
 - dataset_dynamic.py (dataloader and build_graphs_from_batch)
"""
import sys
sys.path.append("/lustre/home/dante/BigDataBowl/src")

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# Config
BACKBONE_PATH = "/lustre/home/dante/compartido/models/backbone_ssl_final.pth"
OUTPUT_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"

# Make Dir if not existant
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Load classes from train_ssl.py
from train_ssl import DynamicEncoder, HIDDEN_DIM, build_graphs_from_batch, dataloader, device

# Load Model
print(f"[INFO] Loading backbone from {BACKBONE_PATH} ...")
encoder = DynamicEncoder(in_dim=6, hidden_dim=HIDDEN_DIM).to(device)
state = torch.load(BACKBONE_PATH, map_location=device)
encoder.load_state_dict(state)
encoder.eval()
print("[INFO] Backbone successfully loaded in eval mode.")

# Extract embeddings per play
embeddings = []
play_ids = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extrayendo embeddings", ncols=100):
        X_t_batch, X_tp1_batch = batch
        graphs_t = build_graphs_from_batch(X_t_batch)

        for g in graphs_t:
            g = g.to(device)
            h = encoder(g.x, g.edge_index, getattr(g, "edge_type", None), None)
            emb = h.mean(dim=0).detach().cpu().numpy()

            pid = getattr(g, "play_id", None)
            play_ids.append(pid)
            embeddings.append(emb)

# Save results
import numpy as np

embeddings = np.stack(embeddings)
df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
df["play_id"] = play_ids

df.to_parquet(OUTPUT_PATH, index=False)
print(f" Embeddings saved in: {OUTPUT_PATH}")
print(f"[INFO] Total proccesed plays: {len(df)}")
