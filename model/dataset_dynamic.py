#!/usr/bin/env python
# coding: utf-8

"""
Shared Module — Dynamic Dataset & Graph Builder
===============================================

This module provides the core infrastructure for loading NFL tracking data and 
converting it into graph structures suitable for Geometric Deep Learning.

Key Components:
1.  DynamicPlayDataset: Loads play data and serves frame pairs (t, t+1).
2.  dynamic_collate: Handles batching of variable-size graphs.
3.  build_graphs_from_batch: Converts raw tensors into PyG Data objects with 
    spatial edges defined by a distance threshold.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.data import Data

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

PROCESSED_PATH = "/lustre/proyectos/p037/datasets/processed/plays_processed.parquet"

BATCH_SIZE = 8
NUM_WORKERS = 4

# Fixed Seeds
torch.manual_seed(42)
np.random.seed(42)


# -----------------------------------------------------------
# DATASET CLASS
# -----------------------------------------------------------

class DynamicPlayDataset(Dataset):
    """
    PyTorch Dataset that dynamically serves frame pairs (X_t, X_t+1) for SSL.
    """
    def __init__(self, parquet_path, seq_len=2):
        super().__init__()
        self.seq_len = seq_len
        print(f"[INFO] Loading processed dataset from {parquet_path}...", flush=True)
        self.data = pd.read_parquet(parquet_path)

        # Validation
        required_cols = ["game_id", "play_id", "frame_id", "nfl_id", "x", "y", "s", "a", "o", "dir"]
        missing = [c for c in required_cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Group by Play
        self.plays = []
        for (gid, pid), df in tqdm(self.data.groupby(["game_id", "play_id"]), desc="Grouping plays"):
            frames = sorted(df["frame_id"].unique())
            self.plays.append({
                "game_id": gid,
                "play_id": pid,
                "frames": frames,
                "df": df
            })

        print(f"Total plays loaded: {len(self.plays)}")

    def __len__(self):
        return len(self.plays)

    def __getitem__(self, idx):
        play = self.plays[idx]
        df = play["df"]
        frames = play["frames"]

        if len(frames) < self.seq_len:
            return None, None

        # Random frame selection
        t = np.random.randint(0, len(frames) - 1)
        f_t, f_tp1 = frames[t], frames[t + 1]

        feat_cols = ["x", "y", "s", "a", "o", "dir"]

        # ID Mapping
        nfl_ids = sorted(df["nfl_id"].unique())
        id_map = {nid: i for i, nid in enumerate(nfl_ids)}

        # Filter Frames
        df_t = df[df["frame_id"] == f_t].copy()
        df_tp1 = df[df["frame_id"] == f_tp1].copy()

        # Padding / Matrix construction
        def fill_players(df_frame):
            mat = np.zeros((len(nfl_ids), len(feat_cols)), dtype=np.float32)
            for _, row in df_frame.iterrows():
                i = id_map[row["nfl_id"]]
                mat[i] = row[feat_cols].values
            return mat

        X_t = fill_players(df_t)
        X_tp1 = fill_players(df_tp1)

        return torch.tensor(X_t), torch.tensor(X_tp1)


# -----------------------------------------------------------
# COLLATE FUNCTION
# -----------------------------------------------------------

def dynamic_collate(batch):
    """Batches plays with different numbers of players."""
    valid_batch = [(x, y) for x, y in batch if x is not None and y is not None]
    if not valid_batch:
        return None, None

    X_t_batch, X_tp1_batch = zip(*valid_batch)
    max_players = max(x.shape[0] for x in X_t_batch)
    feat_dim = X_t_batch[0].shape[1]

    def pad_tensor(t, target_n):
        pad_n = target_n - t.shape[0]
        if pad_n > 0:
            pad = torch.zeros((pad_n, feat_dim))
            return torch.cat([t, pad], dim=0)
        return t

    X_t_padded = torch.stack([pad_tensor(t, max_players) for t in X_t_batch])
    X_tp1_padded = torch.stack([pad_tensor(t, max_players) for t in X_tp1_batch])

    return X_t_padded, X_tp1_padded


# -----------------------------------------------------------
# GLOBAL DATALOADER
# -----------------------------------------------------------

dataset = DynamicPlayDataset(PROCESSED_PATH)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    collate_fn=dynamic_collate
)


# -----------------------------------------------------------
# GRAPH BUILDER
# -----------------------------------------------------------

def build_graphs_from_batch(X_batch, max_distance=10.0):
    """
    Converts batch tensor [B, N, F] -> List of PyG Data objects.
    Edges are created based on Euclidean distance < max_distance.
    """
    graphs = []
    if X_batch is None:
        return graphs

    B, N, F = X_batch.shape

    for b in range(B):
        x = X_batch[b]
        
        # Remove padding (zero rows)
        valid_mask = (x.abs().sum(dim=1) > 0)
        x_valid = x[valid_mask]
        num_nodes = x_valid.size(0)

        if num_nodes <= 1:
            continue

        # Distance Matrix
        pos = x_valid[:, :2]
        dist = torch.cdist(pos, pos, p=2)

        # Edge Creation
        edge_index = (dist < max_distance).nonzero(as_tuple=False).T
        mask = edge_index[0] != edge_index[1] # Remove self-loops
        edge_index = edge_index[:, mask]

        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        data = Data(
            x=x_valid,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes
        )
        graphs.append(data)

    return graphs