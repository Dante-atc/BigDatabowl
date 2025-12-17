#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4 — Baseline Metric Computation
=====================================

This module computes the baseline defensive metrics for each play by projecting
embeddings into the learned Archetype Space (A_ideal).

Key Operations:
1.  ID Preservation: Ensures game_id and play_id are correctly propagated.
2.  Dimensionality Alignment: Loads UMAP embeddings (16D) to match centroids.
3.  Metric Decomposition: Decouples 'Spacing' (Density) from 'Integrity' (Structure).

Inputs:
    - embeddings_playlevel.parquet : Metadata keys.
    - UMAP_embeddings.npy          : 16D vector representations.
    - A_ideal_best.parquet         : Learned cluster centroids.

Outputs:
    - metrics_playlevel_baseline.parquet: DataFrame containing raw geometric scores.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

METADATA_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"
UMAP_PATH = "/lustre/home/dante/compartido/clusters/UMAP_embeddings.npy"
PCA_PATH = "/lustre/home/dante/compartido/clusters/PCA_embeddings.npy"
AIDEAL_PATH = "/lustre/home/dante/compartido/clusters/final/A_ideal_best.parquet"
OUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------

print("[INFO] Loading play metadata...")
df_meta = pd.read_parquet(METADATA_PATH)
df_meta = df_meta.reset_index()

# Standardize column names
rename_map = {"gameId": "game_id", "playId": "play_id", "frameId": "frame_id"}
df_meta.rename(columns=rename_map, inplace=True)

if "game_id" not in df_meta.columns or "play_id" not in df_meta.columns:
    raise KeyError("CRITICAL: 'game_id' or 'play_id' missing from metadata.")

# Preserve IDs
id_cols = ["game_id", "play_id", "frame_id"]
cols_to_keep = [c for c in df_meta.columns if c in id_cols]
df_out = df_meta[cols_to_keep].copy()

# Load Vectors
print(f"[INFO] Loading reduced embeddings...")
if os.path.exists(UMAP_PATH):
    print(f"   -> Loading UMAP embeddings from: {UMAP_PATH}")
    X = np.load(UMAP_PATH)
elif os.path.exists(PCA_PATH):
    print(f"   -> [WARN] UMAP not found. Fallback to PCA embeddings.")
    X = np.load(PCA_PATH)
else:
    raise FileNotFoundError("Critical: No embeddings found.")


# -----------------------------------------------------------
# METRIC COMPUTATION
# -----------------------------------------------------------

print("[INFO] Loading A_ideal centroids...")
aideal = pd.read_parquet(AIDEAL_PATH)
cent_cols = [c for c in aideal.columns if "dim_" in c]
C = aideal[cent_cols].values

if X.shape[1] != C.shape[1]:
    raise ValueError(f"Dimension Mismatch! X: {X.shape}, Centroids: {C.shape}")

print("[INFO] Computing Euclidean distances to archetypes...")
D = pairwise_distances(X, C, metric="euclidean")
D_sorted = np.sort(D, axis=1)

dist_to_centroid = D_sorted[:, 0]  # d1
dist_to_second   = D_sorted[:, 1]  # d2
closest = np.argmin(D, axis=1)

# --- Proxies ---
# Spacing Proxy: How 'tightly' the play fits the ideal archetype.
spacing_proxy = np.exp(-dist_to_centroid)

# Integrity Proxy: Measures 'Tactical Clarity' (Ambiguity).
integrity_proxy = (dist_to_second - dist_to_centroid) / (dist_to_second + 1e-6)

# Baseline Metrics (Uncalibrated)
alpha, beta, gamma = 1.0, 1.0, 1.0
dci_base = np.exp(-alpha * dist_to_centroid)
dis_base = (beta * spacing_proxy + gamma * integrity_proxy) / 2.0


# -----------------------------------------------------------
# EXPORT
# -----------------------------------------------------------

print("[INFO] Exporting baseline metrics...")
df_out["cluster_id"] = closest
df_out["distance_to_ideal"] = dist_to_centroid
df_out["distance_to_second"] = dist_to_second
df_out["spacing_proxy"] = spacing_proxy
df_out["integrity_proxy"] = integrity_proxy
df_out["dci_base"] = dci_base
df_out["dis_base"] = dis_base

df_out.to_parquet(OUT_PATH, index=False)
print(f"[SUCCESS] Metrics saved to: {OUT_PATH}")