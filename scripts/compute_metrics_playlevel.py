#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4.2 — Defensive Metrics Baseline Computation
==================================================

This module computes the baseline defensive metrics for each play using the
learned Defensive Archetypes (A_ideal).

CRITICAL FIXES INCLUDED:
1. ID Preservation: Ensures game_id and play_id are correctly loaded and renamed.
2. Dimensionality Match: Loads UMAP embeddings (16D) to match A_ideal centroids.
3. Metric Decoupling: Computes 'Integrity' (Ambiguity) separately from 'Spacing'.

Inputs:
    - embeddings_playlevel.parquet : Metadata (game_id, play_id).
    - UMAP_embeddings.npy          : 16D vector representations of plays.
    - A_ideal_best.parquet         : Learned cluster centroids (16D).

Outputs:
    - metrics_playlevel_baseline.parquet

Author: Team p037
Competition: NFL Big Data Bowl 2026
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

# Path to metadata (Game IDs, Play IDs)
METADATA_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"

# Path to the correct vector space (UMAP 16D)
# We must use the same embeddings used to generate the clusters.
UMAP_PATH = "/lustre/home/dante/compartido/clusters/UMAP_embeddings.npy"
PCA_PATH = "/lustre/home/dante/compartido/clusters/PCA_embeddings.npy" # Fallback

# Path to learned centroids
AIDEAL_PATH = "/lustre/home/dante/compartido/clusters/final/A_ideal_best.parquet"

# Output path
OUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------

print("[INFO] Loading play metadata...")
# Load the parquet containing embeddings and metadata
df_meta = pd.read_parquet(METADATA_PATH)

# --- CRITICAL FIX: Ensure ID columns exist and are named correctly ---

# 1. Reset index:
# In some processing steps, game_id/play_id might have ended up as the index.
# We reset it to ensure they are available as standard columns.
df_meta = df_meta.reset_index()

# 2. Standardize column names:
# Convert legacy camelCase (e.g., from older BDB datasets) to snake_case.
rename_map = {
    "gameId": "game_id", 
    "playId": "play_id", 
    "frameId": "frame_id"
}
df_meta.rename(columns=rename_map, inplace=True)

# 3. Validation:
# Ensure that the critical identifier columns exist before proceeding.
if "game_id" not in df_meta.columns or "play_id" not in df_meta.columns:
    print(f"[ERROR] Available columns: {df_meta.columns.tolist()}")
    raise KeyError("CRITICAL: 'game_id' or 'play_id' missing from metadata! Check input parquet structure.")

# 4. Column Selection:
# We preserve the identifiers (game_id, play_id) and any other metadata.
# We explicitly drop the raw high-dimensional embeddings ('dim_X') to save memory,
# as we will load the reduced dimensionality embeddings (UMAP) separately.
id_cols = ["game_id", "play_id", "frame_id"]
cols_to_keep = [c for c in df_meta.columns if c in id_cols or "dim_" not in c]

df_out = df_meta[cols_to_keep].copy()
print(f"[INFO] Metadata preserved for {len(df_out)} plays.")


# --- Loading Embeddings (Vector Space) ---

print(f"[INFO] Loading reduced embeddings (Target: 16D)...")
if os.path.exists(UMAP_PATH):
    print(f"   -> Loading UMAP embeddings from: {UMAP_PATH}")
    X = np.load(UMAP_PATH)
elif os.path.exists(PCA_PATH):
    print(f"   -> [WARN] UMAP not found. Fallback to PCA embeddings from: {PCA_PATH}")
    X = np.load(PCA_PATH)
else:
    raise FileNotFoundError("Critical: No UMAP or PCA embeddings found in clusters directory.")

print(f"   -> Input X shape: {X.shape} (Rows, Dimensions)")


# --- Loading Centroids ---

print("[INFO] Loading A_ideal centroids...")
aideal = pd.read_parquet(AIDEAL_PATH)
# Extract only dimension columns
cent_cols = [c for c in aideal.columns if "dim_" in c]
C = aideal[cent_cols].values

print(f"   -> Centroids C shape: {C.shape}")

# Safety Assertion: Dimensions must match
if X.shape[1] != C.shape[1]:
    raise ValueError(f"Dimension Mismatch! X has {X.shape[1]} dims, but Centroids have {C.shape[1]} dims.")


# -----------------------------------------------------------
# STEP 1 — GEOMETRIC DISTANCE & CLUSTERING
# -----------------------------------------------------------

print("[INFO] Computing Euclidean distances to archetypes...")
# Compute distance matrix between every play (X) and every centroid (C)
D = pairwise_distances(X, C, metric="euclidean")

# Sort distances to find best matches
D_sorted = np.sort(D, axis=1)

dist_to_centroid = D_sorted[:, 0]  # Distance to nearest cluster (d1)
dist_to_second   = D_sorted[:, 1]  # Distance to second nearest (d2)

# Assign Cluster ID
closest = np.argmin(D, axis=1)


# -----------------------------------------------------------
# STEP 2 — SPACING COHESION PROXY
# -----------------------------------------------------------
"""
Proxy for Geometric Tightness.
Formula: exp(-distance)
Interpretation: High values indicate the play closely matches the ideal spacing archetype.
"""
spacing_proxy = np.exp(-dist_to_centroid)


# -----------------------------------------------------------
# STEP 3 — EXECUTION QUALITY PROXY
# -----------------------------------------------------------
"""
Secondary proxy for Execution.
Formula: 1 / (1 + distance)
Interpretation: Linear decay score representing how well the defense executed the scheme.
"""
execution_proxy = 1.0 / (1.0 + dist_to_centroid)


# -----------------------------------------------------------
# STEP 4 — INTEGRITY PROXY (TACTICAL AMBIGUITY)
# -----------------------------------------------------------
"""
Integrity Proxy measures "Tactical Clarity".
It penalizes defenses that are stuck between two archetypes (ambiguous).

Formula: (d2 - d1) / d2

- If d1 << d2: High Integrity (Clear scheme intent).
- If d1 ~ d2:  Low Integrity (Ambiguous/Broken structure).
"""
# Add epsilon to prevent division by zero
integrity_proxy = (dist_to_second - dist_to_centroid) / (dist_to_second + 1e-6)


# -----------------------------------------------------------
# STEP 5 — BASELINE DCI & DIS METRICS
# -----------------------------------------------------------
"""
Computing raw unoptimized metrics. 
These will be calibrated by the Evolutionary Algorithm later.

DCI (Defensive Coverage Index): Focuses on geometric collapse (Coverage).
DIS (Defensive Integrity Score): Focuses on structural clarity (Discipline).
"""

alpha = 1.0
beta = 1.0
gamma = 1.0

dci_base = np.exp(-alpha * dist_to_centroid)
dis_base = (beta * spacing_proxy + gamma * integrity_proxy) / 2.0


# -----------------------------------------------------------
# EXPORT
# -----------------------------------------------------------

print("[INFO] Exporting baseline metrics...")

# Append new metrics to metadata dataframe
df_out["cluster_id"] = closest
df_out["distance_to_ideal"] = dist_to_centroid
df_out["distance_to_second"] = dist_to_second

# Proxies
df_out["spacing_proxy"] = spacing_proxy
df_out["execution_proxy"] = execution_proxy
df_out["integrity_proxy"] = integrity_proxy

# Baseline Metrics
df_out["dci_base"] = dci_base
df_out["dis_base"] = dis_base

# Save
df_out.to_parquet(OUT_PATH, index=False)

print(f"\n Success! Baseline metrics saved to:")
print(f"    {OUT_PATH}")
print(f"    Dimensions used: {X.shape[1]}D")