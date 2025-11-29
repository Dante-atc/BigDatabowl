#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 4.2 — Defensive Metrics Baseline Computation
==================================================
This module computes the baseline defensive metrics for each play-level
embedding using the A_ideal centroids learned in Phase 4.1.

Outputs:
- metrics_playlevel_baseline.parquet

These baseline metrics DO NOT perform any optimization. They are the raw
geometric + cohesion + integrity features that the Differential Evolution
optimizer will later calibrate.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

EMBED_PATH = "/lustre/home/dante/compartido/embeddings/embeddings_playlevel.parquet"
AIDEAL_PATH = "/lustre/home/dante/compartido/clusters/final/A_ideal_best.parquet"
OUT_PATH = "/lustre/home/dante/compartido/metrics/metrics_playlevel_baseline.parquet"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------

print("[INFO] Loading embeddings...")
df = pd.read_parquet(EMBED_PATH)

embed_cols = [c for c in df.columns if "dim_" in c]
X = df[embed_cols].values

print(f"[INFO] Loaded {X.shape[0]} plays with {X.shape[1]} embedding dims.")

print("[INFO] Loading A_ideal centroids...")
aideal = pd.read_parquet(AIDEAL_PATH)
C = aideal[[c for c in aideal.columns if "dim_" in c]].values

print(f"[INFO] Loaded {C.shape[0]} defensive archetype centroids.")


# -----------------------------------------------------------
# STEP 1 — DISTANCE TO IDEAL CENTROIDS & AMBIGUITY CHECK
# -----------------------------------------------------------

print("[INFO] Computing distances to archetype centroids...")
D = pairwise_distances(X, C, metric="euclidean")

# We sort distances to find the closest (assigned) and second closest (competitor) archetypes.
# D_sorted[:, 0] -> Distance to the best matching cluster (d1)
# D_sorted[:, 1] -> Distance to the second best matching cluster (d2)
D_sorted = np.sort(D, axis=1)

dist_to_centroid = D_sorted[:, 0]
dist_to_second   = D_sorted[:, 1]

# Get the actual cluster ID of the closest centroid
closest = np.argsort(D, axis=1)[:, 0]


# -----------------------------------------------------------
# STEP 2 — SPACING COHESION PROXY
# -----------------------------------------------------------
"""
This proxy estimates how "tight" the defensive shape is relative to the ideal.

Formula:
    spacing_proxy = exp(-distance)

Interpretation:
- 1.0  → Extremely coherent coverage (perfect match to centroid).
- 0.0  → Blown spacing / large geometric deviations.
"""

spacing_proxy = np.exp(-dist_to_centroid)


# -----------------------------------------------------------
# STEP 3 — EXECUTION QUALITY PROXY
# -----------------------------------------------------------
"""
Execution proxy captures the raw proximity to the cluster, using an inverse
linear decay. This serves as a secondary check for geometric quality.

Formula:
    execution_proxy = 1 / (1 + distance)
"""

execution_proxy = 1.0 / (1.0 + dist_to_centroid)


# -----------------------------------------------------------
# STEP 4 — INTEGRITY PROXY
# -----------------------------------------------------------
"""
The Integrity Proxy measures the Tactic Clarity of the defense.
It answers: Does the defense clearly know what scheme it is playing?

Formula:
    integrity = (d2 - d1) / d2

Logic:
- If d1=2.0 and d2=10.0 -> (8/10) = 0.80. High Integrity. The defense is clearly
  executing Cluster A and looks nothing like Cluster B.
- If d1=5.0 and d2=5.1  -> (0.1/5.1) ~ 0.02. Low Integrity. The defense is
  ambiguous, caught in No Man's Land between two schemes.
"""

# Add epsilon to avoid division by zero in edge cases
integrity_proxy = (dist_to_second - dist_to_centroid) / (dist_to_second + 1e-6)


# -----------------------------------------------------------
# STEP 5 — BASE DCI & DIS (UNOPTIMIZED BASELINES)
# -----------------------------------------------------------
"""
We compute the raw metrics. The DE optimizer will later find the optimal
weights (alpha, beta, gamma) to maximize correlation with EPA/Success.

DCI (Defensive Coverage Index):
    - Purely geometric.
    - Measures how well the defense "collapses" the space towards the ideal.
    - DCI = exp(-alpha * distance)

DIS (Defensive Integrity Score):
    - Purely structural/tactical.
    - Measures Spacing (Cohesion) + Integrity (Clarity).
    - Penalizes defenses that are geometrically tight but tactically ambiguous.
    - DIS = (beta * spacing + gamma * integrity) / 2
"""

alpha = 1.0
beta = 1.0
gamma = 1.0

# DCI: Geometric Coverage Quality
dci_base = np.exp(-alpha * dist_to_centroid)

# DIS: Structural Integrity & Clarity
dis_base = (beta * spacing_proxy + gamma * integrity_proxy) / 2.0


# -----------------------------------------------------------
# EXPORT RESULTS
# -----------------------------------------------------------

print("[INFO] Exporting baseline metrics...")

out_df = df.copy()
out_df["cluster_id"] = closest
out_df["distance_to_ideal"] = dist_to_centroid
out_df["distance_to_second"] = dist_to_second

# Proxies
out_df["spacing_proxy"] = spacing_proxy
out_df["execution_proxy"] = execution_proxy
out_df["integrity_proxy"] = integrity_proxy  

# Baseline Metrics
out_df["dci_base"] = dci_base
out_df["dis_base"] = dis_base

out_df.to_parquet(OUT_PATH, index=False)

print(f"\n Baseline metrics saved successfully to:")
print(f"    {OUT_PATH}")
print(f"    - Included 'integrity_proxy' for DIS calculation.")
print(f"    - Included 'distance_to_second' for ambiguity analysis.\n")